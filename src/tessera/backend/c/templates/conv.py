"""Templates for NHWC convolutions and pooling: output pixels x channels,
reduced over a window (and input channels)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tessera.backend.c.templates.common import (
    PACK_LIMIT,
    VECTOR_SIZES,
    CpuTarget,
    balanced_tile,
    choose,
    choose_bool,
    choose_int,
    loop_kinds,
    nearest,
    parallel_knob,
    parallel_on,
    parallelize,
    read_index,
    register_options,
    schedule_alone,
    tile_options,
    try_apply,
    vectorize_inside,
)
from tessera.backend.c.templates.matmul import tile_loops
from tessera.backend.c.templates.micro import ConvLayout, conv_intrinsic, conv_layout
from tessera.plan import LoweredGroup
from tessera.tir.expr import Var
from tessera.tir.schedule import Schedule
from tessera.tune.space import Config


def schedule_conv2d(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Register-blocked loops over (output column, output channel) tiles:
    ``tile_ow`` columns (unrolled) by ``tile_c`` channels (vectorized) are
    accumulated in a local buffer over the whole window and all input
    channels, then the epilogue runs on the tile. With ``order`` "co" the
    loop over channel strips is outermost and each strip's weights
    (KH x KW x C x ``tile_c``) are copied into a contiguous local buffer
    (``pack_w``, when the strip fits the stack) that every pixel tile
    reuses; "pixels" puts the pixel tiles outside.

    On NEON and SME targets an NHWC convolution is computed by the implicit
    GEMM microkernel instead (``kernel``, see ``_micro_knobs``)."""
    _schedule_windowed(sch, group, cfg, target, rows_default=4, lanes_default=16, weights=True)


def schedule_depthwise_conv2d(
    sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget
) -> None:
    """As ``schedule_conv2d``, reducing over the window only: each channel's
    weights are reused across ``tile_ow`` output columns."""
    _schedule_windowed(sch, group, cfg, target, rows_default=4, lanes_default=16)


def schedule_pool(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Max and average pooling: channels vectorized, the window reduced in a
    local buffer, the average's scaling applied when the tile is stored."""
    _schedule_windowed(sch, group, cfg, target, rows_default=1, lanes_default=16)


def _schedule_windowed(
    sch: Schedule,
    group: LoweredGroup,
    cfg: Config,
    target: CpuTarget,
    *,
    rows_default: int,
    lanes_default: int,
    weights: bool = False,
) -> None:
    main = group.main_block
    assert main is not None
    loops = sch.get_loops(main)
    kinds = loop_kinds(sch, main)
    spatial = [v for v, k in zip(loops, kinds, strict=True) if k == "S"]
    reduce = [v for v, k in zip(loops, kinds, strict=True) if k == "R"]
    n, oh, ow, ch = spatial
    width, channels = sch.extent(ow), sch.extent(ch)
    work = prod(sch.extent(v) for v in loops)

    rows = register_options(width)
    tile_ow = choose_int(cfg, "tile_ow", rows, nearest(rows, rows_default))
    lanes = [v for v in VECTOR_SIZES if channels % v == 0] or [1]
    tile_c = choose_int(cfg, "tile_c", lanes, nearest(lanes, lanes_default))
    if weights:
        order = choose(cfg, "order", ["co", "pixels"], "co")
        pack_w = choose_bool(cfg, "pack_w", True)
    else:
        order, pack_w = "pixels", False
    enabled = parallel_knob(cfg, work)

    layout = conv_layout(sch, main, group) if weights and target.isa != "generic" else None
    if layout is not None:
        micro = _micro_knobs(cfg, channels, target)
        if micro.enabled:
            _schedule_micro(sch, group, layout, spatial, reduce, micro, enabled, target)
            return

    owo, owi = sch.split(ow, [None, tile_ow])
    co, ci = sch.split(ch, [None, tile_c])
    if order == "pixels":
        sch.reorder(owo, co, *reduce, owi, ci)
        tile_loops(sch, main, group.epilogue_block, co, reduce[0], owi, ci)
        parallelize(sch, [n, oh, owo], enabled)
        return
    sch.reorder(co, n, oh, owo, *reduce, owi, ci)
    tile_loops(sch, main, group.epilogue_block, owo, reduce[0], owi, ci)
    # Enough channel strips for every thread: run strips in parallel and pack
    # each strip's weights once. Otherwise output rows run in parallel too,
    # and each row packs the weights it uses.
    strips_only = channels // tile_c >= 2 * target.threads
    w_index = read_index(sch, main, uses=[co, *reduce], avoids=owo)
    strip = prod(sch.extent(v) for v in reduce) * tile_c * 4
    if pack_w and w_index is not None and strip <= PACK_LIMIT:
        sch.compute_at(sch.cache_read(main, w_index, "local"), co if strips_only else oh)
    parallelize(sch, [co] if strips_only else [co, n, oh], enabled)


@dataclass(frozen=True)
class MicroKnobs:
    enabled: bool  # use the convolution microkernel (else the generated loops)
    strip: int  # output channels per call
    threads: int


def _micro_knobs(cfg: Config, channels: int, target: CpuTarget) -> MicroKnobs:
    """The microkernel path's knobs: one call computes the whole output plane of
    an image for a strip of ``micro_c`` output channels; strips run in
    parallel on ``micro_threads`` threads (all of them by default, as for
    matmul). The default strip gives every thread a whole number of strips
    at most 256 channels wide. On NEON the loops stay the default when the
    strips would be narrower than two of the microkernel's 12-wide panels."""
    # NEON's panels are 12 channels wide: below two panels per thread's strip
    # (64-channel layers on four threads) the generated loops are faster.
    narrow = target.isa == "neon" and channels // target.threads < 24
    default = "loops" if narrow else "micro"
    enabled = choose(cfg, "kernel", ["micro", "loops"], default) == "micro"
    counts = sorted({c for c in (1, 2, target.threads) if c <= target.threads})
    threads = choose_int(cfg, "micro_threads", counts, target.threads)
    strips = tile_options(channels, 1024)
    strip = choose_int(cfg, "micro_c", strips, balanced_tile(channels, 1, target.threads, strips))
    return MicroKnobs(enabled, strip, threads)


def _schedule_micro(
    sch: Schedule,
    group: LoweredGroup,
    layout: ConvLayout,
    spatial: list[Var],
    reduce: list[Var],
    knobs: MicroKnobs,
    enabled: bool,
    target: CpuTarget,
) -> None:
    """Strips of output channels computed by the convolution microkernel (an
    implicit GEMM over the whole output plane), the bias written by the
    reduction's init before each call and the epilogue applied after it."""
    main = group.main_block
    assert main is not None
    n, oh, ow, ch = spatial
    co, ci = sch.split(ch, [None, knobs.strip])
    sch.reorder(n, co, oh, ow, ci)
    init = sch.decompose_reduction(main, oh)
    vectorize_inside(sch, init, co)
    epilogue = group.epilogue_block
    if epilogue is not None:
        if try_apply(sch.reverse_compute_at, epilogue, co):
            vectorize_inside(sch, epilogue, co)
        else:
            schedule_alone(sch, epilogue)
    parallel_on(sch, [n, co], enabled, knobs.threads)
    nest = [oh, ow, ci, *reduce]
    sch.tensorize(oh, conv_intrinsic(sch, main, group, layout, nest, target.isa))
