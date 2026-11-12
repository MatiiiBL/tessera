"""Templates for NHWC convolutions and pooling: output pixels x channels,
reduced over a window (and input channels)."""

from __future__ import annotations

from math import prod

from tessera.backend.c.templates.common import (
    VECTOR_SIZES,
    CpuTarget,
    choose_int,
    loop_kinds,
    nearest,
    parallel_knob,
    parallelize,
    register_options,
)
from tessera.backend.c.templates.matmul import tile_loops
from tessera.plan import LoweredGroup
from tessera.tir.schedule import Schedule
from tessera.tune.space import Config


def schedule_conv2d(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Register-blocked loops over (output column, output channel) tiles:
    ``tile_ow`` columns (unrolled) by ``tile_o`` channels (vectorized) are
    accumulated in a local buffer over the whole window and all input
    channels, then the epilogue runs on the tile. Output rows of all images
    run in parallel."""
    _schedule_windowed(sch, group, cfg, rows_default=4, lanes_default=16)


def schedule_depthwise_conv2d(
    sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget
) -> None:
    """As ``schedule_conv2d``, reducing over the window only: each channel's
    weights are reused across ``tile_ow`` output columns."""
    _schedule_windowed(sch, group, cfg, rows_default=4, lanes_default=16)


def schedule_pool(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Max and average pooling: channels vectorized, the window reduced in a
    local buffer, the average's scaling applied when the tile is stored."""
    _schedule_windowed(sch, group, cfg, rows_default=1, lanes_default=16)


def _schedule_windowed(
    sch: Schedule, group: LoweredGroup, cfg: Config, *, rows_default: int, lanes_default: int
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
    enabled = parallel_knob(cfg, work)

    owo, owi = sch.split(ow, [None, tile_ow])
    co, ci = sch.split(ch, [None, tile_c])
    sch.reorder(owo, co, *reduce, owi, ci)
    tile_loops(sch, main, group.epilogue_block, co, reduce[0], owi, ci)
    parallelize(sch, [n, oh, owo], enabled)
