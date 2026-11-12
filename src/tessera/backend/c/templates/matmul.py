"""The matmul template: C[b, i, j] = sum_k A[b, i, k] B[k, j], plus prologue and epilogue."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tessera.backend.c.templates.common import (
    PACK_LIMIT,
    CpuTarget,
    balanced_tile,
    choose,
    choose_bool,
    choose_int,
    divisors,
    iter_var_of,
    loop_kinds,
    nearest,
    parallel_knob,
    parallel_on,
    parallelize,
    place_epilogue,
    read_index,
    register_options,
    schedule_alone,
    tile_options,
    try_apply,
    vectorize_inside,
)
from tessera.backend.c.templates.micro import GemmLayout, gemm_intrinsic, gemm_layout, is_constant
from tessera.plan import LoweredGroup
from tessera.tir.expr import Var
from tessera.tir.schedule import Schedule
from tessera.tune.space import Config


def schedule_matmul(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Register-blocked loops: each ``tile_m`` x ``tile_n`` tile of C is
    accumulated in a local buffer over the whole reduction (rows unrolled,
    columns vectorized), then the epilogue runs on the tile while it is in
    cache.

    ``order`` "nm" walks column tiles outside row tiles: a column strip of B
    (K x ``tile_n``) is then read once per strip and, with ``pack_b``, copied
    into a contiguous local buffer that stays in L1 while every row tile
    uses it; batches and column strips run in parallel. This is what weight
    matrices larger than the caches need (a strip is packed only if it fits
    the stack, see ``PACK_LIMIT``). "mn" walks row tiles outside and runs
    every tile in parallel."""
    main = group.main_block
    assert main is not None
    loops = sch.get_loops(main)
    kinds = loop_kinds(sch, main)
    spatial = [v for v, k in zip(loops, kinds, strict=True) if k == "S"]
    reduce = [v for v, k in zip(loops, kinds, strict=True) if k == "R"]
    *batch, i, j = spatial
    m, n = sch.extent(i), sch.extent(j)
    work = prod(sch.extent(v) for v in loops)

    rows = register_options(m)
    tile_m = choose_int(cfg, "tile_m", rows, nearest(rows, 4))
    columns = [d for d in divisors(n) if d <= 64]
    preferred = [d for d in columns if d % 4 == 0 and d <= 16]
    tile_n = choose_int(
        cfg, "tile_n", columns, max(preferred) if preferred else nearest(columns, 16)
    )
    order = choose(cfg, "order", ["nm", "mn"], "nm")
    pack_b = choose_bool(cfg, "pack_b", True)
    enabled = parallel_knob(cfg, work)

    layout = None
    if target.isa != "generic" and len(reduce) == 1:
        ivs = [iter_var_of(sch, main, v) for v in (i, j, reduce[0])]
        if all(ivs):
            layout = gemm_layout(sch, main, *ivs)  # type: ignore[arg-type]
    if layout is not None:
        batches = prod(sch.extent(v) for v in batch)
        micro = _micro_knobs(cfg, m, n, batches, target)
        if micro.enabled:
            _schedule_micro(sch, group, layout, batch, i, j, reduce[0], micro, enabled, target)
            return

    io, ii = sch.split(i, [None, tile_m])
    jo, ji = sch.split(j, [None, tile_n])
    if order == "nm":
        sch.reorder(jo, io, *reduce, ii, ji)
        tile_loops(sch, main, group.epilogue_block, io, reduce[0], ii, ji)
        b_index = read_index(sch, main, uses=[jo, *reduce], avoids=io)
        strip = prod(sch.extent(v) for v in reduce) * tile_n * 4
        if pack_b and b_index is not None and strip <= PACK_LIMIT:
            sch.compute_at(sch.cache_read(main, b_index, "local"), jo)
        parallelize(sch, [*batch, jo], enabled)
    else:
        sch.reorder(io, jo, *reduce, ii, ji)
        tile_loops(sch, main, group.epilogue_block, jo, reduce[0], ii, ji)
        parallelize(sch, [*batch, io, jo], enabled)


@dataclass(frozen=True)
class MicroKnobs:
    enabled: bool  # use the microkernel (else the generated loops)
    tile_m: int
    tile_n: int
    threads: int


def _micro_knobs(cfg: Config, m: int, n: int, batches: int, target: CpuTarget) -> MicroKnobs:
    """The microkernel path's knobs. Tiles are the unit of parallel work and of
    epilogue fusion; inside a tile the microkernel blocks for the caches
    itself, but it packs its operands at every call, so tiles are wide. By
    default a tile has up to 256 rows (one block of A, which the microkernel
    then packs once per call), and its width gives every thread a whole
    number of tiles at most 1024 columns wide (``balanced_tile``).

    On SME the performance cores share one matrix unit. Tiles are sized for
    one thread, and the tiles run on every thread only when there are no
    more of them than threads: a few wide tiles overlap one thread's packing
    and epilogue with another's streaming computation, but many tiles keep
    all threads contending for the unit (results/cpu/sme_models.md)."""
    enabled = choose(cfg, "kernel", ["micro", "loops"], "micro") == "micro"
    # Tiles need not divide the matrix: an edge tile computes its valid part.
    rows = tile_options(m, 512)
    default_m = min(m, 256)
    row_tiles = batches * -(-m // default_m)
    columns = tile_options(n, 1024)
    sme = target.isa == "sme"
    default_n = balanced_tile(n, row_tiles, 1 if sme else target.threads, columns, 1024)
    tiles = row_tiles * -(-n // default_n)
    counts = sorted({c for c in (1, 2, target.threads) if c <= target.threads})
    default_threads = 1 if sme and tiles > target.threads else target.threads
    threads = choose_int(cfg, "micro_threads", counts, default_threads)
    tile_m = choose_int(cfg, "micro_m", rows, default_m)
    tile_n = choose_int(cfg, "micro_n", columns, default_n)
    return MicroKnobs(enabled, tile_m, tile_n, threads)


def _schedule_micro(
    sch: Schedule,
    group: LoweredGroup,
    layout: GemmLayout,
    batch: list[Var],
    i: Var,
    j: Var,
    k: Var,
    knobs: MicroKnobs,
    enabled: bool,
    target: CpuTarget,
) -> None:
    """Tiles of C computed by the GEMM microkernel, column strips outside row
    tiles, the epilogue applied to each tile after its call."""
    main = group.main_block
    assert main is not None
    io, ii = sch.split(i, [None, knobs.tile_m])
    jo, ji = sch.split(j, [None, knobs.tile_n])
    sch.reorder(jo, io, ii, ji, k)
    init = sch.decompose_reduction(main, ii)
    vectorize_inside(sch, init, io)
    epilogue = group.epilogue_block
    if epilogue is not None:
        if try_apply(sch.reverse_compute_at, epilogue, io):
            vectorize_inside(sch, epilogue, io)
        else:
            schedule_alone(sch, epilogue)
    # Parallelize first: the tensorized block passes every operand by address.
    parallel_on(sch, [*batch, jo, io], enabled, knobs.threads)
    constant = is_constant(group, layout.b)
    intrinsic = gemm_intrinsic(sch, main, layout, [ii, ji, k], target.isa, constant_b=constant)
    sch.tensorize(ii, intrinsic)


def tile_loops(
    sch: Schedule,
    main: str,
    epilogue: str | None,
    tile: Var,
    first_reduce: Var,
    rows: Var | None,
    lanes: Var,
) -> str:
    """Accumulate the tile under ``tile`` in a local buffer: initialize it
    before ``first_reduce``, copy it out (through the epilogue) after the
    reduction, unroll ``rows`` (if any) and vectorize ``lanes``. Returns the
    copy-out block."""
    copy = sch.cache_write(main, 0, "local")
    sch.reverse_compute_at(copy, tile)
    sch.decompose_reduction(main, first_reduce)
    place_epilogue(sch, epilogue, tile)
    if rows is not None:
        sch.unroll(rows)
    sch.vectorize(lanes)
    copy_loops = sch.get_loops(copy)
    inside = copy_loops[copy_loops.index(tile) + 1 :]
    if inside and sch.extent(inside[-1]) > 1:
        sch.vectorize(inside[-1])
    return copy
