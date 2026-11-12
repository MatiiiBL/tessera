"""The matmul template: C[b, i, j] = sum_k A[b, i, k] B[k, j], plus prologue and epilogue."""

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
    place_epilogue,
    register_options,
)
from tessera.plan import LoweredGroup
from tessera.tir.expr import Var
from tessera.tir.schedule import Schedule
from tessera.tune.space import Config


def schedule_matmul(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Register-blocked loops: each ``tile_m`` x ``tile_n`` tile of C is
    accumulated in a local buffer over the whole reduction (rows unrolled,
    columns vectorized), then the epilogue runs on the tile while it is in
    cache. Tiles of all batches run in parallel."""
    main = group.main_block
    assert main is not None
    loops = sch.get_loops(main)
    kinds = loop_kinds(sch, main)
    spatial = [v for v, k in zip(loops, kinds, strict=True) if k == "S"]
    reduce = [v for v, k in zip(loops, kinds, strict=True) if k == "R"]
    *batch, i, j = spatial
    m, n = sch.extent(i), sch.extent(j)
    work = prod(sch.extent(v) for v in loops)

    tile_m = choose_int(cfg, "tile_m", register_options(m), nearest(register_options(m), 4))
    columns = [v for v in VECTOR_SIZES if n % v == 0] or [1]
    tile_n = choose_int(cfg, "tile_n", columns, nearest(columns, 16))
    enabled = parallel_knob(cfg, work)

    io, ii = sch.split(i, [None, tile_m])
    jo, ji = sch.split(j, [None, tile_n])
    sch.reorder(io, jo, *reduce, ii, ji)
    tile_loops(sch, main, group.epilogue_block, jo, reduce[0], ii, ji)
    parallelize(sch, [*batch, io, jo], enabled)


def tile_loops(
    sch: Schedule,
    main: str,
    epilogue: str | None,
    tile: Var,
    first_reduce: Var,
    rows: Var,
    lanes: Var,
) -> str:
    """Accumulate the tile under ``tile`` in a local buffer: initialize it
    before ``first_reduce``, copy it out (through the epilogue) after the
    reduction, unroll ``rows`` and vectorize ``lanes``. Returns the copy-out
    block."""
    copy = sch.cache_write(main, 0, "local")
    sch.reverse_compute_at(copy, tile)
    sch.decompose_reduction(main, first_reduce)
    place_epilogue(sch, epilogue, tile)
    sch.unroll(rows)
    sch.vectorize(lanes)
    copy_loops = sch.get_loops(copy)
    inside = copy_loops[copy_loops.index(tile) + 1 :]
    if inside and sch.extent(inside[-1]) > 1:
        sch.vectorize(inside[-1])
    return copy
