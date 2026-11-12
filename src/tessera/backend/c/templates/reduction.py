"""Templates for reductions: ReduceSum/Mean/Max, and the row-wise softmax and
layernorm, whose stages each reduce a row before the row is normalized."""

from __future__ import annotations

from math import prod

from tessera.backend.c.templates.common import (
    CpuTarget,
    choose_int,
    loop_kinds,
    nearest,
    output_block,
    parallel_knob,
    parallelize,
    place_epilogue,
    vector_options,
)
from tessera.plan import LoweredGroup
from tessera.tir.expr import Var
from tessera.tir.schedule import Schedule
from tessera.tune.space import Config


def schedule_reduce(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """A reduction over the last axes (INNER == 1) keeps its accumulator in a
    local buffer and lets clang vectorize the contiguous reduction loop;
    otherwise the reduction loops move outside the inner output axes, whose
    innermost loop is vectorized (``vector``). Outer output loops run in
    parallel."""
    main = group.main_block
    assert main is not None
    loops = sch.get_loops(main)
    kinds = loop_kinds(sch, main)
    spatial = [v for v, k in zip(loops, kinds, strict=True) if k == "S"]
    reduce = [v for v, k in zip(loops, kinds, strict=True) if k == "R"]
    work = prod(sch.extent(v) for v in loops)
    inner = _inner_loops(sch, spatial, group.dims.get("INNER", 1))
    outer = spatial[: len(spatial) - len(inner)]
    options = vector_options(sch.extent(inner[-1])) if inner else [1]
    vector = choose_int(cfg, "vector", options, nearest(options, 16))
    enabled = parallel_knob(cfg, work)
    epilogue = group.epilogue_block

    if not inner:
        # Contiguous reduction: one accumulator per output element.
        rloop = reduce[0] if len(reduce) == 1 else sch.fuse(*reduce)
        copy = sch.cache_write(main, 0, "local")
        if outer:
            sch.reverse_compute_at(copy, outer[-1])
        sch.decompose_reduction(main, rloop)
        place_epilogue(sch, epilogue, outer[-1] if outer else None)
        parallelize(sch, outer, enabled)
        return

    lanes = inner[-1]
    rest: list[Var] = list(inner[:-1])
    if 1 < vector < sch.extent(lanes):
        head, lanes = sch.split(lanes, [None, vector])
        rest.append(head)
    sch.reorder(*rest, *reduce, lanes)
    if vector > 1:
        sch.vectorize(lanes)
    sch.decompose_reduction(main, reduce[0])
    around = [*outer, *rest]
    place_epilogue(sch, epilogue, around[-1] if around else None)
    parallelize(sch, [*outer, *rest], enabled)


def _inner_loops(sch: Schedule, spatial: list[Var], inner: int) -> list[Var]:
    """The trailing spatial loops whose extents multiply to ``inner`` (the
    output axes after the reduced ones)."""
    out: list[Var] = []
    size = 1
    for loop in reversed(spatial):
        if size >= inner:
            break
        size *= sch.extent(loop)
        out.insert(0, loop)
    return out if size == inner else []


def schedule_rows(sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget) -> None:
    """Softmax and LayerNorm: every reduction stage is computed per row, inside
    the loop over rows of the final stage, so a row is read from cache after
    the first pass. Reductions keep their running value in a local buffer
    (clang vectorizes the contiguous loops); the final stage's row loop is
    vectorized (``vector``); rows run in parallel."""
    out = output_block(sch)
    loops = sch.get_loops(out)
    work = prod(sch.extent(v) for v in loops) * len(sch.blocks())
    contiguous = group.dims.get("INNER", 1) == 1
    columns = loops[-1]
    rows = loops[:-1] if contiguous else loops[:1]
    options = vector_options(sch.extent(columns)) if contiguous else [1]
    vector = choose_int(cfg, "vector", options, nearest(options, 16))
    enabled = parallel_knob(cfg, work)

    stages = [b for b in sch.blocks() if b != out]
    if rows:
        for stage in reversed(stages):
            sch.compute_at(stage, rows[-1])
    for stage in stages:
        kinds = loop_kinds(sch, stage)
        reduce = [v for v, k in zip(sch.get_loops(stage), kinds, strict=True) if k == "R"]
        if reduce:
            sch.decompose_reduction(stage, reduce[0])
    if contiguous and vector > 1:
        if vector == sch.extent(columns):
            sch.vectorize(columns)
        else:
            _, lanes = sch.split(columns, [None, vector])
            sch.vectorize(lanes)
    place_epilogue(sch, group.epilogue_block, rows[-1] if rows else None)
    parallelize(sch, rows, enabled)
