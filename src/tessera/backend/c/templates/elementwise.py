"""Templates for groups without a reduction: elementwise, broadcast and injective."""

from __future__ import annotations

from math import prod

from tessera.backend.c.templates.common import (
    CpuTarget,
    choose_int,
    nearest,
    output_block,
    parallel_knob,
    parallelize,
    try_apply,
    vector_options,
)
from tessera.plan import LoweredGroup
from tessera.tir.schedule import Schedule
from tessera.tune.space import Config


def schedule_injective(
    sch: Schedule, group: LoweredGroup, cfg: Config, *, target: CpuTarget
) -> None:
    """Inline every stage into the output's, vectorize its innermost loop
    (``vector``: the split factor, 1 for none) and run the remaining loops,
    fused, on the thread pool (``parallel``)."""
    out = output_block(sch)
    for name in sch.blocks():
        if name != out:
            try_apply(sch.compute_inline, name)
    loops = sch.get_loops(out)
    if not loops:
        return
    work = prod(sch.extent(v) for v in loops)
    inner = loops[-1]
    extent = sch.extent(inner)
    options = vector_options(extent)
    vector = choose_int(cfg, "vector", options, nearest(options, 16))
    enabled = parallel_knob(cfg, work)
    outer = list(loops[:-1])
    if vector == extent:
        sch.vectorize(inner)
    elif vector > 1:
        rest, lanes = sch.split(inner, [None, vector])
        sch.vectorize(lanes)
        outer.append(rest)
    else:
        outer.append(inner)
    parallelize(sch, outer, enabled)
