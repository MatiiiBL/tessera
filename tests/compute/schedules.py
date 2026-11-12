"""Simple CPU-style and GPU-style schedules for any lowered group.

They are not tuned; they apply the primitives the backend templates use
(tiling, cache_write, epilogue inlining, compute_at of statistics, reduction
decomposition, parallel, vectorize, bind) so that the tests exercise them on
every group of the real models.
"""

from __future__ import annotations

from tessera.compute import ROLE
from tessera.plan import LoweredGroup
from tessera.tir.primitives.common import block_paths, realize_of
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import IterKind, PrimFunc


def roles(func: PrimFunc) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in block_paths(func):
        b = realize_of(path).block
        role = dict(b.annotations).get(ROLE)
        if isinstance(role, str):
            out[role] = b.name
    return out


def _spatial_count(s: Schedule, block: str) -> int:
    from tessera.tir.primitives.common import find_block

    b = realize_of(find_block(s.func, block)).block
    return sum(1 for iv in b.iter_vars if iv.kind == IterKind.SPATIAL)


def _stats(lowered: LoweredGroup) -> list[str]:
    r = roles(lowered.func)
    return [r[k] for k in ("sum", "max", "var", "mean") if k in r]


def cpu(lowered: LoweredGroup, tile: int = 4) -> Schedule:
    s = Schedule(lowered.func)
    main, epilogue = lowered.main_block, lowered.epilogue_block
    if main is None:
        (block,) = s.blocks()
        loops = s.get_loops(block)
        fused = s.fuse(*loops) if len(loops) > 1 else loops[0]
        outer, inner = s.split(fused, [None, 16])
        s.parallel(outer)
        s.vectorize(inner)
        return s
    if lowered.pattern in ("softmax", "layernorm"):
        if epilogue is not None:
            s.reverse_compute_inline(epilogue)
        loops = s.get_loops(main)
        for stat in _stats(lowered):
            s.compute_at(stat, loops[0])
        s.parallel(loops[0])
        return s
    copy = s.cache_write(main, 0, "local")
    if epilogue is not None:
        s.reverse_compute_inline(epilogue)
    loops = s.get_loops(main)
    n = _spatial_count(s, main)
    spatial, reduce = loops[:n], loops[n:]
    if not spatial:
        return s
    tiles = [s.split(v, [None, tile]) for v in spatial[-2:]]
    outer = spatial[:-2] + [t[0] for t in tiles]
    inner = [t[1] for t in tiles]
    s.reorder(*outer, *reduce, *inner)
    s.reverse_compute_at(copy, outer[-1])
    if reduce:
        s.decompose_reduction(main, reduce[0])
    fused = s.fuse(*outer) if len(outer) > 1 else outer[0]
    s.parallel(fused)
    s.vectorize(s.get_loops(copy)[-1])
    return s


def gpu(lowered: LoweredGroup, tile: int = 4) -> Schedule:
    s = Schedule(lowered.func)
    main, epilogue = lowered.main_block, lowered.epilogue_block
    if main is None:
        (block,) = s.blocks()
        loops = s.get_loops(block)
        fused = s.fuse(*loops) if len(loops) > 1 else loops[0]
        outer, inner = s.split(fused, [None, 64])
        s.bind(outer, "threadgroup.x")
        s.bind(inner, "thread.x")
        return s
    if lowered.pattern in ("softmax", "layernorm"):
        if epilogue is not None:
            s.reverse_compute_inline(epilogue)
        loops = s.get_loops(main)
        for stat in _stats(lowered):
            s.compute_at(stat, loops[0])
        s.bind(loops[0], "threadgroup.x")
        for stat in _stats(lowered):
            s.bind(s.get_loops(stat)[-1], "thread.x")
        s.bind(s.get_loops(main)[-1], "thread.x")
        return s
    copy = s.cache_write(main, 0, "local")
    if epilogue is not None:
        s.reverse_compute_inline(epilogue)
    loops = s.get_loops(main)
    n = _spatial_count(s, main)
    spatial, reduce = loops[:n], loops[n:]
    if not spatial:
        return s
    tiles = [s.split(v, [None, tile]) for v in spatial[-2:]]
    outer = spatial[:-2] + [t[0] for t in tiles]
    threads = [t[1] for t in tiles]
    s.reorder(*outer, *threads, *reduce)
    fused = s.fuse(*outer) if len(outer) > 1 else outer[0]
    s.reverse_compute_at(copy, threads[-1])
    if reduce:
        s.decompose_reduction(main, reduce[0])
    s.bind(fused, "threadgroup.x")
    for v, tag in zip(threads, ("thread.y", "thread.x")[-len(threads) :], strict=True):
        s.bind(v, tag)
    return s
