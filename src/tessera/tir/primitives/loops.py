"""Loop primitives: split, fuse, reorder, and loop kinds (parallel, vectorize, ...)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from math import prod
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera.tir.analysis import is_leaf_block
from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import Binary, Compare, Expr, IntImm, Logical, Not, Select, Var, lt
from tessera.tir.functor import StmtMutator, children, free_vars, substitute_stmt, walk_stmts
from tessera.tir.primitives.common import (
    Path,
    ScheduleError,
    analyzer_for,
    block_paths,
    check_reductions,
    conjoin,
    facts,
    find_block,
    find_loop,
    fresh,
    loop_of,
    loop_roles,
    loops_above,
    readers,
    realize_of,
    rebuild,
    replace_for,
    simplify_stmt,
    used_names,
    with_body,
    writers,
)
from tessera.tir.stmt import (
    THREAD_TAGS,
    Annotations,
    Block,
    BlockRealize,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IterKind,
    PrimFunc,
    Stmt,
)

type NDArray = npt.NDArray[Any]


def get_loops(func: PrimFunc, block: str) -> list[Var]:
    return [f.loop_var for f in loops_above(find_block(func, block))]


def const_extent(loop: For) -> int:
    if not isinstance(loop.extent, IntImm):
        raise ScheduleError(f"loop {loop.loop_var.name} has no constant extent")
    return loop.extent.value


def _require_serial(loop: For, what: str) -> None:
    if loop.kind != ForKind.SERIAL or loop.thread_tag is not None:
        raise ScheduleError(f"cannot {what} loop {loop.loop_var.name}: it is {loop.kind.value}")


def _top_realizes(stmt: Stmt) -> list[BlockRealize]:
    """BlockRealizes in ``stmt`` that are not nested inside another block."""
    out: list[BlockRealize] = []
    stack = [stmt]
    while stack:
        s = stack.pop()
        if isinstance(s, BlockRealize):
            out.append(s)
            continue
        stack.extend(children(s))
    return out


def _has_loose_statements(stmt: Stmt) -> bool:
    """True if ``stmt`` has stores or calls outside any block."""
    stack = [stmt]
    while stack:
        s = stack.pop()
        if isinstance(s, BlockRealize):
            continue
        if isinstance(s, BufferStore | Evaluate):
            return True
        stack.extend(children(s))
    return False


def _add_predicate(stmt: Stmt, pred: Expr) -> Stmt:
    targets = {id(r) for r in _top_realizes(stmt)}

    class _Guard(StmtMutator):
        def visit_block_realize(self, s: BlockRealize) -> Stmt:
            if id(s) in targets:
                return BlockRealize(s.iter_values, conjoin(s.predicate, pred), s.block)
            return s

    return _Guard().visit_stmt(stmt)


def _finish(func: PrimFunc, path: Path, new: Stmt) -> PrimFunc:
    """Rebuild with ``new`` at ``path`` and simplify it with the loop ranges in scope."""
    new = simplify_stmt(new, analyzer_for(path))
    return with_body(func, rebuild(path, new))


def split(func: PrimFunc, loop: Var, factors: Sequence[int | None]) -> tuple[PrimFunc, list[Var]]:
    path = find_loop(func, loop)
    node = loop_of(path)
    _require_serial(node, "split")
    extent = const_extent(node)
    facs = list(factors)
    if not facs:
        raise ScheduleError("split needs at least one factor")
    if sum(f is None for f in facs) > 1:
        raise ScheduleError("at most one split factor may be None")
    for f in facs:
        if f is not None and (not isinstance(f, int) or f < 1):
            raise ScheduleError(f"split factors must be positive integers, got {f!r}")
    known = prod(f for f in facs if f is not None)
    if None in facs:
        facs[facs.index(None)] = -(-extent // known)
    sizes = [int(f) for f in facs if f is not None]
    total = prod(sizes)
    if total < extent:
        raise ScheduleError(f"factors {sizes} cover {total} of the {extent} iterations")
    names = used_names(func)
    new_vars = [Var(fresh(f"{loop.name}_{k}", names), loop.dtype) for k in range(len(sizes))]
    fused: Expr = IntImm(0, loop.dtype)
    for k, v in enumerate(new_vars):
        stride = prod(sizes[k + 1 :])
        fused = fused + (v * stride if stride != 1 else v)
    body = substitute_stmt(node.body, {loop: node.min + fused if node.min != IntImm(0) else fused})
    if total > extent:
        if _has_loose_statements(node.body):
            raise ScheduleError(
                f"cannot split loop {loop.name} imperfectly: it has statements outside blocks"
            )
        body = _add_predicate(body, lt(fused, extent))
    stmt: Stmt = body
    for v, size in zip(reversed(new_vars), reversed(sizes), strict=True):
        stmt = For(v, IntImm(0, loop.dtype), IntImm(size, loop.dtype), ForKind.SERIAL, stmt)
    if node.annotations:
        assert isinstance(stmt, For)
        stmt = replace_for(stmt, annotations=node.annotations)
    return _finish(func, path, stmt), new_vars


def _perfect_chain(func: PrimFunc, loops: Sequence[Var]) -> tuple[Path, list[For]]:
    """The path to the outermost of ``loops`` and every loop from it down to the
    innermost of ``loops``, which must form a perfect nest."""
    paths = [find_loop(func, v) for v in loops]
    paths.sort(key=len)
    outer, inner = paths[0], paths[-1]
    for p in paths:
        if any(a is not b for a, b in zip(inner[: len(p)], p, strict=True)):
            raise ScheduleError("the loops are not nested in one another")
    chain = list(inner[len(outer) - 1 :])
    for node in chain:
        if not isinstance(node, For):
            raise ScheduleError("the loops are separated by a block or statement")
    fors = [n for n in chain if isinstance(n, For)]
    for a, b in itertools.pairwise(fors):
        if a.body is not b:
            raise ScheduleError(
                f"loops {a.loop_var.name} and {b.loop_var.name} are not perfectly nested"
            )
    return outer, fors


def fuse(func: PrimFunc, loops: Sequence[Var]) -> tuple[PrimFunc, Var]:
    if not loops:
        raise ScheduleError("fuse needs at least one loop")
    if len(set(map(id, loops))) != len(loops):
        raise ScheduleError("fuse got the same loop twice")
    if len(loops) == 1:
        find_loop(func, loops[0])
        return func, loops[0]
    path, chain = _perfect_chain(func, loops)
    if [f.loop_var for f in chain] != list(loops):
        raise ScheduleError("fuse needs consecutive loops, outermost first")
    for f in chain:
        _require_serial(f, "fuse")
    extents = [const_extent(f) for f in chain]
    for bpath in block_paths(func):
        if not any(n is chain[0] for n in bpath):
            continue
        roles = set(loop_roles(realize_of(bpath), chain)) - {"N"}
        if "X" in roles or {"S", "R"} <= roles:
            name = realize_of(bpath).block.name
            raise ScheduleError(f"cannot fuse spatial and reduction loops of block {name}")
    names = used_names(func)
    dtype = loops[0].dtype
    fused = Var(fresh("_".join(v.name for v in loops) + "_fused", names), dtype)
    mapping: dict[Var, Expr] = {}
    for k, f in enumerate(chain):
        stride = prod(extents[k + 1 :])
        e: Expr = fused // stride if stride != 1 else fused
        if k > 0:
            e = e % extents[k]
        mapping[f.loop_var] = e if f.min == IntImm(0) else f.min + e
    body = substitute_stmt(chain[-1].body, mapping)
    annotations: Annotations = tuple(a for f in chain for a in f.annotations)
    new = For(fused, IntImm(0, dtype), IntImm(prod(extents), dtype), ForKind.SERIAL, body)
    new = replace_for(new, annotations=annotations)
    return _finish(func, path, new), fused


def reorder(func: PrimFunc, loops: Sequence[Var]) -> PrimFunc:
    if len(set(map(id, loops))) != len(loops):
        raise ScheduleError("reorder got the same loop twice")
    if len(loops) < 2:
        if loops:
            find_loop(func, loops[0])
        return func
    path, chain = _perfect_chain(func, loops)
    chain_vars = {f.loop_var for f in chain}
    for f in chain:
        if free_vars(f.min) & chain_vars or free_vars(f.extent) & chain_vars:
            raise ScheduleError(f"the range of loop {f.loop_var.name} depends on another loop")
    wanted = {id(v) for v in loops}
    order = iter(loops)
    by_var = {id(f.loop_var): f for f in chain}
    new_chain = [by_var[id(next(order))] if id(f.loop_var) in wanted else f for f in chain]
    stmt: Stmt = chain[-1].body
    for f in reversed(new_chain):
        stmt = replace_for(f, body=stmt)
    out = with_body(func, rebuild(path, stmt))
    check_reductions(out)
    return out


# -- loop kinds ---------------------------------------------------------------------


def _private(func: PrimFunc, loop: For, buf: object) -> bool:
    """True if every access to ``buf`` is under ``loop`` (it can live per iteration)."""
    if buf in func.params:
        return False
    under = {id(r) for r in _leaf_realizes(loop)}
    accessors = writers(func, buf) + readers(func, buf)  # type: ignore[arg-type]
    return all(id(r) in under for r in accessors)


def _leaf_realizes(stmt: Stmt) -> list[BlockRealize]:
    return [s for s in walk_stmts(stmt) if isinstance(s, BlockRealize) and is_leaf_block(s.block)]


def check_independent(func: PrimFunc, loop: Var, what: str, *, allow_reduce: bool = False) -> None:
    """Different iterations of ``loop`` write different elements (or private buffers)
    and do not take part in a reduction, so they may run concurrently. With
    ``allow_reduce`` the loop may drive a reduction (a cross-thread reduction,
    which the backend lowers to a cooperative one)."""
    lpath = find_loop(func, loop)
    lnode = loop_of(lpath)
    for bpath in block_paths(func):
        if not any(n is lnode for n in bpath):
            continue
        r = realize_of(bpath)
        if not _is_leaf(r.block):
            continue
        loops = loops_above(bpath)
        if lnode not in loops:
            continue
        roles = loop_roles(r, loops)
        role = roles[[f.loop_var for f in loops].index(loop)]
        if role == "R" and allow_reduce:
            continue
        if role in ("R", "X"):
            raise ScheduleError(
                f"cannot {what} loop {loop.name}: it is a reduction loop of block {r.block.name}"
            )
        for buf in facts(r.block).writes:
            if _private(func, lnode, buf):
                continue
            if role == "N":
                raise ScheduleError(
                    f"cannot {what} loop {loop.name}: every iteration writes the same "
                    f"elements of {buf.name} (block {r.block.name})"
                )
            if not _injective(bpath, lnode, loops, roles, r):
                raise ScheduleError(
                    f"cannot {what} loop {loop.name}: iterations may write the same "
                    f"elements of {buf.name} (block {r.block.name})"
                )


def _is_leaf(block: Block) -> bool:
    return is_leaf_block(block)


def _injective(
    bpath: Path, lnode: For, loops: list[For], roles: list[str], r: BlockRealize
) -> bool:
    """Different iterations of ``lnode`` and the spatial loops inside it write
    different elements of the block's domain."""
    start = loops.index(lnode)
    inner = [f for f, role in zip(loops[start:], roles[start:], strict=True) if role == "S"]
    if not all(isinstance(f.min, IntImm) and f.min.value == 0 for f in inner):
        return False
    an = analyzer_for(bpath)
    return (
        _flat_index_injective(an, inner, r)
        or _box_injective(an, inner, r)
        or _enumerated_injective(loops[:start], inner, r)
    )


def _flat_index_injective(an: Analyzer, inner: list[For], r: BlockRealize) -> bool:
    """The row-major position of the written point in the block's domain is an affine
    function of the loops whose coefficients form a mixed radix (each one larger
    than the span of the smaller ones), so no two iterations share a point.

    Fused and split loops pass this test even when the split is imperfect: the
    digits ``f // 24`` and ``f % 24`` of a fused index recombine into ``f``.
    """
    flat: Expr = IntImm(0)
    stride = 1
    for iv, value in reversed(list(zip(r.block.iter_vars, r.iter_values, strict=True))):
        if iv.kind != IterKind.SPATIAL:
            continue
        flat = flat + value * stride
        assert isinstance(iv.dom.extent, IntImm)
        stride *= iv.dom.extent.value
    terms, _ = an.linear(flat)
    loop_vars = {f.loop_var for f in inner}
    coefs: dict[Var, int] = {}
    for t, k in terms.items():
        if isinstance(t, Var) and t in loop_vars:
            coefs[t] = k
        elif free_vars(t) & loop_vars:
            return False
    radix = sorted((abs(coefs.get(f.loop_var, 0)), const_extent(f)) for f in inner)
    span = 0
    for coef, extent in radix:
        if extent == 1:
            continue
        if coef <= span:
            return False
        span += coef * (extent - 1)
    return True


# Concurrent iteration spaces up to this many points are checked exhaustively.
_ENUMERATION_LIMIT = 1 << 20


def _enumerated_injective(outer: list[For], inner: list[For], r: BlockRealize) -> bool:
    """Evaluate the spatial bindings at every iteration (where the predicate holds)
    and check that, for each iteration of the enclosing loops, no point repeats."""
    used = set().union(*(free_vars(v) for v in r.iter_values), free_vars(r.predicate))
    fixed = [f for f in outer if f.loop_var in used]
    loops = fixed + inner
    if not all(isinstance(f.min, IntImm) and f.min.value == 0 for f in loops):
        return False
    extents = [const_extent(f) for f in loops]
    if prod(extents) > _ENUMERATION_LIMIT:
        return False
    if used - {f.loop_var for f in loops}:
        return False
    grid = np.indices(extents).reshape(len(extents), -1)
    env = {f.loop_var: grid[k] for k, f in enumerate(loops)}
    n = grid.shape[1]
    values = [
        v
        for iv, v in zip(r.block.iter_vars, r.iter_values, strict=True)
        if iv.kind == IterKind.SPATIAL
    ]
    try:
        mask = np.broadcast_to(_evaluate(r.predicate, env), (n,)).astype(bool)
        cols = [np.broadcast_to(_evaluate(v, env), (n,)) for v in values]
    except _NotEnumerable:
        return False
    keys = np.stack([*grid[: len(fixed)], *cols])[:, mask]
    if keys.size == 0:
        return True
    return bool(np.unique(keys, axis=1).shape[1] == int(mask.sum()))


class _NotEnumerable(Exception):
    pass


def _evaluate(e: Expr, env: dict[Var, NDArray]) -> NDArray:
    """An integer or boolean index expression evaluated on arrays of loop values."""
    if isinstance(e, IntImm):
        return np.asarray(e.value)
    if isinstance(e, Var):
        if e not in env:
            raise _NotEnumerable
        return env[e]
    if isinstance(e, Binary):
        a, b = _evaluate(e.a, env), _evaluate(e.b, env)
        ops = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "floordiv": np.floor_divide,
            "floormod": np.mod,
            "min": np.minimum,
            "max": np.maximum,
        }
        if e.op not in ops:
            raise _NotEnumerable
        return np.asarray(ops[e.op](a, b))
    if isinstance(e, Compare):
        a, b = _evaluate(e.a, env), _evaluate(e.b, env)
        cmp = {"lt": np.less, "le": np.less_equal, "eq": np.equal, "ne": np.not_equal,
               "gt": np.greater, "ge": np.greater_equal}  # fmt: skip
        return np.asarray(cmp[e.op](a, b))
    if isinstance(e, Logical):
        a, b = _evaluate(e.a, env), _evaluate(e.b, env)
        return np.asarray(np.logical_and(a, b) if e.op == "and" else np.logical_or(a, b))
    if isinstance(e, Not):
        return np.asarray(np.logical_not(_evaluate(e.a, env)))
    if isinstance(e, Select):
        return np.asarray(
            np.where(
                _evaluate(e.cond, env), _evaluate(e.true_value, env), _evaluate(e.false_value, env)
            )
        )
    raise _NotEnumerable


def _box_injective(an: Analyzer, inner: list[For], r: BlockRealize) -> bool:
    """The written points fill a box with exactly as many points as iterations."""
    relaxed = {f.loop_var: Interval(0, const_extent(f) - 1) for f in inner}
    iterations = prod(const_extent(f) for f in inner)
    points = 1
    for iv, value in zip(r.block.iter_vars, r.iter_values, strict=True):
        if iv.kind != IterKind.SPATIAL:
            continue
        lo, hi = an.relax(value, relaxed)
        if lo is None or hi is None:
            return False
        ext = an.simplify(hi - lo + 1)
        if not isinstance(ext, IntImm):
            return False
        points *= ext.value
    return points == iterations


def set_kind(func: PrimFunc, loop: Var, kind: ForKind, thread_tag: str | None = None) -> PrimFunc:
    path = find_loop(func, loop)
    node = loop_of(path)
    const_extent(node)
    if node.kind == kind and node.thread_tag == thread_tag:
        return func
    if node.kind != ForKind.SERIAL:
        raise ScheduleError(f"loop {loop.name} is already {node.kind.value}")
    verb = {
        ForKind.PARALLEL: "parallelize",
        ForKind.VECTORIZED: "vectorize",
        ForKind.UNROLLED: "unroll",
        ForKind.THREAD_BINDING: "bind",
    }[kind]
    if kind == ForKind.VECTORIZED and any(isinstance(s, For) for s in walk_stmts(node.body)):
        raise ScheduleError(f"cannot vectorize loop {loop.name}: it has loops inside")
    if kind == ForKind.THREAD_BINDING:
        if thread_tag not in THREAD_TAGS:
            raise ScheduleError(f"unknown thread axis {thread_tag!r} (one of {THREAD_TAGS})")
        nested = [n for n in path[:-1] if isinstance(n, For)] + [
            s for s in walk_stmts(node.body) if isinstance(s, For)
        ]
        if any(f.thread_tag == thread_tag for f in nested):
            raise ScheduleError(
                f"another loop around or inside {loop.name} is bound to {thread_tag}"
            )
    if kind != ForKind.UNROLLED:
        cross_thread = kind == ForKind.THREAD_BINDING and str(thread_tag).startswith("thread.")
        check_independent(func, loop, verb, allow_reduce=cross_thread)
    new = replace_for(node, kind=kind, thread_tag=thread_tag)
    return with_body(func, rebuild(path, new))


def annotate_loop(func: PrimFunc, loop: Var, key: str, value: int | float | str) -> PrimFunc:
    path = find_loop(func, loop)
    node = loop_of(path)
    items = (*((k, v) for k, v in node.annotations if k != key), (key, value))
    return with_body(func, rebuild(path, replace_for(node, annotations=items)))
