"""Block primitives: compute_inline, reverse_compute_inline, compute_at, reverse_compute_at."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from tessera.tir.analysis import block_accesses, is_leaf_block
from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import Binary, BufferLoad, Compare, Expr, IntImm, Logical, Var, ge, lt
from tessera.tir.functor import StmtMutator, free_vars, substitute, walk_stmts
from tessera.tir.primitives.common import (
    Path,
    ScheduleError,
    analyzer_for,
    check_reductions,
    conjoin,
    exclusive_root,
    facts,
    find_block,
    find_loop,
    flat_seq,
    fresh,
    index_of,
    is_reduction,
    iter_paths,
    loop_of,
    loop_roles,
    loops_above,
    only,
    program_order,
    readers,
    realize_of,
    rebuild,
    refresh_regions,
    replace_block,
    replace_for,
    seq_items,
    simplify_stmt,
    used_names,
    with_body,
    writers,
)
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Stmt,
)

# -- shared checks --------------------------------------------------------------


def _leaf(path: Path, what: str) -> BlockRealize:
    r = realize_of(path)
    if not is_leaf_block(r.block):
        raise ScheduleError(f"cannot {what} block {r.block.name}: it contains other blocks")
    return r


def _single_store(block: Block, what: str) -> BufferStore:
    if not isinstance(block.body, BufferStore):
        raise ScheduleError(f"cannot {what} block {block.name}: its body is not a single store")
    return block.body


def _index_vars(
    block: Block, indices: tuple[Expr, ...], buf: Buffer, what: str, access: str
) -> list[Var | None]:
    """The spatial iteration variable behind each index of an access to ``buf``.

    Indices must be distinct spatial variables, except that a constant 0 on a
    dimension of extent 1 stands for an extent-1 variable the access does not
    otherwise use (or for none, if there is no such variable): simplified IR
    writes ``x[0, i, j]`` for a batch of one.
    """
    spatial = {iv.var for iv in block.iter_vars if iv.kind == IterKind.SPATIAL}
    used = {i for i in indices if isinstance(i, Var)}
    spare = [
        iv.var
        for iv in block.iter_vars
        if iv.kind == IterKind.SPATIAL and iv.dom.extent == IntImm(1) and iv.var not in used
    ]
    dims: list[Var | None] = []
    for d, idx in enumerate(indices):
        if isinstance(idx, Var) and idx in spatial and idx not in dims:
            dims.append(idx)
        elif idx == IntImm(0) and buf.shape[d] == 1:
            dims.append(spare.pop(0) if spare else None)
        else:
            raise ScheduleError(
                f"cannot {what} block {block.name}: index {d} of its {access} {buf.name} "
                "is not a distinct spatial iteration variable"
            )
    return dims


def _store_dims(block: Block, buf: Buffer, what: str) -> list[Var | None]:
    """``_index_vars`` of ``block``'s stores to ``buf``, which must all store to the
    same element."""
    stores = block_accesses(block).stores.get(buf, [])
    if not stores:
        raise ScheduleError(f"block {block.name} does not store to {buf.name}")
    first = stores[0]
    if any(s != first for s in stores):
        raise ScheduleError(f"cannot {what} block {block.name}: it stores to different elements")
    return _index_vars(block, first, buf, what, "store to")


def _unindexed(block: Block, dims: list[Var | None]) -> list[IterVar]:
    """Spatial iteration variables that no index uses (they must have extent 1)."""
    return [iv for iv in block.iter_vars if iv.kind == IterKind.SPATIAL and iv.var not in dims]


def _covers_buffer(block: Block, buf: Buffer, dims: list[Var | None], what: str) -> None:
    doms = {iv.var: iv.dom for iv in block.iter_vars}
    for d, v in enumerate(dims):
        if v is None:
            continue
        dom = doms[v]
        if dom.min != IntImm(0) or dom.extent != IntImm(buf.shape[d]):
            raise ScheduleError(
                f"cannot {what} block {block.name}: it does not compute all of {buf.name}"
            )
    if any(iv.dom.extent != IntImm(1) for iv in _unindexed(block, dims)):
        raise ScheduleError(
            f"cannot {what} block {block.name}: a spatial variable does not index {buf.name}"
        )


def _remove_block(func: PrimFunc, path: Path) -> PrimFunc:
    root = exclusive_root(path)
    return with_body(func, rebuild(path[: root + 1], None))


class _DropAlloc(StmtMutator):
    def __init__(self, buf: Buffer) -> None:
        self.buf = buf

    def visit_block(self, s: Block) -> Stmt:
        inner = super().visit_block(s)
        assert isinstance(inner, Block)
        if self.buf in inner.alloc_buffers:
            allocs = tuple(b for b in inner.alloc_buffers if b is not self.buf)
            return replace_block(inner, alloc_buffers=allocs)
        return inner


def _drop_alloc(func: PrimFunc, buf: Buffer) -> PrimFunc:
    return with_body(func, _DropAlloc(buf).visit_stmt(func.body))


def _finalize(func: PrimFunc) -> PrimFunc:
    body = refresh_regions(simplify_stmt(func.body, Analyzer()))
    out = with_body(func, body)
    check_reductions(out)
    return out


# -- compute_inline -----------------------------------------------------------------


def compute_inline(func: PrimFunc, name: str) -> PrimFunc:
    path = find_block(func, name)
    r = _leaf(path, "inline")
    b = r.block
    if is_reduction(b) or b.init is not None:
        raise ScheduleError(f"cannot inline block {name}: it is a reduction")
    store = _single_store(b, "inline")
    buf = store.buffer
    if buf in func.params:
        raise ScheduleError(f"cannot inline block {name}: it writes the output {buf.name}")
    dims = _store_dims(b, buf, "inline")
    _covers_buffer(b, buf, dims, "inline")
    if buf in facts(b).reads:
        raise ScheduleError(f"cannot inline block {name}: it reads its own output")
    if not only(writers(func, buf), r):
        raise ScheduleError(f"cannot inline block {name}: other blocks write {buf.name}")
    order = program_order(func)
    me = index_of(order, r)
    for c in readers(func, buf):
        if index_of(order, c) < me:
            raise ScheduleError(f"block {c.block.name} reads {buf.name} before {name} writes it")
    for inp in facts(b).reads:
        for w in writers(func, inp):
            if index_of(order, w) > me:
                raise ScheduleError(
                    f"cannot inline block {name}: block {w.block.name} writes its input "
                    f"{inp.name} later"
                )
    value = store.value

    class _Inline(StmtMutator):
        def visit_buffer_load(self, e: BufferLoad) -> Expr:
            idx = self.visit_exprs(e.indices)
            if e.buffer is buf:
                mapping = {v: i for v, i in zip(dims, idx, strict=True) if v is not None}
                return substitute(value, mapping)
            return e if idx is e.indices else BufferLoad(e.buffer, idx)

    out = _remove_block(func, path)
    out = with_body(out, _Inline().visit_stmt(out.body))
    return _finalize(_drop_alloc(out, buf))


# -- reverse_compute_inline ---------------------------------------------------------


def reverse_compute_inline(func: PrimFunc, name: str) -> PrimFunc:
    cpath = find_block(func, name)
    c = _leaf(cpath, "reverse-inline")
    cb = c.block
    if is_reduction(cb) or cb.init is not None:
        raise ScheduleError(f"cannot reverse-inline block {name}: it is a reduction")
    cstore = _single_store(cb, "reverse-inline")
    produced = [
        buf
        for buf in facts(cb).reads
        if buf not in func.params and any(w is not c for w in writers(func, buf))
    ]
    if len(produced) != 1:
        raise ScheduleError(
            f"cannot reverse-inline block {name}: it must read exactly one buffer computed "
            f"in the function (it reads {len(produced)})"
        )
    buf = produced[0]
    ws = writers(func, buf)
    if len(ws) != 1:
        raise ScheduleError(f"cannot reverse-inline block {name}: {buf.name} has several writers")
    p = ws[0]
    pb = p.block
    if not is_leaf_block(pb) or is_reduction(pb) or pb.init is not None:
        raise ScheduleError(
            f"cannot reverse-inline block {name}: its producer {pb.name} is not a "
            "complete spatial block"
        )
    if not only([x for x in readers(func, buf) if x is not p], c):
        raise ScheduleError(f"cannot reverse-inline block {name}: {buf.name} has other readers")
    pstore = _single_store(pb, "reverse-inline into")
    pdims = _store_dims(pb, buf, "reverse-inline into")
    _covers_buffer(pb, buf, pdims, "reverse-inline into")
    loads = block_accesses(cb).loads[buf]
    first = loads[0]
    if any(x != first for x in loads):
        raise ScheduleError(f"cannot reverse-inline block {name}: it reads {buf.name} unevenly")
    cdims = _index_vars(cb, first, buf, "reverse-inline", "read of")
    cdoms = {iv.var: iv.dom for iv in cb.iter_vars}
    for d, cv in enumerate(cdims):
        if cv is not None and (
            cdoms[cv].min != IntImm(0) or cdoms[cv].extent != IntImm(buf.shape[d])
        ):
            raise ScheduleError(f"cannot reverse-inline block {name}: domains do not match")
    if any(iv.dom.extent != IntImm(1) for iv in _unindexed(cb, cdims)):
        raise ScheduleError(
            f"cannot reverse-inline block {name}: it has iteration variables that do not "
            f"index {buf.name}"
        )
    order = program_order(func)
    pi, ci = index_of(order, p), index_of(order, c)
    if pi > ci:
        raise ScheduleError(f"block {name} comes before its producer")
    out_buf = cstore.buffer
    other_inputs = facts(cb).reads - {buf}
    for between in order[pi + 1 : ci]:
        f = facts(between.block)
        if f.writes & other_inputs or out_buf in f.writes or out_buf in f.reads:
            raise ScheduleError(
                f"cannot reverse-inline block {name}: block {between.block.name} runs in between"
            )
    mapping: dict[Var, Expr] = {iv.var: IntImm(0) for iv in _unindexed(cb, cdims)}
    for cv, pv in zip(cdims, pdims, strict=True):
        if cv is not None:
            mapping[cv] = IntImm(0) if pv is None else pv
    pvalue = pstore.value

    class _Replace(StmtMutator):
        def visit_buffer_load(self, e: BufferLoad) -> Expr:
            if e.buffer is buf:
                return pvalue
            return super().visit_buffer_load(e)

    new_value = substitute(_Replace().visit_expr(cstore.value), mapping)
    new_indices = tuple(substitute(i, mapping) for i in cstore.indices)
    new_store = BufferStore(out_buf, new_value, new_indices)
    out = _remove_block(func, cpath)
    ppath = find_block(out, pb.name)
    pr = realize_of(ppath)
    new_p = BlockRealize(pr.iter_values, pr.predicate, replace_block(pr.block, body=new_store))
    out = with_body(out, rebuild(ppath, new_p))
    return _finalize(_drop_alloc(out, buf))


# -- compute_at / reverse_compute_at --------------------------------------------------


@dataclass
class _Region:
    """``[min, min + extent)`` per dimension; ``min`` in the loops outside the target."""

    mins: list[Expr]
    extents: list[int]


def _relaxed_ranges(an: Analyzer, loops: list[For]) -> dict[Var, Interval]:
    out: dict[Var, Interval] = {}
    for f in loops:
        r = an.ranges.get(f.loop_var)
        out[f.loop_var] = r if r is not None else Interval.everything()
    return out


def _access_region(
    path: Path,
    target: For,
    accesses: list[tuple[Expr, ...]],
    shape: tuple[int, ...],
) -> tuple[_Region, Analyzer]:
    """The box of ``accesses`` (indices in the block's iteration variables) touched in
    one iteration of ``target``, by relaxing the loops between ``target`` and the block."""
    r = realize_of(path)
    loops = loops_above(path)
    if target not in loops:
        raise ScheduleError(f"loop {target.loop_var.name} is not around block {r.block.name}")
    inner = loops[loops.index(target) + 1 :]
    an = analyzer_for(path)
    relaxed = _relaxed_ranges(an, inner)
    binding = dict(zip((iv.var for iv in r.block.iter_vars), r.iter_values, strict=True))
    los: list[list[Expr | None]] = [[] for _ in shape]
    his: list[list[Expr | None]] = [[] for _ in shape]
    for idx in accesses:
        for d, e in enumerate(idx):
            lo, hi = an.relax(substitute(e, binding), relaxed)
            los[d].append(lo)
            his[d].append(hi)
    mins: list[Expr] = []
    extents: list[int] = []
    for d, n in enumerate(shape):
        lo_e = _combine(an, los[d], "min")
        hi_e = _combine(an, his[d], "max")
        if lo_e is None or hi_e is None:
            mins.append(IntImm(0))
            extents.append(n)
            continue
        ext = an.simplify(hi_e - lo_e + 1)
        if isinstance(ext, IntImm):
            e_val = ext.value
        else:
            bound = an.bound(ext).hi
            e_val = n if bound is None else bound
        if e_val > n:
            mins.append(IntImm(0))
            extents.append(n)
        else:
            mins.append(lo_e)
            extents.append(max(1, e_val))
    return _Region(mins, extents), an


def _combine(an: Analyzer, items: list[Expr | None], op: str) -> Expr | None:
    if any(x is None for x in items):
        return None
    exprs = [x for x in items if x is not None]
    out = exprs[0]
    for e in exprs[1:]:
        diff = an.simplify(e - out)
        if isinstance(diff, IntImm):
            if (op == "min" and diff.value < 0) or (op == "max" and diff.value > 0):
                out = e
        else:
            out = an.simplify(Binary("min" if op == "min" else "max", out, e))
    return out


def _union(a: _Region | None, b: _Region, an: Analyzer, shape: tuple[int, ...]) -> _Region:
    if a is None:
        return b
    mins: list[Expr] = []
    extents: list[int] = []
    for d, n in enumerate(shape):
        lo = _combine(an, [a.mins[d], b.mins[d]], "min")
        hi = _combine(an, [a.mins[d] + (a.extents[d] - 1), b.mins[d] + (b.extents[d] - 1)], "max")
        assert lo is not None
        assert hi is not None
        ext = an.simplify(hi - lo + 1)
        e_val = ext.value if isinstance(ext, IntImm) else (an.bound(ext).hi or n)
        if e_val > n:
            mins.append(IntImm(0))
            extents.append(n)
        else:
            mins.append(lo)
            extents.append(e_val)
    return _Region(mins, extents)


def _nest_for(
    func: PrimFunc,
    block: Block,
    dim_of: dict[Var, int],
    region: _Region,
    an: Analyzer,
) -> Stmt:
    """Loops and a BlockRealize running ``block`` over ``region`` (mapped through
    ``dim_of``: iteration variable -> buffer dimension) and over the full domain of
    its other variables, with a predicate where the region can leave the domain."""
    names = used_names(func)
    loops: list[tuple[Var, int]] = []
    values: list[Expr] = []
    pred: Expr = TRUE
    inner = an.copy()
    for k, iv in enumerate(block.iter_vars):
        assert isinstance(iv.dom.min, IntImm)
        assert isinstance(iv.dom.extent, IntImm)
        d = dim_of.get(iv.var)
        if d is None:
            lo: Expr = iv.dom.min
            ext = iv.dom.extent.value
        else:
            lo, ext = region.mins[d], region.extents[d]
        if ext == 1:
            value = lo
        else:
            v = Var(fresh(f"ax{k}", names))
            loops.append((v, ext))
            inner.bind(v, Interval(0, ext - 1))
            value = lo + v
        value = inner.simplify(value)
        values.append(value)
        if d is not None:
            dom_lo, dom_hi = iv.dom.min.value, iv.dom.min.value + iv.dom.extent.value
            if not inner.can_prove(ge(value, dom_lo)):
                pred = conjoin(pred, ge(value, dom_lo))
            if not inner.can_prove(lt(value, dom_hi)):
                pred = conjoin(pred, lt(value, dom_hi))
    stmt: Stmt = BlockRealize(tuple(values), inner.simplify(pred), block)
    for v, ext in reversed(loops):
        stmt = For(v, IntImm(0), IntImm(ext), ForKind.SERIAL, stmt)
    return stmt


def _under(path: Path, node: Stmt) -> bool:
    return any(n is node for n in path)


def _block_path(func: PrimFunc, r: BlockRealize) -> Path:
    for p in iter_paths(func.body):
        if p[-1] is r:
            return p
    raise ScheduleError(f"block {r.block.name} not found")


def compute_at(func: PrimFunc, name: str, loop: Var) -> PrimFunc:
    ppath = find_block(func, name)
    p = _leaf(ppath, "compute_at")
    pb = p.block
    if is_reduction(pb) and pb.init is None:
        raise ScheduleError(f"cannot move block {name}: its reduction init was split off")
    written = facts(pb).writes
    if len(written) != 1:
        raise ScheduleError(f"cannot move block {name}: it writes {len(written)} buffers")
    (buf,) = written
    if buf in func.params:
        raise ScheduleError(f"cannot move block {name}: it writes the output {buf.name}")
    dims = _store_dims(pb, buf, "move")
    if any(iv.dom.extent != IntImm(1) for iv in _unindexed(pb, dims)):
        raise ScheduleError(
            f"cannot move block {name}: a spatial variable does not index {buf.name}"
        )
    lpath = find_loop(func, loop)
    target = loop_of(lpath)
    if _under(ppath, target):
        raise ScheduleError(f"block {name} is already under loop {loop.name}")
    if not only(writers(func, buf), p):
        raise ScheduleError(f"cannot move block {name}: other blocks write {buf.name}")
    consumers = [c for c in readers(func, buf) if c is not p]
    if not consumers:
        raise ScheduleError(f"block {name} has no consumer")
    cpaths = [_block_path(func, c) for c in consumers]
    if not all(_under(cp, target) for cp in cpaths):
        raise ScheduleError(f"not every consumer of {buf.name} is under loop {loop.name}")
    order = program_order(func)
    me = index_of(order, p)
    under_target = {id(r) for r in _leaves(target)}
    if any(index_of(order, r) < me for r in order if id(r) in under_target):
        raise ScheduleError(f"loop {loop.name} runs before block {name}")
    for inp in facts(pb).reads:
        for w in writers(func, inp):
            if w is not p and (id(w) in under_target or index_of(order, w) > me):
                raise ScheduleError(
                    f"cannot move block {name}: block {w.block.name} writes its input {inp.name}"
                )
    region: _Region | None = None
    an: Analyzer | None = None
    for c, cp in zip(consumers, cpaths, strict=True):
        loads = block_accesses(c.block).loads[buf]
        reg, an_c = _access_region(cp, target, loads, buf.shape)
        an = an or an_c
        region = _union(region, reg, an, buf.shape)
    assert region is not None
    assert an is not None
    outer = analyzer_for(lpath)
    outer.bind(target.loop_var, an.ranges.get(target.loop_var, Interval.everything()))
    nest = _nest_for(func, pb, {v: d for d, v in enumerate(dims) if v is not None}, region, outer)
    out = _remove_block(func, ppath)
    lpath = find_loop(out, loop)
    target = loop_of(lpath)
    items = seq_items(target.body)
    pos = next(
        (i for i, s in enumerate(items) if any(buf in facts(r.block).reads for r in _leaves(s))),
        len(items),
    )
    new_target = replace_for(target, body=flat_seq([*items[:pos], nest, *items[pos:]]))
    out = with_body(out, rebuild(lpath, new_target))
    return _finalize(out)


def _leaves(stmt: Stmt) -> list[BlockRealize]:
    return [s for s in walk_stmts(stmt) if isinstance(s, BlockRealize) and is_leaf_block(s.block)]


def reverse_compute_at(func: PrimFunc, name: str, loop: Var) -> PrimFunc:
    cpath = find_block(func, name)
    c = _leaf(cpath, "reverse_compute_at")
    cb = c.block
    if is_reduction(cb):
        raise ScheduleError(f"cannot move block {name}: it is a reduction")
    lpath = find_loop(func, loop)
    target = loop_of(lpath)
    if _under(cpath, target):
        raise ScheduleError(f"block {name} is already under loop {loop.name}")
    under = _leaves(target)
    under_ids = {id(r) for r in under}
    reads = facts(cb).reads
    produced = [b for b in reads if any(b in facts(r.block).writes for r in under)]
    if len(produced) != 1:
        raise ScheduleError(
            f"cannot move block {name}: it must read exactly one buffer produced under "
            f"loop {loop.name} (it reads {len(produced)})"
        )
    buf = produced[0]
    producers = writers(func, buf)
    if any(id(w) not in under_ids for w in producers):
        raise ScheduleError(f"cannot move block {name}: {buf.name} is also written elsewhere")
    out_bufs = facts(cb).writes
    for r in under:
        f = facts(r.block)
        if (f.reads | f.writes) & out_bufs:
            raise ScheduleError(
                f"cannot move block {name}: block {r.block.name} under loop {loop.name} "
                "uses its output"
            )
    order = program_order(func)
    me = index_of(order, c)
    last_under = max(index_of(order, r) for r in under)
    if last_under > me:
        raise ScheduleError(f"loop {loop.name} runs after block {name}")
    for between in order[last_under + 1 : me]:
        f = facts(between.block)
        if (f.reads | f.writes) & out_bufs or f.writes & reads:
            raise ScheduleError(
                f"cannot move block {name}: block {between.block.name} runs in between"
            )
    for inp in reads - {buf}:
        for w in writers(func, inp):
            if index_of(order, w) > me or id(w) in under_ids:
                raise ScheduleError(f"cannot move block {name}: {inp.name} is written later")
    region: _Region | None = None
    an: Analyzer | None = None
    for w in producers:
        wpath = _block_path(func, w)
        loops = loops_above(wpath)
        roles = loop_roles(w, loops)
        ti = loops.index(target)
        if any(role in ("R", "X") for role in roles[: ti + 1]):
            raise ScheduleError(
                f"cannot move block {name}: {buf.name} is still being reduced across "
                f"iterations of loop {loop.name}"
            )
        wdims = _store_dims(w.block, buf, "move after")
        stores: list[tuple[Expr, ...]] = [tuple(IntImm(0) if v is None else v for v in wdims)]
        reg, an_w = _access_region(wpath, target, stores, buf.shape)
        iterations = prod(
            _extent(f)
            for f, role in zip(loops[ti + 1 :], roles[ti + 1 :], strict=True)
            if role == "S"
        )
        exact = prod(reg.extents) == iterations and _clips_to_buffer(w, stores[0], buf, an_w)
        if not exact and not _writes_box(w, loops, ti, stores[0], reg, buf):
            raise ScheduleError(
                f"cannot move block {name}: what block {w.block.name} writes in one "
                f"iteration of loop {loop.name} is not a box"
            )
        an = an or an_w
        region = reg if region is None else _union(region, reg, an, buf.shape)
    assert region is not None
    assert an is not None
    loads = block_accesses(cb).loads[buf]
    first = loads[0]
    if any(x != first for x in loads):
        raise ScheduleError(f"cannot move block {name}: it reads {buf.name} unevenly")
    dim_of = {
        v: d for d, v in enumerate(_index_vars(cb, first, buf, "move", "read of")) if v is not None
    }
    outer = analyzer_for(lpath)
    outer.bind(target.loop_var, an.ranges.get(target.loop_var, Interval.everything()))
    nest = _nest_for(func, cb, dim_of, region, outer)
    out = _remove_block(func, cpath)
    lpath = find_loop(out, loop)
    target = loop_of(lpath)
    new_target = replace_for(target, body=flat_seq([*seq_items(target.body), nest]))
    out = with_body(out, rebuild(lpath, new_target))
    return _finalize(out)


def _clips_to_buffer(w: BlockRealize, store: tuple[Expr, ...], buf: Buffer, an: Analyzer) -> bool:
    """Every conjunct of ``w``'s predicate says only that one index of its store to
    ``buf`` is inside the buffer (``i0 * 4 + i1 < 50257`` for an index
    ``i0 * 4 + i1``), or only limits reduction loops: the predicate cuts a tile
    exactly at the buffer's edge."""
    binding = dict(zip((iv.var for iv in w.block.iter_vars), w.iter_values, strict=True))
    points = [an.simplify(substitute(e, binding)) for e in store]
    reduce_loops: set[Var] = set().union(
        *(
            free_vars(v)
            for iv, v in zip(w.block.iter_vars, w.iter_values, strict=True)
            if iv.kind == IterKind.REDUCE
        )
    )
    stack = [w.predicate]
    while stack:
        c = stack.pop()
        if c == TRUE:
            continue
        if free_vars(c) <= reduce_loops:
            continue  # cuts a reduction tile; every point is still written
        if isinstance(c, Logical) and c.op == "and":
            stack.extend((c.a, c.b))
            continue
        if not (isinstance(c, Compare) and c.op in ("lt", "ge") and isinstance(c.b, IntImm)):
            return False
        ok = False
        for d, p in enumerate(points):
            delta = an.simplify(c.a - p)
            if isinstance(delta, IntImm):
                edge = buf.shape[d] if c.op == "lt" else 0
                ok = ok or c.b.value - delta.value == edge
        if not ok:
            return False
    return True


def _writes_box(
    w: BlockRealize,
    loops: list[For],
    ti: int,
    store: tuple[Expr, ...],
    reg: _Region,
    buf: Buffer,
) -> bool:
    """In every iteration of ``loops[ti]``, block ``w`` writes exactly the points of
    ``reg`` that lie in ``buf`` (its predicate cuts partial tiles), checked by
    evaluating every iteration when there are few enough."""
    from tessera.tir.primitives.loops import _ENUMERATION_LIMIT, _evaluate, _NotEnumerable

    binding = dict(zip((iv.var for iv in w.block.iter_vars), w.iter_values, strict=True))
    points = [substitute(e, binding) for e in store]
    used: set[Var] = set().union(
        free_vars(w.predicate), *(free_vars(p) for p in points), *(free_vars(m) for m in reg.mins)
    )
    group = [f for f in loops[: ti + 1] if f.loop_var in used or f is loops[ti]]
    inner = [f for f in loops[ti + 1 :] if f.loop_var in used]
    run = group + inner
    if used - {f.loop_var for f in run}:
        return False
    if not all(isinstance(f.min, IntImm) and f.min.value == 0 for f in run):
        return False
    extents = [_extent(f) for f in run]
    if prod(extents) > _ENUMERATION_LIMIT:
        return False
    grid = np.indices(extents).reshape(len(extents), -1)
    env = {f.loop_var: grid[k] for k, f in enumerate(run)}
    n = grid.shape[1]
    try:
        mask = np.broadcast_to(_evaluate(w.predicate, env), (n,)).astype(bool)
        pts = np.stack([np.broadcast_to(_evaluate(p, env), (n,)) for p in points])
        mins = [np.broadcast_to(_evaluate(m, env), (n,)) for m in reg.mins]
    except _NotEnumerable:
        return False
    volume = np.ones(n, dtype=np.int64)
    for lo, ext, size in zip(mins, reg.extents, buf.shape, strict=True):
        volume *= np.maximum(0, np.minimum(lo + ext, size) - np.maximum(lo, 0))
    gid = np.asarray(np.ravel_multi_index(tuple(grid[: len(group)]), extents[: len(group)]))
    groups = int(np.prod(extents[: len(group)]))
    expected = np.zeros(groups, dtype=np.int64)
    expected[gid] = volume
    try:
        flat = (
            np.ravel_multi_index(pts[:, mask], buf.shape)
            if buf.shape
            else np.zeros(int(mask.sum()), dtype=np.int64)
        )
    except ValueError:
        return False
    keys = np.unique(gid[mask] * max(1, buf.numel) + flat)
    produced = np.bincount(keys // max(1, buf.numel), minlength=groups)
    return bool(np.array_equal(produced, expected))


def _extent(f: For) -> int:
    if not isinstance(f.extent, IntImm):
        raise ScheduleError(f"loop {f.loop_var.name} has no constant extent")
    return f.extent.value


def annotate_block(func: PrimFunc, name: str, key: str, value: int | float | str) -> PrimFunc:
    path = find_block(func, name)
    r = realize_of(path)
    items = (*((k, v) for k, v in r.block.annotations if k != key), (key, value))
    block = replace_block(r.block, annotations=items)
    return with_body(func, rebuild(path, BlockRealize(r.iter_values, r.predicate, block)))
