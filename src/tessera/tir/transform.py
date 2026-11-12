"""Lowering passes run after scheduling: opaque blocks and compact buffers.

``lower_blocks`` removes what makes blocks special: iteration variables are
replaced by their bindings, a reduction's init becomes an ``if`` testing
that the reduction variables are at their start, and predicates become
``if``s. The result has blocks with no iteration variables (scopes for
allocations and names for code), which a code generator can print as plain
loops.

``compact_buffers`` then gives every local and shared buffer the size it
needs: it is allocated at the innermost loop containing all its accesses
(shared buffers stay outside ``thread.*`` loops, since the threads of a
threadgroup share them), its shape is the region touched in one iteration
of that loop, and indices are shifted to the start of that region.
"""

from __future__ import annotations

from dataclasses import dataclass

from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import AddressOf, BufferLoad, Expr, IntImm, Var, eq
from tessera.tir.functor import StmtMutator, substitute, substitute_stmt, walk_exprs
from tessera.tir.primitives.common import (
    Path,
    analyzer_for,
    conjoin,
    fresh,
    iter_paths,
    rebuild,
    replace_block,
    replace_for,
    simplify_stmt,
    with_body,
)
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    Evaluate,
    For,
    IfThenElse,
    IterKind,
    LetStmt,
    PrimFunc,
    Range,
    Stmt,
    seq,
)


class _Opaque(StmtMutator):
    def visit_block_realize(self, s: BlockRealize) -> Stmt:
        inner = super().visit_block_realize(s)
        assert isinstance(inner, BlockRealize)
        b = inner.block
        if not b.iter_vars and inner.predicate == TRUE and b.init is None:
            return inner
        mapping: dict[Var, Expr] = {
            iv.var: v for iv, v in zip(b.iter_vars, inner.iter_values, strict=True)
        }
        body = substitute_stmt(b.body, mapping)
        if b.init is not None:
            cond: Expr = TRUE
            for iv in b.iter_vars:
                if iv.kind == IterKind.REDUCE:
                    cond = conjoin(cond, eq(mapping[iv.var], iv.dom.min))
            body = seq([IfThenElse(cond, substitute_stmt(b.init, mapping)), body])
        if inner.predicate != TRUE:
            body = IfThenElse(inner.predicate, body)
        regions = [
            BufferRegion(
                r.buffer,
                tuple(Range(substitute(x.min, mapping), x.extent) for x in r.region),
            )
            for r in b.reads + b.writes
        ]
        block = replace_block(
            b,
            iter_vars=(),
            reads=tuple(regions[: len(b.reads)]),
            writes=tuple(regions[len(b.reads) :]),
            body=body,
            init=None,
        )
        return BlockRealize((), TRUE, block)


def lower_blocks(func: PrimFunc) -> PrimFunc:
    """``func`` with every block opaque: bindings substituted, inits and predicates
    turned into ``if`` statements. Computes the same result."""
    body = simplify_stmt(_Opaque().visit_stmt(func.body), Analyzer())
    return with_body(func, body)


@dataclass(frozen=True)
class Footprint:
    """Where a buffer is needed: allocate ``extents`` under ``loop`` (``None``: where it
    is allocated now) and access element ``i`` at ``i - mins``."""

    buffer: Buffer
    loop: Var | None
    mins: tuple[Expr, ...]
    extents: tuple[int, ...]


def _accesses(stmt: Stmt, buf: Buffer) -> list[tuple[Path, tuple[Expr, ...]]]:
    out: list[tuple[Path, tuple[Expr, ...]]] = []
    for path in iter_paths(stmt):
        node = path[-1]
        exprs: list[Expr] = []
        if isinstance(node, BufferStore):
            if node.buffer is buf:
                out.append((path, node.indices))
            exprs = [node.value, *node.indices]
        elif isinstance(node, IfThenElse):
            exprs = [node.condition]
        elif isinstance(node, For):
            exprs = [node.min]
        elif isinstance(node, Evaluate | LetStmt):
            exprs = [node.value]
        for e in exprs:
            for sub in walk_exprs(e):
                if isinstance(sub, BufferLoad | AddressOf) and sub.buffer is buf:
                    out.append((path, sub.indices))
    return out


def footprints(func: PrimFunc) -> list[Footprint]:
    """The allocation point and touched region of every local and shared buffer of a
    function whose blocks are opaque (see ``lower_blocks``). Buffers passed to an
    intrinsic by address are left out: their strides are part of the call."""
    out: list[Footprint] = []
    escaped = {e.buffer for e in walk_exprs(func.body) if isinstance(e, AddressOf)}
    for path in iter_paths(func.body):
        node = path[-1]
        if not isinstance(node, Block):
            continue
        for buf in node.alloc_buffers:
            # Intrinsic calls take pointers and strides fixed at tensorize time.
            if buf.scope == "global" or buf in escaped:
                continue
            fp = _footprint(path, buf)
            if fp is not None:
                out.append(fp)
    return out


def _footprint(scope_path: Path, buf: Buffer) -> Footprint | None:
    scope = scope_path[-1]
    accesses = _accesses(scope, buf)
    if not accesses:
        return None
    paths = [scope_path[:-1] + p for p, _ in accesses]
    common = list(paths[0])
    for p in paths[1:]:
        n = 0
        while n < len(common) and n < len(p) and common[n] is p[n]:
            n += 1
        common = common[:n]
    start = len(scope_path)
    loop: For | None = None
    for node in common[start:]:
        if isinstance(node, Block) and node.iter_vars:
            break
        if isinstance(node, For):
            if buf.scope == "shared" and (node.thread_tag or "").startswith("thread."):
                break
            loop = node
    mins: list[Expr] = []
    extents: list[int] = []
    los: list[list[Expr | None]] = [[] for _ in buf.shape]
    his: list[list[Expr | None]] = [[] for _ in buf.shape]
    an_outer: Analyzer | None = None
    for p, idx in zip(paths, (i for _, i in accesses), strict=True):
        an = analyzer_for(p)
        an_outer = an_outer or an
        below = False
        relaxed: dict[Var, Interval] = {}
        for node in p[start - 1 :]:
            if loop is None and node is scope:
                below = True
            if isinstance(node, For):
                if below:
                    relaxed[node.loop_var] = an.ranges.get(node.loop_var, Interval.everything())
                if node is loop:
                    below = True
        for d, e in enumerate(idx):
            lo, hi = an.relax(e, relaxed)
            los[d].append(lo)
            his[d].append(hi)
    assert an_outer is not None
    for d, n in enumerate(buf.shape):
        lo_e = _extreme(an_outer, los[d], "min")
        hi_e = _extreme(an_outer, his[d], "max")
        if lo_e is None or hi_e is None:
            mins.append(IntImm(0))
            extents.append(n)
            continue
        ext = an_outer.simplify(hi_e - lo_e + 1)
        bound = ext.value if isinstance(ext, IntImm) else an_outer.bound(ext).hi
        if bound is None or bound >= n:
            mins.append(IntImm(0))
            extents.append(n)
        else:
            mins.append(lo_e)
            extents.append(max(1, bound))
    return Footprint(buf, None if loop is None else loop.loop_var, tuple(mins), tuple(extents))


def _extreme(an: Analyzer, items: list[Expr | None], op: str) -> Expr | None:
    if any(x is None for x in items):
        return None
    exprs = [x for x in items if x is not None]
    best = exprs[0]
    for e in exprs[1:]:
        diff = an.simplify(e - best)
        if not isinstance(diff, IntImm):
            return None
        if (op == "min" and diff.value < 0) or (op == "max" and diff.value > 0):
            best = e
    return best


class _Shift(StmtMutator):
    def __init__(self, old: Buffer, new: Buffer, mins: tuple[Expr, ...]) -> None:
        self.old, self.new, self.mins = old, new, mins

    def _idx(self, idx: tuple[Expr, ...]) -> tuple[Expr, ...]:
        return tuple(i - m if m != IntImm(0) else i for i, m in zip(idx, self.mins, strict=True))

    def visit_buffer_load(self, e: BufferLoad) -> Expr:
        idx = self.visit_exprs(e.indices)
        if e.buffer is self.old:
            return BufferLoad(self.new, self._idx(idx))
        return e if idx is e.indices else BufferLoad(e.buffer, idx)

    def visit_address_of(self, e: AddressOf) -> Expr:
        idx = self.visit_exprs(e.indices)
        if e.buffer is self.old:
            return AddressOf(self.new, self._idx(idx))
        return e if idx is e.indices else AddressOf(e.buffer, idx)

    def visit_buffer_store(self, s: BufferStore) -> Stmt:
        value = self.visit_expr(s.value)
        idx = self.visit_exprs(s.indices)
        if s.buffer is self.old:
            return BufferStore(self.new, value, self._idx(idx))
        return BufferStore(s.buffer, value, idx)

    def visit_buffer(self, buffer: Buffer) -> Buffer:
        return self.new if buffer is self.old else buffer

    def visit_region(self, region: BufferRegion) -> BufferRegion:
        if region.buffer is not self.old:
            return super().visit_region(region)
        return BufferRegion(self.new, tuple(Range.of(n) for n in self.new.shape))


def compact_buffers(func: PrimFunc) -> PrimFunc:
    """``func`` lowered with ``lower_blocks``, with every local and shared buffer
    allocated where it is needed and shrunk to the region it touches there."""
    func = lower_blocks(func)
    names = {p[-1].block.name for p in iter_paths(func.body) if isinstance(p[-1], BlockRealize)}
    for fp in footprints(func):
        old = fp.buffer
        new = Buffer(old.name, fp.extents, old.dtype, old.scope)
        if new.shape == old.shape and fp.loop is None:
            continue
        func = with_body(func, _Shift(old, new, fp.mins).visit_stmt(func.body))
        if fp.loop is None:
            continue
        func = with_body(func, _DropAlloc(new).visit_stmt(func.body))
        for path in iter_paths(func.body):
            node = path[-1]
            if isinstance(node, For) and node.loop_var is fp.loop:
                scope = Block(
                    fresh(f"{new.name}_scope", names), (), (), (), node.body, None, (new,)
                )
                new_loop = replace_for(node, body=BlockRealize((), TRUE, scope))
                func = with_body(func, rebuild(path, new_loop))
                break
    return with_body(func, simplify_stmt(func.body, Analyzer()))


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
