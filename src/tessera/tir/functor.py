"""Visitors and mutators over loop-IR trees, and variable/buffer substitution.

``StmtVisitor`` walks every statement and expression of a tree;
``StmtMutator`` rebuilds a tree bottom-up and returns the original node
wherever nothing below it changed, so unchanged subtrees keep their identity
(the schedule finds loops and blocks by identity). Subclasses override the
``visit_<node>`` methods they care about and call the base method to recurse.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from tessera.tir.expr import (
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    Expr,
    FloatImm,
    IntImm,
    Logical,
    Not,
    Select,
    Var,
)
from tessera.tir.stmt import (
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    Evaluate,
    For,
    IfThenElse,
    IterVar,
    LetStmt,
    Range,
    SeqStmt,
    Stmt,
)


class ExprVisitor:
    """Walks an expression tree; override ``visit_<node>`` and call ``super()`` to recurse."""

    def visit_expr(self, e: Expr) -> None:
        getattr(self, _EXPR_METHOD[type(e)])(e)

    def visit_buffer(self, buffer: Buffer) -> None:
        """Called for every buffer reference (loads, stores, regions, allocations)."""

    def visit_int_imm(self, e: IntImm) -> None:
        pass

    def visit_float_imm(self, e: FloatImm) -> None:
        pass

    def visit_var(self, e: Var) -> None:
        pass

    def visit_binary(self, e: Binary) -> None:
        self.visit_expr(e.a)
        self.visit_expr(e.b)

    def visit_compare(self, e: Compare) -> None:
        self.visit_expr(e.a)
        self.visit_expr(e.b)

    def visit_logical(self, e: Logical) -> None:
        self.visit_expr(e.a)
        self.visit_expr(e.b)

    def visit_not(self, e: Not) -> None:
        self.visit_expr(e.a)

    def visit_select(self, e: Select) -> None:
        self.visit_expr(e.cond)
        self.visit_expr(e.true_value)
        self.visit_expr(e.false_value)

    def visit_cast(self, e: Cast) -> None:
        self.visit_expr(e.value)

    def visit_call(self, e: Call) -> None:
        for a in e.args:
            self.visit_expr(a)

    def visit_buffer_load(self, e: BufferLoad) -> None:
        self.visit_buffer(e.buffer)
        for i in e.indices:
            self.visit_expr(i)

    def visit_address_of(self, e: AddressOf) -> None:
        self.visit_buffer(e.buffer)
        for i in e.indices:
            self.visit_expr(i)


class StmtVisitor(ExprVisitor):
    """Walks statements and every expression inside them."""

    def visit_stmt(self, s: Stmt) -> None:
        getattr(self, _STMT_METHOD[type(s)])(s)

    def visit_range(self, r: Range) -> None:
        self.visit_expr(r.min)
        self.visit_expr(r.extent)

    def visit_region(self, region: BufferRegion) -> None:
        self.visit_buffer(region.buffer)
        for r in region.region:
            self.visit_range(r)

    def visit_for(self, s: For) -> None:
        self.visit_expr(s.min)
        self.visit_expr(s.extent)
        self.visit_stmt(s.body)

    def visit_block_realize(self, s: BlockRealize) -> None:
        for v in s.iter_values:
            self.visit_expr(v)
        self.visit_expr(s.predicate)
        self.visit_block(s.block)

    def visit_block(self, s: Block) -> None:
        for iv in s.iter_vars:
            self.visit_range(iv.dom)
        for b in s.alloc_buffers:
            self.visit_buffer(b)
        for region in s.reads:
            self.visit_region(region)
        for region in s.writes:
            self.visit_region(region)
        if s.init is not None:
            self.visit_stmt(s.init)
        self.visit_stmt(s.body)

    def visit_buffer_store(self, s: BufferStore) -> None:
        self.visit_buffer(s.buffer)
        self.visit_expr(s.value)
        for i in s.indices:
            self.visit_expr(i)

    def visit_seq_stmt(self, s: SeqStmt) -> None:
        for c in s.seq:
            self.visit_stmt(c)

    def visit_if_then_else(self, s: IfThenElse) -> None:
        self.visit_expr(s.condition)
        self.visit_stmt(s.then_case)
        if s.else_case is not None:
            self.visit_stmt(s.else_case)

    def visit_let_stmt(self, s: LetStmt) -> None:
        self.visit_expr(s.value)
        self.visit_stmt(s.body)

    def visit_evaluate(self, s: Evaluate) -> None:
        self.visit_expr(s.value)


class ExprMutator:
    """Rebuilds expressions; returns the same node when no child changed."""

    def visit_expr(self, e: Expr) -> Expr:
        result: Expr = getattr(self, _EXPR_METHOD[type(e)])(e)
        return result

    def visit_buffer(self, buffer: Buffer) -> Buffer:
        """Maps every buffer reference; the default keeps it."""
        return buffer

    def visit_int_imm(self, e: IntImm) -> Expr:
        return e

    def visit_float_imm(self, e: FloatImm) -> Expr:
        return e

    def visit_var(self, e: Var) -> Expr:
        return e

    def visit_binary(self, e: Binary) -> Expr:
        a, b = self.visit_expr(e.a), self.visit_expr(e.b)
        return e if a is e.a and b is e.b else Binary(e.op, a, b)

    def visit_compare(self, e: Compare) -> Expr:
        a, b = self.visit_expr(e.a), self.visit_expr(e.b)
        return e if a is e.a and b is e.b else Compare(e.op, a, b)

    def visit_logical(self, e: Logical) -> Expr:
        a, b = self.visit_expr(e.a), self.visit_expr(e.b)
        return e if a is e.a and b is e.b else Logical(e.op, a, b)

    def visit_not(self, e: Not) -> Expr:
        a = self.visit_expr(e.a)
        return e if a is e.a else Not(a)

    def visit_select(self, e: Select) -> Expr:
        c = self.visit_expr(e.cond)
        t = self.visit_expr(e.true_value)
        f = self.visit_expr(e.false_value)
        if c is e.cond and t is e.true_value and f is e.false_value:
            return e
        return Select(c, t, f)

    def visit_cast(self, e: Cast) -> Expr:
        v = self.visit_expr(e.value)
        return e if v is e.value else Cast(e.type, v)

    def visit_call(self, e: Call) -> Expr:
        args = self.visit_exprs(e.args)
        return e if args is e.args else Call(e.name, args, e.type)

    def visit_buffer_load(self, e: BufferLoad) -> Expr:
        buf = self.visit_buffer(e.buffer)
        idx = self.visit_exprs(e.indices)
        return e if buf is e.buffer and idx is e.indices else BufferLoad(buf, idx)

    def visit_address_of(self, e: AddressOf) -> Expr:
        buf = self.visit_buffer(e.buffer)
        idx = self.visit_exprs(e.indices)
        return e if buf is e.buffer and idx is e.indices else AddressOf(buf, idx)

    def visit_exprs(self, exprs: tuple[Expr, ...]) -> tuple[Expr, ...]:
        """Maps a tuple of expressions; returns the same tuple when none changed."""
        new = tuple(self.visit_expr(x) for x in exprs)
        return exprs if all(a is b for a, b in zip(new, exprs, strict=True)) else new


class StmtMutator(ExprMutator):
    """Rebuilds statements; returns the same node when nothing below it changed."""

    def visit_stmt(self, s: Stmt) -> Stmt:
        result: Stmt = getattr(self, _STMT_METHOD[type(s)])(s)
        return result

    def visit_range(self, r: Range) -> Range:
        lo, ext = self.visit_expr(r.min), self.visit_expr(r.extent)
        return r if lo is r.min and ext is r.extent else Range(lo, ext)

    def visit_region(self, region: BufferRegion) -> BufferRegion:
        buf = self.visit_buffer(region.buffer)
        ranges = tuple(self.visit_range(r) for r in region.region)
        if buf is region.buffer and all(a is b for a, b in zip(ranges, region.region, strict=True)):
            return region
        return BufferRegion(buf, ranges)

    def visit_iter_var(self, iv: IterVar) -> IterVar:
        dom = self.visit_range(iv.dom)
        return iv if dom is iv.dom else IterVar(iv.var, dom, iv.kind)

    def visit_for(self, s: For) -> Stmt:
        lo, ext = self.visit_expr(s.min), self.visit_expr(s.extent)
        body = self.visit_stmt(s.body)
        if lo is s.min and ext is s.extent and body is s.body:
            return s
        return For(s.loop_var, lo, ext, s.kind, body, s.thread_tag, s.annotations)

    def visit_block_realize(self, s: BlockRealize) -> Stmt:
        values = self.visit_exprs(s.iter_values)
        pred = self.visit_expr(s.predicate)
        block = self.visit_stmt(s.block)
        assert isinstance(block, Block)
        if values is s.iter_values and pred is s.predicate and block is s.block:
            return s
        return BlockRealize(values, pred, block)

    def visit_block(self, s: Block) -> Stmt:
        iter_vars = tuple(self.visit_iter_var(iv) for iv in s.iter_vars)
        reads = tuple(self.visit_region(r) for r in s.reads)
        writes = tuple(self.visit_region(r) for r in s.writes)
        allocs = tuple(self.visit_buffer(b) for b in s.alloc_buffers)
        init = None if s.init is None else self.visit_stmt(s.init)
        body = self.visit_stmt(s.body)
        if (
            _same(iter_vars, s.iter_vars)
            and _same(reads, s.reads)
            and _same(writes, s.writes)
            and _same(allocs, s.alloc_buffers)
            and init is s.init
            and body is s.body
        ):
            return s
        return Block(s.name, iter_vars, reads, writes, body, init, allocs, s.annotations)

    def visit_buffer_store(self, s: BufferStore) -> Stmt:
        buf = self.visit_buffer(s.buffer)
        value = self.visit_expr(s.value)
        idx = self.visit_exprs(s.indices)
        if buf is s.buffer and value is s.value and idx is s.indices:
            return s
        return BufferStore(buf, value, idx)

    def visit_seq_stmt(self, s: SeqStmt) -> Stmt:
        items = tuple(self.visit_stmt(c) for c in s.seq)
        return s if _same(items, s.seq) else SeqStmt(items)

    def visit_if_then_else(self, s: IfThenElse) -> Stmt:
        cond = self.visit_expr(s.condition)
        then = self.visit_stmt(s.then_case)
        other = None if s.else_case is None else self.visit_stmt(s.else_case)
        if cond is s.condition and then is s.then_case and other is s.else_case:
            return s
        return IfThenElse(cond, then, other)

    def visit_let_stmt(self, s: LetStmt) -> Stmt:
        value = self.visit_expr(s.value)
        body = self.visit_stmt(s.body)
        return s if value is s.value and body is s.body else LetStmt(s.var, value, body)

    def visit_evaluate(self, s: Evaluate) -> Stmt:
        value = self.visit_expr(s.value)
        return s if value is s.value else Evaluate(value)


def _same[T](a: tuple[T, ...], b: tuple[T, ...]) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


_EXPR_METHOD: dict[type, str] = {
    IntImm: "visit_int_imm",
    FloatImm: "visit_float_imm",
    Var: "visit_var",
    Binary: "visit_binary",
    Compare: "visit_compare",
    Logical: "visit_logical",
    Not: "visit_not",
    Select: "visit_select",
    Cast: "visit_cast",
    Call: "visit_call",
    BufferLoad: "visit_buffer_load",
    AddressOf: "visit_address_of",
}

_STMT_METHOD: dict[type, str] = {
    For: "visit_for",
    BlockRealize: "visit_block_realize",
    Block: "visit_block",
    BufferStore: "visit_buffer_store",
    SeqStmt: "visit_seq_stmt",
    IfThenElse: "visit_if_then_else",
    LetStmt: "visit_let_stmt",
    Evaluate: "visit_evaluate",
}


class Substitute(StmtMutator):
    """Replaces variables by expressions and buffers by buffers."""

    def __init__(
        self,
        var_map: Mapping[Var, Expr] | None = None,
        buffer_map: Mapping[Buffer, Buffer] | None = None,
    ) -> None:
        self.var_map = var_map or {}
        self.buffer_map = buffer_map or {}

    def visit_var(self, e: Var) -> Expr:
        return self.var_map.get(e, e)

    def visit_buffer(self, buffer: Buffer) -> Buffer:
        return self.buffer_map.get(buffer, buffer)


def substitute(
    node: Expr,
    var_map: Mapping[Var, Expr] | None = None,
    buffer_map: Mapping[Buffer, Buffer] | None = None,
) -> Expr:
    """``node`` with every variable in ``var_map`` and buffer in ``buffer_map`` replaced."""
    if not var_map and not buffer_map:
        return node
    return Substitute(var_map, buffer_map).visit_expr(node)


def substitute_stmt(
    node: Stmt,
    var_map: Mapping[Var, Expr] | None = None,
    buffer_map: Mapping[Buffer, Buffer] | None = None,
) -> Stmt:
    """``substitute`` for statements (regions, bindings and bodies included)."""
    if not var_map and not buffer_map:
        return node
    return Substitute(var_map, buffer_map).visit_stmt(node)


class _ExprCollector(StmtVisitor):
    def __init__(self, fn: Callable[[Expr], None]) -> None:
        self.fn = fn

    def visit_expr(self, e: Expr) -> None:
        self.fn(e)
        super().visit_expr(e)


def walk_exprs(node: Expr | Stmt) -> Iterator[Expr]:
    """Every expression node in ``node``, parents before children."""
    found: list[Expr] = []
    collector = _ExprCollector(found.append)
    if isinstance(node, Expr):
        collector.visit_expr(node)
    else:
        collector.visit_stmt(node)
    return iter(found)


def walk_stmts(node: Stmt) -> Iterator[Stmt]:
    """Every statement node in ``node`` (blocks included), parents before children."""
    stack = [node]
    while stack:
        s = stack.pop()
        yield s
        stack.extend(reversed(children(s)))


def children(s: Stmt) -> list[Stmt]:
    """The direct child statements of ``s`` in program order (a block's init first)."""
    if isinstance(s, For | LetStmt):
        return [s.body]
    if isinstance(s, BlockRealize):
        return [s.block]
    if isinstance(s, Block):
        return [s.body] if s.init is None else [s.init, s.body]
    if isinstance(s, SeqStmt):
        return list(s.seq)
    if isinstance(s, IfThenElse):
        return [s.then_case] if s.else_case is None else [s.then_case, s.else_case]
    return []


def free_vars(node: Expr | Stmt) -> set[Var]:
    """Variables used in ``node`` (including ones it defines)."""
    return {e for e in walk_exprs(node) if isinstance(e, Var)}


def uses_var(node: Expr | Stmt, var: Var) -> bool:
    return any(e is var for e in walk_exprs(node))


def uses_any(node: Expr | Stmt, vars: set[Var] | frozenset[Var]) -> bool:
    return any(isinstance(e, Var) and e in vars for e in walk_exprs(node))
