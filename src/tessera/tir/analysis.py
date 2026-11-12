"""Analyses of loop-IR trees: buffer accesses, block regions, structural identity."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from tessera.tir.arith import Analyzer
from tessera.tir.expr import AddressOf, BufferLoad, Expr, IntImm, Var
from tessera.tir.functor import StmtVisitor, children
from tessera.tir.printer import canonical_text
from tessera.tir.stmt import (
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    IterKind,
    PrimFunc,
    Range,
    Stmt,
)


@dataclass
class Accesses:
    """The loads and stores of a statement, by buffer, in program order."""

    loads: dict[Buffer, list[tuple[Expr, ...]]] = field(default_factory=dict)
    stores: dict[Buffer, list[tuple[Expr, ...]]] = field(default_factory=dict)
    addressed: dict[Buffer, list[tuple[Expr, ...]]] = field(default_factory=dict)

    def buffers(self) -> list[Buffer]:
        seen: list[Buffer] = []
        for group in (self.loads, self.stores, self.addressed):
            for b in group:
                if b not in seen:
                    seen.append(b)
        return seen


class _AccessCollector(StmtVisitor):
    def __init__(self, into_blocks: bool) -> None:
        self.out = Accesses()
        self.into_blocks = into_blocks

    def visit_block(self, s: Block) -> None:
        if self.into_blocks:
            super().visit_block(s)

    def visit_block_realize(self, s: BlockRealize) -> None:
        if self.into_blocks:
            super().visit_block_realize(s)

    def visit_buffer_load(self, e: BufferLoad) -> None:
        self.out.loads.setdefault(e.buffer, []).append(e.indices)
        super().visit_buffer_load(e)

    def visit_address_of(self, e: AddressOf) -> None:
        self.out.addressed.setdefault(e.buffer, []).append(e.indices)
        super().visit_address_of(e)

    def visit_buffer_store(self, s: BufferStore) -> None:
        self.out.stores.setdefault(s.buffer, []).append(s.indices)
        super().visit_buffer_store(s)


def block_accesses(block: Block) -> Accesses:
    """Loads and stores in a block's body and init (not inside nested blocks)."""
    c = _AccessCollector(into_blocks=False)
    if block.init is not None:
        c.visit_stmt(block.init)
    c.visit_stmt(block.body)
    return c.out


def stmt_accesses(stmt: Stmt | Expr) -> Accesses:
    """Every load and store in ``stmt``, nested blocks included."""
    c = _AccessCollector(into_blocks=True)
    if isinstance(stmt, Expr):
        c.visit_expr(stmt)
    else:
        c.visit_stmt(stmt)
    return c.out


def is_leaf_block(block: Block) -> bool:
    """A block with no blocks inside it (a computation, not a scope)."""
    return not any(isinstance(s, BlockRealize) for s in _walk_no_root(block))


def _walk_no_root(block: Block) -> Iterator[Stmt]:
    stack = list(children(block))
    while stack:
        s = stack.pop()
        yield s
        stack.extend(children(s))


def region_cover(
    accesses: Sequence[tuple[Expr, ...]], shape: tuple[int, ...], analyzer: Analyzer
) -> tuple[Range, ...]:
    """The smallest box (per dimension) containing every index tuple in ``accesses``.

    Accesses that differ by constants give an exact symbolic range (``vi:vi + 3``
    for ``vi - 1``, ``vi``, ``vi + 1``); otherwise the range is relaxed over the
    analyzer's variable ranges, and clipped to the buffer.
    """
    ranges: list[Range] = []
    for d, extent in enumerate(shape):
        exprs = [analyzer.simplify(idx[d]) for idx in accesses]
        first = exprs[0]
        offsets: list[int] | None = []
        for e in exprs:
            diff = analyzer.simplify(e - first)
            if isinstance(diff, IntImm) and offsets is not None:
                offsets.append(diff.value)
            else:
                offsets = None
        if offsets is not None:
            lo = analyzer.simplify(first + min(offsets))
            ranges.append(Range(lo, IntImm(max(offsets) - min(offsets) + 1)))
            continue
        b = analyzer.bound(exprs[0])
        for e in exprs[1:]:
            b = b.union(analyzer.bound(e))
        lo_v = 0 if b.lo is None else max(0, b.lo)
        hi_v = extent - 1 if b.hi is None else min(extent - 1, b.hi)
        ranges.append(Range(IntImm(lo_v), IntImm(max(1, hi_v - lo_v + 1))))
    return tuple(ranges)


def infer_regions(block: Block) -> tuple[tuple[BufferRegion, ...], tuple[BufferRegion, ...]]:
    """The read and write regions of a leaf block, from its body.

    Regions are in terms of the block's iteration variables. Following
    TensorIR, a reduction block does not list the buffer it accumulates into
    among its reads.
    """
    acc = block_accesses(block)
    an = Analyzer({iv.var: iv.dom for iv in block.iter_vars})
    writes = tuple(BufferRegion(b, region_cover(idx, b.shape, an)) for b, idx in acc.stores.items())
    is_reduction = any(iv.kind == IterKind.REDUCE for iv in block.iter_vars)
    reads = tuple(
        BufferRegion(b, region_cover(idx, b.shape, an))
        for b, idx in acc.loads.items()
        if not (is_reduction and b in acc.stores)
    )
    return reads, writes


def with_regions(block: Block) -> Block:
    """``block`` with its read/write regions recomputed from its body. Blocks with
    nested blocks or with pointer arguments (tensorized) keep theirs."""
    if not is_leaf_block(block) or block_accesses(block).addressed:
        return block
    reads, writes = infer_regions(block)
    return Block(
        block.name,
        block.iter_vars,
        reads,
        writes,
        block.body,
        block.init,
        block.alloc_buffers,
        block.annotations,
    )


def iter_block_realizes(stmt: Stmt) -> Iterator[BlockRealize]:
    """Every BlockRealize in ``stmt`` in program order."""
    stack = [stmt]
    while stack:
        s = stack.pop()
        if isinstance(s, BlockRealize):
            yield s
        stack.extend(reversed(children(s)))


def defined_vars(stmt: Stmt) -> list[Var]:
    """Loop, iteration and let variables defined in ``stmt``."""
    from tessera.tir.stmt import For, LetStmt

    out: list[Var] = []
    stack = [stmt]
    while stack:
        s = stack.pop()
        if isinstance(s, For):
            out.append(s.loop_var)
        elif isinstance(s, Block):
            out.extend(iv.var for iv in s.iter_vars)
        elif isinstance(s, LetStmt):
            out.append(s.var)
        stack.extend(children(s))
    return out


def structural_hash(func: PrimFunc) -> str:
    """A hash of ``func``'s structure that ignores every name (function, buffers,
    variables, blocks): two groups that differ only in names hash the same."""
    return hashlib.sha256(canonical_text(func).encode()).hexdigest()[:16]


def structural_equal(a: PrimFunc, b: PrimFunc) -> bool:
    """True if ``a`` and ``b`` are the same function up to names."""
    return canonical_text(a) == canonical_text(b)
