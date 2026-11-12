"""Hand-built PrimFuncs for the C backend tests (no schedule primitives needed)."""

from __future__ import annotations

from collections.abc import Sequence

from tessera.dtype import DType
from tessera.tir.expr import Expr, FloatImm, IntImm, Var, lt
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
    seq,
)

S, R = IterKind.SPATIAL, IterKind.REDUCE


def loop(var: Var, extent: int, body: Stmt, kind: ForKind = ForKind.SERIAL) -> For:
    return For(var, IntImm(0), IntImm(extent), kind, body)


def realize(
    name: str,
    iters: Sequence[tuple[Var, int, IterKind, Expr]],
    body: Stmt,
    *,
    init: Stmt | None = None,
    predicate: Expr = TRUE,
    reads: Sequence[Buffer] = (),
    writes: Sequence[Buffer] = (),
    alloc: Sequence[Buffer] = (),
) -> BlockRealize:
    """A block whose iteration variables are (var, extent, kind, bound value)."""
    block = Block(
        name,
        tuple(IterVar(v, Range.of(ext), kind) for v, ext, kind, _ in iters),
        tuple(BufferRegion.full(b) for b in reads),
        tuple(BufferRegion.full(b) for b in writes),
        body,
        init,
        tuple(alloc),
    )
    return BlockRealize(tuple(value for *_, value in iters), predicate, block)


def root(body: Stmt, alloc: Sequence[Buffer] = ()) -> BlockRealize:
    return BlockRealize((), TRUE, Block("root", (), (), (), body, None, tuple(alloc)))


def zero(dtype: DType = DType.float32) -> FloatImm:
    return FloatImm(0.0, dtype)


def matmul(m: int, n: int, k: int, name: str = "matmul") -> PrimFunc:
    """C[m, n] = A[m, k] @ B[k, n], unscheduled: one block with an init."""
    a, b, c = Buffer("A", (m, k)), Buffer("B", (k, n)), Buffer("C", (m, n))
    i, j, r = Var("i"), Var("j"), Var("k")
    vi, vj, vk = Var("vi"), Var("vj"), Var("vk")
    block = realize(
        "C",
        [(vi, m, S, i), (vj, n, S, j), (vk, k, R, r)],
        BufferStore(c, c[vi, vj] + a[vi, vk] * b[vk, vj], (vi, vj)),
        init=BufferStore(c, zero(), (vi, vj)),
        reads=(a, b),
        writes=(c,),
    )
    return PrimFunc(name, (a, b, c), root(loop(i, m, loop(j, n, loop(r, k, block)))))


def tiled_matmul(
    m: int, n: int, k: int, *, tile_m: int = 4, tile_n: int = 16, name: str = "tiled"
) -> PrimFunc:
    """A scheduled matmul: rows split into tiles run in parallel, a ``local``
    accumulator of the full output shape (as ``cache_write`` makes it), the
    reduction init in its own block, the innermost column loop vectorized and
    an imperfect split of the rows guarded by a predicate."""
    a, b, c = Buffer("A", (m, k)), Buffer("B", (k, n)), Buffer("C", (m, n))
    acc = Buffer("C_local", (m, n), scope="local")
    io, jo, ii, ji, r = Var("io"), Var("jo"), Var("ii"), Var("ji"), Var("k")
    mo = -(-m // tile_m)
    assert n % tile_n == 0
    no = n // tile_n

    def rows() -> Expr:
        return io * tile_m + ii

    def cols() -> Expr:
        return jo * tile_n + ji

    def guard() -> Expr:
        return lt(rows(), m) if mo * tile_m != m else TRUE

    vi, vj = Var("vi"), Var("vj")
    init = realize(
        "C_init",
        [(vi, m, S, rows()), (vj, n, S, cols())],
        BufferStore(acc, zero(), (vi, vj)),
        predicate=guard(),
        writes=(acc,),
    )
    ui, uj, uk = Var("vi"), Var("vj"), Var("vk")
    update = realize(
        "C_update",
        [(ui, m, S, rows()), (uj, n, S, cols()), (uk, k, R, r)],
        BufferStore(acc, acc[ui, uj] + a[ui, uk] * b[uk, uj], (ui, uj)),
        predicate=guard(),
        reads=(acc, a, b),
        writes=(acc,),
    )
    wi, wj = Var("vi"), Var("vj")
    store = realize(
        "C_store",
        [(wi, m, S, rows()), (wj, n, S, cols())],
        BufferStore(c, acc[wi, wj], (wi, wj)),
        predicate=guard(),
        reads=(acc,),
        writes=(c,),
    )
    ii2, ji2, ii3, ji3 = Var("ii"), Var("ji"), Var("ii"), Var("ji")
    init_nest = loop(ii, tile_m, loop(ji, tile_n, init, ForKind.VECTORIZED))
    update_nest = loop(r, k, loop(ii2, tile_m, loop(ji2, tile_n, update, ForKind.VECTORIZED)))
    store_nest = loop(ii3, tile_m, loop(ji3, tile_n, store, ForKind.VECTORIZED))
    # The update and store nests use their own loop variables: rebind the
    # index expressions to them.
    update_nest = _rebind(update_nest, {ii: ii2, ji: ji2})
    store_nest = _rebind(store_nest, {ii: ii3, ji: ji3})
    body = loop(io, mo, loop(jo, no, seq([init_nest, update_nest, store_nest])), ForKind.PARALLEL)
    return PrimFunc(name, (a, b, c), root(body, [acc]))


def _rebind(stmt: Stmt, mapping: dict[Var, Var]) -> Stmt:
    """Replace variables in the iteration values of the blocks under ``stmt``."""
    from dataclasses import replace

    from tessera.tir.expr import Binary, Compare

    def sub(e: Expr) -> Expr:
        if isinstance(e, Var):
            return mapping.get(e, e)
        if isinstance(e, Binary):
            return Binary(e.op, sub(e.a), sub(e.b))
        if isinstance(e, Compare):
            return Compare(e.op, sub(e.a), sub(e.b))
        return e

    if isinstance(stmt, For):
        return replace(stmt, body=_rebind(stmt.body, mapping))
    if isinstance(stmt, BlockRealize):
        return BlockRealize(
            tuple(sub(v) for v in stmt.iter_values), sub(stmt.predicate), stmt.block
        )
    return stmt
