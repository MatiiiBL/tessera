"""Loop-IR functions built by hand for the Metal tests, each with a NumPy reference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tessera.tir.expr import Expr, FloatImm, IntImm, Var, call, lt, maximum
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    For,
    ForKind,
    IfThenElse,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
    seq,
)

type Array = npt.NDArray[Any]
type Reference = Callable[[Sequence[Array]], Array]


def loop(
    var: Var,
    extent: int,
    body: Stmt,
    kind: ForKind = ForKind.SERIAL,
    tag: str | None = None,
    start: int = 0,
) -> For:
    return For(var, IntImm(start), IntImm(extent), kind, body, tag)


def bound(var: Var, extent: int, tag: str, body: Stmt) -> For:
    return loop(var, extent, body, ForKind.THREAD_BINDING, tag)


def nest(vars_extents: Sequence[tuple[Var, int]], body: Stmt) -> Stmt:
    for var, extent in reversed(vars_extents):
        body = loop(var, extent, body)
    return body


def block(
    name: str,
    iters: Sequence[tuple[Var, int, Expr, IterKind]],
    body: Stmt,
    *,
    init: Stmt | None = None,
    predicate: Expr = TRUE,
    alloc: Sequence[Buffer] = (),
) -> BlockRealize:
    iter_vars = tuple(IterVar(v, Range.of(extent), kind) for v, extent, _, kind in iters)
    blk = Block(name, iter_vars, (), (), body, init, tuple(alloc))
    return BlockRealize(tuple(value for _, _, value, _ in iters), predicate, blk)


def spatial(var: Var, extent: int, value: Expr) -> tuple[Var, int, Expr, IterKind]:
    return (var, extent, value, IterKind.SPATIAL)


def reduce(var: Var, extent: int, value: Expr) -> tuple[Var, int, Expr, IterKind]:
    return (var, extent, value, IterKind.REDUCE)


def prim_func(
    name: str, params: Sequence[Buffer], body: Stmt, alloc: Sequence[Buffer] = ()
) -> PrimFunc:
    root = Block(
        "root",
        (),
        (),
        tuple(BufferRegion.full(p) for p in params[-1:]),
        body,
        None,
        tuple(alloc),
    )
    return PrimFunc(name, tuple(params), BlockRealize((), TRUE, root))


@dataclass
class Case:
    func: PrimFunc
    reference: Reference
    inputs: list[Array]


def rand(rng: np.random.Generator, *shape: int) -> Array:
    return rng.standard_normal(shape).astype(np.float32)


# -- elementwise ---------------------------------------------------------------------


def add_relu(n: int, rng: np.random.Generator) -> Case:
    """Unscheduled: one thread runs the whole loop."""
    a, b, c = Buffer("a", (n,)), Buffer("b", (n,)), Buffer("c", (n,))
    i, vi = Var("i"), Var("vi")
    body = loop(
        i,
        n,
        block("c", [spatial(vi, n, i)], BufferStore(c, maximum(a[vi] + b[vi], 0.0), (vi,))),
    )
    return Case(
        prim_func("add_relu", [a, b, c], body),
        lambda xs: np.maximum(xs[0] + xs[1], 0),
        [rand(rng, n), rand(rng, n)],
    )


def bound_unary(
    n: int,
    rng: np.random.Generator,
    fn: Callable[[Expr], Expr],
    reference: Callable[[Array], Array],
    *,
    threads: int = 256,
    name: str = "unary",
    low: float = -4.0,
    high: float = 4.0,
) -> Case:
    """``out[i] = fn(x[i])`` with i split over threadgroups and threads (imperfect)."""
    x, out = Buffer("x", (n,)), Buffer("out", (n,))
    bx, tx, vi = Var("bx"), Var("tx"), Var("vi")
    groups = -(-n // threads)
    index = bx * threads + tx
    body = bound(
        bx,
        groups,
        "threadgroup.x",
        bound(
            tx,
            threads,
            "thread.x",
            block(
                "out",
                [spatial(vi, n, index)],
                BufferStore(out, fn(x[vi]), (vi,)),
                predicate=lt(index, n),
            ),
        ),
    )
    data = rng.uniform(low, high, n).astype(np.float32)
    return Case(prim_func(name, [x, out], body), lambda xs: reference(xs[0]), [data])


# -- matmul --------------------------------------------------------------------------


def matmul_naive(m: int, n: int, k: int, rng: np.random.Generator) -> Case:
    """Unscheduled reduction with an init statement."""
    a, b, c = Buffer("A", (m, k)), Buffer("B", (k, n)), Buffer("C", (m, n))
    i, j, r = Var("i"), Var("j"), Var("k")
    vi, vj, vk = Var("vi"), Var("vj"), Var("vk")
    body = nest(
        [(i, m), (j, n), (r, k)],
        block(
            "C",
            [spatial(vi, m, i), spatial(vj, n, j), reduce(vk, k, r)],
            BufferStore(c, c[vi, vj] + a[vi, vk] * b[vk, vj], (vi, vj)),
            init=BufferStore(c, FloatImm(0.0), (vi, vj)),
        ),
    )
    return Case(
        prim_func("matmul", [a, b, c], body),
        lambda xs: xs[0] @ xs[1],
        [rand(rng, m, k), rand(rng, k, n)],
    )


def matmul_bound(m: int, n: int, k: int, rng: np.random.Generator, tile: int = 8) -> Case:
    """One thread per output element; threadgroups of tile x tile threads."""
    a, b, c = Buffer("A", (m, k)), Buffer("B", (k, n)), Buffer("C", (m, n))
    by, bx, ty, tx, r = Var("by"), Var("bx"), Var("ty"), Var("tx"), Var("k")
    vi, vj, vk = Var("vi"), Var("vj"), Var("vk")
    row, col = by * tile + ty, bx * tile + tx
    body = bound(
        by,
        -(-m // tile),
        "threadgroup.y",
        bound(
            bx,
            -(-n // tile),
            "threadgroup.x",
            bound(
                ty,
                tile,
                "thread.y",
                bound(
                    tx,
                    tile,
                    "thread.x",
                    loop(
                        r,
                        k,
                        block(
                            "C",
                            [spatial(vi, m, row), spatial(vj, n, col), reduce(vk, k, r)],
                            BufferStore(c, c[vi, vj] + a[vi, vk] * b[vk, vj], (vi, vj)),
                            init=BufferStore(c, FloatImm(0.0), (vi, vj)),
                            predicate=_and(lt(row, m), lt(col, n)),
                        ),
                    ),
                ),
            ),
        ),
    )
    return Case(
        prim_func("matmul_bound", [a, b, c], body),
        lambda xs: xs[0] @ xs[1],
        [rand(rng, m, k), rand(rng, k, n)],
    )


def _and(a: Expr, b: Expr) -> Expr:
    from tessera.tir.expr import logical_and

    return logical_and(a, b)


def matmul_shared(size: int, rng: np.random.Generator, tile: int = 16) -> Case:
    """Tiles of A and B staged in threadgroup memory, a local accumulator per thread.
    Barriers are left to the code generator."""
    assert size % tile == 0
    a, b, c = Buffer("A", (size, size)), Buffer("B", (size, size)), Buffer("C", (size, size))
    a_sh = Buffer("A_shared", (tile, tile), scope="shared")
    b_sh = Buffer("B_shared", (tile, tile), scope="shared")
    acc = Buffer("acc", (1,), scope="local")
    by, bx, ko, ki = Var("by"), Var("bx"), Var("ko"), Var("ki")

    def threads(body: Callable[[Var, Var], Stmt]) -> Stmt:
        ty, tx = Var("ty"), Var("tx")
        return bound(ty, tile, "thread.y", bound(tx, tile, "thread.x", body(ty, tx)))

    def load(ty: Var, tx: Var) -> Stmt:
        return seq(
            [
                BufferStore(a_sh, a[by * tile + ty, ko * tile + tx], (ty, tx)),
                BufferStore(b_sh, b[ko * tile + ty, bx * tile + tx], (ty, tx)),
            ]
        )

    def update(ty: Var, tx: Var) -> Stmt:
        return loop(
            ki,
            tile,
            BufferStore(acc, acc[0] + a_sh[ty, ki] * b_sh[ki, tx], (IntImm(0),)),
            ForKind.UNROLLED,
        )

    body = bound(
        by,
        size // tile,
        "threadgroup.y",
        bound(
            bx,
            size // tile,
            "threadgroup.x",
            seq(
                [
                    threads(lambda ty, tx: BufferStore(acc, FloatImm(0.0), (IntImm(0),))),
                    loop(ko, size // tile, seq([threads(load), threads(update)])),
                    threads(
                        lambda ty, tx: BufferStore(c, acc[0], (by * tile + ty, bx * tile + tx))
                    ),
                ]
            ),
        ),
    )
    return Case(
        prim_func("matmul_shared", [a, b, c], body, alloc=[a_sh, b_sh, acc]),
        lambda xs: xs[0] @ xs[1],
        [rand(rng, size, size), rand(rng, size, size)],
    )


def row_sum_tree(rows: int, rng: np.random.Generator, threads: int = 256) -> Case:
    """Sum of each row of ``rows x threads`` by a tree in threadgroup memory."""
    x, out = Buffer("x", (rows, threads)), Buffer("out", (rows,))
    red = Buffer("red", (threads,), scope="shared")
    bx = Var("row")
    steps: list[Stmt] = []
    tx0 = Var("tx")
    steps.append(bound(tx0, threads, "thread.x", BufferStore(red, x[bx, tx0], (tx0,))))
    half = threads // 2
    while half >= 1:
        tx = Var("tx")
        steps.append(
            bound(
                tx,
                threads,
                "thread.x",
                IfThenElse(lt(tx, half), BufferStore(red, red[tx] + red[tx + half], (tx,))),
            )
        )
        half //= 2
    tx_last = Var("tx")
    steps.append(
        bound(
            tx_last,
            threads,
            "thread.x",
            IfThenElse(lt(tx_last, 1), BufferStore(out, red[0], (bx,))),
        )
    )
    body = bound(bx, rows, "threadgroup.x", seq(steps))
    return Case(
        prim_func("row_sum", [x, out], body, alloc=[red]),
        lambda xs: xs[0].astype(np.float64).sum(axis=1),
        [rand(rng, rows, threads)],
    )


def exp_then_scale(n: int, rng: np.random.Generator) -> Case:
    """Two blocks with a global intermediate, unscheduled (one thread)."""
    x, out = Buffer("x", (n,)), Buffer("out", (n,))
    tmp = Buffer("tmp", (n,))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    body = seq(
        [
            loop(
                i, n, block("tmp", [spatial(vi, n, i)], BufferStore(tmp, call("exp", x[vi]), (vi,)))
            ),
            loop(j, n, block("out", [spatial(vj, n, j)], BufferStore(out, tmp[vj] * 2.0, (vj,)))),
        ]
    )
    return Case(
        prim_func("exp_scale", [x, out], body, alloc=[tmp]),
        lambda xs: np.exp(xs[0]) * 2,
        [rand(rng, n)],
    )


def _gpu_available() -> bool:
    from tessera.backend.metal import available

    return available()


# Tests that run kernels are skipped where there is no Metal GPU (Linux, CI).
gpu = pytest.mark.skipif(not _gpu_available(), reason="needs a Metal GPU")
requires_gpu = [pytest.mark.metal, gpu]


def run_case(case: Case) -> tuple[Array, Array]:
    """Run ``case`` on the GPU; returns (result, float64 reference)."""
    from tessera.backend.metal.executor import run_func

    out = case.func.params[-1]
    args = [x.copy() for x in case.inputs] + [np.zeros(out.shape, out.dtype.numpy)]
    run_func(case.func, args)
    reference = case.reference([x.astype(np.float64) for x in case.inputs])
    return args[-1], reference


def assert_close(result: Array, reference: Array, rtol: float = 1e-5, atol: float = 1e-5) -> None:
    np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)


def matmul_tiled(
    m: int, n: int, k: int, rng: np.random.Generator, *, threads: int = 8, reg: int = 2, kt: int = 8
) -> Case:
    """The shape a schedule leaves: cache buffers with the whole tensor's shape
    (``A_shared``, ``B_shared``, ``C_local``) used under tile loops. Each of
    ``threads x threads`` threads computes a ``reg x reg`` tile; tiles of A and
    B are loaded cooperatively, ``kt`` columns at a time. The function is
    correct when run sequentially, so the GPU result depends on compaction
    giving each thread its own accumulators."""
    bm = threads * reg
    assert m % bm == 0
    assert n % bm == 0
    assert k % kt == 0
    a, b, c = Buffer("A", (m, k)), Buffer("B", (k, n)), Buffer("C", (m, n))
    a_sh = Buffer("A_shared", (m, k), scope="shared")
    b_sh = Buffer("B_shared", (k, n), scope="shared")
    c_loc = Buffer("C_local", (m, n), scope="local")
    by, bx, ko = Var("by"), Var("bx"), Var("ko")

    def per_thread(body: Callable[[Var, Var], Stmt]) -> Stmt:
        ty, tx = Var("ty"), Var("tx")
        return bound(ty, threads, "thread.y", bound(tx, threads, "thread.x", body(ty, tx)))

    def tile(body: Callable[[Expr, Expr], Stmt], ty: Var, tx: Var) -> Stmt:
        i, j = Var("i"), Var("j")
        row, col = by * bm + ty * reg + i, bx * bm + tx * reg + j
        return loop(i, reg, loop(j, reg, body(row, col), ForKind.UNROLLED), ForKind.UNROLLED)

    def init(ty: Var, tx: Var) -> Stmt:
        return tile(lambda r, c_: BufferStore(c_loc, FloatImm(0.0), (r, c_)), ty, tx)

    def load(src: Buffer, dst: Buffer, rows: int, cols: int, row0: Expr, col0: Expr) -> Stmt:
        ty, tx, e = Var("ty"), Var("tx"), Var("e")
        count = rows * cols // (threads * threads)
        flat = e * (threads * threads) + ty * threads + tx
        r, c_ = row0 + flat // cols, col0 + flat % cols
        return bound(
            ty,
            threads,
            "thread.y",
            bound(tx, threads, "thread.x", loop(e, count, BufferStore(dst, src[r, c_], (r, c_)))),
        )

    def update(ty: Var, tx: Var) -> Stmt:
        kk = Var("kk")

        def body(r: Expr, c_: Expr) -> Stmt:
            kidx = ko * kt + kk
            return BufferStore(c_loc, c_loc[r, c_] + a_sh[r, kidx] * b_sh[kidx, c_], (r, c_))

        return loop(kk, kt, tile(body, ty, tx))

    def store(ty: Var, tx: Var) -> Stmt:
        return tile(lambda r, c_: BufferStore(c, c_loc[r, c_], (r, c_)), ty, tx)

    body = bound(
        by,
        m // bm,
        "threadgroup.y",
        bound(
            bx,
            n // bm,
            "threadgroup.x",
            seq(
                [
                    per_thread(init),
                    loop(
                        ko,
                        k // kt,
                        seq(
                            [
                                load(a, a_sh, bm, kt, by * bm, ko * kt),
                                load(b, b_sh, kt, bm, ko * kt, bx * bm),
                                per_thread(update),
                            ]
                        ),
                    ),
                    per_thread(store),
                ]
            ),
        ),
    )
    return Case(
        prim_func("matmul_tiled", [a, b, c], body, alloc=[a_sh, b_sh, c_loc]),
        lambda xs: xs[0] @ xs[1],
        [rand(rng, m, k), rand(rng, k, n)],
    )
