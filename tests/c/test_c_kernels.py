"""Generated kernels compiled with clang and run against NumPy."""

from __future__ import annotations

import math
import platform
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from c_funcs import R, S, loop, matmul, realize, root, tiled_matmul, zero
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.backend.c import build as cbuild
from tessera.backend.c import run_kernel
from tessera.backend.c.executor import kernel_sources
from tessera.dtype import DType
from tessera.ir.graph import NDArray
from tessera.target import Target
from tessera.tir.expr import (
    MATH_INTRINSICS,
    Binary,
    Cast,
    Compare,
    Expr,
    FloatImm,
    IntImm,
    Var,
    call,
    ge,
    gt,
    logical_and,
    lt,
    maximum,
    minimum,
    select,
)
from tessera.tir.expr import Select as SelectExpr
from tessera.tir.stmt import (
    Buffer,
    BufferStore,
    For,
    ForKind,
    IfThenElse,
    LetStmt,
    PrimFunc,
    seq,
)

rng = np.random.default_rng(0)


def _rand(*shape: int) -> NDArray:
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize(("m", "n", "k"), [(1, 1, 1), (8, 16, 4), (13, 7, 33)])
def test_matmul(m: int, n: int, k: int) -> None:
    a, b, c = _rand(m, k), _rand(k, n), np.zeros((m, n), np.float32)
    run_kernel(matmul(m, n, k), [a, b, c])
    np.testing.assert_allclose(c, a @ b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("m", "threads"), [(16, 1), (16, 4), (10, 4), (3, 4), (37, 3)])
def test_tiled_parallel_matmul(m: int, threads: int) -> None:
    a, b, c = _rand(m, 24), _rand(24, 32), np.zeros((m, 32), np.float32)
    run_kernel(tiled_matmul(m, 32, 24), [a, b, c], Target(num_threads=threads))
    np.testing.assert_allclose(c, a @ b, rtol=1e-5, atol=1e-5)


def _rowsum(rows: int, cols: int, *, decomposed: bool) -> PrimFunc:
    """Y[i] = sum(X[i, :]) per row in parallel, accumulated in a one-element
    ``local`` buffer declared at the root: the accumulator must be private to
    each row (allocated inside the parallel loop) and live across the
    reduction loop."""
    x, y = Buffer("X", (rows, cols)), Buffer("Y", (rows,))
    acc = Buffer("acc", (1,), scope="local")
    i, k = Var("i"), Var("k")
    vi, vk = Var("vi"), Var("vk")
    update = BufferStore(acc, acc[0] + x[vi, vk], (IntImm(0),))
    init = BufferStore(acc, zero(), (IntImm(0),))
    if decomposed:
        wi = Var("vi")
        nest = seq(
            [
                realize("acc_init", [(wi, rows, S, i)], init),
                loop(k, cols, realize("acc", [(vi, rows, S, i), (vk, cols, R, k)], update)),
            ]
        )
    else:
        nest = loop(
            k, cols, realize("acc", [(vi, rows, S, i), (vk, cols, R, k)], update, init=init)
        )
    wi = Var("vi")
    write = realize("Y", [(wi, rows, S, i)], BufferStore(y, acc[0], (wi,)))
    body = loop(i, rows, seq([nest, write]), ForKind.PARALLEL)
    return PrimFunc("rowsum", (x, y), root(body, [acc]))


@pytest.mark.parametrize("decomposed", [False, True])
def test_local_accumulator_is_private_to_each_parallel_iteration(decomposed: bool) -> None:
    x, y = _rand(64, 50), np.zeros(64, np.float32)
    func = _rowsum(64, 50, decomposed=decomposed)
    run_kernel(func, [x, y], Target(num_threads=4))
    np.testing.assert_allclose(y, x.sum(axis=1), rtol=1e-5, atol=1e-5)


def test_global_intermediate_between_two_nests() -> None:
    """T = X * 2 (a ``global`` buffer of the kernel), then Y = relu(T)."""
    x, y = Buffer("X", (100,)), Buffer("Y", (100,))
    t = Buffer("T", (100,))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    first = loop(i, 100, realize("T", [(vi, 100, S, i)], BufferStore(t, x[vi] * 2.0, (vi,))))
    second = loop(
        j,
        100,
        realize("Y", [(vj, 100, S, j)], BufferStore(y, maximum(t[vj], 0.0), (vj,))),
        ForKind.PARALLEL,
    )
    func = PrimFunc("twice_relu", (x, y), root(seq([first, second]), [t]))
    xs, ys = _rand(100), np.zeros(100, np.float32)
    run_kernel(func, [xs, ys], Target(num_threads=4))
    np.testing.assert_array_equal(ys, np.maximum(xs * 2, 0))


def test_select_guards_out_of_bounds_loads() -> None:
    """Y = pad(X, 3) by Select: the guarded loads would be out of bounds."""
    n, pad = 1000, 3
    x, y = Buffer("X", (n,)), Buffer("Y", (n + 2 * pad,))
    i, vi = Var("i"), Var("vi")
    inside = logical_and(ge(vi, pad), lt(vi, n + pad))
    value = select(inside, x[vi - pad], 0.0)
    body = loop(
        i, n + 2 * pad, realize("Y", [(vi, n + 2 * pad, S, i)], BufferStore(y, value, (vi,)))
    )
    xs, ys = _rand(n), np.zeros(n + 2 * pad, np.float32)
    run_kernel(PrimFunc("pad", (x, y), root(body)), [xs, ys])
    np.testing.assert_array_equal(ys, np.pad(xs, pad))


def test_unrolled_and_vectorized_loops() -> None:
    x, y = Buffer("X", (8, 64)), Buffer("Y", (8, 64))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    block = realize(
        "Y", [(vi, 8, S, i), (vj, 64, S, j)], BufferStore(y, x[vi, vj] * x[vi, vj], (vi, vj))
    )
    body = loop(i, 8, loop(j, 64, block, ForKind.VECTORIZED), ForKind.UNROLLED)
    xs, ys = _rand(8, 64), np.zeros((8, 64), np.float32)
    run_kernel(PrimFunc("square", (x, y), root(body)), [xs, ys])
    np.testing.assert_array_equal(ys, xs * xs)


def test_nested_parallel_loops_and_captured_outer_variables() -> None:
    """for b: parallel i: parallel j (serial inside): Y[b, i, j] = X[b, i, j] + b."""
    x, y = Buffer("X", (3, 10, 7)), Buffer("Y", (3, 10, 7))
    b, i, j = Var("b"), Var("i"), Var("j")
    vb, vi, vj = Var("vb"), Var("vi"), Var("vj")
    value = x[vb, vi, vj] + Cast(DType.float32, vb)
    block = realize(
        "Y", [(vb, 3, S, b), (vi, 10, S, i), (vj, 7, S, j)], BufferStore(y, value, (vb, vi, vj))
    )
    body = loop(b, 3, loop(i, 10, loop(j, 7, block, ForKind.PARALLEL), ForKind.PARALLEL))
    xs, ys = _rand(3, 10, 7), np.zeros((3, 10, 7), np.float32)
    run_kernel(PrimFunc("offset", (x, y), root(body)), [xs, ys], Target(num_threads=4))
    np.testing.assert_array_equal(ys, xs + np.arange(3, dtype=np.float32)[:, None, None])


def test_bool_and_integer_buffers() -> None:
    """M = X > 0 (bool); Y = X[I] (gather by int64 indices) where M, else 0."""
    n = 50
    x, idx = Buffer("X", (n,)), Buffer("I", (n,), DType.int64)
    mask, y = Buffer("M", (n,), DType.bool_), Buffer("Y", (n,))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    first = loop(i, n, realize("M", [(vi, n, S, i)], BufferStore(mask, gt(x[vi], 0.0), (vi,))))
    gathered = select(Cast(DType.bool_, mask[vj]), x[idx[vj]], 0.0)
    second = loop(j, n, realize("Y", [(vj, n, S, j)], BufferStore(y, gathered, (vj,))))
    func = PrimFunc("gather", (x, idx, mask, y), root(seq([first, second])))
    xs = _rand(n)
    ids = rng.integers(0, n, n).astype(np.int64)
    ms, ys = np.zeros(n, np.bool_), np.zeros(n, np.float32)
    run_kernel(func, [xs, ids, ms, ys])
    np.testing.assert_array_equal(ms, xs > 0)
    np.testing.assert_array_equal(ys, np.where(xs > 0, xs[ids], 0))


def test_let_and_if() -> None:
    x, y = Buffer("X", (20,)), Buffer("Y", (20,))
    i, t = Var("i"), Var("t", DType.float32)
    store_pos = BufferStore(y, t, (i,))
    store_neg = BufferStore(y, t * -1.0, (i,))
    body = loop(i, 20, LetStmt(t, x[i] * 3.0, IfThenElse(gt(t, 0.0), store_pos, store_neg)))
    xs, ys = _rand(20), np.zeros(20, np.float32)
    run_kernel(PrimFunc("absx3", (x, y), root(body)), [xs, ys])
    np.testing.assert_allclose(ys, np.abs(xs * 3), rtol=1e-6)


def test_float16_cast() -> None:
    x, y = Buffer("X", (64,)), Buffer("Y", (64,), DType.float16)
    i = Var("i")
    body = loop(i, 64, BufferStore(y, Cast(DType.float16, x[i]), (i,)))
    xs, ys = _rand(64) * 100, np.zeros(64, np.float16)
    run_kernel(PrimFunc("to_half", (x, y), root(body)), [xs, ys])
    np.testing.assert_array_equal(ys, xs.astype(np.float16))


_NUMPY_MATH: dict[str, Callable[[NDArray], NDArray]] = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda v: 1 / np.sqrt(v),
    "tanh": np.tanh,
    "erf": np.vectorize(math.erf),
    "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
}


@pytest.mark.parametrize("name", sorted(MATH_INTRINSICS - {"fma"}))
def test_math_intrinsics(name: str) -> None:
    x, y = Buffer("X", (256,)), Buffer("Y", (256,))
    i = Var("i")
    body = loop(i, 256, BufferStore(y, call(name, x[i]), (i,)), ForKind.VECTORIZED)
    xs = (np.abs(_rand(256)) + 0.01) if name in ("log", "sqrt", "rsqrt") else _rand(256) * 3
    ys = np.zeros(256, np.float32)
    run_kernel(PrimFunc(f"math_{name}", (x, y), root(body)), [xs, ys])
    expected = _NUMPY_MATH[name](xs.astype(np.float64))
    np.testing.assert_allclose(ys, expected, rtol=2e-6, atol=1e-7)


def test_fma() -> None:
    x, y = Buffer("X", (16,)), Buffer("Y", (16,))
    i = Var("i")
    body = loop(i, 16, BufferStore(y, call("fma", x[i], x[i], 1.0), (i,)))
    xs, ys = _rand(16), np.zeros(16, np.float32)
    run_kernel(PrimFunc("fma", (x, y), root(body)), [xs, ys])
    np.testing.assert_allclose(ys, xs * xs + 1, rtol=1e-6)


def test_floordiv_and_floormod_round_toward_negative_infinity() -> None:
    n = 2000
    a, b = Buffer("A", (n,), DType.int64), Buffer("B", (n,), DType.int64)
    q, r = Buffer("Q", (n,), DType.int64), Buffer("R", (n,), DType.int64)
    i = Var("i")
    body = loop(
        i,
        n,
        seq(
            [
                BufferStore(q, Binary("floordiv", a[i], b[i]), (i,)),
                BufferStore(r, Binary("floormod", a[i], b[i]), (i,)),
            ]
        ),
    )
    av = rng.integers(-1000, 1000, n).astype(np.int64)
    bv = rng.integers(1, 50, n).astype(np.int64) * rng.choice([-1, 1], n)
    qv, rv = np.zeros(n, np.int64), np.zeros(n, np.int64)
    run_kernel(PrimFunc("divmod", (a, b, q, r), root(body)), [av, bv, qv, rv])
    np.testing.assert_array_equal(qv, av // bv)
    np.testing.assert_array_equal(rv, av % bv)


# -- printing: random expression trees evaluated in C and in Python ----------

_INT_OPS = ["add", "sub", "mul", "min", "max"]


def _int_tree(leaves: list[Expr]) -> st.SearchStrategy[Expr]:
    base = st.sampled_from(leaves) | st.integers(-9, 9).map(IntImm)

    def extend(children: st.SearchStrategy[Expr]) -> st.SearchStrategy[Expr]:
        arith = st.builds(Binary, st.sampled_from(_INT_OPS), children, children)
        divisor = st.sampled_from([2, 3, 4, 7, 8]).map(IntImm)
        division = st.builds(Binary, st.sampled_from(["floordiv", "floormod"]), children, divisor)
        choice = st.builds(
            lambda c, d, t, f: select(lt(c, d), t, f), children, children, children, children
        )
        return arith | division | choice

    return st.recursive(base, extend, max_leaves=8)


def _evaluate(e: Expr, env: dict[Var, int]) -> int:
    if isinstance(e, IntImm):
        return e.value
    if isinstance(e, Var):
        return env[e]
    if isinstance(e, Binary):
        a, b = _evaluate(e.a, env), _evaluate(e.b, env)
        ops: dict[str, Callable[[int, int], int]] = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "floordiv": lambda x, y: x // y,
            "floormod": lambda x, y: x % y,
            "min": min,
            "max": max,
        }
        return ops[e.op](a, b)
    if isinstance(e, SelectExpr):
        assert isinstance(e.cond, Compare)
        cond = _evaluate(e.cond.a, env) < _evaluate(e.cond.b, env)
        return _evaluate(e.true_value if cond else e.false_value, env)
    raise TypeError(e)


@settings(max_examples=20, deadline=None)
@given(st.data())
def test_random_integer_expressions_match_python(data: st.DataObject) -> None:
    """The printer's precedence and parentheses keep each tree's meaning."""
    n = 64
    x = Buffer("X", (n, 2), DType.int64)
    y = Buffer("Y", (n, 4), DType.int64)
    i = Var("i")
    a, b = Var("a"), Var("b")
    exprs = [data.draw(_int_tree([a, b, i])) for _ in range(4)]
    stores = seq([BufferStore(y, e, (i, IntImm(c))) for c, e in enumerate(exprs)])
    body = loop(i, n, LetStmt(a, x[i, 0], LetStmt(b, x[i, 1], stores)))
    xs = rng.integers(-50, 50, (n, 2)).astype(np.int64)
    ys = np.zeros((n, 4), np.int64)
    run_kernel(PrimFunc("exprs", (x, y), root(body)), [xs, ys])
    for row in range(n):
        env = {a: int(xs[row, 0]), b: int(xs[row, 1]), i: row}
        assert [int(v) for v in ys[row]] == [_evaluate(e, env) for e in exprs]


def test_float_subtraction_keeps_its_grouping() -> None:
    x, y = Buffer("X", (4,)), Buffer("Y", (4,))
    i = Var("i")
    # a - (b - c) and (a - b) - c differ; so do a / (b * c) and a / b * c.
    value = (x[i] - (x[i] * 2.0 - 1.0)) + x[i] / (x[i] * 2.0)
    body = loop(i, 4, BufferStore(y, value, (i,)))
    xs, ys = np.array([1, 2, 3, 4], np.float32), np.zeros(4, np.float32)
    run_kernel(PrimFunc("grouping", (x, y), root(body)), [xs, ys])
    np.testing.assert_allclose(ys, (xs - (xs * 2 - 1)) + xs / (xs * 2), rtol=1e-6)


def test_minimum_and_maximum_of_floats() -> None:
    x, y = Buffer("X", (32,)), Buffer("Y", (32,))
    i = Var("i")
    value = minimum(maximum(x[i], -0.5), FloatImm(0.5))
    body = loop(i, 32, BufferStore(y, value, (i,)))
    xs, ys = _rand(32), np.zeros(32, np.float32)
    run_kernel(PrimFunc("clip", (x, y), root(body)), [xs, ys])
    np.testing.assert_array_equal(ys, np.clip(xs, -0.5, 0.5))


def test_vectorized_loop_compiles_to_simd_instructions(tmp_path: Path) -> None:
    """Checks the assembly clang produces for a VECTORIZED loop."""
    x, y = Buffer("X", (1024,)), Buffer("Y", (1024,))
    i = Var("i")
    store = BufferStore(y, x[i] * 3.0 + 1.0, (i,))
    body = For(i, IntImm(0), IntImm(1024), ForKind.VECTORIZED, store)
    target = Target()
    for name, text in kernel_sources(PrimFunc("axpb", (x, y), root(body)), target).items():
        (tmp_path / name).write_text(text)
    flags = [f for f in cbuild.compile_flags(target) if f not in ("-shared", "-fPIC")]
    cmd = [target.cc, *flags, "-S", "kernel.c", "-o", "kernel.s"]
    subprocess.run(cmd, cwd=tmp_path, check=True)
    asm = (tmp_path / "kernel.s").read_text()
    if platform.machine() in ("arm64", "aarch64"):
        assert re.search(r"fmla(\.4s|\s+v\d+\.4s)", asm)
    else:
        assert re.search(r"(mulps|vfmadd\w*ps|addps)", asm)
