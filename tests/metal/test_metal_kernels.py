"""Generated kernels compute what their PrimFunc says, checked against NumPy on the GPU."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from metal_funcs import (
    Array,
    add_relu,
    assert_close,
    bound,
    bound_unary,
    exp_then_scale,
    matmul_bound,
    matmul_naive,
    matmul_shared,
    prim_func,
    requires_gpu,
    row_sum_tree,
    run_case,
)

from tessera.backend.metal.executor import run_func
from tessera.dtype import DType
from tessera.tir.expr import (
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    Expr,
    FloatImm,
    IntImm,
    Select,
    Var,
    call,
    lt,
    select,
)
from tessera.tir.stmt import Buffer, BufferStore, LetStmt, seq

pytestmark = requires_gpu
RNG = np.random.default_rng(1)


@pytest.mark.parametrize(
    "case",
    [
        add_relu(1000, RNG),
        matmul_naive(7, 9, 11, RNG),
        matmul_bound(33, 17, 20, RNG, tile=8),
        matmul_shared(64, RNG, tile=16),
        row_sum_tree(5, RNG, threads=256),
        exp_then_scale(100, RNG),
    ],
    ids=lambda c: c.func.name,
)
def test_hand_written_schedules(case: object) -> None:
    result, reference = run_case(case)  # type: ignore[arg-type]
    assert_close(result, reference, rtol=1e-4, atol=1e-4)


@given(tile=st.sampled_from([2, 4, 8, 16]), tiles=st.integers(1, 4))
@settings(max_examples=12, deadline=None)
def test_shared_memory_matmul_for_any_tile(tile: int, tiles: int) -> None:
    result, reference = run_case(matmul_shared(tile * tiles, np.random.default_rng(tile), tile))
    assert_close(result, reference, rtol=1e-4, atol=1e-4)


@given(log_threads=st.integers(1, 10), rows=st.integers(1, 6))
@settings(max_examples=12, deadline=None)
def test_tree_reduction_for_any_threadgroup_size(log_threads: int, rows: int) -> None:
    result, reference = run_case(row_sum_tree(rows, np.random.default_rng(rows), 2**log_threads))
    assert_close(result, reference, rtol=1e-4, atol=1e-4)


def _erf(x: Array) -> Array:
    return np.vectorize(math.erf)(x)


UNARY: list[tuple[str, Callable[[Expr], Expr], Callable[[Array], Array], float, float]] = [
    ("exp", lambda v: call("exp", v), np.exp, -20, 20),
    ("log", lambda v: call("log", v), np.log, 1e-3, 1e3),
    ("sqrt", lambda v: call("sqrt", v), np.sqrt, 0, 100),
    ("rsqrt", lambda v: call("rsqrt", v), lambda x: 1 / np.sqrt(x), 1e-2, 100),
    ("tanh", lambda v: call("tanh", v), np.tanh, -100, 100),
    ("erf", lambda v: call("erf", v), _erf, -6, 6),
    ("sigmoid", lambda v: call("sigmoid", v), lambda x: 1 / (1 + np.exp(-x)), -100, 100),
    ("abs", lambda v: call("abs", v), np.abs, -5, 5),
    ("floor", lambda v: call("floor", v), np.floor, -5, 5),
    ("ceil", lambda v: call("ceil", v), np.ceil, -5, 5),
    ("fma", lambda v: call("fma", v, v, 1.0), lambda x: x * x + 1, -5, 5),
    ("pow3", lambda v: Binary("pow", v, FloatImm(3.0)), lambda x: x**3, -4, 4),
    ("pow2", lambda v: Binary("pow", v, FloatImm(2.0)), lambda x: x**2, -4, 4),
    ("pow_half", lambda v: Binary("pow", v, FloatImm(0.5)), np.sqrt, 0, 9),
    ("floordiv", lambda v: v // 0.75, lambda x: np.floor_divide(x, 0.75), -5, 5),
    ("floormod", lambda v: v % 0.75, lambda x: np.mod(x, 0.75), -5, 5),
    ("neg", lambda v: -v, np.negative, -5, 5),
]


@pytest.mark.parametrize(
    ("name", "fn", "reference", "low", "high"), UNARY, ids=[u[0] for u in UNARY]
)
def test_math(
    name: str,
    fn: Callable[[Expr], Expr],
    reference: Callable[[Array], Array],
    low: float,
    high: float,
) -> None:
    case = bound_unary(4099, np.random.default_rng(2), fn, reference, name=name, low=low, high=high)
    result, expected = run_case(case)
    assert not np.isnan(result).any()
    assert_close(result, expected, rtol=2e-6 if name != "exp" else 1e-5, atol=1e-6)


def test_infinities_survive_fast_math() -> None:
    x, out = Buffer("x", (8,)), Buffer("out", (1,))
    i = Var("i")
    from metal_funcs import loop

    body = seq(
        [
            BufferStore(out, FloatImm(float("-inf")), (IntImm(0),)),
            loop(i, 8, BufferStore(out, Binary("max", out[0], x[i]), (IntImm(0),))),
        ]
    )
    func = prim_func("running_max", [x, out], body)
    data = np.array([-np.inf, -3, -7, -1, -np.inf, -2, -9, -4], np.float32)
    result = np.zeros(1, np.float32)
    run_func(func, [data, result])
    assert result[0] == -1


# -- random elementwise expressions -----------------------------------------------------

LEAF_A, LEAF_B = Buffer("a", (256,)), Buffer("b", (256,))


def _expressions(index: Var) -> st.SearchStrategy[Expr]:
    leaves = st.one_of(
        st.just(LEAF_A[index]),
        st.just(LEAF_B[index]),
        st.floats(-2, 2, width=32).map(lambda v: FloatImm(float(v))),
    )

    def extend(children: st.SearchStrategy[Expr]) -> st.SearchStrategy[Expr]:
        binary = st.tuples(st.sampled_from(["add", "sub", "mul", "min", "max"]), children, children)
        unary = st.tuples(st.sampled_from(["tanh", "sigmoid", "abs"]), children)
        choose = st.tuples(
            st.sampled_from(["lt", "ge", "eq"]), children, children, children, children
        )
        return st.one_of(
            binary.map(lambda t: Binary(t[0], t[1], t[2])),  # type: ignore[arg-type]
            unary.map(lambda t: call(t[0], t[1])),
            choose.map(lambda t: Select(Compare(t[0], t[1], t[2]), t[3], t[4])),  # type: ignore[arg-type]
        )

    return st.recursive(leaves, extend, max_leaves=10)


def _evaluate(e: Expr, env: dict[Buffer, Array]) -> tuple[Array, Array]:
    """(value, magnitude bound) in float64; the bound scales the tolerance."""
    match e:
        case FloatImm(value, _):
            v = np.full(256, value)
            return v, np.abs(v)
        case BufferLoad(buffer, _):
            v = env[buffer].astype(np.float64)
            return v, np.abs(v)
        case Binary(op, a, b):
            (x, mx), (y, my) = _evaluate(a, env), _evaluate(b, env)
            if op == "add":
                return x + y, mx + my
            if op == "sub":
                return x - y, mx + my
            if op == "mul":
                return x * y, mx * my
            if op == "min":
                return np.minimum(x, y), np.maximum(mx, my)
            return np.maximum(x, y), np.maximum(mx, my)
        case Call(name, (a,), _):
            x, mx = _evaluate(a, env)
            if name == "tanh":
                return np.tanh(x), np.ones_like(x) + mx
            if name == "sigmoid":
                return 1 / (1 + np.exp(-x)), np.ones_like(x) + mx
            return np.abs(x), mx
        case Select(Compare(op, a, b), t, f):
            (x, mx), (y, my) = _evaluate(a, env), _evaluate(b, env)
            (p, mp), (q, mq) = _evaluate(t, env), _evaluate(f, env)
            cond = {"lt": x < y, "ge": x >= y, "eq": x == y}[op]
            # Near a tie the GPU and NumPy may pick different sides; widen the bound there.
            close = np.abs(x - y) <= 1e-4 * (1 + mx + my)
            return np.where(cond, p, q), np.where(close, np.abs(p - q) * 1e5, np.maximum(mp, mq))
    raise AssertionError(e)


@given(st.data())
@settings(max_examples=40, deadline=None)
def test_random_elementwise_expressions(data: st.DataObject) -> None:
    i, vi = Var("i"), Var("vi")
    expr = data.draw(_expressions(vi))
    out = Buffer("out", (256,))
    from metal_funcs import block, spatial

    body = bound(
        i, 256, "thread.x", block("out", [spatial(vi, 256, i)], BufferStore(out, expr, (vi,)))
    )
    func = prim_func("expr", [LEAF_A, LEAF_B, out], body)
    rng = np.random.default_rng(0)
    a = rng.uniform(-2, 2, 256).astype(np.float32)
    b = rng.uniform(-2, 2, 256).astype(np.float32)
    result = np.zeros(256, np.float32)
    run_func(func, [a, b, result])
    expected, magnitude = _evaluate(expr, {LEAF_A: a, LEAF_B: b})
    assert np.all(np.abs(result - expected) <= 1e-5 * (1 + magnitude))


@given(divisor=st.integers(-9, 9).filter(lambda d: d != 0))
@settings(max_examples=12, deadline=None)
def test_integer_floor_division_matches_python(divisor: int) -> None:
    x = Buffer("x", (201,), DType.int32)
    q, r = Buffer("q", (201,), DType.int32), Buffer("r", (201,), DType.int32)
    i = Var("i")
    d = IntImm(divisor, DType.int32)
    body = bound(
        i,
        201,
        "thread.x",
        seq([BufferStore(q, x[i] // d, (i,)), BufferStore(r, x[i] % d, (i,))]),
    )
    func = prim_func("divide", [x, q, r], body)
    data = np.arange(-100, 101, dtype=np.int32)
    quotient, remainder = np.zeros_like(data), np.zeros_like(data)
    run_func(func, [data, quotient, remainder])
    np.testing.assert_array_equal(quotient, data // divisor)
    np.testing.assert_array_equal(remainder, data % divisor)


def test_gather_with_int64_indices_and_casts() -> None:
    table = Buffer("table", (10, 4))
    indices = Buffer("indices", (6,), DType.int64)
    out = Buffer("out", (6, 4))
    mask = Buffer("mask", (6, 4), DType.bool_)
    i, j, k = Var("i"), Var("j"), Var("k")
    from metal_funcs import loop

    # Negative indices count from the end, as in ONNX Gather.
    raw = indices[i]
    index = select(lt(raw, IntImm(0, DType.int64)), raw + 10, raw)
    body = seq(
        [
            loop(i, 6, loop(j, 4, BufferStore(out, table[index, j], (i, j)))),
            loop(k, 24, BufferStore(mask, Cast(DType.bool_, out[k // 4, k % 4]), (k // 4, k % 4))),
        ]
    )
    func = prim_func("gather", [table, indices, out, mask], body)
    data = np.arange(40, dtype=np.float32).reshape(10, 4) - 3
    idx = np.array([0, 9, -1, 3, -10, 0], np.int64)
    result = np.zeros((6, 4), np.float32)
    flags = np.zeros((6, 4), np.bool_)
    run_func(func, [data, idx, result, flags])
    np.testing.assert_array_equal(result, data[idx])
    np.testing.assert_array_equal(flags, data[idx] != 0)


def test_let_statements_and_float16() -> None:
    x, out = Buffer("x", (64,), DType.float16), Buffer("out", (64,))
    i, t = Var("i"), Var("t", DType.float32)
    body = bound(
        i,
        64,
        "thread.x",
        LetStmt(t, Cast(DType.float32, x[i]) * 3.0, BufferStore(out, t + t, (i,))),
    )
    func = prim_func("let", [x, out], body)
    data = np.linspace(-2, 2, 64).astype(np.float16)
    result = np.zeros(64, np.float32)
    run_func(func, [data, result])
    np.testing.assert_allclose(result, data.astype(np.float32) * 6, rtol=1e-6)
