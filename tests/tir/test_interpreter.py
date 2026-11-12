"""The interpreter computes what the IR says, with C-like semantics, and runs intrinsics."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from tessera import tir
from tessera.dtype import DType
from tessera.tir import intrin
from tessera.tir.expr import (
    Binary,
    Call,
    Cast,
    Expr,
    FloatImm,
    IntImm,
    Var,
    call,
    maximum,
)
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    PrimFunc,
    Stmt,
)

from .workloads import NDArray, matmul_bias_relu, run


def elementwise(
    op: Callable[[Expr], Expr], xs: list[float] | list[int], in_dtype: DType, out_dtype: DType
) -> NDArray:
    """Interpret ``out[i] = op(X[i])`` over the four values ``xs``."""
    x, out = Buffer("X", (4,), in_dtype), Buffer("out", (4,), out_dtype)
    i = Var("i")
    loop = For(i, IntImm(0), IntImm(4), ForKind.SERIAL, BufferStore(out, op(x[i]), (i,)))
    f = PrimFunc("f", (x, out), BlockRealize((), TRUE, Block("root", (), (), (), loop)))
    xa = np.array(xs, dtype=in_dtype.numpy)
    res = np.zeros(4, dtype=out_dtype.numpy)
    tir.interpret(f, [xa, res])
    return res


def test_matches_numpy_on_a_matmul() -> None:
    w = matmul_bias_relu(9, 7, 5)
    np.testing.assert_allclose(run(w.func, w), w.expected(), rtol=1e-5, atol=1e-5)


def test_integer_division_rounds_like_the_ir_says() -> None:
    xs = [-7, -1, 0, 7]
    i64 = DType.int64
    assert list(elementwise(lambda x: Binary("floordiv", x, IntImm(2)), xs, i64, i64)) == [
        -4,
        -1,
        0,
        3,
    ]
    assert list(elementwise(lambda x: Binary("floormod", x, IntImm(3)), xs, i64, i64)) == [
        2,
        2,
        0,
        1,
    ]
    assert list(elementwise(lambda x: Binary("div", x, IntImm(2)), xs, i64, i64)) == [
        -3,
        0,
        0,
        3,
    ]


def test_float_semantics_follow_ieee() -> None:
    f32 = DType.float32
    out = elementwise(lambda x: x / FloatImm(0.0), [1.0, -1.0, 0.0, 2.0], f32, f32)
    assert out[0] == math.inf
    assert out[1] == -math.inf
    assert math.isnan(out[2])
    out = elementwise(lambda x: call("exp", x), [1000.0, 0.0, -1000.0, 1.0], f32, f32)
    assert out[0] == math.inf
    assert out[1] == 1.0
    assert out[2] == 0.0
    out = elementwise(lambda x: call("log", x), [0.0, -1.0, 1.0, math.e], f32, f32)
    assert out[0] == -math.inf
    assert math.isnan(out[1])
    assert out[2] == 0.0
    out = elementwise(lambda x: maximum(x, 0.0), [math.nan, -1.0, 2.0, 0.0], f32, f32)
    assert math.isnan(out[0])
    assert list(out[1:]) == [0.0, 2.0, 0.0]
    out = elementwise(lambda x: call("sigmoid", x), [-1000.0, 0.0, 1000.0, 1.0], f32, f32)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 1 / (1 + math.exp(-1))], rtol=1e-6)


def test_casts_truncate_and_wrap() -> None:
    out = elementwise(
        lambda x: Cast(DType.int32, x), [-2.7, 2.7, 0.0, 9.9], DType.float32, DType.int32
    )
    assert list(out) == [-2, 2, 0, 9]
    out = elementwise(lambda x: Cast(DType.uint8, x), [-1, 256, 3, 255], DType.int64, DType.uint8)
    assert list(out) == [255, 0, 3, 255]


def test_unwritten_intermediates_read_as_nan() -> None:
    a, out, t = Buffer("A", (4,)), Buffer("out", (4,)), Buffer("T", (4,))
    i = Var("i")
    # Nothing writes T before it is read.
    loop = For(i, IntImm(0), IntImm(4), ForKind.SERIAL, BufferStore(out, t[i] + a[i], (i,)))
    root = Block("root", (), (), (), loop, None, (t,))
    f = PrimFunc("f", (a, out), BlockRealize((), TRUE, root))
    res = np.zeros(4, np.float32)
    tir.interpret(f, [np.ones(4, np.float32), res])
    assert np.isnan(res).all()


def test_registered_intrinsic_runs_its_reference() -> None:
    def scale_rows(dst: intrin.Pointer, src: intrin.Pointer, n: int, factor: float) -> None:
        dst.view((n,), (1,))[:] = src.view((n,), (1,)) * factor

    intrin.register("test.scale_rows", reference=scale_rows, override=True)
    a, b = Buffer("A", (3, 4)), Buffer("B", (3, 4))
    r = Var("r")
    call_stmt = Evaluate(
        Call("test.scale_rows", (b.address(r, 0), a.address(r, 0), IntImm(4), FloatImm(3.0)))
    )
    loop = For(r, IntImm(0), IntImm(3), ForKind.SERIAL, call_stmt)
    f = PrimFunc("k", (a, b), BlockRealize((), TRUE, Block("root", (), (), (), loop)))
    tir.verify(f)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.zeros((3, 4), np.float32)
    tir.interpret(f, [x, y])
    np.testing.assert_array_equal(y, x * 3)


def test_deep_nests_are_outlined() -> None:
    shape = (1,) * 20 + (2, 3, 2, 2)
    a, b = Buffer("A", shape), Buffer("B", shape)
    vs = [Var(f"i{d}") for d in range(24)]
    stmt: Stmt = BufferStore(b, a[tuple(vs)] + 1.0, tuple(vs))
    for v in reversed(vs):
        stmt = For(v, IntImm(0), IntImm(shape[vs.index(v)]), ForKind.SERIAL, stmt)
    f = PrimFunc("deep", (a, b), BlockRealize((), TRUE, Block("root", (), (), (), stmt)))
    assert "def _sub1(" in tir.python_source(f)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    y = np.zeros_like(x)
    tir.interpret(f, [x, y])
    np.testing.assert_array_equal(y, x + 1)


def test_bounds_checking_catches_out_of_range_indices() -> None:
    a, b = Buffer("A", (4,)), Buffer("B", (4,))
    i = Var("i")
    loop = For(i, IntImm(0), IntImm(4), ForKind.SERIAL, BufferStore(b, a[i - 1], (i,)))
    f = PrimFunc("oob", (a, b), BlockRealize((), TRUE, Block("root", (), (), (), loop)))
    x, y = np.ones(4, np.float32), np.zeros(4, np.float32)
    tir.interpret(f, [x, y])  # unchecked: index -1 silently wraps, as in any flat buffer
    with pytest.raises(IndexError, match="outside"):
        tir.interpret(f, [x, y], check_bounds=True)


def test_arguments_are_checked() -> None:
    w = matmul_bias_relu(2, 2, 2)
    a, b, bias = w.inputs
    out = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError, match="expected 4 arguments"):
        tir.interpret(w.func, [a, b, out])
    with pytest.raises(ValueError, match="shape"):
        tir.interpret(w.func, [a, b, bias, np.zeros((2, 3), np.float32)])
    with pytest.raises(ValueError, match="dtype"):
        tir.interpret(w.func, [a, b, bias, np.zeros((2, 2), np.float64)])


def test_read_only_inputs_are_not_written() -> None:
    w = matmul_bias_relu(3, 3, 3)
    for x in w.inputs:
        x.setflags(write=False)
    out = np.zeros((3, 3), np.float32)
    tir.interpret(w.func, [*w.inputs, out])
    np.testing.assert_allclose(out, w.expected(), rtol=1e-5, atol=1e-5)
