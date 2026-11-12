"""Elementwise and broadcasting operators against ONNX Runtime."""

from __future__ import annotations

import math

import numpy as np
import pytest
from onnx import helper
from ort_helpers import assert_close, floats, run_onnx, run_op

RNG = np.random.default_rng(0)
SHAPES = [(), (7,), (2, 3, 5), (1, 4, 3, 3)]

UNARY = ["Relu", "Sigmoid", "Tanh", "Exp", "Erf", "Neg", "Abs", "HardSwish"]


@pytest.mark.parametrize("op", UNARY)
@pytest.mark.parametrize("shape", SHAPES)
def test_unary_matches_onnx(op: str, shape: tuple[int, ...]) -> None:
    x = floats(RNG, *shape, scale=3.0)
    (expected,) = run_onnx([helper.make_node(op, ["x"], ["y"])], {"x": x}, {"y": np.float32})
    assert_close(run_op(op, [x]), expected)


@pytest.mark.parametrize("op", ["Log", "Sqrt"])
def test_unary_on_positive_inputs(op: str) -> None:
    x = np.abs(floats(RNG, 3, 17)) + 1e-3
    (expected,) = run_onnx([helper.make_node(op, ["x"], ["y"])], {"x": x}, {"y": np.float32})
    assert_close(run_op(op, [x]), expected)


def test_log_and_sqrt_of_edge_values() -> None:
    x = np.array([0.0, -1.0, np.inf, 1.0], np.float32)
    for op in ("Log", "Sqrt"):
        (expected,) = run_onnx([helper.make_node(op, ["x"], ["y"])], {"x": x}, {"y": np.float32})
        assert_close(run_op(op, [x]), expected)


def test_rsqrt_is_reciprocal_of_sqrt() -> None:
    x = np.abs(floats(RNG, 4, 9)) + 0.1
    nodes = [helper.make_node("Sqrt", ["x"], ["s"]), helper.make_node("Reciprocal", ["s"], ["y"])]
    (expected,) = run_onnx(nodes, {"x": x}, {"y": np.float32})
    assert_close(run_op("Rsqrt", [x]), expected)


def test_silu_is_x_times_sigmoid() -> None:
    x = floats(RNG, 5, 11, scale=4.0)
    nodes = [helper.make_node("Sigmoid", ["x"], ["s"]), helper.make_node("Mul", ["x", "s"], ["y"])]
    (expected,) = run_onnx(nodes, {"x": x}, {"y": np.float32})
    assert_close(run_op("Silu", [x]), expected)


@pytest.mark.parametrize(("alpha", "beta"), [(None, None), (0.5, 0.1), (1 / 6, 0.5)])
def test_hard_sigmoid(alpha: float | None, beta: float | None) -> None:
    x = floats(RNG, 6, 7, scale=4.0)
    attrs = {k: v for k, v in (("alpha", alpha), ("beta", beta)) if v is not None}
    node = helper.make_node("HardSigmoid", ["x"], ["y"], **attrs)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32})
    assert_close(run_op("HardSigmoid", [x], attrs), expected)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(approximate: str) -> None:
    x = floats(RNG, 3, 64, scale=3.0)
    node = helper.make_node("Gelu", ["x"], ["y"], approximate=approximate)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, opset=20)
    assert_close(run_op("Gelu", [x], {"approximate": approximate}), expected)


def test_erf_is_accurate_in_float64() -> None:
    x = np.linspace(-6, 6, 101)
    y = run_op("Erf", [x.astype(np.float32)])
    np.testing.assert_allclose(y, [math.erf(float(v)) for v in x.astype(np.float32)], atol=1e-7)


@pytest.mark.parametrize(
    ("lo", "hi"), [(None, None), (0.0, 6.0), (-1.0, None), (None, 0.5), (2.0, 1.0)]
)
def test_clip(lo: float | None, hi: float | None) -> None:
    x = floats(RNG, 4, 8, scale=4.0)
    names = ["x", "lo" if lo is not None else "", "hi" if hi is not None else ""]
    while names[-1] == "":
        names.pop()
    constants = {k: np.float32(v) for k, v in (("lo", lo), ("hi", hi)) if v is not None}
    node = helper.make_node("Clip", names, ["y"])
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants)
    attrs = {k: v for k, v in (("min", lo), ("max", hi)) if v is not None}
    assert_close(run_op("Clip", [x], attrs), expected)


def test_clip_on_integers() -> None:
    x = np.arange(-5, 6, dtype=np.int64)
    np.testing.assert_array_equal(run_op("Clip", [x], {"min": -2.5, "max": 3}), np.clip(x, -2, 3))


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (np.float32, "int64"),
        (np.float32, "bool"),
        (np.int64, "float32"),
        (np.bool_, "float32"),
        (np.float32, "float16"),
        (np.int32, "int64"),
    ],
)
def test_cast(src: type, dst: str) -> None:
    x = (floats(RNG, 3, 5, scale=5.0)).astype(src)
    to = int(helper.np_dtype_to_tensor_dtype(np.dtype(dst)))
    node = helper.make_node("Cast", ["x"], ["y"], to=to)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.dtype(dst)})
    assert_close(run_op("Cast", [x], {"to": dst}), expected)


BINARY = {
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "Maximum": "Max",
    "Minimum": "Min",
}
BROADCAST_PAIRS = [
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),
    ((2, 1, 4), (3, 1)),
    ((1,), (2, 3, 5)),
    ((), (4, 2)),
    ((1, 6, 1, 1), (2, 6, 5, 5)),
]


@pytest.mark.parametrize("op", sorted(BINARY))
@pytest.mark.parametrize(("sa", "sb"), BROADCAST_PAIRS)
def test_binary_broadcasting(op: str, sa: tuple[int, ...], sb: tuple[int, ...]) -> None:
    a, b = floats(RNG, *sa), floats(RNG, *sb)
    if op == "Div":
        b = np.where(np.abs(b) < 0.1, np.float32(0.5), b)
    node = helper.make_node(BINARY[op], ["a", "b"], ["y"])
    (expected,) = run_onnx([node], {"a": a, "b": b}, {"y": np.float32})
    assert_close(run_op(op, [a, b]), expected)


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div"])
def test_integer_arithmetic(op: str) -> None:
    a = RNG.integers(-20, 20, (4, 5)).astype(np.int64)
    b = RNG.integers(1, 7, (5,)).astype(np.int64) * RNG.choice([-1, 1], (5,))
    node = helper.make_node(op, ["a", "b"], ["y"])
    (expected,) = run_onnx([node], {"a": a, "b": b}, {"y": np.int64})
    assert_close(run_op(op, [a, b]), expected)


def test_pow() -> None:
    a = np.abs(floats(RNG, 3, 4)) + 0.5
    for b in (np.float32(2.0), np.float32(0.5), floats(RNG, 4)):
        node = helper.make_node("Pow", ["a", "b"], ["y"])
        (expected,) = run_onnx([node], {"a": a, "b": np.asarray(b)}, {"y": np.float32})
        assert_close(run_op("Pow", [a, np.asarray(b)]), expected, rtol=1e-5)


@pytest.mark.parametrize("op", ["Equal", "Less", "Greater"])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_comparisons(op: str, dtype: type) -> None:
    a = RNG.integers(0, 3, (4, 1, 3)).astype(dtype)
    b = RNG.integers(0, 3, (5, 1)).astype(dtype)
    node = helper.make_node(op, ["a", "b"], ["y"])
    (expected,) = run_onnx([node], {"a": a, "b": b}, {"y": np.bool_})
    assert_close(run_op(op, [a, b]), expected)


def test_equal_on_bools() -> None:
    a = np.array([True, False, True])
    b = np.array([True])
    np.testing.assert_array_equal(run_op("Equal", [a, b]), [True, False, True])


@pytest.mark.parametrize(
    ("sc", "sa", "sb"), [((3, 4), (3, 4), (3, 4)), ((1, 4), (3, 1), ()), ((2, 1, 1), (4,), (3, 4))]
)
def test_where(sc: tuple[int, ...], sa: tuple[int, ...], sb: tuple[int, ...]) -> None:
    c = RNG.integers(0, 2, sc).astype(np.bool_)
    a, b = floats(RNG, *sa), floats(RNG, *sb)
    node = helper.make_node("Where", ["c", "a", "b"], ["y"])
    (expected,) = run_onnx([node], {"c": c, "a": a, "b": b}, {"y": np.float32})
    assert_close(run_op("Where", [c, a, b]), expected)


@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 5, 5), (2, 3, 7)])
@pytest.mark.parametrize("epsilon", [None, 1e-3])
def test_batchnorm(shape: tuple[int, ...], epsilon: float | None) -> None:
    channels = shape[1]
    x = floats(RNG, *shape)
    scale, bias, mean = floats(RNG, channels), floats(RNG, channels), floats(RNG, channels)
    var = np.abs(floats(RNG, channels)) + 0.1
    attrs = {} if epsilon is None else {"epsilon": epsilon}
    node = helper.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], **attrs)
    feeds = {"x": x, "s": scale, "b": bias, "m": mean, "v": var}
    (expected,) = run_onnx([node], feeds, {"y": np.float32}, opset=15)
    assert_close(run_op("BatchNorm", [x, scale, bias, mean, var], attrs), expected, rtol=1e-5)
