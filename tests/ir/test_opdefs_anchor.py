"""Conv2d, MatMul and the import-only ONNX operators against ONNX Runtime."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper
from ort_helpers import Array, assert_close, floats, run_onnx, run_op

RNG = np.random.default_rng(3)

CONVS = [
    # (N, C, H, W), (O, KH, KW), strides, pads, dilations, groups
    ((1, 3, 8, 8), (4, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1),
    ((2, 4, 9, 7), (6, 3, 2), (2, 1), (0, 1, 2, 0), (1, 1), 1),
    ((1, 4, 10, 10), (4, 3, 3), (1, 1), (2, 2, 2, 2), (2, 2), 4),
    ((1, 6, 8, 8), (4, 1, 1), (1, 1), (0, 0, 0, 0), (1, 1), 2),
    ((1, 8, 12, 12), (8, 3, 3), (2, 2), (1, 1, 1, 1), (1, 1), 8),
    ((1, 3, 16, 16), (8, 7, 7), (2, 2), (3, 3, 3, 3), (1, 1), 1),
    ((1, 2, 5, 5), (3, 5, 5), (1, 1), (0, 0, 0, 0), (1, 1), 1),
    ((1, 4, 9, 9), (6, 3, 3), (3, 2), (1, 0, 1, 0), (2, 1), 2),
]


def _to_nhwc(x: Array) -> Array:
    return np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1)))


@pytest.mark.parametrize(("xs", "ws", "strides", "pads", "dilations", "groups"), CONVS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_conv2d(
    xs: tuple[int, ...],
    ws: tuple[int, int, int],
    strides: tuple[int, int],
    pads: tuple[int, ...],
    dilations: tuple[int, int],
    groups: int,
    bias: bool,
    layout: str,
) -> None:
    x = floats(RNG, *xs)
    o, kh, kw = ws
    w = floats(RNG, o, xs[1] // groups, kh, kw)
    b = floats(RNG, o)
    names = ["x", "w", "b"] if bias else ["x", "w"]
    node = helper.make_node(
        "Conv", names, ["y"], strides=strides, pads=pads, dilations=dilations, group=groups
    )
    feeds = {"x": x, "w": w, "b": b} if bias else {"x": x, "w": w}
    (expected,) = run_onnx([node], feeds, {"y": np.float32})
    attrs = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "groups": groups,
        "layout": layout,
    }
    if layout == "NHWC":
        args = [_to_nhwc(x), np.ascontiguousarray(np.transpose(w, (2, 3, 1, 0)))]
        args += [b] if bias else []
        assert_close(run_op("Conv2d", args, attrs), _to_nhwc(expected), rtol=1e-4, atol=1e-5)
    else:
        args = [x, w, b] if bias else [x, w]
        assert_close(run_op("Conv2d", args, attrs), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("sa", "sb"),
    [
        ((3, 4), (4, 5)),
        ((2, 3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 1, 3, 4), (5, 4, 6)),
        ((1, 12, 128, 64), (1, 12, 64, 128)),
        ((1, 1), (1, 1)),
    ],
)
def test_matmul(sa: tuple[int, ...], sb: tuple[int, ...]) -> None:
    a, b = floats(RNG, *sa), floats(RNG, *sb)
    node = helper.make_node("MatMul", ["a", "b"], ["y"])
    (expected,) = run_onnx([node], {"a": a, "b": b}, {"y": np.float32})
    assert_close(run_op("MatMul", [a, b]), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("trans_a", [0, 1])
@pytest.mark.parametrize("trans_b", [0, 1])
@pytest.mark.parametrize("c_shape", [None, (5,), (3, 5), (1, 1)])
@pytest.mark.parametrize(("alpha", "beta"), [(1.0, 1.0), (0.5, 2.0)])
def test_gemm(
    trans_a: int, trans_b: int, c_shape: tuple[int, ...] | None, alpha: float, beta: float
) -> None:
    a = floats(RNG, *((4, 3) if trans_a else (3, 4)))
    b = floats(RNG, *((5, 4) if trans_b else (4, 5)))
    feeds = {"a": a, "b": b}
    if c_shape is not None:
        feeds["c"] = floats(RNG, *c_shape)
    attrs = {"transA": trans_a, "transB": trans_b, "alpha": alpha, "beta": beta}
    node = helper.make_node("Gemm", list(feeds), ["y"], **attrs)
    (expected,) = run_onnx([node], feeds, {"y": np.float32}, opset=13)
    assert_close(run_op("Gemm", list(feeds.values()), attrs), expected, rtol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -3])
def test_flatten(axis: int) -> None:
    x = floats(RNG, 2, 3, 4)
    node = helper.make_node("Flatten", ["x"], ["y"], axis=axis)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, opset=13)
    assert_close(run_op("Flatten", [x], {"axis": axis}), expected)


@pytest.mark.parametrize("axes", [None, (0,), (-1,), (0, 3)])
def test_squeeze(axes: tuple[int, ...] | None) -> None:
    x = floats(RNG, 1, 3, 4, 1)
    names = ["x"] if axes is None else ["x", "axes"]
    constants = {} if axes is None else {"axes": np.array(axes, np.int64)}
    node = helper.make_node("Squeeze", names, ["y"])
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants, opset=13)
    assert_close(run_op("Squeeze", [x], {} if axes is None else {"axes": axes}), expected)


@pytest.mark.parametrize("axes", [(0,), (-1,), (1, 3), (-1, 0)])
def test_unsqueeze(axes: tuple[int, ...]) -> None:
    x = floats(RNG, 3, 4)
    node = helper.make_node("Unsqueeze", ["x", "axes"], ["y"])
    constants = {"axes": np.array(axes, np.int64)}
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants, opset=13)
    assert_close(run_op("Unsqueeze", [x], {"axes": axes}), expected)


@pytest.mark.parametrize("op", ["Identity", "Dropout"])
def test_identity_like(op: str) -> None:
    x = floats(RNG, 2, 5)
    (expected,) = run_onnx([helper.make_node(op, ["x"], ["y"])], {"x": x}, {"y": np.float32})
    assert_close(run_op(op, [x]), expected)
