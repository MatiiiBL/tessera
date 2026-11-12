"""Reductions, Softmax, LayerNorm and pooling against ONNX Runtime."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper
from ort_helpers import Array, assert_close, floats, run_onnx, run_op

RNG = np.random.default_rng(2)
ONNX_REDUCE = {"ReduceSum": "ReduceSum", "ReduceMean": "ReduceMean", "ReduceMax": "ReduceMax"}


@pytest.mark.parametrize("op", sorted(ONNX_REDUCE))
@pytest.mark.parametrize("axes", [None, (0,), (-1,), (1, 2), (-1, 0), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reductions(op: str, axes: tuple[int, ...] | None, keepdims: int) -> None:
    x = floats(RNG, 3, 4, 5)
    names = ["x"] if axes is None else ["x", "axes"]
    constants = {} if axes is None else {"axes": np.array(axes, np.int64)}
    node = helper.make_node(ONNX_REDUCE[op], names, ["y"], keepdims=keepdims)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants, opset=18)
    attrs: dict[str, int | tuple[int, ...]] = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    assert_close(run_op(op, [x], attrs), expected, rtol=1e-5, atol=1e-5)


def test_reduce_sum_integers() -> None:
    x = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    np.testing.assert_array_equal(run_op("ReduceSum", [x], {"axes": (1,)}), x.sum(1, keepdims=True))


@pytest.mark.parametrize(("shape", "axis"), [((4, 10), -1), ((2, 3, 5), 1), ((2, 3, 5), 0)])
def test_softmax(shape: tuple[int, ...], axis: int) -> None:
    x = floats(RNG, *shape, scale=5.0)
    node = helper.make_node("Softmax", ["x"], ["y"], axis=axis)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, opset=13)
    assert_close(run_op("Softmax", [x], {"axis": axis}), expected)


def test_softmax_is_stable_for_large_inputs() -> None:
    x = np.array([[1000.0, 1000.0, -1000.0]], np.float32)
    np.testing.assert_allclose(run_op("Softmax", [x]), [[0.5, 0.5, 0.0]])


@pytest.mark.parametrize(("shape", "axis"), [((2, 8), -1), ((2, 4, 16), 2), ((2, 3, 4, 5), 1)])
@pytest.mark.parametrize("epsilon", [None, 1e-12])
def test_layernorm(shape: tuple[int, ...], axis: int, epsilon: float | None) -> None:
    x = floats(RNG, *shape, scale=2.0) + 0.5
    normalized = shape[axis:]
    scale, bias = floats(RNG, *normalized), floats(RNG, *normalized)
    attrs: dict[str, float | int] = {"axis": axis}
    if epsilon is not None:
        attrs["epsilon"] = epsilon
    node = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"], **attrs)
    feeds = {"x": x, "s": scale, "b": bias}
    (expected,) = run_onnx([node], feeds, {"y": np.float32}, opset=17)
    assert_close(run_op("LayerNorm", [x, scale, bias], attrs), expected, rtol=1e-4, atol=1e-5)


POOLS = [
    # kernel, strides, pads (top, left, bottom, right)
    ((2, 2), (2, 2), (0, 0, 0, 0)),
    ((3, 3), (2, 2), (1, 1, 1, 1)),
    ((3, 2), (1, 2), (0, 1, 2, 0)),
    ((7, 7), (1, 1), (0, 0, 0, 0)),
    ((1, 1), (3, 3), (0, 0, 0, 0)),
]


def _nhwc(x: Array) -> Array:
    return np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1)))


@pytest.mark.parametrize(("kernel", "strides", "pads"), POOLS)
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_maxpool(
    kernel: tuple[int, int], strides: tuple[int, int], pads: tuple[int, ...], layout: str
) -> None:
    x = floats(RNG, 2, 3, 7, 9)
    node = helper.make_node(
        "MaxPool", ["x"], ["y"], kernel_shape=kernel, strides=strides, pads=pads
    )
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, opset=12)
    attrs = {"kernel": kernel, "strides": strides, "pads": pads, "layout": layout}
    if layout == "NHWC":
        assert_close(run_op("MaxPool2d", [_nhwc(x)], attrs), _nhwc(expected))
    else:
        assert_close(run_op("MaxPool2d", [x], attrs), expected)


@pytest.mark.parametrize(("kernel", "strides", "pads"), POOLS)
@pytest.mark.parametrize("count_include_pad", [0, 1])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_avgpool(
    kernel: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, ...],
    count_include_pad: int,
    layout: str,
) -> None:
    x = floats(RNG, 1, 4, 7, 9)
    node = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        kernel_shape=kernel,
        strides=strides,
        pads=pads,
        count_include_pad=count_include_pad,
    )
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, opset=11)
    attrs = {
        "kernel": kernel,
        "strides": strides,
        "pads": pads,
        "layout": layout,
        "count_include_pad": count_include_pad,
    }
    if layout == "NHWC":
        assert_close(run_op("AvgPool2d", [_nhwc(x)], attrs), _nhwc(expected))
    else:
        assert_close(run_op("AvgPool2d", [x], attrs), expected)


def test_pool_accepts_symmetric_pads_and_scalar_strides() -> None:
    x = floats(RNG, 1, 2, 6, 6)
    a = run_op("MaxPool2d", [x], {"kernel": 3, "strides": 2, "pads": (1, 1)})
    b = run_op("MaxPool2d", [x], {"kernel": (3, 3), "strides": (2, 2), "pads": (1, 1, 1, 1)})
    np.testing.assert_array_equal(a, b)
