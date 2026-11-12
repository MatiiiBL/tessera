"""Injective operators against ONNX Runtime, with ONNX's lenient attribute spellings."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper
from ort_helpers import assert_close, floats, run_onnx, run_op

RNG = np.random.default_rng(1)
I64 = np.int64


@pytest.mark.parametrize(
    ("shape", "target"),
    [
        ((2, 3, 4), (6, 4)),
        ((2, 3, 4), (-1,)),
        ((2, 3, 4), (0, -1)),
        ((2, 3, 4), (0, 0, 2, 2)),
        ((1, 12), (3, -1, 2)),
        ((5,), (1, 5, 1)),
    ],
)
def test_reshape(shape: tuple[int, ...], target: tuple[int, ...]) -> None:
    x = floats(RNG, *shape)
    node = helper.make_node("Reshape", ["x", "shape"], ["y"])
    constants = {"shape": np.array(target, I64)}
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants)
    assert_close(run_op("Reshape", [x], {"shape": target}), expected)


@pytest.mark.parametrize(
    ("shape", "perm"),
    [((2, 3), None), ((2, 3, 4), (1, 0, 2)), ((1, 3, 5, 7), (0, 2, 3, 1)), ((2, 3, 4), (-1, 0, 1))],
)
def test_transpose(shape: tuple[int, ...], perm: tuple[int, ...] | None) -> None:
    x = floats(RNG, *shape)
    onnx_perm = None if perm is None else [p % len(shape) for p in perm]
    attrs = {} if perm is None else {"perm": onnx_perm}
    node = helper.make_node("Transpose", ["x"], ["y"], **attrs)
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32})
    assert_close(run_op("Transpose", [x], {} if perm is None else {"perm": perm}), expected)


SLICES = [
    # starts, ends, axes, steps
    ((1,), (3,), None, None),
    ((0, 1), (2, 4), (0, 2), None),
    ((-3,), (-1,), (1,), None),
    ((1,), (2**62,), (-1,), (2,)),
    ((-100,), (100,), (0,), (3,)),
    ((2,), (1,), (2,), None),
    ((0, 0, 0), (1, 5, 5), None, (1, 2, 3)),
]


@pytest.mark.parametrize(("starts", "ends", "axes", "steps"), SLICES)
def test_slice(
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    axes: tuple[int, ...] | None,
    steps: tuple[int, ...] | None,
) -> None:
    x = floats(RNG, 3, 5, 6)
    names = ["x", "starts", "ends"]
    constants = {"starts": np.array(starts, I64), "ends": np.array(ends, I64)}
    attrs: dict[str, tuple[int, ...]] = {"starts": starts, "ends": ends}
    if axes is not None or steps is not None:
        names.append("axes")
        constants["axes"] = np.array(axes or range(len(starts)), I64)
        if axes is not None:
            attrs["axes"] = axes
    if steps is not None:
        names.append("steps")
        constants["steps"] = np.array(steps, I64)
        attrs["steps"] = steps
    node = helper.make_node("Slice", names, ["y"])
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants)
    assert_close(run_op("Slice", [x], attrs), expected)


@pytest.mark.parametrize("axis", [0, 1, -1, -3])
def test_concat(axis: int) -> None:
    base = [3, 4, 5]
    parts = []
    for size in (1, 2, 3):
        shape = list(base)
        shape[axis] = size
        parts.append(floats(RNG, *shape))
    names = [f"x{i}" for i in range(len(parts))]
    node = helper.make_node("Concat", names, ["y"], axis=axis)
    (expected,) = run_onnx([node], dict(zip(names, parts, strict=True)), {"y": np.float32})
    assert_close(run_op("Concat", parts, {"axis": axis}), expected)


@pytest.mark.parametrize(
    ("data_shape", "indices", "axis"),
    [
        ((10, 4), [[0, 3], [9, -1]], 0),
        ((3, 7, 2), [6, 0, -7], 1),
        ((3, 7, 2), 1, -1),
        ((5,), [[[4]]], 0),
    ],
)
def test_gather(data_shape: tuple[int, ...], indices: object, axis: int) -> None:
    x = floats(RNG, *data_shape)
    idx = np.asarray(indices, I64)
    node = helper.make_node("Gather", ["x", "i"], ["y"], axis=axis)
    (expected,) = run_onnx([node], {"x": x, "i": idx}, {"y": np.float32})
    assert_close(run_op("Gather", [x, idx], {"axis": axis}), expected)


def test_gather_int32_indices() -> None:
    x = floats(RNG, 6, 3)
    idx = np.array([5, 0], np.int32)
    node = helper.make_node("Gather", ["x", "i"], ["y"])
    (expected,) = run_onnx([node], {"x": x, "i": idx}, {"y": np.float32})
    assert_close(run_op("Gather", [x, idx]), expected)


@pytest.mark.parametrize(
    ("shape", "target"),
    [((3, 1), (3, 4)), ((3, 1), (2, 1, 6)), ((4,), (2, 1)), ((1, 1), (1,)), ((2, 3), (2, 3))],
)
def test_expand_broadcasts_both_ways(shape: tuple[int, ...], target: tuple[int, ...]) -> None:
    x = floats(RNG, *shape)
    node = helper.make_node("Expand", ["x", "shape"], ["y"])
    constants = {"shape": np.array(target, I64)}
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants)
    assert_close(run_op("Expand", [x], {"shape": target}), expected)


@pytest.mark.parametrize(
    ("shape", "pads", "value"),
    [
        ((3, 4), (1, 0, 2, 3), 0.0),
        ((1, 2, 3, 3), (0, 0, 1, 1, 0, 0, 1, 1), -1.5),
        ((5,), (0, 0), 2.0),
    ],
)
def test_pad(shape: tuple[int, ...], pads: tuple[int, ...], value: float) -> None:
    x = floats(RNG, *shape)
    node = helper.make_node("Pad", ["x", "pads", "value"], ["y"], mode="constant")
    constants = {"pads": np.array(pads, I64), "value": np.float32(value)}
    (expected,) = run_onnx([node], {"x": x}, {"y": np.float32}, constants=constants)
    assert_close(run_op("Pad", [x], {"pads": pads, "value": value}), expected)


def test_pad_integers() -> None:
    x = np.arange(6, dtype=I64).reshape(2, 3)
    y = run_op("Pad", [x], {"pads": (1, 0, 0, 1), "value": 7})
    np.testing.assert_array_equal(y, np.pad(x, ((1, 0), (0, 1)), constant_values=7))


def test_pad_of_a_scalar_is_the_scalar() -> None:
    x = np.array(2.5, np.float32)
    assert run_op("Pad", [x], {"pads": ()}) == x
