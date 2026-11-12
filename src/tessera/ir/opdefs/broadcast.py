"""Broadcasting operators: NumPy broadcasting of the inputs, one output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from tessera.dtype import DType
from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.opdefs._util import (
    Spec,
    arity,
    broadcast_shapes,
    define,
    keep_dtype,
    number,
    require_float,
    require_numeric,
    same_dtype,
)
from tessera.ir.ops import OpKind, ShapeError


def divide(a: NDArray, b: NDArray) -> NDArray:
    """True division for floats; for integers, division truncating toward zero (as C)."""
    if a.dtype.kind in "iu":
        q = np.abs(a) // np.abs(b)
        return np.where((a < 0) != (b < 0), -q, q)
    return np.asarray(a / b)


type BinaryFn = Callable[[NDArray, NDArray], NDArray]

_ARITHMETIC: dict[str, tuple[BinaryFn, str]] = {
    "Add": (np.add, "a + b"),
    "Sub": (np.subtract, "a - b"),
    "Mul": (np.multiply, "a * b"),
    "Div": (divide, "a / b (integers truncate toward zero)"),
    "Pow": (np.power, "a ** b"),
    "Maximum": (np.maximum, "max(a, b), NaN if either is NaN"),
    "Minimum": (np.minimum, "min(a, b), NaN if either is NaN"),
}
_COMPARISON: dict[str, tuple[BinaryFn, str]] = {
    "Equal": (np.equal, "a == b"),
    "Less": (np.less, "a < b"),
    "Greater": (np.greater, "a > b"),
}


def _binary_infer(compare: bool, allow_bool: bool) -> Callable[..., list[TensorType]]:
    def infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
        arity(types, 2)
        dtype = same_dtype(types)
        if not allow_bool:
            require_numeric(types[0], "inputs")
        shape = broadcast_shapes([t.shape for t in types])
        return [TensorType(shape, DType.bool_ if compare else dtype)]

    return infer


def _binary_reference(fn: BinaryFn, compare: bool) -> Callable[..., list[NDArray]]:
    def reference(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
        a, b = xs
        out = np.asarray(fn(a, b))
        return [out.astype(np.bool_) if compare else keep_dtype(out, a)]

    return reference


for _name, (_fn, _doc) in _ARITHMETIC.items():
    define(
        _name,
        OpKind.BROADCAST,
        _binary_infer(False, False),
        _binary_reference(_fn, False),
        doc=_doc,
    )
for _name, (_fn, _doc) in _COMPARISON.items():
    define(
        _name,
        OpKind.BROADCAST,
        _binary_infer(True, _name == "Equal"),
        _binary_reference(_fn, True),
        doc=_doc,
    )


def _where_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 3)
    if types[0].dtype is not DType.bool_:
        raise ShapeError(f"condition must be bool, got {types[0]}")
    dtype = same_dtype(types[1:], "both branches")
    return [TensorType(broadcast_shapes([t.shape for t in types]), dtype)]


define(
    "Where",
    OpKind.BROADCAST,
    _where_infer,
    lambda xs, attrs: [np.where(xs[0], xs[1], xs[2]).astype(xs[1].dtype)],
    doc="cond ? a : b",
)


def _batchnorm_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 5)
    x = types[0]
    require_float(x)
    same_dtype(types)
    if x.rank < 2:
        raise ShapeError(f"input must have rank >= 2 (channels on axis 1), got {x}")
    channels = x.shape[1]
    for what, t in zip(("scale", "bias", "mean", "var"), types[1:], strict=True):
        if t.shape != (channels,):
            raise ShapeError(f"{what} must have shape ({channels},), got {t}")
    return [x]


def _batchnorm(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    x, scale, bias, mean, var = xs
    shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    eps = number(attrs, "epsilon")
    inv = 1 / np.sqrt(var.reshape(shape) + eps)
    y = (x - mean.reshape(shape)) * inv * scale.reshape(shape) + bias.reshape(shape)
    return [keep_dtype(y, x)]


define(
    "BatchNorm",
    OpKind.BROADCAST,
    _batchnorm_infer,
    _batchnorm,
    specs={"epsilon": Spec("float", 1e-5)},
    doc="inference batch normalization over channel axis 1 (x, scale, bias, mean, var)",
)

NAMES = (*_ARITHMETIC, *_COMPARISON, "Where", "BatchNorm")
