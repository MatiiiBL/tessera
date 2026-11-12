"""Elementwise operators: one input, output of the same shape (and type, except Cast)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from tessera.dtype import DType
from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.opdefs._util import (
    Spec,
    Specs,
    arity,
    define,
    keep_dtype,
    number,
    require_float,
    require_numeric,
    string,
)
from tessera.ir.ops import OpKind

_erf_object = np.frompyfunc(math.erf, 1, 1)


def erf(x: NDArray) -> NDArray:
    """The error function, evaluated in float64 (NumPy has no vectorized erf)."""
    return np.asarray(_erf_object(np.asarray(x, np.float64)), np.float64)


def sigmoid(x: NDArray) -> NDArray:
    return np.asarray(1 / (1 + np.exp(-x)))


def gelu(x: NDArray, approximate: str) -> NDArray:
    if approximate == "tanh":
        inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
        return np.asarray(0.5 * x * (1 + np.tanh(inner)))
    return np.asarray(0.5 * x * (1 + erf(x * (1 / math.sqrt(2)))))


def clip(x: NDArray, lo: float, hi: float) -> NDArray:
    if x.dtype.kind in "iub":
        info = np.iinfo(x.dtype) if x.dtype.kind != "b" else None
        if info is not None:
            lo = max(lo, info.min)
            hi = min(hi, info.max)
        low = math.ceil(lo) if math.isfinite(lo) else None
        high = math.floor(hi) if math.isfinite(hi) else None
        y = x if low is None else np.maximum(x, x.dtype.type(low))
        return y if high is None else np.minimum(y, x.dtype.type(high))
    # min(max(x, lo), hi), as ONNX Runtime: when lo > hi every element is hi.
    return np.minimum(np.maximum(x, lo), hi)


type UnaryFn = Callable[[NDArray, Mapping[str, Attr]], Any]

_UNARY: dict[str, tuple[UnaryFn, bool, Specs, str]] = {
    "Relu": (lambda x, a: np.maximum(x, x.dtype.type(0)), False, {}, "max(x, 0)"),
    "Sigmoid": (lambda x, a: sigmoid(x), True, {}, "1 / (1 + exp(-x))"),
    "Tanh": (lambda x, a: np.tanh(x), True, {}, "tanh(x)"),
    "Exp": (lambda x, a: np.exp(x), True, {}, "exp(x)"),
    "Log": (lambda x, a: np.log(x), True, {}, "natural logarithm"),
    "Sqrt": (lambda x, a: np.sqrt(x), True, {}, "square root"),
    "Rsqrt": (lambda x, a: 1 / np.sqrt(x), True, {}, "1 / sqrt(x)"),
    "Erf": (lambda x, a: erf(x), True, {}, "error function"),
    "Neg": (lambda x, a: np.negative(x), False, {}, "-x"),
    "Abs": (lambda x, a: np.abs(x), False, {}, "|x|"),
    "Silu": (lambda x, a: x * sigmoid(x), True, {}, "x * sigmoid(x)"),
    "HardSwish": (
        lambda x, a: x * clip(x * (1 / 6) + 0.5, 0.0, 1.0),
        True,
        {},
        "x * clip(x / 6 + 1/2, 0, 1)",
    ),
    "HardSigmoid": (
        lambda x, a: clip(number(a, "alpha") * x + number(a, "beta"), 0.0, 1.0),
        True,
        {"alpha": Spec("float", 0.2), "beta": Spec("float", 0.5)},
        "clip(alpha * x + beta, 0, 1)",
    ),
    "Gelu": (
        lambda x, a: gelu(x, string(a, "approximate")),
        True,
        {"approximate": Spec("str", "none", ("none", "tanh"))},
        "x * Phi(x), exactly (erf) or with the tanh approximation",
    ),
    "Clip": (
        lambda x, a: clip(x, number(a, "min"), number(a, "max")),
        False,
        {"min": Spec("float", -math.inf), "max": Spec("float", math.inf)},
        "min(max(x, min), max)",
    ),
}


def _unary_infer(float_only: bool) -> Callable[..., list[TensorType]]:
    def infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
        arity(types, 1)
        if float_only:
            require_float(types[0])
        else:
            require_numeric(types[0])
        return [types[0]]

    return infer


def _unary_reference(fn: UnaryFn) -> Callable[..., list[NDArray]]:
    def reference(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
        x = xs[0]
        return [keep_dtype(np.asarray(fn(x, attrs)), x)]

    return reference


for _name, (_fn, _float_only, _specs, _doc) in _UNARY.items():
    define(
        _name,
        OpKind.ELEMWISE,
        _unary_infer(_float_only),
        _unary_reference(_fn),
        specs=_specs,
        doc=_doc,
    )


def _cast_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    return [TensorType(types[0].shape, DType(string(attrs, "to")))]


define(
    "Cast",
    OpKind.ELEMWISE,
    _cast_infer,
    lambda xs, attrs: [np.asarray(xs[0]).astype(DType(string(attrs, "to")).numpy)],
    specs={"to": Spec("str", None, tuple(d.value for d in DType))},
    doc="convert to the element type `to` (float to int truncates toward zero)",
)

NAMES = (*_UNARY, "Cast")
