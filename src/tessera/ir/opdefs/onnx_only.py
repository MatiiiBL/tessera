"""ONNX operators the importer keeps as they are and ``passes.optimize`` rewrites.

They have shape inference and references, so an imported graph can be
verified and interpreted before optimization, but they never reach fusion
(kind OPAQUE): Gemm becomes MatMul + Add, Flatten, Squeeze and Unsqueeze
become Reshape, and Identity and Dropout disappear.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

import numpy as np

from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.opdefs._util import (
    Spec,
    arity,
    broadcast_shapes,
    check_axes,
    define,
    integer,
    ints,
    keep_dtype,
    normalize_axis,
    normalize_specs,
    number,
    require_numeric,
    require_rank,
    same_dtype,
)
from tessera.ir.ops import OpKind, ShapeError

# -- Gemm ---------------------------------------------------------------------

_GEMM = {
    "alpha": Spec("float", 1.0),
    "beta": Spec("float", 1.0),
    "transA": Spec("int", 0, (0, 1)),
    "transB": Spec("int", 0, (0, 1)),
}


def _gemm_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 2, 3)
    a, b = types[0], types[1]
    dtype = same_dtype(types)
    require_numeric(a)
    require_rank(a, 2, "A")
    require_rank(b, 2, "B")
    m, k = a.shape[::-1] if integer(attrs, "transA") else a.shape
    k2, n = b.shape[::-1] if integer(attrs, "transB") else b.shape
    if k != k2:
        raise ShapeError(f"inner dimensions differ: {a} and {b} give {k} != {k2}")
    if len(types) == 3:
        c = types[2]
        if c.rank > 2 or broadcast_shapes([c.shape, (m, n)]) != (m, n):
            raise ShapeError(f"C {c} does not broadcast to ({m}, {n})")
    return [TensorType((m, n), dtype)]


def _gemm(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    a = xs[0].T if integer(attrs, "transA") else xs[0]
    b = xs[1].T if integer(attrs, "transB") else xs[1]
    y = number(attrs, "alpha") * (a @ b)
    if len(xs) == 3:
        y = y + number(attrs, "beta") * xs[2]
    return [keep_dtype(y, xs[0])]


define(
    "Gemm",
    OpKind.OPAQUE,
    _gemm_infer,
    _gemm,
    specs=_GEMM,
    doc="alpha * A' @ B' + beta * C (ONNX)",
)

# -- Flatten ------------------------------------------------------------------

_FLATTEN = {"axis": Spec("int", 1)}


def _flatten_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    """The axis may equal the rank; a negative axis counts from the rank."""
    out = normalize_specs(_FLATTEN, attrs)
    if types:
        rank, axis = types[0].rank, integer(out, "axis")
        if not -rank <= axis <= rank:
            raise ShapeError(f"axis {axis} is out of range for rank {rank}")
        out["axis"] = axis + rank if axis < 0 else axis
    return out


def _flatten_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    axis = integer(attrs, "axis")
    if not 0 <= axis <= x.rank:
        raise ShapeError(f"axis {axis} must be normalized to [0, {x.rank}]")
    return [TensorType((prod(x.shape[:axis]), prod(x.shape[axis:])), x.dtype)]


def _flatten(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    x = xs[0]
    axis = integer(attrs, "axis")
    return [x.reshape(prod(x.shape[:axis]), prod(x.shape[axis:]))]


define(
    "Flatten",
    OpKind.OPAQUE,
    _flatten_infer,
    _flatten,
    specs=_FLATTEN,
    normalize=_flatten_normalize,
    doc="reshape to (prod(shape[:axis]), prod(shape[axis:]))",
)

# -- Squeeze, Unsqueeze -------------------------------------------------------

_AXES = {"axes": Spec("ints", ())}


def _squeeze_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    out = normalize_specs(_AXES, attrs)
    if len(types) == 1:
        shape = types[0].shape
        if "axes" not in attrs:
            out["axes"] = tuple(i for i, d in enumerate(shape) if d == 1)
        else:
            out["axes"] = tuple(sorted({normalize_axis(a, len(shape)) for a in ints(out, "axes")}))
    return out


def _squeeze_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    axes = ints(attrs, "axes")
    check_axes(axes, x.rank)
    if any(x.shape[a] != 1 for a in axes):
        raise ShapeError(f"cannot squeeze axes {axes} of {x}: dimensions must be 1")
    return [TensorType(tuple(d for i, d in enumerate(x.shape) if i not in axes), x.dtype)]


def _unsqueeze_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    out = normalize_specs({"axes": Spec("ints")}, attrs)
    if len(types) == 1:
        given = ints(out, "axes")
        rank = types[0].rank + len(given)
        axes = sorted(normalize_axis(a, rank) for a in given)
        if len(set(axes)) != len(axes):
            raise ShapeError(f"axes {given} repeat an axis")
        out["axes"] = tuple(axes)
    return out


def _unsqueeze_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    axes = ints(attrs, "axes")
    rank = x.rank + len(axes)
    check_axes(axes, rank)
    dims = iter(x.shape)
    return [TensorType(tuple(1 if i in axes else next(dims) for i in range(rank)), x.dtype)]


define(
    "Squeeze",
    OpKind.OPAQUE,
    _squeeze_infer,
    lambda xs, attrs: [np.squeeze(xs[0], axis=ints(attrs, "axes"))],
    specs=_AXES,
    normalize=_squeeze_normalize,
    doc="remove the unit dimensions `axes` (default: all of them)",
)
define(
    "Unsqueeze",
    OpKind.OPAQUE,
    _unsqueeze_infer,
    lambda xs, attrs: [np.expand_dims(xs[0], ints(attrs, "axes"))],
    specs={"axes": Spec("ints")},
    normalize=_unsqueeze_normalize,
    doc="insert unit dimensions at `axes` (positions in the output)",
)

# -- Identity, Dropout --------------------------------------------------------


def _same_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    return [types[0]]


define("Identity", OpKind.OPAQUE, _same_infer, lambda xs, attrs: [xs[0]], doc="x")
define(
    "Dropout",
    OpKind.OPAQUE,
    _same_infer,
    lambda xs, attrs: [xs[0]],
    doc="x (inference: dropout does nothing)",
)

NAMES = ("Gemm", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout")
