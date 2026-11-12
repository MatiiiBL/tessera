"""Injective operators: every output element is one input element (or a constant)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

import numpy as np

from tessera.dtype import DType
from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.opdefs._util import (
    Spec,
    arity,
    axis_normalizer,
    broadcast_shapes,
    check_axis,
    check_perm,
    define,
    integer,
    ints,
    normalize_axis,
    normalize_specs,
    number,
    same_dtype,
)
from tessera.ir.ops import OpKind, ShapeError

# -- Reshape ------------------------------------------------------------------

_RESHAPE = {"shape": Spec("ints")}


def _reshape_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    """ONNX rules: 0 copies the input dimension at the same index, one -1 is inferred."""
    out = normalize_specs(_RESHAPE, attrs)
    if len(types) != 1:
        return out
    src = types[0].shape
    shape = list(ints(out, "shape"))
    for i, d in enumerate(shape):
        if d == 0:
            if i >= len(src):
                raise ShapeError(f"shape {tuple(shape)} copies dimension {i} of {types[0]}")
            shape[i] = src[i]
        elif d < -1:
            raise ShapeError(f"invalid dimension {d} in shape {tuple(shape)}")
    if shape.count(-1) > 1:
        raise ShapeError(f"shape {tuple(shape)} has more than one -1")
    if -1 in shape:
        known = prod(d for d in shape if d != -1)
        if known == 0 or prod(src) % known:
            raise ShapeError(f"cannot reshape {types[0]} to {tuple(shape)}")
        shape[shape.index(-1)] = prod(src) // known
    out["shape"] = tuple(shape)
    return out


def _reshape_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    shape = ints(attrs, "shape")
    if any(d < 0 for d in shape):
        raise ShapeError(f"shape {shape} must be explicit (no -1)")
    if prod(shape) != types[0].numel:
        raise ShapeError(f"cannot reshape {types[0]} ({types[0].numel} elements) to {shape}")
    return [TensorType(shape, types[0].dtype)]


define(
    "Reshape",
    OpKind.INJECTIVE,
    _reshape_infer,
    lambda xs, attrs: [np.reshape(xs[0], ints(attrs, "shape"))],
    specs=_RESHAPE,
    normalize=_reshape_normalize,
    doc="same elements in row-major order, new shape",
)

# -- Transpose ----------------------------------------------------------------

_TRANSPOSE = {"perm": Spec("ints", ())}


def _transpose_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    out = normalize_specs(_TRANSPOSE, attrs)
    if len(types) == 1:
        rank = types[0].rank
        perm = ints(out, "perm")
        if "perm" not in attrs:
            perm = tuple(reversed(range(rank)))
        out["perm"] = tuple(normalize_axis(p, rank, "perm entry") for p in perm)
    return out


def _transpose_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    perm = ints(attrs, "perm")
    check_perm(perm, types[0].rank)
    return [TensorType(tuple(types[0].shape[p] for p in perm), types[0].dtype)]


define(
    "Transpose",
    OpKind.INJECTIVE,
    _transpose_infer,
    lambda xs, attrs: [np.transpose(xs[0], ints(attrs, "perm"))],
    specs=_TRANSPOSE,
    normalize=_transpose_normalize,
    doc="out.shape[i] = x.shape[perm[i]] (default: reversed axes)",
)

# -- Slice --------------------------------------------------------------------

_SLICE = {"starts": Spec("ints"), "ends": Spec("ints"), "steps": Spec("ints")}
_SLICE_LENIENT = {**_SLICE, "steps": Spec("ints", ()), "axes": Spec("ints", ())}


def _slice_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    """ONNX rules: optional axes and steps, negative indices count from the end, clamping."""
    raw = normalize_specs(_SLICE_LENIENT, attrs)
    starts, ends = ints(raw, "starts"), ints(raw, "ends")
    if len(types) != 1:
        return {k: raw[k] for k in _SLICE}
    dims = types[0].shape
    rank = len(dims)
    axes = ints(raw, "axes") if "axes" in attrs else tuple(range(len(starts)))
    steps = ints(raw, "steps") if "steps" in attrs else (1,) * len(starts)
    if not len(starts) == len(ends) == len(axes) == len(steps):
        raise ShapeError(
            f"starts {starts}, ends {ends}, axes {axes} and steps {steps} differ in length"
        )
    out_starts, out_ends, out_steps = [0] * rank, list(dims), [1] * rank
    seen: set[int] = set()
    for axis, start, end, step in zip(axes, starts, ends, steps, strict=True):
        axis = normalize_axis(axis, rank)
        if axis in seen:
            raise ShapeError(f"axis {axis} is sliced twice")
        seen.add(axis)
        if step < 1:
            raise ShapeError(f"step {step} on axis {axis}: only positive steps are supported")
        d = dims[axis]
        start = min(max(start + d if start < 0 else start, 0), d)
        end = min(max(end + d if end < 0 else end, 0), d)
        out_starts[axis], out_ends[axis], out_steps[axis] = start, max(end, start), step
    return {"starts": tuple(out_starts), "ends": tuple(out_ends), "steps": tuple(out_steps)}


def _slice_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    starts, ends, steps = ints(attrs, "starts"), ints(attrs, "ends"), ints(attrs, "steps")
    if not len(starts) == len(ends) == len(steps) == x.rank:
        raise ShapeError(f"starts, ends and steps need one entry per axis of {x}")
    shape = []
    for axis, (start, end, step, d) in enumerate(zip(starts, ends, steps, x.shape, strict=True)):
        if step < 1 or not 0 <= start <= end <= d:
            raise ShapeError(
                f"axis {axis}: need 0 <= start <= end <= {d} and step >= 1, "
                f"got start={start} end={end} step={step}"
            )
        shape.append((end - start + step - 1) // step)
    return [TensorType(tuple(shape), x.dtype)]


def _slice(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    index = tuple(
        slice(s, e, t)
        for s, e, t in zip(
            ints(attrs, "starts"), ints(attrs, "ends"), ints(attrs, "steps"), strict=True
        )
    )
    return [xs[0][index]]


define(
    "Slice",
    OpKind.INJECTIVE,
    _slice_infer,
    _slice,
    specs=_SLICE,
    normalize=_slice_normalize,
    doc="x[starts:ends:steps] on every axis",
)

# -- Concat -------------------------------------------------------------------

_CONCAT = {"axis": Spec("int")}


def _concat_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    if not types:
        raise ShapeError("expects at least one input")
    dtype = same_dtype(types)
    axis = integer(attrs, "axis")
    first = types[0]
    check_axis(axis, first.rank)
    for t in types[1:]:
        if t.rank != first.rank or any(
            a != b for i, (a, b) in enumerate(zip(t.shape, first.shape, strict=True)) if i != axis
        ):
            listed = ", ".join(map(str, types))
            raise ShapeError(f"inputs {listed} differ outside axis {axis}")
    shape = list(first.shape)
    shape[axis] = sum(t.shape[axis] for t in types)
    return [TensorType(tuple(shape), dtype)]


define(
    "Concat",
    OpKind.INJECTIVE,
    _concat_infer,
    lambda xs, attrs: [np.concatenate(list(xs), axis=integer(attrs, "axis"))],
    specs=_CONCAT,
    normalize=axis_normalizer(_CONCAT),
    doc="inputs joined along `axis`",
)

# -- Gather -------------------------------------------------------------------

_GATHER = {"axis": Spec("int", 0)}


def _gather_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 2)
    data, indices = types
    axis = integer(attrs, "axis")
    check_axis(axis, data.rank)
    if indices.dtype not in (DType.int64, DType.int32):
        raise ShapeError(f"indices must be int64 or int32, got {indices}")
    shape = data.shape[:axis] + indices.shape + data.shape[axis + 1 :]
    return [TensorType(shape, data.dtype)]


define(
    "Gather",
    OpKind.INJECTIVE,
    _gather_infer,
    lambda xs, attrs: [np.take(xs[0], xs[1], axis=integer(attrs, "axis"))],
    specs=_GATHER,
    normalize=axis_normalizer(_GATHER),
    doc="data indexed along `axis` by indices (negative indices count from the end)",
)

# -- Expand -------------------------------------------------------------------

_EXPAND = {"shape": Spec("ints")}


def _expand_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    """ONNX Expand broadcasts both ways; the normalized shape is the output shape."""
    out = normalize_specs(_EXPAND, attrs)
    if len(types) == 1:
        out["shape"] = broadcast_shapes([types[0].shape, ints(out, "shape")])
    return out


def _expand_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    shape = ints(attrs, "shape")
    if len(shape) < types[0].rank or broadcast_shapes([types[0].shape, shape]) != shape:
        raise ShapeError(f"{types[0]} does not broadcast to {shape}")
    return [TensorType(shape, types[0].dtype)]


define(
    "Expand",
    OpKind.INJECTIVE,
    _expand_infer,
    lambda xs, attrs: [np.broadcast_to(xs[0], ints(attrs, "shape"))],
    specs=_EXPAND,
    normalize=_expand_normalize,
    doc="x broadcast to `shape`",
)

# -- Pad ----------------------------------------------------------------------

_PAD = {"pads": Spec("ints"), "value": Spec("float", 0.0)}


def _pad_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    pads = ints(attrs, "pads")
    if len(pads) != 2 * x.rank or any(p < 0 for p in pads):
        raise ShapeError(f"pads {pads} must be 2 x rank = {2 * x.rank} non-negative entries")
    shape = tuple(d + pads[i] + pads[i + x.rank] for i, d in enumerate(x.shape))
    return [TensorType(shape, x.dtype)]


def _pad(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    x = xs[0]
    if x.ndim == 0:
        return [x.copy()]
    pads = ints(attrs, "pads")
    widths = [(pads[i], pads[i + x.ndim]) for i in range(x.ndim)]
    value = x.dtype.type(number(attrs, "value"))
    return [np.pad(x, widths, mode="constant", constant_values=value)]


define(
    "Pad",
    OpKind.INJECTIVE,
    _pad_infer,
    _pad,
    specs=_PAD,
    doc="constant padding; pads lists all begins, then all ends (as ONNX)",
)

NAMES = ("Reshape", "Transpose", "Slice", "Concat", "Gather", "Expand", "Pad")
