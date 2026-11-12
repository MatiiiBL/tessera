"""Reductions: sums, means and maxima over axes, Softmax, LayerNorm and 2-D pooling."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.opdefs._util import (
    Spec,
    arity,
    axis_normalizer,
    check_axes,
    check_axis,
    define,
    integer,
    ints,
    keep_dtype,
    normalize_axis,
    normalize_specs,
    number,
    require_float,
    require_numeric,
    require_rank,
    same_dtype,
    string,
)
from tessera.ir.ops import NormalizeFn, OpKind, ShapeError

# -- ReduceSum, ReduceMean, ReduceMax -----------------------------------------

_REDUCE = {"axes": Spec("ints", ()), "keepdims": Spec("int", 1, (0, 1))}


def _reduce_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    """Missing axes mean every axis (ONNX); given axes may be negative."""
    out = normalize_specs(_REDUCE, attrs)
    if len(types) == 1:
        rank = types[0].rank
        if "axes" not in attrs:
            out["axes"] = tuple(range(rank))
        else:
            axes = [normalize_axis(a, rank) for a in ints(out, "axes")]
            if len(set(axes)) != len(axes):
                raise ShapeError(f"axes {ints(out, 'axes')} repeat an axis")
            out["axes"] = tuple(sorted(axes))
    return out


def _reduce_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    require_numeric(x)
    axes = ints(attrs, "axes")
    check_axes(axes, x.rank)
    if integer(attrs, "keepdims"):
        shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
    else:
        shape = tuple(d for i, d in enumerate(x.shape) if i not in axes)
    return [TensorType(shape, x.dtype)]


type ReduceFn = Callable[..., NDArray]


def _reduce_reference(fn: ReduceFn) -> Callable[..., list[NDArray]]:
    def reference(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
        x = xs[0]
        axes = ints(attrs, "axes")
        out = fn(x, axis=axes, keepdims=bool(integer(attrs, "keepdims")))
        return [keep_dtype(np.asarray(out), x)]

    return reference


for _name, _fn, _doc in (
    ("ReduceSum", np.sum, "sum over `axes`"),
    ("ReduceMean", np.mean, "mean over `axes`"),
    ("ReduceMax", np.max, "maximum over `axes`"),
):
    define(
        _name,
        OpKind.REDUCTION,
        _reduce_infer,
        _reduce_reference(_fn),
        specs=_REDUCE,
        normalize=_reduce_normalize,
        doc=_doc,
    )

# -- Softmax ------------------------------------------------------------------

_SOFTMAX = {"axis": Spec("int", -1)}


def _softmax_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    require_float(types[0])
    check_axis(integer(attrs, "axis"), types[0].rank)
    return [types[0]]


def softmax(x: NDArray, axis: int) -> NDArray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return np.asarray(e / np.sum(e, axis=axis, keepdims=True))


define(
    "Softmax",
    OpKind.REDUCTION,
    _softmax_infer,
    lambda xs, attrs: [keep_dtype(softmax(xs[0], integer(attrs, "axis")), xs[0])],
    specs=_SOFTMAX,
    normalize=axis_normalizer(_SOFTMAX),
    doc="exp(x) / sum(exp(x)) along `axis`",
)

# -- LayerNorm ----------------------------------------------------------------

_LAYERNORM = {"axis": Spec("int", -1), "epsilon": Spec("float", 1e-5)}


def _layernorm_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 3)
    x, scale, bias = types
    require_float(x)
    same_dtype(types)
    axis = integer(attrs, "axis")
    check_axis(axis, x.rank)
    normalized = x.shape[axis:]
    for what, t in (("scale", scale), ("bias", bias)):
        if t.shape != normalized:
            raise ShapeError(f"{what} must have shape {normalized} (x.shape[{axis}:]), got {t}")
    return [x]


def _layernorm(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    x, scale, bias = xs
    axes = tuple(range(integer(attrs, "axis"), x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True)
    centered = x - mean
    var = np.mean(centered * centered, axis=axes, keepdims=True)
    y = centered / np.sqrt(var + number(attrs, "epsilon")) * scale + bias
    return [keep_dtype(y, x)]


define(
    "LayerNorm",
    OpKind.REDUCTION,
    _layernorm_infer,
    _layernorm,
    specs=_LAYERNORM,
    normalize=axis_normalizer(_LAYERNORM),
    doc="(x - mean) / sqrt(var + epsilon) * scale + bias over axes axis..rank-1",
)

# -- MaxPool2d, AvgPool2d -----------------------------------------------------

_POOL = {
    "kernel": Spec("ints", length=2),
    "strides": Spec("ints", (1, 1), length=2),
    "pads": Spec("ints", (0, 0, 0, 0), length=4),
    "layout": Spec("str", "NCHW", ("NCHW", "NHWC")),
}
_AVGPOOL = {**_POOL, "count_include_pad": Spec("int", 0, (0, 1))}


def spatial_normalizer(specs: Mapping[str, Spec]) -> NormalizeFn:
    """Pads may be given as (pad_h, pad_w) and per-dimension attributes as one int."""

    def normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
        raw = dict(attrs)
        pads = raw.get("pads")
        if isinstance(pads, tuple | list) and len(pads) == 2:
            raw["pads"] = (*pads, *pads)
        return normalize_specs(specs, raw)

    return normalize


def check_spatial(attrs: Mapping[str, Attr], *keys: str) -> None:
    """Per-dimension attributes have the right length and range."""
    for key in keys:
        values = ints(attrs, key)
        want = 4 if key == "pads" else 2
        low = 0 if key == "pads" else 1
        if len(values) != want or any(v < low for v in values):
            raise ShapeError(f"{key} {values} must be {want} integers >= {low}")


def pool_output(
    size: tuple[int, int],
    kernel: tuple[int, ...],
    strides: tuple[int, ...],
    pads: tuple[int, ...],
    dilations: tuple[int, ...] = (1, 1),
) -> tuple[int, int]:
    """Output height and width of a sliding window (floor rounding)."""
    out = []
    for i in range(2):
        extent = dilations[i] * (kernel[i] - 1) + 1
        padded = size[i] + pads[i] + pads[i + 2]
        if padded < extent:
            raise ShapeError(
                f"window of {kernel[i]} (dilation {dilations[i]}) does not fit "
                f"size {size[i]} with pads {pads[i]}, {pads[i + 2]}"
            )
        out.append((padded - extent) // strides[i] + 1)
    return out[0], out[1]


def _pool_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 1)
    x = types[0]
    require_rank(x, 4)
    require_float(x)
    check_spatial(attrs, "kernel", "strides", "pads")
    kernel, pads = ints(attrs, "kernel"), ints(attrs, "pads")
    for i in range(2):
        if pads[i] >= kernel[i] or pads[i + 2] >= kernel[i]:
            raise ShapeError(f"pads {pads} must be smaller than the kernel {kernel}")
    nhwc = string(attrs, "layout") == "NHWC"
    n, c, h, w = (x.shape[0], x.shape[3], x.shape[1], x.shape[2]) if nhwc else x.shape
    oh, ow = pool_output((h, w), kernel, ints(attrs, "strides"), pads)
    shape = (n, oh, ow, c) if nhwc else (n, c, oh, ow)
    return [TensorType(shape, x.dtype)]


def _windows(x: NDArray, attrs: Mapping[str, Attr], fill: float) -> NDArray:
    """NCHW ``x`` padded with ``fill``: windows of shape [N, C, OH, OW, KH, KW]."""
    kernel, strides, pads = ints(attrs, "kernel"), ints(attrs, "strides"), ints(attrs, "pads")
    widths = ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3]))
    padded = np.pad(x, widths, mode="constant", constant_values=x.dtype.type(fill))
    view = sliding_window_view(padded, kernel, axis=(2, 3))
    return view[:, :, :: strides[0], :: strides[1]]


def _pool_reference(
    reduce: Callable[[NDArray, Mapping[str, Attr]], NDArray],
) -> Callable[..., list[NDArray]]:
    def reference(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
        x = xs[0]
        nhwc = string(attrs, "layout") == "NHWC"
        nchw = np.transpose(x, (0, 3, 1, 2)) if nhwc else x
        y = reduce(nchw, attrs)
        return [keep_dtype(np.transpose(y, (0, 2, 3, 1)) if nhwc else y, x)]

    return reference


def _maxpool(x: NDArray, attrs: Mapping[str, Attr]) -> NDArray:
    return np.asarray(np.max(_windows(x, attrs, -np.inf), axis=(4, 5)))


def _avgpool(x: NDArray, attrs: Mapping[str, Attr]) -> NDArray:
    sums = np.sum(_windows(x, attrs, 0.0), axis=(4, 5))
    if integer(attrs, "count_include_pad"):
        kernel = ints(attrs, "kernel")
        return np.asarray(sums / (kernel[0] * kernel[1]))
    ones = np.ones((1, 1, *x.shape[2:]), x.dtype)
    counts = np.sum(_windows(ones, attrs, 0.0), axis=(4, 5))
    return np.asarray(sums / counts)


define(
    "MaxPool2d",
    OpKind.REDUCTION,
    _pool_infer,
    _pool_reference(_maxpool),
    specs=_POOL,
    normalize=spatial_normalizer(_POOL),
    doc="maximum over kernel windows; padding never wins",
)
define(
    "AvgPool2d",
    OpKind.REDUCTION,
    _pool_infer,
    _pool_reference(_avgpool),
    specs=_AVGPOOL,
    normalize=spatial_normalizer(_AVGPOOL),
    doc="mean over kernel windows; padding counts only with count_include_pad",
)

NAMES = ("ReduceSum", "ReduceMean", "ReduceMax", "Softmax", "LayerNorm", "MaxPool2d", "AvgPool2d")
