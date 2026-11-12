"""Anchors: the compute-heavy operators a kernel template is built around."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.opdefs._util import (
    Spec,
    arity,
    broadcast_shapes,
    define,
    integer,
    ints,
    keep_dtype,
    require_float,
    require_numeric,
    require_rank,
    same_dtype,
    string,
)
from tessera.ir.opdefs.reduction import check_spatial, pool_output, spatial_normalizer
from tessera.ir.ops import OpKind, ShapeError

# -- Conv2d -------------------------------------------------------------------

_CONV = {
    "strides": Spec("ints", (1, 1), length=2),
    "pads": Spec("ints", (0, 0, 0, 0), length=4),
    "dilations": Spec("ints", (1, 1), length=2),
    "groups": Spec("int", 1),
    "layout": Spec("str", "NCHW", ("NCHW", "NHWC")),
}


def conv_dims(
    x: TensorType, w: TensorType, layout: str
) -> tuple[int, int, int, int, int, int, int, int]:
    """(N, C, H, W, O, I, KH, KW) of a convolution in either layout."""
    if layout == "NHWC":
        n, h, wd, c = x.shape
        kh, kw, i, o = w.shape
    else:
        n, c, h, wd = x.shape
        o, i, kh, kw = w.shape
    return n, c, h, wd, o, i, kh, kw


def _conv_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 2, 3)
    x, w = types[0], types[1]
    require_rank(x, 4)
    require_rank(w, 4, "weight")
    require_float(x)
    same_dtype(types)
    check_spatial(attrs, "strides", "pads", "dilations")
    layout = string(attrs, "layout")
    groups = integer(attrs, "groups")
    n, c, h, wd, o, i, kh, kw = conv_dims(x, w, layout)
    wlayout = "HWIO" if layout == "NHWC" else "OIHW"
    if groups < 1 or c % groups or o % groups:
        raise ShapeError(f"groups={groups} must divide input channels {c} and output channels {o}")
    if i * groups != c:
        raise ShapeError(
            f"weight {w} ({wlayout}) has {i} input channels per group; "
            f"input {x} with groups={groups} needs {c // groups}"
        )
    if len(types) == 3 and types[2].shape != (o,):
        raise ShapeError(f"bias must have shape ({o},), got {types[2]}")
    oh, ow = pool_output(
        (h, wd), (kh, kw), ints(attrs, "strides"), ints(attrs, "pads"), ints(attrs, "dilations")
    )
    shape = (n, oh, ow, o) if layout == "NHWC" else (n, o, oh, ow)
    return [TensorType(shape, x.dtype)]


def conv2d_nchw(
    x: NDArray,
    w: NDArray,
    strides: tuple[int, ...],
    pads: tuple[int, ...],
    dilations: tuple[int, ...],
    groups: int,
) -> NDArray:
    """Direct convolution: one batched matrix product per kernel offset."""
    n = x.shape[0]
    o, cg, kh, kw = w.shape
    sh, sw = strides
    dh, dw = dilations
    oh, ow = pool_output(x.shape[2:], (kh, kw), strides, pads, dilations)
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    xg = xp.reshape(n, groups, cg, xp.shape[2], xp.shape[3])
    wg = w.reshape(groups, o // groups, cg, kh, kw)
    out = np.zeros((n, groups, o // groups, oh * ow), np.result_type(x, w))
    for i in range(kh):
        for j in range(kw):
            rows = slice(i * dh, i * dh + sh * (oh - 1) + 1, sh)
            cols = slice(j * dw, j * dw + sw * (ow - 1) + 1, sw)
            patch = xg[:, :, :, rows, cols].reshape(n, groups, cg, oh * ow)
            out += wg[:, :, :, i, j] @ patch
    return out.reshape(n, o, oh, ow)


def _conv(xs: Sequence[NDArray], attrs: Mapping[str, Attr]) -> list[NDArray]:
    x, w = xs[0], xs[1]
    nhwc = string(attrs, "layout") == "NHWC"
    if nhwc:
        x = np.transpose(x, (0, 3, 1, 2))
        w = np.transpose(w, (3, 2, 0, 1))
    y = conv2d_nchw(
        x,
        w,
        ints(attrs, "strides"),
        ints(attrs, "pads"),
        ints(attrs, "dilations"),
        integer(attrs, "groups"),
    )
    if len(xs) == 3:
        y = y + xs[2].reshape(1, -1, 1, 1)
    if nhwc:
        y = np.transpose(y, (0, 2, 3, 1))
    return [keep_dtype(np.asarray(y, order="C"), xs[0])]


define(
    "Conv2d",
    OpKind.ANCHOR,
    _conv_infer,
    _conv,
    specs=_CONV,
    normalize=spatial_normalizer(_CONV),
    doc="2-D convolution (x, w[, bias]); w is OIHW for NCHW and HWIO for NHWC",
)

# -- MatMul -------------------------------------------------------------------


def matmul_shape(a: TensorType, b: TensorType) -> tuple[int, ...]:
    """The output shape of NumPy ``matmul`` on operands of rank >= 2."""
    if a.rank < 2 or b.rank < 2:
        raise ShapeError(f"operands must have rank >= 2, got {a} and {b}")
    (m, k), (k2, n) = a.shape[-2:], b.shape[-2:]
    if k != k2:
        raise ShapeError(f"inner dimensions differ: {a} @ {b} ({k} != {k2})")
    batch = broadcast_shapes([a.shape[:-2], b.shape[:-2]])
    return (*batch, m, n)


def _matmul_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    arity(types, 2)
    dtype = same_dtype(types)
    require_numeric(types[0], "operands")
    return [TensorType(matmul_shape(types[0], types[1]), dtype)]


define(
    "MatMul",
    OpKind.ANCHOR,
    _matmul_infer,
    lambda xs, attrs: [keep_dtype(np.matmul(xs[0], xs[1]), xs[0])],
    doc="NumPy matmul on rank >= 2 operands; batch dimensions broadcast",
)

NAMES = ("Conv2d", "MatMul")
