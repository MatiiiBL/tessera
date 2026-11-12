"""Tensorize intrinsics for the NEON and SME microkernels.

Each intrinsic's description is made from the block it will replace: the
group's own update block (after ``decompose_reduction``) with the loops of
the nest being replaced rebound to fresh description loops and every outer
loop replaced by 0. ``Schedule.tensorize`` then checks, as for any
intrinsic, that the nest computes the description (the indices differ only
by the tile's offsets); what this module adds is the recognition of the
blocks the microkernels can compute, the call that replaces them and the
NumPy reference the interpreter runs instead.

- ``gemm``: ``C[.., i, j] += A[.., i, k] * B[.., k, j]`` (or ``B[.., j, k]``,
  read transposed) on a ``tile_m x tile_n`` tile, any reduction length;
  the unit-stride axes must be the last ones. Calls ``tsr_<isa>_sgemm``
  with the tile's valid rows and columns (fewer on an edge tile).
- ``conv2d``: an NHWC convolution with HWIO weights over the whole output
  plane of one image and a strip of output channels, its padding as
  lower_group writes it (clamped indices under a Select). Calls
  ``tsr_<isa>_conv2d``, which packs im2col panels straight from the image.

Intrinsics are named after the structural hash of their description, so a
template can register the same one any number of times, in any process; an
``impl`` therefore takes buffers only from its ``IntrinMatch``, never from
the nest it was first registered for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tessera.backend.c.alloc import affine, sub_exprs
from tessera.backend.c.codegen import fold, substitute
from tessera.backend.c.microkernels import CONV2D_KERNELS, SGEMM_KERNELS
from tessera.backend.c.templates.common import realize
from tessera.dtype import DType
from tessera.plan import LoweredGroup
from tessera.tir import intrin
from tessera.tir.analysis import structural_hash
from tessera.tir.expr import (
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Expr,
    FloatImm,
    IntImm,
    Select,
    Var,
)
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IterKind,
    PrimFunc,
    Stmt,
)

type NDArray = Any


@dataclass(frozen=True)
class Product:
    """The update ``C[out] += A[lhs] * B[rhs]`` of a block, by buffer."""

    out: BufferStore
    lhs: BufferLoad
    rhs: BufferLoad


def product(block: Block) -> Product | None:
    """The block's update if it is a plain multiply-accumulate of three
    buffers (its init, if any, is decomposed before tensorizing)."""
    body = block.body
    if not isinstance(body, BufferStore):
        return None
    value = body.value
    if not (isinstance(value, Binary) and value.op == "add"):
        return None
    acc, term = value.a, value.b
    if not (
        isinstance(acc, BufferLoad) and acc.buffer is body.buffer and acc.indices == body.indices
    ):
        return None
    if not (isinstance(term, Binary) and term.op == "mul"):
        return None
    if not (isinstance(term.a, BufferLoad) and isinstance(term.b, BufferLoad)):
        return None
    if body.buffer.dtype is not DType.float32:
        return None
    return Product(body, term.a, term.b)


def describe(
    sch: Schedule,
    block: str,
    nest: Sequence[Var],
    params: Sequence[Buffer],
    body: Stmt | None = None,
) -> PrimFunc:
    """The description of the nest rooted at ``nest[0]`` (its loops, outermost
    first): the block (with ``body`` in place of its own, if given), bound to
    fresh loops in place of ``nest`` and to 0 in place of every loop outside
    it."""
    r = realize(sch, block)
    fresh = {v: Var(f"d_{v.name}") for v in nest}
    outer = {
        e: IntImm(0, e.dtype)
        for value in r.iter_values
        for e in sub_exprs(value)
        if isinstance(e, Var) and e not in fresh
    }
    mapping: dict[Var, Expr] = {**outer, **fresh}
    # Folded, so that a binding reads d_j rather than 0 * 32 + d_j.
    values = tuple(fold(substitute(v, mapping)) for v in r.iter_values)
    dblock = r.block if body is None else replace(r.block, body=body, reads=(), writes=())
    stmt: Stmt = BlockRealize(values, TRUE, dblock)
    for v in reversed(nest):
        stmt = For(fresh[v], IntImm(0), IntImm(sch.extent(v)), ForKind.SERIAL, stmt)
    root = BlockRealize((), TRUE, Block("root", (), (), (), stmt))
    return PrimFunc(f"{block}_desc", tuple(params), root)


def _register(kind: str, desc: PrimFunc, impl: intrin.Impl, reference: intrin.Reference) -> str:
    name = f"c.{kind}_{structural_hash(desc)[:16]}"
    if name not in intrin.registered():
        intrin.register(name, desc, impl, reference)
    return name


def _dims_using(load: BufferLoad | BufferStore, var: Var) -> list[int]:
    return [d for d, idx in enumerate(load.indices) if any(e is var for e in sub_exprs(idx))]


def _unit(index: Expr, var: Var) -> bool:
    """Whether ``index`` is ``var`` plus terms without it (so its stride is 1)."""
    if index is var:
        return True
    if isinstance(index, Binary) and index.op == "add":
        a, b = index.a, index.b
        return (_unit(a, var) and not _uses(b, var)) or (_unit(b, var) and not _uses(a, var))
    return False


def _uses(e: Expr, var: Var) -> bool:
    return any(x is var for x in sub_exprs(e))


@dataclass(frozen=True)
class GemmLayout:
    a: Buffer
    b: Buffer
    c: Buffer
    a_rows: int  # dimension of A indexed by i
    b_stride_dim: int  # dimension of B whose stride is ldb
    c_rows: int
    transposed: bool  # B indexed [.., j, k]
    a_is_lhs: bool  # A is the left factor of the block's product


def gemm_layout(sch: Schedule, block: str, i: Var, j: Var, k: Var) -> GemmLayout | None:
    """How the block's operands map onto the GEMM microkernel, if they do: A
    indexed by ``i`` in one dimension and by ``k`` in the last, C by ``i`` in
    one and ``j`` in the last, B by ``k`` in one and ``j`` in the last (or the
    other way round: read transposed), every other index free of i, j, k.
    ``i``, ``j``, ``k`` are the block's iteration variables of the rows,
    columns and reduction."""
    r = realize(sch, block)
    p = product(r.block)
    if p is None:
        return None

    def uses(access: BufferLoad | BufferStore) -> tuple[list[int], list[int], list[int]]:
        return _dims_using(access, i), _dims_using(access, j), _dims_using(access, k)

    def last(access: BufferLoad | BufferStore, var: Var) -> bool:
        return _unit(access.indices[-1], var)

    ci, cj, ck = uses(p.out)
    if len(ci) != 1 or cj != [p.out.buffer.ndim - 1] or ck or not last(p.out, j):
        return None
    for lhs, rhs in ((p.lhs, p.rhs), (p.rhs, p.lhs)):
        a_is_lhs = lhs is p.lhs
        ai, aj, ak = uses(lhs)
        bi, bj, bk = uses(rhs)
        if len(ai) != 1 or aj or ak != [lhs.buffer.ndim - 1] or not last(lhs, k) or bi:
            continue
        top = rhs.buffer.ndim - 1
        if bj == [top] and len(bk) == 1 and bk[0] != top and last(rhs, j):
            return GemmLayout(
                lhs.buffer, rhs.buffer, p.out.buffer, ai[0], bk[0], ci[0], False, a_is_lhs
            )
        if bk == [top] and len(bj) == 1 and bj[0] != top and last(rhs, k):
            return GemmLayout(
                lhs.buffer, rhs.buffer, p.out.buffer, ai[0], bj[0], ci[0], True, a_is_lhs
            )
    return None


def gemm_intrinsic(
    sch: Schedule, block: str, layout: GemmLayout, nest: Sequence[Var], isa: str
) -> str:
    """Register the GEMM intrinsic for the ``(rows, columns, reduction)`` nest
    and return its name."""
    red = sch.extent(nest[2])
    kernel = SGEMM_KERNELS[isa][layout.transposed]
    # The description has a buffer of its own per operand, so two operands that
    # read one buffer (Q and K sliced from one projection) still get a region
    # and a pointer each.
    p = product(realize(sch, block).block)
    assert p is not None
    a = Buffer("a", layout.a.shape, layout.a.dtype)
    b = Buffer("b", layout.b.shape, layout.b.dtype)
    c = Buffer("c", layout.c.shape, layout.c.dtype)
    a_load, b_load = (p.lhs, p.rhs) if layout.a_is_lhs else (p.rhs, p.lhs)
    da, db = BufferLoad(a, a_load.indices), BufferLoad(b, b_load.indices)
    term = da * db if layout.a_is_lhs else db * da
    update = BufferStore(c, c[p.out.indices] + term, p.out.indices)
    desc = describe(sch, block, nest, (a, b, c), update)

    def impl(match: intrin.IntrinMatch) -> Stmt:
        (pa, pb, pc), (sa, sb, sc) = match.pointers, match.strides
        # On an edge tile the valid rows and columns are fewer than the tile's:
        # the microkernel takes them as its sizes, so it never reads past them.
        valid_rows, valid_cols, _ = match.valid
        args = (
            valid_rows, valid_cols, IntImm(red),
            pa, IntImm(sa[layout.a_rows]),
            pb, IntImm(sb[layout.b_stride_dim]),
            pc, IntImm(sc[layout.c_rows]),
            IntImm(1),
        )  # fmt: skip
        return Evaluate(Call(kernel.name, args, DType.int32))

    def reference(a: NDArray, b: NDArray, c: NDArray, *extents: int) -> None:
        # Views of the valid part of the tile; their other dimensions have one
        # element.
        av = a.reshape(a.shape[layout.a_rows], a.shape[-1])
        bv = b.reshape(b.shape[layout.b_stride_dim], b.shape[-1])
        cv = c.reshape(c.shape[layout.c_rows], c.shape[-1])
        product = av.astype(np.float64) @ (bv.T if layout.transposed else bv).astype(np.float64)
        cv += product.astype(np.float32)

    return _register(f"{isa}_sgemm", desc, impl, reference)


@dataclass(frozen=True)
class ConvLayout:
    x: Buffer
    w: Buffer
    y: Buffer


def conv_layout(sch: Schedule, block: str, group: LoweredGroup) -> ConvLayout | None:
    """The operands of an NHWC/HWIO convolution's update block, if it is
    exactly what the convolution microkernel computes:
    ``Y[n, oh, ow, o] += X'[n, ih, iw, c] * W[kh, kw, c, o]`` where ``X'`` is a
    load of the image, alone or under a Select with zero (padding), its row
    index ``oh*SH + kh*DH - PT`` (possibly clamped to the image) and its
    column index likewise. Anything else (a prologue inlined into the load,
    another layout) is left to the generated loops."""
    r = realize(sch, block)
    body, dims = r.block.body, group.dims
    spatial = [iv.var for iv in r.block.iter_vars if iv.kind is IterKind.SPATIAL]
    reduce = [iv.var for iv in r.block.iter_vars if iv.kind is IterKind.REDUCE]
    if len(spatial) != 4 or len(reduce) != 3:
        return None
    if dims.get("G", 1) != 1 or not isinstance(body, BufferStore):
        return None
    n, oh, ow, o = spatial
    kh, kw, c = reduce
    value = body.value
    if not (isinstance(value, Binary) and value.op == "add" and isinstance(value.b, Binary)):
        return None
    acc, term = value.a, value.b
    if not (
        isinstance(acc, BufferLoad) and acc.buffer is body.buffer and acc.indices == body.indices
    ):
        return None
    if term.op != "mul" or not isinstance(term.b, BufferLoad):
        return None
    image = term.a
    if isinstance(image, Select):
        if not (isinstance(image.false_value, FloatImm) and image.false_value.value == 0.0):
            return None
        image = image.true_value
    w, y = term.b, body
    if not isinstance(image, BufferLoad) or body.buffer.dtype is not DType.float32:
        return None
    if tuple(w.indices) != (kh, kw, c, o) or tuple(y.indices) != (n, oh, ow, o):
        return None
    x = image
    if len(x.indices) != 4 or x.indices[0] is not n or x.indices[3] is not c:
        return None
    rows = _window(x.indices[1], oh, kh, dims["SH"], dims.get("DH", 1), dims["PT"], dims["H"])
    cols = _window(x.indices[2], ow, kw, dims["SW"], dims.get("DW", 1), dims["PL"], dims["W"])
    if not (rows and cols):
        return None
    if len({id(x.buffer), id(w.buffer), id(y.buffer)}) != 3:
        return None
    if x.buffer.shape[1:] != (dims["H"], dims["W"], dims["C"]):
        return None
    if w.buffer.shape != (dims["KH"], dims["KW"], dims["C"], dims["O"]):
        return None
    return ConvLayout(x.buffer, w.buffer, y.buffer)


def _window(
    index: Expr, out: Var, tap: Var, stride: int, dilation: int, pad: int, size: int
) -> bool:
    """Whether ``index`` is ``out*stride + tap*dilation - pad``, possibly
    clamped to ``[0, size - 1]`` (on either side or both)."""
    if isinstance(index, Binary) and index.op == "min" and index.b == IntImm(size - 1, index.dtype):
        index = index.a
    if isinstance(index, Binary) and index.op == "max" and index.b == IntImm(0, index.dtype):
        index = index.a
    form = affine(index, lambda v: {v: 1})
    want: dict[Expr | None, int] = {out: stride, tap: dilation, None: -pad}
    return form is not None and _nonzero(form) == _nonzero(want)


def _nonzero(form: dict[Expr | None, int]) -> dict[Expr | None, int]:
    return {k: v for k, v in form.items() if v != 0}


def conv_intrinsic(
    sch: Schedule,
    block: str,
    group: LoweredGroup,
    layout: ConvLayout,
    nest: Sequence[Var],
    isa: str,
) -> str:
    """Register the convolution intrinsic for the nest ``(oh, ow, o, kh, kw, c)``
    (the whole output plane, a strip of channels) and return its name."""
    d = group.dims
    oh, ow = sch.extent(nest[0]), sch.extent(nest[1])
    kernel = CONV2D_KERNELS[isa]
    desc = describe(sch, block, nest, (layout.x, layout.w, layout.y))
    geometry = (d["H"], d["W"], d["C"], d["KH"], d["KW"], d["SH"], d["SW"])
    pads = (d["PT"], d["PL"])
    dilations = (d.get("DH", 1), d.get("DW", 1))

    def impl(match: intrin.IntrinMatch) -> Stmt:
        # Only the match may be used here: the intrinsic is shared by every
        # nest with the same structure, whatever its buffers.
        xr = match.regions[0]
        image = AddressOf(xr.buffer, (xr.region[0].min, IntImm(0), IntImm(0), IntImm(0)))
        _, weights, output = match.pointers
        w_strides, y_strides = match.strides[1], match.strides[2]
        args = (
            IntImm(oh * ow), match.valid[2], IntImm(0),
            image, weights, IntImm(w_strides[2]), output, IntImm(y_strides[2]),
            *(IntImm(g) for g in geometry),
            IntImm(pads[0]), IntImm(pads[1]),
            IntImm(dilations[0]), IntImm(dilations[1]), IntImm(ow), IntImm(1),
        )  # fmt: skip
        return Evaluate(Call(kernel.name, args, DType.int32))

    def reference(x: NDArray, w: NDArray, y: NDArray, *extents: int) -> None:
        h, wd, _, kh, kw, sh, sw = geometry
        xv = x.reshape(x.shape[1:]).astype(np.float64)
        strip = w.shape[-1]  # the valid channels of an edge strip
        out = np.zeros((oh, ow, strip))
        for i in range(kh):
            for j in range(kw):
                rows = np.arange(oh) * sh + i * dilations[0] - pads[0]
                cols = np.arange(ow) * sw + j * dilations[1] - pads[1]
                inside = ((rows >= 0) & (rows < h))[:, None] & ((cols >= 0) & (cols < wd))[None]
                rr = np.clip(rows, 0, xv.shape[0] - 1)
                cc = np.clip(cols, 0, xv.shape[1] - 1)
                patch = xv[rr][:, cc] * inside[..., None]
                out += patch @ w[i, j].astype(np.float64)
        yv = y.reshape(oh, ow, strip)
        yv += out.astype(np.float32)

    return _register(f"{isa}_conv2d", desc, impl, reference)
