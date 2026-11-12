"""Canonicalization: ONNX-only operators become canonical ones, and identities vanish.

Gemm becomes MatMul + Add with its transposes and scale factors folded into
constant operands; Flatten, Squeeze and Unsqueeze become Reshape; Identity
and Dropout disappear; BatchNorm becomes a per-channel Mul + Add (when it
follows a convolution, ``passes.batchnorm`` folds it into the weights first).
The simplification rules remove no-op nodes and merge chains of reshapes and
transposes.
"""

from __future__ import annotations

from math import prod

import numpy as np

from tessera.ir.graph import Node, Value
from tessera.ir.ops import OpKind
from tessera.passes.rewrite import Rewriter, Rule, scalar


def _identity(rw: Rewriter, node: Node) -> Value | None:
    if node.op in ("Identity", "Dropout"):
        return node.inputs[0]
    return None


def _to_reshape(rw: Rewriter, node: Node) -> Value | None:
    if node.op in ("Flatten", "Squeeze", "Unsqueeze"):
        return rw.op("Reshape", node.inputs[0], name=node.name, shape=node.output.shape)
    return None


def transpose(rw: Rewriter, value: Value, perm: tuple[int, ...], name: str) -> Value:
    """``value`` transposed; a constant is relaid out at compile time (even without folding)."""
    if value.data is not None:
        return rw.const(np.asarray(np.transpose(value.data, perm), order="C"), name)
    return rw.op("Transpose", value, name=name, perm=perm)


def scale(rw: Rewriter, value: Value, factor: float, name: str) -> Value:
    """``value * factor``: computed now if it is a constant and folding is on."""
    if factor == 1.0:
        return value
    if value.data is not None and rw.fold:
        return rw.const((value.data * factor).astype(value.data.dtype), name)
    return rw.op("Mul", value, rw.const(factor, f"{name}_factor", like=value), name=name)


def _gemm(rw: Rewriter, node: Node) -> Value | None:
    """Y = alpha * A' B' + beta * C as MatMul and Add, transposes folded into constants."""
    if node.op != "Gemm":
        return None
    a, b = node.inputs[0], node.inputs[1]
    if node.int_attr("transA"):
        a = transpose(rw, a, (1, 0), f"{node.name}_a")
    if node.int_attr("transB"):
        b = transpose(rw, b, (1, 0), f"{node.name}_weight")
    alpha = node.float_attr("alpha")
    if b.data is not None and rw.fold:
        b, alpha = scale(rw, b, alpha, f"{node.name}_weight"), 1.0
    y = rw.op("MatMul", a, b, name=node.name)
    y = scale(rw, y, alpha, f"{node.name}_alpha")
    if len(node.inputs) == 3:
        c = scale(rw, node.inputs[2], node.float_attr("beta"), f"{node.name}_bias")
        y = rw.op("Add", y, c, name=f"{node.name}_bias")
    return y


def _batchnorm(rw: Rewriter, node: Node) -> Value | None:
    """BatchNorm as x * k + (bias - mean * k) with k = scale / sqrt(var + eps), per channel."""
    if node.op != "BatchNorm" or not all(v.data is not None for v in node.inputs[1:]):
        return None
    x = node.inputs[0]
    gamma, beta, mean, var = (np.asarray(v.data) for v in node.inputs[1:])
    k = gamma / np.sqrt(var + node.float_attr("epsilon"))
    shift = beta - mean * k
    shape = (x.shape[1],) + (1,) * (x.type.rank - 2)
    dtype = x.dtype.numpy
    mul = rw.op("Mul", x, rw.const(k.reshape(shape).astype(dtype), f"{node.name}_scale"))
    return rw.op("Add", mul, rw.const(shift.reshape(shape).astype(dtype), f"{node.name}_shift"))


CANONICALIZE: list[Rule] = [_identity, _to_reshape, _gemm, _batchnorm]


# -- simplification -------------------------------------------------------------


def _noop(rw: Rewriter, node: Node) -> Value | None:
    """Operators that return their input unchanged."""
    x = node.inputs[0] if node.inputs else None
    if x is None or x.type != node.output.type:
        return None
    match node.op:
        case "Reshape" | "Expand" | "Cast":
            return x
        case "Transpose":
            return x if node.ints_attr("perm") == tuple(range(x.type.rank)) else None
        case "Slice":
            full = node.ints_attr("starts") == (0,) * x.type.rank
            return x if full and node.ints_attr("steps") == (1,) * x.type.rank else None
        case "Pad":
            return x if not any(node.ints_attr("pads")) else None
        case "Concat":
            return x if len(node.inputs) == 1 else None
    return None


def _neutral_operand(rw: Rewriter, node: Node) -> Value | None:
    """x + 0, x - 0, x * 1 and x / 1, when the constant does not broadcast x."""
    if node.op not in ("Add", "Sub", "Mul", "Div") or not node.output.dtype.is_float:
        return None
    neutral = 1.0 if node.op in ("Mul", "Div") else 0.0
    for keep, other in ((0, 1), (1, 0)):
        if keep == 1 and node.op in ("Sub", "Div"):
            continue
        x, c = node.inputs[keep], node.inputs[other]
        if x.type == node.output.type and scalar(c) == neutral:
            return x
    return None


def _reshape_chain(rw: Rewriter, node: Node) -> Value | None:
    """Reshape(Reshape(x)) is one Reshape of x."""
    if node.op != "Reshape":
        return None
    src = node.inputs[0].producer
    if src is None or src.op != "Reshape":
        return None
    return rw.op("Reshape", src.inputs[0], name=node.name, shape=node.ints_attr("shape"))


def compose_transposes(rw: Rewriter, node: Node) -> Value | None:
    """Transpose(Transpose(x)) is one Transpose; the identity permutation vanishes."""
    if node.op != "Transpose":
        return None
    perm = node.ints_attr("perm")
    src = node.inputs[0].producer
    if src is not None and src.op == "Transpose":
        inner = src.ints_attr("perm")
        perm = tuple(inner[p] for p in perm)
        if perm == tuple(range(len(perm))):
            return src.inputs[0]
        return rw.op("Transpose", src.inputs[0], name=node.name, perm=perm)
    if perm == tuple(range(len(perm))):
        return node.inputs[0]
    return None


def transpose_as_reshape(rw: Rewriter, node: Node) -> Value | None:
    """A Transpose that only moves size-1 axes keeps the element order: it is a Reshape."""
    if node.op != "Transpose":
        return None
    shape = node.inputs[0].shape
    moved = [p for p in node.ints_attr("perm") if shape[p] != 1]
    if moved != sorted(moved):
        return None
    return rw.op("Reshape", node.inputs[0], name=node.name, shape=node.output.shape)


def _row_view(rw: Rewriter, value: Value) -> Value | None:
    """For ``value = Reshape(x)`` of shape [P, K] where x's leading axes multiply to P and
    the others to K: x viewed as [..., K] (keeping the leading axes), else None."""
    node = value.producer
    if node is None or node.op != "Reshape" or value.type.rank != 2:
        return None
    src = node.inputs[0]
    rows, cols = value.shape
    if rows == 1:
        return None  # a single row: there are no batch axes to keep
    # At least two leading axes must stay, or the view is the 2-D Reshape itself.
    for i in range(2, src.type.rank):
        if prod(src.shape[:i]) == rows and prod(src.shape[i:]) == cols:
            if i == src.type.rank - 1:
                return src
            return rw.op("Reshape", src, name=f"{node.name}_rows", shape=(*src.shape[:i], cols))
    return None


def _matmul_of_rows(rw: Rewriter, node: Node) -> Value | None:
    """MatMul(Reshape(x), w) where the Reshape only merges rows is Reshape(MatMul(x, w)),
    so the product keeps the batch axes and its epilogue meets no Reshape."""
    if node.op != "MatMul" or node.inputs[1].type.rank != 2:
        return None
    src = _row_view(rw, node.inputs[0])
    if src is None:
        return None
    y = rw.op("MatMul", src, node.inputs[1], name=node.name)
    return rw.op("Reshape", y, name=f"{node.name}_rows", shape=node.output.shape)


def _sink_reshape(rw: Rewriter, node: Node) -> Value | None:
    """Elementwise ops (and broadcasts of a last-axis vector) commute with a Reshape that
    keeps the last axis; moving the Reshape down lets matrix products keep their epilogues."""
    kind = node.opdef.kind
    if kind not in (OpKind.ELEMWISE, OpKind.BROADCAST) or node.op == "BatchNorm":
        return None
    reshaped = [v for v in node.inputs if v.producer is not None and v.producer.op == "Reshape"]
    if not reshaped:
        return None
    src = reshaped[0].producer.inputs[0]  # type: ignore[union-attr]
    out = node.output
    if out.type.rank == 0 or src.type.rank == 0 or src.shape[-1] != out.shape[-1]:
        return None
    args: list[Value] = []
    for v in node.inputs:
        producer = v.producer
        if (
            producer is not None
            and producer.op == "Reshape"
            and producer.inputs[0].shape == src.shape
        ):
            if v.shape != out.shape:
                return None
            args.append(producer.inputs[0])
        elif all(d == 1 for d in v.shape[:-1]) and v.type.rank <= out.type.rank:
            # Broadcasts along the last axis only: the same on the source's shape.
            args.append(v)
        else:
            return None
    y = rw.op(node.op, *args, name=node.name, **node.attrs)
    if y.shape != src.shape:
        return None
    return rw.op("Reshape", y, name=f"{node.name}_shape", shape=out.shape)


SIMPLIFY: list[Rule] = [
    _noop,
    _neutral_operand,
    _reshape_chain,
    compose_transposes,
    _matmul_of_rows,
    _sink_reshape,
]
TRANSPOSE_AS_RESHAPE: list[Rule] = [transpose_as_reshape]
