"""Canonicalization: ONNX-only operators become canonical ones, and identities vanish.

Gemm becomes MatMul + Add with its transposes and scale factors folded into
constant operands; Flatten, Squeeze and Unsqueeze become Reshape; Identity
and Dropout disappear; BatchNorm becomes a per-channel Mul + Add (when it
follows a convolution, ``passes.batchnorm`` folds it into the weights first).
The simplification rules remove no-op nodes and merge chains of reshapes and
transposes.
"""

from __future__ import annotations

import numpy as np

from tessera.ir.graph import Node, Value
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


SIMPLIFY: list[Rule] = [_noop, _neutral_operand, _reshape_chain, compose_transposes]
TRANSPOSE_AS_RESHAPE: list[Rule] = [transpose_as_reshape]
