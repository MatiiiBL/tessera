"""NCHW to NHWC: convolutions and pooling switch layout, and the transposes cancel.

Every NCHW Conv2d and pooling node becomes ``Transpose(NHWC -> NCHW)`` of the
NHWC operator applied to ``Transpose(NCHW -> NHWC)`` of its input, with the
weight relaid out from OIHW to HWIO at compile time. Transposes then sink
toward the outputs through every operator that does not care about layout:
elementwise operators, broadcasts (constant operands are relaid out),
reductions, Softmax, Concat, Pad and Slice (their axes are renumbered), and
LayerNorm when the transpose keeps its normalized axes in place. Two
transposes in a row compose, so a convolution's output transpose meets the
next convolution's input transpose and both disappear. A transpose that only
moves size-1 axes becomes a Reshape. What remains in a CNN is one transpose
per 4-D graph input and output.
"""

from __future__ import annotations

import numpy as np

from tessera.ir.graph import Attr, Node, Value
from tessera.ir.ops import OpKind
from tessera.passes.canonicalize import compose_transposes, transpose, transpose_as_reshape
from tessera.passes.rewrite import Rewriter, Rule

TO_NHWC = (0, 2, 3, 1)
TO_NCHW = (0, 3, 1, 2)
OIHW_TO_HWIO = (2, 3, 1, 0)


def _convert(rw: Rewriter, node: Node) -> Value | None:
    if node.op not in ("Conv2d", "MaxPool2d", "AvgPool2d") or node.str_attr("layout") != "NCHW":
        return None
    args = [transpose(rw, node.inputs[0], TO_NHWC, f"{node.name}_nhwc")]
    if node.op == "Conv2d":
        args.append(transpose(rw, node.inputs[1], OIHW_TO_HWIO, f"{node.name}_hwio"))
        args += node.inputs[2:]
    y = rw.op(node.op, *args, name=node.name, **{**node.attrs, "layout": "NHWC"})
    return rw.op("Transpose", y, name=f"{node.name}_nchw", perm=TO_NCHW)


def _transposed(value: Value) -> tuple[Value, tuple[int, ...]] | None:
    """(source, perm) when ``value`` is a Transpose."""
    node = value.producer
    if node is None or node.op != "Transpose":
        return None
    return node.inputs[0], node.ints_attr("perm")


def _inverse(perm: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(i) for i in np.argsort(perm))


def _wrap(rw: Rewriter, node: Node, y: Value, perm: tuple[int, ...]) -> Value:
    return rw.op("Transpose", y, name=f"{node.name}_t", perm=perm)


def _sink_elementwise(rw: Rewriter, node: Node) -> Value | None:
    if node.opdef.kind is not OpKind.ELEMWISE or len(node.inputs) != 1:
        return None
    found = _transposed(node.inputs[0])
    if found is None:
        return None
    src, perm = found
    return _wrap(rw, node, rw.op(node.op, src, name=node.name, **node.attrs), perm)


def _relaid(rw: Rewriter, value: Value, perm: tuple[int, ...], rank: int) -> Value | None:
    """An operand that, transposed by ``perm``, broadcasts as ``value`` did (None if none)."""
    found = _transposed(value)
    if found is not None and found[1] == perm and value.type.rank == rank:
        return found[0]
    aligned = (1,) * (rank - value.type.rank) + value.shape
    if value.data is not None:
        data = np.transpose(value.data.reshape(aligned), _inverse(perm))
        return rw.const(np.asarray(data, order="C"), f"{value.name}_relaid")
    if all(d == 1 for d in value.shape):
        return value if value.type.rank == rank else rw.op("Reshape", value, shape=aligned)
    return None


def _common_perm(node: Node, rank: int) -> tuple[int, ...] | None:
    perms = {f[1] for v in node.inputs if (f := _transposed(v)) and v.type.rank == rank}
    return perms.pop() if len(perms) == 1 else None


def _sink_broadcast(rw: Rewriter, node: Node) -> Value | None:
    if node.opdef.kind is not OpKind.BROADCAST or node.op == "BatchNorm":
        return None
    rank = node.output.type.rank
    perm = _common_perm(node, rank)
    if perm is None:
        return None
    args = [_relaid(rw, v, perm, rank) for v in node.inputs]
    if any(a is None for a in args):
        return None
    y = rw.op(node.op, *[a for a in args if a is not None], name=node.name, **node.attrs)
    return _wrap(rw, node, y, perm)


def _sink_reduce(rw: Rewriter, node: Node) -> Value | None:
    if node.op not in ("ReduceSum", "ReduceMean", "ReduceMax"):
        return None
    found = _transposed(node.inputs[0])
    if found is None:
        return None
    src, perm = found
    axes = node.ints_attr("axes")
    keep = node.int_attr("keepdims")
    y = rw.op(
        node.op, src, name=node.name, axes=tuple(sorted(perm[a] for a in axes)), keepdims=keep
    )
    if keep:
        return _wrap(rw, node, y, perm)
    remaining = [perm[i] for i in range(len(perm)) if i not in axes]
    order = sorted(remaining)
    return _wrap(rw, node, y, tuple(order.index(r) for r in remaining))


def _sink_axis_op(rw: Rewriter, node: Node) -> Value | None:
    """Softmax, Concat, Pad, Slice and LayerNorm, with their axes renumbered."""
    if node.op not in ("Softmax", "Concat", "Pad", "Slice", "LayerNorm"):
        return None
    rank = node.output.type.rank
    perm = _common_perm(node, rank)
    if perm is None or rank == 0:
        return None
    src = _transposed(node.inputs[0])
    match node.op:
        case "Softmax":
            if src is None:
                return None
            y = rw.op("Softmax", src[0], name=node.name, axis=perm[node.int_attr("axis")])
        case "LayerNorm":
            axis = node.int_attr("axis")
            if src is None or any(perm[i] != i for i in range(axis, rank)):
                return None
            y = rw.op("LayerNorm", src[0], *node.inputs[1:], name=node.name, **node.attrs)
        case "Concat":
            args = [_relaid(rw, v, perm, rank) for v in node.inputs]
            if any(a is None for a in args):
                return None
            axis = perm[node.int_attr("axis")]
            y = rw.op("Concat", *[a for a in args if a is not None], name=node.name, axis=axis)
        case "Pad" | "Slice":
            if src is None:
                return None
            attrs: dict[str, Attr] = {}
            keys = ("pads",) if node.op == "Pad" else ("starts", "ends", "steps")
            for key in keys:
                values = node.ints_attr(key)
                moved: list[int] = []
                for begin in range(0, len(values), rank):
                    chunk = [0] * rank
                    for i, v in enumerate(values[begin : begin + rank]):
                        chunk[perm[i]] = v
                    moved += chunk
                attrs[key] = tuple(moved)
            if node.op == "Pad":
                attrs["value"] = node.attrs["value"]
            y = rw.op(node.op, src[0], name=node.name, **attrs)
        case _:
            return None
    return _wrap(rw, node, y, perm)


LAYOUT: list[Rule] = [
    _convert,
    compose_transposes,
    transpose_as_reshape,
    _sink_elementwise,
    _sink_broadcast,
    _sink_reduce,
    _sink_axis_op,
]
