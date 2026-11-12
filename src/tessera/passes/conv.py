"""Convolution rewrites: BatchNorm folding and splitting of grouped convolutions."""

from __future__ import annotations

import numpy as np

from tessera.ir.graph import Node, Value
from tessera.passes.rewrite import Rewriter, Rule


def _fold_batchnorm(rw: Rewriter, node: Node) -> Value | None:
    """BatchNorm(Conv2d(x, w, b)) is Conv2d(x, w * k, (b - mean) * k + beta), k = scale / std."""
    if node.op != "BatchNorm" or not rw.fold:
        return None
    conv = node.inputs[0].producer
    if conv is None or conv.op != "Conv2d" or conv.str_attr("layout") != "NCHW":
        return None
    if not rw.single_reader(node.inputs[0]):
        return None
    w = conv.inputs[1].data
    params = [v.data for v in node.inputs[1:]]
    bias = conv.inputs[2].data if len(conv.inputs) == 3 else np.zeros(conv.output.shape[1])
    if w is None or bias is None or any(p is None for p in params):
        return None
    gamma, beta, mean, var = (np.asarray(p, np.float64) for p in params)
    k = gamma / np.sqrt(var + node.float_attr("epsilon"))
    dtype = w.dtype
    weight = rw.const((w * k[:, None, None, None]).astype(dtype), f"{conv.name}_weight")
    shift = rw.const(((bias - mean) * k + beta).astype(dtype), f"{conv.name}_bias")
    return rw.op("Conv2d", conv.inputs[0], weight, shift, name=conv.name, **conv.attrs)


def is_depthwise(node: Node) -> bool:
    groups = node.int_attr("groups")
    nhwc = node.str_attr("layout") == "NHWC"
    channels = node.inputs[0].shape[3 if nhwc else 1]
    return groups == channels == node.output.shape[3 if nhwc else 1]


def _split_groups(rw: Rewriter, node: Node) -> Value | None:
    """A grouped convolution (not depthwise) as one convolution per group, concatenated."""
    if node.op != "Conv2d" or node.int_attr("groups") == 1 or is_depthwise(node):
        return None
    groups = node.int_attr("groups")
    nhwc = node.str_attr("layout") == "NHWC"
    channel_axis = 3 if nhwc else 1
    weight_axis = 3 if nhwc else 0
    x, w = node.inputs[0], node.inputs[1]
    per_in = x.shape[channel_axis] // groups
    per_out = node.output.shape[channel_axis] // groups

    def part(value: Value, axis: int, size: int, g: int, name: str) -> Value:
        start, end = g * size, (g + 1) * size
        if value.data is not None:
            index = [slice(None)] * value.type.rank
            index[axis] = slice(start, end)
            return rw.const(np.asarray(value.data[tuple(index)], order="C"), name)
        return rw.op("Slice", value, name=name, starts=(start,), ends=(end,), axes=(axis,))

    attrs = {**node.attrs, "groups": 1}
    outs = []
    for g in range(groups):
        args = [
            part(x, channel_axis, per_in, g, f"{node.name}_in{g}"),
            part(w, weight_axis, per_out, g, f"{node.name}_w{g}"),
        ]
        if len(node.inputs) == 3:
            args.append(part(node.inputs[2], 0, per_out, g, f"{node.name}_b{g}"))
        outs.append(rw.op("Conv2d", *args, name=f"{node.name}_g{g}", **attrs))
    return rw.op("Concat", *outs, name=node.name, axis=channel_axis)


FOLD_BATCHNORM: list[Rule] = [_fold_batchnorm]
SPLIT_GROUPS: list[Rule] = [_split_groups]
