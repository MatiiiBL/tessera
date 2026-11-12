"""Fused groups lowered by ``compute.lower_group`` from small graphs, with the
graph interpreter's output as the reference, for testing the GPU templates."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera import compute, ir, passes
from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.plan import LoweredGroup
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import PrimFunc
from tessera.tune.space import ConfigSpace, KnobValue, space_of

type Array = npt.NDArray[Any]
F32 = DType.float32


@dataclass
class Workload:
    """One group of a graph: its PrimFunc's arguments and the expected output."""

    lowered: LoweredGroup
    inputs: list[Array]  # one per parameter but the output: graph inputs and constants
    output: Array

    @property
    def func(self) -> PrimFunc:
        return self.lowered.func

    @property
    def pattern(self) -> str:
        return self.lowered.pattern


def lower(graph: Graph, seed: int = 0) -> Workload:
    """The graph's only kernel, lowered, with random inputs and the reference output."""
    graph.verify()
    groups = [g for g in passes.fuse(graph) if not g.is_view]
    if len(groups) != 1:
        raise ValueError(f"{graph.name} fuses into {len(groups)} kernels, expected one")
    group = groups[0]
    feeds = ir.random_inputs(graph, seed)
    args = [v.data if v.data is not None else feeds[v.name] for v in group.inputs]
    output = ir.interpret(graph, feeds, float64=True)[graph.outputs[0].name]
    return Workload(compute.lower_group(group), [np.asarray(a) for a in args], output)


def _weights(rng: np.random.Generator, *shape: int, scale: float = 0.3) -> Array:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def injective() -> Workload:
    g = Graph("transpose_add_sigmoid")
    x = g.add_input("x", TensorType((4, 8, 16), F32))
    z = g.add_input("z", TensorType((16,), F32))
    t = g.op("Transpose", x, perm=(2, 1, 0))
    g.set_outputs([g.op("Sigmoid", g.op("Add", t, g.op("Reshape", z, shape=(16, 1, 1))))])
    return lower(g)


def matmul(m: int, n: int, k: int, *, batch: int = 0, epilogue: bool = True) -> Workload:
    rng = np.random.default_rng(m * n + k)
    g = Graph("matmul")
    lead = (batch,) if batch else ()
    x = g.add_input("x", TensorType((*lead, m, k), F32))
    w = g.add_constant(_weights(rng, k, n), "w")
    y = g.op("MatMul", x, w)
    if epilogue:
        y = g.op("Relu", g.op("Add", y, g.add_constant(_weights(rng, n), "b")))
    g.set_outputs([y])
    return lower(g)


def conv2d(
    shape: tuple[int, int, int, int],
    out_channels: int,
    kernel: int,
    *,
    stride: int = 1,
    pad: int = 0,
) -> Workload:
    rng = np.random.default_rng(sum(shape) + kernel)
    g = Graph("conv")
    x = g.add_input("x", TensorType(shape, F32))
    w = g.add_constant(_weights(rng, kernel, kernel, shape[3], out_channels), "w")
    b = g.add_constant(_weights(rng, out_channels), "b")
    conv = g.op(
        "Conv2d",
        x,
        w,
        b,
        strides=(stride, stride),
        pads=(pad,) * 4,
        dilations=(1, 1),
        groups=1,
        layout="NHWC",
    )
    g.set_outputs([g.op("Relu", conv)])
    return lower(g)


def depthwise(shape: tuple[int, int, int, int], *, stride: int = 1) -> Workload:
    rng = np.random.default_rng(sum(shape) + stride)
    g = Graph("dwconv")
    x = g.add_input("x", TensorType(shape, F32))
    w = g.add_constant(_weights(rng, 3, 3, 1, shape[3]), "w")
    conv = g.op(
        "Conv2d",
        x,
        w,
        strides=(stride, stride),
        pads=(1, 1, 1, 1),
        dilations=(1, 1),
        groups=shape[3],
        layout="NHWC",
    )
    g.set_outputs([g.op("Clip", conv, min=0.0, max=6.0)])
    return lower(g)


def pool(op: str, shape: tuple[int, int, int, int]) -> Workload:
    g = Graph(op.lower())
    x = g.add_input("x", TensorType(shape, F32))
    attrs: dict[str, Any] = {
        "kernel": (3, 3),
        "strides": (2, 2),
        "pads": (1, 1, 1, 1),
        "layout": "NHWC",
    }
    if op == "AvgPool2d":
        attrs["count_include_pad"] = 0
    g.set_outputs([g.op(op, x, **attrs)])
    return lower(g)


def reduce(op: str, shape: tuple[int, ...], axes: tuple[int, ...], keepdims: int = 1) -> Workload:
    g = Graph(op.lower())
    x = g.add_input("x", TensorType(shape, F32))
    g.set_outputs([g.op(op, x, axes=axes, keepdims=keepdims)])
    return lower(g)


def softmax(rows: int, r: int) -> Workload:
    g = Graph("softmax")
    x = g.add_input("x", TensorType((rows, r), F32))
    g.set_outputs([g.op("Softmax", x, axis=1)])
    return lower(g)


def layernorm(rows: int, r: int) -> Workload:
    rng = np.random.default_rng(rows + r)
    g = Graph("layernorm")
    x = g.add_input("x", TensorType((rows, r), F32))
    scale = g.add_constant(_weights(rng, r, scale=1.0), "scale")
    bias = g.add_constant(_weights(rng, r, scale=1.0), "bias")
    g.set_outputs([g.op("LayerNorm", x, scale, bias, axis=1, epsilon=1e-5)])
    return lower(g)


def space(template: Callable[..., None], workload: Workload) -> ConfigSpace:
    """The template's knobs for ``workload``."""
    return space_of(lambda cfg: template(Schedule(workload.func), workload.lowered, cfg))


def configs(
    template: Callable[..., None], workload: Workload, samples: int, seed: int = 0
) -> Iterator[dict[str, KnobValue]]:
    """The default point of the template's space, then ``samples`` random ones."""
    sp = space(template, workload)
    yield sp.default()
    rng = random.Random(seed)
    for _ in range(samples):
        yield sp.sample(rng)
