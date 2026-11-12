"""Graphs of one fused group each, lowered by ``compute.lower_group``, with the
graph interpreter's result as the reference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera import compute, passes
from tessera.dtype import DType
from tessera.ir import Graph, TensorType, interpret
from tessera.ir.graph import Value
from tessera.plan import LoweredGroup
from tessera.tir.stmt import PrimFunc

type NDArray = npt.NDArray[Any]
F32 = DType.float32


@dataclass
class Workload:
    lowered: LoweredGroup
    inputs: list[NDArray]  # one per parameter but the last (the output)
    expected: NDArray
    tolerance: float = 1e-4

    @property
    def func(self) -> PrimFunc:
        return self.lowered.func

    def output(self) -> NDArray:
        out = self.func.params[-1]
        return np.full(out.shape, np.nan, out.dtype.numpy)


def lowered(build: Callable[[Graph, np.random.Generator], Value], seed: int = 0) -> Workload:
    """Build a graph, fuse it, and lower its one kernel group (view groups
    aside); the graph's inputs get random values."""
    rng = np.random.default_rng(seed)
    g = Graph("workload")
    g.set_outputs([build(g, rng)])
    g.verify()
    groups = [grp for grp in passes.fuse(g) if not grp.is_view]
    assert len(groups) == 1, [grp.name for grp in groups]
    group = groups[0]
    feeds = {v.name: rng.standard_normal(v.type.shape).astype(v.type.dtype.numpy) for v in g.inputs}
    expected = interpret(g, feeds)[g.outputs[0].name]
    args = [feeds[v.name] if v.data is None else np.asarray(v.data) for v in group.inputs]
    return Workload(compute.lower_group(group), args, expected)


def _const(g: Graph, rng: np.random.Generator, shape: tuple[int, ...], name: str) -> Value:
    return g.add_constant(rng.standard_normal(shape).astype(np.float32), name)


def matmul(m: int, n: int, k: int, *, batch: int = 0, epilogue: bool = True) -> Workload:
    """relu(A @ W + b), A of shape [batch, m, k] when batch > 0."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        lead = (batch,) if batch else ()
        y = g.op(
            "MatMul", g.add_input("A", TensorType((*lead, m, k), F32)), _const(g, rng, (k, n), "W")
        )
        if epilogue:
            y = g.op("Relu", g.op("Add", y, _const(g, rng, (n,), "b")))
        return y

    return lowered(build)


def attention_scores(heads: int, seq: int, dim: int) -> Workload:
    """Q @ K^T * scale: the transpose of K is a prologue of the matmul."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        q = g.add_input("Q", TensorType((heads, seq, dim), F32))
        kt = g.op("Transpose", g.add_input("K", TensorType((heads, seq, dim), F32)), perm=(0, 2, 1))
        scale = g.add_constant(np.float32([dim**-0.5]), "scale")
        return g.op("Mul", g.op("MatMul", q, kt), scale)

    return lowered(build)


def conv2d(
    n: int,
    h: int,
    w: int,
    c: int,
    o: int,
    kernel: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] = (1, 1, 1, 1),
    dilations: tuple[int, int] = (1, 1),
    *,
    depthwise: bool = False,
    epilogue: bool = True,
) -> Workload:
    """NHWC convolution with HWIO weights and a bias, then relu."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType((n, h, w, c), F32))
        out = c if depthwise else o
        weight = _const(g, rng, (*kernel, 1 if depthwise else c, out), "W")
        y = g.op(
            "Conv2d",
            x,
            weight,
            _const(g, rng, (out,), "b"),
            strides=strides,
            pads=pads,
            dilations=dilations,
            groups=c if depthwise else 1,
            layout="NHWC",
        )
        return g.op("Relu", y) if epilogue else y

    return lowered(build)


def pool(
    kind: str,
    n: int,
    h: int,
    w: int,
    c: int,
    kernel: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (2, 2),
    pads: tuple[int, int, int, int] = (1, 1, 1, 1),
) -> Workload:
    """NHWC MaxPool2d or AvgPool2d (counting padding)."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType((n, h, w, c), F32))
        attrs: dict[str, Any] = {
            "kernel": kernel,
            "strides": strides,
            "pads": pads,
            "layout": "NHWC",
        }
        if kind == "avg":
            return g.op("AvgPool2d", x, count_include_pad=1, **attrs)
        return g.op("MaxPool2d", x, **attrs)

    return lowered(build)


def reduce(kind: str, shape: tuple[int, ...], axes: tuple[int, ...], keepdims: int = 0) -> Workload:
    op = {"sum": "ReduceSum", "mean": "ReduceMean", "max": "ReduceMax"}[kind]

    def build(g: Graph, rng: np.random.Generator) -> Value:
        return g.op(op, g.add_input("X", TensorType(shape, F32)), axes=axes, keepdims=keepdims)

    return lowered(build)


def softmax(m: int, n: int, *, scale: bool = False) -> Workload:
    """Softmax over the last axis (of x * 0.125 with ``scale``: a prologue)."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType((m, n), F32))
        if scale:
            x = g.op("Mul", x, g.add_constant(np.float32([0.125]), "s"))
        return g.op("Softmax", x, axis=1)

    return lowered(build)


def layernorm(m: int, n: int) -> Workload:
    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType((m, n), F32))
        return g.op(
            "LayerNorm",
            x,
            _const(g, rng, (n,), "scale"),
            _const(g, rng, (n,), "bias"),
            axis=1,
            epsilon=1e-5,
        )

    return lowered(build)


def transpose(n: int, h: int, w: int, c: int) -> Workload:
    """sigmoid(x) * y + 1 broadcast over channels, then NHWC -> NCHW."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType((n, h, w, c), F32))
        y = g.op(
            "Add",
            g.op("Mul", g.op("Sigmoid", x), _const(g, rng, (c,), "Y")),
            g.add_constant(np.float32([1.0]), "one"),
        )
        return g.op("Transpose", y, perm=(0, 3, 1, 2))

    return lowered(build)


def gelu(shape: tuple[int, ...], approximate: str = "none") -> Workload:
    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType(shape, F32))
        y = g.op("Gelu", x, approximate=approximate)
        return g.op("Add", y, _const(g, rng, (shape[-1],), "b"))

    return lowered(build)


def gather_rows(rows: int, cols: int, count: int) -> Workload:
    """Embedding lookup: Gather of int64 indices along axis 0."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        table = _const(g, rng, (rows, cols), "table")
        ids = g.add_constant(rng.integers(0, rows, count).astype(np.int64), "ids")
        return g.op("Add", g.op("Gather", table, ids, axis=0), _const(g, rng, (cols,), "pos"))

    return lowered(build)


def concat_slice(n: int, c: int) -> Workload:
    """Concat along channels of a slice and a padded tensor."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("X", TensorType((n, 2 * c), F32))
        a = g.op("Slice", x, starts=(0, 1), ends=(n, c + 1), steps=(1, 1))
        b = g.op("Pad", g.add_input("Y", TensorType((n, c - 2), F32)), pads=(0, 1, 0, 1), value=0.5)
        return g.op("Concat", a, b, axis=1)

    return lowered(build)


def shared_operands(seq: int, dim: int) -> Workload:
    """Q @ K^T with Q and K two column slices of one projection: both operands
    of the product read the same buffer."""

    def build(g: Graph, rng: np.random.Generator) -> Value:
        x = g.add_input("QK", TensorType((seq, 2 * dim), F32))
        q = g.op("Slice", x, starts=(0, 0), ends=(seq, dim), steps=(1, 1))
        k = g.op("Slice", x, starts=(0, dim), ends=(seq, 2 * dim), steps=(1, 1))
        return g.op("MatMul", q, g.op("Transpose", k, perm=(1, 0)))

    return lowered(build)
