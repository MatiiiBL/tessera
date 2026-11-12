"""``tessera bench ops``: single kernels against vendor libraries.

**GEMM.** ``C = A @ B`` in float32 with ``B`` a constant (a weight), for
square sizes and the shapes of the transformer models (sequence 128 against
BERT's and GPT-2's projections and GPT-2's vocabulary). Tessera compiles a
one-node ONNX model per shape for each target; the baselines are NumPy's
``matmul``, which calls Accelerate, and PyTorch's ``mm`` on MPS.
Throughput is ``2·M·N·K`` over the median time.

**ResNet-50 convolutions.** Every convolution kernel of the compiled
ResNet-50, timed by profiling the whole model (each kernel includes its
fused epilogue: bias, residual add, ReLU), against PyTorch's ``conv2d`` of
the same shape on the CPU and on MPS (channels-last, without the epilogue).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tessera.bench.baselines import torch_available
from tessera.bench.report import markdown_table
from tessera.bench.stats import Latency, time_runs
from tessera.driver import lower
from tessera.ir.graph import NDArray, Node

SQUARE = (256, 512, 1024, 2048)
TRANSFORMER = ((128, 768, 768), (128, 3072, 768), (128, 768, 3072), (128, 50257, 768))
GEMM_SHAPES: tuple[tuple[int, int, int], ...] = (
    *((n, n, n) for n in SQUARE),
    *TRANSFORMER,
)  # (M, N, K)
RUNS = 50


@dataclass(frozen=True)
class OpResult:
    op: str  # "gemm 1024x1024x1024", "conv layer1.0.conv2 ..."
    system: str
    latency: Latency | None
    flops: int
    error: str | None = None

    @property
    def gflops(self) -> float | None:
        return self.flops / self.latency.median / 1e9 if self.latency else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "system": self.system,
            "latency": self.latency.as_dict() if self.latency else None,
            "flops": self.flops,
            "gflops": self.gflops,
            "error": self.error,
        }


def gemm_model(m: int, n: int, k: int, seed: int = 0) -> onnx.ModelProto:
    """``y = x @ w`` with ``x`` [M, K] an input and ``w`` [K, N] a constant."""
    w = (np.random.default_rng(seed).standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="gemm")],
        f"gemm_{m}x{n}x{k}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])],
        [numpy_helper.from_array(w, "w")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def bench_gemm(
    shapes: Sequence[tuple[int, int, int]] = GEMM_SHAPES,
    *,
    targets: Sequence[str] = (),
    baselines: Sequence[str] = ("numpy", "pytorch-mps"),
    progress: Callable[[str], None] = lambda message: None,
) -> list[OpResult]:
    """Every GEMM shape on every system."""
    results: list[OpResult] = []
    for m, n, k in shapes:
        op = f"gemm {m}x{n}x{k}"
        flops = 2 * m * n * k
        rng = np.random.default_rng(0)
        x = rng.standard_normal((m, k)).astype(np.float32)
        model = gemm_model(m, n, k)
        w = numpy_helper.to_array(model.graph.initializer[0])
        for system in baselines:
            if system == "pytorch-mps" and not torch_available("mps"):
                continue
            progress(f"{op}: {system}")
            results.append(_attempt(op, system, flops, partial(_gemm_baseline, system, x, w)))
        for target in targets:
            progress(f"{op}: tessera-{target}")
            run = partial(_gemm_tessera, model, target, x)
            results.append(_attempt(op, f"tessera-{target}", flops, run))
    return results


def _gemm_baseline(system: str, x: NDArray, w: NDArray) -> Latency:
    if system == "numpy":
        out = np.empty((x.shape[0], w.shape[1]), np.float32)
        return time_runs(lambda: np.matmul(x, w, out=out), runs=RUNS)
    import torch

    a, b = torch.from_numpy(x).to("mps"), torch.from_numpy(w).to("mps")

    def run() -> None:
        torch.mm(a, b)
        torch.mps.synchronize()

    return time_runs(run, runs=RUNS)


def _gemm_tessera(model: onnx.ModelProto, target: str, x: NDArray) -> Latency:
    compiled = lower(model, target=target).build()
    expected = x.astype(np.float64) @ numpy_helper.to_array(model.graph.initializer[0])
    got = compiled.run({"x": x})["y"]
    error = float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))
    if error > 1e-4:
        raise ValueError(f"wrong result: relative error {error:.2e}")
    return Latency.of(compiled.benchmark({"x": x}, warmup=10, repeat=RUNS))


def _attempt(op: str, system: str, flops: int, fn: Callable[[], Latency]) -> OpResult:
    try:
        return OpResult(op, system, fn(), flops)
    except Exception as e:  # keep the other systems' numbers
        return OpResult(op, system, None, flops, f"{type(e).__name__}: {e}")


def markdown(results: Sequence[OpResult]) -> str:
    """One row per operation, one GFLOP/s column per system."""
    systems = list(dict.fromkeys(r.system for r in results))
    ops = list(dict.fromkeys(r.op for r in results))
    by = {(r.op, r.system): r for r in results}
    rows: list[list[object]] = []
    for op in ops:
        row: list[object] = [op]
        for s in systems:
            r = by.get((op, s))
            if r is None:
                row.append("")
            elif r.gflops is None:
                row.append("failed")
            else:
                row.append(f"{r.gflops:.1f} ({r.latency.median * 1e3:.3f} ms)" if r.latency else "")
        rows.append(row)
    return markdown_table(["operation", *(f"{s} GFLOP/s (ms)" for s in systems)], rows)


@dataclass(frozen=True)
class ConvShape:
    """A convolution's problem size, NHWC input and HWIO weight."""

    h: int
    w: int
    c: int
    o: int
    kh: int
    kw: int
    stride: int
    pads: tuple[int, int, int, int]  # top, left, bottom, right
    groups: int

    @property
    def out_hw(self) -> tuple[int, int]:
        top, left, bottom, right = self.pads
        oh = (self.h + top + bottom - self.kh) // self.stride + 1
        ow = (self.w + left + right - self.kw) // self.stride + 1
        return oh, ow

    @property
    def flops(self) -> int:
        oh, ow = self.out_hw
        return 2 * oh * ow * self.o * self.kh * self.kw * (self.c // self.groups)

    def __str__(self) -> str:
        kind = "dw " if self.groups > 1 else ""
        return f"{kind}{self.kh}x{self.kw}/{self.stride} {self.h}x{self.w}x{self.c} -> {self.o}"


def conv_shape(node: Node) -> ConvShape:
    """The shape of an NHWC ``Conv2d`` node."""
    _, h, w, c = node.inputs[0].shape
    kh, kw, _, o = node.inputs[1].shape
    strides = node.ints_attr("strides")
    pads = node.ints_attr("pads")
    return ConvShape(
        h,
        w,
        c,
        o,
        kh,
        kw,
        strides[0],
        (pads[0], pads[1], pads[2], pads[3]),
        node.int_attr("groups"),
    )


def bench_convs(
    model: Any,
    *,
    target: str,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    baselines: Sequence[str] = ("pytorch-cpu", "pytorch-mps"),
    repeat: int = 20,
    progress: Callable[[str], None] = lambda message: None,
) -> list[OpResult]:
    """Each distinct convolution of ``model``: Tessera's kernel time (the median over
    the kernels of that shape, from a profile of the whole model) and the baselines."""
    from tessera.ir import random_inputs

    progress(f"convolutions: compiling for {target}")
    comp = lower(model, target=target, input_shapes=input_shapes)
    compiled = comp.build()
    times = {
        t.kernel: t.seconds for t in compiled.profile(random_inputs(comp.graph), repeat=repeat)
    }
    by_shape: dict[ConvShape, list[float]] = {}
    for k in comp.kernels:
        group = k.group
        if group is None or group.anchor is None or group.anchor.op != "Conv2d":
            continue
        by_shape.setdefault(conv_shape(group.anchor), []).append(times[k.name])
    results: list[OpResult] = []
    for shape, seconds in by_shape.items():
        op = f"conv {shape} (x{len(seconds)})"
        median = float(np.median(seconds))
        results.append(
            OpResult(
                op,
                f"tessera-{target}",
                Latency(median, min(seconds), max(seconds), repeat),
                shape.flops,
            )
        )
        for system in baselines:
            device = system.removeprefix("pytorch-")
            if torch_available(device):
                progress(f"{op}: {system}")
                run = partial(_conv_baseline, shape, device)
                results.append(_attempt(op, system, shape.flops, run))
    return results


def _conv_baseline(shape: ConvShape, device: str) -> Latency:
    import torch
    import torch.nn.functional as F

    top, left, bottom, right = shape.pads
    if (top, left) != (bottom, right):
        raise ValueError(f"asymmetric padding {shape.pads}")
    x = torch.randn(1, shape.c, shape.h, shape.w).to(device, memory_format=torch.channels_last)
    w = torch.randn(shape.o, shape.c // shape.groups, shape.kh, shape.kw).to(
        device, memory_format=torch.channels_last
    )

    def run() -> None:
        with torch.inference_mode():
            F.conv2d(x, w, stride=shape.stride, padding=(top, left), groups=shape.groups)
        if device == "mps":
            torch.mps.synchronize()

    return time_runs(run, runs=RUNS)
