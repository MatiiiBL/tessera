"""Interpreter speed on a few kernels (``python -m tessera.tir.bench``).

Writes ``results/tir_interpreter.json`` and ``results/tir_interpreter.md``:
for each kernel, the time to translate and compile it to Python, the median
time of a run, the number of times its innermost statement runs and the
time per execution.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from tessera import te
from tessera.compute import lower_group
from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.passes import fuse
from tessera.tir.interpreter import Kernel
from tessera.tir.primitives.common import block_paths
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import For, PrimFunc


@dataclass
class Row:
    kernel: str
    executions: int
    compile_seconds: float
    run_seconds: float

    @property
    def ns_per_execution(self) -> float:
        return self.run_seconds / self.executions * 1e9


def _executions(func: PrimFunc) -> int:
    """How many times leaf blocks run: the product of the loops around each one."""
    total = 0
    for path in block_paths(func):
        loops = [n for n in path if isinstance(n, For)]
        total += prod(n.extent.value for n in loops)  # type: ignore[attr-defined]
    return total


def _matmul(n: int) -> PrimFunc:
    a = te.placeholder((n, n), DType.float32, "A")
    b = te.placeholder((n, n), DType.float32, "B")
    k = te.reduce_axis(n, "k")
    c = te.compute((n, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), "C", ["i", "j"])
    return te.create_prim_func([a, b, c], f"matmul_{n}")


def _tiled(func: PrimFunc) -> PrimFunc:
    s = Schedule(func)
    i, j, k = s.get_loops("C")
    io, ii = s.split(i, [None, 8])
    jo, ji = s.split(j, [None, 8])
    s.reorder(io, jo, k, ii, ji)
    s.decompose_reduction("C", k)
    s.parallel(io)
    s.vectorize(ji)
    return s.func


def _group(build: Callable[[Graph], None]) -> PrimFunc:
    g = Graph("bench")
    build(g)
    (group,) = fuse(g)
    return lower_group(group).func


def _conv(g: Graph) -> None:
    x = g.add_input("x", TensorType((1, 16, 16, 16), DType.float32))
    w = g.add_constant(np.ones((3, 3, 16, 16), np.float32), "w")
    g.set_outputs([g.op("Relu", g.op("Conv2d", x, w, pads=(1, 1, 1, 1), layout="NHWC"))])


def _softmax(g: Graph) -> None:
    x = g.add_input("x", TensorType((64, 512), DType.float32))
    g.set_outputs([g.op("Softmax", x, axis=1)])


def _gelu(g: Graph) -> None:
    x = g.add_input("x", TensorType((256, 256), DType.float32))
    g.set_outputs([g.op("Gelu", x)])


KERNELS: dict[str, Callable[[], PrimFunc]] = {
    "matmul 64x64x64": lambda: _matmul(64),
    "matmul 64x64x64, tiled 8x8": lambda: _tiled(_matmul(64)),
    "conv2d 3x3 NHWC 16x16x16 -> 16, relu": lambda: _group(_conv),
    "softmax 64x512": lambda: _group(_softmax),
    "gelu 256x256": lambda: _group(_gelu),
}


def measure(name: str, func: PrimFunc, repeat: int = 3) -> Row:
    start = time.perf_counter()
    kernel = Kernel(func, check_bounds=False)
    compiled = time.perf_counter() - start
    rng = np.random.default_rng(0)
    args = [rng.standard_normal(p.shape).astype(p.dtype.numpy) for p in func.params]
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        kernel(args)
        times.append(time.perf_counter() - start)
    return Row(name, _executions(func), compiled, statistics.median(times))


def storage_cost(n: int = 200_000) -> dict[str, float]:
    """Nanoseconds per ``x[5] = x[5] + x[7] * x[8]`` from Python on each kind of
    flat storage the interpreter could use."""
    import array
    import timeit

    stores: dict[str, Any] = {
        "array.array": array.array("f", [0.0] * 16),
        "list": [0.0] * 16,
        "numpy": np.zeros(16, np.float32),
    }
    return {
        name: timeit.timeit("x[5] = x[5] + x[7] * x[8]", globals={"x": x}, number=n) / n * 1e9
        for name, x in stores.items()
    }


def _machine() -> dict[str, Any]:
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "load_average_1min": load,
    }


def main(out_dir: Path | None = None, repeat: int = 3) -> list[Row]:
    """Measure every kernel (median of ``repeat`` runs) and write the results files."""
    from tessera.bench.report import results_dir

    rows = [measure(name, build(), repeat) for name, build in KERNELS.items()]
    storage = storage_cost()
    out = out_dir or results_dir()
    out.mkdir(parents=True, exist_ok=True)
    info = _machine()
    data = {
        **info,
        "rows": [{**asdict(r), "ns_per_execution": r.ns_per_execution} for r in rows],
        "storage_ns": storage,
    }
    (out / "tir_interpreter.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Loop-IR interpreter speed",
        "",
        f"Measured {info['date']} on {info['machine']}, Python {info['python']}, "
        f"1-minute load average {info['load_average_1min']:.1f} "
        "(the machine was shared; numbers are provisional).",
        "Regenerate with `uv run python -m tessera.tir.bench`.",
        "",
        "| kernel | block executions | compile (ms) | run (ms) | ns per execution |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.kernel} | {r.executions:,} | {r.compile_seconds * 1e3:.1f} | "
            f"{r.run_seconds * 1e3:.1f} | {r.ns_per_execution:.0f} |"
        )
    lines += [
        "",
        "Cost of one read-modify-write `x[5] = x[5] + x[7] * x[8]` from Python:",
        "",
        "| storage | ns |",
        "|---|---:|",
        *(f"| {name} | {ns:.0f} |" for name, ns in storage.items()),
    ]
    (out / "tir_interpreter.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    for row in main():
        print(f"{row.kernel}: {row.run_seconds * 1e3:.1f} ms, {row.ns_per_execution:.0f} ns")
