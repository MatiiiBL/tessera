"""Kernel benchmarks of the Metal backend, written to ``results/metal/``.

Run ``python -m tessera.backend.metal.bench NAME`` (``uv run --group bench``
for the PyTorch baselines). Each benchmark writes ``NAME.json`` and
``NAME.md``; times are GPU times from Metal's timestamps unless a column says
wall-clock.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from tessera.backend.base import MeasureError
from tessera.backend.metal import runtime
from tessera.backend.metal.codegen import CodegenError, generate
from tessera.backend.metal.measure import measure_kernel
from tessera.backend.metal.templates import TEMPLATES
from tessera.bench import report
from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.plan import LoweredGroup
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import PrimFunc
from tessera.tune.space import ApplyConfig, KnobValue, config_json, space_of

LAUNCH_KERNEL = """\
#include <metal_stdlib>
using namespace metal;
kernel void touch(device float* x [[buffer(0)]], uint i [[thread_index_in_threadgroup]]) {
  if (i == 0) {
    x[0] = x[0] + 1.0f;
  }
}
"""
LAUNCH_COUNTS = (1, 16, 256, 1024)
LAUNCH_MODES = {
    "dependent": "concurrent encoder, a barrier before every launch",
    "independent": "concurrent encoder, no barriers",
    "serial": "serial encoder (Metal orders every launch)",
}


def results_dir() -> Path:
    """``results/metal`` in the checkout (or under ``$TESSERA_RESULTS``)."""
    return report.results_dir() / "metal"


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    return float(np.polyfit(np.asarray(xs, dtype=np.float64), np.asarray(ys), 1)[0])


def bench_launch(repeat: int = 30) -> dict[str, Any]:
    """Cost of one kernel launch inside a command buffer, and of a command buffer."""
    device = runtime.device()
    pipeline = device.compile(LAUNCH_KERNEL).pipeline("touch")
    buffer = device.buffer(4)
    rows: list[dict[str, Any]] = []
    for mode in LAUNCH_MODES:
        gpu: list[float] = []
        wall: list[float] = []
        for count in LAUNCH_COUNTS:
            program = device.program()
            for _ in range(count):
                program.add(
                    pipeline, (1, 1, 1), (32, 1, 1), [(buffer, 0)], barrier=mode != "independent"
                )
            program.run(5, serial=mode == "serial")
            times = program.run(repeat, serial=mode == "serial")
            gpu.append(statistics.median(times.gpu))
            wall.append(statistics.median(times.wall))
        rows.append(
            {
                "mode": mode,
                "counts": list(LAUNCH_COUNTS),
                "gpu_seconds": gpu,
                "wall_seconds": wall,
                "gpu_per_launch": _slope(LAUNCH_COUNTS, gpu),
                "wall_per_launch": _slope(LAUNCH_COUNTS, wall),
                "command_buffer_wall": wall[0],
            }
        )
    return {"device": device.info.name, "repeat": repeat, "modes": rows}


def launch_markdown(data: dict[str, Any], env: dict[str, Any]) -> str:
    table = report.markdown_table(
        [
            "Encoding",
            "GPU per launch (µs)",
            "Wall-clock per launch (µs)",
            "1-launch command buffer, wall-clock (µs)",
        ],
        [
            [
                LAUNCH_MODES[r["mode"]],
                f"{r['gpu_per_launch'] * 1e6:.2f}",
                f"{r['wall_per_launch'] * 1e6:.2f}",
                f"{r['command_buffer_wall'] * 1e6:.0f}",
            ]
            for r in data["modes"]
        ],
    )
    detail = report.markdown_table(
        ["Encoding", *[f"{n} launches: GPU / wall (µs)" for n in LAUNCH_COUNTS]],
        [
            [
                r["mode"],
                *[
                    f"{g * 1e6:.0f} / {w * 1e6:.0f}"
                    for g, w in zip(r["gpu_seconds"], r["wall_seconds"], strict=True)
                ],
            ]
            for r in data["modes"]
        ],
    )
    return "\n".join(
        [
            report.header("Metal: the cost of a kernel launch", env, COMMANDS["launch"]),
            "A one-threadgroup kernel that updates one float, launched 1 to 1024 times in one",
            "command buffer. Per-launch costs are the slope of the median time (over "
            f"{data['repeat']} runs) against the",
            "number of launches; the last column is the round trip of a command buffer with one",
            "launch, from encoding to the CPU seeing it complete.",
            "",
            table,
            "",
            detail,
        ]
    )


# Compute-bound work, where the clock matters: 256 threadgroups of 256 threads, each
# thread running a dependent chain of multiply-adds.
WARMUP_KERNEL = """\
#include <metal_stdlib>
using namespace metal;
kernel void chain(device float* x [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
  float v = x[i];
  for (int k = 0; k < 2048; ++k) {
    v = fma(v, 0.999f, 0.001f);
  }
  x[i] = v;
}
"""
WARMUP_BATCH = 10


def bench_warmup(idle_seconds: float = 3.0, batches: int = 40) -> dict[str, Any]:
    """How an idle GPU's clock reacts to work: per-launch time of a small kernel in
    batches of 50 launches, one command buffer per batch, starting from idle,
    and then again after one continuous command buffer of about 50 ms."""
    import time

    device = runtime.device()
    pipeline = device.compile(WARMUP_KERNEL).pipeline("chain")
    buffer = device.buffer(4 * 256 * 256)

    def program(count: int) -> runtime.Program:
        p = device.program()
        for _ in range(count):
            p.add(pipeline, (256, 1, 1), (256, 1, 1), [(buffer, 0)])
        return p

    batch = program(WARMUP_BATCH)
    trials: list[dict[str, Any]] = []
    for warm in (False, True):
        time.sleep(idle_seconds)
        warm_seconds = 0.0
        if warm:
            single = program(1).run(1).gpu[0]
            count = max(1, min(20000, int(0.05 / max(single, 1e-6))))
            warm_seconds = program(count).run(1).gpu[0]
        elapsed, points = 0.0, []
        for _ in range(batches):
            gpu = batch.run(1).gpu[0]
            elapsed += gpu
            points.append({"elapsed": elapsed, "per_launch": gpu / WARMUP_BATCH})
        trials.append({"warmup_seconds": warm_seconds, "batches": points})
    return {"device": device.info.name, "idle_seconds": idle_seconds, "trials": trials}


def warmup_markdown(data: dict[str, Any], env: dict[str, Any]) -> str:
    cold, warm = data["trials"]
    picks = [0, 1, 2, 4, 9, 19, len(cold["batches"]) - 1]
    rows = [
        [
            i + 1,
            f"{cold['batches'][i]['elapsed'] * 1e3:.1f}",
            f"{cold['batches'][i]['per_launch'] * 1e6:.1f}",
            f"{warm['batches'][i]['per_launch'] * 1e6:.1f}",
        ]
        for i in picks
    ]
    return "\n".join(
        [
            report.header("Metal: GPU clock warmup", env, COMMANDS["warmup"]),
            f"A compute-bound kernel of 256 x 256 threads, in batches of {WARMUP_BATCH} (one "
            "command buffer each,",
            "the CPU waiting for each), after the GPU has been idle for "
            f"{data['idle_seconds']:.0f} s. The last column repeats",
            "the batches after one continuous command buffer of "
            f"{warm['warmup_seconds'] * 1e3:.0f} ms of the same kernel.",
            "",
            report.markdown_table(
                [
                    "Batch",
                    "GPU time so far (ms)",
                    "Per launch, from idle (µs)",
                    "Per launch, after warmup (µs)",
                ],
                rows,
            ),
        ]
    )


# -- kernels from the templates ----------------------------------------------------------


def schedule_default(lowered: LoweredGroup) -> tuple[PrimFunc, dict[str, KnobValue]]:
    """``lowered`` scheduled by its Metal template with the default configuration."""
    template = TEMPLATES[lowered.pattern]
    space = space_of(lambda cfg: template(Schedule(lowered.func), lowered, cfg))
    values = space.default()
    return schedule_with(lowered, values), values


def schedule_with(lowered: LoweredGroup, values: dict[str, KnobValue]) -> PrimFunc:
    sch = Schedule(lowered.func)
    TEMPLATES[lowered.pattern](sch, lowered, ApplyConfig(values))
    return sch.func


def time_kernel(func: PrimFunc) -> float:
    """GPU seconds per launch (see ``measure.measure_kernel``)."""
    return measure_kernel(generate(func), min_seconds=0.1)


def search(
    lowered: LoweredGroup, trials: int, seed: int = 0
) -> tuple[float, float, dict[str, KnobValue]]:
    """(default seconds, best seconds, best configuration) over ``trials`` random
    points of the template's space besides the default."""
    template = TEMPLATES[lowered.pattern]
    space = space_of(lambda cfg: template(Schedule(lowered.func), lowered, cfg))
    default = time_kernel(schedule_with(lowered, space.default()))
    best, best_values = default, space.default()
    rng = random.Random(seed)
    for _ in range(trials):
        values = space.sample(rng)
        try:
            seconds = time_kernel(schedule_with(lowered, values))
        except (ScheduleError, MeasureError, CodegenError, runtime.MetalError):
            continue
        if seconds < best:
            best, best_values = seconds, values
    return default, best, best_values


def _lower_one(graph: Graph, pattern: str) -> LoweredGroup:
    from tessera import compute, passes

    groups = [g for g in passes.fuse(graph) if g.pattern == pattern and not g.is_view]
    return compute.lower_group(groups[0])


def matmul_group(batch: int, m: int, n: int, k: int) -> LoweredGroup:
    """``x @ w`` lowered, with ``x`` of shape ``(batch, m, k)`` (no batch if 0)."""
    g = Graph(f"matmul_{batch}x{m}x{n}x{k}")
    lead = (batch,) if batch else ()
    x = g.add_input("x", TensorType((*lead, m, k), DType.float32))
    w = g.add_constant(np.zeros((k, n), np.float32), "w")
    g.set_outputs([g.op("MatMul", x, w)])
    return _lower_one(g, "matmul")


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.backends.mps.is_available() else None


def mps_seconds(run: Callable[[], object], repeat: int = 50) -> float:
    """GPU seconds of one ``run()`` of MPS work, timed with MPS events."""
    torch = _torch()
    assert torch is not None
    for _ in range(5):
        run()
    torch.mps.synchronize()
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run()
    end.record()
    torch.mps.synchronize()
    return float(start.elapsed_time(end)) * 1e-3 / repeat


SQUARE = (256, 512, 1024, 2048, 4096)
# (batch, M, N, K) of BERT-base and GPT-2 small at 128 tokens.
TRANSFORMER = {
    "attention projection (BERT, GPT-2)": (0, 128, 768, 768),
    "fused QKV projection (GPT-2)": (0, 128, 2304, 768),
    "MLP up (BERT, GPT-2)": (0, 128, 3072, 768),
    "MLP down (BERT, GPT-2)": (0, 128, 768, 3072),
    "attention scores, 12 heads": (12, 128, 128, 64),
    "attention times values, 12 heads": (12, 128, 64, 128),
}


def bench_gemm(trials: int = 0) -> dict[str, Any]:
    """GEMM throughput of the matmul template against ``torch.matmul`` on MPS."""
    torch = _torch()
    shapes = [(f"{n} x {n} x {n}", (0, n, n, n)) for n in SQUARE] + list(TRANSFORMER.items())
    rows: list[dict[str, Any]] = []
    for name, (batch, m, n, k) in shapes:
        flops = 2 * max(batch, 1) * m * n * k
        lowered = matmul_group(batch, m, n, k)
        default, best, values = search(lowered, trials)
        row: dict[str, Any] = {
            "name": name,
            "shape": [batch, m, n, k],
            "flops": flops,
            "default_seconds": default,
            "best_seconds": best,
            "best_config": config_json(values),
        }
        if torch is not None:
            a = torch.randn(*((batch,) if batch else ()), m, k, device="mps")
            b = torch.randn(k, n, device="mps")
            row["mps_seconds"] = mps_seconds(partial(torch.matmul, a, b))
        rows.append(row)
    return {"trials": trials, "torch": None if torch is None else torch.__version__, "shapes": rows}


def gemm_markdown(data: dict[str, Any], env: dict[str, Any]) -> str:
    def gflops(row: dict[str, Any], key: str) -> str:
        seconds = row.get(key)
        return "" if seconds is None else f"{row['flops'] / seconds / 1e9:.0f}"

    headers = ["Shape (batch, M, N, K)", "Tessera default", "MPS"]
    if data["trials"]:
        headers.insert(2, f"Tessera best of {data['trials']}")
    table = []
    for row in data["shapes"]:
        cells = [row["name"], gflops(row, "default_seconds"), gflops(row, "mps_seconds")]
        if data["trials"]:
            cells.insert(2, gflops(row, "best_seconds"))
        table.append(cells)
    return "\n".join(
        [
            report.header("Metal: GEMM throughput (GFLOPS)", env, COMMANDS["gemm"]),
            "float32 C = A @ B, B a constant. GPU time per call: Metal timestamps around "
            "batches of launches",
            f"for Tessera, MPS events around 50 calls for PyTorch {data['torch']}. GFLOPS "
            "count 2 M N K per batch.",
            "",
            report.markdown_table(headers, table),
        ]
    )


# -- threadgroup shape -----------------------------------------------------------------------


def simdgroup_matmul(
    m: int, n: int, k: int, *, groups: tuple[int, int] = (1, 2), frags: tuple[int, int] = (2, 4)
) -> PrimFunc:
    """A matmul with SIMD-group matrices, as tensorizing leaves it: ``groups`` SIMD
    groups per threadgroup (rows on thread.z, columns on thread.y), each holding
    ``frags`` 8 x 8 accumulator tiles, A and B tiles loaded from device memory
    every 8 steps of K. The fragment buffers have the whole tensors' shapes."""
    from tessera.backend.metal.codegen import (
        simdgroup_fill,
        simdgroup_load,
        simdgroup_mma,
        simdgroup_store,
    )
    from tessera.tir.expr import Expr, FloatImm, IntImm, Var
    from tessera.tir.stmt import (
        TRUE,
        Block,
        BlockRealize,
        Buffer,
        BufferRegion,
        For,
        ForKind,
        Stmt,
        seq,
    )

    (gm, gn), (fm, fn) = groups, frags
    bm, bn = gm * fm * 8, gn * fn * 8
    a, b, c = Buffer("A", (m, k)), Buffer("B", (k, n)), Buffer("C", (m, n))
    c_frag = Buffer("C_frag", (m, n), scope="local")
    a_frag = Buffer("A_frag", (m, k), scope="local")
    b_frag = Buffer("B_frag", (k, n), scope="local")
    by, bx, sm, sn, ko = Var("by"), Var("bx"), Var("sm"), Var("sn"), Var("ko")
    row0, col0 = by * bm + sm * (fm * 8), bx * bn + sn * (fn * 8)

    def loop(var: Var, extent: int, body: Stmt, kind: ForKind, tag: str | None = None) -> For:
        return For(var, IntImm(0), IntImm(extent), kind, body, tag)

    def tiles(body: Callable[[Expr, Expr], Stmt]) -> Stmt:
        i, j = Var("i"), Var("j")
        inner = loop(j, fn, body(row0 + i * 8, col0 + j * 8), ForKind.UNROLLED)
        return loop(i, fm, inner, ForKind.UNROLLED)

    def line(body: Callable[[Expr], Stmt], count: int, base: Expr) -> Stmt:
        i = Var("i")
        return loop(i, count, body(base + i * 8), ForKind.UNROLLED)

    kk = ko * 8
    step = seq(
        [
            line(lambda r: simdgroup_load(a_frag.address(r, kk), a.address(r, kk), k), fm, row0),
            line(lambda q: simdgroup_load(b_frag.address(kk, q), b.address(kk, q), n), fn, col0),
            tiles(
                lambda r, q: simdgroup_mma(
                    c_frag.address(r, q),
                    a_frag.address(r, kk),
                    b_frag.address(kk, q),
                    c_frag.address(r, q),
                )
            ),
        ]
    )
    body = seq(
        [
            tiles(lambda r, q: simdgroup_fill(c_frag.address(r, q), FloatImm(0.0))),
            loop(ko, k // 8, step, ForKind.SERIAL),
            tiles(lambda r, q: simdgroup_store(c_frag.address(r, q), c.address(r, q), n)),
        ]
    )
    bound = ForKind.THREAD_BINDING
    nest = loop(
        by,
        m // bm,
        loop(
            bx,
            n // bn,
            loop(sm, gm, loop(sn, gn, body, bound, "thread.y"), bound, "thread.z"),
            bound,
            "threadgroup.x",
        ),
        bound,
        "threadgroup.y",
    )
    root = Block("root", (), (), (BufferRegion.full(c),), nest, None, (c_frag, a_frag, b_frag))
    return PrimFunc("matmul_simdgroup", (a, b, c), BlockRealize((), TRUE, root))


SHAPE_SIZES = (1024, 2048)
SHAPE_GROUPS = ((1, 2), (2, 1), (2, 2))


def bench_shape() -> dict[str, Any]:
    """The same SIMD-group matmul kernels launched with flat and shaped threadgroups."""
    rows: list[dict[str, Any]] = []
    for size in SHAPE_SIZES:
        for groups in SHAPE_GROUPS:
            func = simdgroup_matmul(size, size, size, groups=groups)
            flat = generate(func)
            shaped = generate(func, flat=False)
            rows.append(
                {
                    "size": size,
                    "groups": list(groups),
                    "threads": list(shaped.threads),
                    "flops": 2 * size**3,
                    "flat_seconds": measure_kernel(flat, min_seconds=0.2),
                    "shaped_seconds": measure_kernel(shaped, min_seconds=0.2),
                }
            )
    return {"kernels": rows}


def shape_markdown(data: dict[str, Any], env: dict[str, Any]) -> str:
    table = [
        [
            f"{r['size']}³",
            f"{r['groups'][0]} x {r['groups'][1]}",
            " x ".join(str(t) for t in r["threads"]),
            f"{r['flops'] / r['shaped_seconds'] / 1e9:.0f}",
            f"{r['flops'] / r['flat_seconds'] / 1e9:.0f}",
            f"{r['shaped_seconds'] / r['flat_seconds']:.2f}",
        ]
        for r in data["kernels"]
    ]
    return "\n".join(
        [
            report.header("Metal: threadgroup shape", env, COMMANDS["shape"]),
            "One SIMD-group matrix multiply kernel (each SIMD group holds 2 x 4 accumulator "
            "tiles of 8 x 8,",
            "loads A and B from device memory), generated twice: with the threadgroup "
            "shaped like its thread",
            "loops (32 lanes x SIMD-group columns x SIMD-group rows), and launched flat "
            "with the positions",
            "computed from the thread index. GFLOPS from GPU time.",
            "",
            report.markdown_table(
                ["Size", "SIMD groups", "Shape", "Shaped (GFLOPS)", "Flat (GFLOPS)", "Speedup"],
                table,
            ),
        ]
    )


# -- ResNet-50 convolutions --------------------------------------------------------------

MODELS = Path("data/models")


def resnet_convolutions(path: Path | None = None) -> list[tuple[LoweredGroup, int]]:
    """Every distinct convolution group of ResNet-50 (shape and fused operators),
    with how many times it occurs, in first-seen order."""
    from tessera import compute, passes
    from tessera.frontend import onnx as frontend

    model = path or report.repo_root() / MODELS / "resnet50.onnx"
    if not model.exists():
        raise FileNotFoundError(f"{model} is missing: run scripts/export_models.py")
    graph = passes.optimize(frontend.load(model))
    seen: dict[tuple[Any, ...], list[Any]] = {}
    for group in passes.fuse(graph):
        if group.pattern != "conv2d":
            continue
        lowered = compute.lower_group(group)
        key = (tuple(sorted(lowered.dims.items())), tuple(n.op for n in group.nodes))
        if key in seen:
            seen[key][1] += 1
        else:
            seen[key] = [lowered, 1]
    return [(lowered, count) for lowered, count in seen.values()]


def _conv_name(lowered: LoweredGroup) -> str:
    d = lowered.dims
    ops = "+".join(n.op.lower() for n in lowered.group.nodes[1:])
    return f"{d['H']}x{d['W']}x{d['C']} -> {d['O']}, {d['KH']}x{d['KW']}/{d['SH']}" + (
        f" +{ops}" if ops else ""
    )


def _mps_conv(lowered: LoweredGroup) -> float:
    torch = _torch()
    assert torch is not None
    d = lowered.dims
    x = torch.randn(d["N"], d["C"], d["H"], d["W"], device="mps")
    x = x.to(memory_format=torch.channels_last)
    w = torch.randn(d["O"], d["C"], d["KH"], d["KW"], device="mps")
    b = torch.randn(d["O"], device="mps")
    ops = [n.op for n in lowered.group.nodes[1:]]
    residual = torch.randn(d["N"], d["O"], d["OH"], d["OW"], device="mps")
    residual = residual.to(memory_format=torch.channels_last)
    conv = torch.nn.functional.conv2d

    def run() -> object:
        y = conv(x, w, b, stride=(d["SH"], d["SW"]), padding=(d["PT"], d["PL"]))
        if "Add" in ops:
            y = y + residual
        if "Relu" in ops:
            y = torch.relu(y)
        return y

    return mps_seconds(run)


def bench_conv(trials: int = 0) -> dict[str, Any]:
    """The convolution template on ResNet-50's convolutions against MPS."""
    torch = _torch()
    rows: list[dict[str, Any]] = []
    for lowered, count in resnet_convolutions():
        d = lowered.dims
        flops = 2 * d["N"] * d["OH"] * d["OW"] * d["O"] * d["C"] * d["KH"] * d["KW"]
        default, best, values = search(lowered, trials)
        row: dict[str, Any] = {
            "name": _conv_name(lowered),
            "count": count,
            "dims": dict(d),
            "flops": flops,
            "default_seconds": default,
            "best_seconds": best,
            "best_config": config_json(values),
        }
        if torch is not None:
            row["mps_seconds"] = _mps_conv(lowered)
        rows.append(row)
    return {"trials": trials, "torch": None if torch is None else torch.__version__, "layers": rows}


def conv_markdown(data: dict[str, Any], env: dict[str, Any]) -> str:
    def cell(row: dict[str, Any], key: str) -> str:
        seconds = row.get(key)
        if seconds is None:
            return ""
        return f"{seconds * 1e6:.0f} ({row['flops'] / seconds / 1e9:.0f})"

    headers = ["Layer (NHWC input -> channels, kernel/stride)", "Count", "Tessera default", "MPS"]
    if data["trials"]:
        headers.insert(3, f"Tessera best of {data['trials']}")
    table = []
    totals = {"default_seconds": 0.0, "best_seconds": 0.0, "mps_seconds": 0.0}
    for row in data["layers"]:
        cells = [row["name"], row["count"], cell(row, "default_seconds"), cell(row, "mps_seconds")]
        if data["trials"]:
            cells.insert(3, cell(row, "best_seconds"))
        table.append(cells)
        for key in totals:
            totals[key] += row["count"] * row.get(key, 0.0)
    summary = (
        f"Over the {sum(r['count'] for r in data['layers'])} convolutions of the network: "
        f"Tessera {totals['default_seconds'] * 1e3:.2f} ms with default configurations"
        + (f", {totals['best_seconds'] * 1e3:.2f} ms with the best found" if data["trials"] else "")
        + (f"; MPS {totals['mps_seconds'] * 1e3:.2f} ms." if data["torch"] else ".")
    )
    return "\n".join(
        [
            report.header("Metal: ResNet-50 convolutions", env, COMMANDS["conv"]),
            "Each distinct convolution group of ResNet-50 at 224 x 224, batch 1, float32, "
            "with its fused bias,",
            "residual addition and ReLU. Cells are GPU microseconds per call "
            "(GFLOPS in parentheses); MPS",
            f"runs the same operators with PyTorch {data['torch']} on channels-last tensors.",
            "",
            report.markdown_table(headers, table),
            "",
            summary,
        ]
    )


COMMANDS = {
    "launch": "uv run python -m tessera.backend.metal.bench launch",
    "warmup": "uv run python -m tessera.backend.metal.bench warmup",
    "shape": "uv run python -m tessera.backend.metal.bench shape",
    "gemm": "uv run --group bench python -m tessera.backend.metal.bench gemm",
    "conv": "uv run --group bench python -m tessera.backend.metal.bench conv",
}
# name -> (measure(trials), render(data, environment)); ``trials`` is the number of
# random configurations the template benchmarks try besides the default.
BENCHMARKS: dict[
    str, tuple[Callable[[int], dict[str, Any]], Callable[[dict[str, Any], dict[str, Any]], str]]
] = {
    "launch": (lambda trials: bench_launch(), launch_markdown),
    "warmup": (lambda trials: bench_warmup(), warmup_markdown),
    "shape": (lambda trials: bench_shape(), shape_markdown),
    "gemm": (bench_gemm, gemm_markdown),
    "conv": (bench_conv, conv_markdown),
}


def run(name: str, directory: Path | None = None, *, trials: int = 0) -> Path:
    """Run benchmark ``name`` and write its results; returns the Markdown path."""
    measure, render = BENCHMARKS[name]
    env = report.environment()
    data = measure(trials)
    data["environment"] = env
    return report.write(name, data, render(data, env), directory or results_dir())


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m tessera.backend.metal.bench NAME [NAME ...] | all``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=[*BENCHMARKS, "all"])
    parser.add_argument("--out", type=Path, default=None, help="directory for the results")
    parser.add_argument(
        "--trials", type=int, default=0, help="random configurations to try (gemm, conv)"
    )
    args = parser.parse_args(argv)
    names = list(BENCHMARKS) if "all" in args.names else args.names
    for name in names:
        print(run(name, args.out, trials=args.trials))


if __name__ == "__main__":
    main()
