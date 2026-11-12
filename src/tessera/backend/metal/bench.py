"""Kernel benchmarks of the Metal backend, written to ``results/metal/``.

Run ``python -m tessera.backend.metal.bench NAME`` (``uv run --group bench``
for the PyTorch baselines). Each benchmark writes ``NAME.json`` and
``NAME.md``; times are GPU times from Metal's timestamps unless a column says
wall-clock.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tessera.backend.metal import runtime
from tessera.bench import report

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


COMMANDS = {
    "launch": "uv run python -m tessera.backend.metal.bench launch",
    "warmup": "uv run python -m tessera.backend.metal.bench warmup",
}
BENCHMARKS: dict[
    str, tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any], dict[str, Any]], str]]
] = {
    "launch": (bench_launch, launch_markdown),
    "warmup": (bench_warmup, warmup_markdown),
}


def run(name: str, directory: Path | None = None) -> Path:
    """Run benchmark ``name`` and write its results; returns the Markdown path."""
    measure, render = BENCHMARKS[name]
    env = report.environment()
    data = measure()
    data["environment"] = env
    return report.write(name, data, render(data, env), directory or results_dir())


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m tessera.backend.metal.bench NAME [NAME ...] | all``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=[*BENCHMARKS, "all"])
    parser.add_argument("--out", type=Path, default=None, help="directory for the results")
    args = parser.parse_args(argv)
    names = list(BENCHMARKS) if "all" in args.names else args.names
    for name in names:
        print(run(name, args.out))


if __name__ == "__main__":
    main()
