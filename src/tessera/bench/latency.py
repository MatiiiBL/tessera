"""Timing a compiled model in fresh processes.

macOS sometimes keeps one of the C runtime's SME workers on the efficiency
cluster for a whole process; that worker skips its share of the loops, so
the process runs with one thread fewer, and every run in it is slower
(docs/cpu.md, Limitations). A latency measured in a single process
therefore depends on that one process's placement. ``fresh_process_runs``
times the model in several new processes, a few runs each after a warmup,
and pools the runs, so the median reflects how often placement goes wrong
rather than whether it did once.

Run as ``python -m tessera.bench.latency``, this module is the child: it
reads a JSON request on stdin, compiles (the build cache makes that a load)
and prints the timings as JSON.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tessera.ir.graph import NDArray


def fresh_process_runs(
    model: Path,
    inputs: Mapping[str, NDArray],
    *,
    target: str,
    options: Mapping[str, bool],
    input_shapes: Mapping[str, tuple[int, ...]] | None = None,
    processes: int = 5,
    runs: int = 10,
    warmup: int = 10,
) -> list[float]:
    """Seconds of ``processes`` x ``runs`` timed runs, each process new."""
    seconds: list[float] = []
    with tempfile.TemporaryDirectory(prefix="tessera-latency-") as tmp:
        request = _request(Path(tmp), model, inputs, target, options, input_shapes)
        request.update(runs=runs, warmup=warmup)
        for _ in range(processes):
            seconds += _run(request)
    return seconds


def fresh_process_profile(
    model: Path,
    inputs: Mapping[str, NDArray],
    *,
    target: str,
    options: Mapping[str, bool],
    processes: int = 5,
    repeat: int = 10,
) -> dict[str, list[float]]:
    """Each kernel's median time in each of ``processes`` new processes, by kernel."""
    pooled: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory(prefix="tessera-latency-") as tmp:
        request = _request(Path(tmp), model, inputs, target, options, None)
        request.update(mode="profile", runs=repeat)
        for _ in range(processes):
            for kernel, seconds in _run(request):
                pooled.setdefault(kernel, []).append(seconds)
    return pooled


def _request(
    tmp: Path,
    model: Path,
    inputs: Mapping[str, NDArray],
    target: str,
    options: Mapping[str, bool],
    input_shapes: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, Any]:
    feeds = tmp / "inputs.pickle"
    feeds.write_bytes(pickle.dumps({k: np.asarray(v) for k, v in inputs.items()}))
    return {
        "model": str(model),
        "inputs": str(feeds),
        "target": target,
        "options": dict(options),
        "input_shapes": {k: list(v) for k, v in (input_shapes or {}).items()},
        "mode": "benchmark",
        "runs": 10,
        "warmup": 10,
    }


def _run(request: dict[str, Any]) -> Any:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run(
        [sys.executable, "-m", __spec__.name],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _child(request: dict[str, Any]) -> Any:
    from tessera.driver import lower

    inputs = pickle.loads(Path(request["inputs"]).read_bytes())
    shapes = {k: tuple(v) for k, v in request["input_shapes"].items()} or None
    comp = lower(
        request["model"], target=request["target"], input_shapes=shapes, **request["options"]
    )
    compiled = comp.build()
    if request["mode"] == "profile":
        return [(k.kernel, k.seconds) for k in compiled.profile(inputs, repeat=request["runs"])]
    return compiled.benchmark(inputs, warmup=request["warmup"], repeat=request["runs"])


if __name__ == "__main__":
    print(json.dumps(_child(json.loads(sys.stdin.read()))))
