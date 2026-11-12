"""The kernel benchmarks run and write their result files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from metal_funcs import requires_gpu

from tessera.backend.metal import bench

pytestmark = requires_gpu


def test_launch_benchmark_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "LAUNCH_COUNTS", (1, 4, 16))
    render = bench.BENCHMARKS["launch"][1]
    monkeypatch.setitem(bench.BENCHMARKS, "launch", (lambda: bench.bench_launch(repeat=3), render))
    md = bench.run("launch", tmp_path)
    data = json.loads((tmp_path / "launch.json").read_text())
    assert {m["mode"] for m in data["modes"]} == set(bench.LAUNCH_MODES)
    assert all(m["gpu_per_launch"] > 0 for m in data["modes"] if m["mode"] != "independent")
    assert data["environment"]["machine"]
    text = md.read_text()
    assert text.startswith("# Metal: the cost of a kernel launch")
    assert "python -m tessera.backend.metal.bench launch" in text
