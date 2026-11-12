"""The interpreter benchmark runs and writes its results files."""

from __future__ import annotations

import json
from pathlib import Path

from tessera.tir import bench


def test_bench_writes_results(tmp_path: Path) -> None:
    rows = bench.main(tmp_path, repeat=1)
    assert [r.kernel for r in rows] == list(bench.KERNELS)
    assert all(r.executions > 0 for r in rows)
    assert all(r.run_seconds > 0 for r in rows)
    data = json.loads((tmp_path / "tir_interpreter.json").read_text())
    assert len(data["rows"]) == len(rows)
    assert "| matmul 64x64x64 |" in (tmp_path / "tir_interpreter.md").read_text()
