"""Benchmark statistics, agreement metrics and result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tessera.bench.report import environment, header, markdown_table, write
from tessera.bench.stats import MIN_RUNS, Latency, agreement, time_runs


def test_latency_percentiles() -> None:
    lat = Latency.of([float(x) for x in range(1, 102)])
    assert (lat.p10, lat.median, lat.p90, lat.runs) == (11.0, 51.0, 91.0, 101)
    assert Latency.of([0.001, 0.002, 0.003]).ms() == "2.00 (1.20 to 2.80)"
    with pytest.raises(ValueError, match="no timings"):
        Latency.of([])


def test_time_runs_counts_warmup_and_never_stops_before_the_minimum() -> None:
    calls = []
    lat = time_runs(lambda: calls.append(1), warmup=3, runs=60)
    assert len(calls) == 63
    assert lat.runs == 60
    lat = time_runs(lambda: None, warmup=0, runs=1000, max_seconds=0.0)
    assert lat.runs == MIN_RUNS


def test_agreement() -> None:
    ref = [{"y": np.array([[1.0, 4.0, 2.0]])}, {"y": np.array([[3.0, 0.0, 1.0]])}]
    out = [{"y": np.array([[1.0, 4.1, 2.0]])}, {"y": np.array([[0.5, 2.0, 1.0]])}]
    a = agreement(out, ref, classifier=True)
    assert a.max_abs == pytest.approx(2.5)
    assert a.max_rel == pytest.approx(2.5 / 3.0)
    assert a.top1 == 0.5
    assert agreement(ref, ref).as_dict() == {"max_abs": 0.0, "max_rel": 0.0, "top1": None}
    with pytest.raises(ValueError, match="names differ"):
        agreement([{"z": ref[0]["y"]}], ref[:1])
    with pytest.raises(ValueError, match="shape"):
        agreement([{"y": np.zeros(2)}], ref[:1])


def test_result_files(tmp_path: Path) -> None:
    env = environment(provisional=True)
    assert env["provisional"] is True
    assert env["performance_cores"] >= 1
    text = header("Models", env, "tessera bench models")
    assert "**Provisional:**" in text
    assert "tessera bench models" in text
    assert "Provisional" not in header("Models", {**env, "provisional": False}, "x")
    table = markdown_table(["model", "ms"], [["resnet50", "1.00"]])
    assert table.splitlines() == ["| model | ms |", "|---|---|", "| resnet50 | 1.00 |"]
    md = write("models", {"env": env}, text + table, tmp_path)
    assert md.read_text().endswith("| resnet50 | 1.00 |\n")
    assert json.loads((tmp_path / "models.json").read_text())["env"]["provisional"] is True
