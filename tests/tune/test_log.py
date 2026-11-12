"""Tuning logs: format, append-only resumable files and best-config lookup."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from tessera.tune.log import BestConfigs, Record, TuningLog
from tessera.tune.space import KnobValue

knob_values = st.one_of(
    st.integers(-5, 5000),
    st.booleans(),
    st.sampled_from(["a", "nhwc", "x y"]),
    st.lists(st.integers(1, 512), min_size=1, max_size=4).map(tuple),
)
configs = st.dictionaries(st.text("abcdefgh_", min_size=1, max_size=8), knob_values, max_size=6)


@given(configs, st.one_of(st.none(), st.floats(1e-9, 10.0)))
def test_records_round_trip(config: dict[str, KnobValue], seconds: float | None) -> None:
    when = "2026-11-12T10:00:00Z"
    record = Record("c-sme-t4-Apple_M4", "matmul:abc", config, seconds, None, when)
    assert Record.from_json(record.to_json()) == record


def test_line_format_matches_the_design() -> None:
    line = Record("t", "matmul:1", {"tile_m": (8, 16)}, 0.000412, None).to_json()
    raw = json.loads(line)
    assert list(raw) == ["target", "task", "config", "seconds", "error", "time"]
    assert raw["config"] == {"tile_m": [8, 16]}
    assert raw["time"].endswith("Z")


def test_append_resume_and_partial_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"
    log = TuningLog(path)
    log.append(Record("t", "a", {"x": 1}, 1.0))
    log.append(Record("t", "a", {"x": 2}, None, "build: ScheduleError: no"))
    assert len(log.records()) == 2
    with path.open("a") as f:
        f.write("not json\n")
        f.write('{"target": "t", "task": "a", "config": {"x": 3}, "seconds": 0.5')  # cut off
    assert len(log.records()) == 2
    with path.open("a") as f:
        f.write(', "error": null, "time": ""}\n')
    reopened = TuningLog(path)
    assert [r.config["x"] for r in reopened.records()] == [1, 2, 3]
    assert set(reopened.measured("t", "a")) == {'{"x": 1}', '{"x": 2}', '{"x": 3}'}
    assert reopened.measured("other", "a") == {}


def test_in_memory_log() -> None:
    log = TuningLog()
    log.append(Record("t", "a", {"x": 1}, 1.0))
    assert log.records()[0].time != ""


def test_best_configs_skips_failures_and_rejected_configs(tmp_path: Path) -> None:
    log = TuningLog(tmp_path / "a.jsonl")
    log.append(Record("t", "a", {"x": 1}, 3.0))
    log.append(Record("t", "a", {"x": 2}, 1.0))
    log.append(Record("t", "a", {"x": 3}, None, "measure: timed out after 60 s"))
    TuningLog(tmp_path / "sub" / "b.jsonl").append(Record("t", "b", {"y": 1}, 2.0))
    best = BestConfigs.load(tmp_path, tmp_path / "missing.jsonl")
    assert len(best) == 2
    found = best.lookup("t", "a")
    assert found is not None
    assert found.config == {"x": 2}
    fallback = best.lookup("t", "a", accepts=lambda c: c["x"] != 2)
    assert fallback is not None
    assert fallback.config == {"x": 1}
    assert best.lookup("t", "missing") is None
    assert BestConfigs.load(tmp_path / "a.jsonl").lookup("t", "b") is None
