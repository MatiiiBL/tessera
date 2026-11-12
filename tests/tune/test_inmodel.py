"""Validating tuned configurations inside compiled models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import pytest

from e2e.onnx_models import mlp
from tessera.backend.c import CBackend
from tessera.driver import lower
from tessera.tune.inmodel import validate_in_model
from tessera.tune.log import BestConfigs, Record, TuningLog
from tessera.tune.space import Config


def template(sch: Any, lowered: Any, cfg: Config) -> None:
    loop = next(lp for lp in sch.get_loops(lowered.main_block) if sch.extent(lp) > 1)
    extent = sch.extent(loop)
    sch.split(loop, list(cfg.split("tile", extent, parts=2, default=(1, extent))))


def test_the_faster_configuration_in_the_model_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CBackend, "templates", lambda self, target=None: {"matmul": template})
    path = tmp_path / "mlp.onnx"
    onnx.save(mlp(sizes=(64, 256, 64)), path)
    comp = lower(path, target="c:generic", tuned=False)
    log = TuningLog(tmp_path / "log.jsonl")
    for key, task in comp.tasks.items():
        other = next(c for c in task.space if c != task.space.default())
        log.append(Record(comp.target.key, key, other, 1e-9))  # "tuned" by a lucky measurement
    seen: list[str] = []
    verdicts = validate_in_model([path], "c:generic", log, rounds=2, repeat=3, progress=seen.append)
    assert set(verdicts) == set(comp.tasks)
    assert seen
    best = BestConfigs(log.records())
    for key, (default, tuned) in verdicts.items():
        task = comp.tasks[key]
        chosen = best.lookup(comp.target.key, key)
        assert chosen is not None
        assert chosen.validation is not None
        assert chosen.validation.endswith("in the model")
        expected_default = default <= tuned
        assert (dict(chosen.config) == task.space.default()) is expected_default


def test_nothing_to_validate_without_tuned_configurations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CBackend, "templates", lambda self, target=None: {"matmul": template})
    path = tmp_path / "mlp.onnx"
    onnx.save(mlp(sizes=(64, 256, 64)), path)
    log = TuningLog(tmp_path / "log.jsonl")
    assert validate_in_model([path], "c:generic", log) == {}
    assert log.records() == []
