"""The tessera command, on a model the whole pipeline handles today."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import typer
from typer.testing import CliRunner

from e2e.onnx_models import views_model
from tessera.cli import app
from tessera.cli.app import parse_shapes

runner = CliRunner()


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "views.onnx"
    onnx.save(views_model(), path)
    return path


def invoke(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_run_check(model: Path) -> None:
    out = invoke("run", str(model), "--target", "c:generic", "--check")
    assert "match: worst relative error 0" in out
    assert "(6, 4)" in out


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("graph", "Reshape"),
        ("optimized", "return %y, %z"),
        ("fused", "view"),
        ("tir", "def copy_y("),
        ("code", "==== model.c"),
    ],
)
def test_inspect_stages(model: Path, stage: str, expected: str) -> None:
    assert expected in invoke("inspect", str(model), "-t", "c:generic", "--stage", stage)


def test_inspect_one_group(model: Path) -> None:
    out = invoke("inspect", str(model), "-t", "c:generic", "-s", "tir", "--group", "copy_z")
    assert "copy_z" in out
    assert "copy_y" not in out


def test_compile_writes_sources(model: Path, tmp_path: Path) -> None:
    out = invoke("compile", str(model), "-t", "c:generic", "-o", str(tmp_path / "out"))
    assert "2 (2 copies)" in out
    kernels = json.loads((tmp_path / "out" / "kernels.json").read_text())
    assert [k["name"] for k in kernels["kernels"]] == ["copy_y", "copy_z"]
    assert (tmp_path / "out" / "model.c").read_text().startswith("/*")


def test_profile(model: Path) -> None:
    out = invoke("profile", str(model), "-t", "c:generic", "--repeat", "3")
    assert "copy_y" in out
    assert "over 2 kernels" in out


def test_tune_without_templates(model: Path, tmp_path: Path) -> None:
    out = invoke("tune", str(model), "-t", "c:generic", "--log", str(tmp_path / "log.jsonl"))
    assert "nothing to tune" in out


def test_shapes() -> None:
    assert parse_shapes(["x=1,3,224,224", "mask=1,128"]) == {
        "x": (1, 3, 224, 224),
        "mask": (1, 128),
    }
    assert parse_shapes(None) is None
    for bad in ("x", "=1,2", "x=1,a"):
        with pytest.raises(typer.BadParameter):
            parse_shapes([bad])


def test_missing_model() -> None:
    result = runner.invoke(app, ["run", "nope.onnx"])
    assert result.exit_code != 0


def test_bench_tuning_writes_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESSERA_RESULTS", str(tmp_path))
    out = invoke(
        "bench", "tuning", "--trials", "16", "--seeds", "1", "--real-tasks", "0", "--final"
    )
    assert "tuning.md" in out
    page = (tmp_path / "tuning.md").read_text()
    assert page.startswith("# Tuning")
    assert "Provisional" not in page
    data = json.loads((tmp_path / "tuning.json").read_text())
    assert len(data["synthetic"]) == 3


def test_bench_models_on_a_stand_in_zoo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.test_models_bench import tiny_zoo

    (tmp_path / "models").mkdir()
    zoo = tiny_zoo(tmp_path / "models")
    monkeypatch.setenv("TESSERA_MODELS", str(zoo))
    monkeypatch.setenv("TESSERA_RESULTS", str(tmp_path / "results"))
    invoke("bench", "models", "--model", "resnet50", "--target", "c:generic", "--provisional")
    page = (tmp_path / "results" / "models.md").read_text()
    assert "**Provisional:**" in page
    assert "| tessera-c:generic-untuned |" in page
    assert "| c:generic | fusion off |" in page
    data = json.loads((tmp_path / "results" / "models.json").read_text())
    systems = {s["system"]: s for s in data["models"][0]["systems"]}
    assert systems["tessera-c:generic-untuned"]["agreement"]["top1"] == 1.0
    assert systems["onnxruntime-cpu"]["latency"]["runs"] >= 50


def test_run_check_with_onnx_names(tmp_path: Path) -> None:
    from e2e.onnx_models import dotted_model

    path = tmp_path / "dotted.onnx"
    onnx.save(dotted_model(), path)
    out = invoke("run", str(path), "-t", "c:generic", "--check")
    assert "out.0" in out
    assert "match" in out


def test_tune_shares_one_budget_across_models(tmp_path: Path) -> None:
    from e2e.onnx_models import cnn, mlp
    from tessera.tune.log import TuningLog

    paths = []
    for name, model in (("mlp", mlp()), ("cnn", cnn())):
        paths.append(tmp_path / f"{name}.onnx")
        onnx.save(model, paths[-1])
    log = tmp_path / "c-generic.jsonl"
    args = ["tune", *map(str, paths), "-t", "c:generic", "--trials", "24", "--batch", "8"]
    out = invoke(
        *args,
        "--log",
        str(log),
        "--strategy",
        "random",
        "--validate-rounds",
        "0",
        "--model-rounds",
        "0",
    )
    assert "of 2 models" in out
    records = TuningLog(log).records()
    assert len(records) == 24
    assert {r.task.split(":")[0] for r in records} >= {"matmul"}
    # Resumed: the budget is spent, so only the validation measures anything.
    again = invoke(*args, "--log", str(log), "--strategy", "random", "--validate-rounds", "1")
    assert "speedup" in again
    validated = TuningLog(log).records()[24:]
    assert validated
    assert all(r.ok for r in validated)
    assert {r.validation.split(" ", 1)[1] for r in validated if r.validation} <= {
        "measured alone",
        "in the model",
    }
    assert "inside the models" in again


def test_tune_new_only_skips_tasks_already_in_the_log(tmp_path: Path) -> None:
    from e2e.onnx_models import cnn, mlp
    from tessera.tune.log import TuningLog

    paths = {}
    for name, model in (("mlp", mlp()), ("cnn", cnn())):
        paths[name] = tmp_path / f"{name}.onnx"
        onnx.save(model, paths[name])
    log = tmp_path / "c-generic.jsonl"
    common = ["-t", "c:generic", "--log", str(log), "--strategy", "random"]
    quick = ["--validate-rounds", "0", "--model-rounds", "0"]
    invoke("tune", str(paths["mlp"]), *common, *quick, "--trials", "16")
    mlp_tasks = {r.task for r in TuningLog(log).records()}
    out = invoke(
        "tune",
        str(paths["mlp"]),
        str(paths["cnn"]),
        *common,
        *quick,
        "--trials",
        "16",
        "--new-only",
    )
    assert "of 2 models" in out
    added = TuningLog(log).records()[16:]
    assert added
    assert not {r.task for r in added} & mlp_tasks
