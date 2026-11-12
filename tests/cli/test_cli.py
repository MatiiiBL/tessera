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
