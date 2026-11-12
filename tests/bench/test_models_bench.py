"""The model benchmark on a small model, and its tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest

from e2e.onnx_models import cnn
from tessera.bench.models import (
    CLASSIFIER_SAMPLES,
    ModelResult,
    SystemResult,
    bench_model,
    markdown,
    samples,
)
from tessera.bench.stats import Agreement, Latency
from tessera.bench.zoo import load_manifest


def tiny_zoo(tmp_path: Path) -> Path:
    model = cnn()
    onnx.save(model, tmp_path / "resnet50.onnx")
    image = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype(np.float32)
    np.savez(tmp_path / "resnet50.npz", image=image)
    entry = {
        "onnx": "resnet50.onnx",
        "npz": "resnet50.npz",
        "inputs": [{"name": "image", "shape": [1, 3, 32, 32], "dtype": "float32"}],
        "outputs": [{"name": "logits", "shape": [1, 10]}],
        "torch": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"resnet50": entry}))
    return tmp_path


def test_samples_and_baseline_only_bench(tmp_path: Path) -> None:
    spec = load_manifest(tiny_zoo(tmp_path))["resnet50"]  # named like a classifier
    inputs = samples(spec)
    assert len(inputs) == CLASSIFIER_SAMPLES
    assert np.array_equal(inputs[0]["image"], spec.sample_inputs()["image"])
    seen: list[str] = []
    result = bench_model(spec, targets=[], baselines=["onnxruntime-cpu"], progress=seen.append)
    [ort] = result.systems
    assert ort.latency is not None
    assert ort.latency.runs >= 50
    assert ort.agreement == Agreement(0.0, 0.0, 1.0)
    assert seen == ["resnet50: onnxruntime-cpu"]


def test_failures_are_reported_not_raised(tmp_path: Path) -> None:
    spec = load_manifest(tiny_zoo(tmp_path))["resnet50"]
    result = bench_model(spec, targets=["cuda"], baselines=[], variants=())
    assert [s.system for s in result.systems] == ["tessera-cuda-untuned"]  # no log: no tuned row
    assert all("unknown target" in (s.error or "") for s in result.systems)


def test_markdown_tables() -> None:
    lat = Latency(0.010, 0.009, 0.012, 50)
    systems = [
        SystemResult("onnxruntime-cpu", lat, Agreement(0, 0, 1.0), 4),
        SystemResult(
            "tessera-c:sme",
            Latency(0.005, 0.004, 0.006, 50),
            Agreement(1e-5, 2e-6, 1.0),
            4,
            details={"kernels": 54, "tuned_kernels": 50},
        ),
        SystemResult("pytorch-mps", error="RuntimeError: no device"),
    ]
    variant = {
        "kernels": 54,
        "bytes_moved": 2 * 2**20,
        "arena_bytes": 2**20,
        "intermediate_bytes": 3 * 2**20,
        "latency": lat.as_dict(),
    }
    ablations = {"c:sme": {"full": variant, "fusion off": {"error": "ValueError: x"}}}
    text = markdown([ModelResult("resnet50", systems, ablations)])
    assert "| tessera-c:sme | 5.00 (4.00 to 6.00) | 2.00x | 1.00e-05 | 2.00e-06 | 100% |" in text
    assert "54 kernels, 50 tuned" in text
    assert "| pytorch-mps | failed |" in text
    assert "| c:sme | full | 54 | 2.0 | 1.0 | 3.0 | 10.00 | 50 |" in text
    assert "| c:sme | fusion off | failed |" in text


def test_results_round_trip_through_json() -> None:
    lat = Latency(0.010, 0.009, 0.012, 50)
    result = ModelResult(
        "resnet50",
        [
            SystemResult("onnxruntime-cpu", lat, Agreement(0, 0, 1.0), 4),
            SystemResult("x", error="E", details={"kernels": 3}),
        ],
        {"c:sme": {"full": {"kernels": 3}}},
    )
    assert ModelResult.from_dict(json.loads(json.dumps(result.as_dict()))) == result


def test_pages_merge_models_measured_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tessera.bench.pages import models_page

    zoo = tiny_zoo(tmp_path)
    manifest = json.loads((zoo / "manifest.json").read_text())
    manifest["mobilenet_v2"] = manifest["resnet50"]
    (zoo / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("TESSERA_MODELS", str(zoo))
    out = tmp_path / "results"
    for name in ("mobilenet_v2", "resnet50"):
        models_page(
            [],
            names=[name],
            ablation_targets=["c:generic"],
            variants=["no schedule"],
            provisional=True,
            directory=out,
            progress=lambda message: None,
        )
    data = json.loads((out / "models.json").read_text())
    assert [m["model"] for m in data["models"]] == ["resnet50", "mobilenet_v2"]
    page = (out / "models.md").read_text()
    assert page.index("### resnet50") < page.index("### mobilenet_v2")
    assert "| c:generic | no schedule |" in page
    assert page.count("(provisional).") == 2


def test_reruns_replace_only_what_they_measure() -> None:
    from tessera.bench.pages import merge_model

    first = {"date": "2026-11-12T01:00:00Z", "provisional": True}
    second = {"date": "2026-11-12T02:00:00Z", "provisional": False}
    old = merge_model(
        None,
        {
            "model": "m",
            "systems": [{"system": "a", "v": 1}, {"system": "b", "v": 1}],
            "ablations": {"c:sme": {"no schedule": {"v": 1}}},
        },
        first,
    )
    new = merge_model(
        old,
        {
            "model": "m",
            "systems": [{"system": "b", "v": 2}],
            "ablations": {"c:sme": {"full": {"v": 2}}, "metal": {"full": {"v": 2}}},
        },
        second,
    )
    assert [(s["system"], s["v"], s["measured"]) for s in new["systems"]] == [
        ("b", 2, second["date"]),
        ("a", 1, first["date"]),
    ]
    assert set(new["ablations"]["c:sme"]) == {"no schedule", "full"}
    assert new["ablations"]["c:sme"]["no schedule"]["measured"] == first["date"]
    assert new["runs"] == [first, second]


def test_unscheduled_metal_is_not_measured(tmp_path: Path) -> None:
    from tessera.bench.models import METAL_UNSCHEDULED, ablate

    spec = load_manifest(tiny_zoo(tmp_path))["resnet50"]
    out = ablate(spec, "metal", ["no schedule"], [], [], lambda message: None)
    assert out == {"no schedule": {"system": "no schedule", "skipped": METAL_UNSCHEDULED}}
    text = markdown([ModelResult("resnet50", [], {"metal": out})])
    assert "| metal | no schedule |" in text
    assert "one GPU thread" in text
