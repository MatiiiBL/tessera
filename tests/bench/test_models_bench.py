"""The model benchmark on a small model, and its tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx

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
    result = bench_model(spec, targets=["cuda"], baselines=[], ablations=False)
    assert [s.system for s in result.systems] == ["tessera-cuda", "tessera-cuda-untuned"]
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
        "bytes_moved": 2_000_000,
        "arena_bytes": 1_000_000,
        "intermediate_bytes": 3_000_000,
        "latency": lat.as_dict(),
    }
    ablations = {"c:sme": {"full": variant, "fusion off": {"error": "ValueError: x"}}}
    text = markdown([ModelResult("resnet50", systems, ablations)])
    assert "| tessera-c:sme | 5.00 (4.00 to 6.00) | 2.00x | 1.00e-05 | 2.00e-06 | 100% |" in text
    assert "54 kernels, 50 tuned" in text
    assert "| pytorch-mps | failed |" in text
    assert "| c:sme | full | 54 | 2.0 | 1.0 | 3.0 | 10.00 |" in text
    assert "| c:sme | fusion off | failed |" in text
