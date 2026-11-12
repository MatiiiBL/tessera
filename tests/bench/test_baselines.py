"""The model manifest and the ONNX Runtime and PyTorch baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tessera.bench.baselines import onnxruntime_cpu, torch_available, torch_eager
from tessera.bench.stats import time_runs
from tessera.bench.zoo import MissingModels, ModelSpec, load_manifest


def tiny_model(path: Path) -> np.ndarray:
    """y = relu(x @ w): a one-node-pair ONNX model; returns w."""
    w = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Relu", ["h"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path)
    return w


def write_manifest(root: Path, entries: object) -> None:
    (root / "manifest.json").write_text(json.dumps(entries))


def test_onnxruntime_runner(tmp_path: Path) -> None:
    w = tiny_model(tmp_path / "tiny.onnx")
    runner = onnxruntime_cpu(tmp_path / "tiny.onnx", threads=1)
    x = np.arange(8, dtype=np.float32).reshape(2, 4) - 3
    out = runner.outputs({"x": x})
    np.testing.assert_allclose(out["y"], np.maximum(x @ w, 0), rtol=1e-6)
    assert runner.threads == 1
    assert time_runs(runner.prepare({"x": x}), warmup=1, runs=50).median > 0


def test_manifest_forms_and_inputs(tmp_path: Path) -> None:
    np.savez(tmp_path / "tiny.npz", x=np.ones((2, 4), np.float32), y=np.zeros((2, 3), np.float32))
    entry = {
        "onnx": "data/models/tiny.onnx",
        "npz": "tiny.npz",
        "inputs": [["x", [2, 4], "float32"]],
        "outputs": [{"name": "y", "shape": [2, 3]}],
        "torch": {"library": "torchvision", "name": "resnet50", "weights": "IMAGENET1K_V1"},
    }
    for form in (
        {"tiny": entry},
        [{"name": "tiny", **entry}],
        {"models": [{"name": "tiny", **entry}]},
    ):
        write_manifest(tmp_path, form)
        spec = load_manifest(tmp_path)["tiny"]
        assert spec.inputs == (("x", (2, 4), "float32"),)
        assert spec.outputs == (("y", (2, 3)),)
        assert spec.onnx == tmp_path / "tiny.onnx"
    assert spec.sample_inputs()["x"].shape == (2, 4)
    assert set(spec.torch_outputs()) == {"y"}
    with pytest.raises(MissingModels, match="make models"):
        spec.onnx_path()


def test_random_inputs_respect_types() -> None:
    spec = ModelSpec(
        "bert",
        Path("bert.onnx"),
        Path("bert.npz"),
        (("input_ids", (1, 8), "int64"), ("attention_mask", (1, 8), "int64")),
        (("last_hidden_state", (1, 8, 768)),),
        {},
    )
    a, b = spec.random_inputs(0), spec.random_inputs(0)
    assert a["input_ids"].dtype == np.int64
    assert np.array_equal(a["input_ids"], b["input_ids"])
    assert np.all(a["attention_mask"] == 1)
    assert not spec.classifier


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(MissingModels, match="run `make models` first"):
        load_manifest(tmp_path)


@pytest.mark.skipif(not torch_available("cpu"), reason="PyTorch is not installed")
def test_torch_runner_needs_cached_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    spec = ModelSpec(
        "resnet50",
        tmp_path / "r.onnx",
        tmp_path / "r.npz",
        (("input", (1, 3, 224, 224), "float32"),),
        (("logits", (1, 1000)),),
        {"library": "torchvision", "name": "resnet50", "weights": "IMAGENET1K_V1"},
    )
    with pytest.raises(MissingModels, match="not cached"):
        torch_eager(spec)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=[
                pytest.mark.metal,
                pytest.mark.skipif(not torch_available("mps"), reason="no MPS device"),
            ],
        ),
    ],
)
@pytest.mark.skipif(not torch_available("cpu"), reason="PyTorch is not installed")
def test_torch_runner(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    import torch

    import tessera.bench.baselines as baselines

    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 3)
    monkeypatch.setattr(baselines, "torch_module", lambda spec: linear)
    spec = ModelSpec(
        "tiny", Path("t.onnx"), Path("t.npz"), (("x", (2, 4), "float32"),), (("y", (2, 3)),), {}
    )
    runner = torch_eager(spec, device)
    x = np.ones((2, 4), np.float32)
    out = runner.outputs({"x": x})
    expected = x @ linear.weight.detach().cpu().numpy().T + linear.bias.detach().cpu().numpy()
    np.testing.assert_allclose(out["y"], expected, rtol=1e-5, atol=1e-6)
    assert runner.name == f"pytorch-{device}"
    assert time_runs(runner.prepare({"x": x}), warmup=1, runs=50).runs == 50
