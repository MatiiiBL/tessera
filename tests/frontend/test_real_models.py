"""The four benchmark models: import, optimize, fuse and plan, checked against ONNX Runtime.

Run with ``uv run pytest -m models``; the models come from
``scripts/export_models.py`` (``data/models``, or ``$TESSERA_MODELS``).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest

from tessera.frontend import onnx as onnx_frontend
from tessera.ir import Graph, interpret
from tessera.ir.opdefs import CANONICAL_OPS
from tessera.passes import fuse, optimize, plan_memory, verify_groups

pytestmark = pytest.mark.models

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TESSERA_MODELS", ROOT / "data" / "models"))
NAMES = ["resnet50", "mobilenet_v2", "bert", "gpt2"]


def _entry(name: str) -> dict[str, Any]:
    manifest = MODELS_DIR / "manifest.json"
    if not manifest.exists():
        pytest.skip(f"no exported models in {MODELS_DIR} (run scripts/export_models.py)")
    entries: dict[str, Any] = json.loads(manifest.read_text())
    if name not in entries:
        pytest.skip(f"{name} is not exported")
    return entries[name]


@cache
def _model(name: str) -> tuple[Graph, Graph, dict[str, np.ndarray], np.ndarray]:
    entry = _entry(name)
    path = MODELS_DIR / entry["onnx"]
    data = np.load(MODELS_DIR / entry["npz"])
    inputs = {i["name"]: data[i["name"]] for i in entry["inputs"]}
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (expected,) = session.run(None, inputs)
    imported = onnx_frontend.load(path)
    return imported, optimize(imported), inputs, expected


def _close(actual: np.ndarray, expected: np.ndarray) -> None:
    scale = max(1.0, float(np.abs(expected).max()))
    error = float(np.abs(actual - expected).max())
    assert error <= 1e-4 * scale, f"max abs error {error:.3g} (output scale {scale:.3g})"


@pytest.mark.parametrize("name", NAMES)
def test_optimized_model_matches_onnx_runtime(name: str) -> None:
    _, optimized, inputs, expected = _model(name)
    assert [v.name for v in optimized.inputs] == list(inputs)
    assert {n.op for n in optimized.nodes} <= CANONICAL_OPS
    (out,) = interpret(optimized, inputs).values()
    _close(out, expected)
    if name in ("resnet50", "mobilenet_v2"):
        assert out.argmax() == expected.argmax()


@pytest.mark.parametrize("name", NAMES)
def test_imported_and_unfolded_models_match_onnx_runtime(name: str) -> None:
    imported, _, inputs, expected = _model(name)
    for graph in (imported, optimize(imported, fold=False)):
        (out,) = interpret(graph, inputs).values()
        _close(out, expected)


@pytest.mark.parametrize("name", NAMES)
def test_model_fuses_and_plans(name: str) -> None:
    _, optimized, _, _ = _model(name)
    for enabled in (True, False):
        groups = fuse(optimized, enabled=enabled)
        verify_groups(optimized, groups)
        plan = plan_memory(optimized, groups)
        intermediates = sum(
            g.output.type.nbytes for g in groups if g.output not in optimized.outputs
        )
        assert 0 < plan.arena_bytes <= intermediates


@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_cnns_run_in_nhwc_with_one_transpose(name: str) -> None:
    _, optimized, _, _ = _model(name)
    ops = Counter(n.op for n in optimized.nodes)
    assert ops["Transpose"] == 1
    assert optimized.nodes[0].op == "Transpose"
    assert all(n.attrs["layout"] == "NHWC" for n in optimized.nodes if "layout" in n.attrs)
    patterns = Counter(g.pattern for g in fuse(optimized) if not g.is_view)
    assert patterns["injective"] == 0
    if name == "mobilenet_v2":
        assert patterns["depthwise_conv2d"] == 17


@pytest.mark.parametrize(("name", "approximate"), [("bert", "none"), ("gpt2", "tanh")])
def test_transformers_use_fused_layernorm_and_gelu(name: str, approximate: str) -> None:
    _, optimized, _, _ = _model(name)
    ops = Counter(n.op for n in optimized.nodes)
    assert ops["LayerNorm"] == 25
    assert ops["Gelu"] == 12
    assert ops["Erf"] == ops["Tanh"] == ops["Pow"] == 0
    assert {n.attrs["approximate"] for n in optimized.nodes if n.op == "Gelu"} == {approximate}
    groups = fuse(optimized)
    gelu_epilogues = [g for g in groups if g.pattern == "matmul" and g.nodes[-1].op == "Gelu"]
    assert len(gelu_epilogues) == 12
