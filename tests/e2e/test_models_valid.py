"""The test models themselves: valid ONNX that ONNX Runtime runs, deterministic."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import onnx
import pytest

from e2e.onnx_models import cnn, mlp, onnxruntime_outputs, random_inputs, transformer_block

MODELS: dict[str, Callable[[], onnx.ModelProto]] = {
    "mlp": mlp,
    "cnn": cnn,
    "transformer_erf": transformer_block,
    "transformer_tanh": lambda: transformer_block(gelu="tanh"),
}
SHAPES = {
    "mlp": {"probs": (1, 10)},
    "cnn": {"logits": (1, 10)},
    "transformer_erf": {"y": (1, 16, 64)},
    "transformer_tanh": {"y": (1, 16, 64)},
}


@pytest.mark.parametrize("name", list(MODELS))
def test_model_runs_in_onnxruntime(name: str) -> None:
    model = MODELS[name]()
    out = onnxruntime_outputs(model, random_inputs(model))
    assert {k: v.shape for k, v in out.items()} == SHAPES[name]
    assert all(np.all(np.isfinite(v)) for v in out.values())
    again = MODELS[name]()
    assert again.SerializeToString() == model.SerializeToString()


def test_softmax_output_is_a_distribution() -> None:
    model = mlp()
    probs = onnxruntime_outputs(model, random_inputs(model))["probs"]
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
