"""The parts of scripts/export_models.py that need neither PyTorch nor the network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tessera.bench.zoo import MODELS


def script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "export_models.py"
    spec = importlib.util.spec_from_file_location("export_models", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it runs
    spec.loader.exec_module(module)
    return module


def test_image_input_is_normalized_like_imagenet() -> None:
    m = script()
    image = m.image_input()["input"]
    assert image.shape == (1, 3, 224, 224)
    assert image.dtype == np.float32
    pixels = image[0].transpose(1, 2, 0) * m.IMAGENET_STD + m.IMAGENET_MEAN
    assert pixels.min() >= 0.0
    assert pixels.max() <= 1.0
    assert np.array_equal(image, m.image_input()["input"])


def test_exports_the_benchmark_models() -> None:
    m = script()
    assert tuple(m.EXPORTS) == MODELS
    assert m.OPSET == 17
    assert m.MAX_IR_VERSION <= 9


def test_rejects_unknown_names(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        script().main(["vgg16"])
    assert "unknown models ['vgg16']" in capsys.readouterr().err
