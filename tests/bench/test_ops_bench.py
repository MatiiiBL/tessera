"""The kernel benchmark: GEMM models, convolution shapes, baselines and tables."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import numpy_helper

from e2e.onnx_models import onnxruntime_outputs
from tessera.bench.baselines import torch_available
from tessera.bench.ops import (
    ConvShape,
    OpResult,
    _conv_baseline,
    bench_gemm,
    conv_shape,
    gemm_model,
    markdown,
)
from tessera.bench.stats import Latency
from tessera.dtype import DType
from tessera.ir.graph import Node, TensorType, Value


def test_gemm_model_computes_x_times_w() -> None:
    model = gemm_model(4, 6, 5)
    x = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    w = numpy_helper.to_array(model.graph.initializer[0])
    np.testing.assert_allclose(onnxruntime_outputs(model, {"x": x})["y"], x @ w, rtol=1e-5)


def test_numpy_gemm_baseline() -> None:
    [r] = bench_gemm([(32, 16, 8)], baselines=["numpy"])
    assert r.system == "numpy"
    assert r.flops == 2 * 32 * 16 * 8
    assert r.gflops is not None
    assert r.gflops > 0


def test_conv_shapes() -> None:
    f32 = DType.float32
    x = Value("x", TensorType((1, 56, 56, 64), f32))
    w = Value("w", TensorType((3, 3, 64, 64), f32))
    y = Value("y", TensorType((1, 28, 28, 64), f32))
    attrs = {"strides": (2, 2), "pads": (1, 1, 1, 1), "dilations": (1, 1), "groups": 1}
    shape = conv_shape(Node("Conv2d", [x, w], [y], {**attrs, "layout": "NHWC"}))
    assert shape == ConvShape(56, 56, 64, 64, 3, 3, 2, (1, 1, 1, 1), 1)
    assert shape.out_hw == (28, 28)
    assert shape.flops == 2 * 28 * 28 * 64 * 9 * 64
    assert str(shape) == "3x3/2 56x56x64 -> 64"
    depthwise = ConvShape(14, 14, 32, 32, 3, 3, 1, (1, 1, 1, 1), 32)
    assert depthwise.flops == 2 * 14 * 14 * 32 * 9
    assert str(depthwise).startswith("dw ")


@pytest.mark.skipif(not torch_available("cpu"), reason="PyTorch is not installed")
def test_torch_conv_baseline() -> None:
    lat = _conv_baseline(ConvShape(8, 8, 4, 8, 3, 3, 1, (1, 1, 1, 1), 1), "cpu")
    assert lat.runs >= 50
    with pytest.raises(ValueError, match="asymmetric"):
        _conv_baseline(ConvShape(8, 8, 4, 8, 3, 3, 2, (0, 0, 1, 1), 1), "cpu")


def test_markdown_has_a_column_per_system() -> None:
    lat = Latency(0.001, 0.0009, 0.0011, 50)
    results = [
        OpResult("gemm 8x8x8", "numpy", lat, 2_000_000),
        OpResult("gemm 8x8x8", "tessera-c:sme", None, 2_000_000, "ValueError: x"),
    ]
    text = markdown(results)
    assert "| operation | numpy GFLOP/s (ms) | tessera-c:sme GFLOP/s (ms) |" in text
    assert "| gemm 8x8x8 | 2.0 (1.000 ms) | failed |" in text
