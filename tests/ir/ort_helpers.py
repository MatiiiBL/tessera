"""Build tiny ONNX models with onnx.helper and run them in ONNX Runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from tessera.ir import Attr, Graph, TensorType, interpret

type Array = np.ndarray[Any, Any]


def onnx_type(dtype: np.dtype[Any]) -> int:
    return int(helper.np_dtype_to_tensor_dtype(np.dtype(dtype)))


def run_onnx(
    nodes: Sequence[onnx.NodeProto],
    inputs: Mapping[str, Array],
    outputs: Mapping[str, np.dtype[Any] | type],
    *,
    constants: Mapping[str, Array] | None = None,
    opset: int = 17,
) -> list[Array]:
    """Run ``nodes`` in ONNX Runtime; ``constants`` become initializers."""
    graph = helper.make_graph(
        list(nodes),
        "test",
        [helper.make_tensor_value_info(k, onnx_type(v.dtype), v.shape) for k, v in inputs.items()],
        [
            helper.make_tensor_value_info(k, onnx_type(np.dtype(t)), None)
            for k, t in outputs.items()
        ],
        initializer=[
            numpy_helper.from_array(np.asarray(v), k) for k, v in (constants or {}).items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    feeds = {k: np.asarray(v) for k, v in inputs.items()}
    return [np.asarray(r) for r in session.run(list(outputs), feeds)]


def run_op(op: str, args: Sequence[Array], attrs: Mapping[str, Attr] | None = None) -> Array:
    """Run one Tessera operator through the interpreter (every argument a graph input)."""
    g = Graph(op.lower())
    values = [g.add_input(f"in{i}", TensorType.of(np.asarray(a))) for i, a in enumerate(args)]
    out = g.op(op, *values, **dict(attrs or {}))
    g.set_outputs([out])
    g.verify()
    result = interpret(g, {f"in{i}": np.asarray(a) for i, a in enumerate(args)})
    return result[out.name]


def assert_close(actual: Array, expected: Array, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    if actual.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def floats(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> Array:
    return np.asarray(rng.standard_normal(shape) * scale, np.float32)
