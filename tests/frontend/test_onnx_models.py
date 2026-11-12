"""Whole small models: import, optimize and fuse, compared with ONNX Runtime."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx
import pytest
from onnx_models import Array, cnn, mlp, run_ort, transformer_block

from tessera.frontend import onnx as onnx_frontend
from tessera.ir import interpret
from tessera.ir.opdefs import CANONICAL_OPS
from tessera.passes import fuse, optimize, plan_memory, verify_groups

MODELS: dict[str, Callable[[], tuple[onnx.ModelProto, dict[str, Array]]]] = {
    "mlp": mlp,
    "cnn": cnn,
    "transformer_block": transformer_block,
}
SHAPES: dict[str, dict[str, tuple[int, ...]]] = {"mlp": {"image": (2, 3, 4, 4)}}


def close(a: Any, b: Any) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_import_matches_onnx_runtime(name: str) -> None:
    model, feeds = MODELS[name]()
    expected = run_ort(model, feeds)
    graph = onnx_frontend.load(model, SHAPES.get(name))
    graph.verify()
    assert [v.name for v in graph.inputs] == list(feeds)
    assert [v.name for v in graph.outputs] == list(expected)
    got = interpret(graph, feeds)
    for key, value in expected.items():
        close(got[key], value)


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("fold", [True, False])
def test_optimized_graph_matches_onnx_runtime(name: str, fold: bool) -> None:
    model, feeds = MODELS[name]()
    expected = run_ort(model, feeds)
    graph = optimize(onnx_frontend.load(model, SHAPES.get(name)), fold=fold)
    assert {n.op for n in graph.nodes} <= CANONICAL_OPS
    assert [v.name for v in graph.outputs] == list(expected)
    got = interpret(graph, feeds)
    for key, value in expected.items():
        close(got[key], value)
    groups = fuse(graph)
    verify_groups(graph, groups)
    plan = plan_memory(graph, groups)
    assert plan.arena_bytes >= 0


def test_shape_subgraphs_are_evaluated_at_import() -> None:
    model, _ = transformer_block()
    graph = onnx_frontend.load(model)
    ops = {n.op for n in graph.nodes}
    assert "Shape" not in ops
    assert "Concat" not in ops
    assert all(n.inputs and not all(v.is_constant for v in n.inputs) for n in graph.nodes)


def test_optimize_removes_import_operators() -> None:
    model, _ = mlp()
    imported = onnx_frontend.load(model, SHAPES["mlp"])
    assert {"Gemm", "Flatten", "Dropout"} <= {n.op for n in imported.nodes}
    optimized = optimize(imported)
    # Gemm's alpha and beta are folded into the constant weight and bias.
    assert [n.op for n in optimized.nodes] == [
        "Reshape",
        "MatMul",
        "Add",
        "Relu",
        "MatMul",
        "Add",
        "Tanh",
        "MatMul",
        "Add",
        "Softmax",
    ]
    assert imported.nodes[0].op == "Flatten", "optimize must not modify its argument"


def test_transformer_block_is_recognized() -> None:
    model, _ = transformer_block()
    ops = Counter(n.op for n in optimize(onnx_frontend.load(model)).nodes)
    assert ops["LayerNorm"] == 2
    assert ops["Gelu"] == 1
    assert ops["Erf"] == ops["Pow"] == ops["ReduceMean"] == ops["Sqrt"] == 0


def test_cnn_runs_in_nhwc() -> None:
    model, _ = cnn()
    graph = optimize(onnx_frontend.load(model))
    ops = Counter(n.op for n in graph.nodes)
    assert ops["BatchNorm"] == 0
    assert ops["Transpose"] == 1
    assert graph.nodes[0].op == "Transpose"
    assert all(n.attrs["layout"] == "NHWC" for n in graph.nodes if "layout" in n.attrs)
