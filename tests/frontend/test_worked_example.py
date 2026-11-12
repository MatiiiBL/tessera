"""The worked example of docs/graph.md: a residual CNN block, ONNX to memory plan."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx_models import run_ort

from tessera.frontend import onnx as onnx_frontend
from tessera.ir import interpret
from tessera.passes import fuse, optimize, plan_memory
from tessera.plan import StorageKind


def block() -> onnx.ModelProto:
    rng = np.random.default_rng(0)

    def w(name: str, *shape: int) -> onnx.TensorProto:
        return numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), name)

    var = (np.abs(rng.standard_normal(8)) + 0.5).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "w1"], ["c1"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["c1", "g", "b", "m", "v"], ["bn"], name="bn1"),
        helper.make_node("Relu", ["bn"], ["r1"], name="relu1"),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], name="conv2", pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["c2", "r1"], ["s"], name="residual"),
        helper.make_node("Relu", ["s"], ["r2"], name="relu2"),
        helper.make_node("GlobalAveragePool", ["r2"], ["p"], name="pool"),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wfc", "bfc"], ["logits"], name="fc", transB=1),
    ]
    inits = [
        w("w1", 8, 3, 3, 3),
        w("g", 8),
        w("b", 8),
        w("m", 8),
        numpy_helper.from_array(var, "v"),
        w("w2", 8, 8, 3, 3),
        w("b2", 8),
        w("wfc", 10, 8),
        w("bfc", 10),
    ]
    graph = helper.make_graph(
        nodes,
        "block",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_worked_example() -> None:
    model = block()
    imported = onnx_frontend.load(model)
    assert [n.op for n in imported.nodes] == [
        "Conv2d",
        "BatchNorm",
        "Relu",
        "Conv2d",
        "Add",
        "Relu",
        "ReduceMean",
        "Flatten",
        "Gemm",
    ]
    optimized = optimize(imported)
    assert [(n.op, n.name) for n in optimized.nodes] == [
        ("Transpose", "conv1_nhwc"),
        ("Conv2d", "conv1"),
        ("Relu", "relu1"),
        ("Conv2d", "conv2"),
        ("Add", "residual"),
        ("Relu", "relu2"),
        ("ReduceMean", "pool"),
        ("Reshape", "flatten"),
        ("MatMul", "fc"),
        ("Add", "fc_bias"),
    ]
    groups = fuse(optimized)
    assert [(g.name, g.pattern, [n.op for n in g.nodes]) for g in groups] == [
        ("conv1", "conv2d", ["Transpose", "Conv2d", "Relu"]),
        ("conv2", "conv2d", ["Conv2d", "Add", "Relu"]),
        ("pool", "reduce", ["ReduceMean"]),
        ("fc", "matmul", ["Reshape", "MatMul", "Add"]),
    ]
    plan = plan_memory(optimized, groups)
    arena = {k: (r.offset, r.nbytes) for k, r in plan.refs.items() if r.kind is StorageKind.ARENA}
    assert arena == {"relu1": (0, 32768), "relu2": (32768, 32768), "pool": (0, 32)}
    assert plan.arena_bytes == 65536
    assert len(fuse(optimized, enabled=False)) == 10

    feeds = {"x": np.random.default_rng(1).standard_normal((1, 3, 32, 32)).astype(np.float32)}
    expected = run_ort(model, feeds)["logits"]
    np.testing.assert_allclose(
        interpret(optimized, feeds)["logits"], expected, rtol=1e-4, atol=1e-5
    )
