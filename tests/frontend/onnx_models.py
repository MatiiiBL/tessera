"""Small ONNX models written with onnx.helper, shaped like what PyTorch exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

type Array = np.ndarray[Any, Any]


class ModelBuilder:
    """Accumulates nodes and initializers; names are generated."""

    def __init__(self, seed: int = 0) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def name(self, hint: str) -> str:
        self.count += 1
        return f"/{hint}_{self.count}"

    def weight(self, *shape: int, scale: float = 0.3, data: Array | None = None) -> str:
        name = self.name("w")[1:]
        array = (
            data if data is not None else self.rng.standard_normal(shape).astype(np.float32) * scale
        )
        self.inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def ints(self, *values: int) -> str:
        return self.weight(data=np.array(values, np.int64))

    def index(self, value: int) -> str:
        return self.weight(data=np.array(value, np.int64))

    def scalar(self, value: float) -> str:
        return self.weight(data=np.array(value, np.float32))

    def __call__(self, op: str, *inputs: str, n_out: int = 1, **attrs: Any) -> Any:
        node_name = self.name(op)
        outs = [f"{node_name}_output_{i}" for i in range(n_out)]
        self.nodes.append(helper.make_node(op, list(inputs), outs, name=node_name, **attrs))
        return outs[0] if n_out == 1 else outs

    def model(
        self,
        inputs: Sequence[tuple[str, int, Sequence[int | str]]],
        outputs: Mapping[str, str],
        opset: int = 17,
    ) -> onnx.ModelProto:
        for alias, value in outputs.items():
            self.nodes.append(helper.make_node("Identity", [value], [alias]))
        graph = helper.make_graph(
            self.nodes,
            "test_model",
            [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in outputs],
            initializer=self.inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = 8
        return model


def run_ort(model: onnx.ModelProto, feeds: Mapping[str, Array]) -> dict[str, Array]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(names, dict(feeds)), strict=True))


def mlp() -> tuple[onnx.ModelProto, dict[str, Array]]:
    b = ModelBuilder(1)
    x = b("Flatten", "image", axis=1)
    h = b("Relu", b("Gemm", x, b.weight(32, 48), b.weight(32), transB=1))
    h = b("Gemm", h, b.weight(32, 16), b.weight(16), alpha=0.5, beta=2.0)
    h = b("Dropout", b("Tanh", h))
    y = b("Softmax", b("Add", b("MatMul", h, b.weight(16, 10)), b.weight(10)), axis=-1)
    model = b.model([("image", TensorProto.FLOAT, ["batch", 3, 4, 4])], {"probs": y})
    return model, {"image": b.rng.standard_normal((2, 3, 4, 4)).astype(np.float32)}


def cnn() -> tuple[onnx.ModelProto, dict[str, Array]]:
    b = ModelBuilder(2)

    def bn(x: str, c: int) -> str:
        var = np.abs(b.rng.standard_normal(c)).astype(np.float32) + 0.5
        return b(
            "BatchNormalization",
            x,
            b.weight(c),
            b.weight(c),
            b.weight(c),
            b.weight(data=var),
            epsilon=1e-5,
        )

    x = b("Relu", bn(b("Conv", "input", b.weight(8, 3, 3, 3), pads=[1, 1, 1, 1]), 8))
    x = b("MaxPool", x, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    shortcut = x
    y = b("Conv", x, b.weight(8, 1, 3, 3), b.weight(8), pads=[1, 1, 1, 1], group=8)
    y = b("Clip", bn(y, 8), b.scalar(0.0), b.scalar(6.0))
    y = b("Conv", y, b.weight(8, 8, 1, 1))
    x = b("Relu", b("Add", y, shortcut))
    x = b("Conv", x, b.weight(16, 8, 3, 3), b.weight(16), strides=[2, 2], auto_pad="SAME_UPPER")
    x = b("HardSwish", x)
    x = b("Flatten", b("GlobalAveragePool", x))
    logits = b("Gemm", x, b.weight(10, 16), b.weight(10), transB=1)
    model = b.model([("input", TensorProto.FLOAT, [1, 3, 16, 16])], {"logits": logits})
    return model, {"input": b.rng.standard_normal((1, 3, 16, 16)).astype(np.float32)}


def transformer_block(
    seq: int = 8, dim: int = 16, heads: int = 2
) -> tuple[onnx.ModelProto, dict[str, Array]]:
    """A BERT-style post-norm encoder layer, with exporter-style shape subgraphs."""
    b = ModelBuilder(3)
    head = dim // heads
    x = "hidden"

    def linear(v: str, n_in: int, n_out: int) -> str:
        return b("Add", b("MatMul", v, b.weight(n_in, n_out)), b.weight(n_out))

    def split_heads(v: str) -> str:
        # Shape -> Gather -> Unsqueeze -> Concat: the target shape is computed in the graph.
        shape = b("Shape", v)
        batch = b("Unsqueeze", b("Gather", shape, b.index(0), axis=0), b.ints(0))
        length = b("Unsqueeze", b("Gather", shape, b.index(1), axis=0), b.ints(0))
        target = b("Concat", batch, length, b.ints(heads), b.ints(head), axis=0)
        return b("Transpose", b("Reshape", v, target), perm=[0, 2, 1, 3])

    q, k, v = (split_heads(linear(x, dim, dim)) for _ in range(3))
    mask = b("Cast", b("Unsqueeze", "attention_mask", b.ints(1, 2)), to=TensorProto.FLOAT)
    mask = b("Mul", b("Sub", b.scalar(1.0), mask), b.scalar(-10000.0))
    scale = b("Sqrt", b("Cast", b("Gather", b("Shape", q), b.index(3), axis=0), to=1))
    scores = b("Div", b("MatMul", q, b("Transpose", k, perm=[0, 1, 3, 2])), scale)
    probs = b("Softmax", b("Add", scores, mask), axis=-1)
    context = b("Transpose", b("MatMul", probs, v), perm=[0, 2, 1, 3])
    context = b("Reshape", context, b.ints(0, 0, dim))
    attn = b("Add", linear(context, dim, dim), x)

    # Decomposed LayerNorm, as older exporters write it.
    mean = b("ReduceMean", attn, b.ints(-1))
    centered = b("Sub", attn, mean)
    var = b("ReduceMean", b("Pow", centered, b.scalar(2.0)), b.ints(-1))
    normed = b("Div", centered, b("Sqrt", b("Add", var, b.scalar(1e-12))))
    h = b("Add", b("Mul", normed, b.weight(dim)), b.weight(dim))

    # GELU (erf form), then the output projection and a fused LayerNormalization.
    up = linear(h, dim, 4 * dim)
    erf = b("Erf", b("Div", up, b.scalar(float(np.sqrt(2.0)))))
    gelu = b("Mul", b("Mul", up, b("Add", erf, b.scalar(1.0))), b.scalar(0.5))
    out = b("Add", linear(gelu, 4 * dim, dim), h)
    out = b("LayerNormalization", out, b.weight(dim), b.weight(dim), axis=-1, epsilon=1e-12)
    model = b.model(
        [
            ("hidden", TensorProto.FLOAT, [1, seq, dim]),
            ("attention_mask", TensorProto.INT64, [1, seq]),
        ],
        {"output": out},
        opset=18,
    )
    mask_values = np.ones((1, seq), np.int64)
    mask_values[0, seq - 2 :] = 0
    feeds = {
        "hidden": b.rng.standard_normal((1, seq, dim)).astype(np.float32),
        "attention_mask": mask_values,
    }
    return model, feeds
