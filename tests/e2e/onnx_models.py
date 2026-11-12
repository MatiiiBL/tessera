"""Small ONNX models built with ``onnx.helper`` for the end-to-end tests.

They use the operator forms real exporters emit (Gemm with a transposed
weight, BatchNormalization after Conv, Flatten, GlobalAveragePool, a
decomposed GELU, multi-head attention through Reshape and Transpose) so
that importing, optimizing and fusing them exercises the same paths as the
benchmark models, at sizes that compile and run in seconds.

Models are written with IR version 9: onnx 1.23 defaults to version 14,
which ONNX Runtime 1.31 refuses to load.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
IR_VERSION = 9


class _Builder:
    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.count = 0

    def fresh(self, hint: str) -> str:
        self.count += 1
        return f"{hint}_{self.count}"

    def const(self, array: np.ndarray, hint: str = "c") -> str:
        name = self.fresh(hint)
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def weight(self, *shape: int, scale: float | None = None, hint: str = "w") -> str:
        fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
        s = scale if scale is not None else 1.0 / np.sqrt(fan_in)
        return self.const((self.rng.standard_normal(shape) * s).astype(np.float32), hint)

    def op(
        self, op_type: str, inputs: Sequence[str], hint: str | None = None, **attrs: object
    ) -> str:
        out = self.fresh(hint or op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], name=out, **attrs))
        return out

    def model(
        self, inputs: Sequence[tuple[str, Sequence[int]]], outputs: Sequence[tuple[str, str]]
    ) -> onnx.ModelProto:
        """``outputs`` maps internal value names to the output names to expose."""
        for value, name in outputs:
            self.nodes.append(helper.make_node("Identity", [value], [name], name=name))
        graph = helper.make_graph(
            self.nodes,
            self.name,
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(s)) for n, s in inputs],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for _, n in outputs],
            self.initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
        )
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        return model


def mlp(seed: int = 0, sizes: Sequence[int] = (64, 128, 64, 10)) -> onnx.ModelProto:
    """Gemm + Relu layers, then Softmax: input ``x`` [1, 64], output ``probs`` [1, 10]."""
    b = _Builder("mlp", seed)
    h = "x"
    for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
        w = b.weight(n_out, n_in)
        bias = b.weight(n_out, scale=0.1, hint="b")
        h = b.op("Gemm", [h, w, bias], transB=1)
        if i < len(sizes) - 2:
            h = b.op("Relu", [h])
    return b.model([("x", (1, sizes[0]))], [(b.op("Softmax", [h], axis=-1), "probs")])


def _conv_bn(
    b: _Builder, x: str, c_in: int, c_out: int, *, stride: int = 1, groups: int = 1
) -> str:
    w = b.weight(c_out, c_in // groups, 3, 3)
    y = b.op(
        "Conv",
        [x, w],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[stride, stride],
        group=groups,
    )
    scale = b.const((1.0 + 0.1 * b.rng.standard_normal(c_out)).astype(np.float32), "gamma")
    shift = b.const((0.1 * b.rng.standard_normal(c_out)).astype(np.float32), "beta")
    mean = b.const((0.1 * b.rng.standard_normal(c_out)).astype(np.float32), "mean")
    var = b.const((0.5 + b.rng.random(c_out)).astype(np.float32), "var")
    return b.op("BatchNormalization", [y, scale, shift, mean, var], epsilon=1e-5)


def cnn(seed: int = 0, channels: int = 16, size: int = 32) -> onnx.ModelProto:
    """Conv-BN-ReLU stem, two residual blocks, a strided and a depthwise conv,
    max pooling, global average pooling and a classifier: ``image`` [1, 3, 32, 32]
    to ``logits`` [1, 10]."""
    b = _Builder("cnn", seed)
    c = channels
    h = b.op("Relu", [_conv_bn(b, "image", 3, c)])
    for _ in range(2):
        r = b.op("Relu", [_conv_bn(b, h, c, c)])
        h = b.op("Relu", [b.op("Add", [_conv_bn(b, r, c, c), h])])
    h = b.op("Relu", [_conv_bn(b, h, c, 2 * c, stride=2)])
    h = b.op(
        "Clip",
        [
            _conv_bn(b, h, 2 * c, 2 * c, groups=2 * c),
            b.const(np.float32(0.0)),
            b.const(np.float32(6.0)),
        ],
    )
    h = b.op("MaxPool", [h], kernel_shape=[2, 2], strides=[2, 2])
    h = b.op("Flatten", [b.op("GlobalAveragePool", [h])], axis=1)
    logits = b.op("Gemm", [h, b.weight(10, 2 * c), b.weight(10, scale=0.1, hint="b")], transB=1)
    return b.model([("image", (1, 3, size, size))], [(logits, "logits")])


def transformer_block(
    seed: int = 0, seq: int = 16, dim: int = 64, heads: int = 4, gelu: str = "erf"
) -> onnx.ModelProto:
    """A pre-norm transformer block: LayerNorm, multi-head self-attention with a
    residual, LayerNorm, a GELU MLP with a residual. ``x`` [1, seq, dim] to ``y``.
    ``gelu`` is "erf" (BERT) or "tanh" (GPT-2), both written out elementwise."""
    b = _Builder("transformer_block", seed)
    head = dim // heads

    def layer_norm(x: str) -> str:
        gamma = b.const((1.0 + 0.1 * b.rng.standard_normal(dim)).astype(np.float32), "ln_g")
        beta = b.const((0.1 * b.rng.standard_normal(dim)).astype(np.float32), "ln_b")
        return b.op("LayerNormalization", [x, gamma, beta], axis=-1, epsilon=1e-5)

    def linear(x: str, n_in: int, n_out: int) -> str:
        return b.op(
            "Add",
            [b.op("MatMul", [x, b.weight(n_in, n_out)]), b.weight(n_out, scale=0.1, hint="b")],
        )

    def split_heads(x: str) -> str:
        shape = b.const(np.array([1, seq, heads, head], np.int64), "shape")
        return b.op("Transpose", [b.op("Reshape", [x, shape])], perm=[0, 2, 1, 3])

    def gelu_of(x: str) -> str:
        half = b.const(np.float32(0.5))
        one = b.const(np.float32(1.0))
        if gelu == "erf":
            inner = b.op("Erf", [b.op("Div", [x, b.const(np.float32(np.sqrt(2.0)))])])
        else:
            cube = b.op("Pow", [x, b.const(np.float32(3.0))])
            poly = b.op("Add", [x, b.op("Mul", [cube, b.const(np.float32(0.044715))])])
            inner = b.op("Tanh", [b.op("Mul", [poly, b.const(np.float32(np.sqrt(2 / np.pi)))])])
        return b.op("Mul", [b.op("Mul", [x, half]), b.op("Add", [inner, one])])

    n = layer_norm("x")
    q, k, v = (split_heads(linear(n, dim, dim)) for _ in range(3))
    scores = b.op("MatMul", [q, b.op("Transpose", [k], perm=[0, 1, 3, 2])])
    scores = b.op("Div", [scores, b.const(np.float32(np.sqrt(head)))])
    attn = b.op("MatMul", [b.op("Softmax", [scores], axis=-1), v])
    merged = b.op(
        "Reshape",
        [
            b.op("Transpose", [attn], perm=[0, 2, 1, 3]),
            b.const(np.array([1, seq, dim], np.int64), "shape"),
        ],
    )
    h = b.op("Add", ["x", linear(merged, dim, dim)])
    mlp_out = linear(gelu_of(linear(layer_norm(h), dim, 4 * dim)), 4 * dim, dim)
    y = b.op("Add", [h, mlp_out])
    return b.model([("x", (1, seq, dim))], [(y, "y")])


def views_model() -> onnx.ModelProto:
    """Outputs that are views of the input: no kernel writes them, so the driver copies."""
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["x", "shape"], ["y"]),
            helper.make_node("Identity", ["x"], ["z"]),
        ],
        "views",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in ("y", "z")],
        [numpy_helper.from_array(np.array([6, 4], np.int64), "shape")],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
    )


def random_inputs(model: onnx.ModelProto, seed: int = 0) -> dict[str, np.ndarray]:
    """Standard-normal float32 inputs of every graph input's static shape."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for value in model.graph.input:
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        out[value.name] = rng.standard_normal(shape).astype(np.float32)
    return out


def onnxruntime_outputs(
    model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, inputs), strict=True))
