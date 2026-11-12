"""ONNX import: a ModelProto with static shapes to a Tessera ``Graph``.

Every ONNX node is translated by a converter into Tessera operators. A node
whose inputs are all constants is evaluated during import with the reference
implementations, which is how shape subgraphs (Shape, Gather, Concat,
Reshape on constants, ...) disappear: by the time a Reshape reads its target
shape, the shape is a constant. Operators that only make sense on constants
(Range, ConstantOfShape, Trilu, ...) are evaluated directly and are an error
on non-constant inputs. The graph keeps ONNX semantics and NCHW layout;
``passes.optimize`` canonicalizes it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tessera.dtype import DType
from tessera.ir.graph import Attr, Graph, NDArray, TensorType, Value
from tessera.ir.ops import get_op

_DTYPES: dict[int, DType] = {
    TensorProto.FLOAT: DType.float32,
    TensorProto.FLOAT16: DType.float16,
    TensorProto.DOUBLE: DType.float32,  # computed in float32
    TensorProto.INT64: DType.int64,
    TensorProto.INT32: DType.int32,
    TensorProto.INT8: DType.int8,
    TensorProto.UINT8: DType.uint8,
    TensorProto.BOOL: DType.bool_,
}


class OnnxImportError(ValueError):
    """The model uses something the importer cannot translate."""


def onnx_dtype(code: int, what: str) -> DType:
    try:
        return _DTYPES[code]
    except KeyError:
        name = TensorProto.DataType.Name(code)
        raise OnnxImportError(f"{what}: element type {name} is not supported") from None


def _array(proto: TensorProto) -> NDArray:
    data = numpy_helper.to_array(proto)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    DType.from_numpy(data.dtype)
    return np.asarray(data, order="C")


def _clean(name: str, fallback: str) -> str:
    """A readable identifier from an ONNX name ("/encoder/layer.0/MatMul")."""
    text = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return text or fallback


@dataclass
class OnnxNode:
    """One ONNX node during import: its inputs (None when omitted) and attributes."""

    op: str
    name: str
    inputs: list[Value | None]
    attrs: dict[str, Any]
    outputs: list[str]
    importer: Importer = field(repr=False)

    def __str__(self) -> str:
        return f"{self.op} node {self.name!r}"

    def error(self, message: str) -> OnnxImportError:
        return OnnxImportError(f"{self}: {message}")

    def input(self, i: int) -> Value:
        value = self.inputs[i] if i < len(self.inputs) else None
        if value is None:
            raise self.error(f"input {i} is required")
        return value

    def optional(self, i: int) -> Value | None:
        return self.inputs[i] if i < len(self.inputs) else None

    def constant(self, i: int, what: str) -> NDArray:
        """The data of input ``i``, which must be a constant (shapes are static)."""
        value = self.input(i)
        if value.data is None:
            raise self.error(f"{what} (input {i}) must be a constant; Tessera needs static shapes")
        return value.data

    def ints(self, i: int, what: str) -> tuple[int, ...]:
        return tuple(int(v) for v in self.constant(i, what).reshape(-1))

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    def op_(self, op: str, *inputs: Value, **attrs: Attr) -> Value:
        """Build a Tessera node named after this ONNX node (folded if its inputs are constant)."""
        return self.importer.op(op, inputs, attrs, self.name)

    def const(self, data: NDArray | float | int, like: Value | None = None) -> Value:
        array = np.asarray(data, like.dtype.numpy if like is not None else None)
        return self.importer.graph.add_constant(array, f"{self.name}_const")


type Converter = Callable[[OnnxNode], Sequence[Value | None]]
_CONVERTERS: dict[str, Converter] = {}


def converter(*ops: str) -> Callable[[Converter], Converter]:
    def register(fn: Converter) -> Converter:
        for op in ops:
            _CONVERTERS[op] = fn
        return fn

    return register


class Importer:
    """Translates one ONNX graph."""

    def __init__(self, model: onnx.ModelProto, input_shapes: Mapping[str, Sequence[int]]) -> None:
        self.model = model
        self.opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 1)
        name = _clean(model.graph.name, "model")
        self.graph = Graph(name if name[0].isalpha() else f"m_{name}")
        self.env: dict[str, Value | None] = {}
        self.input_shapes = dict(input_shapes)

    def op(self, op: str, inputs: Sequence[Value], attrs: Mapping[str, Attr], name: str) -> Value:
        g = self.graph
        if inputs and all(v.data is not None for v in inputs):
            opdef = get_op(op)
            types = [v.type for v in inputs]
            normalized = opdef.normalize(types, attrs) if opdef.normalize else dict(attrs)
            (out_type,) = opdef.infer(types, normalized)
            args = [v.data for v in inputs if v.data is not None]
            with np.errstate(all="ignore"):
                (result,) = opdef.reference(args, normalized)
            data = np.asarray(np.asarray(result).astype(out_type.dtype.numpy), order="C")
            return g.add_constant(data, name)
        return g.op(op, *inputs, name=name, **attrs)

    def run(self) -> Graph:
        g, graph = self.graph, self.model.graph
        initializers = {t.name for t in graph.initializer}
        for tensor in graph.initializer:
            self.env[tensor.name] = g.add_constant(_array(tensor), _clean(tensor.name, "const"))
        unknown = set(self.input_shapes) - {i.name for i in graph.input}
        if unknown:
            raise OnnxImportError(f"input_shapes names unknown inputs {sorted(unknown)}")
        for proto in graph.input:
            if proto.name not in initializers:
                self.env[proto.name] = g.add_input(_clean(proto.name, "input"), self._type(proto))
        output_names = [_clean(o.name, "output") for o in graph.output]
        for name in output_names:
            if name not in {v.name for v in g.inputs}:
                g.claim_name(name)

        for proto in graph.node:
            self._convert(proto)

        outputs: list[Value] = []
        for proto, name in zip(graph.output, output_names, strict=True):
            value = self.env.get(proto.name)
            if value is None:
                raise OnnxImportError(f"graph output {proto.name!r} is not computed")
            if value.producer is not None and value.name != name and value not in outputs:
                value.name = name
                if len(value.producer.outputs) == 1:
                    value.producer.name = name
            outputs.append(value)
        g.set_outputs(outputs)
        g.verify()
        return g

    def _type(self, proto: onnx.ValueInfoProto) -> TensorType:
        tensor = proto.type.tensor_type
        dtype = onnx_dtype(tensor.elem_type, f"input {proto.name!r}")
        if proto.name in self.input_shapes:
            shape = tuple(int(d) for d in self.input_shapes[proto.name])
            if len(shape) != len(tensor.shape.dim):
                raise OnnxImportError(
                    f"input {proto.name!r} has rank {len(tensor.shape.dim)}, "
                    f"input_shapes gives {shape}"
                )
            for d, given in zip(tensor.shape.dim, shape, strict=True):
                if d.HasField("dim_value") and d.dim_value != given:
                    raise OnnxImportError(
                        f"input {proto.name!r}: dimension {d.dim_value} != {given}"
                    )
            return TensorType(shape, dtype)
        dims = []
        for d in tensor.shape.dim:
            if not d.HasField("dim_value"):
                label = d.dim_param or "?"
                raise OnnxImportError(
                    f"input {proto.name!r} has a dynamic dimension {label!r}; "
                    "give its shape in input_shapes"
                )
            dims.append(int(d.dim_value))
        return TensorType(tuple(dims), dtype)

    def _convert(self, proto: onnx.NodeProto) -> None:
        name = _clean(proto.name, proto.op_type.lower())
        if proto.domain not in ("", "ai.onnx"):
            raise OnnxImportError(
                f"{proto.op_type} node {name!r}: domain {proto.domain!r} is not supported"
            )
        inputs: list[Value | None] = []
        for i, ref in enumerate(proto.input):
            if not ref:
                inputs.append(None)
                continue
            if ref not in self.env:
                raise OnnxImportError(
                    f"{proto.op_type} node {name!r} reads undefined value {ref!r}"
                )
            value = self.env[ref]
            if value is None:
                raise OnnxImportError(
                    f"{proto.op_type} node {name!r} reads {ref!r} (input {i}), an output "
                    "the importer does not compute"
                )
            inputs.append(value)
        attrs = {a.name: _attribute(a) for a in proto.attribute}
        node = OnnxNode(proto.op_type, name, inputs, attrs, list(proto.output), self)
        fn = _CONVERTERS.get(proto.op_type)
        if fn is None:
            raise node.error("this ONNX operator has no Tessera equivalent")
        results = list(fn(node))
        for i, out in enumerate(proto.output):
            if out:
                self.env[out] = results[i] if i < len(results) else None


def _attribute(attr: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, TensorProto):
        return _array(value)
    if isinstance(value, list):
        return [v.decode() if isinstance(v, bytes) else v for v in value]
    return value


def load(
    model: str | Path | onnx.ModelProto,
    input_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> Graph:
    """Import an ONNX model; ``input_shapes`` fixes dynamic input dimensions."""
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    return Importer(model, input_shapes or {}).run()


# == converters ===================================================================


@converter("Constant")
def _constant(n: OnnxNode) -> list[Value]:
    for key in ("value", "value_float", "value_floats", "value_int", "value_ints"):
        if key in n.attrs:
            data = np.asarray(n.attrs[key])
            if key.startswith("value_int"):
                data = data.astype(np.int64)
            elif key.startswith("value_float"):
                data = data.astype(np.float32)
            return [n.importer.graph.add_constant(data, n.name)]
    raise n.error(f"unsupported attributes {sorted(n.attrs)}")


@converter("Identity", "Dropout")
def _identity(n: OnnxNode) -> list[Value]:
    if n.op == "Dropout" and n.optional(2) is not None:
        training = n.optional(2)
        if training is None or training.data is None or bool(training.data):
            raise n.error("only inference (training_mode false) is supported")
    x = n.input(0)
    return [x if x.data is not None else n.op_(n.op, x)]


_UNARY = {
    "Relu": "Relu",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Exp": "Exp",
    "Log": "Log",
    "Sqrt": "Sqrt",
    "Erf": "Erf",
    "Neg": "Neg",
    "Abs": "Abs",
    "HardSwish": "HardSwish",
}


@converter(*_UNARY)
def _unary(n: OnnxNode) -> list[Value]:
    return [n.op_(_UNARY[n.op], n.input(0))]


@converter("HardSigmoid")
def _hard_sigmoid(n: OnnxNode) -> list[Value]:
    return [n.op_("HardSigmoid", n.input(0), alpha=n.attr("alpha", 0.2), beta=n.attr("beta", 0.5))]


@converter("Gelu")
def _gelu(n: OnnxNode) -> list[Value]:
    return [n.op_("Gelu", n.input(0), approximate=n.attr("approximate", "none"))]


@converter("Reciprocal")
def _reciprocal(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    return [n.op_("Div", n.const(1, like=x), x)]


@converter("Softplus")
def _softplus(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    return [n.op_("Log", n.op_("Add", n.op_("Exp", x), n.const(1, like=x)))]


@converter("LeakyRelu")
def _leaky_relu(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    alpha = float(n.attr("alpha", 0.01))
    zero = n.const(0, like=x)
    return [n.op_("Where", n.op_("Less", x, zero), n.op_("Mul", x, n.const(alpha, like=x)), x)]


@converter("Clip")
def _clip(n: OnnxNode) -> list[Value]:
    if n.importer.opset < 11:
        lo, hi = n.attr("min", -math.inf), n.attr("max", math.inf)
    else:
        lo = float(n.constant(1, "min")) if n.optional(1) is not None else -math.inf
        hi = float(n.constant(2, "max")) if n.optional(2) is not None else math.inf
    return [n.op_("Clip", n.input(0), min=float(lo), max=float(hi))]


@converter("Cast")
def _cast(n: OnnxNode) -> list[Value]:
    to = onnx_dtype(int(n.attr("to")), str(n))
    x = n.input(0)
    return [x if x.dtype is to else n.op_("Cast", x, to=to.value)]


@converter("CastLike")
def _cast_like(n: OnnxNode) -> list[Value]:
    x, like = n.input(0), n.input(1)
    return [x if x.dtype is like.dtype else n.op_("Cast", x, to=like.dtype.value)]


_BINARY = {
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "Equal": "Equal",
    "Less": "Less",
    "Greater": "Greater",
}


@converter(*_BINARY)
def _binary(n: OnnxNode) -> list[Value]:
    return [n.op_(_BINARY[n.op], n.input(0), n.input(1))]


@converter("Pow")
def _pow(n: OnnxNode) -> list[Value]:
    base, exponent = n.input(0), n.input(1)
    if exponent.dtype is not base.dtype:
        exponent = n.op_("Cast", exponent, to=base.dtype.value)
    return [n.op_("Pow", base, exponent)]


def _not(n: OnnxNode, x: Value) -> Value:
    return n.op_("Equal", x, n.const(False))


@converter("LessOrEqual", "GreaterOrEqual")
def _compare_or_equal(n: OnnxNode) -> list[Value]:
    a, b = n.input(0), n.input(1)
    strict = n.op_("Greater" if n.op == "LessOrEqual" else "Less", a, b)
    return [_not(n, strict)]


@converter("Not", "And", "Or", "Xor")
def _logic(n: OnnxNode) -> list[Value]:
    a = n.input(0)
    if n.op == "Not":
        return [_not(n, a)]
    b = n.input(1)
    if n.op == "And":
        return [n.op_("Where", a, b, n.const(False))]
    if n.op == "Or":
        return [n.op_("Where", a, n.const(True), b)]
    return [_not(n, n.op_("Equal", a, b))]


@converter("Max", "Min", "Sum", "Mean")
def _variadic(n: OnnxNode) -> list[Value]:
    op = {"Max": "Maximum", "Min": "Minimum", "Sum": "Add", "Mean": "Add"}[n.op]
    values = [v for v in n.inputs if v is not None]
    out = values[0]
    for v in values[1:]:
        out = n.op_(op, out, v)
    if n.op == "Mean" and len(values) > 1:
        out = n.op_("Div", out, n.const(len(values), like=out))
    return [out]


@converter("Where")
def _where(n: OnnxNode) -> list[Value]:
    return [n.op_("Where", n.input(0), n.input(1), n.input(2))]


@converter("MatMul")
def _matmul(n: OnnxNode) -> list[Value]:
    """NumPy matmul; a vector operand becomes a matrix and its unit axis is dropped after."""
    a, b = n.input(0), n.input(1)
    a_vector, b_vector = a.type.rank == 1, b.type.rank == 1
    if a_vector:
        a = n.op_("Reshape", a, shape=(1, a.shape[0]))
    if b_vector:
        b = n.op_("Reshape", b, shape=(b.shape[0], 1))
    y = n.op_("MatMul", a, b)
    if a_vector or b_vector:
        shape = list(y.shape)
        if b_vector:
            shape.pop(-1)
        if a_vector:
            shape.pop(-1 if b_vector else -2)
        y = n.op_("Reshape", y, shape=tuple(shape))
    return [y]


@converter("Gemm")
def _gemm(n: OnnxNode) -> list[Value]:
    args = [v for v in (n.input(0), n.input(1), n.optional(2)) if v is not None]
    return [
        n.op_(
            "Gemm",
            *args,
            alpha=float(n.attr("alpha", 1.0)),
            beta=float(n.attr("beta", 1.0)),
            transA=int(n.attr("transA", 0)),
            transB=int(n.attr("transB", 0)),
        )
    ]


def _spatial_pads(n: OnnxNode, size: Sequence[int], kernel: Sequence[int]) -> tuple[int, ...]:
    """Explicit (top, left, bottom, right) pads, resolving auto_pad."""
    auto = n.attr("auto_pad", "NOTSET")
    strides = list(n.attr("strides", [1] * len(kernel)))
    dilations = list(n.attr("dilations", [1] * len(kernel)))
    if auto in ("NOTSET", ""):
        pads = list(n.attr("pads", [0] * (2 * len(kernel))))
        return tuple(int(p) for p in pads)
    if auto == "VALID":
        return (0,) * (2 * len(kernel))
    begins, ends = [], []
    for d, k, s, dil in zip(size, kernel, strides, dilations, strict=True):
        out = -(-d // s)
        total = max(0, (out - 1) * s + (k - 1) * dil + 1 - d)
        small, large = total // 2, total - total // 2
        begins.append(large if auto == "SAME_LOWER" else small)
        ends.append(small if auto == "SAME_LOWER" else large)
    return (*begins, *ends)


@converter("Conv")
def _conv(n: OnnxNode) -> list[Value]:
    x, w, bias = n.input(0), n.input(1), n.optional(2)
    if x.type.rank == 3:
        return [_conv1d(n, x, w, bias)]
    if x.type.rank != 4:
        raise n.error(f"only 1-D and 2-D convolutions are supported, input is {x.type}")
    kernel = w.shape[2:]
    pads = _spatial_pads(n, x.shape[2:], kernel)
    args = [x, w] if bias is None else [x, w, bias]
    return [
        n.op_(
            "Conv2d",
            *args,
            strides=tuple(n.attr("strides", [1, 1])),
            pads=pads,
            dilations=tuple(n.attr("dilations", [1, 1])),
            groups=int(n.attr("group", 1)),
        )
    ]


def _conv1d(n: OnnxNode, x: Value, w: Value, bias: Value | None) -> Value:
    """A 1-D convolution as a 2-D one over a height of 1."""
    pads = _spatial_pads(n, x.shape[2:], w.shape[2:])
    x2 = n.op_("Reshape", x, shape=(x.shape[0], x.shape[1], 1, x.shape[2]))
    w2 = n.op_("Reshape", w, shape=(w.shape[0], w.shape[1], 1, w.shape[2]))
    args = [x2, w2] if bias is None else [x2, w2, bias]
    y = n.op_(
        "Conv2d",
        *args,
        strides=(1, int(n.attr("strides", [1])[0])),
        pads=(0, pads[0], 0, pads[1]),
        dilations=(1, int(n.attr("dilations", [1])[0])),
        groups=int(n.attr("group", 1)),
    )
    return n.op_("Reshape", y, shape=(y.shape[0], y.shape[1], y.shape[3]))


@converter("MaxPool", "AveragePool")
def _pool(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    if x.type.rank != 4:
        raise n.error(f"only 2-D pooling is supported, input is {x.type}")
    if any(d != 1 for d in n.attr("dilations", [1, 1])):
        raise n.error("dilated pooling is not supported")
    kernel = tuple(int(k) for k in n.attr("kernel_shape"))
    strides = tuple(int(s) for s in n.attr("strides", [1, 1]))
    pads = list(_spatial_pads(n, x.shape[2:], kernel))
    if n.attr("ceil_mode", 0):
        for i in range(2):
            size = x.shape[2 + i] + pads[i] + pads[i + 2]
            floor_out = (size - kernel[i]) // strides[i] + 1
            ceil_out = -(-(size - kernel[i]) // strides[i]) + 1
            # The last window must start inside the input or the leading padding.
            if (ceil_out - 1) * strides[i] >= x.shape[2 + i] + pads[i]:
                ceil_out -= 1
            if ceil_out > floor_out:
                if n.op == "AveragePool":
                    raise n.error("ceil_mode with a partial last window is not supported")
                pads[i + 2] += (ceil_out - 1) * strides[i] + kernel[i] - size
    if n.op == "MaxPool":
        if len(n.outputs) > 1 and n.outputs[1]:
            raise n.error("the Indices output is not supported")
        return [n.op_("MaxPool2d", x, kernel=kernel, strides=strides, pads=tuple(pads))]
    return [
        n.op_(
            "AvgPool2d",
            x,
            kernel=kernel,
            strides=strides,
            pads=tuple(pads),
            count_include_pad=int(n.attr("count_include_pad", 0)),
        )
    ]


@converter("GlobalAveragePool", "GlobalMaxPool")
def _global_pool(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    op = "ReduceMean" if n.op == "GlobalAveragePool" else "ReduceMax"
    return [n.op_(op, x, axes=tuple(range(2, x.type.rank)), keepdims=1)]


@converter("BatchNormalization")
def _batchnorm(n: OnnxNode) -> list[Value]:
    if n.attr("training_mode", 0) or any(n.outputs[1:]):
        raise n.error("only inference batch normalization is supported")
    args = [n.input(i) for i in range(5)]
    return [n.op_("BatchNorm", *args, epsilon=float(n.attr("epsilon", 1e-5)))]


@converter("LayerNormalization")
def _layernorm(n: OnnxNode) -> list[Value]:
    if any(n.outputs[1:]):
        raise n.error("the Mean and InvStdDev outputs are not supported")
    x = n.input(0)
    axis = int(n.attr("axis", -1))
    axis = axis + x.type.rank if axis < 0 else axis
    normalized = x.shape[axis:]
    scale = _broadcast_to(n, n.input(1), normalized)
    bias_in = n.optional(2)
    bias = (
        _broadcast_to(n, bias_in, normalized)
        if bias_in is not None
        else n.const(np.zeros(normalized), like=x)
    )
    return [n.op_("LayerNorm", x, scale, bias, axis=axis, epsilon=float(n.attr("epsilon", 1e-5)))]


def _broadcast_to(n: OnnxNode, value: Value, shape: tuple[int, ...]) -> Value:
    if value.shape == shape:
        return value
    out = n.op_("Expand", value, shape=shape)
    if out.shape != shape:
        raise n.error(f"{value.type} does not broadcast to {shape}")
    return out


@converter("Softmax", "LogSoftmax")
def _softmax(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    rank = x.type.rank
    if n.importer.opset >= 13:
        y = n.op_("Softmax", x, axis=int(n.attr("axis", -1)))
    else:
        # Before opset 13, Softmax normalizes the 2-D view [prod(:axis), prod(axis:)].
        axis = int(n.attr("axis", 1))
        axis = axis + rank if axis < 0 else axis
        if axis == rank - 1:
            y = n.op_("Softmax", x, axis=axis)
        else:
            rows = int(np.prod(x.shape[:axis]))
            flat = n.op_("Reshape", x, shape=(rows, -1))
            y = n.op_("Reshape", n.op_("Softmax", flat, axis=1), shape=x.shape)
    return [n.op_("Log", y) if n.op == "LogSoftmax" else y]


@converter("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin")
def _reduce(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    axes_from_input = n.op == "ReduceSum" and n.importer.opset >= 13
    axes_from_input |= n.op != "ReduceSum" and n.importer.opset >= 18
    if axes_from_input:
        axes = n.ints(1, "axes") if n.optional(1) is not None else None
    else:
        axes = tuple(n.attr("axes")) if "axes" in n.attrs else None
    if not axes:
        if n.attr("noop_with_empty_axes", 0):
            return [x]
        axes = tuple(range(x.type.rank))
    keepdims = int(n.attr("keepdims", 1))
    if n.op == "ReduceMin":
        return [n.op_("Neg", n.op_("ReduceMax", n.op_("Neg", x), axes=axes, keepdims=keepdims))]
    return [n.op_(n.op, x, axes=axes, keepdims=keepdims)]


@converter("Reshape")
def _reshape(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    shape = list(n.ints(1, "shape") if n.importer.opset >= 5 else n.attr("shape"))
    if n.attr("allowzero", 0) and 0 in shape:
        if -1 in shape:
            raise n.error("allowzero with both 0 and -1 is invalid")
        return [_literal_reshape(n, x, shape)]
    return [n.op_("Reshape", x, shape=tuple(shape))]


def _literal_reshape(n: OnnxNode, x: Value, shape: list[int]) -> Value:
    if int(np.prod(shape)) != x.type.numel:
        raise n.error(f"cannot reshape {x.type} to {tuple(shape)}")
    if x.type.numel == 0:
        return n.const(np.zeros(shape, x.dtype.numpy))
    return n.op_("Reshape", x, shape=tuple(shape))


@converter("Flatten")
def _flatten(n: OnnxNode) -> list[Value]:
    return [n.op_("Flatten", n.input(0), axis=int(n.attr("axis", 1)))]


def _axes(n: OnnxNode) -> tuple[int, ...] | None:
    if n.importer.opset >= 13:
        return n.ints(1, "axes") if n.optional(1) is not None else None
    return tuple(n.attr("axes")) if "axes" in n.attrs else None


@converter("Squeeze")
def _squeeze(n: OnnxNode) -> list[Value]:
    axes = _axes(n)
    x = n.input(0)
    return [n.op_("Squeeze", x) if axes is None else n.op_("Squeeze", x, axes=axes)]


@converter("Unsqueeze")
def _unsqueeze(n: OnnxNode) -> list[Value]:
    axes = _axes(n)
    if axes is None:
        raise n.error("axes are required")
    return [n.op_("Unsqueeze", n.input(0), axes=axes)]


@converter("Transpose")
def _transpose(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    perm = n.attr("perm")
    return [n.op_("Transpose", x) if perm is None else n.op_("Transpose", x, perm=tuple(perm))]


@converter("Concat")
def _concat(n: OnnxNode) -> list[Value]:
    values = [v for v in n.inputs if v is not None and v.type.numel > 0] or [n.input(0)]
    return [n.op_("Concat", *values, axis=int(n.attr("axis")))]


@converter("Split")
def _split(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    axis = int(n.attr("axis", 0))
    axis = axis + x.type.rank if axis < 0 else axis
    dim = x.shape[axis]
    count = len(n.outputs)
    if n.importer.opset >= 13 and n.optional(1) is not None:
        sizes = list(n.ints(1, "split"))
    elif n.importer.opset < 13 and "split" in n.attrs:
        sizes = list(n.attr("split"))
    else:
        count = int(n.attr("num_outputs", count))
        step = -(-dim // count)
        sizes = [min(step, dim - i * step) for i in range(count)]
    if sum(sizes) != dim:
        raise n.error(f"split sizes {sizes} do not add up to {dim}")
    outs, start = [], 0
    for size in sizes:
        outs.append(n.op_("Slice", x, starts=(start,), ends=(start + size,), axes=(axis,)))
        start += size
    return outs


@converter("Slice")
def _slice(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    if n.importer.opset >= 10:
        starts, ends = n.ints(1, "starts"), n.ints(2, "ends")
        axes = n.ints(3, "axes") if n.optional(3) is not None else tuple(range(len(starts)))
        steps = n.ints(4, "steps") if n.optional(4) is not None else (1,) * len(starts)
    else:
        starts, ends = tuple(n.attr("starts")), tuple(n.attr("ends"))
        axes = tuple(n.attr("axes", range(len(starts))))
        steps = (1,) * len(starts)
    forward = [(a, s, e, t) for a, s, e, t in zip(axes, starts, ends, steps, strict=True) if t > 0]
    backward = [(a, s, e, t) for a, s, e, t in zip(axes, starts, ends, steps, strict=True) if t < 0]
    y = x
    if forward:
        a, s, e, t = zip(*forward, strict=True)
        y = n.op_("Slice", x, starts=s, ends=e, axes=a, steps=t)
    for axis, start, end, step in backward:
        # Negative steps: a Gather of the selected indices (in descending order).
        axis = axis + x.type.rank if axis < 0 else axis
        indices = np.arange(y.shape[axis], dtype=np.int64)[slice(start, end, step)]
        y = n.op_("Gather", y, n.const(indices), axis=axis)
    return [y]


@converter("Gather")
def _gather(n: OnnxNode) -> list[Value]:
    return [n.op_("Gather", n.input(0), n.input(1), axis=int(n.attr("axis", 0)))]


@converter("Expand")
def _expand(n: OnnxNode) -> list[Value]:
    return [n.op_("Expand", n.input(0), shape=n.ints(1, "shape"))]


@converter("Tile")
def _tile(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    repeats = n.ints(1, "repeats")
    if x.data is not None:
        return [n.const(np.tile(x.data, repeats))]
    # x[d0, d1] tiled (r0, r1) is x viewed as [1, d0, 1, d1], expanded to [r0, d0, r1, d1].
    spread = tuple(v for d in x.shape for v in (1, d))
    target = tuple(v for r, d in zip(repeats, x.shape, strict=True) for v in (r, d))
    y = n.op_("Expand", n.op_("Reshape", x, shape=spread), shape=target)
    return [n.op_("Reshape", y, shape=tuple(r * d for r, d in zip(repeats, x.shape, strict=True)))]


@converter("Pad")
def _pad(n: OnnxNode) -> list[Value]:
    x = n.input(0)
    if n.attr("mode", "constant") != "constant":
        raise n.error(f"only constant padding is supported, not {n.attr('mode')!r}")
    rank = x.type.rank
    if n.importer.opset >= 11:
        given = n.ints(1, "pads")
        value_in = n.optional(2)
        value = float(n.constant(2, "constant_value")) if value_in is not None else 0.0
        axes = n.ints(3, "axes") if n.optional(3) is not None else tuple(range(rank))
    else:
        given = tuple(n.attr("pads"))
        value = float(n.attr("value", 0.0))
        axes = tuple(range(rank))
    pads = [0] * (2 * rank)
    for i, axis in enumerate(axes):
        axis = axis + rank if axis < 0 else axis
        pads[axis], pads[axis + rank] = given[i], given[i + len(axes)]
    if any(p < 0 for p in pads):
        # Negative pads crop: a Slice, then the remaining positive padding.
        starts = tuple(max(0, -p) for p in pads[:rank])
        ends = tuple(d - max(0, -p) for d, p in zip(x.shape, pads[rank:], strict=True))
        x = n.op_("Slice", x, starts=starts, ends=ends)
        pads = [max(0, p) for p in pads]
    return [n.op_("Pad", x, pads=tuple(pads), value=value)]


# -- operators evaluated at import time ---------------------------------------------


@converter("Shape")
def _shape(n: OnnxNode) -> list[Value]:
    shape = n.input(0).shape
    start, end = int(n.attr("start", 0)), n.attr("end")
    return [n.const(np.asarray(shape[start : None if end is None else int(end)], np.int64))]


@converter("Size")
def _size(n: OnnxNode) -> list[Value]:
    return [n.const(np.asarray(n.input(0).type.numel, np.int64))]


@converter("ConstantOfShape")
def _constant_of_shape(n: OnnxNode) -> list[Value]:
    shape = n.ints(0, "shape")
    fill = n.attr("value")
    fill = np.zeros(1, np.float32) if fill is None else np.asarray(fill).reshape(-1)
    return [n.const(np.full(shape, fill[0], fill.dtype))]


@converter("Range")
def _range(n: OnnxNode) -> list[Value]:
    start, limit, delta = (n.constant(i, w) for i, w in enumerate(("start", "limit", "delta")))
    count = max(0, math.ceil((limit.item() - start.item()) / delta.item()))
    return [n.const((start + delta * np.arange(count)).astype(start.dtype))]


@converter("Trilu")
def _trilu(n: OnnxNode) -> list[Value]:
    x = n.constant(0, "input")
    k = int(n.constant(1, "k")) if n.optional(1) is not None else 0
    fn = np.triu if n.attr("upper", 1) else np.tril
    return [n.const(fn(x, k))]


def _constant_only(fn: Callable[..., NDArray]) -> Converter:
    def convert(n: OnnxNode) -> list[Value]:
        args = [
            n.constant(i, f"input {i}") for i in range(len(n.inputs)) if n.inputs[i] is not None
        ]
        return [n.const(np.asarray(fn(*args)))]

    return convert


_CONSTANT_ONLY: dict[str, Callable[..., Any]] = {
    "Floor": np.floor,
    "Ceil": np.ceil,
    "Round": np.round,
    "Mod": np.fmod,
    "NonZero": lambda x: np.asarray(np.nonzero(x), np.int64),
    "IsNaN": np.isnan,
    "IsInf": np.isinf,
    "Sign": np.sign,
}
for _op, _fn in _CONSTANT_ONLY.items():
    _CONVERTERS[_op] = _constant_only(_fn)


@converter("GatherElements")
def _gather_elements(n: OnnxNode) -> list[Value]:
    data, indices = n.constant(0, "data"), n.constant(1, "indices")
    axis = int(n.attr("axis", 0))
    indices = np.where(indices < 0, indices + data.shape[axis], indices)
    return [n.const(np.take_along_axis(data, indices, axis=axis))]


@converter("CumSum")
def _cumsum(n: OnnxNode) -> list[Value]:
    x = n.constant(0, "input")
    axis = int(n.constant(1, "axis"))
    if n.attr("exclusive", 0) or n.attr("reverse", 0):
        raise n.error("exclusive and reverse cumulative sums are not supported")
    return [n.const(np.cumsum(x, axis=axis).astype(x.dtype))]


SUPPORTED_OPS: frozenset[str] = frozenset(_CONVERTERS)
