"""One ONNX node at a time: import (and optimize) must agree with ONNX Runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx_models import Array, run_ort

from tessera.frontend import onnx as onnx_frontend
from tessera.frontend.onnx import OnnxImportError
from tessera.ir import interpret
from tessera.passes import optimize

RNG = np.random.default_rng(5)
I64 = np.int64


def f32(*shape: int) -> Array:
    return np.asarray(RNG.standard_normal(shape), np.float32)


def model_of(
    nodes: list[onnx.NodeProto],
    inputs: dict[str, Array],
    constants: dict[str, Array] | None = None,
    opset: int = 17,
    outputs: tuple[str, ...] = ("y",),
) -> onnx.ModelProto:
    def info(name: str, a: Array) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(a.dtype), a.shape
        )

    graph = helper.make_graph(
        nodes,
        "case",
        [info(k, v) for k, v in inputs.items()],
        [helper.make_empty_tensor_value_info(o) for o in outputs],
        initializer=[
            numpy_helper.from_array(np.asarray(v), k) for k, v in (constants or {}).items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def node(op: str, inputs: list[str], outputs: list[str] | None = None, **attrs: Any) -> Any:
    return helper.make_node(op, inputs, outputs or ["y"], **attrs)


CASES: dict[str, tuple[list[onnx.NodeProto], dict[str, Array], dict[str, Array], int]] = {
    "reciprocal": ([node("Reciprocal", ["x"])], {"x": f32(3, 4) + 4}, {}, 17),
    "softplus": ([node("Softplus", ["x"])], {"x": f32(3, 4)}, {}, 17),
    "leaky_relu": ([node("LeakyRelu", ["x"], alpha=0.1)], {"x": f32(3, 4)}, {}, 17),
    "clip_opset6": ([node("Clip", ["x"], min=-0.5, max=0.5)], {"x": f32(5)}, {}, 6),
    "clip_max_only": (
        [node("Clip", ["x", "", "hi"])],
        {"x": f32(6)},
        {"hi": np.array(0.2, np.float32)},
        13,
    ),
    "cast_like": (
        [node("CastLike", ["x", "like"])],
        {"x": f32(4) * 5},
        {"like": np.array([1], I64)},
        17,
    ),
    "pow_int_exponent": ([node("Pow", ["x", "e"])], {"x": f32(4)}, {"e": np.array(3, I64)}, 17),
    "less_or_equal": (
        [node("LessOrEqual", ["a", "b"])],
        {"a": np.array([1, 2, 3], I64), "b": np.array([2, 2, 2], I64)},
        {},
        17,
    ),
    "logic": (
        [
            node("Greater", ["a", "b"], ["g"]),
            node("Less", ["a", "b"], ["l"]),
            node("Or", ["g", "l"], ["o"]),
            node("Not", ["o"], ["n"]),
            node("Xor", ["n", "g"], ["xr"]),
            node("And", ["xr", "l"], ["y"]),
        ],
        {"a": np.array([1, 2, 3, 2], I64), "b": np.array([2, 2, 2, 1], I64)},
        {},
        17,
    ),
    "max_min_sum_mean": (
        [
            node("Max", ["a", "b", "c"], ["mx"]),
            node("Min", ["a", "b"], ["mn"]),
            node("Sum", ["mx", "mn", "c"], ["s"]),
            node("Mean", ["s", "a"], ["y"]),
        ],
        {"a": f32(2, 3), "b": f32(3), "c": f32(2, 1)},
        {},
        17,
    ),
    "matmul_vectors": (
        [node("MatMul", ["v", "m"], ["t"]), node("MatMul", ["t", "w"], ["y"])],
        {"v": f32(4), "m": f32(2, 4, 3), "w": f32(3)},
        {},
        17,
    ),
    "conv_same_lower": (
        [node("Conv", ["x", "w"], auto_pad="SAME_LOWER", strides=[2, 2])],
        {"x": f32(1, 2, 7, 6)},
        {"w": f32(3, 2, 4, 3)},
        17,
    ),
    "conv1d": (
        [node("Conv", ["x", "w", "b"], pads=[1, 2], strides=[2], dilations=[2])],
        {"x": f32(2, 3, 11)},
        {"w": f32(4, 3, 3), "b": f32(4)},
        17,
    ),
    "maxpool_ceil": (
        [node("MaxPool", ["x"], kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1)],
        {"x": f32(1, 2, 8, 8)},
        {},
        12,
    ),
    "maxpool_same_upper": (
        [node("MaxPool", ["x"], kernel_shape=[2, 2], strides=[2, 2], auto_pad="SAME_UPPER")],
        {"x": f32(1, 2, 7, 5)},
        {},
        12,
    ),
    "global_max_pool": ([node("GlobalMaxPool", ["x"])], {"x": f32(1, 3, 4, 5)}, {}, 17),
    "layernorm_no_bias": (
        [node("LayerNormalization", ["x", "s"], axis=1)],
        {"x": f32(2, 3, 4)},
        {"s": f32(3, 4)},
        17,
    ),
    "softmax_opset11_axis1": ([node("Softmax", ["x"], axis=1)], {"x": f32(2, 3, 4)}, {}, 11),
    "log_softmax": ([node("LogSoftmax", ["x"], axis=-1)], {"x": f32(2, 5)}, {}, 13),
    "reduce_min": ([node("ReduceMin", ["x"], axes=[1], keepdims=0)], {"x": f32(3, 4)}, {}, 13),
    "reduce_mean_opset13": ([node("ReduceMean", ["x"], axes=[-1])], {"x": f32(3, 4)}, {}, 13),
    "reduce_sum_no_axes": ([node("ReduceSum", ["x"], keepdims=0)], {"x": f32(3, 4)}, {}, 13),
    "reshape_allowzero": (
        [node("Reshape", ["x", "s"], allowzero=1)],
        {"x": f32(2, 6)},
        {"s": np.array([3, 4], I64)},
        14,
    ),
    "squeeze_opset11": ([node("Squeeze", ["x"], axes=[0])], {"x": f32(1, 3)}, {}, 11),
    "unsqueeze_opset11": ([node("Unsqueeze", ["x"], axes=[1])], {"x": f32(2, 3)}, {}, 11),
    "split_sizes": (
        [node("Split", ["x", "s"], ["y", "z"], axis=1)],
        {"x": f32(2, 7)},
        {"s": np.array([3, 4], I64)},
        13,
    ),
    "split_even_opset18": (
        [node("Split", ["x"], ["y", "z", "w"], axis=-1, num_outputs=3)],
        {"x": f32(2, 8)},
        {},
        18,
    ),
    "slice_negative_step": (
        [node("Slice", ["x", "s", "e", "a", "t"])],
        {"x": f32(4, 6)},
        {
            "s": np.array([-1, 1], I64),
            "e": np.array([-100, 5], I64),
            "a": np.array([1, 0], I64),
            "t": np.array([-2, 1], I64),
        },
        13,
    ),
    "slice_opset9": (
        [node("Slice", ["x"], starts=[1], ends=[3], axes=[1])],
        {"x": f32(2, 5)},
        {},
        9,
    ),
    "tile": ([node("Tile", ["x", "r"])], {"x": f32(2, 3)}, {"r": np.array([2, 3], I64)}, 13),
    "pad_negative": (
        [node("Pad", ["x", "p"], mode="constant")],
        {"x": f32(4, 5)},
        {"p": np.array([1, -1, -2, 2], I64)},
        13,
    ),
    "pad_axes_opset18": (
        [node("Pad", ["x", "p", "v", "a"], mode="constant")],
        {"x": f32(2, 3, 4)},
        {"p": np.array([1, 2], I64), "v": np.array(1.5, np.float32), "a": np.array([-1], I64)},
        18,
    ),
    "pad_opset2": (
        [node("Pad", ["x"], pads=[0, 1, 0, 2], value=3.0)],
        {"x": f32(2, 3)},
        {},
        2,
    ),
    "shape_range_constant_of_shape": (
        [
            node("Shape", ["x"], ["s"], start=1),
            node(
                "ConstantOfShape",
                ["s"],
                ["z"],
                value=numpy_helper.from_array(np.array([2.0], np.float32)),
            ),
            node("Range", ["start", "limit", "delta"], ["r"]),
            node("Cast", ["r"], ["rf"], to=TensorProto.FLOAT),
            node("Add", ["z", "rf"], ["zr"]),
            node("Mul", ["x", "zr"], ["y"]),
        ],
        {"x": f32(3, 4)},
        {
            "start": np.array(0, I64),
            "limit": np.array(4, I64),
            "delta": np.array(1, I64),
        },
        17,
    ),
    "trilu_mask": (
        [
            node("Trilu", ["ones"], ["tri"], upper=0),
            node("Cast", ["tri"], ["mask"], to=TensorProto.BOOL),
            node("Where", ["mask", "x", "neg"], ["y"]),
        ],
        {"x": f32(4, 4)},
        {"ones": np.ones((4, 4), np.float32), "neg": np.array(-1e9, np.float32)},
        17,
    ),
    "gather_elements_of_constants": (
        [
            node("GatherElements", ["table", "idx"], ["picked"], axis=1),
            node("Add", ["x", "picked"], ["y"]),
        ],
        {"x": f32(2, 3)},
        {"table": f32(2, 5), "idx": np.array([[0, 4, -1], [2, 2, 1]], I64)},
        13,
    ),
    "dropout_with_ratio": (
        [node("Dropout", ["x", "ratio", "train"])],
        {"x": f32(3)},
        {"ratio": np.array(0.1, np.float32), "train": np.array(False)},
        13,
    ),
    "batchnorm_rank3": (
        [node("BatchNormalization", ["x", "s", "b", "m", "v"])],
        {"x": f32(2, 3, 5)},
        {"s": f32(3), "b": f32(3), "m": f32(3), "v": np.abs(f32(3)) + 1},
        15,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_converter_matches_onnx_runtime(case: str) -> None:
    nodes, inputs, constants, opset = CASES[case]
    outputs = tuple(o for o in nodes[-1].output) if case.startswith("split") else ("y",)
    model = model_of(nodes, inputs, constants, opset, outputs)
    expected = run_ort(model, inputs)
    graph = onnx_frontend.load(model)
    for g in (graph, optimize(graph)):
        got = interpret(g, inputs)
        # By position: an output that optimizes to a graph input takes the input's name.
        for out, value in zip(g.outputs, expected.values(), strict=True):
            assert got[out.name].dtype == value.dtype
            np.testing.assert_allclose(got[out.name], value, rtol=1e-5, atol=1e-6)


def test_dynamic_dimensions_need_input_shapes() -> None:
    model = model_of([node("Relu", ["x"])], {"x": f32(2, 3)})
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    with pytest.raises(OnnxImportError, match="dynamic dimension 'batch'"):
        onnx_frontend.load(model)
    graph = onnx_frontend.load(model, {"x": (5, 3)})
    assert graph.inputs[0].shape == (5, 3)
    with pytest.raises(OnnxImportError, match="dimension 3 != 4"):
        onnx_frontend.load(model, {"x": (5, 4)})
    with pytest.raises(OnnxImportError, match="unknown inputs"):
        onnx_frontend.load(model, {"z": (1,)})


def test_unsupported_operator_names_the_node() -> None:
    model = model_of([helper.make_node("Det", ["x"], ["y"], name="/head/Det")], {"x": f32(3, 3)})
    with pytest.raises(OnnxImportError, match=r"Det node 'head_Det'.*no Tessera equivalent"):
        onnx_frontend.load(model)


def test_shapes_must_be_constant() -> None:
    model = model_of([node("Reshape", ["x", "s"])], {"x": f32(2, 3), "s": np.array([3, 2], I64)})
    with pytest.raises(OnnxImportError, match="must be a constant"):
        onnx_frontend.load(model)


def test_unsupported_second_outputs_are_rejected_when_read() -> None:
    model = model_of(
        [node("MaxPool", ["x"], ["y", "idx"], kernel_shape=[2, 2])],
        {"x": f32(1, 1, 4, 4)},
        opset=12,
        outputs=("y", "idx"),
    )
    with pytest.raises(OnnxImportError, match="Indices output"):
        onnx_frontend.load(model)


def test_load_from_file(tmp_path: Any) -> None:
    model = model_of([node("Relu", ["x"])], {"x": f32(2, 3)})
    path = tmp_path / "relu.onnx"
    onnx.save(model, str(path))
    graph = onnx_frontend.load(path)
    assert [n.op for n in graph.nodes] == ["Relu"]


NAMING_CASES: dict[str, tuple[list[onnx.NodeProto], dict[str, Array], tuple[str, ...]]] = {
    "identity_of_constant": ([node("Identity", ["c"], ["k"])], {"x": f32(3)}, ("k",)),
    "constant_itself": ([], {"x": f32(3)}, ("c",)),
    "identity_of_input": ([node("Identity", ["x"], ["y"])], {"x": f32(3)}, ("y",)),
    "input_itself": ([], {"x": f32(3)}, ("x",)),
    "dropped_chain": (
        [
            node("Dropout", ["x"], ["d"]),
            node("Reshape", ["d", "same"], ["r"]),
            node("Identity", ["r"], ["out"]),
        ],
        {"x": f32(3)},
        ("out",),
    ),
    "same_value_twice": (
        [node("Relu", ["x"], ["a"]), node("Identity", ["a"], ["b"])],
        {"x": f32(3)},
        ("a", "b"),
    ),
    "output_named_like_a_node": (
        [node("Relu", ["x"], ["relu"]), node("Neg", ["relu"], ["Relu"])],
        {"x": f32(3)},
        ("Relu", "relu"),
    ),
}


@pytest.mark.parametrize("case", sorted(NAMING_CASES))
def test_inputs_and_outputs_keep_their_onnx_names(case: str) -> None:
    nodes, inputs, outputs = NAMING_CASES[case]
    constants = {"c": np.arange(3, dtype=np.float32), "same": np.array([3], I64)}
    model = model_of(nodes, inputs, constants, 14, outputs)
    expected = run_ort(model, inputs)
    imported = onnx_frontend.load(model)
    for g in (imported, optimize(imported), optimize(imported, fold=False)):
        assert [v.name for v in g.inputs] == list(inputs)
        assert [v.name for v in g.outputs] == list(outputs)
        got = interpret(g, inputs)
        assert list(got) == list(outputs)
        for name in outputs:
            np.testing.assert_array_equal(got[name], expected[name])
