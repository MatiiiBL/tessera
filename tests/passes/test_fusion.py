"""Fusion: group structure, heuristics, invariants and preserved semantics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from graph_strategies import graphs, run_groups
from hypothesis import HealthCheck, given, settings

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, Value, interpret, random_inputs
from tessera.passes import FusionOptions, fuse, verify_groups
from tessera.plan import FusedGroup

F32 = DType.float32
RNG = np.random.default_rng(0)


def const(g: Graph, *shape: int, name: str = "c") -> Value:
    return g.add_constant(RNG.standard_normal(shape).astype(np.float32), name)


def summary(groups: list[FusedGroup]) -> list[tuple[str, list[str]]]:
    return [(g.pattern, [n.op for n in g.nodes]) for g in groups]


def check(g: Graph, **kwargs: object) -> list[FusedGroup]:
    groups = fuse(g, **kwargs)  # type: ignore[arg-type]
    verify_groups(g, groups)
    feeds = random_inputs(g)
    expected = interpret(g, feeds)
    got = run_groups(g, groups, feeds)
    for name, value in expected.items():
        np.testing.assert_allclose(got[name], value, rtol=1e-5, atol=1e-6)
    return groups


def conv_block() -> Graph:
    g = Graph("block")
    x = g.add_input("x", TensorType((1, 6, 6, 3), F32))
    padded = g.op("Pad", x, pads=(0, 1, 1, 0, 0, 1, 1, 0))
    conv = g.op("Conv2d", padded, const(g, 3, 3, 3, 4), layout="NHWC", name="conv")
    act = g.op("Relu", g.op("Add", conv, const(g, 4)))
    g.set_outputs([g.op("MaxPool2d", act, kernel=2, strides=2, layout="NHWC")])
    return g


def test_conv_takes_pad_prologue_and_bias_relu_epilogue() -> None:
    groups = check(conv_block())
    assert summary(groups) == [
        ("conv2d", ["Pad", "Conv2d", "Add", "Relu"]),
        ("pool", ["MaxPool2d"]),
    ]
    assert groups[0].anchor is not None
    assert groups[0].anchor.op == "Conv2d"
    assert groups[0].name == "conv"


def test_fusion_off_gives_one_node_per_group() -> None:
    g = conv_block()
    groups = check(g, enabled=False)
    assert [len(gr.nodes) for gr in groups] == [1] * len(g.nodes)
    assert [gr.pattern for gr in groups] == [
        "injective",
        "conv2d",
        "injective",
        "injective",
        "pool",
    ]


def attention() -> Graph:
    """Scores of one head: softmax(Q Kᵀ * scale + mask) V, as an exporter emits it."""
    g = Graph("attention")
    q = g.add_input("q", TensorType((1, 2, 8, 4), F32))
    k = g.add_input("k", TensorType((1, 2, 8, 4), F32))
    v = g.add_input("v", TensorType((1, 2, 8, 4), F32))
    mask = g.add_input("mask", TensorType((1, 1, 1, 8), F32))
    kt = g.op("Transpose", k, perm=(0, 1, 3, 2), name="k_t")
    scores = g.op("MatMul", q, kt, name="scores")
    scaled = g.op("Add", g.op("Mul", scores, g.add_constant(np.float32(0.5), "scale")), mask)
    probs = g.op("Softmax", scaled, axis=-1, name="probs")
    g.set_outputs([g.op("MatMul", probs, v, name="context")])
    return g


def test_attention_fuses_transpose_prologue_and_scale_mask_epilogue() -> None:
    assert summary(check(attention())) == [
        ("matmul", ["Transpose", "MatMul", "Mul", "Add"]),
        ("softmax", ["Softmax"]),
        ("matmul", ["MatMul"]),
    ]


def test_epilogue_stops_at_a_value_with_two_readers() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 8), F32))
    h = g.op("Add", g.op("MatMul", x, const(g, 8, 8)), const(g, 8), name="h")
    g.set_outputs([g.op("Add", g.op("MatMul", h, const(g, 8, 8)), h)])
    assert summary(check(g)) == [("matmul", ["MatMul", "Add"]), ("matmul", ["MatMul", "Add"])]


def test_epilogue_never_grows_the_anchor_shape() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 8), F32))
    big = g.add_input("big", TensorType((3, 4, 8), F32))
    g.set_outputs([g.op("Add", g.op("MatMul", x, const(g, 8, 8)), big)])
    assert summary(check(g)) == [("matmul", ["MatMul"]), ("injective", ["Add"])]


def test_cheap_producer_is_duplicated_into_every_reader() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((8, 4), F32))
    xt = g.op("Transpose", x, name="xt")
    a = g.op("MatMul", xt, const(g, 8, 3), name="a")
    b = g.op("MatMul", xt, const(g, 8, 5), name="b")
    g.set_outputs([a, b])
    groups = check(g)
    assert summary(groups) == [("matmul", ["Transpose", "MatMul"])] * 2
    assert groups[0].nodes[0] is groups[1].nodes[0]
    assert all(v.name == "x" or v.is_constant for gr in groups for v in gr.inputs)
    no_dup = check(g, options=FusionOptions(duplicate=False))
    assert summary(no_dup) == [
        ("injective", ["Transpose"]),
        ("matmul", ["MatMul"]),
        ("matmul", ["MatMul"]),
    ]


def test_expensive_producer_is_not_recomputed_inside_a_matmul() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 8), F32))
    g.set_outputs([g.op("MatMul", g.op("Exp", x), const(g, 8, 2))])
    assert summary(check(g)) == [("injective", ["Exp"]), ("matmul", ["MatMul"])]


def test_reductions_take_any_elementwise_prologue() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 8), F32))
    y = g.add_input("y", TensorType((4, 8), F32))
    summed = g.op("Add", g.op("Exp", x), y)
    g.set_outputs([g.op("Softmax", summed)])
    assert summary(check(g)) == [("softmax", ["Exp", "Add", "Softmax"])]


def test_elementwise_diamond_is_one_kernel() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3, 5), F32))
    a = g.op("Sigmoid", x)
    g.set_outputs([g.op("Mul", g.op("Tanh", a), g.op("Exp", a))])
    assert summary(check(g)) == [("injective", ["Sigmoid", "Tanh", "Exp", "Mul"])]


def test_reshape_alone_is_a_view() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 6), F32))
    s = g.op("Softmax", x)
    g.set_outputs([g.op("Reshape", s, shape=(3, 4)), s])
    groups = check(g)
    assert summary(groups) == [("softmax", ["Softmax"]), ("injective", ["Reshape"])]
    assert groups[1].is_view
    assert not groups[0].is_view


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            FusionOptions(epilogue=False),
            [("conv2d", ["Pad", "Conv2d"]), ("pool", ["Add", "Relu", "MaxPool2d"])],
        ),
        (
            FusionOptions(prologue=False),
            [
                ("injective", ["Pad"]),
                ("conv2d", ["Conv2d", "Add", "Relu"]),
                ("pool", ["MaxPool2d"]),
            ],
        ),
    ],
)
def test_options_turn_rules_off(options: FusionOptions, expected: list[object]) -> None:
    assert summary(check(conv_block(), options=options)) == expected


def test_injective_off_keeps_elementwise_nodes_apart() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3, 5), F32))
    g.set_outputs([g.op("Tanh", g.op("Relu", x))])
    groups = check(g, options=FusionOptions(injective=False))
    assert summary(groups) == [("injective", ["Relu"]), ("injective", ["Tanh"])]


def test_patterns_follow_the_anchor() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 4, 6, 6), F32))
    dw = g.op("Conv2d", x, const(g, 4, 1, 3, 3), groups=4)
    red = g.op("ReduceMean", dw, axes=(2, 3))
    ln = g.op("LayerNorm", red, const(g, 1, 1), const(g, 1, 1), axis=2)
    g.set_outputs([ln])
    assert [gr.pattern for gr in check(g)] == ["depthwise_conv2d", "reduce", "layernorm"]


def test_grouped_convolution_has_no_pattern() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 4, 6, 6), F32))
    g.set_outputs([g.op("Conv2d", x, const(g, 6, 2, 3, 3), groups=2)])
    with pytest.raises(ValueError, match="grouped convolution"):
        fuse(g)


def test_import_operators_are_rejected() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 4, 6, 6), F32))
    g.set_outputs([g.op("Flatten", x)])
    with pytest.raises(ValueError, match=r"run passes\.optimize first"):
        fuse(g)


def test_kernel_names_are_c_identifiers() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3,), F32))
    a = g.op("Relu", x, name="float")
    b = g.op("Tanh", a, name="_hidden")
    g.set_outputs([a, b])
    assert [gr.name for gr in fuse(g)] == ["float_k", "hidden"]


def _broken(groups: list[FusedGroup], index: int, **changes: object) -> list[FusedGroup]:
    out = list(groups)
    out[index] = dataclasses.replace(out[index], **changes)  # type: ignore[arg-type]
    return out


def test_verify_groups_catches_broken_partitions() -> None:
    g = conv_block()
    groups = fuse(g)
    conv, pool = groups
    cases = [
        (_broken(groups, 0, pattern="matmul"), "pattern"),
        (_broken(groups, 0, inputs=conv.inputs[::-1]), "first-use order"),
        (_broken(groups, 0, nodes=conv.nodes[::-1]), "topological"),
        (_broken(groups, 0, output=conv.nodes[2].output), "read outside"),
        (_broken(groups, 1, name=conv.name), "unique"),
        ([pool, conv], "before a previous group"),
        ([conv], "in no group"),
        (_broken(groups, 0, nodes=[*conv.nodes, pool.nodes[0]], output=pool.output), "reduction"),
    ]
    for broken, message in cases:
        with pytest.raises(ValueError, match=message):
            verify_groups(g, broken)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graphs())
def test_fusion_preserves_semantics_on_random_graphs(g: Graph) -> None:
    for kwargs in ({}, {"enabled": False}, {"options": FusionOptions(duplicate=False)}):
        check(g, **kwargs)


def test_duplication_is_one_level_deep() -> None:
    """A residual stream: each Add is read by a LayerNorm and by the next Add."""
    g = Graph()
    h = g.add_input("x", TensorType((4, 8), F32))
    outputs = []
    for i in range(4):
        h = g.op("Add", h, g.add_input(f"r{i}", TensorType((4, 8), F32)), name=f"h{i}")
        outputs.append(g.op("LayerNorm", h, const(g, 8), const(g, 8), name=f"ln{i}"))
    g.set_outputs([*outputs, h])
    groups = check(g)
    longest = max(len(gr.nodes) for gr in groups)
    assert longest <= 3, summary(groups)
