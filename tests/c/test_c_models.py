"""Whole graphs through fusion, lowering, the CPU templates, memory planning and
one compiled library, against the graph interpreter."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tessera import compute, passes
from tessera.backend.base import get_backend
from tessera.backend.c import CModel
from tessera.dtype import DType
from tessera.ir import Graph, TensorType, interpret
from tessera.plan import ExecutionPlan, KernelLaunch
from tessera.target import Target
from tessera.tir.schedule import Schedule
from tessera.tune.space import ApplyConfig, DefineConfig

type NDArray = npt.NDArray[Any]
F32 = DType.float32


def compile_graph(g: Graph, target: Target, *, fuse: bool = True) -> CModel:
    """What the driver does, with every template at its defaults."""
    backend = get_backend("c")
    templates = backend.templates(target)
    groups = passes.fuse(g, enabled=fuse)
    memory = passes.plan_memory(g, groups)
    funcs, launches = {}, []
    for group in groups:
        if group.is_view:
            continue
        lowered = compute.lower_group(group)
        template = templates[group.pattern]
        defaults = DefineConfig()
        template(Schedule(lowered.func), lowered, defaults)
        sch = Schedule(lowered.func)
        template(sch, lowered, ApplyConfig(defaults.defaults))
        funcs[group.name] = sch.func
        refs = tuple(memory.refs[v.name] for v in [*group.inputs, group.output])
        launches.append(KernelLaunch(group.name, refs))
    plan = ExecutionPlan(
        g.name,
        launches,
        memory,
        inputs=[(v.name, v.type) for v in g.inputs],
        outputs=[(v.name, v.type) for v in g.outputs],
        funcs=funcs,
    )
    model = backend.build(plan, target)
    assert isinstance(model, CModel)
    return model


def _const(g: Graph, rng: np.random.Generator, *shape: int, scale: float = 0.2) -> Any:
    return g.add_constant((rng.standard_normal(shape) * scale).astype(np.float32))


def small_cnn() -> Graph:
    """conv-relu, max pool, a residual pair of convolutions, global average pool,
    a fully connected layer and softmax."""
    rng = np.random.default_rng(0)
    g = Graph("small_cnn")
    x = g.add_input("image", TensorType((1, 16, 16, 3), F32))
    conv: dict[str, Any] = {"dilations": (1, 1), "groups": 1, "layout": "NHWC"}
    y = g.op(
        "Conv2d", x, _const(g, rng, 3, 3, 3, 16), _const(g, rng, 16),
        strides=(2, 2), pads=(1, 1, 1, 1), **conv,
    )  # fmt: skip
    y = g.op("Relu", y)
    y = g.op("MaxPool2d", y, kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), layout="NHWC")
    z = g.op(
        "Conv2d",
        y,
        _const(g, rng, 3, 3, 16, 16),
        _const(g, rng, 16),
        strides=(1, 1),
        pads=(1, 1, 1, 1),
        **conv,
    )
    z = g.op("Relu", z)
    z = g.op(
        "Conv2d",
        z,
        _const(g, rng, 1, 1, 16, 16),
        _const(g, rng, 16),
        strides=(1, 1),
        pads=(0, 0, 0, 0),
        **conv,
    )
    y = g.op("Relu", g.op("Add", y, z))
    dw = g.op(
        "Conv2d",
        y,
        _const(g, rng, 3, 3, 1, 16),
        _const(g, rng, 16),
        strides=(1, 1),
        pads=(1, 1, 1, 1),
        dilations=(1, 1),
        groups=16,
        layout="NHWC",
    )
    y = g.op("ReduceMean", g.op("Relu", dw), axes=(1, 2), keepdims=0)
    logits = g.op("Add", g.op("MatMul", y, _const(g, rng, 16, 10)), _const(g, rng, 10))
    g.set_outputs([g.op("Softmax", logits, axis=1)])
    return g


def transformer_block(seq: int = 8, dim: int = 32, heads: int = 2) -> Graph:
    """Pre-norm attention and a GELU MLP with residuals."""
    rng = np.random.default_rng(1)
    g = Graph("block")
    x = g.add_input("tokens", TensorType((seq, dim), F32))
    h = g.op(
        "LayerNorm", x, _const(g, rng, dim, scale=1.0), _const(g, rng, dim), axis=1, epsilon=1e-5
    )
    hd = dim // heads

    def project(name: str) -> Any:
        y = g.op("Add", g.op("MatMul", h, _const(g, rng, dim, dim)), _const(g, rng, dim))
        y = g.op("Reshape", y, shape=(seq, heads, hd))
        return g.op("Transpose", y, perm=(1, 0, 2))  # heads, seq, hd

    q, k, v = project("q"), project("k"), project("v")
    scores = g.op("MatMul", q, g.op("Transpose", k, perm=(0, 2, 1)))
    scores = g.op("Mul", scores, g.add_constant(np.float32([hd**-0.5])))
    attn = g.op("MatMul", g.op("Softmax", scores, axis=2), v)  # heads, seq, hd
    attn = g.op("Reshape", g.op("Transpose", attn, perm=(1, 0, 2)), shape=(seq, dim))
    x = g.op("Add", x, g.op("MatMul", attn, _const(g, rng, dim, dim)))
    h = g.op(
        "LayerNorm", x, _const(g, rng, dim, scale=1.0), _const(g, rng, dim), axis=1, epsilon=1e-5
    )
    m = g.op(
        "Gelu",
        g.op("Add", g.op("MatMul", h, _const(g, rng, dim, 4 * dim)), _const(g, rng, 4 * dim)),
        approximate="none",
    )
    x = g.op("Add", x, g.op("MatMul", m, _const(g, rng, 4 * dim, dim)))
    g.set_outputs([x])
    return g


def _check(g: Graph, target: Target, *, fuse: bool = True) -> CModel:
    g.verify()
    rng = np.random.default_rng(2)
    feeds = {v.name: rng.standard_normal(v.type.shape).astype(np.float32) for v in g.inputs}
    model = compile_graph(g, target, fuse=fuse)
    got = model.run(feeds)
    want = interpret(g, feeds)
    for name, value in want.items():
        np.testing.assert_allclose(got[name], value, rtol=2e-4, atol=2e-5)
    return model


@pytest.mark.parametrize("threads", [1, 4])
def test_small_cnn(threads: int) -> None:
    model = _check(small_cnn(), Target(isa="generic", num_threads=threads))
    profile = model.profile({"image": np.zeros((1, 16, 16, 3), np.float32)}, repeat=2)
    assert [p.kernel for p in profile] == [launch.kernel for launch in model.plan.launches]


def test_small_cnn_unfused() -> None:
    _check(small_cnn(), Target(isa="generic", num_threads=2), fuse=False)


@pytest.mark.parametrize("threads", [1, 4])
def test_transformer_block(threads: int) -> None:
    _check(transformer_block(), Target(isa="generic", num_threads=threads))
