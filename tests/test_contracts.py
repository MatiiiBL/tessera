"""Smoke tests of the shared IR records."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.target import Target
from tessera.tir.expr import Binary, IntImm, Var, lt
from tessera.tir.stmt import Buffer
from tessera.tune.space import ApplyConfig, ConfigSpace, DefineConfig, split_options


def test_graph_builds_and_verifies() -> None:
    g = Graph("tiny")
    x = g.add_input("x", TensorType((2, 3), DType.float32))
    b = g.add_constant(np.ones(3, np.float32), "bias")
    y = g.op("Relu", g.op("Add", x, b))
    g.set_outputs([y])
    g.verify()
    assert y.type == TensorType((2, 3), DType.float32)
    assert [n.op for n in g.nodes] == ["Add", "Relu"]
    copy = g.copy()
    copy.verify()
    assert copy.constants()[0].data is b.data
    assert "Relu" in str(g)


def test_verify_rejects_use_before_definition() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4,), DType.float32))
    y = g.op("Relu", x)
    z = g.op("Relu", y)
    g.nodes.reverse()
    g.set_outputs([z])
    with pytest.raises(ValueError, match="before it is defined"):
        g.verify()
    g.topo_sort()
    g.verify()


def test_expressions_compare_structurally_and_vars_by_identity() -> None:
    i, j = Var("i"), Var("i")
    assert i + 1 == i + 1
    assert i + 1 != j + 1
    assert (i * 4 + 2) == Binary("add", Binary("mul", i, IntImm(4)), IntImm(2))
    assert lt(i, 3).dtype is DType.bool_
    buf = Buffer("A", (4, 8))
    assert buf.strides == (8, 1)
    assert buf[i, 2].indices == (i, IntImm(2))


def test_config_space_round_trip() -> None:
    def template(cfg: DefineConfig | ApplyConfig) -> tuple[object, object]:
        return cfg.split("tile", 12, parts=2, default=(3, 4)), cfg.choice("unroll", [0, 4])

    define = DefineConfig()
    assert template(define) == ((3, 4), 0)
    space = ConfigSpace(define.knobs, define.defaults)
    assert space.size == len(split_options(12, 2)) * 2
    points = list(space)
    assert len(points) == space.size
    for index in (0, space.size - 1):
        assert space.index_of(space.get(index)) == index
    assert template(ApplyConfig(space.get(3))) == tuple(space.get(3).values())
    assert all(a * b >= 10 for a, b in split_options(10, 2, "power2"))  # type: ignore[misc]


def test_target_parse() -> None:
    assert Target.parse("c:generic").isa == "generic"
    assert Target.parse("metal").kind == "metal"
    with pytest.raises(ValueError, match="unknown target"):
        Target.parse("cuda")
