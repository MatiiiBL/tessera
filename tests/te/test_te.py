"""Tensor expressions build one block per stage under a root that allocates intermediates."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import te, tir
from tessera.dtype import DType
from tessera.tir.expr import Var
from tessera.tir.stmt import BlockRealize, For, IterKind, IterVar, Range


def test_matmul_stage_structure() -> None:
    a = te.placeholder((3, 5), DType.float32, "A")
    b = te.placeholder((5, 4), DType.float32, "B")
    k = te.reduce_axis(5, "k")
    c = te.compute((3, 4), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), "C")
    f = te.create_prim_func([a, b, c], "mm")
    tir.verify(f)
    assert [p.name for p in f.params] == ["A", "B", "C"]
    root = f.body
    assert isinstance(root, BlockRealize)
    assert root.block.alloc_buffers == ()
    loops = []
    node = root.block.body
    while isinstance(node, For):
        loops.append(node)
        node = node.body
    assert [lp.extent.value for lp in loops] == [3, 4, 5]  # type: ignore[attr-defined]
    assert isinstance(node, BlockRealize)
    block = node.block
    assert block.name == "C"
    assert [iv.kind for iv in block.iter_vars] == [
        IterKind.SPATIAL,
        IterKind.SPATIAL,
        IterKind.REDUCE,
    ]
    assert block.init is not None
    assert [r.buffer.name for r in block.reads] == ["A", "B"]
    x = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
    out = np.zeros((3, 4), np.float32)
    tir.interpret(f, [x, y, out])
    np.testing.assert_allclose(out, x @ y, rtol=1e-5, atol=1e-6)


def test_intermediates_are_allocated_and_stages_ordered() -> None:
    x = te.placeholder((6,), DType.float32, "X")
    t = te.compute((6,), lambda i: x[i] * 2.0, "t")
    u = te.compute((6,), lambda i: t[i] + x[i], "u")
    f = te.create_prim_func([x, u])
    tir.verify(f)
    assert [b.name for b in f.body.block.alloc_buffers] == ["t"]  # type: ignore[attr-defined]
    text = tir.script(f)
    assert text.index("with block('t')") < text.index("with block('u')")


@pytest.mark.parametrize(
    ("reduce", "numpy"), [(te.sum, np.sum), (te.max, np.max), (te.min, np.min)]
)
def test_reductions(reduce: object, numpy: object) -> None:
    x = te.placeholder((4, 7), DType.float32, "X")
    k = te.reduce_axis(7, "k")
    r = te.compute((4,), lambda i: reduce(x[i, k], axis=k), "r")  # type: ignore[operator]
    f = te.create_prim_func([x, r])
    data = np.random.default_rng(2).standard_normal((4, 7)).astype(np.float32)
    out = np.zeros(4, np.float32)
    tir.interpret(f, [data, out])
    np.testing.assert_allclose(out, numpy(data, axis=1), rtol=1e-5)  # type: ignore[operator]


def test_reduce_axes_work_in_index_arithmetic() -> None:
    x = te.placeholder((10,), DType.float32, "X")
    k = te.reduce_axis(3, "k")
    y = te.compute((4,), lambda i: te.sum(x[i * 2 + k], axis=k), "y")
    data = np.arange(10, dtype=np.float32)
    out = np.zeros(4, np.float32)
    tir.interpret(te.create_prim_func([x, y]), [data, out])
    np.testing.assert_array_equal(out, [data[2 * i : 2 * i + 3].sum() for i in range(4)])


def test_scalar_output_of_a_full_reduction() -> None:
    x = te.placeholder((5,), DType.float32, "X")
    k = te.reduce_axis(5, "k")
    s = te.compute((), lambda: te.sum(x[k], axis=k), "s")
    f = te.create_prim_func([x, s])
    tir.verify(f)
    out = np.zeros((), np.float32)
    tir.interpret(f, [np.arange(5, dtype=np.float32), out])
    assert out == 10


def test_errors() -> None:
    x = te.placeholder((4,), DType.float32, "X")
    y = te.compute((4,), lambda i: x[i] + 1.0, "y")
    with pytest.raises(ValueError, match="not a parameter"):
        te.create_prim_func([y])
    k = te.reduce_axis(4, "k")
    with pytest.raises(ValueError, match="whole body"):
        te.compute((1,), lambda i: te.sum(x[k], axis=k) * 2.0, "bad")
    with pytest.raises(IndexError, match="rank"):
        te.compute((4,), lambda i: x[i, i], "bad")
    with pytest.raises(ValueError, match="not a reduction axis"):
        te.sum(x[0], axis=IterVar(Var("i"), Range.of(4)))
