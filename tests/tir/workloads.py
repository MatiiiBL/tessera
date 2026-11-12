"""Small computations used by the loop-IR tests, with NumPy references."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera import te
from tessera.dtype import DType
from tessera.tir.expr import Expr, ge, logical_and, lt, maximum, minimum, select
from tessera.tir.stmt import PrimFunc

type NDArray = npt.NDArray[Any]


@dataclass
class Workload:
    func: PrimFunc
    inputs: list[NDArray]
    reference: Callable[..., NDArray]

    def outputs(self) -> list[NDArray]:
        return [
            np.full(p.shape, np.nan, p.dtype.numpy) for p in self.func.params[len(self.inputs) :]
        ]

    def expected(self) -> NDArray:
        return self.reference(*(x.astype(np.float64) for x in self.inputs))


def _rand(rng: np.random.Generator, *shape: int) -> NDArray:
    return rng.standard_normal(shape).astype(np.float32)


def matmul_bias_relu(m: int, n: int, k: int, seed: int = 0) -> Workload:
    """C = relu(A @ B + bias): a reduction and an elementwise epilogue."""
    a = te.placeholder((m, k), DType.float32, "A")
    b = te.placeholder((k, n), DType.float32, "B")
    bias = te.placeholder((n,), DType.float32, "bias")
    r = te.reduce_axis(k, "k")
    mm = te.compute(
        (m, n), lambda i, j: te.sum(a[i, r] * b[r, j], axis=r), "matmul", axis_names=["i", "j"]
    )
    out = te.compute(
        (m, n), lambda i, j: maximum(mm[i, j] + bias[j], 0.0), "relu", axis_names=["i", "j"]
    )
    rng = np.random.default_rng(seed)
    return Workload(
        te.create_prim_func([a, b, bias, out], "matmul_bias_relu"),
        [_rand(rng, m, k), _rand(rng, k, n), _rand(rng, n)],
        lambda x, y, z: np.maximum(x @ y + z, 0),
    )


def _clamp(e: Expr, n: int) -> Expr:
    return minimum(maximum(e, 0), n - 1)


def box_filter(h: int, w: int, c: int, seed: int = 0) -> Workload:
    """A zero-padded 3x3 box filter: pad stage, reduction, scale epilogue."""
    x = te.placeholder((h, w, c), DType.float32, "X")

    def pad_body(i: Expr, j: Expr, ch: Expr) -> Expr:
        inside = logical_and(
            logical_and(ge(i, 1), lt(i, h + 1)), logical_and(ge(j, 1), lt(j, w + 1))
        )
        return select(inside, x[_clamp(i - 1, h), _clamp(j - 1, w), ch], 0.0)

    pad = te.compute((h + 2, w + 2, c), pad_body, "pad", axis_names=["h", "w", "c"])
    kh, kw = te.reduce_axis(3, "kh"), te.reduce_axis(3, "kw")
    box = te.compute(
        (h, w, c),
        lambda i, j, ch: te.sum(pad[i + kh, j + kw, ch], axis=[kh, kw]),
        "box",
        axis_names=["h", "w", "c"],
    )
    out = te.compute(
        (h, w, c), lambda i, j, ch: box[i, j, ch] * 0.5, "scale", axis_names=["h", "w", "c"]
    )

    def reference(xs: NDArray) -> NDArray:
        xp = np.pad(xs, ((1, 1), (1, 1), (0, 0)))
        return sum(xp[i : i + h, j : j + w] for i in range(3) for j in range(3)) * 0.5

    rng = np.random.default_rng(seed)
    return Workload(te.create_prim_func([x, out], "box_filter"), [_rand(rng, h, w, c)], reference)


def row_softmax(m: int, n: int, seed: int = 0) -> Workload:
    """Softmax over rows: max reduction, exp-sum reduction, normalization."""
    x = te.placeholder((m, n), DType.float32, "X")
    k1, k2 = te.reduce_axis(n, "k"), te.reduce_axis(n, "k")
    mx = te.compute((m,), lambda i: te.max(x[i, k1], axis=k1), "rowmax", axis_names=["i"])
    from tessera.tir.expr import call

    ex = te.compute(
        (m,), lambda i: te.sum(call("exp", x[i, k2] - mx[i]), axis=k2), "rowsum", axis_names=["i"]
    )
    out = te.compute(
        (m, n),
        lambda i, j: call("exp", x[i, j] - mx[i]) / ex[i],
        "softmax",
        axis_names=["i", "j"],
    )

    def reference(xs: NDArray) -> NDArray:
        e = np.exp(xs - xs.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    return Workload(te.create_prim_func([x, out], "row_softmax"), [_rand(rng, m, n)], reference)


def elementwise_chain(m: int, n: int, seed: int = 0) -> Workload:
    """Three elementwise stages with a transpose in the middle."""
    x = te.placeholder((m, n), DType.float32, "X")
    a = te.compute((m, n), lambda i, j: x[i, j] * 2.0, "double", axis_names=["i", "j"])
    t = te.compute((n, m), lambda i, j: a[j, i] + 1.0, "transpose", axis_names=["i", "j"])
    out = te.compute((n, m), lambda i, j: maximum(t[i, j], 0.0), "relu", axis_names=["i", "j"])
    rng = np.random.default_rng(seed)
    return Workload(
        te.create_prim_func([x, out], "elementwise_chain"),
        [_rand(rng, m, n)],
        lambda xs: np.maximum((xs * 2 + 1).T, 0),
    )


def run(func: PrimFunc, w: Workload, *, check_bounds: bool = True) -> NDArray:
    """Interpret ``func`` (a schedule of ``w.func``) on the workload's inputs."""
    from tessera import tir

    outs = w.outputs()
    tir.interpret(func, [*w.inputs, *outs], check_bounds=check_bounds)
    return outs[0]
