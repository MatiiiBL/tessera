"""Every CPU template, on every pattern, at its default and at random points of
its space: interpreted and compiled, the result matches NumPy."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import c_workloads as W
import numpy as np
import pytest
from c_workloads import Workload

from tessera import tir
from tessera.backend.base import Template, get_backend
from tessera.backend.c import run_kernel
from tessera.backend.c.codegen import generate_kernel
from tessera.backend.c.microkernels import host_supports
from tessera.plan import PATTERNS
from tessera.target import CpuIsa, Target
from tessera.tir.schedule import Schedule
from tessera.tune.space import ApplyConfig, ConfigSpace, DefineConfig, KnobValue

WORKLOADS: dict[str, Callable[[], Workload]] = {
    "matmul": lambda: W.matmul(12, 32, 20),
    "matmul_batched": lambda: W.matmul(6, 16, 9, batch=3, epilogue=False),
    "matmul_odd": lambda: W.matmul(7, 10, 5),
    "attention_scores": lambda: W.attention_scores(2, 8, 16),
    "shared_operands": lambda: W.shared_operands(12, 16),
    "conv2d_3x3": lambda: W.conv2d(1, 6, 8, 4, 16),
    "conv2d_strided": lambda: W.conv2d(2, 9, 9, 3, 8, (7, 7), (2, 2), (3, 3, 3, 3)),
    "conv2d_1x1": lambda: W.conv2d(1, 4, 4, 8, 32, (1, 1), (1, 1), (0, 0, 0, 0)),
    "conv2d_dilated": lambda: W.conv2d(
        1, 8, 7, 5, 12, (3, 3), (1, 2), (2, 1, 0, 2), (2, 1), epilogue=False
    ),
    "depthwise": lambda: W.conv2d(1, 8, 8, 16, 16, strides=(2, 2), depthwise=True),
    "maxpool": lambda: W.pool("max", 1, 9, 9, 8),
    "avgpool": lambda: W.pool("avg", 2, 8, 8, 4, (2, 2), (2, 2), (0, 0, 0, 0)),
    "reduce_last": lambda: W.reduce("sum", (6, 40), (1,)),
    "reduce_middle": lambda: W.reduce("mean", (2, 5, 6, 16), (1, 2)),
    "reduce_keepdims": lambda: W.reduce("max", (3, 4, 7), (2,), keepdims=1),
    "softmax": lambda: W.softmax(5, 48, scale=True),
    "layernorm": lambda: W.layernorm(4, 64),
    "transpose": lambda: W.transpose(2, 3, 5, 8),
    "gelu": lambda: W.gelu((4, 32)),
    "gelu_tanh": lambda: W.gelu((3, 20), "tanh"),
    "gather": lambda: W.gather_rows(10, 16, 6),
    "concat_slice": lambda: W.concat_slice(3, 8),
}


def _template(w: Workload, target: Target) -> Template:
    return get_backend("c").templates(target)[w.lowered.pattern]


def _space(w: Workload, target: Target) -> ConfigSpace:
    cfg = DefineConfig()
    _template(w, target)(Schedule(w.func), w.lowered, cfg)
    return ConfigSpace(cfg.knobs, cfg.defaults)


def _scheduled(w: Workload, target: Target, values: dict[str, KnobValue]) -> Schedule:
    sch = Schedule(w.func)
    _template(w, target)(sch, w.lowered, ApplyConfig(values))
    tir.verify(sch.func)
    return sch


def _check(w: Workload, sch: Schedule, target: Target, *, interpret: bool) -> None:
    expected = w.expected
    if interpret:
        out = w.output()
        tir.interpret(sch.func, [*w.inputs, out])
        np.testing.assert_allclose(out, expected, rtol=w.tolerance, atol=w.tolerance)
    out = w.output()
    run_kernel(sch.func, [*w.inputs, out], target)
    np.testing.assert_allclose(out, expected, rtol=w.tolerance, atol=w.tolerance)


def test_every_pattern_has_a_template() -> None:
    assert set(get_backend("c").templates(Target())) == set(PATTERNS)


def _isa(isa: CpuIsa) -> Any:
    marks = [pytest.mark.sme] if isa == "sme" else []
    skip = pytest.mark.skipif(not host_supports(isa), reason=f"this CPU has no {isa}")
    return pytest.param(isa, marks=[*marks, skip])


ISAS = [_isa(isa) for isa in ("generic", "neon", "sme")]


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_default_schedule(name: str, isa: CpuIsa) -> None:
    w = WORKLOADS[name]()
    target = Target(isa=isa, num_threads=4)
    space = _space(w, target)
    _check(w, _scheduled(w, target, space.default()), target, interpret=True)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_random_schedules(name: str, isa: CpuIsa) -> None:
    w = WORKLOADS[name]()
    target = Target(isa=isa, num_threads=3)
    space = _space(w, target)
    rng = random.Random(f"{name}-{isa}")
    points = {space.index_of(space.sample(rng)) for _ in range(5)}
    for index in sorted(points):
        _check(w, _scheduled(w, target, space.get(index)), target, interpret=False)


MICRO = ["matmul", "matmul_batched", "attention_scores", "shared_operands", "conv2d_3x3",
         "conv2d_strided", "conv2d_1x1", "conv2d_dilated"]  # fmt: skip


@pytest.mark.parametrize("isa", ISAS[1:])
@pytest.mark.parametrize("name", MICRO)
def test_microkernels_are_used_by_default(name: str, isa: CpuIsa) -> None:
    w = WORKLOADS[name]()
    target = Target(isa=isa, num_threads=4)
    sch = _scheduled(w, target, _space(w, target).default())
    assert any(line.startswith("tensorize(") for line in sch.trace)
    assert f"tsr_{isa}_" in generate_kernel(sch.func, isa).text


EDGE_TILES = {
    "matmul": (lambda: W.matmul(37, 50, 20), {"micro_m": 16, "micro_n": 32}),
    "attention": (lambda: W.attention_scores(2, 40, 16), {"micro_m": 16, "micro_n": 32}),
    "conv2d": (lambda: W.conv2d(1, 6, 6, 8, 40), {"micro_c": 16}),
}


@pytest.mark.parametrize("isa", ISAS[1:])
@pytest.mark.parametrize("name", sorted(EDGE_TILES))
def test_microkernels_on_edge_tiles(name: str, isa: CpuIsa) -> None:
    """Tiles that do not divide the output: the last tile in each direction
    passes its valid size to the microkernel."""
    make, knobs = EDGE_TILES[name]
    w = make()
    target = Target(isa=isa, num_threads=3)
    sch = _scheduled(w, target, {**_space(w, target).default(), **knobs})
    assert any(line.startswith("tensorize(") for line in sch.trace)
    _check(w, sch, target, interpret=True)


def test_balanced_tiles() -> None:
    from tessera.backend.c.templates.common import balanced_tile, tile_options

    # One tile per thread when that keeps tiles at most 256 wide...
    assert balanced_tile(768, 1, 4, tile_options(768, 1024)) == 192
    # ... else whole numbers of tiles per thread: 16 tiles of 192 over 4 threads.
    assert balanced_tile(3072, 1, 4, tile_options(3072, 1024)) == 192
    # Other tiles count: 12 heads already feed 4 threads, so the whole axis.
    assert balanced_tile(128, 12, 4, tile_options(128, 1024)) == 128
    # No option small enough: the smallest.
    assert balanced_tile(50257, 1, 4, [29, 1733]) == 29
