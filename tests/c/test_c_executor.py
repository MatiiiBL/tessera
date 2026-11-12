"""Whole plans: one library per model, run through ctypes."""

from __future__ import annotations

import numpy as np
import pytest
from c_funcs import S, loop, matmul, realize, root, tiled_matmul

from tessera.backend.base import get_backend
from tessera.backend.c.executor import CModel
from tessera.dtype import DType
from tessera.ir.graph import NDArray, TensorType
from tessera.plan import BufferRef, ExecutionPlan, KernelLaunch, MemoryPlan, StorageKind
from tessera.target import Target
from tessera.tir.expr import Var, maximum
from tessera.tir.stmt import Buffer, BufferStore, ForKind, PrimFunc

rng = np.random.default_rng(1)
F32 = DType.float32


def _bias_relu(m: int, n: int) -> PrimFunc:
    """Y = relu(X + b), rows in parallel."""
    x, b, y = Buffer("X", (m, n)), Buffer("b", (n,)), Buffer("Y", (m, n))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    value = maximum(x[vi, vj] + b[vj], 0.0)
    block = realize("Y", [(vi, m, S, i), (vj, n, S, j)], BufferStore(y, value, (vi, vj)))
    body = loop(i, m, loop(j, n, block, ForKind.VECTORIZED), ForKind.PARALLEL)
    return PrimFunc("bias_relu", (x, b, y), root(body))


def _ref(kind: StorageKind, nbytes: int, offset: int = 0, index: int = -1) -> BufferRef:
    return BufferRef(kind, offset, nbytes, index)


def _two_layer_plan() -> tuple[ExecutionPlan, NDArray, NDArray]:
    """y = relu(x + b) @ W: x an input, b and W constants, the hidden value in
    the arena, y an output."""
    m, n, k = 10, 32, 24
    bias = rng.standard_normal(k).astype(np.float32)
    weight = rng.standard_normal((k, n)).astype(np.float32)
    blob = np.zeros(256 + weight.nbytes, np.uint8)
    blob[: bias.nbytes] = bias.view(np.uint8)
    blob[256:] = weight.view(np.uint8).reshape(-1)
    hidden = m * k * 4
    refs = {
        "x": _ref(StorageKind.INPUT, hidden, index=0),
        "b": _ref(StorageKind.CONSTANT, bias.nbytes),
        "W": _ref(StorageKind.CONSTANT, weight.nbytes, offset=256),
        "h": _ref(StorageKind.ARENA, hidden, offset=512),
        "y": _ref(StorageKind.OUTPUT, m * n * 4, index=0),
    }
    memory = MemoryPlan(refs, arena_bytes=512 + hidden, constants=blob)
    launches = [
        KernelLaunch("bias_relu", (refs["x"], refs["b"], refs["h"])),
        KernelLaunch("tiled", (refs["h"], refs["W"], refs["y"])),
    ]
    plan = ExecutionPlan(
        "two_layer",
        launches,
        memory,
        inputs=[("x", TensorType((m, k), F32))],
        outputs=[("y", TensorType((m, n), F32))],
        funcs={"bias_relu": _bias_relu(m, k), "tiled": tiled_matmul(m, n, k)},
    )
    return plan, bias, weight


@pytest.fixture(scope="module")
def two_layer() -> tuple[CModel, NDArray, NDArray]:
    plan, bias, weight = _two_layer_plan()
    model = get_backend("c").build(plan, Target(num_threads=4))
    assert isinstance(model, CModel)
    return model, bias, weight


def test_run_matches_numpy(two_layer: tuple[CModel, NDArray, NDArray]) -> None:
    model, bias, weight = two_layer
    x = rng.standard_normal((10, 24)).astype(np.float32)
    y = model.run({"x": x})["y"]
    np.testing.assert_allclose(y, np.maximum(x + bias, 0) @ weight, rtol=1e-5, atol=1e-5)
    # Outputs are new arrays each run.
    assert model.run({"x": x})["y"] is not y
    assert model.inputs == [("x", TensorType((10, 24), F32))]
    assert model.target.threads == 4


def test_benchmark_and_profile(two_layer: tuple[CModel, NDArray, NDArray]) -> None:
    model, _, _ = two_layer
    x = rng.standard_normal((10, 24)).astype(np.float32)
    times = model.benchmark({"x": x}, warmup=1, repeat=7)
    assert len(times) == 7
    assert all(t > 0 for t in times)
    profile = model.profile({"x": x}, repeat=3)
    assert [p.kernel for p in profile] == ["bias_relu", "tiled"]
    assert all(p.seconds > 0 for p in profile)


def test_sources_are_one_translation_unit(two_layer: tuple[CModel, NDArray, NDArray]) -> None:
    model, _, _ = two_layer
    sources = model.sources()
    assert set(sources) == {"model.c", "tessera_runtime.h"}
    text = sources["model.c"]
    assert "#define TSR_NUM_THREADS 4" in text
    assert text.index("static void bias_relu(") < text.index("static void tiled(")
    assert "  bias_relu((float*)inputs[0], (float*)constants, (float*)(arena + 512));" in text
    assert "  tiled((float*)(arena + 512), (float*)(constants + 256), (float*)outputs[0]);" in text


def test_input_checks(two_layer: tuple[CModel, NDArray, NDArray]) -> None:
    model, _, _ = two_layer
    with pytest.raises(KeyError, match="missing input 'x'"):
        model.run({})
    with pytest.raises(ValueError, match="has shape"):
        model.run({"x": np.zeros((3, 3), np.float32)})


def test_output_held_in_the_arena_and_kernel_launched_twice() -> None:
    """y = (x @ W) @ W, with the same kernel launched twice and the result left
    in the arena (as a view group's output would be)."""
    n = 8
    func = matmul(n, n, n)
    nbytes = n * n * 4
    weight = rng.standard_normal((n, n)).astype(np.float32)
    refs = {
        "x": _ref(StorageKind.INPUT, nbytes, index=0),
        "W": _ref(StorageKind.CONSTANT, nbytes),
        "h": _ref(StorageKind.ARENA, nbytes),
        "y": _ref(StorageKind.ARENA, nbytes, offset=nbytes),
    }
    plan = ExecutionPlan(
        "twice",
        [
            KernelLaunch("matmul", (refs["x"], refs["W"], refs["h"])),
            KernelLaunch("matmul", (refs["h"], refs["W"], refs["y"])),
        ],
        MemoryPlan(refs, 2 * nbytes, weight.view(np.uint8).reshape(-1)),
        inputs=[("x", TensorType((n, n), F32))],
        outputs=[("y", TensorType((n, n), F32))],
        funcs={"matmul": func},
    )
    model = get_backend("c").build(plan, Target(isa="generic"))
    x = rng.standard_normal((n, n)).astype(np.float32)
    np.testing.assert_allclose(model.run({"x": x})["y"], x @ weight @ weight, rtol=1e-4, atol=1e-4)
    assert model.sources()["model.c"].count("static void matmul(") == 1


def test_plan_errors() -> None:
    plan, _, _ = _two_layer_plan()
    plan.launches.append(KernelLaunch("missing", ()))
    with pytest.raises(ValueError, match="no PrimFunc"):
        get_backend("c").build(plan, Target())
    plan, _, _ = _two_layer_plan()
    small = BufferRef(StorageKind.ARENA, 0, 4)
    plan.launches[0] = KernelLaunch("bias_relu", (small, small, small))
    with pytest.raises(ValueError, match="needs 960 bytes"):
        get_backend("c").build(plan, Target())
    with pytest.raises(ValueError, match="cannot build for a 'metal' target"):
        get_backend("c").build(plan, Target(kind="metal"))
