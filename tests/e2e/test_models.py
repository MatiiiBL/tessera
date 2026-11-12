"""Small models end to end: ONNX -> tessera.compile -> run, against the interpreter
and ONNX Runtime, on portable C, the host's ISA and Metal."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import onnx
import pytest

from e2e.onnx_models import cnn, mlp, onnxruntime_outputs, random_inputs, transformer_block
from tessera.backend.metal.runtime import available as metal_available
from tessera.driver import lower
from tessera.ir import interpret
from tessera.target import host_isa

MODELS: dict[str, Callable[[], onnx.ModelProto]] = {
    "mlp": mlp,
    "cnn": cnn,
    "transformer": transformer_block,
    "transformer_tanh": lambda: transformer_block(gelu="tanh"),
}
TARGETS = [
    "c:generic",
    pytest.param(
        f"c:{host_isa()}",
        marks=pytest.mark.skipif(host_isa() == "generic", reason="no SIMD ISA on this host"),
        id="c:host",
    ),
    pytest.param(
        "metal",
        marks=[pytest.mark.metal, pytest.mark.skipif(not metal_available(), reason="no Metal")],
    ),
]
TOLERANCE = 1e-4  # of the output's largest magnitude; fp32 kernels against a float64 reference


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return float(diff.max() / np.abs(b).max())


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("name", list(MODELS))
def test_model_matches_the_interpreter_and_onnxruntime(name: str, target: str) -> None:
    model = MODELS[name]()
    inputs = random_inputs(model)
    comp = lower(model, target=target)
    out = comp.build().run(inputs)
    reference = interpret(comp.graph, inputs, float64=True)
    ort = onnxruntime_outputs(model, inputs)
    assert set(out) == set(ort)
    for key in out:
        assert out[key].shape == ort[key].shape
        assert relative_error(out[key], reference[key]) < TOLERANCE
        assert relative_error(out[key], ort[key]) < TOLERANCE


@pytest.mark.parametrize("name", ["cnn", "transformer"])
def test_ablations_compute_the_same_function(name: str) -> None:
    model = MODELS[name]()
    inputs = random_inputs(model)
    ort = onnxruntime_outputs(model, inputs)
    full = lower(model, target="c:generic")
    unfused = lower(model, target="c:generic", fuse=False)
    minimal = lower(model, target="c:generic", optimize=False)
    assert len(unfused.kernels) > len(full.kernels)
    assert len(minimal.graph.nodes) >= len(full.graph.nodes)
    for comp in (unfused, minimal):
        out = comp.build().run(inputs)
        for key in ort:
            assert relative_error(out[key], ort[key]) < TOLERANCE


def test_kernels_are_named_after_their_groups() -> None:
    comp = lower(mlp(), target="c:generic")
    names = [k.name for k in comp.kernels]
    assert names == [g.name for g in comp.groups if not g.is_view]
    assert set(comp.plan.funcs) == set(names)
    assert [launch.kernel for launch in comp.plan.launches] == names


def test_tessera_compile_is_the_entry_point() -> None:
    import tessera

    model = mlp()
    inputs = random_inputs(model)
    out = tessera.compile(model, target="c:generic").run(inputs)
    ort = onnxruntime_outputs(model, inputs)
    assert relative_error(out["probs"], ort["probs"]) < TOLERANCE
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = tessera.nope  # type: ignore[attr-defined]


@pytest.mark.models
@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2", "bert", "gpt2"])
@pytest.mark.parametrize("target", ["c:sme", "metal"])
def test_every_ablation_lowers_the_benchmark_models(name: str, target: str) -> None:
    from tessera.bench.zoo import load_manifest

    spec = load_manifest()[name]
    for fuse, optimize in ((True, True), (False, True), (True, False)):
        comp = lower(spec.onnx_path(), target=target, fuse=fuse, optimize=optimize, tuned=False)
        assert comp.kernels
