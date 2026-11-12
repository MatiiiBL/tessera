"""The result pages ``tessera bench`` writes: ``results/{models,ops,tuning}.{md,json}``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tessera.bench import models as models_bench
from tessera.bench import ops as ops_bench
from tessera.bench import tuning as tuning_bench
from tessera.bench.report import environment, header, write
from tessera.bench.zoo import MODELS, load_manifest
from tessera.target import Target

type Progress = Callable[[str], None]


def default_targets() -> list[str]:
    """The host's best C target, and Metal when there is a GPU."""
    from tessera.backend.metal.runtime import available

    host = Target.host()
    return [f"c:{host.isa}", *(["metal"] if available() else [])]


def models_page(
    targets: Sequence[str],
    *,
    names: Sequence[str] = MODELS,
    provisional: bool | None = None,
    directory: Path | None = None,
    progress: Progress = print,
) -> Path:
    """``results/models``: every model on every system, with the ablations."""
    env = environment(provisional=provisional)
    specs = load_manifest()
    results = [
        models_bench.bench_model(specs[name], targets=targets, progress=progress) for name in names
    ]
    command = "tessera bench models " + " ".join(f"--target {t}" for t in targets)
    text = header("Models", env, command.strip()) + "\n" + models_bench.markdown(results)
    data: dict[str, Any] = {"environment": env, "models": [r.as_dict() for r in results]}
    return write("models", data, text, directory)


def ops_page(
    targets: Sequence[str],
    *,
    provisional: bool | None = None,
    directory: Path | None = None,
    progress: Progress = print,
) -> Path:
    """``results/ops``: GEMM throughput and the ResNet-50 convolutions."""
    env = environment(provisional=provisional)
    gemm = ops_bench.bench_gemm(targets=targets, progress=progress)
    convs: list[ops_bench.OpResult] = []
    resnet = load_manifest()["resnet50"]
    for i, target in enumerate(targets):
        baselines = ("pytorch-cpu", "pytorch-mps") if i == 0 else ()
        convs += ops_bench.bench_convs(
            resnet.onnx_path(), target=target, baselines=baselines, progress=progress
        )
    command = "tessera bench ops " + " ".join(f"--target {t}" for t in targets)
    text = (
        header("Kernels", env, command.strip())
        + "\n## GEMM (float32, C = A @ B, B constant)\n\n"
        + ops_bench.markdown(gemm)
        + "\n\n## ResNet-50 convolutions (batch 1, 224x224)\n\n"
        + "Tessera's time is the median over the kernels of that shape in a profile of the "
        + "whole model, epilogue included; PyTorch runs the convolution alone.\n\n"
        + ops_bench.markdown(convs)
    )
    data = {
        "environment": env,
        "gemm": [r.as_dict() for r in gemm],
        "convolutions": [r.as_dict() for r in convs],
    }
    return write("ops", data, text, directory)


def tuning_page(
    *,
    real: Sequence[tuning_bench.Comparison] = (),
    trials: int = 256,
    seeds: int = 5,
    provisional: bool | None = None,
    directory: Path | None = None,
    command: str = "tessera bench tuning",
) -> Path:
    """``results/tuning``: random against model-guided search, synthetic and real tasks."""
    env = environment(provisional=provisional)
    synthetic = tuning_bench.compare_synthetic(trials=trials, seeds=range(seeds))
    parts = [
        header("Tuning: random against model-guided search", env, command),
        "Each cell is the best throughput found so far relative to the reference "
        "(1.000 is the reference), after that many measurements.",
        "## Synthetic tasks (analytic cost, exact optimum)",
        tuning_bench.table(synthetic),
    ]
    if real:
        parts += ["## Real tasks (measured)", tuning_bench.table(real)]
    data = {
        "environment": env,
        "synthetic": [c.as_dict() for c in synthetic],
        "real": [c.as_dict() for c in real],
    }
    return write("tuning", data, "\n\n".join(parts), directory)
