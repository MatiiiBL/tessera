"""The result pages ``tessera bench`` writes: ``results/{models,ops,tuning}.{md,json}``."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tessera.bench import models as models_bench
from tessera.bench import ops as ops_bench
from tessera.bench import tuning as tuning_bench
from tessera.bench.report import environment, header, results_dir, write
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
    ablation_targets: Sequence[str] | None = None,
    variants: Sequence[str] = tuple(models_bench.VARIANTS),
    provisional: bool | None = None,
    directory: Path | None = None,
    progress: Progress = print,
) -> Path:
    """``results/models``: every model on every system, with the ablations.

    Models measured earlier and not named here are kept, each with the
    environment it was measured in, so each model can be measured by its own
    run of the command.
    """
    out = directory if directory is not None else results_dir()
    previous = out / "models.json"
    kept: dict[str, Any] = {}
    if previous.exists():
        kept = {m["model"]: m for m in json.loads(previous.read_text()).get("models", [])}
    specs = load_manifest()
    for name in names:
        env = environment(provisional=provisional)
        result = models_bench.bench_model(
            specs[name],
            targets=targets,
            ablation_targets=ablation_targets,
            variants=variants,
            progress=progress,
        )
        kept[name] = {**result.as_dict(), "environment": env}
    entries = [kept[n] for n in [*MODELS, *sorted(set(kept) - set(MODELS))] if n in kept]
    # The page's provenance is its latest model's; each model states its own below.
    latest = max((e["environment"] for e in entries), key=lambda e: str(e["date"]))
    env = {**latest, "provisional": any(e["environment"]["provisional"] for e in entries)}
    sections = []
    for entry in entries:
        e = entry["environment"]
        when = (
            f"Measured {e['date']} at commit `{e['commit']}`, load average {e['load_average'][0]}"
        )
        sections.append(
            models_bench.markdown([models_bench.ModelResult.from_dict(entry)])
            + f"\n\n{when}{' (provisional)' if e['provisional'] else ''}."
        )
    text = header("Models", env, "tessera bench models") + "\n" + "\n\n".join(sections)
    return write("models", {"environment": env, "models": entries}, text, out)


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
