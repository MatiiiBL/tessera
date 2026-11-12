"""``tessera bench models``: the four benchmark models against the baselines.

For every model and system (ONNX Runtime on the CPU, PyTorch eager on the
CPU and on MPS, Tessera on each requested target, tuned and untuned) this
measures latency (median of at least 50 runs after a warmup, with the 10th
and 90th percentiles) on the model's fixed input, and agreement with ONNX
Runtime: the largest absolute and relative difference of the outputs and,
for the image models, top-1 agreement over ``CLASSIFIER_SAMPLES`` inputs
(the fixed one and seeded random ones).

The ablations compile each model again with one part of the pipeline
turned off and report what changes: fusion (kernel count, bytes moved,
latency), the graph rewrites (``optimize=False``: no folding, no pattern
rewrites, BatchNorm decomposed instead of folded), memory planning (the
arena against the sum of the intermediate tensors it holds) and tuning
(template defaults against the tuning logs).

"Bytes moved" is the sum, over kernel launches, of the bytes of every
argument: what the kernels would read and write if each touched its
arguments once. It is a lower bound of memory traffic that does not depend
on the hardware, and fusion lowers it by keeping intermediates out of
memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from tessera.bench.baselines import Runner, onnxruntime_cpu, torch_available, torch_eager
from tessera.bench.report import markdown_table
from tessera.bench.stats import Agreement, Latency, agreement, time_runs
from tessera.bench.zoo import ModelSpec
from tessera.driver import Compilation, lower
from tessera.ir.graph import NDArray
from tessera.plan import StorageKind

CLASSIFIER_SAMPLES = 16
WARMUP = 10
RUNS = 50


@dataclass
class SystemResult:
    """One system on one model."""

    system: str
    latency: Latency | None = None
    agreement: Agreement | None = None
    threads: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "latency": self.latency.as_dict() if self.latency else None,
            "agreement": self.agreement.as_dict() if self.agreement else None,
            "threads": self.threads,
            "error": self.error,
            **self.details,
        }


@dataclass
class ModelResult:
    model: str
    systems: list[SystemResult]
    ablations: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "systems": [s.as_dict() for s in self.systems],
            "ablations": self.ablations,
        }


def samples(spec: ModelSpec) -> list[dict[str, NDArray]]:
    """The inputs agreement is measured on: the fixed one, plus random ones for classifiers."""
    fixed = spec.sample_inputs()
    if not spec.classifier:
        return [fixed]
    return [fixed, *(spec.random_inputs(seed) for seed in range(1, CLASSIFIER_SAMPLES))]


def measure_runner(
    runner: Runner,
    spec: ModelSpec,
    inputs: Sequence[Mapping[str, NDArray]],
    reference: list[dict[str, NDArray]],
) -> SystemResult:
    """Latency on the first input and agreement with ``reference`` on all of them."""
    outputs = [runner.outputs(x) for x in inputs]
    latency = time_runs(runner.prepare(inputs[0]), warmup=WARMUP, runs=RUNS)
    return SystemResult(
        runner.name,
        latency,
        agreement(outputs, reference, classifier=spec.classifier),
        runner.threads,
    )


def measure_compilation(
    name: str,
    comp: Compilation,
    spec: ModelSpec,
    inputs: Sequence[Mapping[str, NDArray]],
    reference: list[dict[str, NDArray]],
) -> SystemResult:
    """Build ``comp``, then time it and compare its outputs like a baseline."""
    compiled = comp.build()
    outputs = [compiled.run(x) for x in inputs]
    seconds = compiled.benchmark(inputs[0], warmup=WARMUP, repeat=RUNS)
    return SystemResult(
        name,
        Latency.of(seconds),
        agreement(outputs, reference, classifier=spec.classifier),
        comp.target.threads if comp.target.kind == "c" else None,
        details=plan_stats(comp),
    )


def plan_stats(comp: Compilation) -> dict[str, Any]:
    """Kernel count, bytes moved and memory of a compilation (see the module docstring)."""
    moved = sum(ref.nbytes for launch in comp.plan.launches for ref in launch.args)
    held = {
        name: ref.nbytes for name, ref in comp.memory.refs.items() if ref.kind is StorageKind.ARENA
    }
    return {
        "kernels": len(comp.plan.launches),
        "tuned_kernels": comp.tuned_kernels,
        "bytes_moved": moved,
        "arena_bytes": comp.memory.arena_bytes,
        "intermediate_bytes": sum(held.values()),
        "constant_bytes": int(comp.memory.constants.nbytes),
    }


def bench_model(
    spec: ModelSpec,
    *,
    targets: Sequence[str],
    baselines: Sequence[str] = ("onnxruntime-cpu", "pytorch-cpu", "pytorch-mps"),
    ablations: bool = True,
    progress: Callable[[str], None] = lambda message: None,
) -> ModelResult:
    """Every system on one model, then the ablations on each target."""
    inputs = samples(spec)
    reference_runner = onnxruntime_cpu(spec.onnx_path())
    reference = [reference_runner.outputs(x) for x in inputs]
    results: list[SystemResult] = []

    def attempt(name: str, fn: Callable[[], SystemResult]) -> SystemResult:
        progress(f"{spec.name}: {name}")
        try:
            result = fn()
        except Exception as e:  # one failing system must not lose the others' numbers
            result = SystemResult(name, error=f"{type(e).__name__}: {e}")
        results.append(result)
        return result

    def baseline(name: str) -> SystemResult:
        runner = reference_runner
        if name.startswith("pytorch-"):
            runner = torch_eager(spec, name.removeprefix("pytorch-"))
        return measure_runner(runner, spec, inputs, reference)

    def tessera(label: str, target: str, tuned: bool) -> SystemResult:
        comp = lower(spec.onnx_path(), target=target, input_shapes=spec.input_shapes, tuned=tuned)
        return measure_compilation(label, comp, spec, inputs, reference)

    for name in baselines:
        if not name.startswith("pytorch-") or torch_available(name.removeprefix("pytorch-")):
            attempt(name, partial(baseline, name))

    found: dict[str, Any] = {}
    for target in targets:
        for tuned in (True, False):
            label = f"tessera-{target}{'' if tuned else '-untuned'}"
            attempt(label, partial(tessera, label, target, tuned))
        if ablations:
            found[target] = ablate(spec, target, inputs[:1], reference[:1], progress)
    return ModelResult(spec.name, results, found)


def ablate(
    spec: ModelSpec,
    target: str,
    inputs: Sequence[Mapping[str, NDArray]],
    reference: list[dict[str, NDArray]],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Fusion off and graph rewrites off, against the full untuned pipeline."""
    shapes = spec.input_shapes
    variants = {  # name: (fuse, optimize)
        "full": (True, True),
        "fusion off": (False, True),
        "graph rewrites off": (True, False),
    }
    out: dict[str, Any] = {}
    for name, (fuse, optimize) in variants.items():
        progress(f"{spec.name}: {target} ablation, {name}")
        try:
            comp = lower(
                spec.onnx_path(),
                target=target,
                input_shapes=shapes,
                tuned=False,
                fuse=fuse,
                optimize=optimize,
            )
            out[name] = measure_compilation(name, comp, spec, inputs, reference).as_dict()
        except Exception as e:  # report the variant as failed, keep the others
            out[name] = {"system": name, "error": f"{type(e).__name__}: {e}"}
    return out


def markdown(results: Sequence[ModelResult]) -> str:
    """The tables of ``results/models.md`` (without the provenance header)."""
    parts: list[str] = []
    for r in results:
        rows: list[list[object]] = []
        base = next(
            (s.latency.median for s in r.systems if s.system == "onnxruntime-cpu" and s.latency),
            None,
        )
        for s in r.systems:
            if s.latency is None:
                rows.append([s.system, "failed", "", "", "", "", (s.error or "")[:80]])
                continue
            a = s.agreement
            speedup = f"{base / s.latency.median:.2f}x" if base else ""
            rows.append(
                [
                    s.system,
                    s.latency.ms(),
                    speedup,
                    f"{a.max_abs:.2e}" if a else "",
                    f"{a.max_rel:.2e}" if a else "",
                    f"{a.top1:.0%}" if a and a.top1 is not None else "",
                    _notes(s),
                ]
            )
        parts.append(
            f"### {r.model}\n\n"
            + markdown_table(
                [
                    "system",
                    "ms, median (p10 to p90)",
                    "vs ONNX Runtime",
                    "max abs err",
                    "max rel err",
                    "top-1",
                    "notes",
                ],
                rows,
            )
        )
        if r.ablations:
            parts.append(_ablation_table(r))
    return "\n\n".join(parts)


def _notes(s: SystemResult) -> str:
    notes = []
    if s.threads:
        notes.append(f"{s.threads} threads")
    if "kernels" in s.details:
        notes.append(f"{s.details['kernels']} kernels")
        if s.details.get("tuned_kernels"):
            notes.append(f"{s.details['tuned_kernels']} tuned")
    return ", ".join(notes)


def _ablation_table(r: ModelResult) -> str:
    rows: list[list[object]] = []
    for target, variants in r.ablations.items():
        for name, v in variants.items():
            if v.get("error"):
                rows.append([target, name, "failed", "", "", "", v["error"][:80]])
                continue
            lat = v["latency"]
            rows.append(
                [
                    target,
                    name,
                    v["kernels"],
                    f"{v['bytes_moved'] / 1e6:.1f}",
                    f"{v['arena_bytes'] / 1e6:.1f}",
                    f"{v['intermediate_bytes'] / 1e6:.1f}",
                    f"{lat['median'] * 1e3:.2f}",
                ]
            )
    return f"Ablations of {r.model} (untuned):\n\n" + markdown_table(
        [
            "target",
            "pipeline",
            "kernels",
            "bytes moved (MB)",
            "arena (MB)",
            "intermediates (MB)",
            "ms (median)",
        ],
        rows,
    )
