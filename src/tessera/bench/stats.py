"""Latency statistics and output agreement, the numbers every benchmark reports.

Latency is the median of at least 50 timed runs after a warmup, with the
10th and 90th percentiles. Agreement compares a model's outputs with a
reference (ONNX Runtime): the largest absolute difference, the largest
difference relative to the reference's largest magnitude, and, for
classifiers, how often both pick the same top-1 class.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from tessera.ir.graph import NDArray

MIN_RUNS = 50


@dataclass(frozen=True)
class Latency:
    """Seconds per run: median, 10th and 90th percentiles over ``runs`` runs."""

    median: float
    p10: float
    p90: float
    runs: int

    @classmethod
    def of(cls, seconds: Sequence[float]) -> Latency:
        if not seconds:
            raise ValueError("no timings")
        p10, median, p90 = np.percentile(np.asarray(seconds, dtype=np.float64), [10, 50, 90])
        return cls(float(median), float(p10), float(p90), len(seconds))

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def ms(self) -> str:
        """``median ms (p10 to p90)`` for tables."""
        return f"{self.median * 1e3:.2f} ({self.p10 * 1e3:.2f} to {self.p90 * 1e3:.2f})"


def time_runs(
    run: Callable[[], object],
    *,
    warmup: int = 10,
    runs: int = MIN_RUNS,
    max_seconds: float | None = None,
) -> Latency:
    """Time ``run()`` (which must wait for its work to finish) ``runs`` times after
    ``warmup`` untimed calls. ``max_seconds`` stops early, but never before
    ``MIN_RUNS`` timed runs."""
    for _ in range(warmup):
        run()
    samples: list[float] = []
    started = time.perf_counter()
    while len(samples) < runs:
        t0 = time.perf_counter()
        run()
        samples.append(time.perf_counter() - t0)
        over = max_seconds is not None and time.perf_counter() - started > max_seconds
        if over and len(samples) >= MIN_RUNS:
            break
    return Latency.of(samples)


@dataclass(frozen=True)
class Agreement:
    """How closely outputs match a reference, over every output tensor."""

    max_abs: float
    max_rel: float  # max |a - b| / max |b|, per output, worst output
    top1: float | None = None  # fraction of samples with the same argmax (classifiers)

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def agreement(
    outputs: Sequence[Mapping[str, NDArray]],
    references: Sequence[Mapping[str, NDArray]],
    *,
    classifier: bool = False,
) -> Agreement:
    """Compare matching lists of output dictionaries (one per input sample).

    For a classifier, the first output's last axis holds the class scores.
    """
    if len(outputs) != len(references) or not outputs:
        raise ValueError("need the same, non-zero number of outputs and references")
    max_abs = max_rel = 0.0
    same = total = 0
    for out, ref in zip(outputs, references, strict=True):
        if set(out) != set(ref):
            raise ValueError(f"output names differ: {sorted(out)} vs {sorted(ref)}")
        for name in ref:
            a = np.asarray(out[name], dtype=np.float64)
            b = np.asarray(ref[name], dtype=np.float64)
            if a.shape != b.shape:
                raise ValueError(f"output {name!r}: shape {a.shape} != {b.shape}")
            diff = float(np.max(np.abs(a - b))) if a.size else 0.0
            scale = float(np.max(np.abs(b))) if b.size else 0.0
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, diff / scale if scale > 0 else diff)
        if classifier:
            first = next(iter(ref))
            a_cls = np.argmax(np.asarray(out[first]).reshape(-1, out[first].shape[-1]), axis=-1)
            b_cls = np.argmax(np.asarray(ref[first]).reshape(-1, ref[first].shape[-1]), axis=-1)
            same += int(np.sum(a_cls == b_cls))
            total += len(b_cls)
    return Agreement(max_abs, max_rel, same / total if classifier and total else None)
