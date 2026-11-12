"""Validating tuned configurations inside the compiled models.

A configuration is tuned by timing its kernel alone (``Backend.measure``),
and a kernel alone is not always the kernel in its model: weights a model
prepares once at load may be prepared on every call in the timing harness,
caches hold what the previous kernel left, and the thread pool is shared.
So after tuning, each model is compiled twice, with the tuning logs and
with every template's defaults, both are profiled in alternation, and for
every task whose tuned configuration differs from its default the two
in-model times (per kernel, averaged over every kernel of every model that
uses the task) are appended to the log as a validation pass. ``compile``
decides each task by its latest pass, so a configuration that is slower in
the model than the default is not used.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

from tessera.driver import lower
from tessera.ir import random_inputs
from tessera.tune.log import BestConfigs, Record, TuningLog, now
from tessera.tune.space import KnobValue

type Verdict = tuple[float, float]  # seconds per kernel: default, tuned


def validate_in_model(
    models: Sequence[Path],
    target: str,
    log: TuningLog,
    *,
    rounds: int = 3,
    repeat: int = 10,
    progress: Callable[[str], None] = lambda message: None,
) -> dict[str, Verdict]:
    """Profile each model tuned and untuned ``rounds`` times in alternation and log,
    per task whose tuned configuration differs from its default, the in-model time
    per kernel of both; returns them by task key."""
    best = BestConfigs(log.records())
    seconds: dict[str, tuple[list[float], list[float]]] = {}  # task -> default, tuned
    configs: dict[str, tuple[dict[str, KnobValue], dict[str, KnobValue]]] = {}
    target_key = ""
    for model in models:
        tuned = lower(model, target=target, tuning_log=best)
        untuned = lower(model, target=target, tuned=False)
        target_key = tuned.target.key
        pairs = [
            (u, t)
            for u, t in zip(untuned.kernels, tuned.kernels, strict=True)
            if t.tuned and t.task_key is not None and t.config != u.config
        ]
        if not pairs:
            continue
        progress(f"{model.stem}: {len(pairs)} kernels with tuned configurations")
        built_u, built_t = untuned.build(), tuned.build()
        inputs = random_inputs(tuned.graph)
        samples: dict[str, list[float]] = {}
        for _ in range(rounds):
            for prefix, built in (("u", built_u), ("t", built_t)):
                for k in built.profile(inputs, repeat=repeat):
                    samples.setdefault(f"{prefix}:{k.kernel}", []).append(k.seconds)
        for u, t in pairs:
            key = t.task_key
            if key is None or u.config is None or t.config is None:
                continue  # filtered above; this narrows the types
            default_s, tuned_s = seconds.setdefault(key, ([], []))
            default_s.append(min(samples[f"u:{u.name}"]))
            tuned_s.append(min(samples[f"t:{t.name}"]))
            configs[key] = (u.config, t.config)
    label = f"{now()} in the model"
    verdicts: dict[str, Verdict] = {}
    for key, (default_s, tuned_s) in seconds.items():
        default_config, tuned_config = configs[key]
        verdict = (statistics.fmean(default_s), statistics.fmean(tuned_s))
        verdicts[key] = verdict
        for config, value in zip((default_config, tuned_config), verdict, strict=True):
            log.append(Record(target_key, key, config, value, None, validation=label))
    return verdicts
