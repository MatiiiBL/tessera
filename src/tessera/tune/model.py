"""The learned cost model: scores scheduled loop nests from their features.

The model is trained online, during the search, on every measurement made
so far, across all tasks. Times of different tasks are not comparable (a
large convolution is slower than a small one whatever its schedule), so the
target is each measurement's throughput relative to the best measurement of
its task: ``best_seconds / seconds``, 1 for the best configuration so far,
near 0 for slow ones, and 0 for configurations that failed. This scale
spends the model's capacity on telling good configurations apart, which is
what the search needs, and ignores how bad the bad ones are.

The model only has to rank candidates of one task against each other; it
never predicts seconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from tessera.tune.gbdt import GradientBoostedTrees

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Sample:
    """One measurement: the task, the features of its scheduled function, its time."""

    task: str
    features: FloatArray
    seconds: float | None  # None: the candidate failed to build or run


class CostModel(Protocol):
    """Ranks candidates: a higher score means a faster function."""

    @property
    def trained(self) -> bool: ...

    def fit(self, samples: Sequence[Sample]) -> None:
        """Retrain on every sample (replaces the previous fit)."""
        ...

    def predict(self, features: FloatArray) -> FloatArray:
        """Scores of rows of features."""
        ...


def relative_throughput(samples: Sequence[Sample]) -> FloatArray:
    """Each sample's ``best / seconds`` within its task (0 for failures)."""
    best: dict[str, float] = {}
    for s in samples:
        if s.seconds is not None and s.seconds > 0:
            best[s.task] = min(best.get(s.task, s.seconds), s.seconds)
    return np.array(
        [
            best[s.task] / s.seconds if s.seconds is not None and s.seconds > 0 else 0.0
            for s in samples
        ]
    )


class TreeCostModel:
    """Gradient-boosted trees on relative throughput (see the module docstring)."""

    def __init__(self, *, min_samples: int = 8, seed: int = 0, n_trees: int = 120) -> None:
        self.min_samples = min_samples
        self._trees = GradientBoostedTrees(n_trees=n_trees, seed=seed)
        self._trained = False

    @property
    def trained(self) -> bool:
        return self._trained

    def fit(self, samples: Sequence[Sample]) -> None:
        if len(samples) < self.min_samples:
            self._trained = False
            return
        y = relative_throughput(samples)
        if np.ptp(y) == 0.0:
            self._trained = False
            return
        x = np.stack([s.features for s in samples])
        self._trees.fit(x, y)
        self._trained = True

    def predict(self, features: FloatArray) -> FloatArray:
        if not self._trained:
            return np.zeros(len(features))
        return self._trees.predict(features)
