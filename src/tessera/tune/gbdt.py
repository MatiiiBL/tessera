"""Gradient-boosted regression trees in NumPy.

A small implementation of histogram-based gradient boosting with squared
loss, enough for the tuner's cost model: a few thousand rows and a few
hundred features, refitted after every measurement round.

Each feature is cut into at most ``max_bins`` bins at quantiles of the
training data. A tree is grown depth-first; at each node the gradient sums
of every (feature, bin) pair come from one ``bincount``, and the best split
maximizes the usual second-order gain ``GL^2/(HL+l2) + GR^2/(HR+l2) - G^2/(H+l2)``.
The smaller child's histogram is built directly and the sibling's is the
parent's minus it. Trees are stored as flat arrays, so prediction walks all
rows down a tree together, one level at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


@dataclass
class _Tree:
    feature: IntArray  # -1 at leaves
    threshold: FloatArray  # go left when x <= threshold
    left: IntArray
    right: IntArray
    value: FloatArray

    def predict(self, x: FloatArray) -> FloatArray:
        node = np.zeros(len(x), dtype=np.int64)
        rows = np.arange(len(x))
        while True:
            feat = self.feature[node]
            inner = feat >= 0
            if not inner.any():
                return self.value[node]
            f = np.where(inner, feat, 0)
            go_left = x[rows, f] <= self.threshold[node]
            node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)


@dataclass
class GradientBoostedTrees:
    """Least-squares gradient boosting on binned features."""

    n_trees: int = 120
    learning_rate: float = 0.12
    max_depth: int = 6
    min_samples_leaf: int = 3
    l2: float = 1.0
    max_bins: int = 32
    subsample: float = 0.8
    seed: int = 0
    _trees: list[_Tree] = field(default_factory=list, init=False, repr=False)
    _base: float = field(default=0.0, init=False, repr=False)
    _columns: IntArray = field(default_factory=lambda: np.zeros(0, np.int64), init=False)

    @property
    def fitted(self) -> bool:
        return bool(self._trees) or self._base != 0.0

    def fit(
        self, x: npt.ArrayLike, y: npt.ArrayLike, sample_weight: npt.ArrayLike | None = None
    ) -> GradientBoostedTrees:
        """Fit to rows ``x`` (n by features) and targets ``y``; replaces any earlier fit."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if xa.ndim != 2 or len(xa) != len(ya):
            raise ValueError(f"x must be (n, features) and y (n,), got {xa.shape}, {ya.shape}")
        wa = (
            np.ones(len(ya))
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        rng = np.random.default_rng(self.seed)
        self._trees = []
        self._base = float(np.average(ya, weights=wa)) if len(ya) else 0.0
        if len(ya) < 2:
            return self
        # Constant columns can never split; dropping them makes histograms cheaper.
        varying = np.flatnonzero(xa.max(axis=0) > xa.min(axis=0))
        self._columns = varying.astype(np.int64)
        xs = xa[:, varying]
        if xs.shape[1] == 0:
            return self
        edges = [_edges(xs[:, j], self.max_bins) for j in range(xs.shape[1])]
        binned = np.stack(
            [np.searchsorted(e, xs[:, j], side="left") for j, e in enumerate(edges)], axis=1
        ).astype(np.int64)
        pred = np.full(len(ya), self._base)
        for _ in range(self.n_trees):
            rows = np.arange(len(ya))
            if self.subsample < 1.0:
                take = max(2 * self.min_samples_leaf, int(self.subsample * len(ya)))
                rows = np.sort(rng.choice(len(ya), size=min(take, len(ya)), replace=False))
            residual = (ya - pred) * wa
            tree = _grow(binned, edges, residual, wa, rows, self)
            self._trees.append(tree)
            pred += self.learning_rate * tree.predict(xs)
        return self

    def predict(self, x: npt.ArrayLike) -> FloatArray:
        """Predictions for rows ``x`` (with the feature layout used in ``fit``)."""
        xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
        out = np.full(len(xa), self._base)
        if not self._trees:
            return out
        xs = xa[:, self._columns]
        for tree in self._trees:
            out += self.learning_rate * tree.predict(xs)
        return out


def _edges(column: FloatArray, max_bins: int) -> FloatArray:
    """Split thresholds: midpoints between consecutive distinct quantiles."""
    values = np.unique(column)
    if len(values) > max_bins:
        values = np.unique(np.quantile(column, np.linspace(0.0, 1.0, max_bins + 1)))
    return (values[:-1] + values[1:]) / 2.0


def _grow(
    binned: IntArray,
    edges: list[FloatArray],
    residual: FloatArray,
    weight: FloatArray,
    rows: IntArray,
    params: GradientBoostedTrees,
) -> _Tree:
    n_features = binned.shape[1]
    n_bins = params.max_bins + 1
    codes = binned + np.arange(n_features, dtype=np.int64) * n_bins
    size = n_features * n_bins
    # A split after bin b sends bins <= b left; it exists only if the feature has edge b.
    splittable = np.arange(n_bins - 1)[None, :] < np.array([len(e) for e in edges])[:, None]

    def histogram(idx: IntArray) -> FloatArray:
        """Residual sums, weight sums and counts per (feature, bin): (3, features, bins)."""
        flat = codes[idx].ravel()
        g = np.bincount(flat, np.repeat(residual[idx], n_features), size)
        h = np.bincount(flat, np.repeat(weight[idx], n_features), size)
        n = np.bincount(flat, minlength=size).astype(np.float64)
        return np.stack([g, h, n]).reshape(3, n_features, n_bins)

    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []

    def add(f: int, t: float, v: float) -> int:
        feature.append(f)
        threshold.append(t)
        left.append(-1)
        right.append(-1)
        value.append(v)
        return len(feature) - 1

    def node(idx: IntArray, hist: FloatArray, depth: int) -> int:
        g_sum, h_sum = float(hist[0, 0].sum()), float(hist[1, 0].sum())
        if depth >= params.max_depth or len(idx) < 2 * params.min_samples_leaf:
            return add(-1, 0.0, g_sum / (h_sum + params.l2))
        gl, hl, nl = np.cumsum(hist, axis=2)[:, :, :-1]
        gr, hr, nr = g_sum - gl, h_sum - hl, len(idx) - nl
        lam = params.l2
        gain = gl**2 / (hl + lam) + gr**2 / (hr + lam) - g_sum**2 / (h_sum + lam)
        valid = splittable & (nl >= params.min_samples_leaf) & (nr >= params.min_samples_leaf)
        gain = np.where(valid, gain, -np.inf)
        f, b = divmod(int(np.argmax(gain)), n_bins - 1)
        if not gain[f, b] > 1e-12:
            return add(-1, 0.0, g_sum / (h_sum + lam))
        goes_left = binned[idx, f] <= b
        li, ri = idx[goes_left], idx[~goes_left]
        # Build the smaller child's histogram; the sibling's is the difference.
        if len(li) <= len(ri):
            lh = histogram(li)
            rh = hist - lh
        else:
            rh = histogram(ri)
            lh = hist - rh
        me = add(f, float(edges[f][b]), 0.0)
        left[me] = node(li, lh, depth + 1)
        right[me] = node(ri, rh, depth + 1)
        return me

    node(rows, histogram(rows), 0)
    return _Tree(
        np.asarray(feature, dtype=np.int64),
        np.asarray(threshold, dtype=np.float64),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(value, dtype=np.float64),
    )
