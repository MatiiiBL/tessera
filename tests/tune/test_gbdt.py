"""The NumPy gradient-boosted trees."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.tune.gbdt import GradientBoostedTrees


def test_learns_a_nonlinear_function_on_held_out_data() -> None:
    rng = np.random.default_rng(0)

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(3 * x[:, 0]) + x[:, 1] ** 2 + 0.5 * (x[:, 2] > 0.3)

    x = rng.uniform(-1, 1, (800, 20))
    y = f(x) + rng.normal(0, 0.05, len(x))
    test = rng.uniform(-1, 1, (500, 20))
    pred = GradientBoostedTrees().fit(x, y).predict(test)
    r2 = 1 - np.mean((pred - f(test)) ** 2) / np.var(f(test))
    assert r2 > 0.9


def test_fits_a_step_exactly_and_is_deterministic() -> None:
    # Fewer distinct values than bins: every gap between them is a candidate split.
    x = np.repeat(np.arange(20, dtype=float), 5)[:, None]
    y = (x[:, 0] >= 7).astype(float)
    a = GradientBoostedTrees(n_trees=60, learning_rate=0.3, subsample=1.0).fit(x, y)
    assert np.allclose(a.predict(x), y, atol=0.02)
    b = GradientBoostedTrees(n_trees=60, learning_rate=0.3, subsample=1.0).fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))


def test_constant_features_and_targets() -> None:
    x = np.ones((10, 3))
    model = GradientBoostedTrees().fit(x, np.arange(10.0))
    assert np.allclose(model.predict(x), 4.5)
    tiny = GradientBoostedTrees().fit(np.zeros((1, 2)), [3.0])
    assert tiny.predict(np.zeros((4, 2))).tolist() == [3.0] * 4


def test_sample_weights_shift_the_fit() -> None:
    x = np.zeros((20, 1))
    x[10:, 0] = 1.0
    y = np.r_[np.zeros(10), np.ones(10)]
    w = np.r_[np.ones(10), np.full(10, 9.0)]
    model = GradientBoostedTrees(subsample=1.0).fit(x, y, sample_weight=w)
    assert model.predict([[1.0]])[0] > 0.9
    assert model.predict([[0.0]])[0] < 0.1


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="features"):
        GradientBoostedTrees().fit(np.zeros((3, 2)), np.zeros(4))
