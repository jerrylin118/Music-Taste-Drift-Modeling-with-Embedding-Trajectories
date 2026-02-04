"""Unit tests for drift metrics: zero drift for identical vectors, scale invariance for cosine."""

import numpy as np
import pytest

from src.modeling.drift import (
    cosine_distance,
    euclidean_distance,
    gaussian_kl_proxy,
    compute_drift_series,
)


def test_cosine_identical_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_distance(a, a), 0.0)


def test_cosine_scale_invariance() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = 5.0 * a
    assert np.isclose(cosine_distance(a, b), 0.0)


def test_cosine_orthogonal() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.isclose(cosine_distance(a, b), 1.0)


def test_euclidean_identical_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(euclidean_distance(a, a), 0.0)


def test_euclidean_difference() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert np.isclose(euclidean_distance(a, b), 5.0)


def test_gaussian_kl_identical_zero() -> None:
    mu = np.array([1.0, 2.0])
    var = np.array([1.0, 1.0])
    d = gaussian_kl_proxy(mu, var, mu, var)
    assert d >= 0 and np.isclose(d, 0.0)


def test_compute_drift_series_length() -> None:
    T, d = 10, 4
    taste_means = np.random.randn(T, d)
    dispersions = np.ones(T)
    drifts = compute_drift_series(taste_means, dispersions, metric="cosine")
    assert len(drifts) == T - 1


def test_compute_drift_series_constant_taste_zero_drift() -> None:
    T, d = 5, 4
    mu = np.ones(d)
    taste_means = np.broadcast_to(mu, (T, d)).copy()
    dispersions = np.ones(T)
    drifts = compute_drift_series(taste_means, dispersions, metric="cosine")
    assert np.allclose(drifts, 0.0)
    drifts_euc = compute_drift_series(taste_means, dispersions, metric="euclidean")
    assert np.allclose(drifts_euc, 0.0)
