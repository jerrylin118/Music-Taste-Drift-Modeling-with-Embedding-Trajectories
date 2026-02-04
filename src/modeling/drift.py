"""
Drift metrics between consecutive windows: cosine, Euclidean, Gaussian KL proxy.

Definitions:
  - Cosine distance: 1 - cos(μ_t, μ_{t-1}) = 1 - (μ_t · μ_{t-1}) / (||μ_t|| ||μ_{t-1}||)
  - Euclidean distance: ||μ_t - μ_{t-1}||_2
  - Gaussian proxy: treat window as N(μ, diag(σ²)); symmetric KL proxy between consecutive.
"""

from typing import Dict, Literal, Union

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity; 0 for identical direction, scale-invariant."""
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    cos = float(np.dot(a, b) / (an * bn))
    cos = np.clip(cos, -1.0, 1.0)
    return float(1.0 - cos)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_2."""
    return float(np.linalg.norm(a - b))


def _diagonal_var(embeddings: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Diagonal variance (per dimension). Numerically stable: add small epsilon."""
    if len(embeddings) <= 1:
        return np.ones_like(mean) * 1e-8
    var = np.var(embeddings, axis=0)
    return np.maximum(var, 1e-10)


def gaussian_kl_proxy(
    mean_a: np.ndarray,
    var_a: np.ndarray,
    mean_b: np.ndarray,
    var_b: np.ndarray,
) -> float:
    """
    Symmetric KL proxy for two Gaussians N(μ_a, diag(σ_a²)), N(μ_b, diag(σ_b²)).
    KL(A||B) = 0.5 * ( sum(σ_b/σ_a + σ_a/σ_b - 2) + (μ_a-μ_b)'(1/σ_b²)(μ_a-μ_b) ) for diagonal.
    We use 0.5*(KL(A||B)+KL(B||A)) and keep it stable (no log of zero).
    """
    eps = 1e-10
    va = np.maximum(var_a, eps)
    vb = np.maximum(var_b, eps)
    # Symmetric KL for diagonal Gaussians
    term1 = np.sum(va / vb + vb / va - 2) * 0.5
    term2 = np.sum((mean_a - mean_b) ** 2 / vb)
    term3 = np.sum((mean_a - mean_b) ** 2 / va)
    return float(0.25 * (term1 + term2 + term3))


def compute_drift_series(
    taste_means: np.ndarray,
    dispersions: np.ndarray,
    metric: Literal["cosine", "euclidean", "gaussian_kl"] = "cosine",
) -> np.ndarray:
    """
    Compute drift_t for t = 1..T-1 (between window t-1 and t).
    taste_means: (T, d), dispersions: (T,).
    For gaussian_kl we need per-window variances: use dispersion^2 as proxy for avg variance.
    """
    T = taste_means.shape[0]
    if T < 2:
        return np.array([])
    drifts = np.zeros(T - 1)
    for t in range(1, T):
        mu_prev = taste_means[t - 1]
        mu_curr = taste_means[t]
        if metric == "cosine":
            drifts[t - 1] = cosine_distance(mu_prev, mu_curr)
        elif metric == "euclidean":
            drifts[t - 1] = euclidean_distance(mu_prev, mu_curr)
        elif metric == "gaussian_kl":
            # Proxy: diagonal variance from dispersion (same for all dims as scalar proxy)
            d = taste_means.shape[1]
            var_prev = np.full(d, dispersions[t - 1] ** 2 + 1e-10)
            var_curr = np.full(d, dispersions[t] ** 2 + 1e-10)
            drifts[t - 1] = gaussian_kl_proxy(mu_prev, var_prev, mu_curr, var_curr)
        else:
            drifts[t - 1] = cosine_distance(mu_prev, mu_curr)
    return drifts


def compute_all_drift_series(
    taste_means: Dict[int, np.ndarray],
    dispersions: Dict[int, np.ndarray],
    metric: Literal["cosine", "euclidean", "gaussian_kl"] = "cosine",
) -> Dict[int, np.ndarray]:
    """Compute drift series for every user."""
    return {
        uid: compute_drift_series(taste_means[uid], dispersions[uid], metric)
        for uid in taste_means
    }
