"""Modeling: taste representation, drift metrics, change-point detection."""

from .taste import compute_taste_trajectories
from .drift import compute_drift_series, cosine_distance, euclidean_distance, gaussian_kl_proxy
from .changepoint import detect_change_points, ThresholdDetector, CUSUMDetector

__all__ = [
    "compute_taste_trajectories",
    "compute_drift_series",
    "cosine_distance",
    "euclidean_distance",
    "gaussian_kl_proxy",
    "detect_change_points",
    "ThresholdDetector",
    "CUSUMDetector",
]
