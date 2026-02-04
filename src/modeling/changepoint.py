"""
Change-point detection: threshold and CUSUM-style detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class ChangePointResult:
    """Predicted change points and confidence scores."""
    change_points: List[int]  # 0-indexed window indices where change is detected (at start of new regime)
    scores: List[float]  # confidence/score per detected point
    method: str


def _rolling_stats(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling mean and std; pad with nan or extend edges."""
    if len(arr) == 0:
        return np.array([]), np.array([])
    window = min(window, len(arr))
    mean = np.convolve(arr, np.ones(window) / window, mode="same")
    var = np.convolve(arr ** 2, np.ones(window) / window, mode="same") - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    return mean, std


class ThresholdDetector:
    """
    Flag change at t if drift_t > θ and optionally exceeds rolling baseline by k std.
    Detected index t corresponds to "between window t-1 and t" -> report as change at window t.
    """

    def __init__(
        self,
        threshold: float,
        use_baseline: bool = True,
        baseline_window: int = 3,
        k_std: float = 1.5,
    ) -> None:
        self.threshold = threshold
        self.use_baseline = use_baseline
        self.baseline_window = baseline_window
        self.k_std = k_std

    def detect(self, drift_series: np.ndarray) -> ChangePointResult:
        """
        drift_series: length T-1, drift between consecutive windows.
        Return change points as window indices (1..T-1) where we say "change at start of this window".
        """
        if len(drift_series) == 0:
            return ChangePointResult(change_points=[], scores=[], method="threshold")
        above = drift_series >= self.threshold
        if self.use_baseline and len(drift_series) >= self.baseline_window:
            roll_mean, roll_std = _rolling_stats(drift_series, self.baseline_window)
            above = above & (drift_series >= roll_mean + self.k_std * np.maximum(roll_std, 1e-8))
        indices = np.where(above)[0]
        # Report as "change at window t" meaning between t-1 and t
        change_points = [int(t + 1) for t in indices]
        scores = [float(drift_series[t]) for t in indices]
        return ChangePointResult(change_points=change_points, scores=scores, method="threshold")


class CUSUMDetector:
    """
    CUSUM-style: accumulate positive deviations above a reference; reset on detection.
    Flag when cumulative sum exceeds a threshold.
    """

    def __init__(
        self,
        threshold: float,
        reference: Optional[float] = None,
        drift_scale: Optional[float] = None,
    ) -> None:
        self.threshold = threshold
        self.reference = reference  # if None, use rolling mean
        self.drift_scale = drift_scale or 0.1

    def detect(self, drift_series: np.ndarray) -> ChangePointResult:
        if len(drift_series) == 0:
            return ChangePointResult(change_points=[], scores=[], method="cusum")
        ref = self.reference if self.reference is not None else float(np.median(drift_series))
        # CUSUM on positive excess only: S_t = max(0, S_{t-1} + drift_t - ref)
        S = np.zeros(len(drift_series) + 1)
        for t in range(1, len(S)):
            S[t] = max(0.0, S[t - 1] + drift_series[t - 1] - ref)
        above = S[1:] >= self.threshold
        # Get first time crossing in each "run"
        change_points: List[int] = []
        scores_list: List[float] = []
        i = 0
        while i < len(above):
            if above[i]:
                change_points.append(i + 1)
                scores_list.append(float(S[i + 1]))
                # Skip ahead to avoid duplicate nearby
                while i < len(above) and above[i]:
                    i += 1
            i += 1
        return ChangePointResult(change_points=change_points, scores=scores_list, method="cusum")


def detect_change_points(
    drift_series: np.ndarray,
    method: str = "threshold",
    params: Optional[dict] = None,
) -> ChangePointResult:
    """
    Interface: detect_change_points(drift_series, method="threshold"|"cusum", params={...}).
    params: threshold, and method-specific (use_baseline, k_std, baseline_window for threshold;
            reference, drift_scale for cusum).
    """
    params = params or {}
    threshold = params.get("threshold", 0.2)
    if method == "cusum":
        det = CUSUMDetector(
            threshold=threshold,
            reference=params.get("reference"),
            drift_scale=params.get("drift_scale", 0.1),
        )
    else:
        det = ThresholdDetector(
            threshold=threshold,
            use_baseline=params.get("use_baseline", True),
            baseline_window=params.get("baseline_window", 3),
            k_std=params.get("k_std", 1.5),
        )
    return det.detect(drift_series)
