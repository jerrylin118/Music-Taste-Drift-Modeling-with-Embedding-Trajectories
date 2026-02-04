"""Unit tests for change-point detectors."""

import numpy as np
import pytest

from src.modeling.changepoint import (
    detect_change_points,
    ThresholdDetector,
    CUSUMDetector,
    ChangePointResult,
)


def test_threshold_detector_empty() -> None:
    det = ThresholdDetector(threshold=0.5)
    r = det.detect(np.array([]))
    assert r.change_points == []
    assert r.scores == []


def test_threshold_detector_single_spike() -> None:
    # Drift high only at index 2 (between window 2 and 3 -> report 3)
    drift = np.array([0.0, 0.0, 0.9, 0.0, 0.0])
    det = ThresholdDetector(threshold=0.5, use_baseline=False)
    r = det.detect(drift)
    assert 3 in r.change_points
    assert len(r.scores) == len(r.change_points)


def test_cusum_detector_empty() -> None:
    det = CUSUMDetector(threshold=1.0)
    r = det.detect(np.array([]))
    assert r.change_points == []
    assert r.scores == []


def test_detect_change_points_interface() -> None:
    drift = np.array([0.1, 0.2, 0.8, 0.1, 0.2])
    r = detect_change_points(drift, method="threshold", params={"threshold": 0.5, "use_baseline": False})
    assert isinstance(r, ChangePointResult)
    assert r.method == "threshold"
    r2 = detect_change_points(drift, method="cusum", params={"threshold": 0.3})
    assert r2.method == "cusum"
