"""
Evaluation against ground truth: precision, recall, F1, false positive rate.
Tolerance: ±tolerance_windows for matching predicted to true change points.
"""

from typing import Any, Dict, List

import numpy as np


def _match_with_tolerance(
    predicted: List[int],
    true: List[int],
    tolerance: int,
) -> tuple:
    """Count TP: predicted point matches some true within ±tolerance. FP, FN derived."""
    pred_set = set(predicted)
    true_set = set(true)
    matched_true: set = set()
    matched_pred: set = set()
    for p in pred_set:
        for tau in range(p - tolerance, p + tolerance + 1):
            if tau in true_set:
                matched_true.add(tau)
                matched_pred.add(p)
                break
    tp = len(matched_pred)
    fp = len(pred_set - matched_pred)
    fn = len(true_set - matched_true)
    return tp, fp, fn


def evaluate_detection(
    predicted_change_points: List[int],
    true_change_points: List[int],
    tolerance_windows: int = 1,
) -> Dict[str, float]:
    """
    Per-user evaluation. Returns precision, recall, F1, and indicators.
    For stable users (no true change points), we only count FP rate (FP count).
    """
    tp, fp, fn = _match_with_tolerance(
        predicted_change_points, true_change_points, tolerance_windows
    )
    n_true = len(true_change_points)
    n_pred = len(predicted_change_points)
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_true": n_true,
        "n_pred": n_pred,
    }


def aggregate_metrics(
    per_user_metrics: List[Dict[str, float]],
    stable_user_mask: List[bool],
) -> Dict[str, Any]:
    """
    Aggregate over users. Separate stable (no true CPs) for FP rate.
    """
    if not per_user_metrics:
        return {}
    # Users with at least one true change point
    change_users = [m for m, s in zip(per_user_metrics, stable_user_mask) if not s]
    stable_users = [m for m, s in zip(per_user_metrics, stable_user_mask) if s]
    agg: Dict[str, Any] = {}
    if change_users:
        agg["precision"] = float(np.mean([m["precision"] for m in change_users]))
        agg["recall"] = float(np.mean([m["recall"] for m in change_users]))
        agg["f1"] = float(np.mean([m["f1"] for m in change_users]))
        agg["n_change_users"] = len(change_users)
    else:
        agg["precision"] = agg["recall"] = agg["f1"] = 0.0
        agg["n_change_users"] = 0
    # False positive rate on stable users: fraction of stable users with at least one FP
    if stable_users:
        n_stable_with_fp = sum(1 for m in stable_users if m["fp"] > 0)
        agg["fp_rate_stable"] = n_stable_with_fp / len(stable_users)
        agg["n_stable_users"] = len(stable_users)
    else:
        agg["fp_rate_stable"] = 0.0
        agg["n_stable_users"] = 0
    return agg
