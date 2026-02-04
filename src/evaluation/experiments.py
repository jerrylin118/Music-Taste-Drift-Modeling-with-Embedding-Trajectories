"""
Experiment runner: parameter sweeps (epsilon, threshold) and metric collection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import Config
from src.data import SongSpace, simulate_users, UserArchetype
from src.modeling import (
    compute_taste_trajectories,
    compute_drift_series,
    detect_change_points,
)
from src.evaluation.metrics import evaluate_detection, aggregate_metrics


def run_single_config(
    config: Config,
    exploration_rate: float,
    threshold: float,
    drift_metric: str = "cosine",
    detection_method: str = "threshold",
) -> Dict[str, Any]:
    """Generate data, compute taste, drift, detect, evaluate for one (epsilon, theta)."""
    song_space = SongSpace(
        embedding_dim=config.embedding_dim,
        n_clusters=config.n_clusters,
        cluster_std=config.cluster_std,
        n_songs_per_cluster=config.n_songs_per_cluster,
        seed=config.seed,
    ).build_synthetic()

    listens_df, ground_truth = simulate_users(
        song_space,
        n_users=config.n_users,
        n_windows=config.n_windows,
        listens_per_window=config.listens_per_window,
        exploration_rate=exploration_rate,
        archetype_weights=config.archetype_weights,
        abrupt_change_window=config.abrupt_change_window,
        gradual_start_cluster=config.gradual_start_cluster,
        gradual_end_cluster=config.gradual_end_cluster,
        cyclical_period=config.cyclical_period,
        seed=config.seed,
    )

    taste_means, dispersions, _ = compute_taste_trajectories(listens_df, song_space)
    drift_series_by_user: Dict[int, np.ndarray] = {}
    for uid in taste_means:
        drift_series_by_user[uid] = compute_drift_series(
            taste_means[uid], dispersions[uid], metric=drift_metric
        )

    stable_archetype = UserArchetype.STABLE.value
    gt_by_uid = {g["user_id"]: g for g in ground_truth}
    per_user_metrics: List[Dict[str, float]] = []
    stable_mask: List[bool] = []
    for uid in sorted(taste_means.keys()):
        gt = gt_by_uid[uid]
        true_cps = gt["true_change_points"]
        stable_mask.append(gt["archetype"] == stable_archetype)
        pred_result = detect_change_points(
            drift_series_by_user[uid],
            method=detection_method,
            params={"threshold": threshold, "use_baseline": True, "k_std": 1.5},
        )
        m = evaluate_detection(
            pred_result.change_points,
            true_cps,
            tolerance_windows=config.tolerance_windows,
        )
        m["user_id"] = uid
        m["archetype"] = gt["archetype"]
        per_user_metrics.append(m)

    agg = aggregate_metrics(per_user_metrics, stable_mask)
    agg["exploration_rate"] = exploration_rate
    agg["threshold"] = threshold
    agg["drift_metric"] = drift_metric
    agg["detection_method"] = detection_method
    pred_cps_by_user = {
        uid: detect_change_points(
            drift_series_by_user[uid],
            method=detection_method,
            params={"threshold": threshold, "use_baseline": True, "k_std": 1.5},
        ).change_points
        for uid in taste_means
    }
    return {
        "aggregate": agg,
        "per_user": per_user_metrics,
        "ground_truth": ground_truth,
        "taste_means": taste_means,
        "dispersions": dispersions,
        "drift_series": drift_series_by_user,
        "pred_change_points": pred_cps_by_user,
        "listens_df": listens_df,
        "song_space": song_space,
    }


def run_parameter_sweep(config: Config) -> tuple:
    """
    Sweep exploration_rate and threshold; collect metrics.
    Returns (results_list, full_run_data for first config for plotting).
    """
    exploration_rates = [0.0, 0.1, 0.2, 0.3]
    thresholds = config.threshold_grid
    results: List[Dict[str, Any]] = []
    first_run: Dict[str, Any] = {}
    for eps in exploration_rates:
        for th in thresholds:
            run = run_single_config(
                config, exploration_rate=eps, threshold=th,
                drift_metric="cosine", detection_method="threshold",
            )
            results.append(run["aggregate"])
            if not first_run:
                first_run = run
    return results, first_run


def save_metrics_and_summary(
    results: List[Dict[str, Any]],
    metrics_dir: Path,
) -> None:
    """Save report.json and summary.csv."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "report.json", "w") as f:
        json.dump(results, f, indent=2)
    rows = []
    for r in results:
        rows.append({
            "exploration_rate": r["exploration_rate"],
            "threshold": r["threshold"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "fp_rate_stable": r["fp_rate_stable"],
            "n_change_users": r["n_change_users"],
            "n_stable_users": r["n_stable_users"],
        })
    pd.DataFrame(rows).to_csv(metrics_dir / "summary.csv", index=False)
