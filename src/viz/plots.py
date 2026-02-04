"""
Generate figures: drift examples, 2D trajectory, F1 vs threshold.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data import UserArchetype


def _select_example_users(ground_truth: List[Dict], n_per: int = 2) -> Dict[str, List[int]]:
    """Select user ids: n_per stable, n_per abrupt, n_per gradual."""
    by_arch: Dict[str, List[int]] = {
        UserArchetype.STABLE.value: [],
        UserArchetype.ABRUPT_CHANGE.value: [],
        UserArchetype.GRADUAL_DRIFT.value: [],
    }
    for g in ground_truth:
        arch = g["archetype"]
        if arch in by_arch and len(by_arch[arch]) < n_per:
            by_arch[arch].append(g["user_id"])
    return by_arch


def plot_drift_examples(
    run_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Drift time series for 6 users (2 stable, 2 abrupt, 2 gradual) with detected change points.
    """
    drift_series = run_data["drift_series"]
    ground_truth = run_data["ground_truth"]
    pred_cps = run_data["pred_change_points"]
    gt_by_uid = {g["user_id"]: g for g in ground_truth}
    selected = _select_example_users(ground_truth, n_per=2)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = axes.flatten()
    idx = 0
    for arch_name, uids in selected.items():
        for uid in uids:
            if uid not in drift_series or idx >= 6:
                continue
            ax = axes[idx]
            dr = drift_series[uid]
            windows = np.arange(1, len(dr) + 1)
            ax.plot(windows, dr, "b-", linewidth=1.5, label="Drift")
            true_cps = gt_by_uid[uid]["true_change_points"]
            for t in true_cps:
                if 1 <= t <= len(dr):
                    ax.axvline(t, color="green", linestyle="--", alpha=0.8, label="True CP" if t == true_cps[0] else None)
            for t in pred_cps.get(uid, []):
                if 1 <= t <= len(dr):
                    ax.axvline(t, color="red", linestyle=":", alpha=0.8, label="Pred CP" if t == (pred_cps[uid][0] if pred_cps[uid] else 0) else None)
            ax.set_title(f"User {uid} ({arch_name})")
            ax.set_xlabel("Window")
            ax.set_ylabel("Drift")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)
            idx += 1
    for j in range(idx, 6):
        axes[j].set_visible(False)
    plt.suptitle("Drift over time with change points (green=true, red=predicted)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trajectory_examples(
    run_data: Dict[str, Any],
    output_path: Path,
    n_users: int = 4,
) -> None:
    """
    2D projection (PCA) of user embedding trajectory μ_{u,t}; color by time, mark predicted change.
    """
    taste_means = run_data["taste_means"]
    ground_truth = run_data["ground_truth"]
    pred_cps = run_data["pred_change_points"]
    gt_by_uid = {g["user_id"]: g for g in ground_truth}
    # Pick 1 stable, 1 abrupt, 1 gradual, 1 cyclical if available
    arch_order = [UserArchetype.STABLE.value, UserArchetype.ABRUPT_CHANGE.value, UserArchetype.GRADUAL_DRIFT.value, UserArchetype.CYCLICAL.value]
    uids: List[int] = []
    for arch in arch_order:
        for g in ground_truth:
            if g["archetype"] == arch and g["user_id"] not in uids:
                uids.append(g["user_id"])
                break
    uids = uids[:n_users]
    if not uids:
        uids = list(taste_means.keys())[:n_users]
    # Stack all trajectories for global PCA
    all_mus = []
    for uid in uids:
        all_mus.append(taste_means[uid])
    X = np.vstack(all_mus)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9))
    axes = axes.flatten()
    offset = 0
    for i, uid in enumerate(uids):
        if i >= len(axes):
            break
        mu_t = taste_means[uid]
        T = len(mu_t)
        pts = X2[offset : offset + T]
        offset += T
        ax = axes[i]
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(T), cmap="viridis", s=40)
        ax.plot(pts[:, 0], pts[:, 1], "k-", alpha=0.4, linewidth=0.8)
        for t in pred_cps.get(uid, []):
            if 0 <= t < T:
                ax.scatter(pts[t, 0], pts[t, 1], s=120, facecolors="none", edgecolors="red", linewidths=2, label="Pred CP" if t == (pred_cps[uid][0] if pred_cps[uid] else -1) else None)
        arch = gt_by_uid.get(uid, {}).get("archetype", "?")
        ax.set_title(f"User {uid} ({arch})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="Time window")
        ax.legend(loc="best", fontsize=7)
        ax.set_aspect("equal")
    plt.suptitle("2D PCA of taste trajectory μ_{u,t}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_f1_vs_threshold(
    results: List[Dict[str, Any]],
    output_path: Path,
    eps_values: List[float] = (0.0, 0.1, 0.2),
) -> None:
    """
    F1 (and optionally precision/recall) vs detection threshold for different ε.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in eps_values:
        subset = [r for r in results if r["exploration_rate"] == eps]
        if not subset:
            continue
        subset = sorted(subset, key=lambda x: x["threshold"])
        th = [r["threshold"] for r in subset]
        f1 = [r["f1"] for r in subset]
        ax.plot(th, f1, "o-", label=f"ε = {eps}", linewidth=2, markersize=6)
    ax.set_xlabel("Detection threshold θ")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs detection threshold (by exploration rate ε)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
