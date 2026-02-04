"""User behavior simulation with ground-truth archetypes and change points."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .songspace import SongSpace


class UserArchetype(str, Enum):
    STABLE = "stable"
    GRADUAL_DRIFT = "gradual_drift"
    ABRUPT_CHANGE = "abrupt_change"
    CYCLICAL = "cyclical"


def _mixture_for_user(
    archetype: UserArchetype,
    window: int,
    n_windows: int,
    n_clusters: int,
    abrupt_window: int,
    start_cluster: int,
    end_cluster: int,
    period: int,
) -> np.ndarray:
    """Return mixture weights (n_clusters,) for given archetype and window."""
    w = np.zeros(n_clusters)
    if archetype == UserArchetype.STABLE:
        w[start_cluster] = 1.0
        return w
    if archetype == UserArchetype.GRADUAL_DRIFT:
        alpha = window / max(1, n_windows - 1)
        w[start_cluster] = 1.0 - alpha
        w[end_cluster] = alpha
        return w
    if archetype == UserArchetype.ABRUPT_CHANGE:
        if window < abrupt_window:
            w[start_cluster] = 1.0
        else:
            w[end_cluster] = 1.0
        return w
    if archetype == UserArchetype.CYCLICAL:
        phase = (window % period) / max(1, period)
        w[start_cluster] = 1.0 - phase
        w[end_cluster] = phase
        return w
    w[0] = 1.0
    return w


def simulate_users(
    song_space: SongSpace,
    n_users: int = 200,
    n_windows: int = 20,
    listens_per_window: int = 50,
    exploration_rate: float = 0.1,
    archetype_weights: Optional[List[float]] = None,
    abrupt_change_window: int = 10,
    gradual_start_cluster: int = 0,
    gradual_end_cluster: int = 4,
    cyclical_period: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Simulate N users over T windows. Return tidy dataframe and ground-truth list.
    DataFrame columns: user_id, window, song_id, cluster_id.
    Ground truth per user: archetype, true_change_points (list), drift_type (str).
    """
    rng = np.random.default_rng(seed)
    archetypes = list(UserArchetype)
    if archetype_weights is None:
        archetype_weights = [0.25] * 4
    archetype_weights = np.array(archetype_weights, dtype=float)
    archetype_weights = archetype_weights / archetype_weights.sum()
    n_clusters = song_space.n_clusters

    rows: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    for u in range(n_users):
        idx = rng.choice(len(archetypes), p=archetype_weights)
        arch = archetypes[idx]
        true_cps: List[int] = []
        if arch == UserArchetype.ABRUPT_CHANGE:
            true_cps = [abrupt_change_window]
        if arch == UserArchetype.GRADUAL_DRIFT:
            # "Change region" for evaluation: from start to end of drift
            true_cps = [0, n_windows - 1]
        if arch == UserArchetype.CYCLICAL:
            true_cps = [p for p in range(cyclical_period, n_windows, cyclical_period)]

        ground_truth.append({
            "user_id": u,
            "archetype": arch.value,
            "true_change_points": true_cps,
            "drift_type": arch.value,
        })

        for t in range(n_windows):
            mix = _mixture_for_user(
                arch, t, n_windows, n_clusters,
                abrupt_change_window, gradual_start_cluster, gradual_end_cluster,
                cyclical_period,
            )
            song_ids, cluster_ids = song_space.sample_from_clusters(
                mix, listens_per_window, exploration_rate
            )
            for s, c in zip(song_ids, cluster_ids):
                rows.append({
                    "user_id": u,
                    "window": t,
                    "song_id": int(s),
                    "cluster_id": int(c),
                })

    df = pd.DataFrame(rows)
    return df, ground_truth
