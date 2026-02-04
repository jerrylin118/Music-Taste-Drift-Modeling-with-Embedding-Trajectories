"""
Taste representation: per-user per-window mean embedding μ_{u,t} and dispersion.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.songspace import SongSpace


def compute_taste_trajectories(
    listens_df: pd.DataFrame,
    song_space: SongSpace,
    use_exponential_weight: bool = False,
    exp_decay: float = 0.9,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Convert listen events to per-user per-window taste representations.

    For each user u and window t:
      μ_{u,t} = mean of song embeddings listened to in that window
      (optionally exponentially weighted by position within window)
    Dispersion = mean distance of listens to μ (scalar per window).

    Returns:
      taste_means: user_id -> (T, d) array of mean embeddings per window
      dispersions: user_id -> (T,) array of dispersion per window
      window_indices: user_id -> (T,) array of window indices (0..T-1)
    """
    taste_means: Dict[int, np.ndarray] = {}
    dispersions: Dict[int, np.ndarray] = {}
    embeddings = song_space.embeddings
    d = embeddings.shape[1]

    for uid, grp in listens_df.groupby("user_id"):
        windows = sorted(grp["window"].unique())
        T = len(windows)
        mu_t = np.zeros((T, d), dtype=float)
        disp_t = np.zeros(T, dtype=float)
        for i, w in enumerate(windows):
            sub = grp[grp["window"] == w]
            song_ids = sub["song_id"].values
            vecs = embeddings[song_ids]
            n = len(vecs)
            if n == 0:
                mu_t[i] = np.zeros(d)
                disp_t[i] = 0.0
                continue
            if use_exponential_weight and n > 0:
                weights = np.array([exp_decay ** (n - 1 - j) for j in range(n)], dtype=float)
                weights = weights / weights.sum()
                mu_t[i] = (vecs * weights[:, np.newaxis]).sum(axis=0)
            else:
                mu_t[i] = vecs.mean(axis=0)
            dists = np.linalg.norm(vecs - mu_t[i], axis=1)
            disp_t[i] = float(dists.mean())
        taste_means[uid] = mu_t
        dispersions[uid] = disp_t

    return taste_means, dispersions, {uid: np.arange(len(taste_means[uid])) for uid in taste_means}
