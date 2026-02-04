"""Song embedding space: synthetic clusters or optional PCA from CSV."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SongSpace:
    """
    Configurable song embedding space of dimension d.
    Supports (1) synthetic cluster-based embeddings or (2) optional PCA from CSV.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        n_clusters: int = 8,
        cluster_std: float = 0.4,
        n_songs_per_cluster: int = 100,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.cluster_std = cluster_std
        self.n_songs_per_cluster = n_songs_per_cluster
        self.rng = np.random.default_rng(seed)
        self._embeddings: Optional[np.ndarray] = None
        self._song_to_cluster: Optional[np.ndarray] = None
        self._cluster_centers: Optional[np.ndarray] = None

    def build_synthetic(self) -> "SongSpace":
        """Build synthetic embeddings: K clusters, each song from Gaussian around center."""
        d = self.embedding_dim
        K = self.n_clusters
        n_per = self.n_songs_per_cluster
        # Cluster centers: random unit-norm-ish vectors spread in R^d
        centers = self.rng.standard_normal((K, d))
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        self._cluster_centers = centers
        embeddings_list: List[np.ndarray] = []
        cluster_ids: List[int] = []
        for k in range(K):
            pts = centers[k] + self.cluster_std * self.rng.standard_normal((n_per, d))
            embeddings_list.append(pts)
            cluster_ids.extend([k] * n_per)
        self._embeddings = np.vstack(embeddings_list)
        self._song_to_cluster = np.array(cluster_ids)
        return self

    def build_from_csv(
        self,
        csv_path: Path,
        feature_columns: Optional[List[str]] = None,
        target_dim: Optional[int] = None,
    ) -> "SongSpace":
        """
        Load CSV (e.g. audio features), standardize, PCA to target_dim.
        If feature_columns is None, use all numeric columns except index/id.
        """
        df = pd.read_csv(csv_path)
        if feature_columns is None:
            numeric = df.select_dtypes(include=[np.number])
            feature_columns = list(numeric.columns)
        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        d = target_dim or self.embedding_dim
        d = min(d, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=d)
        self._embeddings = pca.fit_transform(X_scaled)
        self.embedding_dim = d
        # No cluster info from CSV; assign dummy single cluster
        self._song_to_cluster = np.zeros(len(self._embeddings), dtype=int)
        self._cluster_centers = None
        return self

    def build(self, csv_path: Optional[Path] = None) -> "SongSpace":
        """Build space: from CSV if path given and exists, else synthetic."""
        if csv_path is not None and Path(csv_path).exists():
            return self.build_from_csv(csv_path, target_dim=self.embedding_dim)
        return self.build_synthetic()

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self.build_synthetic()
        return self._embeddings  # type: ignore

    @property
    def song_to_cluster(self) -> np.ndarray:
        if self._song_to_cluster is None:
            self.build_synthetic()
        return self._song_to_cluster  # type: ignore

    def get_embedding(self, song_id: int) -> np.ndarray:
        return self.embeddings[song_id].copy()

    def get_cluster(self, song_id: int) -> int:
        return int(self.song_to_cluster[song_id])

    def sample_from_clusters(
        self,
        cluster_weights: np.ndarray,
        size: int,
        exploration_rate: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample song indices and return (song_ids, cluster_ids).
        cluster_weights: length K probability vector.
        With probability exploration_rate, sample uniformly over all songs.
        """
        n_songs = len(self.embeddings)
        n_clusters = self.n_clusters
        cluster_weights = np.asarray(cluster_weights, dtype=float)
        if cluster_weights.size != n_clusters:
            cluster_weights = np.broadcast_to(
                np.array([1.0 / n_clusters] * n_clusters), (n_clusters,)
            )
        cluster_weights = cluster_weights / cluster_weights.sum()
        song_ids: List[int] = []
        cluster_ids: List[int] = []
        for _ in range(size):
            if exploration_rate > 0 and self.rng.random() < exploration_rate:
                idx = self.rng.integers(0, n_songs)
                song_ids.append(idx)
                cluster_ids.append(self.get_cluster(idx))
            else:
                k = self.rng.choice(n_clusters, p=cluster_weights)
                # songs in cluster k
                mask = self.song_to_cluster == k
                indices = np.where(mask)[0]
                idx = self.rng.choice(indices)
                song_ids.append(int(idx))
                cluster_ids.append(k)
        return np.array(song_ids), np.array(cluster_ids)

    def n_songs(self) -> int:
        return len(self.embeddings)
