"""Configuration for music taste drift experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Experiment and data generation configuration."""

    # Random seed for reproducibility
    seed: int = 42

    # Song space
    embedding_dim: int = 32
    n_clusters: int = 8
    cluster_std: float = 0.4
    n_songs_per_cluster: int = 100

    # User simulation
    n_users: int = 200
    n_windows: int = 20
    listens_per_window: int = 50
    exploration_rate: float = 0.1

    # Archetype distribution (must sum to 1)
    archetype_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25]
    )  # stable, gradual, abrupt, cyclical

    # Change-point / drift
    abrupt_change_window: int = 10  # for abrupt archetype
    gradual_start_cluster: int = 0
    gradual_end_cluster: int = 4
    cyclical_period: int = 5

    # Detection
    threshold_grid: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    )
    tolerance_windows: int = 1  # ±1 time step for TP

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    reports_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.reports_dir = self.project_root / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.metrics_dir = self.reports_dir / "metrics"
