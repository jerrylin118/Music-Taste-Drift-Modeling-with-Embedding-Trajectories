# Music Taste Drift: Modeling Non-Stationary User Preferences with Embedding Trajectories

A **representation-learning and concept-drift** project that models a user's music "taste" as a time-varying embedding distribution over discrete time windows and detects when taste changes (gradual drift vs abrupt phase shifts). This is **not** a recommender system; the focus is on drift metrics, change-point detection, and validation on synthetic users with known ground truth.

---

## Overview

User preferences in music are non-stationary: tastes evolve over time due to life events, discovery, or mood. This project treats taste as a **trajectory in an embedding space**—each time window yields a summary vector $\mu_{u,t}$ from listened songs—and studies **drift** between consecutive windows to detect change points. We implement multiple drift metrics and two detectors, then validate detection accuracy and robustness under controlled noise (exploration rate) using synthetic user archetypes with known change points.

---

## Problem Framing: Non-Stationarity / Concept Drift

In many user-modeling settings, the distribution of user behavior shifts over time (concept drift). For music listening:

- **Abrupt change**: a user switches from one genre phase to another at a specific time.
- **Gradual drift**: preference mixture shifts smoothly from cluster A toward cluster B.
- **Stable** and **cyclical** users provide baselines for false-positive control.

We do not predict next-song or build a recommender; we **represent** taste per window and **detect when** it changes, which is useful for adaptive systems, A/B analysis, and longitudinal studies.

---

## Method

### 1. Song Space

- **Configurable embedding space** of dimension $d$ (default $d = 32$).
- **Synthetic mode (default)**: $K$ clusters (e.g. genres/moods); each song embedding is sampled from $\mathcal{N}(\mathbf{c}_k, \sigma^2 I)$ around cluster center $\mathbf{c}_k$.
- **Optional**: load a public CSV of audio features and reduce to $d$ dimensions via standardization + PCA. Not required to run.

A `SongSpace` class provides: sample songs by cluster mixture, return embeddings, and cluster IDs.

### 2. User Behavior Simulation (Ground Truth)

- $N$ users (default 200) over $T$ time windows (e.g. 20 “weeks”), $M$ listens per window (default 50).
- **Four user archetypes** with explicit ground truth:
  - **Stable**: same cluster mixture for all $t$.
  - **Gradual drift**: mixture weights shift linearly from cluster A to B across time.
  - **Abrupt change**: mixture switches at a known change point $\tau$.
  - **Cyclical**: preferences oscillate between two clusters.
- **Noise**: exploration rate $\varepsilon$ (probability of sampling outside preferred clusters) and event-level randomness.
- Output: tidy dataframe (user_id, window, song_id, cluster_id) and per-user ground truth (archetype, true change points, drift type).

### 3. Taste Representation (Embedding Trajectories)

For each user $u$ and window $t$, **taste** is summarized as:

- **Mean embedding**  
  $$
  \mu_{u,t} = \frac{1}{|L_{u,t}|} \sum_{s \in L_{u,t}} \mathbf{e}_s
  $$  
  where $L_{u,t}$ is the set of listened songs in that window and $\mathbf{e}_s$ their embeddings. Optionally an exponentially weighted mean within the window can be used.
- **Dispersion**: scalar (e.g. mean distance of listens to $\mu_{u,t}$) for interpretability.

Implemented in `modeling/taste.py`: listen events → per-user per-window $\mu_{u,t}$ and dispersion.

### 4. Drift Metrics (Formulas + Implementation)

Between consecutive windows we compute **drift** for $t = 1, \ldots, T-1$:

1. **Cosine distance**  
   $$
   D_{\cos}(\mu_{t}, \mu_{t-1}) = 1 - \frac{\mu_t \cdot \mu_{t-1}}{\|\mu_t\| \|\mu_{t-1}\|}
   $$  
   Scale-invariant; 0 when directions align.

2. **Euclidean distance**  
   $$
   D_{\text{euc}}(\mu_t, \mu_{t-1}) = \|\mu_t - \mu_{t-1}\|_2
   $$

3. **Gaussian KL proxy**: treat each window’s embeddings as $\mathcal{N}(\mu, \text{diag}(\sigma^2))$; compute a **symmetric KL proxy** between consecutive Gaussians (numerically stable, documented in code).

`compute_drift_series(user_taste)` returns drift values for $t = 1..T-1$.

### 5. Change-Point Detection

- **Threshold detector**: flag change at $t$ if $\text{drift}_t > \theta$ and optionally exceeds a rolling baseline by $k$ standard deviations.
- **CUSUM-style detector**: accumulates positive deviations above a reference; flag when cumulative sum exceeds a threshold.
- Interface: `detect_change_points(drift_series, method=..., params=...)` → predicted change points and scores.

---

## Experiments

- **Evaluation**: detection vs ground truth change points (for archetypes that have changes). Match with **tolerance** $\pm 1$ time step.
- **Metrics**: precision, recall, F1; false-positive rate on **stable** users.
- **Sweeps**:
  - **Exploration rate** $\varepsilon \in \{0.0, 0.1, 0.2, 0.3\}$ to test robustness under noise.
  - **Detection threshold** $\theta$ over a grid (e.g. 0.05–0.5).
  - Optional: embedding dimension $d$ or listens per window $M$ (configurable in `config.py`).
- Outputs: **CSV** summary and **JSON** report (aggregate and per-archetype breakdown).

---

## Results

Figures are generated under `reports/figures/` after running the pipeline.

1. **Drift over time** (`drift_examples.png`): drift time series for 6 example users (2 stable, 2 abrupt, 2 gradual) with **true** (green) and **predicted** (red) change points.

![Drift examples](reports/figures/drift_examples.png)

2. **2D trajectory** (`trajectory_examples.png`): PCA projection of $\mu_{u,t}$ across time; points connected in time order, colored by window; predicted change points marked.

![Trajectory examples](reports/figures/trajectory_examples.png)

3. **Evaluation curve** (`f1_vs_threshold.png`): F1 vs detection threshold $\theta$ for several exploration rates $\varepsilon$.

![F1 vs threshold](reports/figures/f1_vs_threshold.png)

Typical findings: lower $\varepsilon$ yields clearer drift signals and higher F1; increasing $\theta$ reduces false positives but can miss true change points (precision/recall trade-off).

---

## Limitations and Future Work

- **Synthetic data**: real listening has session effects, recency, and platform bias; we do not model those here.
- **Implicit feedback**: “listen” does not imply preference; no dislike or skip modeling.
- **Cold start**: new users or very sparse windows are not explicitly handled.
- **Real datasets**: extending to public listening/audio datasets (e.g. Million Song, Spotify features) would require handling missing data, normalization, and possibly learning embeddings instead of fixed PCA/synthetic clusters.

---

## Reproducibility

**Requirements:** Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`. No API keys.

```bash
pip install -r requirements.txt
python -m src.run_experiments
```

- **Outputs**:
  - `reports/figures/drift_examples.png`, `trajectory_examples.png`, `f1_vs_threshold.png`
  - `reports/metrics/summary.csv`, `reports/metrics/report.json`
- **Runtime**: under 5 minutes for default config (200 users, 20 windows, full threshold × ε sweep).
- **Determinism**: fixed seed in `src/config.py` (default 42).

Optional exploration: open `notebooks/exploration.ipynb` and run cells (set notebook kernel to project interpreter).

---

## Repository Structure

```
music-taste-drift/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── songspace.py
│   │   └── simulate.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── taste.py
│   │   ├── drift.py
│   │   └── changepoint.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── experiments.py
│   ├── viz/
│   │   ├── __init__.py
│   │   └── plots.py
│   └── run_experiments.py
├── reports/
│   ├── figures/   # generated PNGs
│   └── metrics/   # summary.csv, report.json
├── notebooks/
│   └── exploration.ipynb
└── tests/
    ├── test_drift.py
    └── test_changepoint.py
```

Run tests: `pytest tests/ -v`
