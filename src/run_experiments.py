"""
Entry point: generate data, compute taste/drift, detect change points, evaluate, save figures and metrics.
Run: python -m src.run_experiments
"""

from pathlib import Path

from src.config import Config
from src.evaluation.experiments import run_parameter_sweep, save_metrics_and_summary
from src.viz.plots import plot_drift_examples, plot_trajectory_examples, plot_f1_vs_threshold


def main() -> None:
    config = Config()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Running parameter sweep (exploration rate × threshold)...")
    results, first_run = run_parameter_sweep(config)

    print("Saving metrics and summary...")
    save_metrics_and_summary(results, config.metrics_dir)

    print("Generating figures...")
    plot_drift_examples(first_run, config.figures_dir / "drift_examples.png")
    plot_trajectory_examples(first_run, config.figures_dir / "trajectory_examples.png")
    plot_f1_vs_threshold(results, config.figures_dir / "f1_vs_threshold.png")

    print("Done.")
    print(f"  Figures: {config.figures_dir}")
    print(f"  Metrics: {config.metrics_dir}")


if __name__ == "__main__":
    main()
