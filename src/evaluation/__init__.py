"""Evaluation: metrics and experiment runner."""

from .metrics import evaluate_detection, aggregate_metrics
from .experiments import run_parameter_sweep

__all__ = ["evaluate_detection", "aggregate_metrics", "run_parameter_sweep"]
