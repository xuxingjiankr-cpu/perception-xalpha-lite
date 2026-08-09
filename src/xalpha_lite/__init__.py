"""Perception-XAlpha Lite: research-only factor discovery tools."""

from .decision import (
    chronological_research_partitions,
    fit_logistic_probability_model,
    fit_pairwise_topk_weights,
    fit_pairwise_weight_ensemble,
    fit_ridge_score_model,
    probability_metrics,
)
from .discovery import run_discovery
from .pit import align_point_in_time_fundamentals

__all__ = [
    "align_point_in_time_fundamentals",
    "chronological_research_partitions",
    "fit_logistic_probability_model",
    "fit_pairwise_topk_weights",
    "fit_pairwise_weight_ensemble",
    "fit_ridge_score_model",
    "probability_metrics",
    "run_discovery",
]
__version__ = "0.2.0"
