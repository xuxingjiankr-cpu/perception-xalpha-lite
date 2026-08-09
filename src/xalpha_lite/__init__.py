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
from .evidence import (
    BootstrapDesign,
    benjamini_hochberg_qvalues,
    benjamini_yekutieli_qvalues,
    evidence_report,
    romano_wolf_stepdown,
    stationary_bootstrap_mean_intervals,
    white_reality_check,
)
from .pit import align_point_in_time_fundamentals

__all__ = [
    "align_point_in_time_fundamentals",
    "benjamini_hochberg_qvalues",
    "benjamini_yekutieli_qvalues",
    "BootstrapDesign",
    "chronological_research_partitions",
    "fit_logistic_probability_model",
    "fit_pairwise_topk_weights",
    "fit_pairwise_weight_ensemble",
    "fit_ridge_score_model",
    "evidence_report",
    "probability_metrics",
    "romano_wolf_stepdown",
    "run_discovery",
    "stationary_bootstrap_mean_intervals",
    "white_reality_check",
]
__version__ = "0.3.0"
