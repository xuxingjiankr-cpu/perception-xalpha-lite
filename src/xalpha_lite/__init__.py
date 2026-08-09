"""Perception-XAlpha Lite: research-only factor discovery tools."""

from .decision import (
    chronological_research_partitions,
    fit_logistic_probability_model,
    fit_pairwise_topk_weights,
    fit_pairwise_weight_ensemble,
    fit_ridge_score_model,
    probability_metrics,
)
from .book import long_only_book, long_only_weights, universe_benchmark
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
from .forward import (
    append_prediction,
    build_prediction,
    freeze_spec,
    load_spec,
    score_log,
    spec_digest,
    validate_spec,
)
from .pit import align_point_in_time_fundamentals
from .universe import (
    apply_sealed_bar_limits,
    point_in_time_eligibility,
    sealed_bar_limits,
)

__all__ = [
    "BootstrapDesign",
    "align_point_in_time_fundamentals",
    "append_prediction",
    "apply_sealed_bar_limits",
    "benjamini_hochberg_qvalues",
    "benjamini_yekutieli_qvalues",
    "build_prediction",
    "chronological_research_partitions",
    "evidence_report",
    "fit_logistic_probability_model",
    "fit_pairwise_topk_weights",
    "fit_pairwise_weight_ensemble",
    "fit_ridge_score_model",
    "freeze_spec",
    "load_spec",
    "long_only_book",
    "long_only_weights",
    "point_in_time_eligibility",
    "probability_metrics",
    "romano_wolf_stepdown",
    "run_discovery",
    "score_log",
    "sealed_bar_limits",
    "spec_digest",
    "stationary_bootstrap_mean_intervals",
    "universe_benchmark",
    "validate_spec",
    "white_reality_check",
]
__version__ = "0.5.0"
