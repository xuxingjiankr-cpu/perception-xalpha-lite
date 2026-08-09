"""Dependence-aware evidence tests for a searched family of research candidates.

The input is a chronological table of performance differentials where positive
values mean that a candidate outperformed a frozen benchmark.  The module is an
offline falsification tool.  It does not construct positions or submit orders.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapDesign:
    """Frozen resampling design shared by every test in one evidence report."""

    repetitions: int = 2_000
    expected_block_length: float = 10.0
    confidence: float = 0.95
    alpha: float = 0.05
    random_seed: int = 20_260_809

    def validate(self, observations: int) -> None:
        if observations < 20:
            raise ValueError("at least 20 joint observations are required")
        if self.repetitions < 99:
            raise ValueError("repetitions must be at least 99")
        if not 1.0 <= self.expected_block_length <= observations:
            raise ValueError(
                "expected_block_length must be between 1 and the sample size"
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")


def stationary_bootstrap_indices(
    observations: int,
    repetitions: int,
    expected_block_length: float,
    random_seed: int,
) -> np.ndarray:
    """Generate Politis--Romano stationary-bootstrap index paths.

    Each path starts at a uniformly sampled observation.  At every subsequent
    position a new block starts with probability ``1 / expected_block_length``;
    otherwise the previous index advances circularly by one.
    """
    design = BootstrapDesign(
        repetitions=repetitions,
        expected_block_length=expected_block_length,
        random_seed=random_seed,
    )
    design.validate(observations)
    rng = np.random.default_rng(random_seed)
    paths = np.empty((repetitions, observations), dtype=np.int64)
    paths[:, 0] = rng.integers(0, observations, size=repetitions)
    restart_probability = 1.0 / expected_block_length
    for position in range(1, observations):
        restart = rng.random(repetitions) < restart_probability
        continuation = (paths[:, position - 1] + 1) % observations
        fresh = rng.integers(0, observations, size=repetitions)
        paths[:, position] = np.where(restart, fresh, continuation)
    return paths


def _validated_differentials(
    differentials: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(differentials, pd.DataFrame) or differentials.empty:
        raise ValueError("differentials must be a non-empty DataFrame")
    if differentials.columns.duplicated().any():
        raise ValueError("candidate names must be unique")
    candidate_names = [str(column) for column in differentials.columns]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate names must remain unique when serialized")
    frame = differentials.apply(pd.to_numeric, errors="coerce").astype(float)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("the time index must be sorted and unique")
    finite = np.isfinite(frame.to_numpy())
    joint_mask = finite.all(axis=1)
    clean = frame.loc[joint_mask]
    if clean.empty:
        raise ValueError("no joint complete observations remain")
    audit = {
        "input_observations": int(len(frame)),
        "joint_observations": int(len(clean)),
        "dropped_non_joint_observations": int((~joint_mask).sum()),
        "candidate_count": int(frame.shape[1]),
        "candidate_names": candidate_names,
        "first_joint_observation": str(clean.index[0]),
        "last_joint_observation": str(clean.index[-1]),
        "joint_complete_case_policy": True,
    }
    return clean, audit


def _bootstrap_arrays(
    values: np.ndarray, design: BootstrapDesign
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = stationary_bootstrap_indices(
        observations=len(values),
        repetitions=design.repetitions,
        expected_block_length=design.expected_block_length,
        random_seed=design.random_seed,
    )
    means = values.mean(axis=0)
    resampled_means = np.empty((design.repetitions, values.shape[1]), dtype=float)
    target_elements = 2_000_000
    batch_size = max(1, min(design.repetitions, target_elements // max(1, values.size)))
    for start in range(0, design.repetitions, batch_size):
        end = min(design.repetitions, start + batch_size)
        resampled_means[start:end] = values[indices[start:end]].mean(axis=1)
    centered_means = resampled_means - means
    standard_errors = centered_means.std(axis=0, ddof=1)
    return indices, means, resampled_means, standard_errors


def _mean_interval_rows(
    columns: pd.Index,
    means: np.ndarray,
    resampled_means: np.ndarray,
    standard_errors: np.ndarray,
    confidence: float,
) -> dict[str, dict[str, float | None]]:
    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(resampled_means, tail, axis=0)
    upper = np.quantile(resampled_means, 1.0 - tail, axis=0)
    return {
        str(name): {
            "mean_differential": float(means[index]),
            "bootstrap_standard_error": (
                float(standard_errors[index]) if standard_errors[index] > 0 else None
            ),
            "confidence_interval_lower": float(lower[index]),
            "confidence_interval_upper": float(upper[index]),
        }
        for index, name in enumerate(columns)
    }


def stationary_bootstrap_mean_intervals(
    differentials: pd.DataFrame,
    design: BootstrapDesign = BootstrapDesign(),
) -> dict[str, dict[str, float | None]]:
    """Dependence-aware percentile intervals for candidate mean differentials."""
    clean, _ = _validated_differentials(differentials)
    design.validate(len(clean))
    _, means, resampled_means, standard_errors = _bootstrap_arrays(
        clean.to_numpy(), design
    )
    return _mean_interval_rows(
        clean.columns,
        means,
        resampled_means,
        standard_errors,
        design.confidence,
    )


def _white_reality_from_bootstrap(
    columns: pd.Index,
    observations: int,
    means: np.ndarray,
    resampled_means: np.ndarray,
    design: BootstrapDesign,
) -> dict[str, Any]:
    scale = np.sqrt(observations)
    observed = float(scale * np.max(means))
    bootstrap_maximum = scale * np.max(resampled_means - means, axis=1)
    p_value = (1.0 + float(np.sum(bootstrap_maximum >= observed))) / (
        design.repetitions + 1.0
    )
    best = int(np.argmax(means))
    return {
        "test": "White Reality Check",
        "null_hypothesis": "no candidate outperforms the frozen benchmark in expectation",
        "best_candidate": str(columns[best]),
        "best_mean_differential": float(means[best]),
        "test_statistic": observed,
        "p_value": p_value,
        "global_null_rejected": bool(p_value <= design.alpha and means[best] > 0),
    }


def white_reality_check(
    differentials: pd.DataFrame,
    design: BootstrapDesign = BootstrapDesign(),
) -> dict[str, Any]:
    """White's bootstrap test of the searched-family global null.

    The null is that no candidate has a positive expected differential versus
    the benchmark.  The bootstrap is applied jointly to every candidate after
    recentering each differential series at its sample mean.
    """
    clean, _ = _validated_differentials(differentials)
    design.validate(len(clean))
    _, means, resampled_means, _ = _bootstrap_arrays(clean.to_numpy(), design)
    return _white_reality_from_bootstrap(
        clean.columns, len(clean), means, resampled_means, design
    )


def benjamini_hochberg_qvalues(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini--Hochberg adjusted p-values for a named hypothesis family."""
    if not p_values:
        return {}
    names = list(p_values)
    values = np.asarray([p_values[name] for name in names], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return {name: float(output[index]) for index, name in enumerate(names)}


def benjamini_yekutieli_qvalues(p_values: dict[str, float]) -> dict[str, float]:
    """Dependence-robust Benjamini--Yekutieli adjusted p-values.

    The harmonic correction is more conservative than BH but does not require
    independence or positive regression dependence within the hypothesis family.
    """
    bh = benjamini_hochberg_qvalues(p_values)
    harmonic = sum(1.0 / rank for rank in range(1, len(bh) + 1))
    return {name: min(1.0, value * harmonic) for name, value in bh.items()}


def romano_wolf_stepdown(
    differentials: pd.DataFrame,
    design: BootstrapDesign = BootstrapDesign(),
) -> dict[str, Any]:
    """Dependence-aware Romano--Wolf step-down max-t adjustment.

    Studentization uses the shared stationary-bootstrap distribution of means.
    The same resampled paths are used for every candidate, preserving their
    cross-sectional dependence during family-wise error control.
    """
    clean, _ = _validated_differentials(differentials)
    design.validate(len(clean))
    _, means, resampled_means, standard_errors = _bootstrap_arrays(
        clean.to_numpy(), design
    )
    return _romano_wolf_from_bootstrap(
        clean.columns, means, resampled_means, standard_errors, design
    )


def _romano_wolf_from_bootstrap(
    columns: pd.Index,
    means: np.ndarray,
    resampled_means: np.ndarray,
    standard_errors: np.ndarray,
    design: BootstrapDesign,
) -> dict[str, Any]:
    valid = standard_errors > np.finfo(float).eps
    observed_t = np.full_like(means, -np.inf)
    observed_t[valid] = means[valid] / standard_errors[valid]
    centered_t = np.zeros_like(resampled_means)
    centered_t[:, valid] = (resampled_means[:, valid] - means[valid]) / standard_errors[
        valid
    ]
    order = np.argsort(-observed_t)
    adjusted = np.ones(len(means), dtype=float)
    unadjusted = np.ones(len(means), dtype=float)
    for candidate in range(len(means)):
        if valid[candidate]:
            unadjusted[candidate] = (
                1.0 + np.sum(centered_t[:, candidate] >= observed_t[candidate])
            ) / (design.repetitions + 1.0)
    previous = 0.0
    for rank, candidate in enumerate(order):
        if not valid[candidate]:
            adjusted[candidate] = 1.0
            continue
        remaining = order[rank:]
        reference = np.max(centered_t[:, remaining], axis=1)
        raw = (1.0 + np.sum(reference >= observed_t[candidate])) / (
            design.repetitions + 1.0
        )
        previous = max(previous, float(raw))
        adjusted[candidate] = previous
    raw_map = {
        str(name): float(unadjusted[index]) for index, name in enumerate(columns)
    }
    q_values = benjamini_hochberg_qvalues(raw_map)
    dependence_robust_q_values = benjamini_yekutieli_qvalues(raw_map)
    candidates = {}
    for index, name in enumerate(columns):
        candidates[str(name)] = {
            "mean_differential": float(means[index]),
            "studentized_statistic": (
                float(observed_t[index]) if np.isfinite(observed_t[index]) else None
            ),
            "unadjusted_bootstrap_p_value": float(unadjusted[index]),
            "romano_wolf_adjusted_p_value": float(adjusted[index]),
            "benjamini_hochberg_q_value": q_values[str(name)],
            "benjamini_yekutieli_q_value": dependence_robust_q_values[str(name)],
            "familywise_rejected": bool(
                adjusted[index] <= design.alpha and means[index] > 0
            ),
            "fdr_rejected": bool(
                q_values[str(name)] <= design.alpha and means[index] > 0
            ),
            "dependence_robust_fdr_rejected": bool(
                dependence_robust_q_values[str(name)] <= design.alpha
                and means[index] > 0
            ),
        }
    return {
        "test": "Romano-Wolf step-down max-t",
        "familywise_error_rate": design.alpha,
        "candidates": candidates,
    }


def _hash_frame(frame: pd.DataFrame) -> str:
    values_hash = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    columns = json.dumps([str(column) for column in frame.columns]).encode()
    return hashlib.sha256(values_hash + columns).hexdigest()


def evidence_report(
    differentials: pd.DataFrame,
    design: BootstrapDesign = BootstrapDesign(),
) -> dict[str, Any]:
    """Run the complete paper-backed evidence stress test."""
    numeric_input = differentials.apply(pd.to_numeric, errors="coerce").astype(float)
    clean, audit = _validated_differentials(numeric_input)
    design.validate(len(clean))
    _, means, resampled_means, standard_errors = _bootstrap_arrays(
        clean.to_numpy(), design
    )
    intervals = _mean_interval_rows(
        clean.columns,
        means,
        resampled_means,
        standard_errors,
        design.confidence,
    )
    reality_check = _white_reality_from_bootstrap(
        clean.columns, len(clean), means, resampled_means, design
    )
    stepdown = _romano_wolf_from_bootstrap(
        clean.columns, means, resampled_means, standard_errors, design
    )
    config = {
        "repetitions": design.repetitions,
        "expected_block_length": design.expected_block_length,
        "confidence": design.confidence,
        "alpha": design.alpha,
        "random_seed": design.random_seed,
    }
    return {
        "status": "diagnostic_only_research_only_not_trading",
        "orders": [],
        "automatic_trading_changes": [],
        "input_contract": (
            "positive differential means candidate performance exceeds a frozen benchmark"
        ),
        "sample_audit": audit,
        "bootstrap_design": config,
        "mean_intervals": intervals,
        "white_reality_check": reality_check,
        "multiple_testing": stepdown,
        "assumption_audit": {
            "chronological_index_required": True,
            "joint_complete_cases_used": True,
            "shared_bootstrap_paths_preserve_cross_candidate_dependence": True,
            "stationary_weak_dependence_assumption_requires_domain_review": True,
            "block_length_is_a_preregistered_design_choice": True,
            "historical_rejection_does_not_authorize_deployment": True,
        },
        "literature_trace": [
            {
                "method": "stationary_bootstrap",
                "citation": "Politis and Romano (1994)",
                "doi": "10.1080/01621459.1994.10476870",
            },
            {
                "method": "reality_check",
                "citation": "White (2000)",
                "doi": "10.1111/1468-0262.00152",
            },
            {
                "method": "stepdown_max_t",
                "citation": "Romano and Wolf (2005)",
                "doi": "10.1111/j.1468-0262.2005.00615.x",
            },
            {
                "method": "false_discovery_rate",
                "citation": "Benjamini and Hochberg (1995)",
                "doi": "10.1111/j.2517-6161.1995.tb02031.x",
            },
            {
                "method": "false_discovery_rate_under_dependency",
                "citation": "Benjamini and Yekutieli (2001)",
                "doi": "10.1214/aos/1013699998",
            },
        ],
        "run_manifest": {
            "input_sha256": _hash_frame(numeric_input),
            "joint_sample_sha256": _hash_frame(clean),
            "configuration_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }


def write_evidence_report(report: dict[str, Any], output: Path) -> None:
    """Atomically persist a JSON evidence artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
