"""Decision-focused research tools with no execution or broker integration.

The functions in this module operate on frozen, point-in-time factor ranks and
offline future outcomes.  They are intentionally separate from the discovery
pipeline: validation and shadow data must never be passed to a fitting call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RidgeScoreModel:
    """One-dimensional ridge calibration for a frozen composite score."""

    intercept: float
    slope: float

    def predict(self, score: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(score, dtype=float)
        return self.intercept + self.slope * values


@dataclass(frozen=True)
class LogisticProbabilityModel:
    """Small auditable logistic calibrator implemented with NumPy IRLS."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: float
    coefficients: np.ndarray

    def predict_proba(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != len(self.feature_names):
            raise ValueError("probability features do not match the fitted model")
        standardised = (values - self.feature_mean) / self.feature_scale
        logit = self.intercept + standardised @ self.coefficients
        return _sigmoid(logit)


def _normalised_sample_weight(
    sample_weight: np.ndarray | Sequence[float] | None, size: int
) -> np.ndarray:
    if sample_weight is None:
        return np.full(size, 1.0 / max(size, 1), dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (size,) or not np.isfinite(weights).all():
        raise ValueError("sample_weight must be one finite value per row")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("sample_weight must be non-negative with positive mass")
    return weights / float(weights.sum())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def project_bounded_simplex(
    values: np.ndarray | Sequence[float], lower: float, upper: float
) -> np.ndarray:
    """Project values onto sum(w)=1 with identical lower/upper bounds."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or len(vector) == 0:
        raise ValueError("values must be a non-empty finite vector")
    if lower < 0 or upper <= lower:
        raise ValueError("invalid simplex bounds")
    if len(vector) * lower > 1.0 + 1e-12 or len(vector) * upper < 1.0 - 1e-12:
        raise ValueError("bounds do not contain a unit simplex")
    low = float(np.min(vector - upper))
    high = float(np.max(vector - lower))
    for _ in range(200):
        shift = (low + high) / 2.0
        projected = np.clip(vector - shift, lower, upper)
        if float(projected.sum()) > 1.0:
            low = shift
        else:
            high = shift
    projected = np.clip(vector - (low + high) / 2.0, lower, upper)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-10:
        free = np.flatnonzero(
            (projected > lower + 1e-12) & (projected < upper - 1e-12)
        )
        if len(free):
            projected[free] += residual / len(free)
    if abs(float(projected.sum()) - 1.0) > 1e-8:
        raise RuntimeError("bounded simplex projection did not converge")
    return projected


def chronological_research_partitions(
    dates: pd.DatetimeIndex,
    *,
    calibration_days: int,
    audit_days: int,
    validation_days: int,
    shadow_days: int,
    purge_days: int,
    minimum_fit_days: int,
) -> dict[str, Any]:
    """Create fit/calibration/audit/validation/shadow blocks with four purges."""
    index = pd.DatetimeIndex(dates).sort_values().unique()
    counts = (calibration_days, audit_days, validation_days, shadow_days)
    if any(int(value) <= 0 for value in counts):
        raise ValueError("all post-fit partition lengths must be positive")
    if purge_days < 0 or minimum_fit_days <= 0:
        raise ValueError("purge_days and minimum_fit_days are invalid")
    required = minimum_fit_days + sum(counts) + 4 * purge_days
    if len(index) < required:
        raise ValueError(f"need at least {required} dates, found {len(index)}")

    cursor = len(index)
    blocks: dict[str, pd.DatetimeIndex] = {}
    purges: list[pd.DatetimeIndex] = []
    for name, length in reversed(
        tuple(zip(("calibration", "audit", "validation", "shadow"), counts))
    ):
        start = cursor - int(length)
        blocks[name] = index[start:cursor]
        cursor = start
        purge_start = cursor - purge_days
        purges.append(index[purge_start:cursor])
        cursor = purge_start
    blocks["fit"] = index[:cursor]
    if len(blocks["fit"]) < minimum_fit_days:
        raise RuntimeError("fit block is shorter than the declared minimum")
    ordered = {
        name: blocks[name]
        for name in ("fit", "calibration", "audit", "validation", "shadow")
    }
    ordered["purged"] = pd.DatetimeIndex(
        np.concatenate([part.to_numpy() for part in reversed(purges)])
    )
    ordered["partition_audit"] = {
        "purge_days": int(purge_days),
        "fit_days": len(blocks["fit"]),
        "ranges": {
            name: [str(values[0].date()), str(values[-1].date()), len(values)]
            for name, values in blocks.items()
        },
        "shadow_is_fit_input": False,
    }
    return ordered


def _validate_rank_frames(
    factor_ranks: Mapping[str, pd.DataFrame], outcomes: pd.DataFrame
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    if len(factor_ranks) < 2:
        raise ValueError("at least two factor ranks are required")
    names = list(factor_ranks)
    aligned: dict[str, pd.DataFrame] = {}
    for name in names:
        frame = factor_ranks[name]
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"factor {name!r} has duplicate labels")
        aligned[name] = frame.reindex(index=outcomes.index, columns=outcomes.columns)
    return names, aligned


def pairwise_topk_training_arrays(
    factor_ranks: Mapping[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    fit_dates: Sequence[pd.Timestamp],
    *,
    top_count: int = 10,
    boundary_count: int = 20,
    minimum_cross_section: int = 50,
    ordinary_loss_threshold: float = 0.0,
    severe_loss_threshold: float = -0.03,
    ordinary_loss_penalty: float = 0.0,
    severe_loss_penalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build daily-equal positive-vs-boundary pairs using offline outcomes.

    ``outcomes`` is a label table and may contain future returns.  It is used
    only to construct training pairs and must not be reused as a feature.
    """
    if top_count <= 0 or boundary_count <= 0:
        raise ValueError("top_count and boundary_count must be positive")
    if minimum_cross_section <= 0:
        raise ValueError("minimum_cross_section must be positive")
    if severe_loss_threshold > ordinary_loss_threshold:
        raise ValueError("severe loss threshold must not exceed ordinary loss threshold")
    if ordinary_loss_penalty < 0 or severe_loss_penalty < 0:
        raise ValueError("loss penalties must be non-negative")
    names, ranks = _validate_rank_frames(factor_ranks, outcomes)
    differences: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used_dates: list[pd.Timestamp] = []
    for raw_date in pd.DatetimeIndex(fit_dates):
        if raw_date not in outcomes.index:
            continue
        y = outcomes.loc[raw_date].to_numpy(dtype=float)
        x = np.column_stack(
            [ranks[name].loc[raw_date].to_numpy(dtype=float) for name in names]
        )
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(valid.sum()) < max(
            minimum_cross_section, top_count + boundary_count
        ):
            continue
        xv, yv = x[valid], y[valid]
        utility = (
            yv
            - ordinary_loss_penalty * (yv <= ordinary_loss_threshold)
            - severe_loss_penalty * (yv <= severe_loss_threshold)
        )
        order = np.argsort(utility, kind="stable")[::-1]
        positive = xv[order[:top_count]]
        negative = xv[order[top_count : top_count + boundary_count]]
        day_pairs = (positive[:, None, :] - negative[None, :, :]).reshape(
            -1, len(names)
        )
        differences.append(day_pairs)
        weights.append(np.full(len(day_pairs), 1.0 / len(day_pairs)))
        used_dates.append(raw_date)
    if not differences:
        raise ValueError("no fit date met the pairwise cross-section requirement")
    day_count = len(differences)
    return (
        np.concatenate(differences),
        np.concatenate(weights) / day_count,
        {
            "factor_names": names,
            "used_dates": used_dates,
            "used_day_count": day_count,
            "pair_count": int(sum(len(value) for value in differences)),
            "pairs_per_used_day": top_count * boundary_count,
            "label_table_only": True,
        },
    )


def fit_pairwise_topk_weights(
    factor_ranks: Mapping[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    fit_dates: Sequence[pd.Timestamp],
    *,
    prior: np.ndarray | Sequence[float] | None = None,
    minimum_weight: float = 0.01,
    maximum_weight: float = 0.40,
    l2_to_prior: float = 0.05,
    max_iterations: int = 2_000,
    tolerance: float = 1e-10,
    **pair_options: Any,
) -> dict[str, Any]:
    """Fit a non-negative bounded simplex with pairwise logistic loss."""
    differences, sample_weight, audit = pairwise_topk_training_arrays(
        factor_ranks, outcomes, fit_dates, **pair_options
    )
    factor_names = audit["factor_names"]
    size = len(factor_names)
    if l2_to_prior < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid optimiser settings")
    if prior is None:
        prior_values = np.full(size, 1.0 / size)
    else:
        if np.asarray(prior).shape != (size,):
            raise ValueError("prior must contain one value per factor")
        prior_values = project_bounded_simplex(
            np.asarray(prior, dtype=float), minimum_weight, maximum_weight
        )
    weight = project_bounded_simplex(
        prior_values, minimum_weight, maximum_weight
    )
    weighted_gram = differences.T @ (differences * sample_weight[:, None])
    lipschitz = (
        0.25 * float(np.linalg.eigvalsh(weighted_gram).max())
        + 2.0 * l2_to_prior
    )
    step = 1.0 / max(lipschitz, 1e-8)
    converged = False
    for iteration in range(1, max_iterations + 1):
        margin = differences @ weight
        inverse = np.exp(-np.logaddexp(0.0, margin))
        gradient = -(differences.T @ (sample_weight * inverse))
        gradient += 2.0 * l2_to_prior * (weight - prior_values)
        candidate = project_bounded_simplex(
            weight - step * gradient, minimum_weight, maximum_weight
        )
        if float(np.linalg.norm(candidate - weight)) <= tolerance:
            weight = candidate
            converged = True
            break
        weight = candidate
    if not converged:
        raise RuntimeError("pairwise weight optimiser did not converge")
    margin = differences @ weight
    objective = float(
        np.sum(np.logaddexp(0.0, -margin) * sample_weight)
        + l2_to_prior * np.square(weight - prior_values).sum()
    )
    return {
        "factor_names": factor_names,
        "weights": weight,
        "weight_map": dict(zip(factor_names, weight, strict=True)),
        "prior": prior_values,
        "converged": converged,
        "iterations": iteration,
        "objective": objective,
        "audit": audit,
    }


def block_subsample_dates(
    dates: Sequence[pd.Timestamp],
    *,
    block_length: int,
    fraction: float,
    seed: int,
    replica: int,
) -> pd.DatetimeIndex:
    """Select complete chronological blocks without selecting individual rows."""
    index = pd.DatetimeIndex(dates)
    if len(index) == 0 or block_length <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid block bootstrap settings")
    blocks = [
        index[start : start + block_length]
        for start in range(0, len(index), block_length)
    ]
    keep = max(1, int(math.ceil(len(blocks) * fraction)))
    rng = np.random.default_rng(int(seed) + int(replica))
    chosen = sorted(rng.choice(len(blocks), size=keep, replace=False).tolist())
    return pd.DatetimeIndex(
        np.concatenate([blocks[position].to_numpy() for position in chosen])
    )


def fit_pairwise_weight_ensemble(
    factor_ranks: Mapping[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    fit_dates: Sequence[pd.Timestamp],
    *,
    replicas: int = 5,
    block_length: int = 21,
    subsample_fraction: float = 0.8,
    random_seed: int = 7,
    **fit_options: Any,
) -> dict[str, Any]:
    """Fit a full model plus block-subsample replicas for weight uncertainty."""
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    final = fit_pairwise_topk_weights(
        factor_ranks, outcomes, fit_dates, **fit_options
    )
    rows = []
    replica_audits = []
    for replica in range(replicas):
        selected = block_subsample_dates(
            fit_dates,
            block_length=block_length,
            fraction=subsample_fraction,
            seed=random_seed,
            replica=replica,
        )
        fitted = fit_pairwise_topk_weights(
            factor_ranks, outcomes, selected, **fit_options
        )
        rows.append(fitted["weights"])
        replica_audits.append(
            {
                "replica": replica,
                "date_count": len(selected),
                "used_day_count": fitted["audit"]["used_day_count"],
                "converged": fitted["converged"],
            }
        )
    matrix = np.vstack(rows)
    return {
        "factor_names": final["factor_names"],
        "weights": final["weights"],
        "weight_map": final["weight_map"],
        "replica_weights": matrix,
        "replica_mean": matrix.mean(axis=0),
        "replica_std": matrix.std(axis=0, ddof=0),
        "replica_minimum": matrix.min(axis=0),
        "replica_maximum": matrix.max(axis=0),
        "fit_audit": final["audit"],
        "replica_audits": replica_audits,
        "orders": [],
        "automatic_trading_changes": [],
    }


def calibration_rows(
    factor_ranks: Mapping[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten cross-sections while assigning equal total mass to each day."""
    names, ranks = _validate_rank_frames(factor_ranks, outcomes)
    feature_rows: list[np.ndarray] = []
    outcome_rows: list[np.ndarray] = []
    day_sizes: list[int] = []
    for date in pd.DatetimeIndex(dates):
        if date not in outcomes.index:
            continue
        y = outcomes.loc[date].to_numpy(dtype=float)
        x = np.column_stack(
            [ranks[name].loc[date].to_numpy(dtype=float) for name in names]
        )
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if not valid.any():
            continue
        feature_rows.append(x[valid])
        outcome_rows.append(y[valid])
        day_sizes.append(int(valid.sum()))
    if not feature_rows:
        raise ValueError("calibration rows are empty")
    sample_weight = np.concatenate(
        [np.full(size, 1.0 / (len(day_sizes) * size)) for size in day_sizes]
    )
    return (
        np.concatenate(feature_rows),
        np.concatenate(outcome_rows),
        sample_weight,
        names,
    )


def fit_ridge_score_model(
    scores: np.ndarray | Sequence[float],
    outcomes: np.ndarray | Sequence[float],
    *,
    l2: float = 1.0,
    sample_weight: np.ndarray | Sequence[float] | None = None,
) -> RidgeScoreModel:
    """Calibrate a frozen score to an expected outcome on an independent block."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if l2 < 0:
        raise ValueError("ridge penalty must be non-negative")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        raise ValueError("ridge calibration needs at least three finite rows")
    raw_weight = None if sample_weight is None else np.asarray(sample_weight)[valid]
    weight = _normalised_sample_weight(raw_weight, len(x))
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0, float(l2)])
    coefficient = np.linalg.solve(
        design.T @ (design * weight[:, None]) + penalty,
        design.T @ (weight * y),
    )
    return RidgeScoreModel(float(coefficient[0]), float(coefficient[1]))


def fit_logistic_probability_model(
    features: np.ndarray | pd.DataFrame,
    events: np.ndarray | Sequence[bool],
    *,
    feature_names: Sequence[str] | None = None,
    l2: float = 0.05,
    sample_weight: np.ndarray | Sequence[float] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> LogisticProbabilityModel:
    """Fit an L2 logistic probability model on an independent calibration block."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(events, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("features/events have incompatible shapes")
    if l2 < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid logistic optimiser settings")
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < max(20, x.shape[1] + 2):
        raise ValueError("logistic calibration sample is too small")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("events must be binary")
    raw_weight = None if sample_weight is None else np.asarray(sample_weight)[valid]
    weight = _normalised_sample_weight(raw_weight, len(x))
    mean = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average(np.square(x - mean), axis=0, weights=weight))
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    event_rate = float(np.clip(np.average(y, weights=weight), 1e-6, 1 - 1e-6))
    coefficient = np.zeros(design.shape[1])
    coefficient[0] = math.log(event_rate / (1.0 - event_rate))
    regulariser = np.diag([0.0] + [float(l2)] * z.shape[1])
    converged = False
    for _ in range(max_iterations):
        probability = _sigmoid(design @ coefficient)
        curvature = weight * probability * (1.0 - probability)
        gradient = design.T @ (weight * (probability - y))
        gradient += regulariser @ coefficient
        hessian = design.T @ (design * curvature[:, None]) + regulariser
        hessian += np.eye(len(coefficient)) * 1e-10
        step = np.linalg.solve(hessian, gradient)
        coefficient -= step
        if float(np.linalg.norm(step)) <= tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError("logistic probability calibrator did not converge")
    names = tuple(
        feature_names
        if feature_names is not None
        else [f"feature_{index + 1:02d}" for index in range(x.shape[1])]
    )
    if len(names) != x.shape[1]:
        raise ValueError("feature_names length does not match features")
    return LogisticProbabilityModel(
        names,
        mean,
        scale,
        float(coefficient[0]),
        coefficient[1:].copy(),
    )


def probability_metrics(
    events: np.ndarray | Sequence[bool],
    probabilities: np.ndarray | Sequence[float],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Return Brier, LogLoss, AUC and equal-width ECE diagnostics."""
    y = np.asarray(events, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if bins <= 0:
        raise ValueError("bins must be positive")
    valid = np.isfinite(y) & np.isfinite(probability)
    y, probability = y[valid], np.clip(probability[valid], 1e-12, 1 - 1e-12)
    if not len(y) or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("finite binary events and probabilities are required")
    positive, negative = int(y.sum()), int(len(y) - y.sum())
    auc = None
    if positive and negative:
        ranks = pd.Series(probability).rank(method="average").to_numpy()
        auc = float(
            (ranks[y == 1].sum() - positive * (positive + 1) / 2)
            / (positive * negative)
        )
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    ece = 0.0
    bucket_rows = []
    for index in range(bins):
        selected = bucket == index
        if not selected.any():
            continue
        mean_probability = float(probability[selected].mean())
        event_rate = float(y[selected].mean())
        ece += float(selected.mean()) * abs(mean_probability - event_rate)
        bucket_rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(selected.sum()),
                "mean_probability": mean_probability,
                "event_rate": event_rate,
            }
        )
    return {
        "observations": len(y),
        "event_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(np.mean(np.square(probability - y))),
        "log_loss": float(
            -np.mean(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
        ),
        "auc": auc,
        "ece": float(ece),
        "buckets": bucket_rows,
    }
