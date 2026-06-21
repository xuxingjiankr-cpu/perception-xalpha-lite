"""Frozen decision-score probability shadow (record-only).

The model is deliberately small: a Beta-smoothed BUY base rate plus one regularized
total-score likelihood term in log-odds space.  It forecasts before the outcome is
known, then the post-close enrichment job attaches Brier/log-loss diagnostics.

This module cannot gate an order or size a position.  The checked-in model contract
requires ``tradeGateEnabled=false`` and ``positionSizingEnabled=false``; otherwise it
fails closed and returns no forecast.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from run_etf_paper_trading_agent import ROOT, as_float


MODEL_PATH = ROOT / "configs" / "shadow" / "decision_probability_v1.json"
PROBABILITY_FIELDS = [
    "calibration_version", "bayesian_model_version", "probability_model_sha256",
    "probability_success_definition", "prior_prob", "prior_logodds",
    "score_evidence_z", "score_log_likelihood_ratio", "posterior_prob",
    "expected_win", "expected_loss", "estimated_round_trip_cost", "breakeven_prob",
    "expected_value_shadow", "probability_confidence", "probability_shadow_action",
]
PROBABILITY_OUTCOME_FIELDS = ["probability_outcome", "brier_score", "probability_log_loss"]

_MODEL_CACHE: tuple[str, int, dict[str, Any] | None] | None = None


def clamp_probability(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def model_sha256(model: dict[str, Any]) -> str:
    payload = json.dumps(model, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_shadow_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    """Load and validate the record-only model, cached by file modification time."""
    global _MODEL_CACHE
    try:
        mtime = path.stat().st_mtime_ns
        cache_key = str(path.resolve())
        if _MODEL_CACHE and _MODEL_CACHE[0] == cache_key and _MODEL_CACHE[1] == mtime:
            return _MODEL_CACHE[2]
        model = json.loads(path.read_text(encoding="utf-8"))
        safe = (
            isinstance(model, dict)
            and model.get("schemaVersion") == "decision_probability_shadow_v1"
            and model.get("recordOnly") is True
            and model.get("tradeGateEnabled") is False
            and model.get("positionSizingEnabled") is False
        )
        loaded = model if safe else None
        _MODEL_CACHE = (cache_key, mtime, loaded)
        return loaded
    except Exception:
        return None


def forecast_shadow(*, total_score: float, decision_type: str, date: str,
                    model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a preregistered BUY probability forecast; never return a trade action."""
    model = model or load_shadow_model()
    if not model or str(decision_type).upper() != "BUY":
        return {}
    effective = str(model.get("effectiveFrom") or "9999-12-31")
    if str(date) < effective:
        return {}
    transform = model.get("scoreTransform", {})
    center = as_float(transform.get("center"), 0.0)
    scale = max(as_float(transform.get("scale"), 1.0), 1e-12)
    slope = as_float(transform.get("regularizedSlope"), 0.0)
    prior = clamp_probability(as_float(model.get("prior", {}).get("probability"), 0.5))
    z_score = (as_float(total_score) - center) / scale
    log_lr = slope * z_score
    posterior = sigmoid(logit(prior) + log_lr)
    payoff = model.get("payoff", {})
    expected_win = max(0.0, as_float(payoff.get("expectedGrossWin")))
    expected_loss = max(0.0, as_float(payoff.get("expectedGrossLoss")))
    cost = max(0.0, as_float(payoff.get("roundTripCost")))
    denominator = expected_win + expected_loss
    breakeven = (expected_loss + cost) / denominator if denominator > 0 else 1.0
    ev = posterior * expected_win - (1.0 - posterior) * expected_loss - cost
    return {
        "calibration_version": model.get("calibrationVersion"),
        "bayesian_model_version": model.get("bayesianModelVersion"),
        "probability_model_sha256": model_sha256(model),
        "probability_success_definition": model.get("successDefinition"),
        "prior_prob": round(prior, 8),
        "prior_logodds": round(logit(prior), 8),
        "score_evidence_z": round(z_score, 8),
        "score_log_likelihood_ratio": round(log_lr, 8),
        "posterior_prob": round(posterior, 8),
        "expected_win": round(expected_win, 8),
        "expected_loss": round(expected_loss, 8),
        "estimated_round_trip_cost": round(cost, 8),
        "breakeven_prob": round(breakeven, 8),
        "expected_value_shadow": round(ev, 8),
        "probability_confidence": round(as_float(model.get("modelConfidence")), 8),
        "probability_shadow_action": "RECORD_ONLY_NO_TRADE_GATE",
    }


def enrich_probability_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """Attach outcome/error only after prices exist; success means net return > 0."""
    if record.get("posterior_prob") is None or record.get("realized_return") is None:
        return record
    cost = max(0.0, as_float(record.get("estimated_round_trip_cost")))
    outcome = 1 if as_float(record.get("realized_return")) - cost > 0 else 0
    probability = clamp_probability(as_float(record.get("posterior_prob"), 0.5))
    record["probability_outcome"] = outcome
    record["brier_score"] = round((probability - outcome) ** 2, 8)
    record["probability_log_loss"] = round(
        -(outcome * math.log(probability) + (1 - outcome) * math.log(1.0 - probability)), 8
    )
    return record


def auc_score(probabilities: list[float], outcomes: list[int]) -> float | None:
    positives = [p for p, y in zip(probabilities, outcomes) if y == 1]
    negatives = [p for p, y in zip(probabilities, outcomes) if y == 0]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0)
               for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def probability_metrics(probabilities: Iterable[float], outcomes: Iterable[int],
                        *, bin_count: int = 5) -> dict[str, Any]:
    probs = [clamp_probability(value) for value in probabilities]
    ys = [int(value) for value in outcomes]
    if len(probs) != len(ys) or not probs:
        return {"count": 0, "brier": None, "log_loss": None, "auc": None,
                "ece": None, "mean_predicted": None, "actual_rate": None, "bins": []}
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bin_count):
        low, high = index / bin_count, (index + 1) / bin_count
        members = [(p, y) for p, y in zip(probs, ys)
                   if p >= low and (p < high or (index == bin_count - 1 and p <= high))]
        if not members:
            continue
        predicted = sum(p for p, _ in members) / len(members)
        actual = sum(y for _, y in members) / len(members)
        ece += len(members) / len(probs) * abs(predicted - actual)
        bins.append({"low": low, "high": high, "count": len(members),
                     "mean_predicted": predicted, "actual_rate": actual})
    return {
        "count": len(probs),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(probs),
        "log_loss": -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                         for p, y in zip(probs, ys)) / len(probs),
        "auc": auc_score(probs, ys),
        "ece": ece,
        "mean_predicted": sum(probs) / len(probs),
        "actual_rate": sum(ys) / len(ys),
        "bins": bins,
    }
