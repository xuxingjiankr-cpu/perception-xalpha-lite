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
RESEARCH_CONFIG_PATH = ROOT / "configs" / "research" / "decision_probability_forward_v1.json"
PRIOR_REGISTRY_PATH = ROOT / "outputs" / "decision_probability_research" / "bayesian_prior_registry.json"
PROBABILITY_FIELDS = [
    "calibration_version", "bayesian_model_version", "probability_model_sha256",
    "probability_success_definition", "prior_prob", "prior_logodds",
    "global_prior_prob", "prior_segment_posterior_prob", "prior_sample_count",
    "prior_independent_days", "prior_confidence",
    "prior_registry_key", "prior_source", "raw_probability", "calibrated_probability",
    "score_evidence_z", "score_log_likelihood_ratio", "configured_lr_log_update", "configured_lr_names",
    "posterior_prob", "bayes_posterior_prob",
    "expected_win", "expected_loss", "estimated_round_trip_cost", "breakeven_prob",
    "expected_value_shadow", "bayes_expected_value_shadow", "probability_confidence",
    "bayes_posterior_research_only", "bayes_posterior_allowed_for_trade_gate",
    "probability_shadow_action",
]
PROBABILITY_OUTCOME_FIELDS = [
    "outcome_horizon_complete", "outcome_completed_at", "probability_outcome",
    "brier_score", "probability_log_loss", "bayes_brier_score", "bayes_log_loss",
]

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


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def load_research_config(path: Path = RESEARCH_CONFIG_PATH) -> dict[str, Any] | None:
    config = load_json_object(path)
    if not config:
        return None
    safe = (
        config.get("schemaVersion") == "decision_probability_forward_research_v1"
        and config.get("recordOnly") is True
        and config.get("tradeGateEnabled") is False
        and config.get("positionSizingEnabled") is False
        and config.get("bayesPosteriorAllowedForTradeGate") is False
    )
    return config if safe else None


def classify_strategy_type(reason: Any) -> str:
    text = str(reason or "").lower()
    if any(word in text for word in ("pullback", "reversion", "oversold", "low_buy", "回踩", "低吸")):
        return "pullback_reversion"
    if any(word in text for word in ("breakout", "orb", "bollinger", "突破")):
        return "breakout"
    if any(word in text for word in ("momentum", "trend", "动量", "趋势")):
        return "momentum_trend"
    if any(word in text for word in ("sell", "exit", "stop", "止损", "止盈")):
        return "exit"
    return "unclassified"


def classify_symbol_group(code: Any, name: Any) -> str:
    text = str(name or "").lower()
    code_text = str(code or "").zfill(6)
    if code_text.startswith("513") or any(word in text for word in (
            "纳指", "标普", "恒生", "港股", "日经", "德国", "法国", "海外", "qdii")):
        return "cross_border"
    if any(word in text for word in ("黄金", "有色", "煤炭", "石油", "原油", "商品", "资源")):
        return "resource_commodity"
    if any(word in text for word in ("芯片", "半导体", "科技", "人工智能", "ai", "软件", "机器人", "创新药")):
        return "technology_growth"
    if any(word in text for word in ("银行", "证券", "保险", "红利", "央企", "公用事业")):
        return "financial_defensive"
    if any(word in text for word in ("沪深300", "上证", "中证500", "中证1000", "创业板", "科创50", "宽基")):
        return "broad_index"
    return "sector_other"


def prior_registry_key(context: dict[str, Any]) -> str:
    dimensions = ("market_regime", "strategy_type", "holding_horizon", "symbol_group", "signal_direction")
    return "|".join(f"{field}={str(context.get(field) or 'unknown')}" for field in dimensions)


def research_prior(context: dict[str, Any], fallback: float,
                   registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry if registry is not None else load_json_object(PRIOR_REGISTRY_PATH)
    key = prior_registry_key(context)
    segment = (registry or {}).get("jointSegments", {}).get(key)
    if not isinstance(segment, dict):
        return {"probability": fallback, "sample_count": 0, "independent_days": 0,
                "segment_probability": fallback, "confidence": 0.0, "key": key,
                "source": "frozen_global_prior"}
    segment_probability = clamp_probability(as_float(segment.get("posteriorProbability"), fallback))
    confidence = max(0.0, min(1.0, as_float(segment.get("confidence"), 0.0)))
    return {
        "probability": clamp_probability(fallback + confidence * (segment_probability - fallback)),
        "segment_probability": segment_probability,
        "sample_count": int(as_float(segment.get("count"), 0)),
        "independent_days": int(as_float(segment.get("independentDays"), 0)),
        "confidence": confidence,
        "key": key,
        "source": "forward_beta_binomial_joint_segment",
    }


def configured_lr_update(context: dict[str, Any], config: dict[str, Any] | None) -> tuple[float, list[str]]:
    evidence = ((config or {}).get("bayesianPrior", {}) or {}).get("configuredLikelihoodRatios", [])
    total = 0.0
    applied: list[str] = []
    for item in evidence if isinstance(evidence, list) else []:
        if not isinstance(item, dict):
            continue
        conditions = item.get("when", {}) if isinstance(item.get("when"), dict) else {}
        if any(str(context.get(key)) != str(value) for key, value in conditions.items()):
            continue
        lr = max(as_float(item.get("likelihoodRatio"), 1.0), 1e-6)
        discount = max(0.0, min(1.0, as_float(item.get("discount"), 1.0)))
        total += discount * math.log(lr)
        applied.append(str(item.get("name") or "configured_lr"))
    return total, applied


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
            and model.get("bayesPosteriorResearchOnly") is True
            and model.get("bayesPosteriorAllowedForTradeGate") is False
        )
        loaded = model if safe else None
        _MODEL_CACHE = (cache_key, mtime, loaded)
        return loaded
    except Exception:
        return None


def forecast_shadow(*, total_score: float, decision_type: str, date: str,
                    context: dict[str, Any] | None = None,
                    model: dict[str, Any] | None = None,
                    research_config: dict[str, Any] | None = None,
                    prior_registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a preregistered BUY probability forecast; never return a trade action."""
    model = model or load_shadow_model()
    context = context or {}
    signal_direction = str(context.get("signal_direction") or decision_type).upper()
    if not model or signal_direction != "BUY":
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
    raw_probability = sigmoid(logit(prior) + log_lr)
    # DCAL-1.0.1 is diagnostics-only: no new Platt/isotonic transform is fitted or applied.
    calibrated_probability = raw_probability
    research_config = research_config or load_research_config()
    selected_prior = research_prior(context, prior, registry=prior_registry)
    configured_log_update, applied_lrs = configured_lr_update(context, research_config)
    bayes_posterior = sigmoid(
        logit(selected_prior["probability"]) + log_lr + configured_log_update
    )
    payoff = model.get("payoff", {})
    expected_win = max(0.0, as_float(payoff.get("expectedGrossWin")))
    expected_loss = max(0.0, as_float(payoff.get("expectedGrossLoss")))
    cost = max(0.0, as_float(payoff.get("roundTripCost")))
    denominator = expected_win + expected_loss
    breakeven = (expected_loss + cost) / denominator if denominator > 0 else 1.0
    ev = calibrated_probability * expected_win - (1.0 - calibrated_probability) * expected_loss - cost
    bayes_ev = bayes_posterior * expected_win - (1.0 - bayes_posterior) * expected_loss - cost
    return {
        "calibration_version": model.get("calibrationVersion"),
        "bayesian_model_version": model.get("bayesianModelVersion"),
        "probability_model_sha256": model_sha256(model),
        "probability_success_definition": model.get("successDefinition"),
        "prior_prob": round(selected_prior["probability"], 8),
        "prior_logodds": round(logit(selected_prior["probability"]), 8),
        "global_prior_prob": round(prior, 8),
        "prior_segment_posterior_prob": round(selected_prior["segment_probability"], 8),
        "prior_sample_count": selected_prior["sample_count"],
        "prior_independent_days": selected_prior["independent_days"],
        "prior_confidence": round(selected_prior["confidence"], 8),
        "prior_registry_key": selected_prior["key"],
        "prior_source": selected_prior["source"],
        "raw_probability": round(raw_probability, 8),
        "calibrated_probability": round(calibrated_probability, 8),
        "score_evidence_z": round(z_score, 8),
        "score_log_likelihood_ratio": round(log_lr, 8),
        "configured_lr_log_update": round(configured_log_update, 8),
        "configured_lr_names": applied_lrs,
        "posterior_prob": round(calibrated_probability, 8),
        "bayes_posterior_prob": round(bayes_posterior, 8),
        "expected_win": round(expected_win, 8),
        "expected_loss": round(expected_loss, 8),
        "estimated_round_trip_cost": round(cost, 8),
        "breakeven_prob": round(breakeven, 8),
        "expected_value_shadow": round(ev, 8),
        "bayes_expected_value_shadow": round(bayes_ev, 8),
        "probability_confidence": round(as_float(model.get("modelConfidence")), 8),
        "bayes_posterior_research_only": True,
        "bayes_posterior_allowed_for_trade_gate": False,
        "probability_shadow_action": "RECORD_ONLY_NO_TRADE_GATE",
    }


def enrich_probability_outcome(record: dict[str, Any], *, outcome_return: float | None = None,
                               completed_at: str | None = None) -> dict[str, Any]:
    """Attach outcome/error only after prices exist; success means net return > 0."""
    realized = record.get("realized_return") if outcome_return is None else outcome_return
    probability_value = record.get("calibrated_probability", record.get("posterior_prob"))
    if probability_value is None or realized is None:
        return record
    cost = max(0.0, as_float(record.get("estimated_round_trip_cost")))
    outcome = 1 if as_float(realized) - cost > 0 else 0
    probability = clamp_probability(as_float(probability_value, 0.5))
    bayes_probability = clamp_probability(as_float(record.get("bayes_posterior_prob"), probability))
    record["outcome_horizon_complete"] = True
    record["outcome_completed_at"] = completed_at
    record["probability_outcome"] = outcome
    record["brier_score"] = round((probability - outcome) ** 2, 8)
    record["probability_log_loss"] = round(
        -(outcome * math.log(probability) + (1 - outcome) * math.log(1.0 - probability)), 8
    )
    record["bayes_brier_score"] = round((bayes_probability - outcome) ** 2, 8)
    record["bayes_log_loss"] = round(
        -(outcome * math.log(bayes_probability) + (1 - outcome) * math.log(1.0 - bayes_probability)), 8
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
                        *, bin_count: int = 5,
                        bin_edges: Iterable[float] | None = None) -> dict[str, Any]:
    probs = [clamp_probability(value) for value in probabilities]
    ys = [int(value) for value in outcomes]
    if len(probs) != len(ys) or not probs:
        return {"count": 0, "brier": None, "log_loss": None, "auc": None,
                "ece": None, "mce": None, "mean_predicted": None,
                "actual_rate": None, "bins": []}
    edges = [float(value) for value in bin_edges] if bin_edges is not None else [
        index / bin_count for index in range(bin_count + 1)
    ]
    if len(edges) < 2 or edges[0] > 0 or edges[-1] < 1 or any(
            right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("bin edges must be increasing and cover [0,1]")
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        members = [(p, y) for p, y in zip(probs, ys)
                   if p >= low and (p < high or (index == len(edges) - 2 and p <= high))]
        if not members:
            continue
        predicted = sum(p for p, _ in members) / len(members)
        actual = sum(y for _, y in members) / len(members)
        calibration_error = actual - predicted
        absolute_error = abs(calibration_error)
        ece += len(members) / len(probs) * absolute_error
        mce = max(mce, absolute_error)
        brier = sum((p - y) ** 2 for p, y in members) / len(members)
        log_loss = -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                        for p, y in members) / len(members)
        bins.append({"low": low, "high": high, "count": len(members),
                     "mean_predicted": predicted, "actual_rate": actual,
                     "calibration_error": calibration_error,
                     "absolute_calibration_error": absolute_error,
                     "brier": brier, "log_loss": log_loss})
    return {
        "count": len(probs),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(probs),
        "log_loss": -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                         for p, y in zip(probs, ys)) / len(probs),
        "auc": auc_score(probs, ys),
        "ece": ece,
        "mce": mce,
        "mean_predicted": sum(probs) / len(probs),
        "actual_rate": sum(ys) / len(ys),
        "bins": bins,
    }
