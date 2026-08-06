"""Preregistered next-session explosion and downside research for A shares.

The signal is formed after day-t close.  A hypothetical position enters at the next
buyable open and exits at the following sellable open, the shortest executable stock
round trip in this research system.  Separate heads estimate strong gain, a
board-normalised price-limit touch, non-positive return and severe loss.  Every output is
research-only; this module imports no broker code and cannot create an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_pinball_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_two_stage as two_stage  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "perception_xalpha_nextday_explosion_v1.json"
)
SCHEMA_VERSION = "perception_xalpha_nextday_explosion_v1"
CODE_VERSION = "perception_xalpha_nextday_explosion_v1.0"
LABEL_COLUMNS = {
    "target_executable_return",
    "target_next_session_mark_return",
    "label_strong_gain",
    "label_limit_touch",
    "label_non_positive",
    "label_severe_loss",
}


@dataclass
class ProbabilityHead:
    name: str
    label_column: str
    model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression | None
    prior: float
    enabled: bool
    audit: dict[str, Any]


@dataclass
class ExplosionModel:
    probability_heads: dict[str, ProbabilityHead]
    return_model: HistGradientBoostingRegressor
    return_calibrator: Ridge
    return_prior: float
    return_clip: tuple[float, float]
    return_enabled: bool
    tail_model: HistGradientBoostingRegressor
    tail_prior: float
    tail_enabled: bool
    feature_columns: list[str]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected next-session explosion schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("next-session explosion model must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading mutation permissions must remain false")
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("parametersFrozenBeforeHistoricalRun") is not True:
        raise ValueError("parameters must be frozen before the historical run")
    if hypothesis.get("validationAndShadowMayNotTuneParameters") is not True:
        raise ValueError("validation and shadow feedback must remain forbidden")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical research cannot promote")
    data = config["data"]
    if int(data["executableHoldingTradingDays"]) != 1:
        raise ValueError("the shortest executable holding period changed")
    if int(data["maximumLabelLookaheadTradingDays"]) != 2:
        raise ValueError("maximum label lookahead must remain two sessions")
    if int(data["candidatePoolSize"]) != 500:
        raise ValueError("the preregistered candidate pool must remain five hundred")
    if int(data["maximumSelectionsPerDay"]) != 3:
        raise ValueError("the strategy may select at most three names")
    if not math.isclose(float(data["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("A-share round-trip cost changed")
    if data.get("excludeST") is not True:
        raise ValueError("ST stocks must remain excluded")
    expected_limits = {"SH_MAIN": 0.10, "SZ_MAIN": 0.10, "STAR": 0.20, "CHINEXT": 0.20}
    if {key: float(value) for key, value in data["boardLimits"].items()} != expected_limits:
        raise ValueError("board price-limit map changed")
    weights = config["candidateGenerator"]["weights"]
    if not np.isclose(sum(map(float, weights.values())), 1.0):
        raise ValueError("candidate-generator weights must sum to one")
    model = config["model"]
    if int(model["baseToCalibrationPurgeTradingDays"]) < 2:
        raise ValueError("base-to-calibration purge is shorter than the label")
    if int(model["calibrationToAuditPurgeTradingDays"]) < 2:
        raise ValueError("calibration-to-audit purge is shorter than the label")
    if set(model["probabilityHeads"]) != {
        "strong_gain",
        "limit_touch",
        "non_positive",
        "severe_loss",
    }:
        raise ValueError("the four preregistered probability heads changed")
    features = set(config["featureSet"]["columns"])
    if features.intersection(LABEL_COLUMNS):
        raise ValueError("future outcomes entered the feature set")
    policy = config["selectionPolicy"]
    if policy.get("allowCash") is not True or policy.get("neverForceSelections") is not True:
        raise ValueError("selection must fail closed and allow cash")
    if int(policy["maximumSelectionsPerDay"]) != 3:
        raise ValueError("selection maximum changed")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must always be empty")


def board_limit_series(
    security_ids: pd.Index,
    config: dict[str, Any],
) -> pd.Series:
    limits = config["data"]["boardLimits"]
    values: list[float] = []
    for security_id in map(str, security_ids):
        exchange, code = security_id.split(".", 1)
        if exchange == "SH" and code.startswith(("688", "689")):
            board = "STAR"
        elif exchange == "SZ" and code.startswith(("300", "301")):
            board = "CHINEXT"
        elif exchange == "SH":
            board = "SH_MAIN"
        elif exchange == "SZ":
            board = "SZ_MAIN"
        else:
            raise ValueError(f"unsupported exchange in clean A-share panel: {security_id}")
        values.append(float(limits[board]))
    return pd.Series(values, index=security_ids, dtype=float)


def build_outcome_frames(
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Build future labels only; callers must never merge these back into feature frames."""
    close, open_price = panel["close"], panel["open"]
    entry = open_price.shift(-1)
    exit_price = open_price.shift(-2)
    executable = exit_price.div(entry.replace(0.0, np.nan)) - 1.0
    next_mark = close.shift(-1).div(entry.replace(0.0, np.nan)) - 1.0
    next_high_from_signal_close = panel["high"].shift(-1).div(
        close.replace(0.0, np.nan)
    ) - 1.0
    limit_by_symbol = board_limit_series(close.columns, config)
    tolerance = float(config["data"]["limitTouchTolerance"])
    limit_touch = next_high_from_signal_close.ge(limit_by_symbol - tolerance, axis=1)
    buyable, sellable = autonomous.tradability_frames(panel)
    eligible = panel["eligible"]
    valid = (
        eligible
        & eligible.shift(-1).eq(True)
        & buyable.shift(-1).eq(True)
        & sellable.shift(-2).eq(True)
        & executable.notna()
    )
    executable = executable.where(valid)
    next_mark = next_mark.where(valid)
    output = {
        "target_executable_return": executable,
        "target_next_session_mark_return": next_mark,
        "label_strong_gain": executable.ge(
            float(config["data"]["strongGainThreshold"])
        ).where(valid),
        "label_limit_touch": limit_touch.where(valid),
        "label_non_positive": executable.le(0.0).where(valid),
        "label_severe_loss": executable.le(
            float(config["data"]["severeLossThreshold"])
        ).where(valid),
    }
    return {
        key: value.astype(float).replace([np.inf, -np.inf], np.nan)
        for key, value in output.items()
    }


def _range_position(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    low = frame.rolling(window, min_periods=max(3, window // 2)).min()
    high = frame.rolling(window, min_periods=max(3, window // 2)).max()
    return frame.sub(low).div(high.sub(low).replace(0.0, np.nan))


def build_past_only_features(
    panel: dict[str, pd.DataFrame],
    factor_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    factor_ranks = two_stage.build_factor_rank_frames(panel, factor_config)
    base, market = two_stage.build_past_only_feature_frames(
        panel,
        factor_ranks,
        factor_config.get("frozenFactorWeights"),
    )
    close, returns, amount = panel["close"], panel["returns"], panel["amount"]
    previous_amount = amount.shift(1).rolling(20, min_periods=10).mean()
    burst: dict[str, pd.DataFrame] = {
        "burst_return_1": close.div(close.shift(1).replace(0.0, np.nan)) - 1.0,
        "burst_return_3": close.div(close.shift(3).replace(0.0, np.nan)) - 1.0,
        "burst_return_5": close.div(close.shift(5).replace(0.0, np.nan)) - 1.0,
        "burst_return_10": close.div(close.shift(10).replace(0.0, np.nan)) - 1.0,
        "burst_return_20": close.div(close.shift(20).replace(0.0, np.nan)) - 1.0,
        "burst_return_60": close.div(close.shift(60).replace(0.0, np.nan)) - 1.0,
        "burst_volatility_10": returns.rolling(10, min_periods=5).std(),
        "burst_volatility_20": returns.rolling(20, min_periods=10).std(),
        "burst_amount_ratio_1_20": amount.div(previous_amount.replace(0.0, np.nan)),
        "burst_amount_ratio_3_20": amount.rolling(3, min_periods=2)
        .mean()
        .div(previous_amount.replace(0.0, np.nan)),
        "burst_intraday_return": close.div(panel["open"].replace(0.0, np.nan)) - 1.0,
        "burst_open_gap": panel["open"].div(close.shift(1).replace(0.0, np.nan)) - 1.0,
        "burst_range_position_20": _range_position(close, 20),
        "burst_range_position_60": _range_position(close, 60),
        "burst_up_fraction_5": returns.gt(0.0).rolling(5, min_periods=3).mean(),
    }
    eligible = panel["eligible"]
    for name, frame in burst.items():
        base[name] = frame.where(eligible).replace([np.inf, -np.inf], np.nan)
    missing = set(config["featureSet"]["columns"]) - (set(base) | set(market))
    if missing:
        raise KeyError(f"missing preregistered explosion features: {sorted(missing)}")
    return base, market


def candidate_score(
    features: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score = panel["close"] * np.nan
    score.loc[:, :] = 0.0
    available = panel["close"] * 0.0
    for name, weight in config["candidateGenerator"]["weights"].items():
        ranked = features[name].where(panel["eligible"]).rank(axis=1, pct=True)
        score = score.add(ranked.fillna(0.0) * float(weight), fill_value=0.0)
        available = available.add(ranked.notna().astype(float) * float(weight), fill_value=0.0)
    score = score.div(available.replace(0.0, np.nan)).where(panel["eligible"])
    rank = score.rank(axis=1, ascending=False, method="first")
    mask = rank.le(int(config["data"]["candidatePoolSize"]))
    return score, rank, mask


def build_candidate_table(
    panel: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
    market: dict[str, pd.Series],
    outcomes: dict[str, pd.DataFrame],
    score: pd.DataFrame,
    rank: pd.DataFrame,
    mask: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    index = rank.where(mask).stack().index
    index.names = ["date", "securityId"]
    table = pd.DataFrame(index=index)
    table["candidate_score"] = score.where(mask).stack().reindex(index)
    table["candidate_rank"] = rank.where(mask).stack().reindex(index)
    for name in config["featureSet"]["columns"]:
        if name in features:
            table[name] = features[name].where(mask).stack().reindex(index)
    dates = table.index.get_level_values("date")
    for name, series in market.items():
        if name in config["featureSet"]["columns"]:
            table[name] = series.reindex(dates).to_numpy(dtype=float)
    for name, frame in outcomes.items():
        table[name] = frame.where(mask).stack().reindex(index)
    return table.reset_index()


def nested_segments(
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(train_dates)))
    model = config["model"]
    calibration_days = int(model["calibrationTradingDays"])
    audit_days = int(model["reliabilityAuditTradingDays"])
    base_gap = int(model["baseToCalibrationPurgeTradingDays"])
    audit_gap = int(model["calibrationToAuditPurgeTradingDays"])
    audit_start = len(dates) - audit_days
    calibration_end = audit_start - audit_gap
    calibration_start = calibration_end - calibration_days
    base_end = calibration_start - base_gap
    if base_end < int(model["minimumBaseFitTradingDays"]):
        raise RuntimeError("nested base-fit segment is too short")
    base = dates[:base_end]
    calibration = dates[calibration_start:calibration_end]
    audit = dates[audit_start:]
    if not (base[-1] < calibration[0] < calibration[-1] < audit[0]):
        raise RuntimeError("nested segments are not strictly chronological")
    return base, calibration, audit, {
        "baseFitDateRange": [base[0].date().isoformat(), base[-1].date().isoformat()],
        "baseFitTradingDays": len(base),
        "baseToCalibrationPurgeTradingDays": base_gap,
        "calibrationDateRange": [
            calibration[0].date().isoformat(),
            calibration[-1].date().isoformat(),
        ],
        "calibrationTradingDays": len(calibration),
        "calibrationToAuditPurgeTradingDays": audit_gap,
        "reliabilityAuditDateRange": [
            audit[0].date().isoformat(),
            audit[-1].date().isoformat(),
        ],
        "reliabilityAuditTradingDays": len(audit),
    }


def _classifier(config: dict[str, Any]) -> HistGradientBoostingClassifier:
    spec = config["model"]["classifier"]
    return HistGradientBoostingClassifier(
        learning_rate=float(spec["learningRate"]),
        max_iter=int(spec["maxIter"]),
        max_leaf_nodes=int(spec["maxLeafNodes"]),
        min_samples_leaf=int(spec["minSamplesLeaf"]),
        l2_regularization=float(spec["l2Regularization"]),
        random_state=int(spec["randomState"]),
    )


def _regressor(config: dict[str, Any], tail: bool = False) -> HistGradientBoostingRegressor:
    spec = config["model"]["tailRegressor" if tail else "returnRegressor"]
    kwargs: dict[str, Any] = {
        "loss": str(spec["loss"]),
        "learning_rate": float(spec["learningRate"]),
        "max_iter": int(spec["maxIter"]),
        "max_leaf_nodes": int(spec["maxLeafNodes"]),
        "min_samples_leaf": int(spec["minSamplesLeaf"]),
        "l2_regularization": float(spec["l2Regularization"]),
        "random_state": int(spec["randomState"]),
    }
    if tail:
        kwargs["quantile"] = float(spec["quantile"])
    return HistGradientBoostingRegressor(**kwargs)


def probability_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid].astype(int), np.clip(p[valid], 1e-8, 1.0 - 1e-8)
    if len(y) == 0:
        return {"n": 0, "events": 0, "eventRate": None, "brier": None, "logLoss": None, "auc": None, "ece": None}
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for value in range(bins):
        member = bucket == value
        if member.any():
            ece += float(member.mean()) * abs(float(p[member].mean()) - float(y[member].mean()))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {
        "n": len(y),
        "events": int(y.sum()),
        "eventRate": round(float(y.mean()), 8),
        "meanProbability": round(float(p.mean()), 8),
        "brier": round(float(brier_score_loss(y, p)), 8),
        "logLoss": round(float(log_loss(y, p, labels=[0, 1])), 8),
        "auc": round(auc, 8) if auc is not None else None,
        "ece": round(float(ece), 8),
    }


def _probability_reliability(
    name: str,
    labels: pd.Series,
    raw: np.ndarray,
    calibrated: np.ndarray,
    prior: float,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    raw_metrics = probability_metrics(labels, raw)
    calibrated_metrics = probability_metrics(labels, calibrated)
    prior_metrics = probability_metrics(labels, np.full(len(labels), prior))
    gate = config["reliabilityGate"]
    checks = {
        "minimumEvents": calibrated_metrics["events"]
        >= int(gate["minimumAuditEvents"][name]),
        "aucPass": raw_metrics["auc"] is not None
        and raw_metrics["auc"] >= float(gate["minimumProbabilityAuditAuc"]),
        "brierBetterThanPrior": calibrated_metrics["brier"] is not None
        and calibrated_metrics["brier"] < prior_metrics["brier"],
        "logLossBetterThanPrior": calibrated_metrics["logLoss"] is not None
        and calibrated_metrics["logLoss"] < prior_metrics["logLoss"],
    }
    enabled = all(checks.values())
    return enabled, {
        "head": name,
        "enabled": enabled,
        "checks": checks,
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "constantPrior": prior_metrics,
    }


def _return_reliability(
    rows: pd.DataFrame,
    predicted: np.ndarray,
    prior: float,
    slope: float,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    actual = rows["target_executable_return"].to_numpy(dtype=float)
    denominator = float(np.sum((actual - prior) ** 2))
    oos_r2 = 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator if denominator > 0 else None
    scored = rows[["date", "target_executable_return"]].copy()
    scored["prediction"] = predicted
    daily_ic: list[float] = []
    daily_direction: list[float] = []
    for _, group in scored.groupby("date", sort=True):
        if len(group) >= 20 and group["prediction"].nunique() > 1:
            value = group["prediction"].corr(group["target_executable_return"], method="spearman")
            if pd.notna(value):
                daily_ic.append(float(value))
        daily_direction.append(
            float((group["prediction"].mean() > 0.0) == (group["target_executable_return"].mean() > 0.0))
        )
    ic_t = autonomous.newey_west_t(np.asarray(daily_ic), 2) if len(daily_ic) >= 3 else None
    direction = float(np.mean(daily_direction)) if daily_direction else None
    gate = config["reliabilityGate"]
    checks = {
        "positiveOosR2": oos_r2 is not None
        and oos_r2 > float(gate["minimumReturnAuditOosR2Exclusive"]),
        "rankIcHacPass": ic_t is not None
        and ic_t >= float(gate["minimumReturnAuditRankIcHacT"]),
        "directionAccuracyPass": direction is not None
        and direction >= float(gate["minimumReturnAuditDirectionAccuracy"]),
        "positiveCalibrationSlope": slope > 0.0,
    }
    enabled = all(checks.values())
    return enabled, {
        "enabled": enabled,
        "checks": checks,
        "n": len(rows),
        "days": int(rows["date"].nunique()),
        "oosR2VsPrior": round(float(oos_r2), 8) if oos_r2 is not None else None,
        "rankIcMean": round(float(np.mean(daily_ic)), 8) if daily_ic else None,
        "rankIcHacT": round(float(ic_t), 4) if ic_t is not None else None,
        "basketDirectionAccuracy": round(direction, 8) if direction is not None else None,
        "calibrationSlope": round(float(slope), 8),
        "predictedMean": round(float(np.mean(predicted)), 8),
        "actualMean": round(float(np.mean(actual)), 8),
    }


def fit_model(
    table: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> ExplosionModel:
    base_dates, calibration_dates, audit_dates, segment_audit = nested_segments(train_dates, config)
    features = list(config["featureSet"]["columns"])
    labelled = table[table["target_executable_return"].notna()].copy()
    base = labelled[labelled["date"].isin(base_dates)]
    calibration = labelled[labelled["date"].isin(calibration_dates)]
    audit = labelled[labelled["date"].isin(audit_dates)]
    if base.empty or calibration.empty or audit.empty:
        raise RuntimeError("base, calibration or reliability-audit rows are empty")
    x_base = base[features].replace([np.inf, -np.inf], np.nan)
    x_calibration = calibration[features].replace([np.inf, -np.inf], np.nan)
    x_audit = audit[features].replace([np.inf, -np.inf], np.nan)
    head_to_label = {
        name: f"label_{name}" for name in config["model"]["probabilityHeads"]
    }
    heads: dict[str, ProbabilityHead] = {}
    probability_audits: dict[str, Any] = {}
    for name in config["model"]["probabilityHeads"]:
        label = head_to_label[name]
        if label not in labelled:
            raise KeyError(f"probability head {name} is missing its offline label {label}")
        y_base = base[label].astype(int)
        if y_base.nunique() != 2:
            raise RuntimeError(f"probability head {name} requires both classes")
        prior = float(y_base.mean())
        positive_weight = min(
            float(config["model"]["classifier"]["maximumPositiveClassWeight"]),
            (1.0 - prior) / max(prior, 1e-8),
        )
        sample_weight = np.where(y_base.to_numpy() == 1, positive_weight, 1.0)
        model = _classifier(config)
        model.fit(x_base, y_base, sample_weight=sample_weight)
        raw_calibration = model.predict_proba(x_calibration)[:, 1]
        calibrator: IsotonicRegression | None = None
        if calibration[label].nunique() == 2 and np.unique(raw_calibration).size > 1:
            calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            calibrator.fit(raw_calibration, calibration[label].astype(int))
        raw_audit = model.predict_proba(x_audit)[:, 1]
        calibrated_audit = (
            calibrator.predict(raw_audit)
            if calibrator is not None
            else np.full(len(raw_audit), prior)
        )
        enabled, head_audit = _probability_reliability(
            name,
            audit[label].astype(int),
            raw_audit,
            calibrated_audit,
            prior,
            config,
        )
        head_audit.update(
            {
                "baseRows": len(base),
                "calibrationRows": len(calibration),
                "auditRows": len(audit),
                "basePrior": round(prior, 8),
                "positiveClassWeight": round(float(positive_weight), 8),
                "fallbackWhenDisabled": "constant_base_fit_prior",
            }
        )
        probability_audits[name] = head_audit
        heads[name] = ProbabilityHead(name, label, model, calibrator, prior, enabled, head_audit)

    y_return = base["target_executable_return"].astype(float)
    lower_q, upper_q = map(float, config["model"]["targetWinsorQuantiles"])
    lower, upper = map(float, y_return.quantile([lower_q, upper_q]).tolist())
    return_model = _regressor(config)
    return_model.fit(x_base, y_return.clip(lower, upper))
    raw_calibration_return = return_model.predict(x_calibration)
    return_calibrator = Ridge(alpha=float(config["model"]["returnCalibrationAlpha"]))
    return_calibrator.fit(
        (raw_calibration_return * 100.0).reshape(-1, 1),
        calibration["target_executable_return"].clip(lower, upper),
    )
    slope = float(return_calibrator.coef_[0]) * 100.0
    raw_audit_return = return_model.predict(x_audit)
    calibrated_audit_return = np.clip(
        return_calibrator.predict((raw_audit_return * 100.0).reshape(-1, 1)),
        lower,
        upper,
    )
    return_prior = float(y_return.mean())
    return_enabled, return_audit = _return_reliability(
        audit,
        calibrated_audit_return,
        return_prior,
        slope,
        config,
    )
    return_audit.update(
        {
            "baseRows": len(base),
            "calibrationRows": len(calibration),
            "auditRows": len(audit),
            "basePrior": round(return_prior, 8),
            "targetClip": [lower, upper],
            "fallbackWhenDisabled": "constant_base_fit_prior",
        }
    )

    tail_model = _regressor(config, tail=True)
    tail_model.fit(x_base, y_return.clip(lower, upper))
    tail_prediction = np.clip(tail_model.predict(x_audit), lower, upper)
    tail_prior = float(y_return.quantile(float(config["model"]["tailRegressor"]["quantile"])))
    actual_audit = audit["target_executable_return"].to_numpy(dtype=float)
    quantile = float(config["model"]["tailRegressor"]["quantile"])
    coverage = float(np.mean(actual_audit <= tail_prediction))
    model_pinball = float(mean_pinball_loss(actual_audit, tail_prediction, alpha=quantile))
    prior_pinball = float(
        mean_pinball_loss(actual_audit, np.full(len(actual_audit), tail_prior), alpha=quantile)
    )
    gate = config["reliabilityGate"]
    tail_checks = {
        "coveragePass": float(gate["tailAuditMinimumCoverage"])
        <= coverage
        <= float(gate["tailAuditMaximumCoverage"]),
        "pinballBetterThanPrior": model_pinball < prior_pinball,
    }
    tail_enabled = all(tail_checks.values())
    tail_audit = {
        "enabled": tail_enabled,
        "checks": tail_checks,
        "quantile": quantile,
        "auditCoverage": round(coverage, 8),
        "auditPinballLoss": round(model_pinball, 8),
        "constantPriorPinballLoss": round(prior_pinball, 8),
        "basePriorQuantile": round(tail_prior, 8),
        "fallbackWhenDisabled": "constant_base_fit_quantile",
    }
    audit_payload = {
        **segment_audit,
        "featureCount": len(features),
        "baseRows": len(base),
        "calibrationRows": len(calibration),
        "auditRows": len(audit),
        "probabilityHeads": probability_audits,
        "returnHead": return_audit,
        "tailHead": tail_audit,
    }
    return ExplosionModel(
        probability_heads=heads,
        return_model=return_model,
        return_calibrator=return_calibrator,
        return_prior=return_prior,
        return_clip=(lower, upper),
        return_enabled=return_enabled,
        tail_model=tail_model,
        tail_prior=tail_prior,
        tail_enabled=tail_enabled,
        feature_columns=features,
        audit=audit_payload,
    )


def score_rows(model: ExplosionModel, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    x = rows[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    for name, head in model.probability_heads.items():
        raw = head.model.predict_proba(x)[:, 1]
        calibrated = (
            head.calibrator.predict(raw)
            if head.calibrator is not None
            else np.full(len(raw), head.prior)
        )
        effective = calibrated if head.enabled else np.full(len(raw), head.prior)
        output[f"raw_probability_{name}"] = raw
        output[f"calibrated_probability_{name}"] = calibrated
        output[f"effective_probability_{name}"] = effective
        output[f"head_enabled_{name}"] = head.enabled
    raw_return = model.return_model.predict(x)
    calibrated_return = np.clip(
        model.return_calibrator.predict((raw_return * 100.0).reshape(-1, 1)),
        *model.return_clip,
    )
    output["raw_expected_executable_return"] = raw_return
    output["calibrated_expected_executable_return"] = calibrated_return
    output["effective_expected_executable_return"] = (
        calibrated_return if model.return_enabled else model.return_prior
    )
    output["head_enabled_expected_return"] = model.return_enabled
    raw_tail = np.clip(model.tail_model.predict(x), *model.return_clip)
    output["raw_predicted_tenth_percentile_return"] = raw_tail
    output["effective_predicted_tenth_percentile_return"] = (
        raw_tail if model.tail_enabled else model.tail_prior
    )
    output["head_enabled_tail"] = model.tail_enabled
    return output


def candidate_recall(
    mask: pd.DataFrame,
    outcomes: dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    train_mask = pd.Series(mask.index.isin(train_dates), index=mask.index)
    result: dict[str, Any] = {}
    for name, label in (("strongGain", "label_strong_gain"), ("limitTouch", "label_limit_touch")):
        events = outcomes[label].eq(1.0).loc[train_mask]
        captured = events & mask.loc[train_mask]
        total = int(events.sum().sum())
        count = int(captured.sum().sum())
        result[name] = {
            "events": total,
            "captured": count,
            "recall": round(count / total, 8) if total else None,
        }
    generator = config["candidateGenerator"]
    result["checks"] = {
        "strongEventRecallPass": result["strongGain"]["recall"] is not None
        and result["strongGain"]["recall"] >= float(generator["minimumStrongEventRecall"]),
        "limitTouchRecallPass": result["limitTouch"]["recall"] is not None
        and result["limitTouch"]["recall"] >= float(generator["minimumLimitTouchRecall"]),
    }
    result["gatePassed"] = all(result["checks"].values())
    return result


def all_heads_ready(model: ExplosionModel) -> bool:
    return bool(
        model.return_enabled
        and model.tail_enabled
        and all(head.enabled for head in model.probability_heads.values())
    )


def selection_rows(
    rows: pd.DataFrame,
    config: dict[str, Any],
    model_ready: bool,
) -> pd.DataFrame:
    if rows.empty or not model_ready:
        return rows.iloc[0:0].copy()
    policy = config["selectionPolicy"]
    selected = rows[
        rows["effective_expected_executable_return"].ge(
            float(policy["minimumExpectedExecutableReturn"])
        )
        & rows["effective_probability_non_positive"].le(
            float(policy["maximumNonPositiveProbability"])
        )
        & rows["effective_probability_strong_gain"].ge(
            float(policy["minimumStrongGainProbability"])
        )
        & rows["effective_probability_limit_touch"].ge(
            float(policy["minimumLimitTouchProbability"])
        )
        & rows["effective_probability_severe_loss"].le(
            float(policy["maximumSevereLossProbability"])
        )
        & rows["effective_predicted_tenth_percentile_return"].ge(
            float(policy["minimumPredictedTenthPercentileReturn"])
        )
    ].copy()
    selected = selected.sort_values(
        [
            "date",
            "effective_expected_executable_return",
            "effective_probability_limit_touch",
            "effective_probability_strong_gain",
            "effective_probability_non_positive",
        ],
        ascending=[True, False, False, False, True],
    )
    return selected.groupby("date", sort=False).head(
        int(policy["maximumSelectionsPerDay"])
    )


def _outcome_summary(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    valid = rows[rows["target_executable_return"].notna()].copy()
    if valid.empty:
        return {
            "observations": 0,
            "signalDays": 0,
            "meanGrossReturn": None,
            "meanNetReturn": None,
            "winRate": None,
            "strongGainRate": None,
            "limitTouchRate": None,
            "severeLossRate": None,
            "cvar10": None,
        }
    values = valid["target_executable_return"].astype(float)
    tail_count = max(1, int(math.ceil(len(values) * 0.10)))
    return {
        "observations": len(valid),
        "signalDays": int(valid["date"].nunique()),
        "meanGrossReturn": round(float(values.mean()), 8),
        "meanNetReturn": round(float(values.mean() - config["data"]["roundTripCost"]), 8),
        "medianGrossReturn": round(float(values.median()), 8),
        "winRate": round(float(values.gt(0.0).mean()), 8),
        "nonPositiveRate": round(float(values.le(0.0).mean()), 8),
        "strongGainRate": round(float(valid["label_strong_gain"].mean()), 8),
        "limitTouchRate": round(float(valid["label_limit_touch"].mean()), 8),
        "severeLossRate": round(float(valid["label_severe_loss"].mean()), 8),
        "cvar10": round(float(values.nsmallest(tail_count).mean()), 8),
    }


def diagnostic_top(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    return rows.sort_values(
        [
            "date",
            "raw_expected_executable_return",
            "raw_probability_limit_touch",
            "raw_probability_strong_gain",
            "raw_probability_non_positive",
        ],
        ascending=[True, False, False, False, True],
    ).groupby("date", sort=False).head(n)


def period_report(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
    model_ready: bool,
) -> dict[str, Any]:
    rows = predictions[
        predictions["date"].isin(dates)
        & predictions["target_executable_return"].notna()
    ].copy()
    selected = selection_rows(rows, config, model_ready)
    diagnostic = diagnostic_top(rows, int(config["data"]["maximumSelectionsPerDay"]))
    heads: dict[str, Any] = {}
    mapping = {
        "strong_gain": "label_strong_gain",
        "limit_touch": "label_limit_touch",
        "non_positive": "label_non_positive",
        "severe_loss": "label_severe_loss",
    }
    for name, label in mapping.items():
        heads[name] = {
            "effective": probability_metrics(
                rows[label], rows[f"effective_probability_{name}"]
            ),
            "raw": probability_metrics(rows[label], rows[f"raw_probability_{name}"]),
        }
    return {
        "candidateRows": len(rows),
        "probabilityHeads": heads,
        "candidatePoolOutcome": _outcome_summary(rows, config),
        "rawDiagnosticTop3Outcome": _outcome_summary(diagnostic, config),
        "policySelectionOutcome": _outcome_summary(selected, config),
        "selectionCountReduction": round(
            1.0 - len(selected) / max(1, len(diagnostic)), 8
        ),
    }


def build_verdict(
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    model_ready = bool(report["modelReady"])
    gate = config["evaluation"]
    periods: dict[str, Any] = {}
    for name in ("validation", "shadow"):
        selected = report["periods"][name]["policySelectionOutcome"]
        pool = report["periods"][name]["candidatePoolOutcome"]
        checks = {
            "minimumSignalDays": selected["signalDays"]
            >= int(gate["minimumSignalDaysPerPeriod"]),
            "minimumObservations": selected["observations"]
            >= int(gate["minimumSelectedObservationsPerPeriod"]),
            "positiveGrossMean": selected["meanGrossReturn"] is not None
            and selected["meanGrossReturn"] >= float(gate["minimumGrossMeanExecutableReturn"]),
            "positiveNetMean": selected["meanNetReturn"] is not None
            and selected["meanNetReturn"] >= float(gate["minimumNetMeanExecutableReturn"]),
            "winRatePass": selected["winRate"] is not None
            and selected["winRate"] >= float(gate["minimumBasketWinRate"]),
            "lossRatePass": selected.get("nonPositiveRate") is not None
            and selected["nonPositiveRate"] <= float(gate["maximumObservedNonPositiveRate"]),
            "strongGainLift": selected["strongGainRate"] is not None
            and pool["strongGainRate"] is not None
            and selected["strongGainRate"] > pool["strongGainRate"],
            "limitTouchLift": selected["limitTouchRate"] is not None
            and pool["limitTouchRate"] is not None
            and selected["limitTouchRate"] > pool["limitTouchRate"],
        }
        periods[name] = {"checks": checks, "passed": all(checks.values())}
    historical_pass = bool(
        model_ready
        and report["candidateRecall"]["gatePassed"]
        and all(value["passed"] for value in periods.values())
    )
    return {
        "status": "research_only_not_eligible_for_trading",
        "modelReady": model_ready,
        "candidateRecallGatePassed": report["candidateRecall"]["gatePassed"],
        "periods": periods,
        "stableHistoricalHypothesis": historical_pass,
        "decision": (
            "keep_preregistered_forward_shadow_hypothesis"
            if historical_pass
            else "reject_for_trading_keep_diagnostics"
        ),
        "promotionAllowed": False,
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Next-session explosion V1",
        "",
        f"- Status: `{report['status']}`",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Model ready: `{report['modelReady']}`",
        f"- Candidate recall gate: `{report['candidateRecall']['gatePassed']}`",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Orders: `[]`",
        "",
        "## Label contract",
        "",
        "- Signal after day-t close; entry at next buyable open; exit at the following sellable open.",
        "- Strong gain: executable return >= 5%.",
        "- Limit touch: next-session high reaches the board-specific limit within tolerance.",
        "- Non-positive and <= -3% severe-loss heads are estimated separately.",
        "- A sealed-up entry or sealed-down exit is excluded rather than credited.",
        "",
        "## OOS diagnostics",
        "",
        "| Period | Policy obs | Days | Gross mean | Net mean | Win | Strong | Limit touch | Severe loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("validation", "shadow"):
        item = report["periods"][name]["policySelectionOutcome"]
        lines.append(
            f"| {name} | {item['observations']} | {item['signalDays']} | "
            f"{item['meanGrossReturn']} | {item['meanNetReturn']} | {item['winRate']} | "
            f"{item['strongGainRate']} | {item['limitTouchRate']} | {item['severeLossRate']} |"
        )
    lines.extend(["", "## Reliability heads", ""])
    for name, item in report["modelAudit"]["probabilityHeads"].items():
        lines.append(
            f"- {name}: enabled=`{item['enabled']}`, audit AUC=`{item['raw']['auc']}`, "
            f"Brier=`{item['calibrated']['brier']}` vs prior `{item['constantPrior']['brier']}`."
        )
    ret = report["modelAudit"]["returnHead"]
    tail = report["modelAudit"]["tailHead"]
    lines.extend(
        [
            f"- expected return: enabled=`{ret['enabled']}`, OOS R2=`{ret['oosR2VsPrior']}`, Rank-IC t=`{ret['rankIcHacT']}`.",
            f"- tenth percentile: enabled=`{tail['enabled']}`, coverage=`{tail['auditCoverage']}`, pinball=`{tail['auditPinballLoss']}`.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def _name_map(base_config: dict[str, Any]) -> dict[str, str]:
    path = ROOT / base_config["assetUniverse"]["masterPath"]
    names: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        names[str(row.get("securityId"))] = str(row.get("name") or "")
    return names


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    base = load_json(ROOT / config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    factor_config = load_json(ROOT / config["baseFactorConfig"])
    two_stage.validate_config(factor_config)
    features, market = build_past_only_features(panel, factor_config, config)
    outcomes = build_outcome_frames(panel, config)
    score, rank, mask = candidate_score(features, panel, config)
    table = build_candidate_table(
        panel, features, market, outcomes, score, rank, mask, config
    )
    split = autonomous.make_split(panel["close"].index, cog_config)
    train_dates = pd.DatetimeIndex(split.train[split.train].index)
    validation_dates = pd.DatetimeIndex(split.validation[split.validation].index)
    shadow_dates = pd.DatetimeIndex(split.shadow[split.shadow].index)
    recall = candidate_recall(mask, outcomes, train_dates, config)
    model = fit_model(table, train_dates, config)
    model_ready = bool(all_heads_ready(model) and recall["gatePassed"])
    external = table[
        table["date"].isin(validation_dates.union(shadow_dates))
        | table["date"].eq(table["date"].max())
    ].copy()
    predictions = score_rows(model, external)
    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_"
        + now.strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + digest({"config": config, "code": CODE_VERSION})[:10]
    )
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": now.isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": digest(config),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "candidateRecall": recall,
        "featureColumns": list(config["featureSet"]["columns"]),
        "labelDefinitions": {
            "target_executable_return": "open[t+2]/open[t+1]-1 after tradability masks",
            "strong_gain": f"target >= {config['data']['strongGainThreshold']}",
            "limit_touch": "high[t+1]/close[t]-1 reaches board limit within tolerance",
            "non_positive": "target <= 0",
            "severe_loss": f"target <= {config['data']['severeLossThreshold']}",
        },
        "modelAudit": model.audit,
        "modelReady": model_ready,
        "periods": {
            "validation": period_report(
                predictions, validation_dates, config, model_ready
            ),
            "shadow": period_report(predictions, shadow_dates, config, model_ready),
        },
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    report["verdict"] = build_verdict(report, config)
    latest_date = table["date"].max()
    latest = predictions[predictions["date"].eq(latest_date)].copy()
    latest_selected = selection_rows(latest, config, model_ready)
    latest_diagnostic = diagnostic_top(latest, 10).copy()
    names = _name_map(base)
    for frame in (latest_diagnostic, latest_selected):
        frame.insert(2, "name", frame["securityId"].map(names).fillna(""))
    output_root = ROOT / config["output"]["root"] / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    output_columns = [
        "date",
        "securityId",
        "name",
        "candidate_rank",
        "candidate_score",
        "raw_expected_executable_return",
        "calibrated_expected_executable_return",
        "effective_expected_executable_return",
        "raw_probability_strong_gain",
        "effective_probability_strong_gain",
        "raw_probability_limit_touch",
        "effective_probability_limit_touch",
        "raw_probability_non_positive",
        "effective_probability_non_positive",
        "raw_probability_severe_loss",
        "effective_probability_severe_loss",
        "raw_predicted_tenth_percentile_return",
        "effective_predicted_tenth_percentile_return",
    ]
    latest_diagnostic[output_columns].to_csv(
        output_root / "latest_diagnostic_top10.csv", index=False
    )
    latest_selected[output_columns].to_csv(
        output_root / "latest_selected_shadow.csv", index=False
    )
    atomic_write_text(
        output_root / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(output_root / "report.md", markdown_report(report))
    atomic_write_text(
        output_root / "model_manifest.json",
        json.dumps(
            {
                "schemaVersion": "perception_xalpha_nextday_explosion_model_manifest_v1",
                "status": "research_only_not_online_inference",
                "runId": run_id,
                "modelReady": model_ready,
                "modelAudit": model.audit,
                "selectionPolicy": config["selectionPolicy"],
                "orders": [],
                "automaticTradingChanges": [],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args.config.resolve(), args.run_id)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runId": report["runId"],
                "modelReady": report["modelReady"],
                "verdict": report["verdict"],
                "orders": report["orders"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
