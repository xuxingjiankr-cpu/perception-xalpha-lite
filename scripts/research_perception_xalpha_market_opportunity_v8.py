"""Research-only calibrated market-opportunity gate for the frozen V2 stock ranker.

V8 changes one thing only: the deterministic V2 market gate is replaced by a
time-isolated probability that the eligible A-share equal-weight basket will clear
the fixed ten-session round-trip cost.  The V2 cross-sectional stock scores remain
unchanged.  This module is offline diagnostics and can never place an order.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_two_stage as selector_v1  # noqa: E402
import research_perception_xalpha_winrate_tail_v3 as tail_v3  # noqa: E402
import research_perception_xalpha_winrate_v2 as winrate_v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_market_opportunity_v8.json"
)
SCHEMA_VERSION = "perception_xalpha_market_opportunity_v8"
CODE_VERSION = "perception_xalpha_market_opportunity_v8.0"
PRIMARY_POLICY = "v2_rank_top3_calibrated_market_opportunity"


@dataclass
class MarketOpportunityModel:
    pipeline: Pipeline | None
    calibrator: LogisticRegression | None
    feature_columns: list[str]
    training_prior: float
    enabled: bool
    reliability: dict[str, Any]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_research_data_override(
    override: dict[str, Any],
    base_v8_path: Path,
    base_research_path: Path,
) -> None:
    if override.get("schemaVersion") != (
        "perception_xalpha_pit_adjusted_robustness_override_v1"
    ):
        raise ValueError("unexpected PIT-adjusted robustness override schema")
    if override.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("the data robustness override must remain research-only")
    safety = override.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only" or any(
        value is not False
        for key, value in safety.items()
        if key.startswith("may")
    ):
        raise ValueError("the data robustness override received a trading permission")
    if (ROOT / override["baseV8Config"]).resolve() != base_v8_path.resolve():
        raise ValueError("the data override points to a different V8 config")
    if file_sha256(base_v8_path) != override["baseV8ConfigFileSha256"]:
        raise ValueError("the preregistered V8 config changed after the override")
    if (ROOT / override["baseResearchConfig"]).resolve() != base_research_path.resolve():
        raise ValueError("the data override points to a different base research config")
    if file_sha256(base_research_path) != override["baseResearchConfigFileSha256"]:
        raise ValueError("the frozen base research config changed after preregistration")
    hypothesis = override["preregisteredRobustnessHypothesis"]
    if hypothesis.get("onlyChange") != (
        "replace_current_master_raw_price_panel_with_pit_membership_adjusted_price_panel"
    ):
        raise ValueError("the robustness audit may change only its data panel")
    for flag in (
        "factorDefinitionsFrozen",
        "rankModelFrozen",
        "marketOpportunityModelFrozen",
        "probabilityCalibrationFrozen",
        "thresholdsFrozen",
        "walkForwardSplitsFrozen",
        "validationAndShadowMayNotTuneAnything",
    ):
        if hypothesis.get(flag) is not True:
            raise ValueError(f"the robustness preregistration flag changed: {flag}")
    if (
        hypothesis.get("historicalRunCanPromote") is not False
        or hypothesis.get("profitGuaranteeClaimAllowed") is not False
    ):
        raise ValueError("the robustness audit cannot promote or guarantee profit")
    universe = override["assetUniverse"]
    if (
        set(universe.get("exchanges", [])) != {"SH", "SZ"}
        or universe.get("requireAdjustedPrices") is not True
        or universe.get("requirePointInTimeStatus") is not True
        or universe.get("requirePointInTimeMaster") is not True
        or universe.get("failClosedUnlessUnbiasedHistoricalValidationEligible")
        is not True
    ):
        raise ValueError("the robustness universe no longer requires clean PIT data")
    quality = override["qualityGate"]
    if any(value is not True for value in quality.values()):
        raise ValueError("every preregistered robustness data gate must remain enabled")


def require_price_data_audit(override: dict[str, Any]) -> dict[str, Any]:
    data_config = load_json(
        ROOT / "configs" / "research" / "ashare_pit_adjusted_data_v1.json"
    )
    audit_path = ROOT / data_config["paths"]["auditRoot"] / "latest_data_audit.json"
    if not audit_path.exists():
        raise RuntimeError("PIT-adjusted price audit is missing")
    audit = load_json(audit_path)
    validate_price_data_audit_payload(audit)
    return audit


def validate_price_data_audit_payload(audit: dict[str, Any]) -> None:
    if audit.get("historicalResearchEligible") is not True:
        raise RuntimeError(
            "PIT-adjusted price audit is incomplete; model fitting is forbidden"
        )


def validate_corrected_panel_audit(panel_audit: dict[str, Any]) -> None:
    if panel_audit.get("unbiasedHistoricalValidationEligible") is not True:
        raise RuntimeError("the corrected panel failed its unbiased-data gate")
    if panel_audit.get("fundamentalAudit", {}).get(
        "historicalValidationEligible"
    ) is not True:
        raise RuntimeError("the corrected fundamental panel failed its PIT gate")


def validate_config(config: dict[str, Any], base: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected V8 market-opportunity schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V8 must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("V8 output must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all V8 trading mutation permissions must remain false")
    base_path = ROOT / config["baseWinrateConfig"]
    if file_sha256(base_path) != config["baseWinrateConfigFileSha256"]:
        raise ValueError("the frozen V2 base config file changed")
    winrate_v2.validate_config(base)
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("primaryPolicy") != PRIMARY_POLICY:
        raise ValueError("the V8 primary policy changed")
    if hypothesis.get("onlyIncrementVsV2") != (
        "replace_fixed_market_gate_with_calibrated_market_opportunity_probability"
    ):
        raise ValueError("V8 may change only the market-opportunity gate")
    for flag in (
        "v2RankPredictionsMustRemainIdentical",
        "parametersFrozenBeforeHistoricalRun",
        "validationAndShadowMayNotTuneParameters",
    ):
        if hypothesis.get(flag) is not True:
            raise ValueError(f"V8 preregistration flag changed: {flag}")
    if hypothesis.get("profitGuaranteeClaimAllowed") is not False:
        raise ValueError("V8 may not claim guaranteed profits")
    data = config["data"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("V8 horizon must remain ten trading days")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("V8 candidate pool must remain the V2 Top50")
    if int(data["maximumSelectionsPerDay"]) != 3:
        raise ValueError("V8 must remain a Top3-or-empty diagnostic")
    if not np.isclose(float(data["roundTripCost"]), 0.003):
        raise ValueError("V8 round-trip cost changed")
    walk = config["walkForward"]
    for key in (
        "trainingWindowTradingDays",
        "minimumTrainingTradingDays",
        "refitEveryTradingDays",
    ):
        if int(walk[key]) != int(base["rankModel"][key]):
            raise ValueError(f"V8 and frozen V2 rolling calendars differ: {key}")
    if int(walk["outerPurgeTradingDays"]) != int(base["rankModel"]["purgeTradingDays"]):
        raise ValueError("V8 and frozen V2 outer purges differ")
    if min(
        int(walk["outerPurgeTradingDays"]),
        int(walk["baseToCalibrationPurgeTradingDays"]),
        int(walk["calibrationToAuditPurgeTradingDays"]),
    ) < int(data["holdingTradingDays"]):
        raise ValueError("every V8 purge must cover the label horizon")
    model = config["marketOpportunityModel"]
    if (
        model.get("kind") != "LogisticRegression"
        or not np.isclose(float(model["C"]), 0.1)
        or model.get("oneObservationPerTradingDate") is not True
    ):
        raise ValueError("the preregistered V8 market model changed")
    if model.get("classWeight") is not None:
        raise ValueError("V8 class weighting was not preregistered")
    if market_feature_columns(config) != [
        "market_return_20",
        "market_return_60",
        "market_volatility_20",
        "market_breadth_20",
    ]:
        raise ValueError("the frozen V8 feature set changed")
    calibration = config["probabilityCalibration"]
    if calibration.get("kind") != "PlattLogisticRegression":
        raise ValueError("V8 calibration kind changed")
    if calibration.get("fitOnCalibrationSegmentOnly") is not True:
        raise ValueError("V8 calibrator must fit only the calibration segment")
    if calibration.get("auditSegmentNeverFitsModelOrCalibrator") is not True:
        raise ValueError("V8 audit must remain never-fit")
    selection = config["selection"]
    if not np.isclose(float(selection["marketOpportunityThresholdInclusive"]), 0.6):
        raise ValueError("V8 opportunity threshold changed")
    if selection.get("requiresReliabilityGate") is not True:
        raise ValueError("V8 cannot bypass reliability")
    if selection.get("allowZeroSelections") is not True or selection.get("neverFillToThree") is not True:
        raise ValueError("V8 must fail closed and never fill a quota")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("a historical V8 run cannot promote")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("V8 orders must remain empty")


def market_feature_columns(config: dict[str, Any]) -> list[str]:
    columns = list(config["marketOpportunityModel"]["featureColumns"])
    forbidden = {
        "target_return_10d",
        "benchmark_return_10d",
        "label_market_opportunity_10d",
        "target_cross_sectional_rank_10d",
    }
    overlap = forbidden.intersection(columns)
    if overlap:
        raise ValueError(f"future outcomes entered V8 features: {sorted(overlap)}")
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("V8 feature list is empty or duplicated")
    return columns


def nested_segments(
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, dict[str, Any]]:
    walk = config["walkForward"]
    calibration_days = int(walk["calibrationTradingDays"])
    audit_days = int(walk["reliabilityAuditTradingDays"])
    base_gap = int(walk["baseToCalibrationPurgeTradingDays"])
    audit_gap = int(walk["calibrationToAuditPurgeTradingDays"])
    audit_start = len(fit_dates) - audit_days
    calibration_end = audit_start - audit_gap
    calibration_start = calibration_end - calibration_days
    base_end = calibration_start - base_gap
    if base_end < int(walk["minimumBaseFitTradingDays"]):
        raise RuntimeError("V8 nested base-fit segment is too short")
    base = fit_dates[:base_end]
    calibration = fit_dates[calibration_start:calibration_end]
    audit = fit_dates[audit_start:]
    if not (base[-1] < calibration[0] < calibration[-1] < audit[0]):
        raise RuntimeError("nested V8 segments are not strictly chronological")
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
        "auditDateRange": [audit[0].date().isoformat(), audit[-1].date().isoformat()],
        "auditTradingDays": len(audit),
    }


def prepare_market_table(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = market_feature_columns(config)
    required = ["date", "securityId", "benchmark_return_10d", *columns]
    missing = [name for name in required if name not in table]
    if missing:
        raise KeyError(f"missing V8 market table columns: {missing}")
    consistency = table.groupby("date", sort=False)[
        ["benchmark_return_10d", *columns]
    ].nunique(dropna=False)
    if bool(consistency.gt(1).any(axis=None)):
        raise ValueError("V8 market fields are not constant within a date")
    output = (
        table.sort_values(["date", "securityId"])
        .groupby("date", sort=True)
        .head(1)[required]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    output["label_market_opportunity_10d"] = np.where(
        output["benchmark_return_10d"].notna(),
        output["benchmark_return_10d"].gt(float(config["data"]["roundTripCost"])).astype(float),
        np.nan,
    )
    return output


def _clip_bounds(config: dict[str, Any]) -> tuple[float, float]:
    lower, upper = map(float, config["probabilityCalibration"]["probabilityClip"])
    return lower, upper


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if not len(y):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    value = 0.0
    for bucket in range(bins):
        mask = membership == bucket
        if np.any(mask):
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def _calibration_slope(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float).clip(0.001, 0.999)
    if len(y) < 3 or len(np.unique(y)) < 2 or np.allclose(p, p[0]):
        return None
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return float(model.coef_[0, 0])


def _score_probabilities(
    pipeline: Pipeline | None,
    calibrator: LogisticRegression | None,
    rows: pd.DataFrame,
    features: list[str],
    prior: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if pipeline is None:
        raw = np.full(len(rows), prior, dtype=float)
        return raw, raw.copy()
    x = rows[features].replace([np.inf, -np.inf], np.nan)
    raw = pipeline.predict_proba(x)[:, 1]
    if calibrator is None:
        calibrated = np.full(len(rows), prior, dtype=float)
    else:
        score = pipeline.decision_function(x).reshape(-1, 1)
        calibrated = calibrator.predict_proba(score)[:, 1]
    lower, upper = _clip_bounds(config)
    return np.clip(raw, lower, upper), np.clip(calibrated, lower, upper)


def reliability_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    prior: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    valid = rows["label_market_opportunity_10d"].notna().to_numpy()
    y = rows.loc[valid, "label_market_opportunity_10d"].to_numpy(dtype=int)
    p = np.asarray(probabilities, dtype=float)[valid]
    lower, upper = _clip_bounds(config)
    p = np.clip(p, lower, upper)
    prior_p = np.full(len(y), np.clip(prior, lower, upper), dtype=float)
    if not len(y):
        return {"n": 0, "enabled": False, "checks": {"auditDataPresent": False}}
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    brier = float(brier_score_loss(y, p))
    prior_brier = float(brier_score_loss(y, prior_p))
    loss = float(log_loss(y, p, labels=[0, 1]))
    prior_loss = float(log_loss(y, prior_p, labels=[0, 1]))
    ece = _ece(y, p, int(config["probabilityCalibration"]["eceBins"]))
    slope = _calibration_slope(y, p)
    gate = config["reliabilityGate"]
    high = p >= float(gate["highProbabilityThresholdInclusive"])
    high_n = int(high.sum())
    high_hit = float(y[high].mean()) if high_n else None
    checks = {
        "minimumAuditDaysPass": len(y) >= int(gate["minimumAuditDays"]),
        "brierVsPriorPass": brier < prior_brier,
        "logLossVsPriorPass": loss < prior_loss,
        "aucPass": auc is not None and auc >= float(gate["minimumAuditAuc"]),
        "ecePass": ece <= float(gate["maximumAuditEce"]),
        "positiveCalibrationSlopePass": slope is not None and slope > 0.0,
        "highProbabilitySamplePass": high_n >= int(gate["minimumAuditHighProbabilityDays"]),
        "highProbabilityHitRatePass": high_hit is not None
        and high_hit >= float(gate["minimumAuditHighProbabilityHitRate"]),
    }
    return {
        "n": len(y),
        "positiveRate": round(float(y.mean()), 8),
        "meanProbability": round(float(p.mean()), 8),
        "frozenBasePrior": round(float(prior), 8),
        "brier": round(brier, 8),
        "frozenPriorBrier": round(prior_brier, 8),
        "brierImprovement": round(prior_brier - brier, 8),
        "logLoss": round(loss, 8),
        "frozenPriorLogLoss": round(prior_loss, 8),
        "logLossImprovement": round(prior_loss - loss, 8),
        "auc": round(auc, 8) if auc is not None else None,
        "ece": round(ece, 8),
        "calibrationSlope": round(slope, 8) if slope is not None else None,
        "highProbabilityDays": high_n,
        "highProbabilityHitRate": round(high_hit, 8) if high_hit is not None else None,
        "checks": checks,
        "enabled": all(checks.values()),
    }


def fit_market_opportunity_model(
    market_table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> MarketOpportunityModel:
    base_dates, calibration_dates, audit_dates, segment = nested_segments(fit_dates, config)
    features = market_feature_columns(config)
    base = market_table[
        market_table["date"].isin(base_dates)
        & market_table["label_market_opportunity_10d"].notna()
    ].copy()
    calibration = market_table[
        market_table["date"].isin(calibration_dates)
        & market_table["label_market_opportunity_10d"].notna()
    ].copy()
    audit = market_table[
        market_table["date"].isin(audit_dates)
        & market_table["label_market_opportunity_10d"].notna()
    ].copy()
    if base.empty:
        raise RuntimeError("V8 base-fit market table is empty")
    prior = float(base["label_market_opportunity_10d"].mean())
    pipeline: Pipeline | None = None
    calibrator: LogisticRegression | None = None
    failure_reason: str | None = None
    if base["label_market_opportunity_10d"].nunique() < 2:
        failure_reason = "base_fit_has_one_class"
    else:
        spec = config["marketOpportunityModel"]
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=float(spec["C"]),
                        solver=str(spec["solver"]),
                        max_iter=int(spec["maximumIterations"]),
                        class_weight=spec["classWeight"],
                        random_state=int(spec["randomSeed"]),
                    ),
                ),
            ]
        )
        pipeline.fit(
            base[features].replace([np.inf, -np.inf], np.nan),
            base["label_market_opportunity_10d"].astype(int),
        )
        if calibration.empty or calibration["label_market_opportunity_10d"].nunique() < 2:
            failure_reason = "calibration_has_fewer_than_two_classes"
        else:
            calibration_spec = config["probabilityCalibration"]
            score = pipeline.decision_function(
                calibration[features].replace([np.inf, -np.inf], np.nan)
            ).reshape(-1, 1)
            calibrator = LogisticRegression(
                C=float(calibration_spec["C"]),
                solver=str(calibration_spec["solver"]),
                max_iter=int(calibration_spec["maximumIterations"]),
                random_state=int(spec["randomSeed"]),
            )
            calibrator.fit(score, calibration["label_market_opportunity_10d"].astype(int))
    _, audit_probabilities = _score_probabilities(
        pipeline, calibrator, audit, features, prior, config
    )
    reliability = reliability_metrics(audit, audit_probabilities, prior, config)
    enabled = failure_reason is None and bool(reliability["enabled"])
    return MarketOpportunityModel(
        pipeline=pipeline,
        calibrator=calibrator,
        feature_columns=features,
        training_prior=prior,
        enabled=enabled,
        reliability=reliability,
        audit={
            **segment,
            "baseFitRows": len(base),
            "baseFitPositiveRate": round(prior, 8),
            "calibrationRows": len(calibration),
            "auditRows": len(audit),
            "featureCount": len(features),
            "failureReason": failure_reason,
            "reliability": reliability,
        },
    )


def score_market_rows(
    model: MarketOpportunityModel,
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    raw, calibrated = _score_probabilities(
        model.pipeline,
        model.calibrator,
        rows,
        model.feature_columns,
        model.training_prior,
        config,
    )
    output = rows[["date", "benchmark_return_10d", "label_market_opportunity_10d"]].copy()
    output["raw_market_opportunity_probability"] = raw
    output["calibrated_market_opportunity_probability"] = calibrated
    output["frozen_base_prior_probability"] = model.training_prior
    output["market_opportunity_model_enabled"] = model.enabled
    output["audit_brier"] = model.reliability.get("brier")
    output["audit_prior_brier"] = model.reliability.get("frozenPriorBrier")
    output["audit_log_loss"] = model.reliability.get("logLoss")
    output["audit_prior_log_loss"] = model.reliability.get("frozenPriorLogLoss")
    output["audit_auc"] = model.reliability.get("auc")
    output["audit_ece"] = model.reliability.get("ece")
    output["audit_high_probability_days"] = model.reliability.get("highProbabilityDays")
    output["audit_high_probability_hit_rate"] = model.reliability.get("highProbabilityHitRate")
    return output


def rolling_market_opportunity_predictions(
    market_table: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(calendar_dates)))
    walk = config["walkForward"]
    minimum = int(walk["minimumTrainingTradingDays"])
    window = int(walk["trainingWindowTradingDays"])
    purge = int(walk["outerPurgeTradingDays"])
    refit = int(walk["refitEveryTradingDays"])
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold = 0
    for start in range(minimum + purge, len(dates), refit):
        block = dates[start : min(start + refit, len(dates))]
        if block.empty:
            continue
        fit_stop = start - purge
        fit_start = max(0, fit_stop - window)
        fit_dates = dates[fit_start:fit_stop]
        if len(fit_dates) < minimum:
            continue
        block_rows = market_table[market_table["date"].isin(block)].copy()
        if block_rows.empty:
            continue
        model = fit_market_opportunity_model(market_table, fit_dates, config)
        scored = score_market_rows(model, block_rows, config)
        fold += 1
        scored["market_opportunity_walk_forward_fold"] = fold
        predictions.append(scored)
        audits.append(
            {
                "fold": fold,
                "predictionDateRange": [block[0].date().isoformat(), block[-1].date().isoformat()],
                "predictionRows": len(scored),
                "outerPurgeTradingDays": purge,
                "outerGapTradingDays": int(start - (fit_stop - 1) - 1),
                "model": model.audit,
            }
        )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        audits,
    )


def add_market_opportunity_gate(
    rank_predictions: pd.DataFrame,
    market_predictions: pd.DataFrame,
    config: dict[str, Any],
    base: dict[str, Any],
) -> pd.DataFrame:
    # The rank table already carries market features and benchmark_return_10d.
    # Merge only genuinely new date-level fields to avoid silently creating
    # ``_x``/``_y`` columns with ambiguous evaluation semantics.
    market_columns = ["date"] + [
        name
        for name in market_predictions.columns
        if name != "date" and name not in rank_predictions.columns
    ]
    output = rank_predictions.merge(
        market_predictions[market_columns], on="date", how="inner", validate="many_to_one"
    )
    threshold = float(config["selection"]["marketOpportunityThresholdInclusive"])
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    output["fixed_v2_risk_on"] = output["market_return_20"].gt(
        float(base["marketGate"]["marketReturn20MinimumExclusive"])
    ) & output["market_breadth_20"].ge(
        float(base["marketGate"]["marketBreadth20MinimumInclusive"])
    )
    output["market_opportunity_threshold_met"] = output[
        "calibrated_market_opportunity_probability"
    ].ge(threshold)
    output["selectedWithoutReliability"] = (
        output["predicted_order"].le(top_n)
        & output["market_opportunity_threshold_met"]
    )
    output["selectedByPrimaryPolicy"] = (
        output["selectedWithoutReliability"]
        & output["market_opportunity_model_enabled"]
    )
    return output.sort_values(["date", "securityId"]).reset_index(drop=True)


def policy_rows(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    return {
        "v2_rank_top3_all_days": rows[rows["predicted_order"].le(top_n)].copy(),
        "v2_rank_top3_fixed_v2_risk_on": rows[
            rows["predicted_order"].le(top_n) & rows["fixed_v2_risk_on"]
        ].copy(),
        "v2_rank_top3_calibrated_probability_without_reliability": rows[
            rows["selectedWithoutReliability"]
        ].copy(),
        PRIMARY_POLICY: rows[rows["selectedByPrimaryPolicy"]].copy(),
    }


def probability_metrics(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    daily = rows.sort_values("date").drop_duplicates("date")
    valid = daily[
        daily["label_market_opportunity_10d"].notna()
        & daily["calibrated_market_opportunity_probability"].notna()
    ].copy()
    if valid.empty:
        return {
            "n": 0,
            "brier": None,
            "frozenPriorBrier": None,
            "logLoss": None,
            "frozenPriorLogLoss": None,
            "auc": None,
            "ece": None,
        }
    y = valid["label_market_opportunity_10d"].astype(int).to_numpy()
    lower, upper = _clip_bounds(config)
    p = valid["calibrated_market_opportunity_probability"].to_numpy(dtype=float).clip(lower, upper)
    prior = valid["frozen_base_prior_probability"].to_numpy(dtype=float).clip(lower, upper)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    brier = float(brier_score_loss(y, p))
    prior_brier = float(np.mean((prior - y) ** 2))
    loss = float(log_loss(y, p, labels=[0, 1]))
    prior_loss = float(log_loss(y, prior, labels=[0, 1]))
    return {
        "n": len(y),
        "positiveRate": round(float(y.mean()), 8),
        "meanProbability": round(float(p.mean()), 8),
        "brier": round(brier, 8),
        "frozenPriorBrier": round(prior_brier, 8),
        "brierImprovement": round(prior_brier - brier, 8),
        "logLoss": round(loss, 8),
        "frozenPriorLogLoss": round(prior_loss, 8),
        "logLossImprovement": round(prior_loss - loss, 8),
        "auc": round(auc, 8) if auc is not None else None,
        "ece": round(_ece(y, p, int(config["probabilityCalibration"]["eceBins"])), 8),
    }


def probability_buckets(rows: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    daily = rows.sort_values("date").drop_duplicates("date")
    valid = daily[
        daily["label_market_opportunity_10d"].notna()
        & daily["calibrated_market_opportunity_probability"].notna()
    ].copy()
    if valid.empty:
        return []
    valid["bucketLower"] = (
        np.floor(valid["calibrated_market_opportunity_probability"].clip(0.0, 0.999999) * 10.0)
        / 10.0
    )
    output: list[dict[str, Any]] = []
    for lower, group in valid.groupby("bucketLower", sort=True):
        output.append(
            {
                "lower": round(float(lower), 1),
                "upper": round(float(lower + 0.1), 1),
                "n": len(group),
                "meanProbability": round(float(group["calibrated_market_opportunity_probability"].mean()), 8),
                "hitRate": round(float(group["label_market_opportunity_10d"].mean()), 8),
                "meanBenchmarkReturn": round(float(group["benchmark_return_10d"].mean()), 8),
            }
        )
    return output


def _hac_binary_group_difference(
    values: pd.Series,
    gated_dates: set[pd.Timestamp],
    lag: int,
) -> dict[str, Any]:
    sample = values.dropna().sort_index().astype(float)
    if sample.empty:
        return {
            "n": 0,
            "gatedN": 0,
            "excludedN": 0,
            "gatedMean": None,
            "excludedMean": None,
            "delta": None,
            "deltaTHac": None,
        }
    gate = np.asarray([pd.Timestamp(date) in gated_dates for date in sample.index], dtype=float)
    if gate.sum() == 0 or gate.sum() == len(gate):
        return {
            "n": len(sample),
            "gatedN": int(gate.sum()),
            "excludedN": int(len(gate) - gate.sum()),
            "gatedMean": float(sample.to_numpy()[gate == 1].mean()) if gate.sum() else None,
            "excludedMean": float(sample.to_numpy()[gate == 0].mean()) if gate.sum() < len(gate) else None,
            "delta": None,
            "deltaTHac": None,
        }
    y = sample.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(y)), gate])
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((2, 2), dtype=float)
    for index in range(len(y)):
        vector = x[index] * residual[index]
        meat += np.outer(vector, vector)
    maximum_lag = min(int(lag), len(y) - 1)
    for distance in range(1, maximum_lag + 1):
        weight = 1.0 - distance / (maximum_lag + 1.0)
        gamma = np.zeros((2, 2), dtype=float)
        for index in range(distance, len(y)):
            current = x[index] * residual[index]
            previous = x[index - distance] * residual[index - distance]
            gamma += np.outer(current, previous)
        meat += weight * (gamma + gamma.T)
    covariance = inverse @ meat @ inverse
    variance = float(covariance[1, 1])
    statistic = float(beta[1] / np.sqrt(variance)) if variance > 0.0 else None
    gated = y[gate == 1]
    excluded = y[gate == 0]
    return {
        "n": len(y),
        "gatedN": len(gated),
        "excludedN": len(excluded),
        "gatedMean": round(float(gated.mean()), 8),
        "excludedMean": round(float(excluded.mean()), 8),
        "delta": round(float(beta[1]), 8),
        "deltaTHac": round(statistic, 4) if statistic is not None else None,
        "hacLag": maximum_lag,
    }


def gate_discrimination(
    all_top3_basket: pd.Series,
    gated_dates: set[pd.Timestamp],
    lag: int,
    tail_threshold: float,
) -> dict[str, Any]:
    returns = _hac_binary_group_difference(all_top3_basket, gated_dates, lag)
    wins = _hac_binary_group_difference(all_top3_basket.gt(0.0).astype(float), gated_dates, lag)
    tails = _hac_binary_group_difference(
        all_top3_basket.le(tail_threshold).astype(float), gated_dates, lag
    )
    return {
        "return": returns,
        "win": wins,
        "tail": tails,
        "tailLossRateReduction": round(-float(tails["delta"]), 8)
        if tails["delta"] is not None
        else None,
    }


def slot_metrics(selected: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    cost = float(config["data"]["roundTripCost"])
    tail = float(config["data"]["tailLossThreshold"])
    for slot in (1, 2, 3):
        actual = selected.loc[
            selected["predicted_order"].eq(slot), "target_return_10d"
        ].dropna()
        output[f"slot{slot}"] = {
            "n": len(actual),
            "meanReturn": round(float(actual.mean()), 8) if len(actual) else None,
            "medianReturn": round(float(actual.median()), 8) if len(actual) else None,
            "winRate": round(float(actual.gt(0.0).mean()), 8) if len(actual) else None,
            "afterCostWinRate": round(float(actual.gt(cost).mean()), 8) if len(actual) else None,
            "tailLossRate": round(float(actual.le(tail).mean()), 8) if len(actual) else None,
        }
    return output


def period_report(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    eligible: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    labelled = predictions[
        predictions["date"].isin(dates) & predictions["target_return_10d"].notna()
    ].copy()
    policies = policy_rows(labelled, config)
    threshold = float(config["data"]["tailLossThreshold"])
    metrics: dict[str, dict[str, Any]] = {}
    baskets: dict[str, pd.Series] = {}
    for name, selected in policies.items():
        metric = selector_v1.portfolio_metrics(selected, target, one_day, eligible, dates, config)
        basket = winrate_v2.basket_series(selected, target, dates)
        metric.update(tail_v3.tail_outcome_metrics(basket, threshold))
        metrics[name] = metric
        baskets[name] = basket
    primary_rows = policies[PRIMARY_POLICY]
    primary_dates = set(pd.to_datetime(primary_rows["date"].unique()))
    enabled_daily = labelled.sort_values("date").drop_duplicates("date")
    return {
        "candidateRows": len(labelled),
        "modelEnabledCalendarDays": int(
            enabled_daily["market_opportunity_model_enabled"].sum()
        ),
        "probabilityThresholdCalendarDaysWithoutReliability": int(
            enabled_daily["market_opportunity_threshold_met"].sum()
        ),
        "opportunityProbability": probability_metrics(labelled, config),
        "opportunityProbabilityBuckets": probability_buckets(labelled, config),
        "selectedSlots": slot_metrics(primary_rows, config),
        "policies": metrics,
        "primaryIndependentEvents": winrate_v2.independent_event_metrics(
            baskets[PRIMARY_POLICY], calendar_dates, int(config["data"]["holdingTradingDays"])
        ),
        "primaryMonthly": winrate_v2.monthly_metrics(baskets[PRIMARY_POLICY]),
        "gatedVsExcludedV2Top3": gate_discrimination(
            baskets["v2_rank_top3_all_days"],
            primary_dates,
            int(config["evaluation"]["hacLagTradingDays"]),
            threshold,
        ),
    }


def build_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["evaluation"]
    checks: dict[str, Any] = {}
    stable = True
    for period in ("validation", "shadow"):
        block = report["periods"][period]
        primary = block["policies"][PRIMARY_POLICY]
        probability = block["opportunityProbability"]
        comparison = block["gatedVsExcludedV2Top3"]
        coverage = primary["newSignalDays"] / max(1, primary["calendarDays"])
        slot_checks: dict[str, Any] = {}
        for name, metric in block["selectedSlots"].items():
            item = {
                "enoughObservations": metric["n"] >= int(rules["minimumPerSlotObservations"]),
                "positiveMeanReturn": metric["meanReturn"] is not None
                and metric["meanReturn"] > float(rules["minimumPerSlotMeanReturnExclusive"]),
                "winRatePass": metric["winRate"] is not None
                and metric["winRate"] > float(rules["minimumPerSlotWinRateExclusive"]),
            }
            item["slotPass"] = all(item.values())
            slot_checks[name] = item
        period_checks = {
            "enoughSignalDays": primary["newSignalDays"]
            >= int(rules["minimumSignalDaysPerValidationPeriod"]),
            "enoughIndependentEvents": block["primaryIndependentEvents"]["n"]
            >= int(rules["minimumIndependentEventsPerValidationPeriod"]),
            "signalDayCoverage": round(float(coverage), 8),
            "coveragePass": coverage >= float(rules["minimumSignalDayCoverage"]),
            "basketWinRatePass": primary["basketTenDayWinRate"] is not None
            and primary["basketTenDayWinRate"] >= float(rules["minimumBasketWinRate"]),
            "positiveMeanReturn": primary["basketTenDayMeanReturn"] is not None
            and primary["basketTenDayMeanReturn"] > 0.0,
            "positiveCostedCumulativeReturn": primary["costedCumulativeReturn"] > 0.0,
            "probabilityBrierPass": probability["brier"] is not None
            and probability["brier"] < probability["frozenPriorBrier"],
            "probabilityLogLossPass": probability["logLoss"] is not None
            and probability["logLoss"] < probability["frozenPriorLogLoss"],
            "probabilityAucPass": probability["auc"] is not None
            and probability["auc"] >= float(rules["minimumAggregateAuc"]),
            "probabilityEcePass": probability["ece"] is not None
            and probability["ece"] <= float(rules["maximumAggregateEce"]),
            "gatedReturnLift": comparison["return"]["delta"],
            "gatedReturnLiftPass": comparison["return"]["delta"] is not None
            and comparison["return"]["delta"]
            > float(rules["minimumGatedVsExcludedMeanReturnLiftExclusive"]),
            "gatedReturnHacPass": comparison["return"]["deltaTHac"] is not None
            and comparison["return"]["deltaTHac"]
            >= float(rules["minimumHacTForGatedVsExcludedReturnDifference"]),
            "gatedWinLift": comparison["win"]["delta"],
            "gatedWinLiftPass": comparison["win"]["delta"] is not None
            and comparison["win"]["delta"]
            >= float(rules["minimumGatedVsExcludedWinRateLift"]),
            "tailLossRateReduction": comparison["tailLossRateReduction"],
            "tailNoWorsePass": comparison["tailLossRateReduction"] is not None
            and comparison["tailLossRateReduction"] >= 0.0,
            "allThreeSlotsPass": all(item["slotPass"] for item in slot_checks.values()),
            "slotChecks": slot_checks,
        }
        required = [
            "enoughSignalDays",
            "enoughIndependentEvents",
            "coveragePass",
            "basketWinRatePass",
            "positiveMeanReturn",
            "positiveCostedCumulativeReturn",
            "probabilityBrierPass",
            "probabilityLogLossPass",
            "probabilityAucPass",
            "probabilityEcePass",
            "gatedReturnLiftPass",
            "gatedReturnHacPass",
            "gatedWinLiftPass",
            "tailNoWorsePass",
            "allThreeSlotsPass",
        ]
        period_checks["periodPass"] = all(bool(period_checks[name]) for name in required)
        checks[period] = period_checks
        stable = stable and bool(period_checks["periodPass"])
    return {
        "status": "research_only_not_eligible_for_trading",
        "primaryPolicy": PRIMARY_POLICY,
        "stableHistoricalWinRateAndReturnImprovement": bool(stable),
        "profitGuaranteed": False,
        "checks": checks,
        "decision": (
            "retain_for_fresh_forward_shadow_only"
            if stable
            else "reject_for_trading_keep_diagnostics"
        ),
        "promotionAllowed": False,
    }


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(value)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Perception-XAlpha market-opportunity V8",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- Frozen V2 ranks select stocks; V8 only decides whether the long-only market opportunity is reliable.",
        "- Top3 can be empty. No result guarantees profit. Orders are always empty.",
        "",
        "## Policy outcomes",
        "",
        "| Period | Policy | Days | Coverage | 10d mean | Win | Tail | Costed cumulative |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        for name, metric in report["periods"][period]["policies"].items():
            coverage = metric["newSignalDays"] / max(1, metric["calendarDays"])
            lines.append(
                f"| {period} | {name} | {metric['newSignalDays']} | {coverage:.2%} | "
                f"{metric['basketTenDayMeanReturn']} | {metric['basketTenDayWinRate']} | "
                f"{metric['tailLossRate']} | {metric['costedCumulativeReturn']} |"
            )
    lines.extend(
        [
            "",
            "## Probability diagnostics",
            "",
            "| Period | N | Brier | Prior | LogLoss | Prior | AUC | ECE | Enabled days |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("trainWalkForward", "validation", "shadow"):
        block = report["periods"][period]
        metric = block["opportunityProbability"]
        lines.append(
            f"| {period} | {metric['n']} | {metric['brier']} | {metric['frozenPriorBrier']} | "
            f"{metric['logLoss']} | {metric['frozenPriorLogLoss']} | {metric['auc']} | "
            f"{metric['ece']} | {block['modelEnabledCalendarDays']} |"
        )
    lines.extend(
        [
            "",
            "## Gated versus excluded dates for the same V2 Top3",
            "",
            "| Period | Gated N | Excluded N | Gated mean | Excluded mean | Delta | HAC t | Win delta | Tail reduction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("validation", "shadow"):
        metric = report["periods"][period]["gatedVsExcludedV2Top3"]
        returns = metric["return"]
        lines.append(
            f"| {period} | {returns['gatedN']} | {returns['excludedN']} | "
            f"{returns['gatedMean']} | {returns['excludedMean']} | {returns['delta']} | "
            f"{returns['deltaTHac']} | {metric['win']['delta']} | {metric['tailLossRateReduction']} |"
        )
    lines.extend(
        [
            "",
            "## Selected rank slots",
            "",
            "| Period | Slot | N | Mean | Win | After-cost win | Tail |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("validation", "shadow"):
        for slot, metric in report["periods"][period]["selectedSlots"].items():
            lines.append(
                f"| {period} | {slot} | {metric['n']} | {metric['meanReturn']} | "
                f"{metric['winRate']} | {metric['afterCostWinRate']} | {metric['tailLossRate']} |"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Decision: `{report['verdict']['decision']}`",
            f"- Stable historical win-rate and return improvement: `{report['verdict']['stableHistoricalWinRateAndReturnImprovement']}`",
            "- A higher conditional win rate caused only by fewer trades is rejected by the gated-versus-excluded HAC test.",
            "- Historical output cannot promote or connect to trading.",
            "- Profit guaranteed: `false`.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    if "date" in output:
        output["date"] = pd.to_datetime(output["date"]).dt.date.astype(str)
    return json.loads(output.to_json(orient="records"))


def run(
    config_path: Path,
    run_id: str | None = None,
    research_data_override_path: Path | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    base_path = ROOT / config["baseWinrateConfig"]
    base = load_json(base_path)
    validate_config(config, base)
    selector_config = load_json(ROOT / base["baseSelectorConfig"])
    selector_v1.validate_config(selector_config)
    base_research_path = ROOT / selector_config["baseResearchConfig"]
    research_config = load_json(base_research_path)
    data_override = None
    price_data_audit = None
    if research_data_override_path is not None:
        data_override = load_json(research_data_override_path)
        validate_research_data_override(
            data_override,
            config_path,
            base_research_path,
        )
        price_data_audit = require_price_data_audit(data_override)
        research_config = copy.deepcopy(research_config)
        research_config["assetUniverse"] = copy.deepcopy(
            data_override["assetUniverse"]
        )
    _, cog_config = perception.load_base_configs(research_config)
    panel, panel_audit = perception.build_configured_panel(research_config, cog_config)
    if data_override is not None:
        validate_corrected_panel_audit(panel_audit)
    factor_ranks = selector_v1.build_factor_rank_frames(panel, selector_config)
    features, market_features = selector_v1.build_past_only_feature_frames(panel, factor_ranks)
    target, one_day = autonomous.target_frames(panel, cog_config)
    raw_table = selector_v1.build_candidate_table(
        panel,
        features,
        market_features,
        target,
        int(config["data"]["candidatePoolSize"]),
    )
    rank_table = winrate_v2.prepare_rank_table(raw_table, base)
    rank_predictions, rank_audits = winrate_v2.rolling_walk_forward_predictions(
        rank_table, panel["close"].index, base
    )
    market_table = prepare_market_table(rank_table, config)
    market_predictions, market_audits = rolling_market_opportunity_predictions(
        market_table, panel["close"].index, config
    )
    if rank_predictions.empty or market_predictions.empty:
        raise RuntimeError("V8 rank or market-opportunity predictions are empty")
    predictions = add_market_opportunity_gate(
        rank_predictions, market_predictions, config, base
    )
    split = autonomous.make_split(panel["close"].index, cog_config)
    train_dates = pd.DatetimeIndex(split.train[split.train].index)
    validation_dates = pd.DatetimeIndex(split.validation[split.validation].index)
    shadow_dates = pd.DatetimeIndex(split.shadow[split.shadow].index)
    predicted_dates = pd.DatetimeIndex(sorted(pd.unique(predictions["date"])))
    train_walk_dates = train_dates.intersection(predicted_dates)
    run_id = run_id or (
        "run_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + selector_v1.digest({"config": config, "code": CODE_VERSION})[:10]
    )
    latest_date = predictions["date"].max()
    latest = predictions[predictions["date"].eq(latest_date)].copy()
    latest_top3 = latest[latest["predicted_order"].le(3)].sort_values(
        ["predicted_order", "securityId"]
    )
    latest_columns = [
        "date",
        "securityId",
        "candidate_rank",
        "factor_composite",
        "predicted_cross_sectional_rank",
        "predicted_order",
        "market_return_20",
        "market_return_60",
        "market_volatility_20",
        "market_breadth_20",
        "fixed_v2_risk_on",
        "raw_market_opportunity_probability",
        "calibrated_market_opportunity_probability",
        "frozen_base_prior_probability",
        "market_opportunity_model_enabled",
        "audit_brier",
        "audit_prior_brier",
        "audit_log_loss",
        "audit_prior_log_loss",
        "audit_auc",
        "audit_ece",
        "market_opportunity_threshold_met",
        "selectedByPrimaryPolicy",
    ]
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": selector_v1.digest(config),
        "researchDataOverridePath": (
            str(research_data_override_path) if research_data_override_path else None
        ),
        "researchDataOverrideSha256": (
            file_sha256(research_data_override_path)
            if research_data_override_path
            else None
        ),
        "priceDataAudit": price_data_audit,
        "baseWinrateConfigSha256": file_sha256(base_path),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(rank_table),
        "predictionRows": len(predictions),
        "marketPredictionRows": len(market_predictions),
        "marketFeatureDefinition": market_feature_columns(config),
        "labelDefinition": config["data"]["marketOpportunityLabel"],
        "rankWalkForwardAudits": rank_audits,
        "marketOpportunityWalkForwardAudits": market_audits,
        "periods": {},
        "latestDate": latest_date.date().isoformat(),
        "latestTop3": _records(latest_top3[latest_columns]),
        "latestSelectedCount": int(latest_top3["selectedByPrimaryPolicy"].sum()),
        "knownLimitations": (
            data_override["knownLimitations"]
            if data_override is not None
            else config["knownLimitations"]
        ),
        "orders": [],
        "automaticTradingChanges": [],
    }
    for name, dates in {
        "trainWalkForward": train_walk_dates,
        "validation": validation_dates,
        "shadow": shadow_dates,
    }.items():
        report["periods"][name] = period_report(
            predictions,
            dates,
            target,
            one_day,
            panel["eligible"],
            panel["close"].index,
            config,
        )
    report["verdict"] = build_verdict(report, config)
    out = ROOT / config["output"]["root"] / run_id
    atomic_write_text(out / "summary.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(out / "report.md", markdown_report(report))
    atomic_write_text(
        out / "latest_top3.csv",
        latest_top3[latest_columns].to_csv(index=False, lineterminator="\n"),
    )
    market_output = market_predictions.copy()
    market_output["date"] = pd.to_datetime(market_output["date"]).dt.date.astype(str)
    atomic_write_text(
        out / "market_opportunities.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in _records(market_output)) + "\n",
    )
    print(
        json.dumps(
            {
                "runId": run_id,
                "output": str(out),
                "dataRange": report["dataRange"],
                "predictionRows": len(predictions),
                "latestDate": report["latestDate"],
                "latestSelectedTop3": report["latestSelectedCount"],
                "profitGuaranteed": False,
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--research-data-override", type=Path)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.run_id or None,
        (
            args.research_data_override.resolve()
            if args.research_data_override
            else None
        ),
    )


if __name__ == "__main__":
    main()
