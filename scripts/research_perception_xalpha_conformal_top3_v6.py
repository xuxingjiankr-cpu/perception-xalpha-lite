"""Research-only Top10 return intervals and conservative Top3 selection.

V6 decomposes the ten-session stock return into a common market component and
a stock residual-alpha component.  Both Ridge models are fit on a base segment,
signed residual interval offsets are fit on a later calibration segment, and a
third never-fit segment decides whether later intervals may be used for the
positive-lower-bound Top3 policy.  Nothing in this module can trade.
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_expected_utility_v4 as utility_v4  # noqa: E402
import research_perception_xalpha_return_reliability_v5 as reliability_v5  # noqa: E402
import research_perception_xalpha_two_stage as selector_v1  # noqa: E402
import research_perception_xalpha_winrate_tail_v3 as tail_v3  # noqa: E402
import research_perception_xalpha_winrate_v2 as winrate_v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_conformal_top3_v6.json"
)
SCHEMA_VERSION = "perception_xalpha_conformal_top3_v6"
CODE_VERSION = "perception_xalpha_conformal_top3_v6.0"
PRIMARY_POLICY = "conformal_top3_positive_lower_bound"


@dataclass
class ComponentModel:
    pipeline: Pipeline
    feature_columns: list[str]
    target_column: str
    target_clip: tuple[float, float]
    training_prior: float
    audit: dict[str, Any]


@dataclass
class ConformalReturnModel:
    market_model: ComponentModel
    residual_model: ComponentModel
    lower_offset: float
    upper_offset: float
    interval_enabled: bool
    reliability: dict[str, Any]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected conformal Top3 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V6 must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading mutation permissions must remain false")
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("primaryPolicy") != PRIMARY_POLICY:
        raise ValueError("the preregistered primary policy changed")
    if hypothesis.get("parametersFrozenBeforeHistoricalRun") is not True:
        raise ValueError("parameters must be frozen before historical evaluation")
    if hypothesis.get("validationAndShadowMayNotTuneParameters") is not True:
        raise ValueError("evaluation windows may not tune parameters")
    if hypothesis.get("profitGuaranteeClaimAllowed") is not False:
        raise ValueError("V6 may not claim guaranteed profits")
    data = config["data"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the return horizon must remain ten sessions")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("the candidate pool must remain the frozen Top50")
    if int(data["contextTopN"]) != 10:
        raise ValueError("the context set must remain the V2 Top10")
    if int(data["maximumSelectionsPerDay"]) != 3:
        raise ValueError("the confidence book may select at most three stocks")
    if not np.isclose(float(data["roundTripCost"]), 0.003):
        raise ValueError("round-trip cost changed")
    walk = config["walkForward"]
    if int(walk["outerPurgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("outer purge must cover the label horizon")
    for key in ("baseToCalibrationPurgeTradingDays", "calibrationToAuditPurgeTradingDays"):
        if int(walk[key]) < int(data["holdingTradingDays"]):
            raise ValueError(f"nested purge is too short: {key}")
    if int(walk["calibrationTradingDays"]) != 63:
        raise ValueError("calibration segment changed")
    if int(walk["reliabilityAuditTradingDays"]) != 63:
        raise ValueError("reliability audit segment changed")
    for name in ("marketReturnModel", "residualAlphaModel"):
        model = config[name]
        if model.get("kind") != "Ridge" or float(model["alpha"]) != 10.0:
            raise ValueError(f"frozen Ridge specification changed: {name}")
    if config["marketReturnModel"].get("oneObservationPerTradingDate") is not True:
        raise ValueError("market model must use one observation per date")
    fitting = config["modelFitting"]
    if list(map(float, fitting["targetWinsorQuantiles"])) != [0.01, 0.99]:
        raise ValueError("training-only target clipping changed")
    interval = config["splitConformalInterval"]
    if interval.get("kind") != "asymmetric_signed_residual":
        raise ValueError("interval score changed")
    if not np.allclose(
        [interval["nominalCoverage"], interval["lowerResidualQuantile"], interval["upperResidualQuantile"]],
        [0.8, 0.1, 0.9],
    ):
        raise ValueError("frozen interval quantiles changed")
    if interval.get("dateEqualWeighted") is not True:
        raise ValueError("calibration dates must receive equal total weight")
    if interval.get("auditSegmentNeverFitsModelsOrBounds") is not True:
        raise ValueError("audit data may not fit models or interval bounds")
    if interval.get("theoreticalFiniteSampleCoverageClaimed") is not False:
        raise ValueError("dependent returns cannot claim exchangeable coverage")
    selection = config["selection"]
    if selection.get("context") != "frozen_v2_predicted_order_top10":
        raise ValueError("Top10 context changed")
    if selection.get("rankingField") != "predicted_return_lower_10d":
        raise ValueError("confidence ranking field changed")
    if not np.isclose(
        float(selection["requiredLowerBoundStrictlyAbove"]),
        float(data["roundTripCost"]),
    ):
        raise ValueError("positive lower-bound threshold changed")
    if selection.get("requiresReliabilityGate") is not True:
        raise ValueError("selection must fail closed on reliability")
    if selection.get("allowZeroSelections") is not True or selection.get("neverFillToThree") is not True:
        raise ValueError("V6 must allow an empty confidence book")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("a historical run cannot promote")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain permanently empty")
    _ = market_feature_columns(config)
    _ = residual_feature_columns(config)


def _validate_past_only_features(columns: list[str], label: str) -> list[str]:
    forbidden = {
        "target_return_10d",
        "benchmark_return_10d",
        "target_residual_10d",
        "target_cross_sectional_rank_10d",
        "label_tail_loss_10d",
        "label_positive_10d",
    }
    overlap = forbidden.intersection(columns)
    if overlap:
        raise ValueError(f"future outcome entered {label} features: {sorted(overlap)}")
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"{label} feature list is empty or duplicated")
    return columns


def market_feature_columns(config: dict[str, Any]) -> list[str]:
    return _validate_past_only_features(
        list(config["marketReturnModel"]["featureColumns"]), "market"
    )


def residual_feature_columns(config: dict[str, Any]) -> list[str]:
    return _validate_past_only_features(
        list(config["residualAlphaModel"]["featureColumns"]), "residual-alpha"
    )


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
        raise RuntimeError("nested base-fit segment is too short")
    base = fit_dates[:base_end]
    calibration = fit_dates[calibration_start:calibration_end]
    audit = fit_dates[audit_start:]
    if not (base[-1] < calibration[0] < calibration[-1] < audit[0]):
        raise RuntimeError("nested V6 segments are not strictly chronological")
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


def prepare_return_decomposition(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    output["target_residual_10d"] = (
        output["target_return_10d"] - output["benchmark_return_10d"]
    )
    return output


def fit_component_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    feature_columns: list[str],
    target_column: str,
    alpha: float,
    quantiles: list[float],
    one_per_date: bool,
) -> ComponentModel:
    missing = [name for name in [*feature_columns, target_column] if name not in table.columns]
    if missing:
        raise KeyError(f"missing V6 model columns: {missing}")
    rows = table[
        table["date"].isin(fit_dates) & table[target_column].notna()
    ].copy()
    if one_per_date:
        rows = rows.sort_values(["date", "securityId"]).groupby("date", sort=True).head(1)
    if rows.empty:
        raise RuntimeError(f"empty fit table for {target_column}")
    target = rows[target_column].astype(float)
    lower, upper = map(float, target.quantile(quantiles).tolist())
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(
        rows[feature_columns].replace([np.inf, -np.inf], np.nan),
        target.clip(lower, upper),
    )
    prior = float(target.clip(lower, upper).mean())
    return ComponentModel(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_column=target_column,
        target_clip=(lower, upper),
        training_prior=prior,
        audit={
            "target": target_column,
            "fitDateRange": [
                pd.Timestamp(fit_dates[0]).date().isoformat(),
                pd.Timestamp(fit_dates[-1]).date().isoformat(),
            ],
            "fitTradingDays": len(fit_dates),
            "fitRows": len(rows),
            "oneObservationPerDate": one_per_date,
            "featureCount": len(feature_columns),
            "targetClip": [round(lower, 8), round(upper, 8)],
            "trainingPrior": round(prior, 8),
        },
    )


def _score_component(model: ComponentModel, rows: pd.DataFrame, clip: bool) -> np.ndarray:
    values = model.pipeline.predict(
        rows[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    )
    return np.clip(values, *model.target_clip) if clip else values


def score_point_forecast(
    market_model: ComponentModel,
    residual_model: ComponentModel,
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clip = bool(config["modelFitting"]["predictionClipUsesBaseFitComponentBounds"])
    market = _score_component(market_model, rows, clip)
    residual = _score_component(residual_model, rows, clip)
    return market, residual, market + residual


def date_equal_weighted_quantile(
    values: np.ndarray,
    dates: pd.Series,
    quantile: float,
) -> float:
    frame = pd.DataFrame(
        {"value": np.asarray(values, dtype=float), "date": pd.to_datetime(dates).to_numpy()}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise RuntimeError("cannot calibrate an interval from no residuals")
    counts = frame.groupby("date")["value"].transform("size").astype(float)
    frame["weight"] = 1.0 / counts
    frame = frame.sort_values(["value", "date"], kind="stable")
    cumulative = frame["weight"].cumsum().to_numpy(dtype=float)
    threshold = float(quantile) * float(frame["weight"].sum())
    position = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(frame["value"].iloc[min(position, len(frame) - 1)])


def _rank_ic_hac(rows: pd.DataFrame, prediction: np.ndarray, lag: int) -> tuple[float | None, float | None, int]:
    scored = rows[["date", "target_return_10d"]].copy()
    scored["prediction"] = prediction
    daily: list[float] = []
    for _, group in scored.groupby("date", sort=True):
        if len(group) < 3 or group["prediction"].nunique() < 2 or group["target_return_10d"].nunique() < 2:
            continue
        value = group["prediction"].corr(group["target_return_10d"])
        if pd.notna(value):
            daily.append(float(value))
    values = np.asarray(daily, dtype=float)
    statistic = autonomous.newey_west_t(values, lag) if len(values) >= 3 else None
    return (
        float(values.mean()) if len(values) else None,
        float(statistic) if statistic is not None else None,
        len(values),
    )


def reliability_metrics(
    rows: pd.DataFrame,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    actual = rows["target_return_10d"].to_numpy(dtype=float)
    covered = (actual >= lower) & (actual <= upper)
    lower_violated = actual < lower
    upper_violated = actual > upper
    cost = float(config["data"]["roundTripCost"])
    positive = lower > cost
    positive_n = int(positive.sum())
    positive_hit = float((actual[positive] > cost).mean()) if positive_n else None
    ic_mean, ic_t, ic_days = _rank_ic_hac(
        rows, point, int(config["evaluation"]["hacLagTradingDays"])
    )
    coverage = float(covered.mean())
    lower_rate = float(lower_violated.mean())
    upper_rate = float(upper_violated.mean())
    width = float(np.mean(upper - lower))
    gate = config["reliabilityGate"]
    checks = {
        "intervalCoveragePass": coverage >= float(gate["minimumAuditIntervalCoverage"]),
        "lowerViolationPass": lower_rate <= float(gate["maximumAuditLowerBoundViolationRate"]),
        "intervalWidthPass": width <= float(gate["maximumAuditMeanIntervalWidth"]),
        "rankIcHacPass": ic_t is not None and ic_t >= float(gate["minimumAuditPointForecastRankIcHacT"]),
        "positiveLowerSamplePass": positive_n >= int(gate["minimumAuditPositiveLowerBoundObservations"]),
        "positiveLowerHitRatePass": positive_hit is not None and positive_hit >= float(gate["minimumAuditPositiveLowerBoundHitRate"]),
    }
    enabled = all(checks.values())
    return {
        "n": len(rows),
        "days": int(rows["date"].nunique()),
        "intervalCoverage": round(coverage, 8),
        "lowerBoundViolationRate": round(lower_rate, 8),
        "upperBoundViolationRate": round(upper_rate, 8),
        "meanIntervalWidth": round(width, 8),
        "pointForecastMae": round(float(np.mean(np.abs(actual - point))), 8),
        "pointForecastRmse": round(float(np.sqrt(np.mean((actual - point) ** 2))), 8),
        "rankIcMean": round(ic_mean, 8) if ic_mean is not None else None,
        "rankIcHacT": round(ic_t, 4) if ic_t is not None else None,
        "rankIcDays": ic_days,
        "positiveLowerBoundObservations": positive_n,
        "positiveLowerBoundHitRateAfterCost": round(positive_hit, 8) if positive_hit is not None else None,
        "checks": checks,
        "intervalEnabled": enabled,
    }


def fit_conformal_return_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> ConformalReturnModel:
    table = prepare_return_decomposition(table)
    base_dates, calibration_dates, audit_dates, segment = nested_segments(fit_dates, config)
    quantiles = list(map(float, config["modelFitting"]["targetWinsorQuantiles"]))
    market_cfg = config["marketReturnModel"]
    residual_cfg = config["residualAlphaModel"]
    market_model = fit_component_model(
        table,
        base_dates,
        market_feature_columns(config),
        "benchmark_return_10d",
        float(market_cfg["alpha"]),
        quantiles,
        True,
    )
    residual_model = fit_component_model(
        table,
        base_dates,
        residual_feature_columns(config),
        "target_residual_10d",
        float(residual_cfg["alpha"]),
        quantiles,
        False,
    )
    calibration = table[
        table["date"].isin(calibration_dates) & table["target_return_10d"].notna()
    ].copy()
    audit_rows = table[
        table["date"].isin(audit_dates) & table["target_return_10d"].notna()
    ].copy()
    if calibration.empty or audit_rows.empty:
        raise RuntimeError("V6 calibration or reliability audit is empty")
    _, _, calibration_point = score_point_forecast(
        market_model, residual_model, calibration, config
    )
    signed_residual = calibration["target_return_10d"].to_numpy(dtype=float) - calibration_point
    interval_cfg = config["splitConformalInterval"]
    lower_offset = date_equal_weighted_quantile(
        signed_residual, calibration["date"], float(interval_cfg["lowerResidualQuantile"])
    )
    upper_offset = date_equal_weighted_quantile(
        signed_residual, calibration["date"], float(interval_cfg["upperResidualQuantile"])
    )
    if lower_offset > upper_offset:
        raise RuntimeError("calibrated lower residual exceeds upper residual")
    _, _, audit_point = score_point_forecast(market_model, residual_model, audit_rows, config)
    audit_lower = audit_point + lower_offset
    audit_upper = audit_point + upper_offset
    reliability = reliability_metrics(
        audit_rows, audit_point, audit_lower, audit_upper, config
    )
    model_audit = {
        **segment,
        "marketModel": market_model.audit,
        "residualModel": residual_model.audit,
        "calibrationRows": len(calibration),
        "calibrationDates": int(calibration["date"].nunique()),
        "lowerResidualOffset": round(lower_offset, 8),
        "upperResidualOffset": round(upper_offset, 8),
        "reliability": reliability,
    }
    return ConformalReturnModel(
        market_model=market_model,
        residual_model=residual_model,
        lower_offset=lower_offset,
        upper_offset=upper_offset,
        interval_enabled=bool(reliability["intervalEnabled"]),
        reliability=reliability,
        audit=model_audit,
    )


def score_conformal_rows(
    model: ConformalReturnModel,
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    market, residual, point = score_point_forecast(
        model.market_model, model.residual_model, rows, config
    )
    output = rows[["date", "securityId"]].copy()
    output["predicted_market_return_10d"] = market
    output["predicted_residual_alpha_10d"] = residual
    output["predicted_total_return_10d"] = point
    output["predicted_return_lower_10d"] = point + model.lower_offset
    output["predicted_return_upper_10d"] = point + model.upper_offset
    output["predicted_interval_width_10d"] = model.upper_offset - model.lower_offset
    output["interval_model_enabled"] = model.interval_enabled
    output["calibration_lower_residual_offset"] = model.lower_offset
    output["calibration_upper_residual_offset"] = model.upper_offset
    output["audit_interval_coverage"] = model.reliability["intervalCoverage"]
    output["audit_lower_bound_violation_rate"] = model.reliability["lowerBoundViolationRate"]
    output["audit_rank_ic_hac_t"] = model.reliability["rankIcHacT"]
    output["audit_positive_lower_bound_n"] = model.reliability["positiveLowerBoundObservations"]
    output["audit_positive_lower_bound_hit_rate"] = model.reliability["positiveLowerBoundHitRateAfterCost"]
    return output


def rolling_conformal_predictions(
    table: pd.DataFrame,
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
        block_rows = table[table["date"].isin(block)].copy()
        if block_rows.empty:
            continue
        model = fit_conformal_return_model(table, fit_dates, config)
        scored = score_conformal_rows(model, block_rows, config)
        fold += 1
        scored["conformal_walk_forward_fold"] = fold
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


def add_context_and_selection(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = rows.copy().reset_index(drop=True)
    context_n = int(config["data"]["contextTopN"])
    output["v2_top10_context"] = output["predicted_order"].le(context_n)
    output["point_forecast_order_within_top10"] = np.nan
    output["lower_bound_order_within_top10"] = np.nan
    context = output[output["v2_top10_context"]].copy()
    point = context.sort_values(
        ["date", "predicted_total_return_10d", "securityId"],
        ascending=[True, False, True],
    )
    point_order = point.groupby("date", sort=False).cumcount() + 1
    output.loc[point.index, "point_forecast_order_within_top10"] = point_order.astype(float)
    lower = context.sort_values(
        ["date", "predicted_return_lower_10d", "securityId"],
        ascending=[True, False, True],
    )
    lower_order = lower.groupby("date", sort=False).cumcount() + 1
    output.loc[lower.index, "lower_bound_order_within_top10"] = lower_order.astype(float)
    max_n = int(config["data"]["maximumSelectionsPerDay"])
    threshold = float(config["selection"]["requiredLowerBoundStrictlyAbove"])
    output["positive_lower_bound_after_cost"] = output["predicted_return_lower_10d"].gt(threshold)
    output["selectedByPrimaryPolicy"] = (
        output["v2_top10_context"]
        & output["lower_bound_order_within_top10"].le(max_n)
        & output["positive_lower_bound_after_cost"]
        & output["interval_model_enabled"]
    )
    output["predicted_lower_bound_net_after_cost"] = (
        output["predicted_return_lower_10d"] - float(config["data"]["roundTripCost"])
    )
    return output.sort_values(["date", "securityId"]).reset_index(drop=True)


def policy_rows(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top3 = int(config["data"]["maximumSelectionsPerDay"])
    return {
        "v2_rank_top10_all_days": rows[rows["predicted_order"].le(10)].copy(),
        "v2_rank_top3_all_days": rows[rows["predicted_order"].le(top3)].copy(),
        "point_forecast_top3_within_v2_top10": rows[
            rows["point_forecast_order_within_top10"].le(top3)
        ].copy(),
        "lower_bound_top3_within_v2_top10": rows[
            rows["lower_bound_order_within_top10"].le(top3)
        ].copy(),
        PRIMARY_POLICY: rows[rows["selectedByPrimaryPolicy"]].copy(),
    }


def interval_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    valid = rows[
        rows["target_return_10d"].notna()
        & rows["predicted_return_lower_10d"].notna()
        & rows["predicted_return_upper_10d"].notna()
    ].copy()
    if valid.empty:
        return {
            "n": 0,
            "days": 0,
            "coverage": None,
            "lowerBoundViolationRate": None,
            "upperBoundViolationRate": None,
            "meanIntervalWidth": None,
            "meanRealizedMinusLowerBound": None,
        }
    actual = valid["target_return_10d"]
    lower = valid["predicted_return_lower_10d"]
    upper = valid["predicted_return_upper_10d"]
    return {
        "n": len(valid),
        "days": int(valid["date"].nunique()),
        "coverage": round(float(((actual >= lower) & (actual <= upper)).mean()), 8),
        "lowerBoundViolationRate": round(float((actual < lower).mean()), 8),
        "upperBoundViolationRate": round(float((actual > upper).mean()), 8),
        "meanIntervalWidth": round(float((upper - lower).mean()), 8),
        "meanRealizedMinusLowerBound": round(float((actual - lower).mean()), 8),
    }


def slot_metrics(selected: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for slot in (1, 2, 3):
        rows = selected[selected["lower_bound_order_within_top10"].eq(float(slot))]
        actual = rows["target_return_10d"].dropna()
        output[f"slot{slot}"] = {
            "n": len(actual),
            "meanReturn": round(float(actual.mean()), 8) if len(actual) else None,
            "medianReturn": round(float(actual.median()), 8) if len(actual) else None,
            "winRate": round(float(actual.gt(0.0).mean()), 8) if len(actual) else None,
            "afterCostWinRate": round(float(actual.gt(0.003).mean()), 8) if len(actual) else None,
            "tailLossRate": round(float(actual.le(-0.03).mean()), 8) if len(actual) else None,
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
        basket = utility_v4.basket_series(selected, target, dates)
        metric.update(tail_v3.tail_outcome_metrics(basket, threshold))
        metrics[name] = metric
        baskets[name] = basket
    primary_rows = policies[PRIMARY_POLICY]
    primary = baskets[PRIMARY_POLICY]
    return {
        "candidateRows": len(labelled),
        "contextTop10Rows": int(labelled["v2_top10_context"].sum()),
        "intervalEnabledCalendarDays": int(
            labelled.loc[labelled["interval_model_enabled"], "date"].nunique()
        ),
        "intervalDisabledCalendarDays": int(
            labelled.loc[~labelled["interval_model_enabled"], "date"].nunique()
        ),
        "contextTop10Interval": interval_metrics(labelled[labelled["v2_top10_context"]]),
        "selectedInterval": interval_metrics(primary_rows),
        "selectedSlots": slot_metrics(primary_rows),
        "policies": metrics,
        "primaryIndependentEvents": winrate_v2.independent_event_metrics(
            primary, calendar_dates, int(config["data"]["holdingTradingDays"])
        ),
        "primaryMonthly": winrate_v2.monthly_metrics(primary),
        "primaryVsV2Top3SameDays": tail_v3.paired_comparison(
            primary,
            baskets["v2_rank_top3_all_days"],
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
        interval = block["selectedInterval"]
        comparison = block["primaryVsV2Top3SameDays"]
        slots = block["selectedSlots"]
        slot_checks = {}
        for name, metric in slots.items():
            slot_checks[name] = {
                "enoughObservations": metric["n"] >= int(rules["minimumPerSlotObservations"]),
                "positiveMeanReturn": metric["meanReturn"] is not None and metric["meanReturn"] > float(rules["minimumPerSlotMeanReturn"]),
                "winRatePass": metric["winRate"] is not None and metric["winRate"] > float(rules["minimumPerSlotWinRate"]),
            }
            slot_checks[name]["slotPass"] = all(slot_checks[name].values())
        period_checks = {
            "enoughSignalDays": primary["newSignalDays"] >= int(rules["minimumSignalDaysPerValidationPeriod"]),
            "enoughIndependentEvents": block["primaryIndependentEvents"]["n"] >= int(rules["minimumIndependentEventsPerValidationPeriod"]),
            "enoughAverageSelections": primary["averageSelectionsOnSignalDay"] >= float(rules["minimumAverageSelectionsPerSignalDay"]),
            "basketWinRatePass": primary["basketTenDayWinRate"] is not None and primary["basketTenDayWinRate"] >= float(rules["minimumBasketWinRate"]),
            "positiveMeanReturn": primary["basketTenDayMeanReturn"] is not None and primary["basketTenDayMeanReturn"] > 0.0,
            "positiveCostedCumulativeReturn": primary["costedCumulativeReturn"] > 0.0,
            "intervalCoveragePass": interval["coverage"] is not None and interval["coverage"] >= float(rules["minimumIntervalCoverage"]),
            "lowerBoundViolationPass": interval["lowerBoundViolationRate"] is not None and interval["lowerBoundViolationRate"] <= float(rules["maximumLowerBoundViolationRate"]),
            "sameDayComparatorCoveragePass": comparison["commonSignalDays"] == primary["newSignalDays"],
            "meanLiftVsV2Top3": comparison["meanReturnDelta"],
            "meanLiftPass": comparison["meanReturnDelta"] is not None and comparison["meanReturnDelta"] > float(rules["minimumSameDayMeanReturnLiftVsV2Top3"]),
            "returnDifferenceHacPass": comparison["returnDeltaTHac"] is not None and comparison["returnDeltaTHac"] >= float(rules["minimumHacTForSameDayReturnDifference"]),
            "tailLossReductionVsV2Top3": comparison["tailLossRateReduction"],
            "tailNoWorsePass": comparison["tailLossRateReduction"] is not None and comparison["tailLossRateReduction"] >= 0.0,
            "allThreeSlotsPass": all(item["slotPass"] for item in slot_checks.values()),
            "slotChecks": slot_checks,
        }
        required = [
            "enoughSignalDays",
            "enoughIndependentEvents",
            "enoughAverageSelections",
            "basketWinRatePass",
            "positiveMeanReturn",
            "positiveCostedCumulativeReturn",
            "intervalCoveragePass",
            "lowerBoundViolationPass",
            "sameDayComparatorCoveragePass",
            "meanLiftPass",
            "returnDifferenceHacPass",
            "tailNoWorsePass",
            "allThreeSlotsPass",
        ]
        period_checks["periodPass"] = all(bool(period_checks[name]) for name in required)
        checks[period] = period_checks
        stable = stable and bool(period_checks["periodPass"])
    return {
        "status": "research_only_not_eligible_for_trading",
        "primaryPolicy": PRIMARY_POLICY,
        "stableHistoricalTop3ProfitabilityEvidence": bool(stable),
        "profitGuaranteed": False,
        "checks": checks,
        "decision": "retain_for_fresh_forward_shadow_only" if stable else "reject_for_trading_keep_diagnostics",
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
        "# Perception-XAlpha conformal Top3 V6",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- Top10 receives point/lower/upper ten-session return estimates.",
        "- Top3 may contain zero to three stocks; a lower bound is not a profit guarantee.",
        "- Orders: always empty.",
        "",
        "## Primary period results",
        "",
        "| Period | Enabled days | Signal days | Avg selected | 10d mean | Win | Tail | Coverage | Lower violations | Costed cumulative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        block = report["periods"][period]
        metric = block["policies"][PRIMARY_POLICY]
        interval = block["selectedInterval"]
        lines.append(
            f"| {period} | {block['intervalEnabledCalendarDays']} | {metric['newSignalDays']} | "
            f"{metric['averageSelectionsOnSignalDay']} | {metric['basketTenDayMeanReturn']} | "
            f"{metric['basketTenDayWinRate']} | {metric['tailLossRate']} | "
            f"{interval['coverage']} | {interval['lowerBoundViolationRate']} | "
            f"{metric['costedCumulativeReturn']} |"
        )
    lines.extend(
        [
            "",
            "## Policy ablation",
            "",
            "| Period | Policy | Days | Avg selected | Mean | Win | Tail | Costed cumulative |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("trainWalkForward", "validation", "shadow"):
        for name, metric in report["periods"][period]["policies"].items():
            lines.append(
                f"| {period} | {name} | {metric['newSignalDays']} | "
                f"{metric['averageSelectionsOnSignalDay']} | {metric['basketTenDayMeanReturn']} | "
                f"{metric['basketTenDayWinRate']} | {metric['tailLossRate']} | "
                f"{metric['costedCumulativeReturn']} |"
            )
    lines.extend(["", "## Selected slot diagnostics", "", "| Period | Slot | N | Mean | Win | After-cost win | Tail |", "|---|---|---:|---:|---:|---:|---:|"])
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
            f"- Stable historical Top3 evidence: `{report['verdict']['stableHistoricalTop3ProfitabilityEvidence']}`",
            "- Profit guaranteed: `false`.",
            "- Reused validation/shadow windows cannot promote this policy.",
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


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    v5_config = load_json(ROOT / config["baseReliabilityConfig"])
    reliability_v5.validate_config(v5_config)
    utility_config = load_json(ROOT / v5_config["baseExpectedUtilityConfig"])
    utility_v4.validate_config(utility_config)
    tail_config = load_json(ROOT / utility_config["baseTailConfig"])
    tail_v3.validate_config(tail_config)
    winrate_config = load_json(ROOT / tail_config["baseWinrateConfig"])
    winrate_v2.validate_config(winrate_config)
    walk = config["walkForward"]
    for key in ("trainingWindowTradingDays", "minimumTrainingTradingDays", "refitEveryTradingDays"):
        if int(walk[key]) != int(winrate_config["rankModel"][key]):
            raise ValueError(f"V6 and V2 rolling calendars differ: {key}")
        if int(walk[key]) != int(tail_config["tailModel"][key]):
            raise ValueError(f"V6 and V3 rolling calendars differ: {key}")
    if int(walk["outerPurgeTradingDays"]) != int(winrate_config["rankModel"]["purgeTradingDays"]):
        raise ValueError("V6 and V2 outer purges differ")
    selector_config = load_json(ROOT / winrate_config["baseSelectorConfig"])
    selector_v1.validate_config(selector_config)
    research_config = load_json(ROOT / selector_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(research_config)
    panel, panel_audit = perception.build_configured_panel(research_config, cog_config)
    factor_ranks = selector_v1.build_factor_rank_frames(panel, selector_config)
    features, market_features = selector_v1.build_past_only_feature_frames(panel, factor_ranks)
    target, one_day = autonomous.target_frames(panel, cog_config)
    raw_table = selector_v1.build_candidate_table(
        panel, features, market_features, target, int(config["data"]["candidatePoolSize"])
    )
    rank_table = winrate_v2.prepare_rank_table(raw_table, winrate_config)
    table = tail_v3.add_tail_label(rank_table, tail_config)
    table = prepare_return_decomposition(table)
    rank_predictions, rank_audits = winrate_v2.rolling_walk_forward_predictions(
        table, panel["close"].index, winrate_config
    )
    tail_predictions, tail_audits = tail_v3.rolling_tail_predictions(
        table, panel["close"].index, tail_config
    )
    interval_predictions, interval_audits = rolling_conformal_predictions(
        table, panel["close"].index, config
    )
    if rank_predictions.empty or tail_predictions.empty or interval_predictions.empty:
        raise RuntimeError("one or more V6 prediction tables are empty")
    predictions = rank_predictions.merge(
        tail_predictions, on=["date", "securityId"], how="inner", validate="one_to_one"
    ).merge(
        interval_predictions, on=["date", "securityId"], how="inner", validate="one_to_one"
    )
    predictions = add_context_and_selection(predictions, config)
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
    latest_top10 = latest[latest["v2_top10_context"]].sort_values(
        ["predicted_return_lower_10d", "securityId"], ascending=[False, True]
    )
    latest_columns = [
        "date",
        "securityId",
        "predicted_order",
        "predicted_cross_sectional_rank",
        "predicted_tail_probability",
        "predicted_market_return_10d",
        "predicted_residual_alpha_10d",
        "predicted_total_return_10d",
        "predicted_return_lower_10d",
        "predicted_return_upper_10d",
        "predicted_interval_width_10d",
        "predicted_lower_bound_net_after_cost",
        "lower_bound_order_within_top10",
        "interval_model_enabled",
        "audit_interval_coverage",
        "audit_lower_bound_violation_rate",
        "audit_rank_ic_hac_t",
        "audit_positive_lower_bound_n",
        "audit_positive_lower_bound_hit_rate",
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
        "dataRange": [panel["close"].index.min().date().isoformat(), panel["close"].index.max().date().isoformat()],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "predictionRows": len(predictions),
        "marketFeatureDefinition": market_feature_columns(config),
        "residualFeatureDefinition": residual_feature_columns(config),
        "labelDefinition": {
            "total": "open[t+11]/open[t+1]-1; offline label only",
            "market": "same-date eligible A-share mean total return; offline label only",
            "residual": "stock total return minus same-date market return; offline label only",
        },
        "rankWalkForwardAudits": rank_audits,
        "tailWalkForwardAudits": tail_audits,
        "conformalWalkForwardAudits": interval_audits,
        "periods": {},
        "latestDate": latest_date.date().isoformat(),
        "latestTop10": _records(latest_top10[latest_columns]),
        "latestSelectedCount": int(latest_top10["selectedByPrimaryPolicy"].sum()),
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    for name, dates in {
        "trainWalkForward": train_walk_dates,
        "validation": validation_dates,
        "shadow": shadow_dates,
    }.items():
        report["periods"][name] = period_report(
            predictions, dates, target, one_day, panel["eligible"], panel["close"].index, config
        )
    report["verdict"] = build_verdict(report, config)
    out = ROOT / config["output"]["root"] / run_id
    atomic_write_text(out / "summary.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(out / "report.md", markdown_report(report))
    atomic_write_text(
        out / "latest_top10_return_intervals.csv",
        latest_top10[latest_columns].to_csv(index=False, lineterminator="\n"),
    )
    atomic_write_text(
        out / "latest_ranking.csv",
        latest.sort_values(["predicted_order", "securityId"])[latest_columns].to_csv(index=False, lineterminator="\n"),
    )
    print(
        json.dumps(
            {
                "runId": run_id,
                "output": str(out),
                "dataRange": report["dataRange"],
                "predictionRows": len(predictions),
                "latestDate": report["latestDate"],
                "latestTop10": len(latest_top10),
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
    args = parser.parse_args()
    run(args.config.resolve(), args.run_id or None)


if __name__ == "__main__":
    main()
