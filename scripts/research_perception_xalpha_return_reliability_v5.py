"""Nested reliability-gated expected-return research for all A shares.

V5 keeps the frozen V2 rank and V3 tail heads.  Within every outer rolling
training window it reserves, in chronological order, a base-fit segment, a
calibration segment and a never-fit reliability-audit segment, with ten-session
purges between them.  The expected-return head enters the score only when that
audit passes all preregistered checks.  This module is offline diagnostics only.
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
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_expected_utility_v4 as utility_v4  # noqa: E402
import research_perception_xalpha_two_stage as selector_v1  # noqa: E402
import research_perception_xalpha_winrate_tail_v3 as tail_v3  # noqa: E402
import research_perception_xalpha_winrate_v2 as winrate_v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_return_reliability_v5.json"
)
SCHEMA_VERSION = "perception_xalpha_return_reliability_v5"
CODE_VERSION = "perception_xalpha_return_reliability_v5.0"
PRIMARY_POLICY = "reliability_adaptive_top10_gated"


@dataclass
class NestedReturnModel:
    raw_model: utility_v4.ExpectedReturnModel
    calibrator: Ridge
    expected_head_enabled: bool
    reliability: dict[str, Any]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected return-reliability schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("return-reliability study must remain research-only")
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
    model = config["expectedReturnModel"]
    data = config["data"]
    calibration = config["returnCalibration"]
    reliability = config["reliabilityGate"]
    integration = config["integration"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the expected-return horizon must remain ten sessions")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("the frozen candidate pool must remain top fifty")
    if int(data["maximumSelectionsPerDay"]) != 10:
        raise ValueError("the adaptive book must remain top ten")
    if model.get("kind") != "Ridge" or float(model["alpha"]) != 10.0:
        raise ValueError("the base expected-return model changed")
    if int(model["outerPurgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("outer purge must cover the label horizon")
    if int(model["baseToCalibrationPurgeTradingDays"]) < 10:
        raise ValueError("base-to-calibration purge is too short")
    if int(model["calibrationToAuditPurgeTradingDays"]) < 10:
        raise ValueError("calibration-to-audit purge is too short")
    if int(model["calibrationFitTradingDays"]) != 63:
        raise ValueError("calibration-fit segment changed")
    if int(model["reliabilityAuditTradingDays"]) != 63:
        raise ValueError("reliability-audit segment changed")
    if list(map(float, model["targetWinsorQuantiles"])) != [0.01, 0.99]:
        raise ValueError("base-fit target winsorization changed")
    if calibration.get("kind") != "Ridge" or float(calibration["alpha"]) != 10.0:
        raise ValueError("return calibrator changed")
    if float(calibration["inputScale"]) != 100.0:
        raise ValueError("return calibrator input scale changed")
    for flag in (
        "targetClipUsesBaseFitBounds",
        "predictionClipUsesBaseFitBounds",
        "fitUsesCalibrationSegmentOnly",
        "auditSegmentNeverFitsModelOrCalibrator",
    ):
        if calibration.get(flag) is not True:
            raise ValueError(f"nested calibration safety changed: {flag}")
    if float(reliability["minimumAuditOosR2Exclusive"]) != 0.0:
        raise ValueError("reliability OOS R2 threshold changed")
    if float(reliability["minimumAuditRankIcHacT"]) != 1.65:
        raise ValueError("reliability Rank IC threshold changed")
    if float(reliability["minimumAuditBasketDirectionAccuracy"]) != 0.5:
        raise ValueError("reliability direction threshold changed")
    if reliability.get("requirePositiveCalibrationSlope") is not True:
        raise ValueError("calibration slope must remain positive")
    enabled = integration["enabledHeadWeights"]
    disabled = integration["disabledHeadWeights"]
    if not np.allclose(
        [enabled["rank"], enabled["expectedReturn"], enabled["inverseTailRisk"]],
        [0.4, 0.4, 0.2],
    ):
        raise ValueError("enabled-head weights changed")
    if not np.allclose(
        [disabled["rank"], disabled["expectedReturn"], disabled["inverseTailRisk"]],
        [1.0, 0.0, 0.0],
    ):
        raise ValueError("disabled-head fallback changed")
    if not np.isclose(
        float(integration["minimumSelectedMeanExpectedReturn"]),
        float(data["roundTripCost"]),
    ):
        raise ValueError("expected net-return threshold changed")
    if integration.get("allowCash") is not True:
        raise ValueError("the reliability policy must allow cash")
    if integration.get("neverForceTenSelections") is not True:
        raise ValueError("the reliability policy cannot force ten stocks")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("a historical run cannot promote")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain permanently empty")


def expected_return_feature_columns(config: dict[str, Any]) -> list[str]:
    columns = list(config["expectedReturnModel"]["featureColumns"])
    forbidden = {
        "target_return_10d",
        "target_cross_sectional_rank_10d",
        "label_tail_loss_10d",
        "label_positive_10d",
    }
    overlap = forbidden.intersection(columns)
    if overlap:
        raise ValueError(f"future outcome entered V5 features: {sorted(overlap)}")
    return columns


def nested_segments(
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, dict[str, Any]]:
    model = config["expectedReturnModel"]
    calibration_days = int(model["calibrationFitTradingDays"])
    audit_days = int(model["reliabilityAuditTradingDays"])
    base_gap = int(model["baseToCalibrationPurgeTradingDays"])
    audit_gap = int(model["calibrationToAuditPurgeTradingDays"])
    audit_start = len(fit_dates) - audit_days
    calibration_end = audit_start - audit_gap
    calibration_start = calibration_end - calibration_days
    base_end = calibration_start - base_gap
    if base_end < int(model["minimumBaseFitTradingDays"]):
        raise RuntimeError("nested base-fit segment is too short")
    base = fit_dates[:base_end]
    calibration = fit_dates[calibration_start:calibration_end]
    audit = fit_dates[audit_start:]
    if not (base[-1] < calibration[0] < calibration[-1] < audit[0]):
        raise RuntimeError("nested return segments are not strictly chronological")
    return base, calibration, audit, {
        "baseFitDateRange": [base[0].date().isoformat(), base[-1].date().isoformat()],
        "baseFitTradingDays": len(base),
        "baseToCalibrationPurgeTradingDays": base_gap,
        "calibrationFitDateRange": [
            calibration[0].date().isoformat(),
            calibration[-1].date().isoformat(),
        ],
        "calibrationFitTradingDays": len(calibration),
        "calibrationToAuditPurgeTradingDays": audit_gap,
        "reliabilityAuditDateRange": [
            audit[0].date().isoformat(),
            audit[-1].date().isoformat(),
        ],
        "reliabilityAuditTradingDays": len(audit),
    }


def _return_rows(table: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return table[
        table["date"].isin(dates) & table["target_return_10d"].notna()
    ].copy()


def _raw_predictions(
    model: utility_v4.ExpectedReturnModel,
    rows: pd.DataFrame,
) -> np.ndarray:
    return model.pipeline.predict(
        rows[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    )


def _calibrated_predictions(
    raw: np.ndarray,
    calibrator: Ridge,
    raw_model: utility_v4.ExpectedReturnModel,
    config: dict[str, Any],
) -> np.ndarray:
    scale = float(config["returnCalibration"]["inputScale"])
    predicted = calibrator.predict(np.asarray(raw, dtype=float).reshape(-1, 1) * scale)
    if config["returnCalibration"]["predictionClipUsesBaseFitBounds"]:
        predicted = np.clip(predicted, *raw_model.target_clip)
    return predicted


def _reliability_metrics(
    rows: pd.DataFrame,
    predicted: np.ndarray,
    prior: float,
    calibration_slope: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    actual = rows["target_return_10d"].to_numpy(dtype=float)
    denominator = float(np.sum((actual - prior) ** 2))
    oos_r2 = 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator if denominator > 0 else None
    scored = rows[["date", "securityId", "target_return_10d"]].copy()
    scored["prediction"] = predicted
    daily_ic: list[float] = []
    daily_direction: list[float] = []
    for _, group in scored.groupby("date", sort=True):
        if (
            len(group) >= 3
            and group["prediction"].nunique() > 1
            and group["target_return_10d"].nunique() > 1
        ):
            value = group["prediction"].corr(group["target_return_10d"])
            if pd.notna(value):
                daily_ic.append(float(value))
        predicted_basket = float(group["prediction"].mean())
        actual_basket = float(group["target_return_10d"].mean())
        daily_direction.append(float((predicted_basket > 0.0) == (actual_basket > 0.0)))
    ic_values = np.asarray(daily_ic, dtype=float)
    lag = int(config["evaluation"]["hacLagTradingDays"])
    ic_t = autonomous.newey_west_t(ic_values, lag) if len(ic_values) >= 3 else None
    direction = float(np.mean(daily_direction)) if daily_direction else None
    gate = config["reliabilityGate"]
    checks = {
        "positiveOosR2": oos_r2 is not None
        and oos_r2 > float(gate["minimumAuditOosR2Exclusive"]),
        "rankIcHacPass": ic_t is not None
        and ic_t >= float(gate["minimumAuditRankIcHacT"]),
        "directionAccuracyPass": direction is not None
        and direction >= float(gate["minimumAuditBasketDirectionAccuracy"]),
        "positiveCalibrationSlope": calibration_slope > 0.0,
    }
    enabled = all(checks.values())
    return {
        "n": len(rows),
        "days": int(rows["date"].nunique()),
        "predictedMean": round(float(np.mean(predicted)), 8),
        "actualMean": round(float(np.mean(actual)), 8),
        "oosR2VsBaseFitPrior": round(float(oos_r2), 8)
        if oos_r2 is not None
        else None,
        "rankIcMean": round(float(ic_values.mean()), 8) if len(ic_values) else None,
        "rankIcHacT": round(float(ic_t), 4) if ic_t is not None else None,
        "basketDirectionAccuracy": round(float(direction), 8)
        if direction is not None
        else None,
        "calibrationSlope": round(float(calibration_slope), 8),
        "checks": checks,
        "expectedHeadEnabled": enabled,
    }


def fit_nested_return_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> NestedReturnModel:
    base_dates, calibration_dates, audit_dates, segment_audit = nested_segments(
        fit_dates, config
    )
    raw_model = utility_v4.fit_expected_return_model(table, base_dates, config)
    calibration_rows = _return_rows(table, calibration_dates)
    audit_rows = _return_rows(table, audit_dates)
    if calibration_rows.empty or audit_rows.empty:
        raise RuntimeError("nested calibration or reliability audit is empty")
    raw_calibration = _raw_predictions(raw_model, calibration_rows)
    target = calibration_rows["target_return_10d"].astype(float).clip(
        *raw_model.target_clip
    )
    calibration_cfg = config["returnCalibration"]
    scale = float(calibration_cfg["inputScale"])
    calibrator = Ridge(alpha=float(calibration_cfg["alpha"]))
    calibrator.fit(raw_calibration.reshape(-1, 1) * scale, target)
    calibration_slope = float(calibrator.coef_[0]) * scale
    raw_audit = _raw_predictions(raw_model, audit_rows)
    predicted_audit = _calibrated_predictions(
        raw_audit, calibrator, raw_model, config
    )
    reliability = _reliability_metrics(
        audit_rows,
        predicted_audit,
        raw_model.training_prior,
        calibration_slope,
        config,
    )
    audit = {
        **segment_audit,
        "baseModel": raw_model.audit,
        "calibrationRows": len(calibration_rows),
        "calibrationIntercept": round(float(calibrator.intercept_), 8),
        "calibrationSlope": round(calibration_slope, 8),
        "auditRows": len(audit_rows),
        "reliability": reliability,
    }
    return NestedReturnModel(
        raw_model=raw_model,
        calibrator=calibrator,
        expected_head_enabled=bool(reliability["expectedHeadEnabled"]),
        reliability=reliability,
        audit=audit,
    )


def score_nested_return_rows(
    model: NestedReturnModel,
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    output = rows[["date", "securityId"]].copy()
    raw = _raw_predictions(model.raw_model, rows)
    calibrated = _calibrated_predictions(
        raw, model.calibrator, model.raw_model, config
    )
    output["raw_expected_return_10d"] = raw
    output["predicted_expected_return_10d"] = calibrated
    output["training_expected_return_prior"] = model.raw_model.training_prior
    output["expected_head_enabled"] = model.expected_head_enabled
    output["reliability_audit_oos_r2"] = model.reliability[
        "oosR2VsBaseFitPrior"
    ]
    output["reliability_audit_rank_ic_t"] = model.reliability["rankIcHacT"]
    output["reliability_audit_direction_accuracy"] = model.reliability[
        "basketDirectionAccuracy"
    ]
    output["return_calibration_slope"] = model.reliability["calibrationSlope"]
    return output


def rolling_nested_return_predictions(
    table: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(calendar_dates)))
    model_cfg = config["expectedReturnModel"]
    minimum = int(model_cfg["minimumTrainingTradingDays"])
    window = int(model_cfg["trainingWindowTradingDays"])
    purge = int(model_cfg["outerPurgeTradingDays"])
    refit = int(model_cfg["refitEveryTradingDays"])
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
        model = fit_nested_return_model(table, fit_dates, config)
        predicted = score_nested_return_rows(model, block_rows, config)
        fold += 1
        predicted["nested_return_walk_forward_fold"] = fold
        predictions.append(predicted)
        audits.append(
            {
                "fold": fold,
                "predictionDateRange": [
                    block[0].date().isoformat(),
                    block[-1].date().isoformat(),
                ],
                "predictionRows": len(predicted),
                "outerPurgeTradingDays": purge,
                "outerGapTradingDays": int(start - (fit_stop - 1) - 1),
                "model": model.audit,
            }
        )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        audits,
    )


def add_reliability_adaptive_score(
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    output = rows.copy()
    output["expected_return_percentile"] = output.groupby("date", sort=False)[
        "predicted_expected_return_10d"
    ].rank(pct=True, method="average")
    output["inverse_tail_probability_percentile"] = output.groupby(
        "date", sort=False
    )["predicted_tail_probability"].rank(
        pct=True, method="average", ascending=False
    )
    output["fixed_equal_weight_score"] = (
        output["rank_score_percentile"]
        + output["expected_return_percentile"]
        + output["inverse_tail_probability_percentile"]
    ) / 3.0
    enabled_weights = config["integration"]["enabledHeadWeights"]
    enabled_score = (
        float(enabled_weights["rank"]) * output["rank_score_percentile"]
        + float(enabled_weights["expectedReturn"])
        * output["expected_return_percentile"]
        + float(enabled_weights["inverseTailRisk"])
        * output["inverse_tail_probability_percentile"]
    )
    output["reliability_adaptive_score"] = np.where(
        output["expected_head_enabled"],
        enabled_score,
        output["rank_score_percentile"],
    )
    output = output.sort_values(
        ["date", "fixed_equal_weight_score", "securityId"],
        ascending=[True, False, True],
    )
    output["fixed_equal_weight_order"] = output.groupby(
        "date", sort=False
    ).cumcount() + 1
    output = output.sort_values(
        ["date", "reliability_adaptive_score", "securityId"],
        ascending=[True, False, True],
    )
    output["reliability_adaptive_order"] = output.groupby(
        "date", sort=False
    ).cumcount() + 1
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    provisional = output[output["reliability_adaptive_order"].le(top_n)]
    selected_mean = provisional.groupby("date", sort=False)[
        "predicted_expected_return_10d"
    ].mean()
    output["selected_mean_expected_return_10d"] = output["date"].map(selected_mean)
    integration = config["integration"]
    output["risk_on_fallback"] = output["market_return_20"].gt(
        float(integration["marketReturn20MinimumExclusive"])
    ) & output["market_breadth_20"].ge(
        float(integration["marketBreadth20MinimumInclusive"])
    )
    expected_gate = output["selected_mean_expected_return_10d"].gt(
        float(integration["minimumSelectedMeanExpectedReturn"])
    )
    output["reliability_adaptive_trade_gate"] = np.where(
        output["expected_head_enabled"], expected_gate, output["risk_on_fallback"]
    ).astype(bool)
    output["trade_gate_mode"] = np.where(
        output["expected_head_enabled"], "calibrated_expected_return", "v2_risk_on"
    )
    output["predicted_net_return_after_round_trip_cost"] = (
        output["predicted_expected_return_10d"]
        - float(config["data"]["roundTripCost"])
    )
    return output.sort_values(["date", "securityId"]).reset_index(drop=True)


def policy_rows(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    return {
        "frozen_factor_top10_all_days": rows[rows["candidate_rank"].le(top_n)].copy(),
        "rolling_ridge_rank_top10_all_days": rows[
            rows["predicted_order"].le(top_n)
        ].copy(),
        "rolling_ridge_rank_top10_risk_on": rows[
            rows["predicted_order"].le(top_n) & rows["risk_on_fallback"]
        ].copy(),
        "fixed_equal_weight_integrated_top10_all_days": rows[
            rows["fixed_equal_weight_order"].le(top_n)
        ].copy(),
        "reliability_adaptive_top10_all_days": rows[
            rows["reliability_adaptive_order"].le(top_n)
        ].copy(),
        PRIMARY_POLICY: rows[
            rows["reliability_adaptive_order"].le(top_n)
            & rows["reliability_adaptive_trade_gate"]
        ].copy(),
    }


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
        metric = selector_v1.portfolio_metrics(
            selected, target, one_day, eligible, dates, config
        )
        basket = utility_v4.basket_series(selected, target, dates)
        metric.update(tail_v3.tail_outcome_metrics(basket, threshold))
        metrics[name] = metric
        baskets[name] = basket
    primary = baskets[PRIMARY_POLICY]
    lag = int(config["evaluation"]["hacLagTradingDays"])
    horizon = int(config["data"]["holdingTradingDays"])
    enabled_rows = labelled[labelled["expected_head_enabled"]].copy()
    enabled_selected = enabled_rows[
        enabled_rows["reliability_adaptive_order"].le(
            int(config["data"]["maximumSelectionsPerDay"])
        )
    ]
    enabled_dates = pd.DatetimeIndex(
        sorted(pd.unique(enabled_selected["date"]))
    )
    enabled_primary = utility_v4.basket_series(enabled_selected, target, enabled_dates)
    rank_selected_enabled = enabled_rows[enabled_rows["predicted_order"].le(10)]
    enabled_rank = utility_v4.basket_series(
        rank_selected_enabled, target, enabled_dates
    )
    return {
        "candidateRows": len(labelled),
        "headEnabledCalendarDays": int(
            labelled.loc[labelled["expected_head_enabled"], "date"].nunique()
        ),
        "headDisabledCalendarDays": int(
            labelled.loc[~labelled["expected_head_enabled"], "date"].nunique()
        ),
        "primaryExpectedHeadEnabledSignalDays": int(
            policies[PRIMARY_POLICY]
            .loc[policies[PRIMARY_POLICY]["expected_head_enabled"], "date"]
            .nunique()
        ),
        "expectedReturnForecastAll": utility_v4.expected_return_metrics(
            labelled, lag
        ),
        "expectedReturnForecastEnabledOnly": utility_v4.expected_return_metrics(
            enabled_rows, lag
        ),
        "policies": metrics,
        "primaryIndependentEvents": winrate_v2.independent_event_metrics(
            primary, calendar_dates, horizon
        ),
        "primaryMonthly": winrate_v2.monthly_metrics(primary),
        "primaryVsRankTop10SameDays": tail_v3.paired_comparison(
            primary,
            baskets["rolling_ridge_rank_top10_all_days"],
            lag,
            threshold,
        ),
        "enabledSelectionVsRankTop10SameDays": tail_v3.paired_comparison(
            enabled_primary, enabled_rank, lag, threshold
        ),
    }


def build_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["evaluation"]
    checks: dict[str, Any] = {}
    stable = True
    for period in ("validation", "shadow"):
        block = report["periods"][period]
        primary = block["policies"][PRIMARY_POLICY]
        overall = block["primaryVsRankTop10SameDays"]
        enabled = block["enabledSelectionVsRankTop10SameDays"]
        independent = block["primaryIndependentEvents"]
        period_checks = {
            "enoughSignalDays": primary["newSignalDays"]
            >= int(rules["minimumSignalDaysPerValidationPeriod"]),
            "enoughIndependentEvents": independent["n"]
            >= int(rules["minimumIndependentEventsPerValidationPeriod"]),
            "enoughExpectedHeadEnabledSignalDays": block[
                "primaryExpectedHeadEnabledSignalDays"
            ]
            >= int(rules["minimumExpectedHeadEnabledSignalDaysPerPeriod"]),
            "sameDayComparatorCoveragePass": overall["commonSignalDays"]
            == primary["newSignalDays"],
            "meanReturnLiftVsRankTop10": overall["meanReturnDelta"],
            "meanReturnLiftPass": overall["meanReturnDelta"] is not None
            and overall["meanReturnDelta"]
            > float(rules["minimumSameDayMeanReturnLiftVsRankTop10"]),
            "winRateLiftVsRankTop10": overall["winRateDelta"],
            "winRateNoWorsePass": overall["winRateDelta"] is not None
            and overall["winRateDelta"]
            >= float(rules["minimumSameDayWinRateLiftVsRankTop10"]),
            "tailLossReductionVsRankTop10": overall["tailLossRateReduction"],
            "tailLossNoWorsePass": overall["tailLossRateReduction"] is not None
            and overall["tailLossRateReduction"]
            >= float(rules["minimumTailLossRateReductionVsRankTop10"]),
            "overallReturnHacPass": overall["returnDeltaTHac"] is not None
            and overall["returnDeltaTHac"]
            >= float(rules["minimumHacTForSameDayReturnDifference"]),
            "enabledHeadMeanLiftPass": enabled["meanReturnDelta"] is not None
            and enabled["meanReturnDelta"] > 0.0,
            "enabledHeadWinNoWorsePass": enabled["winRateDelta"] is not None
            and enabled["winRateDelta"] >= 0.0,
            "enabledHeadTailNoWorsePass": enabled["tailLossRateReduction"]
            is not None
            and enabled["tailLossRateReduction"] >= 0.0,
            "enabledHeadReturnHacPass": enabled["returnDeltaTHac"] is not None
            and enabled["returnDeltaTHac"]
            >= float(rules["minimumHacTForSameDayReturnDifference"]),
            "positiveMeanReturn": primary["basketTenDayMeanReturn"] is not None
            and primary["basketTenDayMeanReturn"] > 0.0,
            "positiveCostedCumulativeReturn": primary["costedCumulativeReturn"]
            > 0.0,
        }
        required = {
            key for key in period_checks if key.endswith("Pass")
        } | {
            "enoughSignalDays",
            "enoughIndependentEvents",
            "enoughExpectedHeadEnabledSignalDays",
            "positiveMeanReturn",
            "positiveCostedCumulativeReturn",
        }
        period_checks["periodPass"] = all(
            bool(period_checks[name]) for name in required
        )
        checks[period] = period_checks
        stable = stable and period_checks["periodPass"]
    return {
        "status": "research_only_not_eligible_for_trading",
        "primaryPolicy": PRIMARY_POLICY,
        "stableHistoricalReliabilityImprovement": bool(stable),
        "checks": checks,
        "decision": (
            "retain_as_forward_research_hypothesis_only"
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
        "# Perception-XAlpha return reliability V5",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- Expected return uses nested base-fit, calibration-fit and never-fit audit segments.",
        "- The return head enters scoring only after positive audit R2, IC, direction and slope checks.",
        "- Reliability failure falls back to frozen V2 rank plus the frozen V2 risk-on gate.",
        "- Orders: always empty.",
        "",
        "## Period summary",
        "",
        "| Period | Head-enabled days | Head-disabled days | Primary days | 10d mean | Win | Tail | Costed cumulative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        block = report["periods"][period]
        metric = block["policies"][PRIMARY_POLICY]
        lines.append(
            f"| {period} | {block['headEnabledCalendarDays']} | "
            f"{block['headDisabledCalendarDays']} | {metric['newSignalDays']} | "
            f"{metric['basketTenDayMeanReturn']} | {metric['basketTenDayWinRate']} | "
            f"{metric['tailLossRate']} | {metric['costedCumulativeReturn']} |"
        )
    lines.extend(
        [
            "",
            "## Policy ablation",
            "",
            "| Period | Policy | Days | Mean | Win | Tail | Costed cumulative | Max drawdown |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("trainWalkForward", "validation", "shadow"):
        for name, metric in report["periods"][period]["policies"].items():
            lines.append(
                f"| {period} | {name} | {metric['newSignalDays']} | "
                f"{metric['basketTenDayMeanReturn']} | {metric['basketTenDayWinRate']} | "
                f"{metric['tailLossRate']} | {metric['costedCumulativeReturn']} | "
                f"{metric['costedMaximumDrawdown']} |"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Decision: `{report['verdict']['decision']}`",
            f"- Stable validation and shadow improvement: `{report['verdict']['stableHistoricalReliabilityImprovement']}`",
            "- Both the overall policy and the expected-head-enabled subset must beat V2 on identical dates.",
            "- The reused historical windows cannot promote or connect to trading.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    utility_config = load_json(ROOT / config["baseExpectedUtilityConfig"])
    utility_v4.validate_config(utility_config)
    tail_config = load_json(ROOT / utility_config["baseTailConfig"])
    tail_v3.validate_config(tail_config)
    winrate_config = load_json(ROOT / tail_config["baseWinrateConfig"])
    winrate_v2.validate_config(winrate_config)
    for key in (
        "trainingWindowTradingDays",
        "minimumTrainingTradingDays",
        "refitEveryTradingDays",
    ):
        expected = int(config["expectedReturnModel"][key])
        if expected != int(tail_config["tailModel"][key]):
            raise ValueError(f"V5 and tail walk-forward calendars differ: {key}")
        if expected != int(winrate_config["rankModel"][key]):
            raise ValueError(f"V5 and rank walk-forward calendars differ: {key}")
    outer_purge = int(config["expectedReturnModel"]["outerPurgeTradingDays"])
    if outer_purge != int(tail_config["tailModel"]["purgeTradingDays"]):
        raise ValueError("V5 and tail outer purges differ")
    if outer_purge != int(winrate_config["rankModel"]["purgeTradingDays"]):
        raise ValueError("V5 and rank outer purges differ")
    selector_config = load_json(ROOT / winrate_config["baseSelectorConfig"])
    selector_v1.validate_config(selector_config)
    research_config = load_json(ROOT / selector_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(research_config)
    panel, panel_audit = perception.build_configured_panel(research_config, cog_config)
    factor_ranks = selector_v1.build_factor_rank_frames(panel, selector_config)
    features, market_features = selector_v1.build_past_only_feature_frames(
        panel, factor_ranks
    )
    target, one_day = autonomous.target_frames(panel, cog_config)
    raw_table = selector_v1.build_candidate_table(
        panel,
        features,
        market_features,
        target,
        int(config["data"]["candidatePoolSize"]),
    )
    rank_table = winrate_v2.prepare_rank_table(raw_table, winrate_config)
    table = tail_v3.add_tail_label(rank_table, tail_config)
    rank_predictions, rank_audits = winrate_v2.rolling_walk_forward_predictions(
        table, panel["close"].index, winrate_config
    )
    tail_predictions, tail_audits = tail_v3.rolling_tail_predictions(
        table, panel["close"].index, tail_config
    )
    return_predictions, return_audits = rolling_nested_return_predictions(
        table, panel["close"].index, config
    )
    if rank_predictions.empty or tail_predictions.empty or return_predictions.empty:
        raise RuntimeError("one or more V5 prediction tables are empty")
    predictions = rank_predictions.merge(
        tail_predictions,
        on=["date", "securityId"],
        how="inner",
        validate="one_to_one",
    ).merge(
        return_predictions,
        on=["date", "securityId"],
        how="inner",
        validate="one_to_one",
    )
    predictions = tail_v3.add_tail_adjustment(predictions, tail_config)
    predictions = add_reliability_adaptive_score(predictions, config)
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
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": selector_v1.digest(config),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "predictionRows": len(predictions),
        "featureDefinition": expected_return_feature_columns(config),
        "rankWalkForwardAudits": rank_audits,
        "tailWalkForwardAudits": tail_audits,
        "nestedReturnWalkForwardAudits": return_audits,
        "periods": {},
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
    atomic_write_text(
        out / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(out / "report.md", markdown_report(report))
    latest_date = predictions["date"].max()
    latest = predictions[predictions["date"].eq(latest_date)].copy()
    latest["selectedByPrimaryPolicy"] = (
        latest["reliability_adaptive_order"].le(
            int(config["data"]["maximumSelectionsPerDay"])
        )
        & latest["reliability_adaptive_trade_gate"]
    )
    latest_columns = [
        "date",
        "securityId",
        "candidate_rank",
        "factor_composite",
        "predicted_cross_sectional_rank",
        "rank_score_percentile",
        "raw_expected_return_10d",
        "predicted_expected_return_10d",
        "predicted_net_return_after_round_trip_cost",
        "expected_return_percentile",
        "predicted_tail_probability",
        "inverse_tail_probability_percentile",
        "expected_head_enabled",
        "reliability_audit_oos_r2",
        "reliability_audit_rank_ic_t",
        "reliability_audit_direction_accuracy",
        "return_calibration_slope",
        "reliability_adaptive_score",
        "reliability_adaptive_order",
        "selected_mean_expected_return_10d",
        "risk_on_fallback",
        "trade_gate_mode",
        "reliability_adaptive_trade_gate",
        "selectedByPrimaryPolicy",
    ]
    atomic_write_text(
        out / "latest_ranking.csv",
        latest[latest_columns].to_csv(index=False, lineterminator="\n"),
    )
    print(
        json.dumps(
            {
                "runId": run_id,
                "output": str(out),
                "dataRange": report["dataRange"],
                "predictionRows": len(predictions),
                "latestDate": latest_date.date().isoformat(),
                "latestExpectedHeadEnabled": bool(
                    latest["expected_head_enabled"].iloc[0]
                ),
                "latestGateMode": latest["trade_gate_mode"].iloc[0],
                "latestGate": bool(
                    latest["reliability_adaptive_trade_gate"].iloc[0]
                ),
                "latestSelections": int(latest["selectedByPrimaryPolicy"].sum()),
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
