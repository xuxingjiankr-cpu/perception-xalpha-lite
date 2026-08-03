"""Research-only stock-specific return intervals for the frozen V6 Top10.

V7 preserves V6 point forecasts exactly.  Its only increment is a frozen
sqrt-time scale from past 20-session stock volatility; calibration residuals
are divided by that scale before date-equal-weighted interval quantiles are
estimated.  The module is offline diagnostics only and cannot trade.
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_conformal_top3_v6 as conformal_v6  # noqa: E402
import research_perception_xalpha_expected_utility_v4 as utility_v4  # noqa: E402
import research_perception_xalpha_return_reliability_v5 as reliability_v5  # noqa: E402
import research_perception_xalpha_two_stage as selector_v1  # noqa: E402
import research_perception_xalpha_winrate_tail_v3 as tail_v3  # noqa: E402
import research_perception_xalpha_winrate_v2 as winrate_v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_conditional_top3_v7.json"
)
SCHEMA_VERSION = "perception_xalpha_conditional_top3_v7"
CODE_VERSION = "perception_xalpha_conditional_top3_v7.0"
PRIMARY_POLICY = "conditional_conformal_top3_positive_lower_bound"


@dataclass
class FrozenScale:
    source_field: str
    multiplier: float
    holding_days: int
    floor: float
    cap: float
    median: float
    audit: dict[str, Any]


@dataclass
class ConditionalReturnModel:
    market_model: conformal_v6.ComponentModel
    residual_model: conformal_v6.ComponentModel
    scale: FrozenScale
    lower_standardized_offset: float
    upper_standardized_offset: float
    interval_enabled: bool
    reliability: dict[str, Any]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any], base: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected conditional Top3 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V7 must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("V7 output must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all V7 trading mutation permissions must remain false")
    base_path = ROOT / config["baseConformalConfig"]
    if file_sha256(base_path) != config["baseConformalConfigFileSha256"]:
        raise ValueError("the frozen V6 base config file changed")
    conformal_v6.validate_config(base)
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("primaryPolicy") != PRIMARY_POLICY:
        raise ValueError("the V7 primary policy changed")
    if hypothesis.get("onlyIncrementVsV6") != "stock_specific_past_volatility_scale":
        raise ValueError("V7 may change only conditional uncertainty scale")
    if hypothesis.get("v6PointForecastMustRemainIdentical") is not True:
        raise ValueError("V7 point forecasts must remain identical to V6")
    if hypothesis.get("parametersFrozenBeforeHistoricalRun") is not True:
        raise ValueError("V7 parameters must be frozen before evaluation")
    if hypothesis.get("validationAndShadowMayNotTuneParameters") is not True:
        raise ValueError("V7 evaluation windows may not tune parameters")
    if hypothesis.get("profitGuaranteeClaimAllowed") is not False:
        raise ValueError("V7 may not claim guaranteed profits")
    scale = config["conditionalScale"]
    if scale.get("sourceField") != "stock_volatility_20":
        raise ValueError("conditional scale source changed")
    if scale.get("usesOnlySignalDateAndEarlier") is not True:
        raise ValueError("conditional scale must remain past-only")
    if scale.get("formula") != "sqrt_holding_days_times_absolute_stock_volatility_20":
        raise ValueError("conditional scale formula changed")
    if int(scale["holdingDays"]) != int(base["data"]["holdingTradingDays"]):
        raise ValueError("conditional scale horizon differs from V6")
    if not np.allclose(
        [
            scale["multiplier"],
            scale["baseFitFloorQuantile"],
            scale["baseFitCapQuantile"],
            scale["lowerStandardizedResidualQuantile"],
            scale["upperStandardizedResidualQuantile"],
        ],
        [1.0, 0.05, 0.95, 0.1, 0.9],
    ):
        raise ValueError("V7 frozen scale or residual quantiles changed")
    for key in (
        "floorCapFitOnBaseSegmentOnly",
        "dateEqualWeighted",
        "auditSegmentNeverFitsModelsScaleOrBounds",
    ):
        if scale.get(key) is not True:
            raise ValueError(f"V7 time-isolation flag changed: {key}")
    selection = config["selection"]
    if selection.get("rankingField") != "predicted_return_lower_10d":
        raise ValueError("V7 confidence ranking changed")
    if int(selection["maximumSelectionsPerDay"]) != 3:
        raise ValueError("V7 must select at most three stocks")
    if selection.get("allowZeroSelections") is not True or selection.get("neverFillToThree") is not True:
        raise ValueError("V7 must fail closed rather than fill a quota")
    if config["evaluation"].get("allThresholdsInheritedUnchangedFromV6") is not True:
        raise ValueError("V7 evaluation thresholds must remain inherited")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("a V7 historical run cannot promote")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("V7 orders must remain empty")


def fit_frozen_scale(
    table: pd.DataFrame,
    base_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> FrozenScale:
    spec = config["conditionalScale"]
    field = str(spec["sourceField"])
    if field not in table:
        raise KeyError(f"missing V7 conditional scale field: {field}")
    values = table.loc[table["date"].isin(base_dates), field].astype(float).abs()
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    raw = float(spec["multiplier"]) * np.sqrt(float(spec["holdingDays"])) * values
    raw = raw[raw > float(spec["minimumPositiveScale"])]
    if raw.empty:
        raise RuntimeError("V7 base-fit conditional scales are empty")
    floor = float(raw.quantile(float(spec["baseFitFloorQuantile"])))
    cap = float(raw.quantile(float(spec["baseFitCapQuantile"])))
    median = float(raw.median())
    if not (0.0 < floor <= median <= cap):
        raise RuntimeError("invalid V7 base-fit scale floor/median/cap")
    return FrozenScale(
        source_field=field,
        multiplier=float(spec["multiplier"]),
        holding_days=int(spec["holdingDays"]),
        floor=floor,
        cap=cap,
        median=median,
        audit={
            "sourceField": field,
            "fitDateRange": [base_dates[0].date().isoformat(), base_dates[-1].date().isoformat()],
            "fitTradingDays": len(base_dates),
            "fitRows": len(raw),
            "floorQuantile": float(spec["baseFitFloorQuantile"]),
            "capQuantile": float(spec["baseFitCapQuantile"]),
            "floor": round(floor, 8),
            "median": round(median, 8),
            "cap": round(cap, 8),
        },
    )


def score_frozen_scale(scale: FrozenScale, rows: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(rows[scale.source_field], errors="coerce").abs()
    raw = scale.multiplier * np.sqrt(float(scale.holding_days)) * values
    return raw.fillna(scale.median).clip(scale.floor, scale.cap).to_numpy(dtype=float)


def fit_conditional_return_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    base: dict[str, Any],
) -> ConditionalReturnModel:
    table = conformal_v6.prepare_return_decomposition(table)
    base_dates, calibration_dates, audit_dates, segment = conformal_v6.nested_segments(
        fit_dates, base
    )
    quantiles = list(map(float, base["modelFitting"]["targetWinsorQuantiles"]))
    market_model = conformal_v6.fit_component_model(
        table,
        base_dates,
        conformal_v6.market_feature_columns(base),
        "benchmark_return_10d",
        float(base["marketReturnModel"]["alpha"]),
        quantiles,
        True,
    )
    residual_model = conformal_v6.fit_component_model(
        table,
        base_dates,
        conformal_v6.residual_feature_columns(base),
        "target_residual_10d",
        float(base["residualAlphaModel"]["alpha"]),
        quantiles,
        False,
    )
    scale = fit_frozen_scale(table, base_dates, config)
    calibration = table[
        table["date"].isin(calibration_dates) & table["target_return_10d"].notna()
    ].copy()
    audit_rows = table[
        table["date"].isin(audit_dates) & table["target_return_10d"].notna()
    ].copy()
    if calibration.empty or audit_rows.empty:
        raise RuntimeError("V7 calibration or reliability audit is empty")
    _, _, calibration_point = conformal_v6.score_point_forecast(
        market_model, residual_model, calibration, base
    )
    calibration_scale = score_frozen_scale(scale, calibration)
    standardized = (
        calibration["target_return_10d"].to_numpy(dtype=float) - calibration_point
    ) / calibration_scale
    spec = config["conditionalScale"]
    lower_offset = conformal_v6.date_equal_weighted_quantile(
        standardized,
        calibration["date"],
        float(spec["lowerStandardizedResidualQuantile"]),
    )
    upper_offset = conformal_v6.date_equal_weighted_quantile(
        standardized,
        calibration["date"],
        float(spec["upperStandardizedResidualQuantile"]),
    )
    if lower_offset > upper_offset:
        raise RuntimeError("V7 standardized lower offset exceeds upper offset")
    _, _, audit_point = conformal_v6.score_point_forecast(
        market_model, residual_model, audit_rows, base
    )
    audit_scale = score_frozen_scale(scale, audit_rows)
    audit_lower = audit_point + lower_offset * audit_scale
    audit_upper = audit_point + upper_offset * audit_scale
    reliability = conformal_v6.reliability_metrics(
        audit_rows, audit_point, audit_lower, audit_upper, base
    )
    return ConditionalReturnModel(
        market_model=market_model,
        residual_model=residual_model,
        scale=scale,
        lower_standardized_offset=lower_offset,
        upper_standardized_offset=upper_offset,
        interval_enabled=bool(reliability["intervalEnabled"]),
        reliability=reliability,
        audit={
            **segment,
            "marketModel": market_model.audit,
            "residualModel": residual_model.audit,
            "conditionalScale": scale.audit,
            "calibrationRows": len(calibration),
            "calibrationDates": int(calibration["date"].nunique()),
            "lowerStandardizedResidualOffset": round(lower_offset, 8),
            "upperStandardizedResidualOffset": round(upper_offset, 8),
            "reliability": reliability,
        },
    )


def score_conditional_rows(
    model: ConditionalReturnModel,
    rows: pd.DataFrame,
    base: dict[str, Any],
) -> pd.DataFrame:
    market, residual, point = conformal_v6.score_point_forecast(
        model.market_model, model.residual_model, rows, base
    )
    scale = score_frozen_scale(model.scale, rows)
    output = rows[["date", "securityId"]].copy()
    output["predicted_market_return_10d"] = market
    output["predicted_residual_alpha_10d"] = residual
    output["predicted_total_return_10d"] = point
    output["predicted_return_scale_10d"] = scale
    output["predicted_return_lower_10d"] = point + model.lower_standardized_offset * scale
    output["predicted_return_upper_10d"] = point + model.upper_standardized_offset * scale
    output["predicted_interval_width_10d"] = (
        model.upper_standardized_offset - model.lower_standardized_offset
    ) * scale
    output["interval_model_enabled"] = model.interval_enabled
    output["calibration_lower_standardized_offset"] = model.lower_standardized_offset
    output["calibration_upper_standardized_offset"] = model.upper_standardized_offset
    output["scale_floor"] = model.scale.floor
    output["scale_cap"] = model.scale.cap
    output["audit_interval_coverage"] = model.reliability["intervalCoverage"]
    output["audit_lower_bound_violation_rate"] = model.reliability["lowerBoundViolationRate"]
    output["audit_rank_ic_hac_t"] = model.reliability["rankIcHacT"]
    output["audit_positive_lower_bound_n"] = model.reliability["positiveLowerBoundObservations"]
    output["audit_positive_lower_bound_hit_rate"] = model.reliability["positiveLowerBoundHitRateAfterCost"]
    return output


def rolling_conditional_predictions(
    table: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    base: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(calendar_dates)))
    walk = base["walkForward"]
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
        model = fit_conditional_return_model(table, fit_dates, config, base)
        scored = score_conditional_rows(model, block_rows, base)
        fold += 1
        scored["conditional_walk_forward_fold"] = fold
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


def period_report(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    eligible: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    base: dict[str, Any],
) -> dict[str, Any]:
    block = conformal_v6.period_report(
        predictions, dates, target, one_day, eligible, calendar_dates, base
    )
    block["policies"][PRIMARY_POLICY] = copy.deepcopy(
        block["policies"][conformal_v6.PRIMARY_POLICY]
    )
    labelled = predictions[
        predictions["date"].isin(dates) & predictions["target_return_10d"].notna()
    ].copy()
    policies = conformal_v6.policy_rows(labelled, base)
    point = utility_v4.basket_series(
        policies["point_forecast_top3_within_v2_top10"], target, dates
    )
    conditional = utility_v4.basket_series(
        policies["lower_bound_top3_within_v2_top10"], target, dates
    )
    block["conditionalLowerVsPointTop3SameDays"] = tail_v3.paired_comparison(
        conditional,
        point,
        int(base["evaluation"]["hacLagTradingDays"]),
        float(base["data"]["tailLossThreshold"]),
    )
    point_sets = {
        date: frozenset(group["securityId"])
        for date, group in policies["point_forecast_top3_within_v2_top10"].groupby("date")
    }
    lower_sets = {
        date: frozenset(group["securityId"])
        for date, group in policies["lower_bound_top3_within_v2_top10"].groupby("date")
    }
    common = sorted(set(point_sets).intersection(lower_sets))
    changed = sum(point_sets[date] != lower_sets[date] for date in common)
    block["conditionalRankingAudit"] = {
        "commonDays": len(common),
        "changedDays": int(changed),
        "changedDayFraction": round(changed / len(common), 8) if common else None,
        "primarySignalCoverage": round(
            block["policies"][PRIMARY_POLICY]["newSignalDays"] / max(1, len(dates)), 8
        ),
    }
    return block


def build_verdict(report: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    inherited = conformal_v6.build_verdict(report, base)
    inherited["primaryPolicy"] = PRIMARY_POLICY
    inherited["stableHistoricalConditionalTop3Evidence"] = inherited.pop(
        "stableHistoricalTop3ProfitabilityEvidence"
    )
    inherited["pointForecastChangedVsV6"] = False
    inherited["conditionalRankingChanged"] = {
        period: report["periods"][period]["conditionalRankingAudit"]["changedDays"] > 0
        for period in ("validation", "shadow")
    }
    return inherited


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
        "# Perception-XAlpha conditional Top3 V7",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- V6 point forecasts are unchanged; only past-volatility-scaled intervals differ.",
        "- Zero to three selections are allowed; profit is never guaranteed.",
        "- Orders: always empty.",
        "",
        "## Primary results",
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
            f"{metric['basketTenDayWinRate']} | {metric['tailLossRate']} | {interval['coverage']} | "
            f"{interval['lowerBoundViolationRate']} | {metric['costedCumulativeReturn']} |"
        )
    lines.extend(
        [
            "",
            "## Conditional ranking increment",
            "",
            "| Period | Common days | Changed days | Changed fraction | Mean lift vs point Top3 | HAC t |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("validation", "shadow"):
        audit = report["periods"][period]["conditionalRankingAudit"]
        paired = report["periods"][period]["conditionalLowerVsPointTop3SameDays"]
        lines.append(
            f"| {period} | {audit['commonDays']} | {audit['changedDays']} | "
            f"{audit['changedDayFraction']} | {paired['meanReturnDelta']} | {paired['returnDeltaTHac']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Decision: `{report['verdict']['decision']}`",
            f"- Stable historical conditional evidence: `{report['verdict']['stableHistoricalConditionalTop3Evidence']}`",
            "- Profit guaranteed: `false`.",
            "- Historical success cannot promote this policy.",
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
    base_path = ROOT / config["baseConformalConfig"]
    base = load_json(base_path)
    validate_config(config, base)
    v5_config = load_json(ROOT / base["baseReliabilityConfig"])
    reliability_v5.validate_config(v5_config)
    utility_config = load_json(ROOT / v5_config["baseExpectedUtilityConfig"])
    utility_v4.validate_config(utility_config)
    tail_config = load_json(ROOT / utility_config["baseTailConfig"])
    tail_v3.validate_config(tail_config)
    winrate_config = load_json(ROOT / tail_config["baseWinrateConfig"])
    winrate_v2.validate_config(winrate_config)
    selector_config = load_json(ROOT / winrate_config["baseSelectorConfig"])
    selector_v1.validate_config(selector_config)
    research_config = load_json(ROOT / selector_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(research_config)
    panel, panel_audit = perception.build_configured_panel(research_config, cog_config)
    factor_ranks = selector_v1.build_factor_rank_frames(panel, selector_config)
    features, market_features = selector_v1.build_past_only_feature_frames(panel, factor_ranks)
    target, one_day = autonomous.target_frames(panel, cog_config)
    raw_table = selector_v1.build_candidate_table(
        panel, features, market_features, target, int(base["data"]["candidatePoolSize"])
    )
    rank_table = winrate_v2.prepare_rank_table(raw_table, winrate_config)
    table = tail_v3.add_tail_label(rank_table, tail_config)
    table = conformal_v6.prepare_return_decomposition(table)
    rank_predictions, rank_audits = winrate_v2.rolling_walk_forward_predictions(
        table, panel["close"].index, winrate_config
    )
    tail_predictions, tail_audits = tail_v3.rolling_tail_predictions(
        table, panel["close"].index, tail_config
    )
    conditional_predictions, conditional_audits = rolling_conditional_predictions(
        table, panel["close"].index, config, base
    )
    if rank_predictions.empty or tail_predictions.empty or conditional_predictions.empty:
        raise RuntimeError("one or more V7 prediction tables are empty")
    predictions = rank_predictions.merge(
        tail_predictions, on=["date", "securityId"], how="inner", validate="one_to_one"
    ).merge(
        conditional_predictions, on=["date", "securityId"], how="inner", validate="one_to_one"
    )
    predictions = conformal_v6.add_context_and_selection(predictions, base)
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
        "predicted_return_scale_10d",
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
        "baseConformalConfigFileSha256": file_sha256(base_path),
        "dataRange": [panel["close"].index.min().date().isoformat(), panel["close"].index.max().date().isoformat()],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "predictionRows": len(predictions),
        "pointForecastDefinition": "identical_to_v6_market_plus_residual_ridge",
        "conditionalScaleDefinition": config["conditionalScale"],
        "rankWalkForwardAudits": rank_audits,
        "tailWalkForwardAudits": tail_audits,
        "conditionalWalkForwardAudits": conditional_audits,
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
            predictions, dates, target, one_day, panel["eligible"], panel["close"].index, base
        )
    report["verdict"] = build_verdict(report, base)
    out = ROOT / config["output"]["root"] / run_id
    atomic_write_text(out / "summary.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(out / "report.md", markdown_report(report))
    atomic_write_text(
        out / "latest_top10_conditional_intervals.csv",
        latest_top10[latest_columns].to_csv(index=False, lineterminator="\n"),
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
