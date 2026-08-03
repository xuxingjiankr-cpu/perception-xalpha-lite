"""Research-only three-head stock selector with an absolute return forecast.

V4 preserves the frozen V2 relative-rank and V3 downside-tail heads, adds a
purged rolling Ridge estimate of the absolute ten-session return, and combines
the three cross-sectional percentiles with equal preregistered weights.  It is
offline diagnostics only and has no route to orders or production decisions.
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
import research_perception_xalpha_two_stage as selector_v1  # noqa: E402
import research_perception_xalpha_winrate_tail_v3 as tail_v3  # noqa: E402
import research_perception_xalpha_winrate_v2 as winrate_v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_expected_utility_v4.json"
)
SCHEMA_VERSION = "perception_xalpha_expected_utility_v4"
CODE_VERSION = "perception_xalpha_expected_utility_v4.0"
PRIMARY_POLICY = "integrated_top10_positive_net_edge_gate"


@dataclass
class ExpectedReturnModel:
    pipeline: Pipeline
    feature_columns: list[str]
    training_prior: float
    target_clip: tuple[float, float]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected expected-utility research schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("expected-utility study must remain research-only")
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
    data = config["data"]
    model = config["expectedReturnModel"]
    integration = config["integration"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the expected-return horizon must remain ten days")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("the frozen candidate pool must remain top fifty")
    if int(data["maximumSelectionsPerDay"]) != 10:
        raise ValueError("the integrated book must remain top ten")
    if int(model["purgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("purge must cover the complete label horizon")
    if model.get("kind") != "Ridge" or float(model.get("alpha")) != 10.0:
        raise ValueError("the preregistered expected-return model changed")
    if list(map(float, model["targetWinsorQuantiles"])) != [0.01, 0.99]:
        raise ValueError("the training-only target winsorization changed")
    weights = [
        float(integration["rankWeight"]),
        float(integration["expectedReturnWeight"]),
        float(integration["inverseTailRiskWeight"]),
    ]
    if not np.allclose(weights, [1.0 / 3.0] * 3) or not np.isclose(sum(weights), 1.0):
        raise ValueError("the equal-weight three-head formula changed")
    if integration.get(
        "tailRiskUsesWithinDatePercentileNotUncalibratedAbsoluteProbability"
    ) is not True:
        raise ValueError("absolute uncalibrated tail probability cannot enter score")
    if not np.isclose(
        float(integration["minimumSelectedMeanExpectedReturn"]),
        float(data["roundTripCost"]),
    ):
        raise ValueError("the positive-net-edge gate changed")
    if integration.get("allowCash") is not True:
        raise ValueError("the integrated policy must be allowed to abstain")
    if integration.get("neverForceTenSelections") is not True:
        raise ValueError("the integrated policy cannot force ten stocks")
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
        raise ValueError(f"future outcome entered return features: {sorted(overlap)}")
    return columns


def fit_expected_return_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> ExpectedReturnModel:
    columns = expected_return_feature_columns(config)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise KeyError(f"missing expected-return features: {missing}")
    rows = table[
        table["date"].isin(fit_dates) & table["target_return_10d"].notna()
    ].copy()
    if rows.empty:
        raise RuntimeError("expected-return fit table is empty")
    model = config["expectedReturnModel"]
    target = rows["target_return_10d"].astype(float)
    quantiles = list(map(float, model["targetWinsorQuantiles"]))
    lower, upper = map(float, target.quantile(quantiles).tolist())
    clipped = target.clip(lower, upper)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(model["alpha"]))),
        ]
    )
    pipeline.fit(
        rows[columns].replace([np.inf, -np.inf], np.nan),
        clipped,
    )
    prior = float(clipped.mean())
    return ExpectedReturnModel(
        pipeline=pipeline,
        feature_columns=columns,
        training_prior=prior,
        target_clip=(lower, upper),
        audit={
            "fitDateRange": [
                pd.Timestamp(fit_dates[0]).date().isoformat(),
                pd.Timestamp(fit_dates[-1]).date().isoformat(),
            ],
            "fitTradingDays": len(fit_dates),
            "fitRows": len(rows),
            "featureCount": len(columns),
            "target": "open_t_plus_11_over_open_t_plus_1_minus_one",
            "targetWinsorQuantiles": quantiles,
            "targetClip": [round(lower, 8), round(upper, 8)],
            "trainingExpectedReturnPrior": round(prior, 8),
        },
    )


def score_expected_return_rows(
    model: ExpectedReturnModel,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    output = rows[["date", "securityId"]].copy()
    output["predicted_expected_return_10d"] = model.pipeline.predict(
        rows[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    )
    output["training_expected_return_prior"] = model.training_prior
    output["training_target_clip_lower"] = model.target_clip[0]
    output["training_target_clip_upper"] = model.target_clip[1]
    return output


def rolling_expected_return_predictions(
    table: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Refit monthly after a full ten-session label purge."""
    dates = pd.DatetimeIndex(sorted(pd.unique(calendar_dates)))
    model_cfg = config["expectedReturnModel"]
    minimum = int(model_cfg["minimumTrainingTradingDays"])
    window = int(model_cfg["trainingWindowTradingDays"])
    purge = int(model_cfg["purgeTradingDays"])
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
        model = fit_expected_return_model(table, fit_dates, config)
        predicted = score_expected_return_rows(model, block_rows)
        fold += 1
        predicted["expected_return_walk_forward_fold"] = fold
        predictions.append(predicted)
        audits.append(
            {
                "fold": fold,
                "predictionDateRange": [
                    block[0].date().isoformat(),
                    block[-1].date().isoformat(),
                ],
                "predictionRows": len(predicted),
                "purgeTradingDays": purge,
                "gapTradingDays": int(start - (fit_stop - 1) - 1),
                "model": model.audit,
            }
        )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        audits,
    )


def add_integrated_score(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = rows.copy()
    output["expected_return_percentile"] = output.groupby("date", sort=False)[
        "predicted_expected_return_10d"
    ].rank(pct=True, method="average")
    output["inverse_tail_probability_percentile"] = output.groupby(
        "date", sort=False
    )["predicted_tail_probability"].rank(
        pct=True, method="average", ascending=False
    )
    weights = config["integration"]
    output["integrated_expected_utility_score"] = (
        float(weights["rankWeight"]) * output["rank_score_percentile"]
        + float(weights["expectedReturnWeight"])
        * output["expected_return_percentile"]
        + float(weights["inverseTailRiskWeight"])
        * output["inverse_tail_probability_percentile"]
    )
    output = output.sort_values(
        ["date", "integrated_expected_utility_score", "securityId"],
        ascending=[True, False, True],
    )
    output["integrated_order"] = output.groupby("date", sort=False).cumcount() + 1
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    provisional = output[output["integrated_order"].le(top_n)]
    selected_mean = provisional.groupby("date", sort=False)[
        "predicted_expected_return_10d"
    ].mean()
    threshold = float(config["integration"]["minimumSelectedMeanExpectedReturn"])
    output["selected_mean_expected_return_10d"] = output["date"].map(selected_mean)
    output["minimum_expected_return_gate"] = threshold
    output["positive_net_edge_gate"] = output[
        "selected_mean_expected_return_10d"
    ].gt(threshold)
    output["predicted_net_return_after_round_trip_cost"] = (
        output["predicted_expected_return_10d"]
        - float(config["data"]["roundTripCost"])
    )
    return output.sort_values(["date", "securityId"]).reset_index(drop=True)


def policy_rows(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    expected_order = rows.sort_values(
        ["date", "predicted_expected_return_10d", "securityId"],
        ascending=[True, False, True],
    ).copy()
    expected_order["expected_return_order"] = expected_order.groupby(
        "date", sort=False
    ).cumcount() + 1
    expected_ids = expected_order[
        expected_order["expected_return_order"].le(top_n)
    ][["date", "securityId"]]
    expected_key = pd.MultiIndex.from_frame(expected_ids)
    row_key = pd.MultiIndex.from_frame(rows[["date", "securityId"]])
    expected_selected = rows[row_key.isin(expected_key)].copy()
    return {
        "frozen_factor_top10_all_days": rows[rows["candidate_rank"].le(top_n)].copy(),
        "rolling_ridge_rank_top10_all_days": rows[
            rows["predicted_order"].le(top_n)
        ].copy(),
        "tail_adjusted_top10_all_days": rows[
            rows["risk_adjusted_order"].le(top_n)
        ].copy(),
        "expected_return_top10_all_days": expected_selected,
        "integrated_top10_all_days": rows[
            rows["integrated_order"].le(top_n)
        ].copy(),
        PRIMARY_POLICY: rows[
            rows["integrated_order"].le(top_n) & rows["positive_net_edge_gate"]
        ].copy(),
    }


def basket_series(
    selected: pd.DataFrame,
    target: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    mask = selector_v1.selected_mask(selected, target.index, target.columns)
    return target.where(mask).mean(axis=1).reindex(dates).dropna()


def expected_return_metrics(rows: pd.DataFrame, lag: int) -> dict[str, Any]:
    valid = rows[
        rows["target_return_10d"].notna()
        & rows["predicted_expected_return_10d"].notna()
        & rows["training_expected_return_prior"].notna()
    ].copy()
    if valid.empty:
        return {
            "n": 0,
            "predictedMean": None,
            "actualMean": None,
            "bias": None,
            "mae": None,
            "rmse": None,
            "oosR2VsTrainingPrior": None,
            "calibrationIntercept": None,
            "calibrationSlope": None,
            "rankIc": {"days": 0, "mean": None, "tHac": None, "hacLag": lag},
        }
    predicted = valid["predicted_expected_return_10d"].to_numpy(dtype=float)
    actual = valid["target_return_10d"].to_numpy(dtype=float)
    prior = valid["training_expected_return_prior"].to_numpy(dtype=float)
    residual = predicted - actual
    denominator = float(np.sum((actual - prior) ** 2))
    oos_r2 = 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator if denominator > 0 else None
    design = np.column_stack([np.ones(len(predicted)), predicted])
    coefficients = np.linalg.lstsq(design, actual, rcond=None)[0]
    daily_ic: list[float] = []
    for _, group in valid.groupby("date", sort=True):
        if len(group) < 3:
            continue
        value = group["predicted_expected_return_10d"].corr(
            group["target_return_10d"]
        )
        if pd.notna(value):
            daily_ic.append(float(value))
    ic_values = np.asarray(daily_ic, dtype=float)
    ic_t = autonomous.newey_west_t(ic_values, lag) if len(ic_values) >= 3 else None
    return {
        "n": len(valid),
        "predictedMean": round(float(predicted.mean()), 8),
        "actualMean": round(float(actual.mean()), 8),
        "bias": round(float(residual.mean()), 8),
        "mae": round(float(np.mean(np.abs(residual))), 8),
        "rmse": round(float(np.sqrt(np.mean(residual**2))), 8),
        "oosR2VsTrainingPrior": round(float(oos_r2), 8)
        if oos_r2 is not None
        else None,
        "calibrationIntercept": round(float(coefficients[0]), 8),
        "calibrationSlope": round(float(coefficients[1]), 8),
        "rankIc": {
            "days": len(ic_values),
            "mean": round(float(ic_values.mean()), 8) if len(ic_values) else None,
            "positiveDayFraction": round(float((ic_values > 0.0).mean()), 8)
            if len(ic_values)
            else None,
            "tHac": round(float(ic_t), 4) if ic_t is not None else None,
            "hacLag": lag,
        },
    }


def expected_return_buckets(rows: pd.DataFrame) -> list[dict[str, Any]]:
    valid = rows[
        rows["target_return_10d"].notna()
        & rows["expected_return_percentile"].notna()
    ].copy()
    if valid.empty:
        return []
    bucket = np.minimum(
        np.floor(valid["expected_return_percentile"].clip(0.0, 0.999999) * 5.0),
        4,
    ).astype(int)
    valid["bucket"] = bucket
    output: list[dict[str, Any]] = []
    for number in range(5):
        group = valid[valid["bucket"].eq(number)]
        output.append(
            {
                "bucket": number + 1,
                "n": len(group),
                "predictedMean": round(
                    float(group["predicted_expected_return_10d"].mean()), 8
                )
                if len(group)
                else None,
                "actualMean": round(
                    float(group["target_return_10d"].mean()), 8
                )
                if len(group)
                else None,
                "winRate": round(float(group["target_return_10d"].gt(0.0).mean()), 8)
                if len(group)
                else None,
            }
        )
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
        metric = selector_v1.portfolio_metrics(
            selected, target, one_day, eligible, dates, config
        )
        basket = basket_series(selected, target, dates)
        metric.update(tail_v3.tail_outcome_metrics(basket, threshold))
        metrics[name] = metric
        baskets[name] = basket
    primary = baskets[PRIMARY_POLICY]
    lag = int(config["evaluation"]["hacLagTradingDays"])
    horizon = int(config["data"]["holdingTradingDays"])
    return {
        "candidateRows": len(labelled),
        "positiveNetEdgeGateDays": int(
            labelled.loc[labelled["positive_net_edge_gate"], "date"].nunique()
        ),
        "expectedReturnForecast": expected_return_metrics(labelled, lag),
        "expectedReturnBuckets": expected_return_buckets(labelled),
        "policies": metrics,
        "primaryIndependentEvents": winrate_v2.independent_event_metrics(
            primary, calendar_dates, horizon
        ),
        "primaryMonthly": winrate_v2.monthly_metrics(primary),
        "primaryVsFactorTop10SameDays": tail_v3.paired_comparison(
            primary, baskets["frozen_factor_top10_all_days"], lag, threshold
        ),
        "primaryVsRankTop10SameDays": tail_v3.paired_comparison(
            primary, baskets["rolling_ridge_rank_top10_all_days"], lag, threshold
        ),
    }


def build_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["evaluation"]
    checks: dict[str, Any] = {}
    stable = True
    for period in ("validation", "shadow"):
        block = report["periods"][period]
        primary = block["policies"][PRIMARY_POLICY]
        vs_rank = block["primaryVsRankTop10SameDays"]
        forecast = block["expectedReturnForecast"]
        independent = block["primaryIndependentEvents"]
        same_day_control = vs_rank["commonSignalDays"] == primary["newSignalDays"]
        rank_ic_t = forecast["rankIc"]["tHac"]
        period_checks = {
            "enoughSignalDays": primary["newSignalDays"]
            >= int(rules["minimumSignalDaysPerValidationPeriod"]),
            "enoughIndependentEvents": independent["n"]
            >= int(rules["minimumIndependentEventsPerValidationPeriod"]),
            "sameDayComparatorCoveragePass": same_day_control,
            "meanReturnLiftVsRankTop10": vs_rank["meanReturnDelta"],
            "meanReturnLiftPass": vs_rank["meanReturnDelta"] is not None
            and vs_rank["meanReturnDelta"]
            > float(rules["minimumSameDayMeanReturnLiftVsRankTop10"]),
            "winRateLiftVsRankTop10": vs_rank["winRateDelta"],
            "winRateNoWorsePass": vs_rank["winRateDelta"] is not None
            and vs_rank["winRateDelta"]
            >= float(rules["minimumSameDayWinRateLiftVsRankTop10"]),
            "tailLossRateReductionVsRankTop10": vs_rank[
                "tailLossRateReduction"
            ],
            "tailLossNoWorsePass": vs_rank["tailLossRateReduction"] is not None
            and vs_rank["tailLossRateReduction"]
            >= float(rules["minimumTailLossRateReductionVsRankTop10"]),
            "positiveExpectedReturnOosR2Pass": forecast["oosR2VsTrainingPrior"]
            is not None
            and forecast["oosR2VsTrainingPrior"] > 0.0,
            "expectedReturnRankIcHacPass": rank_ic_t is not None
            and rank_ic_t >= float(rules["minimumHacTForExpectedReturnRankIc"]),
            "positiveMeanReturn": primary["basketTenDayMeanReturn"] is not None
            and primary["basketTenDayMeanReturn"] > 0.0,
            "positiveCostedCumulativeReturn": primary["costedCumulativeReturn"]
            > 0.0,
            "sameDayReturnHacPass": vs_rank["returnDeltaTHac"] is not None
            and vs_rank["returnDeltaTHac"]
            >= float(rules["minimumHacTForSameDayReturnDifference"]),
        }
        required = {
            "enoughSignalDays",
            "enoughIndependentEvents",
            "sameDayComparatorCoveragePass",
            "meanReturnLiftPass",
            "winRateNoWorsePass",
            "tailLossNoWorsePass",
            "positiveExpectedReturnOosR2Pass",
            "expectedReturnRankIcHacPass",
            "positiveMeanReturn",
            "positiveCostedCumulativeReturn",
            "sameDayReturnHacPass",
        }
        period_checks["periodPass"] = all(
            bool(period_checks[name]) for name in required
        )
        checks[period] = period_checks
        stable = stable and period_checks["periodPass"]
    return {
        "status": "research_only_not_eligible_for_trading",
        "primaryPolicy": PRIMARY_POLICY,
        "stableHistoricalExpectedUtilityImprovement": bool(stable),
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
        "# Perception-XAlpha expected-utility V4",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- Heads: frozen V2 relative rank + rolling absolute expected return + frozen V3 downside risk.",
        "- Integrated score: equal thirds of rank percentile, expected-return percentile and inverse-tail percentile.",
        "- Primary requires selected Top10 mean expected return above the frozen 30 bps round-trip cost.",
        "- Orders: always empty.",
        "",
        "## Forecast diagnostics",
        "",
        "| Period | N | Predicted mean | Actual mean | Bias | MAE | RMSE | OOS R2 | Rank IC | IC t(HAC) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        metric = report["periods"][period]["expectedReturnForecast"]
        lines.append(
            f"| {period} | {metric['n']} | {metric['predictedMean']} | "
            f"{metric['actualMean']} | {metric['bias']} | {metric['mae']} | "
            f"{metric['rmse']} | {metric['oosR2VsTrainingPrior']} | "
            f"{metric['rankIc']['mean']} | {metric['rankIc']['tHac']} |"
        )
    lines.extend(
        [
            "",
            "## Policy comparison",
            "",
            "| Period | Policy | Days | 10d mean | Win | Tail loss | Costed cumulative | Max drawdown |",
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
            f"- Stable validation and shadow improvement: `{report['verdict']['stableHistoricalExpectedUtilityImprovement']}`",
            "- The forecast must beat its rolling training prior and the integrated book must beat V2 on exactly the same dates; abstention alone cannot pass.",
            "- Historical output cannot promote or connect to trading.",
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
    tail_config = load_json(ROOT / config["baseTailConfig"])
    tail_v3.validate_config(tail_config)
    winrate_config = load_json(ROOT / tail_config["baseWinrateConfig"])
    winrate_v2.validate_config(winrate_config)
    for key in (
        "trainingWindowTradingDays",
        "minimumTrainingTradingDays",
        "refitEveryTradingDays",
        "purgeTradingDays",
    ):
        expected = int(config["expectedReturnModel"][key])
        if expected != int(tail_config["tailModel"][key]):
            raise ValueError(f"return and tail walk-forward calendars differ: {key}")
        if expected != int(winrate_config["rankModel"][key]):
            raise ValueError(f"return and rank walk-forward calendars differ: {key}")
    selector_config = load_json(ROOT / winrate_config["baseSelectorConfig"])
    selector_v1.validate_config(selector_config)
    research_config = load_json(ROOT / selector_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(research_config)
    if int(cog_config["data"]["predictionHorizonTradingDays"]) != int(
        config["data"]["holdingTradingDays"]
    ):
        raise ValueError("base label horizon and V4 return horizon differ")
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
    return_predictions, return_audits = rolling_expected_return_predictions(
        table, panel["close"].index, config
    )
    if rank_predictions.empty or tail_predictions.empty or return_predictions.empty:
        raise RuntimeError("one or more V4 walk-forward prediction tables are empty")
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
    predictions = add_integrated_score(predictions, config)
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
        "baseTailConfigSha256": selector_v1.digest(tail_config),
        "baseWinrateConfigSha256": selector_v1.digest(winrate_config),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "predictionRows": len(predictions),
        "expectedReturnLabelDefinition": "open[t+11]/open[t+1]-1; offline label only",
        "expectedReturnFeatureDefinition": expected_return_feature_columns(config),
        "rankWalkForwardAudits": rank_audits,
        "tailWalkForwardAudits": tail_audits,
        "expectedReturnWalkForwardAudits": return_audits,
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
        latest["integrated_order"].le(int(config["data"]["maximumSelectionsPerDay"]))
        & latest["positive_net_edge_gate"]
    )
    latest_columns = [
        "date",
        "securityId",
        "candidate_rank",
        "factor_composite",
        "predicted_cross_sectional_rank",
        "rank_score_percentile",
        "predicted_expected_return_10d",
        "predicted_net_return_after_round_trip_cost",
        "training_expected_return_prior",
        "expected_return_percentile",
        "predicted_tail_probability",
        "inverse_tail_probability_percentile",
        "integrated_expected_utility_score",
        "integrated_order",
        "selected_mean_expected_return_10d",
        "minimum_expected_return_gate",
        "positive_net_edge_gate",
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
                "latestGate": bool(latest["positive_net_edge_gate"].iloc[0]),
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
