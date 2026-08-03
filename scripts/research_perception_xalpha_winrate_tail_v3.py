"""Preregistered downside-tail extension for the all-A-share stock ranker.

The V2 rolling ridge model is kept unchanged.  This research-only module trains a
second, purged walk-forward classifier for a ten-session loss of at least three
percent, penalises the V2 cross-sectional score by that probability, and permits
cash when the selected basket is riskier than the classifier's training base rate.
It cannot create orders or alter any production decision path.
"""

from __future__ import annotations

import argparse
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_two_stage as selector_v1  # noqa: E402
import research_perception_xalpha_winrate_v2 as winrate_v2  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_winrate_tail_v3.json"
)
SCHEMA_VERSION = "perception_xalpha_winrate_tail_v3"
CODE_VERSION = "perception_xalpha_winrate_tail_v3.0"
PRIMARY_POLICY = "tail_adjusted_top10_adaptive_gate"


@dataclass
class TailModel:
    pipeline: Pipeline
    feature_columns: list[str]
    training_tail_base_rate: float
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected downside-tail research schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("downside-tail study must remain research-only")
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
    model = config["tailModel"]
    risk = config["riskAdjustment"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the outcome horizon must remain ten trading days")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("the frozen candidate pool must remain top fifty")
    if int(data["maximumSelectionsPerDay"]) != 10:
        raise ValueError("the preregistered book must remain top ten")
    if float(data["tailLossThreshold"]) != -0.03:
        raise ValueError("the tail-loss definition changed")
    if int(model["purgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("purge must cover the complete label horizon")
    expected_model = {
        "kind": "LogisticRegression",
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "classWeight": "balanced",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"the preregistered tail model changed: {key}")
    if float(risk["tailProbabilityPenalty"]) != 0.5:
        raise ValueError("the tail-probability penalty changed")
    if float(risk["adaptiveGateMultiplier"]) != 1.0:
        raise ValueError("the adaptive tail gate changed")
    if risk.get("allowCash") is not True or risk.get("neverForceTenSelections") is not True:
        raise ValueError("the primary policy must be allowed to abstain")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("a historical run cannot promote")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain permanently empty")


def tail_feature_columns(config: dict[str, Any]) -> list[str]:
    columns = list(config["tailModel"]["featureColumns"])
    forbidden = {
        "target_return_10d",
        "target_cross_sectional_rank_10d",
        "label_tail_loss_10d",
        "label_positive_10d",
    }
    overlap = forbidden.intersection(columns)
    if overlap:
        raise ValueError(f"future outcome entered tail features: {sorted(overlap)}")
    return columns


def add_tail_label(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = table.copy()
    valid = output["target_return_10d"].notna()
    output["label_tail_loss_10d"] = np.nan
    output.loc[valid, "label_tail_loss_10d"] = output.loc[
        valid, "target_return_10d"
    ].le(float(config["data"]["tailLossThreshold"])).astype(float)
    return output


def fit_tail_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> TailModel:
    columns = tail_feature_columns(config)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise KeyError(f"missing downside-tail features: {missing}")
    rows = table[
        table["date"].isin(fit_dates) & table["label_tail_loss_10d"].notna()
    ].copy()
    if rows.empty:
        raise RuntimeError("tail-model fit table is empty")
    labels = rows["label_tail_loss_10d"].astype(int)
    if labels.nunique() != 2:
        raise RuntimeError("tail-model fit requires both outcome classes")
    model = config["tailModel"]
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(model["C"]),
                    penalty=str(model["penalty"]),
                    solver=str(model["solver"]),
                    class_weight="balanced",
                    max_iter=int(model["maximumIterations"]),
                    random_state=int(model["randomSeed"]),
                ),
            ),
        ]
    )
    pipeline.fit(
        rows[columns].replace([np.inf, -np.inf], np.nan),
        labels,
    )
    base_rate = float(labels.mean())
    return TailModel(
        pipeline=pipeline,
        feature_columns=columns,
        training_tail_base_rate=base_rate,
        audit={
            "fitDateRange": [
                pd.Timestamp(fit_dates[0]).date().isoformat(),
                pd.Timestamp(fit_dates[-1]).date().isoformat(),
            ],
            "fitTradingDays": len(fit_dates),
            "fitRows": len(rows),
            "tailEvents": int(labels.sum()),
            "naturalTailBaseRate": round(base_rate, 8),
            "featureCount": len(columns),
            "target": "open_t_plus_11_over_open_t_plus_1_minus_one_le_minus_three_percent",
        },
    )


def score_tail_rows(model: TailModel, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows[["date", "securityId"]].copy()
    output["predicted_tail_probability"] = model.pipeline.predict_proba(
        rows[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    output["training_tail_base_rate"] = model.training_tail_base_rate
    return output


def rolling_tail_predictions(
    table: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Produce monthly blocks after a full label-horizon purge."""
    dates = pd.DatetimeIndex(sorted(pd.unique(calendar_dates)))
    model_cfg = config["tailModel"]
    minimum = int(model_cfg["minimumTrainingTradingDays"])
    window = int(model_cfg["trainingWindowTradingDays"])
    purge = int(model_cfg["purgeTradingDays"])
    refit = int(model_cfg["refitEveryTradingDays"])
    first_prediction = minimum + purge
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold = 0
    for start in range(first_prediction, len(dates), refit):
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
        model = fit_tail_model(table, fit_dates, config)
        predicted = score_tail_rows(model, block_rows)
        fold += 1
        predicted["tail_walk_forward_fold"] = fold
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


def add_tail_adjustment(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = rows.copy()
    output["rank_score_percentile"] = output.groupby("date", sort=False)[
        "predicted_cross_sectional_rank"
    ].rank(pct=True, method="average")
    penalty = float(config["riskAdjustment"]["tailProbabilityPenalty"])
    output["risk_adjusted_score"] = (
        output["rank_score_percentile"]
        - penalty * output["predicted_tail_probability"]
    )
    output = output.sort_values(
        ["date", "risk_adjusted_score", "securityId"],
        ascending=[True, False, True],
    )
    output["risk_adjusted_order"] = output.groupby("date", sort=False).cumcount() + 1
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    provisional = output[output["risk_adjusted_order"].le(top_n)]
    selected_mean = provisional.groupby("date", sort=False)[
        "predicted_tail_probability"
    ].mean()
    training_base = provisional.groupby("date", sort=False)[
        "training_tail_base_rate"
    ].mean()
    multiplier = float(config["riskAdjustment"]["adaptiveGateMultiplier"])
    output["selected_mean_tail_probability"] = output["date"].map(selected_mean)
    output["adaptive_tail_threshold"] = output["date"].map(training_base) * multiplier
    output["adaptive_tail_gate"] = output["selected_mean_tail_probability"].le(
        output["adaptive_tail_threshold"]
    )
    return output.sort_values(["date", "securityId"]).reset_index(drop=True)


def policy_rows(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    return {
        "frozen_factor_top10_all_days": rows[rows["candidate_rank"].le(top_n)].copy(),
        "rolling_ridge_rank_top10_all_days": rows[
            rows["predicted_order"].le(top_n)
        ].copy(),
        "tail_adjusted_top10_all_days": rows[
            rows["risk_adjusted_order"].le(top_n)
        ].copy(),
        PRIMARY_POLICY: rows[
            rows["risk_adjusted_order"].le(top_n) & rows["adaptive_tail_gate"]
        ].copy(),
    }


def basket_series(
    selected: pd.DataFrame,
    target: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    mask = selector_v1.selected_mask(selected, target.index, target.columns)
    return target.where(mask).mean(axis=1).reindex(dates).dropna()


def probability_metrics(rows: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    valid = rows[
        rows["label_tail_loss_10d"].notna()
        & rows["predicted_tail_probability"].notna()
    ].copy()
    if valid.empty:
        return {
            "n": 0,
            "tailEvents": 0,
            "naturalTailRate": None,
            "meanPredictedProbability": None,
            "brier": None,
            "logLoss": None,
            "auc": None,
            "ece": None,
        }
    labels = valid["label_tail_loss_10d"].astype(int)
    probabilities = valid["predicted_tail_probability"].clip(1e-8, 1.0 - 1e-8)
    auc = roc_auc_score(labels, probabilities) if labels.nunique() == 2 else None
    return {
        "n": len(valid),
        "tailEvents": int(labels.sum()),
        "naturalTailRate": round(float(labels.mean()), 8),
        "meanPredictedProbability": round(float(probabilities.mean()), 8),
        "brier": round(float(np.mean((probabilities - labels) ** 2)), 8),
        "logLoss": round(float(log_loss(labels, probabilities, labels=[0, 1])), 8),
        "auc": round(float(auc), 8) if auc is not None else None,
        "ece": selector_v1.ece_score(labels, probabilities, bins),
    }


def probability_buckets(rows: pd.DataFrame) -> list[dict[str, Any]]:
    valid = rows[
        rows["label_tail_loss_10d"].notna()
        & rows["predicted_tail_probability"].notna()
    ].copy()
    if valid.empty:
        return []
    boundaries = [-np.inf, 0.10, 0.20, 0.30, 0.40, 0.50, np.inf]
    labels = ["<10%", "10-20%", "20-30%", "30-40%", "40-50%", ">=50%"]
    valid["bucket"] = pd.cut(
        valid["predicted_tail_probability"], boundaries, labels=labels, right=False
    )
    output: list[dict[str, Any]] = []
    for label in labels:
        group = valid[valid["bucket"].eq(label)]
        output.append(
            {
                "bucket": label,
                "n": len(group),
                "meanPredictedProbability": round(
                    float(group["predicted_tail_probability"].mean()), 8
                )
                if len(group)
                else None,
                "realizedTailRate": round(
                    float(group["label_tail_loss_10d"].mean()), 8
                )
                if len(group)
                else None,
            }
        )
    return output


def tail_outcome_metrics(values: pd.Series, threshold: float) -> dict[str, Any]:
    sample = values.dropna().astype(float)
    if sample.empty:
        return {
            "tailLossRate": None,
            "averageLoserReturn": None,
            "tenPercentExpectedShortfall": None,
        }
    losers = sample[sample < 0.0]
    tail_count = max(1, int(math.ceil(len(sample) * 0.10)))
    return {
        "tailLossRate": round(float(sample.le(threshold).mean()), 8),
        "averageLoserReturn": round(float(losers.mean()), 8)
        if len(losers)
        else None,
        "tenPercentExpectedShortfall": round(
            float(sample.nsmallest(tail_count).mean()), 8
        ),
    }


def paired_comparison(
    primary: pd.Series,
    comparator: pd.Series,
    lag: int,
    tail_threshold: float,
) -> dict[str, Any]:
    joined = pd.concat(
        [primary.rename("primary"), comparator.rename("comparator")], axis=1
    ).dropna()
    if joined.empty:
        return {
            "commonSignalDays": 0,
            "meanReturnDelta": None,
            "winRateDelta": None,
            "tailLossRateReduction": None,
            "returnDeltaTHac": None,
        }
    delta = (joined["primary"] - joined["comparator"]).to_numpy(dtype=float)
    return_t = autonomous.newey_west_t(delta, lag)
    primary_win = joined["primary"].gt(0.0)
    comparator_win = joined["comparator"].gt(0.0)
    primary_tail = joined["primary"].le(tail_threshold)
    comparator_tail = joined["comparator"].le(tail_threshold)
    return {
        "commonSignalDays": len(joined),
        "primaryWinRate": round(float(primary_win.mean()), 8),
        "comparatorWinRate": round(float(comparator_win.mean()), 8),
        "meanReturnDelta": round(float(delta.mean()), 8),
        "winRateDelta": round(float((primary_win.astype(float) - comparator_win.astype(float)).mean()), 8),
        "primaryTailLossRate": round(float(primary_tail.mean()), 8),
        "comparatorTailLossRate": round(float(comparator_tail.mean()), 8),
        "tailLossRateReduction": round(
            float(comparator_tail.mean() - primary_tail.mean()), 8
        ),
        "returnDeltaTHac": round(float(return_t), 4) if return_t is not None else None,
        "hacLag": lag,
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
        basket = basket_series(selected, target, dates)
        metric.update(tail_outcome_metrics(basket, threshold))
        metrics[name] = metric
        baskets[name] = basket
    primary = baskets[PRIMARY_POLICY]
    lag = int(config["evaluation"]["hacLagTradingDays"])
    horizon = int(config["data"]["holdingTradingDays"])
    return {
        "candidateRows": len(labelled),
        "adaptiveGateDays": int(
            labelled.loc[labelled["adaptive_tail_gate"], "date"].nunique()
        ),
        "tailProbability": probability_metrics(labelled),
        "tailProbabilityBuckets": probability_buckets(labelled),
        "rankIc": winrate_v2.rank_ic_metrics(labelled, lag),
        "policies": metrics,
        "primaryIndependentEvents": winrate_v2.independent_event_metrics(
            primary, calendar_dates, horizon
        ),
        "primaryMonthly": winrate_v2.monthly_metrics(primary),
        "primaryVsFactorTop10SameDays": paired_comparison(
            primary, baskets["frozen_factor_top10_all_days"], lag, threshold
        ),
        "primaryVsRankTop10SameDays": paired_comparison(
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
        vs_factor = block["primaryVsFactorTop10SameDays"]
        vs_rank = block["primaryVsRankTop10SameDays"]
        independent = block["primaryIndependentEvents"]
        same_day_control = (
            vs_factor["commonSignalDays"] == primary["newSignalDays"]
            and vs_rank["commonSignalDays"] == primary["newSignalDays"]
        )
        period_checks = {
            "enoughSignalDays": primary["newSignalDays"]
            >= int(rules["minimumSignalDaysPerValidationPeriod"]),
            "enoughIndependentEvents": independent["n"]
            >= int(rules["minimumIndependentEventsPerValidationPeriod"]),
            "sameDayComparatorCoveragePass": same_day_control,
            "winRateLiftVsFactorTop10": vs_factor["winRateDelta"],
            "winRateLiftPass": vs_factor["winRateDelta"] is not None
            and vs_factor["winRateDelta"]
            >= float(rules["minimumWinRateLiftVsFactorTop10"]),
            "tailLossRateReductionVsRankTop10": vs_rank[
                "tailLossRateReduction"
            ],
            "tailLossReductionPass": vs_rank["tailLossRateReduction"] is not None
            and vs_rank["tailLossRateReduction"]
            >= float(rules["minimumTailLossRateReductionVsRankTop10"]),
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
            "winRateLiftPass",
            "tailLossReductionPass",
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
        "stableHistoricalTailImprovement": bool(stable),
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
        "# Perception-XAlpha downside-tail V3",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- Frozen V2 ridge rank plus a separate purged logistic tail-loss head.",
        "- Primary score: within-date V2 rank percentile minus 0.5 times predicted 10-day loss probability.",
        "- Cash is mandatory when selected Top10 mean tail probability exceeds its training base rate.",
        "- Orders: always empty.",
        "",
        "## Policy comparison",
        "",
        "| Period | Policy | Signal days | 10d mean | 10d win | Tail loss | Avg loser | ES10% | Costed cumulative |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        block = report["periods"].get(period)
        if not block:
            continue
        for name, metric in block["policies"].items():
            lines.append(
                f"| {period} | {name} | {metric['newSignalDays']} | "
                f"{metric['basketTenDayMeanReturn']} | {metric['basketTenDayWinRate']} | "
                f"{metric['tailLossRate']} | {metric['averageLoserReturn']} | "
                f"{metric['tenPercentExpectedShortfall']} | {metric['costedCumulativeReturn']} |"
            )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Period | N | Natural tail | Mean predicted | Brier | LogLoss | AUC | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("trainWalkForward", "validation", "shadow"):
        metric = report["periods"][period]["tailProbability"]
        lines.append(
            f"| {period} | {metric['n']} | {metric['naturalTailRate']} | "
            f"{metric['meanPredictedProbability']} | {metric['brier']} | "
            f"{metric['logLoss']} | {metric['auc']} | {metric['ece']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Decision: `{report['verdict']['decision']}`",
            f"- Stable validation and shadow tail improvement: `{report['verdict']['stableHistoricalTailImprovement']}`",
            "- Fewer trading days cannot pass by itself: all lifts are measured against comparators on the exact same selected dates, with minimum coverage and independent-event gates.",
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
    winrate_config_path = ROOT / config["baseWinrateConfig"]
    winrate_config = load_json(winrate_config_path)
    winrate_v2.validate_config(winrate_config)
    for key in (
        "trainingWindowTradingDays",
        "minimumTrainingTradingDays",
        "refitEveryTradingDays",
        "purgeTradingDays",
    ):
        if int(config["tailModel"][key]) != int(winrate_config["rankModel"][key]):
            raise ValueError(f"rank and tail walk-forward calendars differ: {key}")
    selector_config_path = ROOT / winrate_config["baseSelectorConfig"]
    selector_config = load_json(selector_config_path)
    selector_v1.validate_config(selector_config)
    research_config = load_json(ROOT / selector_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(research_config)
    if int(cog_config["data"]["predictionHorizonTradingDays"]) != int(
        config["data"]["holdingTradingDays"]
    ):
        raise ValueError("base label horizon and downside-tail horizon differ")
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
    table = add_tail_label(rank_table, config)
    rank_predictions, rank_audits = winrate_v2.rolling_walk_forward_predictions(
        table, panel["close"].index, winrate_config
    )
    tail_predictions, tail_audits = rolling_tail_predictions(
        table, panel["close"].index, config
    )
    if rank_predictions.empty or tail_predictions.empty:
        raise RuntimeError("walk-forward prediction table is empty")
    predictions = rank_predictions.merge(
        tail_predictions,
        on=["date", "securityId"],
        how="inner",
        validate="one_to_one",
    )
    predictions = add_tail_adjustment(predictions, config)
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
        "baseWinrateConfigSha256": selector_v1.digest(winrate_config),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "predictionRows": len(predictions),
        "labelDefinition": "open[t+11]/open[t+1]-1 <= -3%; label is offline-only",
        "featureDefinition": tail_feature_columns(config),
        "rankWalkForwardAudits": rank_audits,
        "tailWalkForwardAudits": tail_audits,
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
        latest["risk_adjusted_order"].le(
            int(config["data"]["maximumSelectionsPerDay"])
        )
        & latest["adaptive_tail_gate"]
    )
    latest_columns = [
        "date",
        "securityId",
        "candidate_rank",
        "factor_composite",
        "predicted_cross_sectional_rank",
        "predicted_order",
        "predicted_tail_probability",
        "training_tail_base_rate",
        "rank_score_percentile",
        "risk_adjusted_score",
        "risk_adjusted_order",
        "selected_mean_tail_probability",
        "adaptive_tail_threshold",
        "adaptive_tail_gate",
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
                "latestGate": bool(latest["adaptive_tail_gate"].iloc[0]),
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
