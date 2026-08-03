"""Research-only win-rate study for the all-A-share Perception-XAlpha stack.

The failed V1 selector tried to estimate absolute ten-session returns with one model.
This preregistered V2 separates two jobs: a rolling ridge model ranks the frozen top-50
cross-section, and a deterministic, past-only market-state rule decides whether a
long-only book may be opened.  No artifact from this module is a trading instruction.
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


DEFAULT_CONFIG = ROOT / "configs" / "research" / "perception_xalpha_winrate_v2.json"
SCHEMA_VERSION = "perception_xalpha_winrate_v2"
CODE_VERSION = "perception_xalpha_winrate_v2.0"
PRIMARY_POLICY = "rolling_ridge_rank_top5_risk_on"


@dataclass
class RankModel:
    pipeline: Pipeline
    feature_columns: list[str]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected win-rate research schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("win-rate study must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading mutation permissions must remain false")
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("primaryPolicy") != PRIMARY_POLICY:
        raise ValueError("the preregistered primary policy changed")
    if hypothesis.get("parametersFrozenBeforeHistoricalRun") is not True:
        raise ValueError("parameters must be frozen before the historical run")
    data = config["data"]
    model = config["rankModel"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the target horizon must remain ten trading days")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("the factor candidate pool must remain fifty")
    if int(data["maximumSelectionsPerDay"]) != 5:
        raise ValueError("the precision-first book must remain top five")
    if int(model["purgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("purge must cover the complete label horizon")
    if str(model["kind"]) != "Ridge" or float(model["alpha"]) != 10.0:
        raise ValueError("the preregistered rank model changed")
    gate = config["marketGate"]
    if float(gate["marketReturn20MinimumExclusive"]) != 0.0:
        raise ValueError("market-return gate changed")
    if float(gate["marketBreadth20MinimumInclusive"]) != 0.5:
        raise ValueError("market-breadth gate changed")
    if gate.get("usesOnlySignalDayAndEarlierData") is not True:
        raise ValueError("market gate must remain past-only")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("a historical run cannot promote")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain permanently empty")


def transformed_feature_columns(config: dict[str, Any]) -> list[str]:
    return [f"x_{name}" for name in config["rankModel"]["featureColumns"]]


def prepare_rank_table(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build stationary cross-sectional inputs while keeping labels separate."""
    output = table.copy()
    feature_names = list(config["rankModel"]["featureColumns"])
    missing = [name for name in feature_names if name not in output.columns]
    if missing:
        raise KeyError(f"missing rank-model features: {missing}")
    for name in feature_names:
        target_name = f"x_{name}"
        if name.startswith("stock_"):
            output[target_name] = output.groupby("date", sort=False)[name].rank(
                pct=True, method="average"
            )
        else:
            output[target_name] = output[name]
    output["target_cross_sectional_rank_10d"] = output.groupby(
        "date", sort=False
    )["target_return_10d"].rank(pct=True, method="average")
    return output


def fit_rank_model(
    table: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> RankModel:
    columns = transformed_feature_columns(config)
    rows = table[
        table["date"].isin(fit_dates)
        & table["target_cross_sectional_rank_10d"].notna()
    ].copy()
    if rows.empty:
        raise RuntimeError("rank-model fit table is empty")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(config["rankModel"]["alpha"]))),
        ]
    )
    pipeline.fit(
        rows[columns].replace([np.inf, -np.inf], np.nan),
        rows["target_cross_sectional_rank_10d"].astype(float),
    )
    return RankModel(
        pipeline=pipeline,
        feature_columns=columns,
        audit={
            "fitDateRange": [
                pd.Timestamp(fit_dates[0]).date().isoformat(),
                pd.Timestamp(fit_dates[-1]).date().isoformat(),
            ],
            "fitTradingDays": len(fit_dates),
            "fitRows": len(rows),
            "featureCount": len(columns),
            "target": "within_date_top50_ten_day_return_percentile",
        },
    )


def score_rows(model: RankModel, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["predicted_cross_sectional_rank"] = model.pipeline.predict(
        output[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    )
    output = output.sort_values(
        ["date", "predicted_cross_sectional_rank", "securityId"],
        ascending=[True, False, True],
    )
    output["predicted_order"] = output.groupby("date", sort=False).cumcount() + 1
    return output.sort_values(["date", "securityId"]).reset_index(drop=True)


def rolling_walk_forward_predictions(
    table: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Refit monthly using only labels mature before a full ten-day purge."""
    dates = pd.DatetimeIndex(sorted(pd.unique(calendar_dates)))
    model_cfg = config["rankModel"]
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
        model = fit_rank_model(table, fit_dates, config)
        block_rows = table[table["date"].isin(block)].copy()
        if block_rows.empty:
            continue
        predicted = score_rows(model, block_rows)
        fold += 1
        predicted["walk_forward_fold"] = fold
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


def add_market_gate(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = rows.copy()
    gate = config["marketGate"]
    output["risk_on"] = output["market_return_20"].gt(
        float(gate["marketReturn20MinimumExclusive"])
    ) & output["market_breadth_20"].ge(
        float(gate["marketBreadth20MinimumInclusive"])
    )
    return output


def policy_rows(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_n = int(config["data"]["maximumSelectionsPerDay"])
    return {
        "frozen_factor_top10_all_days": rows[rows["candidate_rank"].le(10)].copy(),
        "frozen_factor_top5_all_days": rows[rows["candidate_rank"].le(top_n)].copy(),
        "frozen_factor_top5_risk_on": rows[
            rows["candidate_rank"].le(top_n) & rows["risk_on"]
        ].copy(),
        "rolling_ridge_rank_top5_all_days": rows[
            rows["predicted_order"].le(top_n)
        ].copy(),
        PRIMARY_POLICY: rows[
            rows["predicted_order"].le(top_n) & rows["risk_on"]
        ].copy(),
    }


def basket_series(
    selected: pd.DataFrame,
    target: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    mask = selector_v1.selected_mask(selected, target.index, target.columns)
    return target.where(mask).mean(axis=1).reindex(dates).dropna()


def independent_event_metrics(
    values: pd.Series,
    calendar_dates: pd.DatetimeIndex,
    horizon: int,
) -> dict[str, Any]:
    positions = pd.Series(np.arange(len(calendar_dates)), index=calendar_dates)
    chosen: list[pd.Timestamp] = []
    last_position: int | None = None
    for date in values.index.sort_values():
        if date not in positions.index:
            continue
        position = int(positions.loc[date])
        if last_position is None or position - last_position >= horizon:
            chosen.append(pd.Timestamp(date))
            last_position = position
    sample = values.reindex(chosen).dropna()
    return {
        "n": len(sample),
        "meanReturn": round(float(sample.mean()), 8) if len(sample) else None,
        "winRate": round(float(sample.gt(0.0).mean()), 8) if len(sample) else None,
    }


def rank_ic_metrics(rows: pd.DataFrame, lag: int) -> dict[str, Any]:
    valid = rows[
        rows["target_cross_sectional_rank_10d"].notna()
        & rows["predicted_cross_sectional_rank"].notna()
    ]
    daily: list[float] = []
    for _, group in valid.groupby("date", sort=True):
        if len(group) < 3:
            continue
        value = group["predicted_cross_sectional_rank"].corr(
            group["target_cross_sectional_rank_10d"]
        )
        if pd.notna(value):
            daily.append(float(value))
    values = np.asarray(daily, dtype=float)
    hac = autonomous.newey_west_t(values, lag) if len(values) >= 3 else None
    return {
        "days": len(values),
        "mean": round(float(values.mean()), 8) if len(values) else None,
        "positiveDayFraction": round(float((values > 0.0).mean()), 8)
        if len(values)
        else None,
        "tHac": round(float(hac), 4) if hac is not None else None,
        "hacLag": lag,
    }


def paired_comparison(
    primary: pd.Series,
    comparator: pd.Series,
    lag: int,
) -> dict[str, Any]:
    joined = pd.concat(
        [primary.rename("primary"), comparator.rename("comparator")], axis=1
    ).dropna()
    if joined.empty:
        return {
            "commonSignalDays": 0,
            "meanReturnDelta": None,
            "winRateDelta": None,
            "returnDeltaTHac": None,
            "winDeltaTHac": None,
        }
    return_delta = (joined["primary"] - joined["comparator"]).to_numpy(dtype=float)
    win_delta = (
        joined["primary"].gt(0.0).astype(float)
        - joined["comparator"].gt(0.0).astype(float)
    ).to_numpy(dtype=float)
    return_t = autonomous.newey_west_t(return_delta, lag)
    win_t = autonomous.newey_west_t(win_delta, lag)
    return {
        "commonSignalDays": len(joined),
        "primaryWinRate": round(float(joined["primary"].gt(0.0).mean()), 8),
        "comparatorWinRate": round(
            float(joined["comparator"].gt(0.0).mean()), 8
        ),
        "meanReturnDelta": round(float(return_delta.mean()), 8),
        "winRateDelta": round(float(win_delta.mean()), 8),
        "returnDeltaTHac": round(float(return_t), 4) if return_t is not None else None,
        "winDeltaTHac": round(float(win_t), 4) if win_t is not None else None,
        "hacLag": lag,
    }


def monthly_metrics(values: pd.Series) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if values.empty:
        return output
    for month, group in values.groupby(values.index.to_period("M")):
        output.append(
            {
                "month": str(month),
                "signalDays": len(group),
                "meanReturn": round(float(group.mean()), 8),
                "winRate": round(float(group.gt(0.0).mean()), 8),
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
    metrics = {
        name: selector_v1.portfolio_metrics(
            selected, target, one_day, eligible, dates, config
        )
        for name, selected in policies.items()
    }
    baskets = {
        name: basket_series(selected, target, dates)
        for name, selected in policies.items()
    }
    primary = baskets[PRIMARY_POLICY]
    lag = int(config["evaluation"]["hacLagTradingDays"])
    horizon = int(config["data"]["holdingTradingDays"])
    return {
        "candidateRows": len(labelled),
        "riskOnDays": int(labelled.loc[labelled["risk_on"], "date"].nunique()),
        "rankIc": rank_ic_metrics(labelled, lag),
        "policies": metrics,
        "primaryIndependentEvents": independent_event_metrics(
            primary, calendar_dates, horizon
        ),
        "primaryMonthly": monthly_metrics(primary),
        "primaryVsFactorTop10All": paired_comparison(
            primary, baskets["frozen_factor_top10_all_days"], lag
        ),
        "primaryVsFactorTop5SameRiskDays": paired_comparison(
            primary, baskets["frozen_factor_top5_risk_on"], lag
        ),
    }


def build_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["evaluation"]
    checks: dict[str, Any] = {}
    stable = True
    for period in ("validation", "shadow"):
        block = report["periods"][period]
        primary = block["policies"][PRIMARY_POLICY]
        baseline = block["policies"]["frozen_factor_top10_all_days"]
        same_day = block["primaryVsFactorTop5SameRiskDays"]
        independent = block["primaryIndependentEvents"]
        win_lift = (
            primary["basketTenDayWinRate"] - baseline["basketTenDayWinRate"]
            if primary["basketTenDayWinRate"] is not None
            and baseline["basketTenDayWinRate"] is not None
            else None
        )
        period_checks = {
            "enoughSignalDays": primary["newSignalDays"]
            >= int(rules["minimumSignalDaysPerValidationPeriod"]),
            "enoughIndependentEvents": independent["n"]
            >= int(rules["minimumIndependentEventsPerValidationPeriod"]),
            "winRateLiftVsTop10": round(float(win_lift), 8)
            if win_lift is not None
            else None,
            "winRateLiftPass": win_lift is not None
            and win_lift >= float(rules["minimumWinRateLiftVsTop10"]),
            "positiveMeanReturn": primary["basketTenDayMeanReturn"] is not None
            and primary["basketTenDayMeanReturn"] > 0.0,
            "positiveCostedCumulativeReturn": primary["costedCumulativeReturn"]
            > 0.0,
            "sameDaySelectionWinLiftPass": same_day["winRateDelta"] is not None
            and same_day["winRateDelta"]
            > float(rules["minimumSameDayWinRateLiftVsFactorTop5"]),
            "sameDayReturnHacPass": same_day["returnDeltaTHac"] is not None
            and same_day["returnDeltaTHac"]
            >= float(rules["minimumHacTForSameDayReturnDifference"]),
        }
        passed = all(
            bool(value)
            for key, value in period_checks.items()
            if key.endswith("Pass")
            or key
            in {
                "enoughSignalDays",
                "enoughIndependentEvents",
                "positiveMeanReturn",
                "positiveCostedCumulativeReturn",
            }
        )
        period_checks["periodPass"] = passed
        checks[period] = period_checks
        stable = stable and passed
    return {
        "status": "research_only_not_eligible_for_trading",
        "primaryPolicy": PRIMARY_POLICY,
        "stableHistoricalWinRateImprovement": stable,
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
        "# Perception-XAlpha Win-Rate V2",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- Primary: rolling three-year ridge rank Top5, enabled only when the past-only 20-day market trend and breadth gate is risk-on.",
        "- Orders: always empty.",
        "",
        "## Win-rate comparison",
        "",
        "| Period | Policy | Signal days | Coverage | 10d mean | 10d win | Costed cumulative | Max drawdown |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        block = report["periods"].get(period)
        if not block:
            continue
        for name, metric in block["policies"].items():
            coverage = metric["newSignalDays"] / max(1, metric["calendarDays"])
            lines.append(
                f"| {period} | {name} | {metric['newSignalDays']} | {coverage:.2%} | "
                f"{metric['basketTenDayMeanReturn']} | {metric['basketTenDayWinRate']} | "
                f"{metric['costedCumulativeReturn']} | {metric['costedMaximumDrawdown']} |"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Decision: `{report['verdict']['decision']}`",
            f"- Stable validation and shadow improvement: `{report['verdict']['stableHistoricalWinRateImprovement']}`",
            "- Lower activity alone is not accepted: the primary must beat factor Top5 on the same risk-on dates and pass coverage, independent-event, HAC, return and cost checks.",
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
    selector_config_path = ROOT / config["baseSelectorConfig"]
    selector_config = load_json(selector_config_path)
    selector_v1.validate_config(selector_config)
    research_config = load_json(ROOT / selector_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(research_config)
    if int(cog_config["data"]["predictionHorizonTradingDays"]) != int(
        config["data"]["holdingTradingDays"]
    ):
        raise ValueError("base label horizon and win-rate horizon differ")
    panel, panel_audit = perception.build_configured_panel(
        research_config, cog_config
    )
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
    table = prepare_rank_table(raw_table, config)
    predictions, audits = rolling_walk_forward_predictions(
        table, panel["close"].index, config
    )
    predictions = add_market_gate(predictions, config)
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
        "walkForwardAudits": audits,
        "periods": {},
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    period_inputs = {
        "trainWalkForward": train_walk_dates,
        "validation": validation_dates,
        "shadow": shadow_dates,
    }
    for name, dates in period_inputs.items():
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
        latest["predicted_order"].le(int(config["data"]["maximumSelectionsPerDay"]))
        & latest["risk_on"]
    )
    latest_columns = [
        "date",
        "securityId",
        "candidate_rank",
        "factor_composite",
        "predicted_cross_sectional_rank",
        "predicted_order",
        "market_return_20",
        "market_breadth_20",
        "risk_on",
        "selectedByPrimaryPolicy",
    ]
    atomic_write_text(
        out / "latest_ranking.csv",
        latest[latest_columns].to_csv(index=False, lineterminator="\n"),
    )
    print(json.dumps({
        "runId": run_id,
        "output": str(out),
        "dataRange": report["dataRange"],
        "predictionRows": len(predictions),
        "verdict": report["verdict"],
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    run(args.config.resolve(), args.run_id or None)


if __name__ == "__main__":
    main()
