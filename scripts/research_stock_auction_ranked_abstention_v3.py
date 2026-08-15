#!/usr/bin/env python3
"""Raw-rank auction utility with a calibrated, fixed abstention policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_stock_auction_multitask_utility_v2 as v2  # noqa: E402
import research_stock_auction_signal_amplification_v1 as auction  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_ranked_abstention_v3.json"
SCHEMA_VERSION = "stock_auction_ranked_abstention_result_v3"
CODE_VERSION = "stock_auction_ranked_abstention_v3_20260815"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_auction_ranked_abstention_v3":
        raise ValueError("unexpected V3 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V3 must remain research-only")
    failure = config["failureBeingCorrected"]
    if not failure.get("parametersFrozenBeforeRun") or not failure.get("singleCandidate"):
        raise ValueError("V3 must contain one preregistered candidate")
    ranking = config["ranking"]
    total = sum(float(value) for key, value in ranking.items() if key.endswith("RankWeight"))
    if abs(total - 1.0) > 1e-12:
        raise ValueError("V3 raw-rank weights must sum to one")
    if ranking.get("calibratedValuesMayDetermineRankOrder") is not False:
        raise ValueError("calibrated values cannot determine V3 rank order")
    gate = config["opportunityFilter"]
    if not gate.get("thresholdsFrozenBeforeRun"):
        raise ValueError("V3 opportunity thresholds must be frozen")
    if not gate.get("maySelectFewerThanMaximum") or not gate.get("mayAbstainEntireDay"):
        raise ValueError("V3 must be allowed to abstain")
    if int(gate["maximumSelectionsPerDay"]) != 10:
        raise ValueError("V3 maximum selection count changed")
    if config["evaluation"].get("matchedCoverageControlRequired") is not True:
        raise ValueError("matched-coverage control is required")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def raw_ranked_utility(
    raw_up: np.ndarray,
    raw_return: np.ndarray,
    raw_tail: np.ndarray,
    ranking: dict[str, Any],
) -> np.ndarray:
    frame = pd.DataFrame(
        {"up": raw_up, "return": raw_return, "tail_safety": 1.0 - raw_tail}
    )
    ranks = frame.rank(pct=True, method="average")
    return (
        float(ranking["probabilityUpRankWeight"]) * ranks["up"]
        + float(ranking["expectedReturnRankWeight"]) * ranks["return"]
        + float(ranking["tailSafetyRankWeight"]) * ranks["tail_safety"]
    ).to_numpy(dtype=float)


def opportunity_mask(
    probability_up: np.ndarray,
    expected_return: np.ndarray,
    probability_tail: np.ndarray,
    gate: dict[str, Any],
) -> np.ndarray:
    return (
        (probability_up >= float(gate["minimumCalibratedProbabilityUp"]))
        & (expected_return >= float(gate["minimumCalibratedExpectedGrossReturn"]))
        & (probability_tail <= float(gate["maximumCalibratedSevereLossProbability"]))
    )


def predict_dates(
    heads: dict[str, Any],
    features: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    template = next(iter(features.values()))
    score = pd.DataFrame(np.nan, index=template.index, columns=template.columns)
    raw_score = score.copy()
    predictions = {
        name: score.copy()
        for name in ("probability_up", "expected_return", "probability_severe_loss")
    }
    audit: list[dict[str, Any]] = []
    for date in dates:
        x = np.column_stack(
            [frame.loc[date].to_numpy(dtype=float) for frame in features.values()]
        )
        valid = np.flatnonzero(np.isfinite(x).all(axis=1))
        if len(valid) < 100:
            continue
        xv = x[valid]
        raw_up = heads["upModel"].predict_proba(xv)[:, 1]
        raw_tail = heads["tailModel"].predict_proba(xv)[:, 1]
        raw_return = heads["returnModel"].predict(xv)
        probability_up = np.clip(heads["upCalibrator"](raw_up), 0.0, 1.0)
        probability_tail = np.clip(heads["tailCalibrator"](raw_tail), 0.0, 1.0)
        expected_return = np.asarray(heads["returnCalibrator"](raw_return), dtype=float)
        utility = raw_ranked_utility(raw_up, raw_return, raw_tail, config["ranking"])
        qualified = opportunity_mask(
            probability_up, expected_return, probability_tail, config["opportunityFilter"]
        )
        columns = template.columns[valid]
        raw_score.loc[date, columns] = utility
        score.loc[date, columns[qualified]] = utility[qualified]
        predictions["probability_up"].loc[date, columns] = probability_up
        predictions["expected_return"].loc[date, columns] = expected_return
        predictions["probability_severe_loss"].loc[date, columns] = probability_tail
        top = np.sort(utility[qualified])[-10:] if qualified.any() else np.array([])
        audit.append(
            {
                "date": date.date().isoformat(),
                "supportRows": int(len(valid)),
                "qualifiedRows": int(qualified.sum()),
                "selectedRows": int(min(10, qualified.sum())),
                "rawUtilityDistinct": int(len(np.unique(np.round(utility, 12)))),
                "selectedUtilitySpread": float(np.ptp(top)) if len(top) > 1 else 0.0,
            }
        )
    return score, raw_score, predictions, {
        "days": audit,
        "predictionRows": int(sum(item["supportRows"] for item in audit)),
        "qualifiedRows": int(sum(item["qualifiedRows"] for item in audit)),
        "selectedRows": int(sum(item["selectedRows"] for item in audit)),
        "zeroSelectionDays": int(sum(item["selectedRows"] == 0 for item in audit)),
        "fullTop10Days": int(sum(item["selectedRows"] == 10 for item in audit)),
    }


def capped_selection_score(score: pd.DataFrame, maximum: int) -> pd.DataFrame:
    selected = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for date in score.index:
        available = score.loc[date].dropna()
        if available.empty:
            continue
        chosen = available.nlargest(min(maximum, len(available))).index
        selected.loc[date, chosen] = available.loc[chosen]
    return selected


def matched_control_score(
    control: pd.DataFrame, support: pd.DataFrame, selected: pd.DataFrame, maximum: int
) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=control.index, columns=control.columns)
    for date in control.index:
        count = min(maximum, int(selected.loc[date].notna().sum()))
        if count == 0:
            continue
        available = control.loc[date].where(support.loc[date].notna()).dropna()
        if len(available) < count:
            continue
        chosen = available.nlargest(count).index
        result.loc[date, chosen] = available.loc[chosen]
    return result


def variable_metrics(
    outcome: pd.DataFrame,
    selected_score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    percentiles: list[np.ndarray] = []
    daily: list[float] = []
    selected_counts: list[int] = []
    majority: list[bool] = []
    unresolved = 0
    evaluation_days = 0
    for date in dates.intersection(selected_score.index):
        market = outcome.loc[date].dropna()
        if len(market) < 100:
            continue
        evaluation_days += 1
        chosen = selected_score.loc[date].dropna().sort_values(ascending=False).index
        if len(chosen) == 0:
            continue
        realized = outcome.loc[date].reindex(chosen)
        unresolved += int(realized.isna().sum())
        realized = realized.dropna()
        if realized.empty:
            continue
        values = realized.to_numpy(dtype=float)
        rows.append(values)
        percentiles.append(market.rank(pct=True).loc[realized.index].to_numpy(dtype=float))
        daily.append(float(values.mean()))
        selected_counts.append(len(values))
        majority.append(bool(np.mean(values > 0.0) > 0.5))
    if not rows:
        return {
            "evaluationDays": evaluation_days,
            "activeTradingDays": 0,
            "activeDayRate": 0.0,
            "selectedRows": 0,
            "unresolvedSelectedRows": unresolved,
        }
    realized = np.concatenate(rows)
    pct = np.concatenate(percentiles)
    evaluation = config["evaluation"]
    cost = float(evaluation["roundTripCost"])
    severe = float(evaluation["severeLossThreshold"])
    return {
        "evaluationDays": evaluation_days,
        "activeTradingDays": len(daily),
        "activeDayRate": len(daily) / evaluation_days,
        "averageSelectionsPerActiveDay": float(np.mean(selected_counts)),
        "fullTop10DayRate": float(np.mean(np.asarray(selected_counts) == 10)),
        "selectedRows": int(len(realized)),
        "unresolvedSelectedRows": unresolved,
        "grossUpRate": float(np.mean(realized > 0.0)),
        "meanGrossReturn": float(np.mean(realized)),
        "meanNetReturnAtConfiguredFriction": float(np.mean(realized) - cost),
        "meanReturnPercentile": float(np.mean(pct)),
        "majorityUpDayRate": float(np.mean(majority)),
        "severeLossRate": float(np.mean(realized <= severe)),
        "worstDailyGrossReturn": float(np.min(daily)),
    }


def block_comparison(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    complete = candidate.get("selectedRows", 0) > 0 and control.get("selectedRows", 0) > 0
    return {
        "winImproved": complete
        and candidate["grossUpRate"] > control["grossUpRate"],
        "grossImproved": complete
        and candidate["meanGrossReturn"] > control["meanGrossReturn"],
    }


def acceptance(
    candidate: dict[str, Any], control: dict[str, Any], blocks: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, bool]:
    spec = config["evaluation"]
    available = candidate.get("selectedRows", 0) > 0 and control.get("selectedRows", 0) > 0
    comparisons = [item["comparison"] for item in blocks]
    return {
        "hasSelections": available,
        "minimumActiveDayRate": candidate.get("activeDayRate", 0.0)
        >= float(spec["minimumActiveDayRate"]),
        "minimumAverageSelections": candidate.get("averageSelectionsPerActiveDay", 0.0)
        >= float(spec["minimumAverageSelectionsPerActiveDay"]),
        "winRateImprovesByFrozenMinimum": available
        and 100.0 * (candidate["grossUpRate"] - control["grossUpRate"])
        >= float(spec["minimumWinRateImprovementPercentagePoints"]),
        "meanGrossImprovesByFrozenMinimum": available
        and 10000.0 * (candidate["meanGrossReturn"] - control["meanGrossReturn"])
        >= float(spec["minimumMeanGrossImprovementBps"]),
        "meanPercentileImproved": available
        and candidate["meanReturnPercentile"] > control["meanReturnPercentile"],
        "severeLossNotWorse": available
        and candidate["severeLossRate"] <= control["severeLossRate"],
        "majorityBlocksImproveWinRate": bool(comparisons)
        and sum(item["winImproved"] for item in comparisons) > len(comparisons) / 2.0,
        "majorityBlocksImproveGross": bool(comparisons)
        and sum(item["grossImproved"] for item in comparisons) > len(comparisons) / 2.0,
    }


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    v2_path = ROOT / config["sourceMultitaskConfig"]
    v2_config = load_json(v2_path)
    v2.validate_config(v2_config)
    auction_config = load_json(ROOT / v2_config["sourceAuctionConfig"])
    auction.validate_config(auction_config)
    source_config = load_json(ROOT / v2_config["sourceWinCaptureConfig"])
    data = win_capture.build_inputs(source_config)
    panel = data["panel"]
    opening = auction.opening_time_mask(
        panel, data["baseConfig"]["assetUniverse"], auction_config
    )
    base_features, route, outcome, complete = auction.build_execution_features(
        panel, data["baselineScore"], opening, auction_config
    )
    features = v2.model_features(base_features, route, complete)
    control_score = base_features["prior_close_rank"].where(complete)
    partitions = discrimination.fixed_partitions(panel["close"].index, data["modelConfig"])
    evaluation_dates = pd.DatetimeIndex(
        sorted(set(partitions["audit"]) | set(partitions["validation"]) | set(partitions["shadow"]))
    )
    all_dates = panel["close"].index
    spec = v2_config["training"]
    lookback = int(spec["lookbackTradingDays"])
    purge = int(spec["outerPurgeTradingDays"])
    refit = int(spec["refitEveryTradingDays"])
    candidate_score = pd.DataFrame(np.nan, index=all_dates, columns=panel["close"].columns)
    raw_score = candidate_score.copy()
    predictions = {
        name: candidate_score.copy()
        for name in ("probability_up", "expected_return", "probability_severe_loss")
    }
    blocks: list[dict[str, Any]] = []
    for block_no, start in enumerate(range(0, len(evaluation_dates), refit)):
        test = evaluation_dates[start : start + refit]
        position = all_dates.get_loc(test[0])
        train_end = position - purge
        train = all_dates[max(0, train_end - lookback) : train_end]
        if len(train) < lookback:
            continue
        heads, fit_audit = v2.fit_heads(features, outcome, train, v2_config, block_no * 10000)
        block_score, block_raw, block_predictions, prediction_audit = predict_dates(
            heads, features, test, config
        )
        selected = capped_selection_score(
            block_score.loc[test], int(config["opportunityFilter"]["maximumSelectionsPerDay"])
        )
        matched = matched_control_score(
            control_score.loc[test],
            block_raw.loc[test],
            selected,
            int(config["opportunityFilter"]["maximumSelectionsPerDay"]),
        )
        candidate_score.loc[test] = selected
        raw_score.loc[test] = block_raw.loc[test]
        for name in predictions:
            predictions[name].loc[test] = block_predictions[name].loc[test]
        candidate_metric = variable_metrics(outcome, selected, test, config)
        control_metric = variable_metrics(outcome, matched, test, config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": fit_audit,
                "prediction": {key: value for key, value in prediction_audit.items() if key != "days"},
                "candidate": candidate_metric,
                "matchedCoverageControl": control_metric,
                "comparison": block_comparison(candidate_metric, control_metric),
            }
        )
        print(f"v3_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[raw_score.loc[evaluation_dates].notna().any(axis=1)]
    maximum = int(config["opportunityFilter"]["maximumSelectionsPerDay"])
    matched_score = matched_control_score(
        control_score.loc[scored_dates],
        raw_score.loc[scored_dates],
        candidate_score.loc[scored_dates],
        maximum,
    )
    candidate = variable_metrics(outcome, candidate_score, scored_dates, config)
    matched = variable_metrics(outcome, matched_score, scored_dates, config)
    full_control = auction.metrics(outcome, control_score.where(raw_score.notna()), scored_dates, auction_config)
    checks = acceptance(candidate, matched, blocks, config)
    passed = bool(all(checks.values()))
    latest_date = scored_dates.max()
    latest = candidate_score.loc[latest_date].dropna().sort_values(ascending=False).head(maximum)
    latest_rows: list[dict[str, Any]] = []
    for rank, security in enumerate(latest.index, start=1):
        latest_rows.append(
            {
                "rank": rank,
                "signalDate": latest_date.date().isoformat(),
                "securityId": security,
                "rawUtilityScore": float(latest.loc[security]),
                "probabilityUp": float(predictions["probability_up"].loc[latest_date, security]),
                "expectedGrossReturn": float(predictions["expected_return"].loc[latest_date, security]),
                "probabilitySevereLoss": float(
                    predictions["probability_severe_loss"].loc[latest_date, security]
                ),
                "status": "research_only_shadow_candidate_not_an_order",
            }
        )
    now = datetime.now().astimezone()
    run_id = run_id or f"run_{now:%Y%m%dT%H%M%S%z}"
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": config["status"],
        "runId": run_id,
        "generatedAt": now.isoformat(),
        "configSha256": digest(config_path),
        "sourceV2ConfigSha256": digest(v2_path),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "evaluationRange": [scored_dates.min().date().isoformat(), scored_dates.max().date().isoformat()],
        "symbolUniverse": int(panel["close"].shape[1]),
        "featureOrder": list(features),
        "ranking": config["ranking"],
        "opportunityFilter": config["opportunityFilter"],
        "candidate": candidate,
        "matchedCoverageControl": matched,
        "fullTop10Control": full_control,
        "checks": checks,
        "passed": passed,
        "decision": "fresh_forward_preregistration_only" if passed else "reject_ranked_abstention_v3",
        "walkForwardBlocks": blocks,
        "latestDiagnosticSelection": latest_rows,
        "historicalWindowsAlreadyViewed": True,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    output = ROOT / config["output"]["root"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(v2.safe(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(latest_rows).to_csv(
        output / "latest_diagnostic_selection.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# Raw-rank auction abstention V3",
        "",
        "> Research-only. No order, position, gate or trading configuration is changed.",
        "",
        f"- run: `{run_id}`",
        f"- decision: `{result['decision']}`",
        f"- active days: {candidate.get('activeTradingDays', 0)}/{candidate.get('evaluationDays', 0)}",
        f"- selected rows: {candidate.get('selectedRows', 0)}",
        "",
        "| cohort | win rate | mean gross | net @ 30bp | severe loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("V3", candidate), ("matched control", matched)):
        if item.get("selectedRows", 0):
            lines.append(
                f"| {label} | {item['grossUpRate']:.2%} | {item['meanGrossReturn']:.4%} | "
                f"{item['meanNetReturnAtConfiguredFriction']:.4%} | {item['severeLossRate']:.2%} |"
            )
    lines += ["", "## Frozen checks", ""]
    lines += [f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(v2.safe(result), ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run(args.config.resolve(), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
