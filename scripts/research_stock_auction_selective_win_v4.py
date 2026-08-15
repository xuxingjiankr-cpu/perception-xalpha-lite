#!/usr/bin/env python3
"""Win-rate-first selective opening-auction classifier, research-only."""

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
import research_stock_auction_ranked_abstention_v3 as v3  # noqa: E402
import research_stock_auction_signal_amplification_v1 as auction  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_selective_win_v4.json"
SCHEMA_VERSION = "stock_auction_selective_win_result_v4"
CODE_VERSION = "stock_auction_selective_win_v4_20260815"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_auction_selective_win_v4":
        raise ValueError("unexpected V4 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V4 must remain research-only")
    hypothesis = config["preregisteredHypothesis"]
    if not hypothesis.get("singleCandidate") or not hypothesis.get("parametersFrozenBeforeRun"):
        raise ValueError("V4 must contain one frozen candidate")
    if hypothesis.get("validationMayTune") is not False:
        raise ValueError("validation cannot tune V4")
    ranking = config["ranking"]
    if ranking.get("source") != "raw_probability_up_head_only":
        raise ValueError("V4 must rank only the raw probability-up head")
    if float(ranking["expectedReturnHeadRankWeight"]) != 0.0 or float(
        ranking["tailHeadRankWeight"]
    ) != 0.0:
        raise ValueError("return and tail heads cannot enter V4 ordering")
    if ranking.get("calibratedProbabilityMayDetermineRankOrder") is not False:
        raise ValueError("calibrated probability cannot determine V4 order")
    selection = config["selectiveCalibration"]
    if selection.get("calibrationUsesTrainingWindowOnly") is not True:
        raise ValueError("selective calibration must be train-only")
    if not selection.get("maySelectFewerThanMaximum") or not selection.get(
        "mayAbstainEntireDay"
    ):
        raise ValueError("V4 must support abstention")
    if config["evaluation"].get("matchedCoverageControlRequired") is not True:
        raise ValueError("matched coverage control is required")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def wilson_lower_bound(hits: int, count: int, z: float) -> float:
    if count <= 0:
        return 0.0
    probability = hits / count
    denominator = 1.0 + z * z / count
    center = probability + z * z / (2.0 * count)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count * count)
    )
    return max(0.0, (center - radius) / denominator)


def percentile_bins(values: np.ndarray, bin_count: int) -> np.ndarray:
    ranks = pd.Series(values).rank(pct=True, method="average").to_numpy(dtype=float)
    return np.clip(np.ceil(ranks * bin_count).astype(int) - 1, 0, bin_count - 1)


def fit_selective_reliability(
    heads: dict[str, Any],
    features: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    calibration_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    v2_config: dict[str, Any],
    seed_offset: int,
) -> list[dict[str, Any]]:
    spec = config["selectiveCalibration"]
    bin_count = int(spec["rankBins"])
    maximum = int(v2_config["training"]["maximumRowsPerTradingDay"])
    rng = np.random.default_rng(int(v2_config["training"]["randomSeed"]) + seed_offset)
    counts = np.zeros(bin_count, dtype=int)
    hits = np.zeros(bin_count, dtype=int)
    used_days = 0
    for date in calibration_dates:
        y = outcome.loc[date].to_numpy(dtype=float)
        x = np.column_stack(
            [frame.loc[date].to_numpy(dtype=float) for frame in features.values()]
        )
        valid = np.flatnonzero(np.isfinite(y) & np.isfinite(x).all(axis=1))
        if len(valid) < 100:
            continue
        raw = heads["upModel"].predict_proba(x[valid])[:, 1]
        bins = percentile_bins(raw, bin_count)
        if len(valid) > maximum:
            chosen = np.sort(rng.choice(np.arange(len(valid)), size=maximum, replace=False))
            bins = bins[chosen]
            labels = y[valid][chosen] > 0.0
        else:
            labels = y[valid] > 0.0
        for position in range(bin_count):
            mask = bins == position
            counts[position] += int(mask.sum())
            hits[position] += int(labels[mask].sum())
        used_days += 1
    z = float(spec["oneSidedNormalCriticalValue"])
    minimum_count = int(spec["minimumBinObservations"])
    minimum_lower = float(spec["minimumWinRateLowerBound"])
    rows: list[dict[str, Any]] = []
    for position in range(bin_count):
        count, hit = int(counts[position]), int(hits[position])
        probability = hit / count if count else 0.0
        lower = wilson_lower_bound(hit, count, z)
        rows.append(
            {
                "bin": position,
                "lowerRankExclusive": position / bin_count,
                "upperRankInclusive": (position + 1) / bin_count,
                "observations": count,
                "wins": hit,
                "empiricalWinRate": probability,
                "oneSidedLowerBound": lower,
                "eligible": count >= minimum_count and lower > minimum_lower,
                "calibrationTradingDays": used_days,
            }
        )
    return rows


def predict_dates(
    heads: dict[str, Any],
    reliability: list[dict[str, Any]],
    features: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    template = next(iter(features.values()))
    selected_score = pd.DataFrame(np.nan, index=template.index, columns=template.columns)
    support_score = selected_score.copy()
    predictions = {
        name: selected_score.copy()
        for name in ("probability_up", "expected_return", "probability_severe_loss", "win_rate_lower_bound")
    }
    bin_count = int(config["selectiveCalibration"]["rankBins"])
    tail_ceiling = float(
        config["selectiveCalibration"]["maximumCalibratedSevereLossProbability"]
    )
    days: list[dict[str, Any]] = []
    for date in dates:
        x = np.column_stack(
            [frame.loc[date].to_numpy(dtype=float) for frame in features.values()]
        )
        valid = np.flatnonzero(np.isfinite(x).all(axis=1))
        if len(valid) < 100:
            continue
        xv = x[valid]
        raw_up = heads["upModel"].predict_proba(xv)[:, 1]
        bins = percentile_bins(raw_up, bin_count)
        raw_tail = heads["tailModel"].predict_proba(xv)[:, 1]
        probability_tail = np.clip(heads["tailCalibrator"](raw_tail), 0.0, 1.0)
        probability_up = np.asarray(
            [reliability[position]["empiricalWinRate"] for position in bins], dtype=float
        )
        lower = np.asarray(
            [reliability[position]["oneSidedLowerBound"] for position in bins], dtype=float
        )
        reliable = np.asarray([reliability[position]["eligible"] for position in bins])
        qualified = reliable & (probability_tail <= tail_ceiling)
        raw_return = heads["returnModel"].predict(xv)
        expected_return = np.asarray(heads["returnCalibrator"](raw_return), dtype=float)
        columns = template.columns[valid]
        support_score.loc[date, columns] = raw_up
        selected_score.loc[date, columns[qualified]] = raw_up[qualified]
        predictions["probability_up"].loc[date, columns] = probability_up
        predictions["expected_return"].loc[date, columns] = expected_return
        predictions["probability_severe_loss"].loc[date, columns] = probability_tail
        predictions["win_rate_lower_bound"].loc[date, columns] = lower
        days.append(
            {
                "date": date.date().isoformat(),
                "supportRows": int(len(valid)),
                "qualifiedRows": int(qualified.sum()),
                "selectedRows": int(min(10, qualified.sum())),
                "rawProbabilityDistinct": int(len(np.unique(np.round(raw_up, 12)))),
            }
        )
    return selected_score, support_score, predictions, {
        "days": days,
        "predictionRows": int(sum(row["supportRows"] for row in days)),
        "qualifiedRows": int(sum(row["qualifiedRows"] for row in days)),
        "selectedRows": int(sum(row["selectedRows"] for row in days)),
        "zeroSelectionDays": int(sum(row["selectedRows"] == 0 for row in days)),
        "fullTop10Days": int(sum(row["selectedRows"] == 10 for row in days)),
    }


def acceptance(
    candidate: dict[str, Any], control: dict[str, Any], blocks: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, bool]:
    spec = config["evaluation"]
    available = candidate.get("selectedRows", 0) > 0 and control.get("selectedRows", 0) > 0
    comparisons = [row["comparison"] for row in blocks]
    return {
        "hasSelections": available,
        "minimumActiveDayRate": candidate.get("activeDayRate", 0.0)
        >= float(spec["minimumActiveDayRate"]),
        "minimumAverageSelections": candidate.get("averageSelectionsPerActiveDay", 0.0)
        >= float(spec["minimumAverageSelectionsPerActiveDay"]),
        "winRateImprovesByFrozenMinimum": available
        and 100.0 * (candidate["grossUpRate"] - control["grossUpRate"])
        >= float(spec["minimumWinRateImprovementPercentagePoints"]),
        "meanGrossNotWorse": available
        and candidate["meanGrossReturn"] >= control["meanGrossReturn"],
        "meanPercentileImproved": available
        and candidate["meanReturnPercentile"] > control["meanReturnPercentile"],
        "severeLossNotWorse": available
        and candidate["severeLossRate"] <= control["severeLossRate"],
        "majorityBlocksImproveWinRate": bool(comparisons)
        and sum(row["winImproved"] for row in comparisons) > len(comparisons) / 2.0,
        "majorityBlocksDoNotWorsenGross": bool(comparisons)
        and sum(row["grossNotWorse"] for row in comparisons) > len(comparisons) / 2.0,
    }


def comparison(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    available = candidate.get("selectedRows", 0) > 0 and control.get("selectedRows", 0) > 0
    return {
        "winImproved": available and candidate["grossUpRate"] > control["grossUpRate"],
        "grossNotWorse": available
        and candidate["meanGrossReturn"] >= control["meanGrossReturn"],
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
    training = v2_config["training"]
    lookback = int(training["lookbackTradingDays"])
    purge = int(training["outerPurgeTradingDays"])
    refit = int(training["refitEveryTradingDays"])
    calibration_days = int(training["calibrationTradingDays"])
    selected_score = pd.DataFrame(np.nan, index=all_dates, columns=panel["close"].columns)
    support_score = selected_score.copy()
    predictions = {
        name: selected_score.copy()
        for name in ("probability_up", "expected_return", "probability_severe_loss", "win_rate_lower_bound")
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
        calibration = train[-calibration_days:]
        reliability = fit_selective_reliability(
            heads,
            features,
            outcome,
            calibration,
            config,
            v2_config,
            block_no * 10000 + 2000,
        )
        block_score, block_support, block_predictions, prediction_audit = predict_dates(
            heads, reliability, features, test, config
        )
        selected = v3.capped_selection_score(
            block_score.loc[test], int(config["selectiveCalibration"]["maximumSelectionsPerDay"])
        )
        matched = v3.matched_control_score(
            control_score.loc[test],
            block_support.loc[test],
            selected,
            int(config["selectiveCalibration"]["maximumSelectionsPerDay"]),
        )
        selected_score.loc[test] = selected
        support_score.loc[test] = block_support.loc[test]
        for name in predictions:
            predictions[name].loc[test] = block_predictions[name].loc[test]
        candidate_metric = v3.variable_metrics(outcome, selected, test, config)
        control_metric = v3.variable_metrics(outcome, matched, test, config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": fit_audit,
                "selectiveReliability": reliability,
                "prediction": {key: value for key, value in prediction_audit.items() if key != "days"},
                "candidate": candidate_metric,
                "matchedCoverageControl": control_metric,
                "comparison": comparison(candidate_metric, control_metric),
            }
        )
        print(f"v4_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[support_score.loc[evaluation_dates].notna().any(axis=1)]
    maximum = int(config["selectiveCalibration"]["maximumSelectionsPerDay"])
    matched_score = v3.matched_control_score(
        control_score.loc[scored_dates],
        support_score.loc[scored_dates],
        selected_score.loc[scored_dates],
        maximum,
    )
    candidate = v3.variable_metrics(outcome, selected_score, scored_dates, config)
    matched = v3.variable_metrics(outcome, matched_score, scored_dates, config)
    full_control = auction.metrics(
        outcome, control_score.where(support_score.notna()), scored_dates, auction_config
    )
    checks = acceptance(candidate, matched, blocks, config)
    passed = bool(all(checks.values()))
    latest_date = scored_dates.max()
    latest = selected_score.loc[latest_date].dropna().sort_values(ascending=False).head(maximum)
    latest_rows: list[dict[str, Any]] = []
    for rank, security in enumerate(latest.index, start=1):
        latest_rows.append(
            {
                "rank": rank,
                "signalDate": latest_date.date().isoformat(),
                "securityId": security,
                "rawProbabilityUpScore": float(latest.loc[security]),
                "calibrationBinWinRate": float(predictions["probability_up"].loc[latest_date, security]),
                "winRateLowerBound": float(
                    predictions["win_rate_lower_bound"].loc[latest_date, security]
                ),
                "expectedGrossReturnDiagnosticOnly": float(
                    predictions["expected_return"].loc[latest_date, security]
                ),
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
        "ranking": config["ranking"],
        "selectiveCalibration": config["selectiveCalibration"],
        "candidate": candidate,
        "matchedCoverageControl": matched,
        "fullTop10Control": full_control,
        "checks": checks,
        "passed": passed,
        "decision": "fresh_forward_preregistration_only" if passed else "reject_selective_win_v4",
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
        "# Selective win classifier V4",
        "",
        "> Research-only. Historical windows were already viewed; passing can only justify a fresh-forward preregistration.",
        "",
        f"- run: `{run_id}`",
        f"- decision: `{result['decision']}`",
        f"- active days: {candidate.get('activeTradingDays', 0)}/{candidate.get('evaluationDays', 0)}",
        "",
        "| cohort | win rate | mean gross | net @ 30bp | severe loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("V4", candidate), ("matched control", matched)):
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
