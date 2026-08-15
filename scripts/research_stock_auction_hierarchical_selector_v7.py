#!/usr/bin/env python3
"""Frozen auction day gate + prior-close rank + tail-only safety filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_stock_auction_expanded_pit_win_v5 as v5  # noqa: E402
import research_stock_auction_market_breadth_gate_v6 as v6  # noqa: E402
import research_stock_auction_multitask_utility_v2 as v2  # noqa: E402
import research_stock_auction_ranked_abstention_v3 as v3  # noqa: E402
import research_stock_auction_selective_win_v4 as v4  # noqa: E402
import research_stock_auction_signal_amplification_v1 as auction  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_hierarchical_selector_v7.json"
SCHEMA_VERSION = "stock_auction_hierarchical_selector_result_v7"
CODE_VERSION = "stock_auction_hierarchical_selector_v7_20260815"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_auction_hierarchical_selector_v7":
        raise ValueError("unexpected V7 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V7 must remain research-only")
    hypothesis = config["preregisteredHypothesis"]
    if not hypothesis.get("singleCandidate") or not hypothesis.get("parametersFrozenBeforeRun"):
        raise ValueError("V7 must contain one frozen candidate")
    if hypothesis.get("validationMayTune") is not False:
        raise ValueError("V7 validation cannot tune")
    if hypothesis.get("noNewFittedParameter") is not True:
        raise ValueError("V7 cannot introduce a fitted parameter")
    policy = config["policy"]
    if policy.get("stockRanking") != "unchanged_prior_close_rank":
        raise ValueError("V7 stock ranking must remain the prior-close rank")
    if float(policy.get("maximumCalibratedSevereLossProbability")) != 0.08:
        raise ValueError("V7 must reuse the frozen V4 tail ceiling")
    if config["evaluation"].get("sameDaySameSupportSameCount") is not True:
        raise ValueError("V7 requires the matched-count control")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def tail_safe_baseline_score(
    baseline: pd.DataFrame,
    tail_probability: pd.DataFrame,
    eligible_days: pd.Series,
    ceiling: float,
) -> pd.DataFrame:
    score = baseline.where(tail_probability.le(ceiling))
    score.loc[eligible_days.index[~eligible_days.to_numpy()]] = np.nan
    return score


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    v6_path = ROOT / config["sourceBreadthGateConfig"]
    v6_config = load_json(v6_path)
    v6.validate_config(v6_config)
    v5_path = ROOT / v6_config["sourceExpandedPitConfig"]
    v5_config = load_json(v5_path)
    v4_path = ROOT / v5_config["sourceSelectiveWinConfig"]
    v4_config = load_json(v4_path)
    v2_path = ROOT / v4_config["sourceMultitaskConfig"]
    v2_config = load_json(v2_path)
    auction_config = load_json(ROOT / v2_config["sourceAuctionConfig"])
    source_config = load_json(ROOT / v2_config["sourceWinCaptureConfig"])
    data = win_capture.build_inputs(source_config)
    panel = data["panel"]
    opening = auction.opening_time_mask(panel, data["baseConfig"]["assetUniverse"], auction_config)
    base_features, route, outcome, complete = auction.build_execution_features(
        panel, data["baselineScore"], opening, auction_config
    )
    auction_features = v2.model_features(base_features, route, complete)
    features, expanded_complete = v5.expanded_features(
        auction_features, data["featureSets"][v5_config["sourceFeatureSet"]], complete, v5_config
    )
    day_features, day_label = v6.build_day_table(
        panel, opening, expanded_complete, outcome, v6_config
    )
    control_score = base_features["prior_close_rank"].where(expanded_complete)
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
    inner_purge = int(training["innerPurgeTradingDays"])
    selected_score = pd.DataFrame(np.nan, index=all_dates, columns=panel["close"].columns)
    support_score = selected_score.copy()
    tail_probability = selected_score.copy()
    day_probability = pd.Series(np.nan, index=all_dates, dtype=float)
    blocks: list[dict[str, Any]] = []
    ceiling = float(config["policy"]["maximumCalibratedSevereLossProbability"])
    maximum = int(config["policy"]["maximumSelectionsPerDay"])
    for block_no, start in enumerate(range(0, len(evaluation_dates), refit)):
        test = evaluation_dates[start : start + refit]
        position = all_dates.get_loc(test[0])
        train_end = position - purge
        train = all_dates[max(0, train_end - lookback) : train_end]
        if len(train) < lookback:
            continue
        heads, fit_audit = v2.fit_heads(features, outcome, train, v2_config, block_no * 10000)
        calibration = train[-calibration_days:]
        base = train[: -calibration_days - inner_purge]
        reliability = v4.fit_selective_reliability(
            heads, features, outcome, calibration, v4_config, v2_config, block_no * 10000 + 2000
        )
        gate, gate_audit = v6.fit_day_gate(day_features, day_label, base, calibration, v6_config)
        _win_score, block_support, predictions, prediction_audit = v4.predict_dates(
            heads, reliability, features, test, v4_config
        )
        eligible_days, probabilities = v6.predict_day_gate(gate, day_features, test)
        block_score = tail_safe_baseline_score(
            control_score.loc[test], predictions["probability_severe_loss"].loc[test], eligible_days, ceiling
        )
        selected = v3.capped_selection_score(block_score, maximum)
        matched = v3.matched_control_score(
            control_score.loc[test], block_support.loc[test], selected, maximum
        )
        selected_score.loc[test] = selected
        support_score.loc[test] = block_support.loc[test]
        tail_probability.loc[test] = predictions["probability_severe_loss"].loc[test]
        day_probability.loc[test] = probabilities
        candidate_metric = v3.variable_metrics(outcome, selected, test, v4_config)
        control_metric = v3.variable_metrics(outcome, matched, test, v4_config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": fit_audit,
                "dayGate": gate_audit,
                "eligibleTestDays": int(eligible_days.sum()),
                "tailSafeRows": int(predictions["probability_severe_loss"].loc[test].le(ceiling).sum().sum()),
                "prediction": {key: value for key, value in prediction_audit.items() if key != "days"},
                "candidate": candidate_metric,
                "matchedCoverageControl": control_metric,
                "comparison": v4.comparison(candidate_metric, control_metric),
            }
        )
        print(f"v7_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[support_score.loc[evaluation_dates].notna().any(axis=1)]
    matched_score = v3.matched_control_score(
        control_score.loc[scored_dates], support_score.loc[scored_dates], selected_score.loc[scored_dates], maximum
    )
    candidate = v3.variable_metrics(outcome, selected_score, scored_dates, v4_config)
    matched = v3.variable_metrics(outcome, matched_score, scored_dates, v4_config)
    checks = v4.acceptance(candidate, matched, blocks, v4_config)
    enabled_blocks = sum(bool(block["dayGate"].get("enabled")) for block in blocks)
    checks["minimumDayGateEnabledBlocks"] = enabled_blocks >= int(
        v6_config["dayGate"]["minimumEnabledWalkForwardBlocks"]
    )
    passed = bool(all(checks.values()))
    latest_date = scored_dates.max()
    latest = selected_score.loc[latest_date].dropna().sort_values(ascending=False).head(maximum)
    latest_rows = [
        {
            "rank": rank,
            "signalDate": latest_date.date().isoformat(),
            "securityId": security,
            "priorCloseRankScore": float(latest.loc[security]),
            "probabilitySevereLoss": float(tail_probability.loc[latest_date, security]),
            "dayGateRawProbability": float(day_probability.loc[latest_date]),
            "status": "research_only_shadow_candidate_not_an_order",
        }
        for rank, security in enumerate(latest.index, start=1)
    ]
    now = datetime.now().astimezone()
    run_id = run_id or f"run_{now:%Y%m%dT%H%M%S%z}"
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": config["status"],
        "runId": run_id,
        "generatedAt": now.isoformat(),
        "configSha256": digest(config_path),
        "sourceV6ConfigSha256": digest(v6_path),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "evaluationRange": [scored_dates.min().date().isoformat(), scored_dates.max().date().isoformat()],
        "symbolUniverse": int(panel["close"].shape[1]),
        "candidate": candidate,
        "matchedCoverageControl": matched,
        "dayGateEnabledBlocks": enabled_blocks,
        "checks": checks,
        "passed": passed,
        "decision": "fresh_forward_preregistration_only" if passed else "reject_auction_hierarchical_selector_v7",
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
        "# Hierarchical opening-auction selector V7",
        "",
        "> Research-only. No trading integration or automatic promotion.",
        "",
        f"- run: `{run_id}`",
        f"- decision: `{result['decision']}`",
        f"- active days: {candidate.get('activeTradingDays', 0)}/{candidate.get('evaluationDays', 0)}",
        "",
        "| cohort | win rate | mean gross | net @ 30bp | severe loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("V7", candidate), ("matched control", matched)):
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
