#!/usr/bin/env python3
"""Expanded PIT-feature selective win classifier, research-only."""

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

import research_stock_auction_multitask_utility_v2 as v2  # noqa: E402
import research_stock_auction_ranked_abstention_v3 as v3  # noqa: E402
import research_stock_auction_selective_win_v4 as v4  # noqa: E402
import research_stock_auction_signal_amplification_v1 as auction  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_expanded_pit_win_v5.json"
SCHEMA_VERSION = "stock_auction_expanded_pit_win_result_v5"
CODE_VERSION = "stock_auction_expanded_pit_win_v5_20260815"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_auction_expanded_pit_win_v5":
        raise ValueError("unexpected V5 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V5 must remain research-only")
    hypothesis = config["preregisteredHypothesis"]
    if not hypothesis.get("singleCandidate") or not hypothesis.get("parametersFrozenBeforeRun"):
        raise ValueError("V5 must contain one frozen candidate")
    if hypothesis.get("validationMayTune") is not False:
        raise ValueError("V5 validation cannot tune")
    features = config["features"]
    if features.get("priorCloseFeaturesMustUseShiftOne") is not True:
        raise ValueError("V5 PIT features must be lagged")
    if features.get("sameSessionHighLowCloseVolumeAmountForbidden") is not True:
        raise ValueError("same-session future fields are forbidden")
    if features.get("missingFeatureMayBeFilledFromFuture") is not False:
        raise ValueError("future feature completion is forbidden")
    if config["evaluation"].get("matchedCoverageControlRequired") is not True:
        raise ValueError("matched coverage control is required")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def expanded_features(
    auction_features: dict[str, pd.DataFrame],
    prior_close_features: dict[str, pd.DataFrame],
    complete: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    expected = int(config["features"]["priorCloseFeatureCount"])
    if len(prior_close_features) != expected:
        raise ValueError(f"expected {expected} prior-close features, got {len(prior_close_features)}")
    lagged = {
        f"prior_pit/{name}": frame.shift(1)
        for name, frame in prior_close_features.items()
    }
    expanded_complete = complete.copy()
    for frame in lagged.values():
        expanded_complete &= frame.notna()
    combined = {
        **{name: frame.where(expanded_complete) for name, frame in auction_features.items()},
        **{name: frame.where(expanded_complete) for name, frame in lagged.items()},
    }
    return combined, expanded_complete


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    v4_path = ROOT / config["sourceSelectiveWinConfig"]
    v4_config = load_json(v4_path)
    v4.validate_config(v4_config)
    v2_path = ROOT / v4_config["sourceMultitaskConfig"]
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
    auction_features = v2.model_features(base_features, route, complete)
    source_feature_set = data["featureSets"][config["sourceFeatureSet"]]
    features, expanded_complete = expanded_features(
        auction_features, source_feature_set, complete, config
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
        reliability = v4.fit_selective_reliability(
            heads,
            features,
            outcome,
            calibration,
            v4_config,
            v2_config,
            block_no * 10000 + 2000,
        )
        block_score, block_support, block_predictions, prediction_audit = v4.predict_dates(
            heads, reliability, features, test, v4_config
        )
        maximum = int(v4_config["selectiveCalibration"]["maximumSelectionsPerDay"])
        selected = v3.capped_selection_score(block_score.loc[test], maximum)
        matched = v3.matched_control_score(
            control_score.loc[test], block_support.loc[test], selected, maximum
        )
        selected_score.loc[test] = selected
        support_score.loc[test] = block_support.loc[test]
        for name in predictions:
            predictions[name].loc[test] = block_predictions[name].loc[test]
        candidate_metric = v3.variable_metrics(outcome, selected, test, v4_config)
        control_metric = v3.variable_metrics(outcome, matched, test, v4_config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": fit_audit,
                "selectiveReliability": reliability,
                "prediction": {key: value for key, value in prediction_audit.items() if key != "days"},
                "candidate": candidate_metric,
                "matchedCoverageControl": control_metric,
                "comparison": v4.comparison(candidate_metric, control_metric),
            }
        )
        print(f"v5_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[support_score.loc[evaluation_dates].notna().any(axis=1)]
    maximum = int(v4_config["selectiveCalibration"]["maximumSelectionsPerDay"])
    matched_score = v3.matched_control_score(
        control_score.loc[scored_dates],
        support_score.loc[scored_dates],
        selected_score.loc[scored_dates],
        maximum,
    )
    candidate = v3.variable_metrics(outcome, selected_score, scored_dates, v4_config)
    matched = v3.variable_metrics(outcome, matched_score, scored_dates, v4_config)
    full_control = auction.metrics(
        outcome, control_score.where(support_score.notna()), scored_dates, auction_config
    )
    checks = v4.acceptance(candidate, matched, blocks, v4_config)
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
        "sourceV4ConfigSha256": digest(v4_path),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "evaluationRange": [scored_dates.min().date().isoformat(), scored_dates.max().date().isoformat()],
        "symbolUniverse": int(panel["close"].shape[1]),
        "auctionFeatureCount": len(auction_features),
        "priorPitFeatureCount": len(source_feature_set),
        "featureCount": len(features),
        "featureOrder": list(features),
        "candidate": candidate,
        "matchedCoverageControl": matched,
        "fullTop10Control": full_control,
        "checks": checks,
        "passed": passed,
        "decision": "fresh_forward_preregistration_only" if passed else "reject_expanded_pit_win_v5",
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
        "# Expanded PIT auction win classifier V5",
        "",
        "> Research-only. The twenty PIT features are lagged one full session.",
        "",
        f"- run: `{run_id}`",
        f"- decision: `{result['decision']}`",
        f"- feature count: {len(features)}",
        f"- active days: {candidate.get('activeTradingDays', 0)}/{candidate.get('evaluationDays', 0)}",
        "",
        "| cohort | win rate | mean gross | net @ 30bp | severe loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("V5", candidate), ("matched control", matched)):
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
