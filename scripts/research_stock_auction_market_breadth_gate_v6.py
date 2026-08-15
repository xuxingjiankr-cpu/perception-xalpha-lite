#!/usr/bin/env python3
"""Hierarchical auction-breadth gate over the frozen V5 stock ranker."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_stock_auction_expanded_pit_win_v5 as v5  # noqa: E402
import research_stock_auction_multitask_utility_v2 as v2  # noqa: E402
import research_stock_auction_ranked_abstention_v3 as v3  # noqa: E402
import research_stock_auction_selective_win_v4 as v4  # noqa: E402
import research_stock_auction_signal_amplification_v1 as auction  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_market_breadth_gate_v6.json"
SCHEMA_VERSION = "stock_auction_market_breadth_gate_result_v6"
CODE_VERSION = "stock_auction_market_breadth_gate_v6_20260815"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_auction_market_breadth_gate_v6":
        raise ValueError("unexpected V6 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V6 must remain research-only")
    hypothesis = config["preregisteredHypothesis"]
    if not hypothesis.get("singleCandidate") or not hypothesis.get("parametersFrozenBeforeRun"):
        raise ValueError("V6 must contain one frozen candidate")
    if hypothesis.get("validationMayTune") is not False:
        raise ValueError("V6 validation cannot tune")
    if hypothesis.get("v5StockModelAndReliabilityUnchanged") is not True:
        raise ValueError("V5 stock ranker must remain unchanged")
    timing = config["timing"]
    if timing.get("priorSessionFeatureShift") != 1:
        raise ValueError("prior-session breadth must be lagged")
    if timing.get("sameSessionCloseHighLowVolumeAmountForbidden") is not True:
        raise ValueError("same-session future fields are forbidden")
    if timing.get("labelMayBeUsedOnlyInsideHistoricalTrainingRows") is not True:
        raise ValueError("day label must remain training-only")
    if config["evaluation"].get("matchedCoverageControlRequired") is not True:
        raise ValueError("matched control is required")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def build_day_table(
    panel: dict[str, pd.DataFrame],
    opening_mask: pd.DataFrame,
    complete: pd.DataFrame,
    outcome: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build 09:25/past-only market features and a separate offline day label."""
    close, open_price = panel["close"], panel["open"]
    gap = open_price.div(close.shift(1).replace(0.0, np.nan)) - 1.0
    available = gap.where(opening_mask & complete)
    count = available.notna().sum(axis=1).replace(0, np.nan)
    threshold = float(config["dayGate"]["largeGapAbsoluteThreshold"])
    prior_intraday = (close.div(open_price.replace(0.0, np.nan)) - 1.0).shift(1)
    prior_available = prior_intraday.where(opening_mask & complete)
    prior_count = prior_available.notna().sum(axis=1).replace(0, np.nan)
    features = pd.DataFrame(
        {
            "auction_positive_gap_breadth": available.gt(0.0).sum(axis=1).div(count),
            "auction_median_gap": available.median(axis=1),
            "auction_gap_iqr": available.quantile(0.75, axis=1)
            - available.quantile(0.25, axis=1),
            "auction_large_positive_gap_breadth": available.gt(threshold).sum(axis=1).div(count),
            "auction_large_negative_gap_breadth": available.lt(-threshold).sum(axis=1).div(count),
            "prior_session_intraday_up_breadth": prior_available.gt(0.0).sum(axis=1).div(prior_count),
        }
    )
    ordered = list(config["dayGate"]["featureColumns"])
    if list(features) != ordered:
        raise ValueError("V6 day feature set changed from preregistration")
    observed_outcome = outcome.where(opening_mask & complete)
    outcome_count = observed_outcome.notna().sum(axis=1).replace(0, np.nan)
    label = observed_outcome.gt(0.0).sum(axis=1).div(outcome_count).gt(0.5).astype(float)
    minimum = int(config["dayGate"]["minimumCrossSectionRows"])
    feature_valid = count.ge(minimum) & features.notna().all(axis=1)
    label_valid = feature_valid & outcome_count.ge(minimum)
    return features.where(feature_valid), label.where(label_valid)


def fit_day_gate(
    features: pd.DataFrame,
    label: pd.Series,
    base_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = config["dayGate"]
    base = features.loc[base_dates].copy()
    y_base = label.loc[base_dates]
    base_valid = y_base.notna() & base.notna().any(axis=1)
    base, y_base = base.loc[base_valid], y_base.loc[base_valid].astype(int)
    if len(base) < 100 or y_base.nunique() < 2:
        return {}, {"enabled": False, "reason": "insufficient_base_days", "baseDays": len(base)}
    medians = base.median(axis=0)
    scaler = StandardScaler().fit(base.fillna(medians))
    model = LogisticRegression(
        C=float(spec["regularizationC"]),
        max_iter=int(spec["maximumIterations"]),
        solver="lbfgs",
    ).fit(scaler.transform(base.fillna(medians)), y_base)
    calibration = features.loc[calibration_dates].copy()
    y_calibration = label.loc[calibration_dates]
    calibration_valid = y_calibration.notna() & calibration.notna().any(axis=1)
    calibration = calibration.loc[calibration_valid]
    y_calibration = y_calibration.loc[calibration_valid].astype(int)
    if calibration.empty:
        return {}, {"enabled": False, "reason": "no_calibration_days", "baseDays": len(base)}
    raw = model.predict_proba(scaler.transform(calibration.fillna(medians)))[:, 1]
    cutoff = float(np.median(raw))
    selected = raw >= cutoff
    selected_days = int(selected.sum())
    wins = int(y_calibration.to_numpy()[selected].sum())
    lower = v4.wilson_lower_bound(
        wins, selected_days, float(spec["oneSidedNormalCriticalValue"])
    )
    enabled = bool(
        selected_days >= int(spec["minimumCalibrationSelectedDays"])
        and lower > float(spec["minimumCalibrationWinRateLowerBound"])
    )
    audit = {
        "enabled": enabled,
        "reason": "reliable_high_opportunity_half" if enabled else "calibration_lower_bound_failed",
        "baseDays": int(len(base)),
        "calibrationDays": int(len(calibration)),
        "calibrationSelectedDays": selected_days,
        "calibrationWins": wins,
        "calibrationWinRate": wins / selected_days if selected_days else None,
        "oneSidedLowerBound": lower,
        "rawProbabilityMedianCutoff": cutoff,
    }
    return {
        "model": model,
        "scaler": scaler,
        "medians": medians,
        "cutoff": cutoff,
        "enabled": enabled,
    }, audit


def predict_day_gate(
    fitted: dict[str, Any], features: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    eligible = pd.Series(False, index=dates)
    probability = pd.Series(np.nan, index=dates, dtype=float)
    if not fitted or not fitted.get("enabled"):
        return eligible, probability
    frame = features.loc[dates].copy()
    valid = frame.notna().any(axis=1)
    if valid.any():
        x = fitted["scaler"].transform(frame.loc[valid].fillna(fitted["medians"]))
        raw = fitted["model"].predict_proba(x)[:, 1]
        probability.loc[valid] = raw
        eligible.loc[valid] = raw >= float(fitted["cutoff"])
    return eligible, probability


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    v5_path = ROOT / config["sourceExpandedPitConfig"]
    v5_config = load_json(v5_path)
    v5.validate_config(v5_config)
    v4_path = ROOT / v5_config["sourceSelectiveWinConfig"]
    v4_config = load_json(v4_path)
    v4.validate_config(v4_config)
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
    day_features, day_label = build_day_table(
        panel, opening, expanded_complete, outcome, config
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
    blocks: list[dict[str, Any]] = []
    day_probabilities = pd.Series(np.nan, index=all_dates, dtype=float)
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
        gate, gate_audit = fit_day_gate(day_features, day_label, base, calibration, config)
        block_score, block_support, _predictions, prediction_audit = v4.predict_dates(
            heads, reliability, features, test, v4_config
        )
        eligible_days, probabilities = predict_day_gate(gate, day_features, test)
        day_probabilities.loc[test] = probabilities
        block_score.loc[test[~eligible_days.to_numpy()]] = np.nan
        maximum = int(v4_config["selectiveCalibration"]["maximumSelectionsPerDay"])
        selected = v3.capped_selection_score(block_score.loc[test], maximum)
        matched = v3.matched_control_score(control_score.loc[test], block_support.loc[test], selected, maximum)
        selected_score.loc[test] = selected
        support_score.loc[test] = block_support.loc[test]
        candidate_metric = v3.variable_metrics(outcome, selected, test, v4_config)
        control_metric = v3.variable_metrics(outcome, matched, test, v4_config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": fit_audit,
                "dayGate": gate_audit,
                "eligibleTestDays": int(eligible_days.sum()),
                "prediction": {key: value for key, value in prediction_audit.items() if key != "days"},
                "candidate": candidate_metric,
                "matchedCoverageControl": control_metric,
                "comparison": v4.comparison(candidate_metric, control_metric),
            }
        )
        print(f"v6_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[support_score.loc[evaluation_dates].notna().any(axis=1)]
    maximum = int(v4_config["selectiveCalibration"]["maximumSelectionsPerDay"])
    matched_score = v3.matched_control_score(
        control_score.loc[scored_dates], support_score.loc[scored_dates], selected_score.loc[scored_dates], maximum
    )
    candidate = v3.variable_metrics(outcome, selected_score, scored_dates, v4_config)
    matched = v3.variable_metrics(outcome, matched_score, scored_dates, v4_config)
    checks = v4.acceptance(candidate, matched, blocks, v4_config)
    enabled_blocks = sum(bool(block["dayGate"].get("enabled")) for block in blocks)
    checks["minimumDayGateEnabledBlocks"] = enabled_blocks >= int(
        config["dayGate"]["minimumEnabledWalkForwardBlocks"]
    )
    passed = bool(all(checks.values()))
    latest_date = scored_dates.max()
    latest = selected_score.loc[latest_date].dropna().sort_values(ascending=False).head(maximum)
    latest_rows = [
        {
            "rank": rank,
            "signalDate": latest_date.date().isoformat(),
            "securityId": security,
            "rawProbabilityUpScore": float(latest.loc[security]),
            "dayGateRawProbability": float(day_probabilities.loc[latest_date]),
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
        "sourceV5ConfigSha256": digest(v5_path),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "evaluationRange": [scored_dates.min().date().isoformat(), scored_dates.max().date().isoformat()],
        "symbolUniverse": int(panel["close"].shape[1]),
        "stockFeatureCount": len(features),
        "dayFeatureCount": len(day_features.columns),
        "candidate": candidate,
        "matchedCoverageControl": matched,
        "dayGateEnabledBlocks": enabled_blocks,
        "checks": checks,
        "passed": passed,
        "decision": "fresh_forward_preregistration_only" if passed else "reject_auction_market_breadth_gate_v6",
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
        "# Opening-auction market breadth gate V6",
        "",
        "> Research-only. No trading integration or automatic promotion.",
        "",
        f"- run: `{run_id}`",
        f"- decision: `{result['decision']}`",
        f"- enabled day-gate blocks: {enabled_blocks}/{len(blocks)}",
        f"- active days: {candidate.get('activeTradingDays', 0)}/{candidate.get('evaluationDays', 0)}",
        "",
        "| cohort | win rate | mean gross | net @ 30bp | severe loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("V6", candidate), ("matched control", matched)):
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
