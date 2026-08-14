#!/usr/bin/env python3
"""One preregistered multi-task correction for auction winner-ranker tail risk."""

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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_stock_auction_signal_amplification_v1 as auction_v1  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_multitask_utility_v2.json"
SCHEMA_VERSION = "stock_auction_multitask_utility_result_v2"
CODE_VERSION = "stock_auction_multitask_utility_v2_20260814"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_auction_multitask_utility_v2":
        raise ValueError("unexpected auction multitask schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("auction multitask study must remain research-only")
    if config["failureBeingCorrected"].get("singleNewCandidate") is not True:
        raise ValueError("V2 must contain exactly one correction candidate")
    if config["timing"].get("sameDayHighLowCloseVolumeAmountForbiddenFromFeatures") is not True:
        raise ValueError("same-session future fields must remain forbidden")
    utility = config["utility"]
    total = sum(float(value) for key, value in utility.items() if key.endswith("RankWeight"))
    if abs(total - 1.0) > 1e-12 or utility.get("weightsFrozenBeforeRun") is not True:
        raise ValueError("frozen utility weights must sum to one")
    if int(config["training"]["outerPurgeTradingDays"]) < 1:
        raise ValueError("positive outer purge is required")
    if config["evaluation"].get("sameAuctionTimeSupportAsControl") is not True:
        raise ValueError("candidate and control support must match")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_features(
    base: dict[str, pd.DataFrame], route: pd.Series, complete: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    result = dict(base)
    index, columns = complete.index, complete.columns
    labels = [
        "negative_gap_quiet",
        "negative_gap_stressed",
        "nonnegative_gap_quiet",
        "nonnegative_gap_stressed",
    ]
    for label in labels:
        scalar = route.eq(label).astype(float).reindex(index)
        result[f"market_route_{label}"] = pd.DataFrame(
            np.repeat(scalar.to_numpy()[:, None], len(columns), axis=1),
            index=index,
            columns=columns,
        ).where(complete)
    gap = base["overnight_gap_robust_z"]
    result["negative_gap_reversal"] = (-gap).clip(lower=0.0).where(complete)
    result["positive_gap_momentum"] = gap.clip(lower=0.0).where(complete)
    return result


def sampled_table(
    features: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    maximum = int(config["training"]["maximumRowsPerTradingDay"])
    rng = np.random.default_rng(int(config["training"]["randomSeed"]) + seed_offset)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    used = 0
    for date in dates:
        y = outcome.loc[date].to_numpy(dtype=float)
        x = np.column_stack([frame.loc[date].to_numpy(dtype=float) for frame in features.values()])
        valid = np.flatnonzero(np.isfinite(y) & np.isfinite(x).all(axis=1))
        if len(valid) < 100:
            continue
        if len(valid) > maximum:
            valid = np.sort(rng.choice(valid, size=maximum, replace=False))
        xs.append(x[valid])
        ys.append(y[valid])
        ws.append(np.full(len(valid), 1.0 / len(valid), dtype=float))
        used += 1
    if not xs:
        raise RuntimeError("no sampled auction rows")
    weights = np.concatenate(ws) / used
    return np.concatenate(xs), np.concatenate(ys), weights, {
        "tradingDays": used,
        "rows": int(sum(len(value) for value in ys)),
        "maximumRowsPerDay": maximum,
    }


def classifier(config: dict[str, Any]) -> HistGradientBoostingClassifier:
    spec = config["model"]
    return HistGradientBoostingClassifier(
        learning_rate=float(spec["learningRate"]),
        max_iter=int(spec["maxIter"]),
        max_leaf_nodes=int(spec["maxLeafNodes"]),
        min_samples_leaf=int(spec["minSamplesLeaf"]),
        l2_regularization=float(spec["l2Regularization"]),
        random_state=int(config["training"]["randomSeed"]),
    )


def regressor(config: dict[str, Any]) -> HistGradientBoostingRegressor:
    spec = config["model"]
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(spec["learningRate"]),
        max_iter=int(spec["maxIter"]),
        max_leaf_nodes=int(spec["maxLeafNodes"]),
        min_samples_leaf=int(spec["minSamplesLeaf"]),
        l2_regularization=float(spec["l2Regularization"]),
        random_state=int(config["training"]["randomSeed"]),
    )


def probability_calibrator(raw: np.ndarray, label: np.ndarray, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    events = int(label.sum())
    minimum = int(config["training"]["minimumCalibrationEvents"])
    if events < minimum or len(label) - events < minimum or len(np.unique(raw)) < 5:
        prior = float(label.mean())
        return lambda value: np.full(len(value), prior), {"kind": "constant_prior", "prior": prior, "events": events}
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, label)
    return model.predict, {"kind": "isotonic", "events": events}


def fit_heads(
    features: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    train: pd.DatetimeIndex,
    config: dict[str, Any],
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_days = int(config["training"]["calibrationTradingDays"])
    inner_purge = int(config["training"]["innerPurgeTradingDays"])
    calibration = train[-calibration_days:]
    base = train[: -calibration_days - inner_purge]
    x_base, y_base, w_base, base_audit = sampled_table(features, outcome, base, config, seed_offset)
    x_cal, y_cal, _w_cal, cal_audit = sampled_table(features, outcome, calibration, config, seed_offset + 1000)
    low, high = np.quantile(y_base, config["training"]["winsorQuantiles"])
    y_winsor = np.clip(y_base, low, high)
    up = classifier(config).fit(x_base, y_base > 0.0, sample_weight=w_base)
    tail_threshold = float(config["model"]["severeLossThreshold"])
    tail = classifier(config).fit(x_base, y_base <= tail_threshold, sample_weight=w_base)
    ret = regressor(config).fit(x_base, y_winsor, sample_weight=w_base)
    raw_up = up.predict_proba(x_cal)[:, 1]
    raw_tail = tail.predict_proba(x_cal)[:, 1]
    raw_ret = ret.predict(x_cal)
    up_cal, up_audit = probability_calibrator(raw_up, (y_cal > 0.0).astype(float), config)
    tail_cal, tail_audit = probability_calibrator(raw_tail, (y_cal <= tail_threshold).astype(float), config)
    ridge = Ridge(alpha=10.0).fit(raw_ret.reshape(-1, 1), y_cal)
    if float(ridge.coef_[0]) <= 0.0:
        mean_return = float(y_cal.mean())
        ret_cal = lambda value: np.full(len(value), mean_return)
        ret_audit = {"kind": "constant_mean", "mean": mean_return, "slope": float(ridge.coef_[0])}
    else:
        ret_cal = lambda value: ridge.predict(np.asarray(value).reshape(-1, 1))
        ret_audit = {"kind": "positive_slope_ridge", "slope": float(ridge.coef_[0]), "intercept": float(ridge.intercept_)}
    calibrated_up = np.clip(up_cal(raw_up), 1e-6, 1.0 - 1e-6)
    calibrated_tail = np.clip(tail_cal(raw_tail), 1e-6, 1.0 - 1e-6)
    audit = {
        "base": base_audit,
        "calibration": cal_audit,
        "baseRange": [base.min().date().isoformat(), base.max().date().isoformat()],
        "calibrationRange": [calibration.min().date().isoformat(), calibration.max().date().isoformat()],
        "innerPurgeTradingDays": inner_purge,
        "upCalibration": up_audit,
        "tailCalibration": tail_audit,
        "returnCalibration": ret_audit,
        "calibrationMetrics": {
            "upAuc": float(roc_auc_score(y_cal > 0.0, raw_up)),
            "upBrier": float(brier_score_loss(y_cal > 0.0, calibrated_up)),
            "upLogLoss": float(log_loss(y_cal > 0.0, calibrated_up)),
            "tailAuc": float(roc_auc_score(y_cal <= tail_threshold, raw_tail)),
            "tailBrier": float(brier_score_loss(y_cal <= tail_threshold, calibrated_tail)),
        },
    }
    return {
        "upModel": up,
        "tailModel": tail,
        "returnModel": ret,
        "upCalibrator": up_cal,
        "tailCalibrator": tail_cal,
        "returnCalibrator": ret_cal,
    }, audit


def score_dates(
    heads: dict[str, Any],
    features: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    template = next(iter(features.values()))
    score = pd.DataFrame(np.nan, index=template.index, columns=template.columns)
    prediction_rows = 0
    for date in dates:
        x = np.column_stack([frame.loc[date].to_numpy(dtype=float) for frame in features.values()])
        valid = np.flatnonzero(np.isfinite(x).all(axis=1))
        if len(valid) < 100:
            continue
        xv = x[valid]
        p_up = np.clip(heads["upCalibrator"](heads["upModel"].predict_proba(xv)[:, 1]), 0.0, 1.0)
        p_tail = np.clip(heads["tailCalibrator"](heads["tailModel"].predict_proba(xv)[:, 1]), 0.0, 1.0)
        expected = heads["returnCalibrator"](heads["returnModel"].predict(xv))
        frame = pd.DataFrame({"p_up": p_up, "expected": expected, "tail_safety": 1.0 - p_tail}, index=template.columns[valid])
        ranks = frame.rank(pct=True, method="average")
        utility = config["utility"]
        score.loc[date, template.columns[valid]] = (
            float(utility["probabilityUpRankWeight"]) * ranks["p_up"]
            + float(utility["expectedReturnRankWeight"]) * ranks["expected"]
            + float(utility["tailSafetyRankWeight"]) * ranks["tail_safety"]
        ).to_numpy()
        prediction_rows += len(valid)
    return score, {"predictionRows": prediction_rows, "tradingDays": len(dates)}


def acceptance(candidate: dict[str, Any], control: dict[str, Any], blocks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, bool]:
    spec = config["evaluation"]
    return {
        "winRateImprovesByFrozenMinimum": 100.0 * (candidate["grossUpRate"] - control["grossUpRate"])
        >= float(spec["minimumWinRateImprovementPercentagePoints"]),
        "meanGrossImprovesByFrozenMinimum": 10000.0 * (candidate["meanGrossReturn"] - control["meanGrossReturn"])
        >= float(spec["minimumMeanGrossImprovementBps"]),
        "meanPercentileImproved": candidate["meanReturnPercentile"] > control["meanReturnPercentile"],
        "severeLossNotWorse": candidate["severeLossRate"] <= control["severeLossRate"],
        "majorityBlocksImproveWinRate": sum(item["winImproved"] for item in blocks) > len(blocks) / 2.0,
        "majorityBlocksImproveGross": sum(item["grossImproved"] for item in blocks) > len(blocks) / 2.0,
    }


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    auction_config = load_json(ROOT / config["sourceAuctionConfig"])
    auction_v1.validate_config(auction_config)
    source_config = load_json(ROOT / config["sourceWinCaptureConfig"])
    data = win_capture.build_inputs(source_config)
    panel = data["panel"]
    opening = auction_v1.opening_time_mask(panel, data["baseConfig"]["assetUniverse"], auction_config)
    base_features, route, outcome, complete = auction_v1.build_execution_features(panel, data["baselineScore"], opening, auction_config)
    features = model_features(base_features, route, complete)
    control_score = base_features["prior_close_rank"].where(complete)
    partitions = discrimination.fixed_partitions(panel["close"].index, data["modelConfig"])
    evaluation_dates = pd.DatetimeIndex(sorted(set(partitions["audit"]) | set(partitions["validation"]) | set(partitions["shadow"])))
    all_dates = panel["close"].index
    spec = config["training"]
    lookback, purge, refit = int(spec["lookbackTradingDays"]), int(spec["outerPurgeTradingDays"]), int(spec["refitEveryTradingDays"])
    candidate_score = pd.DataFrame(np.nan, index=all_dates, columns=panel["close"].columns)
    blocks: list[dict[str, Any]] = []
    for block_no, start in enumerate(range(0, len(evaluation_dates), refit)):
        test = evaluation_dates[start : start + refit]
        position = all_dates.get_loc(test[0])
        train_end = position - purge
        train = all_dates[max(0, train_end - lookback) : train_end]
        if len(train) < lookback:
            continue
        heads, fit_audit = fit_heads(features, outcome, train, config, block_no * 10000)
        block_score, prediction_audit = score_dates(heads, features, test, config)
        candidate_score.loc[test] = block_score.loc[test]
        candidate_metric = auction_v1.metrics(outcome, block_score, test, auction_config)
        control_metric = auction_v1.metrics(outcome, control_score, test, auction_config)
        blocks.append({
            "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
            "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
            "fit": fit_audit,
            "prediction": prediction_audit,
            "candidate": candidate_metric,
            "control": control_metric,
            "winImproved": candidate_metric.get("grossUpRate", -np.inf) > control_metric.get("grossUpRate", np.inf),
            "grossImproved": candidate_metric.get("meanGrossReturn", -np.inf) > control_metric.get("meanGrossReturn", np.inf),
        })
        print(f"multitask_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[candidate_score.loc[evaluation_dates].notna().any(axis=1)]
    control = auction_v1.metrics(outcome, control_score, scored_dates, auction_config)
    candidate = auction_v1.metrics(outcome, candidate_score, scored_dates, auction_config)
    candidate_checks = acceptance(candidate, control, blocks, config)
    passed = bool(all(candidate_checks.values()))
    now = datetime.now().astimezone()
    run_id = run_id or f"run_{now:%Y%m%dT%H%M%S%z}"
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": config["status"],
        "runId": run_id,
        "generatedAt": now.isoformat(),
        "configSha256": digest(config_path),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "evaluationRange": [scored_dates.min().date().isoformat(), scored_dates.max().date().isoformat()],
        "symbolUniverse": int(panel["close"].shape[1]),
        "featureOrder": list(features),
        "control": control,
        "candidate": candidate,
        "checks": candidate_checks,
        "passed": passed,
        "walkForwardBlocks": blocks,
        "decision": "historical_feasibility_pass_requires_real_auction_forward_study" if passed else "reject_multitask_auction_utility",
        "historicalWindowsAlreadyViewed": True,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    root = ROOT / config["output"]["root"] / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "result.json").write_text(json.dumps(safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Opening-auction multi-task utility V2",
        "",
        "> Research-only. The observed daily open is not a guaranteed executable fill.",
        "",
        f"- run: `{run_id}`",
        f"- decision: `{result['decision']}`",
        "",
        "| model | win rate | mean gross | mean net @ 30bp | percentile | severe loss |",
        "|---|---:|---:|---:|---:|---:|",
        f"| auction-time prior-close control | {control['grossUpRate']:.2%} | {control['meanGrossReturn']:.4%} | {control['meanNetReturnAtConfiguredFriction']:.4%} | {control['meanReturnPercentile']:.2%} | {control['severeLossRate']:.2%} |",
        f"| multi-task utility | {candidate['grossUpRate']:.2%} | {candidate['meanGrossReturn']:.4%} | {candidate['meanNetReturnAtConfiguredFriction']:.4%} | {candidate['meanReturnPercentile']:.2%} | {candidate['severeLossRate']:.2%} |",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"runId": run_id, "decision": result["decision"], "output": str(root)}, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.run_id)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
