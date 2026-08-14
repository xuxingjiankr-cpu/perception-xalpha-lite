#!/usr/bin/env python3
"""Research-only opening-auction amplification of the frozen stock Top10 rank."""

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
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_perception_xalpha_twelve_factor_utility_weights_v6 as names_v6  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as win_capture  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_auction_signal_amplification_v1.json"
SCHEMA_VERSION = "stock_auction_signal_amplification_result_v1"
CODE_VERSION = "stock_auction_signal_amplification_v1_20260814"


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
    if config.get("schemaVersion") != "stock_auction_signal_amplification_v1":
        raise ValueError("unexpected auction-amplification schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("auction study must remain research/shadow-only")
    if config["timing"].get("dailyOpenIsNotExecutableAuctionFill") is not True:
        raise ValueError("auction-fill limitation must remain explicit")
    training = config["training"]
    if int(training["purgeTradingDays"]) < 1:
        raise ValueError("positive purge is required")
    if config["features"].get("futureDailyHighLowCloseVolumeForbidden") is not True:
        raise ValueError("same-session future features must remain forbidden")
    if config["evaluation"].get("sameAuctionTimeSupportForControlAndCandidates") is not True:
        raise ValueError("control and challengers must use identical support")
    if config["evaluation"].get("unresolvedSelectedNameMayBeReplaced") is not False:
        raise ValueError("unresolved selections cannot be replaced")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def board_limits(columns: pd.Index, config: dict[str, Any]) -> pd.Series:
    limits = config["boardLimits"]
    values: list[float] = []
    for security_id in map(str, columns):
        exchange, code = security_id.split(".", 1)
        if exchange == "SH" and code.startswith(("688", "689")):
            board = "STAR"
        elif exchange == "SZ" and code.startswith(("300", "301")):
            board = "CHINEXT"
        elif exchange == "SH":
            board = "SH_MAIN"
        elif exchange == "SZ":
            board = "SZ_MAIN"
        else:
            values.append(np.nan)
            continue
        values.append(float(limits[board]))
    return pd.Series(values, index=columns, dtype=float)


def opening_time_mask(
    panel: dict[str, pd.DataFrame], universe: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    """09:25 support without same-session high/low/close/volume/amount."""
    close, open_price, amount = panel["close"], panel["open"], panel["amount"]
    window = int(universe.get("pointInTimeAmountWindow", 60))
    seasoning = int(universe.get("pointInTimeMinimumHistory", 120))
    floor = float(
        universe.get(
            "pointInTimeMinimumAmount", universe.get("minimumMedianDailyAmountCny", 0.0)
        )
    )
    trailing_amount = amount.rolling(window, min_periods=max(5, window // 3)).median().shift(1)
    observed_before_open = close.shift(1).notna() & close.shift(1).gt(0.0)
    seasoned = close.notna().cumsum().shift(1).ge(seasoning)
    membership = panel.get("membership", observed_before_open).fillna(False).astype(bool)
    status = panel.get("trade_status")
    status_ok = (
        status.fillna(0.0).eq(1.0)
        if status is not None
        else open_price.notna() & open_price.gt(0.0)
    )
    is_st = panel.get("is_st")
    not_st = (
        is_st.fillna(1.0).eq(0.0)
        if is_st is not None
        else pd.DataFrame(True, index=close.index, columns=close.columns)
    )
    gap = open_price.div(close.shift(1).replace(0.0, np.nan)) - 1.0
    limit = board_limits(close.columns, config)
    not_locked_limit_up = gap.lt(limit - 0.002, axis=1)
    return (
        trailing_amount.ge(floor)
        & observed_before_open
        & seasoned
        & membership
        & status_ok
        & not_st
        & open_price.notna()
        & open_price.gt(0.0)
        & not_locked_limit_up
    ).fillna(False)


def row_robust_z(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    values = frame.where(mask)
    median = values.median(axis=1)
    deviation = values.sub(median, axis=0).abs().median(axis=1)
    scale = (1.4826 * deviation).replace(0.0, np.nan)
    return values.sub(median, axis=0).div(scale, axis=0).clip(-3.0, 3.0).div(3.0)


def build_execution_features(
    panel: dict[str, pd.DataFrame],
    prior_score: pd.DataFrame,
    opening_mask: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.DataFrame, pd.DataFrame]:
    close, open_price = panel["close"], panel["open"]
    prior = prior_score.shift(1).where(opening_mask)
    prior_rank = prior.rank(axis=1, pct=True)
    gap = open_price.div(close.shift(1).replace(0.0, np.nan)) - 1.0
    gap_z = row_robust_z(gap, opening_mask)
    abs_gap_rank = gap.abs().where(opening_mask).rank(axis=1, pct=True)
    prior_intraday = (close.div(open_price.replace(0.0, np.nan)) - 1.0).shift(1)
    prior_volatility = panel["returns"].rolling(20, min_periods=10).std().shift(1)
    prior_liquidity = panel["amount"].rolling(20, min_periods=10).median().shift(1)
    raw = {
        "prior_close_rank": 2.0 * prior_rank - 1.0,
        "overnight_gap_robust_z": gap_z,
        "overnight_gap_abs_rank": 2.0 * abs_gap_rank - 1.0,
        "prior_intraday_rank": 2.0 * prior_intraday.where(opening_mask).rank(axis=1, pct=True) - 1.0,
        "prior_volatility_rank": 2.0 * prior_volatility.where(opening_mask).rank(axis=1, pct=True) - 1.0,
        "prior_liquidity_rank": 2.0 * prior_liquidity.where(opening_mask).rank(axis=1, pct=True) - 1.0,
        "prior_score_x_gap": (2.0 * prior_rank - 1.0) * gap_z,
    }
    ordered = list(config["features"]["ordered"])
    if set(raw) != set(ordered):
        raise ValueError("auction feature set changed from preregistration")
    complete = opening_mask & prior.notna()
    for frame in raw.values():
        complete &= frame.notna()
    features = {key: raw[key].where(complete) for key in ordered}
    market_gap = gap.where(complete).median(axis=1)
    dispersion = gap.where(complete).sub(market_gap, axis=0).abs().median(axis=1)
    history = int(config["training"]["dispersionHistoryTradingDays"])
    threshold = dispersion.rolling(history, min_periods=max(20, history // 3)).median().shift(1)
    stressed = dispersion.gt(threshold)
    route = pd.Series(index=close.index, dtype="object")
    route.loc[market_gap.lt(0.0) & ~stressed] = "negative_gap_quiet"
    route.loc[market_gap.lt(0.0) & stressed] = "negative_gap_stressed"
    route.loc[market_gap.ge(0.0) & ~stressed] = "nonnegative_gap_quiet"
    route.loc[market_gap.ge(0.0) & stressed] = "nonnegative_gap_stressed"
    outcome = close.div(open_price.replace(0.0, np.nan)) - 1.0
    outcome = outcome.where(close.notna() & open_price.notna())
    return features, route, outcome, complete


def winner_pairs(
    features: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    keys = list(features)
    spec = config["training"]
    differences: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used = 0
    for date in dates:
        y = outcome.loc[date].to_numpy(dtype=float)
        x = np.column_stack([features[key].loc[date].to_numpy(dtype=float) for key in keys])
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(valid.sum()) < int(spec["minimumCrossSectionRows"]):
            continue
        xv, yv = x[valid], y[valid]
        positive = np.flatnonzero(yv > 0.0)
        nonpositive = np.flatnonzero(yv <= 0.0)
        winner_count = int(spec["winnerCount"])
        near_count = int(spec["nearZeroNonPositiveCount"])
        worst_count = int(spec["worstLossCount"])
        if len(positive) < winner_count or len(nonpositive) < near_count + worst_count:
            continue
        positive = positive[np.argsort(yv[positive], kind="stable")[::-1][:winner_count]]
        ordered_negative = nonpositive[np.argsort(yv[nonpositive], kind="stable")[::-1]]
        negative = np.unique(
            np.concatenate([ordered_negative[:near_count], ordered_negative[-worst_count:]])
        )
        day = (xv[positive][:, None, :] - xv[negative][None, :, :]).reshape(-1, len(keys))
        day_weight = np.repeat(np.linspace(2.0, 1.0, len(positive)), len(negative))
        day_weight /= day_weight.sum()
        differences.append(day)
        weights.append(day_weight)
        used += 1
    if not differences:
        raise RuntimeError("no auction winner pairs met the frozen training requirements")
    return np.concatenate(differences), np.concatenate(weights) / used, {
        "usedTradingDays": used,
        "pairCount": int(sum(len(item) for item in differences)),
    }


def fit_coefficients(
    features: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    differences, sample_weight, audit = winner_pairs(features, outcome, dates, config)
    spec = config["training"]
    prior = np.zeros(differences.shape[1], dtype=float)
    prior[0] = 1.0
    scale = float(spec["pairwiseMarginScale"])
    l2 = float(spec["l2ShrinkageToPrior"])

    def objective(coef: np.ndarray) -> float:
        margin = scale * (differences @ coef)
        return float(np.sum(np.logaddexp(0.0, -margin) * sample_weight)) + l2 * float(
            np.square(coef - prior).sum()
        )

    def gradient(coef: np.ndarray) -> np.ndarray:
        margin = scale * (differences @ coef)
        inverse = np.exp(-np.logaddexp(0.0, margin))
        return -scale * (differences.T @ (inverse * sample_weight)) + 2.0 * l2 * (
            coef - prior
        )

    result = minimize(objective, prior, jac=gradient, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"auction pairwise fit failed closed: {result.message}")
    audit.update({"success": True, "objective": float(result.fun), "iterations": int(result.nit)})
    return np.asarray(result.x, dtype=float), audit


def linear_score(features: dict[str, pd.DataFrame], coefficient: np.ndarray) -> pd.DataFrame:
    result = next(iter(features.values())) * 0.0
    for position, frame in enumerate(features.values()):
        result = result.add(frame * float(coefficient[position]))
    return result


def metrics(
    outcome: pd.DataFrame,
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    top_count = int(config["evaluation"]["topCount"])
    minimum = int(config["training"]["minimumCrossSectionRows"])
    rows: list[np.ndarray] = []
    percentiles: list[np.ndarray] = []
    daily: list[float] = []
    intended = unresolved = 0
    majority: list[bool] = []
    for date in dates.intersection(score.index):
        available_score = score.loc[date].dropna()
        if len(available_score) < minimum:
            continue
        chosen = list(available_score.nlargest(top_count).index)
        intended += top_count
        market = outcome.loc[date].dropna()
        selected = outcome.loc[date].reindex(chosen).dropna()
        unresolved += top_count - len(selected)
        if selected.empty or len(market) < minimum:
            continue
        realized = selected.to_numpy(dtype=float)
        rows.append(realized)
        percentiles.append(market.rank(pct=True).loc[selected.index].to_numpy(dtype=float))
        daily.append(float(realized.mean()))
        majority.append(bool(np.mean(realized > 0.0) > 0.5))
    if not rows:
        return {"tradingDays": 0, "selectedRows": 0}
    realized = np.concatenate(rows)
    pct = np.concatenate(percentiles)
    cost = float(config["evaluation"]["roundTripCost"])
    severe = float(config["evaluation"]["severeLossThreshold"])
    return {
        "tradingDays": len(daily),
        "intendedSelectedRows": intended,
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


def checks(candidate: dict[str, Any], control: dict[str, Any], blocks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, bool]:
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
    source_path = ROOT / config["sourceWinCaptureConfig"]
    source_config = load_json(source_path)
    data = win_capture.build_inputs(source_config)
    panel = data["panel"]
    opening = opening_time_mask(panel, data["baseConfig"]["assetUniverse"], config)
    features, route, outcome, complete = build_execution_features(
        panel, data["baselineScore"], opening, config
    )
    control_score = features["prior_close_rank"].where(complete)
    partitions = discrimination.fixed_partitions(panel["close"].index, data["modelConfig"])
    evaluation_dates = pd.DatetimeIndex(
        sorted(set(partitions["audit"]) | set(partitions["validation"]) | set(partitions["shadow"]))
    )
    all_dates = panel["close"].index
    spec = config["training"]
    refit, lookback, purge = (
        int(spec["refitEveryTradingDays"]),
        int(spec["lookbackTradingDays"]),
        int(spec["purgeTradingDays"]),
    )
    global_score = pd.DataFrame(np.nan, index=all_dates, columns=panel["close"].columns)
    routed_score = global_score.copy()
    blocks: dict[str, list[dict[str, Any]]] = {name: [] for name in config["evaluation"]["candidateOrder"]}
    latest_coefficients: dict[str, Any] = {}
    for start in range(0, len(evaluation_dates), refit):
        test = evaluation_dates[start : start + refit]
        position = all_dates.get_loc(test[0])
        train_end = position - purge
        train = all_dates[max(0, train_end - lookback) : train_end]
        if len(train) < lookback:
            continue
        coefficient, fit = fit_coefficients(features, outcome, train, config)
        block_global = linear_score(features, coefficient)
        global_score.loc[test] = block_global.loc[test]
        routed_block = block_global.copy()
        route_fit: dict[str, Any] = {}
        for label in config["routes"]:
            route_dates = train[route.reindex(train).eq(label).to_numpy()]
            if len(route_dates) < int(spec["minimumRouteTradingDays"]):
                route_fit[label] = {"status": "global_fallback", "tradingDays": len(route_dates)}
                continue
            local, local_audit = fit_coefficients(features, outcome, route_dates, config)
            weight = float(spec["routeCoefficientWeight"])
            combined = (1.0 - weight) * coefficient + weight * local
            test_dates = test[route.reindex(test).eq(label).to_numpy()]
            if len(test_dates):
                routed_block.loc[test_dates] = linear_score(features, combined).loc[test_dates]
            route_fit[label] = {
                "status": "fitted",
                "tradingDays": len(route_dates),
                "fit": local_audit,
                "combinedCoefficients": combined.tolist(),
            }
        routed_score.loc[test] = routed_block.loc[test]
        control_metric = metrics(outcome, control_score, test, config)
        for name, score in (
            ("auction_pairwise_global", block_global),
            ("auction_pairwise_routed", routed_block),
        ):
            candidate_metric = metrics(outcome, score, test, config)
            blocks[name].append(
                {
                    "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                    "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                    "candidate": candidate_metric,
                    "control": control_metric,
                    "winImproved": candidate_metric.get("grossUpRate", -np.inf) > control_metric.get("grossUpRate", np.inf),
                    "grossImproved": candidate_metric.get("meanGrossReturn", -np.inf) > control_metric.get("meanGrossReturn", np.inf),
                }
            )
        latest_coefficients = {
            "global": dict(zip(config["features"]["ordered"], coefficient.tolist())),
            "globalFit": fit,
            "routes": route_fit,
        }
        print(f"auction_block_complete {test.min().date()}..{test.max().date()}", flush=True)
    scored_dates = evaluation_dates[global_score.loc[evaluation_dates].notna().any(axis=1)]
    control_metric = metrics(outcome, control_score, scored_dates, config)
    candidates: dict[str, Any] = {}
    for name, score in (
        ("auction_pairwise_global", global_score),
        ("auction_pairwise_routed", routed_score),
    ):
        candidate_metric = metrics(outcome, score, scored_dates, config)
        candidate_checks = checks(candidate_metric, control_metric, blocks[name], config)
        candidates[name] = {
            "metrics": candidate_metric,
            "checks": candidate_checks,
            "passed": bool(all(candidate_checks.values())),
            "walkForwardBlocks": blocks[name],
        }
    passed = [name for name, value in candidates.items() if value["passed"]]
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
        "timing": config["timing"],
        "featureOrder": config["features"]["ordered"],
        "control": control_metric,
        "candidates": candidates,
        "latestCoefficients": latest_coefficients,
        "passingCandidates": passed,
        "decision": "historical_feasibility_pass_requires_real_auction_forward_study" if passed else "reject_auction_amplification_challengers",
        "dailyBarExecutionCaveat": "The auction open cannot be guaranteed after observing it; first-minute and order-book data are required before any execution claim.",
        "historicalWindowsAlreadyViewed": True,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    root = ROOT / config["output"]["root"] / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "result.json").write_text(json.dumps(safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Opening-auction signal amplification V1",
        "",
        "> Research-only feasibility. Daily open is not a guaranteed executable fill.",
        "",
        f"- run: `{run_id}`",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- decision: `{result['decision']}`",
        "",
        "| model | win rate | mean gross | mean net @ 30bp | percentile | severe loss | passed |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| auction-time prior-close control | {control_metric['grossUpRate']:.2%} | {control_metric['meanGrossReturn']:.4%} | {control_metric['meanNetReturnAtConfiguredFriction']:.4%} | {control_metric['meanReturnPercentile']:.2%} | {control_metric['severeLossRate']:.2%} | reference |",
    ]
    for name, value in candidates.items():
        item = value["metrics"]
        lines.append(
            f"| {name} | {item['grossUpRate']:.2%} | {item['meanGrossReturn']:.4%} | {item['meanNetReturnAtConfiguredFriction']:.4%} | {item['meanReturnPercentile']:.2%} | {item['severeLossRate']:.2%} | {'yes' if value['passed'] else 'no'} |"
        )
    lines.extend(["", result["dailyBarExecutionCaveat"], ""])
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
