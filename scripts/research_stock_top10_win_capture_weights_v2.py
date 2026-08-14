#!/usr/bin/env python3
"""Audit accuracy-first Top10 weights and two preregistered factor replacements."""

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

import generate_guarded_weight_top10_forecast_v1 as forecast  # noqa: E402
import generate_sixteen_factor_interaction_top10_v1 as sixteen  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402
import research_sixteen_factor_walkforward_weights_v1 as prior_research  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import research_twelve_factor_fundamental_price_interactions_v1 as interaction_model  # noqa: E402
import research_twelve_factor_fundamental_second_stage_v5 as fundamental_stage  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "research" / "stock_top10_win_capture_weights_v2.json"
SCHEMA_VERSION = "stock_top10_win_capture_weights_result_v2"
CODE_VERSION = "stock_top10_win_capture_weights_v2_20260814"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if config.get("schemaVersion") != "stock_top10_win_capture_weights_v2":
        raise ValueError("unexpected win-capture schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("win-capture study must remain research/shadow-only")
    spec = config["training"]
    if int(spec["purgeTradingDays"]) < 1:
        raise ValueError("positive purge is required")
    if int(spec["lookbackTradingDays"]) < int(spec["refitEveryTradingDays"]):
        raise ValueError("lookback must exceed refit interval")
    if spec.get("historicalOutcomeMaySelectObjective") is not False:
        raise ValueError("historical results cannot choose the objective")
    if spec.get("validationOrShadowMayTuneHyperparameters") is not False:
        raise ValueError("evaluation periods cannot tune hyperparameters")
    if spec.get("sameDayLossMayChangeWeights") is not False:
        raise ValueError("same-day loss cannot change weights")
    outcome = config["outcome"]
    if outcome.get("definition") != "close_t_plus_1_div_open_t_plus_1_minus_one":
        raise ValueError("win-capture target must be next-session open-to-close")
    if outcome.get("futureTradabilityMayFilterSelectionBeforeRanking") is not False:
        raise ValueError("future tradability cannot filter the Top10 rank")
    if outcome.get("unresolvedSelectedNameMayBeReplaced") is not False:
        raise ValueError("unresolved selected names cannot be replaced")
    candidates = list(config["candidateOrder"])
    if candidates != list(config["factorSets"]):
        raise ValueError("factor candidates must have one fixed evaluation order")
    for name in candidates:
        factor = config["factorSets"][name]
        if abs(
            float(factor["priceBlockPrior"])
            + float(factor["directFundamentalBlockPrior"])
            + float(factor["interactionBlockPrior"])
            - 1.0
        ) > 1e-12:
            raise ValueError(f"{name} prior blocks do not sum to one")
        if not 0.0 < float(factor["nonPriceBlockMinimum"]) < float(
            factor["nonPriceBlockMaximum"]
        ) < 1.0:
            raise ValueError(f"{name} non-price bounds invalid")
    if config["evaluation"].get("reportHindsightAndTrailingOnIdenticalDates") is not True:
        raise ValueError("selection exposure comparison is required")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading permissions must remain false")


def next_session_intraday_return(
    panel: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label t after close with t+1 buyable-open to sellable-close return."""
    entry = panel["open"].shift(-1)
    exit_price = panel["close"].shift(-1)
    buyable, sellable = precision.autonomous.tradability_frames(panel)
    resolved = (
        panel["eligible"]
        & panel["eligible"].shift(-1).eq(True)
        & buyable.shift(-1).eq(True)
        & sellable.shift(-1).eq(True)
        & entry.notna()
        & exit_price.notna()
    )
    outcome = exit_price.div(entry.replace(0.0, np.nan)) - 1.0
    return outcome.where(resolved), resolved


def build_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source_weight_config = load_json(ROOT / config["sourceWeightConfig"])
    prior_research.validate_config(source_weight_config)
    ranking_path = ROOT / source_weight_config["sourceRankingConfig"]
    ranking_config = load_json(ranking_path)
    sixteen.validate_config(ranking_config)
    interaction_config = load_json(ROOT / ranking_config["sourceInteractionConfig"])
    interaction_model.validate_config(interaction_config)
    source_config = load_json(ROOT / interaction_config["sourceForecastConfig"])
    guarded_config = load_json(ROOT / source_config["guardedWeightsConfig"])
    frozen, _source, _hash = guarded.load_frozen_config(guarded_config)
    fundamental_config = load_json(ROOT / interaction_config["fundamentalMechanismConfig"])
    model_config = load_json(ROOT / interaction_config["modelTemplate"])
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, execution_eligible = next_session_intraday_return(panel)
    outcome = outcome.where(execution_eligible)
    price_ranks, _static, factor_audit = rolling.compute_rank_book(panel, frozen)
    daily_ic = rolling.factor_daily_ic(price_ranks, outcome)
    price_prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    adaptive_weights, _updates = guarded.guarded_weight_path(
        daily_ic, price_prior, guarded_config
    )
    completion = source_config["forecast"]["missingFeatureCompletion"]
    completed_price, _missing, price_support, completion_audit = forecast.complete_rank_book(
        price_ranks,
        panel["eligible"],
        int(interaction_config["data"]["maximumImputedPriceFactorsForEligibility"]),
        float(completion["allMissingCrossSectionFallbackRank"]),
    )
    family_ranks, fundamental_support, fundamental_audit = fundamental_stage.build_family_scores(
        panel, fundamental_config
    )
    market_ranks = interaction_model.build_market_context_ranks(panel, interaction_config)
    support = price_support & fundamental_support
    for frame in market_ranks.values():
        support &= frame.notna()
    interactions = interaction_model.build_interaction_ranks(
        family_ranks, market_ranks, interaction_config, support
    )
    for frame in interactions.values():
        support &= frame.notna()
    price_features = {key: value.where(support) for key, value in completed_price.items()}
    direct_features = {
        f"fundamental/{key}": value.where(support)
        for key, value in family_ranks.items()
    }
    interaction_features = {
        f"interaction/{key}": value.where(support)
        for key, value in interactions.items()
    }
    feature_sets = {
        "current_16_reweighted": {**price_features, **interaction_features},
        "replace_interactions_with_direct_fundamentals_16": {
            **price_features,
            **direct_features,
        },
        "price_fundamental_interaction_20": {
            **price_features,
            **direct_features,
            **interaction_features,
        },
    }
    twelve_score = guarded.adaptive_score(completed_price, adaptive_weights, panel)
    baseline_score, _interaction_mean = sixteen.combine_scores(
        twelve_score,
        interactions,
        support,
        float(ranking_config["ranking"]["twelveFactorBlockWeight"]),
        float(ranking_config["ranking"]["eachInteractionWeight"]),
    )
    return {
        "panel": panel,
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "completionAudit": completion_audit,
        "fundamentalAudit": fundamental_audit,
        "outcome": outcome.where(support),
        "featureSets": feature_sets,
        "baselineScore": baseline_score,
        "frozen": frozen,
        "modelConfig": model_config,
        "baseConfig": base,
        "rankingConfigPath": ranking_path,
    }


def prior_weights(
    frozen: dict[str, Any], features: dict[str, pd.DataFrame], spec: dict[str, Any]
) -> np.ndarray:
    price_lookup = {
        str(row["factorKey"]): float(row["weight"]) for row in frozen["frozenFactors"]
    }
    direct_keys = [key for key in features if key.startswith("fundamental/")]
    interaction_keys = [key for key in features if key.startswith("interaction/")]
    values: list[float] = []
    for key in features:
        if key.startswith("fundamental/"):
            values.append(float(spec["directFundamentalBlockPrior"]) / len(direct_keys))
        elif key.startswith("interaction/"):
            values.append(float(spec["interactionBlockPrior"]) / len(interaction_keys))
        else:
            values.append(float(spec["priceBlockPrior"]) * price_lookup[key])
    result = np.asarray(values, dtype=float)
    if abs(float(result.sum()) - 1.0) > 1e-8 or np.any(result <= 0.0):
        raise ValueError("candidate prior must be a strictly positive simplex")
    return result


def winner_pairs(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    training = config["training"]
    keys = list(features)
    differences: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    used_days = 0
    used_rows = 0
    winner_count = int(training["winnerCount"])
    near_count = int(training["nearZeroNonPositiveCount"])
    worst_count = int(training["worstLossCount"])
    for date in dates:
        y = returns.loc[date].to_numpy(dtype=float)
        x = np.column_stack(
            [features[key].loc[date].to_numpy(dtype=float) for key in keys]
        )
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(valid.sum()) < int(training["minimumCrossSectionRows"]):
            continue
        xv = x[valid]
        yv = y[valid]
        positive_order = np.flatnonzero(yv > 0.0)
        nonpositive_order = np.flatnonzero(yv <= 0.0)
        if len(positive_order) < winner_count or len(nonpositive_order) < near_count + worst_count:
            continue
        positive_order = positive_order[
            np.argsort(yv[positive_order], kind="stable")[::-1][:winner_count]
        ]
        nonpositive_sorted = nonpositive_order[
            np.argsort(yv[nonpositive_order], kind="stable")[::-1]
        ]
        negative_order = np.unique(
            np.concatenate(
                [nonpositive_sorted[:near_count], nonpositive_sorted[-worst_count:]]
            )
        )
        positive = xv[positive_order]
        negative = xv[negative_order]
        day = (positive[:, None, :] - negative[None, :, :]).reshape(-1, len(keys))
        winner_emphasis = np.linspace(2.0, 1.0, len(positive), dtype=float)
        day_weight = np.repeat(winner_emphasis, len(negative))
        day_weight /= day_weight.sum()
        differences.append(day)
        pair_weights.append(day_weight)
        used_days += 1
        used_rows += int(valid.sum())
    if not differences:
        raise RuntimeError("no training dates met winner-pair requirements")
    return (
        np.concatenate(differences),
        np.concatenate(pair_weights) / used_days,
        {
            "usedTradingDays": used_days,
            "usedRows": used_rows,
            "pairCount": int(sum(len(item) for item in differences)),
            "positiveDefinition": "top_positive_executable_returns",
            "negativeDefinition": "near_zero_nonpositive_plus_worst_losses",
        },
    )


def fit_weights(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    prior: np.ndarray,
    factor_spec: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    differences, sample_weight, audit = winner_pairs(features, returns, dates, config)
    training = config["training"]
    scale = float(training["pairwiseMarginScale"])
    l2 = float(training["l2ShrinkageToPrior"])

    def objective(weight: np.ndarray) -> float:
        margin = scale * (differences @ weight)
        return float(np.sum(np.logaddexp(0.0, -margin) * sample_weight)) + l2 * float(
            np.square(weight - prior).sum()
        )

    def gradient(weight: np.ndarray) -> np.ndarray:
        margin = scale * (differences @ weight)
        inverse = np.exp(-np.logaddexp(0.0, margin))
        return -scale * (differences.T @ (inverse * sample_weight)) + 2.0 * l2 * (
            weight - prior
        )

    lower = float(training["minimumFactorWeight"])
    upper = float(training["maximumFactorWeight"])
    nonprice = np.asarray(
        [
            key.startswith("fundamental/") or key.startswith("interaction/")
            for key in features
        ],
        dtype=bool,
    )
    minimum = float(factor_spec["nonPriceBlockMinimum"])
    maximum = float(factor_spec["nonPriceBlockMaximum"])
    result = minimize(
        objective,
        prior,
        jac=gradient,
        method="SLSQP",
        bounds=[(lower, upper)] * len(prior),
        constraints=[
            {"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            {"type": "ineq", "fun": lambda value: float(value[nonprice].sum() - minimum)},
            {"type": "ineq", "fun": lambda value: float(maximum - value[nonprice].sum())},
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"accuracy-first weight optimizer failed closed: {result.message}")
    weight = np.asarray(result.x, dtype=float)
    nonprice_weight = float(weight[nonprice].sum())
    if (
        abs(float(weight.sum()) - 1.0) > 1e-8
        or np.any(weight < lower - 1e-9)
        or np.any(weight > upper + 1e-9)
        or not minimum - 1e-9 <= nonprice_weight <= maximum + 1e-9
    ):
        raise RuntimeError("accuracy-first optimizer violated frozen constraints")
    audit.update(
        {
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "nonPriceBlockWeight": nonprice_weight,
            "success": True,
        }
    )
    return weight, audit


def weighted_score(features: dict[str, pd.DataFrame], weight: np.ndarray) -> pd.DataFrame:
    keys = list(features)
    result = features[keys[0]].astype(float) * float(weight[0])
    for position, key in enumerate(keys[1:], start=1):
        result = result.add(features[key].astype(float) * float(weight[position]))
    return result


def metrics(
    returns: pd.DataFrame,
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    top_count = int(config["evaluation"]["topCount"])
    minimum = int(config["training"]["minimumCrossSectionRows"])
    selected_returns: list[np.ndarray] = []
    selected_percentiles: list[np.ndarray] = []
    daily_gross: list[float] = []
    daily_best: list[float] = []
    daily_majority_up: list[bool] = []
    daily_true_top10_hit: list[bool] = []
    daily_overlap: list[float] = []
    daily_market_up: list[float] = []
    intended_rows = 0
    unresolved_rows = 0
    for date in dates.intersection(score.index):
        available_score = score.loc[date].dropna()
        if len(available_score) < max(top_count, minimum):
            continue
        chosen_index = list(available_score.nlargest(top_count).index)
        intended_rows += top_count
        market_returns = returns.loc[date].dropna()
        chosen_returns = returns.loc[date].reindex(chosen_index).dropna()
        unresolved_rows += top_count - len(chosen_returns)
        if chosen_returns.empty or len(market_returns) < minimum:
            continue
        values = pd.DataFrame({"return": market_returns})
        actual_top = set(values.nlargest(top_count, "return").index)
        chosen_set = set(chosen_returns.index)
        realized = chosen_returns.to_numpy(dtype=float)
        percentiles = values["return"].rank(pct=True, method="average").loc[chosen_returns.index]
        selected_returns.append(realized)
        selected_percentiles.append(percentiles.to_numpy(dtype=float))
        daily_gross.append(float(realized.mean()))
        daily_best.append(float(realized.max()))
        daily_majority_up.append(bool(np.mean(realized > 0.0) > 0.5))
        overlap = len(actual_top & chosen_set)
        daily_true_top10_hit.append(overlap > 0)
        daily_overlap.append(overlap / top_count)
        daily_market_up.append(float((market_returns > 0.0).mean()))
    if not selected_returns:
        return {"tradingDays": 0, "selectedRows": 0}
    realized = np.concatenate(selected_returns)
    percentiles = np.concatenate(selected_percentiles)
    cost = float(config["evaluation"]["roundTripCost"])
    severe = float(config["evaluation"]["severeLossThreshold"])
    top_decile = float(config["evaluation"]["topReturnDecileThreshold"])
    cumulative = np.cumprod(1.0 + np.asarray(daily_gross) - cost)
    running_max = np.maximum.accumulate(cumulative)
    up_rate = float(np.mean(realized > 0.0))
    return {
        "tradingDays": len(daily_gross),
        "intendedSelectedRows": intended_rows,
        "selectedRows": int(len(realized)),
        "unresolvedSelectedRows": unresolved_rows,
        "resolvedFraction": float(len(realized) / intended_rows) if intended_rows else None,
        "meanGrossReturn": float(realized.mean()),
        "meanNetReturn": float(realized.mean() - cost),
        "grossUpRate": up_rate,
        "marketGrossUpRate": float(np.mean(daily_market_up)),
        "grossUpLiftVsMarket": up_rate - float(np.mean(daily_market_up)),
        "majorityUpDayRate": float(np.mean(daily_majority_up)),
        "meanReturnPercentile": float(percentiles.mean()),
        "topReturnDecileRate": float(np.mean(percentiles >= top_decile)),
        "trueMarketTop10OverlapRate": float(np.mean(daily_overlap)),
        "anyTrueMarketTop10HitDayRate": float(np.mean(daily_true_top10_hit)),
        "meanBestSelectedReturn": float(np.mean(daily_best)),
        "severeLossRate": float(np.mean(realized <= severe)),
        "worstDailyGrossReturn": float(np.min(daily_gross)),
        "maximumDrawdownNet": float(np.min(cumulative / running_max - 1.0)),
    }


def walk_forward(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    prior: np.ndarray,
    factor_spec: dict[str, Any],
    baseline_score: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    training = config["training"]
    all_dates = features[next(iter(features))].index
    score = pd.DataFrame(np.nan, index=all_dates, columns=baseline_score.columns)
    blocks: list[dict[str, Any]] = []
    refit = int(training["refitEveryTradingDays"])
    lookback = int(training["lookbackTradingDays"])
    purge = int(training["purgeTradingDays"])
    for start in range(0, len(evaluation_dates), refit):
        test = evaluation_dates[start : start + refit]
        first = all_dates.get_loc(test[0])
        train_end = first - purge
        train_start = max(0, train_end - lookback)
        train = all_dates[train_start:train_end]
        if len(train) < lookback:
            continue
        weight, audit = fit_weights(features, returns, train, prior, factor_spec, config)
        candidate_score = weighted_score(features, weight)
        score.loc[test] = candidate_score.loc[test]
        candidate = metrics(returns, candidate_score, test, config)
        baseline = metrics(returns, baseline_score, test, config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": audit,
                "weights": weight.tolist(),
                "candidate": candidate,
                "baseline": baseline,
                "upRateImproved": candidate.get("grossUpRate", -np.inf)
                > baseline.get("grossUpRate", np.inf),
                "grossImproved": candidate.get("meanGrossReturn", -np.inf)
                > baseline.get("meanGrossReturn", np.inf),
            }
        )
    return score, blocks


def candidate_checks(
    candidate: dict[str, Any], baseline: dict[str, Any], blocks: list[dict[str, Any]]
) -> dict[str, bool]:
    return {
        "grossUpRateImproved": candidate["grossUpRate"] > baseline["grossUpRate"],
        "meanGrossReturnImproved": candidate["meanGrossReturn"] > baseline["meanGrossReturn"],
        "meanReturnPercentileImproved": candidate["meanReturnPercentile"]
        > baseline["meanReturnPercentile"],
        "trueTop10OverlapImproved": candidate["trueMarketTop10OverlapRate"]
        > baseline["trueMarketTop10OverlapRate"],
        "severeLossNotWorse": candidate["severeLossRate"] <= baseline["severeLossRate"],
        "majorityBlocksImproveUpRate": sum(block["upRateImproved"] for block in blocks)
        > len(blocks) / 2.0,
        "majorityBlocksImproveGross": sum(block["grossImproved"] for block in blocks)
        > len(blocks) / 2.0,
    }


def latest_top10(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    date = score.index.max()
    ranked = score.loc[date].dropna().sort_values(ascending=False).head(10)
    return [
        {
            "rank": rank,
            "signalDate": date.date().isoformat(),
            "securityId": str(security_id),
            "name": names.get(str(security_id), ""),
            "close": float(panel["close"].loc[date, security_id]),
            "score": float(value),
            "outcomeKnownAtSelection": bool(
                np.isfinite(returns.loc[date, security_id])
            ),
            "status": "research_only_shadow_candidate_not_an_order",
        }
        for rank, (security_id, value) in enumerate(ranked.items(), start=1)
    ]


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Stock Top10 win-capture weight audit V2",
        "",
        "> Research-only. No candidate may replace the daily ranking from historical results.",
        "",
        f"- run: `{result['runId']}`",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- decision: `{result['decision']}`",
        f"- historical candidate trials: `{len(result['candidates'])}`",
        "",
        "| model | features | up rate | mean gross | mean percentile | true Top10 overlap | severe loss | passed |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    baseline = result["baseline"]
    lines.append(
        f"| frozen baseline | 16 | {baseline['grossUpRate']:.2%} | {baseline['meanGrossReturn']:.4%} | "
        f"{baseline['meanReturnPercentile']:.2%} | {baseline['trueMarketTop10OverlapRate']:.2%} | "
        f"{baseline['severeLossRate']:.2%} | reference |"
    )
    for name, candidate in result["candidates"].items():
        value = candidate["trailingMetrics"]
        lines.append(
            f"| {name} | {candidate['featureCount']} | {value['grossUpRate']:.2%} | "
            f"{value['meanGrossReturn']:.4%} | {value['meanReturnPercentile']:.2%} | "
            f"{value['trueMarketTop10OverlapRate']:.2%} | {value['severeLossRate']:.2%} | "
            f"{'yes' if candidate['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Every candidate is evaluated on the same dates and always selects ten stocks. "
            "Therefore an improvement cannot be manufactured by trading less.",
            "The hindsight-versus-trailing gap is reported for every candidate and hindsight weights are never published.",
            "All historical windows are already viewed; even a pass only qualifies a separately preregistered fresh-forward shadow study.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    data = build_inputs(config)
    panel = data["panel"]
    outcome = data["outcome"]
    baseline_score = data["baselineScore"]
    partitions = discrimination.fixed_partitions(panel["close"].index, data["modelConfig"])
    evaluation_dates = pd.DatetimeIndex(
        sorted(set(partitions["audit"]) | set(partitions["validation"]) | set(partitions["shadow"]))
    )
    candidates: dict[str, Any] = {}
    used_dates: pd.DatetimeIndex | None = None
    all_dates = panel["close"].index
    purge = int(config["training"]["purgeTradingDays"])
    lookback = int(config["training"]["lookbackTradingDays"])
    latest_train = all_dates[-purge - lookback : -purge]
    names = discrimination.v6.name_map(data["baseConfig"])
    for name in config["candidateOrder"]:
        print(f"candidate_start {name}", flush=True)
        features = data["featureSets"][name]
        factor_spec = config["factorSets"][name]
        prior = prior_weights(data["frozen"], features, factor_spec)
        trailing_score, blocks = walk_forward(
            features,
            outcome,
            evaluation_dates,
            prior,
            factor_spec,
            baseline_score,
            config,
        )
        candidate_dates = pd.DatetimeIndex(
            sorted(
                date
                for block in blocks
                for date in pd.date_range(block["testRange"][0], block["testRange"][1], freq="B")
                if date in evaluation_dates
            )
        ).unique()
        if used_dates is None:
            used_dates = candidate_dates
        elif not used_dates.equals(candidate_dates):
            raise RuntimeError("candidate evaluation dates differ")
        baseline_metrics = metrics(outcome, baseline_score, candidate_dates, config)
        trailing_metrics = metrics(outcome, trailing_score, candidate_dates, config)
        hindsight_weight, hindsight_fit = fit_weights(
            features, outcome, candidate_dates, prior, factor_spec, config
        )
        hindsight_metrics = metrics(
            outcome, weighted_score(features, hindsight_weight), candidate_dates, config
        )
        latest_weight, latest_fit = fit_weights(
            features, outcome, latest_train, prior, factor_spec, config
        )
        latest_score = weighted_score(features, latest_weight)
        checks = candidate_checks(trailing_metrics, baseline_metrics, blocks)
        candidates[name] = {
            "featureCount": len(features),
            "featureOrder": list(features),
            "priorWeights": prior.tolist(),
            "latestTrainingRange": [
                latest_train.min().date().isoformat(),
                latest_train.max().date().isoformat(),
            ],
            "latestWeights": [
                {"factor": key, "weight": float(latest_weight[position])}
                for position, key in enumerate(features)
            ],
            "latestFit": latest_fit,
            "latestTop10": latest_top10(latest_score, outcome, panel, names),
            "walkForwardBlocks": blocks,
            "trailingMetrics": trailing_metrics,
            "hindsightMetrics": hindsight_metrics,
            "hindsightFit": hindsight_fit,
            "selectionOverfitExposure": {
                "meanGrossReturnGap": hindsight_metrics["meanGrossReturn"]
                - trailing_metrics["meanGrossReturn"],
                "grossUpRateGap": hindsight_metrics["grossUpRate"]
                - trailing_metrics["grossUpRate"],
                "invalidHindsightWeightsPublished": False,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
        print(
            f"candidate_done {name} up={trailing_metrics['grossUpRate']:.6f} "
            f"gross={trailing_metrics['meanGrossReturn']:.6f} passed={all(checks.values())}",
            flush=True,
        )
    assert used_dates is not None
    baseline = metrics(outcome, baseline_score, used_dates, config)
    passing = [name for name in config["candidateOrder"] if candidates[name]["passed"]]
    if passing:
        decision = f"historical_challenger_{passing[0]}_pass_fresh_forward_required"
    else:
        decision = "reject_all_weight_and_factor_challengers_keep_observation_only"
    identifier = run_id or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z")
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": identifier,
        "generatedAt": datetime.now().astimezone().isoformat(),
        "configSha256": digest(config_path),
        "sourceRankingConfigSha256": digest(data["rankingConfigPath"]),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "signalDate": all_dates.max().date().isoformat(),
        "intendedTradingSession": (
            all_dates.max() + pd.offsets.BDay(1)
        ).date().isoformat(),
        "panelAudit": data["panelAudit"],
        "factorAudit": data["factorAudit"],
        "fundamentalAudit": data["fundamentalAudit"],
        "evaluationDates": [used_dates.min().date().isoformat(), used_dates.max().date().isoformat()],
        "baseline": baseline,
        "candidates": candidates,
        "passingCandidates": passing,
        "decision": decision,
        "mechanicallyReducedSelectionCount": False,
        "target": config["outcome"],
        "historicalWindowsAlreadyViewed": True,
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    output = ROOT / config["output"]["root"] / identifier
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(report_markdown(result), encoding="utf-8")
    for name, candidate in candidates.items():
        pd.DataFrame(candidate["latestWeights"]).to_csv(
            output / f"{name}_latest_weights.csv", index=False
        )
        pd.DataFrame(candidate["latestTop10"]).to_csv(
            output / f"{name}_latest_top10.csv", index=False
        )
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_id)
    print(
        json.dumps(
            {
                "runId": result["runId"],
                "decision": result["decision"],
                "baseline": result["baseline"],
                "candidates": {
                    name: {
                        "metrics": value["trailingMetrics"],
                        "checks": value["checks"],
                        "passed": value["passed"],
                    }
                    for name, value in result["candidates"].items()
                },
                "eligibleForTrading": False,
                "orders": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
