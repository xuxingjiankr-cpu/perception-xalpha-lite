#!/usr/bin/env python3
"""Train and audit bounded non-equal 16-factor weights without look-ahead."""

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
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import research_twelve_factor_fundamental_price_interactions_v1 as interaction_model  # noqa: E402
import research_twelve_factor_fundamental_second_stage_v5 as fundamental_stage  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "sixteen_factor_walkforward_weights_v1.json"
)
SCHEMA_VERSION = "sixteen_factor_walkforward_weights_result_v1"
CODE_VERSION = "sixteen_factor_walkforward_weights_v1_20260813"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
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
    if config.get("schemaVersion") != "sixteen_factor_walkforward_weights_v1":
        raise ValueError("unexpected sixteen-factor weight schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("weight research must remain research/shadow-only")
    spec = config["training"]
    if int(spec["purgeTradingDays"]) < 1:
        raise ValueError("a positive outcome purge is required")
    if int(spec["lookbackTradingDays"]) < int(spec["refitEveryTradingDays"]):
        raise ValueError("walk-forward lookback must exceed the refit interval")
    lower = float(spec["minimumFactorWeight"])
    upper = float(spec["maximumFactorWeight"])
    if not 0.0 < lower < 1.0 / 16.0 < upper < 1.0:
        raise ValueError("factor bounds must contain the sixteen-factor prior")
    if lower * 16.0 > 1.0 or upper * 16.0 < 1.0:
        raise ValueError("factor bounds cannot form a simplex")
    if not 0.0 < float(spec["minimumInteractionBlockWeight"]) < float(
        spec["maximumInteractionBlockWeight"]
    ) < 1.0:
        raise ValueError("interaction block bounds are invalid")
    if spec.get("validationOrShadowMayTuneHyperparameters") is not False:
        raise ValueError("evaluation periods cannot tune hyperparameters")
    if spec.get("hindsightWeightsMayBePublished") is not False:
        raise ValueError("hindsight weights cannot be published")
    if config["evaluation"].get("reportHindsightAndTrailingOnIdenticalDates") is not True:
        raise ValueError("selection exposure must be reported")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(
        bool(value)
        for key, value in config.get("safety", {}).items()
        if key.startswith("may")
    ):
        raise ValueError("all trading and mutation permissions must remain false")


def prior_weights(
    frozen: dict[str, Any], feature_keys: list[str]
) -> np.ndarray:
    price_lookup = {
        str(row["factorKey"]): float(row["weight"])
        for row in frozen["frozenFactors"]
    }
    result = []
    for key in feature_keys:
        if key.startswith("interaction/"):
            result.append(0.0625)
        else:
            result.append(0.75 * price_lookup[key])
    output = np.asarray(result, dtype=float)
    if len(output) != 16 or abs(float(output.sum()) - 1.0) > 1e-8:
        raise ValueError("sixteen-factor prior is not a simplex")
    return output


def pairwise_arrays(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = config["training"]
    keys = list(features)
    differences: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used_days = 0
    used_rows = 0
    top_count = int(spec["trueTopCount"])
    negative_count = int(spec["nearBoundaryNegativeCount"])
    for date in dates:
        y = returns.loc[date].to_numpy(dtype=float)
        x = np.column_stack(
            [features[key].loc[date].to_numpy(dtype=float) for key in keys]
        )
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(valid.sum()) < max(
            int(spec["minimumCrossSectionRows"]), top_count + negative_count
        ):
            continue
        xv = x[valid]
        yv = y[valid]
        utility = (
            yv
            - float(spec["realizedUtilityPenaltyNonPositive"]) * (yv <= 0.0)
            - float(spec["realizedUtilityPenaltySevereLoss"])
            * (yv <= float(config["evaluation"]["severeLossThreshold"]))
        )
        order = np.argsort(utility, kind="stable")[::-1]
        positive = xv[order[:top_count]]
        negative = xv[order[top_count : top_count + negative_count]]
        day = (positive[:, None, :] - negative[None, :, :]).reshape(-1, len(keys))
        differences.append(day)
        weights.append(np.full(len(day), 1.0 / len(day), dtype=float))
        used_days += 1
        used_rows += int(valid.sum())
    if not differences:
        raise RuntimeError("no training dates met the cross-section requirement")
    return (
        np.concatenate(differences),
        np.concatenate(weights) / used_days,
        {
            "usedTradingDays": used_days,
            "usedRows": used_rows,
            "pairCount": int(sum(len(value) for value in differences)),
        },
    )


def fit_weights(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    prior: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    differences, sample_weight, audit = pairwise_arrays(
        features, returns, dates, config
    )
    spec = config["training"]
    l2 = float(spec["l2ShrinkageToPrior"])

    def objective(weight: np.ndarray) -> float:
        margin = differences @ weight
        return float(np.sum(np.logaddexp(0.0, -margin) * sample_weight)) + l2 * float(
            np.square(weight - prior).sum()
        )

    def gradient(weight: np.ndarray) -> np.ndarray:
        margin = differences @ weight
        inverse = np.exp(-np.logaddexp(0.0, margin))
        return -(differences.T @ (inverse * sample_weight)) + 2.0 * l2 * (
            weight - prior
        )

    lower = float(spec["minimumFactorWeight"])
    upper = float(spec["maximumFactorWeight"])
    interaction_min = float(spec["minimumInteractionBlockWeight"])
    interaction_max = float(spec["maximumInteractionBlockWeight"])
    result = minimize(
        objective,
        prior,
        jac=gradient,
        method="SLSQP",
        bounds=[(lower, upper)] * len(prior),
        constraints=[
            {"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            {
                "type": "ineq",
                "fun": lambda value: float(value[-4:].sum() - interaction_min),
            },
            {
                "type": "ineq",
                "fun": lambda value: float(interaction_max - value[-4:].sum()),
            },
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"weight optimizer failed closed: {result.message}")
    weight = np.asarray(result.x, dtype=float)
    if (
        abs(float(weight.sum()) - 1.0) > 1e-8
        or np.any(weight < lower - 1e-9)
        or np.any(weight > upper + 1e-9)
        or not interaction_min - 1e-9 <= float(weight[-4:].sum()) <= interaction_max + 1e-9
    ):
        raise RuntimeError("weight optimizer violated frozen constraints")
    audit.update(
        {
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "interactionBlockWeight": float(weight[-4:].sum()),
            "success": True,
        }
    )
    return weight, audit


def weighted_score(
    features: dict[str, pd.DataFrame], weight: np.ndarray
) -> pd.DataFrame:
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
    selected: list[np.ndarray] = []
    daily_gross: list[float] = []
    top_count = int(config["training"]["trueTopCount"])
    for date in dates.intersection(score.index):
        values = pd.concat(
            [score.loc[date].rename("score"), returns.loc[date].rename("return")],
            axis=1,
        ).dropna()
        if len(values) < max(top_count, int(config["training"]["minimumCrossSectionRows"])):
            continue
        chosen = values.nlargest(top_count, "score")["return"].to_numpy(dtype=float)
        selected.append(chosen)
        daily_gross.append(float(np.mean(chosen)))
    if not selected:
        return {"tradingDays": 0, "selectedRows": 0}
    values = np.concatenate(selected)
    cost = float(config["evaluation"]["roundTripCost"])
    severe = float(config["evaluation"]["severeLossThreshold"])
    cumulative = np.cumprod(1.0 + np.asarray(daily_gross) - cost)
    running_max = np.maximum.accumulate(cumulative)
    return {
        "tradingDays": len(daily_gross),
        "selectedRows": int(len(values)),
        "meanGrossReturn": float(np.mean(values)),
        "meanNetReturn": float(np.mean(values) - cost),
        "grossUpRate": float(np.mean(values > 0.0)),
        "netWinRate": float(np.mean(values > cost)),
        "nonPositiveRate": float(np.mean(values <= 0.0)),
        "severeLossRate": float(np.mean(values <= severe)),
        "worstDailyGrossReturn": float(np.min(daily_gross)),
        "dailyGrossStd": float(np.std(daily_gross, ddof=1)),
        "maximumDrawdownNet": float(np.min(cumulative / running_max - 1.0)),
    }


def walk_forward(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    prior: np.ndarray,
    baseline_score: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = config["training"]
    all_dates = features[next(iter(features))].index
    score = pd.DataFrame(np.nan, index=all_dates, columns=baseline_score.columns)
    blocks: list[dict[str, Any]] = []
    refit = int(spec["refitEveryTradingDays"])
    lookback = int(spec["lookbackTradingDays"])
    purge = int(spec["purgeTradingDays"])
    for start in range(0, len(evaluation_dates), refit):
        test = evaluation_dates[start : start + refit]
        first = all_dates.get_loc(test[0])
        train_end = first - purge
        train_start = max(0, train_end - lookback)
        train = all_dates[train_start:train_end]
        if len(train) < lookback:
            continue
        weight, audit = fit_weights(features, returns, train, prior, config)
        block_score = weighted_score(features, weight)
        score.loc[test] = block_score.loc[test]
        candidate = metrics(returns, block_score, test, config)
        baseline = metrics(returns, baseline_score, test, config)
        blocks.append(
            {
                "trainRange": [train.min().date().isoformat(), train.max().date().isoformat()],
                "testRange": [test.min().date().isoformat(), test.max().date().isoformat()],
                "fit": audit,
                "weights": weight.tolist(),
                "candidate": candidate,
                "baseline": baseline,
                "grossImproved": candidate.get("meanGrossReturn", -np.inf)
                > baseline.get("meanGrossReturn", np.inf),
            }
        )
    return score, blocks


def replica_dates(
    dates: pd.DatetimeIndex, config: dict[str, Any], replica: int
) -> pd.DatetimeIndex:
    spec = config["training"]
    block = int(spec["blockLengthTradingDays"])
    blocks = [dates[start : start + block] for start in range(0, len(dates), block)]
    keep = max(1, int(math.ceil(len(blocks) * float(spec["blockSubsampleFraction"]))))
    rng = np.random.default_rng(int(spec["randomSeed"]) + replica)
    chosen = sorted(rng.choice(len(blocks), size=keep, replace=False).tolist())
    return pd.DatetimeIndex(np.concatenate([blocks[index].to_numpy() for index in chosen]))


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Sixteen-factor walk-forward weight audit V1",
        "",
        "> Research-only. Hindsight weights are an invalid ceiling and are never published.",
        "",
        f"- run: `{result['runId']}`",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- decision: `{result['decision']}`",
        f"- walk-forward blocks improving gross: `{result['walkForwardSummary']['grossImprovedBlocks']}/{result['walkForwardSummary']['blockCount']}`",
        f"- selection overfit exposure: `{result['selectionOverfitExposure']['meanGrossReturnGap']:.6%}` per pick",
        "",
        "## Latest trailing weights",
        "",
        "| factor | prior | learned | replica min | replica max |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["latestWeights"]:
        lines.append(
            f"| {row['factor']} | {row['prior']:.3%} | {row['learned']:.3%} | "
            f"{row['replicaMinimum']:.3%} | {row['replicaMaximum']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Identical-date comparison",
            "",
            "| method | mean gross | mean net | up rate | severe loss | max drawdown net |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "trailing", "hindsight"):
        value = result["identicalDateMetrics"][name]
        lines.append(
            f"| {name} | {value['meanGrossReturn']:.4%} | {value['meanNetReturn']:.4%} | "
            f"{value['grossUpRate']:.2%} | {value['severeLossRate']:.2%} | "
            f"{value['maximumDrawdownNet']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The latest learned weights use only the trailing window ending before the frozen purge. "
            "They remain a historical challenger because all evaluation windows have already been viewed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    ranking_config_path = ROOT / config["sourceRankingConfig"]
    ranking_config = load_json(ranking_config_path)
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
    print(f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}", flush=True)
    outcome, execution_eligible, _delay = precision.executable_horizon_return(
        panel,
        int(interaction_config["data"]["holdingTradingDays"]),
        int(interaction_config["data"]["maximumExitDelayTradingDays"]),
    )
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
    completed_price, _missing, price_support, _audit = forecast.complete_rank_book(
        price_ranks,
        panel["eligible"],
        int(interaction_config["data"]["maximumImputedPriceFactorsForEligibility"]),
        float(completion["allMissingCrossSectionFallbackRank"]),
    )
    family_ranks, fundamental_support, fundamental_audit = fundamental_stage.build_family_scores(
        panel, fundamental_config
    )
    market_ranks = interaction_model.build_market_context_ranks(panel, interaction_config)
    common_support = price_support & fundamental_support
    for frame in market_ranks.values():
        common_support &= frame.notna()
    interactions = interaction_model.build_interaction_ranks(
        family_ranks, market_ranks, interaction_config, common_support
    )
    for frame in interactions.values():
        common_support &= frame.notna()
    features = interaction_model.feature_books(
        completed_price, interactions, common_support
    )["all_interactions"]
    outcome = outcome.where(common_support)
    twelve_score = guarded.adaptive_score(completed_price, adaptive_weights, panel)
    baseline_score, _interaction_mean = sixteen.combine_scores(
        twelve_score,
        interactions,
        common_support,
        float(ranking_config["ranking"]["twelveFactorBlockWeight"]),
        float(ranking_config["ranking"]["eachInteractionWeight"]),
    )
    feature_keys = list(features)
    prior = prior_weights(frozen, feature_keys)
    partitions = discrimination.fixed_partitions(panel["close"].index, model_config)
    evaluation_dates = pd.DatetimeIndex(
        sorted(
            set(partitions["audit"])
            | set(partitions["validation"])
            | set(partitions["shadow"])
        )
    )
    trailing_score, blocks = walk_forward(
        features, outcome, evaluation_dates, prior, baseline_score, config
    )
    used_dates = pd.DatetimeIndex(
        sorted(
            date
            for block in blocks
            for date in pd.date_range(block["testRange"][0], block["testRange"][1], freq="B")
            if date in evaluation_dates
        )
    ).unique()
    hindsight_weight, hindsight_fit = fit_weights(
        features, outcome, used_dates, prior, config
    )
    hindsight_score = weighted_score(features, hindsight_weight)
    identical = {
        "baseline": metrics(outcome, baseline_score, used_dates, config),
        "trailing": metrics(outcome, trailing_score, used_dates, config),
        "hindsight": metrics(outcome, hindsight_score, used_dates, config),
    }
    all_dates = panel["close"].index
    purge = int(config["training"]["purgeTradingDays"])
    lookback = int(config["training"]["lookbackTradingDays"])
    latest_train = all_dates[-purge - lookback : -purge]
    latest_weight, latest_fit = fit_weights(
        features, outcome, latest_train, prior, config
    )
    replicas: list[np.ndarray] = []
    for index in range(int(config["training"]["blockReplicaCount"])):
        dates = replica_dates(latest_train, config, index)
        replica, _fit = fit_weights(features, outcome, dates, prior, config)
        replicas.append(replica)
    replica_matrix = np.vstack(replicas)
    weight_rows = [
        {
            "factor": key,
            "prior": float(prior[position]),
            "learned": float(latest_weight[position]),
            "replicaMinimum": float(replica_matrix[:, position].min()),
            "replicaMaximum": float(replica_matrix[:, position].max()),
            "replicaStd": float(replica_matrix[:, position].std(ddof=1)),
        }
        for position, key in enumerate(feature_keys)
    ]
    trailing = identical["trailing"]
    baseline = identical["baseline"]
    checks = {
        "trailingMeanGrossImproved": trailing["meanGrossReturn"] > baseline["meanGrossReturn"],
        "trailingUpRateImproved": trailing["grossUpRate"] > baseline["grossUpRate"],
        "trailingSevereLossNotWorse": trailing["severeLossRate"] <= baseline["severeLossRate"],
        "majorityWalkForwardBlocksImproveGross": sum(
            bool(block["grossImproved"]) for block in blocks
        ) > len(blocks) / 2.0,
    }
    passed = all(checks.values())
    identifier = run_id or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z")
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": identifier,
        "generatedAt": datetime.now().astimezone().isoformat(),
        "dataRange": [all_dates.min().date().isoformat(), all_dates.max().date().isoformat()],
        "signalDate": all_dates.max().date().isoformat(),
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "fundamentalAudit": fundamental_audit,
        "configSha256": file_sha256(config_path),
        "sourceRankingConfigSha256": file_sha256(ranking_config_path),
        "featureOrder": feature_keys,
        "latestTrainingRange": [latest_train.min().date().isoformat(), latest_train.max().date().isoformat()],
        "latestFit": latest_fit,
        "latestWeights": weight_rows,
        "walkForwardBlocks": blocks,
        "walkForwardSummary": {
            "blockCount": len(blocks),
            "grossImprovedBlocks": sum(bool(block["grossImproved"]) for block in blocks),
        },
        "hindsightFit": hindsight_fit,
        "hindsightWeights": hindsight_weight.tolist(),
        "identicalDateMetrics": identical,
        "selectionOverfitExposure": {
            "meanGrossReturnGap": identical["hindsight"]["meanGrossReturn"]
            - identical["trailing"]["meanGrossReturn"],
            "invalidHindsightMayBePublished": False,
        },
        "checks": checks,
        "decision": (
            "historical_challenger_pass_fresh_forward_required"
            if passed
            else "reject_learned_weights_keep_current_fixed_ranking"
        ),
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
    pd.DataFrame(weight_rows).to_csv(output / "trained_weights.csv", index=False)
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
                "walkForwardSummary": result["walkForwardSummary"],
                "selectionOverfitExposure": result["selectionOverfitExposure"],
                "latestWeights": result["latestWeights"],
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
