#!/usr/bin/env python3
"""Generate a research-only Top10 reranked by twelve factors plus four interactions."""

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

import generate_guarded_weight_top10_forecast_v1 as forecast  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import research_twelve_factor_fundamental_price_interactions_v1 as interaction_model  # noqa: E402
import research_twelve_factor_fundamental_second_stage_v5 as fundamental_stage  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402
import stock_forecast_dashboard as dashboard  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "sixteen_factor_interaction_ranking_v1.json"
)
SCHEMA_VERSION = "sixteen_factor_interaction_ranking_result_v1"
CODE_VERSION = "sixteen_factor_interaction_ranking_v1_20260813"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
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
    if config.get("schemaVersion") != "sixteen_factor_interaction_ranking_v1":
        raise ValueError("unexpected sixteen-factor ranking schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("sixteen-factor ranking must remain research/shadow-only")
    source = ROOT / str(config["sourceInteractionConfig"])
    if sha256(source) != str(config["sourceInteractionConfigSha256"]).lower():
        raise ValueError("frozen interaction dependency changed")
    ranking = config["ranking"]
    if int(ranking["factorCount"]) != 16 or int(ranking["topCount"]) != 10:
        raise ValueError("ranking must remain a sixteen-factor Top10")
    if int(ranking["interactionCount"]) != 4:
        raise ValueError("exactly four interaction factors are required")
    block = float(ranking["twelveFactorBlockWeight"])
    each = float(ranking["eachInteractionWeight"])
    if abs(block - 0.75) > 1e-12 or abs(each - 0.0625) > 1e-12:
        raise ValueError("sixteen equal factor slots are frozen at 75% plus 4x6.25%")
    if abs(block + int(ranking["interactionCount"]) * each - 1.0) > 1e-12:
        raise ValueError("ranking weights must sum to one")
    if ranking.get("requireAllFourFundamentalInteractions") is not True:
        raise ValueError("all four PIT fundamental interactions must be present")
    if ranking.get("missingInteractionMayBeImputed") is not False:
        raise ValueError("a missing interaction must fail closed")
    if ranking.get("historicalOutcomeMayFitRankingWeights") is not False:
        raise ValueError("historical outcomes cannot fit ranking weights")
    if ranking.get("historicalOutcomeMaySelectInteraction") is not False:
        raise ValueError("historical outcomes cannot select interactions")
    if ranking.get("probabilityStretchingAllowed") is not False:
        raise ValueError("probability stretching is forbidden")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(
        bool(value)
        for key, value in config.get("safety", {}).items()
        if key.startswith("may")
    ):
        raise ValueError("all trading and mutation permissions must remain false")


def combine_scores(
    twelve_factor_score: pd.DataFrame,
    interactions: dict[str, pd.DataFrame],
    support: pd.DataFrame,
    twelve_weight: float = 0.75,
    each_interaction_weight: float = 0.0625,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine 12-factor and interaction blocks without fitting ranking weights."""
    if len(interactions) != 4:
        raise ValueError("four interaction ranks are required")
    mask = support.fillna(False).astype(bool)
    interaction_sum = sum(
        (frame.where(mask).astype(float) for frame in interactions.values()),
        start=pd.DataFrame(0.0, index=mask.index, columns=mask.columns),
    )
    interaction_mean = (interaction_sum / 4.0).where(mask)
    combined = (
        float(twelve_weight) * twelve_factor_score.where(mask).astype(float)
        + float(each_interaction_weight) * interaction_sum
    ).where(mask)
    return combined.astype("float32"), interaction_mean.astype("float32")


def latest_rows(
    combined_score: pd.DataFrame,
    twelve_score: pd.DataFrame,
    interaction_mean: pd.DataFrame,
    interactions: dict[str, pd.DataFrame],
    predictions: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    missing_price_factors: pd.DataFrame,
) -> list[dict[str, Any]]:
    date = combined_score.index.max()
    ranked = combined_score.loc[date].dropna().sort_values(ascending=False)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, value) in enumerate(ranked.items(), start=1):
        values = [
            predictions["expectedReturn"].loc[date, security_id],
            predictions["grossUp"].loc[date, security_id],
            predictions["severeLoss"].loc[date, security_id],
        ]
        if not all(np.isfinite(values)):
            raise RuntimeError(f"sixteen-factor forecast missing for {security_id}")
        expected, probability_up, probability_tail = map(float, values)
        imputed = int(missing_price_factors.loc[date, security_id])
        rows.append(
            {
                "rank": rank,
                "signalDate": date.date().isoformat(),
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": round(float(panel["close"].loc[date, security_id]), 4),
                "adaptiveFactorScore": round(float(value), 8),
                "twelveFactorScore": round(float(twelve_score.loc[date, security_id]), 8),
                "interactionCompositeScore": round(
                    float(interaction_mean.loc[date, security_id]), 8
                ),
                "interactionRanks": {
                    name: round(float(frame.loc[date, security_id]), 8)
                    for name, frame in interactions.items()
                },
                "expectedGrossReturn": round(expected, 8),
                "probabilityUp": round(probability_up, 8),
                "probabilitySevereLoss": round(probability_tail, 8),
                "factorCount": 16,
                "imputedFactorCount": imputed,
                "factorCompletion": (
                    "all_four_interactions_present_price_factors_complete"
                    if imputed == 0
                    else "all_four_interactions_present_price_median_completion"
                ),
                "estimateSource": (
                    "sixteen_factor_fundamental_interaction_multivariate_calibrated"
                ),
                "status": "diagnostic_only_reranked_not_an_order",
            }
        )
    return rows


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = np.asarray([row["expectedGrossReturn"] for row in rows], dtype=float)
    up = np.asarray([row["probabilityUp"] for row in rows], dtype=float)
    tail = np.asarray([row["probabilitySevereLoss"] for row in rows], dtype=float)
    top = slice(0, 10)
    return {
        "securityCount": len(rows),
        "distinctExpectedReturnsAtEightDecimals": int(len(np.unique(expected))),
        "expectedReturnMinimum": float(expected.min()),
        "expectedReturnMaximum": float(expected.max()),
        "expectedReturnSpread": float(expected.max() - expected.min()),
        "probabilityUpMinimum": float(up.min()),
        "probabilityUpMaximum": float(up.max()),
        "probabilityUpSpread": float(up.max() - up.min()),
        "probabilityTailLossMinimum": float(tail.min()),
        "probabilityTailLossMaximum": float(tail.max()),
        "probabilityTailLossSpread": float(tail.max() - tail.min()),
        "top10ExpectedReturnSpread": float(expected[top].max() - expected[top].min()),
        "top10ProbabilityUpSpread": float(up[top].max() - up[top].min()),
        "top10ProbabilityTailLossSpread": float(tail[top].max() - tail[top].min()),
        "mechanicallyStretched": False,
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Sixteen-factor fundamental interaction Top10 V1",
        "",
        "> Research-only reranking. Not an order and not eligible for trading.",
        "",
        f"- signal date: `{result['signalDate']}`",
        f"- intended session: `{result['intendedTradingSession']}`",
        "- weights: guarded twelve-factor block 75%; four interactions 6.25% each",
        "- fundamental availability: strictly after noticeDate/updateDate",
        f"- common supported securities: `{result['latestAllForecastCount']}`",
        f"- reliability: `{result['forecastReliability']['status']}`",
        "",
        "| rank | security | name | 16-score | 12-score | interaction | expected | P(up) | P(tail) |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["latestTop10"]:
        lines.append(
            f"| {row['rank']} | {row['securityId']} | {row['name']} | "
            f"{row['adaptiveFactorScore']:.4f} | {row['twelveFactorScore']:.4f} | "
            f"{row['interactionCompositeScore']:.4f} | "
            f"{row['expectedGrossReturn']:.3%} | {row['probabilityUp']:.2%} | "
            f"{row['probabilitySevereLoss']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The four interaction weights were not fitted from historical outcomes. "
            "This challenger ranking has not passed a fresh-forward promotion gate.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    interaction_config = load_json(ROOT / config["sourceInteractionConfig"])
    interaction_model.validate_config(interaction_config)
    source_config = load_json(ROOT / interaction_config["sourceForecastConfig"])
    forecast.validate_config(source_config)
    guarded_config = load_json(ROOT / source_config["guardedWeightsConfig"])
    guarded.validate_config(guarded_config)
    frozen, _source, _hash = guarded.load_frozen_config(guarded_config)
    model_config = load_json(ROOT / interaction_config["modelTemplate"])
    fundamental_config = load_json(
        ROOT / interaction_config["fundamentalMechanismConfig"]
    )
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, _execution_eligible, _exit_delay = precision.executable_horizon_return(
        panel,
        int(interaction_config["data"]["holdingTradingDays"]),
        int(interaction_config["data"]["maximumExitDelayTradingDays"]),
    )
    price_ranks, _static_score, factor_audit = rolling.compute_rank_book(panel, frozen)
    daily_ic = rolling.factor_daily_ic(price_ranks, outcome)
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    adaptive_weights, weight_updates = guarded.guarded_weight_path(
        daily_ic, prior, guarded_config
    )
    completion = source_config["forecast"]["missingFeatureCompletion"]
    completed_price, missing_count, price_support, completion_audit = (
        forecast.complete_rank_book(
            price_ranks,
            panel["eligible"],
            int(interaction_config["data"]["maximumImputedPriceFactorsForEligibility"]),
            float(completion["allMissingCrossSectionFallbackRank"]),
        )
    )
    twelve_score = guarded.adaptive_score(
        completed_price, adaptive_weights, panel
    ).where(price_support)
    family_ranks, fundamental_support, fundamental_audit = (
        fundamental_stage.build_family_scores(panel, fundamental_config)
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
    ranking = config["ranking"]
    combined_score, interaction_mean = combine_scores(
        twelve_score,
        interactions,
        common_support,
        float(ranking["twelveFactorBlockWeight"]),
        float(ranking["eachInteractionWeight"]),
    )
    feature_book = interaction_model.feature_books(
        completed_price, interactions, common_support
    )["all_interactions"]
    comparable_outcome = outcome.where(common_support)
    partitions = discrimination.fixed_partitions(panel["close"].index, model_config)
    models, fit_audit = discrimination.fit_models(
        feature_book, comparable_outcome, partitions, model_config
    )
    amplitude_audit = forecast.refit_scale_aware_return_calibrator(
        models,
        feature_book,
        comparable_outcome,
        partitions["calibration"],
        model_config,
        source_config["forecast"]["returnAmplitudeCalibration"],
    )
    prediction_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *[
                    set(partitions[name])
                    for name in ("audit", "validation", "shadow")
                ],
                {panel["close"].index.max()},
            )
        )
    )
    predictions = discrimination.predict_multivariate(
        feature_book, prediction_dates, models, model_config
    )
    periods = {
        name: discrimination.evaluate_period(
            predictions,
            comparable_outcome,
            partitions[name].intersection(prediction_dates),
            model_config,
        )
        for name in ("audit", "validation", "shadow")
    }
    rows = latest_rows(
        combined_score,
        twelve_score,
        interaction_mean,
        interactions,
        predictions,
        panel,
        discrimination.v6.name_map(base),
        missing_count,
    )
    latest_date = panel["close"].index.max()
    identifier = run_id or datetime.now().astimezone().strftime(
        "run_%Y%m%dT%H%M%S%z"
    )
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "runId": identifier,
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            latest_date.date().isoformat(),
        ],
        "signalDate": latest_date.date().isoformat(),
        "intendedTradingSession": (latest_date + pd.offsets.BDay(1)).date().isoformat(),
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "fundamentalAudit": fundamental_audit,
        "factorCompletionAudit": completion_audit,
        "rankingDefinition": config["ranking"],
        "weightUpdateCount": int(len(weight_updates)),
        "latestTwelveFactorWeights": safe(adaptive_weights.loc[latest_date].to_dict()),
        "fitAudit": fit_audit,
        "returnAmplitudeCalibrationAudit": amplitude_audit,
        "periodDiagnostics": periods,
        "forecastReliability": forecast.forecast_reliability(periods),
        "latestForecastDistribution": distribution(rows),
        "latestTop10": rows[: int(ranking["topCount"])],
        "latestAllForecastCount": len(rows),
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    output = ROOT / config["output"]["root"] / identifier
    precision.atomic_write(
        output / "result.json",
        json.dumps(safe(result), ensure_ascii=False, indent=2) + "\n",
    )
    precision.atomic_write(output / "report.md", report_markdown(result))
    precision.atomic_write(
        output / "latest_top10.csv",
        pd.DataFrame(result["latestTop10"]).to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_all_forecasts.csv",
        pd.DataFrame(rows).to_csv(index=False, lineterminator="\n"),
    )
    if config["output"].get("publishReadOnlyDashboard") is True:
        dashboard.publish_snapshot(
            result,
            rows,
            source_result_path=output / "result.json",
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
                "signalDate": result["signalDate"],
                "intendedTradingSession": result["intendedTradingSession"],
                "top10": result["latestTop10"],
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
