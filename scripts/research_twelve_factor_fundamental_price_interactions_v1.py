#!/usr/bin/env python3
"""Audit four preregistered PIT fundamental by price-volume interactions.

The current guarded twelve-factor Top10 is held fixed.  Individual interactions are
diagnostic ablations only; the all-interaction model is the sole primary candidate.
"""

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

import generate_guarded_weight_top10_forecast_v1 as forecast  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import research_twelve_factor_fundamental_second_stage_v5 as fundamental_stage  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402
import research_twelve_factor_pit_fundamental_increment_v1 as increment  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "twelve_factor_fundamental_price_interactions_v1.json"
)
SCHEMA_VERSION = "twelve_factor_fundamental_price_interactions_result_v1"
CODE_VERSION = "twelve_factor_fundamental_price_interactions_v1_20260813"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_fundamental_price_interactions_v1":
        raise ValueError("unexpected fundamental-interaction schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("fundamental interactions must remain research/shadow-only")
    for path_key, hash_key in (
        ("sourceForecastConfig", "sourceForecastConfigSha256"),
        ("modelTemplate", "modelTemplateSha256"),
        ("fundamentalMechanismConfig", "fundamentalMechanismConfigSha256"),
    ):
        if sha256(ROOT / config[path_key]) != str(config[hash_key]).lower():
            raise ValueError(f"frozen dependency changed: {path_key}")
    interactions = config["hypothesis"]["interactions"]
    expected = {
        "earnings_volume_confirmation": (
            "earnings_innovation",
            "amount_surprise_20",
        ),
        "growth_momentum_confirmation": ("growth_acceleration", "momentum_20"),
        "quality_low_volatility": ("quality", "low_volatility_20"),
        "cash_quality_reversal": ("cash_flow_quality", "reversal_5"),
    }
    actual = {
        name: (value["fundamentalFamily"], value["marketContext"])
        for name, value in interactions.items()
    }
    if actual != expected:
        raise ValueError("the preregistered mechanism interaction map changed")
    hypothesis = config["hypothesis"]
    if hypothesis.get("historicalOutcomeMaySelectInteraction") is not False:
        raise ValueError("historical outcomes may not select an interaction")
    if hypothesis.get("historicalOutcomeMayFitInteractionWeight") is not False:
        raise ValueError("historical outcomes may not fit interaction weights")
    if hypothesis.get("hyperparameterSearchAllowed") is not False:
        raise ValueError("historical hyperparameter search is forbidden")
    data = config["data"]
    if (
        int(data["priceFactorCount"]) != 12
        or int(data["fundamentalFamilyCount"]) != 4
        or int(data["interactionCount"]) != 4
        or int(data["primaryCandidateFeatureCount"]) != 16
    ):
        raise ValueError("the interaction feature counts changed")
    if data.get("availabilityRule") != (
        "first_market_date_strictly_after_max_notice_update"
    ):
        raise ValueError("fundamental availability must stay strictly causal")
    if data.get("reportDateMayDetermineAvailability") is not False:
        raise ValueError("reportDate may not determine availability")
    training = config["training"]
    if training.get("sameRowsForEveryAblation") is not True:
        raise ValueError("all interaction ablations must use identical rows")
    if training.get("sameTop10ForForecastComparison") is not True:
        raise ValueError("the Top10 must stay fixed")
    if training.get("individualAblationsAreDiagnosticOnly") is not True:
        raise ValueError("individual interactions may not become a winner menu")
    if training.get("postOutcomeWinnerSelectionAllowed") is not False:
        raise ValueError("post-outcome interaction selection is forbidden")
    if training.get("probabilityStretchingAllowed") is not False:
        raise ValueError("probability stretching is forbidden")
    if config["evaluation"].get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical window reuse must be explicit")
    safety = config.get("safety", {})
    if any(bool(value) for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all mutation and trading permissions must remain false")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def build_market_context_ranks(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Build date-t observable market contexts using backward-looking windows only."""
    eligible = panel["eligible"].fillna(False)
    close = panel["close"].where(eligible).astype(float)
    amount = panel["amount"].where(eligible).astype(float).where(lambda value: value > 0)
    amount_spec = config["marketContexts"]["amount_surprise_20"]
    denominator = amount.rolling(
        int(amount_spec["lookbackTradingDays"]),
        min_periods=int(amount_spec["minimumObservations"]),
    ).median().shift(1)
    amount_surprise = np.log(amount) - np.log(denominator)
    momentum_20 = close.div(close.shift(20)).sub(1.0)
    returns = close.pct_change(fill_method=None)
    low_volatility_20 = -returns.rolling(20, min_periods=20).std(ddof=1)
    reversal_5 = -close.div(close.shift(5)).sub(1.0)
    raw = {
        "amount_surprise_20": amount_surprise,
        "momentum_20": momentum_20,
        "low_volatility_20": low_volatility_20,
        "reversal_5": reversal_5,
    }
    return {
        name: value.where(eligible).rank(axis=1, pct=True, method="average").astype("float32")
        for name, value in raw.items()
    }


def build_interaction_ranks(
    family_ranks: dict[str, pd.DataFrame],
    market_ranks: dict[str, pd.DataFrame],
    config: dict[str, Any],
    support: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create four nonnegative conjunction factors and rerank them cross-sectionally."""
    result: dict[str, pd.DataFrame] = {}
    mask = support.fillna(False)
    for name, definition in config["hypothesis"]["interactions"].items():
        family = family_ranks[definition["fundamentalFamily"]].clip(0.0, 1.0)
        market = market_ranks[definition["marketContext"]].clip(0.0, 1.0)
        conjunction = np.sqrt(family.astype(float) * market.astype(float)).where(mask)
        result[name] = conjunction.rank(
            axis=1, pct=True, method="average"
        ).where(mask).astype("float32")
    return result


def feature_books(
    price_ranks: dict[str, pd.DataFrame],
    interactions: dict[str, pd.DataFrame],
    support: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    if len(price_ranks) != 12 or len(interactions) != 4:
        raise ValueError("expected twelve price ranks and four interaction ranks")
    mask = support.fillna(False)
    baseline = {key: value.where(mask) for key, value in price_ranks.items()}
    books: dict[str, dict[str, pd.DataFrame]] = {"baseline": baseline}
    for name, frame in interactions.items():
        books[name] = {**baseline, f"interaction/{name}": frame.where(mask)}
    books["all_interactions"] = {
        **baseline,
        **{f"interaction/{name}": frame.where(mask) for name, frame in interactions.items()},
    }
    return books


def _fit_and_predict(
    books: dict[str, dict[str, pd.DataFrame]],
    outcome: pd.DataFrame,
    partitions: dict[str, pd.DatetimeIndex],
    prediction_dates: pd.DatetimeIndex,
    model_config: dict[str, Any],
    amplitude_config: dict[str, Any],
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    audit: dict[str, Any] = {}
    for name, features in books.items():
        print(f"fit_model {name} features={len(features)}", flush=True)
        models, fit = discrimination.fit_models(features, outcome, partitions, model_config)
        amplitude = forecast.refit_scale_aware_return_calibrator(
            models,
            features,
            outcome,
            partitions["calibration"],
            model_config,
            amplitude_config,
        )
        predictions[name] = discrimination.predict_multivariate(
            features, prediction_dates, models, model_config
        )
        audit[name] = {"fit": fit, "returnAmplitude": amplitude}
    return predictions, audit


def _latest_rows(
    score: pd.DataFrame,
    predictions: dict[str, dict[str, pd.DataFrame]],
    interactions: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    top_count: int,
) -> list[dict[str, Any]]:
    date = score.index.max()
    selected = score.loc[date].dropna().sort_values(ascending=False).head(top_count)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, value) in enumerate(selected.items(), start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "securityId": str(security_id),
            "name": names.get(str(security_id), ""),
            "close": float(panel["close"].loc[date, security_id]),
            "fixedTwelveFactorScore": float(value),
            "interactionRanks": {
                name: float(frame.loc[date, security_id])
                for name, frame in interactions.items()
            },
            "forecasts": {},
            "status": "diagnostic_only_same_fixed_selection_not_an_order",
        }
        for model in ("baseline", "all_interactions"):
            row["forecasts"][model] = {
                "expectedGrossReturn": float(
                    predictions[model]["expectedReturn"].loc[date, security_id]
                ),
                "probabilityUp": float(
                    predictions[model]["grossUp"].loc[date, security_id]
                ),
                "probabilitySevereLoss": float(
                    predictions[model]["severeLoss"].loc[date, security_id]
                ),
            }
        rows.append(row)
    return rows


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PIT fundamental by price-volume interaction study V1",
        "",
        "> Research-only fixed-selection historical rejection study. Not trading.",
        "",
        f"- run: `{result['runId']}`",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- universe: `{result['panelAudit']['acceptedSymbols']}` PIT SH/SZ stocks",
        "- selector: unchanged guarded twelve-factor Top10",
        "- primary candidate: baseline plus all four preregistered interaction ranks",
        f"- verdict: `{result['acceptance']['decision']}`",
        "- eligible for trading: `False`",
        "",
        "## Primary comparison",
        "",
        "| period | scope | model | up AUC | up Brier | tail AUC | return MAE | p(up) spread |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for period in ("audit", "validation", "shadow"):
        for scope in ("overall", "fixedTop10"):
            for model in ("baseline", "all_interactions"):
                value = result["periods"][period][model][scope]
                lines.append(
                    f"| {period} | {scope} | {model} | {value['grossUp']['auc']} | "
                    f"{value['grossUp']['brier']:.8f} | {value['severeLoss']['auc']} | "
                    f"{value['expectedReturnMae']:.8f} | {value['meanProbabilityUpSpread']} |"
                )
    lines.extend(
        [
            "",
            "## Individual interaction ablations",
            "",
            "Individual rows are diagnostics, not a menu from which a historical winner may be selected.",
            "",
            "| period | interaction | overall up AUC | fixed Top10 up AUC | overall tail AUC | fixed Top10 return MAE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for period in ("audit", "validation", "shadow"):
        for name in result["interactionDefinitions"]:
            value = result["periods"][period][name]
            lines.append(
                f"| {period} | {name} | {value['overall']['grossUp']['auc']} | "
                f"{value['fixedTop10']['grossUp']['auc']} | "
                f"{value['overall']['severeLoss']['auc']} | "
                f"{value['fixedTop10']['expectedReturnMae']:.8f} |"
            )
    lines.extend(
        [
            "",
            "All models use identical rows and the same selected Top10. A wider probability range is not evidence unless calibration and discrimination also improve in validation and shadow.",
            "Historical windows are already viewed. Even a pass could only justify a separately preregistered 60-session fresh-forward challenger.",
            "Orders remain `[]`; dashboard, ranking and trading files remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    source_config = load_json(ROOT / config["sourceForecastConfig"])
    forecast.validate_config(source_config)
    guarded_config = load_json(ROOT / source_config["guardedWeightsConfig"])
    guarded.validate_config(guarded_config)
    frozen, _source, _frozen_hash = guarded.load_frozen_config(guarded_config)
    model_config = load_json(ROOT / config["modelTemplate"])
    fundamental_config = load_json(ROOT / config["fundamentalMechanismConfig"])
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, _execution_eligible, _exit_delay = precision.executable_horizon_return(
        panel,
        int(config["data"]["holdingTradingDays"]),
        int(config["data"]["maximumExitDelayTradingDays"]),
    )
    price_ranks, _static_score, factor_audit = rolling.compute_rank_book(panel, frozen)
    daily_ic = rolling.factor_daily_ic(price_ranks, outcome)
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    adaptive_weights, weight_updates = guarded.guarded_weight_path(
        daily_ic, prior, guarded_config
    )
    completion_spec = source_config["forecast"]["missingFeatureCompletion"]
    completed_price, _missing_count, price_support, completion_audit = (
        forecast.complete_rank_book(
            price_ranks,
            panel["eligible"],
            int(config["data"]["maximumImputedPriceFactorsForEligibility"]),
            float(completion_spec["allMissingCrossSectionFallbackRank"]),
        )
    )
    fixed_score = guarded.adaptive_score(
        completed_price, adaptive_weights, panel
    ).where(price_support)
    family_ranks, fundamental_support, fundamental_audit = (
        fundamental_stage.build_family_scores(panel, fundamental_config)
    )
    market_ranks = build_market_context_ranks(panel, config)
    context_support = panel["eligible"].fillna(False)
    for frame in market_ranks.values():
        context_support &= frame.notna()
    common_support = price_support & fundamental_support & context_support
    interactions = build_interaction_ranks(
        family_ranks, market_ranks, config, common_support
    )
    for frame in interactions.values():
        common_support &= frame.notna()
    books = feature_books(completed_price, interactions, common_support)
    fixed_score = fixed_score.where(common_support)
    comparable_outcome = outcome.where(common_support)
    partitions = discrimination.fixed_partitions(panel["close"].index, model_config)
    prediction_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *[set(partitions[name]) for name in ("audit", "validation", "shadow")],
                {panel["close"].index.max()},
            )
        )
    )
    predictions, fit_audit = _fit_and_predict(
        books,
        comparable_outcome,
        partitions,
        prediction_dates,
        model_config,
        source_config["forecast"]["returnAmplitudeCalibration"],
    )
    fixed_selection = discrimination.v6.top_mask(
        fixed_score.reindex(index=prediction_dates), int(config["data"]["topCount"])
    )
    periods: dict[str, Any] = {}
    for period in ("audit", "validation", "shadow"):
        dates = partitions[period].intersection(prediction_dates)
        period_result: dict[str, Any] = {}
        row_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for name, predicted in predictions.items():
            metrics, all_rows, top_rows = increment.evaluate_forecast(
                predicted,
                comparable_outcome,
                dates,
                fixed_selection,
                float(config["data"]["severeLossThreshold"]),
            )
            period_result[name] = metrics
            row_cache[name] = (all_rows, top_rows)
        period_result["pairedLossDiagnostics"] = {
            "overall": increment.paired_loss_diagnostics(
                row_cache["baseline"][0],
                row_cache["all_interactions"][0],
                float(config["data"]["severeLossThreshold"]),
            ),
            "fixedTop10": increment.paired_loss_diagnostics(
                row_cache["baseline"][1],
                row_cache["all_interactions"][1],
                float(config["data"]["severeLossThreshold"]),
            ),
        }
        periods[period] = period_result
    acceptance_input = {
        period: {
            "baseline": periods[period]["baseline"],
            "candidate": periods[period]["all_interactions"],
        }
        for period in periods
    }
    verdict = increment.acceptance(acceptance_input, config)
    latest_rows = _latest_rows(
        fixed_score.reindex(index=prediction_dates),
        predictions,
        interactions,
        panel,
        discrimination.v6.name_map(base),
        int(config["data"]["topCount"]),
    )
    latest_date = panel["close"].index.max()
    identifier = run_id or datetime.now().astimezone().strftime(
        "run_%Y%m%dT%H%M%S%z"
    )
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_historical_rejection_test_not_trading",
        "runId": identifier,
        "generatedAt": datetime.now().astimezone().isoformat(),
        "configSha256": sha256(config_path),
        "dataRange": [
            str(panel["close"].index.min().date()),
            str(panel["close"].index.max().date()),
        ],
        "signalDate": str(latest_date.date()),
        "panelAudit": panel_audit,
        "fundamentalAudit": fundamental_audit,
        "factorAudit": factor_audit,
        "interactionDefinitions": config["hypothesis"]["interactions"],
        "supportAudit": {
            "sameRowsEveryModel": True,
            "sameFixedTop10EveryModel": True,
            "latestPriceSupported": int(price_support.loc[latest_date].sum()),
            "latestFundamentalSupported": int(
                fundamental_support.loc[latest_date].sum()
            ),
            "latestContextSupported": int(context_support.loc[latest_date].sum()),
            "latestCommonSupported": int(common_support.loc[latest_date].sum()),
            "priceCompletion": completion_audit,
        },
        "featureAudit": {
            "bookFeatureCounts": {name: len(value) for name, value in books.items()},
            "strictlyPastOrSameDateOnly": True,
            "amountSurpriseDenominatorEndsAtPreviousSession": True,
            "reportDateUsedForAvailability": False,
            "outcomesUsedToConstructInteractions": False,
        },
        "fitAudit": fit_audit,
        "weightUpdateCount": int(len(weight_updates)),
        "partitions": {
            name: [str(values.min().date()), str(values.max().date()), len(values)]
            for name, values in partitions.items()
        },
        "periods": periods,
        "acceptance": verdict,
        "multipleTesting": {
            "newHistoricalTrials": int(config["training"]["newHistoricalTrials"]),
            "priorFundamentalIntegrationTrialsCharged": int(
                config["training"]["priorFundamentalIntegrationTrialsCharged"]
            ),
            "totalFundamentalIntegrationTrials": int(
                config["training"]["newHistoricalTrials"]
                + config["training"]["priorFundamentalIntegrationTrialsCharged"]
            ),
            "individualAblationsMaySelectWinner": False,
            "pbo": config["evaluation"]["pboStatus"],
            "dsr": config["evaluation"]["dsrStatus"],
        },
        "latestFixedTop10Comparison": latest_rows,
        "dashboardChanged": False,
        "rankingChanged": False,
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    output = ROOT / config["output"]["root"] / identifier
    discrimination.atomic_text(
        output / "result.json",
        json.dumps(discrimination.safe(result), ensure_ascii=False, indent=2) + "\n",
    )
    discrimination.atomic_text(output / "report.md", report_markdown(result))
    return {
        "runId": identifier,
        "report": str(output / "report.md"),
        "decision": verdict["decision"],
        "historicalHypothesisPass": verdict["historicalHypothesisPass"],
        "eligibleForTrading": False,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(
        json.dumps(
            discrimination.safe(run(args.config.resolve(), args.run_id)),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
