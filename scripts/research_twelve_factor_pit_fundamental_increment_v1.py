#!/usr/bin/env python3
"""Audit PIT fundamental increments to the current complete twelve-factor forecast.

This is a forecast-only, fixed-selection historical rejection study.  It cannot publish
the dashboard, change a ranking, create an order, or touch any trading configuration.
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
import research_twelve_factor_rank_discrimination_v1 as rank_audit  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "twelve_factor_pit_fundamental_increment_v1.json"
)
SCHEMA_VERSION = "twelve_factor_pit_fundamental_increment_result_v1"
CODE_VERSION = "twelve_factor_pit_fundamental_increment_v1_20260813"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_pit_fundamental_increment_v1":
        raise ValueError("unexpected fundamental-increment schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("fundamental increment must remain research/shadow-only")
    for path_key, hash_key in (
        ("sourceForecastConfig", "sourceForecastConfigSha256"),
        ("modelTemplate", "modelTemplateSha256"),
        ("fundamentalMechanismConfig", "fundamentalMechanismConfigSha256"),
    ):
        if sha256(ROOT / config[path_key]) != str(config[hash_key]).lower():
            raise ValueError(f"frozen dependency changed: {path_key}")
    hypothesis = config["hypothesis"]
    expected_families = [
        "earnings_innovation",
        "growth_acceleration",
        "quality",
        "cash_flow_quality",
    ]
    if hypothesis.get("fundamentalFamilies") != expected_families:
        raise ValueError("the four preregistered fundamental families changed")
    if hypothesis.get("historicalOutcomeMaySelectFamily") is not False:
        raise ValueError("historical outcomes may not select a family")
    if hypothesis.get("historicalOutcomeMayFitFamilyWeight") is not False:
        raise ValueError("historical outcomes may not fit family weights")
    if hypothesis.get("hyperparameterSearchAllowed") is not False:
        raise ValueError("historical hyperparameter search is forbidden")
    data = config["data"]
    if (
        int(data["priceFactorCount"]) != 12
        or int(data["fundamentalFamilyCount"]) != 4
        or int(data["candidateFeatureCount"]) != 16
    ):
        raise ValueError("the study requires twelve price and four fundamental features")
    if data.get("availabilityRule") != (
        "first_market_date_strictly_after_max_notice_update"
    ):
        raise ValueError("fundamental availability rule must stay strictly causal")
    if data.get("reportDateMayDetermineAvailability") is not False:
        raise ValueError("reportDate may never determine information availability")
    if int(data["maximumImputedPriceFactorsForEligibility"]) > 2:
        raise ValueError("price-factor completion support was widened")
    training = config["training"]
    if training.get("sameRowsForBaselineAndCandidate") is not True:
        raise ValueError("baseline and candidate must use identical rows")
    if training.get("sameTop10ForForecastComparison") is not True:
        raise ValueError("forecast comparison must hold the selected Top10 fixed")
    if training.get("probabilityStretchingAllowed") is not False:
        raise ValueError("mechanical probability stretching is forbidden")
    evaluation = config["evaluation"]
    if evaluation.get("sameSupportAndSameSelectionRequired") is not True:
        raise ValueError("same-support fixed-selection comparison is required")
    if evaluation.get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical reuse must be explicit")
    safety = config.get("safety", {})
    if any(bool(value) for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all mutation and trading permissions must remain false")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def combine_feature_books(
    price_ranks: dict[str, pd.DataFrame],
    family_ranks: dict[str, pd.DataFrame],
    support: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return same-support baseline and candidate books without temporal operations."""
    if len(price_ranks) != 12:
        raise ValueError("baseline feature book must contain exactly twelve factors")
    expected = {
        "earnings_innovation",
        "growth_acceleration",
        "quality",
        "cash_flow_quality",
    }
    if set(family_ranks) != expected:
        raise ValueError("candidate feature book must contain all four families")
    mask = support.fillna(False).astype(bool)
    baseline = {key: value.where(mask) for key, value in price_ranks.items()}
    candidate = dict(baseline)
    candidate.update(
        {f"fundamental/{key}": value.where(mask) for key, value in family_ranks.items()}
    )
    return baseline, candidate


def _stack_predictions(
    predictions: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    selection: pd.DataFrame | None = None,
) -> pd.DataFrame:
    parts: dict[str, pd.Series] = {}
    for name in ("expectedReturn", "grossUp", "severeLoss"):
        frame = predictions[name].reindex(index=dates)
        if selection is not None:
            frame = frame.where(selection.reindex(index=dates).fillna(False))
        parts[name] = frame.stack(future_stack=True)
    outcome = returns.reindex(index=dates)
    if selection is not None:
        outcome = outcome.where(selection.reindex(index=dates).fillna(False))
    parts["return"] = outcome.stack(future_stack=True)
    return pd.concat(parts, axis=1).dropna().reset_index(names=["date", "securityId"])


def _metric_block(rows: pd.DataFrame, severe_threshold: float) -> dict[str, Any]:
    actual = rows["return"].to_numpy(dtype=float)
    p_up = rows["grossUp"].to_numpy(dtype=float)
    p_tail = rows["severeLoss"].to_numpy(dtype=float)
    spread = rows.groupby("date")["grossUp"].agg(lambda values: values.max() - values.min())
    return {
        "rows": int(len(rows)),
        "days": int(rows["date"].nunique()),
        "grossUp": rank_audit.probability_metrics(actual > 0.0, p_up),
        "severeLoss": rank_audit.probability_metrics(
            actual <= severe_threshold, p_tail
        ),
        "expectedReturnMae": float(
            np.mean(np.abs(rows["expectedReturn"].to_numpy(dtype=float) - actual))
        ),
        "actualMeanGrossReturn": float(actual.mean()),
        "actualGrossUpRate": float((actual > 0.0).mean()),
        "actualSevereLossRate": float((actual <= severe_threshold).mean()),
        "meanProbabilityUpSpread": float(spread.mean()) if len(spread) else None,
    }


def evaluate_forecast(
    predictions: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    fixed_selection: pd.DataFrame,
    severe_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    overall = _stack_predictions(predictions, returns, dates)
    selected = _stack_predictions(predictions, returns, dates, fixed_selection)
    return {
        "overall": _metric_block(overall, severe_threshold),
        "fixedTop10": _metric_block(selected, severe_threshold),
    }, overall, selected


def _binary_log_loss(actual: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    actual = np.asarray(actual, dtype=float)
    return -(actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability))


def paired_loss_diagnostics(
    baseline_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    severe_threshold: float,
) -> dict[str, Any]:
    keys = ["date", "securityId", "return"]
    merged = baseline_rows.merge(
        candidate_rows,
        on=keys,
        suffixes=("Baseline", "Candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(baseline_rows) or len(merged) != len(candidate_rows):
        raise RuntimeError("baseline and candidate forecast rows diverged")
    up = merged["return"].gt(0.0).to_numpy(dtype=float)
    tail = merged["return"].le(severe_threshold).to_numpy(dtype=float)
    losses = pd.DataFrame(
        {
            "date": pd.to_datetime(merged["date"]),
            "grossUpBrier": np.square(merged["grossUpCandidate"] - up)
            - np.square(merged["grossUpBaseline"] - up),
            "grossUpLogLoss": _binary_log_loss(up, merged["grossUpCandidate"])
            - _binary_log_loss(up, merged["grossUpBaseline"]),
            "severeLossBrier": np.square(merged["severeLossCandidate"] - tail)
            - np.square(merged["severeLossBaseline"] - tail),
            "expectedReturnAbsoluteError": np.abs(
                merged["expectedReturnCandidate"] - merged["return"]
            )
            - np.abs(merged["expectedReturnBaseline"] - merged["return"]),
        }
    )
    daily = losses.groupby("date", sort=True).mean(numeric_only=True)
    output: dict[str, Any] = {
        "interpretation": "negative_candidate_minus_baseline_is_improvement",
        "pairedRows": int(len(merged)),
        "pairedDays": int(len(daily)),
    }
    for name in daily.columns:
        values = daily[name].to_numpy(dtype=float)
        output[name] = {
            "meanCandidateMinusBaseline": float(values.mean()),
            "neweyWestT": rank_audit.newey_west_t(values, 5),
        }
    return output


def _majority_probability_improved(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[int, dict[str, bool]]:
    checks = {
        "auc": candidate["auc"] is not None
        and baseline["auc"] is not None
        and candidate["auc"] > baseline["auc"],
        "brier": candidate["brier"] < baseline["brier"],
        "logLoss": candidate["logLoss"] < baseline["logLoss"],
    }
    return sum(checks.values()), checks


def acceptance(periods: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for period in config["evaluation"]["primaryRejectOnlyPeriods"]:
        baseline = periods[period]["baseline"]
        candidate = periods[period]["candidate"]
        gross_count, gross_checks = _majority_probability_improved(
            baseline["overall"]["grossUp"], candidate["overall"]["grossUp"]
        )
        tail_count, tail_checks = _majority_probability_improved(
            baseline["overall"]["severeLoss"], candidate["overall"]["severeLoss"]
        )
        top_count, top_checks = _majority_probability_improved(
            baseline["fixedTop10"]["grossUp"],
            candidate["fixedTop10"]["grossUp"],
        )
        item = {
            "overallGrossUpMajority": gross_count >= 2,
            "overallGrossUpDetails": gross_checks,
            "overallSevereLossMajority": tail_count >= 2,
            "overallSevereLossDetails": tail_checks,
            "fixedTop10GrossUpMajority": top_count >= 2,
            "fixedTop10GrossUpDetails": top_checks,
            "overallExpectedReturnMaeImproved": candidate["overall"][
                "expectedReturnMae"
            ]
            < baseline["overall"]["expectedReturnMae"],
            "fixedTop10ExpectedReturnMaeImproved": candidate["fixedTop10"][
                "expectedReturnMae"
            ]
            < baseline["fixedTop10"]["expectedReturnMae"],
            "sameOverallRows": candidate["overall"]["rows"]
            == baseline["overall"]["rows"],
            "sameFixedTop10Rows": candidate["fixedTop10"]["rows"]
            == baseline["fixedTop10"]["rows"],
        }
        item["passed"] = all(
            value for key, value in item.items() if not key.endswith("Details")
        )
        checks[period] = item
    passed = all(item["passed"] for item in checks.values())
    return {
        "externalRejectOnlyChecks": checks,
        "historicalHypothesisPass": passed,
        "decision": (
            "merits_separate_fresh_forward_preregistration_only"
            if passed
            else "reject_fundamental_increment_for_current_forecast"
        ),
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
    }


def latest_fixed_top10(
    selection_score: pd.DataFrame,
    baseline: dict[str, pd.DataFrame],
    candidate: dict[str, pd.DataFrame],
    family_ranks: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    top_count: int,
) -> list[dict[str, Any]]:
    date = selection_score.index.max()
    selected = selection_score.loc[date].dropna().sort_values(ascending=False).head(top_count)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, score) in enumerate(selected.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": float(panel["close"].loc[date, security_id]),
                "fixedTwelveFactorScore": float(score),
                "baseline": {
                    "expectedGrossReturn": float(
                        baseline["expectedReturn"].loc[date, security_id]
                    ),
                    "probabilityUp": float(baseline["grossUp"].loc[date, security_id]),
                    "probabilitySevereLoss": float(
                        baseline["severeLoss"].loc[date, security_id]
                    ),
                },
                "candidate": {
                    "expectedGrossReturn": float(
                        candidate["expectedReturn"].loc[date, security_id]
                    ),
                    "probabilityUp": float(candidate["grossUp"].loc[date, security_id]),
                    "probabilitySevereLoss": float(
                        candidate["severeLoss"].loc[date, security_id]
                    ),
                },
                "fundamentalFamilyRanks": {
                    name: float(frame.loc[date, security_id])
                    for name, frame in family_ranks.items()
                },
                "status": "diagnostic_only_same_fixed_selection_not_an_order",
            }
        )
    return rows


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Twelve-factor forecast plus PIT fundamentals V1",
        "",
        "> Research-only fixed-selection rejection study. No ranking or trading output was changed.",
        "",
        f"- run: `{result['runId']}`",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- universe: `{result['panelAudit']['acceptedSymbols']}` PIT SH/SZ stocks",
        f"- causal statement events: `{result['fundamentalAudit']['statementEvents']}`",
        "- baseline: 12 complete price-factor ranks",
        "- candidate: the same 12 ranks plus 4 PIT fundamental-family ranks",
        "- selection: held fixed to the current guarded twelve-factor score",
        f"- verdict: `{result['acceptance']['decision']}`",
        "- eligible for trading: `False`",
        "",
        "## Forecast comparison",
        "",
        "| period | scope | model | up AUC | up Brier | up LogLoss | up ECE | tail AUC | return MAE | p(up) spread |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("audit", "validation", "shadow"):
        for scope in ("overall", "fixedTop10"):
            for model in ("baseline", "candidate"):
                value = result["periods"][period][model][scope]
                up = value["grossUp"]
                tail = value["severeLoss"]
                lines.append(
                    f"| {period} | {scope} | {model} | {up['auc']} | {up['brier']:.8f} | "
                    f"{up['logLoss']:.8f} | {up['ece']:.8f} | {tail['auc']} | "
                    f"{value['expectedReturnMae']:.8f} | {value['meanProbabilityUpSpread']} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The same stocks, dates and current Top10 are used for both forecast heads. Any metric change therefore comes from the four fundamental inputs, not from skipping names or difficult sessions.",
            "Historical audit, validation and shadow windows have already been viewed. They can reject this increment but cannot validate or promote it. A pass only permits a separately preregistered 60-session fresh-forward challenger.",
            "PBO and DSR are not reported because this run evaluates one preregistered forecast challenger and does not change realised selections or returns. Multiple testing is controlled by a single fixed candidate and simultaneous validation-and-shadow gates.",
            "",
            "Orders remain `[]`; dashboard and trading files remain unchanged.",
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
    completed_price, missing_count, price_support, completion_audit = (
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
    common_support = price_support & fundamental_support
    baseline_features, candidate_features = combine_feature_books(
        completed_price, family_ranks, common_support
    )
    fixed_score = fixed_score.where(common_support)
    comparable_outcome = outcome.where(common_support)
    partitions = discrimination.fixed_partitions(panel["close"].index, model_config)
    baseline_models, baseline_fit = discrimination.fit_models(
        baseline_features, comparable_outcome, partitions, model_config
    )
    candidate_models, candidate_fit = discrimination.fit_models(
        candidate_features, comparable_outcome, partitions, model_config
    )
    amplitude = source_config["forecast"]["returnAmplitudeCalibration"]
    baseline_amplitude = forecast.refit_scale_aware_return_calibrator(
        baseline_models,
        baseline_features,
        comparable_outcome,
        partitions["calibration"],
        model_config,
        amplitude,
    )
    candidate_amplitude = forecast.refit_scale_aware_return_calibrator(
        candidate_models,
        candidate_features,
        comparable_outcome,
        partitions["calibration"],
        model_config,
        amplitude,
    )
    prediction_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *[set(partitions[name]) for name in ("audit", "validation", "shadow")],
                {panel["close"].index.max()},
            )
        )
    )
    baseline_predictions = discrimination.predict_multivariate(
        baseline_features, prediction_dates, baseline_models, model_config
    )
    candidate_predictions = discrimination.predict_multivariate(
        candidate_features, prediction_dates, candidate_models, model_config
    )
    fixed_selection = discrimination.v6.top_mask(
        fixed_score.reindex(index=prediction_dates), int(config["data"]["topCount"])
    )
    periods: dict[str, Any] = {}
    for period in ("audit", "validation", "shadow"):
        dates = partitions[period].intersection(prediction_dates)
        baseline_metrics, baseline_all, baseline_top = evaluate_forecast(
            baseline_predictions,
            comparable_outcome,
            dates,
            fixed_selection,
            float(config["data"]["severeLossThreshold"]),
        )
        candidate_metrics, candidate_all, candidate_top = evaluate_forecast(
            candidate_predictions,
            comparable_outcome,
            dates,
            fixed_selection,
            float(config["data"]["severeLossThreshold"]),
        )
        periods[period] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "pairedLossDiagnostics": {
                "overall": paired_loss_diagnostics(
                    baseline_all,
                    candidate_all,
                    float(config["data"]["severeLossThreshold"]),
                ),
                "fixedTop10": paired_loss_diagnostics(
                    baseline_top,
                    candidate_top,
                    float(config["data"]["severeLossThreshold"]),
                ),
            },
        }
    verdict = acceptance(periods, config)
    latest_rows = latest_fixed_top10(
        fixed_score.reindex(index=prediction_dates),
        baseline_predictions,
        candidate_predictions,
        family_ranks,
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
        "supportAudit": {
            "baselineAndCandidateUseSameRows": True,
            "selectionHeldFixed": True,
            "latestPriceSupported": int(price_support.loc[latest_date].sum()),
            "latestFourFamilySupported": int(fundamental_support.loc[latest_date].sum()),
            "latestCommonSupported": int(common_support.loc[latest_date].sum()),
            "priceCompletion": completion_audit,
        },
        "featureAudit": {
            "baselineFeatureCount": len(baseline_features),
            "candidateFeatureCount": len(candidate_features),
            "candidateAddedFeatures": [
                key for key in candidate_features if key not in baseline_features
            ],
            "availabilityRule": config["data"]["availabilityRule"],
            "reportDateUsedForAvailability": False,
            "pastOnly": True,
        },
        "fitAudit": {
            "baseline": baseline_fit,
            "candidate": candidate_fit,
            "baselineReturnAmplitude": baseline_amplitude,
            "candidateReturnAmplitude": candidate_amplitude,
            "weightUpdateCount": int(len(weight_updates)),
        },
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
