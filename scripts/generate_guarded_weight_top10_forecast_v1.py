#!/usr/bin/env python3
"""Generate a research-only latest Top10 from guarded twelve-factor weights."""

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
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402
import stock_forecast_dashboard as dashboard  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_guarded_top10_forecast_v2.json"
)
SCHEMA_VERSION = "twelve_factor_guarded_top10_forecast_result_v2"
CODE_VERSION = "twelve_factor_guarded_top10_forecast_v2_20260812"


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
    if config.get("schemaVersion") != "twelve_factor_guarded_top10_forecast_v2":
        raise ValueError("unexpected guarded Top10 forecast schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("guarded Top10 forecasts must remain research/shadow-only")
    for path_key, hash_key in (
        ("guardedWeightsConfig", "guardedWeightsConfigSha256"),
        ("forecastModelTemplate", "forecastModelTemplateSha256"),
    ):
        if file_sha256(ROOT / config[path_key]) != str(config[hash_key]).lower():
            raise ValueError(f"frozen dependency changed: {path_key}")
    selection = config["selection"]
    if int(selection["factorCount"]) != 12 or int(selection["topCount"]) != 10:
        raise ValueError("forecast must use the frozen twelve-factor Top10")
    if selection.get("selectionNeverUsesForecastOrFutureOutcome") is not True:
        raise ValueError("selection must not use a forecast or future outcome")
    maximum_imputed = int(selection["maximumImputedFactorsForEligibility"])
    if not 0 <= maximum_imputed < int(selection["factorCount"]):
        raise ValueError("maximum imputed factor count must be bounded")
    forecast = config["forecast"]
    completion = forecast["missingFeatureCompletion"]
    if completion.get("method") != "same_date_cross_sectional_median_rank":
        raise ValueError("factor completion must use the preregistered PIT-neutral method")
    if float(completion.get("allMissingCrossSectionFallbackRank")) != 0.5:
        raise ValueError("all-missing factor completion must remain neutral")
    if completion.get("strictlyPastOrSameDateOnly") is not True:
        raise ValueError("factor completion must remain point-in-time")
    if completion.get("samePolicyForFitCalibrationAndPrediction") is not True:
        raise ValueError("fit and prediction must share one completion policy")
    if completion.get("missingnessMayCreatePositiveOrNegativeSignal") is not False:
        raise ValueError("missingness cannot become a directional signal")
    if completion.get("scalarFallbackAllowed") is not False:
        raise ValueError("the complete twelve-factor policy forbids scalar fallback")
    amplitude = forecast["returnAmplitudeCalibration"]
    if amplitude.get("method") != "calibration_period_zscore_then_fixed_ridge":
        raise ValueError("return amplitude must use the preregistered scale-aware method")
    if float(amplitude.get("ridgeAlpha")) <= 0.0:
        raise ValueError("return amplitude ridge alpha must be positive")
    if amplitude.get("standardizationUsesCalibrationPeriodOnly") is not True:
        raise ValueError("return amplitude standardization must remain train-only")
    if amplitude.get("mechanicalOutputStretchingAllowed") is not False:
        raise ValueError("mechanical forecast stretching is forbidden")
    if forecast.get("hyperparameterSearchAllowed") is not False:
        raise ValueError("forecast hyperparameter search is forbidden")
    if forecast.get("historicalRunCanPromote") is not False:
        raise ValueError("historical forecast cannot promote")
    safety = config.get("safety", {})
    if any(bool(value) for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading and mutation permissions must remain false")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def forecast_reliability(periods: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for period in ("validation", "shadow"):
        probability = periods[period]["probability"]
        checks[period] = {
            "grossUpAucAtLeast052": probability["grossUp"]["auc"] is not None
            and probability["grossUp"]["auc"] >= 0.52,
            "severeLossAucAtLeast060": probability["severeLoss"]["auc"] is not None
            and probability["severeLoss"]["auc"] >= 0.60,
        }
        checks[period]["passed"] = all(checks[period].values())
    reliable = all(item["passed"] for item in checks.values())
    return {
        "status": (
            "historically_discriminative_but_fresh_forward_required"
            if reliable
            else "low_confidence_diagnostic_estimates_only"
        ),
        "checks": checks,
        "eligibleForTrading": False,
    }


def complete_rank_book(
    ranks: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    maximum_imputed_factors: int,
    neutral_fallback_rank: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Complete factor ranks with same-date neutral values and bounded support.

    The operation is cross-sectional and date-local: changing a later date cannot
    revise an earlier completed feature. Missingness itself is never a signed input.
    """
    if not ranks:
        raise ValueError("cannot complete an empty rank book")
    if not 0.0 <= neutral_fallback_rank <= 1.0:
        raise ValueError("neutral fallback rank must be inside [0,1]")
    mask = eligible.fillna(False).astype(bool)
    missing_count = pd.DataFrame(
        0, index=mask.index, columns=mask.columns, dtype=np.int16
    )
    completed: dict[str, pd.DataFrame] = {}
    for key, frame in ranks.items():
        observed = frame.reindex_like(mask).where(mask).astype(float)
        missing = mask & observed.isna()
        missing_count = missing_count.add(missing.astype(np.int16)).astype(np.int16)
        same_date_median = (
            observed.median(axis=1, skipna=True)
            .fillna(float(neutral_fallback_rank))
            .clip(0.0, 1.0)
        )
        filled = observed.T.fillna(same_date_median).T.where(mask)
        completed[key] = filled.astype(np.float32)
    supported = mask & missing_count.le(int(maximum_imputed_factors))
    completed = {key: frame.where(supported) for key, frame in completed.items()}
    latest = mask.index.max()
    latest_eligible = mask.loc[latest]
    latest_supported = supported.loc[latest]
    latest_counts = missing_count.loc[latest].where(latest_eligible).dropna()
    audit = {
        "method": "same_date_cross_sectional_median_rank",
        "factorCount": len(completed),
        "maximumImputedFactorsForEligibility": int(maximum_imputed_factors),
        "neutralFallbackRank": float(neutral_fallback_rank),
        "latestEligibleSecurities": int(latest_eligible.sum()),
        "latestSupportedSecurities": int(latest_supported.sum()),
        "latestExcludedForExcessMissing": int((latest_eligible & ~latest_supported).sum()),
        "latestMissingCountDistribution": {
            str(int(key)): int(value)
            for key, value in latest_counts.value_counts().sort_index().items()
        },
        "strictlyPastOrSameDateOnly": True,
        "missingnessUsedAsDirectionalFeature": False,
    }
    return completed, missing_count, supported, audit


class StandardizedReturnCalibrator:
    """A fixed scale-aware calibration adapter fit on calibration rows only."""

    def __init__(self, mean: float, std: float, model: Ridge) -> None:
        self.mean = float(mean)
        self.std = float(std)
        self.model = model

    def predict(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.model.predict((raw - self.mean) / self.std)


def refit_scale_aware_return_calibrator(
    models: discrimination.Models,
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    calibration_dates: pd.DatetimeIndex,
    model_config: dict[str, Any],
    amplitude_config: dict[str, Any],
) -> dict[str, Any]:
    """Prevent one-dimensional Ridge from collapsing because returns are ~1e-3."""
    x_cal, y_cal, weights, used_days = discrimination.stack_features(
        ranks,
        returns,
        calibration_dates,
        0,
        int(model_config["training"]["deterministicSeed"]),
    )
    raw = models.return_base.predict(x_cal).reshape(-1, 1)
    weight_sum = float(weights.sum())
    mean = float(np.sum(weights * raw[:, 0]) / weight_sum)
    variance = float(np.sum(weights * np.square(raw[:, 0] - mean)) / weight_sum)
    std = math.sqrt(max(variance, 0.0))
    minimum = float(amplitude_config["minimumRawPredictionStd"])
    if not math.isfinite(std) or std < minimum:
        raise RuntimeError(
            f"raw return prediction has insufficient calibration dispersion: {std}"
        )
    standardized = (raw - mean) / std
    calibrated = Ridge(alpha=float(amplitude_config["ridgeAlpha"]))
    calibrated.fit(standardized, y_cal, sample_weight=weights)
    models.return_calibrator = StandardizedReturnCalibrator(mean, std, calibrated)
    return {
        "method": str(amplitude_config["method"]),
        "calibrationRows": int(len(y_cal)),
        "calibrationDays": int(used_days),
        "rawPredictionMean": mean,
        "rawPredictionStd": std,
        "ridgeAlpha": float(amplitude_config["ridgeAlpha"]),
        "calibratedCoefficient": float(np.ravel(calibrated.coef_)[0]),
        "calibratedIntercept": float(calibrated.intercept_),
        "standardizationUsesCalibrationPeriodOnly": True,
    }


def latest_forecast_rows(
    score: pd.DataFrame,
    multivariate: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    missing_factor_count: pd.DataFrame,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    date = score.index.max()
    selected = score.loc[date].dropna().sort_values(ascending=False)
    if limit is not None:
        selected = selected.head(limit)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, factor_score) in enumerate(selected.items(), start=1):
        multi_expected = multivariate["expectedReturn"].loc[date, security_id]
        if not np.isfinite(multi_expected):
            raise RuntimeError(
                f"complete twelve-factor forecast missing for selected {security_id}"
            )
        imputed_count = int(missing_factor_count.loc[date, security_id])
        rows.append(
            {
                "rank": rank,
                "signalDate": date.date().isoformat(),
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": round(float(panel["close"].loc[date, security_id]), 4),
                "adaptiveFactorScore": round(float(factor_score), 8),
                "expectedGrossReturn": round(
                    float(multivariate["expectedReturn"].loc[date, security_id]), 8
                ),
                "probabilityUp": round(
                    float(multivariate["grossUp"].loc[date, security_id]), 8
                ),
                "probabilitySevereLoss": round(
                    float(multivariate["severeLoss"].loc[date, security_id]), 8
                ),
                "factorCount": 12,
                "imputedFactorCount": imputed_count,
                "factorCompletion": (
                    "none_observed_all_twelve"
                    if imputed_count == 0
                    else "same_date_cross_sectional_median_rank"
                ),
                "estimateSource": "twelve_rank_multivariate_calibrated",
                "status": "diagnostic_only_not_an_order",
            }
        )
    return rows


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Guarded Twelve-Factor Latest Top10 Forecast",
        "",
        f"- Signal date: `{result['signalDate']}`",
        f"- Intended session: `{result['intendedTradingSession']}`",
        f"- Reliability: `{result['forecastReliability']['status']}`",
        "- Ranking: frozen guarded twelve-factor score only",
        "- Forecast horizon: next buyable open to following sellable open",
        "- Tail loss: executable gross return <= -3%",
        "- Research-only; not an order; no guarantee of profit",
        "",
        "| Rank | Security | Name | Close | Score | Expected gross | P(up) | P(tail loss) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["latestTop10"]:
        lines.append(
            f"| {row['rank']} | {row['securityId']} | {row['name']} | {row['close']} | "
            f"{row['adaptiveFactorScore']:.4f} | {row['expectedGrossReturn']:.3%} | "
            f"{row['probabilityUp']:.2%} | {row['probabilitySevereLoss']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The probabilities are calibrated historical estimates. When the validation or shadow AUC gate fails, they remain descriptive diagnostics and must not be treated as trade eligibility.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    guarded_config = load_json(ROOT / config["guardedWeightsConfig"])
    guarded.validate_config(guarded_config)
    frozen, source, _ = guarded.load_frozen_config(guarded_config)
    model_config = load_json(ROOT / config["forecastModelTemplate"])
    ablation_config = load_json(ROOT / model_config["frozenAblationConfig"])
    _factors, _weights, _v6_config = discrimination.ablation.frozen_factors_and_weights(
        ablation_config
    )
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, _execution_eligible, _exit_delay = precision.executable_horizon_return(
        panel, 1, 5
    )
    ranks, _static_score, factor_audit = rolling.compute_rank_book(panel, frozen)
    daily_ic = rolling.factor_daily_ic(ranks, outcome)
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    adaptive_weights, weight_updates = guarded.guarded_weight_path(
        daily_ic, prior, guarded_config
    )
    completion = config["forecast"]["missingFeatureCompletion"]
    completed_ranks, missing_factor_count, completion_support, completion_audit = (
        complete_rank_book(
            ranks,
            panel["eligible"],
            int(config["selection"]["maximumImputedFactorsForEligibility"]),
            float(completion["allMissingCrossSectionFallbackRank"]),
        )
    )
    score = guarded.adaptive_score(completed_ranks, adaptive_weights, panel).where(
        completion_support
    )
    partitions = discrimination.fixed_partitions(panel["close"].index, model_config)
    models, fit_audit = discrimination.fit_models(
        completed_ranks, outcome.where(completion_support), partitions, model_config
    )
    amplitude_audit = refit_scale_aware_return_calibrator(
        models,
        completed_ranks,
        outcome.where(completion_support),
        partitions["calibration"],
        model_config,
        config["forecast"]["returnAmplitudeCalibration"],
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
    multivariate = discrimination.predict_multivariate(
        completed_ranks, prediction_dates, models, model_config
    )
    periods = {
        name: discrimination.evaluate_period(
            multivariate, outcome, partitions[name].intersection(prediction_dates), model_config
        )
        for name in ("audit", "validation", "shadow")
    }
    latest_date = panel["close"].index.max()
    intended = latest_date + pd.offsets.BDay(1)
    latest_all_forecasts = latest_forecast_rows(
        score,
        multivariate,
        panel,
        discrimination.v6.name_map(base),
        missing_factor_count,
    )
    expected_values = np.asarray(
        [row["expectedGrossReturn"] for row in latest_all_forecasts], dtype=float
    )
    probability_up_values = np.asarray(
        [row["probabilityUp"] for row in latest_all_forecasts], dtype=float
    )
    probability_tail_values = np.asarray(
        [row["probabilitySevereLoss"] for row in latest_all_forecasts], dtype=float
    )
    latest_forecast_distribution = {
        "securityCount": int(len(expected_values)),
        "distinctExpectedReturnsAtEightDecimals": int(
            len(np.unique(expected_values))
        ),
        "expectedReturnCrossSectionStd": float(np.std(expected_values, ddof=0)),
        "expectedReturnMinimum": float(np.min(expected_values)),
        "expectedReturnMaximum": float(np.max(expected_values)),
        "expectedReturnSpread": float(np.max(expected_values) - np.min(expected_values)),
        "probabilityUpMinimum": float(np.min(probability_up_values)),
        "probabilityUpMaximum": float(np.max(probability_up_values)),
        "probabilityUpSpread": float(
            np.max(probability_up_values) - np.min(probability_up_values)
        ),
        "probabilityTailLossMinimum": float(np.min(probability_tail_values)),
        "probabilityTailLossMaximum": float(np.max(probability_tail_values)),
        "probabilityTailLossSpread": float(
            np.max(probability_tail_values) - np.min(probability_tail_values)
        ),
        "mechanicallyStretched": False,
    }
    top10_expected = expected_values[: int(config["selection"]["topCount"])]
    top10_probability_up = probability_up_values[
        : int(config["selection"]["topCount"])
    ]
    top10_probability_tail = probability_tail_values[
        : int(config["selection"]["topCount"])
    ]
    latest_forecast_distribution.update(
        {
            "top10ExpectedReturnSpread": float(
                np.max(top10_expected) - np.min(top10_expected)
            ),
            "top10ProbabilityUpSpread": float(
                np.max(top10_probability_up) - np.min(top10_probability_up)
            ),
            "top10ProbabilityTailLossSpread": float(
                np.max(top10_probability_tail) - np.min(top10_probability_tail)
            ),
        }
    )
    top10 = latest_all_forecasts[: int(config["selection"]["topCount"])]
    reliability = forecast_reliability(periods)
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "runId": run_id
        or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z"),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            latest_date.date().isoformat(),
        ],
        "signalDate": latest_date.date().isoformat(),
        "intendedTradingSession": intended.date().isoformat(),
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "factorCompletionAudit": completion_audit,
        "fitAudit": fit_audit,
        "returnAmplitudeCalibrationAudit": amplitude_audit,
        "latestForecastDistribution": latest_forecast_distribution,
        "weightUpdateCount": int(len(weight_updates)),
        "latestWeights": safe(
            adaptive_weights.loc[latest_date].to_dict()
        ),
        "periodDiagnostics": periods,
        "forecastReliability": reliability,
        "latestTop10": top10,
        "latestAllForecastCount": len(latest_all_forecasts),
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    root = ROOT / config["output"]["root"] / result["runId"]
    precision.atomic_write(
        root / "result.json", json.dumps(safe(result), ensure_ascii=False, indent=2) + "\n"
    )
    precision.atomic_write(root / "report.md", markdown_report(result))
    precision.atomic_write(
        root / "latest_top10.csv",
        pd.DataFrame(top10).to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        root / "latest_all_forecasts.csv",
        pd.DataFrame(latest_all_forecasts).to_csv(index=False, lineterminator="\n"),
    )
    dashboard.publish_snapshot(
        result,
        latest_all_forecasts,
        source_result_path=root / "result.json",
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
                "forecastReliability": result["forecastReliability"]["status"],
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
