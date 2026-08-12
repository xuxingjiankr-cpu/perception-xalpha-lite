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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_guarded_top10_forecast_v1.json"
)
SCHEMA_VERSION = "twelve_factor_guarded_top10_forecast_result_v1"
CODE_VERSION = "twelve_factor_guarded_top10_forecast_v1_20260812"


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
    if config.get("schemaVersion") != "twelve_factor_guarded_top10_forecast_v1":
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
    forecast = config["forecast"]
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


def latest_forecast_rows(
    score: pd.DataFrame,
    multivariate: dict[str, pd.DataFrame],
    scalar: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    date = score.index.max()
    selected = score.loc[date].dropna().sort_values(ascending=False)
    if limit is not None:
        selected = selected.head(limit)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, factor_score) in enumerate(selected.items(), start=1):
        multi_expected = multivariate["expectedReturn"].loc[date, security_id]
        use_multi = np.isfinite(multi_expected)
        source = multivariate if use_multi else scalar
        rows.append(
            {
                "rank": rank,
                "signalDate": date.date().isoformat(),
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": round(float(panel["close"].loc[date, security_id]), 4),
                "adaptiveFactorScore": round(float(factor_score), 8),
                "expectedGrossReturn": round(
                    float(source["expectedReturn"].loc[date, security_id]), 8
                ),
                "probabilityUp": round(float(source["grossUp"].loc[date, security_id]), 8),
                "probabilitySevereLoss": round(
                    float(source["severeLoss"].loc[date, security_id]), 8
                ),
                "estimateSource": (
                    "twelve_rank_multivariate_calibrated"
                    if use_multi
                    else "guarded_score_scalar_calibrated_fallback"
                ),
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
    _factors, _weights, v6_config = discrimination.ablation.frozen_factors_and_weights(
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
    score = guarded.adaptive_score(ranks, adaptive_weights, panel)
    partitions = discrimination.fixed_partitions(panel["close"].index, model_config)
    models, fit_audit = discrimination.fit_models(
        ranks, outcome, partitions, model_config
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
        ranks, prediction_dates, models, model_config
    )
    scalar = discrimination.scalar_predictions(
        score,
        outcome,
        partitions["calibration"],
        prediction_dates,
        v6_config,
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
        scalar,
        panel,
        discrimination.v6.name_map(base),
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
        "fitAudit": fit_audit,
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
