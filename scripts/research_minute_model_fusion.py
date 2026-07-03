"""Ablate HMM-state and Black-Litterman fusion into the minute forecast.

All models and the universe are fitted on the registered training period. The
30-minute OOS replay uses completed bars, next-bar entry and a 12 bps cost.
This is offline record-only research and cannot change trading behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_hmm_nn_bl import (
    GaussianHMM1D,
    black_litterman_posterior,
    covariance_asof,
    load_panel,
    optimize_long_only,
    select_training_universe,
)
from research_minute_forecast_shadow import (
    ROOT,
    build_feature_frames,
    build_samples,
    calibrated_probability,
    daily_performance,
    evaluate_policy,
    fit_temporal_calibrated_classifier,
    make_regressor,
    probability_metrics,
)
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "minute_model_fusion_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "minute_model_fusion"
HMM_FEATURES = [
    "hmm_bear_probability",
    "hmm_bull_probability",
    "hmm_expected_return",
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def add_causal_hmm_features(
    frames: dict[str, Any],
    *,
    train_end: str,
    hmm_config: dict[str, Any],
) -> dict[str, Any]:
    market = frames["market_ret_1"]
    dates = pd.Series(market.index.strftime("%Y-%m-%d"), index=market.index)
    hmm = GaussianHMM1D(
        n_states=int(hmm_config["states"]),
        n_iter=int(hmm_config["iterations"]),
        variance_floor=float(hmm_config["varianceFloor"]),
    )
    sequences = [
        market[dates == trade_date].to_numpy(dtype=float)
        for trade_date in sorted(set(dates[dates <= train_end]))
    ]
    hmm.fit(sequences)
    probabilities = pd.DataFrame(
        index=market.index,
        columns=[f"state_{index}" for index in range(hmm.n_states)],
        dtype=float,
    )
    for trade_date in sorted(set(dates)):
        mask = dates == trade_date
        probabilities.loc[mask, :] = hmm.filter_probabilities(
            market[mask].to_numpy(dtype=float)
        )
    output = dict(frames)
    output["hmm_bear_probability"] = probabilities.iloc[:, 0]
    output["hmm_bull_probability"] = probabilities.iloc[:, -1]
    output["hmm_expected_return"] = pd.Series(
        probabilities.to_numpy(dtype=float) @ hmm.means,
        index=market.index,
    )
    return output


def fit_predictions(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    cost: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = (
        train["gross_forward_return"].to_numpy(dtype=float) > cost
    ).astype(int)
    classifier, calibrator, calibration_audit = (
        fit_temporal_calibrated_classifier(train, features, labels, config)
    )
    regressor = make_regressor(config)
    regressor.fit(
        train[features].to_numpy(dtype=float),
        train["gross_forward_return"].to_numpy(dtype=float) * 10_000.0,
    )
    train_expected = (
        regressor.predict(train[features].to_numpy(dtype=float)) / 10_000.0
    )
    expected = (
        regressor.predict(oos[features].to_numpy(dtype=float)) / 10_000.0
    )
    probability = calibrated_probability(
        classifier, calibrator, oos[features].to_numpy(dtype=float)
    )
    residual_q05 = float(
        np.quantile(
            train["gross_forward_return"].to_numpy(dtype=float)
            - train_expected,
            float(config["models"]["downsideQuantile"]),
        )
    )
    scored = oos.copy()
    scored["probability"] = probability
    scored["expected_gross_return"] = expected
    scored["expected_net_return"] = expected - cost
    scored["downside_net_quantile"] = expected + residual_q05 - cost
    oos_labels = (
        oos["gross_forward_return"].to_numpy(dtype=float) > cost
    ).astype(int)
    audit = {
        "forecast": probability_metrics(
            oos_labels, probability, float(np.mean(labels))
        ),
        "calibration": calibration_audit,
        "trainingResidualQ05": residual_q05,
    }
    return scored, audit


def evaluate_hmm_bl(
    samples: pd.DataFrame,
    frames: dict[str, Any],
    *,
    horizon_bars: int,
    cost: float,
    policy_config: dict[str, Any],
    bl_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    daily: dict[str, float] = defaultdict(float)
    allocations: list[dict[str, Any]] = []
    tranche = min(1.0, 1.0 / max(1, horizon_bars))
    for timestamp, group in samples.groupby("timestamp", sort=True):
        eligible = group[
            (group["probability"] >= float(policy_config["minimumProbability"]))
            & (
                group["expected_net_return"]
                >= float(policy_config["minimumExpectedNetReturn"])
            )
            & (
                group["downside_net_quantile"]
                >= float(policy_config["minimumDownsideNetQuantile"])
            )
        ].copy()
        if eligible.empty:
            continue
        eligible = eligible.sort_values("stockCode")
        codes = eligible["stockCode"].tolist()
        expected = eligible["expected_gross_return"].to_numpy(dtype=float)
        covariance = covariance_asof(
            frames["ret_1"],
            pd.Timestamp(timestamp),
            codes,
            int(bl_config["covarianceLookbackBars"]),
            float(bl_config["covarianceRidge"]),
            horizon_bars,
        )
        residual_variance = max(
            float(np.var(eligible["expected_gross_return"].to_numpy(dtype=float))),
            1e-10,
        )
        posterior = black_litterman_posterior(
            covariance,
            expected,
            tau=float(bl_config["tau"]),
            risk_aversion=float(bl_config["riskAversion"]),
            view_confidence=float(bl_config["viewConfidence"]),
            residual_variance=residual_variance,
        )
        weights = optimize_long_only(
            posterior,
            covariance,
            round_trip_cost=cost,
            risk_aversion=float(bl_config["riskAversion"]),
            max_weight=float(bl_config.get("maxWeightPerAsset", 0.25)),
            max_assets=int(policy_config["maximumAssetsPerTimestamp"]),
        )
        if weights.sum() <= 0:
            continue
        if weights.sum() > tranche:
            weights *= tranche / weights.sum()
        bear = float(eligible["hmm_bear_probability"].iloc[0])
        bull = float(eligible["hmm_bull_probability"].iloc[0])
        regime_exposure = float(
            np.clip(0.25 + 0.75 * bull - 0.25 * bear, 0.0, 1.0)
        )
        weights *= regime_exposure
        actual = eligible["gross_forward_return"].to_numpy(dtype=float)
        interval_net = float(weights @ actual - weights.sum() * cost)
        trade_date = str(eligible["trade_date"].iloc[0])
        daily[trade_date] += interval_net
        for index in np.where(weights > 1e-12)[0]:
            allocations.append(
                {
                    "variant": "new_plus_hmm_and_bl",
                    "trade_date": trade_date,
                    "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "stockCode": codes[index],
                    "weight": float(weights[index]),
                    "probability": float(eligible["probability"].iloc[index]),
                    "expected_gross_return": float(expected[index]),
                    "bl_posterior": float(posterior[index]),
                    "actual_net_return": float(actual[index] - cost),
                    "regime_exposure": regime_exposure,
                }
            )
    dates = sorted(samples["trade_date"].unique())
    complete = {day: float(daily.get(day, 0.0)) for day in dates}
    performance = daily_performance(complete)
    trade_returns = np.asarray(
        [row["actual_net_return"] for row in allocations], dtype=float
    )
    performance.update(
        {
            "selectedTrades": len(allocations),
            "averageNetTrade": (
                float(np.mean(trade_returns)) if len(trade_returns) else None
            ),
            "winTradeRate": (
                float(np.mean(trade_returns > 0))
                if len(trade_returns)
                else None
            ),
            "averageRegimeExposure": (
                float(
                    np.mean(
                        [row["regime_exposure"] for row in allocations]
                    )
                )
                if allocations
                else 0.0
            ),
        }
    )
    return performance, allocations, complete


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Minute Forecast + HMM + Black-Litterman Fusion",
        "",
        "Status: `diagnostic_only / fixed ablation / no live change`",
        "",
        "- Common horizon: 30 minutes.",
        "- Next-bar entry and 12 bps round-trip cost for every variant.",
        "- Old MLP is intentionally not restacked; HMM supplies state and BL "
        "supplies shrinkage/allocation.",
        "",
        "| Variant | Trades | Avg net trade | Net return | Sharpe | Worst day | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["variants"].items():
        lines.append(
            f"| {name} | {result.get('selectedTrades', 0)} | "
            f"{result.get('averageNetTrade')} | "
            f"{result.get('netPortfolioReturn', 0):.2%} | "
            f"{result.get('dailySharpe')} | "
            f"{result.get('worstDay', 0):.2%} | "
            f"{result.get('maxDrawdown', 0):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "Lower loss caused only by lower exposure or fewer trades is not "
            "counted as alpha.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "No live config, order path, sizing rule, overlay or execution lock "
            "was changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / config["baseConfig"]).read_text(encoding="utf-8"))
    old = json.loads((ROOT / config["oldModelConfig"]).read_text(encoding="utf-8"))
    data_cfg = base["data"]
    quotes = ROOT / data_cfg["quotes"]
    selected, universe_audit = select_training_universe(
        quotes,
        str(data_cfg["trainEnd"]),
        int(data_cfg["universeSize"]),
    )
    panel = load_panel(quotes, selected)
    base_frames = build_feature_frames(panel)
    fused_frames = add_causal_hmm_features(
        base_frames,
        train_end=str(data_cfg["trainEnd"]),
        hmm_config=old["hmm"],
    )
    horizon = int(config["horizonBars"])
    base_features = list(base["features"])
    fused_features = base_features + HMM_FEATURES
    samples = build_samples(fused_frames, fused_features, horizon)
    train = samples[
        (samples["trade_date"] >= data_cfg["trainStart"])
        & (samples["trade_date"] <= data_cfg["trainEnd"])
    ].copy()
    oos = samples[
        (samples["trade_date"] >= data_cfg["oosStart"])
        & (samples["trade_date"] <= data_cfg["oosEnd"])
    ].copy()
    cost = float(base["targets"]["roundTripCostBps"]) / 10_000.0
    new_scored, new_audit = fit_predictions(
        train, oos, base_features, base, cost
    )
    fused_scored, fused_audit = fit_predictions(
        train, oos, fused_features, base, cost
    )
    policy = base["shadowPolicy"]
    new_metrics, new_rows, new_daily = evaluate_policy(
        new_scored,
        horizon_bars=horizon,
        cost=cost,
        probability_threshold=float(policy["minimumProbability"]),
        expected_net_threshold=float(policy["minimumExpectedNetReturn"]),
        downside_net_threshold=float(policy["minimumDownsideNetQuantile"]),
        max_assets=int(policy["maximumAssetsPerTimestamp"]),
    )
    fused_metrics, fused_rows, fused_daily = evaluate_policy(
        fused_scored,
        horizon_bars=horizon,
        cost=cost,
        probability_threshold=float(policy["minimumProbability"]),
        expected_net_threshold=float(policy["minimumExpectedNetReturn"]),
        downside_net_threshold=float(policy["minimumDownsideNetQuantile"]),
        max_assets=int(policy["maximumAssetsPerTimestamp"]),
    )
    bl_metrics, bl_rows, bl_daily = evaluate_hmm_bl(
        fused_scored,
        fused_frames,
        horizon_bars=horizon,
        cost=cost,
        policy_config=policy,
        bl_config={
            **old["blackLitterman"],
            "maxWeightPerAsset": old["execution"]["maxWeightPerAsset"],
        },
    )
    variants = {
        "new_linear": new_metrics,
        "new_plus_hmm_features": fused_metrics,
        "new_plus_hmm_and_bl": bl_metrics,
    }
    daily_by_variant = {
        "new_linear": new_daily,
        "new_plus_hmm_features": fused_daily,
        "new_plus_hmm_and_bl": bl_daily,
    }
    gates_cfg = config["evidenceGates"]
    gates: dict[str, dict[str, bool]] = {}
    evidence: dict[str, Any] = {}
    cash = {day: 0.0 for day in new_daily}
    for name, metrics in variants.items():
        gates[name] = {
            "minimumOosDays": metrics["days"]
            >= int(gates_cfg["minimumOosDays"]),
            "minimumSelectedTrades": metrics["selectedTrades"]
            >= int(gates_cfg["minimumSelectedTrades"]),
            "positiveAverageNetTrade": metrics["averageNetTrade"] is not None
            and metrics["averageNetTrade"] > 0,
            "positiveNetPortfolioReturn": metrics["netPortfolioReturn"] > 0,
            "positiveDailySharpe": metrics["dailySharpe"] is not None
            and metrics["dailySharpe"] > 0,
            "freshUnseenForwardWindow": not bool(
                data_cfg["oosWindowPreviouslyReused"]
            ),
        }
        evidence[name] = {
            "gates": gates[name],
            "dmVsNewLinear": diebold_mariano_hln(
                new_daily, daily_by_variant[name], alpha=0.05
            ),
            "dsrVsCash": deflated_sharpe_diagnostic(
                cash, daily_by_variant[name], n_trials=3, alpha=0.10
            ),
        }
    historical_passes = [
        name
        for name, checks in gates.items()
        if all(
            value
            for key, value in checks.items()
            if key != "freshUnseenForwardWindow"
        )
    ]
    verdict = (
        "historical_fusion_candidate_forward_only"
        if historical_passes
        else "fusion_does_not_create_validated_edge"
    )
    report = {
        "schemaVersion": "minute_model_fusion_result_v1",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "config": str(config_path),
        "split": {
            "trainStart": data_cfg["trainStart"],
            "trainEnd": data_cfg["trainEnd"],
            "oosStart": data_cfg["oosStart"],
            "oosEnd": data_cfg["oosEnd"],
            "oosWindowPreviouslyReused": bool(
                data_cfg["oosWindowPreviouslyReused"]
            ),
        },
        "universe": {
            "count": len(selected),
            "selectionUsesOos": universe_audit["selectionUsesOos"],
        },
        "horizonMinutes": horizon * 5,
        "modelAudit": {
            "newLinear": new_audit,
            "hmmFeatures": fused_audit,
            "oldNeuralPredictionIncluded": False,
        },
        "variants": variants,
        "evidence": evidence,
        "historicalPassingVariants": historical_passes,
        "verdict": verdict,
        "verdictReason": (
            "A fixed fusion variant passed the numerical historical gates, but "
            "the reused test window permits forward shadow only."
            if historical_passes
            else "No fixed fusion variant produced enough positive cost-adjusted "
            "trades with positive absolute portfolio return."
        ),
        "limitations": config["knownLimitations"],
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "minute_model_fusion_result.json", report)
    rows = []
    for row in new_rows:
        rows.append({"variant": "new_linear", **row})
    for row in fused_rows:
        rows.append({"variant": "new_plus_hmm_features", **row})
    rows.extend(bl_rows)
    write_csv(output_dir / "minute_model_fusion_allocations.csv", rows)
    write_csv(
        output_dir / "minute_model_fusion_daily.csv",
        [
            {
                "trade_date": day,
                **{
                    name: daily_by_variant[name][day]
                    for name in config["variants"]
                },
            }
            for day in sorted(new_daily)
        ],
    )
    (output_dir / "minute_model_fusion_report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "variants": variants,
                "historicalPassingVariants": historical_passes,
                "output": str(
                    output_dir / "minute_model_fusion_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
