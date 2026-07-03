"""Causal common-factor/HMM conditional tail-risk shadow diagnostic.

The model does not assume that ETF bars are unconditionally independent.
Instead, each next five-minute ETF return is represented as a market-factor
component plus an idiosyncratic innovation.  The market distribution is
propagated by a training-frozen HMM and the idiosyncratic variance is updated
causally with an EWMA after each completed bar.

This script is offline research.  It cannot place orders, change sizing, write
an overlay or modify any live configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score

from research_hmm_nn_bl import (
    ROOT,
    GaussianHMM1D,
    _within_day_ratio,
    load_panel,
    select_training_universe,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "conditional_minute_tail_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "conditional_minute_tail"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def gaussian_mixture_tail_probability(
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    loss_threshold: float,
) -> float:
    """Return model P(X <= -loss_threshold) for a Gaussian mixture."""
    probabilities = norm.cdf(
        (-float(loss_threshold) - np.asarray(means, dtype=float))
        / np.sqrt(np.maximum(np.asarray(variances, dtype=float), 1e-15))
    )
    return float(
        np.clip(
            np.asarray(weights, dtype=float) @ probabilities,
            0.0,
            1.0,
        )
    )


def gaussian_mixture_chernoff_bound(
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    loss_threshold: float,
    lambdas: np.ndarray,
) -> float:
    """Conditional Chernoff bound for a fitted Gaussian mixture."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(weights.sum()), 1e-300)
    means = np.asarray(means, dtype=float)
    variances = np.maximum(np.asarray(variances, dtype=float), 1e-15)
    positive = np.asarray(lambdas, dtype=float)
    positive = positive[positive > 0]
    if loss_threshold <= 0 or not len(positive):
        return 1.0
    log_mgf = logsumexp(
        np.log(np.maximum(weights, 1e-300))[:, None]
        - means[:, None] * positive[None, :]
        + 0.5 * variances[:, None] * positive[None, :] ** 2,
        axis=0,
    )
    log_bounds = -positive * float(loss_threshold) + log_mgf
    return float(np.clip(math.exp(float(np.min(log_bounds))), 0.0, 1.0))


def ewma_variance_update(
    previous: float,
    residual: float,
    half_life_bars: float,
    variance_floor: float,
) -> float:
    """Update variance only after the residual has become observable."""
    decay = math.exp(math.log(0.5) / float(half_life_bars))
    return max(
        float(variance_floor),
        decay * float(previous) + (1.0 - decay) * float(residual) ** 2,
    )


def fit_factor_models(
    returns: pd.DataFrame,
    market: pd.Series,
    training_mask: pd.Series,
    variance_floor: float,
) -> dict[str, dict[str, float]]:
    models: dict[str, dict[str, float]] = {}
    for code in returns.columns:
        sample = pd.DataFrame(
            {
                "asset": returns.loc[training_mask, code],
                "market": market.loc[training_mask],
            }
        ).dropna()
        if len(sample) < 20:
            continue
        design = np.column_stack(
            [np.ones(len(sample)), sample["market"].to_numpy(dtype=float)]
        )
        target = sample["asset"].to_numpy(dtype=float)
        alpha, beta = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ np.asarray([alpha, beta])
        models[str(code)] = {
            "alpha": float(alpha),
            "beta": float(beta),
            "unconditionalMean": float(np.mean(target)),
            "unconditionalVariance": max(
                float(np.var(target, ddof=1)), variance_floor
            ),
            "residualVariance": max(
                float(np.var(residual, ddof=1)), variance_floor
            ),
        }
    return models


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def lag_one_correlation(values: np.ndarray) -> float | None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return None
    left, right = clean[:-1], clean[1:]
    if np.std(left) <= 0 or np.std(right) <= 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def average_absolute_off_diagonal_correlation(frame: pd.DataFrame) -> float:
    correlation = frame.corr(min_periods=20).to_numpy(dtype=float)
    if correlation.size == 0:
        return 0.0
    mask = ~np.eye(len(correlation), dtype=bool) & np.isfinite(correlation)
    return float(np.mean(np.abs(correlation[mask]))) if np.any(mask) else 0.0


def build_predictions(
    returns: pd.DataFrame,
    market: pd.Series,
    dates: pd.Series,
    factor_models: dict[str, dict[str, float]],
    hmm: GaussianHMM1D,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    thresholds = [float(value) for value in model_cfg["lossThresholdsPct"]]
    primary = float(model_cfg["primaryLossThresholdPct"])
    lambdas = np.geomspace(
        float(model_cfg["chernoffLambdaMinimum"]),
        float(model_cfg["chernoffLambdaMaximum"]),
        int(model_cfg["chernoffLambdaCount"]),
    )
    residual_variance = {
        code: model["residualVariance"]
        for code, model in factor_models.items()
    }
    asset_variance = {
        code: model["unconditionalVariance"]
        for code, model in factor_models.items()
    }
    rows: list[dict[str, Any]] = []
    oos_dates = sorted(
        set(
            dates[
                (dates >= str(data_cfg["oosStart"]))
                & (dates <= str(data_cfg["oosEnd"]))
            ]
        )
    )
    for trade_date in oos_dates:
        prior = hmm.start_probability.copy()
        for timestamp in returns.index[dates == trade_date]:
            market_return = market.at[timestamp]
            if not np.isfinite(market_return):
                continue
            observed: list[tuple[str, float]] = []
            for code, fitted in factor_models.items():
                actual = returns.at[timestamp, code]
                if not np.isfinite(actual):
                    continue
                alpha = fitted["alpha"]
                beta = fitted["beta"]
                means = alpha + beta * hmm.means
                variances = np.maximum(
                    beta * beta * hmm.variances + residual_variance[code],
                    float(model_cfg["varianceFloor"]),
                )
                conditional_mean = float(prior @ means)
                conditional_variance = max(
                    float(
                        prior @ (variances + means**2)
                        - conditional_mean**2
                    ),
                    float(model_cfg["varianceFloor"]),
                )
                factor_residual = float(
                    actual - alpha - beta * float(market_return)
                )
                row: dict[str, Any] = {
                    "trade_date": str(trade_date),
                    "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "stockCode": code,
                    "actual_return": float(actual),
                    "market_return": float(market_return),
                    "factor_residual": factor_residual,
                    "standardized_factor_residual": factor_residual
                    / math.sqrt(residual_variance[code]),
                    "conditional_mean": conditional_mean,
                    "conditional_sigma": math.sqrt(conditional_variance),
                }
                for threshold in thresholds:
                    suffix = f"{threshold:.4f}"
                    row[f"conditional_probability_{suffix}"] = (
                        gaussian_mixture_tail_probability(
                            prior, means, variances, threshold
                        )
                    )
                    row[f"static_probability_{suffix}"] = float(
                        norm.cdf(
                            (
                                -threshold
                                - fitted["unconditionalMean"]
                            )
                            / math.sqrt(fitted["unconditionalVariance"])
                        )
                    )
                    row[f"ewma_only_probability_{suffix}"] = float(
                        norm.cdf(
                            (
                                -threshold
                                - fitted["unconditionalMean"]
                            )
                            / math.sqrt(asset_variance[code])
                        )
                    )
                    row[f"tail_event_{suffix}"] = int(actual <= -threshold)
                    if math.isclose(threshold, primary):
                        row[f"conditional_chernoff_bound_{suffix}"] = (
                            gaussian_mixture_chernoff_bound(
                                prior,
                                means,
                                variances,
                                threshold,
                                lambdas,
                            )
                        )
                rows.append(row)
                observed.append((code, factor_residual))

            for code, residual in observed:
                residual_variance[code] = ewma_variance_update(
                    residual_variance[code],
                    residual,
                    float(model_cfg["idiosyncraticVarianceHalfLifeBars"]),
                    float(model_cfg["varianceFloor"]),
                )
                fitted = factor_models[code]
                actual = returns.at[timestamp, code]
                asset_variance[code] = ewma_variance_update(
                    asset_variance[code],
                    float(actual) - fitted["unconditionalMean"],
                    float(model_cfg["idiosyncraticVarianceHalfLifeBars"]),
                    float(model_cfg["varianceFloor"]),
                )

            clipped_market = float(np.clip(market_return, -0.05, 0.05))
            emission_log = hmm._emission_log_probability(
                np.asarray([clipped_market])
            )[0]
            likelihood = np.exp(emission_log - np.max(emission_log))
            posterior = prior * likelihood
            posterior /= max(float(posterior.sum()), 1e-300)
            prior = posterior @ hmm.transition
    return rows


def dependence_audit(predictions: pd.DataFrame) -> dict[str, Any]:
    raw_square: list[float] = []
    residual_square: list[float] = []
    for _, group in predictions.groupby("stockCode", sort=False):
        ordered = group.sort_values("timestamp")
        raw = lag_one_correlation(
            ordered["actual_return"].to_numpy(dtype=float) ** 2
        )
        residual = lag_one_correlation(
            ordered["standardized_factor_residual"].to_numpy(dtype=float)
            ** 2
        )
        if raw is not None:
            raw_square.append(raw)
        if residual is not None:
            residual_square.append(residual)
    raw_panel = predictions.pivot(
        index="timestamp", columns="stockCode", values="actual_return"
    )
    residual_panel = predictions.pivot(
        index="timestamp", columns="stockCode", values="factor_residual"
    )
    return {
        "medianLag1SquaredReturnCorrelation": float(
            np.median(raw_square)
        ),
        "medianLag1SquaredStandardizedResidualCorrelation": float(
            np.median(residual_square)
        ),
        "rawCrossSectionalAbsoluteCorrelation": (
            average_absolute_off_diagonal_correlation(raw_panel)
        ),
        "factorResidualCrossSectionalAbsoluteCorrelation": (
            average_absolute_off_diagonal_correlation(residual_panel)
        ),
    }


def threshold_metrics(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    suffix = f"{threshold:.4f}"
    labels = predictions[f"tail_event_{suffix}"].to_numpy(dtype=int)
    conditional = predictions[
        f"conditional_probability_{suffix}"
    ].to_numpy(dtype=float)
    static = predictions[f"static_probability_{suffix}"].to_numpy(dtype=float)
    ewma_only = predictions[
        f"ewma_only_probability_{suffix}"
    ].to_numpy(dtype=float)
    per_code: list[dict[str, Any]] = []
    for code, group in predictions.groupby("stockCode", sort=True):
        code_labels = group[f"tail_event_{suffix}"].to_numpy(dtype=int)
        conditional_auc = safe_auc(
            code_labels,
            group[f"conditional_probability_{suffix}"].to_numpy(dtype=float),
        )
        static_auc = safe_auc(
            code_labels,
            group[f"static_probability_{suffix}"].to_numpy(dtype=float),
        )
        if conditional_auc is not None:
            per_code.append(
                {
                    "stockCode": code,
                    "events": int(np.sum(code_labels)),
                    "conditionalAuc": conditional_auc,
                    "staticAuc": static_auc,
                }
            )

    daily_rows: list[dict[str, Any]] = []
    for trade_date, group in predictions.groupby("trade_date", sort=True):
        day_labels = group[f"tail_event_{suffix}"].to_numpy(dtype=float)
        day_conditional = group[
            f"conditional_probability_{suffix}"
        ].to_numpy(dtype=float)
        day_static = group[f"static_probability_{suffix}"].to_numpy(dtype=float)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "actualTailRate": float(np.mean(day_labels)),
                "predictedTailRate": float(np.mean(day_conditional)),
                "conditionalBrier": float(
                    np.mean((day_labels - day_conditional) ** 2)
                ),
                "staticBrier": float(
                    np.mean((day_labels - day_static) ** 2)
                ),
            }
        )
    daily = pd.DataFrame(daily_rows)
    improvement = (
        daily["staticBrier"] - daily["conditionalBrier"]
    ).to_numpy(dtype=float)
    quartile = pd.qcut(
        predictions[f"conditional_probability_{suffix}"],
        4,
        labels=False,
        duplicates="drop",
    )
    bins: list[dict[str, Any]] = []
    for bucket, group in predictions.assign(risk_quartile=quartile).groupby(
        "risk_quartile", observed=True, sort=True
    ):
        bins.append(
            {
                "quartile": int(bucket) + 1,
                "observations": len(group),
                "meanPredictedProbability": float(
                    group[f"conditional_probability_{suffix}"].mean()
                ),
                "actualTailRate": float(
                    group[f"tail_event_{suffix}"].mean()
                ),
            }
        )
    return {
        "lossThresholdPct": threshold,
        "observations": len(labels),
        "tailEvents": int(np.sum(labels)),
        "actualTailRate": float(np.mean(labels)),
        "conditional": {
            "meanProbability": float(np.mean(conditional)),
            "auc": safe_auc(labels, conditional),
            "brier": float(brier_score_loss(labels, conditional)),
        },
        "static": {
            "meanProbability": float(np.mean(static)),
            "auc": safe_auc(labels, static),
            "brier": float(brier_score_loss(labels, static)),
        },
        "ewmaOnly": {
            "meanProbability": float(np.mean(ewma_only)),
            "auc": safe_auc(labels, ewma_only),
            "brier": float(brier_score_loss(labels, ewma_only)),
        },
        "medianPerCodeConditionalAuc": (
            float(np.median([row["conditionalAuc"] for row in per_code]))
            if per_code
            else None
        ),
        "codesWithDefinedAuc": len(per_code),
        "meanDailyBrierImprovement": float(np.mean(improvement)),
        "dailyBrierImprovementT": (
            float(
                np.mean(improvement)
                / (np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
            )
            if len(improvement) > 1 and np.std(improvement, ddof=1) > 0
            else None
        ),
        "dailyPredictedVsActualCorrelation": float(
            daily["predictedTailRate"].corr(daily["actualTailRate"])
        ),
        "riskQuartiles": bins,
        "daily": daily_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Conditional Minute Tail-Risk Research",
        "",
        "Status: `diagnostic_only / reused OOS / no live change`",
        "",
        "The model treats bars as conditional innovations, not unconditional IID "
        "rows. Dependence is retained through a common market factor, a filtered "
        "HMM state distribution and lagged EWMA idiosyncratic variance.",
        "",
        "| Loss in next 5m | Events | Conditional / EWMA / static AUC | "
        "Conditional / EWMA / static Brier | Daily Brier improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["thresholds"].values():
        lines.append(
            f"| {item['lossThresholdPct']:.2%} | {item['tailEvents']} | "
            f"{item['conditional']['auc']:.4f} / "
            f"{item['ewmaOnly']['auc']:.4f} / "
            f"{item['static']['auc']:.4f} | "
            f"{item['conditional']['brier']:.6f} / "
            f"{item['ewmaOnly']['brier']:.6f} / "
            f"{item['static']['brier']:.6f} | "
            f"{item['meanDailyBrierImprovement']:.6f} |"
        )
    audit = report["dependenceAudit"]
    lines.extend(
        [
            "",
            "## Dependence audit",
            "",
            f"- Median lag-1 squared-return correlation: "
            f"{audit['medianLag1SquaredReturnCorrelation']:.4f}.",
            f"- After factor removal and causal variance standardisation: "
            f"{audit['medianLag1SquaredStandardizedResidualCorrelation']:.4f}.",
            f"- Mean absolute cross-sectional correlation: "
            f"{audit['rawCrossSectionalAbsoluteCorrelation']:.4f} raw versus "
            f"{audit['factorResidualCrossSectionalAbsoluteCorrelation']:.4f} "
            "after common-factor removal.",
            "",
            "## Verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "The result is a risk ranking only. It does not show that avoiding "
            "high-risk bars improves cost-adjusted return.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "No live config, overlay, order path, execution lock or position "
            "sizing rule was changed.",
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
    data_cfg = config["data"]
    model_cfg = config["model"]
    quotes = ROOT / data_cfg["quotes"]
    selected, universe_audit = select_training_universe(
        quotes,
        str(data_cfg["trainEnd"]),
        int(data_cfg["universeSize"]),
    )
    panel = load_panel(quotes, selected)
    prices = panel.pivot(
        index="timestamp", columns="stockCode", values="close"
    ).sort_index()
    returns = _within_day_ratio(prices, 1)
    market = returns.median(axis=1, skipna=True)
    dates = pd.Series(
        returns.index.strftime("%Y-%m-%d"), index=returns.index
    )
    training_mask = (
        (dates >= str(data_cfg["trainStart"]))
        & (dates <= str(data_cfg["trainEnd"]))
    )
    sequences = [
        market[dates == trade_date]
        .dropna()
        .clip(-0.05, 0.05)
        .to_numpy(dtype=float)
        for trade_date in sorted(set(dates[training_mask]))
    ]
    hmm = GaussianHMM1D(
        n_states=int(model_cfg["states"]),
        n_iter=int(model_cfg["hmmIterations"]),
        variance_floor=float(model_cfg["varianceFloor"]),
    ).fit(sequences)
    factor_models = fit_factor_models(
        returns,
        market,
        training_mask,
        float(model_cfg["varianceFloor"]),
    )
    rows = build_predictions(
        returns, market, dates, factor_models, hmm, config
    )
    if not rows:
        raise RuntimeError("no conditional minute predictions")
    predictions = pd.DataFrame(rows)
    dependence = dependence_audit(predictions)
    threshold_results = {
        f"{threshold:.4f}": threshold_metrics(predictions, threshold)
        for threshold in [
            float(value) for value in model_cfg["lossThresholdsPct"]
        ]
    }
    primary_key = f"{float(model_cfg['primaryLossThresholdPct']):.4f}"
    primary = threshold_results[primary_key]
    chernoff_column = f"conditional_chernoff_bound_{primary_key}"
    probability_column = f"conditional_probability_{primary_key}"
    chernoff_audit = {
        "meanConditionalProbability": float(
            predictions[probability_column].mean()
        ),
        "meanConditionalChernoffBound": float(
            predictions[chernoff_column].mean()
        ),
        "boundDominatesModelProbabilityFraction": float(
            np.mean(
                predictions[chernoff_column].to_numpy(dtype=float)
                + 1e-12
                >= predictions[probability_column].to_numpy(dtype=float)
            )
        ),
        "modelConditionalNotDistributionFree": True,
    }
    gates_cfg = config["evidenceGates"]
    gates = {
        "minimumOosDays": predictions["trade_date"].nunique()
        >= int(gates_cfg["minimumOosDays"]),
        "minimumPrimaryTailEvents": primary["tailEvents"]
        >= int(gates_cfg["minimumPrimaryTailEvents"]),
        "conditionalAucAboveStatic": primary["conditional"]["auc"]
        > primary["static"]["auc"],
        "conditionalBrierBelowStatic": primary["conditional"]["brier"]
        < primary["static"]["brier"],
        "conditionalAucAboveEwmaOnly": primary["conditional"]["auc"]
        > primary["ewmaOnly"]["auc"],
        "conditionalBrierBelowEwmaOnly": primary["conditional"]["brier"]
        < primary["ewmaOnly"]["brier"],
        "positiveDailyBrierImprovement": primary[
            "meanDailyBrierImprovement"
        ]
        > 0,
        "lowerCrossSectionalResidualCorrelation": dependence[
            "factorResidualCrossSectionalAbsoluteCorrelation"
        ]
        < dependence["rawCrossSectionalAbsoluteCorrelation"],
        "freshUnseenForwardWindow": not bool(
            data_cfg["oosWindowPreviouslyReused"]
        ),
    }
    historical_gates = {
        key: value
        for key, value in gates.items()
        if key != "freshUnseenForwardWindow"
    }
    supported = all(historical_gates.values())
    verdict = (
        "conditional_dependence_risk_signal_supported_forward_shadow_required"
        if supported
        else "conditional_dependence_risk_signal_not_supported"
    )
    report = {
        "schemaVersion": "conditional_minute_tail_result_v1",
        "generatedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
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
            "count": len(factor_models),
            "selectionUsesOos": universe_audit["selectionUsesOos"],
            "codes": sorted(factor_models),
        },
        "observations": len(predictions),
        "oosDays": predictions["trade_date"].nunique(),
        "hmm": {
            "means": [float(value) for value in hmm.means],
            "standardDeviations": [
                float(math.sqrt(value)) for value in hmm.variances
            ],
            "transition": hmm.transition.tolist(),
        },
        "dependenceAudit": dependence,
        "thresholds": threshold_results,
        "conditionalChernoffAudit": chernoff_audit,
        "evidence": {"gates": gates},
        "verdict": verdict,
        "verdictReason": (
            "The causal factor/HMM/EWMA model improved historical next-bar "
            "tail-risk ranking and calibration at the preregistered primary "
            "threshold, but the reused OOS window permits forward shadow use only."
            if supported
            else "The conditional model failed at least one sample, calibration, "
            "ranking or dependence-reduction gate."
        ),
        "limitations": config["knownLimitations"],
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "conditional_minute_tail_result.json", report)
    write_csv(
        output_dir / "conditional_minute_tail_predictions.csv",
        predictions.to_dict("records"),
    )
    daily_rows: list[dict[str, Any]] = []
    for key, result in threshold_results.items():
        for row in result["daily"]:
            daily_rows.append({"loss_threshold": key, **row})
    write_csv(
        output_dir / "conditional_minute_tail_daily.csv", daily_rows
    )
    (output_dir / "conditional_minute_tail_report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "observations": report["observations"],
                "oosDays": report["oosDays"],
                "dependenceAudit": dependence,
                "primaryThreshold": primary,
                "conditionalChernoffAudit": chernoff_audit,
                "gates": gates,
                "output": str(
                    output_dir / "conditional_minute_tail_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
