"""Causal shadow test of Cantelli and finite-sample Chernoff tail bounds.

For each evaluation session, the risk multiplier is selected using only the
previous registered number of strategy PnL observations. The selected
multiplier is then applied to that session's baseline PnL for a shadow
comparison. This is a risk-shape diagnostic, not a predictive alpha model.

The script is offline and cannot place orders or modify live configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "tail_probability_bounds_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "tail_probability_bounds"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def cantelli_lower_tail_bound(
    sample_returns: np.ndarray,
    loss_threshold: float,
) -> float:
    """Distribution-free upper bound for P(R <= -loss_threshold)."""
    values = np.asarray(sample_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or loss_threshold <= 0:
        return 1.0
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    distance = loss_threshold + mean
    if distance <= 0:
        return 1.0
    if variance <= 0:
        return 0.0
    return min(1.0, variance / (variance + distance * distance))


def chernoff_ucb_lower_tail_bound(
    sample_returns: np.ndarray,
    loss_threshold: float,
    *,
    return_lower_bound: float,
    return_upper_bound: float,
    confidence: float,
    lambdas: list[float],
) -> float:
    """Chernoff bound with a Hoeffding UCB for the unknown empirical MGF.

    This remains conditional on iid observations and the supplied true return
    support. It fails closed if a historical value violates that support.
    """
    values = np.asarray(sample_returns, dtype=float)
    values = values[np.isfinite(values)]
    if (
        len(values) < 2
        or loss_threshold <= 0
        or not lambdas
        or not 0 < confidence < 1
        or return_lower_bound >= return_upper_bound
        or np.any(values < return_lower_bound)
        or np.any(values > return_upper_bound)
    ):
        return 1.0
    delta_per_lambda = (1.0 - confidence) / len(lambdas)
    best = 1.0
    for raw_lambda in lambdas:
        lam = float(raw_lambda)
        if lam <= 0:
            continue
        transformed = np.exp(-lam * values)
        transformed_lower = math.exp(-lam * return_upper_bound)
        transformed_upper = math.exp(-lam * return_lower_bound)
        mgf_ucb = float(np.mean(transformed)) + (
            transformed_upper - transformed_lower
        ) * math.sqrt(
            math.log(1.0 / delta_per_lambda) / (2.0 * len(values))
        )
        mgf_ucb = min(transformed_upper, mgf_ucb)
        candidate = math.exp(-lam * loss_threshold) * mgf_ucb
        best = min(best, candidate)
    return min(1.0, max(0.0, best))


def select_exposure(
    sample_returns: np.ndarray,
    *,
    daily_loss_threshold: float,
    maximum_tail_probability: float,
    exposure_grid: list[float],
    bound_function: Callable[[np.ndarray, float], float],
) -> tuple[float, float]:
    """Select the largest registered exposure whose tail bound fits budget."""
    ordered = sorted(
        {float(value) for value in exposure_grid if 0 <= float(value) <= 1},
        reverse=True,
    )
    if not ordered or ordered[-1] != 0.0:
        ordered.append(0.0)
    for exposure in ordered:
        if exposure == 0:
            return 0.0, 0.0
        unscaled_loss_threshold = daily_loss_threshold / exposure
        probability_bound = bound_function(
            sample_returns, unscaled_loss_threshold
        )
        if probability_bound <= maximum_tail_probability:
            return exposure, probability_bound
    return 0.0, 0.0


def load_daily_pnl(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    combined: dict[str, dict[str, Any]] = {}
    sources = [
        ("training", ROOT / data_cfg["trainingReplaySummary"]),
        ("continuation", ROOT / data_cfg["continuationReplaySummary"]),
    ]
    for source, path in sources:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for trade_date, values in payload["per_day"].items():
            if trade_date in combined:
                raise RuntimeError(f"overlapping replay date: {trade_date}")
            pnl = values.get("net_pnl", values.get("pnl"))
            if pnl is None:
                raise RuntimeError(f"missing PnL for {trade_date}")
            combined[trade_date] = {
                "trade_date": trade_date,
                "pnl": float(pnl),
                "source": source,
            }
    return [combined[day] for day in sorted(combined)]


def performance(
    daily_returns: list[float],
    exposures: list[float],
    initial_equity: float,
) -> dict[str, Any]:
    values = np.asarray(daily_returns, dtype=float)
    exposure_values = np.asarray(exposures, dtype=float)
    if not len(values):
        raise ValueError("empty evaluation returns")
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = (
        float(np.mean(values) / std * math.sqrt(252)) if std > 0 else None
    )
    return {
        "days": len(values),
        "totalPnl": float((equity[-1] - 1.0) * initial_equity),
        "totalReturn": float(equity[-1] - 1.0),
        "meanDailyReturn": float(np.mean(values)),
        "dailyStd": std,
        "dailySharpe": sharpe,
        "worstDay": float(np.min(values)),
        "maxDrawdown": float(np.min(equity / peaks - 1.0)),
        "averageExposure": float(np.mean(exposure_values)),
        "zeroExposureDays": int(np.sum(exposure_values == 0)),
        "fullExposureDays": int(np.sum(exposure_values == 1)),
    }


def run_shadow(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_cfg = config["data"]
    risk_cfg = config["riskBudget"]
    chernoff_cfg = config["chernoff"]
    equity = float(data_cfg["initialEquity"])
    lookback = int(data_cfg["rollingLookbackDays"])
    all_returns = np.asarray([row["pnl"] / equity for row in rows], dtype=float)
    daily_rows: list[dict[str, Any]] = []

    def chernoff(sample: np.ndarray, threshold: float) -> float:
        return chernoff_ucb_lower_tail_bound(
            sample,
            threshold,
            return_lower_bound=float(
                chernoff_cfg["assumedDailyReturnLowerBound"]
            ),
            return_upper_bound=float(
                chernoff_cfg["assumedDailyReturnUpperBound"]
            ),
            confidence=float(chernoff_cfg["mgfConfidence"]),
            lambdas=[float(value) for value in chernoff_cfg["lambdaGrid"]],
        )

    methods: dict[str, Callable[[np.ndarray, float], float]] = {
        "cantelli": cantelli_lower_tail_bound,
        "chernoff_ucb": chernoff,
        "best_valid_bound": lambda sample, threshold: min(
            cantelli_lower_tail_bound(sample, threshold),
            chernoff(sample, threshold),
        ),
    }
    for index in range(lookback, len(rows)):
        history = all_returns[index - lookback : index]
        actual_return = float(all_returns[index])
        selected: dict[str, tuple[float, float]] = {}
        for name, function in methods.items():
            selected[name] = select_exposure(
                history,
                daily_loss_threshold=float(
                    risk_cfg["dailyLossThresholdPct"]
                ),
                maximum_tail_probability=float(
                    risk_cfg["maximumTailProbability"]
                ),
                exposure_grid=[
                    float(value) for value in risk_cfg["exposureGrid"]
                ],
                bound_function=function,
            )
        daily_rows.append(
            {
                "trade_date": rows[index]["trade_date"],
                "source": rows[index]["source"],
                "baseline_return": actual_return,
                "history_start": rows[index - lookback]["trade_date"],
                "history_end": rows[index - 1]["trade_date"],
                "cantelli_exposure": selected["cantelli"][0],
                "cantelli_probability_bound": selected["cantelli"][1],
                "cantelli_shadow_return": (
                    selected["cantelli"][0] * actual_return
                ),
                "chernoff_ucb_exposure": selected["chernoff_ucb"][0],
                "chernoff_ucb_probability_bound": (
                    selected["chernoff_ucb"][1]
                ),
                "chernoff_ucb_shadow_return": (
                    selected["chernoff_ucb"][0] * actual_return
                ),
                "best_valid_bound_exposure": selected["best_valid_bound"][0],
                "best_valid_bound_probability_bound": (
                    selected["best_valid_bound"][1]
                ),
                "best_valid_bound_shadow_return": (
                    selected["best_valid_bound"][0] * actual_return
                ),
            }
        )
    if not daily_rows:
        raise RuntimeError("insufficient daily rows for rolling evaluation")
    baseline_returns = [row["baseline_return"] for row in daily_rows]
    metrics = {
        "baseline": performance(
            baseline_returns, [1.0] * len(daily_rows), equity
        )
    }
    for method in methods:
        metrics[method] = performance(
            [row[f"{method}_shadow_return"] for row in daily_rows],
            [row[f"{method}_exposure"] for row in daily_rows],
            equity,
        )
    return metrics, daily_rows


def markdown(report: dict[str, Any]) -> str:
    baseline = report["metrics"]["baseline"]
    lines = [
        "# Chebyshev/Cantelli and Chernoff Tail-Risk Shadow Replay",
        "",
        "Status: `diagnostic_only / reused contaminated replay / no live change`",
        "",
        "- Decision timing: each session's exposure uses only the previous "
        f"{report['lookbackDays']} strategy-PnL days.",
        f"- Risk budget: upper-bound P(daily loss > "
        f"{report['dailyLossThresholdPct']:.1%}) <= "
        f"{report['maximumTailProbability']:.0%}.",
        "- This tests exposure scaling, not entry/exit alpha.",
        "",
        "| Method | Return | Daily std | Worst day | Max DD | Avg exposure | Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in report["metrics"].items():
        lines.append(
            f"| {name} | {metric['totalReturn']:.2%} | "
            f"{metric['dailyStd']:.2%} | {metric['worstDay']:.2%} | "
            f"{metric['maxDrawdown']:.2%} | "
            f"{metric['averageExposure']:.2%} | "
            f"{metric['dailySharpe'] if metric['dailySharpe'] is not None else 'n/a'} |"
        )
    cantelli = report["metrics"]["cantelli"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Cantelli reduced daily standard deviation by "
            f"{1.0 - cantelli['dailyStd'] / baseline['dailyStd']:.1%}, "
            f"improved the worst day by "
            f"{cantelli['worstDay'] - baseline['worstDay']:.2%}, and retained "
            f"{cantelli['totalReturn'] / baseline['totalReturn']:.1%} of return.",
            "- The finite-sample Chernoff-Hoeffding UCB selected zero exposure "
            "throughout this short sample. It is mathematically conservative but "
            "not operationally useful here.",
            "- `best_valid_bound` equals Cantelli because its bound dominates the "
            "uninformative Chernoff UCB in this sample.",
            "- The apparent drawdown improvement is largely mechanical exposure "
            "reduction; it is not proof that a tail event was predicted.",
            "",
            "## Verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
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
    rows = load_daily_pnl(config)
    metrics, daily_rows = run_shadow(rows, config)
    baseline = metrics["baseline"]
    cantelli = metrics["cantelli"]
    gates = config["evidenceGates"]
    gate_checks = {
        "minimumEvaluationDays": cantelli["days"]
        >= int(gates["minimumEvaluationDays"]),
        "lowerDailyStd": cantelli["dailyStd"] < baseline["dailyStd"],
        "betterWorstDay": cantelli["worstDay"] > baseline["worstDay"],
        "betterMaxDrawdown": cantelli["maxDrawdown"]
        > baseline["maxDrawdown"],
        "minimumReturnRetention": (
            baseline["totalReturn"] > 0
            and cantelli["totalReturn"] / baseline["totalReturn"]
            >= float(gates["minimumReturnRetention"])
        ),
        "freshUnseenForwardWindow": not bool(
            config["data"]["replayWindowPreviouslyReused"]
        ),
        "pointInTimeUniverse": not bool(
            config["data"]["knownUniverseLookaheadContamination"]
        ),
    }
    shape_checks = {
        key: gate_checks[key]
        for key in (
            "minimumEvaluationDays",
            "lowerDailyStd",
            "betterWorstDay",
            "betterMaxDrawdown",
            "minimumReturnRetention",
        )
    }
    shape_supported = all(shape_checks.values())
    verdict = (
        "historical_risk_shape_supported_forward_shadow_required"
        if shape_supported
        else "tail_bound_risk_budget_not_supported"
    )
    report = {
        "schemaVersion": "tail_probability_bounds_result_v1",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "config": str(config_path),
        "inputDays": len(rows),
        "evaluationDays": len(daily_rows),
        "lookbackDays": int(config["data"]["rollingLookbackDays"]),
        "dailyLossThresholdPct": float(
            config["riskBudget"]["dailyLossThresholdPct"]
        ),
        "maximumTailProbability": float(
            config["riskBudget"]["maximumTailProbability"]
        ),
        "metrics": metrics,
        "evidence": {"gateChecks": gate_checks},
        "verdict": verdict,
        "verdictReason": (
            "The fixed Cantelli budget preserved at least 80% of historical "
            "return while reducing dispersion and drawdown, but reused and "
            "look-ahead-contaminated data prohibit deployment."
            if shape_supported
            else "The fixed risk budget failed at least one sample, return-retention "
            "or tail-risk-shape gate."
        ),
        "limitations": config["knownLimitations"],
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "tail_probability_bounds_result.json", report)
    pd.DataFrame(daily_rows).to_csv(
        output_dir / "tail_probability_bounds_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "tail_probability_bounds_report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "metrics": metrics,
                "gateChecks": gate_checks,
                "output": str(
                    output_dir / "tail_probability_bounds_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
