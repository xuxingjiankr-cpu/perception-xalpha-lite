"""Validate lower variance/drawdown on the frozen 2026-05-21..06-18 OOS split.

Reads replay summaries only.  No broker/API calls, no config/overlay writes, and
no strategy evolution.  The paired circular-block bootstrap preserves alignment
between each candidate and the baseline while retaining short local sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "outputs" / "t0_replay"
CONFIG_DIR = ROOT / "outputs" / "t0_strategy_evolution" / "candidate_configs"
OUT_DIR = ROOT / "outputs" / "t0_strategy_evolution"
OOS_START = "2026-05-21"
OOS_END = "2026-06-18"

SPECS = {
    "baseline": ("oos_variance_baseline_summary.json", "oos_train_baseline_current.json"),
    "i01": ("oos_variance_i01_summary.json", "oos_train_cmaes_g01_i01.json"),
    "i02": ("oos_variance_i02_summary.json", "oos_train_cmaes_g01_i02.json"),
    "i03": ("oos_variance_i03_summary.json", "oos_train_cmaes_g01_i03.json"),
    "i05": ("oos_variance_i05_summary.json", "oos_train_cmaes_g01_i05.json"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_cumulative_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def daily_net_pnl(summary: dict[str, Any], roundtrip_cost_bps: float) -> dict[str, float]:
    rate = roundtrip_cost_bps / 10_000.0
    out: dict[str, float] = {}
    for day, node in sorted(summary.get("per_day", {}).items()):
        gross = float(node.get("gross_pnl", node.get("pnl", 0.0)))
        notional = float(node.get("buy_notional", 0.0)) + float(node.get("sell_notional", 0.0))
        out[day] = gross - 0.5 * rate * notional
    return out


def metrics(values_by_day: dict[str, float]) -> dict[str, Any]:
    days = sorted(values_by_day)
    values = [float(values_by_day[day]) for day in days]
    if not values:
        raise ValueError("empty daily PnL")
    worst = min(values)
    return {
        "start_date": days[0],
        "end_date": days[-1],
        "n_days": len(days),
        "total_pnl": round(sum(values), 2),
        "daily_mean": round(statistics.mean(values), 2),
        "daily_std_sample": round(statistics.stdev(values), 2) if len(values) >= 2 else 0.0,
        "worst_day_pnl": round(worst, 2),
        "worst_day_date": days[values.index(worst)],
        "winning_days": sum(value > 0 for value in values),
        "losing_days": sum(value < 0 for value in values),
        "zero_days": sum(value == 0 for value in values),
        "winning_day_rate": round(sum(value > 0 for value in values) / len(values), 4),
        "max_cumulative_drawdown": round(max_cumulative_drawdown(values), 2),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def paired_block_bootstrap(
    baseline: list[float],
    candidate: list[float],
    *,
    reps: int = 10_000,
    block_length: int = 5,
    seed: int = 20260620,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("paired bootstrap requires equal series with at least two observations")
    n = len(baseline)
    block_length = max(1, min(block_length, n))
    rng = random.Random(seed)
    std_diffs: list[float] = []
    drawdown_diffs: list[float] = []
    pnl_diffs: list[float] = []
    for _ in range(reps):
        indices: list[int] = []
        while len(indices) < n:
            start = rng.randrange(n)
            indices.extend((start + offset) % n for offset in range(block_length))
        indices = indices[:n]
        base_sample = [baseline[index] for index in indices]
        cand_sample = [candidate[index] for index in indices]
        std_diffs.append(statistics.stdev(cand_sample) - statistics.stdev(base_sample))
        # Drawdowns are negative, so candidate - baseline > 0 means less severe.
        drawdown_diffs.append(max_cumulative_drawdown(cand_sample) - max_cumulative_drawdown(base_sample))
        pnl_diffs.append(sum(cand_sample) - sum(base_sample))
    return {
        "method": "paired_circular_block_bootstrap",
        "reps": reps,
        "block_length_days": block_length,
        "seed": seed,
        "daily_std_difference_candidate_minus_baseline": {
            "ci_95": [round(_percentile(std_diffs, 0.025), 2), round(_percentile(std_diffs, 0.975), 2)],
            "probability_candidate_lower": round(sum(value < 0 for value in std_diffs) / reps, 4),
        },
        "max_drawdown_difference_candidate_minus_baseline": {
            "ci_95": [round(_percentile(drawdown_diffs, 0.025), 2), round(_percentile(drawdown_diffs, 0.975), 2)],
            "probability_candidate_less_severe": round(sum(value > 0 for value in drawdown_diffs) / reps, 4),
        },
        "total_pnl_difference_candidate_minus_baseline": {
            "ci_95": [round(_percentile(pnl_diffs, 0.025), 2), round(_percentile(pnl_diffs, 0.975), 2)],
            "probability_candidate_higher": round(sum(value > 0 for value in pnl_diffs) / reps, 4),
        },
    }


def build_report(reps: int = 10_000) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    daily: dict[str, dict[str, float]] = {}
    rows: dict[str, dict[str, Any]] = {}
    config_hashes: dict[str, str] = {}
    common_days: list[str] | None = None
    for name, (summary_name, config_name) in SPECS.items():
        summary_path = REPLAY_DIR / summary_name
        config_path = CONFIG_DIR / config_name
        summary = load_json(summary_path)
        summaries[name] = summary
        day_map = {day: float(node.get("pnl", 0.0)) for day, node in sorted(summary["per_day"].items())}
        days = list(day_map)
        if common_days is None:
            common_days = days
        elif days != common_days:
            raise ValueError(f"OOS dates differ for {name}")
        if days[0] != OOS_START or days[-1] != OOS_END:
            raise ValueError(f"unexpected OOS boundary for {name}: {days[0]}..{days[-1]}")
        daily[name] = day_map
        gross = metrics(day_map)
        net12_map = daily_net_pnl(summary, 12.0)
        net12 = metrics(net12_map)
        rows[name] = {**gross, "net_12bps": net12}
        config_hashes[name] = sha256(config_path)

    baseline = rows["baseline"]
    base_values = list(daily["baseline"].values())
    comparisons: dict[str, Any] = {}
    passing: list[str] = []
    for name in ("i01", "i02", "i03", "i05"):
        row = rows[name]
        bootstrap = paired_block_bootstrap(base_values, list(daily[name].values()), reps=reps)
        std_lower = row["daily_std_sample"] < baseline["daily_std_sample"]
        worst_better = row["worst_day_pnl"] > baseline["worst_day_pnl"]
        std_ci_upper = bootstrap["daily_std_difference_candidate_minus_baseline"]["ci_95"][1]
        std_bootstrap_supported = std_ci_upper < 0
        defensible = bool(std_lower and worst_better and std_bootstrap_supported)
        if defensible:
            passing.append(name)
        comparisons[name] = {
            "total_pnl_delta": round(row["total_pnl"] - baseline["total_pnl"], 2),
            "daily_std_delta": round(row["daily_std_sample"] - baseline["daily_std_sample"], 2),
            "daily_std_reduction_pct": round((1 - row["daily_std_sample"] / baseline["daily_std_sample"]) * 100, 2),
            "worst_day_improvement": round(row["worst_day_pnl"] - baseline["worst_day_pnl"], 2),
            "max_drawdown_improvement": round(row["max_cumulative_drawdown"] - baseline["max_cumulative_drawdown"], 2),
            "lower_daily_std": std_lower,
            "better_worst_day": worst_better,
            "bootstrap_std_supported": std_bootstrap_supported,
            "defensible_lower_variance_pass": defensible,
            "bootstrap": bootstrap,
        }
    status = "oos_lower_variance_supported" if passing else "oos_lower_variance_not_supported"
    return {
        "schema_version": "oos_variance_validation_v1",
        "paper_trading_only": True,
        "diagnostic_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "split": {
            "training": ["2026-03-23", "2026-05-20"],
            "oos_test": [OOS_START, OOS_END],
            "oos_days": len(common_days or []),
            "optimizer_saw_oos": False,
        },
        "status": status,
        "passing_candidates": passing,
        "metrics": rows,
        "comparisons_vs_baseline": comparisons,
        "frozen_config_sha256": config_hashes,
        "decision": {
            "lower_variance_direction_supported": bool(passing),
            "preferred_candidate_for_evidence": passing[0] if len(passing) == 1 else passing,
            "recommendation": (
                "freeze the lower-variance hypothesis; optimize future research for risk-adjusted return/drawdown, "
                "then validate on new post-2026-06-18 data"
            ) if passing else "abandon parameter tuning and move to signal/regime or pure risk controls",
            "new_evolution_started": False,
            "new_evolution_reason": (
                "the OOS set has now been used to choose the lower-variance objective; reusing it would contaminate validation"
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# OOS Variance / Drawdown Validation",
        "",
        "Paper-only diagnostic. Frozen train: 2026-03-23..05-20; untouched OOS: 2026-05-21..06-18.",
        "",
        "| config | worst day | daily std | max cumulative DD | total PnL | winning days | net PnL @12bps | net std @12bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["metrics"].items():
        net = row["net_12bps"]
        lines.append(
            f"| {name} | {row['worst_day_pnl']:.2f} | {row['daily_std_sample']:.2f} | "
            f"{row['max_cumulative_drawdown']:.2f} | {row['total_pnl']:.2f} | "
            f"{row['winning_days']}/{row['n_days']} | {net['total_pnl']:.2f} | {net['daily_std_sample']:.2f} |"
        )
    lines.extend(["", f"- Status: `{report['status']}`", f"- Passing: `{report['passing_candidates']}`", ""])
    for name, comparison in report["comparisons_vs_baseline"].items():
        lines.append(
            f"- {name}: std {comparison['daily_std_reduction_pct']:+.2f}% reduction; "
            f"worst-day improvement {comparison['worst_day_improvement']:+.2f}; "
            f"defensible pass={comparison['defensible_lower_variance_pass']}"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        report["decision"]["recommendation"] + ".",
        "",
        "A new optimizer run is intentionally not started: the OOS period has now informed objective selection. "
        "Any new risk-adjusted candidate requires prospective data after 2026-06-18.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen OOS variance and drawdown")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    args = parser.parse_args()
    report = build_report(max(100, args.bootstrap_reps))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "oos_variance_validation.json"
    md_path = OUT_DIR / "oos_variance_validation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "passing_candidates": report["passing_candidates"],
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
