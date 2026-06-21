"""DSI-0008: fixed June test of high-vs-low frozen BUY scores.

The May-derived thresholds are fixed at high >= 71 and low <= 65. Results use BUY
next-snapshot-to-close returns less 14bps. Both pooled and same-day comparisons are shown
because pooled separation can be caused by market-day composition rather than ETF selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from run_etf_paper_trading_agent import ROOT, as_float


HIGH_MIN = 71.0
LOW_MAX = 65.0
COST = 0.0014
BOOTSTRAPS = 20_000
SEED = 20260621
DEFAULT_SCORES = (
    ROOT / "outputs" / "decision_score_pseudo_forward" / "20260506_20260618" / "scores"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_score_separation"


def load_june_buys(score_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(score_dir.glob("decision_scores_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (isinstance(row, dict) and str(row.get("date")) >= "2026-06-01"
                    and row.get("decision_type") == "BUY"
                    and row.get("realized_return") is not None):
                row = dict(row)
                row["net_return"] = as_float(row.get("realized_return")) - COST
                rows.append(row)
    return rows


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([as_float(row.get("net_return")) for row in rows], dtype=float)
    return {
        "count": len(rows),
        "days": len({str(row.get("date")) for row in rows}),
        "mean_net_return": float(np.mean(values)) if len(values) else None,
        "median_net_return": float(np.median(values)) if len(values) else None,
        "win_rate": float(np.mean(values > 0)) if len(values) else None,
        "mean_mae": float(np.mean([as_float(row.get("max_adverse_excursion")) for row in rows])) if rows else None,
        "mean_mfe": float(np.mean([as_float(row.get("max_favorable_excursion")) for row in rows])) if rows else None,
    }


def day_means(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, float]:
    result: dict[str, float] = {}
    for day in sorted({str(row.get("date")) for row in rows}):
        values = [as_float(row.get("net_return")) for row in rows
                  if str(row.get("date")) == day and predicate(row)]
        if values:
            result[day] = float(np.mean(values))
    return result


def cluster_bootstrap(rows: list[dict[str, Any]], *, n_boot: int = BOOTSTRAPS,
                      seed: int = SEED) -> dict[str, Any]:
    days = sorted({str(row.get("date")) for row in rows})
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(days, len(days), replace=True)
        high: list[float] = []
        low: list[float] = []
        for day in sampled:
            high.extend(as_float(row.get("net_return")) for row in rows
                        if str(row.get("date")) == day and as_float(row.get("total_score")) >= HIGH_MIN)
            low.extend(as_float(row.get("net_return")) for row in rows
                       if str(row.get("date")) == day and as_float(row.get("total_score")) <= LOW_MAX)
        if high and low:
            differences.append(float(np.mean(high) - np.mean(low)))
    values = np.array(differences, dtype=float)
    return {
        "n_boot": len(values),
        "seed": seed,
        "ci_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "median": float(np.median(values)),
        "one_sided_probability_difference_le_zero": float(np.mean(values <= 0)),
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high = [row for row in rows if as_float(row.get("total_score")) >= HIGH_MIN]
    low = [row for row in rows if as_float(row.get("total_score")) <= LOW_MAX]
    high_stats, low_stats = group_stats(high), group_stats(low)
    pooled_difference = high_stats["mean_net_return"] - low_stats["mean_net_return"]
    high_daily = day_means(rows, lambda row: as_float(row.get("total_score")) >= HIGH_MIN)
    low_daily = day_means(rows, lambda row: as_float(row.get("total_score")) <= LOW_MAX)
    paired_days = sorted(set(high_daily) & set(low_daily))
    paired = {day: high_daily[day] - low_daily[day] for day in paired_days}
    bootstrap = cluster_bootstrap(rows)
    return {
        "iteration_id": "DSI-0008",
        "status": "diagnostic_only",
        "hypothesis": "Frozen high-score BUY decisions outperform frozen low-score BUY decisions after cost",
        "window": "2026-06-01/2026-06-18",
        "thresholds": {"high_min": HIGH_MIN, "low_max": LOW_MAX, "round_trip_cost": COST},
        "sample": {"buy_decisions": len(rows), "trading_days": len({row.get('date') for row in rows})},
        "high": high_stats,
        "low": low_stats,
        "pooled_high_minus_low": pooled_difference,
        "day_balanced": {
            "high_mean": float(np.mean(list(high_daily.values()))),
            "low_mean": float(np.mean(list(low_daily.values()))),
            "paired_day_count": len(paired),
            "paired_high_minus_low_mean": float(np.mean(list(paired.values()))) if paired else None,
            "paired_differences": paired,
        },
        "day_cluster_bootstrap": bootstrap,
        "verdict": {
            "pooled_high_better": pooled_difference > 0,
            "cluster_ci_excludes_zero": bootstrap["ci_95"][0] > 0,
            "same_day_paired_high_better": bool(paired and np.mean(list(paired.values())) > 0),
            "validated": False,
            "reason": "only 13 test days; clustered CI includes zero and same-day paired mean is not positive",
        },
        "live_changes": False,
        "trade_gate_enabled": False,
    }


def render(result: dict[str, Any]) -> str:
    high, low = result["high"], result["low"]
    daily, boot, verdict = result["day_balanced"], result["day_cluster_bootstrap"], result["verdict"]
    return "\n".join([
        "# DSI-0008: High-vs-Low BUY Score Separation",
        "",
        "Status: `diagnostic_only / not_validated`",
        "",
        "Fixed June point-in-time test; BUY only; 14bps deducted; high >=71, low <=65.",
        "",
        "| group | decisions | days | mean net | median net | win rate | mean MAE | mean MFE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| high | {high['count']} | {high['days']} | {high['mean_net_return']:.4%} | {high['median_net_return']:.4%} | {high['win_rate']:.2%} | {high['mean_mae']:.4%} | {high['mean_mfe']:.4%} |",
        f"| low | {low['count']} | {low['days']} | {low['mean_net_return']:.4%} | {low['median_net_return']:.4%} | {low['win_rate']:.2%} | {low['mean_mae']:.4%} | {low['mean_mfe']:.4%} |",
        "",
        f"- pooled high-minus-low: {result['pooled_high_minus_low']:.4%}",
        f"- day-balanced high mean: {daily['high_mean']:.4%}; low mean: {daily['low_mean']:.4%}",
        f"- same-day paired difference ({daily['paired_day_count']} days): {daily['paired_high_minus_low_mean']:.4%}",
        f"- day-cluster bootstrap 95% CI: [{boot['ci_95'][0]:.4%}, {boot['ci_95'][1]:.4%}]",
        f"- one-sided probability difference <= 0: {boot['one_sided_probability_difference_le_zero']:.2%}",
        "",
        "## Verdict",
        "",
        f"- pooled_high_better: `{verdict['pooled_high_better']}`",
        f"- cluster_ci_excludes_zero: `{verdict['cluster_ci_excludes_zero']}`",
        f"- same_day_paired_high_better: `{verdict['same_day_paired_high_better']}`",
        f"- validated: `{verdict['validated']}`",
        "",
        "High scores look better in the pooled sample, but not after controlling for trading day. "
        "The score currently appears to capture favorable market days more than superior same-day ETF selection. "
        "Keep it shadow-only and do not gate orders.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fixed high and low BUY score groups.")
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = analyze(load_june_buys(Path(args.scores)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "dsi_0008_high_vs_low.json"
    md_path = output / "dsi_0008_high_vs_low.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(result)
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
