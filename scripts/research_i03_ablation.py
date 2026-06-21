"""Explain i03's OOS variance reduction with four isolated parameter groups.

Offline replay only. The frozen OOS window has already been observed, so the
minimal combination produced here is exploratory and explicitly requires new
post-2026-06-18 prospective validation before any shadow/live use.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from analyze_oos_variance import daily_net_pnl, metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "outputs" / "t0_strategy_evolution" / "candidate_configs"
REPLAY_DIR = ROOT / "outputs" / "t0_replay"
BASE = CONFIG_DIR / "oos_train_baseline_current.json"
I03 = CONFIG_DIR / "oos_train_cmaes_g01_i03.json"
QUOTES = REPLAY_DIR / "yahoo_60d_quotes.jsonl"
OUT_JSON = ROOT / "outputs" / "t0_strategy_evolution" / "i03_group_ablation.json"
OUT_MD = ROOT / "outputs" / "t0_strategy_evolution" / "i03_group_ablation.md"
OOS_START, OOS_END = "2026-05-21", "2026-06-18"

GROUPS = {
    "L1_risk_per_trade": [
        "strategy.bracket.risk_per_trade_pct", "strategy.bracket.risk_per_trade_pct_chaos_day",
    ],
    "L2_earlier_profit": [
        "strategy.bracket.target1_r_multiple", "strategy.bracket.target2_r_multiple",
        "strategy.profit_trailing_drawdown_pct", "strategy.profit_exit_score_threshold",
    ],
    "L3_faster_loss": [
        "strategy.loss_exit_score_threshold", "strategy.loss_review_after_minutes",
        "strategy.deceleration_exit_threshold",
    ],
    "L4_stress_selectivity": [
        "strategy.market_correlation_stress.avg_abs_corr_threshold", "strategy.entry_score_threshold",
        "strategy.entry_momentum_pct", "strategy.min_hold_minutes",
    ],
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(obj: dict[str, Any], path: str) -> Any:
    node: Any = obj
    for part in path.split("."):
        node = node[part]
    return node


def set_path(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = obj
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = copy.deepcopy(value)


def build_config(base: dict[str, Any], i03: dict[str, Any], groups: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    for group in groups:
        for path in GROUPS[group]:
            set_path(cfg, path, get_path(i03, path))
    return cfg


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_replay(config: Path, label: str, rerun: bool) -> Path:
    summary = REPLAY_DIR / f"{label}_summary.json"
    if summary.exists() and not rerun:
        return summary
    command = [sys.executable, str(ROOT / "scripts" / "replay_t0_decisions.py"),
               "--config", str(config), "--quotes", str(QUOTES),
               "--start-date", OOS_START, "--end-date", OOS_END,
               "--label", label, "--output-detail", "summary"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1800, check=False)
    if completed.returncode != 0 or not summary.exists():
        raise RuntimeError(f"replay failed for {label}: {completed.stderr[-2000:]} {completed.stdout[-2000:]}")
    return summary


def summary_metrics(path: Path) -> dict[str, Any]:
    summary = load(path)
    daily = {day: float(node.get("pnl", 0.0)) for day, node in sorted(summary["per_day"].items())}
    row = metrics(daily)
    net = metrics(daily_net_pnl(summary, 12.0))
    std = row["daily_std_sample"]
    row["daily_sharpe"] = round(row["daily_mean"] / std * math.sqrt(252), 4) if std else None
    row["net_12bps"] = net
    return row


def contribution(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "daily_std_reduction_pct": round((1 - row["daily_std_sample"] / base["daily_std_sample"]) * 100, 2),
        "worst_day_improvement": round(row["worst_day_pnl"] - base["worst_day_pnl"], 2),
        "total_pnl_delta": round(row["total_pnl"] - base["total_pnl"], 2),
        "sharpe_delta": round((row["daily_sharpe"] or 0) - (base["daily_sharpe"] or 0), 4),
    }


def group_score(base: dict[str, Any], row: dict[str, Any]) -> float:
    c = contribution(base, row)
    pnl_penalty = max(0.0, -c["total_pnl_delta"] / max(1.0, abs(base["total_pnl"]))) * 20
    return c["daily_std_reduction_pct"] + max(0.0, c["worst_day_improvement"] / 500.0) - pnl_penalty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    base, i03 = load(BASE), load(I03)
    configs = {"baseline": BASE, "i03": I03}
    changed_values = {}
    for group, paths in GROUPS.items():
        config_path = CONFIG_DIR / f"oos_i03_ablation_{group}.json"
        write_json(config_path, build_config(base, i03, [group]))
        configs[group] = config_path
        changed_values[group] = {path: {"baseline": get_path(base, path), "i03": get_path(i03, path)} for path in paths}
    labels = {name: f"i03_ablation_{name}" for name in configs}
    labels["baseline"] = "oos_variance_baseline"
    labels["i03"] = "oos_variance_i03"
    rows = {name: summary_metrics(run_replay(path, labels[name], args.rerun)) for name, path in configs.items()}
    ranked = sorted(GROUPS, key=lambda name: group_score(rows["baseline"], rows[name]), reverse=True)
    drivers = ranked[:2]
    # Diagnostic reruns must not rewrite a tracked candidate or create a new
    # promotion target. The derived combination lives only in replay outputs.
    minimal_path = REPLAY_DIR / "i03_ablation_minimal_runtime.json"
    minimal_cfg = build_config(base, i03, drivers)
    minimal_cfg["research_metadata"] = {
        "diagnostic_only": True, "selected_using_observed_oos": True,
        "groups": drivers, "requires_prospective_data_after": OOS_END,
    }
    write_json(minimal_path, minimal_cfg)
    rows["minimal_exploratory"] = summary_metrics(
        run_replay(minimal_path, "i03_ablation_minimal_exploratory", args.rerun))
    comparisons = {name: contribution(rows["baseline"], row) for name, row in rows.items() if name != "baseline"}
    report = {
        "schemaVersion": "i03_group_ablation_v1", "paperTradingOnly": True,
        "status": "diagnostic_only", "liveReady": False, "formalStrategyAllowed": False,
        "oos": [OOS_START, OOS_END], "oosAlreadyObserved": True,
        "metrics": rows, "comparisonsVsBaseline": comparisons,
        "changedValues": changed_values, "rankedGroups": ranked, "selectedDriverGroups": drivers,
        "minimalConfig": str(minimal_path),
        "decision": "exploratory only; validate on new post-2026-06-18 prospective data before shadow or live use",
        "methodLimitation": "group attribution is non-additive; the minimal combination was selected on the observed 21-day OOS window",
        "contaminated_warning": "Patched execution was rerun, but the legacy Yahoo60 source universe cannot recover ETFs excluded by its old full-day-turnover gate.",
    }
    write_json(OUT_JSON, report)
    lines = ["# i03 OOS Variance Mechanism — Group Ablation", "",
             "> CONTAMINATED WARNING: patched execution, but legacy Yahoo60 universe remains full-day-turnover contaminated.", "",
             "Observed OOS 2026-05-21..06-18; diagnostic only.", "",
             "| config | daily std | worst day | total PnL | Sharpe | std reduction | worst-day improvement |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        comp = comparisons.get(name, {"daily_std_reduction_pct": 0, "worst_day_improvement": 0})
        lines.append(f"| {name} | {row['daily_std_sample']:.2f} | {row['worst_day_pnl']:.2f} | "
                     f"{row['total_pnl']:.2f} | {row['daily_sharpe']} | {comp['daily_std_reduction_pct']:+.2f}% | "
                     f"{comp['worst_day_improvement']:+.2f} |")
    lines += ["", f"Driver groups (exploratory): `{drivers}`", "",
              f"Minimal config: `{minimal_path}`", "", "## Hard decision", "", report["decision"] + "."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
