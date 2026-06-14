"""Paper-only T+0 strategy parameter evolution.

This script runs offline replays over historical quote snapshots and writes a
bounded strategy overlay for the intraday ETF paper agent. It never calls the
paper-trading API and never changes execution locks. The live/paper agent will
only apply an overlay whose status is approved_for_paper_auto_apply and whose
strategy paths are allowlisted in configs/t0_intraday_paper_agent.json.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float, load_json, write_json


DEFAULT_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "t0_strategy_evolution"
DEFAULT_REPLAY_OUT = ROOT / "outputs" / "t0_replay"


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)
    return dst


def flatten_overlay(obj: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            out.extend(flatten_overlay(val, path))
        else:
            out.append((path, val))
    return out


def candidate_overlays() -> list[dict[str, Any]]:
    """Predefined candidate overlays. No search over live outcomes."""
    return [
        {
            "name": "baseline_current",
            "description": "Current locked config.",
            "overlay": {},
        },
        {
            "name": "precision_entry_gate",
            "description": "Fewer entries: require stronger momentum and higher entry score.",
            "overlay": {
                "entry_momentum_pct": 0.0020,
                "entry_score_threshold": 60,
                "cross_etf_divergence_threshold": 0.0025,
                "consolidation": {"breakout_buffer_pct": 0.0015},
                "bracket": {"risk_per_trade_pct": 0.0035, "risk_per_trade_pct_chaos_day": 0.0018},
            },
        },
        {
            "name": "patient_exit",
            "description": "Avoid immediate loss selling unless sell_score confirmation is stronger.",
            "overlay": {
                "min_hold_minutes": 20,
                "loss_review_after_minutes": 25,
                "loss_exit_score_threshold": 82,
                "profit_exit_score_threshold": 75,
                "profit_trailing_drawdown_pct": -0.007,
                "deceleration_exit_threshold": -0.003,
            },
        },
        {
            "name": "balanced_precision",
            "description": "Moderately stricter entries plus more patient exits.",
            "overlay": {
                "entry_momentum_pct": 0.0018,
                "entry_score_threshold": 58,
                "min_hold_minutes": 15,
                "loss_review_after_minutes": 20,
                "loss_exit_score_threshold": 78,
                "profit_exit_score_threshold": 72,
                "bracket": {"risk_per_trade_pct": 0.0035, "risk_per_trade_pct_chaos_day": 0.0018},
            },
        },
        {
            "name": "tight_risk_cut",
            "description": "Lower risk budget while keeping signal logic mostly unchanged.",
            "overlay": {
                "entry_score_threshold": 55,
                "bracket": {"risk_per_trade_pct": 0.0030, "risk_per_trade_pct_chaos_day": 0.0015},
                "indicators": {"intraday_atr": {"stop_multiplier": 1.4}},
            },
        },
        {
            "name": "trend_quality",
            "description": "Require better quality breakouts and stricter correlation stress.",
            "overlay": {
                "entry_score_threshold": 60,
                "consolidation": {"max_range_pct": 0.0025, "breakout_buffer_pct": 0.0015},
                "indicators": {
                    "bollinger_squeeze": {"squeeze_bandwidth_pct": 0.005, "breakout_buffer_pct": 0.0008}
                },
                "market_correlation_stress": {"avg_abs_corr_threshold": 0.68},
            },
        },
    ]


def validate_allowed_paths(base_cfg: dict[str, Any], overlay: dict[str, Any]) -> list[str]:
    allowed = set(base_cfg.get("self_iteration", {}).get("allowed_strategy_paths", []))
    if not allowed:
        return []
    return [path for path, _ in flatten_overlay(overlay) if path not in allowed]


def write_candidate_config(base_cfg: dict[str, Any], overlay: dict[str, Any], path: Path) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["self_iteration"] = copy.deepcopy(cfg.get("self_iteration", {}))
    cfg["self_iteration"]["auto_apply_changes"] = False
    deep_merge(cfg["strategy"], overlay)
    write_json(path, cfg)


def run_replay(config_path: Path, label: str, date_filter: str | None) -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "replay_t0_decisions.py"), "--config", str(config_path), "--label", label]
    if date_filter:
        cmd.extend(["--date", date_filter])
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=120)
    return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr).strip()


def load_replay_summary(label: str) -> dict[str, Any]:
    path = DEFAULT_REPLAY_OUT / f"{label}_summary.json"
    return load_json(path) if path.exists() else {}


def summarize_candidate(name: str, overlay: dict[str, Any], summary: dict[str, Any], ok: bool, log: str) -> dict[str, Any]:
    trades = summary.get("trades", []) if isinstance(summary.get("trades"), list) else []
    pnl_values = [as_float(t.get("pnl")) for t in trades if isinstance(t, dict)]
    winning = sum(1 for x in pnl_values if x > 0)
    losing = sum(1 for x in pnl_values if x < 0)
    entries = sum(int(as_float(d.get("entries"))) for d in (summary.get("per_day", {}) or {}).values() if isinstance(d, dict))
    max_day_loss = min([as_float(d.get("pnl")) for d in (summary.get("per_day", {}) or {}).values() if isinstance(d, dict)] or [0.0])
    total_pnl = as_float(summary.get("total_pnl"))
    open_count = len(summary.get("open_positions_at_end", {}) or {})
    win_rate = winning / len(pnl_values) if pnl_values else 0.0
    # Precision-first paper objective: prefer less realized loss, fewer loss exits,
    # no unresolved open position, and enough actual entries to avoid "do nothing" wins.
    score = (
        total_pnl
        - 0.8 * abs(min(max_day_loss, 0.0))
        - 250.0 * losing
        + 150.0 * winning
        + 300.0 * win_rate
        - 100.0 * open_count
        - 20.0 * max(entries - 2, 0)
    )
    return {
        "candidate": name,
        "ok": ok,
        "overlay": overlay,
        "rounds_total": int(as_float(summary.get("rounds_total"))),
        "entries": entries,
        "trades": len(pnl_values),
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "max_day_loss": round(max_day_loss, 2),
        "open_positions_at_end": open_count,
        "objective_score": round(score, 2),
        "replay_log_tail": log[-2000:],
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = [
        "candidate", "ok", "rounds_total", "entries", "trades", "winning_trades",
        "losing_trades", "win_rate", "total_pnl", "max_day_loss",
        "open_positions_at_end", "objective_score",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["candidates"]
    lines = [
        "# T0 Strategy Evolution Report",
        "",
        "Paper trading research only. Not live-ready, not a formal strategy, and not investment advice.",
        "",
        f"- Created at: {report['created_at']}",
        f"- Status: `{report['status']}`",
        f"- Selected candidate: `{report.get('selected_candidate')}`",
        f"- Decision reason: {report.get('decision_reason')}",
        "",
        "## Candidate Replay Results",
        "",
        "| candidate | entries | trades | win_rate | total_pnl | max_day_loss | open | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: as_float(x.get("objective_score")), reverse=True):
        lines.append(
            f"| {row['candidate']} | {row['entries']} | {row['trades']} | {row['win_rate']:.2%} | "
            f"{row['total_pnl']:.2f} | {row['max_day_loss']:.2f} | {row['open_positions_at_end']} | "
            f"{row['objective_score']:.2f} |"
        )
    lines.extend([
        "",
        "## Applied Overlay",
        "",
        "```json",
        json.dumps(report.get("strategy_overlay", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Guardrails",
        "",
        "- Only allowlisted `strategy` paths can be auto-applied by the agent.",
        "- `mode`, `execution_enabled`, `risk`, shared execution guard, and order submission locks are not writable through this overlay.",
        "- Evidence gates can downgrade the result to `diagnostic_only`; in that case the agent ignores the overlay.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-only T0 strategy evolution via offline replay")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--date", default=None, help="optional YYYY-MM-DD replay subset")
    parser.add_argument("--label-prefix", default=None)
    args = parser.parse_args()

    base_cfg = load_json(Path(args.config))
    si = base_cfg.get("self_iteration", {})
    out_dir = DEFAULT_OUT_DIR
    cfg_dir = out_dir / "candidate_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.label_prefix or f"evolution_{stamp}"
    rows: list[dict[str, Any]] = []
    invalid: dict[str, list[str]] = {}

    for cand in candidate_overlays():
        name = cand["name"]
        overlay = cand["overlay"]
        bad_paths = validate_allowed_paths(base_cfg, overlay)
        if bad_paths:
            invalid[name] = bad_paths
            rows.append({
                "candidate": name,
                "ok": False,
                "overlay": overlay,
                "rounds_total": 0,
                "entries": 0,
                "trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "max_day_loss": 0.0,
                "open_positions_at_end": 0,
                "objective_score": -1_000_000.0,
                "replay_log_tail": f"invalid overlay paths: {bad_paths}",
            })
            continue
        cfg_path = cfg_dir / f"{prefix}_{name}.json"
        write_candidate_config(base_cfg, overlay, cfg_path)
        label = f"{prefix}_{name}"
        ok, log = run_replay(cfg_path, label, args.date)
        summary = load_replay_summary(label) if ok else {}
        rows.append(summarize_candidate(name, overlay, summary, ok, log))

    rows_ok = [r for r in rows if r.get("ok")]
    baseline = next((r for r in rows_ok if r["candidate"] == "baseline_current"), None)
    selected = max(rows_ok, key=lambda x: as_float(x.get("objective_score"))) if rows_ok else None

    status = "diagnostic_only"
    decision_reason = "no_valid_replay"
    selected_overlay: dict[str, Any] = {}
    if baseline and selected:
        min_trades = int(as_float(si.get("min_replay_trades_for_auto_apply"), 2))
        min_entries = int(as_float(si.get("min_candidate_entries"), 1))
        min_improvement = as_float(si.get("min_score_improvement"), 100)
        improvement = as_float(selected.get("objective_score")) - as_float(baseline.get("objective_score"))
        selected_overlay = selected.get("overlay", {}) if isinstance(selected.get("overlay"), dict) else {}
        if selected["candidate"] == "baseline_current":
            decision_reason = "baseline_ranked_best"
            selected_overlay = {}
        elif int(selected.get("trades", 0)) < min_trades:
            decision_reason = f"selected_trades_below_min:{selected.get('trades')}<{min_trades}"
        elif int(selected.get("entries", 0)) < min_entries:
            decision_reason = f"selected_entries_below_min:{selected.get('entries')}<{min_entries}"
        elif improvement < min_improvement:
            decision_reason = f"score_improvement_below_min:{improvement:.2f}<{min_improvement:.2f}"
        elif as_float(selected.get("total_pnl")) < as_float(baseline.get("total_pnl")):
            decision_reason = "selected_total_pnl_worse_than_baseline"
        elif int(selected.get("losing_trades", 0)) > int(baseline.get("losing_trades", 0)):
            decision_reason = "selected_has_more_losing_trades_than_baseline"
        elif int(selected.get("open_positions_at_end", 0)) > int(baseline.get("open_positions_at_end", 0)):
            decision_reason = "selected_leaves_more_open_positions_than_baseline"
        else:
            status = "approved_for_paper_auto_apply"
            decision_reason = f"selected_improved_objective_by:{improvement:.2f}"

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_advice": False,
        "status": status,
        "decision_reason": decision_reason,
        "selected_candidate": selected.get("candidate") if selected else None,
        "baseline_candidate": baseline,
        "selected_candidate_metrics": selected,
        "strategy_overlay": selected_overlay if status == "approved_for_paper_auto_apply" else {},
        "suggested_strategy_overlay": selected_overlay,
        "invalid_overlay_paths": invalid,
        "candidates": rows,
        "note": "Offline replay over historical snapshots; small sample is not a profitability claim.",
    }

    write_json(out_dir / "latest_strategy_overlay.json", report)
    write_json(out_dir / f"{prefix}_summary.json", report)
    write_csv_rows(out_dir / f"{prefix}_candidate_results.csv", rows)
    write_markdown(out_dir / f"{prefix}_summary.md", report)
    write_markdown(out_dir / "latest_strategy_evolution.md", report)

    print(json.dumps({
        "status": status,
        "selected_candidate": report["selected_candidate"],
        "decision_reason": decision_reason,
        "overlay": str(out_dir / "latest_strategy_overlay.json"),
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
