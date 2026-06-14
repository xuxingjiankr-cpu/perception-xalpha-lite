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
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float, load_json, write_json


DEFAULT_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "t0_strategy_evolution"
DEFAULT_REPLAY_OUT = ROOT / "outputs" / "t0_replay"


PARAM_SPACE = [
    {"path": "entry_momentum_pct", "type": "float", "lo": 0.0012, "hi": 0.0028},
    {"path": "exit_momentum_pct", "type": "float", "lo": -0.0020, "hi": -0.0005},
    {"path": "entry_score_threshold", "type": "int", "lo": 48, "hi": 72},
    {"path": "loss_exit_score_threshold", "type": "int", "lo": 65, "hi": 90},
    {"path": "profit_exit_score_threshold", "type": "int", "lo": 58, "hi": 84},
    {"path": "min_profit_exit_pct", "type": "float", "lo": 0.002, "hi": 0.006},
    {"path": "min_hold_minutes", "type": "int", "lo": 8, "hi": 35},
    {"path": "loss_review_after_minutes", "type": "int", "lo": 10, "hi": 40},
    {"path": "profit_trailing_drawdown_pct", "type": "float", "lo": -0.012, "hi": -0.004},
    {"path": "deceleration_exit_threshold", "type": "float", "lo": -0.004, "hi": -0.001},
    {"path": "cross_etf_divergence_threshold", "type": "float", "lo": 0.0015, "hi": 0.0035},
    {"path": "consolidation.max_range_pct", "type": "float", "lo": 0.0020, "hi": 0.0040},
    {"path": "consolidation.breakout_buffer_pct", "type": "float", "lo": 0.0005, "hi": 0.0020},
    {"path": "indicators.rolling_vwap.require_price_above_for_entry", "type": "bool", "lo": 0.0, "hi": 1.0},
    {"path": "indicators.intraday_atr.stop_multiplier", "type": "float", "lo": 1.0, "hi": 1.8},
    {"path": "indicators.intraday_atr.min_stop_pct", "type": "float", "lo": 0.0015, "hi": 0.0035},
    {"path": "indicators.bollinger_squeeze.squeeze_bandwidth_pct", "type": "float", "lo": 0.0040, "hi": 0.0080},
    {"path": "indicators.bollinger_squeeze.breakout_buffer_pct", "type": "float", "lo": 0.0003, "hi": 0.0012},
    {"path": "market_correlation_stress.avg_abs_corr_threshold", "type": "float", "lo": 0.60, "hi": 0.85},
    {"path": "bracket.risk_per_trade_pct", "type": "float", "lo": 0.0020, "hi": 0.0045},
    {"path": "bracket.risk_per_trade_pct_chaos_day", "type": "float", "lo": 0.0010, "hi": 0.0022},
    {"path": "bracket.target1_r_multiple", "type": "float", "lo": 0.8, "hi": 1.4},
    {"path": "bracket.target2_r_multiple", "type": "float", "lo": 1.6, "hi": 2.6},
    {"path": "bracket.reentry_cooldown_minutes", "type": "int", "lo": 15, "hi": 60},
]


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


def get_nested(obj: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_nested(obj: dict[str, Any], dotted_path: str, value: Any) -> None:
    cur = obj
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


def normalize_param(value: Any, spec: dict[str, Any]) -> float:
    typ = spec["type"]
    if typ == "bool":
        return 1.0 if bool(value) else 0.0
    lo = as_float(spec["lo"])
    hi = as_float(spec["hi"])
    val = as_float(value, (lo + hi) / 2.0)
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def denormalize_param(x: float, spec: dict[str, Any]) -> Any:
    x = max(0.0, min(1.0, x))
    typ = spec["type"]
    if typ == "bool":
        return bool(x >= 0.5)
    lo = as_float(spec["lo"])
    hi = as_float(spec["hi"])
    val = lo + x * (hi - lo)
    if typ == "int":
        return int(round(val))
    return round(val, 6)


def overlay_from_vector(vec: list[float]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    for x, spec in zip(vec, PARAM_SPACE):
        set_nested(overlay, spec["path"], denormalize_param(x, spec))
    return overlay


def vector_from_strategy(strategy: dict[str, Any]) -> list[float]:
    vec: list[float] = []
    for spec in PARAM_SPACE:
        default_val = denormalize_param(0.5, spec)
        vec.append(normalize_param(get_nested(strategy, spec["path"], default_val), spec))
    return vec


def candidate_overlays() -> list[dict[str, Any]]:
    """Predefined candidate overlays. No search over live outcomes.

    The cs_ne_* candidates are a fixed-budget, deterministic trial inspired by
    recent cs.NE themes:
    - mixed categorical/continuous black-box optimization: combine discrete
      toggles and continuous thresholds;
    - dynamic-environment local EA: mutate only near the current policy;
    - multi-objective evolutionary selection: keep separate precision, risk,
      and diversity candidates instead of one opaque optimizer;
    - CMA-ES stopping-criteria caution: small fixed budget, no repeated search
      until more paper data arrives.
    """
    base = [
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
    base.extend([
        {
            "name": "cs_ne_local_mutation_entry_plus",
            "description": "Dynamic-EA style small local mutation: slightly stricter entry, same exit.",
            "overlay": {
                "entry_momentum_pct": 0.0017,
                "entry_score_threshold": 56,
                "consolidation": {"breakout_buffer_pct": 0.0012},
                "indicators": {"bollinger_squeeze": {"breakout_buffer_pct": 0.0007}},
            },
        },
        {
            "name": "cs_ne_local_mutation_exit_plus",
            "description": "Dynamic-EA style small local mutation: more confirmation before selling losers.",
            "overlay": {
                "min_hold_minutes": 15,
                "loss_review_after_minutes": 20,
                "loss_exit_score_threshold": 76,
                "deceleration_exit_threshold": -0.0025,
            },
        },
        {
            "name": "cs_ne_mixed_categorical_vwap_relaxed",
            "description": "Mixed categorical/continuous trial: relax VWAP hard gate, compensate with stronger score gate.",
            "overlay": {
                "entry_score_threshold": 64,
                "entry_momentum_pct": 0.0022,
                "indicators": {"rolling_vwap": {"require_price_above_for_entry": False}},
                "market_correlation_stress": {"avg_abs_corr_threshold": 0.68},
            },
        },
        {
            "name": "cs_ne_quality_diversity_gold_hk",
            "description": "Quality-diversity proxy: broader breakout criteria but lower risk budget.",
            "overlay": {
                "entry_score_threshold": 54,
                "cross_etf_divergence_threshold": 0.0018,
                "consolidation": {"max_range_pct": 0.0028},
                "bracket": {"risk_per_trade_pct": 0.0028, "risk_per_trade_pct_chaos_day": 0.0014},
            },
        },
        {
            "name": "cs_ne_risk_first_low_budget",
            "description": "Risk-first candidate: preserve signals but reduce position risk and widen ATR stop sanity.",
            "overlay": {
                "entry_score_threshold": 55,
                "bracket": {"risk_per_trade_pct": 0.0025, "risk_per_trade_pct_chaos_day": 0.0012},
                "indicators": {"intraday_atr": {"stop_multiplier": 1.5, "min_stop_pct": 0.0025}},
            },
        },
        {
            "name": "cs_ne_patient_profit_capture",
            "description": "Profit-capture mutation: avoid early profit exits unless drawdown confirmation is stronger.",
            "overlay": {
                "profit_exit_score_threshold": 78,
                "min_profit_exit_pct": 0.004,
                "profit_trailing_drawdown_pct": -0.008,
                "bracket": {"target1_r_multiple": 1.2, "target2_r_multiple": 2.2},
            },
        },
    ])
    return base


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
    per_day = summary.get("per_day", {}) or {}
    entries = sum(int(as_float(d.get("entries"))) for d in per_day.values() if isinstance(d, dict))
    max_day_loss = min([as_float(d.get("pnl")) for d in per_day.values() if isinstance(d, dict)] or [0.0])
    distinct_days = len([k for k in per_day if isinstance(per_day.get(k), dict)])
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
        "distinct_days": distinct_days,
        "objective_score": round(score, 2),
        "replay_log_tail": log[-2000:],
    }


def evaluate_candidate(
    base_cfg: dict[str, Any],
    cfg_dir: Path,
    prefix: str,
    name: str,
    overlay: dict[str, Any],
    date_filter: str | None,
) -> dict[str, Any]:
    bad_paths = validate_allowed_paths(base_cfg, overlay)
    if bad_paths:
        return {
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
            "invalid_overlay_paths": bad_paths,
        }
    cfg_path = cfg_dir / f"{prefix}_{name}.json"
    write_candidate_config(base_cfg, overlay, cfg_path)
    label = f"{prefix}_{name}"
    ok, log = run_replay(cfg_path, label, date_filter)
    summary = load_replay_summary(label) if ok else {}
    return summarize_candidate(name, overlay, summary, ok, log)


def cma_es_blackbox_candidates(
    base_cfg: dict[str, Any],
    cfg_dir: Path,
    prefix: str,
    date_filter: str | None,
    generations: int,
    population_size: int,
    seed: int,
    sigma0: float,
) -> list[dict[str, Any]]:
    """Run a bounded diagonal CMA-ES black-box optimization over allowlisted strategy params.

    This is deliberately offline and small-budget. It is a real distributional
    optimizer: samples a population, evaluates by replay, selects elites, and
    adapts the search mean and per-dimension variance across generations.
    """
    rng = random.Random(seed)
    mean = vector_from_strategy(base_cfg.get("strategy", {}))
    dim = len(mean)
    variance = [1.0 for _ in range(dim)]
    rows: list[dict[str, Any]] = []
    mu = max(2, population_size // 2)

    for gen in range(generations):
        gen_rows: list[dict[str, Any]] = []
        sigma = sigma0 * (0.85 ** gen)
        for idx in range(population_size):
            vec = [
                max(0.0, min(1.0, mean[j] + rng.gauss(0.0, sigma * (variance[j] ** 0.5))))
                for j in range(dim)
            ]
            overlay = overlay_from_vector(vec)
            name = f"cmaes_g{gen + 1:02d}_i{idx + 1:02d}"
            row = evaluate_candidate(base_cfg, cfg_dir, prefix, name, overlay, date_filter)
            row["optimizer"] = "bounded_diagonal_cma_es"
            row["generation"] = gen + 1
            gen_rows.append(row)
            rows.append(row)

        elites = sorted(
            [r for r in gen_rows if r.get("ok")],
            key=lambda r: as_float(r.get("objective_score")),
            reverse=True,
        )[:mu]
        if not elites:
            break

        elite_vecs = [vector_from_overlay(row.get("overlay", {}), base_cfg.get("strategy", {})) for row in elites]
        new_mean = [sum(v[j] for v in elite_vecs) / len(elite_vecs) for j in range(dim)]
        new_variance: list[float] = []
        for j in range(dim):
            centered = [(v[j] - new_mean[j]) ** 2 for v in elite_vecs]
            # Keep non-zero exploration while shrinking noisy dimensions.
            new_variance.append(max(0.05, min(1.5, sum(centered) / len(centered) + 0.10 * variance[j])))
        mean = [0.65 * mean[j] + 0.35 * new_mean[j] for j in range(dim)]
        variance = new_variance

    return rows


def vector_from_overlay(overlay: dict[str, Any], base_strategy: dict[str, Any]) -> list[float]:
    merged = copy.deepcopy(base_strategy)
    deep_merge(merged, overlay)
    return vector_from_strategy(merged)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = [
        "candidate", "optimizer", "generation", "ok", "rounds_total", "entries", "trades", "winning_trades",
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
        "| candidate | optimizer | entries | trades | win_rate | total_pnl | max_day_loss | open | score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: as_float(x.get("objective_score")), reverse=True):
        lines.append(
            f"| {row['candidate']} | {row.get('optimizer', 'fixed_grid')} | {row['entries']} | {row['trades']} | {row['win_rate']:.2%} | "
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
        "- CMA-ES candidates are evaluated offline by replay only; they never call the paper-trading API.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-only T0 strategy evolution via offline replay")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--date", default=None, help="optional YYYY-MM-DD replay subset")
    parser.add_argument("--label-prefix", default=None)
    parser.add_argument("--optimizer", choices=["fixed", "cmaes", "both"], default=None)
    parser.add_argument("--cma-generations", type=int, default=None)
    parser.add_argument("--cma-population", type=int, default=None)
    parser.add_argument("--cma-seed", type=int, default=None)
    parser.add_argument("--cma-sigma", type=float, default=None)
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

    optimizer_cfg = si.get("blackbox_optimizer", {}) if isinstance(si.get("blackbox_optimizer", {}), dict) else {}
    optimizer_mode = args.optimizer or str(optimizer_cfg.get("mode", "both"))

    if optimizer_mode in {"fixed", "both"}:
        fixed_candidates = candidate_overlays()
    elif optimizer_mode == "cmaes":
        fixed_candidates = [candidate_overlays()[0]]
    else:
        fixed_candidates = []

    if fixed_candidates:
        for cand in fixed_candidates:
            name = cand["name"]
            row = evaluate_candidate(base_cfg, cfg_dir, prefix, name, cand["overlay"], args.date)
            row["optimizer"] = "fixed_grid"
            if row.get("invalid_overlay_paths"):
                invalid[name] = row["invalid_overlay_paths"]
            rows.append(row)

    if optimizer_mode in {"cmaes", "both"}:
        cma_rows = cma_es_blackbox_candidates(
            base_cfg=base_cfg,
            cfg_dir=cfg_dir,
            prefix=prefix,
            date_filter=args.date,
            generations=args.cma_generations or int(as_float(optimizer_cfg.get("generations"), 2)),
            population_size=args.cma_population or int(as_float(optimizer_cfg.get("population_size"), 4)),
            seed=args.cma_seed or int(as_float(optimizer_cfg.get("seed"), 20260614)),
            sigma0=args.cma_sigma if args.cma_sigma is not None else as_float(optimizer_cfg.get("sigma0"), 0.22),
        )
        for row in cma_rows:
            if row.get("invalid_overlay_paths"):
                invalid[row["candidate"]] = row["invalid_overlay_paths"]
        rows.extend(cma_rows)

    rows_ok = [r for r in rows if r.get("ok")]
    baseline = next((r for r in rows_ok if r["candidate"] == "baseline_current"), None)
    selected = max(rows_ok, key=lambda x: as_float(x.get("objective_score"))) if rows_ok else None

    status = "diagnostic_only"
    decision_reason = "no_valid_replay"
    selected_overlay: dict[str, Any] = {}
    if baseline and selected:
        min_trades = int(as_float(si.get("min_replay_trades_for_auto_apply"), 2))
        min_entries = int(as_float(si.get("min_candidate_entries"), 1))
        min_distinct_days = int(as_float(si.get("min_distinct_days"), 0))
        min_improvement = as_float(si.get("min_score_improvement"), 100)
        max_day_loss_worsening_allowed = as_float(si.get("max_day_loss_worsening_allowed"), 0.0)
        improvement = as_float(selected.get("objective_score")) - as_float(baseline.get("objective_score"))
        max_day_loss_worsening = as_float(baseline.get("max_day_loss")) - as_float(selected.get("max_day_loss"))
        selected_overlay = selected.get("overlay", {}) if isinstance(selected.get("overlay"), dict) else {}
        if selected["candidate"] == "baseline_current":
            decision_reason = "baseline_ranked_best"
            selected_overlay = {}
        elif int(selected.get("trades", 0)) < min_trades:
            decision_reason = f"selected_trades_below_min:{selected.get('trades')}<{min_trades}"
        elif int(selected.get("entries", 0)) < min_entries:
            decision_reason = f"selected_entries_below_min:{selected.get('entries')}<{min_entries}"
        elif int(selected.get("distinct_days", 0)) < min_distinct_days:
            decision_reason = f"sample_distinct_days_below_min:{selected.get('distinct_days')}<{min_distinct_days}"
        elif improvement < min_improvement:
            decision_reason = f"score_improvement_below_min:{improvement:.2f}<{min_improvement:.2f}"
        elif as_float(selected.get("total_pnl")) < as_float(baseline.get("total_pnl")):
            decision_reason = "selected_total_pnl_worse_than_baseline"
        elif max_day_loss_worsening > max_day_loss_worsening_allowed:
            decision_reason = (
                f"selected_max_day_loss_worsening_too_large:"
                f"{max_day_loss_worsening:.2f}>{max_day_loss_worsening_allowed:.2f}"
            )
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
        "auto_apply_risk_limits": {
            "max_day_loss_worsening_allowed": as_float(si.get("max_day_loss_worsening_allowed"), 0.0),
        },
        "strategy_overlay": selected_overlay if status == "approved_for_paper_auto_apply" else {},
        "suggested_strategy_overlay": selected_overlay,
        "optimizer_mode": optimizer_mode,
        "cma_es": {
            "enabled": optimizer_mode in {"cmaes", "both"},
            "generations": args.cma_generations or int(as_float(optimizer_cfg.get("generations"), 2)),
            "population_size": args.cma_population or int(as_float(optimizer_cfg.get("population_size"), 4)),
            "seed": args.cma_seed or int(as_float(optimizer_cfg.get("seed"), 20260614)),
            "sigma0": args.cma_sigma if args.cma_sigma is not None else as_float(optimizer_cfg.get("sigma0"), 0.22),
            "param_space_size": len(PARAM_SPACE),
            "implementation": "bounded_diagonal_cma_es_offline_replay",
        },
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
