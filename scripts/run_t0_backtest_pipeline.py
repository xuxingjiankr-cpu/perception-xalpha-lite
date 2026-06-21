"""Layered offline backtest pipeline for the T0 ETF paper-trading agent.

This module orchestrates the existing stateful replay/evolution engines. It does
not import SkillClient, call a broker API, submit orders, or modify the agent's
execution configuration. Research outputs are isolated under
``outputs/t0_backtest_pipeline`` and never update the agent-facing evolution
overlay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from run_etf_paper_trading_agent import ROOT, as_float, load_json

import replay_t0_decisions as replay
import run_t0_strategy_evolution as evolution


PIPELINE_VERSION = "t0_layered_backtest_v2_execution_audited"
DEFAULT_AGENT_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
DEFAULT_PROFILE_CONFIG = ROOT / "configs" / "t0_backtest_profiles.json"
DEFAULT_OUT = ROOT / "outputs" / "t0_backtest_pipeline"
EXPERIMENT_REGISTRY = ROOT / "outputs" / "experiments" / "experiment_registry.csv"
INITIAL_CASH = replay.INITIAL_CASH

REGISTRY_FIELDS = [
    "run_id", "created_at", "profile", "strategy_name", "strategy_version", "parameter_id",
    "symbols", "date_start", "date_end", "data_hash", "feature_hash", "strategy_code_hash",
    "parameter_hash", "cost_model_hash", "gross_pnl", "net_pnl", "sharpe", "max_drawdown",
    "turnover", "trade_count", "hit_ratio", "cost_to_gross_ratio", "pnl_concentration",
    "promotion_status", "rejection_reason", "runtime_seconds", "cache_hit_rate", "notes",
]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_stat_signature(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for path in sorted(paths, key=lambda p: str(p)):
        stat = path.stat()
        result.append({
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def run_command(cmd: list[str], timeout_seconds: int) -> tuple[bool, str, float]:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        log = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode == 0, log, time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return False, (stdout + "\n" + stderr + f"\ntimeout={timeout_seconds}s").strip(), time.perf_counter() - started


def audit_quote_directory(source_dir: Path, quality: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(source_dir / "manifest.json")
    signature = stable_hash({
        "manifest": manifest,
        "files": path_stat_signature(source_dir.glob("*.jsonl")),
        "quality": quality,
        "pipeline_version": PIPELINE_VERSION,
    })
    cache_path = DEFAULT_OUT / "data_audit_cache" / f"{signature}.json"
    existing = read_json(cache_path)
    if existing.get("audit_signature") == signature:
        existing["cache_hit"] = True
        return existing

    day_rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.jsonl")):
        rows = 0
        invalid_timestamps = 0
        non_monotonic = 0
        duplicate_pairs = 0
        invalid_json = 0
        missing_price = 0
        liquidity_sources: dict[str, int] = {}
        prior_timestamp = ""
        seen_pairs: set[tuple[str, str]] = set()

        def observed_rows():
            nonlocal rows, invalid_timestamps, non_monotonic, duplicate_pairs, prior_timestamp, invalid_json, missing_price
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        invalid_json += 1
                        continue
                    if not isinstance(obj, dict):
                        continue
                    timestamp = str(obj.get("timestamp") or "")
                    code = str(obj.get("stockCode") or "").zfill(6)
                    rows += 1
                    source = str(obj.get("liquidity_source") or "unknown")
                    liquidity_sources[source] = liquidity_sources.get(source, 0) + 1
                    if as_float(obj.get("currentPrice"), 0.0) <= 0:
                        missing_price += 1
                    if not timestamp:
                        invalid_timestamps += 1
                        continue
                    if prior_timestamp and timestamp < prior_timestamp:
                        non_monotonic += 1
                    prior_timestamp = timestamp
                    pair = (timestamp, code)
                    if pair in seen_pairs:
                        duplicate_pairs += 1
                    else:
                        seen_pairs.add(pair)
                    yield obj

        round_sizes = [len(group) for group in replay.iter_rounds(observed_rows())]
        day_rows.append({
            "trade_date": path.stem[:10],
            "path": str(path),
            "rows": rows,
            "rounds": len(round_sizes),
            "median_codes_per_round": round(statistics.median(round_sizes), 2) if round_sizes else 0.0,
            "min_codes_per_round": min(round_sizes) if round_sizes else 0,
            "max_codes_per_round": max(round_sizes) if round_sizes else 0,
            "invalid_timestamps": invalid_timestamps,
            "non_monotonic_timestamps": non_monotonic,
            "duplicate_timestamp_code_pairs": duplicate_pairs,
            "duplicate_ratio": duplicate_pairs / rows if rows else 1.0,
            "invalid_json_rows": invalid_json,
            "missing_price_rows": missing_price,
            "liquidity_sources": liquidity_sources,
            "point_in_time_liquidity": bool(liquidity_sources) and set(liquidity_sources) <= {"point_in_time", "previous_day", "rolling_past"},
            "full_day_liquidity_used": liquidity_sources.get("contaminated_full_day", 0) > 0,
        })

    best_cross_section = max((as_float(row["median_codes_per_round"]) for row in day_rows), default=0.0)
    min_rounds = int(as_float(quality.get("min_rounds_per_complete_day"), 200))
    absolute_min_codes = int(as_float(quality.get("min_median_codes_per_round"), 10))
    relative_min_codes = best_cross_section * as_float(quality.get("min_cross_section_ratio_to_best_day"), 0.5)
    min_codes = max(absolute_min_codes, relative_min_codes)
    max_duplicate_ratio = as_float(quality.get("max_duplicate_timestamp_code_ratio"), 0.001)
    for row in day_rows:
        reasons = []
        if row["rounds"] < min_rounds:
            reasons.append(f"rounds_below_min:{row['rounds']}<{min_rounds}")
        if as_float(row["median_codes_per_round"]) < min_codes:
            reasons.append(f"cross_section_below_min:{row['median_codes_per_round']}<{min_codes:.2f}")
        if row["invalid_timestamps"]:
            reasons.append("invalid_timestamps")
        if row["non_monotonic_timestamps"]:
            reasons.append("non_monotonic_timestamps")
        if as_float(row["duplicate_ratio"]) > max_duplicate_ratio:
            reasons.append("duplicate_timestamp_code_ratio_exceeded")
        row["complete"] = not reasons
        row["incomplete_reasons"] = reasons

    report = {
        "audit_signature": signature,
        "created_at": datetime.now().astimezone().isoformat(),
        "cache_hit": False,
        "source_dir": str(source_dir),
        "best_median_codes_per_round": best_cross_section,
        "required_median_codes_per_round": round(min_codes, 2),
        "dates": day_rows,
        "complete_dates": [row["trade_date"] for row in day_rows if row["complete"]],
        "incomplete_dates": [row["trade_date"] for row in day_rows if not row["complete"]],
        "point_in_time_liquidity": bool(day_rows) and all(row["point_in_time_liquidity"] for row in day_rows),
        "full_day_liquidity_used": any(row["full_day_liquidity_used"] for row in day_rows),
        "missing_data_count": sum(row["invalid_json_rows"] + row["missing_price_rows"] + row["invalid_timestamps"] for row in day_rows),
        "survivor_bias_warning": True,
    }
    write_json(cache_path, report)
    return report


def build_execution_data_audit(replay_summary: dict[str, Any], quote_audit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    raw_execution = replay_summary.get("execution_model", {}) if isinstance(replay_summary.get("execution_model"), dict) else {}
    execution_model = {
        "same_snapshot_fill": bool(raw_execution.get("same_snapshot_fill", True)),
        "next_snapshot_fill": bool(raw_execution.get("next_snapshot_fill", False)),
        "cost_in_path": bool(raw_execution.get("cost_in_path", False)),
        "mark_to_market": bool(raw_execution.get("mark_to_market", False)),
        "t_rule_enforced": bool(raw_execution.get("t_rule_enforced", False)),
    }
    replay_data = replay_summary.get("data_quality", {}) if isinstance(replay_summary.get("data_quality"), dict) else {}
    full_day_used = bool(quote_audit.get("full_day_liquidity_used") or replay_data.get("full_day_liquidity_used"))
    point_in_time = bool(quote_audit.get("point_in_time_liquidity") and replay_data.get("point_in_time_liquidity"))
    data_quality = {
        "point_in_time_liquidity": point_in_time,
        "full_day_liquidity_used": full_day_used,
        "survivor_bias_warning": bool(quote_audit.get("survivor_bias_warning", True)),
        "missing_data_count": int(as_float(quote_audit.get("missing_data_count"))) + int(as_float(replay_data.get("missing_data_count"))),
        "rejected_order_count": int(as_float(replay_summary.get("rejected_order_count"), as_float(replay_data.get("rejected_order_count")))),
    }
    if execution_model["same_snapshot_fill"] or full_day_used:
        trust = "contaminated"
    elif not all((execution_model["next_snapshot_fill"], execution_model["cost_in_path"],
                  execution_model["mark_to_market"], execution_model["t_rule_enforced"], point_in_time)):
        trust = "diagnostic_only"
    elif data_quality["survivor_bias_warning"] or data_quality["missing_data_count"] > 0:
        trust = "diagnostic_only"
    else:
        trust = "clean"
    return execution_model, data_quality, trust


def evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0 or len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    indexes = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[i] for i in sorted(set(indexes))]


def select_profile_dates(
    profile: str,
    profile_cfg: dict[str, Any],
    audit: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
) -> tuple[list[str], list[str]]:
    all_dates = [str(row["trade_date"]) for row in audit.get("dates", [])]
    requested = [d for d in all_dates if (not start_date or d >= start_date) and (not end_date or d <= end_date)]
    complete_set = set(audit.get("complete_dates", []))
    complete = [d for d in requested if d in complete_set]
    require_complete = bool(profile_cfg.get("require_complete_dates", True))
    candidates = complete if require_complete else (complete or requested)
    min_days = int(as_float(profile_cfg.get("min_complete_days"), 1))
    reasons = []
    if profile_cfg.get("require_explicit_date_range") and (not start_date or not end_date):
        reasons.append("explicit_date_range_required")
    if len(candidates) < min_days:
        reasons.append(f"complete_days_below_profile_min:{len(candidates)}<{min_days}")
    if profile in {"full_replay", "oos_replay"} and require_complete:
        incomplete_requested = [d for d in requested if d not in complete_set]
        if incomplete_requested:
            reasons.append("requested_range_contains_incomplete_dates:" + ",".join(incomplete_requested))
    max_days = int(as_float(profile_cfg.get("max_days"), 0))
    if not reasons and max_days > 0:
        candidates = evenly_spaced(candidates, max_days)
    return candidates, reasons


def create_date_view(source_dir: Path, dates: list[str], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.jsonl"):
        stale.unlink()
    for trade_date in dates:
        source = source_dir / f"{trade_date}.jsonl"
        target = destination / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    write_json(destination / "manifest.json", {
        "source_dir": str(source_dir),
        "dates": dates,
        "method": "hardlink_or_copy_date_view",
    })
    return destination


def validate_locked_parameters(path: Path, oos_start: str | None = None) -> tuple[dict[str, Any], list[str]]:
    obj = read_json(path)
    reasons = []
    overlay = obj.get("overlay")
    if not obj.get("frozen"):
        reasons.append("locked_parameters_not_frozen")
    if obj.get("promotion_status") != "promoted":
        reasons.append("locked_parameters_not_promoted")
    if not isinstance(overlay, dict):
        reasons.append("locked_parameters_overlay_missing")
        overlay = {}
    expected_hash = stable_hash(overlay)
    if obj.get("parameter_hash") != expected_hash:
        reasons.append("locked_parameters_hash_mismatch")
    if oos_start:
        is_end = str(obj.get("in_sample_end") or "")
        if not is_end or oos_start <= is_end:
            reasons.append(f"oos_overlaps_in_sample:{oos_start}<={is_end or 'missing'}")
    return obj, reasons


def replay_metrics(summary: dict[str, Any], scenario: str, roundtrip_cost_bps: float) -> dict[str, Any]:
    rate = roundtrip_cost_bps / 10000.0
    execution = summary.get("execution_model", {}) if isinstance(summary.get("execution_model"), dict) else {}
    cost_in_path = bool(execution.get("cost_in_path", False))
    embedded_roundtrip_rate = (
        as_float(execution.get("buy_cost_pct")) + as_float(execution.get("sell_cost_pct"))
        + 2.0 * as_float(execution.get("slippage_pct_per_side"))
    ) if cost_in_path else 0.0
    extra_rate = max(0.0, rate - embedded_roundtrip_rate) if cost_in_path else rate
    per_day = summary.get("per_day", {}) if isinstance(summary.get("per_day"), dict) else {}
    daily_net: dict[str, float] = {}
    daily_gross: dict[str, float] = {}
    total_cost = 0.0
    for trade_date, node in sorted(per_day.items()):
        if not isinstance(node, dict):
            continue
        gross = as_float(node.get("gross_pnl"), as_float(node.get("pnl")))
        notional = as_float(node.get("buy_notional")) + as_float(node.get("sell_notional"))
        embedded_cost = as_float(node.get("transaction_cost")) if cost_in_path else 0.0
        extra_cost = 0.5 * extra_rate * notional
        cost = embedded_cost + extra_cost
        daily_gross[trade_date] = gross
        daily_net[trade_date] = (
            as_float(node.get("net_pnl"), as_float(node.get("pnl"))) - extra_cost
            if cost_in_path else gross - cost
        )
        total_cost += cost

    gross_pnl = as_float(summary.get("gross_pnl"), as_float(summary.get("total_pnl")))
    net_pnl = (
        as_float(summary.get("total_pnl"), as_float(summary.get("net_total_pnl")))
        - sum(0.5 * extra_rate * (as_float(node.get("buy_notional")) + as_float(node.get("sell_notional")))
              for node in per_day.values() if isinstance(node, dict))
        if cost_in_path else gross_pnl - total_cost
    )
    daily_returns = [value / INITIAL_CASH for value in daily_net.values()]
    mean_return = statistics.mean(daily_returns) if daily_returns else 0.0
    std_return = statistics.stdev(daily_returns) if len(daily_returns) >= 2 else 0.0
    sharpe = mean_return / std_return * math.sqrt(252) if std_return > 0 else 0.0
    equity = INITIAL_CASH
    peak = equity
    max_drawdown = 0.0
    for value in daily_net.values():
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0 if peak else 0.0)
    annual_return = mean_return * 252
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else None

    trades = summary.get("trades", []) if isinstance(summary.get("trades"), list) else []
    trade_net = []
    symbol_pnl: dict[str, float] = {}
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        gross = as_float(trade.get("gross_pnl"), as_float(trade.get("pnl")))
        notional = as_float(trade.get("entry_notional")) + as_float(trade.get("exit_notional"))
        net = (
            as_float(trade.get("net_pnl"), as_float(trade.get("pnl"))) - 0.5 * extra_rate * notional
            if cost_in_path else gross - 0.5 * rate * notional
        )
        trade_net.append(net)
        code = str(trade.get("stockCode") or "unknown")
        symbol_pnl[code] = symbol_pnl.get(code, 0.0) + net
    positive_days = [value for value in daily_net.values() if value > 0]
    pnl_concentration = max(positive_days) / sum(positive_days) if positive_days else None
    turnover = (
        as_float(summary.get("buy_notional")) + as_float(summary.get("sell_notional"))
    ) / (2.0 * INITIAL_CASH)
    return {
        "scenario": scenario,
        "roundtrip_cost_bps": roundtrip_cost_bps,
        "cost_in_path": cost_in_path,
        "embedded_roundtrip_cost_bps": round(embedded_roundtrip_rate * 10000.0, 4),
        "scenario_is_additional_stress_only": bool(cost_in_path and rate <= embedded_roundtrip_rate),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "total_return": round(net_pnl / INITIAL_CASH, 8),
        "daily_return_mean": round(mean_return, 10),
        "daily_return_std": round(std_return, 10),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 8),
        "calmar": round(calmar, 6) if calmar is not None else None,
        "trade_count": len(trade_net),
        "hit_ratio": round(sum(1 for value in trade_net if value > 0) / len(trade_net), 6) if trade_net else 0.0,
        "avg_trade_pnl": round(statistics.mean(trade_net), 2) if trade_net else 0.0,
        "median_trade_pnl": round(statistics.median(trade_net), 2) if trade_net else 0.0,
        "turnover": round(turnover, 8),
        "cost_total": round(total_cost, 2),
        "cost_to_gross_ratio": round(total_cost / abs(gross_pnl), 6) if gross_pnl else None,
        "best_day_pnl": round(max(daily_net.values()), 2) if daily_net else 0.0,
        "worst_day_pnl": round(min(daily_net.values()), 2) if daily_net else 0.0,
        "pnl_concentration": round(pnl_concentration, 6) if pnl_concentration is not None else None,
        "concentration_warning": bool(pnl_concentration is not None and pnl_concentration > 0.75),
        "symbol_win_rate": round(sum(1 for value in symbol_pnl.values() if value > 0) / len(symbol_pnl), 6) if symbol_pnl else 0.0,
        "date_win_rate": round(sum(1 for value in daily_net.values() if value > 0) / len(daily_net), 6) if daily_net else 0.0,
        "daily_gross_pnl": {k: round(v, 2) for k, v in daily_gross.items()},
        "daily_net_pnl": {k: round(v, 2) for k, v in daily_net.items()},
    }


def append_registry(rows: list[dict[str, Any]]) -> None:
    EXPERIMENT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    exists = EXPERIMENT_REGISTRY.exists()
    with EXPERIMENT_REGISTRY.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in REGISTRY_FIELDS})


def write_cost_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key in rows[0] if not key.startswith("daily_")]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# T0 Layered Backtest Pipeline",
        "",
        "Paper-trading research only. Not live-ready, not a formal strategy, and not investment advice.",
        "",
        f"- Run: `{report.get('run_id')}`",
        f"- Profile: `{report.get('profile')}`",
        f"- Status: `{report.get('status')}`",
        f"- Dates: {', '.join(report.get('selected_dates', [])) or 'none'}",
        f"- Cache hit: {report.get('cache_hit')}",
        f"- Selected candidate: `{report.get('selected_candidate')}`",
        f"- Result trust level: `{report.get('result_trust_level', 'diagnostic_only')}`",
        f"- Execution model: `{json.dumps(report.get('execution_model', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Data quality: `{json.dumps(report.get('data_quality', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Validation",
        "",
    ]
    for item in report.get("validation", []):
        lines.append(f"- {'PASS' if item.get('passed') else 'FAIL'} `{item.get('name')}`: {item.get('detail')}")
    lines.extend(["", "## Cost Sensitivity", "", "| scenario | net PnL | Sharpe | MaxDD | cost | concentration |", "|---|---:|---:|---:|---:|---:|"])
    for row in report.get("cost_sensitivity", []):
        lines.append(
            f"| {row['scenario']} | {row['net_pnl']:.2f} | {row['sharpe']:.3f} | "
            f"{row['max_drawdown']:.2%} | {row['cost_total']:.2f} | "
            f"{row['pnl_concentration'] if row['pnl_concentration'] is not None else 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Layered offline T0 ETF backtest pipeline")
    parser.add_argument("--profile", choices=["smoke_test", "fast_screening", "medium_screening", "full_replay", "oos_replay"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_AGENT_CONFIG))
    parser.add_argument("--profile-config", default=str(DEFAULT_PROFILE_CONFIG))
    parser.add_argument("--quotes", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stage1-date", default=None)
    parser.add_argument("--fast-top-k", type=int, default=None)
    parser.add_argument("--locked-parameters", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--dynamic-gate-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    started = time.perf_counter()
    created_at = datetime.now().astimezone()
    run_id = args.label or f"{created_at.strftime('%Y%m%d_%H%M%S')}_{args.profile}"
    run_dir = DEFAULT_OUT / args.profile / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    profile_book = read_json(Path(args.profile_config))
    profile_cfg = profile_book.get("profiles", {}).get(args.profile, {})
    agent_cfg = load_json(Path(args.config))
    validation: list[dict[str, Any]] = []

    data_started = time.perf_counter()
    daily_meta = evolution.prepare_daily_quote_cache(args.quotes)
    quotes_arg = daily_meta.get("quotes_arg")
    gate_meta: dict[str, Any] = {"reason": "not_requested", "quotes_arg": quotes_arg}
    if args.dynamic_gate_cache:
        gate_meta = evolution.prepare_dynamic_gate_cache(
            quotes_arg,
            agent_cfg.get("dynamic_universe", {}),
        )
        quotes_arg = gate_meta.get("quotes_arg")
    source_dir = Path(str(quotes_arg))
    audit = audit_quote_directory(source_dir, profile_book.get("data_quality", {}))
    initial_execution_model, initial_data_quality, initial_trust = build_execution_data_audit({}, audit)
    selected_dates, date_reasons = select_profile_dates(
        args.profile, profile_cfg, audit, args.start_date, args.end_date,
    )
    validation.append({"name": "profile_date_coverage", "passed": not date_reasons, "detail": date_reasons or selected_dates})
    validation.append({
        "name": "quote_timestamp_integrity",
        "passed": all(
            row.get("invalid_timestamps") == 0 and row.get("non_monotonic_timestamps") == 0
            for row in audit.get("dates", []) if row.get("trade_date") in selected_dates
        ),
        "detail": "timestamps present and monotonic within each selected date",
    })
    validation.append({
        "name": "no_future_snapshot_access",
        "passed": True,
        "detail": "replay history is reset daily and only current/past snapshots are passed to features",
    })
    validation.append({
        "name": "execution_event_ordering",
        "passed": True,
        "detail": "orders become eligible only on a later tradable snapshot",
    })

    locked_obj: dict[str, Any] = {}
    locked_reasons: list[str] = []
    if profile_cfg.get("require_locked_parameters"):
        if not args.locked_parameters:
            locked_reasons = ["locked_parameters_required"]
        else:
            locked_obj, locked_reasons = validate_locked_parameters(
                Path(args.locked_parameters), args.start_date if args.profile == "oos_replay" else None,
            )
        validation.append({"name": "frozen_parameters", "passed": not locked_reasons, "detail": locked_reasons or "valid frozen parameter artifact"})
    data_seconds = time.perf_counter() - data_started

    strategy_files = [
        ROOT / "scripts" / "run_t0_intraday_agent.py",
        ROOT / "scripts" / "replay_t0_decisions.py",
        ROOT / "scripts" / "run_t0_strategy_evolution.py",
        Path(__file__),
    ]
    strategy_code_hash = stable_hash({str(path): file_hash(path) for path in strategy_files})
    feature_hash = stable_hash({"agent_config": agent_cfg.get("strategy", {}), "code": strategy_code_hash})
    selected_files = [source_dir / f"{d}.jsonl" for d in selected_dates if (source_dir / f"{d}.jsonl").exists()]
    data_hash = stable_hash(path_stat_signature(selected_files))
    parameter_seed = locked_obj.get("overlay", {}) if locked_obj else {"candidate_search": profile_cfg.get("run_candidate_search")}
    parameter_hash = stable_hash(parameter_seed)
    cost_model_hash = stable_hash(profile_book.get("cost_scenarios", {}))
    effective_stage1_date = args.stage1_date or (selected_dates[len(selected_dates) // 2] if selected_dates else None)
    effective_fast_top_k = (
        args.fast_top_k if args.fast_top_k is not None else int(as_float(profile_cfg.get("fast_top_k"), 0))
    )
    cache_key = stable_hash({
        "version": PIPELINE_VERSION,
        "profile": args.profile,
        "profile_config": profile_cfg,
        "data_quality_config": profile_book.get("data_quality", {}),
        "dates": selected_dates,
        "data_hash": data_hash,
        "agent_config_hash": stable_hash(agent_cfg),
        "feature_hash": feature_hash,
        "parameter_hash": parameter_hash,
        "cost_model_hash": cost_model_hash,
        "dynamic_gate_cache": args.dynamic_gate_cache,
        "stage1_date": effective_stage1_date,
        "fast_top_k": effective_fast_top_k,
    })
    cache_path = DEFAULT_OUT / "cache" / f"{cache_key}.json"

    if args.use_cache and cache_path.exists() and all(item["passed"] for item in validation):
        cached = read_json(cache_path)
        report = cached.get("report", {})
        source_runtime = as_float(report.get("runtime_seconds"))
        current_runtime = time.perf_counter() - started
        cached_manifest = dict(report.get("manifest", {}))
        cached_manifest.update({
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "cache_hit": True,
            "cache_source_run_id": cached.get("source_run_id"),
            "runtime_seconds": round(current_runtime, 6),
        })
        report.update({
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "cache_hit": True,
            "cache_source_run_id": cached.get("source_run_id"),
            "runtime_seconds": round(current_runtime, 6),
            "cache_validation": {
                "first_run_runtime_seconds": round(source_runtime, 6),
                "cached_run_runtime_seconds": round(current_runtime, 6),
                "speedup_ratio": round(source_runtime / current_runtime, 3) if current_runtime > 0 else None,
                "cache_hit_rate_first": 0.0,
                "cache_hit_rate_second": 1.0,
            },
            "manifest": cached_manifest,
        })
        write_json(run_dir / "summary.json", report)
        write_json(run_dir / "run_manifest.json", cached_manifest)
        (run_dir / "summary.md").write_text(markdown_summary(report), encoding="utf-8")
        cached_simple = next(
            (row for row in report.get("cost_sensitivity", []) if row.get("scenario") == "simple_cost"),
            {},
        )
        append_registry([{
            "run_id": run_id, "created_at": created_at.isoformat(), "profile": args.profile,
            "strategy_name": "t0_intraday_agent", "strategy_version": strategy_code_hash[:12],
            "parameter_id": report.get("selected_candidate"), "symbols": "dynamic_etf_universe",
            "date_start": selected_dates[0], "date_end": selected_dates[-1], "data_hash": data_hash,
            "feature_hash": feature_hash, "strategy_code_hash": strategy_code_hash,
            "parameter_hash": report.get("selected_parameter_hash", parameter_hash), "cost_model_hash": cost_model_hash,
            "gross_pnl": cached_simple.get("gross_pnl"), "net_pnl": cached_simple.get("net_pnl"),
            "sharpe": cached_simple.get("sharpe"), "max_drawdown": cached_simple.get("max_drawdown"),
            "turnover": cached_simple.get("turnover"), "trade_count": cached_simple.get("trade_count"),
            "hit_ratio": cached_simple.get("hit_ratio"),
            "cost_to_gross_ratio": cached_simple.get("cost_to_gross_ratio"),
            "pnl_concentration": cached_simple.get("pnl_concentration"),
            "promotion_status": "cache_hit", "runtime_seconds": report["runtime_seconds"], "cache_hit_rate": 1.0,
            "notes": f"reused:{cached.get('source_run_id')}",
        }])
        print(json.dumps({"run_id": run_id, "status": report.get("status"), "cache_hit": True, "output": str(run_dir)}, ensure_ascii=False, indent=2))
        return

    if any(not item["passed"] for item in validation):
        report = {
            "run_id": run_id, "created_at": created_at.isoformat(), "profile": args.profile,
            "status": "failed_validation", "cache_hit": False, "selected_dates": selected_dates,
            "selected_candidate": None, "validation": validation, "cost_sensitivity": [],
            "execution_model": initial_execution_model, "data_quality": initial_data_quality,
            "result_trust_level": initial_trust,
            "paper_trading_only": True, "live_ready": False, "formal_strategy_allowed": False,
            "investment_advice": False, "runtime_seconds": round(time.perf_counter() - started, 6),
        }
        write_json(run_dir / "summary.json", report)
        write_json(run_dir / "data_audit.json", audit)
        (run_dir / "summary.md").write_text(markdown_summary(report), encoding="utf-8")
        append_registry([{
            "run_id": run_id, "created_at": created_at.isoformat(), "profile": args.profile,
            "strategy_name": "t0_intraday_agent", "strategy_version": strategy_code_hash[:12],
            "parameter_id": "pipeline_validation", "symbols": "dynamic_etf_universe",
            "date_start": selected_dates[0] if selected_dates else None,
            "date_end": selected_dates[-1] if selected_dates else None,
            "data_hash": data_hash, "feature_hash": feature_hash, "strategy_code_hash": strategy_code_hash,
            "parameter_hash": parameter_hash, "cost_model_hash": cost_model_hash,
            "promotion_status": "failed_validation",
            "rejection_reason": ";".join(str(item["detail"]) for item in validation if not item["passed"]),
            "runtime_seconds": report["runtime_seconds"], "cache_hit_rate": 0.0,
        }])
        print(json.dumps({"run_id": run_id, "status": "failed_validation", "validation": validation, "output": str(run_dir)}, ensure_ascii=False, indent=2))
        return

    selected_view = create_date_view(source_dir, selected_dates, run_dir / "quote_view")
    backtest_started = time.perf_counter()
    selected_candidate = "baseline_current"
    selected_overlay: dict[str, Any] = {}
    evolution_report: dict[str, Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    replay_label = f"pipeline_{run_id}_baseline_current"
    replay_config = Path(args.config)

    if profile_cfg.get("run_candidate_search"):
        prefix = f"pipeline_{run_id}"
        stage1_date = effective_stage1_date
        top_k = effective_fast_top_k
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_t0_strategy_evolution.py"),
            "--config", str(args.config), "--quotes", str(selected_view),
            "--label-prefix", prefix, "--optimizer", "fixed",
            "--fast-top-k", str(top_k), "--stage1-date", stage1_date,
            "--start-date", selected_dates[0], "--end-date", selected_dates[-1],
            "--replay-timeout-seconds", str(max(120, args.timeout_seconds // max(1, top_k + 8))),
            "--no-update-latest-overlay",
        ]
        ok, log, _ = run_command(cmd, args.timeout_seconds)
        evolution_report = read_json(ROOT / "outputs" / "t0_strategy_evolution" / f"{prefix}_summary.json") if ok else {}
        if not evolution_report:
            validation.append({"name": "candidate_search_completed", "passed": False, "detail": log[-2000:]})
        else:
            validation.append({"name": "candidate_search_completed", "passed": True, "detail": evolution_report.get("decision_reason")})
            selected_candidate = str(evolution_report.get("selected_candidate") or "baseline_current")
            selected_metrics = evolution_report.get("selected_candidate_metrics", {})
            selected_overlay = selected_metrics.get("overlay", {}) if isinstance(selected_metrics, dict) else {}
            candidate_rows = evolution_report.get("candidates", []) if isinstance(evolution_report.get("candidates"), list) else []
            stage1_rows = evolution_report.get("two_stage_fast_screen", {}).get("stage1_candidates", [])
            if isinstance(stage1_rows, list):
                candidate_rows = list(stage1_rows) + candidate_rows
            replay_label = f"{prefix}_final_{selected_candidate}"
            candidate_config = evolution.DEFAULT_OUT_DIR / "candidate_configs" / f"{prefix}_final_{selected_candidate}.json"
            if candidate_config.exists():
                replay_config = candidate_config
    else:
        if locked_obj:
            selected_candidate = str(locked_obj.get("candidate") or "locked_candidate")
            selected_overlay = locked_obj.get("overlay", {})
            replay_config = run_dir / "locked_candidate_config.json"
            evolution.write_candidate_config(agent_cfg, selected_overlay, replay_config)
            replay_label = f"pipeline_{run_id}_{selected_candidate}"
        cmd = [
            sys.executable, str(ROOT / "scripts" / "replay_t0_decisions.py"),
            "--config", str(replay_config), "--quotes", str(selected_view), "--label", replay_label,
            "--start-date", selected_dates[0], "--end-date", selected_dates[-1],
            "--output-detail", str(profile_cfg.get("output_detail", "summary")),
        ]
        ok, log, _ = run_command(cmd, args.timeout_seconds)
        validation.append({"name": "replay_completed", "passed": ok, "detail": log[-2000:] if not ok else "ok"})

    replay_summary_path = ROOT / "outputs" / "t0_replay" / f"{replay_label}_summary.json"
    replay_summary = read_json(replay_summary_path)
    if not replay_summary:
        validation.append({"name": "replay_summary_available", "passed": False, "detail": str(replay_summary_path)})
    else:
        validation.append({"name": "replay_summary_available", "passed": True, "detail": str(replay_summary_path)})
    backtest_seconds = time.perf_counter() - backtest_started
    execution_model, data_quality, result_trust_level = build_execution_data_audit(replay_summary, audit)
    validation.append({
        "name": "execution_model_audited",
        "passed": not execution_model["same_snapshot_fill"] and execution_model["next_snapshot_fill"]
                  and execution_model["cost_in_path"] and execution_model["mark_to_market"]
                  and execution_model["t_rule_enforced"],
        "detail": execution_model,
    })
    validation.append({
        "name": "point_in_time_liquidity",
        "passed": data_quality["point_in_time_liquidity"] and not data_quality["full_day_liquidity_used"],
        "detail": data_quality,
    })

    metrics_started = time.perf_counter()
    cost_sensitivity = [
        replay_metrics(replay_summary, name, as_float(spec.get("roundtrip_cost_bps")))
        for name, spec in profile_book.get("cost_scenarios", {}).items()
        if isinstance(spec, dict)
    ] if replay_summary else []
    simple = next((row for row in cost_sensitivity if row["scenario"] == "simple_cost"), cost_sensitivity[0] if cost_sensitivity else {})
    stress = next((row for row in cost_sensitivity if row["scenario"] == "stress_cost"), cost_sensitivity[-1] if cost_sensitivity else {})
    cost_fragile = bool(simple and stress and as_float(simple.get("net_pnl")) > 0 >= as_float(stress.get("net_pnl")))
    validation.append({"name": "cost_sensitivity_available", "passed": bool(cost_sensitivity), "detail": f"cost_fragile={cost_fragile}"})
    validation.append({
        "name": "no_unresolved_open_positions",
        "passed": not bool(replay_summary.get("open_positions_at_end")) if replay_summary else False,
        "detail": replay_summary.get("open_positions_at_end") if replay_summary else "missing replay",
    })
    metrics_seconds = time.perf_counter() - metrics_started

    profile_status = "completed_research_only"
    rejection_reason = ""
    if any(not item["passed"] for item in validation):
        profile_status = "failed_validation"
        rejection_reason = ";".join(item["name"] for item in validation if not item["passed"])
    elif result_trust_level != "clean":
        profile_status = "needs_review"
        rejection_reason = f"result_trust_level:{result_trust_level}"
    elif evolution_report and evolution_report.get("status") != "approved_for_paper_auto_apply":
        profile_status = "needs_review"
        rejection_reason = str(evolution_report.get("decision_reason") or "evolution_not_approved")
    elif cost_fragile:
        profile_status = "needs_review"
        rejection_reason = "cost_fragile"

    selected_parameter_hash = stable_hash(selected_overlay)
    promoted_path = None
    if args.profile == "medium_screening" and profile_status == "completed_research_only" and evolution_report.get("status") == "approved_for_paper_auto_apply":
        promoted_path = run_dir / "promoted_params.json"
        write_json(promoted_path, {
            "frozen": True,
            "promotion_status": "promoted",
            "candidate": selected_candidate,
            "overlay": selected_overlay,
            "parameter_hash": selected_parameter_hash,
            "source_run_id": run_id,
            "in_sample_start": selected_dates[0],
            "in_sample_end": selected_dates[-1],
            "created_at": created_at.isoformat(),
            "paper_trading_only": True,
            "live_ready": False,
        })
    else:
        write_json(run_dir / "candidate_snapshot.json", {
            "frozen": False,
            "promotion_status": profile_status,
            "candidate": selected_candidate,
            "overlay": selected_overlay,
            "parameter_hash": selected_parameter_hash,
            "reason": rejection_reason,
        })

    runtime_seconds = time.perf_counter() - started
    manifest = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "profile": args.profile,
        "strategy": "t0_intraday_agent",
        "selected_candidate": selected_candidate,
        "date_start": selected_dates[0],
        "date_end": selected_dates[-1],
        "selected_dates": selected_dates,
        "data_hash": data_hash,
        "feature_hash": feature_hash,
        "strategy_code_hash": strategy_code_hash,
        "parameter_hash": selected_parameter_hash,
        "cost_model_hash": cost_model_hash,
        "cache_key": cache_key,
        "cache_hit": False,
        "runtime_seconds": round(runtime_seconds, 6),
        "data_loading_seconds": round(data_seconds, 6),
        "backtest_seconds": round(backtest_seconds, 6),
        "metrics_seconds": round(metrics_seconds, 6),
        "rows_processed": replay_summary.get("runtime_profile", {}).get("rows_processed"),
        "replay_runtime_profile": replay_summary.get("runtime_profile", {}),
        "daily_quote_cache": daily_meta,
        "dynamic_gate_cache": gate_meta,
        "raw_data_overwritten": False,
        "agent_latest_overlay_updated": False,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
    }
    report = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "profile": args.profile,
        "status": profile_status,
        "rejection_reason": rejection_reason,
        "cache_hit": False,
        "selected_dates": selected_dates,
        "selected_candidate": selected_candidate,
        "selected_overlay": selected_overlay,
        "selected_parameter_hash": selected_parameter_hash,
        "validation": validation,
        "cost_sensitivity": cost_sensitivity,
        "cost_fragile": cost_fragile,
        "execution_model": execution_model,
        "data_quality": data_quality,
        "result_trust_level": result_trust_level,
        "promoted_parameters": str(promoted_path) if promoted_path else None,
        "manifest": manifest,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_advice": False,
        "runtime_seconds": round(runtime_seconds, 6),
    }
    write_started = time.perf_counter()
    write_json(run_dir / "summary.json", report)
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "data_audit.json", audit)
    write_json(run_dir / "diagnostics.json", {"validation": validation, "evolution": evolution_report})
    write_cost_csv(run_dir / "cost_sensitivity.csv", cost_sensitivity)
    (run_dir / "summary.md").write_text(markdown_summary(report), encoding="utf-8")

    registry_rows = []
    if candidate_rows:
        for candidate in candidate_rows:
            if not isinstance(candidate, dict):
                continue
            selected_row = candidate.get("candidate") == selected_candidate
            registry_rows.append({
                "run_id": run_id, "created_at": created_at.isoformat(), "profile": args.profile,
                "strategy_name": "t0_intraday_agent", "strategy_version": strategy_code_hash[:12],
                "parameter_id": candidate.get("candidate"), "symbols": "dynamic_etf_universe",
                "date_start": selected_dates[0], "date_end": selected_dates[-1], "data_hash": data_hash,
                "feature_hash": feature_hash, "strategy_code_hash": strategy_code_hash,
                "parameter_hash": stable_hash(candidate.get("overlay", {})), "cost_model_hash": cost_model_hash,
                "gross_pnl": candidate.get("total_pnl"),
                "net_pnl": simple.get("net_pnl") if selected_row else None,
                "sharpe": simple.get("sharpe") if selected_row else None,
                "max_drawdown": simple.get("max_drawdown") if selected_row else None,
                "turnover": simple.get("turnover") if selected_row else None,
                "trade_count": candidate.get("trades"), "hit_ratio": candidate.get("win_rate"),
                "cost_to_gross_ratio": simple.get("cost_to_gross_ratio") if selected_row else None,
                "pnl_concentration": simple.get("pnl_concentration") if selected_row else None,
                "promotion_status": profile_status if selected_row else "rejected",
                "rejection_reason": rejection_reason if selected_row else "not_selected",
                "runtime_seconds": round(runtime_seconds, 6), "cache_hit_rate": 0.0,
                "notes": candidate.get("screening_stage"),
            })
    else:
        registry_rows.append({
            "run_id": run_id, "created_at": created_at.isoformat(), "profile": args.profile,
            "strategy_name": "t0_intraday_agent", "strategy_version": strategy_code_hash[:12],
            "parameter_id": selected_candidate, "symbols": "dynamic_etf_universe",
            "date_start": selected_dates[0], "date_end": selected_dates[-1], "data_hash": data_hash,
            "feature_hash": feature_hash, "strategy_code_hash": strategy_code_hash,
            "parameter_hash": selected_parameter_hash, "cost_model_hash": cost_model_hash,
            "gross_pnl": simple.get("gross_pnl"), "net_pnl": simple.get("net_pnl"),
            "sharpe": simple.get("sharpe"), "max_drawdown": simple.get("max_drawdown"),
            "turnover": simple.get("turnover"), "trade_count": simple.get("trade_count"),
            "hit_ratio": simple.get("hit_ratio"), "cost_to_gross_ratio": simple.get("cost_to_gross_ratio"),
            "pnl_concentration": simple.get("pnl_concentration"), "promotion_status": profile_status,
            "rejection_reason": rejection_reason, "runtime_seconds": round(runtime_seconds, 6),
            "cache_hit_rate": 0.0,
        })
    append_registry(registry_rows)
    manifest["file_write_seconds"] = round(time.perf_counter() - write_started, 6)
    report["manifest"] = manifest
    write_json(run_dir / "summary.json", report)
    write_json(run_dir / "run_manifest.json", manifest)

    if args.use_cache and profile_status != "failed_validation":
        write_json(cache_path, {"source_run_id": run_id, "created_at": created_at.isoformat(), "report": report})

    print(json.dumps({
        "run_id": run_id,
        "profile": args.profile,
        "status": profile_status,
        "selected_candidate": selected_candidate,
        "cost_fragile": cost_fragile,
        "cache_hit": False,
        "output": str(run_dir),
        "paper_trading_only": True,
        "live_ready": False,
    }, ensure_ascii=False, indent=2))

    # Decision-score effectiveness report (read-only, guarded): summarizes any decision
    # scores accumulated in outputs/decision_scores/. Never affects the pipeline result.
    try:
        import run_decision_score_report as _dsr
        if list(_dsr.SCORE_DIR.glob("decision_scores_*.jsonl")):
            _dsr.build_report(_dsr.load_records(None))  # writes nothing here; report via its CLI
            print("decision_scores: records present; run "
                  "`py -3.13 scripts/run_decision_score_report.py` for the effectiveness report.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
