"""Zetra-style support/resistance and retracement shadow diagnostics.

Offline-only research script. It reads T+0 ETF replay quote snapshots and replay
decisions, then computes rolling support/resistance, Fibonacci pivot, and
retracement diagnostics inspired by zeta-zetra/code.

No paper-trading API calls are made. No orders are submitted/cancelled. No
production config is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float, write_json
from replay_t0_dinapoli_shadow import extract_order_events, load_jsonl, locate_index, parse_dt


DEFAULT_QUOTES = ROOT / "outputs" / "t0_replay" / "replay_20etf_20260615_quotes.jsonl"
DEFAULT_DECISIONS = ROOT / "outputs" / "t0_replay" / "replay_20etf_20260615_decisions.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "t0_replay"


@dataclass
class ZetaSrConfig:
    lookback_bars: int = 45
    pivot_left_bars: int = 3
    pivot_right_bars: int = 3
    min_swing_pct: float = 0.003
    zone_threshold_pct: float = 0.0015
    breakout_buffer_pct: float = 0.0008
    break_buffer_pct: float = 0.0010
    gap_bars: int = 5


def build_history(quotes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quotes:
        code = str(row.get("stockCode", "")).zfill(6)
        ts = parse_dt(row.get("timestamp"))
        px = as_float(row.get("currentPrice"), 0.0)
        if not code or ts is None or px <= 0:
            continue
        node = dict(row)
        node["_dt"] = ts
        by_code[code].append(node)
    for code in by_code:
        by_code[code].sort(key=lambda r: r["_dt"])
    return by_code


def confirmed_local_levels(prices: list[float], left: int, right: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return confirmed pivot highs/lows using only data in ``prices``.

    A pivot at index c is confirmed only after ``right`` later observations are
    present in the same history window. This avoids using future data at the
    current replay timestamp.
    """
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    n = len(prices)
    if n < left + right + 1:
        return highs, lows
    for center in range(left, n - right):
        window = prices[center - left : center + right + 1]
        val = prices[center]
        if val == max(window) and window.count(val) == 1:
            highs.append((center, val))
        if val == min(window) and window.count(val) == 1:
            lows.append((center, val))
    return highs, lows


def nearest_level(levels: list[tuple[int, float]], current: float, below: bool) -> tuple[int | None, float | None, float | None]:
    candidates = [(idx, val) for idx, val in levels if val < current] if below else [(idx, val) for idx, val in levels if val > current]
    if not candidates:
        return None, None, None
    idx, val = max(candidates, key=lambda x: x[1]) if below else min(candidates, key=lambda x: x[1])
    dist = current / val - 1.0 if below and val else val / current - 1.0 if current else None
    return idx, val, dist


def compute_zeta_sr(history: list[dict[str, Any]], idx: int | None, cfg: ZetaSrConfig) -> dict[str, Any]:
    if idx is None or idx < max(8, cfg.pivot_left_bars + cfg.pivot_right_bars + 2):
        return {"available": False, "reason": "insufficient_history"}

    start = max(0, idx - cfg.lookback_bars + 1)
    window = history[start : idx + 1]
    prices = [as_float(r.get("currentPrice"), 0.0) for r in window]
    if len(prices) < max(8, cfg.pivot_left_bars + cfg.pivot_right_bars + 2) or any(p <= 0 for p in prices):
        return {"available": False, "reason": "bad_or_insufficient_prices"}

    current = prices[-1]
    pivot_high = max(prices)
    pivot_low = min(prices)
    pivot_high_rel = prices.index(pivot_high)
    pivot_low_rel = prices.index(pivot_low)
    swing = pivot_high - pivot_low
    swing_pct = swing / pivot_low if pivot_low > 0 else 0.0
    pp = (pivot_high + pivot_low + current) / 3.0
    fib_r1 = pp + 0.382 * swing
    fib_r2 = pp + 0.618 * swing
    fib_s1 = pp - 0.382 * swing
    fib_s2 = pp - 0.618 * swing

    local_highs, local_lows = confirmed_local_levels(prices, cfg.pivot_left_bars, cfg.pivot_right_bars)
    support_idx_rel, support, support_dist = nearest_level(local_lows, current, below=True)
    resistance_idx_rel, resistance, resistance_dist = nearest_level(local_highs, current, below=False)

    trend_up = pivot_low_rel < pivot_high_rel and swing_pct >= cfg.min_swing_pct
    trend_down = pivot_high_rel < pivot_low_rel and swing_pct >= cfg.min_swing_pct
    retracement_618 = pivot_high - 0.618 * swing if swing > 0 else None
    retracement_382 = pivot_high - 0.382 * swing if swing > 0 else None
    bounce_from_retracement = False
    retracement_confirmed = False
    bars_since_high = len(prices) - 1 - pivot_high_rel
    if trend_up and retracement_618:
        near_retracement = abs(current / retracement_618 - 1.0) <= cfg.zone_threshold_pct
        bounce_from_retracement = current > retracement_618 and bars_since_high >= cfg.gap_bars
        retracement_confirmed = near_retracement and bounce_from_retracement

    near_support = bool(support and support_dist is not None and support_dist <= cfg.zone_threshold_pct)
    near_resistance = bool(resistance and resistance_dist is not None and resistance_dist <= cfg.zone_threshold_pct)
    fib_breakout = current > fib_r1 * (1.0 + cfg.breakout_buffer_pct)
    resistance_breakout = bool(resistance and current > resistance * (1.0 + cfg.breakout_buffer_pct))
    support_break = bool(support and current < support * (1.0 - cfg.break_buffer_pct))
    fib_support_break = current < fib_s1 * (1.0 - cfg.break_buffer_pct)
    structure_break_confirmed = support_break or fib_support_break

    return {
        "available": True,
        "reason": "ok",
        "current": round(current, 6),
        "lookback_start_idx": start,
        "lookback_bars": len(prices),
        "pivot_high": round(pivot_high, 6),
        "pivot_low": round(pivot_low, 6),
        "pivot_high_idx": start + pivot_high_rel,
        "pivot_low_idx": start + pivot_low_rel,
        "swing_pct": round(swing_pct, 6),
        "trend_up": trend_up,
        "trend_down": trend_down,
        "fib_pp": round(pp, 6),
        "fib_s1": round(fib_s1, 6),
        "fib_s2": round(fib_s2, 6),
        "fib_r1": round(fib_r1, 6),
        "fib_r2": round(fib_r2, 6),
        "retracement_382": round(retracement_382, 6) if retracement_382 else None,
        "retracement_618": round(retracement_618, 6) if retracement_618 else None,
        "support": round(support, 6) if support else None,
        "support_idx": start + support_idx_rel if support_idx_rel is not None else None,
        "support_distance_pct": round(support_dist, 6) if support_dist is not None else None,
        "resistance": round(resistance, 6) if resistance else None,
        "resistance_idx": start + resistance_idx_rel if resistance_idx_rel is not None else None,
        "resistance_distance_pct": round(resistance_dist, 6) if resistance_dist is not None else None,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "fib_breakout": fib_breakout,
        "resistance_breakout": resistance_breakout,
        "retracement_confirmed": retracement_confirmed,
        "bounce_from_retracement": bounce_from_retracement,
        "structure_break_confirmed": structure_break_confirmed,
        "support_break": support_break,
        "fib_support_break": fib_support_break,
        "local_high_count": len(local_highs),
        "local_low_count": len(local_lows),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    approved = [r for r in rows if r.get("decision_approved")]
    entries = [r for r in approved if r.get("direction") == "buy"]
    sells = [r for r in approved if r.get("direction") == "sell"]
    return {
        "approved_entries": len(entries),
        "approved_sells": len(sells),
        "entry_retracement_confirmed": sum(1 for r in entries if r.get("retracement_confirmed")),
        "entry_breakout_confirmed": sum(1 for r in entries if r.get("fib_breakout") or r.get("resistance_breakout")),
        "entry_near_support": sum(1 for r in entries if r.get("near_support")),
        "sell_structure_break_confirmed": sum(1 for r in sells if r.get("structure_break_confirmed")),
        "sell_would_delay_by_structure": sum(1 for r in sells if not r.get("structure_break_confirmed")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "decision_timestamp", "direction", "stockCode", "name", "quantity", "price",
        "reason", "decision_approved", "available", "zeta_reason", "current",
        "trend_up", "trend_down", "pivot_high", "pivot_low", "swing_pct",
        "fib_pp", "fib_s1", "fib_s2", "fib_r1", "fib_r2",
        "retracement_382", "retracement_618", "support", "support_distance_pct",
        "resistance", "resistance_distance_pct", "near_support", "near_resistance",
        "fib_breakout", "resistance_breakout", "retracement_confirmed",
        "structure_break_confirmed", "support_break", "fib_support_break",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Zetra S/R Retracement Shadow Replay",
        "",
        "Paper trading research only. Not live-ready, not a formal strategy, and not investment advice.",
        "",
        f"- Created at: {report['created_at']}",
        f"- Quotes: `{report['quotes_path']}`",
        f"- Decisions: `{report['decisions_path']}`",
        f"- Quote rows: {report['quote_rows']}",
        f"- Code count: {report['code_count']}",
        "",
        "## Fixed Rules",
        "",
        f"- Lookback bars: {report['zeta_config']['lookback_bars']}",
        f"- Confirmed pivot window: left={report['zeta_config']['pivot_left_bars']}, right={report['zeta_config']['pivot_right_bars']}",
        f"- Minimum swing: {report['zeta_config']['min_swing_pct']:.4%}",
        f"- Zone threshold: {report['zeta_config']['zone_threshold_pct']:.4%}",
        "- Fibonacci pivot: PP, S1/S2, R1/R2 using rolling high/low/current.",
        "- Local support/resistance pivots are delayed-confirmed, so no future bars are used at replay time.",
        "",
        "## Summary",
        "",
        f"- Approved entries: {report['summary']['approved_entries']}",
        f"- Entry retracement confirmed: {report['summary']['entry_retracement_confirmed']}",
        f"- Entry breakout confirmed: {report['summary']['entry_breakout_confirmed']}",
        f"- Entry near support: {report['summary']['entry_near_support']}",
        f"- Approved sells: {report['summary']['approved_sells']}",
        f"- Sell structure break confirmed: {report['summary']['sell_structure_break_confirmed']}",
        f"- Sells structure would delay: {report['summary']['sell_would_delay_by_structure']}",
        "",
        "## Order Diagnostics",
        "",
        "| time | side | code | price | agent reason | structural read | note |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        if not row.get("decision_approved"):
            continue
        if row.get("direction") == "buy":
            flags = []
            if row.get("retracement_confirmed"):
                flags.append("retracement")
            if row.get("fib_breakout") or row.get("resistance_breakout"):
                flags.append("breakout")
            if row.get("near_support"):
                flags.append("near_support")
            verdict = "+".join(flags) if flags else "not_confirmed"
        else:
            verdict = "structure_break" if row.get("structure_break_confirmed") else "would_delay"
        note = row.get("zeta_reason")
        lines.append(
            f"| {row.get('decision_timestamp')} | {row.get('direction')} | {row.get('stockCode')} | "
            f"{row.get('price')} | {row.get('reason')} | {verdict} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zetra-style S/R retracement shadow diagnostics over T0 replay decisions")
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--label", default="zeta_sr_shadow_20260615_20etf")
    parser.add_argument("--lookback-bars", type=int, default=45)
    parser.add_argument("--pivot-left-bars", type=int, default=3)
    parser.add_argument("--pivot-right-bars", type=int, default=3)
    parser.add_argument("--min-swing-pct", type=float, default=0.003)
    parser.add_argument("--zone-threshold-pct", type=float, default=0.0015)
    args = parser.parse_args()

    quotes_path = Path(args.quotes)
    decisions_path = Path(args.decisions)
    quotes = load_jsonl(quotes_path)
    decisions = load_jsonl(decisions_path)
    history = build_history(quotes)
    cfg = ZetaSrConfig(
        lookback_bars=args.lookback_bars,
        pivot_left_bars=args.pivot_left_bars,
        pivot_right_bars=args.pivot_right_bars,
        min_swing_pct=args.min_swing_pct,
        zone_threshold_pct=args.zone_threshold_pct,
    )

    rows: list[dict[str, Any]] = []
    for event in extract_order_events(decisions):
        code = str(event.get("stockCode", "")).zfill(6)
        ts = parse_dt(event.get("decision_timestamp"))
        hist = history.get(code, [])
        idx = locate_index(hist, ts) if ts else None
        diag = compute_zeta_sr(hist, idx, cfg) if idx is not None else {"available": False, "reason": "no_quote_at_decision_time"}
        row = dict(event)
        row.update({
            "available": diag.get("available"),
            "zeta_reason": diag.get("reason"),
        })
        row.update({k: v for k, v in diag.items() if k not in {"reason", "available"}})
        rows.append(row)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_advice": False,
        "task": "zeta_sr_retracement_shadow_replay",
        "source_inspiration": "zeta-zetra/code support-resistance and retracement modules",
        "quotes_path": str(quotes_path),
        "decisions_path": str(decisions_path),
        "quote_rows": len(quotes),
        "decision_rows": len(decisions),
        "code_count": len(history),
        "zeta_config": cfg.__dict__,
        "summary": summarize_rows(rows),
        "diagnostics": rows,
        "note": "Shadow-only diagnostic over synthetic L1 replay quotes. It validates structural timing logic only, not execution quality.",
    }

    out_dir = DEFAULT_OUT_DIR
    json_path = out_dir / f"{args.label}.json"
    csv_path = out_dir / f"{args.label}.csv"
    md_path = out_dir / f"{args.label}.md"
    write_json(json_path, report)
    write_csv(csv_path, rows)
    write_md(md_path, report, rows)
    print(json.dumps({
        "summary": report["summary"],
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
