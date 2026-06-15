"""DiNapoli-level shadow diagnostics for T+0 ETF replay data.

Offline-only analysis. This script reads replay quote snapshots and replay
decisions, computes mechanical DiNapoli-style retracement/extension levels, and
reports whether historical replay entries/exits were structurally confirmed.

It never calls the paper-trading API, never submits/cancels orders, and never
modifies production agent configuration.
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


DEFAULT_QUOTES = ROOT / "outputs" / "t0_replay" / "replay_20etf_20260615_quotes.jsonl"
DEFAULT_DECISIONS = ROOT / "outputs" / "t0_replay" / "replay_20etf_20260615_decisions.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "t0_replay"


@dataclass
class DinapoliConfig:
    lookback_bars: int = 30
    reaction_window_bars: int = 12
    min_swing_pct: float = 0.003
    bounce_buffer_pct: float = 0.0005
    invalidation_buffer_pct: float = 0.0005
    extension_warning: str = "COP"


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def quote_key(row: dict[str, Any]) -> tuple[str, str]:
    ts = parse_dt(row.get("timestamp"))
    code = str(row.get("stockCode", "")).zfill(6)
    return ((ts.isoformat() if ts else str(row.get("timestamp"))), code)


def build_history(quotes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quotes:
        code = str(row.get("stockCode", "")).zfill(6)
        if not code:
            continue
        px = as_float(row.get("currentPrice"), 0.0)
        ts = parse_dt(row.get("timestamp"))
        if px <= 0 or ts is None:
            continue
        node = dict(row)
        node["_dt"] = ts
        by_code[code].append(node)
    for code in by_code:
        by_code[code].sort(key=lambda r: r["_dt"])
    return by_code


def locate_index(history: list[dict[str, Any]], ts: datetime) -> int | None:
    # Exact timestamp match is expected for replay rows; fall back to the last
    # observation not after the decision timestamp.
    last_idx: int | None = None
    for idx, row in enumerate(history):
        row_ts = row.get("_dt")
        if row_ts == ts:
            return idx
        if row_ts and row_ts <= ts:
            last_idx = idx
        if row_ts and row_ts > ts:
            break
    return last_idx


def compute_dinapoli(
    history: list[dict[str, Any]],
    idx: int,
    cfg: DinapoliConfig,
) -> dict[str, Any]:
    if idx is None or idx < max(6, cfg.lookback_bars // 2):
        return {"available": False, "reason": "insufficient_history"}

    start = max(0, idx - cfg.lookback_bars + 1)
    window = history[start : idx + 1]
    prices = [as_float(r.get("currentPrice"), 0.0) for r in window]
    if len(prices) < max(6, cfg.lookback_bars // 2) or any(p <= 0 for p in prices):
        return {"available": False, "reason": "bad_or_insufficient_prices"}

    low_idx_rel = min(range(len(prices)), key=lambda i: prices[i])
    high_idx_rel = max(range(len(prices)), key=lambda i: prices[i])
    low = prices[low_idx_rel]
    high = prices[high_idx_rel]
    current = prices[-1]
    swing = high - low
    swing_pct = swing / low if low > 0 else 0.0
    trend_up = low_idx_rel < high_idx_rel and swing_pct >= cfg.min_swing_pct

    if not trend_up or swing <= 0:
        return {
            "available": False,
            "reason": "no_valid_bullish_swing",
            "current": current,
            "swing_low": low,
            "swing_high": high,
            "swing_pct": swing_pct,
            "low_idx": start + low_idx_rel,
            "high_idx": start + high_idx_rel,
        }

    after_high_start = start + high_idx_rel
    reaction_start = max(after_high_start, idx - cfg.reaction_window_bars + 1)
    reaction_window = history[reaction_start : idx + 1]
    reaction_prices = [as_float(r.get("currentPrice"), 0.0) for r in reaction_window if as_float(r.get("currentPrice"), 0.0) > 0]
    reaction_low = min(reaction_prices) if reaction_prices else current

    retrace_382 = high - 0.382 * swing
    retrace_618 = high - 0.618 * swing
    cop = reaction_low + 0.618 * swing
    op = reaction_low + swing
    xop = reaction_low + 1.618 * swing

    zone_hi = max(retrace_382, retrace_618)
    zone_lo = min(retrace_382, retrace_618)
    touched_zone = reaction_low <= zone_hi and reaction_low >= zone_lo * (1 - cfg.invalidation_buffer_pct)
    bounced = current >= reaction_low * (1 + cfg.bounce_buffer_pct)
    structure_invalidated = current < retrace_618 * (1 - cfg.invalidation_buffer_pct)
    below_reaction_low = current < reaction_low * (1 - cfg.invalidation_buffer_pct)
    extended_beyond_cop = current >= cop
    extended_beyond_op = current >= op
    entry_confirmed = touched_zone and bounced and not structure_invalidated and not extended_beyond_op
    sell_confirmed = structure_invalidated or below_reaction_low
    would_delay_exit = not sell_confirmed

    return {
        "available": True,
        "reason": "ok",
        "current": round(current, 6),
        "swing_low": round(low, 6),
        "swing_high": round(high, 6),
        "swing_pct": round(swing_pct, 6),
        "reaction_low": round(reaction_low, 6),
        "retracement_382": round(retrace_382, 6),
        "retracement_618": round(retrace_618, 6),
        "cop": round(cop, 6),
        "op": round(op, 6),
        "xop": round(xop, 6),
        "touched_retracement_zone": touched_zone,
        "bounced_from_reaction": bounced,
        "structure_invalidated": structure_invalidated,
        "below_reaction_low": below_reaction_low,
        "extended_beyond_cop": extended_beyond_cop,
        "extended_beyond_op": extended_beyond_op,
        "entry_confirmed": entry_confirmed,
        "sell_confirmed": sell_confirmed,
        "would_delay_exit": would_delay_exit,
        "low_idx": start + low_idx_rel,
        "high_idx": start + high_idx_rel,
        "lookback_bars": cfg.lookback_bars,
        "reaction_window_bars": cfg.reaction_window_bars,
    }


def extract_order_events(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in decisions:
        ts = row.get("timestamp")
        for order in row.get("orders") or []:
            if not isinstance(order, dict):
                continue
            event = dict(order)
            event["decision_timestamp"] = ts
            event["decision_approved"] = bool(row.get("approved"))
            event["decision_reason"] = row.get("reason")
            events.append(event)
    return events


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [r for r in rows if r.get("direction") == "buy" and r.get("decision_approved")]
    sells = [r for r in rows if r.get("direction") == "sell" and r.get("decision_approved")]
    return {
        "approved_entries": len(entries),
        "approved_sells": len(sells),
        "entry_confirmed": sum(1 for r in entries if r.get("entry_confirmed")),
        "entry_rejected_or_unavailable": sum(1 for r in entries if not r.get("entry_confirmed")),
        "sell_confirmed": sum(1 for r in sells if r.get("sell_confirmed")),
        "sell_would_delay": sum(1 for r in sells if r.get("would_delay_exit")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "decision_timestamp", "direction", "stockCode", "name", "quantity", "price",
        "reason", "decision_approved", "available", "dinapoli_reason", "current",
        "swing_low", "swing_high", "swing_pct", "reaction_low", "retracement_382",
        "retracement_618", "cop", "op", "xop", "entry_confirmed",
        "sell_confirmed", "would_delay_exit", "structure_invalidated",
        "extended_beyond_cop", "extended_beyond_op",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DiNapoli Shadow Replay",
        "",
        "Paper trading research only. Not live-ready, not a formal strategy, and not investment advice.",
        "",
        f"- Created at: {report['created_at']}",
        f"- Quotes: `{report['quotes_path']}`",
        f"- Decisions: `{report['decisions_path']}`",
        f"- Codes: {report['code_count']}",
        f"- Quote rows: {report['quote_rows']}",
        "",
        "## Fixed Rules",
        "",
        f"- Lookback bars: {report['dinapoli_config']['lookback_bars']}",
        f"- Reaction window bars: {report['dinapoli_config']['reaction_window_bars']}",
        f"- Minimum bullish swing: {report['dinapoli_config']['min_swing_pct']:.4%}",
        "- Retracement zone: 38.2% to 61.8%",
        "- Objectives: COP 61.8%, OP 100%, XOP 161.8%",
        "",
        "## Summary",
        "",
        f"- Approved entries: {report['summary']['approved_entries']}",
        f"- Entry confirmed by DiNapoli: {report['summary']['entry_confirmed']}",
        f"- Entry rejected/unavailable by DiNapoli: {report['summary']['entry_rejected_or_unavailable']}",
        f"- Approved sells: {report['summary']['approved_sells']}",
        f"- Sell confirmed by structure break: {report['summary']['sell_confirmed']}",
        f"- Sells DiNapoli would delay: {report['summary']['sell_would_delay']}",
        "",
        "## Order Diagnostics",
        "",
        "| time | side | code | price | agent reason | dinapoli | note |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        if not row.get("decision_approved"):
            continue
        if row.get("direction") == "buy":
            verdict = "entry_confirmed" if row.get("entry_confirmed") else "entry_not_confirmed"
        else:
            verdict = "sell_confirmed" if row.get("sell_confirmed") else "would_delay_exit"
        note = row.get("dinapoli_reason")
        lines.append(
            f"| {row.get('decision_timestamp')} | {row.get('direction')} | {row.get('stockCode')} | "
            f"{row.get('price')} | {row.get('reason')} | {verdict} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="DiNapoli shadow diagnostics over T0 replay decisions")
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--label", default="dinapoli_shadow_20260615_20etf")
    parser.add_argument("--lookback-bars", type=int, default=30)
    parser.add_argument("--reaction-window-bars", type=int, default=12)
    parser.add_argument("--min-swing-pct", type=float, default=0.003)
    args = parser.parse_args()

    quotes_path = Path(args.quotes)
    decisions_path = Path(args.decisions)
    quotes = load_jsonl(quotes_path)
    decisions = load_jsonl(decisions_path)
    history = build_history(quotes)
    cfg = DinapoliConfig(
        lookback_bars=args.lookback_bars,
        reaction_window_bars=args.reaction_window_bars,
        min_swing_pct=args.min_swing_pct,
    )

    order_events = extract_order_events(decisions)
    rows: list[dict[str, Any]] = []
    for event in order_events:
        code = str(event.get("stockCode", "")).zfill(6)
        ts = parse_dt(event.get("decision_timestamp"))
        hist = history.get(code, [])
        idx = locate_index(hist, ts) if ts else None
        diag = compute_dinapoli(hist, idx, cfg) if idx is not None else {"available": False, "reason": "no_quote_at_decision_time"}
        row = dict(event)
        row.update({
            "available": diag.get("available"),
            "dinapoli_reason": diag.get("reason"),
        })
        row.update({k: v for k, v in diag.items() if k not in {"reason", "available"}})
        rows.append(row)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_advice": False,
        "task": "dinapoli_shadow_replay",
        "quotes_path": str(quotes_path),
        "decisions_path": str(decisions_path),
        "quote_rows": len(quotes),
        "decision_rows": len(decisions),
        "code_count": len(history),
        "dinapoli_config": cfg.__dict__,
        "summary": summarize_rows(rows),
        "diagnostics": rows,
        "note": "Shadow-only diagnostic. Synthetic replay quotes may not represent true historical bid/ask.",
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
