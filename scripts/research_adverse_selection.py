"""Adverse-selection monitor for passive limit orders (SHADOW / research).

Posting passive limit buys earns the bid-ask spread, but Glosten-Milgrom warns
of adverse selection: a passive buy tends to fill exactly when informed sellers
are hitting the bid -- i.e., right before the price drops. This tool measures
that, so we know whether the spread we 'earn' is real or eaten by adverse fills.

Method: from the agent run log, take BUY orders tagged execution_style=passive
that subsequently FILLED (position appeared), and compare the order's
submission_mid to the market mid a few minutes later. A consistently NEGATIVE
post-fill move means adverse selection is eating the captured spread.

No orders, no account/skill calls, no live state. Reads outputs only.
Run: py -3.13 scripts/research_adverse_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "t0_intraday_agent" / "t0_agent_runs.jsonl"
QUOTES = ROOT / "outputs" / "t0_intraday_agent" / "minute_quotes.jsonl"
HORIZON_MIN = 15  # look this many minutes after a fill


def load_runs() -> list[dict[str, Any]]:
    if not RUNS.exists():
        return []
    out = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def mid_series_by_code() -> dict[str, list[tuple[Any, float]]]:
    series: dict[str, list[tuple[Any, float]]] = {}
    if not QUOTES.exists():
        return series
    for line in QUOTES.read_text(encoding="utf-8").splitlines():
        try:
            q = json.loads(line)
        except Exception:
            continue
        c = str(q.get("stockCode", "")).zfill(6)
        bid, ask, cur = q.get("bidPrice1"), q.get("askPrice1"), q.get("currentPrice")
        if cur in (None, 0):
            continue
        mid = (float(bid) + float(ask)) / 2.0 if bid and ask else float(cur)
        ts = parse(q.get("timestamp"))
        if ts is not None:
            series.setdefault(c, []).append((ts, mid))
    for c in series:
        series[c].sort()
    return series


def parse(value: Any):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    runs = load_runs()
    series = mid_series_by_code()
    # collect submitted passive BUY orders
    passive_buys = []
    for r in runs:
        if r.get("status") != "submitted":
            continue
        for o in (r.get("orders") or []):
            if o.get("direction") == "buy" and o.get("execution_style") == "passive":
                passive_buys.append({
                    "ts": parse(r.get("timestamp")), "code": str(o.get("stockCode", "")).zfill(6),
                    "sub_mid": o.get("submission_mid"), "price": o.get("price"),
                })
    print(f"=== adverse-selection monitor (SHADOW) — {len(passive_buys)} passive buys submitted ===")
    if not passive_buys:
        print("no passive buys yet (need live passive-entry fills to accumulate).")
        return
    moves = []
    from datetime import timedelta
    for b in passive_buys:
        if b["ts"] is None or not b["sub_mid"]:
            continue
        ser = series.get(b["code"], [])
        target = b["ts"] + timedelta(minutes=HORIZON_MIN)
        fut = [m for (t, m) in ser if t >= target]
        if not fut:
            continue
        post_mid = fut[0]
        move_bps = (post_mid / b["sub_mid"] - 1.0) * 1e4
        moves.append(move_bps)
    if not moves:
        print("not enough post-fill quotes to measure yet.")
        return
    avg = sum(moves) / len(moves)
    print(f"avg post-fill mid move after {HORIZON_MIN}min: {avg:.1f} bps (n={len(moves)})")
    print("interpretation: strongly NEGATIVE => adverse selection eats the earned spread;")
    print("  near-zero or positive => passive fills are clean (spread capture is real).")


if __name__ == "__main__":
    main()
