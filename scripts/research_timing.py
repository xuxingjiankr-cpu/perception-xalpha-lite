"""Research: entry/exit timing ACCURACY -- how close do we buy-to-low and sell-to-high,
and does a pullback entry beat a breakout entry? (offline / SHADOW)

We can't buy the exact bottom or sell the exact top (nobody can). The measurable goal
is buying NEARER the low and selling NEARER the high, on average. This probe quantifies
that with a "range percentile": for an entry at price P over its forward window
[t..close], pct = (P - low) / (high - low). 0% = we bought the forward low (ideal),
100% = we bought the forward high (worst). For an exit, higher pct = sold nearer the
high (ideal).

It compares, on the recent top-momentum population (what our system chases):
  ENTRY: breakout (enter at the signal price) vs pullback (wait for a small dip, then
         enter) -- on entry percentile, forward return, and fill rate.
  EXIT : hold-to-close vs a trailing/deceleration exit -- on exit percentile and return.

STRICTLY offline: NO orders, NO broker calls, NO agent state. diagnostic_only; needs
MANY days + the promotion gates before any live wiring.

Run: py -3.13 scripts/research_timing.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float
from research_early_entry import _china_minute, REQUIRED_PROMOTION_GATES, RESEARCH_VERSION
from point_in_time_liquidity import point_in_time_liquidity_gate

SNAP_DIR = ROOT / "data" / "market" / "eastmoney" / "full_market" / "snapshots"
OUT_DIR = ROOT / "outputs" / "research_timing"
MIN_DAYS_FOR_STATISTICAL_TESTING = 20
SESSION_CLOSE = 15 * 60


def range_percentile(price: float, lo: float, hi: float) -> float | None:
    """Where `price` sits in [lo, hi]: 0.0 = at the low, 1.0 = at the high."""
    if hi <= lo:
        return None
    return max(0.0, min(1.0, (price - lo) / (hi - lo)))


def pullback_entry(path: list[float], pullback_frac: float, wait: int) -> tuple[float, int] | None:
    """From a signal at path[0], wait up to `wait` steps for a dip to
    signal*(1-pullback_frac); return (entry_price, index) or None if no dip."""
    if not path:
        return None
    signal = path[0]
    target = signal * (1 - pullback_frac)
    for i in range(1, min(len(path), wait + 1)):
        if path[i] <= target:
            return path[i], i
    return None


def trailing_exit(path: list[float], trail_frac: float) -> tuple[float, int]:
    """Exit when price falls trail_frac from the running peak ('loses steam'); else exit
    at the last price. Return (exit_price, index)."""
    if not path:
        return 0.0, 0
    peak = path[0]
    for i in range(1, len(path)):
        peak = max(peak, path[i])
        if peak > path[0] and path[i] <= peak * (1 - trail_frac):
            return path[i], i
    return path[-1], len(path) - 1


def load_day(day_dir: Path) -> dict[str, dict[str, Any]]:
    by_code: dict[str, dict[str, Any]] = {}
    for gz in sorted(day_dir.glob("*_etf.csv.gz")):
        try:
            with gzip.open(gz, "rb") as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        except Exception:
            continue
        for r in rows:
            minute = _china_minute(r)
            price = as_float(r.get("currentPrice"), 0.0)
            if minute is None or price <= 0:
                continue
            code = str(r.get("stockCode", "")).zfill(6)
            node = by_code.setdefault(code, {"amount_by_min": {}, "by_min": {}})
            node["by_min"][minute] = price
            node["amount_by_min"][minute] = as_float(r.get("amount"), 0.0)
    return by_code


def price_at(by_min: dict[int, float], minute: int) -> float | None:
    best_m = best_p = None
    for m, p in by_min.items():
        if m <= minute and (best_m is None or m > best_m):
            best_m, best_p = m, p
    return best_p


def _forward_path(by_min: dict[int, float], start: int, end: int) -> list[float]:
    return [by_min[m] for m in sorted(by_min) if start <= m <= end]


def analyze_day(by_code: dict[str, dict[str, Any]], *, decision_minutes: list[int], window: int,
                min_amount: float, top_frac: float, pullback_frac: float, pullback_wait: int,
                trail_frac: float, cost_pct: float) -> dict[str, list[float]]:
    cost = cost_pct / 100.0
    out: dict[str, list[float]] = {
        "entry_pct_breakout": [], "entry_pct_pullback": [], "pullback_filled": [],
        "fwd_ret_breakout": [], "fwd_ret_pullback": [],
        "exit_pct_hold": [], "exit_pct_trail": [], "ret_hold": [], "ret_trail": [],
    }
    for t in decision_minutes:
        eligible = {
            c: n for c, n in by_code.items()
            if point_in_time_liquidity_gate(n["amount_by_min"], t, min_amount)[0]
            and sum(1 for minute in n["by_min"] if minute <= t) >= 2
        }
        if len(eligible) < 20:
            continue
        recent = []
        for c, n in eligible.items():
            p_now, p_past = price_at(n["by_min"], t), price_at(n["by_min"], t - window)
            if p_now and p_past and p_past > 0:
                recent.append((c, p_now / p_past - 1.0))
        if len(recent) < 20:
            continue
        recent.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(recent) * top_frac))
        for c, _ in recent[:k]:
            path = _forward_path(eligible[c]["by_min"], t, SESSION_CLOSE)
            if len(path) < 3:
                continue
            lo, hi = min(path), max(path)
            entry_b = path[0]
            pb = range_percentile(entry_b, lo, hi)
            if pb is not None:
                out["entry_pct_breakout"].append(pb * 100)
                out["fwd_ret_breakout"].append((path[-1] / entry_b - 1.0 - cost) * 100)
            # pullback entry on the same signal
            pull = pullback_entry(path, pullback_frac, pullback_wait)
            out["pullback_filled"].append(1.0 if pull else 0.0)
            if pull:
                entry_p, idx = pull
                rem = path[idx:]
                lo2, hi2 = min(rem), max(rem)
                pp = range_percentile(entry_p, lo2, hi2)
                if pp is not None:
                    out["entry_pct_pullback"].append(pp * 100)
                    out["fwd_ret_pullback"].append((rem[-1] / entry_p - 1.0 - cost) * 100)
            # exit accuracy on the breakout entry, held forward
            eh = range_percentile(path[-1], lo, hi)
            if eh is not None:
                out["exit_pct_hold"].append(eh * 100)
                out["ret_hold"].append((path[-1] / entry_b - 1.0 - cost) * 100)
            ex_price, _ = trailing_exit(path, trail_frac)
            et = range_percentile(ex_price, lo, hi)
            if et is not None:
                out["exit_pct_trail"].append(et * 100)
                out["ret_trail"].append((ex_price / entry_b - 1.0 - cost) * 100)
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(buckets: dict[str, list[float]], days: int, params: dict[str, Any]) -> dict[str, Any]:
    m = {k: _mean(v) for k, v in buckets.items()}
    enough = days >= MIN_DAYS_FOR_STATISTICAL_TESTING
    gate_status = "not_run_requires_replay_integration" if enough else "not_run_insufficient_days"
    return {
        "research_version": RESEARCH_VERSION, "generated_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True, "status": "diagnostic_only", "edge_validated": False,
        "live_ready": False, "formal_strategy_allowed": False, "order_submit_calls_made": False,
        "params": params, "sample_days": days, "liquidity_source": "point_in_time",
        "n_entries": len(buckets.get("entry_pct_breakout", [])),
        "entry_accuracy": {
            "breakout_entry_pct_in_range": m["entry_pct_breakout"],   # lower = nearer the low
            "pullback_entry_pct_in_range": m["entry_pct_pullback"],
            "pullback_fill_rate": _mean(buckets["pullback_filled"]),
            "breakout_fwd_ret_pct": m["fwd_ret_breakout"],
            "pullback_fwd_ret_pct": m["fwd_ret_pullback"],
        },
        "exit_accuracy": {
            "hold_to_close_exit_pct_in_range": m["exit_pct_hold"],     # higher = nearer the high
            "trailing_exit_pct_in_range": m["exit_pct_trail"],
            "hold_ret_pct": m["ret_hold"],
            "trailing_ret_pct": m["ret_trail"],
        },
        "statistical_readiness": {
            "minimum_days_before_statistical_testing": MIN_DAYS_FOR_STATISTICAL_TESTING,
            "sample_sufficient_for_statistical_testing": enough,
            "required_promotion_gates": {g: {"status": gate_status, "passed": False} for g in REQUIRED_PROMOTION_GATES},
            "verdict": "no_validated_edge",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Entry/exit timing-accuracy research (offline, no orders).")
    ap.add_argument("--decision-minutes", default="600,630,660,690,720,780,810,840")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-amount", type=float, default=50_000_000.0)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--pullback-frac", type=float, default=0.004)   # wait for a 0.4% dip
    ap.add_argument("--pullback-wait", type=int, default=15)
    ap.add_argument("--trail-frac", type=float, default=0.006)      # exit on a 0.6% drop from peak
    ap.add_argument("--cost-pct", type=float, default=0.05)
    args = ap.parse_args()

    decision_minutes = [int(x) for x in str(args.decision_minutes).split(",") if x.strip()]
    buckets: dict[str, list[float]] = {}
    days = 0
    for day_dir in sorted(p for p in SNAP_DIR.glob("*") if p.is_dir()):
        by_code = load_day(day_dir)
        if not by_code:
            continue
        b = analyze_day(by_code, decision_minutes=decision_minutes, window=args.window,
                        min_amount=args.min_amount, top_frac=args.top_frac, pullback_frac=args.pullback_frac,
                        pullback_wait=args.pullback_wait, trail_frac=args.trail_frac, cost_pct=args.cost_pct)
        if any(b.values()):
            days += 1
            for k, v in b.items():
                buckets.setdefault(k, []).extend(v)

    print("=== entry/exit timing-accuracy research (SHADOW, no orders) ===")
    print("percentile-in-range: ENTRY lower=nearer low (ideal); EXIT higher=nearer high (ideal)")
    summary = summarize(buckets, days, vars(args))
    if not summary["n_entries"]:
        print("no usable samples yet.")
        return
    ea, xa = summary["entry_accuracy"], summary["exit_accuracy"]
    print(f"  entries n={summary['n_entries']} over {days} days")
    print(f"  ENTRY  breakout={ea['breakout_entry_pct_in_range']}%-in-range fwd={ea['breakout_fwd_ret_pct']}% | "
          f"pullback={ea['pullback_entry_pct_in_range']}%-in-range fwd={ea['pullback_fwd_ret_pct']}% "
          f"(fill {ea['pullback_fill_rate']})")
    print(f"  EXIT   hold_to_close={xa['hold_to_close_exit_pct_in_range']}%-in-range ret={xa['hold_ret_pct']}% | "
          f"trailing={xa['trailing_exit_pct_in_range']}%-in-range ret={xa['trailing_ret_pct']}%")
    print(f"  verdict: {summary['statistical_readiness']['verdict']} (need >= {MIN_DAYS_FOR_STATISTICAL_TESTING} days + gates)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "timing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"log: {OUT_DIR / 'timing_summary.json'}")


if __name__ == "__main__":
    main()
