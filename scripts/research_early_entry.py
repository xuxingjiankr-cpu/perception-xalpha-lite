"""Research: does entering EARLY (on relative-strength + volume surge) beat our
current LATE breakout entry -- both on forward return and on entry PRICE?

Motivation (from a live post-mortem): on 2026-06-18 the agent bought 588030 at
14:36 @ 2.034, but that name's whole move was 1.991 -> ~2.03 by 11:26 -- we bought
~2% above the morning base, hours after the move was over. The hypothesis: the
day's leaders are identifiable EARLY by relative strength + a volume surge
(accumulation shows in volume before price), so entering near 10:00 gets a better
price than waiting for the mature afternoon breakout.

This is STRICTLY offline research: it reads the full-market ETF snapshots (which
carry volume/amount for every ETF every minute) and computes, per trade day:
  Q1 (predictive): do the top early-signal names out-return the universe from the
      early decision time to the close?
  Q2 (entry price): for those names, is the early (decision-time) price meaningfully
      cheaper than the price at which a late "new-high breakout" would have fired?

NO broker calls, NO orders, NO agent state. Reads snapshot files only.
Run: py -3.13 scripts/research_early_entry.py [--early-end 10:00] [--breakout-after 13:00]
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

SNAP_DIR = ROOT / "data" / "market" / "eastmoney" / "full_market" / "snapshots"
OUT_DIR = ROOT / "outputs" / "research_early_entry"


def _china_minute(row: dict[str, Any]) -> int | None:
    """Minute-of-day in China time from source_quote_time (+08), e.g. 09:35 -> 575."""
    txt = str(row.get("source_quote_time") or row.get("collected_at") or "")
    try:
        dt = datetime.fromisoformat(txt)
    except Exception:
        return None
    # source_quote_time is +08 (exchange); collected_at is +09 (host). Normalize to +08.
    if dt.utcoffset() is not None:
        from datetime import timezone, timedelta
        dt = dt.astimezone(timezone(timedelta(hours=8)))
    return dt.hour * 60 + dt.minute


def _zscores(vals: list[float]) -> list[float]:
    n = len(vals)
    if n == 0:
        return []
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5
    return [0.0] * n if sd <= 0 else [(v - m) / sd for v in vals]


def load_day(day_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """code -> time-sorted list of {minute, price, volume, amount, open, high, low, name}."""
    by_code: dict[str, list[dict[str, Any]]] = {}
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
            by_code.setdefault(code, []).append({
                "minute": minute, "price": price,
                "volume": as_float(r.get("volume"), 0.0),
                "amount": as_float(r.get("amount"), 0.0),
                "open": as_float(r.get("open"), 0.0),
                "high": as_float(r.get("high"), 0.0),
                "low": as_float(r.get("low"), 0.0),
                "name": r.get("name"),
            })
    for code in by_code:
        by_code[code].sort(key=lambda x: x["minute"])
    return by_code


def _at_or_before(series: list[dict[str, Any]], minute: int) -> dict[str, Any] | None:
    chosen = None
    for s in series:
        if s["minute"] <= minute:
            chosen = s
        else:
            break
    return chosen


def analyze_day(day: str, by_code: dict[str, list[dict[str, Any]]], *, early_end: int,
                breakout_after: int, min_amount: float, min_price: float,
                top_frac: float, breakout_pct: float) -> dict[str, Any] | None:
    rows = []
    for code, series in by_code.items():
        if len(series) < 5:
            continue
        op = series[0]["open"] or series[0]["price"]
        early = _at_or_before(series, early_end)
        close = series[-1]
        if not early or op <= 0:
            continue
        full_amount = max(s["amount"] for s in series)
        if full_amount < min_amount or close["price"] < min_price:
            continue
        early_ret = early["price"] / op - 1.0
        elapsed = max(1, early["minute"] - (9 * 60 + 30))
        early_vol_rate = early["volume"] / elapsed  # cumulative volume per minute so far
        fwd_ret = close["price"] / early["price"] - 1.0  # enter at early, hold to close
        # late breakout entry proxy: first new-session-high after `breakout_after`
        # that is also >= breakout_pct above the open (a "momentum/ORB" style fire).
        late_price = None
        run_high = max(s["high"] for s in series if s["minute"] <= breakout_after) if any(s["minute"] <= breakout_after for s in series) else op
        for s in series:
            if s["minute"] <= breakout_after:
                continue
            if s["price"] > run_high and s["price"] >= op * (1 + breakout_pct):
                late_price = s["price"]
                break
            run_high = max(run_high, s["high"])
        rows.append({
            "code": code, "name": early.get("name"), "early_ret": early_ret,
            "early_vol_rate": early_vol_rate, "fwd_ret": fwd_ret,
            "early_price": early["price"], "late_price": late_price,
        })
    if len(rows) < 20:
        return None
    zr = _zscores([x["early_ret"] for x in rows])
    zv = _zscores([x["early_vol_rate"] for x in rows])
    for i, x in enumerate(rows):
        x["early_score"] = zr[i] + zv[i]
    rows.sort(key=lambda x: x["early_score"], reverse=True)
    n_top = max(1, int(len(rows) * top_frac))
    top = rows[:n_top]
    uni_fwd = sum(x["fwd_ret"] for x in rows) / len(rows)
    top_fwd = sum(x["fwd_ret"] for x in top) / len(top)
    # entry-price improvement: among top names that DID later break out, how much
    # cheaper was the early price vs the late breakout price?
    pairs = [(x["early_price"], x["late_price"]) for x in top if x["late_price"]]
    price_improve = (sum((lp / ep - 1.0) for ep, lp in pairs) / len(pairs)) if pairs else None
    return {
        "date": day, "eligible": len(rows), "top_n": n_top,
        "universe_fwd_ret_pct": round(uni_fwd * 100, 3),
        "top_early_fwd_ret_pct": round(top_fwd * 100, 3),
        "early_edge_pct": round((top_fwd - uni_fwd) * 100, 3),
        "top_that_later_broke_out": len(pairs),
        "late_vs_early_price_premium_pct": round(price_improve * 100, 3) if price_improve is not None else None,
        "top_names": [f"{x['code']}({x['name']})" for x in top[:8]],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Research: early relative-strength + volume entry vs late breakout.")
    ap.add_argument("--early-end", default="10:00", help="China-time HH:MM of the early decision point")
    ap.add_argument("--breakout-after", default="13:00", help="late breakout only counts after this HH:MM")
    ap.add_argument("--min-amount", type=float, default=50_000_000.0)
    ap.add_argument("--min-price", type=float, default=0.3)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--breakout-pct", type=float, default=0.015)
    args = ap.parse_args()

    def hhmm(v: str) -> int:
        h, m = (int(x) for x in v.split(":"))
        return h * 60 + m

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    early_end, breakout_after = hhmm(args.early_end), hhmm(args.breakout_after)
    results = []
    for day_dir in sorted(p for p in SNAP_DIR.glob("*") if p.is_dir()):
        by_code = load_day(day_dir)
        if not by_code:
            continue
        r = analyze_day(day_dir.name, by_code, early_end=early_end, breakout_after=breakout_after,
                        min_amount=args.min_amount, min_price=args.min_price,
                        top_frac=args.top_frac, breakout_pct=args.breakout_pct)
        if r:
            results.append(r)

    print("=== early relative-strength + volume entry vs late breakout (SHADOW research) ===")
    print(f"early decision @ {args.early_end} CST | late breakout only after {args.breakout_after} | top {int(args.top_frac*100)}%")
    if not results:
        print("no usable days yet (need full-market snapshots with intraday coverage).")
        return
    edges, premiums = [], []
    for r in results:
        edges.append(r["early_edge_pct"])
        if r["late_vs_early_price_premium_pct"] is not None:
            premiums.append(r["late_vs_early_price_premium_pct"])
        print(f"  {r['date']}: eligible={r['eligible']} top={r['top_n']} | "
              f"Q1 top_fwd={r['top_early_fwd_ret_pct']}% vs uni={r['universe_fwd_ret_pct']}% "
              f"(edge {r['early_edge_pct']:+}%) | Q2 late-vs-early price premium="
              f"{r['late_vs_early_price_premium_pct']}% (n={r['top_that_later_broke_out']})")
        print(f"      top early names: {', '.join(r['top_names'])}")
    avg_edge = sum(edges) / len(edges) if edges else 0.0
    avg_prem = sum(premiums) / len(premiums) if premiums else None
    print("--- summary ---")
    print(f"Q1 predictive: avg early-signal forward EDGE over universe = {avg_edge:+.3f}% across {len(edges)} days")
    print(f"Q2 entry price: avg late-breakout price was {avg_prem if avg_prem is None else round(avg_prem,3)}% "
          f"ABOVE the early-entry price (positive => early entry is cheaper)")
    print("verdict guidance: need Q1 edge>0 (early signal predicts) AND Q2 premium>0 (early price better),")
    print("  consistently across MANY days, before wiring an early-entry into live gating (DSR-gated).")
    print(f"NOTE: only {len(results)} day(s) of data -- directional read only, NOT statistically conclusive.")
    out_path = OUT_DIR / "early_entry_research.json"
    out_path.write_text(json.dumps({"params": vars(args), "results": results,
                                    "avg_edge_pct": avg_edge, "avg_late_premium_pct": avg_prem},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"log: {out_path}")


if __name__ == "__main__":
    main()
