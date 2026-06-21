"""Convert the June TDX full-universe minute bars into agent quote rounds, so the
decision engine can be REPLAYED over every June trading day (not just the 4 live
snapshot days). Offline; writes a single time-ordered minute_quotes JSONL.

Source: data/market/eastmoney/minute/2026-06/etf/*.csv.gz (per-ETF 1-min OHLCV bars,
TDX). We emit one cross-section per 5-minute mark during the session (matching the live
~5-min cadence), liquidity-gated at each timestamp using cumulative amount observed so far
so the replay universe ~ the live dynamic universe. Volume/amount are cumulated within
the day (snapshot semantics); bid/ask are synthesized = close (no historical book), so
spread ~ 0 -- a tight-spread assumption appropriate for the liquid names we keep.

NO orders, NO broker calls. Reads bars, writes one JSONL the replay consumes.
Run: py -3.13 scripts/convert_june_to_quotes.py
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from pathlib import Path

from run_etf_paper_trading_agent import ROOT, as_float
from point_in_time_liquidity import point_in_time_liquidity_gate

TDX_DIR = ROOT / "data" / "market" / "eastmoney" / "minute" / "2026-06" / "etf"
OUT = ROOT / "outputs" / "t0_replay" / "june_full_quotes.jsonl"
MIN_LIQUIDITY_AMOUNT = 50_000_000.0  # elapsed-session-scaled point-in-time gate
SESSION_MARKS = (                      # 5-min cadence China-time minutes within session
    [m for m in range(9 * 60 + 35, 11 * 60 + 31, 5)] +
    [m for m in range(13 * 60 + 5, 15 * 60 + 1, 5)]
)


def _clean(x) -> float:
    v = as_float(x, 0.0)
    return v if v >= 1e-6 else 0.0  # TDX writes ~5.9e-39 for empty minutes


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(TDX_DIR.glob("*.csv.gz"))
    # data[code] = {"name","exch", "days": {date: {minute: (close, cumvol, cumamt)}}, "lastclose": {date: close}}
    data: dict[str, dict] = {}
    for gz in files:
        try:
            with gzip.open(gz, "rb") as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        except Exception:
            continue
        for r in rows:
            dt = str(r.get("datetime", ""))
            if len(dt) < 16:
                continue
            date, hm = dt[:10], dt[11:16]
            try:
                minute = int(hm[:2]) * 60 + int(hm[3:5])
            except ValueError:
                continue
            code = str(r.get("stockCode", "")).zfill(6)
            close = _clean(r.get("close"))
            if close <= 0:
                continue
            node = data.setdefault(code, {"name": r.get("name"),
                                          "exch": "SH" if str(r.get("market")).strip() == "1" else "SZ",
                                          "days": {}, "lastclose": {}})
            day = node["days"].setdefault(date, {})
            prev = day.get("__cum__", (0.0, 0.0))
            cumvol = prev[0] + _clean(r.get("volume"))
            cumamt = prev[1] + _clean(r.get("amount"))
            day["__cum__"] = (cumvol, cumamt)
            day[minute] = (close, cumvol, cumamt)
            node["lastclose"][date] = close

    all_dates = sorted({d for n in data.values() for d in n["days"]})
    prev_date = {all_dates[i]: all_dates[i - 1] for i in range(1, len(all_dates))}

    def at_or_before(day: dict, minute: int):
        chosen = None
        for m in sorted(k for k in day if isinstance(k, int)):
            if m <= minute:
                chosen = day[m]
            else:
                break
        return chosen

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    per_day_counts: dict[str, int] = {}
    per_day_codes: dict[str, set[str]] = {}
    with OUT.open("w", encoding="utf-8") as out:
        for date in all_dates:
            for minute in SESSION_MARKS:
                ts = f"{date}T{minute // 60:02d}:{minute % 60:02d}:00+08:00"
                for code, node in data.items():
                    if date not in node["days"]:
                        continue
                    day = node["days"][date]
                    bar = at_or_before(day, minute)
                    if not bar:
                        continue
                    close, cumvol, cumamt = bar
                    liquid, liquidity_source, _ = point_in_time_liquidity_gate(
                        {minute: cumamt}, minute, MIN_LIQUIDITY_AMOUNT,
                    )
                    if not liquid:
                        continue
                    pd = prev_date.get(date)
                    prev_close = node["lastclose"].get(pd) if pd else None
                    if not prev_close:
                        first = day.get(min(k for k in day if isinstance(k, int)))
                        prev_close = first[0] if first else close
                    # synthesize a tight book (~8bps) consistent with spread_pct, since
                    # TDX bars carry no order book; spread_pct must be present or the
                    # liquidity filter drops every quote (it is set in the live
                    # normalize_quote, which the offline replay path does not run).
                    half = round(close * 0.0004, 4)
                    out.write(json.dumps({
                        "timestamp": ts, "stockCode": code, "exchange": node["exch"],
                        "name": node["name"], "currentPrice": close, "prevClose": prev_close,
                        "bidPrice1": round(close - half, 4), "askPrice1": round(close + half, 4),
                        "spread_pct": 0.0008, "volume": cumvol, "amount": cumamt,
                        "change_pct": round((close / prev_close - 1.0) * 100, 4) if prev_close else 0.0,
                        "quote_ok": True, "isSuspended": False,
                        "liquidity_source": liquidity_source,
                    }, ensure_ascii=False) + "\n")
                    n_rows += 1
                    per_day_counts[date] = per_day_counts.get(date, 0) + 1
                    per_day_codes.setdefault(date, set()).add(code)

    print(f"=== converted June TDX bars -> {OUT} ===")
    print(f"trading days: {len(all_dates)} ({all_dates[0]}..{all_dates[-1]}) | rows: {n_rows}")
    for d in all_dates:
        uni = len(per_day_codes.get(d, set()))
        print(f"  {d}: universe={uni} rows={per_day_counts.get(d, 0)}")


if __name__ == "__main__":
    main()
