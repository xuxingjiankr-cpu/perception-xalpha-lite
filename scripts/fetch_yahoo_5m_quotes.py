"""Fetch ~60 trading days of 5-MINUTE A-share ETF bars from Yahoo Finance and emit
agent quote rounds, so the strategy can be backtested/evolved over a real multi-month
span (not just the 14 TDX June days). Offline research data prep: no orders/broker.

Why Yahoo 5m: months of 1-min history is unavailable on every free source (Yahoo's
1m is capped at ~7d), but Yahoo serves 5m bars ~60d back -- and our backtest already
runs at 5-min cadence, so 5m is sufficient. Volume is populated for the liquid names;
amount is reconstructed as volume*close (Yahoo A-share volume is in shares).

Universe = the liquid codes the June backtest used (read from june_full_quotes.jsonl).
Symbols map to Yahoo as <code>.SS (Shanghai 5/6...) / <code>.SZ (Shenzhen 1...).

Output: one time-ordered minute_quotes JSONL the replay/evolution consume, liquidity-
gated to the day's tradable names (cumulative full-day amount >= floor), with a
synthesized ~8bps book (Yahoo has no order book) so the liquidity filter passes.

Run: py -3.13 scripts/fetch_yahoo_5m_quotes.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from run_etf_paper_trading_agent import ROOT, as_float

CST = timezone(timedelta(hours=8))
JUNE_QUOTES = ROOT / "outputs" / "t0_replay" / "june_full_quotes.jsonl"
OUT = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
MIN_FULL_DAY_AMOUNT = 50_000_000.0
SESSION_MARKS = (
    [m for m in range(9 * 60 + 35, 11 * 60 + 31, 5)] +
    [m for m in range(13 * 60 + 5, 15 * 60 + 1, 5)]
)


def yahoo_symbol(code: str) -> str:
    return f"{code}.SS" if code[:1] in ("5", "6") else f"{code}.SZ"


def load_universe() -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    if JUNE_QUOTES.exists():
        for line in JUNE_QUOTES.open(encoding="utf-8"):
            try:
                q = json.loads(line)
            except Exception:
                continue
            c = str(q.get("stockCode", "")).zfill(6)
            if c and c not in seen:
                seen[c] = q.get("exchange", "SH")
    return sorted(seen.items())


def fetch_5m(symbol: str, timeout: float = 20.0) -> list[tuple[datetime, float, float]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=60d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    d = json.loads(raw)
    res = (d.get("chart", {}).get("result") or [None])[0]
    if not res or "timestamp" not in res:
        return []
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    out: list[tuple[datetime, float, float]] = []
    for i, t in enumerate(ts):
        c = q["close"][i]
        if c is None or c <= 0:
            continue
        v = q["volume"][i] or 0.0
        out.append((datetime.fromtimestamp(t, CST), float(c), float(v)))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    universe = load_universe()
    print(f"fetching {len(universe)} liquid ETFs from Yahoo (5m, 60d)...")
    # data[code] = {"exch", "days": {date: {minute: (close, cumvol, cumamt)}}, "fullamt": {date: amt}, "lastclose": {date: close}}
    data: dict[str, dict] = {}
    ok = fail = 0
    for n, (code, exch) in enumerate(universe):
        try:
            bars = fetch_5m(yahoo_symbol(code))
        except Exception:
            bars = []
        if not bars:
            fail += 1
            time.sleep(0.25)
            continue
        ok += 1
        node = data.setdefault(code, {"exch": exch, "days": {}, "fullamt": {}, "lastclose": {}})
        cur_day = None
        cumvol = cumamt = 0.0
        for dt, close, vol in bars:
            d = dt.strftime("%Y-%m-%d")
            minute = dt.hour * 60 + dt.minute
            if not (9 * 60 + 30 <= minute <= 15 * 60):
                continue
            if d != cur_day:
                cur_day, cumvol, cumamt = d, 0.0, 0.0
            cumvol += vol
            cumamt += vol * close
            node["days"].setdefault(d, {})[minute] = (close, cumvol, cumamt)
            node["fullamt"][d] = cumamt
            node["lastclose"][d] = close
        if n % 50 == 0:
            print(f"  {n}/{len(universe)} ok={ok} fail={fail}")
        time.sleep(0.2)

    all_dates = sorted({d for nd in data.values() for d in nd["days"]})
    if not all_dates:
        print("no data fetched.")
        return
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
    per_day: dict[str, int] = {}
    with OUT.open("w", encoding="utf-8") as out:
        for date in all_dates:
            uni = [c for c, nd in data.items() if nd["fullamt"].get(date, 0.0) >= MIN_FULL_DAY_AMOUNT]
            for minute in SESSION_MARKS:
                ts = f"{date}T{minute // 60:02d}:{minute % 60:02d}:00+08:00"
                for code in uni:
                    nd = data[code]
                    bar = at_or_before(nd["days"][date], minute)
                    if not bar:
                        continue
                    close, cumvol, cumamt = bar
                    pd = prev_date.get(date)
                    prev_close = nd["lastclose"].get(pd) if pd else None
                    if not prev_close:
                        first = nd["days"][date].get(min(k for k in nd["days"][date] if isinstance(k, int)))
                        prev_close = first[0] if first else close
                    half = round(close * 0.0004, 4)
                    out.write(json.dumps({
                        "timestamp": ts, "stockCode": code, "exchange": nd["exch"],
                        "name": code, "currentPrice": close, "prevClose": prev_close,
                        "bidPrice1": round(close - half, 4), "askPrice1": round(close + half, 4),
                        "spread_pct": 0.0008, "volume": cumvol, "amount": cumamt,
                        "change_pct": round((close / prev_close - 1.0) * 100, 4) if prev_close else 0.0,
                        "quote_ok": True, "isSuspended": False,
                    }, ensure_ascii=False) + "\n")
                    n_rows += 1
                    per_day[date] = per_day.get(date, 0) + 1

    print(f"=== Yahoo 5m -> {OUT} ===")
    print(f"fetched ok={ok} fail={fail} | trading days={len(all_dates)} ({all_dates[0]}..{all_dates[-1]}) | rows={n_rows}")
    for d in all_dates:
        uni = len([c for c, nd in data.items() if nd["fullamt"].get(d, 0.0) >= MIN_FULL_DAY_AMOUNT])
        print(f"  {d}: universe={uni} rows={per_day.get(d, 0)}")


if __name__ == "__main__":
    main()
