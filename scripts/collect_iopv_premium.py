"""ETF IOPV / premium-discount collector -- the one untested, free-data spot-edge line.

Cross-border (QDII) ETFs trade at a premium/discount to intraday fair value (IOPV) -- often
LARGE (纳指 +7.7%, 标普 +4.5%, 中概 -3.4% at last close), driven by QDII quota constraints.
The premium carries predictive info (SSRN 2273930) and is NOT a price-only signal, so it's
genuinely untested by everything we've done.

Source: akshare fund_etf_spot_em() -- one call returns all ETFs with `IOPV实时估值` (real-time
IOPV) and `基金折价率` (premium/discount %). Direct, no field-guessing. eastmoney is intermittent,
so failures are tolerated (the collector just skips that poll). Data capture only; no orders.

The research question is whether the intraday premium CHANGE mean-reverts beyond cost (the level
is persistent/structural; the tradeable part, if any, is the intraday deviation).

Run during session: py -3.13 scripts/collect_iopv_premium.py [--once] [--poll-seconds 60] [--until 15:00]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from run_etf_paper_trading_agent import ROOT

CST = timezone(timedelta(hours=8))
OUT_DIR = ROOT / "outputs" / "iopv_premium"
# liquid cross-border / QDII ETFs where premium is largest + most predictive
UNIVERSE = {"513050", "513180", "513100", "513500", "159941", "159920", "513330",
            "159792", "513060", "159866", "513130", "159509", "159632", "513090"}


def fetch() -> list[dict]:
    import akshare as ak
    df = ak.fund_etf_spot_em()
    now = datetime.now(CST)
    rows: list[dict] = []
    for _, r in df.iterrows():
        code = str(r.get("代码", "")).zfill(6)
        if code not in UNIVERSE:
            continue
        price = r.get("最新价"); iopv = r.get("IOPV实时估值"); disc = r.get("基金折价率")
        try:
            price = float(price); iopv = float(iopv)
        except Exception:
            continue
        if price <= 0 or iopv <= 0:
            continue
        rows.append({
            "ts": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
            "code": code, "name": str(r.get("名称", "")),
            "price": price, "iopv": round(iopv, 4),
            "premium_pct": round((price / iopv - 1.0) * 100, 4),   # + = price above NAV (premium)
            "discount_field": _f(disc),
            "change_pct": _f(r.get("涨跌幅")), "amount": _f(r.get("成交额")),
        })
    return rows


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--until", default="15:00")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write(rows):
        if not rows:
            return 0
        with (OUT_DIR / f"iopv_{rows[0]['date']}.jsonl").open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    if args.once:
        rows = fetch()
        write(rows)
        for r in sorted(rows, key=lambda x: -abs(x["premium_pct"]))[:10]:
            print(f"  {r['code']} {r['name']}: price={r['price']} iopv={r['iopv']} premium={r['premium_pct']:+.2f}%")
        print(f"wrote {len(rows)} rows")
        return 0

    hh, mm = (int(x) for x in args.until.split(":"))
    polls = 0
    while True:
        now = datetime.now(CST)
        if now.hour > hh or (now.hour == hh and now.minute >= mm):
            break
        m = now.hour * 60 + now.minute
        if (9 * 60 + 30) <= m <= (11 * 60 + 30) or (13 * 60) <= m <= (15 * 60):
            try:
                write(fetch()); polls += 1
            except Exception as e:
                print(f"  poll error (tolerated): {type(e).__name__}")
        time.sleep(args.poll_seconds)
    print(f"done: {polls} polls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
