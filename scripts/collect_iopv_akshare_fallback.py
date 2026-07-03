"""FALLBACK ETF IOPV / premium collector via akshare fund_etf_spot_em (eastmoney push2).

Why this exists again after the 2026-06-30 dedup removed its predecessor: the authoritative
collect_etf_iopv_pcf.py feed (eastmoney f131) delivered ZERO usable IOPV rows in its first 5
sessions (06-29..07-03: 85k price rows, iopv/premium_pct all None -- the f131 batch endpoint is
consistently refused from this host, "Remote end closed connection without response"), while
the akshare endpoint demonstrably worked when the predecessor ran on 06-28. This is a FALLBACK
data source with its own directory and explicit lineage tag, not a replacement: rows carry
iopv_source=akshare_fund_etf_spot_em, and the premium harness (research_iopv_premium.py) reads
the primary feed first. If the primary feed comes alive, this one is redundant again and can be
unregistered ("ETF IOPV Akshare Fallback" task).

eastmoney drops connections intermittently, so each poll retries 3x with backoff and failures
are tolerated (skip the poll, keep the loop). Data-only: NO broker calls, NO orders.

Run during session: py -3.13 scripts/collect_iopv_akshare_fallback.py [--once] [--poll-seconds 60]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta

from run_etf_paper_trading_agent import ROOT

CST = timezone(timedelta(hours=8))
OUT_DIR = ROOT / "data" / "research" / "etf_iopv_akshare"
RETRIES = 3
BACKOFF_S = 20.0
# liquid cross-border / QDII ETFs where premium is largest + most predictive
UNIVERSE = {"513050", "513180", "513100", "513500", "159941", "159920", "513330",
            "159792", "513060", "159866", "513130", "159509", "159632", "513090"}


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def fetch() -> list[dict]:
    import akshare as ak
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            df = ak.fund_etf_spot_em()
            break
        except Exception as exc:   # eastmoney intermittently drops connections
            last_exc = exc
            time.sleep(BACKOFF_S * (attempt + 1))
    else:
        raise last_exc if last_exc else RuntimeError("fetch failed")
    now = datetime.now(CST)
    rows: list[dict] = []
    for _, r in df.iterrows():
        code = str(r.get("代码", "")).zfill(6)
        if code not in UNIVERSE:
            continue
        price, iopv = _f(r.get("最新价")), _f(r.get("IOPV实时估值"))
        if not price or not iopv or price <= 0 or iopv <= 0:
            continue
        rows.append({
            "ts": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
            "code": code, "name": str(r.get("名称", "")),
            "price": price, "iopv": round(iopv, 4),
            "premium_pct": round((price / iopv - 1.0) * 100, 4),
            "discount_field": _f(r.get("基金折价率")),
            "change_pct": _f(r.get("涨跌幅")), "amount": _f(r.get("成交额")),
            "iopv_source": "akshare_fund_etf_spot_em",
        })
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--until", default="15:00")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write(rows: list[dict]) -> int:
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
    polls = ok = 0
    while True:
        now = datetime.now(CST)
        if now.hour > hh or (now.hour == hh and now.minute >= mm):
            break
        m = now.hour * 60 + now.minute
        if (9 * 60 + 30) <= m <= (11 * 60 + 30) or (13 * 60) <= m <= (15 * 60):
            polls += 1
            try:
                ok += 1 if write(fetch()) else 0
            except Exception as exc:
                print(f"  poll error (tolerated): {type(exc).__name__}", flush=True)
        time.sleep(max(5.0, float(args.poll_seconds)))
    print(f"done: {ok}/{polls} successful polls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
