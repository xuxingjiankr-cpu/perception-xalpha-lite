"""Free L2-lite order-book DEPTH collector (sina 5-level 量价) -- the real microstructure
data we lacked. Polls sina for the watchlist (the nightly daily-momentum pool + a liquid
core), parses 5-level bid/ask volumes+prices, computes order-book imbalance (OBI), and appends
to a depth log for the OBI research harness. No broker, no orders -- data capture only.

Sina list= returns many codes in one request: var hq_str_sh510300="name,open,prevclose,cur,
high,low,bid,ask,vol,amt, buy1vol,buy1px,...,buy5vol,buy5px, sell1vol,sell1px,...,sell5vol,
sell5px, date,time,...". OBI = (Σbid5 − Σask5)/(Σbid5 + Σask5) in (−1,1): >0 = buy pressure.

Run during market hours: py -3.13 scripts/collect_l2_depth.py [--once] [--poll-seconds 5]
                                                                [--until 15:00]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from run_etf_paper_trading_agent import ROOT

CST = timezone(timedelta(hours=8))
OUT_DIR = ROOT / "outputs" / "l2_depth"

# OBI SCOPE = liquid T+0 ETFs only. OBI is a round-trip (做T) signal, so the universe MUST be
# T+0 (buy+sell same day) -- excludes all domestic equity ETFs (沪深300/科创/创业/行业 = T+1,
# incl. the momentum pool semis). And it must be TIGHT-spread/liquid, since the whole question
# is edge-vs-spread. So: the most liquid cross-border (US/HK QDII) + gold ETFs.
# Selection principle: T+0 (round-trippable) AND tight spread. Tick is ¥0.001, so spread% ≈
# 0.001/price -- sub-¥1 ETFs have 9-14bps half-spreads that bury OBI's few-bp edge. So we
# prefer HIGH-PRICED T+0 names: gold (sub-1bp), then US-index (~2bps). A few active HK/commodity
# names are kept so the sim (which deducts each name's REAL spread) can show where OBI survives.
OBI_T0_UNIVERSE = [
    # 黄金 (最紧价差 <1bp, T+0) -- OBI 最有机会的地方
    "518880",  # 黄金ETF (~¥8.5)
    "159934",  # 黄金ETF易方达 (~¥8)
    # 跨境美股 (高价, ~2-3bp, T+0)
    "513500",  # 标普500ETF (~¥2.5)
    "513100",  # 纳指ETF (~¥2.2)
    "159941",  # 纳指ETF广发 (~¥1.6)
    # 跨境港股 (较高价/活跃; 价差由 sim 按真实扣)
    "159920",  # 恒生ETF (~¥1.4, ~3.6bp)
    "513050",  # 中概互联网ETF (~¥1.0, ~5bp -- 活跃,sim判)
    "513180",  # 恒生科技ETF (~¥0.57, ~9bp -- 活跃但价差宽,sim判)
    # 商品 (T+0, 波动)
    "159985",  # 豆粕ETF
    "501018",  # 南方原油
]


def sina_symbol(code: str) -> str:
    return ("sh" if code[:1] in ("5", "6") else "sz") + code


def watchlist() -> list[str]:
    """OBI trading scope: liquid T+0 ETFs only (round-trippable, tight-spread)."""
    seen, out = set(), []
    for c in OBI_T0_UNIVERSE:
        c = str(c).zfill(6)
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def fetch_depth(codes: list[str], timeout: float = 10.0) -> list[dict]:
    url = "https://hq.sinajs.cn/list=" + ",".join(sina_symbol(c) for c in codes)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Referer": "https://finance.sina.com.cn"})
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode("gbk", "ignore")
    now = datetime.now(CST)
    rows: list[dict] = []
    for line in raw.split("\n"):
        if 'hq_str_' not in line or '="' not in line:
            continue
        sym = line.split("hq_str_")[1].split("=")[0]
        code = sym[2:]
        body = line.split('"')[1] if '"' in line else ""
        f = body.split(",")
        if len(f) < 32:
            continue
        try:
            cur = float(f[3]); prev = float(f[2])
            if cur <= 0:
                continue
            bid_v = [float(f[10 + 2 * i]) for i in range(5)]
            bid_p = [float(f[11 + 2 * i]) for i in range(5)]
            ask_v = [float(f[20 + 2 * i]) for i in range(5)]
            ask_p = [float(f[21 + 2 * i]) for i in range(5)]
        except Exception:
            continue
        sb, sa = sum(bid_v), sum(ask_v)
        obi = (sb - sa) / (sb + sa) if (sb + sa) > 0 else None
        b1, a1 = bid_p[0], ask_p[0]
        micro = (a1 * sb + b1 * sa) / (sb + sa) if (sb + sa) > 0 and a1 > 0 and b1 > 0 else cur
        rows.append({
            "ts": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
            "code": code, "current": cur, "prevClose": prev,
            "bid1": b1, "ask1": a1,
            "half_spread_bps": round((a1 - b1) / 2 / ((a1 + b1) / 2) * 1e4, 3) if a1 > b1 > 0 else None,
            "bid_vol5": round(sb, 1), "ask_vol5": round(sa, 1),
            "obi": round(obi, 5) if obi is not None else None,
            "micro_price": round(micro, 5),
            "micro_dev_bps": round((micro / cur - 1) * 1e4, 3) if cur > 0 else None,
        })
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--until", default="15:00")
    args = ap.parse_args()
    codes = watchlist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"L2 depth collector: {len(codes)} codes, poll {args.poll_seconds}s until {args.until}")

    def append(rows):
        if not rows:
            return 0
        day = rows[0]["date"]
        with (OUT_DIR / f"depth_{day}.jsonl").open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    if args.once:
        rows = fetch_depth(codes)
        n = append(rows)
        for r in rows[:8]:
            print(f"  {r['code']} cur={r['current']} OBI={r['obi']} half_spread={r['half_spread_bps']}bps "
                  f"micro_dev={r['micro_dev_bps']}bps")
        print(f"wrote {n} depth rows")
        return 0

    hh, mm = (int(x) for x in args.until.split(":"))
    polls = 0
    while True:
        now = datetime.now(CST)
        if now.hour > hh or (now.hour == hh and now.minute >= mm):
            break
        # only poll during continuous session
        m = now.hour * 60 + now.minute
        if (9 * 60 + 30) <= m <= (11 * 60 + 30) or (13 * 60) <= m <= (15 * 60):
            try:
                append(fetch_depth(codes)); polls += 1
            except Exception as e:
                print(f"  poll error: {e}")
        time.sleep(args.poll_seconds)
    print(f"done: {polls} polls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
