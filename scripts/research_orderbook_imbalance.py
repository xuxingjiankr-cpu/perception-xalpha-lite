"""Order-book imbalance (OBI) signal audit -- the genuine intraday microstructure test, on
REAL free 5-level depth captured by collect_l2_depth.py.

Question: does OBI = (Σbid5 − Σask5)/(Σbid5 + Σask5) predict the next-few-seconds forward
return, and does the edge clear the REAL half-spread you'd cross? OBI is one of the few signals
with genuine intraday predictive power -- but it lives at sub-minute horizons where you must
provide liquidity (passive) to avoid paying the spread. We report the forward IC (time-series,
pooled with day clustering) AND the edge in bps vs the contemporaneous half-spread.

Refuses a conclusion below MIN_OBS / MIN_DAYS -- accumulates forward from the collector.

Run: py -3.13 scripts/research_orderbook_imbalance.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT

DEPTH_DIR = ROOT / "outputs" / "l2_depth"
OUT = ROOT / "outputs" / "l2_depth"
HORIZONS = [1, 3, 6, 12]      # polls ahead (×poll_seconds)
MIN_OBS = 2000
MIN_DAYS = 3


def load_series():
    """code -> list of (date, ts, micro_price, obi, half_spread_bps, micro_dev_bps), time-ordered."""
    series = defaultdict(list)
    for p in sorted(DEPTH_DIR.glob("depth_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            mp = r.get("micro_price") or r.get("current")
            if r.get("obi") is None or not mp:
                continue
            series[r["code"]].append((r.get("date"), r.get("ts"), float(mp),
                                      float(r["obi"]),
                                      r.get("half_spread_bps"), r.get("micro_dev_bps")))
    return series


COMMISSION_RT = 0.0006   # 万六 round-trip commission (万三/side x2)
ACCOUNT_CAPITAL = 200_000   # notional ISOLATED OBI paper account (~20% of a 1M competition), monitor-only
ACCOUNT_HORIZON = 3         # polls held per OBI trade for the tracked account


def paper_account(series, horizon=ACCOUNT_HORIZON, capital=ACCOUNT_CAPITAL):
    """Self-contained, MONITORABLE OBI paper account (isolated from the competition).
    Equal-weights the day's taker trades (realistic fills: enter ask, exit bid, net spread+commission),
    compounds into a daily equity curve. Gives clean standalone attribution -- what OBI actually
    makes/loses -- which is impossible if OBI is buried inside the bold-play account."""
    by_day = defaultdict(list)
    blotter = []
    for code, s in series.items():
        for i in range(len(s)):
            if s[i][3] is None or s[i][3] < 0.3:
                continue
            j = i + horizon
            if j >= len(s) or s[j][0] != s[i][0]:
                continue
            gross = s[j][2] / s[i][2] - 1.0
            net = gross - (s[i][4] or 0) / 1e4 - (s[j][4] or 0) / 1e4 - COMMISSION_RT
            by_day[s[i][0]].append(net)
            blotter.append({"date": s[i][0], "code": code, "entry_ts": s[i][1],
                            "obi": round(s[i][3], 3), "net_bps": round(net * 1e4, 2)})
    days = sorted(by_day)
    equity = capital
    curve = []
    for d in days:
        day_ret = float(np.mean(by_day[d]))      # equal-weight that day's signals
        equity *= (1 + day_ret)
        curve.append({"date": d, "trades": len(by_day[d]), "day_return_pct": round(day_ret * 100, 4),
                      "day_pnl_yuan": round(capital * day_ret, 1), "equity": round(equity, 1)})
    total_ret = (equity / capital - 1) if days else None
    eq = np.array([c["equity"] for c in curve]) if curve else np.array([])
    maxdd = float((eq / np.maximum.accumulate(eq) - 1).min()) if eq.size else None
    return {"capital": capital, "horizon_polls": horizon, "days": len(days),
            "total_trades": len(blotter), "total_return_pct": round(total_ret * 100, 3) if total_ret is not None else None,
            "final_equity": round(equity, 1) if days else capital,
            "max_drawdown_pct": round(maxdd * 100, 3) if maxdd is not None else None,
            "daily_curve": curve, "recent_blotter": blotter[-15:]}


def simulate_obi_trade(series, horizon, thr=0.3):
    """Long-only OBI trade NET OF REAL SPREAD (the actual trade, not just IC).
    TAKER (realistic floor): enter at ask, exit at bid -> cross the half-spread on BOTH legs +
      commission. This is what a retail taker actually nets.
    MAKER (optimistic ceiling): enter at bid (EARN the entry half-spread), exit at mid, pay only
      commission. Ignores adverse selection / non-fills, so it OVERSTATES -- a maker mostly fills
      when OBI is about to be WRONG. Truth sits between, near the taker for a 5s-poll retail.
    Long-only (no A-share ETF shorting): act only on buy-pressure (OBI>=thr)."""
    taker, maker, days = [], [], []
    for code, s in series.items():
        for i in range(len(s)):
            if s[i][3] is None or s[i][3] < thr:
                continue
            j = i + horizon
            if j >= len(s) or s[j][0] != s[i][0]:
                continue
            gross = s[j][2] / s[i][2] - 1.0
            hs_i = (s[i][4] or 0.0) / 1e4
            hs_j = (s[j][4] or 0.0) / 1e4
            taker.append(gross - hs_i - hs_j - COMMISSION_RT)
            maker.append(gross + hs_i - COMMISSION_RT)
            days.append(s[i][0])

    def agg(x):
        if len(x) < 30:
            return {"n": len(x)}
        a = np.array(x)
        return {"n": len(a), "mean_bps": round(float(a.mean()) * 1e4, 3),
                "pooled_t": round(float(a.mean() / a.std() * np.sqrt(a.size)), 2) if a.std() else None,
                "hit_rate": round(float(np.mean(a > 0)), 3),
                "total_pct": round(float(a.sum()) * 100, 3)}
    return {"trades": len(taker), "thr": thr,
            "taker": agg(taker), "maker_optimistic": agg(maker)}


def analyze():
    series = load_series()
    # build (signal, fwd_return, day) tuples per horizon, pooled across codes
    rows = defaultdict(list)        # h -> list of (obi, fwd, day, half_spread)
    rows_md = defaultdict(list)     # micro_dev signal
    n_obs = 0
    days = set()
    for code, s in series.items():
        # only consecutive same-day points (don't cross day boundary or gaps)
        for i in range(len(s)):
            days.add(s[i][0])
            for h in HORIZONS:
                j = i + h
                if j >= len(s) or s[j][0] != s[i][0]:
                    continue
                fwd = s[j][2] / s[i][2] - 1.0
                rows[h].append((s[i][3], fwd, s[i][0], s[i][4]))
                if s[i][5] is not None:
                    rows_md[h].append((s[i][5], fwd, s[i][0]))
            n_obs += 1

    def ic_stats(data):
        if len(data) < 50:
            return {"n": len(data)}
        x = np.array([d[0] for d in data]); y = np.array([d[1] for d in data])
        # pooled Spearman
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        ic = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else None
        # day-clustered t: mean daily IC / std * sqrt(days)
        byday = defaultdict(list)
        for s_, f_, d_, *_ in data:
            byday[d_].append((s_, f_))
        daily_ic = []
        for d_, vv in byday.items():
            if len(vv) < 30:
                continue
            xa = np.array([a for a, _ in vv]); ya = np.array([b for _, b in vv])
            rxa = np.argsort(np.argsort(xa)).astype(float); rya = np.argsort(np.argsort(ya)).astype(float)
            if rxa.std() and rya.std():
                daily_ic.append(float(np.corrcoef(rxa, rya)[0, 1]))
        dic = np.array(daily_ic)
        # edge proxy: top-vs-bottom OBI tertile forward return spread (bps), gross
        hi = np.quantile(x, 2 / 3); lo = np.quantile(x, 1 / 3)
        spread_bps = float((y[x >= hi].mean() - y[x <= lo].mean()) * 1e4) if (x >= hi).any() and (x <= lo).any() else None
        return {"n": len(data), "pooled_ic": round(ic, 4) if ic is not None else None,
                "n_days": int(dic.size),
                "mean_daily_ic": round(float(dic.mean()), 4) if dic.size else None,
                "day_clustered_t": round(float(dic.mean() / dic.std() * np.sqrt(dic.size)), 2) if dic.size > 1 and dic.std() else None,
                "top_minus_bottom_bps": round(spread_bps, 3) if spread_bps is not None else None}

    half_spreads = [d[3] for h in rows for d in rows[h] if d[3] is not None]
    med_half = float(np.median(half_spreads)) if half_spreads else None
    result = {
        "depth_rows": sum(len(s) for s in series.values()),
        "codes": len(series), "days": sorted(days),
        "median_half_spread_bps": round(med_half, 3) if med_half else None,
        "sufficient": sum(len(s) for s in series.values()) >= MIN_OBS and len(days) >= MIN_DAYS,
        "obi_ic": {f"+{h}": ic_stats(rows[h]) for h in HORIZONS},
        "micro_dev_ic": {f"+{h}": ic_stats(rows_md[h]) for h in HORIZONS},
        "obi_trade": {f"+{h}": simulate_obi_trade(series, h) for h in HORIZONS},
        "paper_account": paper_account(series),
    }
    return result


def render(r) -> str:
    L = ["# Order-Book Imbalance (OBI) Audit -- real free 5-level depth", "",
         f"Depth rows: {r['depth_rows']} | codes: {r['codes']} | days: {len(r['days'])} "
         f"({r['days'][:1]}..{r['days'][-1:]}) | median half-spread: {r['median_half_spread_bps']} bps", ""]
    if not r["sufficient"]:
        L += [f"**STATUS: insufficient data** (need >= {MIN_OBS} rows and >= {MIN_DAYS} days). "
              "The collector accumulates forward each session; re-run this after a few days.", ""]
    L += ["## OBI -> forward return (pooled IC, day-clustered t, top-bottom bps)",
          "| horizon | n | pooled IC | days | mean daily IC | clustered t | top-bottom bps |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for h, s in r["obi_ic"].items():
        L.append(f"| {h} | {s.get('n')} | {s.get('pooled_ic')} | {s.get('n_days')} | "
                 f"{s.get('mean_daily_ic')} | {s.get('day_clustered_t')} | {s.get('top_minus_bottom_bps')} |")
    L += ["", "## OBI TRADE simulation -- long-only, NET of real spread (per-trade bps)",
          "| horizon | trades | TAKER mean bps | taker t | taker hit | MAKER* mean bps | maker hit |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for h, s in r["obi_trade"].items():
        tk, mk = s.get("taker", {}), s.get("maker_optimistic", {})
        L.append(f"| {h} | {s.get('trades')} | {tk.get('mean_bps')} | {tk.get('pooled_t')} | "
                 f"{tk.get('hit_rate')} | {mk.get('mean_bps')} | {mk.get('hit_rate')} |")
    L += ["", "_TAKER = enter at ask / exit at bid (realistic retail floor). "
          "MAKER* = enter at bid earning the spread, exit at mid (OPTIMISTIC ceiling; ignores "
          "adverse selection -- a maker mostly fills when OBI is wrong, so the real number is "
          "well below this, near the taker)._", ""]
    pa = r.get("paper_account", {})
    L += ["", "## OBI PAPER ACCOUNT (isolated, monitorable — NOT the competition account)",
          f"Notional capital ¥{pa.get('capital'):,} | hold {pa.get('horizon_polls')} polls | "
          f"days {pa.get('days')} | trades {pa.get('total_trades')} | "
          f"**total return {pa.get('total_return_pct')}%** | final equity ¥{pa.get('final_equity'):,} | "
          f"maxDD {pa.get('max_drawdown_pct')}%", ""]
    if pa.get("daily_curve"):
        L += ["| date | trades | day return | day P&L ¥ | equity ¥ |", "|---|--:|--:|--:|--:|"]
        for c in pa["daily_curve"][-12:]:
            L.append(f"| {c['date']} | {c['trades']} | {c['day_return_pct']}% | {c['day_pnl_yuan']:,} | {c['equity']:,} |")
    else:
        L.append("_no trades yet — accumulates forward from the collector each session._")
    L += ["", "## Read", "",
          "OBI is real microstructure: a clustered-t>2 with a top-bottom spread that EXCEEDS the "
          f"median half-spread ({r['median_half_spread_bps']} bps) -- and only capturable PASSIVELY "
          "(provide liquidity, don't cross) -- would be the first genuine intraday edge. If the "
          "spread is smaller than the half-spread, it's real but only harvestable as a maker, not "
          "a taker. Gate any use with purged OOS + PBO + realistic (passive) fills.",
          "", "_Diagnostic only; no live change._", ""]
    return "\n".join(L)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    r = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "obi_audit.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(r)
    (OUT / "obi_audit.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
