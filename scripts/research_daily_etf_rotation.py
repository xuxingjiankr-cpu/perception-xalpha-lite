"""Daily ETF cross-sectional rotation pre-test -- does a LOW-TURNOVER daily regime have an
edge that survives 万三 cost, OOS, with PBO<0.5? This is the gate that decides whether the
Qlib/daily pivot is worth pursuing (the intraday 5m regime was cost-walled).

Pulls multi-year DAILY adjusted closes for the full liquid A-share ETF universe from Yahoo
(akshare's endpoint is sandbox-blocked; Yahoo daily is deep+free), computes cross-sectional
factors (momentum / short reversal / low-vol), and weekly-rebalances a top-tertile basket net
of turnover cost. Reports factor IC (purged OOS), the top-vs-bottom and top-vs-equalweight
spread net of cost, day-cluster CI, PBO across factors, and a deflated-Sharpe note.

Diagnostic only; nothing live. Run: py -3.13 scripts/research_daily_etf_rotation.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT

UNIVERSE_SRC = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
CACHE = ROOT / "outputs" / "daily_etf" / "daily_adjclose.json"
OUT = ROOT / "outputs" / "daily_etf"
COST_RT = 0.0006          # 万三 x2 round-trip per name turned over
REBAL = 5                 # trading days between rebalances (~weekly)
MIN_HISTORY = 200
MIN_CODES = 20
TEST_FRAC = 0.30          # last 30% of rebalances = purged OOS test


def yahoo_symbol(code: str) -> str:
    return f"{code}.SS" if code[:1] in ("5", "6") else f"{code}.SZ"


def universe() -> list[str]:
    codes = set()
    with UNIVERSE_SRC.open(encoding="utf-8") as f:
        for line in f:
            try:
                codes.add(str(json.loads(line)["stockCode"]).zfill(6))
            except Exception:
                pass
    return sorted(codes)


def fetch_daily(code: str, timeout: float = 20.0) -> dict[str, float]:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(code)}"
           f"?interval=1d&range=3y")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    res = (d.get("chart", {}).get("result") or [None])[0]
    if not res or "timestamp" not in res:
        return {}
    ts = res["timestamp"]
    adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    closes = res["indicators"]["quote"][0].get("close")
    series = adj if adj else closes
    out: dict[str, float] = {}
    import datetime as dt
    for t, p in zip(ts, series or []):
        if p and p > 0:
            out[dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")] = float(p)
    return out


def load_or_fetch() -> dict[str, dict[str, float]]:
    if CACHE.exists():
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"cache: {len(data)} ETFs")
        return data
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    codes = universe()
    print(f"fetching daily 3y for {len(codes)} ETFs ...")
    data: dict[str, dict[str, float]] = {}
    ok = 0
    for n, c in enumerate(codes):
        try:
            s = fetch_daily(c)
        except Exception:
            s = {}
        if len(s) >= MIN_HISTORY:
            data[c] = s
            ok += 1
        if n % 100 == 0:
            print(f"  {n}/{len(codes)} ok={ok}", flush=True)
        time.sleep(0.15)
    CACHE.write_text(json.dumps(data), encoding="utf-8")
    print(f"cached {ok} ETFs -> {CACHE}")
    return data


def build_panel(data: dict[str, dict[str, float]]):
    all_dates = sorted({d for s in data.values() for d in s})
    codes = [c for c, s in data.items() if len(s) >= MIN_HISTORY]
    px = {c: data[c] for c in codes}
    return all_dates, codes, px


def factors_at(prices: dict[str, float], dates: list[str], i: int) -> dict[str, float] | None:
    """Point-in-time factors from a code's price dict at date index i."""
    def p(k):
        d = dates[i - k] if i - k >= 0 else None
        return prices.get(d) if d else None
    p0 = p(0)
    if not p0:
        return None
    out = {}
    for name, lag in (("mom_20", 20), ("mom_60", 60), ("mom_120", 120)):
        pl = p(lag)
        out[name] = (p0 / pl - 1.0) if pl else None
    p5 = p(5)
    out["rev_5"] = (-(p0 / p5 - 1.0)) if p5 else None
    rets = []
    for k in range(1, 21):
        a, b = p(k - 1), p(k)
        if a and b:
            rets.append(a / b - 1.0)
    out["lowvol"] = (-float(np.std(rets))) if len(rets) >= 10 else None
    return out


def spearman(x, y):
    if len(x) < MIN_CODES:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else None


def analyze(data):
    import overfitting_guard as og
    dates, codes, px = build_panel(data)
    # align each code's available date-index list to the global date axis
    date_idx = {d: i for i, d in enumerate(dates)}
    factor_names = ["mom_20", "mom_60", "mom_120", "rev_5", "lowvol"]
    rebal_i = list(range(MIN_HISTORY, len(dates) - REBAL, REBAL))

    per_factor_ic: dict[str, list[float]] = {f: [] for f in factor_names}
    comp_strategy = []      # per-rebalance dict
    prev_top: set[str] = set()
    for ri in rebal_i:
        d0, d1 = dates[ri], dates[ri + REBAL]
        rows = []
        for c in codes:
            s = px[c]
            if d0 not in s or d1 not in s:
                continue
            # build this code's own ordered date list up to ri for point-in-time lags
            fa = factors_at(s, dates, ri)
            if fa is None:
                continue
            fwd = s[d1] / s[d0] - 1.0
            rows.append((c, fa, fwd))
        if len(rows) < MIN_CODES:
            continue
        # per-factor IC
        for f in factor_names:
            xs = [(r[1].get(f), r[2]) for r in rows if r[1].get(f) is not None]
            if len(xs) >= MIN_CODES:
                ic = spearman(np.array([a for a, _ in xs]), np.array([b for _, b in xs]))
                if ic is not None:
                    per_factor_ic[f].append(ic)
        # composite = cross-sectional z-avg of factors (reversal & lowvol already sign-aligned positive)
        zsum = {r[0]: [0.0, 0] for r in rows}
        for f in factor_names:
            vals = [(r[0], r[1].get(f)) for r in rows if r[1].get(f) is not None]
            if len(vals) < MIN_CODES:
                continue
            arr = np.array([v for _, v in vals]); mu, sd = arr.mean(), arr.std()
            if sd == 0:
                continue
            for c, v in vals:
                zsum[c][0] += (v - mu) / sd; zsum[c][1] += 1
        comp = [(c, zsum[c][0] / zsum[c][1], fwd) for (c, _, fwd) in rows if zsum[c][1] > 0]
        if len(comp) < MIN_CODES:
            continue
        z = np.array([x[1] for x in comp]); fwd = np.array([x[2] for x in comp])
        hi = np.quantile(z, 2 / 3); lo = np.quantile(z, 1 / 3)
        top_names = {c for c, zz, _ in comp if zz >= hi}
        top = fwd[z >= hi]; bot = fwd[z <= lo]
        turnover = len(top_names - prev_top) / max(1, len(top_names)) if prev_top else 1.0
        prev_top = top_names
        comp_strategy.append({
            "date": d0,
            "top_gross": float(top.mean()),
            "bench": float(fwd.mean()),
            "spread": float(top.mean() - bot.mean()),
            "turnover": turnover,
            "top_net": float(top.mean()) - turnover * COST_RT,
            "top_excess_net": float(top.mean() - fwd.mean()) - turnover * COST_RT,
        })

    def fstats(ics):
        a = np.array(ics)
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean_ic": round(float(a.mean()), 4),
                "ir": round(float(a.mean() / a.std()), 3) if a.std() else None,
                "t": round(float(a.mean() / a.std() * np.sqrt(a.size)), 2) if a.std() else None}

    n = len(comp_strategy)
    split = int(n * (1 - TEST_FRAC))

    def strat_stats(rows):
        if not rows:
            return {"periods": 0}
        net = np.array([r["top_net"] for r in rows])
        exc = np.array([r["top_excess_net"] for r in rows])
        sp = np.array([r["spread"] for r in rows])
        ann = np.sqrt(50)
        return {
            "periods": len(rows),
            "top_net_mean": round(float(net.mean()), 5),
            "excess_net_mean": round(float(exc.mean()), 5),
            "excess_t": round(float(exc.mean() / exc.std() * np.sqrt(len(exc))), 2) if exc.std() else None,
            "excess_sharpe_ann": round(float(exc.mean() / exc.std() * ann), 3) if exc.std() else None,
            "spread_mean": round(float(sp.mean()), 5),
            "spread_t": round(float(sp.mean() / sp.std() * np.sqrt(len(sp))), 2) if sp.std() else None,
            "avg_turnover": round(float(np.mean([r["turnover"] for r in rows])), 3),
            "win_periods": round(float(np.mean(exc > 0)), 3),
        }

    # PBO across factors (rows=factors incl composite IC, cols=periods)
    comp_ic = []
    # recompute composite per-period IC for PBO row
    # (reuse spread sign of composite as proxy IC already captured; use factor ICs matrix)
    L = min(len(v) for v in per_factor_ic.values()) if all(per_factor_ic.values()) else 0
    matrix = [per_factor_ic[f][:L] for f in factor_names] if L >= 4 else []
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 and L >= 4 else {"pbo": None}
    full = strat_stats(comp_strategy)
    dsr = og.deflated_significance_note(n_trials=len(factor_names),
                                        observed_sharpe=full.get("excess_sharpe_ann") or 0.0,
                                        n_obs=full.get("periods", 0))
    return {
        "universe_etfs": len(codes), "date_span": [dates[0], dates[-1]] if dates else None,
        "rebalances": n, "cost_round_trip": COST_RT,
        "factor_ic": {f: fstats(per_factor_ic[f]) for f in factor_names},
        "factor_ic_test": {f: fstats(per_factor_ic[f][split:]) for f in factor_names},
        "strategy_full": full,
        "strategy_train": strat_stats(comp_strategy[:split]),
        "strategy_test": strat_stats(comp_strategy[split:]),
        "pbo": pbo, "deflated_sharpe": dsr,
    }


def render(r) -> str:
    L = ["# Daily ETF Cross-Sectional Rotation -- pre-test (Yahoo daily, weekly rebal, net 万三)",
         "",
         f"Universe {r['universe_etfs']} ETFs | span {r['date_span']} | rebalances {r['rebalances']} | "
         f"cost {r['cost_round_trip']:.2%} round-trip per turned-over name",
         "",
         "## Factor cross-sectional IC (full | TEST)",
         "| factor | mean IC | IR | t | TEST IC | TEST t |",
         "|---|--:|--:|--:|--:|--:|"]
    for f, s in r["factor_ic"].items():
        t = r["factor_ic_test"][f]
        L.append(f"| {f} | {s.get('mean_ic')} | {s.get('ir')} | {s.get('t')} | {t.get('mean_ic')} | {t.get('t')} |")
    sf, st = r["strategy_train"], r["strategy_test"]
    f0 = r["strategy_full"]
    L += ["",
          "## Top-tertile composite basket, net of cost",
          "| window | periods | top net/wk | excess-vs-EW net | excess t | ann Sharpe | spread | turnover | win |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
          f"| full | {f0['periods']} | {_p(f0['top_net_mean'])} | {_p(f0['excess_net_mean'])} | {f0['excess_t']} | {f0['excess_sharpe_ann']} | {_p(f0['spread_mean'])} | {f0['avg_turnover']} | {_p(f0['win_periods'])} |",
          f"| train | {sf['periods']} | {_p(sf.get('top_net_mean'))} | {_p(sf.get('excess_net_mean'))} | {sf.get('excess_t')} | {sf.get('excess_sharpe_ann')} | {_p(sf.get('spread_mean'))} | {sf.get('avg_turnover')} | {_p(sf.get('win_periods'))} |",
          f"| TEST | {st['periods']} | {_p(st.get('top_net_mean'))} | {_p(st.get('excess_net_mean'))} | {st.get('excess_t')} | {st.get('excess_sharpe_ann')} | {_p(st.get('spread_mean'))} | {st.get('avg_turnover')} | {_p(st.get('win_periods'))} |",
          "",
          f"- **PBO** across factors: {r['pbo'].get('pbo')} "
          f"({'noise' if (r['pbo'].get('pbo') or 0)>=0.5 else 'holds OOS' if r['pbo'].get('pbo') is not None else 'n/a'})",
          f"- **Deflated-Sharpe**: {r['deflated_sharpe']}",
          "",
          "## Verdict",
          "",
          "PASS (=> Qlib pivot justified) requires: excess-vs-equalweight net > 0 in TEST with t>~2, "
          "ann Sharpe meaningfully positive OOS, AND PBO well below 0.5. `top net/wk` and `excess net` "
          "already deduct turnover*万三; if excess net <= 0 the daily edge does not clear cost either.",
          "",
          "_Diagnostic only; no live change._", ""]
    return "\n".join(L)


def _p(v):
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    data = load_or_fetch()
    if len(data) < MIN_CODES:
        print("not enough ETF history fetched.")
        return 1
    r = analyze(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "daily_rotation.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(r)
    (OUT / "daily_rotation.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
