"""ETF premium / IOPV reversion audit -- does the cross-border ETF premium predict forward
return, and does the edge clear cost? Reads collect_iopv_premium.py logs.

Two honest tests:
  (1) CROSS-SECTIONAL: rank ETFs by premium each snapshot; do LOW-premium (cheap vs NAV) names
      outperform HIGH-premium ones next interval? (negative IC of premium->forward return).
  (2) TIME-SERIES: does a name's premium deviation from its own intraday mean revert?
Caveat: QDII premium is largely STRUCTURAL (quota-driven, persistent) and retail can't
create/redeem or short -- so the level may not revert on a tradeable horizon. The data decides.

Cost-gated (8 bps breakeven), day-clustered; refuses a verdict below MIN_OBS/MIN_DAYS.
Diagnostic only; no live gating. Run: py -3.13 scripts/research_iopv_premium.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT

DEPTH_DIR = ROOT / "outputs" / "iopv_premium"
HORIZONS = [1, 3, 6]      # polls ahead
COST_RT = 0.0008          # ~8 bps breakeven (the candidate-edge floor from the reversal suite)
MIN_OBS = 1500
MIN_DAYS = 3


def load():
    series = defaultdict(list)   # code -> [(date, ts, price, premium)]
    for p in sorted(DEPTH_DIR.glob("iopv_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("price") and r.get("premium_pct") is not None:
                series[r["code"]].append((r.get("date"), r.get("ts"),
                                          float(r["price"]), float(r["premium_pct"])))
    return series


def spearman(x, y):
    if len(x) < 6:
        return None
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else None


def analyze():
    series = load()
    rows = sum(len(s) for s in series.values())
    days = sorted({d for s in series.values() for (d, *_ ) in s})
    # build per (date,ts) panels for cross-sectional IC: premium vs forward return
    by_dt = defaultdict(list)   # (date,ts) -> [(code, premium, price)]
    code_idx = {}               # code -> ordered list for forward lookups
    for code, s in series.items():
        s.sort()
        code_idx[code] = s
        for i, (d, ts, px, prem) in enumerate(s):
            by_dt[(d, ts)].append((code, prem, px, i))
    out = {"rows": rows, "days": days, "codes": len(series),
           "sufficient": rows >= MIN_OBS and len(days) >= MIN_DAYS}
    xsec = {}
    for h in HORIZONS:
        daily_ic = defaultdict(list)
        for (d, ts), members in by_dt.items():
            xs, ys = [], []
            for code, prem, px, i in members:
                s = code_idx[code]
                j = i + h
                if j < len(s) and s[j][0] == d:    # same-day forward
                    fwd = s[j][2] / px - 1.0
                    xs.append(prem); ys.append(fwd)
            ic = spearman(np.array(xs), np.array(ys)) if len(xs) >= 6 else None
            if ic is not None:
                daily_ic[d].append(ic)
        per_day = [np.mean(v) for v in daily_ic.values() if v]
        a = np.array(per_day)
        # top-bottom: buy low-premium (cheap), forward return spread net of cost
        xsec[f"+{h}"] = {
            "panels": int(sum(len(v) for v in daily_ic.values())),
            "mean_ic": round(float(a.mean()), 4) if a.size else None,
            "day_clustered_t": round(float(a.mean() / a.std() * np.sqrt(a.size)), 2) if a.size > 1 and a.std() else None,
            "n_days": int(a.size),
            "note": "negative IC = low-premium(cheap) names outperform (premium reverts)",
        }
    out["cross_sectional_premium_ic"] = xsec
    return out


def render(r) -> str:
    L = ["# ETF Premium / IOPV Reversion Audit", "",
         f"rows: {r['rows']} | codes: {r['codes']} | days: {len(r['days'])} "
         f"({r['days'][:1]}..{r['days'][-1:]})", ""]
    if not r["sufficient"]:
        L += [f"**STATUS: insufficient data** (need >= {MIN_OBS} rows and >= {MIN_DAYS} days). "
              "Collector accumulates forward each session; re-run after a few days.", ""]
    L += ["## Cross-sectional: premium -> forward return (does cheap-vs-NAV outperform?)",
          "| horizon | panels | mean IC | clustered t | days |", "|---|--:|--:|--:|--:|"]
    for h, s in r["cross_sectional_premium_ic"].items():
        L.append(f"| {h} | {s.get('panels')} | {s.get('mean_ic')} | {s.get('day_clustered_t')} | {s.get('n_days')} |")
    L += ["", "## Read", "",
          "A robust NEGATIVE IC (low premium -> higher forward return), |t|>2 day-clustered, with a "
          f"top-bottom spread exceeding ~8 bps cost, would be the first cost-surviving spot signal. "
          "BUT QDII premium is structural/persistent and retail can't create-redeem or short, so the "
          "tradeable form is long-only 'buy the relative discount' -- and the level may not revert on "
          "a tradeable horizon. Gate any use with purged OOS + the 8 bps breakeven. Diagnostic only.",
          ""]
    return "\n".join(L)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    r = analyze()
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    (DEPTH_DIR / "iopv_premium_audit.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(r)
    (DEPTH_DIR / "iopv_premium_audit.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
