"""Two-year policy-semantics probe of DEFER-WEAK selling, before any mechanism is built:
when a non-urgent sell wants to fire while the name trades BELOW its session VWAP (the
statistically worst moment to sell: weak-decile bars carry +20.5bps t=9.0 remaining-to-close
vs average, research_sell_operator_sweep.py), defer the sell until the FIRST bar back at/above
session VWAP (sell into strength) or the session close, whichever comes first.

Exact semantics probed per (code, day, bar) where close < session_vwap:
  immediate : sell at this bar's close.
  deferred  : sell at the first later bar with close >= running session VWAP; if none, sell
              at the day's close (guaranteed completion -- mirrors a 14:45 fail-open in live).
  improvement_bps = (deferred_px / immediate_px - 1) * 1e4   (spread cancels: one sell either way)

Two populations: ALL weak bars, and TROUBLED weak bars (also >=0.5% below session high --
closer to where the live sell score actually fires). Day-clustered stats + the tail the sweep
could not show: P5/P10 of improvement and the share of bars where deferring lost > 50bps
(the all-day-slide case; live hard stops still cap that path, but the probe must expose it).
Both halves (split 2025-07-01) must agree for this to advance to the 60d replay gate.

STRICTLY offline; reads mootdx 5-min bars for the 18 traded codes. No config change.
Run: py -3.13 scripts/research_sell_defer_weak_2y.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT

BARS_DIR = ROOT / "data" / "market" / "mootdx" / "bars_5m"
OUT_DIR = ROOT / "outputs" / "sell_operator_sweep"
SPLIT = "2025-07-01"
LAST_BARS_EXCLUDED = 3      # a sell wanted in the final ~15min just executes; nothing to defer
TROUBLE_DD = -0.005         # troubled = also >=0.5% below session high


def process_code(code: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in (BARS_DIR / f"{code}.jsonl").read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    df["date"] = df["dt"].str[:10]
    df = df.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    out = []
    for d, g in df.groupby("date"):
        c = g["close"].to_numpy()
        v = g["vol"].to_numpy()
        a = g["amount"].to_numpy()
        n = len(c)
        if n < 20:
            continue
        vwap = np.cumsum(a) / np.where(np.cumsum(v) > 0, np.cumsum(v), np.nan)
        sess_high = np.maximum.accumulate(g["high"].to_numpy())
        day_close = c[-1]
        weak = c < vwap
        for i in range(6, n - LAST_BARS_EXCLUDED):
            if not weak[i] or not np.isfinite(vwap[i]):
                continue
            # deferred exit: first later bar at/above the running session VWAP, else day close
            exit_px = day_close
            for j in range(i + 1, n):
                if np.isfinite(vwap[j]) and c[j] >= vwap[j]:
                    exit_px = c[j]
                    break
            out.append({
                "date": d,
                "improve_bps": (exit_px / c[i] - 1.0) * 1e4,
                "troubled": bool(c[i] / sess_high[i] - 1.0 <= TROUBLE_DD),
            })
    return pd.DataFrame(out)


def day_stats(df: pd.DataFrame) -> dict:
    per_day = df.groupby("date")["improve_bps"].mean()
    if len(per_day) < 60:
        return {"days": len(per_day)}
    t = float(per_day.mean() / per_day.std(ddof=1) * np.sqrt(len(per_day)))
    h1 = per_day[per_day.index < SPLIT]
    h2 = per_day[per_day.index >= SPLIT]
    def tt(x):
        return round(float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))), 2) if len(x) > 30 and x.std(ddof=1) else None
    return {
        "days": len(per_day), "bars": len(df),
        "mean_bps": round(float(per_day.mean()), 2), "day_t": round(t, 2),
        "h1": {"days": len(h1), "mean_bps": round(float(h1.mean()), 2), "t": tt(h1)},
        "h2": {"days": len(h2), "mean_bps": round(float(h2.mean()), 2), "t": tt(h2)},
        "bar_p5_bps": round(float(df["improve_bps"].quantile(0.05)), 1),
        "bar_p10_bps": round(float(df["improve_bps"].quantile(0.10)), 1),
        "share_lose_gt50bps": round(float((df["improve_bps"] < -50).mean()), 4),
        "share_deferred_to_close": None,   # filled by caller when tracked
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    frames = [process_code(p.stem) for p in sorted(BARS_DIR.glob("*.jsonl"))]
    data = pd.concat(frames, ignore_index=True)
    print(f"weak-state sell opportunities: {len(data)} bars over {data['date'].nunique()} days [{time.time()-t0:.0f}s]", flush=True)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(), "status": "diagnostic_only",
        "policy": "defer weak-state (below session VWAP) non-urgent sells to first at/above-VWAP bar, else day close",
        "all_weak": day_stats(data),
        "troubled_weak": day_stats(data[data["troubled"]]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sell_defer_weak_2y.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ALL WEAK      :", json.dumps(result["all_weak"], ensure_ascii=False))
    print("TROUBLED WEAK :", json.dumps(result["troubled_weak"], ensure_ascii=False))
    print(f"log: {OUT_DIR / 'sell_defer_weak_2y.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
