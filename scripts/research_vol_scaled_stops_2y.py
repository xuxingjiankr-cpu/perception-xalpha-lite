"""Two-year probe: should stops be volatility-scaled per code? (application test of the
Markov/magnitude findings + the stop-vs-noise-band audit of 2026-07-05.)

Audit context: the fixed -2% emergency stop sits INSIDE the daily noise band for the
high-vol names (513180: worst-entry MAE beyond -2% on 16% of days; 588000/588030/159546:
19-31%) but OUTSIDE it for gold/US-index names (~2-5%). Hypothesis: widening stops to
-K x trailing-20d-sigma per code reduces noise stop-outs and improves net outcome.

Probe: agent-like entries (top-decile 20-min momentum bars, up to 3/day/code, entry at bar
close) on the 18 traded codes x ~480 days of 5-min bars; exits compared per entry:
  hold_to_close | fixed -2% stop | vol-scaled stop = -clamp(2.5 x sigma20, 1.2%..4.5%).
Day-clustered paired differences + tail quantiles (P1/P5 per trade, worst day).

RESULT (2026-07-05 run): momentum-chase entries on this universe are -14.5bps/trade to
close (t=-5.6) before exits even matter (reversal-dominated tape -- consistent with every
prior line). Fixed -2% slightly IMPROVES the mean (-13.2bps, truncation wins: what falls
2% from a momentum-chase entry tends to keep falling) and caps the tail (trade P1 -2.00%
vs -3.13% hold; worst day -1.64% vs -2.55%). Vol-scaled (median level -3.74%) is so wide
it barely binds: == hold (paired -0.07bps, t=-0.35) and LOSES 1.44bps/trade vs fixed
(t=-2.79) with fatter tails. VERDICT: vol-scaling REJECTED; the fixed -2% emergency stop
is validated as well-placed -- the audit's "inside the noise band" concern does not
translate into losses because noise-stop-outs are offset by genuine-break truncation.
No config change. Diagnostic only.

Run: py -3.13 scripts/research_vol_scaled_stops_2y.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT

BARS = ROOT / "data" / "market" / "mootdx" / "bars_5m"
K_SIGMA, FLOOR, CAP = 2.5, 0.012, 0.045


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    all_trades = []
    for p in sorted(BARS.glob("*.jsonl")):
        df = pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()])
        df["date"] = df["dt"].str[:10]
        daily = df.groupby("date").agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last"))
        daily["sigma20"] = ((daily["hi"] - daily["lo"]) / daily["cl"]).rolling(20).mean().shift(1)
        for d, g in df.groupby("date"):
            s20 = daily.loc[d, "sigma20"]
            if not np.isfinite(s20) or len(g) < 30:
                continue
            c = g["close"].to_numpy(); lo = g["low"].to_numpy()
            n = len(c); day_close = c[-1]
            mom = c[4:n - 8] / c[:n - 12] - 1.0
            thr = np.quantile(mom, 0.9)
            for idx in np.where(mom >= max(thr, 0.001))[0][:3]:
                i = idx + 4
                entry = c[i]
                hold_ret = day_close / entry - 1.0
                def stopped(stop_pct):
                    level = entry * (1 + stop_pct)
                    for j in range(i + 1, n):
                        if lo[j] <= level:
                            return level / entry - 1.0
                    return hold_ret
                vol_stop = -min(max(K_SIGMA * s20, FLOOR), CAP)
                all_trades.append({"date": d, "hold": hold_ret, "fix2": stopped(-0.02),
                                   "vol": stopped(vol_stop)})
    tr = pd.DataFrame(all_trades)
    print(f"trades {len(tr)} | days {tr['date'].nunique()}")
    for col, label in [("hold", "hold_to_close"), ("fix2", "fixed -2%"), ("vol", "vol-scaled")]:
        per_day = tr.groupby("date")[col].mean()
        t = per_day.mean() / per_day.std(ddof=1) * np.sqrt(len(per_day))
        print(f"{label:<14} mean {tr[col].mean()*1e4:+7.2f}bps day-t {t:+.2f} "
              f"P1 {tr[col].quantile(0.01)*100:+.2f}% worst-day {per_day.min()*100:+.2f}%")
    for a, b in [("vol", "fix2"), ("vol", "hold"), ("fix2", "hold")]:
        d_ = tr.groupby("date").apply(lambda g: (g[a] - g[b]).mean(), include_groups=False)
        t = d_.mean() / d_.std(ddof=1) * np.sqrt(len(d_))
        print(f"paired {a}-{b}: {d_.mean()*1e4:+.2f}bps day-t {t:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
