"""Markov vs Chebyshev vs Chernoff on minute bars -- which concept actually helps
minute-level anticipation, tested through the standard gates.

Conceptual ground truth this script quantifies (user question: trading data as
"sequentially continuous yet mutually independent"):
  Only a MARKOV CHAIN is a *model* (it can encode sequential structure and SIMULATE forward
  paths). Chebyshev / Chernoff (and Markov's inequality) are tail BOUNDS -- they cap
  P(|X|>=x) from moments; they predict nothing and simulate nothing. Their correct use is
  stop/size risk budgeting. The "continuous yet unrelated" intuition is the martingale
  property: direction ~uncorrelated, magnitude strongly dependent (volatility clustering).

Four preregistered measurements on mootdx 1-min bars (18 traded codes, ~4.5 months) with the
5-min 2-year panel as a long-sample cross-check:
  1. DEPENDENCE DECOMPOSITION: per-code likelihood-ratio of a first-order Markov chain vs an
     iid multinomial, separately for SIGN states (direction) and |r|-quantile states
     (magnitude). Expectation under martingale reality: magnitude LR >> sign LR.
  2. PREDICTION: walk-forward 5-state chain (states by trailing-window return quantiles;
     transition matrix + state-conditional next-bar mean re-estimated daily from a trailing
     15-session window; strictly causal). Forecast E[r_{t+1}|s_t]. Metrics: pooled-per-day IC
     (day-clustered t) and net-of-cost verdict for a taker round trip (15.5bp) on the top-|f|
     decile -- the cost wall every prior minute-scale signal died on.
  3. SIMULATION CALIBRATION: Monte-Carlo the chain 30 minutes ahead each hour; PIT-style
     check: share of realized 30-min moves inside the simulated 10-90% band (target ~0.80).
     Simulation is a DISTRIBUTION tool, not a point forecast.
  4. BOUND TIGHTNESS: for 30-min |move| >= 2/3 trailing sigma: empirical exceedance vs
     Markov-ineq (on |X|), Chebyshev, and sub-Gaussian Chernoff bounds from trailing moments.
     Valid = bound >= empirical; tight = smallest valid.

Diagnostic only. Run: py -3.13 scripts/research_markov_minute.py [--freq 1m|5m]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT

OUT_DIR = ROOT / "outputs" / "markov_minute"
N_STATES = 5
TRAIL_SESSIONS = 15
COST_RT_BPS = 15.5
SIM_HORIZON = 30          # bars ahead for simulation (30x1min or 30x5min)
SIM_PATHS = 400


def load(freq: str) -> dict[str, pd.DataFrame]:
    bars_dir = ROOT / "data" / "market" / "mootdx" / f"bars_{freq}"
    out = {}
    for p in sorted(bars_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        df = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
        df["date"] = df["dt"].str[:10]
        df["r"] = df["close"].pct_change()
        out[p.stem] = df.dropna(subset=["r"]).reset_index(drop=True)
    return out


def lr_markov_vs_iid(states: np.ndarray, k: int) -> float:
    """2*(LL_markov - LL_iid); df=(k-1)*k - (k-1) but we report the statistic per 1000 obs."""
    n = len(states)
    if n < 200:
        return np.nan
    counts = np.zeros((k, k))
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    row = counts.sum(axis=1, keepdims=True)
    P = counts / np.where(row > 0, row, 1)
    marg = counts.sum(axis=0) / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_m = np.nansum(counts * np.log(np.where(P > 0, P, 1)))
        ll_0 = np.nansum(counts.sum(axis=0) * np.log(np.where(marg > 0, marg, 1)))
    return float(2.0 * (ll_m - ll_0) / (n / 1000.0))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freq", default="1m", choices=["1m", "5m"])
    args = ap.parse_args()
    t0 = time.time()
    data = load(args.freq)
    print(f"codes {len(data)} | freq {args.freq} [{time.time()-t0:.0f}s]", flush=True)

    # ---- 1. dependence decomposition -------------------------------------------------
    dep = []
    for code, df in data.items():
        r = df["r"].to_numpy()
        sign = (r > 0).astype(int)                       # 2-state direction chain
        q = pd.qcut(pd.Series(np.abs(r)), 4, labels=False, duplicates="drop").to_numpy()  # magnitude
        q = np.where(np.isfinite(q), q, 0).astype(int)
        dep.append({"code": code,
                    "lr_sign_per1k": round(lr_markov_vs_iid(sign, 2), 1),
                    "lr_mag_per1k": round(lr_markov_vs_iid(q, int(np.nanmax(q)) + 1), 1)})
    lr_sign = float(np.nanmedian([d["lr_sign_per1k"] for d in dep]))
    lr_mag = float(np.nanmedian([d["lr_mag_per1k"] for d in dep]))
    print(f"1) DEPENDENCE median LR/1k obs: direction(sign)={lr_sign:.1f} vs magnitude(|r|)={lr_mag:.1f}", flush=True)

    # ---- 2. walk-forward Markov prediction -------------------------------------------
    ic_by_day: dict[str, list[float]] = {}
    net_rows = []
    for code, df in data.items():
        dates = df["date"].unique()
        for di in range(TRAIL_SESSIONS, len(dates)):
            train = df[df["date"].isin(dates[di - TRAIL_SESSIONS:di])]
            test = df[df["date"] == dates[di]]
            if len(train) < 500 or len(test) < 30:
                continue
            edges = np.quantile(train["r"], np.linspace(0, 1, N_STATES + 1)[1:-1])
            tr_s = np.digitize(train["r"], edges)
            te_s = np.digitize(test["r"], edges)
            counts = np.zeros((N_STATES, N_STATES))
            for a, b in zip(tr_s[:-1], tr_s[1:]):
                counts[a, b] += 1
            row = counts.sum(axis=1, keepdims=True)
            P = counts / np.where(row > 0, row, 1)
            cond_mean = np.array([train["r"][tr_s == s].mean() if (tr_s == s).any() else 0.0
                                  for s in range(N_STATES)])
            f = (P @ cond_mean)[te_s[:-1]]                # E[r_{t+1} | s_t]
            realized = test["r"].to_numpy()[1:]
            if len(f) < 20 or np.std(f) == 0:
                continue
            ic = pd.Series(f).corr(pd.Series(realized), method="spearman")
            if pd.notna(ic):
                ic_by_day.setdefault(dates[di], []).append(float(ic))
            thr = np.quantile(np.abs(f), 0.9)
            m = np.abs(f) >= thr
            if m.any():
                net_rows.append({"date": dates[di],
                                 "gross_bps": float(np.mean(np.sign(f[m]) * realized[m]) * 1e4)})
    day_ic = pd.Series({d: np.mean(v) for d, v in ic_by_day.items()}).sort_index()
    t_ic = float(day_ic.mean() / day_ic.std(ddof=1) * np.sqrt(len(day_ic))) if day_ic.std(ddof=1) else np.nan
    nets = pd.DataFrame(net_rows).groupby("date")["gross_bps"].mean()
    t_net = float(nets.mean() / nets.std(ddof=1) * np.sqrt(len(nets))) if len(nets) > 30 and nets.std(ddof=1) else np.nan
    print(f"2) PREDICTION days={len(day_ic)} IC={day_ic.mean():.4f} t={t_ic:.2f} | "
          f"top-decile gross {nets.mean():.2f}bps/bar-trade t={t_net:.2f} vs taker cost {COST_RT_BPS}bps", flush=True)

    # ---- 3. simulation calibration ----------------------------------------------------
    hits = tot = 0
    rng = np.random.default_rng(11)
    for code, df in data.items():
        dates = df["date"].unique()
        for di in range(TRAIL_SESSIONS, len(dates), 3):
            train = df[df["date"].isin(dates[di - TRAIL_SESSIONS:di])]
            test = df[df["date"] == dates[di]].reset_index(drop=True)
            if len(train) < 500 or len(test) < SIM_HORIZON + 10:
                continue
            edges = np.quantile(train["r"], np.linspace(0, 1, N_STATES + 1)[1:-1])
            tr_s = np.digitize(train["r"], edges)
            counts = np.zeros((N_STATES, N_STATES))
            for a, b in zip(tr_s[:-1], tr_s[1:]):
                counts[a, b] += 1
            P = counts / np.where(counts.sum(axis=1, keepdims=True) > 0, counts.sum(axis=1, keepdims=True), 1)
            pools = [train["r"][tr_s == s].to_numpy() for s in range(N_STATES)]
            for start in range(0, len(test) - SIM_HORIZON, SIM_HORIZON):
                s0 = int(np.digitize([test["r"].iloc[start]], edges)[0])
                sims = np.zeros(SIM_PATHS)
                for p_i in range(SIM_PATHS):
                    s, acc = s0, 0.0
                    for _ in range(SIM_HORIZON):
                        s = rng.choice(N_STATES, p=P[s] if P[s].sum() > 0.99 else np.ones(N_STATES) / N_STATES)
                        pool = pools[s]
                        acc += pool[rng.integers(len(pool))] if len(pool) else 0.0
                    sims[p_i] = acc
                real = float(test["close"].iloc[start + SIM_HORIZON] / test["close"].iloc[start] - 1.0)
                lo, hi = np.quantile(sims, [0.10, 0.90])
                hits += int(lo <= real <= hi)
                tot += 1
    print(f"3) SIMULATION 10-90% band coverage: {hits}/{tot} = {hits/max(tot,1):.3f} (target ~0.80)", flush=True)

    # ---- 4. bound tightness on 30-bar |move| -----------------------------------------
    emp2 = emp3 = n_obs = 0
    for code, df in data.items():
        r30 = df["close"].pct_change(SIM_HORIZON).dropna()
        sig = r30.rolling(500).std().shift(1)
        z = (r30.abs() / sig).dropna()
        emp2 += int((z >= 2).sum()); emp3 += int((z >= 3).sum()); n_obs += len(z)
    e2, e3 = emp2 / n_obs, emp3 / n_obs
    bounds = {
        "empirical": {"2sig": round(e2, 4), "3sig": round(e3, 4)},
        "markov_ineq_on_|X|": {"2sig": round(1 / 2 * np.sqrt(2 / np.pi), 4), "3sig": round(1 / 3 * np.sqrt(2 / np.pi), 4)},
        "chebyshev": {"2sig": 0.25, "3sig": round(1 / 9, 4)},
        "chernoff_subgaussian": {"2sig": round(2 * np.exp(-2.0), 4), "3sig": round(2 * np.exp(-4.5), 4)},
    }
    print(f"4) BOUNDS 30-bar |move| exceedance: {json.dumps(bounds)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"markov_minute_{args.freq}.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(), "status": "diagnostic_only",
        "freq": args.freq, "dependence_median_lr_per1k": {"sign": lr_sign, "magnitude": lr_mag},
        "dependence_per_code": dep,
        "prediction": {"days": len(day_ic), "mean_ic": round(float(day_ic.mean()), 4),
                        "day_t": round(t_ic, 2), "top_decile_gross_bps": round(float(nets.mean()), 2),
                        "gross_t": round(t_net, 2), "taker_cost_bps": COST_RT_BPS},
        "simulation_band_coverage": round(hits / max(tot, 1), 3),
        "bounds_30bar": bounds,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done [{time.time()-t0:.0f}s] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
