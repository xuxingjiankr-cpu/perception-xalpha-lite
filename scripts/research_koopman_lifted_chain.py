"""EDMD / Koopman-lifted transition regression -- the exact object the user proposes
("use the Koopman operator to strengthen Markov + LPPLS + HMM + Laplace"): a linear
operator fitted on a lifted dictionary of observables, walk-forward, predicting the
next-bar return. Dictionary = Markov 5-state indicators (the plain chain's basis) +
Laplace exponential kernels (5 half-lives) + volatility / drawdown observables (the
HMM/EWS-style magnitude state). Mathematically: the stochastic Koopman operator IS the
Markov transition operator; richer dictionaries change the approximation basis, not the
information content of the same scalar price series.

Preregistered judgment: compare against the plain 5-state chain from
research_markov_minute.py on the SAME data/protocol: (a) does the lifted operator raise
OOS day-clustered IC materially? (b) does top-|forecast|-decile GROSS bps/trade approach
the 15.5bp taker cost? The ceiling hypothesis says no: the chain already estimates the
conditional law to within noise (day-t 26-34), so basis sophistication cannot create
information -- only new DATA can. Diagnostic only.

Run: py -3.13 scripts/research_koopman_lifted_chain.py [--freq 1m|5m]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT
from research_markov_minute import load, TRAIL_SESSIONS, N_STATES, COST_RT_BPS

OUT_DIR = ROOT / "outputs" / "markov_minute"
HALF_LIVES = [2, 6, 12, 24, 60]
RIDGE = 5.0


def lift(df: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Dictionary of observables at each bar (all causal)."""
    r = df["r"].to_numpy()
    c = df["close"].to_numpy()
    feats = []
    s = np.digitize(r, edges)
    for k in range(N_STATES):                      # Markov-state indicator basis
        feats.append((s == k).astype(float))
    rs = pd.Series(r)
    for h in HALF_LIVES:                           # Laplace exponential-kernel basis
        feats.append(rs.ewm(halflife=h, min_periods=2).mean().to_numpy())
    feats.append(rs.rolling(12).std().to_numpy())  # magnitude observable (HMM-ish state)
    dn = rs.clip(upper=0.0)
    feats.append((dn.pow(2).rolling(12).sum() / rs.pow(2).rolling(12).sum().replace(0, np.nan)).to_numpy())
    g = df.groupby("date")["high"].cummax().to_numpy()
    feats.append(c / g - 1.0)                      # drawdown observable
    return np.column_stack(feats)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freq", default="1m", choices=["1m", "5m"])
    args = ap.parse_args()
    t0 = time.time()
    data = load(args.freq)

    ic_by_day: dict[str, list[float]] = {}
    net_rows = []
    for code, df in data.items():
        df = df.reset_index(drop=True)
        dates = df["date"].unique()
        for di in range(TRAIL_SESSIONS, len(dates)):
            tr_mask = df["date"].isin(dates[di - TRAIL_SESSIONS:di]).to_numpy()
            te_mask = (df["date"] == dates[di]).to_numpy()
            train, test = df[tr_mask], df[te_mask]
            if len(train) < 600 or len(test) < 30:
                continue
            edges = np.quantile(train["r"], np.linspace(0, 1, N_STATES + 1)[1:-1])
            X_all = lift(df, edges)
            Xtr, ytr = X_all[tr_mask][:-1], train["r"].to_numpy()[1:]
            m = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
            Xtr, ytr = Xtr[m], ytr[m]
            if len(ytr) < 400:
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            Z = (Xtr - mu) / sd
            beta = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(Z.shape[1]), Z.T @ ytr)
            Xte = X_all[te_mask][:-1]
            realized = test["r"].to_numpy()[1:]
            mte = np.isfinite(Xte).all(axis=1)
            f = np.full(len(Xte), np.nan)
            f[mte] = ((Xte[mte] - mu) / sd) @ beta
            ok = np.isfinite(f) & np.isfinite(realized)
            if ok.sum() < 20 or np.nanstd(f[ok]) == 0:
                continue
            ic = pd.Series(f[ok]).corr(pd.Series(realized[ok]), method="spearman")
            if pd.notna(ic):
                ic_by_day.setdefault(dates[di], []).append(float(ic))
            thr = np.nanquantile(np.abs(f[ok]), 0.9)
            sel = ok & (np.abs(f) >= thr)
            if sel.any():
                net_rows.append({"date": dates[di],
                                 "gross_bps": float(np.mean(np.sign(f[sel]) * realized[sel]) * 1e4)})
    day_ic = pd.Series({d: np.mean(v) for d, v in ic_by_day.items()}).sort_index()
    t_ic = float(day_ic.mean() / day_ic.std(ddof=1) * np.sqrt(len(day_ic)))
    nets = pd.DataFrame(net_rows).groupby("date")["gross_bps"].mean()
    t_net = float(nets.mean() / nets.std(ddof=1) * np.sqrt(len(nets)))
    print(f"KOOPMAN-LIFTED ({args.freq}) days={len(day_ic)} IC={day_ic.mean():.4f} t={t_ic:.2f} | "
          f"top-decile gross {nets.mean():.2f}bps t={t_net:.2f} vs cost {COST_RT_BPS}bps [{time.time()-t0:.0f}s]")
    ref = json.load(open(OUT_DIR / f"markov_minute_{args.freq}.json", encoding="utf-8"))["prediction"]
    print(f"PLAIN CHAIN  ({args.freq}) days={ref['days']} IC={ref['mean_ic']} t={ref['day_t']} | "
          f"top-decile gross {ref['top_decile_gross_bps']}bps")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"koopman_lifted_{args.freq}.json").write_text(json.dumps({
        "freq": args.freq, "days": len(day_ic), "mean_ic": round(float(day_ic.mean()), 4),
        "day_t": round(t_ic, 2), "top_decile_gross_bps": round(float(nets.mean()), 2),
        "gross_t": round(t_net, 2), "plain_chain_ref": ref, "status": "diagnostic_only",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
