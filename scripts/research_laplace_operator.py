"""Laplace-operator feature study (distilled from ARTEMIS's Laplace Neural Operator encoder,
arXiv 2603.18107) -- does a Laplace-domain representation of price history add PREDICTIVE
INCREMENT over plain momentum features, on the full ETF daily panel, through our gates?

Distillation rationale: the LNO encoder's core object is a pole-residue (exponential-decay
basis) representation of the input signal. The honest cheap version is (a) LAPLACE BASIS:
projections of past returns onto exponential kernels exp(-tau/h) across a half-life grid
(multi-scale EMA coefficients = real-pole residues), and (b) DISCRETE LAPLACIAN: second
difference of log price at several scales (curvature). A full neural LNO would mostly test
optimization noise on our panel sizes; if the basis carries no linear increment here, wrapping
it in a neural operator will not conjure one (and ARTEMIS's own real-market results -- RankIC
0.04 Jane Street, -0.06 Optiver -- do not suggest otherwise).

Protocol (same discipline as the alpha-zoo scan):
  - Panel: mootdx daily bars, tradable filter (>=250 obs, median amount >= 30M CNY).
  - Implementation-lagged target: ret(t+1 close -> t+2 close).
  - Walk-forward ridge per model, refit every 21 days on a trailing 750-day window, predicting
    the cross-section one day at a time. No test-period information touches any fit.
  - Models: MOM (5/21/63d returns), LAPLACE (5 half-life exponential projections + 2-scale
    log-price Laplacian), BOTH. The judged quantity is OOS day-clustered IC and, decisively,
    the INCREMENT of BOTH over MOM (paired per-day IC difference, day-clustered t) plus
    long-only top-decile net spread at 15.5bp x turnover.
Diagnostic only; nothing here can gate/size/execute.

Run: py -3.13 scripts/research_laplace_operator.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT
from research_alpha_zoo_daily import build_panel, COST_RT, DECILE

OUT_DIR = ROOT / "outputs" / "laplace_operator"
HALF_LIVES = [2, 5, 10, 21, 63]
LAP_SCALES = [5, 21]
LOOKBACK = 130          # feature window (>= 2*63)
TRAIN_WIN = 750         # trailing fit window (days)
REFIT_EVERY = 21
RIDGE_L2 = 10.0
TEST_START = "2023-01-01"   # walk-forward predictions evaluated from here (fits use trailing data only)


def laplace_features(ret: pd.DataFrame, logc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cross-sectionally comparable Laplace-domain features, all causal."""
    feats: dict[str, pd.DataFrame] = {}
    for h in HALF_LIVES:
        # exponential projection of returns = real-pole residue at decay ln2/h (EWM mean)
        feats[f"lap_h{h}"] = ret.ewm(halflife=h, min_periods=max(3, h)).mean()
    for s in LAP_SCALES:
        # discrete Laplacian of log price at scale s: curvature = x_t - 2 x_{t-s} + x_{t-2s}
        feats[f"lapl_s{s}"] = logc - 2.0 * logc.shift(s) + logc.shift(2 * s)
    return feats


def momentum_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f"mom_{w}": close / close.shift(w) - 1.0 for w in (5, 21, 63)}


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-5, 5)


def walk_forward_ic(feats: dict[str, pd.DataFrame], fwd: pd.DataFrame, valid: pd.DataFrame):
    """Rolling-ridge cross-sectional forecaster; returns per-day OOS predictions (wide)."""
    names = sorted(feats)
    Z = {k: zscore_xs(feats[k].where(valid)) for k in names}
    dates = fwd.index
    preds = pd.DataFrame(np.nan, index=dates, columns=fwd.columns)
    beta = None
    start_i = max(LOOKBACK, 60)
    for i in range(start_i, len(dates) - 2):
        d = dates[i]
        if beta is None or (i - start_i) % REFIT_EVERY == 0:
            lo = max(0, i - TRAIN_WIN)
            X_rows, y_rows = [], []
            for j in range(lo, i - 2):          # -2: target uses t+1..t+2, keep it strictly past
                dj = dates[j]
                xs = np.column_stack([Z[k].loc[dj].values for k in names])
                y = fwd.loc[dj].values
                m = np.isfinite(xs).all(axis=1) & np.isfinite(y)
                if m.sum() >= 30:
                    X_rows.append(xs[m]); y_rows.append(y[m])
            if not X_rows:
                continue
            X = np.vstack(X_rows); y = np.concatenate(y_rows)
            A = X.T @ X + RIDGE_L2 * len(y) / 1000.0 * np.eye(len(names))
            beta = np.linalg.solve(A, X.T @ y)
        xs = np.column_stack([Z[k].loc[d].values for k in names])
        m = np.isfinite(xs).all(axis=1)
        row = np.full(xs.shape[0], np.nan)
        row[m] = xs[m] @ beta
        preds.loc[d] = row
    return preds


def daily_ic(preds: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    both = preds.notna() & fwd.notna()
    ok = both.sum(axis=1) >= 30
    return preds[ok].corrwith(fwd[ok], axis=1, method="spearman").dropna()


def long_net(preds: pd.DataFrame, fwd: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    ranks = preds.rank(axis=1)
    hi = ranks.ge(ranks.quantile(1 - DECILE, axis=1), axis=0)
    w = hi.div(hi.sum(axis=1), axis=0).fillna(0.0)
    ret = (w * fwd).sum(axis=1)
    bench = fwd[valid].mean(axis=1)
    turnover = w.diff().abs().sum(axis=1) / 2.0
    return (ret - bench - turnover * COST_RT).dropna()


def block(s: pd.Series) -> dict:
    s = s[s.index >= TEST_START]
    if len(s) < 60:
        return {"n": len(s)}
    t = float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if s.std(ddof=1) else None
    return {"n": len(s), "mean": round(float(s.mean()), 6), "t": round(t, 2) if t else None,
            "ir_ann": round(float(s.mean() / s.std(ddof=1) * np.sqrt(244)), 3) if s.std(ddof=1) else None}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    panel = build_panel()
    close = panel["close"]
    valid = close.notna() & (close > 0)
    ret = close.pct_change(fill_method=None)
    logc = np.log(close.where(close > 0))
    fwd = close.shift(-2) / close.shift(-1) - 1.0
    print(f"panel {close.shape[1]} x {close.shape[0]} [{time.time()-t0:.0f}s]", flush=True)

    models = {
        "MOM": momentum_features(close),
        "LAPLACE": laplace_features(ret, logc),
    }
    models["BOTH"] = {**models["MOM"], **models["LAPLACE"]}

    out = {"generated_at": datetime.now().astimezone().isoformat(), "status": "diagnostic_only",
           "test_start": TEST_START, "models": {}}
    ics = {}
    for name, feats in models.items():
        preds = walk_forward_ic(feats, fwd, valid)
        ic = daily_ic(preds, fwd)
        ics[name] = ic
        net = long_net(preds[preds.index >= TEST_START], fwd, valid)
        out["models"][name] = {"oos_ic": block(ic), "long_net": block(net)}
        print(f"{name:<8} OOS IC {out['models'][name]['oos_ic']} | long-net {out['models'][name]['long_net']}", flush=True)

    inc = (ics["BOTH"] - ics["MOM"]).dropna()
    out["increment_BOTH_minus_MOM"] = block(inc)
    print(f"INCREMENT (BOTH-MOM paired daily IC): {out['increment_BOTH_minus_MOM']}")
    lap_only_inc = (ics["LAPLACE"] - ics["MOM"]).dropna()
    out["laplace_vs_mom"] = block(lap_only_inc)
    print(f"LAPLACE vs MOM (paired daily IC diff): {out['laplace_vs_mom']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "laplace_operator_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done [{time.time()-t0:.0f}s] -> {OUT_DIR / 'laplace_operator_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
