"""Factor-ensemble accuracy study: can COMBINING the zoo's weak-but-real signals raise
out-of-sample prediction accuracy where single factors (zoo scan) and representation changes
(Laplace study) could not? Grinold: IR ~ IC x sqrt(breadth) -- breadth is the one lever not
yet used: 116/454 factors carry |TEST IC t|>=2 individually.

Protocol (preregistered):
  Pass A (cache): every zoo factor -> per-day cross-sectional rank panel (float16 npz,
    outputs/factor_cache/) + per-day Spearman IC series vs the implementation-lagged target
    ret(t+1 close -> t+2 close). Pure function of trailing data.
  Pass B (walk-forward ensemble): every 21 days, select top-K factors by |trailing 500-day
    ICIR| and take sign(trailing ICIR); ensemble score = weighted mean of sign-adjusted rank
    panels (weights: equal or |ICIR|). Selection, signs and weights use ONLY the trailing
    window. Variants: K in {10, 30, 60} x {equal, icir} = 6 configs (small grid, reported in
    full -- no cherry-pick).
  Judgment (fixed before running): a config "improves accuracy" only if OOS (>= 2023-01-01)
    daily IC day-clustered t >= 2.5 AND clearly above the MOM baseline harness, AND long-only
    top-decile net of 15.5bp x turnover has IR > 0 with a majority of calendar years positive.
    Otherwise verdict: breadth does not rescue accuracy on this universe.

Diagnostic only. Run: py -3.13 scripts/research_factor_ensemble.py [--rebuild-cache]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT
from research_alpha_zoo_daily import build_panel, list_factors, COST_RT, DECILE

sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))

CACHE = ROOT / "outputs" / "factor_cache"
OUT_DIR = ROOT / "outputs" / "factor_ensemble"
TEST_START = "2023-01-01"
REFIT_EVERY = 21
TRAIL = 500
CONFIGS = [(k, w) for k in (10, 30, 60) for w in ("equal", "icir")]


def build_cache(panel: dict, fwd: pd.DataFrame, valid: pd.DataFrame) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    close = panel["close"]
    ic_rows = {}
    factors = list_factors(["alpha101", "gtja191", "qlib158", "academic"])
    t0 = time.time()
    for i, (zoo, name) in enumerate(factors):
        key = f"{zoo}__{name}"
        f_npz = CACHE / f"{key}.npz"
        try:
            sig = importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(panel).where(valid)
            ranks = sig.rank(axis=1, pct=True)   # 0..1 cross-sectional rank
            n_ok = ranks.notna().sum(axis=1)
            usable = n_ok >= 30
            ic = sig[usable].corrwith(fwd[usable], axis=1, method="spearman")
            np.savez_compressed(f_npz, r=ranks.to_numpy(dtype=np.float16))
            ic_rows[key] = ic
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  cache {i+1}/{len(factors)} [{time.time()-t0:.0f}s]", flush=True)
    ics = pd.DataFrame(ic_rows, index=close.index)
    ics.to_pickle(CACHE / "daily_ic.pkl")
    meta = {"index": [str(d.date()) for d in close.index], "columns": list(close.columns)}
    (CACHE / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def block(s: pd.Series, start: str = TEST_START) -> dict:
    s = s[s.index >= start].dropna()
    if len(s) < 60:
        return {"n": len(s)}
    t = float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if s.std(ddof=1) else None
    return {"n": len(s), "mean": round(float(s.mean()), 5), "t": round(t, 2) if t else None,
            "ir_ann": round(float(s.mean() / s.std(ddof=1) * np.sqrt(244)), 3) if s.std(ddof=1) else None}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    panel = build_panel()
    close = panel["close"]
    valid = close.notna() & (close > 0)
    fwd = close.shift(-2) / close.shift(-1) - 1.0
    print(f"panel {close.shape[1]} x {close.shape[0]} [{time.time()-t0:.0f}s]", flush=True)

    if args.rebuild_cache or not (CACHE / "daily_ic.pkl").exists():
        print("building factor cache ...", flush=True)
        build_cache(panel, fwd, valid)

    ics: pd.DataFrame = pd.read_pickle(CACHE / "daily_ic.pkl")
    keys = list(ics.columns)
    dates = close.index
    print(f"cached factors: {len(keys)}", flush=True)

    # walk-forward selection schedule
    sched = {}   # refit date index -> list[(key, sign, weight)]
    for i in range(TRAIL, len(dates), REFIT_EVERY):
        window = ics.iloc[max(0, i - TRAIL):i]
        mu, sd = window.mean(), window.std(ddof=1)
        icir = (mu / sd.replace(0, np.nan)).dropna()
        sched[i] = icir
    print(f"refit points: {len(sched)}", flush=True)

    # load rank panels lazily with a small LRU (union of selected keys is modest)
    loaded: dict[str, np.ndarray] = {}
    def get_ranks(key: str) -> np.ndarray:
        if key not in loaded:
            loaded[key] = np.load(CACHE / f"{key}.npz")["r"].astype(np.float32)
        return loaded[key]

    results = {}
    ens_ic_series = {}
    for K, wmode in CONFIGS:
        preds = np.full(close.shape, np.nan, dtype=np.float32)
        refit_is = sorted(sched)
        for ri, i0 in enumerate(refit_is):
            icir = sched[i0]
            top = icir.abs().sort_values(ascending=False).head(K)
            i1 = refit_is[ri + 1] if ri + 1 < len(refit_is) else len(dates)
            acc = np.zeros((i1 - i0, close.shape[1]), dtype=np.float32)
            wsum = 0.0
            for key in top.index:
                sign = 1.0 if icir[key] > 0 else -1.0
                w = 1.0 if wmode == "equal" else float(abs(icir[key]))
                r = get_ranks(key)[i0:i1]
                r = np.where(np.isfinite(r), r, 0.5)   # neutral for missing
                acc += w * sign * r
                wsum += w
            preds[i0:i1] = acc / max(wsum, 1e-9)
        P = pd.DataFrame(preds, index=dates, columns=close.columns).where(valid)
        both_ok = (P.notna() & fwd.notna()).sum(axis=1) >= 30
        ic = P[both_ok].corrwith(fwd[both_ok], axis=1, method="spearman").dropna()
        ranks = P.rank(axis=1)
        hi = ranks.ge(ranks.quantile(1 - DECILE, axis=1), axis=0)
        w_ = hi.div(hi.sum(axis=1), axis=0).fillna(0.0)
        net = ((w_ * fwd).sum(axis=1) - fwd[valid].mean(axis=1) - (w_.diff().abs().sum(axis=1) / 2.0) * COST_RT)
        yearly = {str(y): round(float(v.mean() * 244 * 1e4), 1)
                  for y, v in net[net.index >= TEST_START].groupby(net[net.index >= TEST_START].index.year)}
        name = f"K{K}_{wmode}"
        results[name] = {"oos_ic": block(ic), "long_net": block(net), "yearly_net_bps_ann": yearly}
        ens_ic_series[name] = ic
        print(f"{name:<12} IC {results[name]['oos_ic']} | net {results[name]['long_net']} | yearly {yearly}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ensemble_report.json").write_text(json.dumps(
        {"generated_at": datetime.now().astimezone().isoformat(), "status": "diagnostic_only",
         "test_start": TEST_START, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done [{time.time()-t0:.0f}s] -> {OUT_DIR / 'ensemble_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
