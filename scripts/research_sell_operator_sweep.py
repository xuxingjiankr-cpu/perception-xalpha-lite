"""Operator sweep for SELL timing: can any causal operator on 5-min price/volume history
predict the REMAINING-TO-CLOSE return of a held name, so a sell can fire before the drop?

Why this evades the cost wall that killed the entry-side signal lines: our validated sell
framework is hold-to-close by default, so "sell at bar t" vs "sell at close" pays the same
one-way spread either way -- the timing difference is pure. An operator whose danger-decile
bars carry significantly NEGATIVE forward-to-close returns is directly harvestable on the
sell side (unlike entry signals, which must clear a full round-trip).

Data: mootdx 5-min bars, the 18 actually-traded codes (per the population lesson: validate on
the book the sell engine actually holds), 2024-06..2026-07 (~490 days).

17 fixed operators (no fitted parameters -> no train/test contamination; multiple-testing
handled by requiring BOTH half-samples same-sign AND pooled day-clustered |t| >= 3):
  momentum (1/3/6/12 bars), EMA deviation (6/24), log-price Laplacian curvature (3/12),
  rolling vol (12), downside-semivariance share (12), drawdown-from-session-high,
  signed down-run length, volume z (vs trailing day), Amihud illiquidity (12),
  bar-range z (12), amount-signed flow (12), rolling skew (24), VWAP deviation (session).

Per operator: per-day Spearman IC of signal vs remaining-to-close return (pooled across codes
within the day; one IC per day; day-clustered t across ~490 days, reported per half), plus the
economic read: mean remaining-to-close return of the within-day WORST-decile bars minus the
day's average (does selling the danger decile early actually save money, in bps).
Bars in the final 30 minutes are excluded (no meaningful remainder). Diagnostic only.

Run: py -3.13 scripts/research_sell_operator_sweep.py
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
SPLIT = "2025-07-01"     # two halves for the same-sign stability requirement
MIN_ROWS_PER_DAY = 60
DANGER_DECILE = 0.10


def load_code(code: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in (BARS_DIR / f"{code}.jsonl").read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].str[:10]
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def operators(df: pd.DataFrame) -> pd.DataFrame:
    """All causal; grouped by date where the operator is session-scoped."""
    g = df.groupby("date", group_keys=False)
    c, h, l, v, a = df["close"], df["high"], df["low"], df["vol"], df["amount"]
    logc = np.log(c)
    r1 = c.pct_change()
    out = pd.DataFrame(index=df.index)
    for k in (1, 3, 6, 12):
        out[f"mom_{k}"] = c / c.shift(k) - 1.0
    for hl in (6, 24):
        out[f"ema_dev_{hl}"] = c / c.ewm(halflife=hl, min_periods=hl).mean() - 1.0
    for s in (3, 12):
        out[f"curv_{s}"] = logc - 2.0 * logc.shift(s) + logc.shift(2 * s)
    out["vol_12"] = r1.rolling(12).std()
    dn = r1.clip(upper=0.0)
    out["semivar_12"] = (dn.pow(2).rolling(12).sum() / r1.pow(2).rolling(12).sum().replace(0, np.nan))
    out["dd_high"] = c / g["high"].cummax() - 1.0
    sign = np.sign(r1).fillna(0.0)
    run = sign.copy()
    run = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    out["down_run"] = np.where(sign < 0, run, 0.0)
    out["vol_z"] = (v - v.rolling(48).mean()) / v.rolling(48).std().replace(0, np.nan)
    out["amihud_12"] = (r1.abs() / a.replace(0, np.nan)).rolling(12).mean()
    rng = (h - l) / c
    out["rng_z"] = (rng - rng.rolling(48).mean()) / rng.rolling(48).std().replace(0, np.nan)
    out["flow_12"] = (sign * a).rolling(12).sum() / a.rolling(12).sum().replace(0, np.nan)
    out["skew_24"] = r1.rolling(24).skew()
    sess_vwap = g.apply(lambda x: (x["amount"].cumsum() / x["vol"].cumsum().replace(0, np.nan))).reset_index(drop=True)
    out["vwap_dev"] = c / sess_vwap.values - 1.0
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    frames = []
    for path in sorted(BARS_DIR.glob("*.jsonl")):
        df = load_code(path.stem)
        ops = operators(df)
        g = df.groupby("date")
        day_close = g["close"].transform("last")
        bar_no = g.cumcount()
        bars_in_day = g["close"].transform("size")
        rem = day_close / df["close"] - 1.0
        keep = bar_no < (bars_in_day - 6)   # exclude final ~30min
        block = ops[keep].copy()
        block["fwd_close"] = rem[keep]
        block["date"] = df["date"][keep]
        frames.append(block)
    data = pd.concat(frames, ignore_index=True)
    op_names = [c for c in data.columns if c not in ("fwd_close", "date")]
    print(f"rows {len(data)} | days {data['date'].nunique()} | operators {len(op_names)} [{time.time()-t0:.0f}s]", flush=True)

    results = []
    for op in op_names:
        sub = data[["date", op, "fwd_close"]].dropna()
        ics, danger = {}, {}
        for d, grp in sub.groupby("date"):
            if len(grp) < MIN_ROWS_PER_DAY:
                continue
            ic = grp[op].corr(grp["fwd_close"], method="spearman")
            if pd.isna(ic):
                continue
            ics[d] = ic
            thr = grp[op].quantile(DANGER_DECILE)
            lo = grp[grp[op] <= thr]["fwd_close"]
            danger[d] = (lo.mean() - grp["fwd_close"].mean()) * 1e4   # bps saved-if-sold-early when negative
        s_ic = pd.Series(ics).sort_index()
        s_dg = pd.Series(danger).sort_index()
        def t_of(x):
            return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 30 and x.std(ddof=1) else np.nan
        h1_ic, h2_ic = s_ic[s_ic.index < SPLIT], s_ic[s_ic.index >= SPLIT]
        results.append({
            "op": op, "days": len(s_ic),
            "ic_mean": round(float(s_ic.mean()), 4), "ic_t": round(t_of(s_ic), 2),
            "ic_t_h1": round(t_of(h1_ic), 2), "ic_t_h2": round(t_of(h2_ic), 2),
            "same_sign": bool(np.sign(h1_ic.mean()) == np.sign(h2_ic.mean())) if len(h1_ic) and len(h2_ic) else False,
            "danger_decile_bps": round(float(s_dg.mean()), 2), "danger_t": round(t_of(s_dg), 2),
        })

    results.sort(key=lambda r: -abs(r["ic_t"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sweep_results.json").write_text(json.dumps(
        {"generated_at": datetime.now().astimezone().isoformat(), "status": "diagnostic_only",
         "split": SPLIT, "results": results}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"{'operator':<14}{'IC':>8}{'t':>7}{'t_H1':>7}{'t_H2':>7}{'sign=':>6}{'danger bps':>11}{'t':>7}")
    for r in results:
        print(f"{r['op']:<14}{r['ic_mean']:>8.4f}{r['ic_t']:>7.2f}{r['ic_t_h1']:>7.2f}{r['ic_t_h2']:>7.2f}"
              f"{str(r['same_sign']):>6}{r['danger_decile_bps']:>11.2f}{r['danger_t']:>7.2f}")
    passers = [r for r in results if abs(r["ic_t"]) >= 3 and r["same_sign"] and abs(r["ic_t_h1"]) >= 1.5 and abs(r["ic_t_h2"]) >= 1.5]
    print(f"\npass rule (|t|>=3 pooled, same sign, |t|>=1.5 each half): {[r['op'] for r in passers] or 'NONE'}")
    print(f"done [{time.time()-t0:.0f}s] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
