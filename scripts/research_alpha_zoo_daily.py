"""Alpha-zoo scan: 456 vendored formulaic factors (Kakushadze 101 / GTJA 191 / Qlib 158 /
academic 10, from HKUDS/Vibe-Trading, MIT) run through OUR validation pipeline on the full
non-money A-share ETF universe's mootdx daily bars (~1500 codes, back to 2019 where listed).

Pipeline (preregistered before first run):
  1. Panel: daily OHLCV+amount -> vwap, returns. Tradability filter: >=250 obs and median
     daily amount >= 30M CNY. All factors computed with the vendored lookahead-banned ops.
  2. Signal timing: factor value from day t's close data; earliest execution is day t+1, so
     the graded IC is signal(t) vs close-to-close return t+1 -> t+2 (implementation-lagged).
  3. Per-factor stats: daily cross-sectional Spearman IC -> day-clustered mean / t / ICIR;
     long-short decile daily return series (edge existence, gross); LONG-ONLY top-decile
     minus equal-weight-universe net of turnover x 15.5bp round-trip (harvestability -- A-share
     retail cannot short).
  4. Selection discipline: factors are ranked by TRAIN-window ICIR only (train <= 2024-12-31);
     the report shows TEST-window (2025-01-01..) stats for the train-selected top list. PBO
     (combinatorial_symmetric_pbo) across the FULL zoo's daily L-S matrix and a Deflated
     Sharpe note with n_trials=456 quantify the zoo-level multiple-testing burden.
  5. Nothing here touches live config/sizing/execution. A factor only graduates to the next
     step (replay integration / forward shadow) if its TEST ICIR and net long-only spread
     survive, and then only via the standing evidence gates.

Run: py -3.13 scripts/research_alpha_zoo_daily.py [--zoos alpha101,gtja191,qlib158,academic]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT
import overfitting_guard as og

sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))

BARS_DIR = ROOT / "data" / "market" / "mootdx" / "bars_1d"
OUT_DIR = ROOT / "outputs" / "alpha_zoo"
TRAIN_END = "2024-12-31"          # train <= this < test (18-month OOS)
MIN_OBS = 250
MIN_MEDIAN_AMOUNT = 30_000_000.0
COST_RT = 0.00155                 # 15.5bp round-trip on turned-over notional
DECILE = 0.10
TOP_REPORT = 25


def build_panel() -> dict[str, pd.DataFrame]:
    frames: dict[str, dict[str, pd.Series]] = {k: {} for k in ("open", "high", "low", "close", "volume", "amount")}
    for path in sorted(BARS_DIR.glob("*.jsonl")):
        code = path.stem
        recs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs.append(r)
        if len(recs) < MIN_OBS:
            continue
        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["dt"].str[:10])
        df = df.drop_duplicates("date").set_index("date").sort_index()
        if float(df["amount"].median()) < MIN_MEDIAN_AMOUNT:
            continue
        for k in frames:
            frames[k][code] = df[k if k != "volume" else "vol"]
    panel = {k: pd.DataFrame(v).sort_index() for k, v in frames.items()}
    close = panel["close"]
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    panel["returns"] = close.pct_change()
    return panel


def list_factors(zoos: list[str]) -> list[tuple[str, str]]:
    out = []
    for zoo in zoos:
        zoo_dir = ROOT / "scripts" / "vendor" / "vibe_factors" / "src" / "factors" / "zoo" / zoo
        for f in sorted(zoo_dir.glob("*.py")):
            if f.stem != "__init__":
                out.append((zoo, f.stem))
    return out


def evaluate(sig: pd.DataFrame, fwd: pd.DataFrame, close_valid: pd.DataFrame) -> dict | None:
    """Per-day cross-sectional Spearman IC + decile long-short / long-only daily returns."""
    sig = sig.where(close_valid)
    ranks = sig.rank(axis=1)
    n_valid = ranks.notna().sum(axis=1)
    usable = n_valid >= 30
    if usable.sum() < 120:
        return None
    ic = sig[usable].corrwith(fwd[usable], axis=1, method="spearman")
    # decile portfolios, next-day-lagged forward returns already aligned in fwd
    q_hi = ranks.ge(ranks.quantile(1 - DECILE, axis=1), axis=0) & usable.values[:, None]
    q_lo = ranks.le(ranks.quantile(DECILE, axis=1), axis=0) & usable.values[:, None]
    w_hi = q_hi.div(q_hi.sum(axis=1), axis=0).fillna(0.0)
    ret_hi = (w_hi * fwd).sum(axis=1)
    ret_lo = (q_lo.div(q_lo.sum(axis=1), axis=0).fillna(0.0) * fwd).sum(axis=1)
    ls = (ret_hi - ret_lo)[usable]
    bench = fwd[usable].mean(axis=1)
    turnover = (w_hi.diff().abs().sum(axis=1) / 2.0)[usable]
    long_net = (ret_hi[usable] - bench) - turnover * COST_RT
    return {"ic": ic.dropna(), "ls": ls.dropna(), "long_net": long_net.dropna()}


def stats(series: pd.Series, split: str) -> dict:
    tr = series[series.index <= split]
    te = series[series.index > split]
    def block(s):
        if len(s) < 60:
            return {"n": len(s)}
        t = float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if s.std(ddof=1) else None
        return {"n": len(s), "mean": round(float(s.mean()), 6),
                "t": round(t, 2) if t is not None else None,
                "ir_ann": round(float(s.mean() / s.std(ddof=1) * np.sqrt(244)), 3) if s.std(ddof=1) else None}
    return {"train": block(tr), "test": block(te)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zoos", default="alpha101,gtja191,qlib158,academic")
    args = ap.parse_args()

    t0 = time.time()
    panel = build_panel()
    close = panel["close"]
    print(f"panel: {close.shape[1]} codes x {close.shape[0]} days "
          f"({close.index.min():%Y-%m-%d}..{close.index.max():%Y-%m-%d}) [{time.time()-t0:.0f}s]", flush=True)
    # implementation-lagged forward return: signal(t) -> ret(t+1 close -> t+2 close)
    fwd = close.shift(-2) / close.shift(-1) - 1.0
    close_valid = close.notna() & (close > 0)

    results = []
    ls_matrix: list[list[float]] = []
    common_index = None
    factors = list_factors([z.strip() for z in args.zoos.split(",") if z.strip()])
    print(f"scanning {len(factors)} factors ...", flush=True)
    for i, (zoo, name) in enumerate(factors):
        try:
            mod = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
            sig = mod.compute(panel)
            ev = evaluate(sig, fwd, close_valid)
        except Exception as exc:
            results.append({"zoo": zoo, "name": name, "error": str(exc)[:120]})
            continue
        if ev is None:
            results.append({"zoo": zoo, "name": name, "error": "insufficient_panel"})
            continue
        rec = {"zoo": zoo, "name": name,
               "ic": stats(ev["ic"], TRAIN_END),
               "ls": stats(ev["ls"], TRAIN_END),
               "long_net": stats(ev["long_net"], TRAIN_END)}
        results.append(rec)
        if common_index is None:
            common_index = ev["ls"].index
        ls_matrix.append(ev["ls"].reindex(common_index).fillna(0.0).tolist())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(factors)} [{time.time()-t0:.0f}s]", flush=True)

    ok = [r for r in results if "error" not in r and r["ic"]["train"].get("t") is not None]
    # rank by TRAIN IC-IR only; TEST stats are then honest OOS for the selection
    ok.sort(key=lambda r: -abs(r["ic"]["train"].get("ir_ann") or 0.0))
    pbo = og.combinatorial_symmetric_pbo(ls_matrix, n_blocks=8) if len(ls_matrix) >= 2 else {"pbo": None}
    best_test = max((r for r in ok[:TOP_REPORT]), key=lambda r: (r["long_net"]["test"].get("ir_ann") or -9), default=None)
    dsr = (og.deflated_significance_note(n_trials=len(ok), observed_sharpe=(best_test["long_net"]["test"].get("ir_ann") or 0.0),
                                          n_obs=best_test["long_net"]["test"].get("n") or 1) if best_test else {"note": "n/a"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "zoo_scan_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    L = ["# Alpha-Zoo Daily Scan -- full ETF universe, implementation-lagged, train-ranked/test-reported", "",
         f"panel {close.shape[1]} codes x {close.shape[0]} days | train<= {TRAIN_END} < test | "
         f"cost {COST_RT:.4%} RT x turnover | factors scanned {len(factors)}, usable {len(ok)}", "",
         "Top-25 by |TRAIN IC-IR| (TEST columns are out-of-sample for this ranking):", "",
         "| factor | train IC t | train ICIR | TEST IC t | TEST ICIR | TEST L-S t | TEST long-net IR |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for r in ok[:TOP_REPORT]:
        L.append(f"| {r['zoo']}/{r['name']} | {r['ic']['train'].get('t')} | {r['ic']['train'].get('ir_ann')} | "
                 f"{r['ic']['test'].get('t')} | {r['ic']['test'].get('ir_ann')} | "
                 f"{r['ls']['test'].get('t')} | {r['long_net']['test'].get('ir_ann')} |")
    n_test_sig = sum(1 for r in ok if abs(r["ic"]["test"].get("t") or 0) >= 2.0)
    L += ["",
          f"- factors with |TEST IC t| >= 2: {n_test_sig} / {len(ok)} (chance at 5%: ~{round(0.05*len(ok))})",
          f"- **zoo-level PBO** (L-S daily matrix): {pbo.get('pbo')}",
          f"- **DSR note (best test long-net among train-top25, n_trials={len(ok)})**: {dsr}", "",
          "## Read", "",
          "A factor graduates ONLY if: TEST IC |t|>=2 AND TEST long-only net IR meaningfully >0 AND it "
          "survives the zoo-level multiple-testing discount (DSR) -- then replay integration + forward "
          "shadow, per the standing gates. Diagnostic only; no live change from this scan.", ""]
    (OUT_DIR / "zoo_scan_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"done in {time.time()-t0:.0f}s | outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
