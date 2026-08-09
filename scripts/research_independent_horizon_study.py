"""Independent (Claude-line) walk-forward study of net-of-cost factor returns by horizon.

Protocol is fixed in docs/independent_net_factor_study_preregistration.md and must not be
changed after seeing results. Summary of the design that matters:

  * The object of study is the HOLDING HORIZON, not the factor. Factor choice is made
    mechanically inside each training window (top K by trailing net-of-cost IR, equal
    weights), so a positive result cannot be a weight-fitting or factor-picking artifact.
  * Selection is strictly walk-forward: at each annual rebalance only trailing data is read,
    and the following year is scored out of sample. The concatenated forward segments are
    the track record.
  * Cost is 30 bps round trip charged on realised turnover at every horizon. Lengthening the
    horizon lowers turnover, not the rate -- that is precisely the mechanism under test.

Deliberately separate from the Codex perception/XAlpha line: separate scripts, separate
output namespace, no shared state, registry or config.

Diagnostic only. No orders, no promotion, no trading configuration.

Run: py -3.13 scripts/research_independent_horizon_study.py [--symbols N]
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
from research_nextday_holding_factors import eligibility, load_panel
import overfitting_guard as og

sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))

OUT_DIR = ROOT / "outputs" / "independent_research"
HORIZONS = (1, 5, 10, 20)
COST_RT = 0.003
DECILE = 0.10
TOP_K = 10
FIRST_FORWARD_YEAR = 2021          # needs >= 2 years of trailing history to rank factors
ZOOS = ("alpha101", "gtja191", "qlib158", "academic")
# preregistered viability thresholds
MIN_IR = 0.5
MAX_DRAWDOWN = -0.20


def zoo_factors() -> list[tuple[str, str]]:
    out = []
    for zoo in ZOOS:
        directory = ROOT / "scripts" / "vendor" / "vibe_factors" / "src" / "factors" / "zoo" / zoo
        out += [(zoo, p.stem) for p in sorted(directory.glob("*.py")) if p.stem != "__init__"]
    return out


def horizon_label(panel: dict[str, pd.DataFrame], eligible: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One unit of the strategy: buy open[t+1], sell open[t+1+h], T+1-compatible for h >= 1."""
    open_, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
    forward = open_.shift(-(horizon + 1)) / open_.shift(-1) - 1.0
    sealed = high.eq(low) & high.notna()
    previous = close.shift(1)
    buyable = ~(sealed & close.gt(previous))
    sellable = ~(sealed & close.lt(previous))
    return forward.where(buyable.shift(-1) & sellable.shift(-(horizon + 1)) & eligible)


def daily_net(signal: pd.DataFrame, label: pd.DataFrame, eligible: pd.DataFrame,
              horizon: int) -> pd.Series | None:
    """Net-of-cost daily excess for an equal-weight top-decile book held `horizon` days."""
    signal = signal.where(eligible).replace([np.inf, -np.inf], np.nan)
    usable = (signal.notna() & label.notna()).sum(axis=1) >= 50
    if int(usable.sum()) < 250:
        return None
    ranks = signal.rank(axis=1, pct=True)
    top = ranks.ge(1.0 - DECILE) & usable.values[:, None]
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if horizon > 1:                      # overlapping tranches: 1/h of the book turns daily
        weights = weights.rolling(horizon).mean().fillna(0.0)
        weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    per_day = label / horizon            # express an h-day label as a daily rate
    book = (weights * per_day).sum(axis=1, min_count=1)
    bench = per_day.where(eligible)[usable].mean(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    return (book - bench - turnover * COST_RT)[usable].dropna()


def stats(series: pd.Series) -> dict:
    if len(series) < 60:
        return {"n": len(series)}
    equity = (1.0 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "n": len(series),
        "mean_bps": round(float(series.mean()) * 1e4, 3),
        "ir_ann": round(float(series.mean() / std * np.sqrt(244)), 3) if std else None,
        "max_drawdown_pct": round(float((equity / equity.cummax() - 1.0).min()) * 100, 2),
        "win_rate": round(float((series > 0).mean()), 3),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=1200)
    args = parser.parse_args()
    started = time.time()

    panel = load_panel(args.symbols)
    eligible = eligibility(panel)
    close = panel["close"]
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    panel["returns"] = close.pct_change(fill_method=None)
    print(f"panel {close.shape[1]} x {close.shape[0]} "
          f"({close.index.min():%Y-%m-%d}..{close.index.max():%Y-%m-%d}) [{time.time()-started:.0f}s]",
          flush=True)

    factors = zoo_factors()
    signals: dict[str, pd.DataFrame] = {}
    for zoo, name in factors:
        try:
            signals[f"{zoo}/{name}"] = importlib.import_module(
                f"src.factors.zoo.{zoo}.{name}"
            ).compute(panel)
        except Exception:
            continue
    print(f"computed {len(signals)} / {len(factors)} factors [{time.time()-started:.0f}s]", flush=True)

    years = sorted({d.year for d in close.index if d.year >= FIRST_FORWARD_YEAR})
    report: dict[str, dict] = {}
    for horizon in HORIZONS:
        label = horizon_label(panel, eligible, horizon)
        series: dict[str, pd.Series] = {}
        for name, signal in signals.items():
            net = daily_net(signal, label, eligible, horizon)
            if net is not None:
                series[name] = net
        forward_parts: list[pd.Series] = []
        picks_by_year: dict[str, list[str]] = {}
        for year in years:
            train_end = pd.Timestamp(f"{year}-01-01")
            ranked = []
            for name, net in series.items():
                trailing = net[net.index < train_end]
                if len(trailing) < 250:
                    continue
                std = trailing.std(ddof=1)
                if std:
                    ranked.append((float(trailing.mean() / std * np.sqrt(244)), name))
            if len(ranked) < TOP_K:
                continue
            ranked.sort(reverse=True)
            chosen = [name for _ir, name in ranked[:TOP_K]]
            picks_by_year[str(year)] = chosen
            window = pd.concat(
                [series[name][(series[name].index >= train_end)
                              & (series[name].index < pd.Timestamp(f"{year + 1}-01-01"))]
                 for name in chosen], axis=1
            ).mean(axis=1).dropna()
            if len(window):
                forward_parts.append(window)
        if not forward_parts:
            continue
        track = pd.concat(forward_parts).sort_index()
        block = stats(track)
        dsr = og.deflated_significance_note(
            n_trials=len(series), observed_sharpe=block.get("ir_ann") or 0.0,
            n_obs=block.get("n") or 1,
        )
        passes = (
            (block.get("mean_bps") or -1) > 0
            and (block.get("ir_ann") or -1) >= MIN_IR
            and (block.get("max_drawdown_pct") or -99) >= MAX_DRAWDOWN * 100
            and str(dsr.get("flag")) != "consistent_with_luck"
        )
        report[f"h{horizon}"] = {
            "walk_forward_oos": block, "dsr": dsr, "factorsScored": len(series),
            "picksByYear": picks_by_year, "viablePerPreregisteredRule": bool(passes),
        }
        print(f"  h={horizon:<3} OOS net {str(block.get('mean_bps')):>8}bps "
              f"IR={str(block.get('ir_ann')):>7} maxDD={str(block.get('max_drawdown_pct')):>8}% "
              f"win={block.get('win_rate')} | DSR {dsr.get('flag')} | VIABLE={passes}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "horizon_study.json").write_text(json.dumps({
        "generatedAt": datetime.now().astimezone().isoformat(),
        "line": "claude_independent", "status": "diagnostic_only_research_only_not_trading",
        "preregistration": "docs/independent_net_factor_study_preregistration.md",
        "roundTripCost": COST_RT, "topK": TOP_K, "selection": "trailing_net_ir_only_equal_weight",
        "horizons": report, "orders": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    viable = [k for k, v in report.items() if v["viablePerPreregisteredRule"]]
    print(f"\nVIABLE HORIZONS: {viable or 'NONE'}")
    print(f"done [{time.time()-started:.0f}s] -> {OUT_DIR / 'horizon_study.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
