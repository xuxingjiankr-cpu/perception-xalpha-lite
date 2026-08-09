"""Run the full 456-factor zoo at the shortest executable A-share holding period.

Extends research_nextday_holding_factors.py from 18 hand-built candidates to the vendored
library (Kakushadze 101, GTJA 191, Qlib 158, academic 10) on the same PIT-adjusted panel and
the same accounting: buy open[t+1], sell open[t+2], point-in-time membership, ST and
trade-status exclusion, sealed-limit legs dropped, 30bps round trip charged on realised
turnover, train only ranks and every reported number comes from the 2025+ test window.

The hand-built scan found gross top-decile excess topping out near 3 bps/day against a cost
of 0.8-25.5 bps/day, and outcomes ordered by turnover rather than by signal strength. This
run asks whether a much larger and more varied factor library changes that, and reports the
diagnostic that actually matters at a one-day hold: gross excess per unit of turnover, i.e.
how much a factor earns for each unit of cost it must pay.

Diagnostic only. No orders, no promotion, no config.

Run: py -3.13 scripts/research_nextday_zoo_scan.py [--symbols N]
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
from research_nextday_holding_factors import (
    COST_RT,
    DECILE,
    TRAIN_END,
    eligibility,
    labels,
    load_panel,
)

sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))
OUT_DIR = ROOT / "outputs" / "nextday_holding_factors"
ZOOS = ("alpha101", "gtja191", "qlib158", "academic")


def zoo_factors() -> list[tuple[str, str]]:
    out = []
    for zoo in ZOOS:
        directory = ROOT / "scripts" / "vendor" / "vibe_factors" / "src" / "factors" / "zoo" / zoo
        for path in sorted(directory.glob("*.py")):
            if path.stem != "__init__":
                out.append((zoo, path.stem))
    return out


def score(signal: pd.DataFrame, label: pd.DataFrame, eligible: pd.DataFrame) -> dict | None:
    signal = signal.where(eligible).replace([np.inf, -np.inf], np.nan)
    usable = (signal.notna() & label.notna()).sum(axis=1) >= 50
    if int(usable.sum()) < 200:
        return None
    ranks = signal.rank(axis=1, pct=True)
    top = ranks.ge(1.0 - DECILE) & usable.values[:, None]
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book = (weights * label).sum(axis=1, min_count=1)
    bench = label.where(eligible)[usable].mean(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    gross = (book - bench)[usable]
    net = (book - bench - turnover * COST_RT)[usable]
    ic = signal[usable].corrwith(label[usable], axis=1, method="spearman").dropna()

    def window(series: pd.Series, test: bool) -> pd.Series:
        return series[series.index > TRAIN_END] if test else series[series.index <= TRAIN_END]

    def block(series: pd.Series) -> dict:
        part = series.dropna()
        if len(part) < 60:
            return {"n": len(part)}
        equity = (1.0 + part).cumprod()
        std = part.std(ddof=1)
        return {
            "n": len(part),
            "mean_bps": round(float(part.mean()) * 1e4, 3),
            "ir_ann": round(float(part.mean() / std * np.sqrt(244)), 3) if std else None,
            "max_drawdown_pct": round(float((equity / equity.cummax() - 1.0).min()) * 100, 2),
            "win_rate": round(float((part > 0).mean()), 3),
        }

    ic_test = window(ic, True)
    ic_std = ic_test.std(ddof=1) if len(ic_test) > 2 else None
    turnover_test = window(turnover[usable], True).mean()
    gross_test = block(window(gross, True))
    return {
        "ic_test_t": round(float(ic_test.mean() / ic_std * np.sqrt(len(ic_test))), 2)
        if ic_std else None,
        "turnover_per_day": round(float(turnover_test), 4),
        "gross_test": gross_test,
        "net_test": block(window(net, True)),
        "gross_train_bps": block(window(gross, False)).get("mean_bps"),
        # how much gross excess the factor earns per unit of cost it is forced to pay
        "gross_per_cost": round(
            float(gross_test.get("mean_bps", 0.0)) / max(float(turnover_test) * COST_RT * 1e4, 1e-9), 3
        ) if gross_test.get("mean_bps") is not None else None,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=1200)
    args = parser.parse_args()
    started = time.time()

    panel = load_panel(args.symbols)
    eligible = eligibility(panel)
    label = labels(panel, eligible)
    close = panel["close"]
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    panel["returns"] = close.pct_change(fill_method=None)
    print(f"panel {close.shape[1]} x {close.shape[0]} | labelled {int(label.notna().sum().sum()):,} "
          f"[{time.time()-started:.0f}s]", flush=True)

    factors = zoo_factors()
    print(f"scanning {len(factors)} zoo factors ...", flush=True)
    results: dict[str, dict] = {}
    for index, (zoo, name) in enumerate(factors):
        try:
            module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
            stats = score(module.compute(panel), label, eligible)
        except Exception:
            continue
        if stats:
            results[f"{zoo}/{name}"] = stats
        if (index + 1) % 100 == 0:
            print(f"  {index+1}/{len(factors)} [{time.time()-started:.0f}s]", flush=True)

    ranked = sorted(
        (item for item in results.items() if item[1]["net_test"].get("mean_bps") is not None),
        key=lambda item: -item[1]["net_test"]["mean_bps"],
    )
    positive = [item for item in ranked if item[1]["net_test"]["mean_bps"] > 0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "nextday_zoo_scan.json").write_text(json.dumps({
        "generatedAt": datetime.now().astimezone().isoformat(),
        "status": "diagnostic_only_research_only_not_trading",
        "holding": "buy open[t+1], sell open[t+2]",
        "roundTripCost": COST_RT, "trainEnd": TRAIN_END,
        "scanned": len(factors), "usable": len(results),
        "netPositiveCount": len(positive),
        "factors": results, "orders": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nusable {len(results)} / scanned {len(factors)} | NET-POSITIVE: {len(positive)}")
    print(f"\n{'factor':<28}{'ICt':>6}{'gross':>8}{'turn':>7}{'NET bps':>9}{'IR':>8}{'maxDD%':>9}{'g/cost':>8}")
    for name, stats in ranked[:15]:
        g, n = stats["gross_test"], stats["net_test"]
        print(f"{name:<28}{str(stats['ic_test_t']):>6}{str(g.get('mean_bps')):>8}"
              f"{stats['turnover_per_day']:>7.3f}{n.get('mean_bps'):>9}{str(n.get('ir_ann')):>8}"
              f"{str(n.get('max_drawdown_pct')):>9}{str(stats['gross_per_cost']):>8}")
    print(f"\ndone [{time.time()-started:.0f}s] -> {OUT_DIR / 'nextday_zoo_scan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
