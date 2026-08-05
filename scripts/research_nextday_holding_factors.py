"""Search for factors that pay over the SHORTEST executable A-share holding period.

The question this answers: buy on the next open, sell on the open after (the minimum a T+1
account can do), and ask which signal maximises that return while keeping drawdown low.

Discipline carried over from the audited pipeline, because a one-day horizon is the setting
where sloppy accounting flatters a strategy most:
  * PIT-adjusted BaoStock panel: backward-adjusted prices (corporate actions cannot be read
    as reversals), point-in-time ST and trade-status flags, delisted names retained.
  * Point-in-time membership: trailing-60-session median amount (shifted), seasoning, the
    bar actually traded, not ST, status normal. No whole-history filter decides membership.
  * Tradability: a sealed bar (high == low) is a locked limit on every board, so a
    sealed-up entry cannot be bought and a sealed-down exit cannot be sold. Those legs are
    dropped rather than priced.
  * Cost: 30 bps round trip (commission 6 + stamp duty 5 + spread), charged on realised
    turnover. At a one-day hold turnover is near 1.0, so the cost hurdle is roughly 30 bps
    PER DAY -- the central fact this study must confront rather than hide.
  * Labels are one-day and therefore non-overlapping: no HAC correction is needed here,
    unlike the 10-day pipeline.
  * Train <= 2024-12-31 decides nothing except ranking; every number reported for the
    selected factors is from the untouched 2025+ test window.

Reports per factor: rank IC and day-clustered t, top-decile gross and net excess, annualised
IR, max drawdown of the daily net series, and daily win rate -- criteria 1 (gain), 2
(drawdown) and 3 (next-day exit) respectively.

Diagnostic only. Nothing here can size, gate or place an order.

Run: py -3.13 scripts/research_nextday_holding_factors.py [--symbols N]
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

BARS = ROOT / "data" / "market" / "ashare_research" / "baostock_pit_adjusted" / "bars_1d_backward_adjusted"
OUT_DIR = ROOT / "outputs" / "nextday_holding_factors"
TRAIN_END = "2024-12-31"
COST_RT = 0.003
DECILE = 0.10
MIN_AMOUNT = 3e7
SEASONING = 120
AMOUNT_WINDOW = 60


def load_panel(limit: int | None) -> dict[str, pd.DataFrame]:
    files = sorted(BARS.glob("*.jsonl"))
    if limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    fields = {k: {} for k in ("open", "high", "low", "close", "volume", "amount", "is_st", "status")}
    for path in files:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if len(rows) < 300:
            continue
        frame = pd.DataFrame(rows)
        if not {"dt", "open", "high", "low", "close", "vol", "amount"}.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["dt"].astype(str).str[:10], errors="coerce")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        code = path.stem
        for key, column in (("open", "open"), ("high", "high"), ("low", "low"),
                             ("close", "close"), ("volume", "vol"), ("amount", "amount")):
            fields[key][code] = pd.to_numeric(frame[column], errors="coerce")
        fields["is_st"][code] = pd.to_numeric(frame.get("isST", 0), errors="coerce")
        fields["status"][code] = pd.to_numeric(frame.get("tradeStatus", 1), errors="coerce")
    return {k: pd.DataFrame(v).sort_index() for k, v in fields.items()}


def eligibility(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close, amount = panel["close"], panel["amount"]
    observed = close.notna() & close.gt(0)
    trailing = amount.rolling(AMOUNT_WINDOW, min_periods=20).median().shift(1)
    seasoned = observed.cumsum().shift(1).ge(SEASONING)
    traded = panel["volume"].fillna(0).gt(0) & amount.fillna(0).gt(0)
    not_st = panel["is_st"].fillna(0).eq(0)
    normal = panel["status"].fillna(1).eq(1)
    return (trailing.ge(MIN_AMOUNT) & seasoned & observed & traded & not_st & normal).fillna(False)


def labels(panel: dict[str, pd.DataFrame], eligible: pd.DataFrame) -> pd.DataFrame:
    """Buy open[t+1], sell open[t+2] -- the shortest hold a T+1 account can execute."""
    open_, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
    forward = open_.shift(-2) / open_.shift(-1) - 1.0
    sealed = high.eq(low) & high.notna()
    previous = close.shift(1)
    buyable = ~(sealed & close.gt(previous))
    sellable = ~(sealed & close.lt(previous))
    return forward.where(buyable.shift(-1) & sellable.shift(-2) & eligible)


def build_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close, open_, high, low = panel["close"], panel["open"], panel["high"], panel["low"]
    amount, volume = panel["amount"], panel["volume"]
    returns = close.pct_change(fill_method=None)
    rng = (high - low) / close.replace(0, np.nan)
    factors: dict[str, pd.DataFrame] = {}
    for window in (1, 3, 5, 10, 20):
        factors[f"reversal_{window}d"] = -(close / close.shift(window) - 1.0)
    factors["overnight_gap"] = -(open_ / close.shift(1) - 1.0)
    factors["intraday_move"] = -(close / open_ - 1.0)
    factors["close_position_in_range"] = (close - low) / (high - low).replace(0, np.nan)
    factors["amount_surge"] = amount / amount.rolling(20).mean().shift(1)
    factors["amount_shrink"] = -factors["amount_surge"]
    factors["volatility_20d"] = -returns.rolling(20).std()
    factors["range_compression"] = -(rng / rng.rolling(20).mean().shift(1))
    factors["illiquidity"] = -(returns.abs() / amount.replace(0, np.nan)).rolling(20).mean()
    factors["downside_semivol"] = -returns.clip(upper=0).pow(2).rolling(20).sum()
    factors["drawdown_from_20d_high"] = close / close.rolling(20).max() - 1.0
    factors["turnover_trend"] = -(amount.rolling(5).mean() / amount.rolling(60).mean().shift(1))
    factors["reversal_x_lowvol"] = (
        -(close / close.shift(5) - 1.0) * (-returns.rolling(20).std()).rank(axis=1, pct=True)
    )
    factors["gap_fade_x_volume"] = (
        -(open_ / close.shift(1) - 1.0) * (amount / amount.rolling(20).mean().shift(1)).rank(axis=1, pct=True)
    )
    return factors


def evaluate(signal: pd.DataFrame, label: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    signal = signal.where(eligible).replace([np.inf, -np.inf], np.nan)
    usable = (signal.notna() & label.notna()).sum(axis=1) >= 50
    if int(usable.sum()) < 200:
        return {}
    ic = signal[usable].corrwith(label[usable], axis=1, method="spearman").dropna()
    ranks = signal.rank(axis=1, pct=True)
    top = ranks.ge(1.0 - DECILE) & usable.values[:, None]
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book = (weights * label).sum(axis=1, min_count=1)
    bench = label.where(eligible)[usable].mean(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    gross = (book - bench)[usable].dropna()
    net = (book - bench - turnover * COST_RT)[usable].dropna()

    def block(series: pd.Series, window: str) -> dict:
        part = series[series.index <= TRAIN_END] if window == "train" else series[series.index > TRAIN_END]
        if len(part) < 60:
            return {"n": len(part)}
        equity = (1.0 + part).cumprod()
        drawdown = float((equity / equity.cummax() - 1.0).min())
        std = part.std(ddof=1)
        return {
            "n": len(part),
            "mean_bps": round(float(part.mean()) * 1e4, 2),
            "t": round(float(part.mean() / std * np.sqrt(len(part))), 2) if std else None,
            "ir_ann": round(float(part.mean() / std * np.sqrt(244)), 3) if std else None,
            "max_drawdown_pct": round(drawdown * 100, 2),
            "win_rate": round(float((part > 0).mean()), 3),
        }

    ic_train = ic[ic.index <= TRAIN_END]
    ic_test = ic[ic.index > TRAIN_END]
    def ic_block(part):
        if len(part) < 60:
            return {"n": len(part)}
        std = part.std(ddof=1)
        return {"n": len(part), "mean": round(float(part.mean()), 5),
                "t": round(float(part.mean() / std * np.sqrt(len(part))), 2) if std else None}
    return {
        "ic": {"train": ic_block(ic_train), "test": ic_block(ic_test)},
        "gross": {"train": block(gross, "train"), "test": block(gross, "test")},
        "net": {"train": block(net, "train"), "test": block(net, "test")},
        "turnover_per_day": round(float(turnover[usable].mean()), 3),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=1200)
    args = parser.parse_args()
    started = time.time()

    panel = load_panel(args.symbols)
    close = panel["close"]
    eligible = eligibility(panel)
    label = labels(panel, eligible)
    print(f"panel {close.shape[1]} symbols x {close.shape[0]} sessions "
          f"({close.index.min():%Y-%m-%d}..{close.index.max():%Y-%m-%d}) "
          f"| eligible name-days {int(eligible.sum().sum()):,} "
          f"| labelled {int(label.notna().sum().sum()):,} [{time.time()-started:.0f}s]", flush=True)

    results = {}
    for name, signal in build_factors(panel).items():
        stats = evaluate(signal, label, eligible)
        if stats:
            results[name] = stats
            t = stats["net"]["test"]
            print(f"  {name:<26} testIC t={str(stats['ic']['test'].get('t')):>6} | "
                  f"net {str(t.get('mean_bps')):>7}bps IR={str(t.get('ir_ann')):>7} "
                  f"maxDD={str(t.get('max_drawdown_pct')):>7}% win={t.get('win_rate')}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now().astimezone().isoformat(),
        "status": "diagnostic_only_research_only_not_trading",
        "holding": "buy open[t+1], sell open[t+2] (shortest T+1-executable hold)",
        "roundTripCost": COST_RT,
        "trainEnd": TRAIN_END,
        "panel": {"symbols": int(close.shape[1]), "sessions": int(close.shape[0])},
        "factors": results,
        "orders": [],
    }
    (OUT_DIR / "nextday_factor_scan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\ndone [{time.time()-started:.0f}s] -> {OUT_DIR / 'nextday_factor_scan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
