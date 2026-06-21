"""Research: regime-conditional entry edge (offline / SHADOW).

Our live system only does momentum/breakout entries -- which win in TRENDING markets
and bleed in CHOP (the 2026-06-18 159546 chase died in an afternoon range). Hypothesis:
classify the intraday regime and the right STYLE flips -- momentum in a trend, mean-
reversion (or stand aside) in chop. If true, a regime filter raises expectancy by not
trading the wrong style.

We classify INTRADAY (per decision window), not per-day, so a few days still yield many
samples from the cross-section. Regime is read from an equal-weight "market" level via
Kaufman's Efficiency Ratio (|net move| / summed path): high ER + a move => trend;
otherwise chop. At each decision point we then compare two LONG entry styles' forward
return net of cost: MOMENTUM (buy recent top-decile) vs REVERSION (buy recent bottom-
decile). The question: momentum_fwd > reversion_fwd in trend, and the reverse in chop?

STRICTLY offline: NO orders, NO broker calls, NO agent state. diagnostic_only;
needs MANY days + the promotion gates before any live wiring.

Run: py -3.13 scripts/research_regime.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float
from research_early_entry import _china_minute, REQUIRED_PROMOTION_GATES, RESEARCH_VERSION
from point_in_time_liquidity import point_in_time_liquidity_gate

SNAP_DIR = ROOT / "data" / "market" / "eastmoney" / "full_market" / "snapshots"
OUT_DIR = ROOT / "outputs" / "research_regime"
MIN_DAYS_FOR_STATISTICAL_TESTING = 20


def load_day(day_dir: Path) -> dict[str, dict[str, Any]]:
    by_code: dict[str, dict[str, Any]] = {}
    for gz in sorted(day_dir.glob("*_etf.csv.gz")):
        try:
            with gzip.open(gz, "rb") as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        except Exception:
            continue
        for r in rows:
            minute = _china_minute(r)
            price = as_float(r.get("currentPrice"), 0.0)
            op = as_float(r.get("open"), 0.0)
            if minute is None or price <= 0 or op <= 0:
                continue
            code = str(r.get("stockCode", "")).zfill(6)
            node = by_code.setdefault(code, {"open": op, "amount_by_min": {}, "by_min": {}})
            node["by_min"][minute] = price
            node["amount_by_min"][minute] = as_float(r.get("amount"), 0.0)
    return by_code


def price_at(by_min: dict[int, float], minute: int) -> float | None:
    best_m, best_p = None, None
    for m, p in by_min.items():
        if m <= minute and (best_m is None or m > best_m):
            best_m, best_p = m, p
    return best_p


def efficiency_ratio(levels: list[float]) -> float:
    if len(levels) < 2:
        return 0.0
    net = abs(levels[-1] - levels[0])
    path = sum(abs(levels[i] - levels[i - 1]) for i in range(1, len(levels)))
    return (net / path) if path > 0 else 0.0


def classify_regime(net_move: float, er: float, *, er_thr: float, move_thr: float) -> str:
    if abs(net_move) >= move_thr and er >= er_thr:
        return "trend_up" if net_move > 0 else "trend_down"
    return "chop"


def analyze_day(by_code: dict[str, dict[str, Any]], *, decision_start: int, decision_end: int,
                step: int, window: int, horizon: int, min_amount: float, top_frac: float,
                er_thr: float, move_thr: float, roundtrip_cost_pct: float) -> dict[str, list[float]]:
    """Return {regime+"::"+style: [forward_returns_pct_net_of_cost...]}."""
    def market_level(eligible: dict[str, dict[str, Any]], minute: int) -> float | None:
        rets = []
        for n in eligible.values():
            p = price_at(n["by_min"], minute)
            if p and n["open"] > 0:
                rets.append(p / n["open"] - 1.0)
        return (sum(rets) / len(rets)) if rets else None

    buckets: dict[str, list[float]] = {}
    t = decision_start
    while t <= decision_end:
        eligible = {
            c: n for c, n in by_code.items()
            if point_in_time_liquidity_gate(n["amount_by_min"], t, min_amount)[0]
            and sum(1 for minute in n["by_min"] if minute <= t) >= 2
        }
        minutes = sorted({m for n in eligible.values() for m in n["by_min"] if m <= t})
        lvls = [market_level(eligible, m) for m in range(t - window, t + 1) if m in minutes]
        lvls = [x for x in lvls if x is not None]
        if len(lvls) >= 3:
            regime = classify_regime(lvls[-1] - lvls[0], efficiency_ratio(lvls), er_thr=er_thr, move_thr=move_thr)
            recent = []
            for c, n in eligible.items():
                p_now, p_past = price_at(n["by_min"], t), price_at(n["by_min"], t - window)
                p_fwd = price_at(n["by_min"], t + horizon)
                if p_now and p_past and p_fwd and p_past > 0 and p_now > 0:
                    recent.append((c, p_now / p_past - 1.0, p_fwd / p_now - 1.0))
            if len(recent) >= 20:
                recent.sort(key=lambda x: x[1], reverse=True)
                k = max(1, int(len(recent) * top_frac))
                mom = [x[2] for x in recent[:k]]            # buy recent winners (momentum, long)
                rev = [x[2] for x in recent[-k:]]           # buy recent losers (reversion, long)
                cost = roundtrip_cost_pct / 100.0
                buckets.setdefault(f"{regime}::momentum", []).extend([(m - cost) * 100 for m in mom])
                buckets.setdefault(f"{regime}::reversion", []).extend([(r - cost) * 100 for r in rev])
        t += step
    return buckets


def summarize(all_buckets: dict[str, list[float]], days: int, params: dict[str, Any]) -> dict[str, Any]:
    per = {}
    for key, vals in sorted(all_buckets.items()):
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        hit = sum(1 for v in vals if v > 0) / len(vals)
        per[key] = {"n": len(vals), "mean_fwd_net_pct": round(mean, 4), "hit_rate": round(hit, 3)}
    # hypothesis read: momentum-minus-reversion edge per regime
    spreads = {}
    for regime in ("trend_up", "trend_down", "chop"):
        m, r = per.get(f"{regime}::momentum"), per.get(f"{regime}::reversion")
        if m and r:
            spreads[regime] = round(m["mean_fwd_net_pct"] - r["mean_fwd_net_pct"], 4)
    enough = days >= MIN_DAYS_FOR_STATISTICAL_TESTING
    gate_status = "not_run_requires_replay_integration" if enough else "not_run_insufficient_days"
    return {
        "research_version": RESEARCH_VERSION, "generated_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True, "status": "diagnostic_only", "edge_validated": False,
        "live_ready": False, "formal_strategy_allowed": False, "order_submit_calls_made": False,
        "params": params, "sample_days": days, "per_regime_style": per,
        "liquidity_source": "point_in_time",
        "momentum_minus_reversion_pct_by_regime": spreads,
        "statistical_readiness": {
            "minimum_days_before_statistical_testing": MIN_DAYS_FOR_STATISTICAL_TESTING,
            "sample_sufficient_for_statistical_testing": enough,
            "required_promotion_gates": {g: {"status": gate_status, "passed": False} for g in REQUIRED_PROMOTION_GATES},
            "verdict": "no_validated_edge",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Regime-conditional entry research (offline, no orders).")
    ap.add_argument("--decision-start", default="10:00")
    ap.add_argument("--decision-end", default="14:30")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--min-amount", type=float, default=50_000_000.0)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--er-thr", type=float, default=0.4)
    ap.add_argument("--move-thr", type=float, default=0.003)
    ap.add_argument("--roundtrip-cost-pct", type=float, default=0.15)
    args = ap.parse_args()

    def hhmm(v: str) -> int:
        h, m = (int(x) for x in v.split(":"))
        return h * 60 + m

    all_buckets: dict[str, list[float]] = {}
    days = 0
    for day_dir in sorted(p for p in SNAP_DIR.glob("*") if p.is_dir()):
        by_code = load_day(day_dir)
        if not by_code:
            continue
        b = analyze_day(by_code, decision_start=hhmm(args.decision_start), decision_end=hhmm(args.decision_end),
                        step=args.step, window=args.window, horizon=args.horizon, min_amount=args.min_amount,
                        top_frac=args.top_frac, er_thr=args.er_thr, move_thr=args.move_thr,
                        roundtrip_cost_pct=args.roundtrip_cost_pct)
        if b:
            days += 1
            for k, v in b.items():
                all_buckets.setdefault(k, []).extend(v)

    print("=== regime-conditional entry edge (SHADOW research, no orders) ===")
    print(f"momentum=buy recent top-{int(args.top_frac*100)}% | reversion=buy recent bottom | net of {args.roundtrip_cost_pct}% cost")
    summary = summarize(all_buckets, days, vars(args))
    if not summary["per_regime_style"]:
        print("no usable samples yet.")
        return
    for key, s in summary["per_regime_style"].items():
        print(f"  {key:22s}: n={s['n']:5d} mean_fwd_net={s['mean_fwd_net_pct']:+.3f}% hit={s['hit_rate']}")
    print("--- hypothesis (momentum - reversion, by regime; want >0 in trend, <0 in chop) ---")
    for regime, spread in summary["momentum_minus_reversion_pct_by_regime"].items():
        print(f"  {regime:12s}: momentum - reversion = {spread:+.3f}%")
    print(f"verdict: {summary['statistical_readiness']['verdict']} (sample_days={days}; "
          f"need >= {MIN_DAYS_FOR_STATISTICAL_TESTING} + gates before any live wiring)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "regime_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"log: {OUT_DIR / 'regime_summary.json'}")


if __name__ == "__main__":
    main()
