"""Research: position sizing across concurrent holdings -- equal-weight vs inverse-
volatility vs vol-targeting (offline / SHADOW).

We hold up to 5 names; equal-weight lets the most volatile name dominate portfolio
risk. Theory (risk parity; Moreira-Muir 2017 "Volatility-Managed Portfolios"): weight
by inverse volatility and/or scale total exposure down when volatility is high to
improve risk-adjusted return (Sharpe), not necessarily raw return.

Method (offline): at each decision minute build a basket of the recent top-decile
momentum names (the entry population). Weight it three ways -- equal, inverse-vol,
and vol-targeted (lever the equal basket toward a target vol) -- and record each
basket's forward return to a horizon, net of cost. Aggregate across baskets: mean,
std, hit, and a Sharpe-like mean/std. The question: do IV / vol-target raise mean/std
(risk-adjusted) vs equal weight?

STRICTLY offline: NO orders, NO broker calls, NO agent state. diagnostic_only; needs
MANY days + promotion gates before any live wiring.

Run: py -3.13 scripts/research_sizing.py
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
OUT_DIR = ROOT / "outputs" / "research_sizing"
MIN_DAYS_FOR_STATISTICAL_TESTING = 20


def inverse_vol_weights(vols: list[float], eps: float = 1e-4) -> list[float]:
    """Weights proportional to 1/vol, normalized to sum to 1 (risk parity, uncorrelated)."""
    inv = [1.0 / (max(v, 0.0) + eps) for v in vols]
    total = sum(inv)
    return [x / total for x in inv] if total > 0 else [1.0 / len(vols)] * len(vols)


def basket_return(weights: list[float], fwds: list[float]) -> float:
    return sum(w * f for w, f in zip(weights, fwds))


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
            if minute is None or price <= 0:
                continue
            code = str(r.get("stockCode", "")).zfill(6)
            node = by_code.setdefault(code, {"amount_by_min": {}, "by_min": {}})
            node["by_min"][minute] = price
            node["amount_by_min"][minute] = as_float(r.get("amount"), 0.0)
    return by_code


def price_at(by_min: dict[int, float], minute: int) -> float | None:
    best_m = best_p = None
    for m, p in by_min.items():
        if m <= minute and (best_m is None or m > best_m):
            best_m, best_p = m, p
    return best_p


def recent_vol(by_min: dict[int, float], t: int, window: int) -> float | None:
    pts = sorted((m, p) for m, p in by_min.items() if t - window <= m <= t and p > 0)
    if len(pts) < 3:
        return None
    rets = [pts[i][1] / pts[i - 1][1] - 1.0 for i in range(1, len(pts)) if pts[i - 1][1] > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    return (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5


def analyze_day(by_code: dict[str, dict[str, Any]], *, decision_minutes: list[int], window: int,
                horizon: int, min_amount: float, basket_k: int, target_vol: float, max_lev: float,
                cost_pct: float) -> dict[str, list[float]]:
    cost = cost_pct / 100.0
    buckets: dict[str, list[float]] = {"equal_weight": [], "inverse_vol": [], "vol_targeted": []}
    for t in decision_minutes:
        eligible = {
            c: n for c, n in by_code.items()
            if point_in_time_liquidity_gate(n["amount_by_min"], t, min_amount)[0]
            and sum(1 for minute in n["by_min"] if minute <= t) >= 3
        }
        cand = []
        for c, n in eligible.items():
            p_now, p_past = price_at(n["by_min"], t), price_at(n["by_min"], t - window)
            p_fwd = price_at(n["by_min"], t + horizon)
            vol = recent_vol(n["by_min"], t, window)
            if p_now and p_past and p_fwd and vol and p_past > 0 and p_now > 0:
                cand.append((c, p_now / p_past - 1.0, p_fwd / p_now - 1.0, vol))
        if len(cand) < basket_k:
            continue
        cand.sort(key=lambda x: x[1], reverse=True)  # momentum rank
        basket = cand[:basket_k]
        fwds = [x[2] for x in basket]
        vols = [x[3] for x in basket]
        ew = [1.0 / basket_k] * basket_k
        iv = inverse_vol_weights(vols)
        ew_ret = basket_return(ew, fwds)
        iv_ret = basket_return(iv, fwds)
        basket_vol = sum(v / basket_k for v in vols) or 1e-9  # crude (ignores correlation)
        lev = min(max_lev, target_vol / basket_vol)
        buckets["equal_weight"].append((ew_ret - cost) * 100)
        buckets["inverse_vol"].append((iv_ret - cost) * 100)
        buckets["vol_targeted"].append((ew_ret * lev - cost * lev) * 100)
    return buckets


def _stats(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    return {"n": n, "mean_net_pct": round(mean, 4), "std_pct": round(sd, 4),
            "sharpe_like": round(mean / sd, 4) if sd > 0 else None,
            "hit_rate": round(sum(1 for v in vals if v > 0) / n, 3)}


def summarize(all_buckets: dict[str, list[float]], days: int, params: dict[str, Any]) -> dict[str, Any]:
    per = {k: _stats(v) for k, v in all_buckets.items() if v}
    base = per.get("equal_weight", {}).get("sharpe_like")
    vs_ew = {k: round((s["sharpe_like"] or 0) - base, 4) for k, s in per.items()} if base is not None else {}
    enough = days >= MIN_DAYS_FOR_STATISTICAL_TESTING
    gate_status = "not_run_requires_replay_integration" if enough else "not_run_insufficient_days"
    return {
        "research_version": RESEARCH_VERSION, "generated_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True, "status": "diagnostic_only", "edge_validated": False,
        "liquidity_source": "point_in_time",
        "live_ready": False, "formal_strategy_allowed": False, "order_submit_calls_made": False,
        "params": params, "sample_days": days, "per_scheme": per, "sharpe_minus_equal_weight": vs_ew,
        "statistical_readiness": {
            "minimum_days_before_statistical_testing": MIN_DAYS_FOR_STATISTICAL_TESTING,
            "sample_sufficient_for_statistical_testing": enough,
            "required_promotion_gates": {g: {"status": gate_status, "passed": False} for g in REQUIRED_PROMOTION_GATES},
            "verdict": "no_validated_edge",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Position-sizing research (offline, no orders).")
    ap.add_argument("--decision-minutes", default="600,630,660,690,780,810,840")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--min-amount", type=float, default=50_000_000.0)
    ap.add_argument("--basket-k", type=int, default=5)
    ap.add_argument("--target-vol", type=float, default=0.004)
    ap.add_argument("--max-lev", type=float, default=1.0)
    ap.add_argument("--cost-pct", type=float, default=0.05)
    args = ap.parse_args()

    decision_minutes = [int(x) for x in str(args.decision_minutes).split(",") if x.strip()]
    all_buckets: dict[str, list[float]] = {}
    days = 0
    for day_dir in sorted(p for p in SNAP_DIR.glob("*") if p.is_dir()):
        by_code = load_day(day_dir)
        if not by_code:
            continue
        b = analyze_day(by_code, decision_minutes=decision_minutes, window=args.window, horizon=args.horizon,
                        min_amount=args.min_amount, basket_k=args.basket_k, target_vol=args.target_vol,
                        max_lev=args.max_lev, cost_pct=args.cost_pct)
        if any(b.values()):
            days += 1
            for k, v in b.items():
                all_buckets.setdefault(k, []).extend(v)

    print("=== position-sizing research (SHADOW, no orders) ===")
    print(f"basket = top-{args.basket_k} momentum; horizon {args.horizon}min; target_vol {args.target_vol}; cost {args.cost_pct}%")
    summary = summarize(all_buckets, days, vars(args))
    if not summary["per_scheme"]:
        print("no usable baskets yet.")
        return
    for name in ("equal_weight", "inverse_vol", "vol_targeted"):
        s = summary["per_scheme"].get(name)
        if s:
            d = summary["sharpe_minus_equal_weight"].get(name, 0.0)
            print(f"  {name:14s}: n={s['n']:4d} mean_net={s['mean_net_pct']:+.3f}% std={s['std_pct']} "
                  f"sharpe={s['sharpe_like']} (vs EW {d:+.4f}) hit={s['hit_rate']}")
    print(f"verdict: {summary['statistical_readiness']['verdict']} (sample_days={days}; "
          f"need >= {MIN_DAYS_FOR_STATISTICAL_TESTING} + gates). Moreira-Muir: judge on Sharpe, not raw return.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sizing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"log: {OUT_DIR / 'sizing_summary.json'}")


if __name__ == "__main__":
    main()
