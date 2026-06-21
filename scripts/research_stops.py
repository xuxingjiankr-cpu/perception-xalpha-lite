"""Research: do stop-loss / trailing / take-profit exits improve our exit edge?
(offline / SHADOW)

Kaminski-Lo (2014, "When Do Stop-Loss Rules Stop Losses?"): a stop helps when returns
are momentum-like / negatively skewed (it cuts the fat left tail) but HURTS when returns
mean-revert (it sells the bottom right before the bounce). So a stop is not free -- it
must be tested on OUR actual entry population.

Method (offline): mimic the system's momentum entries -- at each decision minute, "enter"
the recent top-decile names at the current price and follow each one's intraday path to
the close under several exit rules. Compare the realized return DISTRIBUTION (mean, std,
hit rate, left-tail p5) net of cost across rules. The question: does any stop/trail beat
plain hold-to-close in expectancy AND tail, on our population?

STRICTLY offline: NO orders, NO broker calls, NO agent state. diagnostic_only; needs
MANY days + promotion gates before any live wiring.

Run: py -3.13 scripts/research_stops.py
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
OUT_DIR = ROOT / "outputs" / "research_stops"
MIN_DAYS_FOR_STATISTICAL_TESTING = 20
SESSION_CLOSE = 15 * 60  # 15:00 China time


def simulate_exit(path: list[float], *, stop: float | None = None, trail: float | None = None,
                  take: float | None = None) -> float:
    """Realized return (fraction) for an entry at path[0], walking the price path. The
    first triggered rule wins; otherwise exit at the last price (hold-to-close).
    stop/trail/take are fractions (0.01 = 1%). Fills are assumed at the trigger level."""
    if not path or path[0] <= 0:
        return 0.0
    entry = path[0]
    peak = entry
    for p in path[1:]:
        peak = max(peak, p)
        if take is not None and p >= entry * (1 + take):
            return take
        if stop is not None and p <= entry * (1 - stop):
            return -stop
        if trail is not None and peak > entry and p <= peak * (1 - trail):
            return peak * (1 - trail) / entry - 1.0
    return path[-1] / entry - 1.0


RULES: dict[str, dict[str, Any]] = {
    "hold_to_close": {},
    "stop_1.0%": {"stop": 0.010},
    "stop_1.5%": {"stop": 0.015},
    "stop_2.0%": {"stop": 0.020},
    "trail_1.0%": {"trail": 0.010},
    "trail_1.5%": {"trail": 0.015},
    "tp2.0%_stop1.0%": {"take": 0.020, "stop": 0.010},
}


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


def _sorted_path(by_min: dict[int, float], start: int, end: int) -> list[float]:
    return [by_min[m] for m in sorted(by_min) if start <= m <= end]


def price_at(by_min: dict[int, float], minute: int) -> float | None:
    best_m = best_p = None
    for m, p in by_min.items():
        if m <= minute and (best_m is None or m > best_m):
            best_m, best_p = m, p
    return best_p


def analyze_day(by_code: dict[str, dict[str, Any]], *, decision_minutes: list[int], window: int,
                min_amount: float, top_frac: float, cost_pct: float) -> dict[str, list[float]]:
    cost = cost_pct / 100.0
    buckets: dict[str, list[float]] = {}
    for t in decision_minutes:
        eligible = {
            c: n for c, n in by_code.items()
            if point_in_time_liquidity_gate(n["amount_by_min"], t, min_amount)[0]
            and sum(1 for minute in n["by_min"] if minute <= t) >= 2
        }
        if len(eligible) < 20:
            continue
        recent = []
        for c, n in eligible.items():
            p_now, p_past = price_at(n["by_min"], t), price_at(n["by_min"], t - window)
            if p_now and p_past and p_past > 0:
                recent.append((c, p_now / p_past - 1.0))
        if len(recent) < 20:
            continue
        recent.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(recent) * top_frac))
        for c, _ in recent[:k]:  # momentum entry population
            path = _sorted_path(by_code[c]["by_min"], t, SESSION_CLOSE)
            if len(path) < 2:
                continue
            for rule_name, params in RULES.items():
                buckets.setdefault(rule_name, []).append((simulate_exit(path, **params) - cost) * 100)
    return buckets


def _stats(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    s = sorted(vals)
    p5 = s[max(0, int(n * 0.05) - 1)]
    return {"n": n, "mean_net_pct": round(mean, 4), "std_pct": round(sd, 4),
            "hit_rate": round(sum(1 for v in vals if v > 0) / n, 3), "p5_pct": round(p5, 4)}


def summarize(all_buckets: dict[str, list[float]], days: int, params: dict[str, Any]) -> dict[str, Any]:
    per = {name: _stats(vals) for name, vals in all_buckets.items() if vals}
    base = per.get("hold_to_close", {}).get("mean_net_pct")
    vs_hold = {name: round(s["mean_net_pct"] - base, 4) for name, s in per.items() if base is not None}
    enough = days >= MIN_DAYS_FOR_STATISTICAL_TESTING
    gate_status = "not_run_requires_replay_integration" if enough else "not_run_insufficient_days"
    return {
        "research_version": RESEARCH_VERSION, "generated_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True, "status": "diagnostic_only", "edge_validated": False,
        "liquidity_source": "point_in_time",
        "live_ready": False, "formal_strategy_allowed": False, "order_submit_calls_made": False,
        "params": params, "sample_days": days, "per_rule": per,
        "mean_minus_hold_pct": vs_hold,
        "statistical_readiness": {
            "minimum_days_before_statistical_testing": MIN_DAYS_FOR_STATISTICAL_TESTING,
            "sample_sufficient_for_statistical_testing": enough,
            "required_promotion_gates": {g: {"status": gate_status, "passed": False} for g in REQUIRED_PROMOTION_GATES},
            "verdict": "no_validated_edge",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Stop-loss / trailing / take-profit exit research (offline).")
    ap.add_argument("--decision-minutes", default="600,630,660,780,810,840")  # 10:00..14:00 China
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-amount", type=float, default=50_000_000.0)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--cost-pct", type=float, default=0.05)
    args = ap.parse_args()

    decision_minutes = [int(x) for x in str(args.decision_minutes).split(",") if x.strip()]
    all_buckets: dict[str, list[float]] = {}
    days = 0
    for day_dir in sorted(p for p in SNAP_DIR.glob("*") if p.is_dir()):
        by_code = load_day(day_dir)
        if not by_code:
            continue
        b = analyze_day(by_code, decision_minutes=decision_minutes, window=args.window,
                        min_amount=args.min_amount, top_frac=args.top_frac, cost_pct=args.cost_pct)
        if b:
            days += 1
            for k, v in b.items():
                all_buckets.setdefault(k, []).extend(v)

    print("=== stop-loss / exit-rule research (SHADOW, no orders) ===")
    print(f"momentum entry population (recent top-{int(args.top_frac*100)}%), hold to close under each rule; cost {args.cost_pct}%")
    summary = summarize(all_buckets, days, vars(args))
    if not summary["per_rule"]:
        print("no usable samples yet.")
        return
    for name in RULES:
        s = summary["per_rule"].get(name)
        if s:
            d = summary["mean_minus_hold_pct"].get(name, 0.0)
            print(f"  {name:16s}: n={s['n']:5d} mean_net={s['mean_net_pct']:+.3f}% (vs hold {d:+.3f}%) "
                  f"hit={s['hit_rate']} p5(tail)={s['p5_pct']:+.3f}% std={s['std_pct']}")
    print(f"verdict: {summary['statistical_readiness']['verdict']} (sample_days={days}; "
          f"need >= {MIN_DAYS_FOR_STATISTICAL_TESTING} + gates). Kaminski-Lo: a stop must beat hold on "
          f"mean AND tail to be worth it.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stops_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"log: {OUT_DIR / 'stops_summary.json'}")


if __name__ == "__main__":
    main()
