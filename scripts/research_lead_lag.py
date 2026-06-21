"""Research: lead-lag predictability in the ETF cross-section (offline / SHADOW).

Hypothesis (extends the "buy earlier" thread): within a sector, the most LIQUID
ETF (the leader / bellwether) tends to move FIRST; the less-liquid followers catch
up with a lag. If true, a leader's recent move predicts its followers' next-window
move -- letting us enter a laggard EARLY, near its base, instead of chasing the
mature breakout hours later (the 2026-06-18 588030 failure mode).

Method, per trade day (offline, reads full-market snapshots only):
  - group ETFs by sector (config keyword map); the leader = highest full-day amount.
  - at decision minutes t (from --decision-start, every --step minutes), compute the
    leader's trailing return over --window, and each follower's FORWARD return over a
    set of horizons; measure (a) leader->follower correlation, (b) the conditional
    follower forward return when the leader is up beyond --leader-threshold, and (c)
    that return NET of a round-trip cost. The horizon with the best net return is the
    estimated "lead time".

STRICTLY research: NO orders, NO broker calls, NO agent state. Mirrors
research_early_entry.py conventions: diagnostic_only, statistical_readiness gates,
accumulates across days; needs MANY days + the promotion gates before anything is
wired into live trading.

Run: py -3.13 scripts/research_lead_lag.py
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

from run_etf_paper_trading_agent import ROOT, as_float, load_json
from run_t0_intraday_agent import classify_etf_sector
from research_early_entry import _china_minute, REQUIRED_PROMOTION_GATES, RESEARCH_VERSION
from point_in_time_liquidity import point_in_time_liquidity_gate, value_at_or_before

SNAP_DIR = ROOT / "data" / "market" / "eastmoney" / "full_market" / "snapshots"
OUT_DIR = ROOT / "outputs" / "research_lead_lag"
HORIZONS = (5, 10, 15, 30)  # minutes ahead to test for the follower (lead-time scan)
MIN_DAYS_FOR_STATISTICAL_TESTING = 20


def load_day(day_dir: Path) -> dict[str, dict[str, Any]]:
    """Code-level point-in-time amount and price series."""
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
            node = by_code.setdefault(code, {"name": r.get("name"), "amount_by_min": {}, "series": []})
            node["series"].append((minute, price))
            node["amount_by_min"][minute] = as_float(r.get("amount"), 0.0)
    for node in by_code.values():
        node["series"].sort(key=lambda x: x[0])
    return by_code


def price_at(series: list[tuple[int, float]], minute: int) -> float | None:
    """Last price at or before `minute` (step function); None if none yet."""
    chosen = None
    for m, p in series:
        if m <= minute:
            chosen = p
        else:
            break
    return chosen


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx <= 0 or sy <= 0:
        return None
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    return cov / (sx * sy)


def analyze_day(day: str, by_code: dict[str, dict[str, Any]], *, keyword_map: dict[str, Any],
                decision_start: int, decision_end: int, step: int, window: int,
                leader_threshold: float, min_amount: float, min_members: int,
                roundtrip_cost_pct: float) -> dict[str, Any] | None:
    # Sector membership is static; liquidity and leader choice are recomputed at
    # each decision time from information available then.
    sectors: dict[str, list[str]] = {}
    for code, node in by_code.items():
        sector = classify_etf_sector(node.get("name"), keyword_map)
        if sector == "other":
            continue  # only test real sectors with a meaningful leader
        sectors.setdefault(sector, []).append(code)

    # collect (leader_signal, follower_forward) pairs per horizon
    pairs_by_h: dict[int, list[tuple[float, float]]] = {h: [] for h in HORIZONS}
    sectors_seen: set[str] = set()
    for sector, codes in sectors.items():
        t = decision_start
        while t <= decision_end:
            eligible_codes = [
                code for code in codes
                if point_in_time_liquidity_gate(by_code[code]["amount_by_min"], t, min_amount)[0]
                and sum(1 for minute, _ in by_code[code]["series"] if minute <= t) >= 2
            ]
            if len(eligible_codes) < min_members:
                t += step
                continue
            leader = max(
                eligible_codes,
                key=lambda code: value_at_or_before(by_code[code]["amount_by_min"], t) or 0.0,
            )
            followers = [code for code in eligible_codes if code != leader]
            lead_series = by_code[leader]["series"]
            sectors_seen.add(sector)
            p_now = price_at(lead_series, t)
            p_past = price_at(lead_series, t - window)
            if p_now and p_past and p_past > 0:
                leader_sig = p_now / p_past - 1.0
                for fcode in followers:
                    fser = by_code[fcode]["series"]
                    f_now = price_at(fser, t)
                    if not f_now or f_now <= 0:
                        continue
                    for h in HORIZONS:
                        f_fwd_px = price_at(fser, t + h)
                        if f_fwd_px and f_fwd_px > 0:
                            pairs_by_h[h].append((leader_sig, f_fwd_px / f_now - 1.0))
            t += step

    sectors_used = len(sectors_seen)
    if sectors_used == 0 or all(len(v) < 20 for v in pairs_by_h.values()):
        return None

    per_h = {}
    for h, pairs in pairs_by_h.items():
        if len(pairs) < 20:
            continue
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        corr = _corr(xs, ys)
        cond = [b for a, b in pairs if a >= leader_threshold]  # leader up -> buy follower
        cond_mean = (sum(cond) / len(cond)) if cond else None
        net = (cond_mean * 100 - roundtrip_cost_pct) if cond_mean is not None else None
        hit = (sum(1 for b in cond if b > 0) / len(cond)) if cond else None
        per_h[h] = {
            "pairs": len(pairs), "corr": round(corr, 4) if corr is not None else None,
            "leader_up_signals": len(cond),
            "follower_fwd_pct": round(cond_mean * 100, 4) if cond_mean is not None else None,
            "net_of_cost_pct": round(net, 4) if net is not None else None,
            "hit_rate": round(hit, 3) if hit is not None else None,
        }
    if not per_h:
        return None
    # Lead time = the horizon where the leader->follower CORRELATION peaks (timing).
    # Tradability is a separate question: the horizon with the best net-of-cost return.
    corr_h = max((h for h in per_h if per_h[h]["corr"] is not None),
                 key=lambda h: per_h[h]["corr"], default=None)
    net_h = max((h for h in per_h if per_h[h]["net_of_cost_pct"] is not None),
                key=lambda h: per_h[h]["net_of_cost_pct"], default=None)
    return {
        "date": day, "sectors_used": sectors_used,
        "roundtrip_cost_pct": roundtrip_cost_pct,
        "best_lead_minutes": corr_h,                                   # timing (corr peak)
        "best_corr": per_h[corr_h]["corr"] if corr_h else None,
        "best_net_horizon_minutes": net_h,                            # tradability
        "best_net_of_cost_pct": per_h[net_h]["net_of_cost_pct"] if net_h else None,
        "best_hit_rate": per_h[net_h]["hit_rate"] if net_h else None,
        "by_horizon": per_h,
    }


def summarize(results: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda r: str(r.get("date") or ""))
    nets = [as_float(r.get("best_net_of_cost_pct")) for r in ordered if r.get("best_net_of_cost_pct") is not None]
    corrs = [as_float(r.get("best_corr")) for r in ordered if r.get("best_corr") is not None]
    enough = len(nets) >= MIN_DAYS_FOR_STATISTICAL_TESTING
    gate_status = "not_run_requires_replay_integration" if enough else "not_run_insufficient_days"
    return {
        "research_version": RESEARCH_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True, "status": "diagnostic_only",
        "liquidity_source": "point_in_time",
        "edge_validated": False, "live_ready": False, "formal_strategy_allowed": False,
        "order_submit_calls_made": False, "params": params, "results": ordered,
        "sample_days": len(nets),
        "avg_best_net_of_cost_pct": round(sum(nets) / len(nets), 4) if nets else None,
        "positive_net_days": sum(1 for v in nets if v > 0),
        "avg_best_corr": round(sum(corrs) / len(corrs), 4) if corrs else None,
        "statistical_readiness": {
            "minimum_days_before_statistical_testing": MIN_DAYS_FOR_STATISTICAL_TESTING,
            "sample_sufficient_for_statistical_testing": enough,
            "required_promotion_gates": {g: {"status": gate_status, "passed": False} for g in REQUIRED_PROMOTION_GATES},
            "verdict": "no_validated_edge",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Lead-lag predictability research (offline, no orders).")
    ap.add_argument("--decision-start", default="10:00")
    ap.add_argument("--decision-end", default="14:30")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--leader-threshold", type=float, default=0.002)
    ap.add_argument("--min-amount", type=float, default=50_000_000.0)
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--roundtrip-cost-pct", type=float, default=0.15)
    args = ap.parse_args()

    def hhmm(v: str) -> int:
        h, m = (int(x) for x in v.split(":"))
        return h * 60 + m

    cfg = load_json(ROOT / "configs" / "t0_intraday_paper_agent.json")
    keyword_map = (cfg.get("sector_diversification", {}) or {}).get("keyword_map", {})
    params = vars(args)
    results = []
    for day_dir in sorted(p for p in SNAP_DIR.glob("*") if p.is_dir()):
        by_code = load_day(day_dir)
        if not by_code:
            continue
        r = analyze_day(day_dir.name, by_code, keyword_map=keyword_map,
                        decision_start=hhmm(args.decision_start), decision_end=hhmm(args.decision_end),
                        step=args.step, window=args.window, leader_threshold=args.leader_threshold,
                        min_amount=args.min_amount, min_members=args.min_members,
                        roundtrip_cost_pct=args.roundtrip_cost_pct)
        if r:
            results.append(r)

    print("=== ETF lead-lag predictability (SHADOW research, no orders) ===")
    print(f"leader=most-liquid in sector | follower fwd horizons={HORIZONS}min | cost={args.roundtrip_cost_pct}%")
    if not results:
        print("no usable days yet.")
        return
    for r in results:
        print(f"  {r['date']}: sectors={r['sectors_used']} | best lead={r['best_lead_minutes']}min "
              f"corr={r['best_corr']} follower_net_of_cost={r['best_net_of_cost_pct']}% hit={r['best_hit_rate']}")
    summary = summarize(results, params)
    print("--- summary ---")
    print(f"avg best follower net-of-cost = {summary['avg_best_net_of_cost_pct']}% over {summary['sample_days']} days "
          f"({summary['positive_net_days']} positive); avg corr = {summary['avg_best_corr']}")
    print(f"verdict: {summary['statistical_readiness']['verdict']} "
          f"(need >= {MIN_DAYS_FOR_STATISTICAL_TESTING} days + promotion gates before any live wiring)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lead_lag_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"log: {OUT_DIR / 'lead_lag_summary.json'}")


if __name__ == "__main__":
    main()
