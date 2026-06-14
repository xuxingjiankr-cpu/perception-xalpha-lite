"""Offline missingness stress replay for the T+0 ETF paper agent.

This script implements robustness tests inspired by Bernoulli amputation:
it damages already-collected quote snapshots with deterministic missingness
patterns, then replays build_decision() locally. It never calls SkillClient,
never submits/cancels orders, and never writes outputs/t0_intraday_agent/t0_state.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, as_float, load_json, write_json
from replay_t0_decisions import (
    DEFAULT_QUOTES,
    FAKE_PENDING,
    INITIAL_CASH,
    OUT_DIR,
    apply_buy_fill,
    apply_sell_fill,
    fake_balance,
    fake_positions,
    group_rounds,
    load_rows,
    record_post_submit,
)

import run_t0_intraday_agent as agent


STRESS_OUT_DIR = ROOT / "outputs" / "t0_missingness_stress"


def stable_unit(*parts: Any) -> float:
    raw = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def mark_quote_missing(q: dict[str, Any], reason: str) -> dict[str, Any]:
    q = dict(q)
    q["quote_ok"] = False
    q["currentPrice"] = 0.0
    q["prevClose"] = 0.0
    q["bidPrice1"] = 0.0
    q["askPrice1"] = 0.0
    q["midpoint"] = 0.0
    q["spread_pct"] = None
    q["momentum"] = None
    q["momentum_available"] = False
    q["missingness_stress_reason"] = reason
    q["quote_error"] = {
        "category": "stress_replay",
        "message": reason,
    }
    return q


def remove_bid_ask(q: dict[str, Any], reason: str) -> dict[str, Any]:
    q = dict(q)
    q["bidPrice1"] = 0.0
    q["askPrice1"] = 0.0
    q["midpoint"] = as_float(q.get("currentPrice"), 0.0)
    q["spread_pct"] = None
    q["missingness_stress_reason"] = reason
    return q


def stale_quote(q: dict[str, Any], minutes: int = 10) -> dict[str, Any]:
    q = dict(q)
    dt = agent.parse_iso_dt(q.get("timestamp"))
    if dt is not None:
        q["timestamp"] = (dt - timedelta(minutes=minutes)).isoformat()
    q["missingness_stress_reason"] = f"timestamp_stale_{minutes}m"
    return q


def apply_missingness(round_quotes: list[dict[str, Any]], scenario: str, round_idx: int) -> list[dict[str, Any]]:
    damaged: list[dict[str, Any]] = []
    for q in round_quotes:
        code = str(q.get("stockCode", "")).zfill(6)
        asset_class = q.get("asset_class")
        ts = q.get("timestamp")
        if scenario == "baseline":
            damaged.append(dict(q))
        elif scenario == "random_quote_missing_20pct":
            if stable_unit(scenario, code, ts) < 0.20:
                damaged.append(mark_quote_missing(q, "random_quote_missing_20pct"))
            else:
                damaged.append(dict(q))
        elif scenario == "block_hk_etf_missing_every_5th_round":
            if round_idx % 5 == 0 and asset_class == "hk_etf":
                damaged.append(mark_quote_missing(q, "block_hk_etf_missing_every_5th_round"))
            else:
                damaged.append(dict(q))
        elif scenario == "bid_ask_missing_30pct":
            if stable_unit(scenario, code, ts) < 0.30:
                damaged.append(remove_bid_ask(q, "bid_ask_missing_30pct"))
            else:
                damaged.append(dict(q))
        elif scenario == "quota_afternoon_all_missing":
            dt = agent.parse_iso_dt(q.get("timestamp"))
            local = dt.astimezone(ZoneInfo("Asia/Shanghai")) if dt else None
            if local and (local.hour, local.minute) >= (14, 0):
                damaged.append(mark_quote_missing(q, "quota_afternoon_all_missing"))
            else:
                damaged.append(dict(q))
        elif scenario == "stale_all_quotes_10m":
            damaged.append(stale_quote(q, 10))
        elif scenario == "zero_price_one_quote_10pct":
            if stable_unit(scenario, code, ts) < 0.10:
                damaged.append(mark_quote_missing(q, "zero_price_one_quote_10pct"))
            else:
                damaged.append(dict(q))
        else:
            raise ValueError(f"unknown scenario: {scenario}")
    return damaged


def check_invariants(decision: dict[str, Any], quotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    orders = decision.get("orders") if isinstance(decision.get("orders"), list) else []
    ranked = decision.get("ranked") if isinstance(decision.get("ranked"), list) else []
    missing_codes = {
        str(q.get("stockCode", "")).zfill(6)
        for q in quotes
        if not q.get("quote_ok") or as_float(q.get("currentPrice"), 0.0) <= 0
    }
    ranked_codes = {str(q.get("stockCode", "")).zfill(6) for q in ranked}
    leaked = sorted(missing_codes & ranked_codes)
    if leaked:
        failures.append({"name": "missing_quote_not_ranked", "codes": leaked})

    for order in orders:
        price = as_float(order.get("price"), 0.0)
        qty = int(as_float(order.get("quantity"), 0.0))
        if price <= 0:
            failures.append({"name": "order_price_positive", "order": order})
        if qty <= 0:
            failures.append({"name": "order_quantity_positive", "order": order})
        if order.get("direction") == "buy" and str(order.get("stockCode", "")).zfill(6) in missing_codes:
            failures.append({"name": "no_buy_on_missing_quote", "order": order})

    if any(q.get("quote_ok") and q.get("missingness_stress_reason") == "timestamp_stale_10m" for q in quotes):
        check = next((c for c in decision.get("risk_checks", []) if c.get("name") == "quote_freshness"), None)
        if not check or check.get("passed"):
            failures.append({"name": "stale_quotes_block_entry", "quote_freshness": check})

    if all(not q.get("quote_ok") for q in quotes):
        if decision.get("approved_for_submit"):
            failures.append({"name": "all_missing_not_approved", "approved": decision.get("approved_for_submit")})
    return failures


def run_scenario(cfg: dict[str, Any], rounds: list[list[dict[str, Any]]], scenario: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    sim = {"cash": INITIAL_CASH, "positions": {}}
    history: list[dict[str, Any]] = []
    lookback = int(cfg["strategy"]["lookback_minutes"])
    submit_seq = 0
    counters: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    approved_orders = 0
    orders_built = 0
    per_day: dict[str, dict[str, Any]] = {}

    for idx, rnd in enumerate(rounds, start=1):
        ts = agent.parse_iso_dt(rnd[-1].get("timestamp"))
        if ts is None:
            continue
        replay_now = ts.astimezone(ZoneInfo("Asia/Shanghai"))
        agent.set_replay_now(replay_now)
        trade_date = replay_now.strftime("%Y-%m-%d")
        day = per_day.setdefault(trade_date, {"rounds": 0, "approved": 0, "orders": 0, "failures": 0})
        day["rounds"] += 1

        raw_quotes = apply_missingness([dict(q) for q in rnd], scenario, idx)
        quotes = agent.compute_snapshot_momentum(raw_quotes, history, lookback, cfg["strategy"])
        market_correlation_stress = agent.compute_market_correlation_stress(
            history,
            quotes,
            cfg.get("strategy", {}).get("market_correlation_stress", {}),
        )
        history.extend(quotes)
        agent.update_orb_state(state, trade_date, quotes, replay_now, True)

        total_assets = sim["cash"] + sum(
            as_float(p.get("costPrice")) * as_float(p.get("quantity"))
            for p in sim["positions"].values()
        )
        decision = agent.build_decision(
            cfg,
            quotes,
            fake_balance(total_assets, sim["cash"]),
            fake_positions(sim["positions"]),
            FAKE_PENDING,
            state,
            market_correlation_stress,
        )
        for c in decision.get("risk_checks", []):
            if not c.get("passed"):
                counters[f"risk_fail:{c.get('name', '?')}"] += 1
        round_failures = check_invariants(decision, quotes)
        if round_failures:
            failures.append({
                "timestamp": replay_now.isoformat(),
                "scenario": scenario,
                "failures": round_failures,
                "reason": decision.get("state_machine", {}).get("reason"),
            })
            day["failures"] += len(round_failures)

        orders = decision.get("orders") if isinstance(decision.get("orders"), list) else []
        if orders:
            orders_built += len(orders)
            day["orders"] += len(orders)
        if decision.get("approved_for_submit") and orders:
            approved_orders += len(orders)
            day["approved"] += len(orders)
            for order in orders:
                submit_seq += 1
                if order.get("direction") == "buy":
                    apply_buy_fill(sim, order)
                elif order.get("direction") == "sell":
                    apply_sell_fill(sim, order)
                record_post_submit(state, cfg, decision, order, submit_seq)

    agent.set_replay_now(None)
    return {
        "scenario": scenario,
        "rounds": sum(v["rounds"] for v in per_day.values()),
        "orders_built": orders_built,
        "approved_orders": approved_orders,
        "invariant_failures": failures,
        "invariant_failure_count": sum(len(x["failures"]) for x in failures),
        "risk_failure_counts": dict(counters.most_common()),
        "per_day": per_day,
        "final_open_positions": sim["positions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline T+0 missingness stress replay")
    parser.add_argument("--config", default=str(agent.DEFAULT_CONFIG))
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--date", default=None, help="replay a single trade date YYYY-MM-DD")
    parser.add_argument("--label", default="missingness_stress")
    parser.add_argument("--scenario", action="append", default=None, help="run one scenario; can repeat")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    rows = load_rows(Path(args.quotes), args.date)
    rounds = group_rounds(rows)
    scenarios = args.scenario or [
        "baseline",
        "random_quote_missing_20pct",
        "block_hk_etf_missing_every_5th_round",
        "bid_ask_missing_30pct",
        "quota_afternoon_all_missing",
        "stale_all_quotes_10m",
        "zero_price_one_quote_10pct",
    ]

    STRESS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for scenario in scenarios:
            results.append(run_scenario(cfg, rounds, scenario))
    finally:
        agent.set_replay_now(None)

    summary = {
        "label": args.label,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "api_calls_made": False,
        "order_submit_calls_made": False,
        "state_file_touched": False,
        "quotes_file": str(Path(args.quotes)),
        "rounds_total": len(rounds),
        "scenarios": results,
        "all_invariants_passed": all(r["invariant_failure_count"] == 0 for r in results),
    }
    out_json = STRESS_OUT_DIR / f"{args.label}_summary.json"
    out_md = STRESS_OUT_DIR / f"{args.label}_summary.md"
    write_json(out_json, summary)
    lines = [
        f"# T0 Missingness Stress Replay - {args.label}",
        "",
        "Paper trading offline diagnostics only. No API calls and no order submissions were made.",
        "",
        f"- Rounds: {len(rounds)}",
        f"- All invariants passed: {summary['all_invariants_passed']}",
        "",
        "## Scenarios",
    ]
    for result in results:
        lines.extend([
            "",
            f"### {result['scenario']}",
            f"- Orders built: {result['orders_built']}",
            f"- Approved local simulated orders: {result['approved_orders']}",
            f"- Invariant failures: {result['invariant_failure_count']}",
            f"- Top risk failures: {list(result['risk_failure_counts'].items())[:5]}",
        ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "summary": str(out_json),
        "markdown": str(out_md),
        "all_invariants_passed": summary["all_invariants_passed"],
        "scenario_count": len(results),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
