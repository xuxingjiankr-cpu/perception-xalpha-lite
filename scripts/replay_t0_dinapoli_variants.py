"""Replay DiNapoli confirmation variants against T+0 ETF decisions.

Offline-only. This compares baseline T+0 replay behavior with several
DiNapoli-style structural confirmation overlays:

1. baseline
2. patient_exit
3. dinapoli_entry_filter
4. dinapoli_sell_delay
5. dinapoli_entry_and_sell

No paper-trading API calls are made. No production config is modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, as_float, load_json, write_json

import run_t0_intraday_agent as agent
from replay_t0_decisions import (
    FAKE_PENDING,
    INITIAL_CASH,
    apply_buy_fill,
    apply_sell_fill,
    fake_balance,
    fake_positions,
    group_rounds,
    load_rows,
    record_post_submit,
)
from replay_t0_dinapoli_shadow import DinapoliConfig, compute_dinapoli


DEFAULT_CONFIG = ROOT / "outputs" / "t0_replay" / "config_replay_20etf_20260615.json"
DEFAULT_QUOTES = ROOT / "outputs" / "t0_replay" / "replay_20etf_20260615_quotes.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "t0_replay"

PATIENT_EXIT_OVERLAY = {
    "min_hold_minutes": 20,
    "loss_review_after_minutes": 25,
    "loss_exit_score_threshold": 82,
    "profit_exit_score_threshold": 75,
    "profit_trailing_drawdown_pct": -0.007,
    "deceleration_exit_threshold": -0.003,
}

HARD_SELL_REASONS = {
    "kill_switch_liquidation",
    "emergency_stop_exit",
    "unrecognized_position_state_liquidation",
    "bracket_stop_loss",
    "breakeven_stop_after_t1",
    "force_close_pre_market_close",
    "intraday_momentum_eod_exit",
}


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)
    return dst


def last_prices_from_history(history_by_code: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for code, rows in history_by_code.items():
        if rows:
            out[code] = as_float(rows[-1].get("currentPrice"), 0.0)
    return out


def quote_history_append(history_by_code: dict[str, list[dict[str, Any]]], quotes: list[dict[str, Any]]) -> None:
    for q in quotes:
        code = str(q.get("stockCode", "")).zfill(6)
        ts = agent.parse_iso_dt(q.get("timestamp"))
        px = as_float(q.get("currentPrice"), 0.0)
        if not code or ts is None or px <= 0:
            continue
        node = dict(q)
        node["_dt"] = ts
        history_by_code[code].append(node)


def dinapoli_for_order(
    order: dict[str, Any],
    history_by_code: dict[str, list[dict[str, Any]]],
    cfg: DinapoliConfig,
) -> dict[str, Any]:
    code = str(order.get("stockCode", "")).zfill(6)
    hist = history_by_code.get(code, [])
    if not hist:
        return {"available": False, "reason": "no_quote_history"}
    return compute_dinapoli(hist, len(hist) - 1, cfg)


def should_block_order(
    variant: str,
    order: dict[str, Any],
    dinapoli: dict[str, Any],
) -> tuple[bool, str]:
    direction = order.get("direction")
    reason = str(order.get("reason") or "")
    if direction == "buy" and variant in {"dinapoli_entry_filter", "dinapoli_entry_and_sell"}:
        if not bool(dinapoli.get("entry_confirmed")):
            return True, "dinapoli_entry_not_confirmed"

    if direction == "sell" and variant in {"dinapoli_sell_delay", "dinapoli_entry_and_sell"}:
        pnl_pct = as_float(order.get("pnl_pct"), 0.0)
        if pnl_pct < 0 and reason not in HARD_SELL_REASONS and not bool(dinapoli.get("sell_confirmed")):
            return True, "dinapoli_loss_sell_not_structurally_confirmed"

    return False, ""


def mark_to_market(sim: dict[str, Any], last_prices: dict[str, float]) -> dict[str, Any]:
    open_positions: dict[str, Any] = sim.get("positions", {})
    mtm_rows: list[dict[str, Any]] = []
    value = as_float(sim.get("cash"), 0.0)
    unrealized = 0.0
    for code, pos in open_positions.items():
        qty = int(as_float(pos.get("quantity"), 0.0))
        cost = as_float(pos.get("costPrice"), 0.0)
        last = as_float(last_prices.get(code), cost)
        mkt = qty * last
        pnl = qty * (last - cost)
        value += mkt
        unrealized += pnl
        mtm_rows.append({
            "stockCode": code,
            "quantity": qty,
            "costPrice": round(cost, 6),
            "lastPrice": round(last, 6),
            "marketValue": round(mkt, 2),
            "unrealizedPnl": round(pnl, 2),
            "unrealizedPct": round(last / cost - 1.0, 6) if cost > 0 else None,
        })
    return {
        "marked_final_assets": round(value, 2),
        "unrealized_pnl": round(unrealized, 2),
        "open_positions_marked": mtm_rows,
    }


def replay_variant(
    base_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    variant: str,
    dinapoli_cfg: DinapoliConfig,
    out_path: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if variant == "patient_exit":
        deep_merge(cfg.setdefault("strategy", {}), PATIENT_EXIT_OVERLAY)

    rounds = group_rounds(rows)
    state: dict[str, Any] = {}
    sim = {"cash": INITIAL_CASH, "positions": {}}
    indicator_history: list[dict[str, Any]] = []
    dinapoli_history_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lookback = int(cfg["strategy"]["lookback_minutes"])
    submit_seq = 0
    per_day: dict[str, dict[str, Any]] = {}
    fail_counter: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    trades: list[dict[str, Any]] = []
    blocked_orders: list[dict[str, Any]] = []
    order_events: list[dict[str, Any]] = []

    out_path.write_text("", encoding="utf-8")
    try:
        for rnd in rounds:
            ts = agent.parse_iso_dt(rnd[-1].get("timestamp"))
            if ts is None:
                continue
            replay_now = ts.astimezone(ZoneInfo("Asia/Shanghai"))
            agent.set_replay_now(replay_now)
            trade_date = replay_now.strftime("%Y-%m-%d")
            day = per_day.setdefault(trade_date, {
                "rounds": 0,
                "entries": 0,
                "exits": Counter(),
                "pnl": 0.0,
                "blocked": Counter(),
            })
            day["rounds"] += 1

            raw_quotes = [dict(q) for q in rnd]
            quotes = agent.compute_snapshot_momentum(raw_quotes, indicator_history, lookback, cfg["strategy"])
            market_correlation_stress = agent.compute_market_correlation_stress(
                indicator_history,
                quotes,
                cfg.get("strategy", {}).get("market_correlation_stress", {}),
            )
            indicator_history.extend(quotes)
            quote_history_append(dinapoli_history_by_code, quotes)

            agent.update_orb_state(state, trade_date, quotes, replay_now, True)
            total_assets = as_float(sim["cash"], 0.0) + sum(
                as_float(p.get("costPrice")) * as_float(p.get("quantity")) for p in sim["positions"].values()
            )
            decision = agent.build_decision(
                cfg,
                quotes,
                fake_balance(total_assets, sim["cash"]),
                fake_positions(sim["positions"]),
                FAKE_PENDING,
                state,
                market_correlation_stress,
                history=indicator_history,
            )
            actions[decision.get("action") or decision.get("state_machine", {}).get("action", "?")] += 1
            for check in decision.get("risk_checks", []):
                if not check.get("passed"):
                    fail_counter[check.get("name", "?")] += 1

            orders_to_fill: list[dict[str, Any]] = []
            decision_orders = decision.get("orders") or []
            if decision.get("approved_for_submit") and decision_orders:
                for order in decision_orders:
                    dinapoli = dinapoli_for_order(order, dinapoli_history_by_code, dinapoli_cfg)
                    blocked, block_reason = should_block_order(variant, order, dinapoli)
                    event = {
                        "timestamp": replay_now.isoformat(),
                        "variant": variant,
                        "direction": order.get("direction"),
                        "stockCode": order.get("stockCode"),
                        "name": order.get("name"),
                        "quantity": order.get("quantity"),
                        "price": order.get("price"),
                        "reason": order.get("reason"),
                        "pnl_pct": order.get("pnl_pct"),
                        "r_multiple": order.get("r_multiple"),
                        "dinapoli_available": dinapoli.get("available"),
                        "dinapoli_reason": dinapoli.get("reason"),
                        "entry_confirmed": dinapoli.get("entry_confirmed"),
                        "sell_confirmed": dinapoli.get("sell_confirmed"),
                        "would_delay_exit": dinapoli.get("would_delay_exit"),
                        "structure_invalidated": dinapoli.get("structure_invalidated"),
                        "blocked_by_variant": blocked,
                        "block_reason": block_reason,
                    }
                    event.update({
                        "dinapoli_current": dinapoli.get("current"),
                        "retracement_382": dinapoli.get("retracement_382"),
                        "retracement_618": dinapoli.get("retracement_618"),
                        "cop": dinapoli.get("cop"),
                        "op": dinapoli.get("op"),
                    })
                    order_events.append(event)
                    if blocked:
                        day["blocked"][block_reason] += 1
                        blocked_orders.append(event)
                    else:
                        orders_to_fill.append(order)

            for order in orders_to_fill:
                submit_seq += 1
                if order.get("direction") == "buy":
                    apply_buy_fill(sim, order)
                    day["entries"] += 1
                elif order.get("direction") == "sell":
                    pnl = apply_sell_fill(sim, order)
                    day["pnl"] += pnl
                    day["exits"][order.get("reason", "?")] += 1
                    trades.append({
                        "trade_date": trade_date,
                        "stockCode": order.get("stockCode"),
                        "reason": order.get("reason"),
                        "pnl": round(pnl, 2),
                        "r_multiple": order.get("r_multiple"),
                    })
                record_post_submit(state, cfg, decision, order, submit_seq)

            agent.append_jsonl(out_path, {
                "timestamp": replay_now.isoformat(),
                "variant": variant,
                "approved": bool(decision.get("approved_for_submit")),
                "original_orders": decision_orders,
                "filled_orders": orders_to_fill,
                "blocked_orders": [x for x in order_events if x.get("timestamp") == replay_now.isoformat() and x.get("blocked_by_variant")],
                "failed_checks": [c.get("name") for c in decision.get("risk_checks", []) if not c.get("passed")],
            })
    finally:
        agent.set_replay_now(None)

    last_prices = last_prices_from_history(dinapoli_history_by_code)
    realized = round(sum(d["pnl"] for d in per_day.values()), 2)
    mtm = mark_to_market(sim, last_prices)
    summary = {
        "variant": variant,
        "rounds_total": sum(d["rounds"] for d in per_day.values()),
        "per_day": {
            td: {
                "rounds": d["rounds"],
                "entries": d["entries"],
                "exits": dict(d["exits"]),
                "blocked": dict(d["blocked"]),
                "realized_pnl": round(d["pnl"], 2),
            } for td, d in sorted(per_day.items())
        },
        "entries": sum(d["entries"] for d in per_day.values()),
        "closed_trades": len(trades),
        "winning_trades": sum(1 for t in trades if as_float(t.get("pnl")) > 0),
        "losing_trades": sum(1 for t in trades if as_float(t.get("pnl")) < 0),
        "realized_pnl": realized,
        "unrealized_pnl": mtm["unrealized_pnl"],
        "realized_plus_mtm": round(realized + mtm["unrealized_pnl"], 2),
        "marked_final_assets": mtm["marked_final_assets"],
        "open_positions_at_end": {c: int(as_float(p.get("quantity"))) for c, p in sim["positions"].items()},
        "open_positions_marked": mtm["open_positions_marked"],
        "trades": trades,
        "blocked_orders": blocked_orders,
        "order_events": order_events,
        "top_blocking_checks": fail_counter.most_common(10),
        "actions": dict(actions),
    }
    return summary


def write_comparison_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "rounds_total",
        "entries",
        "closed_trades",
        "winning_trades",
        "losing_trades",
        "realized_pnl",
        "unrealized_pnl",
        "realized_plus_mtm",
        "marked_final_assets",
        "open_positions_at_end",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k) for k in fields})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DiNapoli Variant Replay Comparison",
        "",
        "Paper trading research only. Not live-ready, not a formal strategy, and not investment advice.",
        "",
        f"- Created at: {report['created_at']}",
        f"- Config: `{report['config_path']}`",
        f"- Quotes: `{report['quotes_path']}`",
        f"- Rounds: {report['rounds_total']}",
        "",
        "## Results",
        "",
        "| variant | entries | closed | realized | unrealized | realized+MTM | open positions |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in report["summaries"]:
        lines.append(
            f"| {s['variant']} | {s['entries']} | {s['closed_trades']} | "
            f"{s['realized_pnl']:.2f} | {s['unrealized_pnl']:.2f} | "
            f"{s['realized_plus_mtm']:.2f} | {json.dumps(s['open_positions_at_end'], ensure_ascii=False)} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `dinapoli_entry_filter` blocks BUY orders when no valid 38.2%-61.8% reaction/bounce confirmation exists.",
        "- `dinapoli_sell_delay` blocks non-hard loss sells when DiNapoli structure is not invalidated.",
        "- Hard sell reasons such as emergency stop, bracket stop, kill switch, and forced close are never blocked.",
        "- Synthetic replay bid/ask comes from minute bars, so this validates timing logic only, not real fill quality.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DiNapoli variant replay comparison")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--date", default="2026-06-15")
    parser.add_argument("--label", default="dinapoli_variant_compare_20260615_20etf")
    parser.add_argument("--lookback-bars", type=int, default=30)
    parser.add_argument("--reaction-window-bars", type=int, default=12)
    parser.add_argument("--min-swing-pct", type=float, default=0.003)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    quotes_path = Path(args.quotes)
    base_cfg = load_json(cfg_path)
    rows = load_rows(quotes_path, args.date)
    if not rows:
        raise SystemExit(f"no replay quote rows found: {quotes_path}")

    dinapoli_cfg = DinapoliConfig(
        lookback_bars=args.lookback_bars,
        reaction_window_bars=args.reaction_window_bars,
        min_swing_pct=args.min_swing_pct,
    )
    variants = [
        "baseline",
        "patient_exit",
        "dinapoli_entry_filter",
        "dinapoli_sell_delay",
        "dinapoli_entry_and_sell",
    ]

    out_dir = DEFAULT_OUT_DIR
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_path = out_dir / f"{args.label}_{variant}_decisions.jsonl"
        summaries.append(replay_variant(base_cfg, rows, variant, dinapoli_cfg, variant_path))

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_advice": False,
        "task": "dinapoli_variant_replay_comparison",
        "config_path": str(cfg_path),
        "quotes_path": str(quotes_path),
        "date": args.date,
        "rounds_total": group_rounds(rows).__len__(),
        "dinapoli_config": dinapoli_cfg.__dict__,
        "patient_exit_overlay": PATIENT_EXIT_OVERLAY,
        "variants": variants,
        "summaries": summaries,
        "note": "Offline replay over synthetic L1 quotes from minute bars; not a profitability claim.",
    }
    json_path = out_dir / f"{args.label}.json"
    csv_path = out_dir / f"{args.label}.csv"
    md_path = out_dir / f"{args.label}.md"
    write_json(json_path, report)
    write_comparison_csv(csv_path, summaries)
    write_markdown(md_path, report)
    print(json.dumps({
        "results": [
            {
                "variant": s["variant"],
                "entries": s["entries"],
                "closed_trades": s["closed_trades"],
                "realized_pnl": s["realized_pnl"],
                "unrealized_pnl": s["unrealized_pnl"],
                "realized_plus_mtm": s["realized_plus_mtm"],
                "open_positions_at_end": s["open_positions_at_end"],
            } for s in summaries
        ],
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
