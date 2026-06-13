"""Offline decision replayer for the T+0 intraday paper agent.

Feeds historical quote snapshots from minute_quotes.jsonl back through
build_decision() with a virtual clock, simulating fills locally. Pure
offline analysis: no SkillClient, no network, no order submission, and
it never touches outputs/t0_intraday_agent/t0_state.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, as_float, load_json

import run_t0_intraday_agent as agent

DEFAULT_QUOTES = ROOT / "outputs" / "t0_intraday_agent" / "minute_quotes.jsonl"
OUT_DIR = ROOT / "outputs" / "t0_replay"
INITIAL_CASH = 1_000_000.0


def load_rows(path: Path, date_filter: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or not obj.get("timestamp"):
            continue
        if date_filter and not str(obj["timestamp"]).startswith(date_filter):
            continue
        rows.append(obj)
    return rows


def group_rounds(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Rows are appended per agent run; start a new round on code repeat or >60s gap."""
    rounds: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_ts: datetime | None = None
    for row in rows:
        ts = agent.parse_iso_dt(row.get("timestamp"))
        code = str(row.get("stockCode", "")).zfill(6)
        gap = (ts - last_ts).total_seconds() if ts and last_ts else 0.0
        if current and (code in seen or gap > 60):
            rounds.append(current)
            current = []
            seen = set()
        current.append(row)
        seen.add(code)
        if ts:
            last_ts = ts
    if current:
        rounds.append(current)
    return rounds


def fake_balance(total: float, cash: float) -> dict[str, Any]:
    return {"ok": True, "data": {"totalAssets": total, "availableBalance": cash}}


def fake_positions(positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "data": {"positions": list(positions.values())}}


FAKE_PENDING = {"ok": True, "data": {"orders": []}}


def apply_buy_fill(sim: dict[str, Any], order: dict[str, Any]) -> None:
    code = str(order.get("stockCode", "")).zfill(6)
    qty = int(as_float(order.get("quantity")))
    px = as_float(order.get("price"))
    sim["cash"] -= px * qty
    pos = sim["positions"].get(code)
    if pos is None:
        pos = {"stockCode": code, "stockName": order.get("name"), "exchange": order.get("exchange", "SH"),
               "quantity": 0, "availableQuantity": 0, "costPrice": px}
        sim["positions"][code] = pos
    prev_qty = int(as_float(pos.get("quantity")))
    pos["costPrice"] = (as_float(pos.get("costPrice")) * prev_qty + px * qty) / (prev_qty + qty) if prev_qty + qty > 0 else px
    pos["quantity"] = prev_qty + qty
    pos["availableQuantity"] = int(as_float(pos.get("availableQuantity"))) + qty


def apply_sell_fill(sim: dict[str, Any], order: dict[str, Any]) -> float:
    code = str(order.get("stockCode", "")).zfill(6)
    qty = int(as_float(order.get("quantity")))
    px = as_float(order.get("price"))
    sim["cash"] += px * qty
    pos = sim["positions"].get(code)
    pnl = 0.0
    if pos:
        cost = as_float(order.get("cost_price"), as_float(pos.get("costPrice")))
        pnl = (px - cost) * qty
        pos["quantity"] = int(as_float(pos.get("quantity"))) - qty
        pos["availableQuantity"] = max(0, int(as_float(pos.get("availableQuantity"))) - qty)
        if pos["quantity"] <= 0:
            del sim["positions"][code]
    return pnl


def record_post_submit(state: dict[str, Any], cfg: dict[str, Any], decision: dict[str, Any],
                       order: dict[str, Any], submit_seq: int) -> None:
    """Mirror the post-submit state updates of run_agent (bracket meta, cooldowns, stop list)."""
    trade_date = decision["trade_date"]
    fake_submit = {"ok": True, "data": {"orderId": f"REPLAY{submit_seq}"}}
    agent.increment_daily_state(state, "submitted_orders_by_date", trade_date)
    state["last_submit_at"] = agent.now_iso()
    if order.get("direction") == "buy":
        agent.record_t0_buy_submission(state, trade_date, order, fake_submit)
        bm = order.get("bracket")
        if bm:
            bcode = str(order.get("stockCode", "")).zfill(6)
            inv_node = agent.t0_inventory_for_code(state, trade_date, bcode)
            inv_node.setdefault("r_value", bm.get("r_value"))
            inv_node.setdefault("stop_price", bm.get("stop_price"))
            inv_node.setdefault("target1_price", bm.get("target1_price"))
            inv_node.setdefault("target2_price", bm.get("target2_price"))
            inv_node.setdefault("t1_filled", False)
            inv_node.setdefault("stop_moved_to_breakeven", False)
        agent.increment_daily_state(state, "entries_by_date", trade_date)
    elif order.get("direction") == "sell":
        agent.record_t0_sell_submission(state, trade_date, order, fake_submit)
        sell_code = str(order.get("stockCode", "")).zfill(6)
        if order.get("reason") == "bracket_t1_partial_exit":
            inv_node = agent.t0_inventory_for_code(state, trade_date, sell_code)
            inv_node["t1_filled"] = True
            inv_node["stop_moved_to_breakeven"] = True
        last_sell = state.get("last_sell_at_by_code")
        if not isinstance(last_sell, dict):
            last_sell = {}
        last_sell[sell_code] = agent.now_iso()
        state["last_sell_at_by_code"] = last_sell
        stop_reasons = {"confirmed_loss_score_exit", "emergency_stop_exit", "bracket_stop_loss", "breakeven_stop_after_t1"}
        reason_is_stop = order.get("reason") in stop_reasons or (
            order.get("reason") == "unified_sell_score_exit" and as_float(order.get("pnl_pct"), 0.0) < 0
        )
        bracket_cfg = cfg.get("strategy", {}).get("bracket", {})
        if bool(bracket_cfg.get("no_reentry_after_stop_loss_same_day", True)) and reason_is_stop:
            stopped_map = state.get("stopped_out_today_by_date")
            if not isinstance(stopped_map, dict):
                stopped_map = {}
            today_stopped = stopped_map.get(trade_date, [])
            if not isinstance(today_stopped, list):
                today_stopped = []
            if sell_code not in today_stopped:
                today_stopped.append(sell_code)
            stopped_map[trade_date] = today_stopped
            state["stopped_out_today_by_date"] = stopped_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay of T+0 agent decisions (no API, no orders)")
    parser.add_argument("--config", default=str(agent.DEFAULT_CONFIG))
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--date", default=None, help="replay a single trade date YYYY-MM-DD")
    parser.add_argument("--label", default="replay", help="suffix for output filenames")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    rows = load_rows(Path(args.quotes), args.date)
    rounds = group_rounds(rows)
    if not rounds:
        print("no quote rounds found")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions_path = OUT_DIR / f"{args.label}_decisions.jsonl"
    summary_path = OUT_DIR / f"{args.label}_summary.json"
    decisions_path.write_text("", encoding="utf-8")

    state: dict[str, Any] = {}
    sim = {"cash": INITIAL_CASH, "positions": {}}
    history: list[dict[str, Any]] = []
    lookback = int(cfg["strategy"]["lookback_minutes"])
    submit_seq = 0
    per_day: dict[str, dict[str, Any]] = {}
    fail_counter: Counter[str] = Counter()
    trades: list[dict[str, Any]] = []

    try:
        for rnd in rounds:
            ts = agent.parse_iso_dt(rnd[-1].get("timestamp"))
            if ts is None:
                continue
            replay_now = ts.astimezone(ZoneInfo("Asia/Shanghai"))
            agent.set_replay_now(replay_now)
            trade_date = replay_now.strftime("%Y-%m-%d")
            day = per_day.setdefault(trade_date, {
                "rounds": 0, "entries": 0, "exits": Counter(), "pnl": 0.0,
                "entry_score_pass": 0, "actions": Counter(),
            })
            day["rounds"] += 1

            raw_quotes = [dict(q) for q in rnd]
            quotes = agent.compute_snapshot_momentum(raw_quotes, history, lookback, cfg["strategy"])
            history.extend(quotes)

            session = {"in_regular_session": True}
            agent.update_orb_state(state, trade_date, quotes, replay_now, True)

            total_assets = sim["cash"] + sum(
                as_float(p.get("costPrice")) * as_float(p.get("quantity")) for p in sim["positions"].values()
            )
            decision = agent.build_decision(
                cfg, quotes,
                fake_balance(total_assets, sim["cash"]),
                fake_positions(sim["positions"]),
                FAKE_PENDING,
                state,
            )
            day["actions"][decision.get("action") or decision.get("state_machine", {}).get("action", "?")] += 1
            for c in decision.get("risk_checks", []):
                if not c.get("passed"):
                    fail_counter[c.get("name", "?")] += 1
                if c.get("name") == "entry_score_gate" and c.get("passed"):
                    day["entry_score_pass"] += 1

            orders = decision.get("orders") or []
            if decision.get("approved_for_submit") and orders:
                for order in orders:
                    submit_seq += 1
                    if order.get("direction") == "buy":
                        apply_buy_fill(sim, order)
                        day["entries"] += 1
                    else:
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

            agent.append_jsonl(decisions_path, {
                "timestamp": replay_now.isoformat(),
                "action": orders[0]["direction"] if orders else "hold",
                "reason": orders[0].get("reason") if orders else decision.get("state_machine", {}).get("reason"),
                "approved": bool(decision.get("approved_for_submit")),
                "sell_score": decision.get("sell_score"),
                "carry_allowed": decision.get("carry_allowed"),
                "failed_checks": [c.get("name") for c in decision.get("risk_checks", []) if not c.get("passed")],
                "orders": orders,
            })
    finally:
        agent.set_replay_now(None)

    final_assets = sim["cash"] + sum(
        as_float(p.get("costPrice")) * as_float(p.get("quantity")) for p in sim["positions"].values()
    )
    summary = {
        "label": args.label,
        "rounds_total": sum(d["rounds"] for d in per_day.values()),
        "per_day": {
            td: {
                "rounds": d["rounds"],
                "entries": d["entries"],
                "entry_score_pass_rounds": d["entry_score_pass"],
                "exits": dict(d["exits"]),
                "pnl": round(d["pnl"], 2),
            } for td, d in sorted(per_day.items())
        },
        "trades": trades,
        "r_multiples": [t["r_multiple"] for t in trades if t.get("r_multiple") is not None],
        "total_pnl": round(sum(d["pnl"] for d in per_day.values()), 2),
        "final_assets": round(final_assets, 2),
        "open_positions_at_end": {c: int(as_float(p.get("quantity"))) for c, p in sim["positions"].items()},
        "top_blocking_checks": fail_counter.most_common(10),
        "note": "simulated fills at limit price; small sample; not a profitability claim",
    }
    agent.write_json(summary_path, summary)

    print(f"rounds: {summary['rounds_total']}")
    for td, d in summary["per_day"].items():
        print(f"{td}: rounds={d['rounds']} entries={d['entries']} score_pass={d['entry_score_pass_rounds']} exits={d['exits']} pnl={d['pnl']}")
    print(f"total_pnl={summary['total_pnl']} final_assets={summary['final_assets']} open_at_end={summary['open_positions_at_end']}")
    print("top blocking checks:", summary["top_blocking_checks"][:5])
    print(f"outputs: {decisions_path} | {summary_path}")


if __name__ == "__main__":
    main()
