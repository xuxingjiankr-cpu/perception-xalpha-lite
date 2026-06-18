"""Offline decision replayer for the T+0 intraday paper agent.

Feeds historical quote snapshots from legacy or daily minute quote logs back through
build_decision() with a virtual clock, simulating fills locally. Pure
offline analysis: no SkillClient, no network, no order submission, and
it never touches outputs/t0_intraday_agent/t0_state.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, as_float, load_json

import run_t0_intraday_agent as agent

DEFAULT_QUOTES = ROOT / "outputs" / "t0_intraday_agent"
OUT_DIR = ROOT / "outputs" / "t0_replay"
INITIAL_CASH = 1_000_000.0


def _date_allowed(ts_text: str, date_filter: str | None, start_date: str | None, end_date: str | None) -> bool:
    trade_date = ts_text[:10]
    if date_filter and trade_date != date_filter:
        return False
    if start_date and trade_date < start_date:
        return False
    if end_date and trade_date > end_date:
        return False
    return True


def quote_source_paths(path: Path, date_filter: str | None = None) -> list[Path]:
    if not path.exists():
        return []
    if path.is_dir():
        # Agent output directories also contain large non-quote JSONL logs. Prefer
        # the legacy/daily minute quote family there, while retaining support for
        # evolution cache directories whose files are simply YYYY-MM-DD.jsonl.
        candidates = sorted(path.glob("minute_quotes*.jsonl"))
        if not candidates:
            candidates = sorted(path.glob("*.jsonl"))
        if date_filter:
            matched = [candidate for candidate in candidates if date_filter in candidate.name]
            if matched:
                return matched
        return candidates
    return [path]


def iter_quote_rows(
    path: Path,
    date_filter: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[dict[str, Any]]:
    for source in quote_source_paths(path, date_filter):
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict) or not obj.get("timestamp"):
                    continue
                ts_text = str(obj["timestamp"])
                if not _date_allowed(ts_text, date_filter, start_date, end_date):
                    continue
                yield obj


def load_rows(path: Path, date_filter: str | None) -> list[dict[str, Any]]:
    return list(iter_quote_rows(path, date_filter))


def iter_rounds(rows: Iterator[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Rows are appended per agent run; start a new round on code repeat or >60s gap."""
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_ts: datetime | None = None
    for row in rows:
        ts = agent.parse_iso_dt(row.get("timestamp"))
        code = str(row.get("stockCode", "")).zfill(6)
        gap = (ts - last_ts).total_seconds() if ts and last_ts else 0.0
        if current and (code in seen or gap > 60):
            yield current
            current = []
            seen = set()
        current.append(row)
        seen.add(code)
        if ts:
            last_ts = ts
    if current:
        yield current


def group_rounds(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return list(iter_rounds(iter(rows)))


def fake_balance(total: float, cash: float) -> dict[str, Any]:
    return {"ok": True, "data": {"totalAssets": total, "availableBalance": cash}}


def fake_positions(positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "data": {"positions": list(positions.values())}}


FAKE_PENDING = {"ok": True, "data": {"orders": []}}


def apply_buy_fill(sim: dict[str, Any], order: dict[str, Any]) -> None:
    code = str(order.get("stockCode", "")).zfill(6)
    qty = int(as_float(order.get("quantity")))
    px = as_float(order.get("price"))
    notional = px * qty
    sim["cash"] -= notional
    sim["buy_notional"] = as_float(sim.get("buy_notional")) + notional
    pos = sim["positions"].get(code)
    if pos is None:
        pos = {"stockCode": code, "stockName": order.get("name"), "exchange": order.get("exchange", "SH"),
               "quantity": 0, "availableQuantity": 0, "costPrice": px}
        sim["positions"][code] = pos
    prev_qty = int(as_float(pos.get("quantity")))
    pos["costPrice"] = (as_float(pos.get("costPrice")) * prev_qty + px * qty) / (prev_qty + qty) if prev_qty + qty > 0 else px
    pos["quantity"] = prev_qty + qty
    pos["availableQuantity"] = int(as_float(pos.get("availableQuantity"))) + qty
    sim["_last_fill"] = {"notional": notional, "quantity": float(qty), "price": px}


def apply_sell_fill(sim: dict[str, Any], order: dict[str, Any]) -> float:
    code = str(order.get("stockCode", "")).zfill(6)
    qty = int(as_float(order.get("quantity")))
    px = as_float(order.get("price"))
    notional = px * qty
    sim["cash"] += notional
    sim["sell_notional"] = as_float(sim.get("sell_notional")) + notional
    pos = sim["positions"].get(code)
    pnl = 0.0
    if pos:
        cost = as_float(order.get("cost_price"), as_float(pos.get("costPrice")))
        pnl = (px - cost) * qty
        pos["quantity"] = int(as_float(pos.get("quantity"))) - qty
        pos["availableQuantity"] = max(0, int(as_float(pos.get("availableQuantity"))) - qty)
        if pos["quantity"] <= 0:
            del sim["positions"][code]
    sim["_last_fill"] = {
        "gross_pnl": pnl,
        "quantity": float(qty),
        "entry_price": cost if pos else 0.0,
        "exit_price": px,
        "entry_notional": cost * qty if pos else 0.0,
        "exit_notional": notional,
    }
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
        if order.get("im_trade"):
            bcode = str(order.get("stockCode", "")).zfill(6)
            inv_node = agent.t0_inventory_for_code(state, trade_date, bcode)
            inv_node["im_trade"] = True
            agent.increment_daily_state(state, "intraday_momentum_by_date", trade_date)
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
    parser.add_argument("--start-date", default=None, help="inclusive replay start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="inclusive replay end date YYYY-MM-DD")
    parser.add_argument("--label", default="replay", help="suffix for output filenames")
    parser.add_argument("--output-detail", choices=["summary", "full"], default="full",
                        help="summary skips per-round decisions JSONL; full preserves legacy output")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    round_iter = iter_rounds(iter_quote_rows(Path(args.quotes), args.date, args.start_date, args.end_date))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions_path = OUT_DIR / f"{args.label}_decisions.jsonl"
    summary_path = OUT_DIR / f"{args.label}_summary.json"
    if args.output_detail == "full":
        decisions_path.write_text("", encoding="utf-8")

    state: dict[str, Any] = {}
    sim = {"cash": INITIAL_CASH, "positions": {}, "buy_notional": 0.0, "sell_notional": 0.0}
    history: list[dict[str, Any]] = []
    history_trade_date: str | None = None
    history_universe_max = 0
    lookback = int(cfg["strategy"]["lookback_minutes"])
    indicator_windows = [
        lookback,
        120,
        int(as_float(cfg.get("strategy", {}).get("market_correlation_stress", {}).get("window_snapshots"), 30)),
    ]
    max_history_snapshots = max(indicator_windows)
    submit_seq = 0
    per_day: dict[str, dict[str, Any]] = {}
    fail_counter: Counter[str] = Counter()
    trades: list[dict[str, Any]] = []
    rows_processed = 0
    feature_seconds = 0.0
    signal_seconds = 0.0
    file_write_seconds = 0.0
    replay_started = time.perf_counter()

    try:
        for rnd in round_iter:
            ts = agent.parse_iso_dt(rnd[-1].get("timestamp"))
            if ts is None:
                continue
            replay_now = ts.astimezone(ZoneInfo("Asia/Shanghai"))
            agent.set_replay_now(replay_now)
            trade_date = replay_now.strftime("%Y-%m-%d")
            if history_trade_date != trade_date:
                history = []
                history_trade_date = trade_date
                history_universe_max = 0
            day = per_day.setdefault(trade_date, {
                "rounds": 0, "entries": 0, "exits": Counter(), "pnl": 0.0,
                "entry_score_pass": 0, "actions": Counter(),
                "buy_notional": 0.0, "sell_notional": 0.0,
            })
            day["rounds"] += 1
            rows_processed += len(rnd)

            raw_quotes = [dict(q) for q in rnd]
            feature_started = time.perf_counter()
            quotes = agent.compute_snapshot_momentum(raw_quotes, history, lookback, cfg["strategy"])
            market_correlation_stress = agent.compute_market_correlation_stress(
                history,
                quotes,
                cfg.get("strategy", {}).get("market_correlation_stress", {}),
            )
            feature_seconds += time.perf_counter() - feature_started
            history.extend(quotes)
            history_universe_max = max(history_universe_max, len(quotes))
            history_row_cap = max_history_snapshots * max(1, history_universe_max)
            if len(history) > history_row_cap:
                history = history[-history_row_cap:]

            session = {"in_regular_session": True}
            agent.update_orb_state(state, trade_date, quotes, replay_now, True)

            total_assets = sim["cash"] + sum(
                as_float(p.get("costPrice")) * as_float(p.get("quantity")) for p in sim["positions"].values()
            )
            signal_started = time.perf_counter()
            decision = agent.build_decision(
                cfg, quotes,
                fake_balance(total_assets, sim["cash"]),
                fake_positions(sim["positions"]),
                FAKE_PENDING,
                state,
                market_correlation_stress,
                history=history,
            )
            signal_seconds += time.perf_counter() - signal_started
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
                        fill = sim["_last_fill"]
                        day["buy_notional"] += fill["notional"]
                        day["entries"] += 1
                    else:
                        pnl = apply_sell_fill(sim, order)
                        fill = sim["_last_fill"]
                        day["sell_notional"] += fill["exit_notional"]
                        day["pnl"] += pnl
                        day["exits"][order.get("reason", "?")] += 1
                        trades.append({
                            "trade_date": trade_date,
                            "stockCode": order.get("stockCode"),
                            "reason": order.get("reason"),
                            "pnl": round(pnl, 2),
                            "gross_pnl": round(pnl, 2),
                            "quantity": int(fill["quantity"]),
                            "entry_price": round(fill["entry_price"], 6),
                            "exit_price": round(fill["exit_price"], 6),
                            "entry_notional": round(fill["entry_notional"], 2),
                            "exit_notional": round(fill["exit_notional"], 2),
                            "r_multiple": order.get("r_multiple"),
                        })
                    record_post_submit(state, cfg, decision, order, submit_seq)

            if args.output_detail == "full":
                write_started = time.perf_counter()
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
                file_write_seconds += time.perf_counter() - write_started
    finally:
        agent.set_replay_now(None)

    if not per_day:
        print("no quote rounds found")
        return

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
                "gross_pnl": round(d["pnl"], 2),
                "buy_notional": round(d["buy_notional"], 2),
                "sell_notional": round(d["sell_notional"], 2),
            } for td, d in sorted(per_day.items())
        },
        "trades": trades,
        "r_multiples": [t["r_multiple"] for t in trades if t.get("r_multiple") is not None],
        "total_pnl": round(sum(d["pnl"] for d in per_day.values()), 2),
        "gross_pnl": round(sum(d["pnl"] for d in per_day.values()), 2),
        "buy_notional": round(as_float(sim.get("buy_notional")), 2),
        "sell_notional": round(as_float(sim.get("sell_notional")), 2),
        "turnover": round(
            (as_float(sim.get("buy_notional")) + as_float(sim.get("sell_notional"))) / (2.0 * INITIAL_CASH),
            6,
        ),
        "final_assets": round(final_assets, 2),
        "open_positions_at_end": {c: int(as_float(p.get("quantity"))) for c, p in sim["positions"].items()},
        "top_blocking_checks": fail_counter.most_common(10),
        "runtime_profile": {
            "total_runtime_seconds": round(time.perf_counter() - replay_started, 6),
            "feature_calculation_seconds": round(feature_seconds, 6),
            "signal_generation_seconds": round(signal_seconds, 6),
            "file_write_seconds": round(file_write_seconds, 6),
            "rows_processed": rows_processed,
            "output_detail": args.output_detail,
        },
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
