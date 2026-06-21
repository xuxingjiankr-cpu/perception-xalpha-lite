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
from datetime import datetime, timedelta
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


def replay_execution_settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (cfg or {}).get("replay_execution", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "same_snapshot_fill": False,
        "next_snapshot_fill": True,
        "cost_in_path": True,
        "mark_to_market": True,
        "t_rule_enforced": True,
        "default_sell_rule": str(raw.get("default_sell_rule", "T1")).upper(),
        "t0_allowlist": {str(x).zfill(6) for x in raw.get("t0_allowlist", []) if str(x).strip()},
        "buy_cost_pct": as_float(raw.get("buy_cost_pct"), 0.0005),
        "sell_cost_pct": as_float(raw.get("sell_cost_pct"), 0.0005),
        "slippage_pct_per_side": as_float(raw.get("slippage_pct_per_side"), 0.0002),
        "order_time_in_force_snapshots": max(1, int(as_float(raw.get("order_time_in_force_snapshots"), 1))),
    }


def _ensure_sim(sim: dict[str, Any]) -> dict[str, Any]:
    sim.setdefault("initial_cash", as_float(sim.get("cash"), INITIAL_CASH))
    sim.setdefault("cash", as_float(sim.get("initial_cash"), INITIAL_CASH))
    sim.setdefault("positions", {})
    sim.setdefault("buy_notional", 0.0)
    sim.setdefault("sell_notional", 0.0)
    sim.setdefault("transaction_cost_total", 0.0)
    sim.setdefault("gross_realized_pnl", 0.0)
    sim.setdefault("net_realized_pnl", 0.0)
    sim.setdefault("last_prices", {})
    sim.setdefault("equity_peak", as_float(sim.get("initial_cash"), INITIAL_CASH))
    sim.setdefault("max_drawdown", 0.0)
    sim.setdefault("equity_curve", [])
    sim.setdefault("lots", {})
    return sim


def _parse_replay_dt(value: Any) -> datetime:
    parsed = agent.parse_iso_dt(value)
    if parsed is not None:
        return parsed.astimezone(ZoneInfo("Asia/Shanghai"))
    return datetime(2000, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))


def _available_sell_time(fill_time: datetime, code: str, settings: dict[str, Any]) -> datetime:
    if code in settings["t0_allowlist"]:
        return fill_time
    next_date = fill_time.date() + timedelta(days=1)
    return datetime.combine(next_date, datetime.min.time(), tzinfo=ZoneInfo("Asia/Shanghai"))


def _position_lots(sim: dict[str, Any], code: str) -> list[dict[str, Any]]:
    lots = sim.setdefault("lots", {}).setdefault(code, [])
    return lots if isinstance(lots, list) else []


def _rebuild_position(sim: dict[str, Any], code: str, now: datetime | None = None) -> None:
    pos = sim.get("positions", {}).get(code)
    if not isinstance(pos, dict):
        return
    lots = [lot for lot in _position_lots(sim, code) if int(as_float(lot.get("remaining_qty"))) > 0]
    sim["lots"][code] = lots
    qty = sum(int(as_float(lot.get("remaining_qty"))) for lot in lots)
    if qty <= 0:
        sim["positions"].pop(code, None)
        sim["lots"].pop(code, None)
        return
    pos["quantity"] = qty
    pos["costPrice"] = sum(as_float(lot.get("net_unit_cost")) * int(as_float(lot.get("remaining_qty"))) for lot in lots) / qty
    pos["grossCostPrice"] = sum(as_float(lot.get("gross_unit_cost")) * int(as_float(lot.get("remaining_qty"))) for lot in lots) / qty
    if now is not None:
        pos["availableQuantity"] = sum(
            int(as_float(lot.get("remaining_qty")))
            for lot in lots
            if _parse_replay_dt(lot.get("available_sell_time")) <= now
        )


def refresh_available_quantities(sim: dict[str, Any], now: datetime) -> None:
    _ensure_sim(sim)
    for code in list(sim["positions"]):
        _rebuild_position(sim, code, now)


def _consume_lots(sim: dict[str, Any], code: str, qty: int, now: datetime | None = None,
                  enforce_available: bool = False) -> tuple[float, float, int]:
    remaining = max(0, int(qty))
    gross_basis = 0.0
    net_basis = 0.0
    consumed = 0
    for lot in _position_lots(sim, code):
        if remaining <= 0:
            break
        lot_qty = int(as_float(lot.get("remaining_qty")))
        if lot_qty <= 0:
            continue
        if enforce_available and now is not None and _parse_replay_dt(lot.get("available_sell_time")) > now:
            continue
        take = min(remaining, lot_qty)
        lot["remaining_qty"] = lot_qty - take
        remaining -= take
        consumed += take
        gross_basis += take * as_float(lot.get("gross_unit_cost"))
        net_basis += take * as_float(lot.get("net_unit_cost"))
    return gross_basis, net_basis, consumed


def extend_replay_universe(cfg: dict[str, Any], quotes: list[dict[str, Any]]) -> int:
    """Make dynamically selected quote codes visible to position/exit logic.

    Live runs replace the static seed with the resolved dynamic universe before
    calling build_decision. Offline replay receives quote rounds directly, so it
    must mirror that step or dynamic buys become invisible holdings that can
    never be evaluated for exit.
    """
    universe = cfg.get("universe")
    if not isinstance(universe, list):
        universe = []
        cfg["universe"] = universe
    known = {str(item.get("stockCode", "")).zfill(6) for item in universe if isinstance(item, dict)}
    added = 0
    for quote in quotes:
        code = str(quote.get("stockCode", "")).zfill(6)
        if not code or code in known:
            continue
        universe.append({
            "stockCode": code,
            "exchange": str(quote.get("exchange") or "SH").upper(),
            "name": quote.get("name") or code,
            "asset_class": quote.get("asset_class") or "dynamic",
        })
        known.add(code)
        added += 1
    return added


def apply_buy_fill(sim: dict[str, Any], order: dict[str, Any], *, fill_price: float | None = None,
                   fill_time: datetime | None = None, cfg: dict[str, Any] | None = None) -> None:
    _ensure_sim(sim)
    settings = replay_execution_settings(cfg)
    code = str(order.get("stockCode", "")).zfill(6)
    qty = int(as_float(order.get("quantity")))
    px = as_float(fill_price, as_float(order.get("price")))
    when = fill_time or _parse_replay_dt(order.get("fill_time") or order.get("timestamp"))
    notional = px * qty
    cash_cost = notional * (1.0 + settings["buy_cost_pct"] + settings["slippage_pct_per_side"])
    fee = cash_cost - notional
    sim["cash"] -= cash_cost
    sim["buy_notional"] = as_float(sim.get("buy_notional")) + notional
    sim["transaction_cost_total"] = as_float(sim.get("transaction_cost_total")) + fee
    pos = sim["positions"].get(code)
    if pos is None:
        pos = {"stockCode": code, "stockName": order.get("name"), "exchange": order.get("exchange", "SH"),
               "quantity": 0, "availableQuantity": 0, "costPrice": px, "grossCostPrice": px}
        sim["positions"][code] = pos
    available_time = _available_sell_time(when, code, settings)
    _position_lots(sim, code).append({
        "buy_time": when.isoformat(), "qty": qty, "remaining_qty": qty,
        "available_sell_time": available_time.isoformat(),
        "sell_rule": "T0" if code in settings["t0_allowlist"] else settings["default_sell_rule"],
        "gross_unit_cost": px, "net_unit_cost": cash_cost / qty if qty > 0 else px,
    })
    _rebuild_position(sim, code, when)
    sim["_last_fill"] = {
        "notional": notional, "cash_effect": -cash_cost, "transaction_cost": fee,
        "quantity": float(qty), "price": px, "fill_time": when.isoformat(),
        "available_sell_time": available_time.isoformat(),
    }


def apply_sell_fill(sim: dict[str, Any], order: dict[str, Any], *, fill_price: float | None = None,
                    fill_time: datetime | None = None, cfg: dict[str, Any] | None = None,
                    enforce_available: bool = False) -> float:
    _ensure_sim(sim)
    settings = replay_execution_settings(cfg)
    code = str(order.get("stockCode", "")).zfill(6)
    qty = int(as_float(order.get("quantity")))
    px = as_float(fill_price, as_float(order.get("price")))
    when = fill_time or _parse_replay_dt(order.get("fill_time") or order.get("timestamp"))
    notional = px * qty
    pos = sim["positions"].get(code)
    fallback_cost = as_float(order.get("cost_price"), as_float(pos.get("costPrice")) if pos else 0.0)
    gross_basis, net_basis, consumed = _consume_lots(sim, code, qty, when, enforce_available)
    if consumed <= 0 and pos and not sim.get("lots", {}).get(code):
        consumed = min(qty, int(as_float(pos.get("quantity"))))
        gross_basis = fallback_cost * consumed
        net_basis = fallback_cost * consumed
        pos["quantity"] = int(as_float(pos.get("quantity"))) - consumed
    qty = consumed
    notional = px * qty
    cash_proceeds = notional * (1.0 - settings["sell_cost_pct"] - settings["slippage_pct_per_side"])
    fee = notional - cash_proceeds
    sim["cash"] += cash_proceeds
    sim["sell_notional"] = as_float(sim.get("sell_notional")) + notional
    sim["transaction_cost_total"] = as_float(sim.get("transaction_cost_total")) + fee
    gross_pnl = notional - gross_basis
    net_pnl = cash_proceeds - net_basis
    sim["gross_realized_pnl"] = as_float(sim.get("gross_realized_pnl")) + gross_pnl
    sim["net_realized_pnl"] = as_float(sim.get("net_realized_pnl")) + net_pnl
    _rebuild_position(sim, code, when)
    sim["_last_fill"] = {
        "gross_pnl": gross_pnl, "net_pnl": net_pnl,
        "transaction_cost": fee, "cash_effect": cash_proceeds,
        "quantity": float(qty),
        "entry_price": gross_basis / qty if qty > 0 else fallback_cost,
        "net_entry_price": net_basis / qty if qty > 0 else fallback_cost,
        "exit_price": px,
        "entry_notional": gross_basis,
        "exit_notional": notional,
        "fill_time": when.isoformat(),
    }
    return gross_pnl


def mark_to_market(sim: dict[str, Any], quotes: list[dict[str, Any]], timestamp: datetime) -> dict[str, float]:
    _ensure_sim(sim)
    for quote in quotes:
        code = str(quote.get("stockCode", "")).zfill(6)
        price = as_float(quote.get("currentPrice"), 0.0)
        if code and price > 0:
            sim["last_prices"][code] = price
    refresh_available_quantities(sim, timestamp)
    market_value = 0.0
    gross_basis = 0.0
    net_basis = 0.0
    for code, pos in sim["positions"].items():
        qty = int(as_float(pos.get("quantity")))
        mark = as_float(sim["last_prices"].get(code), as_float(pos.get("grossCostPrice")))
        market_value += mark * qty
        for lot in _position_lots(sim, code):
            lot_qty = int(as_float(lot.get("remaining_qty")))
            gross_basis += as_float(lot.get("gross_unit_cost")) * lot_qty
            net_basis += as_float(lot.get("net_unit_cost")) * lot_qty
    gross_unrealized = market_value - gross_basis
    unrealized = market_value - net_basis
    equity = as_float(sim.get("cash")) + market_value
    peak = max(as_float(sim.get("equity_peak"), equity), equity)
    drawdown = equity / peak - 1.0 if peak > 0 else 0.0
    sim.update({
        "market_value": market_value, "gross_unrealized_pnl": gross_unrealized,
        "unrealized_pnl": unrealized, "total_equity": equity,
        "equity_peak": peak, "drawdown": drawdown,
        "max_drawdown": min(as_float(sim.get("max_drawdown"), 0.0), drawdown),
    })
    point = {
        "timestamp": timestamp.isoformat(), "cash": round(as_float(sim.get("cash")), 2),
        "market_value": round(market_value, 2), "equity": round(equity, 2),
        "realized_pnl": round(as_float(sim.get("net_realized_pnl")), 2),
        "unrealized_pnl": round(unrealized, 2), "drawdown": round(drawdown, 8),
    }
    sim["equity_curve"].append(point)
    return point


def make_order_intent(order: dict[str, Any], signal_time: datetime, sequence: int,
                      decision: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "order_id": f"REPLAY{sequence}", "signal_time": signal_time.isoformat(),
        "order_time": signal_time.isoformat(), "eligible_fill_time": None, "fill_time": None,
        "order_price": as_float(order.get("price")), "fill_price": None,
        "qty": int(as_float(order.get("quantity"))), "filled_qty": 0,
        "side": str(order.get("direction") or "").lower(), "status": "pending",
        "reject_reason": None, "expire_reason": None, "stockCode": str(order.get("stockCode", "")).zfill(6),
        "exchange": order.get("exchange"), "reason": order.get("reason"),
        "_order": dict(order), "_decision": decision or {},
    }


def public_order_record(intent: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in intent.items() if not k.startswith("_")}


def fake_pending_orders(pending: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "data": {"orders": [
        {"orderId": row["order_id"], "stockCode": row["stockCode"], "exchange": row.get("exchange"),
         "direction": row["side"], "quantity": row["qty"], "price": row["order_price"], "status": "pending"}
        for row in pending if row.get("status") == "pending"
    ]}}


def process_pending_orders(sim: dict[str, Any], pending: list[dict[str, Any]], quotes: list[dict[str, Any]],
                           now: datetime, cfg: dict[str, Any], state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate intents only on a later tradable snapshot. Returns (remaining, terminal)."""
    settings = replay_execution_settings(cfg)
    quote_by_code = {str(q.get("stockCode", "")).zfill(6): q for q in quotes}
    remaining: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    refresh_available_quantities(sim, now)
    for intent in pending:
        signal_time = _parse_replay_dt(intent.get("signal_time"))
        if now <= signal_time:
            remaining.append(intent)
            continue
        if now.date() != signal_time.date():
            intent["status"] = "expired"
            intent["expire_reason"] = "session_ended_before_eligible_fill"
            terminal.append(intent)
            continue
        quote = quote_by_code.get(intent["stockCode"])
        if quote is None:
            remaining.append(intent)
            continue
        intent["eligible_fill_time"] = now.isoformat()
        if quote.get("quote_ok") is False or bool(quote.get("isSuspended")):
            intent["status"] = "rejected"
            intent["reject_reason"] = "suspended_or_invalid_quote"
            terminal.append(intent)
            continue
        current = as_float(quote.get("currentPrice"), 0.0)
        has_flow = "volume" in quote or "amount" in quote
        no_flow = has_flow and as_float(quote.get("volume"), 0.0) <= 0 and as_float(quote.get("amount"), 0.0) <= 0
        if current <= 0 or no_flow:
            intent["status"] = "rejected"
            intent["reject_reason"] = "invalid_price_or_no_trade"
            terminal.append(intent)
            continue
        side = intent["side"]
        executable = as_float(quote.get("askPrice1"), current) if side == "buy" else as_float(quote.get("bidPrice1"), current)
        if executable <= 0:
            intent["status"] = "rejected"
            intent["reject_reason"] = "invalid_executable_price"
            terminal.append(intent)
            continue
        order = dict(intent["_order"])
        order_type = str(order.get("orderType") or order.get("order_type") or "limit").lower()
        limit_price = as_float(intent.get("order_price"))
        marketable = order_type == "market" or (side == "buy" and executable <= limit_price) or (side == "sell" and executable >= limit_price)
        if not marketable:
            intent["status"] = "expired"
            intent["expire_reason"] = "next_snapshot_limit_not_marketable"
            terminal.append(intent)
            continue
        requested_qty = int(as_float(intent.get("qty")))
        fill_qty = requested_qty
        if side == "sell":
            refresh_available_quantities(sim, now)
            pos = sim.get("positions", {}).get(intent["stockCode"], {})
            available = int(as_float(pos.get("availableQuantity"), 0.0)) if isinstance(pos, dict) else 0
            if available <= 0:
                intent["status"] = "rejected"
                intent["reject_reason"] = "insufficient_available_qty_t_rule"
                terminal.append(intent)
                continue
            fill_qty = min(requested_qty, available)
        else:
            required_cash = executable * requested_qty * (
                1.0 + settings["buy_cost_pct"] + settings["slippage_pct_per_side"]
            )
            if required_cash > as_float(sim.get("cash")):
                intent["status"] = "rejected"
                intent["reject_reason"] = "insufficient_cash_including_cost"
                terminal.append(intent)
                continue
        order["quantity"] = fill_qty
        if side == "buy":
            apply_buy_fill(sim, order, fill_price=executable, fill_time=now, cfg=cfg)
        else:
            apply_sell_fill(sim, order, fill_price=executable, fill_time=now, cfg=cfg, enforce_available=True)
            fill_qty = int(as_float(sim.get("_last_fill", {}).get("quantity")))
        if fill_qty <= 0:
            intent["status"] = "rejected"
            intent["reject_reason"] = "zero_fill_after_constraints"
            terminal.append(intent)
            continue
        intent["fill_time"] = now.isoformat()
        intent["fill_price"] = executable
        intent["filled_qty"] = fill_qty
        intent["_fill"] = dict(sim.get("_last_fill", {}))
        intent["status"] = "filled" if fill_qty == requested_qty else "partially_filled"
        if fill_qty < requested_qty:
            intent["expire_reason"] = "remaining_qty_exceeds_available_qty"
        record_post_submit(state, cfg, intent["_decision"], order, int(intent["order_id"].replace("REPLAY", "")))
        if side == "sell":
            trade_date = now.strftime("%Y-%m-%d")
            realized = state.setdefault("realized_pnl_by_date", {})
            realized[trade_date] = as_float(realized.get(trade_date)) + as_float(sim.get("_last_fill", {}).get("net_pnl"))
        terminal.append(intent)
    return remaining, terminal


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
    parser.add_argument("--decision-scores", action="store_true",
                        help="also emit Decision Scoring System records (record-only, diagnostic)")
    parser.add_argument("--output-dir", default=str(OUT_DIR),
                        help="isolated replay output directory; defaults to outputs/t0_replay")
    parser.add_argument("--decision-score-output-dir", default="",
                        help="isolated decision-score directory; default keeps legacy outputs/decision_scores")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    source_execution_locks = {"mode": cfg.get("mode"), "execution_enabled": cfg.get("execution_enabled")}
    replay_cfg = cfg.get("replay_execution", {}) if isinstance(cfg.get("replay_execution"), dict) else {}
    offline_lock_override = bool(replay_cfg.get("evaluate_strategy_locks_offline", False))
    if offline_lock_override:
        # Local-only decision evaluation. This process has no broker client and never
        # writes the config, overlay, or live agent state.
        cfg["mode"] = "paper_execute"
        cfg["execution_enabled"] = True
    execution_settings = replay_execution_settings(cfg)
    round_iter = iter_rounds(iter_quote_rows(Path(args.quotes), args.date, args.start_date, args.end_date))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / f"{args.label}_decisions.jsonl"
    summary_path = output_dir / f"{args.label}_summary.json"
    if args.output_detail == "full":
        decisions_path.write_text("", encoding="utf-8")

    state: dict[str, Any] = {}
    sim = _ensure_sim({"initial_cash": INITIAL_CASH, "cash": INITIAL_CASH, "positions": {},
                       "buy_notional": 0.0, "sell_notional": 0.0})
    history: list[dict[str, Any]] = []
    history_by_code: dict[str, list[dict[str, Any]]] = {}
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
    pending: list[dict[str, Any]] = []
    order_records: list[dict[str, Any]] = []
    liquidity_sources: Counter[str] = Counter()
    rows_processed = 0
    missing_data_count = 0
    feature_seconds = 0.0
    momentum_seconds = 0.0
    correlation_seconds = 0.0
    signal_seconds = 0.0
    file_write_seconds = 0.0
    replay_started = time.perf_counter()
    last_replay_now: datetime | None = None

    def day_node(trade_date: str) -> dict[str, Any]:
        return per_day.setdefault(trade_date, {
            "rounds": 0, "entries": 0, "exits": Counter(), "pnl": 0.0,
            "entry_score_pass": 0, "actions": Counter(),
            "buy_notional": 0.0, "sell_notional": 0.0,
            "filled_orders": 0, "rejected_orders": 0, "expired_orders": 0,
            "start_equity": as_float(sim.get("total_equity"), as_float(sim.get("cash"))),
            "start_gross_realized": as_float(sim.get("gross_realized_pnl")),
            "start_net_realized": as_float(sim.get("net_realized_pnl")),
            "start_cost": as_float(sim.get("transaction_cost_total")),
        })

    def register_terminal(intent: dict[str, Any], fallback_date: str) -> None:
        status = str(intent.get("status"))
        event_time = str(intent.get("fill_time") or intent.get("eligible_fill_time") or intent.get("signal_time") or fallback_date)
        td = event_time[:10] if len(event_time) >= 10 else fallback_date
        day = day_node(td)
        if status == "rejected":
            day["rejected_orders"] += 1
        elif status == "expired":
            day["expired_orders"] += 1
        elif status in {"filled", "partially_filled"}:
            day["filled_orders"] += 1
            fill = intent.get("_fill", {}) if isinstance(intent.get("_fill"), dict) else {}
            if intent.get("side") == "buy":
                day["entries"] += 1
                day["buy_notional"] += as_float(fill.get("notional"))
            else:
                day["sell_notional"] += as_float(fill.get("exit_notional"))
                day["exits"][intent.get("reason") or "?"] += 1
                trades.append({
                    "trade_date": td, "stockCode": intent.get("stockCode"),
                    "reason": intent.get("reason"),
                    "pnl": round(as_float(fill.get("net_pnl")), 2),
                    "gross_pnl": round(as_float(fill.get("gross_pnl")), 2),
                    "net_pnl": round(as_float(fill.get("net_pnl")), 2),
                    "quantity": int(as_float(fill.get("quantity"))),
                    "entry_price": round(as_float(fill.get("entry_price")), 6),
                    "net_entry_price": round(as_float(fill.get("net_entry_price")), 6),
                    "exit_price": round(as_float(fill.get("exit_price")), 6),
                    "entry_notional": round(as_float(fill.get("entry_notional")), 2),
                    "exit_notional": round(as_float(fill.get("exit_notional")), 2),
                    "transaction_cost": round(as_float(fill.get("transaction_cost")), 2),
                    "signal_time": intent.get("signal_time"), "fill_time": intent.get("fill_time"),
                    "r_multiple": intent.get("_order", {}).get("r_multiple") if isinstance(intent.get("_order"), dict) else None,
                })
        order_records.append(public_order_record(intent))

    def update_day_equity(trade_date: str, point: dict[str, float]) -> None:
        day = day_node(trade_date)
        net_equity_pnl = as_float(point.get("equity")) - as_float(day.get("start_equity"))
        day_cost = as_float(sim.get("transaction_cost_total")) - as_float(day.get("start_cost"))
        day.update({
            "cash": round(as_float(point.get("cash")), 2),
            "market_value": round(as_float(point.get("market_value")), 2),
            "equity": round(as_float(point.get("equity")), 2),
            "pnl": round(net_equity_pnl, 2), "net_pnl": round(net_equity_pnl, 2),
            "gross_pnl": round(net_equity_pnl + day_cost, 2),
            "gross_realized_pnl": round(as_float(sim.get("gross_realized_pnl")) - as_float(day.get("start_gross_realized")), 2),
            "net_realized_pnl": round(as_float(sim.get("net_realized_pnl")) - as_float(day.get("start_net_realized")), 2),
            "unrealized_pnl": round(as_float(point.get("unrealized_pnl")), 2),
            "transaction_cost": round(day_cost, 2), "drawdown": point.get("drawdown"),
        })

    decision_score_records: list[dict[str, Any]] = []
    try:
        for rnd in round_iter:
            ts = agent.parse_iso_dt(rnd[-1].get("timestamp"))
            if ts is None:
                continue
            replay_now = ts.astimezone(ZoneInfo("Asia/Shanghai"))
            last_replay_now = replay_now
            agent.set_replay_now(replay_now)
            trade_date = replay_now.strftime("%Y-%m-%d")
            if history_trade_date != trade_date:
                history = []
                history_by_code = {}
                history_trade_date = trade_date
                history_universe_max = 0
            day = day_node(trade_date)
            day["rounds"] += 1
            rows_processed += len(rnd)

            raw_quotes = [dict(q) for q in rnd]
            for quote in raw_quotes:
                source = str(quote.get("liquidity_source") or "unknown")
                liquidity_sources[source] += 1
                if not str(quote.get("stockCode") or "") or as_float(quote.get("currentPrice"), 0.0) <= 0:
                    missing_data_count += 1

            pending, terminal = process_pending_orders(sim, pending, raw_quotes, replay_now, cfg, state)
            for intent in terminal:
                register_terminal(intent, trade_date)
            extend_replay_universe(cfg, raw_quotes)
            feature_started = time.perf_counter()
            momentum_started = time.perf_counter()
            quotes = agent.compute_snapshot_momentum(
                raw_quotes,
                history,
                lookback,
                cfg["strategy"],
                history_by_code=history_by_code,
                lazy_indicators=True,
            )
            momentum_seconds += time.perf_counter() - momentum_started
            correlation_started = time.perf_counter()
            market_correlation_stress = agent.compute_market_correlation_stress(
                history,
                quotes,
                cfg.get("strategy", {}).get("market_correlation_stress", {}),
                history_by_code=history_by_code,
            )
            correlation_seconds += time.perf_counter() - correlation_started
            feature_seconds += time.perf_counter() - feature_started
            history.extend(quotes)
            history_universe_max = max(history_universe_max, len(quotes))
            history_row_cap = max_history_snapshots * max(1, history_universe_max)
            if len(history) > history_row_cap:
                history = history[-history_row_cap:]

            session = {"in_regular_session": True}
            agent.update_orb_state(state, trade_date, quotes, replay_now, True)

            point = mark_to_market(sim, raw_quotes, replay_now)
            update_day_equity(trade_date, point)
            total_assets = as_float(sim.get("total_equity"), as_float(sim.get("cash")))
            signal_started = time.perf_counter()
            decision = agent.build_decision(
                cfg, quotes,
                fake_balance(total_assets, sim["cash"]),
                fake_positions(sim["positions"]),
                fake_pending_orders(pending),
                state,
                market_correlation_stress,
                history=history,
                feature_history_by_code=history_by_code,
            )
            signal_seconds += time.perf_counter() - signal_started
            if args.decision_scores:
                try:
                    import decision_scoring as _ds
                    _td = replay_now.strftime("%Y-%m-%d")
                    for _ctx in _ds.contexts_from_decision(
                            cfg, decision, trade_date=_td,
                            timestamp=replay_now.strftime("%H:%M:%S")):
                        _ctx["same_snapshot_fill"] = execution_settings["same_snapshot_fill"]
                        decision_score_records.append(_ds.score_decision(_ctx))
                except Exception:
                    pass
            for quote in quotes:
                code = str(quote.get("stockCode", "")).zfill(6)
                if not code:
                    continue
                code_history = history_by_code.setdefault(code, [])
                code_history.append(quote)
                if len(code_history) > max_history_snapshots:
                    del code_history[:-max_history_snapshots]
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
                    pending.append(make_order_intent(order, replay_now, submit_seq, decision))

            if args.output_detail == "full":
                write_started = time.perf_counter()
                agent.append_jsonl(decisions_path, {
                    "timestamp": replay_now.isoformat(),
                    "action": "order_intent" if orders else "hold",
                    "reason": orders[0].get("reason") if orders else decision.get("state_machine", {}).get("reason"),
                    "approved": bool(decision.get("approved_for_submit")),
                    "sell_score": decision.get("sell_score"),
                    "carry_allowed": decision.get("carry_allowed"),
                    "failed_checks": [c.get("name") for c in decision.get("risk_checks", []) if not c.get("passed")],
                    "orders": orders,
                    "order_intents": [public_order_record(row) for row in pending if row.get("signal_time") == replay_now.isoformat()],
                    "equity": point,
                })
                file_write_seconds += time.perf_counter() - write_started
    finally:
        agent.set_replay_now(None)

    if not per_day:
        print("no quote rounds found")
        return

    if pending:
        for intent in pending:
            intent["status"] = "expired"
            intent["expire_reason"] = "end_of_replay_no_next_snapshot"
            register_terminal(intent, str(intent.get("signal_time", ""))[:10])
        pending = []
    if last_replay_now is not None:
        refresh_available_quantities(sim, last_replay_now)
    final_assets = as_float(sim.get("total_equity"), as_float(sim.get("cash")))
    transaction_cost_total = as_float(sim.get("transaction_cost_total"))
    net_total_pnl = final_assets - INITIAL_CASH
    gross_total_pnl = net_total_pnl + transaction_cost_total
    source_text = str(Path(args.quotes)).lower()
    legacy_full_day_source = "yahoo_60d_quotes" in source_text or "june_all_etf_quotes" in source_text
    full_day_used = legacy_full_day_source or liquidity_sources.get("contaminated_full_day", 0) > 0
    known_sources = {key for key, count in liquidity_sources.items() if count > 0 and key != "unknown"}
    point_in_time_liquidity = bool(known_sources) and known_sources <= {"point_in_time", "previous_day", "rolling_past"} and not full_day_used
    rejected_count = sum(1 for row in order_records if row.get("status") == "rejected")
    expired_count = sum(1 for row in order_records if row.get("status") == "expired")
    execution_model = {
        "same_snapshot_fill": False, "next_snapshot_fill": True,
        "cost_in_path": True, "mark_to_market": True, "t_rule_enforced": True,
        "default_sell_rule": execution_settings["default_sell_rule"],
        "t0_allowlist_count": len(execution_settings["t0_allowlist"]),
        "buy_cost_pct": execution_settings["buy_cost_pct"],
        "sell_cost_pct": execution_settings["sell_cost_pct"],
        "slippage_pct_per_side": execution_settings["slippage_pct_per_side"],
        "source_execution_locks": source_execution_locks,
        "offline_lock_override": offline_lock_override,
    }
    data_quality = {
        "point_in_time_liquidity": point_in_time_liquidity,
        "full_day_liquidity_used": full_day_used,
        "liquidity_sources": dict(liquidity_sources),
        "survivor_bias_warning": True,
        "missing_data_count": missing_data_count,
        "rejected_order_count": rejected_count,
    }
    result_trust_level = "contaminated" if full_day_used else (
        "diagnostic_only" if not point_in_time_liquidity or data_quality["survivor_bias_warning"] else "clean"
    )
    if args.decision_scores and decision_score_records:
        try:
            import decision_scoring as _ds
            from collections import defaultdict as _dd
            _ds.enrich_from_quotes(decision_score_records, Path(args.quotes))  # fill forward outcomes
            by_date: dict[str, list[dict[str, Any]]] = _dd(list)
            for r in decision_score_records:
                by_date[str(r.get("date"))].append(r)
            score_output_dir = Path(args.decision_score_output_dir) if args.decision_score_output_dir else _ds.OUT_DIR
            for d, recs in by_date.items():
                _ds.write_scores(recs, d, out_dir=score_output_dir)
            _enriched = sum(1 for r in decision_score_records if r.get("realized_return") is not None)
            print(f"decision_scores: wrote {len(decision_score_records)} records "
                  f"({_enriched} with outcomes) across {len(by_date)} day(s)")
        except Exception as _e:
            print(f"decision_scores: skipped ({_e})")
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
                "gross_pnl": round(as_float(d.get("gross_pnl"), as_float(d.get("pnl"))), 2),
                "net_pnl": round(as_float(d.get("net_pnl"), as_float(d.get("pnl"))), 2),
                "gross_realized_pnl": round(as_float(d.get("gross_realized_pnl")), 2),
                "net_realized_pnl": round(as_float(d.get("net_realized_pnl")), 2),
                "unrealized_pnl": round(as_float(d.get("unrealized_pnl")), 2),
                "cash": round(as_float(d.get("cash")), 2),
                "market_value": round(as_float(d.get("market_value")), 2),
                "equity": round(as_float(d.get("equity")), 2),
                "drawdown": d.get("drawdown"),
                "transaction_cost": round(as_float(d.get("transaction_cost")), 2),
                "buy_notional": round(d["buy_notional"], 2),
                "sell_notional": round(d["sell_notional"], 2),
                "filled_order_count": d["filled_orders"],
                "rejected_order_count": d["rejected_orders"],
                "expired_order_count": d["expired_orders"],
            } for td, d in sorted(per_day.items())
        },
        "trades": trades,
        "order_lifecycle": order_records,
        "equity_curve": sim.get("equity_curve", []),
        "r_multiples": [t["r_multiple"] for t in trades if t.get("r_multiple") is not None],
        "total_pnl": round(net_total_pnl, 2),
        "gross_pnl": round(gross_total_pnl, 2),
        "gross_realized_pnl": round(as_float(sim.get("gross_realized_pnl")), 2),
        "net_realized_pnl": round(as_float(sim.get("net_realized_pnl")), 2),
        "unrealized_pnl": round(as_float(sim.get("unrealized_pnl")), 2),
        "total_equity": round(final_assets, 2),
        "max_drawdown": round(as_float(sim.get("max_drawdown")), 8),
        "trade_count": len(trades),
        "rejected_order_count": rejected_count,
        "expired_order_count": expired_count,
        "transaction_cost_total": round(transaction_cost_total, 2),
        "buy_notional": round(as_float(sim.get("buy_notional")), 2),
        "sell_notional": round(as_float(sim.get("sell_notional")), 2),
        "turnover": round(
            (as_float(sim.get("buy_notional")) + as_float(sim.get("sell_notional"))) / (2.0 * INITIAL_CASH),
            6,
        ),
        "final_assets": round(final_assets, 2),
        "open_positions_at_end": {c: int(as_float(p.get("quantity"))) for c, p in sim["positions"].items()},
        "execution_model": execution_model,
        "data_quality": data_quality,
        "result_trust_level": result_trust_level,
        "top_blocking_checks": fail_counter.most_common(10),
        "runtime_profile": {
            "total_runtime_seconds": round(time.perf_counter() - replay_started, 6),
            "feature_calculation_seconds": round(feature_seconds, 6),
            "momentum_feature_seconds": round(momentum_seconds, 6),
            "correlation_feature_seconds": round(correlation_seconds, 6),
            "signal_generation_seconds": round(signal_seconds, 6),
            "file_write_seconds": round(file_write_seconds, 6),
            "rows_processed": rows_processed,
            "output_detail": args.output_detail,
        },
        "note": "next-snapshot conservative limit simulation with in-path costs and T-rule lots; diagnostic, not a profitability claim",
    }
    agent.write_json(summary_path, summary)

    print(f"rounds: {summary['rounds_total']}")
    for td, d in summary["per_day"].items():
        print(f"{td}: rounds={d['rounds']} entries={d['entries']} score_pass={d['entry_score_pass_rounds']} exits={d['exits']} net_equity_pnl={d['net_pnl']}")
    print(f"net_total_pnl={summary['total_pnl']} total_equity={summary['total_equity']} max_drawdown={summary['max_drawdown']} open_at_end={summary['open_positions_at_end']}")
    print(f"orders: trades={summary['trade_count']} rejected={rejected_count} expired={expired_count} trust={result_trust_level}")
    print("top blocking checks:", summary["top_blocking_checks"][:5])
    print(f"outputs: {decisions_path} | {summary_path}")


if __name__ == "__main__":
    main()
