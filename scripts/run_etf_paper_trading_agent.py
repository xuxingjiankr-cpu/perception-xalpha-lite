"""ETF-only paper-trading quant agent wrapper.

This script does not perform live trading. It wraps the installed
a-share-paper-trading skill and defaults to dry-run mode. Set both
`mode=paper_execute` and `execution_enabled=true` in the config to submit
simulated paper orders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from shared_paper_trading_guard import SharedExecutionGuard, tag_order_owner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "etf_paper_trading_agent.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def cn_market_session(now: datetime | None = None) -> dict[str, Any]:
    """Return a conservative A-share session check for execution gating."""
    local = (now or datetime.now(tz=ZoneInfo("Asia/Shanghai"))).astimezone(ZoneInfo("Asia/Shanghai"))
    minutes = local.hour * 60 + local.minute
    morning = 9 * 60 + 30 <= minutes <= 11 * 60 + 30
    afternoon = 13 * 60 <= minutes <= 15 * 60
    weekday = local.weekday() < 5
    return {
        "exchange_timezone": "Asia/Shanghai",
        "local_time": local.isoformat(),
        "weekday": weekday,
        "in_regular_session": bool(weekday and (morning or afternoon)),
        "session": "morning" if weekday and morning else ("afternoon" if weekday and afternoon else "closed"),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def stable_order_id(order: dict[str, Any], timestamp: str) -> str:
    raw = json.dumps(
        {
            "timestamp": timestamp,
            "direction": order.get("direction"),
            "stockCode": order.get("stockCode"),
            "exchange": order.get("exchange"),
            "quantity": order.get("quantity"),
            "orderType": order.get("orderType"),
            "price": order.get("price"),
            "reason": order.get("reason"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def append_order_blotter(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            try:
                fields = next(reader)
            except StopIteration:
                fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_blotter_rows(
    result: dict[str, Any],
    submit_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    timestamp = str(result.get("timestamp") or now_iso())
    orders = result.get("orders", [])
    if not isinstance(orders, list):
        return []
    submit_results = submit_results or []
    rows: list[dict[str, Any]] = []
    for idx, order in enumerate(orders):
        submit = submit_results[idx] if idx < len(submit_results) else None
        submit_data = extract_data(submit) if isinstance(submit, dict) else {}
        rows.append({
            "timestamp": timestamp,
            "local_order_id": stable_order_id(order, timestamp),
            "mode": result.get("mode"),
            "agent_status": result.get("status"),
            "event_type": "submitted" if submit else "planned",
            "direction": order.get("direction"),
            "stockCode": order.get("stockCode"),
            "exchange": order.get("exchange"),
            "name": order.get("name"),
            "t0_eligible": order.get("t0_eligible"),
            "asset_class": order.get("asset_class"),
            "quantity": order.get("quantity"),
            "orderType": order.get("orderType"),
            "price": order.get("price"),
            "score": order.get("score"),
            "reason": order.get("reason"),
            "target_position_value": order.get("target_position_value"),
            "current_position_value": order.get("current_position_value"),
            "submit_ok": submit.get("ok") if isinstance(submit, dict) else None,
            "broker_order_id": submit_data.get("orderId"),
            "broker_status": submit_data.get("status"),
            "submit_error": json.dumps(submit.get("error"), ensure_ascii=False) if isinstance(submit, dict) and submit.get("error") else None,
            "paper_trading_only": True,
            "live_ready": False,
        })
    return rows


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def parse_tool_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {"ok": False, "data": None, "error": {"category": "empty", "message": "empty stdout"}}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "data": None, "error": {"category": "parse", "message": str(exc), "raw": text[:500]}}
    if "ok" in obj:
        return obj
    if str(obj.get("code", "")) not in {"", "0"}:
        return {"ok": False, "data": None, "error": {"category": "api", "message": obj.get("msg") or obj.get("message"), "raw": obj}}
    return {"ok": True, "data": obj.get("data", obj), "error": None}


@dataclass
class SkillClient:
    python_cmd: list[str]
    script: Path

    def call(self, tool: str, *args: str) -> dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [*self.python_cmd, str(self.script), tool, *args]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
        parsed = parse_tool_json(proc.stdout)
        parsed["_cmd_tool"] = tool
        parsed["_returncode"] = proc.returncode
        if proc.stderr.strip():
            parsed["_stderr"] = proc.stderr.strip()[:1000]
        return parsed

    def get_balance(self) -> dict[str, Any]:
        return self.call("getAccountBalance")

    def get_positions(self) -> dict[str, Any]:
        return self.call("getPositions")

    def get_quote(self, code: str, exchange: str) -> dict[str, Any]:
        return self.call("getQuote", "--stock-code", code, "--exchange", exchange)

    def cancel_all_pending(self) -> dict[str, Any]:
        return self.call("cancelAllPendingOrders")

    def submit_order(self, direction: str, code: str, exchange: str, quantity: int, order_type: str, price: float | None) -> dict[str, Any]:
        args = ["--direction", direction, "--stock-code", code, "--exchange", exchange, "--quantity", str(quantity), "--order-type", order_type]
        if order_type == "limit":
            if price is None:
                raise ValueError("limit order requires price")
            args += ["--price", f"{price:.3f}"]
        return self.call("submitOrder", *args)


def python_cmd_from_config(value: str) -> list[str]:
    parts = value.split()
    return parts or [sys.executable]


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def extract_data(resp: dict[str, Any]) -> dict[str, Any]:
    data = resp.get("data")
    return data if isinstance(data, dict) else {}


def trade_date_cn() -> str:
    return datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def is_quota_exhausted_response(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    err = resp.get("error")
    if not isinstance(err, dict):
        return False
    raw = err.get("raw")
    code = str(raw.get("code")) if isinstance(raw, dict) else ""
    message = str(err.get("message") or "")
    return code == "1002" or "配额" in message or "quota" in message.lower()


def any_quota_exhausted(responses: list[Any]) -> bool:
    return any(is_quota_exhausted_response(x) for x in responses)


def quota_state_path(output_dir: Path, cfg: dict[str, Any]) -> Path:
    return output_dir / cfg.get("outputs", {}).get("quota_state", "quota_backoff_state.json")


def quota_backoff_status(path: Path, trade_date: str) -> dict[str, Any]:
    state = load_state(path)
    bucket = state.get("quota_exhausted_by_date", {})
    day = bucket.get(trade_date, {}) if isinstance(bucket, dict) else {}
    active = bool(isinstance(day, dict) and day.get("active"))
    return {
        "active": active,
        "trade_date": trade_date,
        "state_path": str(path),
        "detail": day if isinstance(day, dict) else {},
    }


def mark_quota_exhausted(path: Path, trade_date: str, source: str, response: Any = None) -> dict[str, Any]:
    state = load_state(path)
    bucket = state.get("quota_exhausted_by_date", {})
    if not isinstance(bucket, dict):
        bucket = {}
    error = response.get("error") if isinstance(response, dict) else None
    bucket[trade_date] = {
        "active": True,
        "detected_at": now_iso(),
        "source": source,
        "error": error,
        "resume_rule": "next_trade_date_only",
    }
    state["quota_exhausted_by_date"] = bucket
    save_state(path, state)
    return quota_backoff_status(path, trade_date)


def quota_backoff_result(
    cfg: dict[str, Any],
    mode: str,
    execute: bool,
    output_dir: Path,
    reason: str,
    quota_status: dict[str, Any],
    quotes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "agent_name": cfg.get("agent_name"),
        "mode": mode,
        "execution_enabled": bool(cfg.get("execution_enabled")),
        "cli_execute": bool(execute),
        "asset_type": cfg.get("asset_type"),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "status": "quota_exhausted_backoff",
        "reason": reason,
        "quota_backoff": quota_status,
        "quotes": quotes or [],
        "orders": [],
        "submit_results": [],
        "execution_report": {
            "attempted": False,
            "submitted_count": 0,
            "submit_results": [],
            "blocked_reason": reason,
        },
        "order_blotter_rows": [],
        "quota_backoff_policy": "no_further_api_calls_until_next_trade_date",
        "output_dir": str(output_dir),
    }


def market_closed_no_api_result(cfg: dict[str, Any], mode: str, execute: bool, output_dir: Path, session: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "agent_name": cfg.get("agent_name"),
        "mode": mode,
        "execution_enabled": bool(cfg.get("execution_enabled")),
        "cli_execute": bool(execute),
        "asset_type": cfg.get("asset_type"),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "status": "market_closed_no_api",
        "reason": "outside_regular_trading_session_no_api_calls",
        "session": session,
        "approved_for_submit": False,
        "risk_checks": [{
            "name": "regular_trading_session",
            "passed": False,
            "detail": session,
        }],
        "quotes": [],
        "orders": [],
        "submit_results": [],
        "execution_report": {
            "attempted": False,
            "submitted_count": 0,
            "submit_results": [],
            "blocked_reason": "outside_regular_trading_session_no_api_calls",
        },
        "order_blotter_rows": [],
        "output_dir": str(output_dir),
    }


def normalize_quote(etf: dict[str, Any], resp: dict[str, Any],
                    price_history: dict[str, dict[str, float]] | None = None,
                    momentum_days: int = 0) -> dict[str, Any]:
    data = extract_data(resp)
    current = as_float(data.get("currentPrice"))
    prev = as_float(data.get("prevClose"))
    intraday_ret = (current / prev - 1.0) if current > 0 and prev > 0 else 0.0
    is_suspended = bool(data.get("isSuspended", False))

    # --- Signal selection (competition param) ---
    if not is_suspended and momentum_days > 0 and price_history is not None:
        mom = compute_momentum_score(etf["stockCode"], current, price_history, momentum_days)
        signal = mom if mom is not None else intraday_ret
        signal_type = "momentum_5d" if mom is not None else "intraday_ret_fallback"
    else:
        signal = intraday_ret
        signal_type = "intraday_ret"

    return {
        "stockCode": etf["stockCode"],
        "exchange": etf["exchange"],
        "name": data.get("stockName") or etf.get("name", ""),
        "t0_eligible": bool(etf.get("t0_eligible", False)),
        "asset_class": etf.get("asset_class"),
        "currentPrice": current,
        "prevClose": prev,
        "bidPrice1": as_float(data.get("bidPrice1"), current),
        "askPrice1": as_float(data.get("askPrice1"), current),
        "isSuspended": is_suspended,
        "intraday_ret": intraday_ret,
        "signal_type": signal_type,
        "score": signal if not is_suspended else -999.0,
        "quote_ok": bool(resp.get("ok")),
        "quote_error": None if resp.get("ok") else resp.get("error"),
    }


def positions_by_code(resp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = extract_data(resp)
    positions = data.get("positions", [])
    if not isinstance(positions, list):
        positions = []
    return {str(p.get("stockCode")).zfill(6): p for p in positions if p.get("stockCode")}


def account_assets(resp: dict[str, Any]) -> tuple[float, float]:
    data = extract_data(resp)
    total = as_float(data.get("totalAssets") or data.get("totalAsset") or data.get("initialCapital"))
    available = as_float(data.get("availableBalance"))
    return total, available


# ---------------------------------------------------------------------------
# Price-history helpers (for N-day momentum signal)
# ---------------------------------------------------------------------------

def load_price_history(path: Path) -> dict[str, dict[str, float]]:
    """Return {date_str: {stockCode: prevClose}}. Empty dict if missing/corrupt."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_price_history(path: Path, history: dict[str, dict[str, float]],
                       max_days: int = 30) -> None:
    """Persist history, pruning entries older than max_days trading sessions."""
    dates = sorted(history.keys())
    for old_date in dates[:-max_days]:
        history.pop(old_date, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def update_price_history(history: dict[str, dict[str, float]],
                         date_str: str, quotes: list[dict[str, Any]]) -> None:
    """Record today's prevClose for each ETF; prevClose = yesterday's official close."""
    day_data = {
        q["stockCode"]: q["prevClose"]
        for q in quotes
        if q.get("quote_ok") and as_float(q.get("prevClose")) > 0
    }
    if day_data:
        history[date_str] = day_data


def compute_momentum_score(stock_code: str, current_price: float,
                           history: dict[str, dict[str, float]],
                           n_days: int) -> float | None:
    """N-day momentum = currentPrice / prevClose[n_days_ago] − 1.
    Returns None when insufficient history (caller falls back to intraday_ret)."""
    dates = sorted(history.keys())
    if len(dates) < n_days:
        return None
    anchor_close = history[dates[-n_days]].get(stock_code)
    if not anchor_close or anchor_close <= 0:
        return None
    return current_price / anchor_close - 1.0


def build_grid_regime_report(
    cfg: dict[str, Any],
    quotes: list[dict[str, Any]],
    price_history: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Read-only range/breakout diagnostic inspired by dynamic grid trading.

    This does not modify ranking, target weights, orders, or risk gates.
    It is intended to flag whether an ETF looks range-bound or has broken
    out of its recent price range, because grid-style logic is fragile in
    directional regimes.
    """
    regime_cfg = cfg.get("strategy", {}).get("grid_regime", {})
    enabled = bool(regime_cfg.get("enabled", True))
    lookback_days = int(regime_cfg.get("lookback_days", 5))
    breakout_threshold = as_float(regime_cfg.get("breakout_threshold_pct"), 0.002)
    report: dict[str, Any] = {
        "enabled": enabled,
        "diagnostic_only": True,
        "affects_signal": False,
        "affects_orders": False,
        "source": "Dynamic-Grid-Trading-inspired range/breakout diagnostic",
        "lookback_days": lookback_days,
        "breakout_threshold_pct": breakout_threshold,
        "items": [],
    }
    if not enabled:
        return report

    dates = sorted(price_history.keys())
    for q in quotes:
        code = str(q.get("stockCode", "")).zfill(6)
        close_series = [
            as_float(price_history.get(day, {}).get(code))
            for day in dates[-lookback_days:]
            if as_float(price_history.get(day, {}).get(code)) > 0
        ]
        current = as_float(q.get("currentPrice"))
        if current > 0:
            close_series.append(current)
        if len(close_series) < max(3, min(lookback_days, 3)):
            item = {
                "stockCode": code,
                "exchange": q.get("exchange"),
                "name": q.get("name"),
                "available_points": len(close_series),
                "status": "insufficient_history",
                "grid_unsuitable_reason": "insufficient_price_history",
            }
            report["items"].append(item)
            continue

        recent_high = max(close_series[:-1]) if len(close_series) > 1 else current
        recent_low = min(close_series[:-1]) if len(close_series) > 1 else current
        range_mid = (recent_high + recent_low) / 2.0 if recent_high > 0 and recent_low > 0 else 0.0
        recent_range_pct = (recent_high - recent_low) / range_mid if range_mid > 0 else 0.0
        breakout_up = current > recent_high * (1.0 + breakout_threshold)
        breakout_down = current < recent_low * (1.0 - breakout_threshold)
        range_bound_score = max(0.0, 1.0 - min(recent_range_pct / 0.10, 1.0))
        if breakout_up:
            status = "breakout_up"
            reason = "upside_breakout_dynamic_grid_reset_needed"
        elif breakout_down:
            status = "breakout_down"
            reason = "downside_breakout_dynamic_grid_reset_needed"
        elif recent_range_pct <= 0.03:
            status = "range_bound"
            reason = "grid_style_diagnostic_possible_but_not_used_for_orders"
        else:
            status = "wide_range"
            reason = "range_too_wide_for_static_grid_assumption"

        report["items"].append({
            "stockCode": code,
            "exchange": q.get("exchange"),
            "name": q.get("name"),
            "available_points": len(close_series),
            "currentPrice": current,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "recent_range_pct": recent_range_pct,
            "range_bound_score": range_bound_score,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "status": status,
            "grid_unsuitable_reason": reason,
        })
    report["summary"] = {
        "range_bound_count": sum(1 for x in report["items"] if x.get("status") == "range_bound"),
        "breakout_up_count": sum(1 for x in report["items"] if x.get("status") == "breakout_up"),
        "breakout_down_count": sum(1 for x in report["items"] if x.get("status") == "breakout_down"),
        "insufficient_history_count": sum(1 for x in report["items"] if x.get("status") == "insufficient_history"),
    }
    return report


def round_lot(quantity: float, lot: int) -> int:
    if lot <= 0:
        return int(quantity)
    return int(quantity // lot * lot)


def build_plan(cfg: dict[str, Any], quotes: list[dict[str, Any]], balance: dict[str, Any], positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    risk = cfg["risk"]
    strat = cfg["strategy"]
    total_assets, available_cash = account_assets(balance)
    if total_assets <= 0:
        return {"ok": False, "reason": "account_total_assets_unavailable", "orders": []}

    active_quotes = [q for q in quotes if q["quote_ok"] and not q["isSuspended"] and q["currentPrice"] > 0]
    ranked = sorted(active_quotes, key=lambda x: x["score"], reverse=True)
    target_n = max(1, int(strat.get("target_holdings", 3)))
    force_build = bool(strat.get("force_build_position", True))
    entry_threshold = as_float(strat.get("entry_score_threshold_pct"), 0.0)
    if force_build:
        selected = ranked[:target_n]
    else:
        selected = [q for q in ranked if as_float(q.get("score"), -999.0) >= entry_threshold][:target_n]

    cash_reserve = as_float(strat.get("cash_reserve_pct"), 0.05)
    max_pos_pct = as_float(risk.get("max_position_pct"), 0.25)
    max_order_pct = as_float(risk.get("max_single_order_pct"), 0.10)
    drift = as_float(strat.get("rebalance_drift_threshold_pct"), 0.03)
    lot = int(risk.get("quantity_lot", 100))
    min_qty = int(risk.get("min_order_quantity", 100))
    order_type = str(risk.get("order_type", "limit"))
    slip = as_float(risk.get("limit_price_slippage_pct"), 0.002)

    investable = total_assets * max(0.0, 1.0 - cash_reserve)
    target_value = min(total_assets * max_pos_pct, investable / max(1, len(selected)))
    max_order_value = total_assets * max_order_pct

    orders: list[dict[str, Any]] = []
    selected_codes = {q["stockCode"] for q in selected}
    quote_by_code = {q["stockCode"]: q for q in active_quotes}

    if bool(risk.get("stop_loss_enabled", True)):
        stop_loss_pct = as_float(risk.get("stop_loss_pct"), -0.03)
        stop_orders: list[dict[str, Any]] = []
        for code, pos in positions.items():
            q = quote_by_code.get(code)
            if not q:
                continue
            qty = round_lot(as_float(pos.get("availableQuantity")), lot)
            if qty < min_qty:
                continue
            cost_price = as_float(pos.get("costPrice"))
            current = as_float(q.get("currentPrice"))
            loss_pct = current / cost_price - 1.0 if cost_price > 0 and current > 0 else None
            if loss_pct is not None and loss_pct <= stop_loss_pct:
                px = q["bidPrice1"] if q["bidPrice1"] > 0 else q["currentPrice"]
                if order_type == "limit":
                    px *= 1.0 - slip
                stop_orders.append({
                    "direction": "sell",
                    "stockCode": code,
                    "exchange": q["exchange"],
                    "name": q["name"],
                    "quantity": qty,
                    "orderType": order_type,
                    "price": round(px, 3) if order_type == "limit" else None,
                    "score": q["score"],
                    "t0_eligible": bool(q.get("t0_eligible", False)),
                    "asset_class": q.get("asset_class"),
                    "reason": "stop_loss_exit",
                    "loss_pct": loss_pct,
                    "stop_loss_pct": stop_loss_pct,
                    "current_position_value": as_float(pos.get("marketValue")),
                    "target_position_value": 0.0,
                })
        if stop_orders:
            return {
                "ok": True,
                "total_assets": total_assets,
                "available_cash": available_cash,
                "selected": selected,
                "orders": stop_orders[: int(risk.get("max_daily_stop_loss_sells", 1))],
                "priority": "stop_loss_exit",
            }

    for q in selected:
        pos = positions.get(q["stockCode"], {})
        current_value = as_float(pos.get("marketValue"))
        diff = target_value - current_value
        if abs(diff) < total_assets * drift:
            continue
        direction = "buy" if diff > 0 else "sell"
        trade_value = min(abs(diff), max_order_value)
        px = q["askPrice1"] if direction == "buy" else q["bidPrice1"]
        if px <= 0:
            px = q["currentPrice"]
        if order_type == "limit":
            px = px * (1.0 + slip) if direction == "buy" else px * (1.0 - slip)
        qty = round_lot(trade_value / px, lot)
        if qty < min_qty:
            continue
        if direction == "sell":
            qty = min(qty, int(as_float(pos.get("availableQuantity"))))
            qty = round_lot(qty, lot)
            if qty < min_qty:
                continue
        orders.append({
            "direction": direction,
            "stockCode": q["stockCode"],
            "exchange": q["exchange"],
            "name": q["name"],
            "quantity": qty,
            "orderType": order_type,
            "price": round(px, 3) if order_type == "limit" else None,
            "score": q["score"],
            "t0_eligible": bool(q.get("t0_eligible", False)),
            "asset_class": q.get("asset_class"),
            "reason": "target_weight_rebalance",
            "current_position_value": current_value,
            "target_position_value": target_value,
        })

    for code, pos in positions.items():
        if code in selected_codes or code not in quote_by_code:
            continue
        qty = round_lot(as_float(pos.get("availableQuantity")), lot)
        if qty < min_qty:
            continue
        q = quote_by_code[code]
        px = q["bidPrice1"] if q["bidPrice1"] > 0 else q["currentPrice"]
        if order_type == "limit":
            px *= 1.0 - slip
        orders.append({
            "direction": "sell",
            "stockCode": code,
            "exchange": q["exchange"],
            "name": q["name"],
            "quantity": qty,
            "orderType": order_type,
            "price": round(px, 3) if order_type == "limit" else None,
            "score": q["score"],
            "t0_eligible": bool(q.get("t0_eligible", False)),
            "asset_class": q.get("asset_class"),
            "reason": "not_in_selected_etf_set",
            "current_position_value": as_float(pos.get("marketValue")),
            "target_position_value": 0.0,
        })

    orders = orders[: int(risk.get("max_daily_orders", 5))]
    defensive_cash_threshold = as_float(strat.get("defensive_cash_threshold_pct"), -0.005)
    all_scores = [
        as_float(q.get("score"))
        for q in active_quotes
        if str(q.get("signal_type", "")).startswith("momentum_") and q.get("score") is not None
    ]
    score_cfg = strat.get("score", {})
    required_history_count = int(score_cfg.get("momentum_days", 0)) if not score_cfg.get("use_intraday_return", True) else 0
    require_full_momentum = bool(score_cfg.get("require_full_momentum_history_for_execute", True))
    momentum_warmup_active = bool(require_full_momentum and required_history_count > 0 and len(all_scores) < max(4, int(strat.get("target_holdings", 3))))
    positive_momentum_count = sum(1 for x in all_scores if x >= entry_threshold)
    min_positive_count = int(strat.get("min_positive_momentum_count_for_buy", strat.get("min_selected_with_positive_score", 1)))
    market_breadth_block_active = bool(len(all_scores) >= max(4, min_positive_count) and positive_momentum_count < min_positive_count)
    cash_defense_active = len(all_scores) >= 4 and max(all_scores) < defensive_cash_threshold
    max_score_observed = max(all_scores) if all_scores else None
    non_risk_orders_blocked_reason = None
    if cash_defense_active:
        orders = [o for o in orders if o.get("direction") != "buy"]
        non_risk_orders_blocked_reason = "cash_defense_active"
    if momentum_warmup_active or market_breadth_block_active:
        # During warm-up or weak breadth, preserve only explicit risk exits.
        orders = [o for o in orders if o.get("reason") == "stop_loss_exit"]
        non_risk_orders_blocked_reason = "momentum_warmup_active" if momentum_warmup_active else "market_breadth_block_active"
    target_holdings_actual = 0 if cash_defense_active else len(selected)
    return {
        "ok": True,
        "total_assets": total_assets,
        "available_cash": available_cash,
        "selected": selected,
        "target_holdings_actual": target_holdings_actual,
        "orders": orders,
        "cash_defense_active": cash_defense_active,
        "cash_defense_reason": "all_momentum_scores_below_threshold" if cash_defense_active else None,
        "cash_defense_threshold": defensive_cash_threshold,
        "max_score_observed": max_score_observed,
        "cash_defense_score_count": len(all_scores),
        "momentum_warmup_active": momentum_warmup_active,
        "required_history_count": required_history_count,
        "valid_momentum_score_count": len(all_scores),
        "market_breadth_block_active": market_breadth_block_active,
        "positive_momentum_count": positive_momentum_count,
        "min_positive_momentum_count_for_buy": min_positive_count,
        "non_risk_orders_blocked_reason": non_risk_orders_blocked_reason,
    }


def build_analysis_report(cfg: dict[str, Any], quotes: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted([q for q in quotes if q.get("quote_ok")], key=lambda x: x.get("score", -999), reverse=True)
    selected = plan.get("selected", [])
    target_weight = 1.0 / len(selected) if selected else 0.0
    return {
        "framework": "AI-Trader-inspired local state machine",
        "quantmuse_inspired": {
            "strategy_result_schema": True,
            "factor_weighting_used": False,
            "external_quantmuse_dependency": False,
        },
        "zipline_inspired": {
            "lifecycle": ["initialize", "before_trading_start", "handle_data", "order_target", "analyze"],
            "order_target_semantics": True,
            "external_zipline_dependency": False,
        },
        "cloud_ai_trader_used": False,
        "signal_type": cfg.get("strategy", {}).get("type"),
        "asset_scope": "ETF_ONLY",
        "universe_size": len(cfg.get("universe", [])),
        "quote_ok_count": sum(1 for q in quotes if q.get("quote_ok")),
        "suspended_count": sum(1 for q in quotes if q.get("isSuspended")),
        "top_ranked": [
            {
                "stockCode": q.get("stockCode"),
                "exchange": q.get("exchange"),
                "name": q.get("name"),
                "score": q.get("score"),
                "currentPrice": q.get("currentPrice"),
                "intraday_ret": q.get("intraday_ret"),
            }
            for q in ranked[:5]
        ],
        "selected": [
            {
                "stockCode": q.get("stockCode"),
                "exchange": q.get("exchange"),
                "name": q.get("name"),
                "score": q.get("score"),
                "target_weight_within_selected": target_weight,
            }
            for q in selected
        ],
        "strategy_result": {
            "strategy_name": cfg.get("strategy", {}).get("type"),
            "selected_symbols": [f"{q.get('stockCode')}.{q.get('exchange')}" for q in selected],
            "weights": {f"{q.get('stockCode')}.{q.get('exchange')}": target_weight for q in selected},
            "parameters": cfg.get("strategy", {}),
            "performance_metrics": {
                "selected_count": len(selected),
                "planned_order_count": len(plan.get("orders", [])),
                "total_weight": round(target_weight * len(selected), 8) if selected else 0.0,
            },
            "metadata": {
                "paper_trading_only": True,
                "live_ready": False,
            },
        },
    }


def portfolio_risk_metrics(cfg: dict[str, Any], plan: dict[str, Any], balance: dict[str, Any]) -> dict[str, Any]:
    total_assets, available_cash = account_assets(balance)
    orders = plan.get("orders", []) if isinstance(plan.get("orders"), list) else []
    notionals = []
    for order in orders:
        price = as_float(order.get("price"))
        qty = as_float(order.get("quantity"))
        notionals.append(price * qty)
    gross_order_notional = sum(notionals)
    max_order_notional = max(notionals) if notionals else 0.0
    selected = plan.get("selected", []) if isinstance(plan.get("selected"), list) else []
    target_values = [as_float(o.get("target_position_value")) for o in orders]
    max_target_value = max(target_values) if target_values else 0.0
    return {
        "total_assets": total_assets,
        "available_cash": available_cash,
        "selected_count": len(selected),
        "planned_order_count": len(orders),
        "gross_order_notional": gross_order_notional,
        "gross_order_notional_pct": gross_order_notional / total_assets if total_assets > 0 else 0.0,
        "max_order_notional": max_order_notional,
        "max_order_notional_pct": max_order_notional / total_assets if total_assets > 0 else 0.0,
        "max_target_position_value": max_target_value,
        "max_target_position_pct": max_target_value / total_assets if total_assets > 0 else 0.0,
        "configured_max_position_pct": as_float(cfg.get("risk", {}).get("max_position_pct")),
        "configured_max_single_order_pct": as_float(cfg.get("risk", {}).get("max_single_order_pct")),
        "configured_max_daily_orders": int(cfg.get("risk", {}).get("max_daily_orders", 0)),
    }


def zipline_style_lifecycle_report(
    cfg: dict[str, Any],
    mode: str,
    quotes: list[dict[str, Any]],
    plan: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Expose a Zipline-like strategy lifecycle without importing Zipline."""
    session = gate.get("session", {})
    return {
        "initialize": {
            "strategy": cfg.get("strategy", {}).get("type"),
            "universe_size": len(cfg.get("universe", [])),
            "asset_scope": "ETF_ONLY",
            "mode": mode,
        },
        "before_trading_start": {
            "session": session,
            "trading_hours_only": bool(cfg.get("trading_hours_only", True)),
        },
        "handle_data": {
            "quote_count": len(quotes),
            "quote_ok_count": sum(1 for q in quotes if q.get("quote_ok")),
            "suspended_count": sum(1 for q in quotes if q.get("isSuspended")),
        },
        "order_target": {
            "target_holdings": cfg.get("strategy", {}).get("target_holdings"),
            "cash_reserve_pct": cfg.get("strategy", {}).get("cash_reserve_pct"),
            "planned_order_count": len(plan.get("orders", [])) if isinstance(plan.get("orders"), list) else 0,
            "target_position_values": [
                {
                    "symbol": f"{o.get('stockCode')}.{o.get('exchange')}",
                    "target_position_value": o.get("target_position_value"),
                    "current_position_value": o.get("current_position_value"),
                }
                for o in (plan.get("orders", []) if isinstance(plan.get("orders"), list) else [])
            ],
        },
        "trading_controls": {
            "checks": gate.get("checks", []),
            "approved_for_submit": gate.get("approved_for_submit", False),
        },
        "analyze": {
            "status": "planned" if plan.get("ok") else "blocked",
            "paper_trading_only": True,
            "live_ready": False,
            "formal_strategy_allowed": False,
        },
    }


def risk_gate(
    cfg: dict[str, Any],
    mode: str,
    execute: bool,
    kill_switch_active: bool,
    plan: dict[str, Any],
    quotes: list[dict[str, Any]],
    balance: dict[str, Any],
) -> dict[str, Any]:
    risk = cfg.get("risk", {})
    session = cn_market_session()
    total_assets, available_cash = account_assets(balance)
    whitelist = {(str(x["stockCode"]).zfill(6), x["exchange"]) for x in cfg.get("universe", [])}
    orders = plan.get("orders", []) if plan.get("ok") else []

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add("kill_switch_inactive", not kill_switch_active)
    add("paper_mode_only", mode in {"observe", "paper_dry_run", "paper_execute"}, mode)
    add("execution_double_confirmed", mode != "paper_execute" or (bool(cfg.get("execution_enabled")) and execute), {
        "mode": mode,
        "config_execution_enabled": bool(cfg.get("execution_enabled")),
        "cli_execute": bool(execute),
    })
    if cfg.get("trading_hours_only", True):
        add("regular_trading_session", session["in_regular_session"], session)
    else:
        add("regular_trading_session", True, {"disabled_by_config": True, **session})
    add("account_assets_available", total_assets > 0, {"total_assets": total_assets, "available_cash": available_cash})
    add("quotes_available", any(q.get("quote_ok") and not q.get("isSuspended") for q in quotes))
    add("plan_ok", bool(plan.get("ok")), plan.get("reason"))
    add("daily_order_limit", len(orders) <= int(risk.get("max_daily_orders", 5)), len(orders))
    add("limit_orders_only", all(o.get("orderType") == "limit" for o in orders), [o.get("orderType") for o in orders])
    add("etf_whitelist_only", all((str(o.get("stockCode")).zfill(6), o.get("exchange")) in whitelist for o in orders), orders)
    add("positive_lot_quantities", all(int(o.get("quantity", 0)) >= int(risk.get("min_order_quantity", 100)) for o in orders), orders)
    add("available_cash_nonnegative", available_cash >= 0, available_cash)

    approved = all(c["passed"] for c in checks)
    return {
        "approved_for_submit": bool(approved and mode == "paper_execute"),
        "approved_for_planning": all(c["passed"] for c in checks if c["name"] not in {"execution_double_confirmed", "regular_trading_session"}),
        "checks": checks,
        "portfolio_risk_metrics": portfolio_risk_metrics(cfg, plan, balance),
        "session": session,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
    }


def execution_cooldown_check(cfg: dict[str, Any], output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Return whether automated paper execution is allowed by local cadence state."""
    control = cfg.get("execution_control", {})
    state_name = str(control.get("state_file", "execution_state.json"))
    state_path = output_dir / state_name
    state = load_state(state_path)
    today = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    orders = plan.get("orders", []) if isinstance(plan.get("orders"), list) else []
    if not orders:
        return {
            "enabled": bool(control.get("enabled", True)),
            "state_file": str(state_path),
            "trade_date": today,
            "lane": "no_orders",
            "orders_count": 0,
            "allowed": True,
            "reason": "no_orders_to_execute",
        }

    non_t0_orders = [
        f"{o.get('stockCode')}.{o.get('exchange')}"
        for o in orders
        if not bool(o.get("t0_eligible", False))
    ]
    all_orders_stop_loss = all(o.get("direction") == "sell" and o.get("reason") == "stop_loss_exit" for o in orders)
    all_orders_t0 = not non_t0_orders
    lane = "stop_loss_exit" if all_orders_stop_loss else ("t0_intraday" if all_orders_t0 else "t1_or_mixed")

    if lane == "stop_loss_exit":
        max_runs = int(control.get("stop_loss_extra_sells_per_trade_date", 1))
        runs_by_date = state.get("successful_stop_loss_sells_by_date", {})
        last_key = "last_successful_stop_loss_sell_at"
        min_minutes = 0
    elif lane == "t0_intraday":
        max_runs = int(control.get("t0_max_successful_executes_per_trade_date", 3))
        runs_by_date = state.get("successful_t0_executes_by_date", {})
        last_key = "last_successful_t0_execute_at"
        min_minutes = int(control.get("t0_min_minutes_between_executes", 60))
    else:
        max_runs = int(control.get("max_successful_executes_per_trade_date", 1))
        runs_by_date = state.get("successful_executes_by_date", {})
        last_key = "last_successful_execute_at"
        min_minutes = 0

    if not isinstance(runs_by_date, dict):
        runs_by_date = {}
    count_today = int(runs_by_date.get(today, 0) or 0)
    count_allowed = count_today < max_runs

    interval_allowed = True
    minutes_since_last = None
    if min_minutes > 0 and state.get(last_key):
        try:
            last_dt = datetime.fromisoformat(str(state[last_key]))
            now_dt = datetime.now().astimezone()
            minutes_since_last = (now_dt - last_dt.astimezone()).total_seconds() / 60.0
            interval_allowed = minutes_since_last >= min_minutes
        except Exception:
            interval_allowed = False

    allowed = count_allowed and interval_allowed
    if not count_allowed:
        reason = "daily_execute_cooldown_active"
    elif not interval_allowed:
        reason = "t0_min_interval_active"
    else:
        reason = "ok"
    return {
        "enabled": bool(control.get("enabled", True)),
        "state_file": str(state_path),
        "trade_date": today,
        "lane": lane,
        "orders_count": len(orders),
        "all_orders_t0": all_orders_t0,
        "all_orders_stop_loss": all_orders_stop_loss,
        "non_t0_orders": non_t0_orders,
        "max_successful_executes_per_trade_date": max_runs,
        "successful_executes_today": count_today,
        "min_minutes_between_executes": min_minutes,
        "minutes_since_last_execute": minutes_since_last,
        "allowed": bool(allowed),
        "reason": reason,
    }


def mark_execution_success(cfg: dict[str, Any], output_dir: Path, result: dict[str, Any]) -> None:
    control = cfg.get("execution_control", {})
    if not bool(control.get("enabled", True)):
        return
    state_path = output_dir / str(control.get("state_file", "execution_state.json"))
    state = load_state(state_path)
    today = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    cooldown = result.get("execution_cooldown") if isinstance(result.get("execution_cooldown"), dict) else {}
    lane = cooldown.get("lane", "t1_or_mixed")
    if lane == "stop_loss_exit":
        runs_key = "successful_stop_loss_sells_by_date"
        last_key = "last_successful_stop_loss_sell_at"
    elif lane == "t0_intraday":
        runs_key = "successful_t0_executes_by_date"
        last_key = "last_successful_t0_execute_at"
    else:
        runs_key = "successful_executes_by_date"
        last_key = "last_successful_execute_at"
    runs_by_date = state.get(runs_key, {})
    if not isinstance(runs_by_date, dict):
        runs_by_date = {}
    runs_by_date[today] = int(runs_by_date.get(today, 0) or 0) + 1
    state[runs_key] = runs_by_date
    state[last_key] = now_iso()
    state["last_successful_execute_lane"] = lane
    state["last_status"] = result.get("status")
    state["last_orders_count"] = len(result.get("orders", [])) if isinstance(result.get("orders"), list) else 0
    save_state(state_path, state)


def run_agent(config_path: Path, override_mode: str | None = None, execute: bool = False) -> dict[str, Any]:
    cfg = load_json(config_path)
    mode = override_mode or cfg.get("mode", "paper_dry_run")
    output_dir = ROOT / cfg["outputs"]["dir"]
    kill_switch = ROOT / cfg["risk"]["kill_switch_file"]
    output_dir.mkdir(parents=True, exist_ok=True)

    script = expand_path(cfg["skill"]["script"])
    client = SkillClient(python_cmd_from_config(cfg["skill"].get("python", sys.executable)), script)
    result: dict[str, Any] = {
        "timestamp": now_iso(),
        "agent_name": cfg.get("agent_name"),
        "mode": mode,
        "execution_enabled": bool(cfg.get("execution_enabled")),
        "cli_execute": bool(execute),
        "agent_architecture": "AI-Trader-inspired signal/risk/execution/audit state machine",
        "cloud_ai_trader_used": False,
        "asset_type": "ETF_ONLY",
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "kill_switch_active": kill_switch.exists(),
    }
    if kill_switch.exists():
        result.update({"status": "blocked", "reason": "kill_switch_active", "orders": []})
        return result

    session = cn_market_session()
    if cfg.get("trading_hours_only", True) and not session.get("in_regular_session"):
        return market_closed_no_api_result(cfg, mode, execute, output_dir, session)

    today_str = trade_date_cn()
    quota_path = quota_state_path(output_dir, cfg)
    quota_status = quota_backoff_status(quota_path, today_str)
    result["quota_backoff"] = quota_status
    if quota_status.get("active"):
        return quota_backoff_result(
            cfg,
            mode,
            execute,
            output_dir,
            "quota_exhausted_backoff_active",
            quota_status,
        )

    # --- Price history (for momentum signal) ---
    price_history_path = output_dir / cfg["outputs"].get("price_history", "price_history.json")
    score_cfg = cfg.get("strategy", {}).get("score", {})
    momentum_days = int(score_cfg.get("momentum_days", 0)) if not score_cfg.get("use_intraday_return", True) else 0
    price_history = load_price_history(price_history_path)
    result["momentum_days"] = momentum_days
    result["price_history_entries"] = len(price_history)

    balance = client.get_balance()
    positions_resp = client.get_positions()
    result["balance_response_ok"] = bool(balance.get("ok"))
    result["positions_response_ok"] = bool(positions_resp.get("ok"))
    if not balance.get("ok") or not positions_resp.get("ok"):
        if any_quota_exhausted([balance, positions_resp]):
            quota_status = mark_quota_exhausted(quota_path, today_str, "account_or_position_query", balance if is_quota_exhausted_response(balance) else positions_resp)
            return quota_backoff_result(
                cfg,
                mode,
                execute,
                output_dir,
                "quota_exhausted_detected_account_or_position_query",
                quota_status,
            )
        result.update({
            "status": "blocked",
            "reason": "account_or_position_query_failed",
            "balance_error": balance.get("error"),
            "positions_error": positions_resp.get("error"),
            "orders": [],
        })
        return result

    quotes = []
    quote_responses = []
    for etf in cfg["universe"]:
        resp = client.get_quote(etf["stockCode"], etf["exchange"])
        quote_responses.append(resp)
        quotes.append(normalize_quote(etf, resp, price_history, momentum_days))
    if any_quota_exhausted(quote_responses):
        quota_resp = next((x for x in quote_responses if is_quota_exhausted_response(x)), None)
        quota_status = mark_quota_exhausted(quota_path, today_str, "quote_query", quota_resp)
        return quota_backoff_result(
            cfg,
            mode,
            execute,
            output_dir,
            "quota_exhausted_detected_quote_query",
            quota_status,
            quotes,
        )

    # Update and persist price history with today's prevClose values
    update_price_history(price_history, today_str, quotes)
    save_price_history(price_history_path, price_history)
    result["price_history_updated_date"] = today_str
    grid_regime_report = build_grid_regime_report(cfg, quotes, price_history)

    positions = positions_by_code(positions_resp)
    plan = build_plan(cfg, quotes, balance, positions)
    analysis_report = build_analysis_report(cfg, quotes, plan)
    gate = risk_gate(cfg, mode, execute, kill_switch.exists(), plan, quotes, balance)
    cooldown = execution_cooldown_check(cfg, output_dir, plan)
    if mode == "paper_execute" and execute and cooldown.get("enabled", True):
        gate["checks"].append({
            "name": "daily_execution_cooldown_clear",
            "passed": bool(cooldown.get("allowed")),
            "detail": cooldown,
        })
        gate["approved_for_submit"] = bool(gate.get("approved_for_submit") and cooldown.get("allowed"))
    result["quotes"] = quotes
    result["grid_regime_report"] = grid_regime_report
    result["analysis_report"] = analysis_report
    result["plan"] = plan
    result["cash_defense_active"] = bool(plan.get("cash_defense_active", False))
    result["cash_defense_reason"] = plan.get("cash_defense_reason")
    result["cash_defense_threshold"] = plan.get("cash_defense_threshold")
    result["max_score_observed"] = plan.get("max_score_observed")
    result["momentum_warmup_active"] = bool(plan.get("momentum_warmup_active", False))
    result["valid_momentum_score_count"] = plan.get("valid_momentum_score_count")
    result["required_history_count"] = plan.get("required_history_count")
    result["market_breadth_block_active"] = bool(plan.get("market_breadth_block_active", False))
    result["positive_momentum_count"] = plan.get("positive_momentum_count")
    result["min_positive_momentum_count_for_buy"] = plan.get("min_positive_momentum_count_for_buy")
    result["non_risk_orders_blocked_reason"] = plan.get("non_risk_orders_blocked_reason")
    result["risk_report"] = gate
    result["execution_cooldown"] = cooldown
    result["zipline_lifecycle_report"] = zipline_style_lifecycle_report(cfg, mode, quotes, plan, gate)
    agent_name = str(cfg.get("agent_name", "etf_paper_trading_agent"))
    tag_order_owner(plan.get("orders", []) if isinstance(plan.get("orders"), list) else [], agent_name, today_str)
    result["orders"] = plan.get("orders", [])
    result["status"] = "planned" if plan.get("ok") else "blocked"
    result["reason"] = plan.get("reason")

    submit_results = []
    can_execute = bool(gate.get("approved_for_submit"))
    result["cancel_pending_response"] = {
        "attempted": False,
        "reason": "not_in_approved_paper_execute_path",
    }
    with SharedExecutionGuard(
        cfg,
        agent_name=agent_name,
        trade_date=today_str,
        orders=plan.get("orders", []) if isinstance(plan.get("orders"), list) else [],
        can_execute=bool(can_execute and plan.get("ok") and plan.get("orders")),
    ) as shared_guard:
        result["shared_execution_guard"] = shared_guard.report
        if can_execute and plan.get("ok") and plan.get("orders") and not shared_guard.allowed:
            result["status"] = "paper_execute_blocked_shared_execution_guard"
        elif can_execute and plan.get("ok") and plan.get("orders"):
            if cfg.get("cancel_pending_before_rebalance", False):
                cancel_resp = client.cancel_all_pending()
                result["cancel_pending_response"] = {
                    "attempted": True,
                    "ok": cancel_resp.get("ok"),
                    "error": cancel_resp.get("error"),
                }
                if not cancel_resp.get("ok"):
                    result["status"] = "paper_execute_blocked_cancel_pending_failed"
                    result["submit_results"] = []
                    result["execution_report"] = {
                        "attempted": False,
                        "submitted_count": 0,
                        "submit_results": [],
                        "blocked_reason": "cancel_pending_failed",
                    }
                    result["order_blotter_rows"] = build_blotter_rows(result, [])
                    shared_guard.record_submit_results([], status=result["status"], cancel_result=result["cancel_pending_response"])
                    return result
            for order in plan["orders"]:
                submit_results.append(client.submit_order(
                    order["direction"], order["stockCode"], order["exchange"], int(order["quantity"]), order["orderType"], order.get("price")
                ))
            result["status"] = "submitted"
            mark_execution_success(cfg, output_dir, result)
            shared_guard.record_submit_results(submit_results, status=result["status"], cancel_result=result["cancel_pending_response"])
        elif mode == "paper_execute" and not can_execute:
            result["status"] = "paper_execute_blocked_by_risk_gate"
    result["submit_results"] = submit_results
    result["execution_report"] = {
        "attempted": bool(can_execute and plan.get("ok")),
        "submitted_count": len(submit_results),
        "submit_results": submit_results,
    }
    result["order_blotter_rows"] = build_blotter_rows(result, submit_results)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF-only paper trading quant agent")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=["observe", "paper_dry_run", "paper_execute"], default=None)
    parser.add_argument("--execute", action="store_true", help="Allow simulated submitOrder when mode=paper_execute")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json(cfg_path)
    out_dir = ROOT / cfg["outputs"]["dir"]
    latest = out_dir / cfg["outputs"]["latest"]
    jsonl = out_dir / cfg["outputs"]["jsonl"]
    orders_csv = out_dir / cfg["outputs"]["orders"]
    blotter_csv = out_dir / cfg["outputs"].get("blotter", "order_blotter.csv")

    result = run_agent(cfg_path, args.mode, args.execute)
    write_json(latest, result)
    append_jsonl(jsonl, result)
    write_csv(orders_csv, result.get("orders", []))
    append_order_blotter(blotter_csv, result.get("order_blotter_rows", []))
    print(json.dumps({
        "status": result.get("status"),
        "reason": result.get("reason"),
        "mode": result.get("mode"),
        "orders_count": len(result.get("orders", [])),
        "latest": str(latest),
        "blotter": str(blotter_csv),
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
