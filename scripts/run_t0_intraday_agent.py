"""T+0 ETF intraday paper-trading agent.

This is a paper-trading research agent only. It collects minute-level quote
snapshots and can submit simulated paper orders when all gates pass. It is
not live-ready and is not an investment recommendation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import numpy as np
except ImportError:  # pragma: no cover - production fallback when NumPy is absent
    np = None

from run_etf_paper_trading_agent import (
    ROOT,
    SkillClient,
    account_assets,
    append_order_blotter,
    as_float,
    any_quota_exhausted,
    build_blotter_rows,
    cn_market_session,
    expand_path,
    extract_data,
    fetch_quote_responses,
    is_quota_exhausted_response,
    load_json,
    load_state,
    mark_quota_exhausted,
    market_closed_no_api_result,
    positions_by_code,
    python_cmd_from_config,
    quota_backoff_result,
    quota_backoff_status,
    quota_state_path,
    round_lot,
    trade_date_cn,
    write_csv,
    write_json,
)
from shared_paper_trading_guard import SharedExecutionGuard, tag_order_owner
from select_t0_universe import resolve_agent_universe


DEFAULT_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
EXPECTED_EVOLUTION_GATE_VERSION = "dm_hln_mcs_spa_v2"

# Asset classes that are managed for EXIT only and must never be ranked for a
# new T0 BUY. bond_etf: intraday range < round-trip cost. exit_only_t1: a T+1
# broad-based holding parked in the T0 universe purely so the unified sell_score
# engine can liquidate it (buying it intraday would be unsellable same day).
ENTRY_EXCLUDED_ASSET_CLASSES = {"bond_etf", "exit_only_t1"}

MONEY_LIKE_ETF_NAME_RE = re.compile(
    r"(货币|现金|理财|保证金|添利|快线|收益宝|现金通|银华日利|华宝添益|建信添益)"
)

# Entry-oriented + BUY-budget/data checks that must NEVER block a SELL exit
# (stop-loss / profit / liquidation). Otherwise a position is forced to carry
# overnight exactly when risk controls fire. quote_freshness is handled
# separately: bypassed for SELL only on unconditional exits (see check_applies).
SELL_BYPASS_CHECKS = {
    # entry-oriented
    "open_quiet_period_passed",
    "no_new_entry_afternoon_cutoff",
    "broad_market_not_declining",
    "no_existing_non_t0_position_for_entry",
    "quote_liquidity_filter",
    "rolling_vwap_entry_filter",
    "execution_quality_filter",
    "safe_policy_shield",
    "market_correlation_stress_filter",
    "full_market_entry_guard",
    "t0_entry_eligibility_filter",
    "sector_diversification_entry_filter",
    "reentry_cooldown",
    "daily_entry_limit",
    "entry_score_gate",
    "skip_date_guard",
    "kill_switch_inactive",
    # BUY budget/count caps: gate new risk only, never block an exit
    "balance_ok",
    "daily_loss_limit",
    "daily_order_limit",
    "daily_round_trip_limit",
    "no_pending_t0_orders",
}

# Normal-momentum entry gates that an intraday-momentum BUY bypasses (it has its
# own first-half-hour-return signal). Risk-off guards (broad market, correlation
# stress, skip date) and budget/limit/data/session gates are deliberately NOT here.
IM_BUY_BYPASS_CHECKS = {
    "no_new_entry_afternoon_cutoff",
    "entry_score_gate",
    "rolling_vwap_entry_filter",
    "reentry_cooldown",
}

DEFAULT_SELL_THROTTLE_BYPASS_REASONS = {
    "kill_switch_liquidation",
    "emergency_stop_exit",
    "unrecognized_position_state_liquidation",
    "intraday_momentum_eod_exit",
}

DEFAULT_EVOLUTION_ALLOWED_STRATEGY_PATHS = {
    "entry_momentum_pct",
    "exit_momentum_pct",
    "min_profit_exit_pct",
    "min_hold_minutes",
    "loss_review_after_minutes",
    "loss_exit_score_threshold",
    "profit_exit_score_threshold",
    "profit_trailing_drawdown_pct",
    "deceleration_exit_threshold",
    "entry_score_threshold",
    "cross_etf_divergence_threshold",
    "consolidation.max_range_pct",
    "consolidation.breakout_buffer_pct",
    "indicators.rolling_vwap.require_price_above_for_entry",
    "indicators.intraday_atr.stop_multiplier",
    "indicators.intraday_atr.min_stop_pct",
    "indicators.bollinger_squeeze.squeeze_bandwidth_pct",
    "indicators.bollinger_squeeze.breakout_buffer_pct",
    "market_correlation_stress.avg_abs_corr_threshold",
    "bracket.risk_per_trade_pct",
    "bracket.risk_per_trade_pct_chaos_day",
    "bracket.target1_r_multiple",
    "bracket.target2_r_multiple",
    "bracket.reentry_cooldown_minutes",
}

# Apply-time hard bounds (defense-in-depth). Even though the evolution producer
# samples within these ranges, the overlay file is an untrusted trust boundary:
# any numeric leaf outside its [lo, hi] is rejected (not applied). Mirrors
# PARAM_SPACE in scripts/run_t0_strategy_evolution.py. Paths absent here are
# accepted as-is only if booleans/enumerable allowlisted leaves.
EVOLUTION_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "entry_momentum_pct": (0.0012, 0.0028),
    "exit_momentum_pct": (-0.0020, -0.0005),
    "entry_score_threshold": (48, 72),
    "alpha101_conviction_threshold": (0.3, 0.8),
    "loss_exit_score_threshold": (65, 90),
    "profit_exit_score_threshold": (58, 84),
    "min_profit_exit_pct": (0.002, 0.006),
    "min_hold_minutes": (8, 35),
    "loss_review_after_minutes": (10, 40),
    "profit_trailing_drawdown_pct": (-0.012, -0.004),
    "deceleration_exit_threshold": (-0.004, -0.001),
    "cross_etf_divergence_threshold": (0.0015, 0.0035),
    "consolidation.max_range_pct": (0.0020, 0.0040),
    "consolidation.breakout_buffer_pct": (0.0005, 0.0020),
    "indicators.intraday_atr.stop_multiplier": (1.0, 1.8),
    "indicators.intraday_atr.min_stop_pct": (0.0015, 0.0035),
    "indicators.bollinger_squeeze.squeeze_bandwidth_pct": (0.0040, 0.0080),
    "indicators.bollinger_squeeze.breakout_buffer_pct": (0.0003, 0.0012),
    "market_correlation_stress.avg_abs_corr_threshold": (0.60, 0.85),
    "bracket.risk_per_trade_pct": (0.0020, 0.0045),
    "bracket.risk_per_trade_pct_chaos_day": (0.0010, 0.0022),
    "bracket.target1_r_multiple": (0.8, 1.4),
    "bracket.target2_r_multiple": (1.6, 2.6),
    "bracket.reentry_cooldown_minutes": (15, 60),
}


# Virtual clock for offline replay (scripts/replay_t0_decisions.py).
# The live agent never calls set_replay_now, so real-path behavior is unchanged.
_REPLAY_NOW: datetime | None = None


def set_replay_now(dt: datetime | None) -> None:
    global _REPLAY_NOW
    _REPLAY_NOW = dt


def current_dt() -> datetime:
    return _REPLAY_NOW if _REPLAY_NOW is not None else datetime.now().astimezone()


def now_iso() -> str:
    return current_dt().isoformat()


def _flatten_strategy_overlay(obj: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            leaves.extend(_flatten_strategy_overlay(val, path))
        else:
            leaves.append((path, val))
    return leaves


def _set_nested_strategy_value(strategy: dict[str, Any], dotted_path: str, val: Any) -> None:
    cur = strategy
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = val


def apply_evolution_overlay_if_enabled(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply a post-close paper-only strategy overlay.

    This deliberately permits only strategy-parameter leaves from an allowlist.
    Execution mode, execution_enabled, risk limits, order routing, and safety
    locks are not writable through the overlay.
    """
    si = cfg.get("self_iteration", {})
    meta: dict[str, Any] = {
        "enabled": bool(si.get("enabled", False)),
        "auto_apply_changes": bool(si.get("auto_apply_changes", False)),
        "applied": False,
        "reason": "not_enabled",
    }
    if not meta["enabled"]:
        cfg["_evolution_overlay"] = meta
        return cfg
    if not meta["auto_apply_changes"]:
        meta["reason"] = "auto_apply_disabled"
        cfg["_evolution_overlay"] = meta
        return cfg

    overlay_rel = si.get("overlay_path", "outputs/t0_strategy_evolution/latest_strategy_overlay.json")
    overlay_path = Path(str(overlay_rel))
    if not overlay_path.is_absolute():
        overlay_path = ROOT / overlay_path
    meta["overlay_path"] = str(overlay_path)
    if not overlay_path.exists():
        meta["reason"] = "overlay_missing"
        cfg["_evolution_overlay"] = meta
        return cfg

    try:
        overlay = load_json(overlay_path)
    except Exception as exc:
        meta["reason"] = f"overlay_read_failed:{exc}"
        cfg["_evolution_overlay"] = meta
        return cfg

    if overlay.get("status") != "approved_for_paper_auto_apply":
        meta["reason"] = f"overlay_status_not_approved:{overlay.get('status')}"
        meta["selected_candidate"] = overlay.get("selected_candidate")
        cfg["_evolution_overlay"] = meta
        return cfg
    if overlay.get("paper_trading_only") is not True:
        meta["reason"] = "overlay_missing_paper_trading_only"
        cfg["_evolution_overlay"] = meta
        return cfg
    required_gate_version = str(si.get("required_gate_version", EXPECTED_EVOLUTION_GATE_VERSION))
    meta["required_gate_version"] = required_gate_version
    if overlay.get("gate_version") != required_gate_version:
        meta["reason"] = f"overlay_gate_version_mismatch:{overlay.get('gate_version')}!={required_gate_version}"
        meta["selected_candidate"] = overlay.get("selected_candidate")
        cfg["_evolution_overlay"] = meta
        return cfg
    diagnostics_source = "selection_diagnostics"
    diagnostics = overlay.get("selection_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics_source = "auto_apply_risk_limits"
        diagnostics = overlay.get("auto_apply_risk_limits", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    meta["selection_diagnostics_source"] = diagnostics_source
    required_diag_flags = {
        "require_diebold_mariano_significant": True,
        "require_baseline_excluded_from_mcs": True,
        "require_spa_reject": True,
        "require_deflated_sharpe_significant": True,
    }
    missing_required_flags = [
        key for key, expected in required_diag_flags.items()
        if diagnostics.get(key) is not expected
    ]
    if missing_required_flags:
        meta["reason"] = f"overlay_missing_required_selection_flags:{missing_required_flags}"
        meta["selected_candidate"] = overlay.get("selected_candidate")
        cfg["_evolution_overlay"] = meta
        return cfg

    allowed = set(si.get("allowed_strategy_paths") or DEFAULT_EVOLUTION_ALLOWED_STRATEGY_PATHS)
    raw_overlay = overlay.get("strategy_overlay", {})
    if not isinstance(raw_overlay, dict):
        meta["reason"] = "strategy_overlay_not_object"
        cfg["_evolution_overlay"] = meta
        return cfg

    applied: dict[str, Any] = {}
    rejected: list[str] = []
    out_of_bounds: list[str] = []
    for path, val in _flatten_strategy_overlay(raw_overlay):
        if path not in allowed:
            rejected.append(path)
            continue
        bounds = EVOLUTION_PARAM_BOUNDS.get(path)
        if bounds is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
            lo, hi = bounds
            if not (lo <= float(val) <= hi):
                out_of_bounds.append(f"{path}={val}∉[{lo},{hi}]")
                continue
        _set_nested_strategy_value(cfg["strategy"], path, val)
        applied[path] = val

    meta.update({
        "applied": bool(applied),
        "reason": "applied" if applied else "no_allowed_changes",
        "selected_candidate": overlay.get("selected_candidate"),
        "created_at": overlay.get("created_at"),
        "applied_paths": applied,
        "rejected_paths": rejected,
        "out_of_bounds_paths": out_of_bounds,
    })
    cfg["_evolution_overlay"] = meta
    return cfg


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def call_pending_orders(client: SkillClient) -> dict[str, Any]:
    return client.call("listPendingOrders")


def call_trade_history(client: SkillClient, start_date: str, end_date: str) -> dict[str, Any]:
    return client.call("listTradeHistory", "--start-date", start_date, "--end-date", end_date)


def t0_quota_backoff_result(
    cfg: dict[str, Any],
    execute: bool,
    out_dir: Path,
    reason: str,
    quota_status: dict[str, Any],
    quotes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    session = cn_market_session()
    local_time = exchange_local_time(session)
    base = quota_backoff_result(
        cfg,
        cfg.get("mode", "paper_dry_run"),
        execute,
        out_dir,
        reason,
        quota_status,
        quotes,
    )
    base.update({
        "agent_name": cfg.get("agent_name"),
        "trade_date": trade_date_cn(),
        "session": session,
        "system_time": {
            "system_timezone": "Asia/Seoul",
            "system_local_time": system_local_time().isoformat(),
            "exchange_timezone": "Asia/Shanghai",
            "exchange_local_time": local_time.isoformat(),
            "time_guard_basis": "exchange_local_time",
        },
        "ranked": [],
        "positions_t0": {},
        "t0_inventory": {},
        "t0_sellable_by_code": {},
        "pending_t0_orders": [],
        "volume_filter_status": "quota_exhausted_backoff",
        "state_machine": {
            "state": "quota_backoff",
            "action": "hold",
            "reason": reason,
        },
        "risk_checks": [
            {
                "name": "quota_backoff_inactive",
                "passed": False,
                "detail": quota_status,
            }
        ],
        "approved_for_submit": False,
        "fill_reconciliation": {
            "attempted": False,
            "reason": "quota_exhausted_backoff",
        },
        "cancel_result": {
            "attempted": False,
            "reason": "quota_exhausted_backoff",
        },
    })
    return base


def t0_market_closed_no_api_result(
    cfg: dict[str, Any],
    execute: bool,
    out_dir: Path,
    session: dict[str, Any],
) -> dict[str, Any]:
    local_time = exchange_local_time(session)
    base = market_closed_no_api_result(
        cfg,
        cfg.get("mode", "paper_dry_run"),
        execute,
        out_dir,
        session,
    )
    base.update({
        "agent_name": cfg.get("agent_name"),
        "trade_date": trade_date_cn(),
        "session": session,
        "system_time": {
            "system_timezone": "Asia/Seoul",
            "system_local_time": system_local_time().isoformat(),
            "exchange_timezone": "Asia/Shanghai",
            "exchange_local_time": local_time.isoformat(),
            "time_guard_basis": "exchange_local_time",
        },
        "ranked": [],
        "positions_t0": {},
        "t0_inventory": {},
        "t0_sellable_by_code": {},
        "pending_t0_orders": [],
        "volume_filter_status": "market_closed_no_api",
        "state_machine": {
            "state": "market_closed",
            "action": "hold",
            "reason": "outside_regular_trading_session_no_api_calls",
        },
        "risk_checks": [{
            "name": "regular_trading_session",
            "passed": False,
            "detail": session,
        }],
        "approved_for_submit": False,
        "fill_reconciliation": {
            "attempted": False,
            "reason": "market_closed_no_api",
        },
        "cancel_result": {
            "attempted": False,
            "reason": "market_closed_no_api",
        },
    })
    return base


def t0_api_budget_result(
    cfg: dict[str, Any],
    execute: bool,
    out_dir: Path,
    trade_date: str,
    runs_today: int,
    max_runs: int,
) -> dict[str, Any]:
    session = cn_market_session()
    local_time = exchange_local_time(session)
    return {
        "timestamp": now_iso(),
        "agent_name": cfg.get("agent_name"),
        "mode": cfg.get("mode", "paper_dry_run"),
        "execution_enabled": bool(cfg.get("execution_enabled")),
        "cli_execute": bool(execute),
        "asset_type": cfg.get("asset_type"),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "status": "api_run_budget_backoff",
        "reason": "max_daily_api_runs_reached",
        "trade_date": trade_date,
        "api_run_budget": {
            "runs_today": runs_today,
            "max_daily_api_runs": max_runs,
            "policy": "no_further_api_calls_until_next_trade_date",
        },
        "session": session,
        "system_time": {
            "system_timezone": "Asia/Seoul",
            "system_local_time": system_local_time().isoformat(),
            "exchange_timezone": "Asia/Shanghai",
            "exchange_local_time": local_time.isoformat(),
            "time_guard_basis": "exchange_local_time",
        },
        "quotes": [],
        "ranked": [],
        "positions_t0": {},
        "t0_inventory": {},
        "t0_sellable_by_code": {},
        "pending_t0_orders": [],
        "volume_filter_status": "api_run_budget_backoff",
        "state_machine": {
            "state": "api_budget_backoff",
            "action": "hold",
            "reason": "max_daily_api_runs_reached",
        },
        "risk_checks": [{
            "name": "daily_api_run_budget_available",
            "passed": False,
            "detail": {"runs_today": runs_today, "max_daily_api_runs": max_runs},
        }],
        "approved_for_submit": False,
        "orders": [],
        "submit_results": [],
        "cancel_result": {"attempted": False, "reason": "api_run_budget_backoff"},
        "fill_reconciliation": {"attempted": False, "reason": "api_run_budget_backoff"},
        "output_dir": str(out_dir),
    }


def t0_quote_only_broker_query_throttled_result(
    cfg: dict[str, Any],
    execute: bool,
    out_dir: Path,
    trade_date: str,
    session: dict[str, Any],
    quotes: list[dict[str, Any]],
    market_data_report: dict[str, Any],
    throttle_detail: dict[str, Any],
) -> dict[str, Any]:
    local_time = exchange_local_time(session)
    return {
        "timestamp": now_iso(),
        "agent_name": cfg.get("agent_name"),
        "mode": cfg.get("mode", "paper_dry_run"),
        "execution_enabled": bool(cfg.get("execution_enabled")),
        "cli_execute": bool(execute),
        "asset_type": cfg.get("asset_type"),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "status": "quote_only_broker_query_throttled",
        "reason": "huatai_broker_query_interval_active",
        "trade_date": trade_date,
        "session": session,
        "system_time": {
            "system_timezone": "Asia/Seoul",
            "system_local_time": system_local_time().isoformat(),
            "exchange_timezone": "Asia/Shanghai",
            "exchange_local_time": local_time.isoformat(),
            "time_guard_basis": "exchange_local_time",
        },
        "quotes": quotes,
        "market_data_report": market_data_report,
        "broker_query_throttle": throttle_detail,
        "ranked": [],
        "positions_t0": {},
        "t0_inventory": {},
        "t0_sellable_by_code": {},
        "pending_t0_orders": [],
        "volume_filter_status": "eastmoney_quote_only",
        "state_machine": {
            "state": "quote_only",
            "action": "hold",
            "reason": "huatai_broker_query_interval_active",
        },
        "risk_checks": [{
            "name": "huatai_broker_query_interval_clear",
            "passed": False,
            "detail": throttle_detail,
        }],
        "approved_for_submit": False,
        "orders": [],
        "submit_results": [],
        "cancel_result": {"attempted": False, "reason": "broker_query_throttled"},
        "fill_reconciliation": {"attempted": False, "reason": "broker_query_throttled"},
        "output_dir": str(out_dir),
    }


def normalize_quote(etf: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
    data = extract_data(resp)
    current = as_float(data.get("currentPrice"))
    bid = as_float(data.get("bidPrice1"), current)
    ask = as_float(data.get("askPrice1"), current)
    spread_pct = (ask - bid) / current if current > 0 and ask > 0 and bid > 0 else None
    midpoint = (bid + ask) / 2.0 if bid > 0 and ask > 0 else current
    volume = data.get("volume") or data.get("turnoverVolume") or data.get("成交量")
    amount = data.get("amount") or data.get("turnoverAmount") or data.get("成交额")
    return {
        "timestamp": now_iso(),
        "stockCode": etf["stockCode"],
        "exchange": etf["exchange"],
        "name": data.get("stockName") or etf.get("name"),
        "asset_class": etf.get("asset_class"),
        "currentPrice": current,
        "prevClose": as_float(data.get("prevClose")),
        "bidPrice1": bid,
        "askPrice1": ask,
        "midpoint": midpoint,
        "spread_pct": spread_pct,
        "change_pct": as_float(data.get("change")) / 100.0,
        "isSuspended": bool(data.get("isSuspended", False)),
        "quote_ok": bool(resp.get("ok")),
        "volume": as_float(volume, -1.0) if volume is not None else None,
        "amount": as_float(amount, -1.0) if amount is not None else None,
        "raw_has_volume_field": volume is not None,
        "raw_has_amount_field": amount is not None,
        "quote_error": None if resp.get("ok") else resp.get("error"),
        "quote_source": data.get("source") or resp.get("_market_data_provider") or resp.get("_cmd_tool"),
        "source_quote_time": data.get("source_quote_time"),
    }


def load_recent_quotes(path: Path, lookback_rows: int = 2000, *, universe_size: int = 0,
                       min_per_code: int = 60) -> list[dict[str, Any]]:
    """Load recent minute-quote rows. The row budget scales with universe size so
    per-code history depth stays sufficient for the indicators (rolling VWAP 5+,
    ATR/Bollinger 20, correlation-stress 30) -- a fixed 2000-row cap would starve
    them once the dynamic universe grows to hundreds of names (~7 rows/code).
    Reads via a bounded tail (deque) so memory stays O(budget), not O(filesize)."""
    if not path.exists():
        return []
    from collections import deque
    effective = max(lookback_rows, int(universe_size) * int(min_per_code))
    with path.open("r", encoding="utf-8") as fh:
        lines = deque(fh, maxlen=effective)
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def dated_output_path(out_dir: Path, template: Any, trade_date: str) -> Path:
    """Resolve an output filename for a trading day.

    New configs should use a ``{trade_date}`` placeholder. Legacy filenames are
    still rolled by appending the date before the extension, so an old
    ``minute_quotes.jsonl`` setting becomes ``minute_quotes_YYYY-MM-DD.jsonl``.
    """
    name = str(template or "").strip()
    if not name:
        raise ValueError("empty output filename template")
    if "{trade_date}" in name:
        return out_dir / name.format(trade_date=trade_date)
    path = out_dir / name
    if path.suffix:
        return path.with_name(f"{path.stem}_{trade_date}{path.suffix}")
    return path.with_name(f"{path.name}_{trade_date}")


def classify_etf_sector(name: Any, keyword_map: dict[str, Any] | None = None) -> str:
    """Classify an ETF name into a coarse sector bucket using configured keywords."""
    text = str(name or "")
    if not text:
        return "other"
    if isinstance(keyword_map, dict):
        for keyword, sector in keyword_map.items():
            kw = str(keyword or "").strip()
            if kw and kw in text:
                sec = str(sector or "other").strip()
                return sec or "other"
    return "other"


def compute_consolidation_box(hist: list[dict[str, Any]], ccfg: dict[str, Any]) -> dict[str, Any] | None:
    if not ccfg.get("enabled"):
        return None
    win = int(as_float(ccfg.get("window_snapshots", 6), 6))
    if win < 3 or len(hist) < win:
        return None
    window = hist[-win:]
    prices = [as_float(r.get("currentPrice")) for r in window]
    if any(p <= 0 for p in prices):
        return None
    first_ts = parse_iso_dt(window[0].get("timestamp"))
    last_ts = parse_iso_dt(window[-1].get("timestamp"))
    # 数据断档(跨午休/停更)时窗口跨度异常, 放弃箱体
    if first_ts is None or last_ts is None or (last_ts - first_ts).total_seconds() > win * 7 * 60:
        return None
    box_high = max(prices)
    box_low = min(prices)
    box_mid = (box_high + box_low) / 2.0
    if box_mid <= 0:
        return None
    box_range_pct = (box_high - box_low) / box_mid
    if box_range_pct > as_float(ccfg.get("max_range_pct", 0.003)):
        return None
    return {
        "high": box_high,
        "low": box_low,
        "mid": round(box_mid, 4),
        "range_pct": round(box_range_pct, 5),
        "window_snapshots": win,
    }


def contiguous_recent_window(hist: list[dict[str, Any]], window: int, max_gap_minutes: float | None = None) -> list[dict[str, Any]]:
    if window <= 0 or len(hist) < window:
        return []
    rows = hist[-window:]
    if max_gap_minutes is None:
        max_gap_minutes = window * 7
    first_ts = parse_iso_dt(rows[0].get("timestamp"))
    last_ts = parse_iso_dt(rows[-1].get("timestamp"))
    if first_ts is None or last_ts is None:
        return []
    if (last_ts - first_ts).total_seconds() > max_gap_minutes * 60:
        return []
    return rows


def inventory_skew_ticks(exec_cfg: dict[str, Any], inventory_ratio: float, volatility: float,
                         minutes_to_close: float | None) -> int:
    """Avellaneda-Stoikov (2006) inventory term, adapted long-only and tick-quantized.

    A-S skews a dealer's reservation price away from inventory by q*gamma*sigma^2*(T-t).
    We are a LONG-ONLY buyer, so the only actionable direction is to post our passive
    BUY *lower* (less eager to add) as our book fills up. The raw A-S term is sub-tick
    at intraday ETF scale, so we express it as a small, bounded number of TICKS that
    grows with: inventory_ratio (q, how full vs target_holdings), volatility (sigma,
    via ATR%), and time-to-close (T-t, more remaining horizon => more inventory risk).
    Returns 0 (no skew) when the feature is disabled or inventory is empty. This only
    ever lowers fill probability on an OPTIONAL entry, so it carries no downside risk;
    by design it stays ~0 for the first holdings and only bites near capacity."""
    skew_cfg = exec_cfg.get("inventory_skew", {}) if isinstance(exec_cfg.get("inventory_skew"), dict) else {}
    if not bool(skew_cfg.get("enabled", False)) or inventory_ratio <= 0:
        return 0
    gamma = as_float(skew_cfg.get("risk_aversion", 2.0), 2.0)
    ref_sigma = as_float(skew_cfg.get("reference_volatility", 0.01), 0.01)
    vol_cap = as_float(skew_cfg.get("vol_multiplier_cap", 3.0), 3.0)
    max_skew = int(as_float(skew_cfg.get("max_skew_ticks", 2), 2))
    session_minutes = as_float(skew_cfg.get("session_minutes", 240.0), 240.0)
    vol_mult = min(volatility / ref_sigma, vol_cap) if ref_sigma > 0 and volatility > 0 else 1.0
    time_frac = max(0.0, min(1.0, minutes_to_close / session_minutes)) if minutes_to_close is not None and session_minutes > 0 else 1.0
    skew_raw = gamma * min(1.0, inventory_ratio) * vol_mult * time_frac
    return max(0, min(max_skew, int(round(skew_raw))))


def passive_entry_price(q: dict[str, Any], risk: dict[str, Any], exec_cfg: dict[str, Any], *,
                        inventory_ratio: float = 0.0, volatility: float = 0.0,
                        minutes_to_close: float | None = None) -> tuple[float, str, float]:
    """Entry limit price. Passive (post at the bid -> EARN the spread instead of
    paying it; ~5-26 bps for our universe) when enabled and a valid book exists.
    Falls back to the aggressive cross (ask x (1+slippage)) otherwise. Returns
    (price, execution_style, mid). Entries are optional so non-fill has no downside;
    exits stay aggressive (handled separately) for fill certainty.

    When execution.inventory_skew is enabled, the passive bid is pushed down by an
    Avellaneda-Stoikov inventory term (see inventory_skew_ticks): the fuller our book
    (vs target_holdings) and the more volatile/early the session, the more conservative
    we post -- never above the ask, never below zero."""
    ask = as_float(q.get("askPrice1"), q.get("currentPrice"))
    bid = as_float(q.get("bidPrice1"), 0.0)
    slip = as_float(risk.get("limit_price_slippage_pct"), 0.001)
    aggressive = round(ask * (1.0 + slip), 3)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ask
    if not bool(exec_cfg.get("passive_entry_enabled", True)) or bid <= 0 or ask <= 0 or ask <= bid:
        return aggressive, "aggressive", mid
    tick = as_float(exec_cfg.get("tick_size", 0.001), 0.001)
    offset = int(as_float(exec_cfg.get("passive_offset_ticks", 0), 0))
    passive = bid + offset * tick
    passive = min(passive, ask - tick)  # must stay strictly passive (below the ask)
    skew_ticks = inventory_skew_ticks(exec_cfg, inventory_ratio, volatility, minutes_to_close)
    if skew_ticks > 0:
        passive = passive - skew_ticks * tick
    if passive <= 0:
        return aggressive, "aggressive", mid
    return round(passive, 3), "passive", mid


def verify_sell_orders_against_broker(orders: list[dict[str, Any]], broker_avail: dict[str, float],
                                      pos_ok: bool, min_qty: int, lot: int) -> list[dict[str, Any]]:
    """Drop/cap SELL orders against the broker's ACTUAL availableQuantity before
    submission, so we never queue a stuck pending order for phantom local
    inventory. Buys pass through. If positions could not be fetched, only
    unconditional exits (emergency/kill_switch/IM-eod) are allowed through."""
    out: list[dict[str, Any]] = []
    for o in orders:
        if o.get("direction") != "sell":
            out.append(o)
            continue
        code = str(o.get("stockCode", "")).zfill(6)
        if not pos_ok:
            if bool(o.get("unconditional_exit")):
                out.append(o)
            else:
                o["skipped_reason"] = "position_unverified"
            continue
        avail = broker_avail.get(code, 0.0)
        if avail < min_qty:
            o["skipped_reason"] = f"broker_holds_insufficient:{avail}"
            continue
        if int(as_float(o.get("quantity"))) > avail:
            o["quantity"] = round_lot(avail, lot)
            o["quantity_capped_to_broker_available"] = True
        out.append(o)
    return out


def compute_rolling_vwap(hist: list[dict[str, Any]], current_quote: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg.get("enabled", True):
        return {"available": False, "status": "disabled"}
    window = int(as_float(cfg.get("window_snapshots", 20), 20))
    min_points = int(as_float(cfg.get("min_points", 5), 5))
    rows = contiguous_recent_window(hist + [current_quote], window)
    if len(rows) < min_points:
        return {"available": False, "status": "insufficient_history", "window_snapshots": window}

    pairs = [(as_float(r.get("volume"), 0.0), as_float(r.get("amount"), 0.0)) for r in rows]
    if not all(v > 0 and a > 0 for v, a in pairs):
        return {"available": False, "status": "missing_volume_or_amount", "window_snapshots": window}

    delta_amount = 0.0
    delta_volume = 0.0
    for (prev_v, prev_a), (cur_v, cur_a) in zip(pairs, pairs[1:]):
        dv = cur_v - prev_v
        da = cur_a - prev_a
        if dv > 0 and da > 0:
            delta_volume += dv
            delta_amount += da
    if delta_volume > 0 and delta_amount > 0:
        vwap = delta_amount / delta_volume
        method = "positive_cumulative_deltas"
    else:
        total_volume = sum(v for v, _ in pairs)
        total_amount = sum(a for _, a in pairs)
        if total_volume <= 0 or total_amount <= 0:
            return {"available": False, "status": "invalid_volume_amount", "window_snapshots": window}
        vwap = total_amount / total_volume
        method = "sum_amount_over_sum_volume"

    current = as_float(current_quote.get("currentPrice"), 0.0)
    return {
        "available": bool(vwap > 0),
        "status": "available" if vwap > 0 else "invalid_vwap",
        "vwap": vwap if vwap > 0 else None,
        "price_above_vwap": bool(current > 0 and vwap > 0 and current >= vwap),
        "distance_pct": (current / vwap - 1.0) if current > 0 and vwap > 0 else None,
        "method": method,
        "window_snapshots": window,
    }


def compute_atr_proxy(hist: list[dict[str, Any]], current_quote: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg.get("enabled", True):
        return {"available": False, "status": "disabled"}
    window = int(as_float(cfg.get("window_snapshots", 20), 20))
    rows = contiguous_recent_window(hist + [current_quote], window)
    if len(rows) < window:
        return {"available": False, "status": "insufficient_history", "window_snapshots": window}
    prices = [as_float(r.get("currentPrice"), 0.0) for r in rows]
    if any(p <= 0 for p in prices):
        return {"available": False, "status": "invalid_price", "window_snapshots": window}
    abs_rets = [abs(cur / prev - 1.0) for prev, cur in zip(prices, prices[1:]) if prev > 0 and cur > 0]
    if not abs_rets:
        return {"available": False, "status": "no_returns", "window_snapshots": window}
    atr_pct = sum(abs_rets) / len(abs_rets)
    multiplier = as_float(cfg.get("stop_multiplier", 1.2), 1.2)
    min_stop_pct = as_float(cfg.get("min_stop_pct", 0.002), 0.002)
    stop_distance_pct = max(min_stop_pct, multiplier * atr_pct)
    return {
        "available": True,
        "status": "available",
        "atr_pct": atr_pct,
        "stop_distance_pct": stop_distance_pct,
        "stop_multiplier": multiplier,
        "min_stop_pct": min_stop_pct,
        "window_snapshots": window,
    }


def compute_bollinger_squeeze(hist: list[dict[str, Any]], current_quote: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg.get("enabled", True):
        return {"available": False, "status": "disabled"}
    window = int(as_float(cfg.get("window_snapshots", 20), 20))
    rows = contiguous_recent_window(hist, window)
    if len(rows) < window:
        return {"available": False, "status": "insufficient_history", "window_snapshots": window}
    prices = [as_float(r.get("currentPrice"), 0.0) for r in rows]
    current = as_float(current_quote.get("currentPrice"), 0.0)
    if current <= 0 or any(p <= 0 for p in prices):
        return {"available": False, "status": "invalid_price", "window_snapshots": window}
    mid = sum(prices) / len(prices)
    if mid <= 0:
        return {"available": False, "status": "invalid_mid", "window_snapshots": window}
    variance = sum((p - mid) ** 2 for p in prices) / len(prices)
    std = math.sqrt(variance)
    std_mult = as_float(cfg.get("std_mult", 2.0), 2.0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bandwidth = (upper - lower) / mid if mid > 0 else None
    squeeze_threshold = as_float(cfg.get("squeeze_bandwidth_pct", 0.006), 0.006)
    breakout_buffer = as_float(cfg.get("breakout_buffer_pct", 0.0005), 0.0005)
    squeeze = bandwidth is not None and bandwidth <= squeeze_threshold
    breakout = bool(squeeze and current > upper * (1.0 + breakout_buffer))
    return {
        "available": True,
        "status": "available",
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "bandwidth_pct": bandwidth,
        "squeeze": squeeze,
        "breakout": breakout,
        "std_mult": std_mult,
        "squeeze_bandwidth_pct": squeeze_threshold,
        "breakout_buffer_pct": breakout_buffer,
        "window_snapshots": window,
    }


def compute_market_correlation_stress(
    history: list[dict[str, Any]],
    quotes: list[dict[str, Any]],
    cfg: dict[str, Any],
    history_by_code: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Diagnose high cross-ETF correlation regimes from local quote snapshots.

    This is a defensive market-state filter, not an alpha signal. It only uses
    already-collected local quote rows and current quotes. Data gaps, bad quotes,
    and insufficient samples return an unavailable diagnostic rather than a block.
    """
    if not cfg.get("enabled", True):
        return {"available": False, "status": "disabled", "block_new_buy": False}

    window = int(as_float(cfg.get("window_snapshots", 30), 30))
    min_assets = int(as_float(cfg.get("min_assets", 4), 4))
    min_snapshots = int(as_float(cfg.get("min_snapshots", max(8, min(window, 10))), max(8, min(window, 10))))
    corr_threshold = as_float(cfg.get("avg_abs_corr_threshold", 0.75), 0.75)
    breadth_max = int(as_float(cfg.get("breadth_positive_count_max", 1), 1))
    max_gap_seconds = as_float(cfg.get("max_snapshot_gap_seconds", 480), 480)

    if history_by_code is None:
        source_rows = history + quotes
    else:
        # Replay optimization: the caller already owns an equivalent per-code
        # index, so avoid rebuilding/scanning the full cross-sectional history
        # on every minute. Live callers keep using the legacy list path.
        source_rows = []
        for quote in quotes:
            code = str(quote.get("stockCode", "")).zfill(6)
            source_rows.extend(history_by_code.get(code, [])[-window:])
            source_rows.append(quote)
    rows = [
        r for r in source_rows
        if r.get("quote_ok") and r.get("asset_class") not in ENTRY_EXCLUDED_ASSET_CLASSES and as_float(r.get("currentPrice"), 0.0) > 0
    ]
    snapshots: dict[str, dict[str, float]] = {}
    snapshot_times: dict[str, datetime] = {}
    for row in rows:
        ts_text = str(row.get("timestamp") or "")
        key = ts_text[:16] if len(ts_text) >= 16 and ts_text[10:11] == "T" else ""
        if not key:
            dt = parse_iso_dt(ts_text)
            if dt is None:
                continue
            key = dt.strftime("%Y-%m-%dT%H:%M")
        else:
            dt = snapshot_times.get(key)
            if dt is None:
                dt = parse_iso_dt(ts_text)
                if dt is None:
                    continue
        # The agent queries ETFs sequentially; grouping to the minute captures a
        # single scan without requiring identical second-level timestamps.
        code = str(row.get("stockCode", "")).zfill(6)
        if not code:
            continue
        snapshots.setdefault(key, {})[code] = as_float(row.get("currentPrice"), 0.0)
        snapshot_times[key] = dt

    ordered_keys = sorted(snapshots, key=lambda k: snapshot_times[k])
    if len(ordered_keys) < min_snapshots:
        return {
            "available": False,
            "status": "insufficient_snapshots",
            "block_new_buy": False,
            "snapshots": len(ordered_keys),
            "min_snapshots": min_snapshots,
        }

    recent_keys = ordered_keys[-window:]
    contiguous_keys: list[str] = []
    prev_dt: datetime | None = None
    for key in recent_keys:
        dt = snapshot_times[key]
        if prev_dt is not None and (dt - prev_dt).total_seconds() > max_gap_seconds:
            contiguous_keys = []
        contiguous_keys.append(key)
        prev_dt = dt
    if len(contiguous_keys) < min_snapshots:
        return {
            "available": False,
            "status": "insufficient_contiguous_snapshots",
            "block_new_buy": False,
            "snapshots": len(contiguous_keys),
            "min_snapshots": min_snapshots,
        }

    common_codes = set(snapshots[contiguous_keys[0]].keys())
    for key in contiguous_keys[1:]:
        common_codes &= set(snapshots[key].keys())
    common_codes = {code for code in common_codes if all(as_float(snapshots[key].get(code), 0.0) > 0 for key in contiguous_keys)}
    if len(common_codes) < min_assets:
        return {
            "available": False,
            "status": "insufficient_assets",
            "block_new_buy": False,
            "asset_count": len(common_codes),
            "min_assets": min_assets,
        }

    returns_by_code: dict[str, list[float]] = {}
    for code in sorted(common_codes):
        prices = [as_float(snapshots[key].get(code), 0.0) for key in contiguous_keys]
        returns = [cur / prev - 1.0 for prev, cur in zip(prices, prices[1:]) if prev > 0 and cur > 0]
        if len(returns) >= min_snapshots - 1:
            returns_by_code[code] = returns
    if len(returns_by_code) < min_assets:
        return {
            "available": False,
            "status": "insufficient_return_series",
            "block_new_buy": False,
            "asset_count": len(returns_by_code),
            "min_assets": min_assets,
        }

    def corr(xs: list[float], ys: list[float]) -> float | None:
        n = min(len(xs), len(ys))
        if n < 3:
            return None
        x = xs[-n:]
        y = ys[-n:]
        mx = sum(x) / n
        my = sum(y) / n
        vx = sum((v - mx) ** 2 for v in x)
        vy = sum((v - my) ** 2 for v in y)
        if vx <= 0 or vy <= 0:
            return None
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)

    codes = sorted(returns_by_code)
    pair_corrs: list[float] = []
    correlation_engine = "python_pairwise"
    if np is not None and len(codes) >= 2:
        matrix = np.asarray([returns_by_code[code] for code in codes], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_matrix = np.corrcoef(matrix)
        upper = np.abs(corr_matrix[np.triu_indices(len(codes), k=1)])
        finite = upper[np.isfinite(upper)]
        pair_corrs = finite.tolist()
        correlation_engine = "numpy_exact_matrix"
    else:
        for i, code_a in enumerate(codes):
            for code_b in codes[i + 1:]:
                val = corr(returns_by_code[code_a], returns_by_code[code_b])
                if val is not None:
                    pair_corrs.append(abs(val))
    if not pair_corrs:
        return {"available": False, "status": "no_valid_pair_correlations", "block_new_buy": False}

    avg_abs_corr = sum(pair_corrs) / len(pair_corrs)
    max_abs_corr = max(pair_corrs)
    non_bond_quotes = [q for q in quotes if q.get("quote_ok") and q.get("asset_class") not in ENTRY_EXCLUDED_ASSET_CLASSES]
    positive_count = sum(1 for q in non_bond_quotes if as_float(q.get("change_pct")) > -0.005)
    block_new_buy = avg_abs_corr > corr_threshold and positive_count <= breadth_max
    return {
        "available": True,
        "status": "available",
        "block_new_buy": bool(block_new_buy),
        "reason": "correlation_stress_market_beta_dominant" if block_new_buy else "correlation_stress_not_triggered",
        "avg_abs_corr": avg_abs_corr,
        "max_abs_corr": max_abs_corr,
        "pair_count": len(pair_corrs),
        "correlation_engine": correlation_engine,
        "asset_count": len(codes),
        "snapshots": len(contiguous_keys),
        "window_snapshots": window,
        "avg_abs_corr_threshold": corr_threshold,
        "positive_count": positive_count,
        "breadth_positive_count_max": breadth_max,
        "codes": codes,
    }


def compute_first_half_hour_return(hist: list[dict[str, Any]], current_quote: dict[str, Any], im_cfg: dict[str, Any]) -> float | None:
    """Market-intraday-momentum signal (Gao-Han-Li-Zhou 2018): first-half-hour
    return from the previous close, r = price(~10:00) / prevClose - 1. Returns
    None if today's pre-10:00 snapshot or prevClose is unavailable. Data-self-
    sufficient (own price only)."""
    if not im_cfg.get("enabled"):
        return None
    prev_close = as_float(current_quote.get("prevClose"))
    if prev_close <= 0:
        return None
    end_hhmm = str(im_cfg.get("first_half_hour_end", "10:00"))
    try:
        eh, em = (int(x) for x in end_hhmm.split(":"))
    except Exception:
        eh, em = 10, 0
    cutoff_minute = eh * 60 + em
    today = current_dt().astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    candidates: list[tuple[int, float]] = []
    for r in hist:
        ts_text = str(r.get("timestamp") or "")
        if ts_text[:10] != today:
            continue
        try:
            minute = int(ts_text[11:13]) * 60 + int(ts_text[14:16])
        except (TypeError, ValueError):
            ts = parse_iso_dt(ts_text)
            if ts is None:
                continue
            ts_sh = ts.astimezone(ZoneInfo("Asia/Shanghai"))
            minute = ts_sh.hour * 60 + ts_sh.minute
        if minute > cutoff_minute:
            continue
        candidates.append((minute, as_float(r.get("currentPrice"))))
    if not candidates:
        return None
    candidates.sort()
    price_first = candidates[-1][1]  # snapshot closest to (and at/before) 10:00
    if price_first <= 0:
        return None
    return round(price_first / prev_close - 1.0, 5)


def compute_alpha101_conviction(hist: list[dict[str, Any]], current_quote: dict[str, Any]) -> float | None:
    """Alpha#101 (Kakushadze 2015), single-name and adapted to the live session:
    (close - open) / (high - low + eps). We map close=current price, open=first
    snapshot today, high/low=intraday extrema (incl. current). Range ~[-1, 1]:
    +1 = opened at the low and now at the high (strong up-conviction), -1 the reverse.

    We use ONLY this non-cross-sectional formula from the 101: the paper's alphas are
    overwhelmingly rank()-based across a broad universe, which is meaningless at our
    N=7 ETF breadth. This is a long-only momentum CONFIRMATION feature (orthogonal to
    short-window momentum: it captures intraday directional persistence), not a
    standalone signal. Returns None when today's session history is too thin."""
    cur = as_float(current_quote.get("currentPrice"))
    if cur <= 0:
        return None
    today = current_dt().astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    prices: list[float] = []
    for r in hist:
        ts_text = str(r.get("timestamp") or "")
        if ts_text[:10] != today:
            continue
        p = as_float(r.get("currentPrice"))
        if p > 0:
            prices.append(p)
    prices.append(cur)
    if len(prices) < 3:
        return None
    session_open = prices[0]
    rng = max(prices) - min(prices)
    if rng <= 0:
        return 0.0
    return round((cur - session_open) / (rng + 1e-9), 4)


def compute_volume_surge(hist: list[dict[str, Any]], current_quote: dict[str, Any],
                         recent_n: int = 3, baseline_n: int = 10) -> float | None:
    """Intraday volume-surge proxy (the early-accumulation tell). Eastmoney volume is
    CUMULATIVE within the day, so we difference it into per-snapshot increments and
    compare the recent average increment to an earlier baseline: >1 means volume is
    accelerating (often LEADS the price move), ~1 flat, <1 fading. Cumulative resets
    or noise are clamped to 0. Returns None until enough same-day volume history
    exists. This is logged for research first; it is NOT yet an entry gate."""
    cur_vol = as_float(current_quote.get("volume"), 0.0)
    vols = [as_float(r.get("volume"), 0.0) for r in hist]
    vols.append(cur_vol)
    vols = [v for v in vols if v > 0]
    if len(vols) < baseline_n + recent_n + 1:
        return None
    incs: list[float] = []
    for a, b in zip(vols, vols[1:]):
        d = b - a
        incs.append(d if d > 0 else 0.0)
    if len(incs) < baseline_n + recent_n:
        return None
    recent = incs[-recent_n:]
    baseline = incs[:-recent_n]
    recent_avg = sum(recent) / len(recent)
    baseline_avg = (sum(baseline) / len(baseline)) if baseline else 0.0
    if baseline_avg <= 0:
        return None
    return round(recent_avg / baseline_avg, 3)


def _hhmm_to_minutes(value: str, default: int) -> int:
    try:
        h, m = (int(x) for x in str(value).split(":"))
        return h * 60 + m
    except Exception:
        return default


def im_in_entry_window(local_time: datetime, im_cfg: dict[str, Any]) -> bool:
    """True inside the intraday-momentum entry window [entry_time, entry_time+window_minutes)."""
    if not im_cfg.get("enabled"):
        return False
    start = _hhmm_to_minutes(im_cfg.get("entry_time", "14:30"), 14 * 60 + 30)
    width = int(as_float(im_cfg.get("entry_window_minutes", 10), 10))
    now_m = local_time.hour * 60 + local_time.minute
    return start <= now_m < start + width


def im_force_exit_due(local_time: datetime, im_cfg: dict[str, Any]) -> bool:
    """True at/after the intraday-momentum exit time (IM is an intraday-only bet)."""
    exit_m = _hhmm_to_minutes(im_cfg.get("exit_time", "14:55"), 14 * 60 + 55)
    now_m = local_time.hour * 60 + local_time.minute
    return now_m >= exit_m


def enrich_snapshot_features(
    quote: dict[str, Any],
    hist: list[dict[str, Any]],
    strategy_cfg: dict[str, Any],
) -> dict[str, Any]:
    indicators_cfg = strategy_cfg.get("indicators", {}) if isinstance(strategy_cfg.get("indicators", {}), dict) else {}
    consolidation_cfg = strategy_cfg.get("consolidation", {}) if isinstance(strategy_cfg.get("consolidation", {}), dict) else {}
    quote["consolidation_box"] = compute_consolidation_box(hist, consolidation_cfg or {})
    quote["first_half_hour_return"] = compute_first_half_hour_return(
        hist,
        quote,
        strategy_cfg.get("intraday_momentum", {}) if isinstance(strategy_cfg.get("intraday_momentum", {}), dict) else {},
    )
    quote["alpha101_conviction"] = compute_alpha101_conviction(hist, quote)
    quote["volume_surge"] = compute_volume_surge(hist, quote)
    vwap_result = compute_rolling_vwap(hist, quote, indicators_cfg.get("rolling_vwap", {}))
    atr_result = compute_atr_proxy(hist, quote, indicators_cfg.get("intraday_atr", {}))
    bollinger_result = compute_bollinger_squeeze(hist, quote, indicators_cfg.get("bollinger_squeeze", {}))
    quote["rolling_vwap"] = vwap_result.get("vwap")
    quote["rolling_vwap_available"] = bool(vwap_result.get("available"))
    quote["rolling_vwap_status"] = vwap_result.get("status")
    quote["price_above_vwap"] = vwap_result.get("price_above_vwap")
    quote["vwap_distance_pct"] = vwap_result.get("distance_pct")
    quote["vwap_diagnostic"] = vwap_result
    quote["atr_pct"] = atr_result.get("atr_pct")
    quote["atr_stop_distance_pct"] = atr_result.get("stop_distance_pct")
    quote["atr_available"] = bool(atr_result.get("available"))
    quote["atr_status"] = atr_result.get("status")
    quote["atr_diagnostic"] = atr_result
    quote["bollinger_squeeze"] = bollinger_result
    quote["bollinger_squeeze_available"] = bool(bollinger_result.get("available"))
    quote["bollinger_squeeze_active"] = bool(bollinger_result.get("squeeze"))
    quote["bollinger_squeeze_breakout"] = bool(bollinger_result.get("breakout"))
    quote["_snapshot_features_enriched"] = True
    return quote


def compute_snapshot_momentum(
    quotes: list[dict[str, Any]],
    history: list[dict[str, Any]],
    lookback_minutes: int,
    strategy_cfg: dict[str, Any] | None = None,
    history_by_code: dict[str, list[dict[str, Any]]] | None = None,
    lazy_indicators: bool = False,
) -> list[dict[str, Any]]:
    strategy_cfg = strategy_cfg or {}
    if "window_snapshots" in strategy_cfg and "consolidation" not in strategy_cfg:
        # Backward compatible path for old callers that passed only the consolidation config.
        strategy_cfg = {"consolidation": strategy_cfg}
    if history_by_code is None:
        by_code: dict[str, list[dict[str, Any]]] = {}
        for row in history:
            code = str(row.get("stockCode", "")).zfill(6)
            if code:
                by_code.setdefault(code, []).append(row)
    else:
        by_code = history_by_code
    out = []
    for q in quotes:
        code = str(q.get("stockCode", "")).zfill(6)
        hist = by_code.get(code, [])
        anchor = hist[-lookback_minutes] if len(hist) >= lookback_minutes else None
        current = as_float(q.get("currentPrice"))
        anchor_price = as_float(anchor.get("currentPrice")) if anchor else 0.0
        momentum = current / anchor_price - 1.0 if current > 0 and anchor_price > 0 else None
        midpoint = as_float(q.get("midpoint"), current)
        midpoint_anchor = hist[-3] if len(hist) >= 3 else None
        midpoint_anchor_value = as_float(midpoint_anchor.get("midpoint")) if midpoint_anchor else 0.0
        if midpoint_anchor and midpoint_anchor_value <= 0:
            anchor_bid = as_float(midpoint_anchor.get("bidPrice1"))
            anchor_ask = as_float(midpoint_anchor.get("askPrice1"))
            anchor_current = as_float(midpoint_anchor.get("currentPrice"))
            midpoint_anchor_value = (anchor_bid + anchor_ask) / 2.0 if anchor_bid > 0 and anchor_ask > 0 else anchor_current
        bid_pressure_3m = midpoint - midpoint_anchor_value if midpoint_anchor_value > 0 else None
        bid_pressure_3m_pct = bid_pressure_3m / midpoint_anchor_value if bid_pressure_3m is not None and midpoint_anchor_value > 0 else None
        acceleration = None
        if len(hist) >= 10:
            price_t_minus_5 = as_float(hist[-5].get("currentPrice"))
            price_t_minus_10 = as_float(hist[-10].get("currentPrice"))
            mom_5m_t = current / price_t_minus_5 - 1.0 if current > 0 and price_t_minus_5 > 0 else None
            mom_5m_prev = price_t_minus_5 / price_t_minus_10 - 1.0 if price_t_minus_5 > 0 and price_t_minus_10 > 0 else None
            if mom_5m_t is not None and mom_5m_prev is not None:
                acceleration = mom_5m_t - mom_5m_prev
        q2 = dict(q)
        q2["volume"] = as_float(q.get("volume"), 0.0)
        q2["amount"] = as_float(q.get("amount"), 0.0)
        q2["turnover_pct"] = as_float(q.get("turnover_pct"), 0.0)
        q2["lookback_minutes"] = lookback_minutes
        q2["momentum"] = momentum
        q2["momentum_available"] = momentum is not None
        q2["midpoint"] = midpoint
        q2["bid_pressure_3m"] = bid_pressure_3m
        q2["bid_pressure_3m_pct"] = bid_pressure_3m_pct
        q2["acceleration"] = acceleration
        if lazy_indicators:
            q2["_snapshot_features_enriched"] = False
        else:
            enrich_snapshot_features(q2, hist, strategy_cfg)
        if q2["momentum_available"]:
            q2["signal_type"] = f"snapshot_momentum_{lookback_minutes}m"
        else:
            q2["momentum"] = as_float(q2.get("change_pct"), 0.0)
            q2["momentum_available"] = True
            q2["signal_type"] = "change_pct_fallback"
        out.append(q2)
    return out


def daily_state_bucket(state: dict[str, Any], key: str, trade_date: str) -> int:
    bucket = state.get(key, {})
    if not isinstance(bucket, dict):
        return 0
    return int(bucket.get(trade_date, 0) or 0)


def daily_state_float_bucket(state: dict[str, Any], key: str, trade_date: str) -> float:
    bucket = state.get(key, {})
    if not isinstance(bucket, dict):
        return 0.0
    return as_float(bucket.get(trade_date), 0.0)


def increment_daily_state(state: dict[str, Any], key: str, trade_date: str) -> None:
    bucket = state.get(key, {})
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[trade_date] = int(bucket.get(trade_date, 0) or 0) + 1
    state[key] = bucket


def exchange_local_time(session: dict[str, Any]) -> datetime:
    parsed = parse_iso_dt(session.get("local_time"))
    if parsed is None:
        return datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    return parsed.astimezone(ZoneInfo("Asia/Shanghai"))


def system_local_time() -> datetime:
    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


def t0_inventory_day(state: dict[str, Any], trade_date: str) -> dict[str, Any]:
    root = state.get("t0_inventory_by_date")
    if not isinstance(root, dict):
        root = {}
    day = root.get(trade_date)
    if not isinstance(day, dict):
        day = {}
    root[trade_date] = day
    state["t0_inventory_by_date"] = root
    return day


def t0_inventory_for_code(state: dict[str, Any], trade_date: str, code: str) -> dict[str, Any]:
    day = t0_inventory_day(state, trade_date)
    zcode = str(code).zfill(6)
    node = day.get(zcode)
    if not isinstance(node, dict):
        node = {}
    day[zcode] = node
    return node


def t0_inventory_remaining_qty(state: dict[str, Any], trade_date: str, code: str) -> int:
    day = state.get("t0_inventory_by_date", {}).get(trade_date, {})
    node = day.get(str(code).zfill(6), {}) if isinstance(day, dict) else {}
    if not isinstance(node, dict):
        return 0
    if node.get("fill_reconciliation_ok") is True:
        buy_qty = int(as_float(node.get("buy_quantity_filled"), 0.0))
        sell_qty = int(as_float(node.get("sell_quantity_filled"), 0.0))
    else:
        buy_qty = int(as_float(node.get("buy_quantity_submitted"), 0.0))
        sell_qty = int(as_float(node.get("sell_quantity_submitted"), 0.0))
    return max(0, buy_qty - sell_qty)


def t0_sellable_quantity(state: dict[str, Any], trade_date: str, code: str, held_pos: dict[str, Any], lot: int) -> int:
    remaining = t0_inventory_remaining_qty(state, trade_date, code)
    if remaining <= 0:
        return 0
    node = state.get("t0_inventory_by_date", {}).get(trade_date, {}).get(str(code).zfill(6), {})
    baseline_available = as_float(node.get("baseline_available_quantity"), 0.0) if isinstance(node, dict) else 0.0
    current_available = as_float(held_pos.get("availableQuantity"), 0.0)
    broker_excess = max(0, int(current_available - baseline_available))
    return round_lot(min(remaining, broker_excess), lot)


def record_t0_buy_submission(state: dict[str, Any], trade_date: str, order: dict[str, Any], submit: dict[str, Any]) -> None:
    if not submit.get("ok"):
        return
    code = str(order.get("stockCode", "")).zfill(6)
    node = t0_inventory_for_code(state, trade_date, code)
    data = extract_data(submit) or {}
    order_id = data.get("orderId")
    node["buy_quantity_submitted"] = int(as_float(node.get("buy_quantity_submitted"), 0.0)) + int(as_float(order.get("quantity"), 0.0))
    node.setdefault("sell_quantity_submitted", int(as_float(node.get("sell_quantity_submitted"), 0.0)))
    node.setdefault("baseline_available_quantity", as_float(order.get("baseline_available_quantity"), 0.0))
    order_ids = node.get("buy_order_ids")
    if not isinstance(order_ids, list):
        order_ids = []
    if order_id and str(order_id) not in {str(x) for x in order_ids}:
        order_ids.append(str(order_id))
    node["buy_order_ids"] = order_ids
    node["last_buy_price"] = as_float(order.get("price"), 0.0)
    node["last_buy_at"] = now_iso()
    node.setdefault("first_buy_at", node["last_buy_at"])
    node.setdefault("entry_price", node["last_buy_price"])
    node["highest_price_since_entry"] = max(
        as_float(node.get("highest_price_since_entry"), 0.0),
        node["last_buy_price"],
    )
    node["last_buy_order_id"] = order_id
    node["inventory_scope"] = "t0_submitted_buy_only"


def record_t0_sell_submission(state: dict[str, Any], trade_date: str, order: dict[str, Any], submit: dict[str, Any]) -> None:
    if not submit.get("ok"):
        return
    code = str(order.get("stockCode", "")).zfill(6)
    node = t0_inventory_for_code(state, trade_date, code)
    data = extract_data(submit) or {}
    order_id = data.get("orderId")
    node["sell_quantity_submitted"] = int(as_float(node.get("sell_quantity_submitted"), 0.0)) + int(as_float(order.get("quantity"), 0.0))
    order_ids = node.get("sell_order_ids")
    if not isinstance(order_ids, list):
        order_ids = []
    if order_id and str(order_id) not in {str(x) for x in order_ids}:
        order_ids.append(str(order_id))
    node["sell_order_ids"] = order_ids
    node["last_sell_price"] = as_float(order.get("price"), 0.0)
    node["last_sell_at"] = now_iso()
    node["last_sell_order_id"] = order_id


def trade_rows(resp: dict[str, Any]) -> list[dict[str, Any]]:
    data = extract_data(resp)
    trades = data.get("trades", [])
    return trades if isinstance(trades, list) else []


def reconcile_t0_inventory_from_trade_history(state: dict[str, Any], trade_date: str, trades_resp: dict[str, Any]) -> dict[str, Any]:
    day = state.get("t0_inventory_by_date", {}).get(trade_date, {})
    if not isinstance(day, dict) or not day:
        return {"attempted": False, "reason": "no_t0_inventory_for_trade_date"}
    if not trades_resp.get("ok"):
        result = {"attempted": True, "ok": False, "error": trades_resp.get("error")}
        state["last_fill_reconciliation"] = {**result, "trade_date": trade_date, "timestamp": now_iso()}
        return result

    trades = trade_rows(trades_resp)
    reconciled_codes: list[str] = []
    for code, node in day.items():
        if not isinstance(node, dict):
            continue
        buy_ids = {str(x) for x in node.get("buy_order_ids", []) if x}
        sell_ids = {str(x) for x in node.get("sell_order_ids", []) if x}
        if node.get("last_buy_order_id"):
            buy_ids.add(str(node.get("last_buy_order_id")))
        if node.get("last_sell_order_id"):
            sell_ids.add(str(node.get("last_sell_order_id")))
        if not buy_ids and not sell_ids:
            continue

        buy_qty = 0
        sell_qty = 0
        buy_amount = 0.0
        sell_amount = 0.0
        matched_order_ids: list[str] = []
        for tr in trades:
            order_id = str(tr.get("orderId") or tr.get("tradeId") or "")
            if not order_id:
                continue
            direction = str(tr.get("direction") or "").lower()
            qty = int(as_float(tr.get("filledQuantity", tr.get("quantity")), 0.0))
            px = as_float(tr.get("filledPrice", tr.get("price")), 0.0)
            amount = as_float(tr.get("filledAmount", tr.get("amount")), px * qty)
            if order_id in buy_ids and direction == "buy":
                buy_qty += qty
                buy_amount += amount
                matched_order_ids.append(order_id)
            elif order_id in sell_ids and direction == "sell":
                sell_qty += qty
                sell_amount += amount
                matched_order_ids.append(order_id)
        node["buy_quantity_filled"] = buy_qty
        node["sell_quantity_filled"] = sell_qty
        node["filled_buy_vwap"] = buy_amount / buy_qty if buy_qty > 0 else None
        node["filled_sell_vwap"] = sell_amount / sell_qty if sell_qty > 0 else None
        node["filled_remaining_qty"] = max(0, buy_qty - sell_qty)
        node["matched_trade_order_ids"] = sorted(set(matched_order_ids))
        node["fill_reconciliation_ok"] = True
        node["fill_reconciled_at"] = now_iso()
        # Book realized PnL ONLY on the newly-confirmed filled sell quantity (true
        # fills, not submissions). Idempotent via realized_booked_sell_qty.
        booked = int(as_float(node.get("realized_booked_sell_qty"), 0.0))
        newly = sell_qty - booked
        sell_vwap = node.get("filled_sell_vwap")
        cost_basis = (
            as_float(node.get("realization_cost_price"), 0.0)
            or as_float(node.get("filled_buy_vwap"), 0.0)
            or as_float(node.get("entry_price"), 0.0)
        )
        if newly > 0 and sell_vwap and cost_basis > 0:
            realized = (as_float(sell_vwap) - cost_basis) * newly
            add_daily_state_float(state, "today_realized_pnl", trade_date, realized)
            if realized < 0:
                record_negative_action_sample(state, trade_date, {"stockCode": code, "direction": "sell", "reason": "fill_confirmed_loss"}, realized)
            rtp = state.get("realized_trade_pnls")
            if not isinstance(rtp, list):
                rtp = []
            rtp.append(round(realized, 2))
            state["realized_trade_pnls"] = rtp[-100:]
            node["realized_booked_sell_qty"] = sell_qty
        reconciled_codes.append(str(code).zfill(6))

    result = {
        "attempted": True,
        "ok": True,
        "reconciled_codes": reconciled_codes,
        "trade_count": len(trades),
    }
    state["last_fill_reconciliation"] = {**result, "trade_date": trade_date, "timestamp": now_iso()}
    return result


def maybe_reconcile_t0_fills(cfg: dict[str, Any], client: SkillClient, state: dict[str, Any], trade_date: str, force: bool = False) -> dict[str, Any]:
    risk = cfg.get("risk", {})
    if not bool(risk.get("reconcile_fills_from_trade_history", True)):
        return {"attempted": False, "reason": "fill_reconciliation_disabled"}
    day = state.get("t0_inventory_by_date", {}).get(trade_date, {})
    if not isinstance(day, dict) or not day:
        return {"attempted": False, "reason": "no_t0_inventory_for_trade_date"}
    min_interval = int(risk.get("fill_reconciliation_min_interval_minutes", 3))
    last = state.get("last_fill_reconciliation", {})
    since_last = minutes_since(last.get("timestamp")) if isinstance(last, dict) and last.get("trade_date") == trade_date else None
    if not force and since_last is not None and since_last < min_interval:
        return {"attempted": False, "reason": "fill_reconciliation_interval_active", "minutes_since_last": since_last}
    resp = call_trade_history(client, trade_date, trade_date)
    return reconcile_t0_inventory_from_trade_history(state, trade_date, resp)


def add_daily_state_float(state: dict[str, Any], key: str, trade_date: str, value: float) -> None:
    bucket = state.get(key, {})
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[trade_date] = as_float(bucket.get(trade_date), 0.0) + float(value)
    state[key] = bucket


def parse_iso_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).astimezone()
    except Exception:
        return None


def _cfg_path(value: Any, default: str) -> Path:
    raw = str(value or default)
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def is_money_like_etf_name(name: Any) -> bool:
    return bool(MONEY_LIKE_ETF_NAME_RE.search(str(name or "")))


def _read_full_market_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def evaluate_full_market_entry_guard(strategy: dict[str, Any], trade_date: str, local_time: datetime) -> dict[str, Any]:
    """Point-in-time full-market ETF breadth guard for new BUYs only.

    Uses the latest successful local Eastmoney full-market ETF snapshot at or
    before the decision time. It is a defensive entry guard, not an alpha signal.
    Unavailable data only blocks when explicitly configured to fail closed.
    """
    cfg = strategy.get("full_market_entry_guard", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return {"enabled": False, "status": "disabled", "block_new_buy": False}
    if _REPLAY_NOW is not None and not cfg.get("enabled_in_replay", False):
        return {"enabled": False, "status": "disabled_in_replay", "block_new_buy": False}

    runs_path = _cfg_path(cfg.get("snapshot_runs_path"), "outputs/eastmoney_full_market/snapshot_runs.jsonl")
    block_on_unavailable = bool(cfg.get("block_on_unavailable", False))
    if not runs_path.exists():
        return {
            "enabled": True, "status": "snapshot_runs_missing", "block_new_buy": block_on_unavailable,
            "snapshot_runs_path": str(runs_path),
        }

    max_age_minutes = as_float(cfg.get("max_snapshot_age_minutes"), 150.0)
    min_non_money_etfs = int(as_float(cfg.get("min_non_money_etfs"), 300))
    min_active_amount = as_float(cfg.get("min_active_amount"), 1_000_000.0)
    min_active_etfs = int(as_float(cfg.get("min_active_etfs"), 200))
    up_frac_max = as_float(cfg.get("risk_off_up_frac_max"), 0.35)
    median_max = as_float(cfg.get("risk_off_median_change_pct_max"), -0.35)
    lt_minus1_min = as_float(cfg.get("risk_off_down_gt_1pct_frac_min"), 0.55)
    block_on_risk_off = bool(cfg.get("block_on_risk_off", True))

    latest: tuple[datetime, Path, int] | None = None
    try:
        for line in runs_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            run = json.loads(line)
            session = run.get("session") if isinstance(run.get("session"), dict) else {}
            if (
                session.get("trade_date") != trade_date
                or run.get("scope") != "etf"
                or run.get("status") != "ok"
                or int(as_float(run.get("row_count"), 0)) <= 0
            ):
                continue
            snap_dt = parse_iso_dt(session.get("exchange_local_time"))
            out_path = Path(str(run.get("output_file") or ""))
            if snap_dt is None or not out_path.exists():
                continue
            snap_dt_sh = snap_dt.astimezone(ZoneInfo("Asia/Shanghai"))
            # Never use a future snapshot in replay or delayed runs.
            if snap_dt_sh > local_time:
                continue
            latest = (snap_dt_sh, out_path, int(as_float(run.get("row_count"), 0)))
    except Exception as exc:
        return {
            "enabled": True, "status": "snapshot_runs_parse_error",
            "block_new_buy": block_on_unavailable, "error": str(exc), "snapshot_runs_path": str(runs_path),
        }

    if latest is None:
        return {
            "enabled": True, "status": "no_point_in_time_snapshot",
            "block_new_buy": block_on_unavailable, "trade_date": trade_date, "snapshot_runs_path": str(runs_path),
        }

    snap_dt, out_path, raw_rows = latest
    age_minutes = max(0.0, (local_time - snap_dt).total_seconds() / 60.0)
    if max_age_minutes > 0 and age_minutes > max_age_minutes:
        return {
            "enabled": True, "status": "snapshot_stale", "block_new_buy": block_on_unavailable,
            "snapshot_time": snap_dt.isoformat(), "age_minutes": round(age_minutes, 2),
            "max_snapshot_age_minutes": max_age_minutes, "snapshot_file": str(out_path),
        }

    try:
        rows = _read_full_market_csv_rows(out_path)
    except Exception as exc:
        return {
            "enabled": True, "status": "snapshot_read_error", "block_new_buy": block_on_unavailable,
            "error": str(exc), "snapshot_file": str(out_path),
        }

    all_changes: list[float] = []
    active_changes: list[float] = []
    for row in rows:
        name = row.get("name")
        if is_money_like_etf_name(name):
            continue
        price = as_float(row.get("currentPrice"), 0.0)
        chg = as_float(row.get("change_pct"), math.nan)
        if not math.isfinite(chg) or price <= 0:
            continue
        all_changes.append(chg)
        if as_float(row.get("amount"), 0.0) >= min_active_amount:
            active_changes.append(chg)

    changes = active_changes if len(active_changes) >= min_active_etfs else all_changes
    if len(all_changes) < min_non_money_etfs or not changes:
        return {
            "enabled": True, "status": "insufficient_full_market_rows",
            "block_new_buy": block_on_unavailable, "raw_rows": raw_rows,
            "non_money_count": len(all_changes), "active_count": len(active_changes),
            "min_non_money_etfs": min_non_money_etfs, "min_active_etfs": min_active_etfs,
            "snapshot_file": str(out_path), "snapshot_time": snap_dt.isoformat(),
        }

    ordered = sorted(changes)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    up_frac = sum(1 for x in ordered if x > 0) / n
    down_frac = sum(1 for x in ordered if x < 0) / n
    avg = sum(ordered) / n
    lt_minus1_frac = sum(1 for x in ordered if x <= -1.0) / n
    risk_off = bool((up_frac <= up_frac_max and median <= median_max) or lt_minus1_frac >= lt_minus1_min)
    mode = "risk_off" if risk_off else ("selective_or_neutral" if up_frac < 0.55 else "risk_on")
    return {
        "enabled": True,
        "status": "available",
        "block_new_buy": bool(block_on_risk_off and risk_off),
        "mode": mode,
        "snapshot_time": snap_dt.isoformat(),
        "age_minutes": round(age_minutes, 2),
        "snapshot_file": str(out_path),
        "raw_rows": raw_rows,
        "non_money_count": len(all_changes),
        "active_count": len(active_changes),
        "sample_used": "active_amount" if len(active_changes) >= min_active_etfs else "all_non_money",
        "sample_count": n,
        "up_frac": round(up_frac, 4),
        "down_frac": round(down_frac, 4),
        "avg_change_pct": round(avg, 4),
        "median_change_pct": round(median, 4),
        "down_gt_1pct_frac": round(lt_minus1_frac, 4),
        "rule": {
            "risk_off_up_frac_max": up_frac_max,
            "risk_off_median_change_pct_max": median_max,
            "risk_off_down_gt_1pct_frac_min": lt_minus1_min,
            "block_on_risk_off": block_on_risk_off,
            "block_on_unavailable": block_on_unavailable,
        },
    }


def t0_entry_eligibility(q: dict[str, Any], strategy: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Conservative whitelist for ETFs that may be opened by the T0 agent.

    If the broker cannot make same-day sellability reliable for a class (e.g.
    dynamic domestic industry ETFs), the T0 agent must not open it as an intraday
    position. This does not affect selling existing holdings.
    """
    cfg = strategy.get("t0_entry_eligibility", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return True, {"enabled": False, "status": "disabled"}
    code = str(q.get("stockCode", "")).zfill(6)
    asset_class = str(q.get("asset_class") or "unknown")
    allow_codes = {str(x).zfill(6) for x in (cfg.get("allow_stock_codes") or []) if str(x).strip()}
    block_codes = {str(x).zfill(6) for x in (cfg.get("block_stock_codes") or []) if str(x).strip()}
    allow_classes = {str(x) for x in (cfg.get("allow_asset_classes") or [])}
    block_classes = {str(x) for x in (cfg.get("block_asset_classes") or [])}
    block_unknown = bool(cfg.get("block_unknown_asset_class", True))
    if code in block_codes:
        return False, {"enabled": True, "status": "blocked_code", "stockCode": code, "asset_class": asset_class}
    if code in allow_codes:
        return True, {"enabled": True, "status": "allowlisted_code", "stockCode": code, "asset_class": asset_class}
    if asset_class in allow_classes:
        return True, {"enabled": True, "status": "allowlisted_asset_class", "stockCode": code, "asset_class": asset_class}
    if asset_class in block_classes or (block_unknown and asset_class == "unknown"):
        return False, {
            "enabled": True, "status": "blocked_asset_class", "stockCode": code, "asset_class": asset_class,
            "reason": cfg.get("reason", "asset_class_not_confirmed_same_day_sellable_for_t0_entry"),
        }
    return True, {"enabled": True, "status": "not_blocked", "stockCode": code, "asset_class": asset_class}


def quote_for_code(quotes: list[dict[str, Any]], code: str | None) -> dict[str, Any] | None:
    if not code:
        return None
    zcode = str(code).zfill(6)
    return next((x for x in quotes if str(x.get("stockCode", "")).zfill(6) == zcode), None)


def minutes_since(value: Any) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value)).astimezone()
        return (current_dt() - dt).total_seconds() / 60.0
    except Exception:
        return None


def t0_inventory_node_read(state: dict[str, Any], trade_date: str, code: str) -> dict[str, Any]:
    day = state.get("t0_inventory_by_date", {}).get(trade_date, {})
    node = day.get(str(code).zfill(6), {}) if isinstance(day, dict) else {}
    return node if isinstance(node, dict) else {}


def t0_entry_price(node: dict[str, Any], held_pos: dict[str, Any]) -> float:
    return (
        as_float(node.get("filled_buy_vwap"), 0.0) or
        as_float(node.get("entry_price"), 0.0) or
        as_float(node.get("last_buy_price"), 0.0) or
        as_float(held_pos.get("costPrice"), 0.0)
    )


def safe_sell_reference_price(q: dict[str, Any] | None, current: float, node: dict[str, Any], held_pos: dict[str, Any]) -> float:
    """Return a non-zero sell reference price from quote, last mark, then cost anchors."""
    return (
        (as_float(q.get("bidPrice1"), 0.0) if q else 0.0) or
        current or
        as_float(node.get("last_mark_price"), 0.0) or
        t0_entry_price(node, held_pos)
    )


def update_t0_mark_state(state: dict[str, Any], trade_date: str, code: str, current_price: float) -> dict[str, Any]:
    node = t0_inventory_for_code(state, trade_date, code)
    if current_price > 0:
        node["last_mark_price"] = current_price
        node["last_mark_at"] = now_iso()
        node["highest_price_since_entry"] = max(as_float(node.get("highest_price_since_entry"), 0.0), current_price)
    return node


def score_loss_exit(
    *,
    strategy: dict[str, Any],
    filters: dict[str, Any],
    local_time: datetime,
    state: dict[str, Any],
    trade_date: str,
    q: dict[str, Any] | None,
    code: str,
    pnl_pct: float,
    current: float,
    holding_minutes: float | None,
    broad_market_not_declining: bool,
) -> dict[str, Any]:
    weights = strategy.get("exit_score_weights", {})
    stop_loss = abs(as_float(strategy.get("stop_loss_pct"), -0.012)) or 0.012
    mom = as_float(q.get("momentum"), 0.0) if q else 0.0
    acceleration_raw = q.get("acceleration") if q else None
    acceleration = as_float(acceleration_raw) if acceleration_raw is not None else None
    bid_pressure_raw = q.get("bid_pressure_3m_pct") if q else None
    bid_pressure = as_float(bid_pressure_raw) if bid_pressure_raw is not None else None
    spread = q.get("spread_pct") if q else None
    rolling_vwap_available = bool(q.get("rolling_vwap_available")) if q else False
    price_above_vwap = q.get("price_above_vwap") if q else None
    max_spread = as_float(filters.get("max_spread_pct"), 0.0015)
    decel_threshold = as_float(strategy.get("deceleration_exit_threshold"), -0.002)
    loss_review_after = as_float(strategy.get("loss_review_after_minutes"), 15)

    orb = get_orb(state, trade_date, code)
    structure_score = 0.0
    structure_reason = "none"
    if orb and current > 0:
        if current < as_float(orb.get("low")):
            structure_score = as_float(weights.get("structure_break", 25), 25)
            structure_reason = "below_orb_low"
        elif current < as_float(orb.get("midpoint")):
            structure_score = as_float(weights.get("structure_break", 25), 25) * 0.5
            structure_reason = "below_orb_midpoint"

    loss_depth_score = 0.0
    if pnl_pct < 0:
        loss_depth_score = min(abs(pnl_pct) / stop_loss, 1.0) * as_float(weights.get("loss_depth", 15), 15)

    momentum_score = 0.0
    momentum_reason = "none"
    if mom <= as_float(strategy.get("exit_momentum_pct"), -0.001) and acceleration is not None and acceleration <= decel_threshold:
        momentum_score = as_float(weights.get("momentum_reversal", 20), 20)
        momentum_reason = "momentum_and_acceleration_negative"
    elif mom < 0:
        momentum_score = as_float(weights.get("momentum_reversal", 20), 20) * 0.5
        momentum_reason = "momentum_negative"

    bid_score = as_float(weights.get("bid_pressure_negative", 15), 15) if bid_pressure is not None and bid_pressure <= 0 else 0.0
    breadth_score = as_float(weights.get("market_breadth_negative", 10), 10) if not broad_market_not_declining else 0.0
    time_score = as_float(weights.get("time_confirmation", 10), 10) if holding_minutes is not None and holding_minutes >= loss_review_after else 0.0
    liquidity_score = 0.0
    if spread is None or as_float(spread, 999.0) > max_spread:
        liquidity_score = as_float(weights.get("liquidity_deterioration", 5), 5)

    components = {
        "loss_depth": loss_depth_score,
        "structure_break": structure_score,
        "momentum_reversal": momentum_score,
        "bid_pressure_negative": bid_score,
        "market_breadth_negative": breadth_score,
        "time_confirmation": time_score,
        "liquidity_deterioration": liquidity_score,
    }
    total = sum(as_float(x, 0.0) for x in components.values())
    return {
        "score": total,
        "threshold": as_float(strategy.get("loss_exit_score_threshold"), 70),
        "components": components,
        "details": {
            "pnl_pct": pnl_pct,
            "momentum": mom,
            "acceleration": acceleration,
            "bid_pressure_3m_pct": bid_pressure,
            "holding_minutes": holding_minutes,
            "structure_reason": structure_reason,
            "momentum_reason": momentum_reason,
            "local_time": str(local_time),
        },
    }


def score_profit_exit(
    *,
    strategy: dict[str, Any],
    local_time: datetime,
    q: dict[str, Any] | None,
    pnl_pct: float,
    highest_price: float,
    current: float,
    holding_minutes: float | None,
) -> dict[str, Any]:
    weights = strategy.get("exit_score_weights", {})
    mom = as_float(q.get("momentum"), 0.0) if q else 0.0
    acceleration_raw = q.get("acceleration") if q else None
    acceleration = as_float(acceleration_raw) if acceleration_raw is not None else None
    bid_pressure_raw = q.get("bid_pressure_3m_pct") if q else None
    bid_pressure = as_float(bid_pressure_raw) if bid_pressure_raw is not None else None
    min_profit = as_float(strategy.get("min_profit_exit_pct"), 0.003)
    trailing_drawdown = as_float(strategy.get("profit_trailing_drawdown_pct"), -0.005)
    decel_threshold = as_float(strategy.get("deceleration_exit_threshold"), -0.002)
    drawdown_from_high = current / highest_price - 1.0 if highest_price > 0 and current > 0 else 0.0
    near_close = (local_time.hour == 14 and local_time.minute >= 0) or local_time.hour >= 15

    components = {
        "profit_floor": as_float(weights.get("profit_floor", 20), 20) if pnl_pct >= min_profit else 0.0,
        "drawdown_from_high": as_float(weights.get("drawdown_from_high", 25), 25) if drawdown_from_high <= trailing_drawdown else 0.0,
        "profit_momentum_negative": as_float(weights.get("profit_momentum_negative", 15), 15) if mom <= as_float(strategy.get("exit_momentum_pct"), -0.001) else 0.0,
        "profit_acceleration_negative": as_float(weights.get("profit_acceleration_negative", 15), 15) if acceleration is not None and acceleration <= decel_threshold else 0.0,
        "profit_bid_pressure_negative": as_float(weights.get("profit_bid_pressure_negative", 15), 15) if bid_pressure is not None and bid_pressure <= 0 else 0.0,
        "near_close": as_float(weights.get("near_close", 10), 10) if near_close else 0.0,
    }
    total = sum(as_float(x, 0.0) for x in components.values())
    return {
        "score": total,
        "threshold": as_float(strategy.get("profit_exit_score_threshold"), 65),
        "components": components,
        "details": {
            "pnl_pct": pnl_pct,
            "drawdown_from_high": drawdown_from_high,
            "momentum": mom,
            "acceleration": acceleration,
            "bid_pressure_3m_pct": bid_pressure,
            "holding_minutes": holding_minutes,
            "local_time": str(local_time),
        },
    }


def score_unified_sell(
    *,
    strategy: dict[str, Any],
    filters: dict[str, Any],
    local_time: datetime,
    state: dict[str, Any],
    trade_date: str,
    q: dict[str, Any] | None,
    code: str,
    pnl_pct: float,
    current: float,
    highest_price: float,
) -> dict[str, Any]:
    """统一 sell_score 出场引擎 (unified_sell_score): 适用于当日T0仓位与隔夜持仓。

    near_close 只是加权项, 不能单独触发卖出; 阈值按盈亏方向取
    loss_exit_score_threshold / profit_exit_score_threshold。
    """
    weights = strategy.get("exit_score_weights", {})
    stop_loss = abs(as_float(strategy.get("stop_loss_pct"), -0.012)) or 0.012
    mom = as_float(q.get("momentum"), 0.0) if q else 0.0
    acceleration_raw = q.get("acceleration") if q else None
    acceleration = as_float(acceleration_raw) if acceleration_raw is not None else None
    bid_pressure_raw = q.get("bid_pressure_3m_pct") if q else None
    bid_pressure = as_float(bid_pressure_raw) if bid_pressure_raw is not None else None
    spread = q.get("spread_pct") if q else None
    rolling_vwap_available = bool(q.get("rolling_vwap_available")) if q else False
    price_above_vwap = q.get("price_above_vwap") if q else None
    max_spread = as_float(filters.get("max_spread_pct"), 0.0015)
    decel_threshold = as_float(strategy.get("deceleration_exit_threshold"), -0.002)
    trailing_drawdown = as_float(strategy.get("profit_trailing_drawdown_pct"), -0.005)

    loss_depth_score = 0.0
    if pnl_pct < 0:
        loss_depth_score = min(abs(pnl_pct) / stop_loss, 1.0) * as_float(weights.get("loss_depth", 15), 15)

    orb = get_orb(state, trade_date, code)
    structure_score = 0.0
    structure_reason = "none"
    if orb and current > 0:
        if current < as_float(orb.get("low")):
            structure_score = as_float(weights.get("structure_break", 25), 25)
            structure_reason = "below_orb_low"
        elif current < as_float(orb.get("midpoint")):
            structure_score = as_float(weights.get("structure_break", 25), 25) * 0.5
            structure_reason = "below_orb_midpoint"

    momentum_score = 0.0
    momentum_reason = "none"
    if mom <= as_float(strategy.get("exit_momentum_pct"), -0.001):
        momentum_score = as_float(weights.get("momentum_reversal", 20), 20)
        momentum_reason = "momentum_below_exit_threshold"
    elif mom < 0:
        momentum_score = as_float(weights.get("momentum_reversal", 20), 20) * 0.5
        momentum_reason = "momentum_negative"

    bid_score = as_float(weights.get("bid_pressure_negative", 15), 15) if bid_pressure is not None and bid_pressure <= 0 else 0.0
    accel_score = as_float(weights.get("acceleration_negative", 15), 15) if acceleration is not None and acceleration <= decel_threshold else 0.0
    liquidity_score = 0.0
    if spread is None or as_float(spread, 999.0) > max_spread:
        liquidity_score = as_float(weights.get("liquidity_deterioration", 15), 15)
    vwap_score = 0.0
    if rolling_vwap_available and price_above_vwap is False:
        vwap_score = as_float(weights.get("vwap_breakdown", 15), 15)

    near_close_score = 0.0
    if (local_time.hour == 14 and local_time.minute >= 50) or local_time.hour >= 15:
        near_close_score = as_float(weights.get("near_close_1450", 35), 35)
    elif local_time.hour == 14 and local_time.minute >= 20:
        near_close_score = as_float(weights.get("near_close_1420", 20), 20)

    profit_drawdown_score = 0.0
    drawdown_from_high = current / highest_price - 1.0 if highest_price > 0 and current > 0 else 0.0
    if pnl_pct > 0 and drawdown_from_high < 0 and trailing_drawdown < 0:
        profit_drawdown_score = min(abs(drawdown_from_high) / abs(trailing_drawdown), 1.0) * as_float(weights.get("profit_drawdown", 25), 25)

    components = {
        "loss_depth": loss_depth_score,
        "structure_break": structure_score,
        "momentum_reversal": momentum_score,
        "bid_pressure_negative": bid_score,
        "acceleration_negative": accel_score,
        "liquidity_deterioration": liquidity_score,
        "vwap_breakdown": vwap_score,
        "near_close": near_close_score,
        "profit_drawdown": profit_drawdown_score,
    }
    total = sum(as_float(x, 0.0) for x in components.values())
    threshold = as_float(strategy.get("loss_exit_score_threshold"), 70) if pnl_pct < 0 else as_float(strategy.get("profit_exit_score_threshold"), 65)
    return {
        "score": round(total, 2),
        "threshold": threshold,
        "components": components,
        "near_close_component": near_close_score,
        "details": {
            "pnl_pct": pnl_pct,
            "momentum": mom,
            "acceleration": acceleration,
            "bid_pressure_3m_pct": bid_pressure,
            "spread_pct": spread,
            "rolling_vwap": q.get("rolling_vwap") if q else None,
            "vwap_distance_pct": q.get("vwap_distance_pct") if q else None,
            "price_above_vwap": price_above_vwap,
            "drawdown_from_high": drawdown_from_high,
            "structure_reason": structure_reason,
            "momentum_reason": momentum_reason,
            "local_time": str(local_time),
        },
    }


def compute_kelly_scale(realized_pnls: list[float], kelly_cfg: dict[str, Any]) -> dict[str, Any]:
    """Fractional-Kelly position-size multiplier in [floor, 1.0].

    Decision-theoretic sizing robust to estimation error (arXiv:2107.08827,
    1612.07194): a half/quarter Kelly fraction shrinks risk when the edge is
    uncertain. CRITICAL SAFETY PROPERTY: this can only ever REDUCE the configured
    risk_per_trade_pct, never inflate it on a noisy small-sample edge. With no
    demonstrated edge (current state) it returns the floor — tiny but non-zero
    so data keeps accumulating.
    """
    floor = as_float(kelly_cfg.get("floor_scale", 0.5))
    detail: dict[str, Any] = {"scale": floor, "reason": "floor", "n_trades": len(realized_pnls)}
    if not bool(kelly_cfg.get("enabled", True)):
        detail.update({"scale": 1.0, "reason": "kelly_disabled"})
        return detail
    min_trades = int(as_float(kelly_cfg.get("min_trades_for_kelly", 8), 8))
    n = len(realized_pnls)
    if n < min_trades:
        detail["reason"] = f"insufficient_trades:{n}<{min_trades}"
        return detail
    wins = [p for p in realized_pnls if p > 0]
    losses = [-p for p in realized_pnls if p < 0]
    win_rate = len(wins) / n
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    if avg_loss <= 0:
        # no losing trades in window -> payoff ratio unbounded; Kelly fraction = win_rate
        f_star = win_rate
    else:
        b = avg_win / avg_loss
        f_star = win_rate - (1.0 - win_rate) / b if b > 0 else -1.0
    # Empirical-Bayes / James-Stein shrinkage of the estimated edge toward zero with
    # strength decreasing in sample size (arXiv:2512.25042, 1110.3460). At small n the
    # raw f_star is noisy; shrinking it toward 0 makes a lucky win-streak size up more
    # slowly and converges to the empirical estimate as trades accumulate. This can only
    # reduce a positive edge (slower upsizing); a negative edge stays <=0 -> floor.
    f_raw = f_star
    pseudo = as_float(kelly_cfg.get("edge_shrinkage_pseudo_trades", 10))
    if pseudo > 0:
        f_star = f_star * (n / (n + pseudo))
    # Interpolate scale from floor (no/poor edge) up to 1.0 (configured risk) as the
    # demonstrated Kelly fraction f_star approaches f_target. Half-Kelly robustness is
    # encoded by requiring a solid edge (f_target) before allowing full configured risk,
    # plus the hard cap at 1.0 (Kelly never inflates beyond the configured budget).
    f_target = as_float(kelly_cfg.get("f_star_for_full_risk", 0.25))
    progress = max(0.0, f_star) / f_target if f_target > 0 else 0.0
    scale = floor + (1.0 - floor) * min(1.0, progress)
    scale = max(floor, min(1.0, scale))
    detail.update({
        "scale": round(scale, 4), "reason": "ok", "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "f_star": round(f_star, 4), "f_star_raw": round(f_raw, 4), "f_target": f_target,
        "shrinkage_pseudo_trades": pseudo,
    })
    return detail


def _market_breadth_up_frac(quotes: list[dict[str, Any]]) -> float | None:
    """Fraction of this snapshot's quotes that are up on the day (broad-strength gauge)."""
    vals = [as_float(q.get("change_pct")) for q in quotes if q and q.get("change_pct") is not None]
    return (sum(1 for x in vals if x > 0) / len(vals)) if vals else None


def apply_sell_logic_v2(
    strategy: dict[str, Any], state: dict[str, Any], trade_date: str, code: str,
    sell_score_result: dict[str, Any], sell_threshold: float,
    quotes: list[dict[str, Any]], local_time: datetime,
) -> tuple[bool, float, str]:
    """Gate a TIMING sell (`unified_sell_score_exit`) to cut whipsaw. Returns
    (do_sell, sell_fraction, reason). Hard stops/kill/emergency never reach here, so
    they are untouched. When `sell_logic_v2.enabled` is false this is a no-op that
    returns (True, 1.0, "unified_sell_score_exit") -> byte-identical baseline behavior.

    Levers (all data already in scope; no signature changes): H5 require >=N independent
    negative components; H3 in strong breadth require score >= threshold+delta; H2 require
    the signal to persist across consecutive snapshots; H4 scale out a fraction and trail
    the rest. Pending state lives in state['pending_sell_v2_by_date'][date][code]."""
    cfg = strategy.get("sell_logic_v2", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return True, 1.0, "unified_sell_score_exit"
    min_components = max(1, int(as_float(cfg.get("min_independent_negative_components"), 1)))
    confirm_n = max(1, int(as_float(cfg.get("confirmation_snapshots"), 2)))
    max_gap = int(as_float(cfg.get("confirmation_max_gap_minutes"), 7))
    strong_breadth = as_float(cfg.get("strong_tape_breadth"), 1.01)   # >1 disables H3
    delta = as_float(cfg.get("strong_tape_threshold_delta"), 0.0)
    scale = as_float(cfg.get("scale_out_fraction"), 1.0)
    pend = state.setdefault("pending_sell_v2_by_date", {}).setdefault(trade_date, {})

    components = sell_score_result.get("components", {}) if isinstance(sell_score_result.get("components"), dict) else {}
    neg_keys = ("loss_depth", "structure_break", "momentum_reversal", "bid_pressure_negative",
                "acceleration_negative", "liquidity_deterioration", "vwap_breakdown",
                "profit_drawdown", "drawdown_from_high", "profit_momentum_negative",
                "profit_acceleration_negative", "profit_bid_pressure_negative")
    n_neg = sum(1 for k in neg_keys if as_float(components.get(k)) > 0)
    if n_neg < min_components:
        pend.pop(code, None)
        return False, 0.0, "v2_suppress_insufficient_components"

    breadth = _market_breadth_up_frac(quotes)
    score = as_float(sell_score_result.get("score"))
    if breadth is not None and breadth >= strong_breadth and score < sell_threshold + delta:
        pend.pop(code, None)
        return False, 0.0, "v2_suppress_strong_tape"

    now_min = local_time.hour * 60 + local_time.minute
    rec = pend.get(code) if isinstance(pend.get(code), dict) else None
    count = int(rec.get("count", 0)) + 1 if rec and now_min - int(rec.get("last_min", -10_000)) <= max_gap else 1
    if count < confirm_n:
        pend[code] = {"last_min": now_min, "count": count}
        return False, 0.0, "v2_awaiting_confirmation"
    pend.pop(code, None)
    frac = scale if 0.0 < scale < 1.0 else 1.0
    return True, frac, "unified_sell_score_exit" if frac >= 1.0 else "unified_sell_score_exit_v2_scaled"


def evaluate_exit_for_code(
    code: str,
    *,
    strategy: dict[str, Any],
    filters: dict[str, Any],
    risk: dict[str, Any],
    state: dict[str, Any],
    trade_date: str,
    local_time: datetime,
    quotes: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
    sellable_by_code: dict[str, int],
    lot_size: int,
    kill_switch_active: bool,
) -> dict[str, Any]:
    """Unified sell_score evaluation for ONE held position.

    Returns {"order": dict|None, "reason": str, "carry_allowed": bool,
    "sell_score": float, "sell_score_components": dict}. Pure per-position;
    callers loop over every held code so multiple overnight positions are
    all evaluated for exit in the same run (no single-position starvation).
    """
    held_pos = positions.get(code) or {}
    q = quote_for_code(quotes, code)
    available_qty = sellable_by_code.get(code, 0)
    current = as_float(q.get("currentPrice")) if q else 0.0
    node = update_t0_mark_state(state, trade_date, code, current)
    cost_price = t0_entry_price(node, held_pos)
    pnl_pct = current / cost_price - 1.0 if cost_price > 0 and current > 0 else 0.0
    mom = as_float(q.get("momentum"), 0.0) if q else 0.0
    acceleration_raw = q.get("acceleration") if q else None
    acceleration = as_float(acceleration_raw) if acceleration_raw is not None else None
    bid_pressure_raw = q.get("bid_pressure_3m_pct") if q else None
    bid_pressure = as_float(bid_pressure_raw) if bid_pressure_raw is not None else None
    highest_price = as_float(node.get("highest_price_since_entry"), current)
    drawdown_from_high = current / highest_price - 1.0 if highest_price > 0 and current > 0 else 0.0
    holding_minutes = minutes_since(node.get("first_buy_at") or node.get("last_buy_at"))
    min_hold_minutes = as_float(strategy.get("min_hold_minutes"), 10)
    emergency_stop_pct = as_float(strategy.get("emergency_stop_pct"), -0.02)
    node_r_value = as_float(node.get("r_value"), 0.0)
    sell_score_result = score_unified_sell(
        strategy=strategy,
        filters=filters,
        local_time=local_time,
        state=state,
        trade_date=trade_date,
        q=q,
        code=code,
        pnl_pct=pnl_pct,
        current=current,
        highest_price=highest_price,
    )
    sell_score_val = as_float(sell_score_result.get("score"), 0.0)
    sell_threshold = as_float(sell_score_result.get("threshold"), 70)
    near_close_comp = as_float(sell_score_result.get("near_close_component"), 0.0)
    score_pass = sell_score_val >= sell_threshold and (sell_score_val - near_close_comp) > 0
    min_hold_passed = holding_minutes is None or holding_minutes >= min_hold_minutes
    im_cfg = strategy.get("intraday_momentum", {}) if isinstance(strategy.get("intraday_momentum", {}), dict) else {}
    is_im_position = bool(node.get("im_trade"))
    exit_reason = None
    unconditional_exit = False
    if current <= 0:
        exit_reason = None  # 行情坏点(价格为0/缺失)时不卖
    elif kill_switch_active:
        exit_reason = "kill_switch_liquidation"
        unconditional_exit = True
    elif is_im_position and im_force_exit_due(local_time, im_cfg):
        # intraday-momentum is an intraday-only bet -> force exit by exit_time, never carry overnight
        exit_reason = "intraday_momentum_eod_exit"
        unconditional_exit = True
    elif cost_price <= 0:
        exit_reason = "unrecognized_position_state_liquidation"
        unconditional_exit = True
    elif pnl_pct <= emergency_stop_pct:
        exit_reason = "emergency_stop_exit"
        unconditional_exit = True
    elif not min_hold_passed:
        exit_reason = None
    elif score_pass:
        exit_reason = "unified_sell_score_exit"
    # v2 timing-sell de-noising (default OFF). Only gates the unified_sell_score_exit
    # timing sell; hard stops/kill/emergency/unrecognized above are never routed here.
    sell_fraction = 1.0
    if exit_reason == "unified_sell_score_exit":
        do_sell, sell_fraction, exit_reason = apply_sell_logic_v2(
            strategy, state, trade_date, code, sell_score_result,
            sell_threshold, quotes, local_time)
        if not do_sell:
            exit_reason = None
    carry_allowed = exit_reason is None
    px = safe_sell_reference_price(q, current, node, held_pos)
    px *= 1.0 - as_float(risk["limit_price_slippage_pct"])
    order: dict[str, Any] | None = None
    if available_qty >= int(risk["min_order_quantity"]) and exit_reason and px > 0:
        scaled_qty = round_lot(available_qty * sell_fraction, lot_size)
        qty = scaled_qty if scaled_qty >= int(risk["min_order_quantity"]) else round_lot(available_qty, lot_size)
        r_multiple = round((px - cost_price) / node_r_value, 2) if node_r_value > 0 else None
        order = {
            "direction": "sell",
            "stockCode": code,
            "exchange": q.get("exchange") if q else held_pos.get("exchange", "SH"),
            "name": q.get("name") if q else held_pos.get("stockName"),
            "quantity": qty,
            "orderType": risk["order_type"],
            "price": round(px, 3),
            "reason": exit_reason,
            "momentum": mom,
            "pnl_pct": pnl_pct,
            "acceleration": acceleration,
            "bid_pressure_3m_pct": bid_pressure,
            "sell_score": sell_score_val,
            "sell_score_components": sell_score_result.get("components"),
            "sell_score_threshold": sell_threshold,
            "exit_policy": "unified_sell_score",
            "unconditional_exit": unconditional_exit,
            "holding_minutes": holding_minutes,
            "min_hold_minutes": min_hold_minutes,
            "highest_price_since_entry": highest_price,
            "drawdown_from_high": drawdown_from_high,
            "cost_price": cost_price,
            "r_multiple": r_multiple,
            "t0_inventory_sellable_qty": available_qty,
            "sell_scope": "unified_sell_score_all_held",
            "t0_eligible": True,
            "execution_style": "aggressive",  # exits always cross for fill certainty (passive exits are v2)
            "submission_mid": round((as_float(q.get("bidPrice1"), current) + as_float(q.get("askPrice1"), current)) / 2.0, 4) if q else round(current, 4),
        }
        hold_reason = exit_reason
    else:
        if exit_reason and px <= 0:
            hold_reason = "no_valid_sell_price"
        elif exit_reason and available_qty < int(risk["min_order_quantity"]):
            hold_reason = "sellable_below_min_order_quantity"
        elif not min_hold_passed:
            hold_reason = "min_hold_active_no_exit"
        else:
            hold_reason = "carry_allowed_sell_score_not_met"
    if q is not None and order is None:
        q["held_position_diagnostics"] = {
            "pnl_pct": pnl_pct,
            "sell_score": sell_score_val,
            "sell_score_components": sell_score_result.get("components"),
            "sell_score_threshold": sell_threshold,
            "carry_allowed": carry_allowed,
            "holding_minutes": holding_minutes,
            "min_hold_minutes": min_hold_minutes,
            "highest_price_since_entry": highest_price,
            "drawdown_from_high": drawdown_from_high,
            "exit_policy": "unified_sell_score",
        }
    return {
        "order": order,
        "reason": hold_reason,
        "carry_allowed": carry_allowed,
        "sell_score": sell_score_val,
        "sell_score_components": sell_score_result.get("components"),
        "unconditional_exit": unconditional_exit,
    }


def score_entry(
    *,
    strategy: dict[str, Any],
    filters: dict[str, Any],
    q: dict[str, Any] | None,
    orb: dict[str, Any] | None,
    broad_market_not_declining: bool,
    consolidation_entry_allowed: bool = False,
) -> dict[str, Any]:
    weights = strategy.get("entry_score_weights", {})
    mom = as_float(q.get("momentum"), 0.0) if q else 0.0
    acceleration_raw = q.get("acceleration") if q else None
    acceleration = as_float(acceleration_raw) if acceleration_raw is not None else None
    bid_pressure_raw = q.get("bid_pressure_3m_pct") if q else None
    bid_pressure = as_float(bid_pressure_raw) if bid_pressure_raw is not None else None
    spread = q.get("spread_pct") if q else None
    max_spread = as_float(filters.get("max_spread_pct"), 0.0015)
    current = as_float(q.get("currentPrice"), 0.0) if q else 0.0
    cross_divergence = as_float(q.get("cross_etf_divergence_pct"), 0.0) if q else 0.0

    orb_score = 0.0
    orb_reason = "none"
    if orb and current > 0:
        if current > as_float(orb.get("high")) * 1.001 and mom > 0 and (bid_pressure or 0.0) > 0:
            orb_score = as_float(weights.get("orb_breakout", 25), 25)
            orb_reason = "full_breakout"
        elif current > as_float(orb.get("midpoint")) and mom > 0:
            orb_score = as_float(weights.get("orb_breakout", 25), 25) * 0.5
            orb_reason = "above_midpoint"

    entry_mom_thr = as_float(strategy.get("entry_momentum_pct"), 0.0015)
    mom_score = 0.0
    if mom >= entry_mom_thr * 2:
        mom_score = as_float(weights.get("momentum_strength", 20), 20)
    elif mom >= entry_mom_thr:
        mom_score = as_float(weights.get("momentum_strength", 20), 20) * 0.5

    bid_score = as_float(weights.get("bid_pressure_positive", 20), 20) if bid_pressure is not None and bid_pressure > 0 else 0.0
    accel_score = as_float(weights.get("acceleration_positive", 15), 15) if acceleration is not None and acceleration > 0 else 0.0
    spread_score = as_float(weights.get("tight_spread", 10), 10) if spread is not None and spread < max_spread * 0.5 else 0.0
    bollinger_score = 0.0
    bollinger_reason = "none"
    bollinger = q.get("bollinger_squeeze") if q else None
    if isinstance(bollinger, dict) and bollinger.get("available"):
        if bollinger.get("squeeze") and bollinger.get("breakout") and mom > 0:
            bollinger_score = as_float(weights.get("bollinger_squeeze_breakout", 15), 15)
            bollinger_reason = "squeeze_breakout"
        elif bollinger.get("squeeze"):
            bollinger_reason = "squeeze_no_breakout"
        else:
            bollinger_reason = "not_squeeze"

    divergence_thr = as_float(strategy.get("cross_etf_divergence_threshold", 0.002), 0.002)
    divergence_score = 0.0
    if divergence_thr > 0:
        if cross_divergence >= divergence_thr:
            divergence_score = as_float(weights.get("cross_etf_divergence", 15), 15)
        elif cross_divergence > 0:
            divergence_score = as_float(weights.get("cross_etf_divergence", 15), 15) * (cross_divergence / divergence_thr)

    breadth_score = as_float(weights.get("market_breadth_positive", 10), 10) if broad_market_not_declining else 0.0

    # Alpha#101 intraday conviction (single-name, long-only momentum confirmation).
    conviction_raw = q.get("alpha101_conviction") if q else None
    conviction = as_float(conviction_raw) if conviction_raw is not None else None
    conv_thr = as_float(strategy.get("alpha101_conviction_threshold", 0.5), 0.5)
    conv_weight = as_float(weights.get("alpha101_intraday_conviction", 0), 0)
    conviction_score = 0.0
    conviction_reason = "none"
    if conviction is not None and conv_thr > 0 and conv_weight > 0 and mom > 0:
        if conviction >= conv_thr:
            conviction_score = conv_weight
            conviction_reason = "strong_intraday_conviction"
        elif conviction > 0:
            conviction_score = conv_weight * (conviction / conv_thr)
            conviction_reason = "weak_intraday_conviction"

    cons_cfg = strategy.get("consolidation", {})
    box = q.get("consolidation_box") if q else None
    cons_score = 0.0
    cons_reason = "none"
    if cons_cfg.get("enabled") and consolidation_entry_allowed and isinstance(box, dict) and current > 0:
        cons_buffer = as_float(cons_cfg.get("breakout_buffer_pct", 0.001))
        if current > as_float(box.get("high")) * (1.0 + cons_buffer) and mom > 0:
            cons_score = as_float(weights.get("consolidation_breakout", 20), 20)
            cons_reason = "box_breakout"
        else:
            cons_reason = "box_present_no_breakout"

    components = {
        "consolidation_breakout": cons_score,
        "orb_breakout": orb_score,
        "momentum_strength": mom_score,
        "bid_pressure_positive": bid_score,
        "acceleration_positive": accel_score,
        "tight_spread": spread_score,
        "bollinger_squeeze_breakout": bollinger_score,
        "cross_etf_divergence": divergence_score,
        "market_breadth_positive": breadth_score,
        "alpha101_intraday_conviction": conviction_score,
    }
    total = sum(as_float(x, 0.0) for x in components.values())
    return {
        "score": round(total, 2),
        "threshold": as_float(strategy.get("entry_score_threshold"), 50),
        "components": components,
        "details": {
            "momentum": mom,
            "bid_pressure_3m_pct": bid_pressure,
            "acceleration": acceleration,
            "spread_pct": spread,
            "rolling_vwap": q.get("rolling_vwap") if q else None,
            "price_above_vwap": q.get("price_above_vwap") if q else None,
            "vwap_distance_pct": q.get("vwap_distance_pct") if q else None,
            "bollinger_reason": bollinger_reason,
            "bollinger_squeeze": bollinger,
            "cross_etf_divergence_pct": cross_divergence,
            "orb_reason": orb_reason,
            "orb_high": orb.get("high") if orb else None,
            "consolidation_reason": cons_reason,
            "consolidation_box": box,
            "alpha101_conviction": conviction,
            "alpha101_conviction_reason": conviction_reason,
        },
    }


def t0_positions(positions_resp: dict[str, Any], universe: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_pos = positions_by_code(positions_resp)
    codes = {str(x["stockCode"]).zfill(6) for x in universe}
    return {code: pos for code, pos in all_pos.items() if code in codes and as_float(pos.get("quantity")) > 0}


def update_orb_state(state: dict[str, Any], trade_date: str, quotes: list[dict[str, Any]], local_time: datetime, in_regular_session: bool) -> None:
    orb_root = state.get("ORB")
    if not isinstance(orb_root, dict):
        orb_root = {}
    day = orb_root.get(trade_date)
    if not isinstance(day, dict):
        day = {"prices": [], "by_code": {}}
    prices = day.get("prices")
    if not isinstance(prices, list):
        prices = []
    by_code = day.get("by_code")
    if not isinstance(by_code, dict):
        by_code = {}

    collect_open_range = bool(in_regular_session and local_time.hour == 9 and 30 <= local_time.minute < 45)
    finalized = bool((local_time.hour == 9 and local_time.minute >= 45) or local_time.hour >= 10)
    if collect_open_range:
        for q in quotes:
            current = as_float(q.get("currentPrice"))
            if current <= 0 or not q.get("quote_ok"):
                continue
            prices.append(current)
            code = str(q.get("stockCode", "")).zfill(6)
            node = by_code.get(code)
            if not isinstance(node, dict):
                node = {"prices": []}
            code_prices = node.get("prices")
            if not isinstance(code_prices, list):
                code_prices = []
            code_prices.append(current)
            node["prices"] = code_prices
            node["high"] = max(code_prices)
            node["low"] = min(code_prices)
            node["midpoint"] = (node["high"] + node["low"]) / 2.0
            node["finalized"] = finalized
            by_code[code] = node

    if prices:
        day["prices"] = prices
        day["high"] = max(prices)
        day["low"] = min(prices)
        day["midpoint"] = (day["high"] + day["low"]) / 2.0
    day["finalized"] = finalized
    for node in by_code.values():
        if isinstance(node, dict) and node.get("prices"):
            node["finalized"] = finalized
    day["by_code"] = by_code
    orb_root[trade_date] = day
    state["ORB"] = orb_root


def get_orb(state: dict[str, Any], trade_date: str, stock_code: str | None = None) -> dict[str, Any] | None:
    day = state.get("ORB", {}).get(trade_date) if isinstance(state.get("ORB"), dict) else None
    if not isinstance(day, dict):
        return None
    node: dict[str, Any] | None = day
    if stock_code is not None:
        by_code = day.get("by_code", {})
        node = by_code.get(str(stock_code).zfill(6)) if isinstance(by_code, dict) else None
    if not isinstance(node, dict) or not node.get("finalized"):
        return None
    high = as_float(node.get("high"))
    low = as_float(node.get("low"))
    midpoint = as_float(node.get("midpoint"))
    if high <= 0 or low <= 0 or midpoint <= 0:
        return None
    return {"high": high, "low": low, "midpoint": midpoint, "finalized": True}


def compute_execution_quality(q: dict[str, Any], filters: dict[str, Any], risk: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    """Microstructure quality gate for BUY orders.

    This is deliberately a risk filter, not an alpha signal. It blocks obviously
    bad execution states (wide spread, invalid book, high expected crossing
    cost) while leaving SELL exits governed by their own risk path.
    """
    cfg = strategy.get("execution_quality", {}) if isinstance(strategy.get("execution_quality"), dict) else {}
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "passed": True, "score": 100.0, "status": "disabled"}

    max_spread_pct = as_float(filters.get("max_spread_pct"), 0.0015)
    min_score = as_float(cfg.get("min_score"), 60.0)
    max_slippage_bps = as_float(cfg.get("max_expected_slippage_bps"), 12.0)
    shadow_only = bool(cfg.get("shadow_only", False))
    current = as_float(q.get("currentPrice"))
    bid = as_float(q.get("bidPrice1"))
    ask = as_float(q.get("askPrice1"))
    midpoint = as_float(q.get("midpoint"))
    spread_pct = q.get("spread_pct")
    spread = as_float(spread_pct) if spread_pct is not None else None
    penalties: dict[str, float] = {}

    if not q.get("quote_ok") or q.get("isSuspended"):
        penalties["bad_quote"] = as_float(cfg.get("bad_quote_penalty"), 100.0)
    if current <= 0:
        penalties["invalid_current_price"] = 100.0
    if bid <= 0 or ask <= 0 or ask < bid:
        penalties["invalid_book"] = 50.0

    if spread is None or spread < 0:
        penalties["missing_spread"] = 25.0
        spread_bps = None
    else:
        spread_bps = spread * 10000.0
        excess_spread_bps = max(0.0, spread_bps - max_spread_pct * 10000.0)
        if excess_spread_bps > 0:
            penalties["spread_excess"] = excess_spread_bps * as_float(cfg.get("spread_penalty_per_bp"), 2.0)

    mid = midpoint if midpoint > 0 else ((bid + ask) / 2.0 if bid > 0 and ask > 0 else current)
    if mid > 0 and ask > 0:
        half_spread_bps = max(0.0, (ask - bid) / (2.0 * mid) * 10000.0) if bid > 0 else 0.0
        crossing_bps = max(0.0, (ask - mid) / mid * 10000.0)
    else:
        half_spread_bps = 999.0
        crossing_bps = 999.0
    expected_slippage_bps = max(half_spread_bps, crossing_bps) + as_float(risk.get("limit_price_slippage_pct"), 0.001) * 10000.0
    if expected_slippage_bps > max_slippage_bps:
        penalties["expected_slippage_excess"] = (expected_slippage_bps - max_slippage_bps) * as_float(cfg.get("slippage_penalty_per_bp"), 1.5)

    bid_pressure = q.get("bid_pressure_3m_pct")
    if bid_pressure is not None and as_float(bid_pressure) < 0:
        penalties["negative_bid_pressure"] = as_float(cfg.get("negative_bid_pressure_penalty"), 10.0)

    score = max(0.0, 100.0 - sum(penalties.values()))
    would_pass = score >= min_score and expected_slippage_bps <= max_slippage_bps
    return {
        "enabled": True,
        "passed": True if shadow_only else bool(would_pass),
        "would_pass": bool(would_pass),
        "shadow_only": shadow_only,
        "score": round(score, 2),
        "min_score": min_score,
        "spread_bps": round(spread_bps, 3) if spread_bps is not None else None,
        "expected_slippage_bps": round(expected_slippage_bps, 3),
        "max_expected_slippage_bps": max_slippage_bps,
        "penalties": penalties,
        "status": "ok" if would_pass else "blocked_microstructure_cost",
    }


def _recent_history_for_code(history: list[dict[str, Any]], code: str, max_rows: int) -> list[dict[str, Any]]:
    zcode = str(code).zfill(6)
    if history:
        first_code = str(history[0].get("stockCode", "")).zfill(6)
        last_code = str(history[-1].get("stockCode", "")).zfill(6)
        if first_code == zcode and last_code == zcode:
            return history[-max_rows:]
    out: list[dict[str, Any]] = []
    for row in reversed(history or []):
        if str(row.get("stockCode", "")).zfill(6) == zcode:
            out.append(row)
            if len(out) >= max_rows:
                break
    out.reverse()
    return out


def compute_safe_policy_shield(
    q: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    state: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Safe offline-RL style action shield.

    We do not run an RL policy live. This guard only rejects candidate BUYs that
    are far outside the observed replay distribution or repeat a very recent
    losing action on the same ETF.
    """
    cfg = strategy.get("safe_policy_shield", {}) if isinstance(strategy.get("safe_policy_shield"), dict) else {}
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "passed": True, "status": "disabled"}
    if not q:
        return {"enabled": True, "passed": False, "status": "no_candidate"}

    code = str(q.get("stockCode", "")).zfill(6)
    min_rows = int(as_float(cfg.get("min_history_snapshots"), 20))
    max_rows = int(as_float(cfg.get("max_history_snapshots"), 120))
    max_abs_z_allowed = as_float(cfg.get("max_zscore_abs"), 4.0)
    shadow_only = bool(cfg.get("shadow_only", False))
    features = cfg.get("features")
    if not isinstance(features, list) or not features:
        features = ["momentum", "spread_pct", "bid_pressure_3m_pct", "vwap_distance_pct", "acceleration"]

    hist_rows = _recent_history_for_code(history or [], code, max_rows)
    zscores: dict[str, float] = {}
    insufficient: list[str] = []
    for feat in features:
        cur = q.get(feat)
        if cur is None:
            insufficient.append(str(feat))
            continue
        vals = [as_float(r.get(feat)) for r in hist_rows if r.get(feat) is not None]
        if len(vals) < min_rows:
            insufficient.append(str(feat))
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        sd = math.sqrt(var)
        if sd <= 1e-12:
            continue
        zscores[str(feat)] = (as_float(cur) - mean) / sd

    max_abs_z = max((abs(v) for v in zscores.values()), default=0.0)
    ood_block = bool(zscores) and max_abs_z > max_abs_z_allowed

    recent_loss_block = False
    recent_losses = []
    block_minutes = as_float(cfg.get("negative_sample_block_minutes"), 60)
    samples = state.get("negative_action_samples")
    if isinstance(samples, list):
        for sample in reversed(samples[-100:]):
            if str(sample.get("stockCode", "")).zfill(6) != code:
                continue
            mins = minutes_since(sample.get("timestamp"))
            if mins is not None and mins <= block_minutes:
                recent_loss_block = True
                recent_losses.append({
                    "timestamp": sample.get("timestamp"),
                    "minutes_since": round(mins, 1),
                    "pnl_estimate": sample.get("pnl_estimate"),
                    "reason": sample.get("reason"),
                })
                break

    would_pass = not ood_block and not recent_loss_block
    return {
        "enabled": True,
        "passed": True if shadow_only else bool(would_pass),
        "would_pass": bool(would_pass),
        "shadow_only": shadow_only,
        "status": "ok" if would_pass else ("recent_negative_sample_block" if recent_loss_block else "ood_feature_state"),
        "stockCode": code,
        "history_rows": len(hist_rows),
        "min_history_snapshots": min_rows,
        "max_abs_z": round(max_abs_z, 3),
        "max_zscore_abs": max_abs_z_allowed,
        "zscores": {k: round(v, 3) for k, v in zscores.items()},
        "insufficient_features": insufficient,
        "recent_negative_samples": recent_losses,
    }


def record_negative_action_sample(state: dict[str, Any], trade_date: str, order: dict[str, Any], pnl_estimate: float) -> None:
    samples = state.get("negative_action_samples")
    if not isinstance(samples, list):
        samples = []
    samples.append({
        "timestamp": now_iso(),
        "trade_date": trade_date,
        "stockCode": str(order.get("stockCode", "")).zfill(6),
        "direction": order.get("direction"),
        "reason": order.get("reason"),
        "pnl_estimate": round(as_float(pnl_estimate), 2),
        "pnl_pct": order.get("pnl_pct"),
        "cost_price": order.get("cost_price"),
        "fill_price_estimate": order.get("price"),
        "quantity": order.get("quantity"),
    })
    state["negative_action_samples"] = samples[-200:]


def build_decision(
    cfg: dict[str, Any],
    quotes: list[dict[str, Any]],
    balance: dict[str, Any],
    positions_resp: dict[str, Any],
    pending_resp: dict[str, Any],
    state: dict[str, Any],
    market_correlation_stress: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    feature_history_by_code: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    session = cn_market_session(_REPLAY_NOW)
    local_time = exchange_local_time(session)
    strategy = cfg["strategy"]
    indicators_cfg = strategy.get("indicators", {}) if isinstance(strategy.get("indicators", {}), dict) else {}
    filters = cfg["filters"]
    risk = cfg["risk"]
    trade_date = current_dt().astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    total_assets, available_cash = account_assets(balance)
    positions = t0_positions(positions_resp, cfg["universe"])
    lot_size = int(risk["quantity_lot"])

    pending_data = extract_data(pending_resp)
    pending_orders = pending_data.get("orders", [])
    if not isinstance(pending_orders, list):
        pending_orders = []
    t0_codes = {str(x["stockCode"]).zfill(6) for x in cfg["universe"]}
    pending_t0 = [o for o in pending_orders if str(o.get("stockCode", "")).zfill(6) in t0_codes]

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    kill_switch = ROOT / risk["kill_switch_file"]
    add("paper_execute_enabled", cfg.get("mode") == "paper_execute" and bool(cfg.get("execution_enabled")))
    add("regular_trading_session", session["in_regular_session"], session)
    open_quiet_passed = (local_time.hour > 9) or (local_time.hour == 9 and local_time.minute >= 45)
    add("open_quiet_period_passed", open_quiet_passed, {
        "rule": "no_entry_before_09:45",
        "exchange_local_time": str(local_time),
        "system_local_time": str(system_local_time()),
    })
    # 全局API配额(全平台共享)曾在14:06耗尽; 入场截止提前到14:00, 强平提前到14:20, 给平仓留配额缓冲
    no_new_entry_after_cutoff = not (local_time.hour >= 14)
    add("no_new_entry_afternoon_cutoff", no_new_entry_after_cutoff, {
        "rule": "no_new_entry_after_14:00",
        "exchange_local_time": str(local_time),
        "system_local_time": str(system_local_time()),
    })
    add("kill_switch_inactive", not kill_switch.exists(), str(kill_switch))
    add("balance_ok", bool(balance.get("ok")) and total_assets > 0, {"total_assets": total_assets, "available_cash": available_cash})
    add("positions_ok", bool(positions_resp.get("ok")))
    add("pending_orders_ok", bool(pending_resp.get("ok")))

    now_dt = current_dt()
    stale_codes: list[str] = []
    max_age_seconds = 0.0
    for q in quotes:
        if not q.get("quote_ok"):
            continue
        quote_dt = parse_iso_dt(q.get("timestamp"))
        age_seconds = (now_dt - quote_dt).total_seconds() if quote_dt else 999999.0
        max_age_seconds = max(max_age_seconds, age_seconds)
        if age_seconds > 300:
            stale_codes.append(str(q.get("stockCode", "")).zfill(6))
    add("quote_freshness", not stale_codes, {
        "stale_codes": stale_codes,
        "max_age_seconds": max_age_seconds,
    })

    # Daily trade-FREQUENCY caps: a value <= 0 means UNLIMITED (per user: no daily
    # trading cap, so the wide-universe / 5-holding strategy can fill freely). The daily
    # LOSS circuit-breaker and the API-run quota are separate safety controls and remain.
    max_daily_orders = int(risk["max_daily_submitted_orders"])
    max_daily_round_trips = int(risk["max_daily_round_trips"])
    daily_orders = daily_state_bucket(state, "submitted_orders_by_date", trade_date)
    daily_round_trips = daily_state_bucket(state, "round_trips_by_date", trade_date)
    add("daily_order_limit", max_daily_orders <= 0 or daily_orders < max_daily_orders, daily_orders)
    add("daily_round_trip_limit", max_daily_round_trips <= 0 or daily_round_trips < max_daily_round_trips, daily_round_trips)
    # 独立的卖出额度: daily_order_limit 对 SELL 放行后, 用此项防止卖出循环滥用 (出场仍受限但额度更宽)
    daily_sell_orders = daily_state_bucket(state, "sell_orders_by_date", trade_date)
    max_daily_sell_orders = int(risk.get("max_daily_sell_orders", 12))
    add("daily_sell_order_limit", max_daily_sell_orders <= 0 or daily_sell_orders < max_daily_sell_orders, {
        "sell_orders_today": daily_sell_orders,
        "max_daily_sell_orders": max_daily_sell_orders,
        "unlimited": max_daily_sell_orders <= 0,
    })
    today_realized_pnl = daily_state_float_bucket(state, "today_realized_pnl", trade_date)
    max_daily_loss_pct = as_float(risk.get("max_daily_loss_pct"), -0.02)
    add("daily_loss_limit", (today_realized_pnl / total_assets) > max_daily_loss_pct if total_assets > 0 else False, {
        "today_realized_pnl": today_realized_pnl,
        "limit": max_daily_loss_pct * total_assets,
    })

    if pending_t0:
        add("no_pending_t0_orders", False, {"pending_count": len(pending_t0)})
    else:
        add("no_pending_t0_orders", True, {"pending_count": 0})

    volume_fields_available = any(q.get("raw_has_volume_field") or q.get("raw_has_amount_field") for q in quotes)
    volume_required = bool(filters.get("require_volume_field", False))
    add("volume_filter_available_or_not_required", volume_fields_available or not volume_required, {
        "volume_fields_available": volume_fields_available,
        "require_volume_field": volume_required,
        "status": "available" if volume_fields_available else "unavailable_api_field",
    })

    history_by_code: dict[str, list[dict[str, Any]]] = {}
    for row in history or []:
        hcode = str(row.get("stockCode", "")).zfill(6)
        if hcode:
            history_by_code.setdefault(hcode, []).append(row)

    def ensure_entry_diagnostics(q: dict[str, Any] | None) -> None:
        if not q:
            return
        if not isinstance(q.get("execution_quality"), dict):
            eq = compute_execution_quality(q, filters, risk, strategy)
            q["execution_quality"] = eq
            q["execution_quality_score"] = eq.get("score")
            q["expected_slippage_bps"] = eq.get("expected_slippage_bps")
        if not isinstance(q.get("safe_policy_shield"), dict):
            qcode = str(q.get("stockCode", "")).zfill(6)
            shield = compute_safe_policy_shield(q, history_by_code.get(qcode, []), state, strategy)
            q["safe_policy_shield"] = shield
            q["safe_policy_status"] = shield.get("status")

    def ensure_snapshot_features(q: dict[str, Any] | None) -> None:
        if not q or q.get("_snapshot_features_enriched", True):
            return
        qcode = str(q.get("stockCode", "")).zfill(6)
        enrich_snapshot_features(q, (feature_history_by_code or {}).get(qcode, []), strategy)

    for q in quotes:
        q.setdefault("execution_quality", None)
        q.setdefault("safe_policy_shield", None)

    liquid_quotes = []
    ranking_exclusions = []
    t0_entry_blocked_candidates: list[dict[str, Any]] = []
    raw_blocked_codes = strategy.get("entry_blocked_codes") or []
    entry_blocked_codes = {str(x).zfill(6) for x in raw_blocked_codes if str(x).strip()}
    blocked_code_reasons = strategy.get("entry_blocked_code_reasons", {})
    if not isinstance(blocked_code_reasons, dict):
        blocked_code_reasons = {}
    for q in quotes:
        spread = q.get("spread_pct")
        spread_ok = spread is not None and spread >= 0 and spread <= as_float(filters.get("max_spread_pct"), 0.0015)
        volume_ok = True
        if volume_fields_available:
            volume_ok = as_float(q.get("volume"), 0) >= as_float(filters.get("min_volume"), 0) and as_float(q.get("amount"), 0) >= as_float(filters.get("min_amount"), 0)
        qcode = str(q.get("stockCode", "")).zfill(6)
        if qcode in entry_blocked_codes:
            ranking_exclusions.append({
                "stockCode": q.get("stockCode"),
                "asset_class": q.get("asset_class"),
                "excluded_reason": blocked_code_reasons.get(qcode, "entry_blocked_by_config"),
            })
            continue
        if q.get("asset_class") in ENTRY_EXCLUDED_ASSET_CLASSES:
            ranking_exclusions.append({
                "stockCode": q.get("stockCode"),
                "asset_class": q.get("asset_class"),
                "excluded_reason": "exit_only_or_bond_not_eligible_for_t0_entry",
            })
            continue
        t0_entry_ok, t0_entry_detail = t0_entry_eligibility(q, strategy)
        if not t0_entry_ok:
            detail = {
                "stockCode": q.get("stockCode"),
                "asset_class": q.get("asset_class"),
                "excluded_reason": "blocked_t0_entry_ineligible_asset_class",
                **t0_entry_detail,
            }
            ranking_exclusions.append(detail)
            t0_entry_blocked_candidates.append(detail)
            continue
        if q.get("quote_ok") and not q.get("isSuspended") and spread_ok and volume_ok and q.get("momentum_available"):
            liquid_quotes.append(q)

    add("quote_liquidity_filter", bool(liquid_quotes), {
        "passed_count": len(liquid_quotes),
        "max_spread_pct": filters.get("max_spread_pct"),
        "volume_filter_status": "available" if volume_fields_available else "unavailable_api_field",
        "ranking_exclusions": ranking_exclusions,
    })
    t0_eligibility_cfg = strategy.get("t0_entry_eligibility", {})
    t0_entry_filter_available = (
        not isinstance(t0_eligibility_cfg, dict)
        or not t0_eligibility_cfg.get("enabled", False)
        or bool(liquid_quotes)
        or not t0_entry_blocked_candidates
    )
    add("t0_entry_eligibility_filter", t0_entry_filter_available, {
        "enabled": bool(isinstance(t0_eligibility_cfg, dict) and t0_eligibility_cfg.get("enabled", False)),
        "blocked_count": len(t0_entry_blocked_candidates),
        "blocked_candidates": t0_entry_blocked_candidates[:20],
        "policy": t0_eligibility_cfg if isinstance(t0_eligibility_cfg, dict) else {},
    })
    non_bond_quotes = [q for q in quotes if q.get("asset_class") not in ENTRY_EXCLUDED_ASSET_CLASSES]
    positive_count = sum(1 for q in non_bond_quotes if as_float(q.get("change_pct")) > -0.005)
    broad_market_not_declining = positive_count >= 1
    add("broad_market_not_declining", broad_market_not_declining, {
        "positive_count": positive_count,
        "total_non_bond": len(non_bond_quotes),
    })
    market_correlation_stress = market_correlation_stress or {
        "available": False,
        "status": "not_computed",
        "block_new_buy": False,
    }
    correlation_stress_ok = not bool(market_correlation_stress.get("block_new_buy"))
    add("market_correlation_stress_filter", correlation_stress_ok, market_correlation_stress)
    full_market_guard = evaluate_full_market_entry_guard(strategy, trade_date, local_time)
    full_market_entry_ok = not bool(full_market_guard.get("block_new_buy"))
    add("full_market_entry_guard", full_market_entry_ok, full_market_guard)

    action = "hold"
    reason = "no_signal"
    order: dict[str, Any] | None = None
    cross_pairs: dict[str, str] = {
        "513500": "513100", "513100": "513500",
        "513050": "513330", "513330": "513050",
    }
    quote_by_code: dict[str, dict[str, Any]] = {str(q.get("stockCode", "")).zfill(6): q for q in quotes}
    name_by_code: dict[str, str] = {}
    for etf in cfg.get("universe", []):
        code = str(etf.get("stockCode", "")).zfill(6)
        if code:
            name_by_code[code] = str(etf.get("name") or code)
    for code, q in quote_by_code.items():
        if q.get("name"):
            name_by_code[code] = str(q.get("name"))
    for code, pos in positions.items():
        pos_name = pos.get("stockName") or pos.get("name") or pos.get("securityName")
        if pos_name:
            name_by_code[code] = str(pos_name)
    for q in liquid_quotes:
        code = str(q.get("stockCode", "")).zfill(6)
        partner_code = cross_pairs.get(code)
        if partner_code:
            partner_q = quote_by_code.get(partner_code)
            if partner_q:
                q["cross_etf_divergence_pct"] = round(
                    as_float(partner_q.get("momentum"), 0.0) - as_float(q.get("momentum"), 0.0), 5
                )
            else:
                q["cross_etf_divergence_pct"] = 0.0
        else:
            q["cross_etf_divergence_pct"] = 0.0
    ranked = sorted(liquid_quotes, key=lambda x: as_float(x.get("momentum"), -999), reverse=True)
    # Holdings first, so the entry candidate excludes names we already hold: with
    # target_holdings>1 we add a NEW name each entry run, up to the cap.
    sellable_by_code = {}
    for code, pos in positions.items():
        t0_qty = t0_sellable_quantity(state, trade_date, code, pos, lot_size)
        overnight_qty = round_lot(as_float(pos.get("availableQuantity"), 0.0), lot_size)
        sellable_by_code[code] = max(t0_qty, overnight_qty)
    held_codes = [code for code, qty in sellable_by_code.items() if qty >= int(risk["min_order_quantity"])]
    held_code = held_codes[0] if held_codes else None
    held_pos = positions.get(held_code) if held_code else None
    target_holdings = max(1, int(as_float(strategy.get("target_holdings"), 1)))
    # COMMITTED holdings = every name we hold or bought today, INCLUDING same-day T+1
    # buys that are not yet sellable. held_codes only counts SELLABLE positions (for the
    # exit engine); the target_holdings cap and "don't re-buy what we hold" must use the
    # full committed set, otherwise T+1 names (most of the universe) never count as held
    # and we over-accumulate well past target_holdings.
    committed_codes: set[str] = {c for c, pos in positions.items() if as_float(pos.get("quantity"), 0.0) > 0}
    _today_inv = state.get("t0_inventory_by_date", {})
    if isinstance(_today_inv, dict) and isinstance(_today_inv.get(trade_date), dict):
        for code, node in _today_inv[trade_date].items():
            if isinstance(node, dict) and as_float(node.get("buy_quantity_submitted")) - as_float(node.get("sell_quantity_submitted")) > 0:
                committed_codes.add(str(code).zfill(6))
    committed_codes |= set(held_codes)
    # 存量(legacy)持仓单独成桶:它们仍在 committed_codes 内(=照常防重复买入、板块计数、
    # 由卖出引擎按各自条件退出),但 NOT counted against the target_holdings 新增配额。
    # 这样"现在就豪赌"——新增集中仓位不被存量挤占;存量清空后该名自然离开 committed_codes。
    legacy_inventory_codes = {str(c).zfill(6) for c in (strategy.get("legacy_inventory_codes") or [])}
    budget_codes: set[str] = committed_codes - legacy_inventory_codes
    add("legacy_inventory_split", True, {
        "legacy_held": sorted(legacy_inventory_codes & committed_codes),
        "new_book_codes": sorted(budget_codes),
        "new_book_count": len(budget_codes),
        "target_holdings": int(as_float(strategy.get("target_holdings"), 1)),
        "new_slots_open": max(0, int(as_float(strategy.get("target_holdings"), 1)) - len(budget_codes)),
    })
    sector_cfg = cfg.get("sector_diversification", {})
    if not isinstance(sector_cfg, dict):
        sector_cfg = {}
    sector_enabled = bool(sector_cfg.get("enabled", False))
    sector_keyword_map = sector_cfg.get("keyword_map") if isinstance(sector_cfg.get("keyword_map"), dict) else {}
    sector_max_per_sector = max(0, int(as_float(sector_cfg.get("max_per_sector"), 2)))
    raw_unlimited = sector_cfg.get("unlimited_sectors", ["other"])
    if isinstance(raw_unlimited, list):
        unlimited_sectors = {str(x) for x in raw_unlimited}
    else:
        unlimited_sectors = {"other"}
    held_sector_counts: dict[str, int] = {}
    held_sector_details: list[dict[str, Any]] = []
    for code in committed_codes:  # sector concentration counts ALL holdings, incl T+1
        name = name_by_code.get(code, code)
        sector = classify_etf_sector(name, sector_keyword_map)
        held_sector_counts[sector] = held_sector_counts.get(sector, 0) + 1
        held_sector_details.append({"stockCode": code, "name": name, "sector": sector})
    sector_blocked_candidates: list[dict[str, Any]] = []
    sector_blocked_codes: set[str] = set()

    def entry_sector_status(q: dict[str, Any], *, record: bool = True) -> tuple[bool, str]:
        code = str(q.get("stockCode", "")).zfill(6)
        name = str(q.get("name") or name_by_code.get(code, code))
        sector = classify_etf_sector(name, sector_keyword_map)
        held_count = int(held_sector_counts.get(sector, 0))
        limit_applies = (
            sector_enabled
            and sector_max_per_sector > 0
            and sector not in unlimited_sectors
        )
        blocked = bool(limit_applies and held_count >= sector_max_per_sector)
        if blocked and record and code not in sector_blocked_codes:
            sector_blocked_codes.add(code)
            sector_blocked_candidates.append({
                "stockCode": code,
                "name": name,
                "sector": sector,
                "held_sector_count": held_count,
                "max_per_sector": sector_max_per_sector,
                "blocked_reason": "blocked_sector_concentration",
            })
        return blocked, sector

    # entry candidate = highest-momentum liquid quote NOT already held and not
    # blocked by current holding sector concentration. This is entry-only; held
    # names are still evaluated by the sell engine above.
    nonheld_ranked_count = 0
    best = None
    best_sector = None
    for q in ranked:
        qcode = str(q.get("stockCode", "")).zfill(6)
        if qcode in committed_codes:  # never re-buy a name we already hold (incl T+1)
            continue
        nonheld_ranked_count += 1
        blocked, qsector = entry_sector_status(q)
        if blocked:
            continue
        best = q
        best_sector = qsector
        break
    sector_diversification_detail = {
        "enabled": sector_enabled,
        "max_per_sector": sector_max_per_sector,
        "unlimited_sectors": sorted(unlimited_sectors),
        "held_sector_counts": held_sector_counts,
        "held_sector_details": held_sector_details,
        "nonheld_ranked_count": nonheld_ranked_count,
        "blocked_candidates": sector_blocked_candidates,
        "selected_stockCode": best.get("stockCode") if best else None,
        "selected_sector": best_sector,
    }
    sector_entry_ok = (
        (not sector_enabled)
        or len(budget_codes) >= target_holdings
        or nonheld_ranked_count == 0
        or best is not None
    )
    add("sector_diversification_entry_filter", sector_entry_ok, sector_diversification_detail)
    ensure_snapshot_features(best)
    ensure_entry_diagnostics(best)
    best_execution_quality = best.get("execution_quality") if best and isinstance(best.get("execution_quality"), dict) else {
        "enabled": True, "passed": False, "status": "no_ranked_quote"
    }
    add("execution_quality_filter", bool(best_execution_quality.get("passed")), best_execution_quality)
    best_safe_policy = best.get("safe_policy_shield") if best and isinstance(best.get("safe_policy_shield"), dict) else {
        "enabled": True, "passed": False, "status": "no_ranked_quote"
    }
    add("safe_policy_shield", bool(best_safe_policy.get("passed")), best_safe_policy)
    vwap_cfg = indicators_cfg.get("rolling_vwap", {}) if isinstance(indicators_cfg.get("rolling_vwap", {}), dict) else {}
    vwap_filter_enabled = bool(vwap_cfg.get("require_price_above_for_entry", True))
    if best and vwap_filter_enabled and best.get("rolling_vwap_available"):
        vwap_entry_ok = bool(best.get("price_above_vwap"))
        vwap_entry_detail = {
            "stockCode": best.get("stockCode"),
            "rolling_vwap": best.get("rolling_vwap"),
            "currentPrice": best.get("currentPrice"),
            "vwap_distance_pct": best.get("vwap_distance_pct"),
            "status": "available",
            "rule": "entry_requires_price_above_vwap",
        }
    else:
        vwap_entry_ok = True
        vwap_entry_detail = {
            "stockCode": best.get("stockCode") if best else None,
            "status": best.get("rolling_vwap_status") if best else "no_ranked_quote",
            "enabled": vwap_filter_enabled,
            "rule": "vwap_unavailable_or_disabled_does_not_block_entry",
        }
    add("rolling_vwap_entry_filter", vwap_entry_ok, vwap_entry_detail)
    best_code = str(best.get("stockCode", "")).zfill(6) if best else None
    best_existing_position_qty = as_float(positions.get(best_code, {}).get("quantity"), 0.0) if best_code else 0.0
    best_t0_remaining_qty = t0_inventory_remaining_qty(state, trade_date, best_code) if best_code else 0
    best_has_non_t0_position = bool(best and best_existing_position_qty > 0 and best_t0_remaining_qty <= 0)
    add("no_existing_non_t0_position_for_entry", not best_has_non_t0_position, {
        "stockCode": best_code,
        "existing_position_qty": best_existing_position_qty,
        "t0_inventory_remaining_qty": best_t0_remaining_qty,
        "rule": "t0_entry_must_not_mingle_with_existing_non_t0_position",
    })
    bracket_cfg = strategy.get("bracket", {})
    reentry_cooldown_mins = as_float(bracket_cfg.get("reentry_cooldown_minutes", 30))
    last_sell_by_code_map = state.get("last_sell_at_by_code") if isinstance(state.get("last_sell_at_by_code"), dict) else {}
    stopped_today_list = []
    if isinstance(state.get("stopped_out_today_by_date"), dict):
        raw_stopped = state["stopped_out_today_by_date"].get(trade_date, [])
        stopped_today_list = raw_stopped if isinstance(raw_stopped, list) else []
    cooldown_ok = True
    cooldown_detail: dict[str, Any] = {"reason": "ok"}
    if best_code:
        last_sell_ts = last_sell_by_code_map.get(best_code)
        mins_since_sell = minutes_since(last_sell_ts)
        if best_code in stopped_today_list and bool(bracket_cfg.get("no_reentry_after_stop_loss_same_day", True)):
            cooldown_ok = False
            cooldown_detail = {"reason": "stopped_out_no_reentry_today", "stockCode": best_code}
        elif mins_since_sell is not None and mins_since_sell < reentry_cooldown_mins:
            cooldown_ok = False
            cooldown_detail = {"reason": "cooldown_active", "minutes_since_sell": round(mins_since_sell, 1), "cooldown_minutes": reentry_cooldown_mins}
        else:
            cooldown_detail = {"reason": "ok", "minutes_since_sell": round(mins_since_sell, 1) if mins_since_sell is not None else None}
    add("reentry_cooldown", cooldown_ok, cooldown_detail)
    max_entries_per_day = int(bracket_cfg.get("max_entries_per_day", 2))
    entries_today = daily_state_bucket(state, "entries_by_date", trade_date)
    # max_entries_per_day <= 0 => UNLIMITED daily entries (no daily trading cap).
    add("daily_entry_limit", max_entries_per_day <= 0 or entries_today < max_entries_per_day,
        {"entries_today": entries_today, "max_entries_per_day": max_entries_per_day, "unlimited": max_entries_per_day <= 0})

    # 统一 sell_score 出场引擎: 遍历所有持仓, 每仓独立评估。
    # 订单提交层再按 max_sell_orders_per_run 节流，避免同一轮把多仓一起卖出。
    sell_orders: list[dict[str, Any]] = []
    deferred_sell_orders: list[dict[str, Any]] = []
    sell_throttle_detail: dict[str, Any] = {"enabled": False}
    sell_score_by_code: dict[str, Any] = {}
    carry_allowed_by_code: dict[str, bool] = {}
    quote_stale = any(not c.get("passed") for c in checks if c.get("name") == "quote_freshness")

    if held_codes:
        for code in held_codes:
            ensure_snapshot_features(quote_by_code.get(code))
            res = evaluate_exit_for_code(
                code,
                strategy=strategy,
                filters=filters,
                risk=risk,
                state=state,
                trade_date=trade_date,
                local_time=local_time,
                quotes=quotes,
                positions=positions,
                sellable_by_code=sellable_by_code,
                lot_size=lot_size,
                kill_switch_active=kill_switch.exists(),
            )
            sell_score_by_code[code] = res.get("sell_score")
            carry_allowed_by_code[code] = bool(res.get("carry_allowed"))
            o = res.get("order")
            if o is not None:
                # 行情陈旧时只放行无条件出场(emergency/kill_switch/异常态), 走兜底价;
                # 普通 sell_score 出场等下一轮新鲜行情, 不在 stale 下盲卖
                if quote_stale and not bool(o.get("unconditional_exit")):
                    continue
                sell_orders.append(o)
        if sell_orders:
            max_sell_orders_per_run = max(1, int(as_float(strategy.get("max_sell_orders_per_run"), 1)))
            sell_throttle_interval_minutes = max(0.0, as_float(strategy.get("sell_throttle_interval_minutes"), 5))
            raw_bypass_reasons = strategy.get("sell_throttle_bypass_reasons")
            if isinstance(raw_bypass_reasons, list):
                sell_throttle_bypass_reasons = {str(x) for x in raw_bypass_reasons}
            else:
                sell_throttle_bypass_reasons = set(DEFAULT_SELL_THROTTLE_BYPASS_REASONS)

            def sell_order_priority(o: dict[str, Any]) -> tuple[float, float, float]:
                reason_rank = {
                    "kill_switch_liquidation": 100.0,
                    "emergency_stop_exit": 95.0,
                    "unrecognized_position_state_liquidation": 90.0,
                    "intraday_momentum_eod_exit": 85.0,
                    "unified_sell_score_exit": 50.0,
                }.get(str(o.get("reason")), 10.0)
                pnl_pct = as_float(o.get("pnl_pct"), 0.0)
                loss_urgency = max(0.0, -pnl_pct) * 1000.0
                return (reason_rank, as_float(o.get("sell_score"), 0.0), loss_urgency)

            ranked_sell_orders = sorted(sell_orders, key=sell_order_priority, reverse=True)
            bypass_sell_orders = [
                o for o in ranked_sell_orders
                if str(o.get("reason")) in sell_throttle_bypass_reasons or bool(o.get("sell_throttle_bypass"))
            ]
            bypass_order_ids = {id(o) for o in bypass_sell_orders}
            regular_sell_orders = [o for o in ranked_sell_orders if id(o) not in bypass_order_ids]
            for o in bypass_sell_orders:
                o["sell_throttle_bypass"] = True
                o["sell_throttle_bypass_reason"] = "emergency_or_forced_exit"
            last_sell_bucket = state.get("last_throttled_sell_at_by_date")
            if not isinstance(last_sell_bucket, dict):
                last_sell_bucket = {}
            last_throttled_sell_at = last_sell_bucket.get(trade_date)
            last_throttled_dt = parse_iso_dt(last_throttled_sell_at)
            minutes_since_throttled_sell = None
            if last_throttled_dt is not None:
                minutes_since_throttled_sell = (local_time - last_throttled_dt).total_seconds() / 60.0
            interval_active = (
                minutes_since_throttled_sell is not None
                and sell_throttle_interval_minutes > 0
                and minutes_since_throttled_sell < sell_throttle_interval_minutes
            )
            if interval_active:
                selected_regular_orders = []
                interval_deferred_orders = regular_sell_orders
            else:
                selected_regular_orders = regular_sell_orders[:max_sell_orders_per_run]
                interval_deferred_orders = regular_sell_orders[max_sell_orders_per_run:]
            for o in selected_regular_orders:
                o["sell_throttle_bypass"] = False
            sell_orders = bypass_sell_orders + selected_regular_orders
            deferred_sell_orders = interval_deferred_orders
            sell_throttle_detail = {
                "enabled": True,
                "max_sell_orders_per_run": max_sell_orders_per_run,
                "sell_throttle_interval_minutes": sell_throttle_interval_minutes,
                "bypass_reasons": sorted(sell_throttle_bypass_reasons),
                "interval_active": interval_active,
                "last_throttled_sell_at": last_throttled_sell_at,
                "minutes_since_last_throttled_sell": round(minutes_since_throttled_sell, 2) if minutes_since_throttled_sell is not None else None,
                "eligible_sell_orders": len(ranked_sell_orders),
                "bypass_sell_orders_count": len(bypass_sell_orders),
                "regular_sell_orders_count": len(regular_sell_orders),
                "selected_sell_orders": [
                    {"stockCode": o.get("stockCode"), "reason": o.get("reason"), "sell_score": o.get("sell_score"), "pnl_pct": o.get("pnl_pct"), "sell_throttle_bypass": o.get("sell_throttle_bypass")}
                    for o in sell_orders
                ],
                "deferred_sell_orders": [
                    {"stockCode": o.get("stockCode"), "reason": o.get("reason"), "sell_score": o.get("sell_score"), "pnl_pct": o.get("pnl_pct"), "deferred_reason": "sell_throttle_interval_active" if interval_active else "max_sell_orders_per_run"}
                    for o in deferred_sell_orders
                ],
                "policy": "emergency_sells_bypass_regular_sells_one_per_interval",
            }
            if sell_orders:
                action = "sell"
                reason = sell_orders[0].get("reason")
            else:
                action = "hold"
                reason = "sell_throttle_interval_active"
    # Entry-or-hold: enter a NEW name only when nothing is selling/deferred this
    # run AND we are under target_holdings AND a non-held candidate exists. This
    # is how concurrent holdings build up (one new name per entry run).
    if (not sell_orders) and (not deferred_sell_orders) and best is not None and len(budget_codes) < target_holdings:
        # --- Intraday-momentum last-half-hour entry (Gao-Han-Li-Zhou 2018) ---
        # Faithful: trades the last half-hour by the first-half-hour return sign.
        # Long-only, at most once/day, reserved path that bypasses the 14:00 cutoff,
        # HK ETFs preferred (concurrent-hours, clean signal). Forced out by exit_time.
        im_cfg = strategy.get("intraday_momentum", {}) if isinstance(strategy.get("intraday_momentum", {}), dict) else {}
        im_done = bool(daily_state_bucket(state, "intraday_momentum_by_date", trade_date))
        im_window = im_in_entry_window(local_time, im_cfg)
        im_reserved_runs = int(as_float(im_cfg.get("reserved_api_runs", 6), 6))
        im_min_ret = as_float(im_cfg.get("min_first_half_return", 0.001))
        if bool(im_cfg.get("enabled")) and im_window and not im_done:
            hk_pref = bool(im_cfg.get("hk_etf_preferred", True))
            im_pool = []
            for q in liquid_quotes:
                if q.get("first_half_hour_return") is None and feature_history_by_code is not None:
                    qcode = str(q.get("stockCode", "")).zfill(6)
                    q["first_half_hour_return"] = compute_first_half_hour_return(
                        feature_history_by_code.get(qcode, []), q, im_cfg
                    )
                fhr = q.get("first_half_hour_return")
                if fhr is None or as_float(fhr) < im_min_ret:
                    continue
                qcode = str(q.get("stockCode", "")).zfill(6)
                existing_qty = as_float(positions.get(qcode, {}).get("quantity"), 0.0)
                if existing_qty > 0:  # never mingle IM with an existing position
                    continue
                blocked_by_sector, _sector = entry_sector_status(q)
                if blocked_by_sector:
                    continue
                im_pool.append(q)
            im_pool.sort(key=lambda x: (
                0 if (hk_pref and x.get("asset_class") == "hk_etf") else 1,
                -as_float(x.get("first_half_hour_return")),
            ))
            best_im = im_pool[0] if im_pool else None
            add("intraday_momentum_window", True, {
                "in_window": True, "candidates": len(im_pool),
                "selected": best_im.get("stockCode") if best_im else None,
                "min_first_half_return": im_min_ret,
            })
            ensure_snapshot_features(best_im)
            ensure_entry_diagnostics(best_im)
            if best_im is not None and not bool((best_im.get("execution_quality") or {}).get("passed")):
                reason = "blocked_execution_quality"
            elif best_im is not None and not bool((best_im.get("safe_policy_shield") or {}).get("passed")):
                reason = "blocked_safe_policy_shield"
            elif best_im is not None and not full_market_entry_ok:
                reason = "blocked_full_market_risk_off"
            elif best_im is not None:
                im_inv_ratio = (len(held_codes) / target_holdings) if target_holdings > 0 else 0.0
                im_mins_to_close = max(0.0, 15 * 60 - (local_time.hour * 60 + local_time.minute))
                px, im_exec_style, im_mid = passive_entry_price(
                    best_im, risk, strategy.get("execution", {}),
                    inventory_ratio=im_inv_ratio, volatility=as_float(best_im.get("atr_pct"), 0.0),
                    minutes_to_close=im_mins_to_close)
                kelly_cfg = strategy.get("kelly_sizing", {})
                realized_pnls = state.get("realized_trade_pnls") if isinstance(state.get("realized_trade_pnls"), list) else []
                kelly_scale = as_float(compute_kelly_scale(realized_pnls, kelly_cfg).get("scale"), 1.0)
                im_position_pct = as_float(im_cfg.get("position_pct", 0.05))
                target_value = min(
                    total_assets * im_position_pct * kelly_scale,
                    total_assets * as_float(risk["max_single_order_pct"]),
                    available_cash * 0.95,
                )
                qty = round_lot(target_value / px, lot_size) if px > 0 else 0
                if qty >= int(risk["min_order_quantity"]) and px > 0:
                    action = "buy"
                    reason = "entry_intraday_momentum"
                    order = {
                        "direction": "buy",
                        "stockCode": best_im["stockCode"],
                        "exchange": best_im["exchange"],
                        "name": best_im["name"],
                        "quantity": qty,
                        "orderType": risk["order_type"],
                        "price": round(px, 3),
                        "reason": reason,
                        "signal_source": "intraday_momentum",
                        "first_half_hour_return": as_float(best_im.get("first_half_hour_return")),
                        "im_trade": True,
                        "im_exit_time": str(im_cfg.get("exit_time", "14:55")),
                        "execution_quality": best_im.get("execution_quality"),
                        "safe_policy_shield": best_im.get("safe_policy_shield"),
                        "kelly_scale": kelly_scale,
                        "baseline_available_quantity": as_float(positions.get(str(best_im["stockCode"]).zfill(6), {}).get("availableQuantity"), 0.0),
                        "inventory_scope": "t0_intraday_inventory_only",
                        "t0_eligible": True,
                        "execution_style": im_exec_style,
                        "submission_mid": round(im_mid, 4),
                    }
        orb = get_orb(state, trade_date, best.get("stockCode") if best else None)
        best_price = as_float(best.get("currentPrice")) if best else 0.0
        orb_breakout = bool(
            orb is not None and
            best_price > as_float(orb.get("high")) * 1.001 and
            as_float(best.get("momentum")) > 0 and
            as_float(best.get("bid_pressure_3m_pct") or 0) > 0
        ) if best else False
        cons_cfg = strategy.get("consolidation", {})
        earliest_entry = str(cons_cfg.get("earliest_entry_time", "10:00"))
        try:
            cons_h, cons_m = (int(x) for x in earliest_entry.split(":"))
        except Exception:
            cons_h, cons_m = 10, 0
        consolidation_entry_allowed = (local_time.hour, local_time.minute) >= (cons_h, cons_m)
        cons_box = best.get("consolidation_box") if best and isinstance(best.get("consolidation_box"), dict) else None
        cons_buffer = as_float(cons_cfg.get("breakout_buffer_pct", 0.001))
        cons_breakout = bool(
            cons_cfg.get("enabled") and cons_box and consolidation_entry_allowed and
            best_price > as_float(cons_box.get("high")) * (1.0 + cons_buffer) and
            as_float(best.get("momentum")) > 0
        ) if best else False
        bollinger_breakout = bool(best and best.get("bollinger_squeeze_breakout") and as_float(best.get("momentum")) > 0)
        if best and orb_breakout:
            signal_source = "orb_breakout"
        elif cons_breakout:
            signal_source = "consolidation_breakout"
        elif bollinger_breakout:
            signal_source = "bollinger_squeeze_breakout"
        else:
            signal_source = "momentum_fallback"
        entry_score = score_entry(
            strategy=strategy,
            filters=filters,
            q=best,
            orb=orb,
            broad_market_not_declining=broad_market_not_declining,
            consolidation_entry_allowed=consolidation_entry_allowed,
        ) if best else {"score": 0.0, "threshold": as_float(strategy.get("entry_score_threshold"), 50), "components": {}, "details": {}}
        entry_score_passed = as_float(entry_score.get("score"), 0.0) >= as_float(entry_score.get("threshold"), 50)
        add("entry_score_gate", entry_score_passed, entry_score)
        skip_dates: list[str] = strategy.get("skip_dates") or []
        today_skipped = trade_date in skip_dates
        add("skip_date_guard", not today_skipped, {"trade_date": trade_date, "skip_dates": skip_dates, "skipped": today_skipped})
        gap_guard_pct = as_float(strategy.get("overnight_gap_guard_pct"), 0.0)
        gap_reduce_factor = 1.0
        gap_detail: dict[str, Any] = {"overnight_gap_guard_pct": gap_guard_pct, "gap_reduce_factor": 1.0}
        if gap_guard_pct > 0 and best:
            best_change_abs = abs(as_float(best.get("change_pct"), 0.0))
            if best_change_abs > gap_guard_pct:
                gap_reduce_factor = 0.5
            gap_detail = {"overnight_gap_guard_pct": gap_guard_pct, "best_change_abs": best_change_abs, "gap_reduce_factor": gap_reduce_factor}
        current_entry_checks_passed = all(c.get("passed") for c in checks)
        if order is not None:
            pass  # intraday-momentum order already built above; skip normal entry chain
        elif best and open_quiet_passed and no_new_entry_after_cutoff and current_entry_checks_passed and not today_skipped and entry_score_passed:
            inv_ratio = (len(held_codes) / target_holdings) if target_holdings > 0 else 0.0
            mins_to_close = max(0.0, 15 * 60 - (local_time.hour * 60 + local_time.minute))
            px, entry_exec_style, entry_mid = passive_entry_price(
                best, risk, strategy.get("execution", {}),
                inventory_ratio=inv_ratio, volatility=as_float(best.get("atr_pct"), 0.0),
                minutes_to_close=mins_to_close)
            bracket_meta: dict[str, Any] | None = None
            bracket_skip_reason: str | None = None
            orb_for_best = get_orb(state, trade_date, best.get("stockCode"))
            # 止损价来源优先级: ORB突破→ORB中点; 盘整突破→箱体下沿; 其余信号但有ORB→ORB中点
            stop_px = 0.0
            stop_source = None
            range_pct_entry = 0.0
            if orb_for_best and orb_breakout:
                stop_px = orb_for_best["midpoint"]
                stop_source = "orb_midpoint"
                range_pct_entry = (orb_for_best["high"] - orb_for_best["low"]) / orb_for_best["midpoint"]
            elif cons_breakout and cons_box:
                stop_px = as_float(cons_box.get("low"))
                stop_source = "consolidation_box_low"
                range_pct_entry = as_float(cons_box.get("range_pct"))
            elif orb_for_best:
                stop_px = orb_for_best["midpoint"]
                stop_source = "orb_midpoint"
                range_pct_entry = (orb_for_best["high"] - orb_for_best["low"]) / orb_for_best["midpoint"]
            if bracket_cfg.get("enabled") and stop_px > 0 and px > 0:
                R_val = px - stop_px
                min_orb_pct = as_float(bracket_cfg.get("min_orb_range_pct", 0.003))
                chaos_orb_pct = as_float(bracket_cfg.get("chaos_orb_range_pct", 0.015))
                atr_stop_distance_pct = as_float(best.get("atr_stop_distance_pct"), 0.0) if best else 0.0
                atr_available = bool(best.get("atr_available")) if best else False
                if stop_source == "orb_midpoint" and range_pct_entry < min_orb_pct:
                    bracket_skip_reason = f"orb_range_too_narrow:{range_pct_entry:.4f}<{min_orb_pct}"
                elif stop_source == "consolidation_box_low" and R_val < px * 0.002:
                    bracket_skip_reason = f"consolidation_r_too_small:{R_val:.5f}"
                elif atr_available and atr_stop_distance_pct > 0 and R_val < px * atr_stop_distance_pct:
                    bracket_skip_reason = f"atr_r_too_small:{R_val:.5f}<{px * atr_stop_distance_pct:.5f}"
                elif R_val < px * 0.001:
                    bracket_skip_reason = f"invalid_r_distance:{R_val:.5f}"
                else:
                    is_chaos = stop_source == "orb_midpoint" and range_pct_entry > chaos_orb_pct
                    risk_pct_key = "risk_per_trade_pct_chaos_day" if is_chaos else "risk_per_trade_pct"
                    # Fractional-Kelly + drawdown scaling: only ever reduces risk, never inflates.
                    kelly_cfg = strategy.get("kelly_sizing", {})
                    realized_pnls = state.get("realized_trade_pnls") if isinstance(state.get("realized_trade_pnls"), list) else []
                    kelly_detail = compute_kelly_scale(realized_pnls, kelly_cfg)
                    kelly_scale = as_float(kelly_detail.get("scale"), 1.0)
                    dd_factor = 1.0
                    today_pnl = daily_state_float_bucket(state, "today_realized_pnl", trade_date)
                    if bool(kelly_cfg.get("enabled", True)) and today_pnl < 0:
                        dd_factor = as_float(kelly_cfg.get("drawdown_factor", 0.5))
                    risk_pct_val = as_float(bracket_cfg.get(risk_pct_key, 0.004)) * gap_reduce_factor * kelly_scale * dd_factor
                    risk_budget = total_assets * risk_pct_val
                    qty = round_lot(min(
                        risk_budget / R_val,
                        total_assets * as_float(strategy["max_position_pct"]) / px,
                        total_assets * as_float(risk["max_single_order_pct"]) / px,
                        available_cash * 0.95 / px,
                    ), lot_size)
                    t1_mult = as_float(bracket_cfg.get("target1_r_multiple", 1.0))
                    t2_mult = as_float(bracket_cfg.get("target2_r_multiple", 2.0))
                    bracket_meta = {
                        "entry_price": px,
                        "stop_price": round(stop_px, 3),
                        "stop_source": stop_source,
                        "r_value": round(R_val, 4),
                        "target1_price": round(px + t1_mult * R_val, 3),
                        "target2_price": round(px + t2_mult * R_val, 3),
                        "risk_budget": round(risk_budget, 2),
                        "risk_pct": risk_pct_val,
                        "range_pct": round(range_pct_entry, 5),
                        "atr_available": atr_available,
                        "atr_pct": best.get("atr_pct") if best else None,
                        "atr_stop_distance_pct": atr_stop_distance_pct if atr_available else None,
                        "chaos_day": is_chaos,
                        "kelly_scale": kelly_scale,
                        "kelly_detail": kelly_detail,
                        "drawdown_factor": dd_factor,
                    }
            else:
                target_value = min(
                    total_assets * as_float(strategy["max_position_pct"]),
                    total_assets * as_float(risk["max_single_order_pct"]),
                    available_cash * 0.95,
                )
                qty = round_lot(target_value / px, lot_size)
            if bracket_skip_reason:
                reason = bracket_skip_reason
            elif qty >= int(risk["min_order_quantity"]):
                action = "buy"
                if orb_breakout:
                    reason = "entry_orb_breakout_passed"
                elif cons_breakout:
                    reason = "entry_consolidation_breakout_passed"
                elif bollinger_breakout:
                    reason = "entry_bollinger_squeeze_breakout_passed"
                else:
                    reason = "entry_momentum_spread_passed"
                order = {
                    "direction": "buy",
                    "stockCode": best["stockCode"],
                    "exchange": best["exchange"],
                    "name": best["name"],
                    "quantity": qty,
                    "orderType": risk["order_type"],
                    "price": round(px, 3),
                    "reason": reason,
                    "momentum": best.get("momentum"),
                    "spread_pct": best.get("spread_pct"),
                    "signal_source": signal_source,
                    "orb_high": orb["high"] if orb else None,
                    "consolidation_box": cons_box,
                    "rolling_vwap": best.get("rolling_vwap"),
                    "price_above_vwap": best.get("price_above_vwap"),
                    "vwap_distance_pct": best.get("vwap_distance_pct"),
                    "atr_pct": best.get("atr_pct"),
                    "atr_stop_distance_pct": best.get("atr_stop_distance_pct"),
                    "bollinger_squeeze": best.get("bollinger_squeeze"),
                    "bid_pressure_3m_pct": best.get("bid_pressure_3m_pct"),
                    "acceleration": best.get("acceleration"),
                    "entry_score": entry_score,
                    "execution_quality": best.get("execution_quality"),
                    "safe_policy_shield": best.get("safe_policy_shield"),
                    "baseline_available_quantity": as_float(positions.get(str(best["stockCode"]).zfill(6), {}).get("availableQuantity"), 0.0),
                    "inventory_scope": "t0_intraday_inventory_only",
                    "t0_eligible": True,
                    "bracket": bracket_meta,
                    "execution_style": entry_exec_style,
                    "submission_mid": round(entry_mid, 4),
                }
        elif best and not open_quiet_passed:
            reason = "blocked_open_quiet_period"
        elif best and not no_new_entry_after_cutoff:
            reason = "blocked_no_new_entry_after_14_30"
        elif best and not broad_market_not_declining:
            reason = "blocked_broad_market_declining"
        elif best and not correlation_stress_ok:
            reason = "blocked_market_correlation_stress"
        elif best and not full_market_entry_ok:
            reason = "blocked_full_market_risk_off"
        elif not best and t0_entry_blocked_candidates and len(budget_codes) < target_holdings:
            reason = "blocked_t0_entry_ineligible_asset_class"
        elif not best and sector_enabled and sector_blocked_candidates and len(budget_codes) < target_holdings:
            reason = "blocked_sector_concentration"
        elif best and not bool((best.get("execution_quality") or {}).get("passed")):
            reason = "blocked_execution_quality"
        elif best and not bool((best.get("safe_policy_shield") or {}).get("passed")):
            reason = "blocked_safe_policy_shield"
        else:
            reason = "no_entry_signal"
    elif action != "sell" and not deferred_sell_orders:
        # Not entering and nothing pending to sell: explain why we hold. (When a
        # sell is throttle-deferred we keep the throttle reason set above.)
        action = "hold"
        reason = (
            "at_target_holdings_capacity" if len(budget_codes) >= target_holdings
            else "blocked_t0_entry_ineligible_asset_class" if t0_entry_blocked_candidates
            else "blocked_sector_concentration" if sector_enabled and sector_blocked_candidates
            else "carry_allowed_sell_score_not_met" if held_codes
            else "no_entry_candidate"
        )

    # A decision is EITHER one BUY (when adding) OR throttled SELLs (one+ per run).
    orders_list: list[dict[str, Any]] = sell_orders if sell_orders else ([order] if order else [])
    is_sell_batch = bool(orders_list) and all(o.get("direction") == "sell" for o in orders_list)
    all_unconditional = is_sell_batch and all(bool(o.get("unconditional_exit")) for o in orders_list)

    if orders_list:
        add("order_built", True, orders_list[0] if len(orders_list) == 1 else {"orders_count": len(orders_list), "orders": orders_list})
    else:
        add("order_built", False, reason)

    is_im_buy = bool(orders_list) and len(orders_list) == 1 and orders_list[0].get("im_trade") and orders_list[0].get("direction") == "buy"

    def check_applies(c: dict[str, Any]) -> bool:
        if is_sell_batch:
            name = c.get("name")
            if name in SELL_BYPASS_CHECKS:
                return True
            # 行情陈旧时: 整批均为无条件出场才放行(已在上游过滤掉非无条件单)
            if name == "quote_freshness" and all_unconditional:
                return True
        # Intraday-momentum buy has its OWN signal (first-half-hour return), so it
        # bypasses the normal-momentum entry gates and the afternoon cutoff (by design).
        # Genuine risk-off guards still apply: broad_market_not_declining,
        # market_correlation_stress, skip_date, daily limits, session, kill switch,
        # data integrity, quote_freshness, no_pending, liquidity, no-mingling.
        if is_im_buy and c.get("name") in IM_BUY_BYPASS_CHECKS:
            return True
        return bool(c.get("passed"))

    approved = all(check_applies(c) for c in checks) and bool(orders_list)
    return {
        "timestamp": now_iso(),
        "agent_name": cfg["agent_name"],
        "mode": cfg["mode"],
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "session": session,
        "system_time": {
            "system_timezone": "Asia/Seoul",
            "system_local_time": system_local_time().isoformat(),
            "exchange_timezone": "Asia/Shanghai",
            "exchange_local_time": local_time.isoformat(),
            "time_guard_basis": "exchange_local_time",
        },
        "trade_date": trade_date,
        "quotes": quotes,
        "ranked": ranked,
        "positions_t0": positions,
        "t0_inventory": state.get("t0_inventory_by_date", {}).get(trade_date, {}),
        "t0_sellable_by_code": sellable_by_code,
        "pending_t0_orders": pending_t0,
        "volume_filter_status": "available" if volume_fields_available else "unavailable_api_field",
        "state_machine": {
            "state": "pending_order" if pending_t0 else ("holding" if held_code else "flat"),
            "action": action,
            "reason": reason,
        },
        "risk_checks": checks,
        "market_correlation_stress": market_correlation_stress,
        "sector_diversification": sector_diversification_detail,
        "approved_for_submit": bool(approved),
        "orders": orders_list,
        "exit_policy": "unified_sell_score",
        "held_codes": held_codes,
        "sell_score_by_code": sell_score_by_code,
        "carry_allowed_by_code": carry_allowed_by_code,
        "sell_throttle": sell_throttle_detail,
        "deferred_sell_orders": deferred_sell_orders,
        "carry_allowed": (all(carry_allowed_by_code.values()) if carry_allowed_by_code else None),
    }


def maybe_cancel_pending(cfg: dict[str, Any], client: SkillClient, state: dict[str, Any], pending_count: int, decision: dict[str, Any]) -> dict[str, Any]:
    risk = cfg["risk"]
    if pending_count <= 0 or not bool(risk.get("auto_cancel_pending", True)):
        return {"attempted": False, "reason": "no_pending_or_disabled"}
    trade_date = decision["trade_date"]
    cancels_today = daily_state_bucket(state, "cancels_by_date", trade_date)
    since_last = minutes_since(state.get("last_cancel_at"))
    if cancels_today >= int(risk["max_daily_cancels"]):
        return {"attempted": False, "reason": "daily_cancel_limit_reached", "cancels_today": cancels_today}
    if since_last is not None and since_last < int(risk["min_minutes_between_cancels"]):
        return {"attempted": False, "reason": "cancel_interval_active", "minutes_since_last_cancel": since_last}
    resp = client.cancel_all_pending()
    if resp.get("ok"):
        increment_daily_state(state, "cancels_by_date", trade_date)
        state["last_cancel_at"] = now_iso()
    return {"attempted": True, "ok": resp.get("ok"), "error": resp.get("error")}


def run_agent(config_path: Path, execute: bool = False) -> dict[str, Any]:
    cfg = load_json(config_path)
    cfg = apply_evolution_overlay_if_enabled(cfg)
    out_dir = ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    client = SkillClient(python_cmd_from_config(cfg["skill"].get("python", sys.executable)), expand_path(cfg["skill"]["script"]))
    state_path = out_dir / cfg["outputs"]["state"]
    state = load_state(state_path)
    state.setdefault("ORB", {})
    trade_date = trade_date_cn()
    session = cn_market_session()
    if cfg.get("trading_hours_only", True) and not session.get("in_regular_session"):
        save_state(state_path, state)
        return t0_market_closed_no_api_result(cfg, execute, out_dir, session)

    # Widen the tradable universe data-driven from the full-market snapshot (top-N
    # by cross-sectional liquidity+momentum) UNION held codes (never orphan an exit).
    # Falls back to the static config universe if selection is unavailable. This only
    # changes WHICH names are considered; all safety/risk gates downstream are intact.
    universe_meta: dict[str, Any] = {"mode": "static"}
    if cfg.get("dynamic_universe", {}).get("enabled"):
        resolved_universe, universe_meta = resolve_agent_universe(cfg, state, trade_date)
        cfg["universe"] = resolved_universe
        state["last_universe_selection"] = universe_meta

    quota_path = quota_state_path(out_dir, cfg)
    quota_status = quota_backoff_status(quota_path, trade_date)
    if quota_status.get("active"):
        quote_responses, market_data_report = fetch_quote_responses(cfg, client)
        quotes = [
            normalize_quote(etf, resp)
            for etf, resp in zip(cfg["universe"], quote_responses)
        ]
        minute_path = dated_output_path(out_dir, cfg["outputs"]["minute_quotes_jsonl"], trade_date)
        minute_csv_path = dated_output_path(out_dir, cfg["outputs"]["minute_quotes_csv"], trade_date)
        history = load_recent_quotes(minute_path, universe_size=len(cfg["universe"]))
        quotes = compute_snapshot_momentum(quotes, history, int(cfg["strategy"]["lookback_minutes"]), cfg["strategy"])
        for q in quotes:
            q["liquidity_source"] = "point_in_time" if universe_meta.get("mode") == "dynamic" else "unknown"
            append_jsonl(minute_path, q)
        append_csv(minute_csv_path, quotes)
        local_time = exchange_local_time(session)
        update_orb_state(state, trade_date, quotes, local_time, bool(session.get("in_regular_session")))
        save_state(state_path, state)
        out = t0_quota_backoff_result(
            cfg,
            execute,
            out_dir,
            "quota_exhausted_backoff_active",
            quota_status,
            quotes,
        )
        out["market_data_report"] = market_data_report
        return out

    max_api_runs = int(cfg.get("risk", {}).get("max_daily_api_runs", 80))
    api_runs_today = daily_state_bucket(state, "api_runs_by_date", trade_date)
    if api_runs_today >= max_api_runs:
        save_state(state_path, state)
        return t0_api_budget_result(cfg, execute, out_dir, trade_date, api_runs_today, max_api_runs)
    increment_daily_state(state, "api_runs_by_date", trade_date)
    save_state(state_path, state)

    quote_responses, market_data_report = fetch_quote_responses(cfg, client)
    quotes = [
        normalize_quote(etf, resp)
        for etf, resp in zip(cfg["universe"], quote_responses)
    ]

    minute_path = dated_output_path(out_dir, cfg["outputs"]["minute_quotes_jsonl"], trade_date)
    minute_csv_path = dated_output_path(out_dir, cfg["outputs"]["minute_quotes_csv"], trade_date)
    history = load_recent_quotes(minute_path, universe_size=len(cfg["universe"]))
    quotes = compute_snapshot_momentum(quotes, history, int(cfg["strategy"]["lookback_minutes"]), cfg["strategy"])
    market_correlation_stress = compute_market_correlation_stress(
        history,
        quotes,
        cfg.get("strategy", {}).get("market_correlation_stress", {}),
    )
    for q in quotes:
        q["liquidity_source"] = "point_in_time" if universe_meta.get("mode") == "dynamic" else "unknown"
        append_jsonl(minute_path, q)
    append_csv(minute_csv_path, quotes)
    if any_quota_exhausted(quote_responses):
        quota_resp = next((x for x in quote_responses if is_quota_exhausted_response(x)), None)
        quota_status = mark_quota_exhausted(quota_path, trade_date, "t0_quote_query", quota_resp)
        save_state(state_path, state)
        return t0_quota_backoff_result(
            cfg,
            execute,
            out_dir,
            "quota_exhausted_detected_quote_query",
            quota_status,
            quotes,
        )

    local_time = exchange_local_time(session)
    update_orb_state(state, trade_date, quotes, local_time, bool(session.get("in_regular_session")))

    md_cfg = cfg.get("market_data", {}) if isinstance(cfg.get("market_data"), dict) else {}
    broker_query_interval = max(0.0, as_float(md_cfg.get("huatai_broker_query_min_interval_seconds"), 0.0))
    last_broker_query_at = state.get("last_huatai_broker_query_at")
    last_broker_dt = parse_iso_dt(last_broker_query_at)
    seconds_since_broker_query = None
    if last_broker_dt is not None:
        seconds_since_broker_query = (local_time - last_broker_dt).total_seconds()
    broker_query_throttled = (
        broker_query_interval > 0
        and seconds_since_broker_query is not None
        and seconds_since_broker_query < broker_query_interval
    )
    broker_query_throttle_detail = {
        "enabled": broker_query_interval > 0,
        "min_interval_seconds": broker_query_interval,
        "last_huatai_broker_query_at": last_broker_query_at,
        "seconds_since_last_query": round(seconds_since_broker_query, 2) if seconds_since_broker_query is not None else None,
        "market_data_provider": market_data_report.get("provider"),
        "policy": "eastmoney_quotes_every_run_huatai_broker_query_interval",
    }
    if broker_query_throttled:
        save_state(state_path, state)
        return t0_quote_only_broker_query_throttled_result(
            cfg,
            execute,
            out_dir,
            trade_date,
            session,
            quotes,
            market_data_report,
            {**broker_query_throttle_detail, "throttled": True},
        )

    balance = client.get_balance()
    positions = client.get_positions()
    pending = call_pending_orders(client)
    fill_reconciliation = maybe_reconcile_t0_fills(cfg, client, state, trade_date)
    state["last_huatai_broker_query_at"] = local_time.isoformat()
    api_responses = [balance, positions, pending]
    if isinstance(fill_reconciliation, dict) and fill_reconciliation.get("ok") is False:
        api_responses.append({"error": fill_reconciliation.get("error")})
    if any_quota_exhausted(api_responses):
        quota_resp = next((x for x in api_responses if is_quota_exhausted_response(x)), None)
        quota_status = mark_quota_exhausted(quota_path, trade_date, "t0_account_position_pending_or_fill_query", quota_resp)
        save_state(state_path, state)
        return t0_quota_backoff_result(
            cfg,
            execute,
            out_dir,
            "quota_exhausted_detected_account_position_pending_or_fill_query",
            quota_status,
            quotes,
        )
    decision = build_decision(cfg, quotes, balance, positions, pending, state, market_correlation_stress, history=history)
    decision["market_data_report"] = market_data_report
    decision["broker_query_throttle"] = {**broker_query_throttle_detail, "throttled": False}
    decision["fill_reconciliation"] = fill_reconciliation
    decision["evolution_overlay"] = cfg.get("_evolution_overlay", {"applied": False})
    # Decision Scoring System hook: RECORD-ONLY. Wrapped so it can NEVER affect the
    # decision, the orders, or the live run; gated by config (default off). It only
    # appends an auditable decision_score to outputs/decision_scores/.
    if cfg.get("decision_scoring", {}).get("enabled", False):
        try:
            import decision_scoring as _ds
            _ts = exchange_local_time(session).strftime("%H:%M:%S")
            _contexts = _ds.contexts_from_decision(
                cfg, decision, trade_date=trade_date, timestamp=_ts,
            )
            _ds.append_scores([_ds.score_decision(_ctx) for _ctx in _contexts], trade_date)
        except Exception:
            pass  # never let scoring touch the live agent
    agent_name = str(cfg.get("agent_name", "t0_intraday_paper_agent"))
    tag_order_owner(decision.get("orders", []) if isinstance(decision.get("orders"), list) else [], agent_name, decision.get("trade_date", trade_date))
    submit_results: list[dict[str, Any]] = []
    cancel_result = {"attempted": False, "reason": "not_needed"}

    with SharedExecutionGuard(
        cfg,
        agent_name=agent_name,
        trade_date=decision.get("trade_date", trade_date),
        orders=decision.get("orders", []) if isinstance(decision.get("orders"), list) else [],
        can_execute=bool(execute and decision.get("approved_for_submit")),
    ) as shared_guard:
        decision["shared_execution_guard"] = shared_guard.report
        if execute and decision.get("approved_for_submit") and not shared_guard.allowed:
            decision["status"] = "blocked_shared_execution_guard"
        elif execute and decision.get("approved_for_submit"):
            cancel_result = maybe_cancel_pending(cfg, client, state, len(decision.get("pending_t0_orders", [])), decision)
            if cancel_result.get("attempted") and not cancel_result.get("ok"):
                decision["status"] = "blocked_cancel_failed"
            else:
                # Quota-efficient pre-SELL verification: confirm the broker actually
                # holds the position before submitting a sell, so we never queue a
                # stuck pending order against phantom local inventory (today's 513100).
                # One getPositions, ONLY on runs that submit a sell.
                risk_cfg = cfg["risk"]
                if any(o.get("direction") == "sell" for o in decision["orders"]):
                    pos_resp = client.get_positions()
                    decision["pre_sell_position_check"] = {"attempted": True, "ok": bool(pos_resp.get("ok"))}
                    broker_avail = {
                        str(p.get("stockCode", "")).zfill(6): as_float(p.get("availableQuantity"), 0.0)
                        for p in (extract_data(pos_resp).get("positions") or [])
                    } if pos_resp.get("ok") else {}
                    decision["orders"] = verify_sell_orders_against_broker(
                        decision["orders"], broker_avail, bool(pos_resp.get("ok")),
                        int(risk_cfg["min_order_quantity"]), int(risk_cfg["quantity_lot"]),
                    )
                for order in decision["orders"]:
                    submit_results.append(client.submit_order(order["direction"], order["stockCode"], order["exchange"], int(order["quantity"]), order["orderType"], order.get("price")))
                decision["status"] = "submitted" if submit_results else ("no_submit_after_position_check" if not decision["orders"] else "planned_no_submit")
                if submit_results:
                    increment_daily_state(state, "submitted_orders_by_date", decision["trade_date"])
                    state["last_submit_at"] = now_iso()
                    for order, submit in zip(decision["orders"], submit_results):
                        if not submit.get("ok"):
                            continue
                        if order.get("direction") == "buy":
                            record_t0_buy_submission(state, decision["trade_date"], order, submit)
                            bm = order.get("bracket")
                            if bm:
                                bcode = str(order.get("stockCode", "")).zfill(6)
                                inv_node = t0_inventory_for_code(state, decision["trade_date"], bcode)
                                inv_node.setdefault("r_value", bm.get("r_value"))
                                inv_node.setdefault("stop_price", bm.get("stop_price"))
                                inv_node.setdefault("target1_price", bm.get("target1_price"))
                                inv_node.setdefault("target2_price", bm.get("target2_price"))
                                inv_node.setdefault("t1_filled", False)
                                inv_node.setdefault("stop_moved_to_breakeven", False)
                            if order.get("im_trade"):
                                # tag inventory so the held branch force-exits it by exit_time,
                                # and mark IM done so it fires at most once per day
                                bcode = str(order.get("stockCode", "")).zfill(6)
                                inv_node = t0_inventory_for_code(state, decision["trade_date"], bcode)
                                inv_node["im_trade"] = True
                                increment_daily_state(state, "intraday_momentum_by_date", decision["trade_date"])
                            increment_daily_state(state, "entries_by_date", decision["trade_date"])
                        elif order.get("direction") == "sell":
                            record_t0_sell_submission(state, decision["trade_date"], order, submit)
                            increment_daily_state(state, "sell_orders_by_date", decision["trade_date"])
                            # Realized PnL is booked ONLY on confirmed fills (see
                            # reconcile_t0_inventory_from_trade_history). A submitted
                            # limit order is NOT a fill -- booking it here produced
                            # phantom PnL when the order did not execute. Store the
                            # cost basis so fill reconciliation can compute true PnL.
                            sell_code = str(order.get("stockCode", "")).zfill(6)
                            _inv = t0_inventory_for_code(state, decision["trade_date"], sell_code)
                            _inv["realization_cost_price"] = as_float(order.get("cost_price"))
                            est = (as_float(order.get("price")) - as_float(order.get("cost_price"))) * int(as_float(order.get("quantity")))
                            add_daily_state_float(state, "today_estimated_pnl_submitted", decision["trade_date"], est)
                            if order.get("reason") == "bracket_t1_partial_exit":
                                inv_node = t0_inventory_for_code(state, decision["trade_date"], sell_code)
                                inv_node["t1_filled"] = True
                                inv_node["stop_moved_to_breakeven"] = True
                            last_sell_state = state.get("last_sell_at_by_code")
                            if not isinstance(last_sell_state, dict):
                                last_sell_state = {}
                            last_sell_state[sell_code] = now_iso()
                            state["last_sell_at_by_code"] = last_sell_state
                            if not bool(order.get("sell_throttle_bypass")):
                                last_throttled_bucket = state.get("last_throttled_sell_at_by_date")
                                if not isinstance(last_throttled_bucket, dict):
                                    last_throttled_bucket = {}
                                last_throttled_bucket[decision["trade_date"]] = now_iso()
                                state["last_throttled_sell_at_by_date"] = last_throttled_bucket
                            stop_reasons = {"confirmed_loss_score_exit", "emergency_stop_exit", "bracket_stop_loss", "breakeven_stop_after_t1"}
                            reason_is_stop = order.get("reason") in stop_reasons or (
                                order.get("reason") == "unified_sell_score_exit" and as_float(order.get("pnl_pct"), 0.0) < 0
                            )
                            bracket_cfg_run = cfg.get("strategy", {}).get("bracket", {})
                            if bool(bracket_cfg_run.get("no_reentry_after_stop_loss_same_day", True)) and reason_is_stop:
                                stopped_map = state.get("stopped_out_today_by_date")
                                if not isinstance(stopped_map, dict):
                                    stopped_map = {}
                                today_stopped = stopped_map.get(decision["trade_date"], [])
                                if not isinstance(today_stopped, list):
                                    today_stopped = []
                                if sell_code not in today_stopped:
                                    today_stopped.append(sell_code)
                                stopped_map[decision["trade_date"]] = today_stopped
                                state["stopped_out_today_by_date"] = stopped_map
            # After a BUY fills the order book, refresh the broker balance so the
            # same-day available cash is known (quota-efficient: only after a buy).
            if any(o.get("direction") == "buy" for o, s in zip(decision["orders"], submit_results) if s.get("ok")):
                bal = client.get_balance()
                if bal.get("ok"):
                    ta, av = account_assets(bal)
                    state["last_known_balance"] = {"total_assets": ta, "available_cash": av, "at": now_iso()}
                    decision["post_buy_balance"] = state["last_known_balance"]
            shared_guard.record_submit_results(submit_results, status=decision.get("status", "unknown"), cancel_result=cancel_result)
        else:
            decision["status"] = "planned" if decision.get("orders") else "observe"

    decision["cli_execute"] = bool(execute)
    decision["cancel_result"] = cancel_result
    decision["submit_results"] = submit_results
    save_state(state_path, state)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="T+0 ETF intraday paper agent")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    cfg = load_json(Path(args.config))
    out_dir = ROOT / cfg["outputs"]["dir"]
    latest = out_dir / cfg["outputs"]["latest"]
    runs = out_dir / cfg["outputs"]["jsonl"]
    orders_csv = out_dir / cfg["outputs"]["orders"]
    blotter = out_dir / cfg["outputs"]["blotter"]

    result = run_agent(Path(args.config), args.execute)
    write_json(latest, result)
    append_jsonl(runs, result)
    write_csv(orders_csv, result.get("orders", []))
    append_order_blotter(blotter, build_blotter_rows(result, result.get("submit_results", [])))
    print(json.dumps({
        "status": result.get("status"),
        "state": result.get("state_machine", {}).get("state"),
        "action": result.get("state_machine", {}).get("action"),
        "approved_for_submit": result.get("approved_for_submit"),
        "orders_count": len(result.get("orders", [])),
        "volume_filter_status": result.get("volume_filter_status"),
        "system_local_time": result.get("system_time", {}).get("system_local_time"),
        "exchange_local_time": result.get("system_time", {}).get("exchange_local_time"),
        "latest": str(latest),
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
