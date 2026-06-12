"""Status view for the ETF-only paper trading agent.

This script never submits orders. It runs the existing agent in dry-run mode
or reads the latest decision file, then prints a compact status summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import (
    DEFAULT_CONFIG,
    ROOT,
    append_order_blotter,
    load_json,
    run_agent,
    write_json,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _money(value: Any) -> str:
    return f"{_as_float(value):,.2f}"


def _pct(value: Any) -> str:
    return f"{_as_float(value) * 100:.2f}%"


def _extract_balance(result: dict[str, Any]) -> dict[str, Any]:
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    return {
        "total_assets": plan.get("total_assets"),
        "available_cash": plan.get("available_cash"),
    }


def _risk_failed_checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    report = result.get("risk_report") if isinstance(result.get("risk_report"), dict) else {}
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return []
    return [c for c in checks if isinstance(c, dict) and not c.get("passed")]


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    quotes = result.get("quotes", [])
    if not isinstance(quotes, list):
        quotes = []
    orders = result.get("orders", [])
    if not isinstance(orders, list):
        orders = []
    failed = _risk_failed_checks(result)
    balance = _extract_balance(result)
    grid_report = result.get("grid_regime_report")
    if not isinstance(grid_report, dict):
        grid_report = {}
    grid_items = grid_report.get("items", [])
    if not isinstance(grid_items, list):
        grid_items = []
    cooldown = result.get("execution_cooldown")
    if not isinstance(cooldown, dict):
        cooldown = {}

    ranked = sorted(
        [q for q in quotes if isinstance(q, dict)],
        key=lambda q: _as_float(q.get("score"), -999.0),
        reverse=True,
    )

    return {
        "timestamp": result.get("timestamp"),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "mode": result.get("mode"),
        "asset_type": result.get("asset_type", "ETF_ONLY"),
        "total_assets": balance["total_assets"],
        "available_cash": balance["available_cash"],
        "quote_count": len(quotes),
        "planned_orders": len(orders),
        "cash_defense_active": bool(result.get("cash_defense_active", False)),
        "cash_defense_reason": result.get("cash_defense_reason"),
        "cash_defense_threshold": result.get("cash_defense_threshold"),
        "max_score_observed": result.get("max_score_observed"),
        "risk_failed_checks": [c.get("name") for c in failed],
        "grid_regime_summary": grid_report.get("summary", {}),
        "grid_regime_top": grid_items[:5],
        "execution_cooldown": cooldown,
        "top_ranked": ranked[:5],
        "orders": orders,
        "live_ready": False,
        "formal_strategy_allowed": False,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("ETF Paper Agent Status")
    print("=" * 24)
    print(f"timestamp: {summary.get('timestamp')}")
    print(f"status: {summary.get('status')}")
    if summary.get("reason"):
        print(f"reason: {summary.get('reason')}")
    print(f"mode: {summary.get('mode')}")
    print(f"asset_type: {summary.get('asset_type')}")
    if summary.get("total_assets") is not None:
        print(f"total_assets: {_money(summary.get('total_assets'))}")
    if summary.get("available_cash") is not None:
        print(f"available_cash: {_money(summary.get('available_cash'))}")
    print(f"quote_count: {summary.get('quote_count')}")
    print(f"planned_orders: {summary.get('planned_orders')}")

    failed = summary.get("risk_failed_checks") or []
    print(f"risk_failed_checks: {', '.join(failed) if failed else 'none'}")

    cooldown = summary.get("execution_cooldown") or {}
    if cooldown:
        print(
            "execution_cooldown: lane={lane} allowed={allowed} reason={reason}".format(
                lane=cooldown.get("lane"),
                allowed=str(cooldown.get("allowed")).lower(),
                reason=cooldown.get("reason"),
            )
        )

    grid_summary = summary.get("grid_regime_summary") or {}
    print("\nGrid regime diagnostic")
    if grid_summary:
        print(
            "range_bound={range_bound_count} breakout_up={breakout_up_count} "
            "breakout_down={breakout_down_count} insufficient={insufficient_history_count}".format(
                range_bound_count=grid_summary.get("range_bound_count", 0),
                breakout_up_count=grid_summary.get("breakout_up_count", 0),
                breakout_down_count=grid_summary.get("breakout_down_count", 0),
                insufficient_history_count=grid_summary.get("insufficient_history_count", 0),
            )
        )
    else:
        print("not available")

    print("\nTop ETF ranks")
    for i, q in enumerate(summary.get("top_ranked") or [], 1):
        print(
            f"{i}. {q.get('stockCode')}.{q.get('exchange')} "
            f"{q.get('name', '')} price={q.get('currentPrice')} score={_pct(q.get('score'))}"
        )

    print("\nPlanned orders")
    orders = summary.get("orders") or []
    if not orders:
        print("none")
    for order in orders:
        print(
            f"{order.get('direction')} {order.get('stockCode')}.{order.get('exchange')} "
            f"qty={order.get('quantity')} type={order.get('orderType')} price={order.get('price')}"
        )

    print("\nSafety")
    print("paper_trading_only: true")
    print("live_ready: false")
    print("formal_strategy_allowed: false")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show ETF paper agent status without submitting orders")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--refresh", action="store_true", help="Run a fresh paper_dry_run before showing status")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    cfg_path = Path(args.config)
    cfg = load_json(cfg_path)
    out_dir = ROOT / cfg["outputs"]["dir"]
    latest = out_dir / cfg["outputs"]["latest"]
    status_path = out_dir / "latest_status.json"
    blotter_path = out_dir / cfg["outputs"].get("blotter", "order_blotter.csv")

    if args.refresh or not latest.exists():
        result = run_agent(cfg_path, "paper_dry_run", execute=False)
        write_json(latest, result)
        append_order_blotter(blotter_path, result.get("order_blotter_rows", []))
    else:
        result = load_json(latest)

    summary = _summarize(result)
    write_json(status_path, summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_summary(summary)
        print(f"\nstatus_file: {status_path}")


if __name__ == "__main__":
    main()
