"""Local holdings calibration report for paper trading.

Reads local logs/state only. It does not call trading APIs, submit orders,
cancel orders, or modify strategy parameters. When API quota is exhausted,
this report explicitly marks submitted orders as not externally confirmed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def today_cn() -> str:
    return datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def row_date(row: dict[str, Any]) -> str:
    for key in ("timestamp", "trade_date", "submitTime", "submit_time"):
        value = str(row.get(key) or "")
        if len(value) >= 10:
            return value[:10]
    return ""


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def quota_for_date(path: Path, date: str) -> dict[str, Any]:
    state = read_json(path)
    node = state.get("quota_exhausted_by_date", {}).get(date, {})
    return node if isinstance(node, dict) else {}


def latest_run_with_order_state(path: Path, date: str) -> list[dict[str, Any]]:
    out = []
    for run in read_jsonl(path):
        if row_date(run) != date:
            continue
        if run.get("orders") or run.get("submit_results"):
            out.append(run)
    return out


def build_report(date: str) -> dict[str, Any]:
    t0_state = read_json(ROOT / "outputs/t0_intraday_agent/t0_state.json")
    t0_latest = read_json(ROOT / "outputs/t0_intraday_agent/latest_t0_decision.json")
    lf_latest = read_json(ROOT / "outputs/paper_trading_agent/latest_decision.json")
    t0_inventory = t0_state.get("t0_inventory_by_date", {}).get(date, {})
    if not isinstance(t0_inventory, dict):
        t0_inventory = {}

    t0_blotter = [r for r in read_csv(ROOT / "outputs/t0_intraday_agent/t0_order_blotter.csv") if row_date(r) == date]
    lf_blotter = [r for r in read_csv(ROOT / "outputs/paper_trading_agent/order_blotter.csv") if row_date(r) == date]
    submitted_runs = latest_run_with_order_state(ROOT / "outputs/t0_intraday_agent/t0_agent_runs.jsonl", date)

    positions: list[dict[str, Any]] = []
    pending_or_unconfirmed: list[dict[str, Any]] = []
    for code, node in sorted(t0_inventory.items()):
        if not isinstance(node, dict):
            continue
        filled_qty = int(as_float(node.get("filled_remaining_qty")))
        submitted_qty = int(as_float(node.get("buy_quantity_submitted")))
        sell_submitted = int(as_float(node.get("sell_quantity_submitted")))
        entry_price = as_float(node.get("filled_buy_vwap"), as_float(node.get("entry_price")))
        record = {
            "stockCode": code,
            "buy_order_ids": node.get("buy_order_ids", []),
            "submitted_buy_qty": submitted_qty,
            "confirmed_filled_remaining_qty": filled_qty,
            "submitted_sell_qty": sell_submitted,
            "entry_price_local": entry_price,
            "stop_price": node.get("stop_price"),
            "target1_price": node.get("target1_price"),
            "target2_price": node.get("target2_price"),
            "fill_reconciliation_ok": node.get("fill_reconciliation_ok"),
            "fill_reconciled_at": node.get("fill_reconciled_at"),
            "confirmation_status": "confirmed_position" if filled_qty > 0 else "submitted_not_confirmed_filled",
        }
        if filled_qty > 0:
            positions.append(record)
        elif submitted_qty > 0 or sell_submitted > 0:
            pending_or_unconfirmed.append(record)

    t0_quota = quota_for_date(ROOT / "outputs/t0_intraday_agent/quota_backoff_state.json", date)
    lf_quota = quota_for_date(ROOT / "outputs/paper_trading_agent/quota_backoff_state.json", date)

    return {
        "date": date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "api_calls_made": False,
        "calibration_source": "local_logs_and_state_only",
        "external_broker_confirmation_available": False,
        "external_broker_confirmation_blocked_reason": "quota_exhausted_backoff" if t0_quota or lf_quota else "not_queried_by_design",
        "confirmed_positions_local": positions,
        "pending_or_unconfirmed_orders_local": pending_or_unconfirmed,
        "today_t0_blotter_rows": t0_blotter,
        "today_low_frequency_blotter_rows": lf_blotter,
        "submitted_runs_with_api_responses": submitted_runs,
        "quota_status": {
            "t0": t0_quota,
            "low_frequency": lf_quota,
        },
        "latest": {
            "t0": {
                "timestamp": t0_latest.get("timestamp"),
                "status": t0_latest.get("status"),
                "state_machine": t0_latest.get("state_machine"),
                "orders": len(t0_latest.get("orders") or []),
                "submit_results": len(t0_latest.get("submit_results") or []),
            },
            "low_frequency": {
                "timestamp": lf_latest.get("timestamp"),
                "status": lf_latest.get("status"),
                "reason": lf_latest.get("reason"),
                "orders": len(lf_latest.get("orders") or []),
                "submit_results": len(lf_latest.get("submit_results") or []),
            },
        },
        "summary": {
            "confirmed_position_count_local": len(positions),
            "pending_or_unconfirmed_count_local": len(pending_or_unconfirmed),
            "t0_blotter_rows": len(t0_blotter),
            "low_frequency_blotter_rows": len(lf_blotter),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Holdings Calibration {report['date']}",
        "",
        "Paper trading diagnostics only. This report reads local logs/state only and does not call trading APIs.",
        "",
        "## Summary",
        "",
        f"- Confirmed local positions: {report['summary']['confirmed_position_count_local']}",
        f"- Pending or unconfirmed local orders: {report['summary']['pending_or_unconfirmed_count_local']}",
        f"- External broker confirmation available: {report['external_broker_confirmation_available']}",
        f"- Blocked reason: {report['external_broker_confirmation_blocked_reason']}",
        "",
        "## Confirmed Local Positions",
        "",
    ]
    if report["confirmed_positions_local"]:
        for pos in report["confirmed_positions_local"]:
            lines.append(f"- {pos['stockCode']}: qty={pos['confirmed_filled_remaining_qty']}, entry={pos['entry_price_local']}")
    else:
        lines.append("- None confirmed from local fill reconciliation.")
    lines.extend(["", "## Pending Or Unconfirmed", ""])
    if report["pending_or_unconfirmed_orders_local"]:
        for pos in report["pending_or_unconfirmed_orders_local"]:
            lines.append(
                f"- {pos['stockCode']}: submitted_buy_qty={pos['submitted_buy_qty']}, "
                f"confirmed_filled_remaining_qty={pos['confirmed_filled_remaining_qty']}, "
                f"orders={pos['buy_order_ids']}, status={pos['confirmation_status']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Safety Flags", ""])
    lines.extend([
        f"- paper_trading_only: {report['paper_trading_only']}",
        f"- live_ready: {report['live_ready']}",
        f"- formal_strategy_allowed: {report['formal_strategy_allowed']}",
        f"- investment_recommendation: {report['investment_recommendation']}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local holdings calibration report")
    parser.add_argument("--date", default=today_cn())
    args = parser.parse_args()
    out_dir = ROOT / "outputs" / "holdings_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.date)
    stem = args.date.replace("-", "")
    json_path = out_dir / f"{stem}_holdings_calibration.json"
    md_path = out_dir / f"{stem}_holdings_calibration.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({
        "status": "written",
        "date": args.date,
        "json": str(json_path),
        "markdown": str(md_path),
        "api_calls_made": False,
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
