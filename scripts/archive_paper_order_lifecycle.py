"""Build an idempotent paper-order lifecycle and adverse-selection research ledger.

Inputs are existing local paper records only:

* shared execution intent/submission events;
* intraday-agent pending-order snapshots and broker-confirmed fill evidence;
* the reconciled T0 inventory state for legacy fills;
* point-in-time minute quote logs for post-order markouts.

The script never calls the broker, never submits/cancels an order, and never writes
trading config, state, overlays or risk gates. Unobserved final states remain
``submitted_unresolved`` rather than being guessed as filled or cancelled.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CN = ZoneInfo("Asia/Shanghai")
INTENTS = ROOT / "outputs" / "shared_order_router" / "order_intents.jsonl"
RUNS = ROOT / "outputs" / "t0_intraday_agent" / "t0_agent_runs.jsonl"
STATE = ROOT / "outputs" / "t0_intraday_agent" / "t0_state.json"
QUOTES = ROOT / "outputs" / "t0_intraday_agent"
OUT = ROOT / "outputs" / "execution_research"
HORIZONS_MINUTES = (1, 5, 10, 30)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_time(value: Any, default_timezone: ZoneInfo = CN) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip().replace("Z", "+00:00")
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(raw, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed.astimezone(CN)


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def new_order(order_id: str) -> dict[str, Any]:
    return {
        "schemaVersion": "paper_order_lifecycle_v1",
        "orderId": order_id,
        "status": "submitted_unresolved",
        "submit_ok": True,
        "pending_observations": 0,
        "fill_confirmed": False,
        "fill_evidence": None,
        "filledQuantity": None,
        "filledPrice": None,
        "filledTime": None,
        "fill_time_known": False,
    }


def load_submissions(path: Path = INTENTS) -> dict[str, dict[str, Any]]:
    orders: dict[str, dict[str, Any]] = {}
    for event in read_jsonl(path):
        if event.get("event_type") != "submit_results_recorded":
            continue
        intents = event.get("orders") if isinstance(event.get("orders"), list) else []
        results = event.get("submit_results") if isinstance(event.get("submit_results"), list) else []
        for index, intent in enumerate(intents):
            if not isinstance(intent, dict):
                continue
            submit = results[index] if index < len(results) and isinstance(results[index], dict) else {}
            data = submit.get("data") if isinstance(submit.get("data"), dict) else {}
            order_id = str(data.get("orderId") or "")
            if not order_id:
                continue
            row = orders.setdefault(order_id, new_order(order_id))
            submitted = parse_time(data.get("submitTime")) or parse_time(event.get("timestamp"))
            row.update(
                {
                    "trade_date": str(event.get("trade_date") or ""),
                    "agent_name": event.get("agent_name"),
                    "submitted_at": submitted.isoformat() if submitted else event.get("timestamp"),
                    "stockCode": str(intent.get("stockCode") or "").zfill(6),
                    "exchange": intent.get("exchange"),
                    "name": intent.get("name") or data.get("stockName"),
                    "direction": str(intent.get("direction") or "").lower(),
                    "orderType": intent.get("orderType"),
                    "limitPrice": as_float(intent.get("price")),
                    "quantity": int(as_float(intent.get("quantity")) or 0),
                    "execution_style": intent.get("execution_style"),
                    "submission_mid": as_float(intent.get("submission_mid")),
                    "reason": intent.get("reason"),
                    "submit_ok": bool(submit.get("ok")),
                    "broker_submit_status": data.get("status"),
                    "submit_error": submit.get("error"),
                }
            )
            if not submit.get("ok"):
                row["status"] = "submit_rejected"
    return orders


def apply_run_observations(
    orders: dict[str, dict[str, Any]],
    path: Path = RUNS,
) -> None:
    for run in read_jsonl(path):
        observed_at = parse_time(run.get("timestamp"))
        pending = run.get("pending_t0_orders")
        for item in pending if isinstance(pending, list) else []:
            if not isinstance(item, dict):
                continue
            order_id = str(item.get("orderId") or "")
            if not order_id:
                continue
            row = orders.setdefault(order_id, new_order(order_id))
            first = parse_time(row.get("first_pending_seen_at"))
            row["first_pending_seen_at"] = (
                min(first, observed_at).isoformat()
                if first is not None and observed_at is not None
                else (first or observed_at).isoformat()
                if (first or observed_at) is not None
                else None
            )
            row["last_pending_seen_at"] = observed_at.isoformat() if observed_at else run.get("timestamp")
            row["pending_observations"] = int(row.get("pending_observations") or 0) + 1
            row["pending_max_filled_quantity"] = max(
                int(row.get("pending_max_filled_quantity") or 0),
                int(as_float(item.get("filledQuantity")) or 0),
            )
            if not row.get("fill_confirmed"):
                row["status"] = "pending_observed"
            for source_key, target_key in (
                ("stockCode", "stockCode"),
                ("exchange", "exchange"),
                ("direction", "direction"),
                ("price", "limitPrice"),
                ("quantity", "quantity"),
            ):
                if row.get(target_key) in (None, "", 0) and item.get(source_key) not in (None, ""):
                    row[target_key] = item[source_key]

        reconciliation = run.get("fill_reconciliation")
        confirmed = (
            reconciliation.get("confirmed_trades")
            if isinstance(reconciliation, dict)
            and isinstance(reconciliation.get("confirmed_trades"), list)
            else []
        )
        for trade in confirmed:
            if not isinstance(trade, dict):
                continue
            order_id = str(trade.get("orderId") or "")
            if not order_id:
                continue
            row = orders.setdefault(order_id, new_order(order_id))
            fill_time = parse_time(trade.get("filledTime"))
            row.update(
                {
                    "status": "filled_confirmed_exact",
                    "fill_confirmed": True,
                    "fill_evidence": "broker_trade_history_record",
                    "filledQuantity": int(as_float(trade.get("filledQuantity")) or 0),
                    "filledPrice": as_float(trade.get("filledPrice")),
                    "filledAmount": as_float(trade.get("filledAmount")),
                    "fee": as_float(trade.get("fee")),
                    "filledTime": fill_time.isoformat() if fill_time else trade.get("filledTime"),
                    "fill_time_known": fill_time is not None,
                }
            )


def apply_legacy_reconciled_state(
    orders: dict[str, dict[str, Any]],
    state_path: Path = STATE,
) -> None:
    state = read_json(state_path)
    by_date = state.get("t0_inventory_by_date")
    if not isinstance(by_date, dict):
        return
    for trade_date, code_nodes in by_date.items():
        if not isinstance(code_nodes, dict):
            continue
        for code, node in code_nodes.items():
            if not isinstance(node, dict) or not node.get("fill_reconciliation_ok"):
                continue
            matched = {str(value) for value in node.get("matched_trade_order_ids", []) if value}
            buy_ids = [str(value) for value in node.get("buy_order_ids", []) if value]
            sell_ids = [str(value) for value in node.get("sell_order_ids", []) if value]
            for direction, identifiers, quantity_key, price_key in (
                ("buy", buy_ids, "buy_quantity_filled", "filled_buy_vwap"),
                ("sell", sell_ids, "sell_quantity_filled", "filled_sell_vwap"),
            ):
                matched_direction = [order_id for order_id in identifiers if order_id in matched]
                for order_id in matched_direction:
                    row = orders.setdefault(order_id, new_order(order_id))
                    if row.get("status") == "filled_confirmed_exact":
                        continue
                    quantity = (
                        int(as_float(node.get(quantity_key)) or 0)
                        if len(matched_direction) == 1
                        else None
                    )
                    row.update(
                        {
                            "trade_date": row.get("trade_date") or trade_date,
                            "stockCode": row.get("stockCode") or str(code).zfill(6),
                            "direction": row.get("direction") or direction,
                            "status": "filled_confirmed_aggregate",
                            "fill_confirmed": True,
                            "fill_evidence": "reconciled_inventory_aggregate",
                            "filledQuantity": quantity,
                            "filledPrice": as_float(node.get(price_key)),
                            "fill_confirmed_at": node.get("fill_reconciled_at"),
                            "fill_time_known": False,
                        }
                    )


def quote_time(row: dict[str, Any]) -> datetime | None:
    # The collector timestamp is when the strategy could observe the quote.
    return parse_time(row.get("timestamp") or row.get("collected_at"))


def quote_mid(row: dict[str, Any]) -> float | None:
    bid = as_float(row.get("bidPrice1", row.get("bid")))
    ask = as_float(row.get("askPrice1", row.get("ask")))
    if bid is not None and ask is not None and ask >= bid > 0:
        return (bid + ask) / 2.0
    current = as_float(row.get("currentPrice", row.get("current")))
    return current if current is not None and current > 0 else None


def load_quotes_for_day(trade_date: str, quote_dir: Path = QUOTES) -> dict[str, list[tuple[datetime, float]]]:
    path = quote_dir / f"minute_quotes_{trade_date}.jsonl"
    by_code: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in read_jsonl(path):
        code = str(row.get("stockCode") or row.get("code") or "").zfill(6)
        timestamp = quote_time(row)
        midpoint = quote_mid(row)
        if len(code) != 6 or timestamp is None or midpoint is None:
            continue
        by_code[code].append((timestamp, midpoint))
    for values in by_code.values():
        values.sort(key=lambda item: item[0])
    return by_code


def first_at_or_after(
    observations: list[tuple[datetime, float]],
    target: datetime,
) -> tuple[datetime, float] | None:
    for timestamp, price in observations:
        if timestamp >= target:
            return timestamp, price
    return None


def enrich_execution_and_markouts(
    orders: dict[str, dict[str, Any]],
    quote_dir: Path = QUOTES,
    horizons: tuple[int, ...] = HORIZONS_MINUTES,
) -> None:
    quote_cache: dict[str, dict[str, list[tuple[datetime, float]]]] = {}
    for row in orders.values():
        trade_date = str(row.get("trade_date") or "")
        code = str(row.get("stockCode") or "").zfill(6)
        submitted = parse_time(row.get("submitted_at"))
        if not trade_date or not code or submitted is None:
            continue
        if trade_date not in quote_cache:
            quote_cache[trade_date] = load_quotes_for_day(trade_date, quote_dir)
        observations = quote_cache[trade_date].get(code, [])

        submission_mid = as_float(row.get("submission_mid"))
        if submission_mid is None:
            at_submit = first_at_or_after(observations, submitted)
            submission_mid = at_submit[1] if at_submit else None
            row["submission_mid_source"] = "first_recorded_quote_at_or_after_submit"
        else:
            row["submission_mid_source"] = "order_intent"
        row["submission_mid"] = submission_mid

        fill_price = as_float(row.get("filledPrice"))
        direction = str(row.get("direction") or "").lower()
        if row.get("fill_confirmed") and fill_price and submission_mid and direction in {"buy", "sell"}:
            signed_cost = (
                fill_price / submission_mid - 1.0
                if direction == "buy"
                else submission_mid / fill_price - 1.0
            )
            row["implementation_shortfall_bps"] = round(signed_cost * 10_000.0, 6)

        exact_fill = parse_time(row.get("filledTime")) if row.get("fill_time_known") else None
        reference_time = exact_fill or submitted
        reference_price = fill_price if row.get("fill_confirmed") and fill_price else submission_mid
        row["markout_reference"] = (
            "exact_fill_time_and_price"
            if exact_fill is not None and fill_price
            else "submission_time_with_confirmed_fill_price"
            if row.get("fill_confirmed") and fill_price
            else "submission_time_and_mid"
        )
        if exact_fill is not None:
            row["fill_delay_seconds"] = round((exact_fill - submitted).total_seconds(), 3)
        if reference_price is None or reference_price <= 0 or direction not in {"buy", "sell"}:
            continue
        for horizon in horizons:
            observation = first_at_or_after(
                observations,
                reference_time + timedelta(minutes=horizon),
            )
            if observation is None:
                row[f"markout_{horizon}m_bps"] = None
                row[f"markout_{horizon}m_observed_at"] = None
                continue
            observed_at, future_mid = observation
            signed_return = (
                future_mid / reference_price - 1.0
                if direction == "buy"
                else reference_price / future_mid - 1.0
            )
            row[f"markout_{horizon}m_bps"] = round(signed_return * 10_000.0, 6)
            row[f"markout_{horizon}m_observed_at"] = observed_at.isoformat()


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    submitted = [row for row in rows if row.get("submit_ok")]
    confirmed = [row for row in submitted if row.get("fill_confirmed")]
    statuses: dict[str, int] = defaultdict(int)
    for row in rows:
        statuses[str(row.get("status") or "unknown")] += 1
    return {
        "schemaVersion": "paper_order_lifecycle_summary_v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "orders": len(rows),
        "submitted_ok": len(submitted),
        "confirmed_fills": len(confirmed),
        "confirmed_fill_rate": round(len(confirmed) / len(submitted), 6) if submitted else None,
        "unresolved_orders": sum(
            row.get("status") in {"submitted_unresolved", "pending_observed"} for row in submitted
        ),
        "status_counts": dict(sorted(statuses.items())),
        "implementation_shortfall_bps": describe(
            finite_values(confirmed, "implementation_shortfall_bps")
        ),
        "confirmed_fill_markouts_bps": {
            f"{horizon}m": describe(finite_values(confirmed, f"markout_{horizon}m_bps"))
            for horizon in HORIZONS_MINUTES
        },
        "limitations": [
            "Legacy reconciled fills have confirmed price/quantity but often lack exact fill time.",
            "Pending disappearance is never assumed to mean fill or cancellation.",
            "Minute markouts use locally observed point-in-time quotes, not message-level queue data.",
            "The ledger is research-only and cannot change order style, gating, sizing or overlays.",
        ],
        "liveChanges": False,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper Order Lifecycle and Adverse-Selection Ledger",
        "",
        "Status: `diagnostic_only / local paper records / no broker calls`",
        "",
        f"- Orders: {summary['orders']}",
        f"- Submitted OK: {summary['submitted_ok']}",
        f"- Confirmed fills: {summary['confirmed_fills']}",
        f"- Confirmed fill rate: {summary['confirmed_fill_rate']}",
        f"- Unresolved orders: {summary['unresolved_orders']}",
        f"- Median implementation shortfall: "
        f"{summary['implementation_shortfall_bps']['median']} bps",
        "",
        "## Confirmed-fill signed markouts",
        "",
        "| Horizon | Count | Mean bps | Median bps |",
        "|---|---:|---:|---:|",
    ]
    for horizon, stats in summary["confirmed_fill_markouts_bps"].items():
        lines.append(
            f"| {horizon} | {stats['count']} | {stats['mean']} | {stats['median']} |"
        )
    lines.extend(["", "## Status counts", ""])
    for status, count in summary["status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def build_lifecycle(
    intents_path: Path = INTENTS,
    runs_path: Path = RUNS,
    state_path: Path = STATE,
    quote_dir: Path = QUOTES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    orders = load_submissions(intents_path)
    apply_run_observations(orders, runs_path)
    apply_legacy_reconciled_state(orders, state_path)
    enrich_execution_and_markouts(orders, quote_dir)
    rows = sorted(
        orders.values(),
        key=lambda row: (str(row.get("submitted_at") or ""), str(row.get("orderId") or "")),
    )
    return rows, summarize(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    rows, summary = build_lifecycle()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(
        output / "paper_order_lifecycle.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    atomic_json(output / "paper_order_lifecycle_summary.json", summary)
    atomic_text(output / "paper_order_lifecycle_report.md", markdown(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "orders": summary["orders"],
                "confirmed_fills": summary["confirmed_fills"],
                "unresolved_orders": summary["unresolved_orders"],
                "output": str(output / "paper_order_lifecycle_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
