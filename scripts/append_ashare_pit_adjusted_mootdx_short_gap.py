"""Append a strictly bounded raw-TDX short gap to the PIT adjusted research panel.

This is a research-data continuity adapter for temporary BaoStock outages.  It
never rewrites an existing observation.  A raw row is normalized to the most
recent BaoStock backward-adjusted price scale only when at least two recent,
official BaoStock/raw overlap sessions agree on that scale.  Ambiguous symbols
are skipped and reported rather than guessed.

The adapter is deliberately isolated from trading configuration, orders,
positions, overlays, and execution code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = (
    ROOT
    / "data"
    / "market"
    / "ashare_research"
    / "baostock_pit_adjusted"
    / "master"
    / "ashare_pit_master_latest.jsonl"
)
DEFAULT_PIT_ROOT = (
    ROOT
    / "data"
    / "market"
    / "ashare_research"
    / "baostock_pit_adjusted"
    / "bars_1d_backward_adjusted"
)
DEFAULT_RAW_ROOT = ROOT / "data" / "market" / "ashare_research" / "bars_1d_raw"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs" / "edge_research" / "ashare_pit_adjusted_short_gap"
)

SCHEMA_VERSION = "ashare_pit_adjusted_short_gap_v1"
OFFICIAL_SOURCE = "baostock_query_history_k_data_plus"
ADJUSTMENT = "backward_adjusted_baostock_pctchg_method"
PRICE_FIELDS = ("open", "high", "low", "close")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read a small JSONL suffix without loading a long price history."""
    if limit < 1 or not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunks: list[bytes] = []
        newline_count = 0
        while position > 0 and newline_count <= limit:
            size = min(65536, position)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    data = b"".join(reversed(chunks)).decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in data.splitlines()[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_price_row(row: dict[str, Any]) -> bool:
    values = [_number(row.get(field)) for field in PRICE_FIELDS]
    if any(value is None or value <= 0.0 for value in values):
        return False
    open_, high, low, close = (float(value) for value in values)
    return high >= max(open_, close, low) and low <= min(open_, close, high)


def estimate_recent_scale(
    pit_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    minimum_official_sessions: int,
    maximum_relative_error: float,
) -> tuple[float | None, float | None, str | None]:
    """Estimate one price scale from recent official BaoStock overlap rows."""
    raw_by_date = {str(row.get("dt")): row for row in raw_rows}
    sessions: list[tuple[str, list[float]]] = []
    for pit in pit_rows:
        if pit.get("source") != OFFICIAL_SOURCE:
            continue
        raw = raw_by_date.get(str(pit.get("dt")))
        if raw is None or not _valid_price_row(pit) or not _valid_price_row(raw):
            continue
        session_ratios: list[float] = []
        for field in PRICE_FIELDS:
            raw_value = float(raw[field])
            pit_value = float(pit[field])
            session_ratios.append(pit_value / raw_value)
        sessions.append((str(pit.get("dt")), session_ratios))
    if len(sessions) < minimum_official_sessions:
        return None, None, "insufficient_recent_official_overlap"

    # Backward-adjusted/raw scale changes at a corporate action.  Anchor on the
    # latest official session and use only its most recent stable suffix rather
    # than averaging across an older adjustment regime.
    ratios: list[float] = []
    stable_sessions = 0
    anchor: float | None = None
    for _, session_ratios in sorted(sessions, reverse=True):
        session_scale = sum(session_ratios) / len(session_ratios)
        within_error = max(abs(value / session_scale - 1.0) for value in session_ratios)
        if within_error > maximum_relative_error:
            break
        if anchor is None:
            anchor = session_scale
        elif abs(session_scale / anchor - 1.0) > maximum_relative_error:
            break
        ratios.extend(session_ratios)
        stable_sessions += 1
    if stable_sessions < minimum_official_sessions:
        return None, None, "unstable_recent_adjustment_scale"
    scale = sum(ratios) / len(ratios)
    if not math.isfinite(scale) or scale <= 0.0:
        return None, None, "invalid_scale"
    relative_error = max(abs(value / scale - 1.0) for value in ratios)
    if relative_error > maximum_relative_error:
        return None, relative_error, "unstable_recent_adjustment_scale"
    return scale, relative_error, None


def build_adjusted_row(
    raw: dict[str, Any],
    previous_adjusted_close: float,
    scale: float,
    last_is_st: int,
    reconciliation_error: float,
) -> dict[str, Any]:
    prices = {field: float(raw[field]) * scale for field in PRICE_FIELDS}
    close = prices["close"]
    raw_volume = float(raw.get("vol") or 0.0)
    amount = float(raw.get("amount") or 0.0)
    return {
        "adjustflag": "1",
        "adjustment": ADJUSTMENT,
        "adjustmentDerivation": "mootdx_raw_normalized_to_recent_official_baostock_scale",
        "amount": amount,
        "amountSource": raw.get("amountSource", "exchange_reported_via_tdx"),
        "amountUnit": "CNY",
        "close": close,
        "dt": str(raw["dt"]),
        "exchange": str(raw["exchange"]),
        "high": prices["high"],
        "isST": int(last_is_st),
        "low": prices["low"],
        "open": prices["open"],
        "pctChangePct": (close / previous_adjusted_close - 1.0) * 100.0,
        "pointInTimeStatus": True,
        "pointInTimeStatusSource": "last_baostock_status_carried_forward_short_gap",
        "preclose": previous_adjusted_close,
        "providerCode": f"{str(raw['exchange']).lower()}.{raw['stockCode']}",
        "researchOnly": True,
        "securityId": str(raw["securityId"]),
        "source": "mootdx_raw_normalized_to_baostock_scale",
        "sourceReconciliationMaxRelativeError": reconciliation_error,
        "stockCode": str(raw["stockCode"]),
        "tradeStatus": int(raw_volume > 0.0),
        "vol": raw_volume * 100.0,
        "volumeUnit": "shares",
        "vwap": sum(prices.values()) / 4.0,
        "vwapSource": "adjusted_ohlc4_proxy_not_true_transaction_vwap",
    }


def plan_symbol_append(
    pit_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    target_date: str,
    *,
    maximum_gap_sessions: int,
    minimum_official_sessions: int,
    maximum_relative_error: float,
    maximum_absolute_raw_return: float,
) -> tuple[list[dict[str, Any]], str | None, float | None]:
    if not pit_rows:
        return [], "missing_pit_history", None
    if not raw_rows:
        return [], "missing_raw_history", None
    pit_rows = sorted(pit_rows, key=lambda row: str(row.get("dt")))
    raw_rows = sorted(raw_rows, key=lambda row: str(row.get("dt")))
    last_pit_date = str(pit_rows[-1].get("dt"))
    missing = [
        row
        for row in raw_rows
        if last_pit_date < str(row.get("dt")) <= target_date
    ]
    if not missing:
        return [], None, None
    if len(missing) > maximum_gap_sessions:
        return [], "gap_exceeds_frozen_session_limit", None
    if any(not _valid_price_row(row) for row in missing):
        return [], "invalid_raw_ohlc", None

    scale, reconciliation_error, scale_error = estimate_recent_scale(
        pit_rows,
        raw_rows,
        minimum_official_sessions=minimum_official_sessions,
        maximum_relative_error=maximum_relative_error,
    )
    if scale_error is not None or scale is None or reconciliation_error is None:
        return [], scale_error or "invalid_scale", reconciliation_error

    raw_by_date = {str(row.get("dt")): row for row in raw_rows}
    prior_raw = raw_by_date.get(last_pit_date)
    if prior_raw is None or not _valid_price_row(prior_raw):
        return [], "missing_raw_anchor_on_last_pit_date", reconciliation_error

    previous_raw_close = float(prior_raw["close"])
    previous_adjusted_close = float(pit_rows[-1]["close"])
    last_is_st = int(pit_rows[-1].get("isST") or 0)
    appended: list[dict[str, Any]] = []
    for raw in missing:
        raw_return = float(raw["close"]) / previous_raw_close - 1.0
        if abs(raw_return) > maximum_absolute_raw_return:
            return [], "suspected_corporate_action_or_abnormal_raw_return", reconciliation_error
        adjusted = build_adjusted_row(
            raw,
            previous_adjusted_close,
            scale,
            last_is_st,
            reconciliation_error,
        )
        appended.append(adjusted)
        previous_raw_close = float(raw["close"])
        previous_adjusted_close = float(adjusted["close"])
    return appended, None, reconciliation_error


def run(args: argparse.Namespace) -> dict[str, Any]:
    master = read_jsonl(args.master)
    if not master:
        raise RuntimeError("PIT master is missing or empty")
    target_date = args.target_date
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    counters: Counter[str] = Counter()
    appended_rows = 0
    appended_symbols = 0
    target_observed = 0
    target_covered = 0
    reconciliation_errors: list[float] = []
    planned_writes: list[tuple[Path, list[dict[str, Any]]]] = []

    for security in master:
        exchange = str(security.get("exchange") or "")
        code = str(security.get("stockCode") or "")
        if exchange not in {"SH", "SZ"} or len(code) != 6:
            counters["outside_supported_universe"] += 1
            continue
        if str(security.get("listingDate") or "9999-99-99") > target_date:
            counters["not_yet_listed"] += 1
            continue
        delisting = str(security.get("delistingDate") or "")
        if delisting and delisting < target_date:
            counters["already_delisted"] += 1
            continue

        filename = f"{exchange}_{code}.jsonl"
        pit_path = args.pit_root / filename
        raw_path = args.raw_root / filename
        pit_tail = read_jsonl_tail(pit_path, args.overlap_lookback_rows)
        raw_tail = read_jsonl_tail(raw_path, args.overlap_lookback_rows + args.maximum_gap_sessions)
        if raw_tail and str(raw_tail[-1].get("dt")) == target_date:
            target_observed += 1
        if pit_tail and str(pit_tail[-1].get("dt")) >= target_date:
            target_covered += 1
            counters["already_current"] += 1
            continue

        appended, reason, error = plan_symbol_append(
            pit_tail,
            raw_tail,
            target_date,
            maximum_gap_sessions=args.maximum_gap_sessions,
            minimum_official_sessions=args.minimum_official_sessions,
            maximum_relative_error=args.maximum_relative_error,
            maximum_absolute_raw_return=args.maximum_absolute_raw_return,
        )
        if reason is not None:
            counters[reason] += 1
            continue
        if not appended:
            counters["no_new_observed_rows"] += 1
            continue
        if str(appended[-1]["dt"]) == target_date:
            target_covered += 1
        reconciliation_errors.append(float(error or 0.0))
        appended_rows += len(appended)
        appended_symbols += 1
        planned_writes.append((pit_path, appended))

    coverage = target_covered / target_observed if target_observed else 0.0
    eligible = target_observed > 0 and coverage >= args.minimum_target_coverage
    status = "dry_run" if args.dry_run else "complete"
    if not eligible:
        status = "failed_closed"

    summary = {
        "schemaVersion": SCHEMA_VERSION,
        "status": status,
        "researchOnly": True,
        "generatedAt": now_iso(),
        "runId": run_id,
        "targetDate": target_date,
        "dryRun": bool(args.dry_run),
        "masterCount": len(master),
        "targetDateRawSymbolsObserved": target_observed,
        "targetDatePitSymbolsCovered": target_covered,
        "targetDateCoverage": round(coverage, 8),
        "minimumTargetCoverage": args.minimum_target_coverage,
        "qualityGatePassed": eligible,
        "plannedOrAppendedSymbols": appended_symbols,
        "plannedOrAppendedRows": appended_rows,
        "maximumReconciliationRelativeError": max(reconciliation_errors, default=None),
        "skipAndStatusCounts": dict(sorted(counters.items())),
        "limitations": [
            "This is a bounded continuity fallback, not a replacement for vendor-adjusted history.",
            "Point-in-time ST status is carried forward only across the short gap.",
            "A corporate action inside the missing interval can evade a pure price-jump guard; such rows require later BaoStock reconciliation.",
            "Outputs remain research-only and cannot create orders or alter trading state.",
        ],
        "orders": [],
        "automaticTradingChanges": [],
    }
    output = args.output_root / run_id / "summary.json"
    atomic_json(output, summary)
    if not eligible:
        raise RuntimeError(
            f"short-gap quality gate failed: coverage={coverage:.6f}, "
            f"required={args.minimum_target_coverage:.6f}; summary={output}"
        )
    if not args.dry_run:
        for path, appended in planned_writes:
            atomic_jsonl(path, read_jsonl(path) + appended)
    print(json.dumps({**summary, "summaryPath": str(output)}, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", default=date.today().isoformat())
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--pit-root", type=Path, default=DEFAULT_PIT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overlap-lookback-rows", type=int, default=24)
    parser.add_argument("--maximum-gap-sessions", type=int, default=5)
    parser.add_argument("--minimum-official-sessions", type=int, default=2)
    parser.add_argument("--maximum-relative-error", type=float, default=0.0005)
    parser.add_argument("--maximum-absolute-raw-return", type=float, default=0.25)
    parser.add_argument("--minimum-target-coverage", type=float, default=0.98)
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed_closed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
