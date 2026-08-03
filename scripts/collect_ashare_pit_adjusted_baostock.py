"""Collect an isolated PIT-membership, adjusted SH/SZ A-share research dataset.

The script uses BaoStock only for offline data research.  It never reads a trading
configuration, touches an order path, or overwrites the existing raw TDX/Sina files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "ashare_pit_adjusted_data_v1.json"
SCHEMA_VERSION = "ashare_pit_adjusted_data_v1"
SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def resolve_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / config["paths"][key]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected PIT adjusted data schema")
    if config.get("status") != "research_only_not_trading":
        raise ValueError("PIT adjusted collection must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("collector output must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("collector cannot receive a trading permission")
    provider = config["provider"]
    if provider.get("name") != "BaoStock" or provider.get("priceAdjustmentFlag") != "1":
        raise ValueError("clean V1 requires BaoStock backward-adjusted prices")
    if set(config["universe"]["exchanges"]) != {"SH", "SZ"}:
        raise ValueError("clean V1 is deliberately restricted to SH/SZ")
    if config["universe"].get("securityType") != "1":
        raise ValueError("clean V1 may include only type-1 stocks")
    if config["normalization"].get("volumeUnit") != "shares":
        raise ValueError("clean V1 volume unit changed")
    roots = [resolve_path(config, key).resolve() for key in ("masterRoot", "barsRoot", "auditRoot")]
    if len(set(roots)) != len(roots):
        raise ValueError("master, bars and audit roots must be isolated")
    forbidden = (ROOT / "data" / "market" / "ashare_research" / "bars_1d_raw").resolve()
    if any(root == forbidden or forbidden in root.parents for root in roots):
        raise ValueError("clean V1 cannot overwrite the raw bars directory")


def _a_share_code(code: str) -> tuple[str, str] | None:
    code = str(code).lower()
    if code.startswith("sh.") and code[3:].startswith(SH_PREFIXES):
        return "SH", code[3:]
    if code.startswith("sz.") and code[3:].startswith(SZ_PREFIXES):
        return "SZ", code[3:]
    return None


def normalize_master_record(
    item: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    if str(item.get("type")) != str(config["universe"]["securityType"]):
        return None
    parsed = _a_share_code(str(item.get("code") or ""))
    if parsed is None:
        return None
    exchange, stock_code = parsed
    listing = str(item.get("ipoDate") or "")[:10]
    delisting = str(item.get("outDate") or "")[:10]
    start = str(config["universe"]["startDate"])
    if not listing:
        return None
    if delisting and delisting < start:
        return None
    return {
        "securityId": f"{exchange}.{stock_code}",
        "providerCode": str(item["code"]).lower(),
        "exchange": exchange,
        "stockCode": stock_code,
        "name": str(item.get("code_name") or "").strip(),
        "listingDate": listing,
        "delistingDate": delisting or None,
        "currentDiscoverable": str(item.get("status")) == "1",
        "delisted": bool(delisting),
        "securityType": "A_SHARE_STOCK",
        "providerSecurityType": str(item.get("type")),
        "pointInTimeMembership": True,
        "membershipRule": config["universe"]["pointInTimeMembershipRule"],
        "discoverySource": "baostock_query_stock_basic",
        "researchOnly": True,
    }


def fetch_master(config: dict[str, Any]) -> list[dict[str, Any]]:
    import baostock as bs

    result = bs.query_stock_basic()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock master failed: {result.error_code} {result.error_msg}")
    rows: list[dict[str, Any]] = []
    while result.next():
        item = dict(zip(result.fields, result.get_row_data()))
        normalized = normalize_master_record(item, config)
        if normalized is not None:
            rows.append(normalized)
    return sorted(rows, key=lambda row: row["securityId"])


def save_master(rows: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    generated = now_iso()
    for row in rows:
        row["masterGeneratedAt"] = generated
    root = resolve_path(config, "masterRoot")
    dated = root / f"ashare_pit_master_{date.today().strftime('%Y%m%d')}.jsonl"
    atomic_jsonl(dated, rows)
    atomic_jsonl(resolve_path(config, "latestMaster"), rows)
    return dated


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_history_record(
    item: dict[str, Any], security: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    prices = [_as_float(item.get(name)) for name in ("open", "high", "low", "close")]
    volume = _as_float(item.get("volume"))
    amount = _as_float(item.get("amount"))
    preclose = _as_float(item.get("preclose"))
    trade_status = _as_int(item.get("tradestatus"))
    is_st = _as_int(item.get("isST"))
    dt = str(item.get("date") or "")[:10]
    if (
        not dt
        or any(value is None or value <= 0.0 for value in prices)
        or volume is None
        or volume < 0.0
        or amount is None
        or amount < 0.0
        or preclose is None
        or preclose <= 0.0
        or trade_status not in {0, 1}
        or is_st not in {0, 1}
        or str(item.get("adjustflag")) != str(config["provider"]["priceAdjustmentFlag"])
    ):
        return None
    open_px, high_px, low_px, close_px = (float(value) for value in prices)
    if high_px < max(open_px, low_px, close_px) or low_px > min(open_px, high_px, close_px):
        return None
    return {
        "dt": dt,
        "securityId": security["securityId"],
        "providerCode": security["providerCode"],
        "exchange": security["exchange"],
        "stockCode": security["stockCode"],
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "preclose": float(preclose),
        "vol": float(volume),
        "amount": float(amount),
        "vwap": (open_px + high_px + low_px + close_px) / 4.0,
        "vwapSource": "adjusted_ohlc4_proxy_not_true_transaction_vwap",
        "volumeUnit": "shares",
        "amountUnit": "CNY",
        "amountSource": "exchange_reported_via_baostock",
        "adjustment": "backward_adjusted_baostock_pctchg_method",
        "adjustflag": "1",
        "tradeStatus": trade_status,
        "isST": is_st,
        "pctChangePct": _as_float(item.get("pctChg")),
        "source": "baostock_query_history_k_data_plus",
        "pointInTimeStatus": True,
        "researchOnly": True,
    }


def _query_history(
    bs: Any,
    security: dict[str, Any],
    start_date: str,
    end_date: str,
    config: dict[str, Any],
    retries: int,
) -> list[dict[str, Any]]:
    fields = ",".join(config["provider"]["fields"])
    last_error = "unknown"
    for attempt in range(max(1, retries)):
        result = bs.query_history_k_data_plus(
            security["providerCode"],
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=str(config["provider"]["frequency"]),
            adjustflag=str(config["provider"]["priceAdjustmentFlag"]),
        )
        if result.error_code == "0":
            output: list[dict[str, Any]] = []
            while result.next():
                item = dict(zip(result.fields, result.get_row_data()))
                parsed = parse_history_record(item, security, config)
                if parsed is not None:
                    output.append(parsed)
            return output
        last_error = f"{result.error_code} {result.error_msg}"
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(last_error)


def collect_one(
    bs: Any,
    security: dict[str, Any],
    config: dict[str, Any],
    end_date: str,
    force: bool,
    retries: int,
) -> dict[str, Any]:
    path = resolve_path(config, "barsRoot") / f"{security['exchange']}_{security['stockCode']}.jsonl"
    old = [] if force else read_jsonl(path)
    start = max(str(config["universe"]["startDate"]), str(security["listingDate"]))
    if old:
        start = max(start, (datetime.fromisoformat(str(old[-1]["dt"])[:10]) + timedelta(days=1)).date().isoformat())
    stop = end_date
    if security.get("delistingDate"):
        stop = min(stop, str(security["delistingDate"]))
    if start > stop:
        return {
            "securityId": security["securityId"],
            "status": "skipped_current",
            "rows": len(old),
            "first": old[0]["dt"] if old else None,
            "last": old[-1]["dt"] if old else None,
            "path": str(path),
        }
    try:
        fresh = _query_history(bs, security, start, stop, config, retries)
        merged = {str(row["dt"]): row for row in [*old, *fresh]}
        rows = [merged[key] for key in sorted(merged)]
        if rows:
            atomic_jsonl(path, rows)
        return {
            "securityId": security["securityId"],
            "status": "ok" if fresh else "no_new_rows" if old else "no_rows",
            "rows": len(rows),
            "addedRows": len(fresh),
            "first": rows[0]["dt"] if rows else None,
            "last": rows[-1]["dt"] if rows else None,
            "path": str(path),
        }
    except Exception as exc:
        return {
            "securityId": security["securityId"],
            "status": "failed",
            "rows": len(old),
            "error": str(exc)[:500],
            "path": str(path),
        }


def audit(config: dict[str, Any]) -> dict[str, Any]:
    master_path = resolve_path(config, "latestMaster")
    master = read_jsonl(master_path)
    files = 0
    symbols_500 = 0
    total_rows = 0
    invalid_rows = 0
    missing_status_rows = 0
    wrong_adjustment_rows = 0
    delisted_files = 0
    for security in master:
        path = resolve_path(config, "barsRoot") / f"{security['exchange']}_{security['stockCode']}.jsonl"
        rows = read_jsonl(path)
        if not rows:
            continue
        files += 1
        total_rows += len(rows)
        symbols_500 += int(len(rows) >= 500)
        delisted_files += int(bool(security.get("delisted")))
        for row in rows:
            missing_status_rows += int(row.get("isST") not in {0, 1} or row.get("tradeStatus") not in {0, 1})
            wrong_adjustment_rows += int(row.get("adjustment") != "backward_adjusted_baostock_pctchg_method")
            invalid_rows += int(
                not row.get("dt")
                or min(float(row.get(key) or 0.0) for key in ("open", "high", "low", "close")) <= 0.0
            )
    coverage = files / len(master) if master else 0.0
    invalid_fraction = invalid_rows / total_rows if total_rows else 1.0
    gate = config["qualityGate"]
    checks = {
        "masterPresent": bool(master),
        "masterFileCoveragePass": coverage >= float(gate["minimumMasterFileCoverage"]),
        "longHistorySymbolCountPass": symbols_500 >= int(gate["minimumSymbolsWithAtLeast500Rows"]),
        "invalidRowFractionPass": invalid_fraction <= float(gate["maximumInvalidRowFraction"]),
        "allRowsAdjustedPass": total_rows > 0 and wrong_adjustment_rows == 0,
        "allRowsPointInTimeStatusPass": total_rows > 0 and missing_status_rows == 0,
        "listingDateCompletePass": bool(master) and all(row.get("listingDate") for row in master),
    }
    payload = {
        "schemaVersion": "ashare_pit_adjusted_data_audit_v1",
        "status": "diagnostic_only_research_only",
        "generatedAt": now_iso(),
        "configSha256": digest_file(DEFAULT_CONFIG),
        "masterSha256": digest_file(master_path),
        "masterCount": len(master),
        "currentMasterRows": sum(int(bool(row.get("currentDiscoverable"))) for row in master),
        "delistedMasterRows": sum(int(bool(row.get("delisted"))) for row in master),
        "barFileCount": files,
        "delistedBarFileCount": delisted_files,
        "masterFileCoverage": round(coverage, 8),
        "symbolsWithAtLeast500Rows": symbols_500,
        "totalRows": total_rows,
        "invalidRows": invalid_rows,
        "invalidRowFraction": round(invalid_fraction, 10),
        "missingPointInTimeStatusRows": missing_status_rows,
        "wrongAdjustmentRows": wrong_adjustment_rows,
        "checks": checks,
        "historicalResearchEligible": all(checks.values()),
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(resolve_path(config, "auditRoot") / "latest_data_audit.json", payload)
    return payload


def _login() -> Any:
    import baostock as bs

    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {result.error_code} {result.error_msg}")
    return bs


def backfill(config: dict[str, Any], args: argparse.Namespace, bs: Any) -> dict[str, Any]:
    master = read_jsonl(resolve_path(config, "latestMaster"))
    if not master:
        raise RuntimeError("PIT master missing; run --mode master first")
    codes = {value.strip().upper() for value in args.codes.split(",") if value.strip()}
    if codes:
        master = [
            row
            for row in master
            if row["securityId"].upper() in codes or row["stockCode"] in codes
        ]
    if args.max_codes:
        master = master[: args.max_codes]
    end_date = args.end_date or date.today().isoformat()
    results: list[dict[str, Any]] = []
    started = now_iso()
    for number, security in enumerate(master, start=1):
        result = collect_one(bs, security, config, end_date, args.force, args.retries)
        results.append(result)
        if number == 1 or number % max(1, args.progress_every) == 0 or result["status"] == "failed":
            print(json.dumps({"progress": f"{number}/{len(master)}", **result}, ensure_ascii=False), flush=True)
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    summary = {
        "schemaVersion": "ashare_pit_adjusted_collection_summary_v1",
        "status": "research_only",
        "startedAt": started,
        "completedAt": now_iso(),
        "requested": len(master),
        "endDate": end_date,
        "force": bool(args.force),
        "statusCounts": counts,
        "failedSecurityIds": [row["securityId"] for row in results if row["status"] == "failed"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(resolve_path(config, "auditRoot") / "latest_collection_summary.json", summary)
    append_jsonl(resolve_path(config, "manifest"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("master", "backfill", "audit", "all"), default="audit")
    parser.add_argument("--codes", default="")
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_config(config)
    bs = None
    try:
        if args.mode in {"master", "backfill", "all"}:
            bs = _login()
        if args.mode in {"master", "all"}:
            master = fetch_master(config)
            path = save_master(master, config)
            print(json.dumps({
                "savedMaster": str(path),
                "masterCount": len(master),
                "current": sum(int(row["currentDiscoverable"]) for row in master),
                "delisted": sum(int(row["delisted"]) for row in master),
            }, ensure_ascii=False, indent=2))
        if args.mode in {"backfill", "all"}:
            summary = backfill(config, args, bs)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.mode in {"audit", "all"}:
            print(json.dumps(audit(config), ensure_ascii=False, indent=2))
    finally:
        if bs is not None:
            bs.logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
