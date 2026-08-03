"""Collect point-in-time A-share financial indicators for offline factor research.

The collector uses AkShare's Eastmoney financial-analysis endpoint because it
returns both NOTICE_DATE and UPDATE_DATE.  It writes one atomic UTF-8 JSONL file
per security and can resume safely after interruption.  It never imports or
touches a trading agent.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data/market/ashare_research/master/ashare_master_latest.jsonl"
OUTPUT_ROOT = ROOT / "data/market/ashare_research/fundamentals_pit"
MANIFEST = OUTPUT_ROOT / "collection_manifest.jsonl"
SUMMARY = OUTPUT_ROOT / "latest_collection_summary.json"

FIELD_MAP = {
    "EPSJB": "epsYtd",
    "BPS": "bookValuePerShare",
    "TOTALOPERATEREVE": "revenueYtd",
    "PARENTNETPROFIT": "parentNetProfitYtd",
    "TOTALOPERATEREVETZ": "revenueYoyPct",
    "PARENTNETPROFITTZ": "netProfitYoyPct",
    "ROEJQ": "roePct",
    "ROIC": "roicPct",
    "XSMLL": "grossMarginPct",
    "XSJLL": "netMarginPct",
    "ZCFZL": "debtAssetRatioPct",
    "LD": "currentRatio",
    "SD": "quickRatio",
    "JYXJLYYSR": "operatingCashToRevenue",
    "XJLLB": "operatingCashToNetProfit",
    "YSZKZZTS": "receivableTurnoverDays",
    "CHZZTS": "inventoryTurnoverDays",
    "ZZCZZTS": "totalAssetTurnover",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def append_manifest(payload: dict[str, Any], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_master(path: Path = MASTER) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("securityId") and row.get("stockCode") and row.get("exchange"):
            rows.append(row)
    return sorted(rows, key=lambda row: str(row["securityId"]))


def as_number(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def as_date(value: Any) -> str | None:
    stamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(stamp) else stamp.date().isoformat()


def normalize_frame(frame: pd.DataFrame, security: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        report_date = as_date(raw.get("REPORT_DATE"))
        notice_date = as_date(raw.get("NOTICE_DATE"))
        update_date = as_date(raw.get("UPDATE_DATE"))
        # Missing disclosure dates are rejected rather than guessed from quarter end.
        if not report_date or not notice_date:
            continue
        row: dict[str, Any] = {
            "schemaVersion": "ashare_fundamental_pit_v1",
            "status": "research_only_not_trading",
            "securityId": security["securityId"],
            "exchange": security["exchange"],
            "stockCode": str(security["stockCode"]).zfill(6),
            "name": security.get("name"),
            "board": security.get("board"),
            "reportDate": report_date,
            "noticeDate": notice_date,
            "updateDate": update_date or notice_date,
            "reportType": raw.get("REPORT_TYPE"),
            "currency": raw.get("CURRENCY"),
            "source": "akshare_eastmoney_financial_analysis_indicator",
            "collectedAt": now_iso(),
        }
        for source, target in FIELD_MAP.items():
            row[target] = as_number(raw.get(source))
        rows.append(row)
    rows.sort(key=lambda row: (row["noticeDate"], row["reportDate"], row["updateDate"]))
    return rows


def collect_one(
    security: dict[str, Any],
    retries: int,
    output_root: Path = OUTPUT_ROOT,
) -> tuple[str, int, str | None]:
    import akshare as ak

    symbol = f"{str(security['stockCode']).zfill(6)}.{security['exchange']}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frame = ak.stock_financial_analysis_indicator_em(
                symbol=symbol, indicator="按报告期"
            )
            rows = normalize_frame(frame, security)
            if not rows:
                raise RuntimeError("endpoint returned no rows with a disclosure date")
            path = output_root / (
                f"{security['exchange']}_"
                f"{str(security['stockCode']).zfill(6)}.jsonl"
            )
            atomic_jsonl(path, rows)
            return str(security["securityId"]), len(rows), None
        except Exception as error:  # isolated public-endpoint failure
            last_error = error
            if attempt < retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    return str(security["securityId"]), 0, f"{type(last_error).__name__}: {last_error}"


def run_collect(args: argparse.Namespace) -> int:
    master_path = args.master_path.resolve()
    output_root = args.output_root.resolve()
    master = load_master(master_path)
    full_master = list(master)
    if args.max_symbols:
        master = master[: args.max_symbols]
    if args.resume and not args.force:
        master = [
            row
            for row in master
            if not (
                output_root
                / f"{row['exchange']}_{str(row['stockCode']).zfill(6)}.jsonl"
            ).exists()
        ]
    started = now_iso()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(collect_one, row, args.retries, output_root): row
            for row in master
        }
        for index, future in enumerate(as_completed(futures), start=1):
            security_id, count, error = future.result()
            if error:
                failures.append({"securityId": security_id, "reason": error})
            else:
                successes.append({"securityId": security_id, "statementRows": count})
            if index % 25 == 0 or index == len(futures):
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "requested": len(futures),
                            "succeeded": len(successes),
                            "failed": len(failures),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    total_files = len(list(output_root.glob("??_*.jsonl")))
    master_files = sum(
        int(
            (
                output_root
                / f"{row['exchange']}_{str(row['stockCode']).zfill(6)}.jsonl"
            ).exists()
        )
        for row in full_master
    )
    summary = {
        "schemaVersion": "ashare_fundamental_collection_summary_v1",
        "status": "research_only_not_trading",
        "startedAt": started,
        "finishedAt": now_iso(),
        "masterPath": str(master_path),
        "outputRoot": str(output_root),
        "masterCount": len(full_master),
        "requestedThisRun": len(master),
        "succeededThisRun": len(successes),
        "failedThisRun": len(failures),
        "totalSymbolFiles": total_files,
        "totalSelectedMasterFiles": master_files,
        "coverageOfSelectedMaster": round(
            master_files / max(1, len(full_master)), 8
        ),
        "failures": failures,
        "availabilityRule": "first market date strictly after max(noticeDate, updateDate)",
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(output_root / args.summary_name, summary)
    append_manifest(summary, output_root / "collection_manifest.jsonl")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 3


def run_audit(args: argparse.Namespace) -> int:
    master_path = args.master_path.resolve()
    output_root = args.output_root.resolve()
    master = load_master(master_path)
    master_count = len(master)
    files = [
        output_root
        / f"{row['exchange']}_{str(row['stockCode']).zfill(6)}.jsonl"
        for row in master
    ]
    files = [path for path in files if path.exists()]
    statement_rows = 0
    invalid_json = 0
    missing_notice = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            statement_rows += 1
            missing_notice += int(not bool(row.get("noticeDate")))
    audit = {
        "schemaVersion": "ashare_fundamental_collection_audit_v1",
        "status": "research_only_not_trading",
        "masterPath": str(master_path),
        "outputRoot": str(output_root),
        "masterCount": master_count,
        "symbolFiles": len(files),
        "coverage": round(len(files) / max(1, master_count), 8),
        "statementRows": statement_rows,
        "invalidJsonRows": invalid_json,
        "missingNoticeDateRows": missing_notice,
        "pointInTimeEligible": bool(
            master_count > 0
            and len(files) == master_count
            and invalid_json == 0
            and missing_notice == 0
        ),
        "orders": [],
        "automaticTradingChanges": [],
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["pointInTimeEligible"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "audit"), default="collect")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--master-path", type=Path, default=MASTER)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-name", default=SUMMARY.name)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Path(args.summary_name).name != args.summary_name or not args.summary_name.endswith(
        ".json"
    ):
        raise ValueError("summary-name must be a plain JSON filename")
    if not args.master_path.exists():
        raise FileNotFoundError(args.master_path)
    return run_audit(args) if args.mode == "audit" else run_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
