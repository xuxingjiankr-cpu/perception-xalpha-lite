"""Collect a current SH/SZ/BJ A-share master and raw daily research bars.

This is a market-data utility only.  It never imports a trading agent, reads an
account, writes a strategy overlay, or submits an order.

SH/SZ security discovery and daily bars use the local mootdx/TDX client.  BJ
security discovery uses the public Beijing Stock Exchange table exposed by
AkShare, and BJ daily bars use Sina's public K-line endpoint because mootdx does
not support the Beijing market.

The master is a *current discoverable universe*, not a historical point-in-time
constituent database.  Delisted securities are therefore incomplete and every
audit/report records that survivorship limitation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "market" / "ashare_research"
MASTER_ROOT = DATA_ROOT / "master"
BARS_ROOT = DATA_ROOT / "bars_1d_raw"
# Preregistered: the unattended weekly run continues on isolated symbol failures and aborts
# only on a coverage collapse. 5 unreachable BJ symbols out of 5,539 must not stop mining.
MINIMUM_BACKFILL_COVERAGE = 0.995
LATEST_SESSION_CACHE: list[str | None] = [None]


def latest_expected_session(today: date | None = None) -> str:
    """Most recent CLOSED XSHG session -- the only defensible freshness target.

    Using the calendar date makes every weekend and holiday look stale, so the collector
    re-requested all 5,539 symbols on Saturday 2026-08-01. Falls back to a weekday walk-back
    if the calendar package is unavailable, which is still strictly better than today().
    """
    moment = today or date.today()
    try:
        import exchange_calendars as xcals
        import pandas as pd

        calendar = xcals.get_calendar("XSHG")
        stamp = pd.Timestamp(moment.isoformat())
        session = calendar.previous_close(stamp + pd.Timedelta(days=1)).date()
        return session.isoformat()
    except Exception:
        probe = moment
        while probe.weekday() >= 5:
            probe -= timedelta(days=1)
        return probe.isoformat()
OUTPUT_ROOT = ROOT / "outputs" / "edge_research" / "ashare_data_audit"
LATEST_MASTER = MASTER_ROOT / "ashare_master_latest.jsonl"
MANIFEST = DATA_ROOT / "collection_manifest.jsonl"

PAGE = 800
TDX_DAILY_FREQUENCY = 9
SINA_ENDPOINT = (
    "https://quotes.sina.cn/cn/api/openapi.php/"
    "CN_MarketDataService.getKLineData"
)

SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
BJ_PREFIXES = ("43", "83", "87", "88", "92")
_THREAD_LOCAL = threading.local()


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
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
        for row in rows
    )
    atomic_text(path, text)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            + "\n"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def classify_security(exchange: str, code: str) -> tuple[str, str] | None:
    code = str(code).zfill(6)
    if exchange == "SH" and code.startswith(SH_PREFIXES):
        return ("STAR" if code.startswith(("688", "689")) else "SH_MAIN", "SH")
    if exchange == "SZ" and code.startswith(SZ_PREFIXES):
        return (
            "CHINEXT" if code.startswith(("300", "301")) else "SZ_MAIN",
            "SZ",
        )
    if exchange == "BJ" and code.startswith(BJ_PREFIXES):
        return ("BSE", "BJ")
    return None


def normalize_name(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def discover_sh_sz() -> list[dict[str, Any]]:
    from mootdx.quotes import Quotes

    client = Quotes.factory(market="std")
    rows: list[dict[str, Any]] = []
    for market_number, exchange in ((0, "SZ"), (1, "SH")):
        frame = client.stocks(market=market_number)
        for item in frame.to_dict("records"):
            code = str(item.get("code") or "").zfill(6)
            classification = classify_security(exchange, code)
            if classification is None:
                continue
            board, _ = classification
            rows.append(
                {
                    "securityId": f"{exchange}.{code}",
                    "exchange": exchange,
                    "marketNumber": market_number,
                    "stockCode": code,
                    "name": normalize_name(item.get("name")),
                    "board": board,
                    "listingDate": None,
                    "industry": None,
                    "discoverySource": "mootdx_tdx_current_master",
                    "currentDiscoverable": True,
                }
            )
    return rows


def discover_bj() -> list[dict[str, Any]]:
    import akshare as ak

    frame = ak.stock_info_bj_name_code()
    rows: list[dict[str, Any]] = []
    for item in frame.to_dict("records"):
        code = str(item.get("证券代码") or "").zfill(6)
        if classify_security("BJ", code) is None:
            continue
        listing = item.get("上市日期")
        rows.append(
            {
                "securityId": f"BJ.{code}",
                "exchange": "BJ",
                "marketNumber": 2,
                "stockCode": code,
                "name": normalize_name(item.get("证券简称")),
                "board": "BSE",
                "listingDate": (
                    listing.date().isoformat()
                    if hasattr(listing, "date")
                    else str(listing)[:10] if listing not in (None, "") else None
                ),
                "industry": normalize_name(item.get("所属行业")) or None,
                "region": normalize_name(item.get("地区")) or None,
                "discoverySource": "beijing_stock_exchange_via_akshare",
                "currentDiscoverable": True,
            }
        )
    return rows


def build_master(include_bj: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = discover_sh_sz()
    skipped: list[dict[str, Any]] = []
    if include_bj:
        try:
            rows.extend(discover_bj())
        except Exception as exc:
            skipped.append(
                {
                    "component": "BJ",
                    "reason": "master_provider_failed",
                    "detail": str(exc)[:300],
                }
            )
    deduplicated = {row["securityId"]: row for row in rows}
    ordered = sorted(
        deduplicated.values(),
        key=lambda row: (row["exchange"], row["stockCode"]),
    )
    generated = now_iso()
    for row in ordered:
        row["masterGeneratedAt"] = generated
        row["universeDefinition"] = "current_discoverable_a_shares"
        row["pointInTimeMembership"] = False
        row["researchOnly"] = True
    counts: dict[str, int] = {}
    for row in ordered:
        counts[row["exchange"]] = counts.get(row["exchange"], 0) + 1
    audit = {
        "schemaVersion": "ashare_research_master_audit_v1",
        "status": "research_only",
        "generatedAt": generated,
        "securityCount": len(ordered),
        "exchangeCounts": counts,
        "duplicateSecurityIds": len(rows) - len(ordered),
        "skipped": skipped,
        "survivorshipWarning": (
            "This is the current discoverable SH/SZ/BJ A-share universe. "
            "Historically delisted securities and point-in-time ST status are incomplete."
        ),
        "orders": [],
        "automaticTradingChanges": [],
    }
    return ordered, audit


def save_master(rows: list[dict[str, Any]], audit: dict[str, Any]) -> Path:
    stamp = date.today().strftime("%Y%m%d")
    dated = MASTER_ROOT / f"ashare_master_{stamp}.jsonl"
    atomic_jsonl(dated, rows)
    atomic_jsonl(LATEST_MASTER, rows)
    atomic_json(OUTPUT_ROOT / "latest_master_audit.json", audit)
    return dated


def thread_tdx_client() -> Any:
    client = getattr(_THREAD_LOCAL, "tdx_client", None)
    if client is None:
        from mootdx.quotes import Quotes

        client = Quotes.factory(market="std")
        _THREAD_LOCAL.tdx_client = client
    return client


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fetch_tdx_daily(
    security: dict[str, Any], max_pages: int, retries: int
) -> list[dict[str, Any]]:
    client = thread_tdx_client()
    raw: list[dict[str, Any]] = []
    code = security["stockCode"]
    for page in range(max_pages):
        frame = None
        error: Exception | None = None
        for attempt in range(retries):
            try:
                frame = client.bars(
                    symbol=code,
                    frequency=TDX_DAILY_FREQUENCY,
                    offset=PAGE,
                    start=page * PAGE,
                )
                error = None
                break
            except Exception as exc:
                error = exc
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
        if error is not None:
            raise error
        if frame is None or frame.empty:
            break
        for dt, row in frame.iterrows():
            raw.append(
                {
                    "dt": str(dt)[:10],
                    "open": as_float(row.get("open")),
                    "high": as_float(row.get("high")),
                    "low": as_float(row.get("low")),
                    "close": as_float(row.get("close")),
                    "vol": as_float(row.get("vol")),
                    "amount": as_float(row.get("amount")),
                    "amountSource": "exchange_reported_via_tdx",
                    "volumeUnit": "hands_100_shares",
                    "adjustment": "none_raw",
                    "source": "mootdx_tdx",
                }
            )
        if len(frame) < PAGE:
            break
    return raw


def fetch_sina_daily(
    security: dict[str, Any], maximum_rows: int, timeout_seconds: float
) -> list[dict[str, Any]]:
    symbol = security["exchange"].lower() + security["stockCode"]
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "scale": "240",
            "ma": "no",
            "datalen": str(maximum_rows),
        }
    )
    request = urllib.request.Request(
        SINA_ENDPOINT + "?" + params,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn/",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    source_rows = ((payload.get("result") or {}).get("data") or [])
    rows: list[dict[str, Any]] = []
    for item in source_rows:
        open_px = as_float(item.get("open"))
        high_px = as_float(item.get("high"))
        low_px = as_float(item.get("low"))
        close_px = as_float(item.get("close"))
        volume = as_float(item.get("volume"))
        typical = (open_px + high_px + low_px + close_px) / 4.0
        rows.append(
            {
                "dt": str(item.get("day") or "")[:10],
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "vol": volume,
                "amount": typical * volume,
                "amountSource": "ohlc4_times_volume_estimate",
                "volumeUnit": "shares",
                "adjustment": "none_raw",
                "source": "sina_public_kline",
            }
        )
    return rows


def valid_daily_row(row: dict[str, Any]) -> bool:
    values = [as_float(row.get(key)) for key in ("open", "high", "low", "close")]
    if not row.get("dt") or min(values) <= 0:
        return False
    open_px, high_px, low_px, close_px = values
    return (
        high_px >= max(open_px, low_px, close_px)
        and low_px <= min(open_px, high_px, close_px)
        and as_float(row.get("vol")) >= 0
        and as_float(row.get("amount")) >= 0
    )


def filter_dates(
    rows: list[dict[str, Any]], start_date: str | None, end_date: str | None
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        dt = str(row.get("dt") or "")[:10]
        if start_date and dt < start_date:
            continue
        if end_date and dt > end_date:
            continue
        if valid_daily_row(row):
            selected[dt] = row
    return [selected[key] for key in sorted(selected)]


def collect_one(
    security: dict[str, Any],
    max_pages: int,
    retries: int,
    bj_maximum_rows: int,
    timeout_seconds: float,
    start_date: str | None,
    end_date: str | None,
    force: bool,
) -> dict[str, Any]:
    path = BARS_ROOT / (
        f"{security['exchange']}_{security['stockCode']}.jsonl"
    )
    old = load_jsonl(path) if path.exists() else []
    if old and not force:
        last = str(old[-1].get("dt") or "")[:10] if old else None
        # Resolved once per backfill and passed in; recomputing per symbol reloaded the
        # exchange calendar thousands of times per run.
        freshness_target = end_date or LATEST_SESSION_CACHE[0] or latest_expected_session()
        if last and last >= freshness_target:
            return {
                "securityId": security["securityId"],
                "status": "skipped_current",
                "rows": len(old),
                "first": old[0].get("dt") if old else None,
                "last": last,
                "path": str(path),
            }
    try:
        if security["exchange"] in {"SH", "SZ"}:
            pages = 1 if old and not force else max_pages
            rows = fetch_tdx_daily(security, pages, retries)
        else:
            rows = fetch_sina_daily(
                security, bj_maximum_rows, timeout_seconds
            )
        rows = filter_dates(old + rows, start_date, end_date)
        for row in rows:
            row.update(
                {
                    "securityId": security["securityId"],
                    "exchange": security["exchange"],
                    "stockCode": security["stockCode"],
                    "board": security["board"],
                    "researchOnly": True,
                }
            )
        # A vendor outage returns zero rows, and filter_dates applies to what is
        # PERSISTED rather than to what is fetched, so both paths previously wrote a
        # shorter file over a longer one and silently destroyed history. On
        # 2026-09-12 that emptied 2250 of 5566 symbol files while the TDX endpoint
        # was returning nothing for every SH/SZ name. Never shrink an existing file
        # unless --force says to.
        if old and len(rows) < len(old) and not force:
            return {
                "securityId": security["securityId"],
                "status": "refused_shrinking_overwrite",
                "rows": len(old),
                "fetchedRows": len(rows),
                "first": old[0].get("dt"),
                "last": old[-1].get("dt"),
                "path": str(path),
            }
        atomic_jsonl(path, rows)
        return {
            "securityId": security["securityId"],
            "status": "ok" if rows else "no_rows",
            "rows": len(rows),
            "first": rows[0]["dt"] if rows else None,
            "last": rows[-1]["dt"] if rows else None,
            "path": str(path),
        }
    except Exception as exc:
        return {
            "securityId": security["securityId"],
            "status": "failed",
            "rows": 0,
            "error": str(exc)[:500],
            "path": str(path),
        }


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    LATEST_SESSION_CACHE[0] = args.end_date or latest_expected_session()
    master = load_jsonl(LATEST_MASTER)
    if not master:
        raise RuntimeError("A-share master is missing; run --mode master first")
    if args.exchanges:
        allowed = {
            value.strip().upper()
            for value in args.exchanges.split(",")
            if value.strip()
        }
        master = [row for row in master if row["exchange"] in allowed]
    if args.codes:
        codes = {
            value.strip().zfill(6)
            for value in args.codes.split(",")
            if value.strip()
        }
        master = [row for row in master if row["stockCode"] in codes]
    if args.max_codes:
        master = master[: args.max_codes]
    BARS_ROOT.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                collect_one,
                security,
                args.max_pages,
                args.retries,
                args.bj_maximum_rows,
                args.timeout_seconds,
                args.start_date,
                args.end_date,
                args.force,
            ): security
            for security in master
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {"progress": f"{index}/{len(futures)}", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    counts: dict[str, int] = {}
    for result in results:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "schemaVersion": "ashare_daily_collection_summary_v1",
        "status": "research_only",
        "startedAt": started,
        "completedAt": now_iso(),
        "requested": len(master),
        "statusCounts": counts,
        "results": sorted(results, key=lambda row: row["securityId"]),
        "adjustment": "none_raw",
        "survivorshipWarning": (
            "Collection follows the current master and therefore omits some "
            "historically delisted securities."
        ),
        "orders": [],
        "automaticTradingChanges": [],
    }
    # Coverage must live in the persisted artefact: it was previously computed in main()
    # AFTER this write, so the summary on disk never carried it and an unattended run left
    # no durable record of how complete the collection actually was.
    failed = int(counts.get("failed", 0))
    attempted = sum(int(value) for value in counts.values()) or 1
    coverage = 1.0 - failed / attempted
    summary["freshnessTarget"] = LATEST_SESSION_CACHE[0]
    summary["coverage"] = round(coverage, 6)
    summary["minimumCoverage"] = MINIMUM_BACKFILL_COVERAGE
    summary["failedSecurityIds"] = sorted(
        row["securityId"] for row in results if row.get("status") == "failed"
    )[:200]
    summary["collectorStatus"] = (
        "ok" if failed == 0
        else "degraded_within_tolerance" if coverage >= MINIMUM_BACKFILL_COVERAGE
        else "insufficient_coverage"
    )
    atomic_json(OUTPUT_ROOT / "latest_collection_summary.json", summary)
    append_jsonl(MANIFEST, {key: value for key, value in summary.items() if key != "results"})
    return summary


def audit() -> dict[str, Any]:
    master = load_jsonl(LATEST_MASTER)
    by_exchange: dict[str, int] = {}
    files_by_exchange: dict[str, int] = {}
    row_counts: list[int] = []
    estimated_amount_files = 0
    for row in master:
        exchange = str(row.get("exchange"))
        by_exchange[exchange] = by_exchange.get(exchange, 0) + 1
        path = BARS_ROOT / f"{exchange}_{row.get('stockCode')}.jsonl"
        if not path.exists():
            continue
        files_by_exchange[exchange] = files_by_exchange.get(exchange, 0) + 1
        rows = load_jsonl(path)
        row_counts.append(len(rows))
        if any(
            item.get("amountSource") == "ohlc4_times_volume_estimate"
            for item in rows[:1]
        ):
            estimated_amount_files += 1
    covered = sum(files_by_exchange.values())
    coverage = covered / len(master) if master else 0.0
    payload = {
        "schemaVersion": "ashare_daily_data_audit_v1",
        "status": "diagnostic_only_research_only",
        "generatedAt": now_iso(),
        "masterCount": len(master),
        "masterByExchange": by_exchange,
        "barFileCount": covered,
        "barFilesByExchange": files_by_exchange,
        "masterCoverage": round(coverage, 8),
        "minimumRows": min(row_counts) if row_counts else 0,
        "medianRows": (
            sorted(row_counts)[len(row_counts) // 2] if row_counts else 0
        ),
        "estimatedAmountFiles": estimated_amount_files,
        "historicalValidationEligible": bool(
            master and coverage >= 0.95 and all(by_exchange.get(x, 0) > 0 for x in ("SH", "SZ", "BJ"))
        ),
        "limitations": [
            "current-master survivorship bias",
            "historical point-in-time ST status is unavailable",
            "raw prices are unadjusted; corporate-action jumps require filtering",
            "BJ amount is an OHLC4 x volume estimate when the source omits amount",
        ],
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(OUTPUT_ROOT / "latest_data_audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("master", "backfill", "audit", "all"), default="audit"
    )
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--codes", default="")
    parser.add_argument("--exchanges", default="")
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--bj-maximum-rows", type=int, default=1023)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"master", "all"}:
        rows, master_audit = build_master(include_bj=not args.exclude_bj)
        path = save_master(rows, master_audit)
        print(
            json.dumps(
                {"savedMaster": str(path), **master_audit},
                ensure_ascii=False,
                indent=2,
            )
        )
        if master_audit["skipped"]:
            return 2
    if args.mode in {"backfill", "all"}:
        summary = backfill(args)
        print(json.dumps({
            "backfillCoverage": summary["coverage"],
            "collectorStatus": summary["collectorStatus"],
            "freshnessTarget": summary["freshnessTarget"],
            "failed": int(summary["statusCounts"].get("failed", 0)),
        }, ensure_ascii=False))
        # A handful of unreachable symbols (typically BJ timeouts) must not abort the whole
        # unattended weekly run: on 2026-08-01 five failures out of 5,539 returned exit 3 and
        # the miner never started. Only a genuine coverage collapse is fatal.
        if float(summary["coverage"]) < MINIMUM_BACKFILL_COVERAGE:
            return 3
    if args.mode in {"audit", "all"}:
        print(json.dumps(audit(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
