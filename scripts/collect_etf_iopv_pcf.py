"""Collect point-in-time ETF IOPV/premium snapshots and official SSE PCF files.

Research-only market-data pipeline:

* confirmed non-money T0 universe comes from the audited local master;
* SSE PCF XML files are downloaded from the official exchange endpoint once per
  trading day and preserved byte-for-byte with a parsed manifest;
* public vendor field ``f131`` is recorded as a vendor IOPV value, with its field
  identity and validation status explicit; it is never presented as an official
  direct-feed value;
* market price, bid, ask, vendor IOPV and premium are stored with source quote
  timestamps and freshness checks.

There are no account, position, order, cancel, strategy-config or overlay calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from collect_l2_depth import (
    CST,
    in_continuous_session,
    is_xshg_session,
    load_confirmed_t0_universe,
    now_cn,
    universe_fingerprint,
)
from run_etf_paper_trading_agent import ROOT


DATA_ROOT = ROOT / "data" / "research"
IOPV_DIR = DATA_ROOT / "etf_iopv"
PCF_DIR = DATA_ROOT / "etf_pcf"
COVERAGE_DIR = ROOT / "outputs" / "etf_iopv_pcf"
EASTMONEY_TOKEN = "fa5fd1943c7b386f172d6893dbfba10b"
EASTMONEY_ENDPOINTS = (
    "https://push2.eastmoney.com/api/qt/ulist.np/get",
    "https://push2his.eastmoney.com/api/qt/ulist.np/get",
)
SSE_PCF_ENDPOINT = "https://query.sse.com.cn/etfDownload/downloadETF2Bulletin.do"
DEFAULT_MAX_QUOTE_AGE_SECONDS = 180
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def eastmoney_secid(row: dict[str, Any]) -> str:
    market = "1" if row["exchange"] == "SH" else "0"
    return f"{market}.{row['code']}"


def source_quote_time(value: Any) -> datetime | None:
    try:
        stamp = int(float(value))
    except (TypeError, ValueError):
        return None
    if stamp <= 0:
        return None
    return datetime.fromtimestamp(stamp, CST)


def parse_iopv_items(
    items: list[dict[str, Any]],
    universe: list[dict[str, Any]],
    collected_at: datetime,
    *,
    max_quote_age_seconds: int = DEFAULT_MAX_QUOTE_AGE_SECONDS,
    pcf_manifest: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    metadata = {row["code"]: row for row in universe}
    pcf = pcf_manifest or {}
    parsed: list[dict[str, Any]] = []
    for item in items:
        code = str(item.get("f12") or "").zfill(6)
        meta = metadata.get(code)
        if meta is None:
            continue
        current = as_float(item.get("f2"))
        bid = as_float(item.get("f31"))
        ask = as_float(item.get("f32"))
        previous_close = as_float(item.get("f18"))
        previous_nav = as_float(item.get("f130"))
        vendor_iopv = as_float(item.get("f131"))
        quote_time = source_quote_time(item.get("f124"))
        age = (
            (collected_at - quote_time).total_seconds()
            if quote_time is not None
            else None
        )
        fresh = bool(
            quote_time is not None
            and quote_time.date() == collected_at.date()
            and age is not None
            and -5 <= age <= max_quote_age_seconds
        )
        publish_flag = pcf.get(code, {}).get("publish_iopv")
        validation = (
            "vendor_field_with_official_sse_publish_flag"
            if meta["exchange"] == "SH" and publish_flag is True
            else "vendor_field_official_pcf_says_not_published"
            if meta["exchange"] == "SH" and publish_flag is False
            else "vendor_field_no_official_public_pcf_crosscheck"
        )

        def premium(price: float | None) -> float | None:
            if price is None or vendor_iopv is None or vendor_iopv <= 0:
                return None
            return round((price / vendor_iopv - 1.0) * 100.0, 6)

        parsed.append(
            {
                "schemaVersion": "etf_iopv_snapshot_v1",
                "collected_at": collected_at.isoformat(),
                "trade_date": collected_at.strftime("%Y-%m-%d"),
                "source_quote_time": quote_time.isoformat() if quote_time else None,
                "quote_age_seconds": round(age, 3) if age is not None else None,
                "is_fresh": fresh,
                "stockCode": code,
                "exchange": meta["exchange"],
                "name": meta.get("name") or item.get("f14"),
                "asset_class": meta.get("asset_class"),
                "current": current,
                "bid1": bid,
                "ask1": ask,
                "previous_close": previous_close,
                "previous_nav": previous_nav,
                "iopv": vendor_iopv,
                "iopv_source": "eastmoney_public_quote_field_f131",
                "iopv_source_tier": "vendor_not_direct_exchange_feed",
                "iopv_validation_status": validation,
                "official_sse_publish_iopv": publish_flag,
                "premium_pct": premium(current),
                "bid_premium_pct": premium(bid),
                "ask_premium_pct": premium(ask),
                "source": "eastmoney_public_quote",
            }
        )
    return parsed


def fetch_iopv(
    universe: list[dict[str, Any]],
    *,
    collected_at: datetime | None = None,
    batch_size: int = 50,
    timeout: float = 10.0,
    max_quote_age_seconds: int = DEFAULT_MAX_QUOTE_AGE_SECONDS,
    pcf_manifest: dict[str, dict[str, Any]] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    moment = collected_at or now_cn()
    all_items: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index in range(0, len(universe), max(1, batch_size)):
        batch = universe[index : index + max(1, batch_size)]
        params = urllib.parse.urlencode(
            {
                "fltt": "2",
                "invt": "2",
                "fields": "f2,f12,f13,f14,f18,f31,f32,f124,f130,f131",
                "secids": ",".join(eastmoney_secid(row) for row in batch),
                "ut": EASTMONEY_TOKEN,
                "_": str(int(moment.timestamp() * 1000)),
            }
        )
        success = False
        last_error: str | None = None
        for endpoint in EASTMONEY_ENDPOINTS:
            started = time.perf_counter()
            try:
                request = urllib.request.Request(
                    endpoint + "?" + params,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Referer": "https://quote.eastmoney.com/",
                    },
                )
                response = opener(request, timeout=timeout)
                payload = json.loads(response.read().decode("utf-8", "replace"))
                items = payload.get("data", {}).get("diff", [])
                if not isinstance(items, list):
                    items = []
                all_items.extend(item for item in items if isinstance(item, dict))
                reports.append(
                    {
                        "batch": index // max(1, batch_size) + 1,
                        "requested": len(batch),
                        "returned": len(items),
                        "host": urllib.parse.urlparse(endpoint).netloc,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "ok": bool(items),
                    }
                )
                if items:
                    success = True
                    break
            except Exception as exc:
                last_error = str(exc)
        if not success:
            reports.append(
                {
                    "batch": index // max(1, batch_size) + 1,
                    "requested": len(batch),
                    "returned": 0,
                    "ok": False,
                    "error": last_error or "no endpoint returned data",
                }
            )
    rows = parse_iopv_items(
        all_items,
        universe,
        moment,
        max_quote_age_seconds=max_quote_age_seconds,
        pcf_manifest=pcf_manifest,
    )
    deduplicated = {row["stockCode"]: row for row in rows}
    return [deduplicated[code] for code in sorted(deduplicated)], {"batches": reports}


def xml_text(root: ET.Element, tag: str) -> str | None:
    node = root.find(tag)
    return node.text.strip() if node is not None and node.text else None


def parse_sse_pcf(content: bytes) -> dict[str, Any]:
    root = ET.fromstring(content)
    components = root.findall("./ComponentList/Component")
    creation_rates = [
        as_float(xml_text(component, "CreationPremiumRate"))
        for component in components
    ]
    redemption_rates = [
        as_float(xml_text(component, "RedemptionDiscountRate"))
        for component in components
    ]
    substitution_counts: dict[str, int] = {}
    for component in components:
        flag = xml_text(component, "SubstitutionFlag") or "missing"
        substitution_counts[flag] = substitution_counts.get(flag, 0) + 1
    return {
        "fund_code": xml_text(root, "FundInstrumentID"),
        "trading_day": xml_text(root, "TradingDay"),
        "previous_trading_day": xml_text(root, "PreTradingDay"),
        "creation_redemption_unit": as_float(xml_text(root, "CreationRedemptionUnit")),
        "nav_per_creation_unit": as_float(xml_text(root, "NAVperCU")),
        "nav": as_float(xml_text(root, "NAV")),
        "previous_cash_component": as_float(xml_text(root, "PreCashComponent")),
        "estimated_cash_component": as_float(xml_text(root, "EstimatedCashComponent")),
        "max_cash_ratio": as_float(xml_text(root, "MaxCashRatio")),
        "creation_limit": as_float(xml_text(root, "CreationLimit")),
        "redemption_limit": as_float(xml_text(root, "RedemptionLimit")),
        "publish_iopv": xml_text(root, "PublishIOPVFlag") == "1",
        "creation_redemption_switch": xml_text(root, "CreationRedemptionSwitch"),
        "creation_redemption_mechanism": xml_text(root, "CreationRedemptionMechanism"),
        "record_number": int(as_float(xml_text(root, "RecordNumber")) or 0),
        "parsed_components": len(components),
        "substitution_flag_counts": dict(sorted(substitution_counts.items())),
        "max_creation_premium_rate": max(
            (value for value in creation_rates if value is not None),
            default=None,
        ),
        "max_redemption_discount_rate": max(
            (value for value in redemption_rates if value is not None),
            default=None,
        ),
    }


def download_sse_pcf(
    row: dict[str, Any],
    *,
    trade_date: str,
    timeout: float = 20.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    code = row["code"]
    url = SSE_PCF_ENDPOINT + "?" + urllib.parse.urlencode({"fundCode": code})
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.sse.com.cn/disclosure/fund/etflist/",
        },
    )
    try:
        response = opener(request, timeout=timeout)
        content = response.read()
        metadata = parse_sse_pcf(content)
        filename = f"ssepcf_{code}_{metadata.get('trading_day') or trade_date.replace('-', '')}.xml"
        filename = SAFE_FILENAME.sub("_", filename)
        path = PCF_DIR / trade_date / filename
        atomic_bytes(path, content)
        expected_day = trade_date.replace("-", "")
        return {
            "stockCode": code,
            "exchange": "SH",
            "asset_class": row.get("asset_class"),
            "ok": True,
            "fresh_for_requested_trade_date": metadata.get("trading_day") == expected_day,
            "file": str(path),
            "source_url": url,
            **metadata,
        }
    except Exception as exc:
        return {
            "stockCode": code,
            "exchange": "SH",
            "asset_class": row.get("asset_class"),
            "ok": False,
            "fresh_for_requested_trade_date": False,
            "error": str(exc),
            "source_url": url,
        }


def collect_pcf(
    universe: list[dict[str, Any]],
    trade_date: str,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    sse = [row for row in universe if row["exchange"] == "SH"]
    sz = [row for row in universe if row["exchange"] == "SZ"]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(
            executor.map(
                lambda row: download_sse_pcf(row, trade_date=trade_date),
                sse,
            )
        )
    manifest = {
        "schemaVersion": "etf_pcf_manifest_v1",
        "collected_at": now_cn().isoformat(),
        "trade_date": trade_date,
        "universe_fingerprint": universe_fingerprint(universe),
        "expected_sse_count": len(sse),
        "downloaded_sse_count": sum(row.get("ok") for row in results),
        "fresh_sse_count": sum(row.get("fresh_for_requested_trade_date") for row in results),
        "sz_count_without_official_public_download_endpoint": len(sz),
        "sz_status": "not_collected_no_stable_official_public_endpoint",
        "rows": sorted(results, key=lambda row: row["stockCode"]),
        "source": "Shanghai Stock Exchange official PCF download",
        "source_url": SSE_PCF_ENDPOINT,
        "status": "diagnostic_only",
    }
    atomic_json(PCF_DIR / trade_date / "manifest.json", manifest)
    return manifest


def manifest_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("rows") if isinstance(manifest.get("rows"), list) else []
    return {
        str(row.get("stockCode") or ""): row
        for row in rows
        if isinstance(row, dict) and row.get("stockCode")
    }


def collect_iopv_once(
    universe: list[dict[str, Any]],
    *,
    pcf_manifest: dict[str, Any],
    max_quote_age_seconds: int,
    batch_size: int,
    write: bool = True,
) -> dict[str, Any]:
    moment = now_cn()
    rows, provider = fetch_iopv(
        universe,
        collected_at=moment,
        batch_size=batch_size,
        max_quote_age_seconds=max_quote_age_seconds,
        pcf_manifest=manifest_lookup(pcf_manifest),
    )
    expected = {row["code"] for row in universe}
    returned = {row["stockCode"] for row in rows}
    fresh = {row["stockCode"] for row in rows if row.get("is_fresh")}
    usable = {
        row["stockCode"]
        for row in rows
        if row.get("is_fresh") and as_float(row.get("iopv")) not in (None, 0.0)
    }
    coverage = {
        "schemaVersion": "etf_iopv_coverage_v1",
        "collected_at": moment.isoformat(),
        "trade_date": moment.strftime("%Y-%m-%d"),
        "expected_count": len(expected),
        "returned_count": len(returned),
        "fresh_count": len(fresh),
        "usable_iopv_count": len(usable),
        "fresh_coverage_rate": round(len(fresh) / len(expected), 6) if expected else 0.0,
        "usable_iopv_rate": round(len(usable) / len(expected), 6) if expected else 0.0,
        "missing_codes": sorted(expected - returned),
        "stale_codes": sorted(returned - fresh),
        "missing_iopv_codes": sorted(fresh - usable),
        "provider": provider,
        "status": "ok" if usable == expected else "partial_coverage",
    }
    if write:
        day = moment.strftime("%Y-%m-%d")
        append_jsonl(IOPV_DIR / f"iopv_{day}.jsonl", [row for row in rows if row.get("is_fresh")])
        append_jsonl(COVERAGE_DIR / f"iopv_coverage_{day}.jsonl", [coverage])
    return coverage


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("both", "pcf", "iopv"), default="both")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--until", default="15:00")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--pcf-workers", type=int, default=4)
    parser.add_argument("--max-quote-age-seconds", type=int, default=DEFAULT_MAX_QUOTE_AGE_SECONDS)
    args = parser.parse_args()

    universe = load_confirmed_t0_universe()
    moment = now_cn()
    day = moment.strftime("%Y-%m-%d")
    if not is_xshg_session(moment.date()):
        print(
            json.dumps(
                {
                    "status": "skipped_non_trading_day",
                    "trade_date": day,
                    "universe_count": len(universe),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    manifest_path = PCF_DIR / day / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if args.mode in {"both", "pcf"} and not args.dry_run:
        manifest = collect_pcf(universe, day, workers=args.pcf_workers)
        print(
            json.dumps(
                {
                    "pcf_expected_sse": manifest["expected_sse_count"],
                    "pcf_downloaded_sse": manifest["downloaded_sse_count"],
                    "pcf_fresh_sse": manifest["fresh_sse_count"],
                    "pcf_sz_not_collected": manifest[
                        "sz_count_without_official_public_download_endpoint"
                    ],
                },
                ensure_ascii=False,
            )
        )

    if args.mode == "pcf":
        return 0

    def poll() -> dict[str, Any]:
        return collect_iopv_once(
            universe,
            pcf_manifest=manifest,
            max_quote_age_seconds=args.max_quote_age_seconds,
            batch_size=args.batch_size,
            write=not args.dry_run,
        )

    if args.once:
        current = now_cn()
        if not in_continuous_session(current):
            print(
                json.dumps(
                    {
                        "status": "skipped_outside_continuous_session",
                        "trade_date": day,
                        "universe_count": len(universe),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(json.dumps(poll(), ensure_ascii=False, indent=2))
        return 0

    end_hour, end_minute = (int(value) for value in args.until.split(":"))
    polls = 0
    while True:
        current = now_cn()
        if not is_xshg_session(current.date()):
            break
        if (current.hour, current.minute) >= (end_hour, end_minute):
            break
        if not in_continuous_session(current):
            time.sleep(min(30.0, max(1.0, args.poll_seconds)))
            continue
        started = time.perf_counter()
        try:
            result = poll()
            polls += 1
            print(
                f"{result['collected_at']} iopv={result['usable_iopv_count']}/"
                f"{result['expected_count']} fresh={result['fresh_count']} "
                f"status={result['status']}"
            )
        except Exception as exc:
            print(f"IOPV poll error: {exc}", file=sys.stderr)
        elapsed = time.perf_counter() - started
        time.sleep(max(1.0, args.poll_seconds - elapsed))
    print(json.dumps({"status": "completed", "polls": polls}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
