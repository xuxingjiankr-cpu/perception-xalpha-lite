"""
Eastmoney full-market data collector for paper-trading research.

This script is market-data only. It does not access Huatai paper-trading
account/order APIs, does not submit/cancel orders, and does not modify trading
agent state. It is intended to preserve Huatai quota by collecting quotes from
Eastmoney while keeping execution/account queries on the paper-trading API.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "market" / "eastmoney"
OUTPUT_ROOT = ROOT / "outputs" / "eastmoney_full_market"

EASTMONEY_TOKEN = "fa5fd1943c7b386f172d6893dbfba10b"
CN_TZ = ZoneInfo("Asia/Shanghai")

ENDPOINTS = [
    "https://push2.eastmoney.com/api/qt/clist/get",
    "https://push2his.eastmoney.com/api/qt/clist/get",
    "http://push2his.eastmoney.com/api/qt/clist/get",
]
KLINE_ENDPOINTS = [
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    "http://push2his.eastmoney.com/api/qt/stock/kline/get",
]

FS_MAP = {
    "ashare": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
    "etf": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
}

CLIST_FIELDS = [
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
    "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21",
    "f23", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38",
    "f39", "f40", "f41", "f42", "f43", "f44", "f45", "f46", "f47",
    "f48", "f49", "f50", "f57", "f58", "f62", "f66", "f69", "f72",
    "f75", "f78", "f81", "f84", "f87", "f100", "f102", "f103",
    "f104", "f105", "f124",
]

SNAPSHOT_COLUMNS = [
    "collected_at",
    "trade_date",
    "source_quote_time",
    "scope",
    "secid",
    "market",
    "stockCode",
    "name",
    "currentPrice",
    "change_pct",
    "change_abs",
    "volume",
    "amount",
    "amplitude_pct",
    "turnover_pct",
    "open",
    "high",
    "low",
    "prevClose",
    "bidPrice1",
    "askPrice1",
    "source",
]

MINUTE_COLUMNS = [
    "datetime",
    "secid",
    "market",
    "stockCode",
    "name",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude_pct",
    "change_pct",
    "change_abs",
    "turnover_pct",
    "source",
]


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "-"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def clean(value: Any, default: Any = None) -> Any:
    return default if value in (None, "", "-") else value


def eastmoney_quote_time(value: Any) -> str | None:
    try:
        ts = int(float(value))
    except Exception:
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, CN_TZ).isoformat()


def http_json(endpoints: list[str], params: dict[str, Any], timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = urllib.parse.urlencode(params)
    attempts: list[dict[str, Any]] = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }
    last_error: dict[str, Any] | None = None
    for endpoint in endpoints:
        parsed = urllib.parse.urlparse(endpoint)
        attempt = {"url_host": parsed.netloc, "scheme": parsed.scheme}
        try:
            req = urllib.request.Request(endpoint + "?" + encoded, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            attempt["ok"] = True
            attempts.append(attempt)
            return payload, {"attempts": attempts, "endpoint_used": endpoint, "ok": True}
        except Exception as exc:
            last_error = {"category": "network", "message": str(exc)}
            attempt.update({"ok": False, "error": last_error})
            attempts.append(attempt)
    return {}, {"attempts": attempts, "ok": False, "error": last_error or {"message": "all endpoints failed"}}


def market_session(now: datetime | None = None) -> dict[str, Any]:
    t = now or now_cn()
    is_weekday = t.weekday() < 5
    morning = (t.hour == 9 and t.minute >= 30) or (t.hour == 10) or (t.hour == 11 and t.minute <= 30)
    afternoon = (t.hour == 13) or (t.hour == 14) or (t.hour == 15 and t.minute == 0)
    return {
        "exchange_timezone": "Asia/Shanghai",
        "exchange_local_time": t.isoformat(),
        "trade_date": t.strftime("%Y-%m-%d"),
        "is_weekday": is_weekday,
        "in_regular_session": bool(is_weekday and (morning or afternoon)),
        "in_lunch_break": bool(is_weekday and ((t.hour == 11 and t.minute > 30) or t.hour == 12)),
    }


def scopes_for(scope: str) -> list[str]:
    if scope == "all":
        return ["ashare", "etf"]
    if scope in FS_MAP:
        return [scope]
    raise ValueError(f"unsupported scope: {scope}")


def secid_from_row(row: dict[str, Any]) -> str:
    code = str(row.get("f12") or "").zfill(6)
    market = infer_market(code, row.get("f13"))
    return f"{market}.{code}"


def infer_market(code: str, raw_market: Any = None) -> str:
    market = str(raw_market or "")
    if market in {"0", "1"}:
        return market
    zcode = str(code or "").zfill(6)
    if zcode.startswith(("5", "6", "9")):
        return "1"
    if zcode.startswith(("0", "1", "2", "3")):
        return "0"
    return market


def normalize_snapshot_row(row: dict[str, Any], scope: str, collected_at: str, trade_date: str) -> dict[str, Any]:
    return {
        "collected_at": collected_at,
        "trade_date": trade_date,
        "source_quote_time": eastmoney_quote_time(row.get("f124")),
        "scope": scope,
        "secid": secid_from_row(row),
        "market": infer_market(str(row.get("f12") or "").zfill(6), row.get("f13")),
        "stockCode": str(row.get("f12") or "").zfill(6),
        "name": clean(row.get("f14"), ""),
        "currentPrice": as_float(clean(row.get("f2"))),
        "change_pct": as_float(clean(row.get("f3"))),
        "change_abs": as_float(clean(row.get("f4"))),
        "volume": as_float(clean(row.get("f5"))),
        "amount": as_float(clean(row.get("f6"))),
        "amplitude_pct": as_float(clean(row.get("f7"))),
        "turnover_pct": as_float(clean(row.get("f8"))),
        "open": as_float(clean(row.get("f17"))),
        "high": as_float(clean(row.get("f15"))),
        "low": as_float(clean(row.get("f16"))),
        "prevClose": as_float(clean(row.get("f18"))),
        "bidPrice1": as_float(clean(row.get("f31"))),
        "askPrice1": as_float(clean(row.get("f32"))),
        "source": "eastmoney",
    }


def fetch_scope_snapshot(
    scope: str,
    page_size: int,
    timeout_seconds: float,
    max_pages: int = 0,
    page_retries: int = 3,
    retry_sleep_seconds: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"scope": scope, "page_size": page_size, "pages": []}
    total = None
    page = 1
    failed_pages: list[int] = []
    while True:
        params = {
            "pn": str(page),
            "pz": str(page_size),
            "po": "1",
            "np": "1",
            "ut": EASTMONEY_TOKEN,
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": FS_MAP[scope],
            "fields": ",".join(CLIST_FIELDS),
            "_": str(int(time.time() * 1000)),
        }
        payload: dict[str, Any] = {}
        req_meta: dict[str, Any] = {}
        diff: list[dict[str, Any]] = []
        page_attempts: list[dict[str, Any]] = []
        for attempt_no in range(1, page_retries + 1):
            payload, req_meta = http_json(ENDPOINTS, params, timeout_seconds)
            data = payload.get("data") or {}
            raw_diff = data.get("diff") or []
            diff = raw_diff if isinstance(raw_diff, list) else []
            page_attempts.append({
                "attempt_no": attempt_no,
                "row_count": len(diff),
                "request": req_meta,
            })
            if diff or req_meta.get("ok"):
                break
            if attempt_no < page_retries and retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
        data = payload.get("data") or {}
        if total is None:
            total = int(data.get("total") or 0)
            meta["total_reported"] = total
        rows.extend(diff)
        meta["pages"].append({
            "page": page,
            "row_count": len(diff),
            "request": req_meta,
            "page_attempts": page_attempts,
        })
        if not diff and not req_meta.get("ok"):
            failed_pages.append(page)
            if total is None:
                meta["stopped_on_failed_page"] = page
                break
            if max_pages and page >= max_pages:
                break
            if total is not None and page * page_size >= total:
                break
            page += 1
            continue
        if not diff:
            break
        if max_pages and page >= max_pages:
            meta["truncated_by_max_pages"] = True
            break
        if total is not None and len(rows) >= total:
            break
        page += 1
    meta["row_count"] = len(rows)
    meta["failed_pages"] = failed_pages
    meta["complete"] = bool(total is not None and len(rows) >= total and not failed_pages)
    return rows, meta


def write_csv_gz(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    session = market_session()
    if args.only_session and not session["in_regular_session"]:
        summary = {
            "task": "eastmoney_full_market_snapshot",
            "status": "skipped_outside_regular_session",
            "scope": args.scope,
            "session": session,
            "paper_trading_only": True,
            "live_ready": False,
            "formal_strategy_allowed": False,
        }
        write_json(OUTPUT_ROOT / "latest_snapshot_summary.json", summary)
        append_jsonl(OUTPUT_ROOT / "snapshot_runs.jsonl", summary)
        return summary

    collected_at = now_iso()
    trade_date = session["trade_date"]
    stamp = now_cn().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict[str, Any]] = []
    scope_meta: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scope in scopes_for(args.scope):
        raw_rows, meta = fetch_scope_snapshot(
            scope,
            args.page_size,
            args.timeout_seconds,
            args.max_pages,
            args.page_retries,
            args.retry_sleep_seconds,
        )
        scope_meta.append(meta)
        for raw in raw_rows:
            norm = normalize_snapshot_row(raw, scope, collected_at, trade_date)
            key = norm["secid"]
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(norm)

    out_dir = DATA_ROOT / "full_market" / "snapshots" / trade_date
    path = out_dir / f"eastmoney_full_market_{stamp}_{args.scope}.csv.gz"
    write_csv_gz(path, all_rows, SNAPSHOT_COLUMNS)
    summary = {
        "task": "eastmoney_full_market_snapshot",
        "status": "ok" if all_rows else "no_rows",
        "scope": args.scope,
        "row_count": len(all_rows),
        "output_file": str(path),
        "session": session,
        "scope_meta": scope_meta,
        "source": "eastmoney",
        "huatai_api_used": False,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "created_at": now_iso(),
    }
    write_json(OUTPUT_ROOT / "latest_snapshot_summary.json", summary)
    append_jsonl(OUTPUT_ROOT / "snapshot_runs.jsonl", summary)
    append_jsonl(DATA_ROOT / "full_market" / "snapshot_manifest.jsonl", summary)
    return summary


def discover_universe(
    scope: str,
    page_size: int,
    timeout_seconds: float,
    max_codes: int = 0,
    page_retries: int = 3,
    retry_sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sub_scope in scopes_for(scope):
        raw_rows, _ = fetch_scope_snapshot(
            sub_scope,
            page_size,
            timeout_seconds,
            page_retries=page_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        for raw in raw_rows:
            secid = secid_from_row(raw)
            code = str(raw.get("f12") or "").zfill(6)
            if not code or code == "000000":
                continue
            market = infer_market(code, raw.get("f13"))
            rows.append({
                "scope": sub_scope,
                "secid": secid,
                "market": market,
                "stockCode": code,
                "name": clean(raw.get("f14"), ""),
            })
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[row["secid"]] = row
    values = sorted(dedup.values(), key=lambda x: (x["scope"], x["secid"]))
    return values[:max_codes] if max_codes else values


def tracked_universe_from_configs() -> list[dict[str, Any]]:
    paths = [
        ROOT / "configs" / "t0_intraday_paper_agent.json",
        ROOT / "configs" / "etf_paper_trading_agent.json",
    ]
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for item in cfg.get("universe", []):
            code = str(item.get("stockCode") or "").zfill(6)
            exchange = str(item.get("exchange") or "SH").upper()
            market = "1" if exchange == "SH" else "0"
            out[f"{market}.{code}"] = {
                "scope": "tracked",
                "secid": f"{market}.{code}",
                "market": market,
                "stockCode": code,
                "name": item.get("name", ""),
            }
    return sorted(out.values(), key=lambda x: x["secid"])


def month_bounds(month: str) -> tuple[str, str]:
    y, m = month.split("-")
    year = int(y)
    mon = int(m)
    begin = f"{year:04d}{mon:02d}01"
    if mon == 12:
        end = f"{year:04d}1231"
    else:
        # End can safely be the first day of the next month minus API-side filtering.
        end = f"{year:04d}{mon + 1:02d}01"
    if month == now_cn().strftime("%Y-%m"):
        end = now_cn().strftime("%Y%m%d")
    return begin, end


def fetch_minute_kline(secid: str, begin: str, end: str, timeout_seconds: float) -> tuple[list[str], dict[str, Any]]:
    params = {
        "secid": secid,
        "klt": "1",
        "fqt": "0",
        "beg": begin,
        "end": end,
        "ut": EASTMONEY_TOKEN,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "_": str(int(time.time() * 1000)),
    }
    payload, meta = http_json(KLINE_ENDPOINTS, params, timeout_seconds)
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not isinstance(klines, list):
        klines = []
    meta.update({
        "secid": secid,
        "stockCode": data.get("code"),
        "name": data.get("name"),
        "row_count": len(klines),
    })
    return klines, meta


def parse_kline_row(line: str, sec: dict[str, Any]) -> dict[str, Any]:
    parts = line.split(",")
    while len(parts) < 11:
        parts.append("")
    return {
        "datetime": parts[0],
        "secid": sec["secid"],
        "market": sec["market"],
        "stockCode": sec["stockCode"],
        "name": sec.get("name", ""),
        "open": as_float(parts[1]),
        "close": as_float(parts[2]),
        "high": as_float(parts[3]),
        "low": as_float(parts[4]),
        "volume": as_float(parts[5]),
        "amount": as_float(parts[6]),
        "amplitude_pct": as_float(parts[7]),
        "change_pct": as_float(parts[8]),
        "change_abs": as_float(parts[9]),
        "turnover_pct": as_float(parts[10]),
        "source": "eastmoney",
    }


def universe_for_backfill(args: argparse.Namespace) -> list[dict[str, Any]]:
    universe_file = getattr(args, "universe_file", "")
    if universe_file:
        universe = read_jsonl(Path(universe_file))
    elif args.scope == "tracked":
        universe = tracked_universe_from_configs()
    else:
        universe = discover_universe(
            args.scope,
            args.page_size,
            args.timeout_seconds,
            page_retries=args.page_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
    if args.max_codes:
        universe = universe[:args.max_codes]
    return universe


def plan_backfill(args: argparse.Namespace) -> dict[str, Any]:
    universe = universe_for_backfill(args)
    begin, end = month_bounds(args.month)
    universe_file = DATA_ROOT / "universe" / f"eastmoney_universe_{args.scope}_{now_cn().strftime('%Y%m%d')}.jsonl"
    write_jsonl(universe_file, universe)
    summary = {
        "task": "eastmoney_minute_backfill_plan",
        "status": "ok",
        "scope": args.scope,
        "month": args.month,
        "begin": begin,
        "end": end,
        "security_count": len(universe),
        "universe_file": str(universe_file),
        "estimated_requests": len(universe),
        "sleep_seconds": args.sleep_seconds,
        "rough_min_runtime_seconds": round(len(universe) * args.sleep_seconds, 1),
        "sample": universe[:10],
        "source": "eastmoney",
        "huatai_api_used": False,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "created_at": now_iso(),
    }
    write_json(OUTPUT_ROOT / f"backfill_plan_{args.month}_{args.scope}.json", summary)
    return summary


def backfill_minute(args: argparse.Namespace) -> dict[str, Any]:
    universe = universe_for_backfill(args)
    begin, end = month_bounds(args.month)
    base_dir = DATA_ROOT / "minute" / args.month / args.scope
    ensure_dir(base_dir)
    runs: list[dict[str, Any]] = []
    ok = 0
    skipped = 0
    failed = 0
    rows_total = 0
    for idx, sec in enumerate(universe, start=1):
        out_file = base_dir / f"{sec['secid'].replace('.', '_')}_{sec['stockCode']}.csv.gz"
        if out_file.exists() and not args.overwrite:
            skipped += 1
            runs.append({"secid": sec["secid"], "status": "skipped_exists", "output_file": str(out_file)})
            continue
        klines, meta = fetch_minute_kline(sec["secid"], begin, end, args.timeout_seconds)
        if klines:
            rows = [parse_kline_row(x, sec) for x in klines]
            write_csv_gz(out_file, rows, MINUTE_COLUMNS)
            ok += 1
            rows_total += len(rows)
            status = "ok"
        else:
            failed += 1
            status = "no_rows"
        run = {
            "idx": idx,
            "total": len(universe),
            "secid": sec["secid"],
            "stockCode": sec["stockCode"],
            "name": sec.get("name"),
            "status": status,
            "row_count": meta.get("row_count", 0),
            "output_file": str(out_file) if status == "ok" else None,
            "request_meta": meta,
            "created_at": now_iso(),
        }
        runs.append(run)
        append_jsonl(OUTPUT_ROOT / f"backfill_runs_{args.month}_{args.scope}.jsonl", run)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    summary = {
        "task": "eastmoney_minute_backfill",
        "status": "ok" if ok else "no_successful_files",
        "scope": args.scope,
        "month": args.month,
        "begin": begin,
        "end": end,
        "security_count": len(universe),
        "ok_count": ok,
        "skipped_count": skipped,
        "failed_count": failed,
        "row_count": rows_total,
        "output_dir": str(base_dir),
        "source": "eastmoney",
        "huatai_api_used": False,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "created_at": now_iso(),
    }
    write_json(OUTPUT_ROOT / f"latest_backfill_summary_{args.month}_{args.scope}.json", summary)
    append_jsonl(DATA_ROOT / "minute" / "backfill_manifest.jsonl", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Eastmoney full-market quote snapshots and minute backfills.")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot")
    snap.add_argument("--scope", choices=["ashare", "etf", "all"], default="all")
    snap.add_argument("--page-size", type=int, default=1000)
    snap.add_argument("--max-pages", type=int, default=0)
    snap.add_argument("--timeout-seconds", type=float, default=8.0)
    snap.add_argument("--page-retries", type=int, default=3)
    snap.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    snap.add_argument("--only-session", action="store_true")

    plan = sub.add_parser("plan-backfill")
    plan.add_argument("--scope", choices=["tracked", "etf", "ashare", "all"], default="tracked")
    plan.add_argument("--month", default=now_cn().strftime("%Y-%m"))
    plan.add_argument("--page-size", type=int, default=1000)
    plan.add_argument("--timeout-seconds", type=float, default=8.0)
    plan.add_argument("--page-retries", type=int, default=3)
    plan.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    plan.add_argument("--sleep-seconds", type=float, default=0.15)
    plan.add_argument("--max-codes", type=int, default=0)
    plan.add_argument("--universe-file", default="")

    backfill = sub.add_parser("backfill-minute")
    backfill.add_argument("--scope", choices=["tracked", "etf", "ashare", "all"], default="tracked")
    backfill.add_argument("--month", default=now_cn().strftime("%Y-%m"))
    backfill.add_argument("--page-size", type=int, default=1000)
    backfill.add_argument("--timeout-seconds", type=float, default=8.0)
    backfill.add_argument("--page-retries", type=int, default=3)
    backfill.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    backfill.add_argument("--sleep-seconds", type=float, default=0.15)
    backfill.add_argument("--max-codes", type=int, default=0)
    backfill.add_argument("--universe-file", default="")
    backfill.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "snapshot":
        summary = collect_snapshot(args)
    elif args.command == "plan-backfill":
        summary = plan_backfill(args)
    elif args.command == "backfill-minute":
        summary = backfill_minute(args)
    else:
        raise ValueError(args.command)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
