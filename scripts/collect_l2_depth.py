"""Forward five-level order-book collector for the confirmed A-share ETF T+0 universe.

The collector is market-data only: it never imports the paper broker client, never
submits/cancels orders, and never changes trading state or strategy configuration.

The universe is rebuilt from the locally audited T0 master on every process start.
Each poll writes:

* ``depth_YYYY-MM-DD.jsonl``: fresh five-level books, backward compatible with the
  existing OBI research script.
* ``coverage_YYYY-MM-DD.jsonl``: expected/returned/fresh/missing/stale coverage.
* ``universe_YYYY-MM-DD.json``: the exact point-in-time universe used that day.

The XSHG calendar and China exchange clock fail closed. Weekend/holiday/stale Sina
quotes cannot enter a trading-day depth file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from run_etf_paper_trading_agent import ROOT


CST = timezone(timedelta(hours=8))
OUT_DIR = ROOT / "outputs" / "l2_depth"
MASTER = ROOT / "outputs" / "edge_research" / "t0_etf_master_latest.jsonl"
CODE_RE = re.compile(r"^\d{6}$")
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_QUOTE_AGE_SECONDS = 180


def now_cn() -> datetime:
    return datetime.now(CST)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_confirmed_t0_universe(
    master_path: Path = MASTER,
    asset_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load only product-level confirmed, non-money T0 ETFs from the audited master."""
    if not master_path.exists():
        raise FileNotFoundError(f"confirmed T0 master not found: {master_path}")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for line in master_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        code = str(row.get("code") or "").zfill(6)
        exchange = str(row.get("exchange") or "").upper()
        asset_class = str(row.get("asset_class") or "unknown")
        if not row.get("t0_confirmed") or row.get("is_money_like"):
            continue
        if asset_classes and asset_class not in asset_classes:
            continue
        if not CODE_RE.fullmatch(code) or exchange not in {"SH", "SZ"}:
            continue
        selected[(exchange, code)] = {
            "code": code,
            "exchange": exchange,
            "name": str(row.get("name") or ""),
            "asset_class": asset_class,
            "confirmation_basis": row.get("confirmation_basis"),
        }
    if not selected:
        raise RuntimeError("confirmed T0 universe is empty; refusing incomplete fallback")
    return [selected[key] for key in sorted(selected)]


def universe_fingerprint(universe: list[dict[str, Any]]) -> str:
    canonical = json.dumps(universe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_xshg_session(day: date) -> bool:
    """Use the exchange calendar; dependency/calendar failures stop collection."""
    import exchange_calendars as xcals
    import pandas as pd

    calendar = xcals.get_calendar("XSHG")
    return bool(calendar.is_session(pd.Timestamp(day.isoformat())))


def in_continuous_session(moment: datetime) -> bool:
    minute = moment.hour * 60 + moment.minute
    return (9 * 60 + 30) <= minute <= (11 * 60 + 30) or (13 * 60) <= minute <= (15 * 60)


def sina_symbol(row: dict[str, Any]) -> str:
    return str(row["exchange"]).lower() + str(row["code"])


def chunks(rows: list[Any], size: int) -> list[list[Any]]:
    width = max(1, int(size))
    return [rows[index : index + width] for index in range(0, len(rows), width)]


def parse_source_time(fields: list[str]) -> datetime | None:
    if len(fields) <= 31:
        return None
    raw_date = fields[30].strip()
    raw_time = fields[31].strip()
    try:
        return datetime.fromisoformat(f"{raw_date}T{raw_time}").replace(tzinfo=CST)
    except (TypeError, ValueError):
        return None


def parse_sina_payload(
    raw: str,
    universe_by_code: dict[str, dict[str, Any]],
    collected_at: datetime,
    max_quote_age_seconds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if "hq_str_" not in line or '="' not in line:
            continue
        symbol = line.split("hq_str_", 1)[1].split("=", 1)[0]
        code = symbol[2:]
        meta = universe_by_code.get(code)
        if meta is None:
            continue
        body = line.split('"', 2)[1] if '"' in line else ""
        fields = body.split(",")
        if len(fields) < 32:
            continue
        try:
            current = float(fields[3])
            previous_close = float(fields[2])
            bid_volumes = [float(fields[10 + 2 * level]) for level in range(5)]
            bid_prices = [float(fields[11 + 2 * level]) for level in range(5)]
            ask_volumes = [float(fields[20 + 2 * level]) for level in range(5)]
            ask_prices = [float(fields[21 + 2 * level]) for level in range(5)]
        except (TypeError, ValueError, IndexError):
            continue
        if current <= 0:
            continue
        source_time = parse_source_time(fields)
        quote_age = (
            (collected_at - source_time).total_seconds()
            if source_time is not None
            else None
        )
        fresh = bool(
            source_time is not None
            and source_time.date() == collected_at.date()
            and quote_age is not None
            and -5 <= quote_age <= max_quote_age_seconds
        )
        total_bid = sum(max(0.0, value) for value in bid_volumes)
        total_ask = sum(max(0.0, value) for value in ask_volumes)
        denominator = total_bid + total_ask
        imbalance = (total_bid - total_ask) / denominator if denominator > 0 else None
        bid1, ask1 = bid_prices[0], ask_prices[0]
        midpoint = (bid1 + ask1) / 2.0 if ask1 >= bid1 > 0 else current
        micro_price = (
            (ask1 * total_bid + bid1 * total_ask) / denominator
            if denominator > 0 and ask1 > 0 and bid1 > 0
            else current
        )
        rows.append(
            {
                "schemaVersion": "t0_l2_depth_v2",
                "collected_at": collected_at.isoformat(),
                "source_quote_time": source_time.isoformat() if source_time else None,
                "quote_age_seconds": round(quote_age, 3) if quote_age is not None else None,
                "is_fresh": fresh,
                "date": collected_at.strftime("%Y-%m-%d"),
                "ts": collected_at.strftime("%H:%M:%S"),
                "code": code,
                "exchange": meta["exchange"],
                "name": meta.get("name"),
                "asset_class": meta.get("asset_class"),
                "current": current,
                "prevClose": previous_close,
                "bid1": bid1,
                "ask1": ask1,
                "midpoint": midpoint,
                "half_spread_bps": round(
                    (ask1 - bid1) / (ask1 + bid1) * 10_000.0, 6
                )
                if ask1 >= bid1 > 0
                else None,
                "bid_prices": bid_prices,
                "bid_volumes": bid_volumes,
                "ask_prices": ask_prices,
                "ask_volumes": ask_volumes,
                "bid_levels": sum(price > 0 and volume > 0 for price, volume in zip(bid_prices, bid_volumes)),
                "ask_levels": sum(price > 0 and volume > 0 for price, volume in zip(ask_prices, ask_volumes)),
                "bid_vol5": round(total_bid, 3),
                "ask_vol5": round(total_ask, 3),
                "obi": round(imbalance, 8) if imbalance is not None else None,
                "micro_price": round(micro_price, 6),
                "micro_dev_bps": round((micro_price / midpoint - 1.0) * 10_000.0, 6)
                if midpoint > 0
                else None,
                "source": "sina_5level",
            }
        )
    return rows


def fetch_depth(
    universe: list[dict[str, Any]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 10.0,
    max_quote_age_seconds: int = DEFAULT_MAX_QUOTE_AGE_SECONDS,
    opener: Callable[..., Any] = urllib.request.urlopen,
    collected_at: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    moment = collected_at or now_cn()
    by_code = {row["code"]: row for row in universe}
    all_rows: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(chunks(universe, batch_size), start=1):
        url = "https://hq.sinajs.cn/list=" + ",".join(sina_symbol(row) for row in batch)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.sina.com.cn",
            },
        )
        started = time.perf_counter()
        try:
            response = opener(request, timeout=timeout)
            raw = response.read().decode("gbk", "ignore")
            parsed = parse_sina_payload(raw, by_code, moment, max_quote_age_seconds)
            all_rows.extend(parsed)
            batch_reports.append(
                {
                    "batch": batch_number,
                    "requested": len(batch),
                    "returned": len(parsed),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "ok": True,
                }
            )
        except Exception as exc:
            batch_reports.append(
                {
                    "batch": batch_number,
                    "requested": len(batch),
                    "returned": 0,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "ok": False,
                    "error": str(exc),
                }
            )
    deduplicated = {row["code"]: row for row in all_rows}
    return [deduplicated[code] for code in sorted(deduplicated)], {
        "provider": "sina_5level",
        "batches": batch_reports,
    }


def coverage_record(
    universe: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fetch_meta: dict[str, Any],
    collected_at: datetime,
) -> dict[str, Any]:
    expected = {row["code"] for row in universe}
    returned = {row["code"] for row in rows}
    fresh = {row["code"] for row in rows if row.get("is_fresh")}
    complete = {
        row["code"]
        for row in rows
        if row.get("is_fresh") and row.get("bid_levels") == 5 and row.get("ask_levels") == 5
    }
    return {
        "schemaVersion": "t0_l2_coverage_v1",
        "collected_at": collected_at.isoformat(),
        "trade_date": collected_at.strftime("%Y-%m-%d"),
        "universe_fingerprint": universe_fingerprint(universe),
        "expected_count": len(expected),
        "returned_count": len(returned),
        "fresh_count": len(fresh),
        "complete_five_level_count": len(complete),
        "fresh_coverage_rate": round(len(fresh) / len(expected), 6) if expected else 0.0,
        "complete_book_rate": round(len(complete) / len(expected), 6) if expected else 0.0,
        "missing_codes": sorted(expected - returned),
        "stale_codes": sorted(returned - fresh),
        "provider": fetch_meta,
        "status": "ok" if fresh == expected else "partial_coverage",
    }


def write_universe_snapshot(universe: list[dict[str, Any]], trade_date: str) -> Path:
    path = OUT_DIR / f"universe_{trade_date}.json"
    atomic_json(
        path,
        {
            "schemaVersion": "t0_l2_universe_v1",
            "trade_date": trade_date,
            "master": str(MASTER),
            "fingerprint": universe_fingerprint(universe),
            "count": len(universe),
            "etfs": universe,
        },
    )
    return path


def collect_once(
    universe: list[dict[str, Any]],
    *,
    batch_size: int,
    timeout: float,
    max_quote_age_seconds: int,
    write: bool = True,
    collected_at: datetime | None = None,
) -> dict[str, Any]:
    moment = collected_at or now_cn()
    rows, fetch_meta = fetch_depth(
        universe,
        batch_size=batch_size,
        timeout=timeout,
        max_quote_age_seconds=max_quote_age_seconds,
        collected_at=moment,
    )
    coverage = coverage_record(universe, rows, fetch_meta, moment)
    fresh_rows = [row for row in rows if row.get("is_fresh")]
    if write:
        trade_date = moment.strftime("%Y-%m-%d")
        append_jsonl(OUT_DIR / f"depth_{trade_date}.jsonl", fresh_rows)
        append_jsonl(OUT_DIR / f"coverage_{trade_date}.jsonl", [coverage])
    return {**coverage, "written_depth_rows": len(fresh_rows) if write else 0}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="fetch and audit without writing")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--until", default="15:00")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-quote-age-seconds", type=int, default=DEFAULT_MAX_QUOTE_AGE_SECONDS)
    parser.add_argument(
        "--asset-classes",
        default="",
        help="optional comma-separated subset; default is every confirmed non-money T0 ETF",
    )
    args = parser.parse_args()

    classes = {value.strip() for value in args.asset_classes.split(",") if value.strip()} or None
    universe = load_confirmed_t0_universe(asset_classes=classes)
    moment = now_cn()
    if not is_xshg_session(moment.date()):
        print(
            json.dumps(
                {
                    "status": "skipped_non_trading_day",
                    "trade_date": moment.strftime("%Y-%m-%d"),
                    "expected_count": len(universe),
                    "written_depth_rows": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe_path = write_universe_snapshot(universe, moment.strftime("%Y-%m-%d"))
    print(
        f"L2 depth collector: {len(universe)} confirmed T0 ETFs, "
        f"batch={args.batch_size}, poll={args.poll_seconds}s, universe={universe_path}"
    )

    if args.once:
        if not in_continuous_session(moment):
            print(
                json.dumps(
                    {
                        "status": "skipped_outside_continuous_session",
                        "trade_date": moment.strftime("%Y-%m-%d"),
                        "expected_count": len(universe),
                        "written_depth_rows": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        result = collect_once(
            universe,
            batch_size=args.batch_size,
            timeout=args.timeout_seconds,
            max_quote_age_seconds=args.max_quote_age_seconds,
            write=not args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    end_hour, end_minute = (int(value) for value in args.until.split(":"))
    polls = 0
    while True:
        moment = now_cn()
        if not is_xshg_session(moment.date()):
            break
        if (moment.hour, moment.minute) >= (end_hour, end_minute):
            break
        if not in_continuous_session(moment):
            time.sleep(min(30.0, max(1.0, args.poll_seconds)))
            continue
        started = time.perf_counter()
        try:
            result = collect_once(
                universe,
                batch_size=args.batch_size,
                timeout=args.timeout_seconds,
                max_quote_age_seconds=args.max_quote_age_seconds,
                write=True,
            )
            polls += 1
            print(
                f"{result['collected_at']} fresh={result['fresh_count']}/"
                f"{result['expected_count']} complete5={result['complete_five_level_count']} "
                f"status={result['status']}"
            )
        except Exception as exc:
            print(f"poll error: {exc}", file=sys.stderr)
        elapsed = time.perf_counter() - started
        time.sleep(max(0.5, args.poll_seconds - elapsed))
    print(json.dumps({"status": "completed", "polls": polls}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
