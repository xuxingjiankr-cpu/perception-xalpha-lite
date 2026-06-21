"""Build offline replay quote snapshots from local ETF minute bars.

This is market-data conversion only. It does not call broker/account/order APIs
and does not write agent state. The output is consumed by replay_t0_decisions.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import copy
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def market_from_exchange(exchange: str) -> str:
    return "1" if exchange.upper() == "SH" else "0"


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "-"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def read_minute_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("datetime"):
                rows.append(row)
    rows.sort(key=lambda x: str(x.get("datetime")))
    return rows


def prev_close_by_day(rows: list[dict[str, Any]]) -> dict[str, float]:
    day_last: dict[str, float] = {}
    for row in rows:
        dt = str(row.get("datetime"))
        day = dt[:10]
        close_px = as_float(row.get("close"))
        if close_px > 0:
            day_last[day] = close_px
    out: dict[str, float] = {}
    last_close: float | None = None
    for day in sorted(day_last):
        out[day] = last_close if last_close and last_close > 0 else day_last[day]
        last_close = day_last[day]
    return out


def quote_from_bar(etf: dict[str, Any], row: dict[str, Any], prev_close: float) -> dict[str, Any]:
    close_px = as_float(row.get("close"))
    tick = 0.001
    bid = max(0.0, close_px - tick)
    ask = close_px + tick if close_px > 0 else 0.0
    spread_pct = (ask - bid) / close_px if close_px > 0 and ask > 0 and bid > 0 else None
    dt_raw = str(row["datetime"])
    fmt = "%Y-%m-%d %H:%M:%S" if len(dt_raw) >= 19 else "%Y-%m-%d %H:%M"
    dt = datetime.strptime(dt_raw, fmt).replace(tzinfo=CN_TZ)
    change_pct = close_px / prev_close - 1.0 if close_px > 0 and prev_close > 0 else 0.0
    return {
        "timestamp": dt.isoformat(),
        "trade_date": dt.strftime("%Y-%m-%d"),
        "stockCode": str(etf.get("stockCode", "")).zfill(6),
        "exchange": str(etf.get("exchange", "SH")).upper(),
        "name": etf.get("name", ""),
        "asset_class": etf.get("asset_class", ""),
        "source": row.get("source") or "local_minute",
        "liquidity_source": "point_in_time",
        "quote_ok": close_px > 0,
        "isSuspended": False,
        "currentPrice": close_px,
        "prevClose": prev_close,
        "open": as_float(row.get("open")),
        "high": as_float(row.get("high")),
        "low": as_float(row.get("low")),
        "bidPrice1": round(bid, 3),
        "askPrice1": round(ask, 3),
        "midpoint": close_px,
        "spread_pct": spread_pct,
        "change_pct": change_pct,
        "change": change_pct * 100.0,
        "volume": as_float(row.get("volume")),
        "amount": as_float(row.get("amount")),
        "raw_has_volume_field": True,
        "raw_has_amount_field": True,
        "synthetic_order_book": True,
    }


def load_replay_universe(config_path: Path, universe_file: Path | None = None) -> list[dict[str, Any]]:
    cfg = load_json(config_path)
    static = {
        str(item.get("stockCode", "")).zfill(6): item
        for item in cfg.get("universe", [])
        if item.get("stockCode")
    }
    if universe_file is None:
        return list(static.values())
    rows: list[dict[str, Any]] = []
    for line in universe_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        code = str(item.get("stockCode") or item.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        seed = static.get(code, {})
        item_exchange = str(item.get("exchange") or seed.get("exchange") or "SH").upper()
        market = str(item.get("market", "1" if item_exchange == "SH" else "0"))
        rows.append({
            "stockCode": code,
            "exchange": "SH" if market == "1" else "SZ",
            "name": item.get("name") or seed.get("name", ""),
            "asset_class": seed.get("asset_class", "dynamic"),
        })
    return sorted(rows, key=lambda x: (x["exchange"], x["stockCode"]))


def cumulative_quotes(
    etf: dict[str, Any],
    rows: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> Iterator[dict[str, Any]]:
    """Convert per-minute bars into live-shaped quotes with same-day cumulative flow fields."""
    prev_map = prev_close_by_day(rows)
    active_day = ""
    cumulative_volume = 0.0
    cumulative_amount = 0.0
    for row in rows:
        dt = str(row.get("datetime") or "")
        day = dt[:10]
        if day != active_day:
            active_day = day
            cumulative_volume = 0.0
            cumulative_amount = 0.0
        cumulative_volume += max(0.0, as_float(row.get("volume")))
        cumulative_amount += max(0.0, as_float(row.get("amount")))
        if day < start_date:
            continue
        if day > end_date:
            break
        prev_close = prev_map.get(day, as_float(row.get("open")))
        quote = quote_from_bar(etf, row, prev_close)
        quote["volume"] = cumulative_volume
        quote["amount"] = cumulative_amount
        quote["minute_volume"] = as_float(row.get("volume"))
        quote["minute_amount"] = as_float(row.get("amount"))
        yield quote


def _session_fraction(timestamp: str) -> float:
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    now_min = hour * 60 + minute
    if now_min <= 9 * 60 + 30:
        elapsed = 0
    elif now_min <= 11 * 60 + 30:
        elapsed = now_min - (9 * 60 + 30)
    elif now_min <= 13 * 60:
        elapsed = 120
    elif now_min <= 15 * 60:
        elapsed = 120 + now_min - 13 * 60
    else:
        elapsed = 240
    return max(0.05, min(1.0, elapsed / 240.0))


def passes_dynamic_gate(quote: dict[str, Any], dyn_cfg: dict[str, Any]) -> bool:
    name = str(quote.get("name") or "")
    for keyword in dyn_cfg.get("name_exclude_keywords", []):
        keyword = str(keyword)
        if keyword and keyword in name and not (keyword == "现金" and "现金流" in name):
            return False
    price = as_float(quote.get("currentPrice"))
    bid = as_float(quote.get("bidPrice1"))
    ask = as_float(quote.get("askPrice1"))
    if price <= as_float(dyn_cfg.get("min_price"), 0.3) or bid <= 0 or ask <= bid:
        return False
    spread = (ask - bid) / ((ask + bid) / 2.0)
    if spread > as_float(dyn_cfg.get("max_spread_pct"), 0.004):
        return False
    required_amount = as_float(dyn_cfg.get("min_amount_yuan"), 50_000_000.0) * _session_fraction(
        str(quote.get("timestamp") or "")
    )
    return as_float(quote.get("amount")) >= required_amount


def resample_quotes(quotes: list[dict[str, Any]], interval_minutes: int) -> list[dict[str, Any]]:
    """Create live-cadence snapshots using only the last quote known at each mark."""
    if interval_minutes <= 0 or not quotes:
        return quotes
    day = str(quotes[0].get("trade_date") or str(quotes[0].get("timestamp"))[:10])
    by_minute = {
        int(str(quote["timestamp"])[11:13]) * 60 + int(str(quote["timestamp"])[14:16]): quote
        for quote in quotes
    }
    marks = (
        list(range(9 * 60 + 30 + interval_minutes, 11 * 60 + 30 + 1, interval_minutes))
        + list(range(13 * 60 + interval_minutes, 15 * 60 + 1, interval_minutes))
    )
    available = sorted(by_minute)
    result: list[dict[str, Any]] = []
    for mark in marks:
        prior = [minute for minute in available if minute <= mark]
        if not prior:
            continue
        source_minute = prior[-1]
        if mark - source_minute > interval_minutes:
            continue
        quote = copy.deepcopy(by_minute[source_minute])
        quote["source_timestamp"] = quote["timestamp"]
        quote["timestamp"] = (
            f"{day}T{mark // 60:02d}:{mark % 60:02d}:00+08:00"
        )
        quote["snapshot_interval_minutes"] = interval_minutes
        result.append(quote)
    return result


def build_replay_directory(
    *,
    config_path: Path,
    minute_dir: Path,
    universe_file: Path | None,
    output_dir: Path,
    start_date: str,
    end_date: str,
    dynamic_gate: bool,
    snapshot_interval_minutes: int = 0,
) -> dict[str, Any]:
    cfg = load_json(config_path)
    universe = load_replay_universe(config_path, universe_file)
    dyn_cfg = cfg.get("dynamic_universe", {}) if isinstance(cfg.get("dynamic_universe"), dict) else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(output_dir.glob("*.jsonl")) + list(output_dir.glob("*.unsorted")):
        stale.unlink()

    handles: dict[str, Any] = {}
    rows_by_date: dict[str, int] = defaultdict(int)
    codes_by_date: dict[str, int] = defaultdict(int)
    missing: list[dict[str, Any]] = []
    loaded_symbols = 0
    try:
        for etf in universe:
            code = str(etf.get("stockCode", "")).zfill(6)
            market = market_from_exchange(str(etf.get("exchange", "SH")))
            path = minute_dir / f"{market}_{code}_{code}.csv.gz"
            rows = read_minute_file(path)
            if not rows:
                missing.append({"stockCode": code, "name": etf.get("name"), "path": str(path)})
                continue
            loaded_symbols += 1
            by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for quote in cumulative_quotes(etf, rows, start_date, end_date):
                by_day[str(quote["trade_date"])].append(quote)
            for day, quotes in by_day.items():
                quotes = resample_quotes(quotes, snapshot_interval_minutes)
                if dynamic_gate:
                    quotes = [quote for quote in quotes if passes_dynamic_gate(quote, dyn_cfg)]
                if not quotes:
                    continue
                if day not in handles:
                    handles[day] = (output_dir / f"{day}.unsorted").open("w", encoding="utf-8")
                handle = handles[day]
                for quote in quotes:
                    payload = json.dumps(quote, ensure_ascii=False, sort_keys=True)
                    handle.write(f"{quote['timestamp']}\t{quote['stockCode']}\t{payload}\n")
                    rows_by_date[day] += 1
                codes_by_date[day] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for day in sorted(rows_by_date):
        staging = output_dir / f"{day}.unsorted"
        lines = staging.read_text(encoding="utf-8").splitlines()
        lines.sort()
        with (output_dir / f"{day}.jsonl").open("w", encoding="utf-8") as out:
            for line in lines:
                out.write(line.split("\t", 2)[2] + "\n")
        staging.unlink()

    summary = {
        "task": "build_t0_replay_quotes_from_minute_data",
        "config": str(config_path),
        "minute_dir": str(minute_dir),
        "universe_file": str(universe_file) if universe_file else None,
        "output_dir": str(output_dir),
        "start_date": start_date,
        "end_date": end_date,
        "configured_universe": len(universe),
        "loaded_symbols": loaded_symbols,
        "missing_symbols": missing,
        "dates": sorted(rows_by_date),
        "rows_by_date": dict(sorted(rows_by_date.items())),
        "eligible_codes_by_date": dict(sorted(codes_by_date.items())),
        "rows_written": sum(rows_by_date.values()),
        "dynamic_gate": dynamic_gate,
        "snapshot_interval_minutes": snapshot_interval_minutes,
        "flow_fields": "same_day_cumulative_from_minute_bars",
        "synthetic_order_book": True,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build T0 replay quotes from local ETF minute csv.gz files.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "t0_intraday_paper_agent.json"))
    parser.add_argument("--minute-dir", default=str(ROOT / "data" / "market" / "eastmoney" / "minute" / "2026-06" / "etf"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "t0_replay" / "replay_from_minute_20etf_202606.jsonl"))
    parser.add_argument("--output-dir", default="", help="write sorted per-day JSONL replay files")
    parser.add_argument("--universe-file", default="", help="optional JSONL full-market ETF universe")
    parser.add_argument("--dynamic-gate", action="store_true", help="retain codes that pass live-shaped liquidity gates")
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default="2026-06-18")
    parser.add_argument("--snapshot-interval-minutes", type=int, default=0,
                        help="resample each symbol to fixed live cadence; 0 keeps raw bars")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_json(config_path)
    minute_dir = Path(args.minute_dir)
    universe_file = Path(args.universe_file) if args.universe_file else None
    if args.output_dir:
        summary = build_replay_directory(
            config_path=config_path,
            minute_dir=minute_dir,
            universe_file=universe_file,
            output_dir=Path(args.output_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            dynamic_gate=args.dynamic_gate,
            snapshot_interval_minutes=args.snapshot_interval_minutes,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    by_ts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for etf in load_replay_universe(config_path, universe_file):
        code = str(etf.get("stockCode", "")).zfill(6)
        market = market_from_exchange(str(etf.get("exchange", "SH")))
        path = minute_dir / f"{market}_{code}_{code}.csv.gz"
        rows = read_minute_file(path)
        if not rows:
            missing.append({"stockCode": code, "name": etf.get("name"), "path": str(path)})
            continue
        kept = 0
        raw_quotes = list(cumulative_quotes(etf, rows, args.start_date, args.end_date))
        by_day_quotes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for quote in raw_quotes:
            by_day_quotes[str(quote["trade_date"])].append(quote)
        for day_quotes in by_day_quotes.values():
            for quote in resample_quotes(day_quotes, args.snapshot_interval_minutes):
                if args.dynamic_gate and not passes_dynamic_gate(quote, cfg.get("dynamic_universe", {})):
                    continue
                by_ts[str(quote["timestamp"])].append(quote)
                kept += 1
        loaded.append({"stockCode": code, "name": etf.get("name"), "rows": kept, "path": str(path)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", encoding="utf-8") as fh:
        for ts in sorted(by_ts):
            for quote in sorted(by_ts[ts], key=lambda x: x["stockCode"]):
                fh.write(json.dumps(quote, ensure_ascii=False, sort_keys=True) + "\n")
                rows_written += 1

    summary = {
        "task": "build_t0_replay_quotes_from_minute_data",
        "config": str(Path(args.config)),
        "minute_dir": str(minute_dir),
        "output": str(output),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "configured_universe": len(load_replay_universe(config_path, universe_file)),
        "loaded_symbols": len(loaded),
        "missing_symbols": missing,
        "timestamp_rounds": len(by_ts),
        "rows_written": rows_written,
        "loaded": loaded,
        "synthetic_order_book": True,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
