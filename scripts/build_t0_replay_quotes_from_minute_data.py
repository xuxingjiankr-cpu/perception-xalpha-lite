"""Build offline replay quote snapshots from local ETF minute bars.

This is market-data conversion only. It does not call broker/account/order APIs
and does not write agent state. The output is consumed by replay_t0_decisions.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build T0 replay quotes from local ETF minute csv.gz files.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "t0_intraday_paper_agent.json"))
    parser.add_argument("--minute-dir", default=str(ROOT / "data" / "market" / "eastmoney" / "minute" / "2026-06" / "etf"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "t0_replay" / "replay_from_minute_20etf_202606.jsonl"))
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default="2026-06-15")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    minute_dir = Path(args.minute_dir)
    by_ts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for etf in cfg.get("universe", []):
        code = str(etf.get("stockCode", "")).zfill(6)
        market = market_from_exchange(str(etf.get("exchange", "SH")))
        path = minute_dir / f"{market}_{code}_{code}.csv.gz"
        rows = read_minute_file(path)
        if not rows:
            missing.append({"stockCode": code, "name": etf.get("name"), "path": str(path)})
            continue
        prev_map = prev_close_by_day(rows)
        kept = 0
        for row in rows:
            dt = str(row["datetime"])
            day = dt[:10]
            if day < args.start_date or day > args.end_date:
                continue
            prev_close = prev_map.get(day, as_float(row.get("open")))
            by_ts[dt].append(quote_from_bar(etf, row, prev_close))
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
        "configured_universe": len(cfg.get("universe", [])),
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
