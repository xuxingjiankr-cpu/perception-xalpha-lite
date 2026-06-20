"""Fetch point-in-time 5-minute bars for officially confirmed T+0 ETFs.

Unlike ``fetch_yahoo_5m_quotes.py``, this research collector never selects an
instrument using its end-of-day turnover.  Membership is fixed by the official
exchange classification master, and listing dates are respected.  It has no
broker/order imports and cannot trade.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CN = timezone(timedelta(hours=8))
MASTER = ROOT / "outputs" / "edge_research" / "t0_etf_master_latest.jsonl"
OUT = ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
SUMMARY = ROOT / "outputs" / "edge_research" / "t0_research_quotes_summary.json"


def yahoo_symbol(code: str, exchange: str) -> str:
    return f"{code}.SS" if exchange == "SH" else f"{code}.SZ"


def load_confirmed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("t0_confirmed") and not row.get("is_money_like"):
                rows.append(row)
    keys = [(str(row["code"]), str(row["exchange"])) for row in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError("confirmed master is empty or contains duplicate composite keys")
    return sorted(rows, key=lambda row: (row["exchange"], row["code"]))


def fetch_one(item: dict[str, Any], timeout: float, retries: int) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    code, exchange = str(item["code"]), str(item["exchange"])
    symbol = yahoo_symbol(code, exchange)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=60d"
    error: str | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = (payload.get("chart", {}).get("result") or [None])[0]
            if not result:
                raise ValueError(str(payload.get("chart", {}).get("error") or "empty chart result"))
            timestamps = result.get("timestamp") or []
            quote = (result.get("indicators", {}).get("quote") or [{}])[0]
            closes, volumes = quote.get("close") or [], quote.get("volume") or []
            rows: list[dict[str, Any]] = []
            for index, raw_ts in enumerate(timestamps):
                close = closes[index] if index < len(closes) else None
                volume = volumes[index] if index < len(volumes) else None
                if close is None or float(close) <= 0:
                    continue
                dt = datetime.fromtimestamp(int(raw_ts), CN)
                minute = dt.hour * 60 + dt.minute
                if not (9 * 60 + 30 <= minute <= 11 * 60 + 30 or 13 * 60 <= minute <= 15 * 60):
                    continue
                trade_date = dt.date().isoformat()
                listing = str(item.get("listing_date") or "").replace("-", "")
                if listing and trade_date.replace("-", "") < listing:
                    continue
                rows.append({
                    "timestamp": dt.isoformat(timespec="seconds"),
                    "trade_date": trade_date,
                    "stockCode": code,
                    "exchange": exchange,
                    "name": item.get("name"),
                    "asset_class": item.get("asset_class"),
                    "close": float(close),
                    "bar_volume": float(volume or 0.0),
                    "source": "yahoo_finance_chart_5m",
                    "selection_rule": "official_t0_class_only_no_full_day_liquidity_gate",
                })
            if not rows:
                raise ValueError("no regular-session bars")
            return item, rows, None
        except Exception as exc:  # network/provider errors are recorded, not hidden
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.75 * (attempt + 1))
    return item, [], error


def enrich_point_in_time(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add only information available at or before each bar."""
    rows.sort(key=lambda row: (row["stockCode"], row["trade_date"], row["timestamp"]))
    output: list[dict[str, Any]] = []
    last_close: dict[str, float] = {}
    current_key: tuple[str, str] | None = None
    cumulative_volume = cumulative_amount = 0.0
    prior_close: float | None = None
    day_last: float | None = None
    for row in rows:
        key = (row["stockCode"], row["trade_date"])
        if key != current_key:
            if current_key is not None and day_last is not None:
                last_close[current_key[0]] = day_last
            current_key = key
            cumulative_volume = cumulative_amount = 0.0
            prior_close = last_close.get(row["stockCode"])
        volume = max(0.0, float(row["bar_volume"]))
        close = float(row["close"])
        cumulative_volume += volume
        cumulative_amount += volume * close
        day_last = close
        row["prev_close"] = prior_close
        row["cumulative_volume"] = cumulative_volume
        row["cumulative_amount"] = cumulative_amount
        output.append(row)
    return sorted(output, key=lambda row: (row["timestamp"], row["exchange"], row["stockCode"]))


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default=str(MASTER))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--summary", default=str(SUMMARY))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    instruments = load_confirmed(Path(args.master))
    fetched: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_one, item, args.timeout_seconds, args.retries): item
            for item in instruments
        }
        for completed, future in enumerate(as_completed(futures), 1):
            item, rows, error = future.result()
            fetched.extend(rows)
            if error:
                failures.append({"code": item["code"], "exchange": item["exchange"], "error": error})
            if completed % 25 == 0 or completed == len(instruments):
                print(f"completed={completed}/{len(instruments)} bars={len(fetched)} failures={len(failures)}")
    enriched = enrich_point_in_time(fetched)
    out_path = Path(args.out)
    atomic_jsonl(out_path, enriched)
    dates = sorted({row["trade_date"] for row in enriched})
    codes = sorted({(row["stockCode"], row["exchange"]) for row in enriched})
    summary = {
        "schemaVersion": "t0_research_quotes_v1",
        "generatedAt": datetime.now(CN).isoformat(timespec="seconds"),
        "source": "Yahoo Finance chart 5m",
        "diagnosticOnly": True,
        "selectionRule": "official_t0_class_only_no_full_day_liquidity_gate",
        "futureDayTurnoverUsed": False,
        "instrumentsRequested": len(instruments),
        "instrumentsFetched": len(codes),
        "failures": failures,
        "rows": len(enriched),
        "tradeDates": len(dates),
        "dateStart": dates[0] if dates else None,
        "dateEnd": dates[-1] if dates else None,
        "output": str(out_path),
        "knownLimitations": [
            "Yahoo has no historical bid/ask or order-book depth",
            "current exchange product list may introduce survivorship bias for delisted funds",
            "first 5-minute close is not an executable opening-auction price",
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if enriched else 1


if __name__ == "__main__":
    raise SystemExit(main())
