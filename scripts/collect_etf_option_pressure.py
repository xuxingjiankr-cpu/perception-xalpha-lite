"""Collect forward SSE ETF-option pressure, volatility and unsigned-gamma states.

The exchange endpoint supplies the current option underlyings and expiry months.
Sina's public quote endpoint supplies contract lists and point-in-time contract
quotes. Source tiers are explicit in every record.

Outputs are research-only:

* raw normalized contract snapshots;
* call/put volume and open-interest pressure by underlying/expiry;
* Black-Scholes implied volatility, delta and gamma using a fixed documented rate;
* put-minus-call 25-delta IV skew where both wings are observable;
* unsigned gamma-open-interest exposure (dealer position sign is unknowable).

No account, position, order, cancel, trading config or overlay is read or written.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, time as datetime_time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from collect_l2_depth import CST, in_continuous_session, is_xshg_session, now_cn
from run_etf_paper_trading_agent import ROOT


SSE_HQ = "https://yunhq.sse.com.cn:32042"
SINA_HQ = "https://hq.sinajs.cn/list="
OUT_DIR = ROOT / "data" / "research" / "etf_options"
COVERAGE_DIR = ROOT / "outputs" / "etf_options"
CN = ZoneInfo("Asia/Shanghai")
CONTRACT_MULTIPLIER = 10_000
DEFAULT_RISK_FREE_RATE = 0.015
DEFAULT_DIVIDEND_YIELD = 0.0
DEFAULT_MAX_AGE_SECONDS = 180


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def black_scholes(
    option_type: str,
    spot: float,
    strike: float,
    years: float,
    rate: float,
    dividend: float,
    sigma: float,
) -> tuple[float, float, float]:
    if min(spot, strike, years, sigma) <= 0:
        return 0.0, 0.0, 0.0
    root_t = math.sqrt(years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend + 0.5 * sigma * sigma) * years
    ) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discount_r = math.exp(-rate * years)
    discount_q = math.exp(-dividend * years)
    if option_type == "call":
        price = spot * discount_q * norm_cdf(d1) - strike * discount_r * norm_cdf(d2)
        delta = discount_q * norm_cdf(d1)
    else:
        price = strike * discount_r * norm_cdf(-d2) - spot * discount_q * norm_cdf(-d1)
        delta = -discount_q * norm_cdf(-d1)
    gamma = discount_q * norm_pdf(d1) / (spot * sigma * root_t)
    return price, delta, gamma


def implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    years: float,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend: float = DEFAULT_DIVIDEND_YIELD,
) -> float | None:
    if min(market_price, spot, strike, years) <= 0:
        return None
    intrinsic = (
        max(0.0, spot * math.exp(-dividend * years) - strike * math.exp(-rate * years))
        if option_type == "call"
        else max(0.0, strike * math.exp(-rate * years) - spot * math.exp(-dividend * years))
    )
    if market_price + 1e-8 < intrinsic:
        return None
    low, high = 0.0001, 5.0
    high_price = black_scholes(
        option_type, spot, strike, years, rate, dividend, high
    )[0]
    if market_price > high_price:
        return None
    for _ in range(80):
        middle = (low + high) / 2.0
        price = black_scholes(
            option_type, spot, strike, years, rate, dividend, middle
        )[0]
        if price < market_price:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def http_json(
    url: str,
    *,
    timeout: float = 10.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.sse.com.cn/assortment/options/price/",
        },
    )
    response = opener(request, timeout=timeout)
    payload = json.loads(response.read().decode("utf-8", "replace"))
    return payload if isinstance(payload, dict) else {}


def official_underlyings_and_expiries(
    *,
    timeout: float = 10.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    underlying_url = (
        SSE_HQ
        + "/v1/sho/list/exchange/underlyingstock?"
        + urllib.parse.urlencode({"select": "stockid"})
    )
    expiry_url = (
        SSE_HQ
        + "/v1/sho/list/exchange/stockexpire?"
        + urllib.parse.urlencode({"select": "stockid,expiremonth"})
    )
    underlying_payload = http_json(underlying_url, timeout=timeout, opener=opener)
    expiry_payload = http_json(expiry_url, timeout=timeout, opener=opener)
    underlyings = {
        str(row[0])
        for row in underlying_payload.get("list", [])
        if isinstance(row, list) and row
    }
    mapping: dict[str, list[str]] = {code: [] for code in sorted(underlyings)}
    for row in expiry_payload.get("list", []):
        if not isinstance(row, list) or len(row) < 2:
            continue
        code = str(row[0])
        month = str(row[1])
        if code in mapping and len(month) == 6:
            mapping[code].append(month)
    for code in mapping:
        mapping[code] = sorted(set(mapping[code]))
    return mapping, {
        "source": "Shanghai Stock Exchange official quote endpoint",
        "source_tier": "official_exchange",
        "date": expiry_payload.get("date"),
        "time": expiry_payload.get("time"),
        "underlying_url": underlying_url,
        "expiry_url": expiry_url,
    }


def sina_text(
    symbols: list[str],
    *,
    timeout: float = 10.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, list[str]]:
    if not symbols:
        return {}
    request = urllib.request.Request(
        SINA_HQ + ",".join(symbols),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn",
        },
    )
    response = opener(request, timeout=timeout)
    raw = response.read().decode("gbk", "replace")
    result: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if "hq_str_" not in line or '="' not in line:
            continue
        symbol = line.split("hq_str_", 1)[1].split("=", 1)[0]
        body = line.split('"', 2)[1] if '"' in line else ""
        result[symbol] = body.split(",") if body else []
    return result


def option_list_symbols(expiries: dict[str, list[str]]) -> list[str]:
    symbols: list[str] = []
    for underlying, months in sorted(expiries.items()):
        for month in months:
            short_month = month[2:]
            symbols.append(f"OP_UP_{underlying}{short_month}")
            symbols.append(f"OP_DOWN_{underlying}{short_month}")
    return symbols


def parse_contract_lists(
    raw: dict[str, list[str]],
    expiries: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    contracts: dict[str, dict[str, str]] = {}
    for underlying, months in expiries.items():
        for month in months:
            short_month = month[2:]
            for symbol, option_type in (
                (f"OP_UP_{underlying}{short_month}", "call"),
                (f"OP_DOWN_{underlying}{short_month}", "put"),
            ):
                for contract_symbol in raw.get(symbol, []):
                    contract_symbol = contract_symbol.strip()
                    if not contract_symbol.startswith("CON_OP_"):
                        continue
                    contracts[contract_symbol] = {
                        "underlying": underlying,
                        "expiry_month": month,
                        "option_type": option_type,
                    }
    return contracts


def parse_underlying_quotes(
    raw: dict[str, list[str]],
    collected_at: datetime,
) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for symbol, fields in raw.items():
        if not (symbol.startswith("sh") and len(symbol) == 8 and len(fields) >= 32):
            continue
        code = symbol[2:]
        try:
            source = datetime.fromisoformat(f"{fields[30]}T{fields[31]}").replace(tzinfo=CST)
        except (TypeError, ValueError):
            source = None
        parsed[code] = {
            "name": fields[0],
            "spot": as_float(fields[3]),
            "previous_close": as_float(fields[2]),
            "source_quote_time": source,
            "quote_age_seconds": (
                (collected_at - source).total_seconds() if source is not None else None
            ),
        }
    return parsed


def parse_contract_quote(
    symbol: str,
    fields: list[str],
    metadata: dict[str, str],
    spot_row: dict[str, Any],
    collected_at: datetime,
    *,
    risk_free_rate: float,
    dividend_yield: float,
    max_quote_age_seconds: int,
    snapshot_trade_date_verified: bool = True,
) -> dict[str, Any] | None:
    if len(fields) < 51:
        return None
    source_time: datetime | None
    try:
        source_time = datetime.strptime(fields[32], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=CN
        )
    except (TypeError, ValueError):
        source_time = None
    age = (
        (collected_at - source_time).total_seconds()
        if source_time is not None
        else None
    )
    # Sina field 32 is the last TRADE time, not necessarily the HTTP snapshot
    # generation time. Treating inactive contracts as stale would select on
    # liquidity and bias put/call OI. Snapshot freshness is instead anchored to
    # the official SSE calendar payload date obtained in the same poll. Keep the
    # last-trade recency as a separate feature.
    fresh = bool(snapshot_trade_date_verified)
    last_trade_recent = bool(
        source_time is not None
        and source_time.date() == collected_at.date()
        and age is not None
        and -5 <= age <= max_quote_age_seconds
    )
    bid = as_float(fields[1])
    last = as_float(fields[2])
    ask = as_float(fields[3])
    strike = as_float(fields[7])
    volume = as_float(fields[5])
    open_interest = as_float(fields[41])
    amount = as_float(fields[42])
    expiry_text = fields[46]
    try:
        expiry = date.fromisoformat(expiry_text)
    except ValueError:
        return None
    spot = as_float(spot_row.get("spot"))
    if spot is None or strike is None or strike <= 0:
        return None
    expiry_close = datetime.combine(expiry, datetime_time(15, 0), tzinfo=CN)
    years = max((expiry_close - collected_at).total_seconds(), 0.0) / (
        365.0 * 24.0 * 60.0 * 60.0
    )
    option_price = (
        (bid + ask) / 2.0
        if bid is not None and ask is not None and ask >= bid > 0
        else last
    )
    option_type = metadata["option_type"]
    iv = (
        implied_volatility(
            option_type,
            option_price,
            spot,
            strike,
            years,
            risk_free_rate,
            dividend_yield,
        )
        if option_price is not None
        else None
    )
    delta = gamma = None
    if iv is not None:
        _, delta, gamma = black_scholes(
            option_type,
            spot,
            strike,
            years,
            risk_free_rate,
            dividend_yield,
            iv,
        )
    return {
        "schemaVersion": "etf_option_contract_snapshot_v1",
        "collected_at": collected_at.isoformat(),
        "trade_date": collected_at.strftime("%Y-%m-%d"),
        "source_quote_time": source_time.isoformat() if source_time else None,
        "source_quote_time_semantics": "last_trade_time",
        "quote_age_seconds": round(age, 3) if age is not None else None,
        "last_trade_recent": last_trade_recent,
        "is_fresh": fresh,
        "contract_symbol": symbol,
        "contract_id": symbol.removeprefix("CON_OP_"),
        "contract_name": fields[37],
        "underlying": metadata["underlying"],
        "underlying_name": spot_row.get("name"),
        "underlying_price": spot,
        "expiry_month": metadata["expiry_month"],
        "expiry_date": expiry_text,
        "years_to_expiry": round(years, 9),
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "last": last,
        "previous_settlement": as_float(fields[8]),
        "volume": volume,
        "open_interest": open_interest,
        "amount": amount,
        "implied_volatility": round(iv, 8) if iv is not None else None,
        "delta": round(delta, 8) if delta is not None else None,
        "gamma": round(gamma, 8) if gamma is not None else None,
        "risk_free_rate_assumption": risk_free_rate,
        "dividend_yield_assumption": dividend_yield,
        "contract_multiplier": CONTRACT_MULTIPLIER,
        "source": "sina_public_option_quote",
        "source_tier": "vendor_not_direct_exchange_feed",
    }


def weighted_mean(rows: list[dict[str, Any]], value: str, weight: str) -> float | None:
    pairs = [
        (as_float(row.get(value)), as_float(row.get(weight)))
        for row in rows
    ]
    valid = [(x, w) for x, w in pairs if x is not None and w is not None and w > 0]
    denominator = sum(weight_value for _, weight_value in valid)
    if denominator <= 0:
        return None
    return sum(value_item * weight_value for value_item, weight_value in valid) / denominator


def closest_delta_iv(rows: list[dict[str, Any]], target_abs_delta: float) -> float | None:
    candidates = [
        row
        for row in rows
        if as_float(row.get("delta")) is not None
        and as_float(row.get("implied_volatility")) is not None
    ]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda row: abs(abs(float(row["delta"])) - target_abs_delta),
    )
    return float(selected["implied_volatility"])


def aggregate_option_states(
    rows: list[dict[str, Any]],
    collected_at: datetime,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_fresh"):
            grouped[(row["underlying"], row["expiry_month"])].append(row)
    states: list[dict[str, Any]] = []
    for (underlying, expiry_month), contracts in sorted(grouped.items()):
        calls = [row for row in contracts if row["option_type"] == "call"]
        puts = [row for row in contracts if row["option_type"] == "put"]
        call_volume = sum(as_float(row.get("volume")) or 0.0 for row in calls)
        put_volume = sum(as_float(row.get("volume")) or 0.0 for row in puts)
        call_oi = sum(as_float(row.get("open_interest")) or 0.0 for row in calls)
        put_oi = sum(as_float(row.get("open_interest")) or 0.0 for row in puts)
        call25 = closest_delta_iv(calls, 0.25)
        put25 = closest_delta_iv(puts, 0.25)
        unsigned_gamma_exposure = sum(
            (as_float(row.get("gamma")) or 0.0)
            * (as_float(row.get("open_interest")) or 0.0)
            * CONTRACT_MULTIPLIER
            * float(row["underlying_price"]) ** 2
            * 0.01
            for row in contracts
        )
        states.append(
            {
                "schemaVersion": "etf_option_pressure_state_v1",
                "collected_at": collected_at.isoformat(),
                "trade_date": collected_at.strftime("%Y-%m-%d"),
                "underlying": underlying,
                "expiry_month": expiry_month,
                "contracts": len(contracts),
                "call_volume": call_volume,
                "put_volume": put_volume,
                "put_call_volume_ratio": round(put_volume / call_volume, 6)
                if call_volume > 0
                else None,
                "call_open_interest": call_oi,
                "put_open_interest": put_oi,
                "put_call_open_interest_ratio": round(put_oi / call_oi, 6)
                if call_oi > 0
                else None,
                "call_volume_weighted_iv": weighted_mean(
                    calls, "implied_volatility", "volume"
                ),
                "put_volume_weighted_iv": weighted_mean(
                    puts, "implied_volatility", "volume"
                ),
                "call_25delta_iv": call25,
                "put_25delta_iv": put25,
                "put_minus_call_25delta_iv": round(put25 - call25, 8)
                if put25 is not None and call25 is not None
                else None,
                "unsigned_gamma_oi_exposure_1pct_move": round(
                    unsigned_gamma_exposure, 6
                ),
                "gamma_sign_status": "unknown_no_dealer_position_sign",
                "status": "diagnostic_only",
            }
        )
    return states


def fetch_snapshot(
    *,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    max_quote_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    timeout: float = 10.0,
    collected_at: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    moment = collected_at or now_cn()
    expiries, official_meta = official_underlyings_and_expiries(timeout=timeout)
    list_raw = sina_text(option_list_symbols(expiries), timeout=timeout)
    contracts = parse_contract_lists(list_raw, expiries)
    underlying_raw = sina_text(
        ["sh" + code for code in sorted(expiries)],
        timeout=timeout,
    )
    spot_rows = parse_underlying_quotes(underlying_raw, moment)
    contract_raw: dict[str, list[str]] = {}
    symbols = sorted(contracts)
    for index in range(0, len(symbols), 80):
        contract_raw.update(sina_text(symbols[index : index + 80], timeout=timeout))
    normalized: list[dict[str, Any]] = []
    official_date = str(official_meta.get("date") or "")
    official_trade_date_verified = (
        official_date == moment.strftime("%Y%m%d")
    )
    for symbol, metadata in contracts.items():
        spot = spot_rows.get(metadata["underlying"])
        fields = contract_raw.get(symbol)
        if spot is None or fields is None:
            continue
        row = parse_contract_quote(
            symbol,
            fields,
            metadata,
            spot,
            moment,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            max_quote_age_seconds=max_quote_age_seconds,
            snapshot_trade_date_verified=official_trade_date_verified,
        )
        if row is not None:
            normalized.append(row)
    states = aggregate_option_states(normalized, moment)
    expected = len(contracts)
    fresh = sum(row.get("is_fresh") for row in normalized)
    coverage = {
        "schemaVersion": "etf_option_coverage_v1",
        "collected_at": moment.isoformat(),
        "trade_date": moment.strftime("%Y-%m-%d"),
        "official_underlyings": sorted(expiries),
        "official_expiry_pairs": sum(len(months) for months in expiries.values()),
        "discovered_contracts": expected,
        "normalized_contracts": len(normalized),
        "fresh_contracts": fresh,
        "fresh_coverage_rate": round(fresh / expected, 6) if expected else 0.0,
        "pressure_states": len(states),
        "official_trade_date_matches_collection_date": official_trade_date_verified,
        "official_contract_calendar": official_meta,
        "quote_source": "Sina public option quote",
        "quote_source_tier": "vendor_not_direct_exchange_feed",
        "status": "ok" if expected > 0 and fresh == expected else "partial_coverage",
    }
    return normalized, states, coverage


def collect_once(
    *,
    risk_free_rate: float,
    dividend_yield: float,
    max_quote_age_seconds: int,
    timeout: float,
    write: bool = True,
) -> dict[str, Any]:
    rows, states, coverage = fetch_snapshot(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        max_quote_age_seconds=max_quote_age_seconds,
        timeout=timeout,
    )
    if write:
        day = coverage["trade_date"]
        append_jsonl(
            OUT_DIR / f"option_contracts_{day}.jsonl",
            [row for row in rows if row.get("is_fresh")],
        )
        append_jsonl(OUT_DIR / f"option_states_{day}.jsonl", states)
        append_jsonl(COVERAGE_DIR / f"coverage_{day}.jsonl", [coverage])
    return coverage


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--until", default="15:00")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-quote-age-seconds", type=int, default=DEFAULT_MAX_AGE_SECONDS)
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    parser.add_argument("--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD)
    args = parser.parse_args()

    moment = now_cn()
    if not is_xshg_session(moment.date()):
        print(
            json.dumps(
                {
                    "status": "skipped_non_trading_day",
                    "trade_date": moment.strftime("%Y-%m-%d"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.once:
        if not in_continuous_session(moment):
            print(
                json.dumps(
                    {
                        "status": "skipped_outside_continuous_session",
                        "trade_date": moment.strftime("%Y-%m-%d"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(
            json.dumps(
                collect_once(
                    risk_free_rate=args.risk_free_rate,
                    dividend_yield=args.dividend_yield,
                    max_quote_age_seconds=args.max_quote_age_seconds,
                    timeout=args.timeout_seconds,
                    write=not args.dry_run,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
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
            coverage = collect_once(
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
                max_quote_age_seconds=args.max_quote_age_seconds,
                timeout=args.timeout_seconds,
                write=True,
            )
            polls += 1
            print(
                f"{coverage['collected_at']} contracts={coverage['fresh_contracts']}/"
                f"{coverage['discovered_contracts']} states={coverage['pressure_states']} "
                f"status={coverage['status']}"
            )
        except Exception as exc:
            print(f"option collector error: {exc}", file=sys.stderr)
        elapsed = time.perf_counter() - started
        time.sleep(max(1.0, args.poll_seconds - elapsed))
    print(json.dumps({"status": "completed", "polls": polls}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
