"""Build a fail-closed research master for A-share listed ETFs.

The master deliberately separates official product classification from name
heuristics.  An ETF is marked ``t0_confirmed`` only when an exchange product
classification and an exchange trading rule jointly support same-day resale.
The output is diagnostic research data and is not read by the trading agent.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CN = ZoneInfo("Asia/Shanghai")
UNIVERSE_DIR = ROOT / "data" / "market" / "eastmoney" / "universe"
OFFICIAL_DIR = ROOT / "data" / "research" / "t0_etf_master" / "official"
OUT_DIR = ROOT / "outputs" / "edge_research"
YAHOO_QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"

SSE_LIST_URL = "https://query.sse.com.cn/commonSoaQuery.do"
SSE_LIST_PAGE = "https://www.sse.com.cn/assortment/fund/etf/list/"
SZSE_LIST_PAGE = "https://www.szse.cn/market/product/list/etfList/index.html"
SZSE_RULE_URL = "https://docs.static.szse.cn/www/lawrules/rule/fund/trade/W020220610536157903323.pdf"
SSE_RULE_URL = "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20250519_10779396.shtml"
SSE_BOND_RULE_URL = "https://etf.sse.com.cn/fund/learning/knowledge/c/5704296.shtml"

# SSE's official FUND_LIST classification values.  The trading-rule evidence
# permits same-day resale for bond, gold, eligible cross-border and commodity
# futures ETFs.  34 explicitly says intraday round-trip; 35 explicitly says no.
SSE_CLASS = {
    "01": ("domestic_equity", "explicit_t1", "single-market domestic equity ETF"),
    "02": ("bond", "confirmed", "single-market bond ETF"),
    "03": ("domestic_equity", "explicit_t1", "cross-market domestic equity ETF"),
    "04": ("cross_border", "confirmed", "legacy cross-border ETF class"),
    "05": ("money_market", "excluded_money", "exchange-traded money fund"),
    "06": ("gold", "confirmed", "gold ETF"),
    "07": ("money_market", "excluded_money", "exchange-traded money fund"),
    "08": ("domestic_equity", "explicit_t1", "domestic/HK mixed equity ETF"),
    "09": ("domestic_equity", "explicit_t1", "STAR Market equity ETF"),
    "31": ("domestic_equity", "explicit_t1", "domestic equity ETF including STAR"),
    "32": ("bond", "confirmed", "cross-market bond ETF"),
    "33": ("cross_border", "confirmed", "cross-border ETF"),
    "34": ("cross_border", "confirmed", "multi-market cross-border ETF (intraday round-trip)"),
    "35": ("cross_border", "explicit_t1", "multi-market cross-border ETF (non-intraday)"),
    "36": ("bond", "confirmed", "interbank cross-market bond ETF"),
    "37": ("bond", "confirmed", "cash-subscription bond ETF"),
    "38": ("commodity_futures", "confirmed", "commodity futures ETF"),
}

MONEY_TERMS = ("货币", "现金管理", "理财", "添富快线", "保证金")
NAME_HINTS = (
    ("bond_candidate", ("债", "国债", "政金债", "信用债", "可转债")),
    ("gold_candidate", ("黄金",)),
    ("commodity_candidate", ("商品", "豆粕", "有色期货", "能源化工")),
    ("cross_border_candidate", ("纳指", "纳斯达克", "标普", "日经", "德国", "法国", "恒生", "港股", "中概", "沙特", "东南亚")),
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)
    json.loads(path.read_text(encoding="utf-8"))


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def latest_file(directory: Path, pattern: str) -> Path | None:
    paths = sorted(directory.glob(pattern))
    return paths[-1] if paths else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def fetch_sse_official(timeout_seconds: float = 60.0) -> list[dict[str, Any]]:
    params = {
        "isPagination": "true",
        "sqlId": "FUND_LIST",
        "pageHelp.pageSize": "3000",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.endPage": "1",
        "fundType": "00",
        "subClass": ",".join(SSE_CLASS),
    }
    req = Request(
        f"{SSE_LIST_URL}?{urlencode(params)}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": SSE_LIST_PAGE, "Accept": "application/json"},
    )
    with urlopen(req, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("pageHelp", {}).get("data") or payload.get("result") or []
    if not isinstance(rows, list) or len(rows) < 100:
        raise RuntimeError(f"SSE FUND_LIST returned an implausible row count: {len(rows) if isinstance(rows, list) else 'non-list'}")
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        code = str(row.get("fundCode", "")).zfill(6)
        subclass = str(row.get("subClass", ""))
        if re.fullmatch(r"\d{6}", code) and subclass in SSE_CLASS:
            cleaned.append({
                "stockCode": code,
                "exchange": "SH",
                "name": row.get("secNameFull") or row.get("fundAbbr") or code,
                "subClass": subclass,
                "listingDate": row.get("listingDate"),
                "benchmarkIndex": row.get("INDEX_NAME") or None,
                "benchmarkCode": row.get("INDEX_CODE") or None,
                "manager": row.get("companyName") or None,
            })
    if len({row["stockCode"] for row in cleaned}) != len(cleaned):
        raise RuntimeError("duplicate composite codes in SSE official response")
    return sorted(cleaned, key=lambda row: row["stockCode"])


def load_sse_official(refresh: bool, as_of: str) -> tuple[list[dict[str, Any]], Path, bool]:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    path = OFFICIAL_DIR / f"sse_fund_list_{as_of.replace('-', '')}.json"
    if refresh or not path.exists():
        rows = fetch_sse_official()
        atomic_json(path, {
            "schemaVersion": "sse_official_fund_list_v1",
            "fetchedAt": datetime.now(CN).isoformat(timespec="seconds"),
            "sourceUrl": SSE_LIST_PAGE,
            "apiUrl": SSE_LIST_URL,
            "rows": rows,
        })
        return rows, path, True
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"], path, False


def is_money_like(name: str) -> bool:
    text = str(name or "")
    if "现金流" in text:
        return False
    return any(term in text for term in MONEY_TERMS)


def name_candidate(name: str) -> str:
    if is_money_like(name):
        return "money_market_candidate"
    for label, terms in NAME_HINTS:
        if any(term in str(name or "") for term in terms):
            return label
    return "unclassified_candidate"


def liquidity_20d(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Compute point-in-time turnover; reject Yahoo's synthetic spread as evidence."""
    if not path.exists():
        return {}, {"available": False, "reason": "quote_file_missing"}
    daily_amount: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    dates: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                code = str(row.get("stockCode", "")).zfill(6)
                exchange = str(row.get("exchange", ""))
                date = str(row.get("timestamp", ""))[:10]
                amount = float(row.get("amount") or 0.0)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if re.fullmatch(r"\d{6}", code) and exchange in {"SH", "SZ"} and len(date) == 10 and amount >= 0:
                key = (code, exchange)
                daily_amount[key][date] = max(amount, daily_amount[key].get(date, 0.0))
                dates.add(date)
                row_count += 1
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, by_date in daily_amount.items():
        chosen = sorted(by_date)[-20:]
        result[key] = {
            "average_turnover_20d": round(statistics.fmean(by_date[date] for date in chosen), 2),
            "turnover_observation_days": len(chosen),
            "turnover_window_start": chosen[0],
            "turnover_window_end": chosen[-1],
            "average_spread": None,
            "spread_note": "unavailable: Yahoo replay bid/ask was synthetic and is not valid spread evidence",
        }
    return result, {
        "available": bool(result),
        "source": str(path),
        "rows": row_count,
        "trade_dates": len(dates),
        "date_start": min(dates) if dates else None,
        "date_end": max(dates) if dates else None,
        "instruments_with_turnover": len(result),
        "spread_is_real": False,
    }


def official_row_to_master(row: dict[str, Any], liquidity: dict[str, Any]) -> dict[str, Any]:
    asset_class, status, description = SSE_CLASS[row["subClass"]]
    money = status == "excluded_money"
    sources = [SSE_LIST_PAGE, SSE_RULE_URL]
    if asset_class == "bond":
        sources.append(SSE_BOND_RULE_URL)
    return {
        "code": row["stockCode"],
        "name": row["name"],
        "exchange": "SH",
        "asset_class": asset_class,
        "asset_class_candidate": asset_class,
        "classification_method": "sse_official_fund_subclass",
        "official_subclass": row["subClass"],
        "listing_date": row.get("listingDate"),
        "t0_status": status,
        "t0_confirmed": status == "confirmed" and not money,
        "is_money_like": money,
        "confirmation_basis": description,
        "average_turnover_20d": liquidity.get("average_turnover_20d"),
        "turnover_observation_days": liquidity.get("turnover_observation_days", 0),
        "average_spread": liquidity.get("average_spread"),
        "spread_note": liquidity.get("spread_note", "no local spread observations"),
        "premium_discount_available": False,
        "underlying_market_hours": None,
        "has_night_reference": True if asset_class in {"gold", "commodity_futures"} else None,
        "benchmark_index": row.get("benchmarkIndex"),
        "benchmark_code": row.get("benchmarkCode"),
        "data_source": sources,
    }


def pending_row_to_master(row: dict[str, Any], exchange: str, liquidity: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("name") or row.get("stockCode") or "")
    money = is_money_like(name)
    return {
        "code": str(row.get("stockCode", "")).zfill(6),
        "name": name,
        "exchange": exchange,
        "asset_class": "unknown",
        "asset_class_candidate": name_candidate(name),
        "classification_method": "name_hint_not_confirmation",
        "official_subclass": None,
        "listing_date": None,
        "t0_status": "excluded_money" if money else "pending_verification",
        "t0_confirmed": False,
        "is_money_like": money,
        "confirmation_basis": "no product-level official T+0 classification available locally; name hint cannot confirm T+0",
        "average_turnover_20d": liquidity.get("average_turnover_20d"),
        "turnover_observation_days": liquidity.get("turnover_observation_days", 0),
        "average_spread": None,
        "spread_note": liquidity.get("spread_note", "no local spread observations"),
        "premium_discount_available": False,
        "underlying_market_hours": None,
        "has_night_reference": None,
        "benchmark_index": None,
        "benchmark_code": None,
        "data_source": [SZSE_LIST_PAGE if exchange == "SZ" else SSE_LIST_PAGE],
    }


def build_master(universe: list[dict[str, Any]], sse_rows: list[dict[str, Any]], liquidity: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    local: dict[tuple[str, str], dict[str, Any]] = {}
    for row in universe:
        code = str(row.get("stockCode", "")).zfill(6)
        exchange = "SH" if str(row.get("market", "")) == "1" or code.startswith("5") else "SZ"
        local[(code, exchange)] = row
    official = {(row["stockCode"], "SH"): row for row in sse_rows}
    keys = sorted(set(local) | set(official))
    rows: list[dict[str, Any]] = []
    for key in keys:
        code, exchange = key
        if key in official:
            item = official_row_to_master(official[key], liquidity.get(key, {}))
        else:
            item = pending_row_to_master(local[key], exchange, liquidity.get(key, {}))
        item["in_local_non_money_universe"] = key in local
        item["paper_research_only"] = True
        rows.append(item)
    return rows


def validate_master(rows: list[dict[str, Any]]) -> None:
    required = {
        "code", "name", "exchange", "asset_class", "t0_confirmed", "is_money_like",
        "average_turnover_20d", "average_spread", "premium_discount_available",
        "underlying_market_hours", "has_night_reference", "benchmark_index", "data_source",
    }
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"master row {index} missing fields: {sorted(missing)}")
        key = (str(row["code"]), str(row["exchange"]))
        if key in seen:
            raise ValueError(f"duplicate master composite key: {key}")
        seen.add(key)
        if not re.fullmatch(r"\d{6}", key[0]) or key[1] not in {"SH", "SZ"}:
            raise ValueError(f"invalid master composite key: {key}")
        if row["t0_confirmed"] and (row["is_money_like"] or row["t0_status"] != "confirmed"):
            raise ValueError(f"unsafe T0 confirmation: {key}")
        if row["t0_confirmed"] and row["classification_method"] != "sse_official_fund_subclass":
            raise ValueError(f"T0 confirmation lacks official product class: {key}")


def count_real_watchlists() -> tuple[int, list[str]]:
    inbox = ROOT / "data" / "research" / "chatgpt_etf_watchlist" / "inbox"
    valid: list[str] = []
    for path in sorted(inbox.glob("*.json")) if inbox.exists() else []:
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schemaVersion") == "chatgpt_etf_watchlist_v1" and payload.get("etfs"):
                valid.append(path.name)
        except (OSError, json.JSONDecodeError):
            pass
    return len(valid), valid


def readiness_report(master: list[dict[str, Any]], universe_path: Path, official_path: Path, liquidity_meta: dict[str, Any]) -> dict[str, Any]:
    real_watchlists, watchlist_files = count_real_watchlists()
    counts = defaultdict(int)
    for row in master:
        counts[row["t0_status"]] += 1
    confirmed_local = sum(bool(row["t0_confirmed"] and row["in_local_non_money_universe"]) for row in master)
    confirmed_liquid = sum(bool(
        row["t0_confirmed"] and row["in_local_non_money_universe"]
        and int(row.get("turnover_observation_days") or 0) >= 20
    ) for row in master)
    quote_days = int(liquidity_meta.get("trade_dates") or 0)
    return {
        "schemaVersion": "t0_edge_research_readiness_v1",
        "generatedAt": datetime.now(CN).isoformat(timespec="seconds"),
        "diagnosticOnly": True,
        "liveReady": False,
        "formalStrategyAllowed": False,
        "inputs": {
            "localUniverse": str(universe_path),
            "officialSseClassification": str(official_path),
            "liquidity": liquidity_meta,
            "realChatgptWatchlistDays": real_watchlists,
            "realChatgptWatchlistFiles": watchlist_files,
        },
        "t0Master": {
            "rows": len(master),
            "confirmed": counts["confirmed"],
            "confirmedInLocalNonMoneyUniverse": confirmed_local,
            "confirmedWith20DayTurnover": confirmed_liquid,
            "explicitT1": counts["explicit_t1"],
            "pendingVerification": counts["pending_verification"],
            "excludedMoney": counts["excluded_money"],
            "limitation": "SZSE public list lacks a product-level T+0 class; SZ names remain fail-closed pending verification",
        },
        "directions": [
            {"id": 1, "name": "watchlist effectiveness", "status": "blocked", "evidence": f"{real_watchlists} real watchlist days; need point-in-time history"},
            {"id": 2, "name": "T+0 ETF master", "status": "partial", "evidence": f"{counts['confirmed']} confirmed; {counts['pending_verification']} pending"},
            {"id": 3, "name": "cross-market lead signals", "status": "blocked", "evidence": "no timestamp-aligned futures/FX/yield series"},
            {"id": 4, "name": "premium/discount filter", "status": "blocked", "evidence": "no point-in-time IOPV/NAV series"},
            {"id": 5, "name": "opening 30 minutes", "status": "partially_ready", "evidence": f"{quote_days} Yahoo 5-minute days; no validated watchlist split"},
            {"id": 6, "name": "trend vs mean reversion", "status": "partially_ready", "evidence": f"{quote_days} days; only officially classified SH instruments can be grouped safely"},
            {"id": 7, "name": "liquidity/slippage", "status": "partial", "evidence": "20-day turnover available for some instruments; historical spread/depth unavailable"},
            {"id": 8, "name": "news decay", "status": "blocked", "evidence": f"{real_watchlists} real watchlist days"},
            {"id": 9, "name": "ETF relative strength", "status": "partially_ready", "evidence": "price/turnover usable; theme, premium and news relevance history absent"},
            {"id": 10, "name": "kill switch", "status": "partial", "evidence": "daily-loss and execution safeguards exist; premium/news/data-source gates require missing inputs"},
        ],
        "nextDataActions": [
            "resolve SZ product-level categories from official fund announcements or a documented exchange field",
            "accumulate at least 20 point-in-time ChatGPT watchlist trading days",
            "collect real bid/ask/depth and IOPV/NAV rather than synthetic replay values",
            "add timestamp-aligned external futures, FX, yields and commodity references",
        ],
    }


def report_markdown(report: dict[str, Any]) -> str:
    master = report["t0Master"]
    lines = [
        "# T+0 ETF Edge Research — Phase 1 Readiness",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        "Research only. No live configuration or execution path is changed.",
        "",
        "## T+0 master",
        "",
        f"- Total rows: {master['rows']}",
        f"- Officially confirmed T+0: {master['confirmed']}",
        f"- Confirmed and present in local non-money universe: {master['confirmedInLocalNonMoneyUniverse']}",
        f"- Confirmed with 20 turnover observations: {master['confirmedWith20DayTurnover']}",
        f"- Explicit T+1: {master['explicitT1']}",
        f"- Pending product verification: {master['pendingVerification']}",
        f"- Money-like excluded: {master['excludedMoney']}",
        f"- Limitation: {master['limitation']}",
        "",
        "## Research directions",
        "",
        "| # | Direction | Status | Evidence |",
        "|---:|---|---|---|",
    ]
    for row in report["directions"]:
        lines.append(f"| {row['id']} | {row['name']} | {row['status']} | {row['evidence']} |")
    lines += ["", "## Next data actions", ""]
    lines += [f"- {item}" for item in report["nextDataActions"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-official", action="store_true", help="refresh SSE official fund classification")
    parser.add_argument("--as-of", default=datetime.now(CN).date().isoformat())
    parser.add_argument("--universe-file", default="")
    parser.add_argument("--quotes", default=str(YAHOO_QUOTES))
    args = parser.parse_args()

    universe_path = Path(args.universe_file) if args.universe_file else latest_file(UNIVERSE_DIR, "eastmoney_universe_etf_*.jsonl")
    if not universe_path or not universe_path.exists():
        raise FileNotFoundError("no local Eastmoney ETF universe")
    universe = read_jsonl(universe_path)
    sse_rows, official_path, refreshed = load_sse_official(args.refresh_official, args.as_of)
    liquidity, liquidity_meta = liquidity_20d(Path(args.quotes))
    master = build_master(universe, sse_rows, liquidity)
    validate_master(master)
    report = readiness_report(master, universe_path, official_path, liquidity_meta)

    stamp = args.as_of.replace("-", "")
    master_path = OUT_DIR / f"t0_etf_master_{stamp}.jsonl"
    latest_path = OUT_DIR / "t0_etf_master_latest.jsonl"
    confirmed_path = OUT_DIR / f"t0_etf_confirmed_pool_{stamp}.jsonl"
    confirmed_latest_path = OUT_DIR / "t0_etf_confirmed_pool_latest.jsonl"
    report_path = OUT_DIR / f"phase1_readiness_{stamp}.json"
    markdown_path = OUT_DIR / f"phase1_readiness_{stamp}.md"
    atomic_jsonl(master_path, master)
    atomic_jsonl(latest_path, master)
    confirmed = [row for row in master if row["t0_confirmed"] and row["in_local_non_money_universe"]]
    atomic_jsonl(confirmed_path, confirmed)
    atomic_jsonl(confirmed_latest_path, confirmed)
    atomic_json(report_path, report)
    markdown_path.write_text(report_markdown(report), encoding="utf-8")

    print(json.dumps({
        "master": str(master_path),
        "master_latest": str(latest_path),
        "confirmed_pool": str(confirmed_path),
        "confirmed_pool_latest": str(confirmed_latest_path),
        "readiness": str(report_path),
        "readiness_markdown": str(markdown_path),
        "official_cache": str(official_path),
        "official_refreshed": refreshed,
        "counts": report["t0Master"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
