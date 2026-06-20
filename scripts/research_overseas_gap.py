"""Overseas close -> A-share T+0 ETF overnight-gap research (offline only).

For A-share date D, reference data is restricted to the latest completed
overseas daily bar whose exchange-local date is strictly before D. ETF bars come
from the fixed official-T0 dataset without an end-of-day turnover universe gate.
No broker, order, agent state, live config or overlay is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from overfitting_guard import combinatorial_symmetric_pbo
from run_t0_strategy_evolution import deflated_sharpe_diagnostic


ROOT = Path(__file__).resolve().parents[1]
SH = timezone(timedelta(hours=8))
TRAIN_END = "2026-05-20"
OOS_START = "2026-05-21"
OOS_END = "2026-06-18"
ETF_QUOTES = ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
REF_CACHE = ROOT / "data" / "research" / "overseas_gap" / "yahoo_daily_refs_20260301_20260618.json"
OUT_JSON = ROOT / "outputs" / "edge_research" / "overseas_gap_oos_20260323_20260618.json"
OUT_MD = ROOT / "outputs" / "edge_research" / "overseas_gap_oos_20260323_20260618.md"
BASE_COST_BPS = 12.0
REFERENCES = {
    "IXIC": {"symbol": "^IXIC", "timezone": "America/New_York"},
    "GSPC": {"symbol": "^GSPC", "timezone": "America/New_York"},
    "ES": {"symbol": "ES=F", "timezone": "America/New_York"},
    "NQ": {"symbol": "NQ=F", "timezone": "America/New_York"},
    "SOX": {"symbol": "^SOX", "timezone": "America/New_York"},
    "GOLD": {"symbol": "GC=F", "timezone": "America/New_York"},
    "DXY": {"symbol": "DX-Y.NYB", "timezone": "America/New_York"},
}
ETF_MAP = {
    "513100": {"name": "纳指ETF", "primary": "IXIC", "references": ["IXIC", "NQ", "SOX", "DXY"]},
    "513500": {"name": "标普500ETF", "primary": "GSPC", "references": ["GSPC", "ES", "DXY"]},
    "518880": {"name": "黄金ETF", "primary": "GOLD", "references": ["GOLD", "DXY"]},
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(content)
        temp = Path(handle.name)
    temp.replace(path)


def fetch_daily(symbol: str, tz_name: str, start: str, end: str) -> list[dict[str, Any]]:
    start_ts = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int((datetime.fromisoformat(end) + timedelta(days=2)).replace(tzinfo=timezone.utc).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo returned no daily data for {symbol}")
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    tz = ZoneInfo(tz_name)
    rows = []
    previous = None
    for index, stamp in enumerate(timestamps):
        close = closes[index] if index < len(closes) else None
        if close is None or float(close) <= 0:
            continue
        local_date = datetime.fromtimestamp(int(stamp), tz).date().isoformat()
        ret = float(close) / previous - 1.0 if previous else None
        rows.append({"date": local_date, "close": float(close), "return": ret})
        previous = float(close)
    return rows


def load_references(path: Path, refresh: bool) -> dict[str, Any]:
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    series = {}
    failures = {}
    for name, meta in REFERENCES.items():
        try:
            series[name] = fetch_daily(meta["symbol"], meta["timezone"], "2026-03-01", OOS_END)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    payload = {
        "schemaVersion": "overseas_daily_reference_v1",
        "fetchedAt": datetime.now(SH).isoformat(timespec="seconds"),
        "source": "Yahoo Finance chart 1d",
        "series": series,
        "failures": failures,
    }
    atomic_json(path, payload)
    return payload


def load_etf_days(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            code = str(row.get("stockCode", ""))
            if code in ETF_MAP:
                grouped[(code, str(row["trade_date"]))].append(row)
    out = []
    for (code, trade_date), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.sort(key=lambda row: row["timestamp"])
        previous = rows[0].get("prev_close")
        if not previous or float(previous) <= 0 or len(rows) < 40:
            continue
        prices = [float(row["close"]) for row in rows]
        opening, closing = prices[0], prices[-1]
        out.append({
            "date": trade_date, "code": code, "name": ETF_MAP[code]["name"],
            "open": opening, "close": closing, "prev_close": float(previous),
            "gap": opening / float(previous) - 1.0,
            "open_close": closing / opening - 1.0,
            "open_high": max(prices) / opening - 1.0,
            "gap_fade": closing / float(previous) - 1.0,
        })
    return out


def strictly_prior_return(rows: list[dict[str, Any]], a_date: str) -> tuple[str, float] | None:
    eligible = [row for row in rows if row.get("date", "") < a_date and row.get("return") is not None]
    if not eligible:
        return None
    latest = max(eligible, key=lambda row: row["date"])
    return str(latest["date"]), float(latest["return"])


def align(features: list[dict[str, Any]], refs: dict[str, Any]) -> list[dict[str, Any]]:
    series = refs.get("series") or {}
    aligned = []
    for row in features:
        item = dict(row)
        used_dates = {}
        complete = True
        for ref in ETF_MAP[row["code"]]["references"]:
            match = strictly_prior_return(series.get(ref, []), row["date"])
            if match is None:
                complete = False
                break
            used_dates[ref], item[f"ref_{ref}"] = match
        if complete:
            item["reference_dates"] = used_dates
            aligned.append(item)
    return aligned


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx > 0 and sy > 0 else None


def explanation(rows: list[dict[str, Any]], ref: str) -> dict[str, Any]:
    x = [row[f"ref_{ref}"] for row in rows]
    y = [row["gap"] for row in rows]
    corr = correlation(x, y)
    return {"samples": len(rows), "correlation": round(corr, 4) if corr is not None else None,
            "r_squared": round(corr * corr, 4) if corr is not None else None}


def rule_defs(ref: str) -> dict[str, Any]:
    return {
        "positive_any_continuation": lambda row: row[f"ref_{ref}"] > 0,
        "positive_050_continuation": lambda row: row[f"ref_{ref}"] > 0.005,
        "positive_100_continuation": lambda row: row[f"ref_{ref}"] > 0.01,
        "negative_any_rebound": lambda row: row[f"ref_{ref}"] < 0,
        "negative_050_rebound": lambda row: row[f"ref_{ref}"] < -0.005,
        "positive_050_high_gap_continuation": lambda row: row[f"ref_{ref}"] > 0.005 and row["gap"] > 0.005,
    }


def daily_rule_returns(rows: list[dict[str, Any]], ref: str, cost_bps: float) -> dict[str, dict[str, float]]:
    result = {}
    for name, predicate in rule_defs(ref).items():
        result[name] = {row["date"]: row["open_close"] - cost_bps / 10_000.0 for row in rows if predicate(row)}
    return result


def performance(values: dict[str, float]) -> dict[str, Any]:
    xs = list(values.values())
    if not xs:
        return {"trades": 0, "mean_pct": None, "win_rate": None, "sharpe": None}
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {
        "trades": len(xs), "mean_pct": round(statistics.fmean(xs) * 100, 4),
        "median_pct": round(statistics.median(xs) * 100, 4),
        "win_rate": round(sum(value > 0 for value in xs) / len(xs), 4),
        "sharpe": round(statistics.fmean(xs) / sd * math.sqrt(252), 4) if sd > 0 else None,
        "tail_p05_pct": round(sorted(xs)[max(0, int(len(xs) * 0.05) - 1)] * 100, 4),
    }


def pbo_for_train(rule_returns: dict[str, dict[str, float]], train_dates: list[str]) -> dict[str, Any]:
    matrix = [[values.get(day, 0.0) for day in train_dates] for values in rule_returns.values()]
    return combinatorial_symmetric_pbo(matrix, n_blocks=8)


def analyze_code(code: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = ETF_MAP[code]["primary"]
    train = [row for row in rows if row["date"] <= TRAIN_END]
    oos = [row for row in rows if OOS_START <= row["date"] <= OOS_END]
    all_rules = daily_rule_returns(rows, primary, BASE_COST_BPS)
    train_rules = daily_rule_returns(train, primary, BASE_COST_BPS)
    oos_rules = daily_rule_returns(oos, primary, BASE_COST_BPS)
    selected = max(train_rules, key=lambda name: statistics.fmean(train_rules[name].values()) if train_rules[name] else -1e9)
    oos_dates = sorted({row["date"] for row in oos})
    baseline = {day: 0.0 for day in oos_dates}
    selected_full = {day: oos_rules[selected].get(day, 0.0) for day in oos_dates}
    dsr = deflated_sharpe_diagnostic(baseline, selected_full, n_trials=len(train_rules), alpha=0.10)
    return {
        "primaryReference": primary,
        "samples": {"train": len(train), "oos": len(oos)},
        "explanation": {ref: {"train": explanation(train, ref), "oos": explanation(oos, ref)}
                        for ref in ETF_MAP[code]["references"]},
        "descriptive": {
            "oosMeanGapPct": round(statistics.fmean(row["gap"] for row in oos) * 100, 4) if oos else None,
            "oosMeanOpenClosePct": round(statistics.fmean(row["open_close"] for row in oos) * 100, 4) if oos else None,
            "oosMeanOpenHighPct": round(statistics.fmean(row["open_high"] for row in oos) * 100, 4) if oos else None,
        },
        "rules": {name: {"train": performance(train_rules[name]), "oos": performance(oos_rules[name])}
                  for name in train_rules},
        "trainSelectedRule": selected,
        "selectedRuleOOS": performance(oos_rules[selected]),
        "pboTrain": pbo_for_train(all_rules, sorted({row["date"] for row in train})),
        "dsrOOS": dsr,
        "statisticalPass": bool(dsr.get("significant") and performance(oos_rules[selected])["trades"] >= 8),
    }


def build_report(aligned: list[dict[str, Any]], refs: dict[str, Any]) -> dict[str, Any]:
    results = {code: analyze_code(code, [row for row in aligned if row["code"] == code]) for code in ETF_MAP}
    all_pass = all(result["statisticalPass"] for result in results.values())
    premium_available = False
    return {
        "schemaVersion": "overseas_gap_oos_research_v1",
        "generatedAt": datetime.now(SH).isoformat(timespec="seconds"),
        "paperTradingOnly": True, "status": "diagnostic_only", "edgeValidated": False,
        "liveReady": False, "formalStrategyAllowed": False, "orderSubmitCallsMade": False,
        "data": {"etfSource": str(ETF_QUOTES), "referenceSource": refs.get("source"),
                 "train": ["2026-03-23", TRAIN_END], "oos": [OOS_START, OOS_END],
                 "alignedSamples": len(aligned), "futureDayTurnoverUsed": False},
        "costModel": {"roundtripBps": BASE_COST_BPS},
        "alignmentRule": "latest completed overseas exchange-local daily bar with date strictly before A-share date",
        "premiumFilter": {"available": premium_available, "status": "not_run_missing_point_in_time_iopv"},
        "results": results,
        "promotion": {"statisticalRulesPass": all_pass, "premiumGatePass": premium_available,
                      "passed": False, "reason": "point-in-time IOPV/premium history unavailable"},
        "conclusion": "No promotable edge: statistical results are diagnostic and the required premium gate cannot be tested.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Overseas Close → A-share ETF Gap Research", "",
             "Offline diagnostic; OOS 2026-05-21..2026-06-18; 12 bps round-trip cost.", "",
             "| ETF | Primary | OOS gap corr | Train-selected rule | OOS trades | OOS net mean | DSR pass |",
             "|---|---|---:|---|---:|---:|---|"]
    for code, result in report["results"].items():
        primary = result["primaryReference"]
        corr = result["explanation"][primary]["oos"]["correlation"]
        perf = result["selectedRuleOOS"]
        lines.append(f"| {code} {ETF_MAP[code]['name']} | {primary} | {corr} | {result['trainSelectedRule']} | "
                     f"{perf['trades']} | {perf['mean_pct']}% | {result['dsrOOS'].get('significant', False)} |")
    lines += ["", "## Decision", "", f"- {report['conclusion']}",
              "- Premium filter: not run; historical point-in-time IOPV is absent.",
              "- No signal/config/overlay change is allowed from this result."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", default=str(ETF_QUOTES))
    parser.add_argument("--reference-cache", default=str(REF_CACHE))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()
    refs = load_references(Path(args.reference_cache), args.refresh)
    features = load_etf_days(Path(args.quotes))
    aligned = align(features, refs)
    if not aligned:
        raise RuntimeError("no aligned overseas/ETF samples")
    report = build_report(aligned, refs)
    atomic_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
