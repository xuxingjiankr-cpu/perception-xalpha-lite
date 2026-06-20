"""Build the next-session ETF observation pool (system 20 + ChatGPT 10).

This is a research/observation input only.  It never calls a broker, submits an
order, or changes the paper agent's entry gates.  The system leg is rebuilt from
the latest Eastmoney full-market snapshot using the existing dynamic-universe
ranking.  The optional ChatGPT leg is validated against the locally collected
non-money ETF master before it is merged.

Examples:
  py -3.13 scripts/build_t0_observation_pool.py
  py -3.13 scripts/build_t0_observation_pool.py --chatgpt-file watchlist.json
  Get-Clipboard | py -3.13 scripts/build_t0_observation_pool.py --chatgpt-file -
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, load_json
from select_t0_universe import select_dynamic_universe
from archive_t0_observation_pool import DEFAULT_HISTORY, archive_document


SH = ZoneInfo("Asia/Shanghai")
DEFAULT_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
DEFAULT_INBOX = ROOT / "data" / "research" / "chatgpt_etf_watchlist" / "inbox"
DEFAULT_OUT = ROOT / "outputs" / "t0_observation_pool"
CODE_RE = re.compile(r"^\d{6}$")


def exchange_from_market(value: Any) -> str:
    return "SH" if str(value).strip() == "1" else "SZ"


def normalize_exchange(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {"1": "SH", "SH": "SH", "SSE": "SH", "0": "SZ", "SZ": "SZ", "SZSE": "SZ"}
    return aliases.get(text, "")


def latest_universe_file(universe_dir: Path) -> Path | None:
    files = sorted(universe_dir.glob("eastmoney_universe_etf_*.jsonl"))
    return files[-1] if files else None


def load_non_money_master(universe_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Path]:
    path = latest_universe_file(universe_dir)
    if path is None:
        raise FileNotFoundError(f"no ETF universe file under {universe_dir}")
    master: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        code = str(row.get("stockCode", "")).strip().zfill(6)
        exchange = exchange_from_market(row.get("market"))
        if CODE_RE.fullmatch(code):
            master[(code, exchange)] = row
    return master, path


def latest_snapshot_date(snapshot_dir: Path) -> str:
    dates = sorted(
        p.name for p in snapshot_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name) and any(p.glob("*_etf.csv.gz"))
    ) if snapshot_dir.exists() else []
    if not dates:
        raise FileNotFoundError(f"no ETF snapshots under {snapshot_dir}")
    return dates[-1]


def build_system_20(cfg: dict[str, Any], limit: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dyn = deepcopy(cfg.get("dynamic_universe", {}))
    dyn["top_n"] = limit
    snapshot_dir = ROOT / dyn.get("snapshot_dir", "data/market/eastmoney/full_market/snapshots")
    universe_dir = ROOT / dyn.get("universe_dir", "data/market/eastmoney/universe")
    as_of = latest_snapshot_date(snapshot_dir)
    result = select_dynamic_universe(snapshot_dir, universe_dir, as_of, dyn)
    if not result.get("meta", {}).get("ok"):
        raise RuntimeError(f"system ranking failed: {result.get('meta')}")
    return result["selected"][:limit], result["meta"]


def read_payload(path_text: str | None, inbox: Path) -> tuple[dict[str, Any] | None, str | None]:
    if path_text == "-":
        return json.load(sys.stdin), "stdin"
    if path_text:
        path = Path(path_text)
        return load_json(path), str(path.resolve())
    files = sorted(inbox.glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name)) if inbox.exists() else []
    if not files:
        return None, None
    path = files[-1]
    return load_json(path), str(path.resolve())


def validate_chatgpt_payload(
    payload: dict[str, Any] | None,
    master: dict[tuple[str, str], dict[str, Any]],
    limit: int = 10,
    today: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if payload is None:
        return [], [], {}
    if not isinstance(payload, dict):
        raise ValueError("ChatGPT payload must be a JSON object")
    rows = payload.get("etfs")
    if not isinstance(rows, list):
        raise ValueError("ChatGPT payload.etfs must be an array")
    if len(rows) > limit:
        raise ValueError(f"ChatGPT payload has {len(rows)} ETFs; maximum is {limit}")

    meta = {
        "schemaVersion": payload.get("schemaVersion"),
        "asOfDate": payload.get("asOfDate"),
        "effectiveDate": payload.get("effectiveDate"),
        "generatedAt": payload.get("generatedAt"),
    }
    if meta["schemaVersion"] != "chatgpt_etf_watchlist_v1":
        raise ValueError("unsupported ChatGPT payload schemaVersion")
    effective = str(meta.get("effectiveDate") or "")
    try:
        datetime.strptime(effective, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("ChatGPT payload.effectiveDate must be YYYY-MM-DD") from exc
    today = today or datetime.now(tz=SH).strftime("%Y-%m-%d")
    if effective < today:
        return [], [{"reason": "stale_effective_date", "effectiveDate": effective, "today": today}], meta

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            rejected.append({"index": index, "reason": "not_an_object"})
            continue
        code = str(raw.get("stockCode", "")).strip()
        exchange = normalize_exchange(raw.get("exchange"))
        key = (code, exchange)
        reason = str(raw.get("reason") or raw.get("newsThesis") or "").strip()
        urls = raw.get("sourceUrls", [])
        if not CODE_RE.fullmatch(code):
            rejected.append({"index": index, "stockCode": code, "reason": "invalid_code"})
        elif not exchange:
            rejected.append({"index": index, "stockCode": code, "reason": "invalid_exchange"})
        elif key not in master:
            rejected.append({"index": index, "stockCode": code, "exchange": exchange,
                             "reason": "not_in_local_non_money_etf_master"})
        elif not reason:
            rejected.append({"index": index, "stockCode": code, "exchange": exchange,
                             "reason": "missing_news_reason"})
        elif not isinstance(urls, list) or not any(str(url).startswith(("http://", "https://")) for url in urls):
            rejected.append({"index": index, "stockCode": code, "exchange": exchange,
                             "reason": "missing_source_url"})
        elif key in seen:
            rejected.append({"index": index, "stockCode": code, "exchange": exchange,
                             "reason": "duplicate_inside_chatgpt_list"})
        else:
            seen.add(key)
            canonical = master[key]
            accepted.append({
                "stockCode": code,
                "exchange": exchange,
                "name": canonical.get("name") or raw.get("name") or code,
                "reason": reason,
                "risks": raw.get("risks", []),
                "newsDrivers": raw.get("newsDrivers", []),
                "sourceUrls": [str(url) for url in urls if str(url).startswith(("http://", "https://"))],
            })
    return accepted, rejected, meta


def merge_observation_pool(
    system_rows: list[dict[str, Any]],
    chatgpt_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    combined: list[dict[str, Any]] = []
    index: dict[tuple[str, str], int] = {}
    for rank, row in enumerate(system_rows, 1):
        item = dict(row)
        item["sources"] = ["system_rank20"]
        item["systemRank"] = rank
        key = (str(item.get("stockCode", "")).zfill(6), normalize_exchange(item.get("exchange")))
        index[key] = len(combined)
        combined.append(item)

    overlaps = 0
    for row in chatgpt_rows:
        key = (row["stockCode"], row["exchange"])
        if key in index:
            overlaps += 1
            item = combined[index[key]]
            item["sources"].append("chatgpt_news10")
            item["chatgptResearch"] = {k: v for k, v in row.items() if k not in {"stockCode", "exchange", "name"}}
        else:
            item = dict(row)
            item["sources"] = ["chatgpt_news10"]
            combined.append(item)
            index[key] = len(combined) - 1
    return combined, overlaps


def build_document(
    system_rows: list[dict[str, Any]],
    system_meta: dict[str, Any],
    chatgpt_rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    chatgpt_meta: dict[str, Any],
    master_path: Path,
    input_source: str | None,
) -> dict[str, Any]:
    combined, overlaps = merge_observation_pool(system_rows, chatgpt_rows)
    return {
        "schemaVersion": "t0_etf_observation_pool_v1",
        "generatedAt": datetime.now(tz=SH).isoformat(timespec="seconds"),
        "asOfDate": system_meta.get("trade_date"),
        "effectiveSession": chatgpt_meta.get("effectiveDate") or "next_trading_session",
        "status": "ready" if len(chatgpt_rows) == 10 else "awaiting_or_partial_chatgpt_list",
        "paperTradingOnly": True,
        "diagnosticOnly": True,
        "tradeGateEnabled": False,
        "liveReady": False,
        "formalStrategyAllowed": False,
        "counts": {
            "system": len(system_rows),
            "chatgptAccepted": len(chatgpt_rows),
            "chatgptRejected": len(rejected),
            "overlaps": overlaps,
            "combinedUnique": len(combined),
        },
        "systemSelectionMeta": system_meta,
        "chatgptInputMeta": {**chatgpt_meta, "sourceFile": input_source},
        "validation": {"nonMoneyMaster": str(master_path.resolve()), "rejected": rejected},
        "system20": system_rows,
        "chatgpt10": chatgpt_rows,
        "combined": combined,
    }


def publish(document: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = str(document.get("effectiveSession") or document.get("asOfDate") or "unknown")
    safe_session = re.sub(r"[^0-9A-Za-z_-]", "_", session)
    dated = out_dir / f"observation_pool_{safe_session}.json"
    latest = out_dir / "latest_observation_pool.json"
    content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    dated.write_text(content, encoding="utf-8")
    latest.write_text(content, encoding="utf-8")
    return dated, latest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Build system-20 + ChatGPT-10 ETF observation pool")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--chatgpt-file", help="JSON file path, or '-' to read JSON from stdin")
    ap.add_argument("--inbox", help="Used when --chatgpt-file is omitted; defaults to config")
    ap.add_argument("--output-dir", help="Defaults to the directory containing config observation_pool.output")
    ap.add_argument("--history-dir", default=str(DEFAULT_HISTORY),
                    help="Point-in-time daily archive and ETF-level history directory")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    pool_cfg = cfg.get("observation_pool", {})
    system_limit = int(pool_cfg.get("system_limit", 20))
    chatgpt_limit = int(pool_cfg.get("chatgpt_limit", 10))
    inbox = Path(args.inbox) if args.inbox else ROOT / pool_cfg.get("inbox", str(DEFAULT_INBOX))
    configured_output = ROOT / pool_cfg.get("output", str(DEFAULT_OUT / "latest_observation_pool.json"))
    out_dir = Path(args.output_dir) if args.output_dir else configured_output.parent
    inbox.mkdir(parents=True, exist_ok=True)
    universe_dir = ROOT / cfg.get("dynamic_universe", {}).get("universe_dir", "data/market/eastmoney/universe")
    master, master_path = load_non_money_master(universe_dir)
    system_rows, system_meta = build_system_20(cfg, system_limit)
    payload, source = read_payload(args.chatgpt_file, inbox)
    chatgpt_rows, rejected, chatgpt_meta = validate_chatgpt_payload(payload, master, chatgpt_limit)
    document = build_document(system_rows, system_meta, chatgpt_rows, rejected, chatgpt_meta, master_path, source)
    dated, latest = publish(document, out_dir)
    archive = archive_document(document, Path(args.history_dir), source_path=latest)
    print(json.dumps(document["counts"], ensure_ascii=False))
    print(f"status={document['status']} as_of={document['asOfDate']} effective={document['effectiveSession']}")
    print(f"output={dated}")
    print(f"latest={latest}")
    print(f"history={archive['dailyFile']} selections={archive['currentDaySelections']} changed={archive['changed']}")


if __name__ == "__main__":
    main()
