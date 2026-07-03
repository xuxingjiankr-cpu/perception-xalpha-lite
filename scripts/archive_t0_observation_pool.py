"""Archive the daily ETF observation pool for later point-in-time research.

Research-only data plumbing: no broker imports, account access or order calls.
Each run writes one canonical daily snapshot and one ETF-level JSONL file, then
rebuilds a deduplicated cumulative selection history. Re-running the same input
is idempotent; a corrected same-day pool atomically replaces that day's records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SH = ZoneInfo("Asia/Shanghai")
DEFAULT_INPUT = ROOT / "outputs" / "t0_observation_pool" / "latest_observation_pool.json"
DEFAULT_HISTORY = ROOT / "data" / "research" / "t0_observation_pool_history"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CODE_RE = re.compile(r"^\d{6}$")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_write_if_changed(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return False
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(content)
        temp = Path(handle.name)
    temp.replace(path)
    return True


def _valid_date(value: Any) -> str | None:
    text = str(value or "")
    if not DATE_RE.fullmatch(text):
        return None
    try:
        date.fromisoformat(text)
    except ValueError:
        return None
    return text


def resolve_selection_date(document: dict[str, Any]) -> tuple[str, str]:
    effective = _valid_date(document.get("effectiveSession"))
    if effective:
        return effective, "effectiveSession"
    chatgpt_effective = _valid_date((document.get("chatgptInputMeta") or {}).get("effectiveDate"))
    if chatgpt_effective:
        return chatgpt_effective, "chatgptInputMeta.effectiveDate"
    research_effective = _valid_date((document.get("researchInputMeta") or {}).get("effectiveDate"))
    if research_effective:
        return research_effective, "researchInputMeta.effectiveDate"
    generated = str(document.get("generatedAt") or "")[:10]
    generated_date = _valid_date(generated)
    if generated_date:
        # The builder can run on weekends with the literal label
        # "next_trading_session". Resolve it with the same XSHG calendar used by
        # the ChatGPT watchlist generator instead of creating weekend samples.
        from generate_chatgpt_etf_watchlist import market_day_context

        context = market_day_context(date.fromisoformat(generated_date))
        return str(context["effective_date"]), "generatedAt_next_XSHG_session"
    raise ValueError("observation pool lacks a resolvable effective trading session")


def validate_document(document: dict[str, Any]) -> None:
    if not isinstance(document, dict) or document.get("schemaVersion") != "t0_etf_observation_pool_v1":
        raise ValueError("unsupported observation-pool schema")
    if not isinstance(document.get("combined"), list):
        raise ValueError("observation pool combined must be an array")
    if document.get("paperTradingOnly") is not True or document.get("diagnosticOnly") is not True:
        raise ValueError("observation pool is not marked research/paper only")
    if document.get("tradeGateEnabled") is not False or document.get("liveReady") is not False:
        raise ValueError("refusing to archive a live/trade-gating document")


def _chatgpt_by_key(document: dict[str, Any]) -> dict[tuple[str, str], tuple[int, dict[str, Any]]]:
    result: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for rank, row in enumerate(document.get("chatgpt10") or [], 1):
        if not isinstance(row, dict):
            continue
        key = (str(row.get("stockCode", "")).zfill(6), str(row.get("exchange", "")).upper())
        result[key] = (rank, row)
    return result


def _local_news_by_key(document: dict[str, Any]) -> dict[tuple[str, str], tuple[int, dict[str, Any]]]:
    result: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for rank, row in enumerate(document.get("localNews10") or [], 1):
        if not isinstance(row, dict):
            continue
        key = (str(row.get("stockCode", "")).zfill(6), str(row.get("exchange", "")).upper())
        result[key] = (rank, row)
    return result


def selection_records(
    document: dict[str, Any],
    selection_date: str,
    selection_date_basis: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    chatgpt = _chatgpt_by_key(document)
    local_news = _local_news_by_key(document)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for combined_rank, row in enumerate(document.get("combined") or [], 1):
        if not isinstance(row, dict):
            raise ValueError(f"combined[{combined_rank - 1}] is not an object")
        code = str(row.get("stockCode", "")).zfill(6)
        exchange = str(row.get("exchange", "")).upper()
        key = (code, exchange)
        if not CODE_RE.fullmatch(code) or exchange not in {"SH", "SZ"}:
            raise ValueError(f"invalid ETF composite key: {key}")
        if key in seen:
            raise ValueError(f"duplicate ETF composite key in combined pool: {key}")
        seen.add(key)
        chatgpt_rank, chatgpt_row = chatgpt.get(key, (None, {}))
        local_news_rank, local_news_row = local_news.get(key, (None, {}))
        if isinstance(row.get("localNewsResearch"), dict):
            research = row["localNewsResearch"]
        elif isinstance(row.get("chatgptResearch"), dict):
            research = row["chatgptResearch"]
        elif local_news_row:
            research = local_news_row
        else:
            research = chatgpt_row
        sources = [str(value) for value in (row.get("sources") or []) if str(value)]
        records.append({
            "schemaVersion": "t0_observation_selection_v1",
            "selectionDate": selection_date,
            "selectionDateBasis": selection_date_basis,
            "observationPoolGeneratedAt": document.get("generatedAt"),
            "asOfDate": document.get("asOfDate"),
            "stockCode": code,
            "exchange": exchange,
            "name": row.get("name") or code,
            "sources": sources,
            "combinedRank": combined_rank,
            "systemRank": row.get("systemRank"),
            "chatgptRank": chatgpt_rank,
            "localNewsRank": local_news_rank,
            "researchRank": local_news_rank if local_news_rank is not None else chatgpt_rank,
            "rankScore": row.get("rank_score"),
            "changePct": row.get("change_pct"),
            "conviction": row.get("conviction"),
            "amount": row.get("amount"),
            "sourceNews": {
                "reason": research.get("reason") if isinstance(research, dict) else None,
                "newsDrivers": research.get("newsDrivers", []) if isinstance(research, dict) else [],
                "risks": research.get("risks", []) if isinstance(research, dict) else [],
                "sourceUrls": research.get("sourceUrls", []) if isinstance(research, dict) else [],
                "localScore": research.get("localScore") if isinstance(research, dict) else None,
                "scoreBreakdown": research.get("scoreBreakdown", {}) if isinstance(research, dict) else {},
                "evidenceConfidence": research.get("evidenceConfidence") if isinstance(research, dict) else None,
                "newsDirection": research.get("newsDirection") if isinstance(research, dict) else None,
                "sourceItems": research.get("sourceItems", []) if isinstance(research, dict) else [],
            },
            "sourceDocumentSha256": source_sha256,
            "paperTradingOnly": True,
            "diagnosticOnly": True,
        })
    return records


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) for row in rows)


def generated_at(document: dict[str, Any]) -> datetime:
    text = str(document.get("generatedAt") or "").strip().replace("Z", "+00:00")
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("observation pool generatedAt must be ISO-8601") from exc
    if value.tzinfo is None:
        raise ValueError("observation pool generatedAt must include timezone")
    return value.astimezone(SH)


def is_pre_session(document: dict[str, Any], selection_date: str) -> bool:
    session_open = datetime.combine(date.fromisoformat(selection_date), time(9, 30), tzinfo=SH)
    return generated_at(document) < session_open


def rebuild_cumulative(history_dir: Path) -> tuple[Path, int, int, bool]:
    rows: list[dict[str, Any]] = []
    dates: set[str] = set()
    for path in sorted((history_dir / "selections").glob("selections_*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid selection record at {path}:{number}")
            rows.append(row)
            dates.add(str(row.get("selectionDate")))
    rows.sort(key=lambda row: (str(row.get("selectionDate")), int(row.get("combinedRank") or 0),
                               str(row.get("exchange")), str(row.get("stockCode"))))
    cumulative = history_dir / "selection_history.jsonl"
    changed = atomic_write_if_changed(cumulative, _jsonl_bytes(rows))
    return cumulative, len(dates), len(rows), changed


def rebuild_manifest(history_dir: Path) -> tuple[Path, bool]:
    days: list[dict[str, Any]] = []
    for path in sorted((history_dir / "daily").glob("observation_pool_*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        records_path = history_dir / "selections" / f"selections_{wrapper['selectionDate']}.jsonl"
        count = sum(1 for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip())
        days.append({
            "selectionDate": wrapper["selectionDate"],
            "selectionDateBasis": wrapper["selectionDateBasis"],
            "sourceDocumentSha256": wrapper["sourceDocumentSha256"],
            "selectionCount": count,
            "dailyFile": str(path.relative_to(history_dir)),
            "selectionsFile": str(records_path.relative_to(history_dir)),
        })
    manifest = {
        "schemaVersion": "t0_observation_history_manifest_v1",
        "paperTradingOnly": True,
        "diagnosticOnly": True,
        "days": days,
        "dayCount": len(days),
        "selectionCount": sum(int(row["selectionCount"]) for row in days),
    }
    path = history_dir / "manifest.json"
    return path, atomic_write_if_changed(path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def archive_document(
    document: dict[str, Any],
    history_dir: Path = DEFAULT_HISTORY,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    validate_document(document)
    selection_date, date_basis = resolve_selection_date(document)
    digest = content_hash(document)
    daily_path = history_dir / "daily" / f"observation_pool_{selection_date}.json"
    selections_path = history_dir / "selections" / f"selections_{selection_date}.jsonl"
    existing = json.loads(daily_path.read_text(encoding="utf-8")) if daily_path.exists() else None
    same = isinstance(existing, dict) and existing.get("sourceDocumentSha256") == digest
    pre_session = is_pre_session(document, selection_date)
    canonical_allowed = existing is None or same or pre_session
    archived_at = existing.get("archivedAt") if same else datetime.now(tz=SH).isoformat(timespec="seconds")
    wrapper = {
        "schemaVersion": "t0_observation_pool_archive_v1",
        "selectionDate": selection_date,
        "selectionDateBasis": date_basis,
        "archivedAt": archived_at,
        "sourcePath": str(source_path.resolve()) if source_path else None,
        "sourceDocumentSha256": digest,
        "paperTradingOnly": True,
        "diagnosticOnly": True,
        "observationPool": document,
    }
    records = selection_records(document, selection_date, date_basis, digest)
    revision_stamp = generated_at(document).strftime("%Y%m%dT%H%M%S%z")
    revision_path = history_dir / "revisions" / selection_date / f"observation_pool_{revision_stamp}_{digest[:12]}.json"
    revision_changed = atomic_write_if_changed(
        revision_path, json.dumps(wrapper, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    daily_changed = False
    selections_changed = False
    if canonical_allowed:
        daily_changed = atomic_write_if_changed(
            daily_path, json.dumps(wrapper, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        )
        selections_changed = atomic_write_if_changed(selections_path, _jsonl_bytes(records))
    cumulative, day_count, selection_count, cumulative_changed = rebuild_cumulative(history_dir)
    manifest, manifest_changed = rebuild_manifest(history_dir)
    return {
        "selectionDate": selection_date,
        "selectionDateBasis": date_basis,
        "dailyFile": str(daily_path),
        "selectionsFile": str(selections_path),
        "revisionFile": str(revision_path),
        "cumulativeFile": str(cumulative),
        "manifestFile": str(manifest),
        "dayCount": day_count,
        "selectionCount": selection_count,
        "currentDaySelections": len(records),
        "changed": bool(revision_changed or daily_changed or selections_changed or cumulative_changed or manifest_changed),
        "sameSourceDocument": same,
        "preSessionDocument": pre_session,
        "canonicalUpdated": bool(daily_changed or selections_changed),
        "lateRevisionOnly": bool(existing is not None and not same and not pre_session),
    }


def archive_file(input_path: Path = DEFAULT_INPUT, history_dir: Path = DEFAULT_HISTORY) -> dict[str, Any]:
    document = json.loads(input_path.read_text(encoding="utf-8"))
    return archive_document(document, history_dir, source_path=input_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY))
    args = parser.parse_args()
    result = archive_file(Path(args.input), Path(args.history_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
