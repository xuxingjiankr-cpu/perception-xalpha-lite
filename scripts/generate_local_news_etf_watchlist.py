"""Build a local, no-model-API ETF news watchlist.

The pipeline reads public Google News RSS feeds, scores the evidence with a
versioned deterministic rule set, joins the score to the previous completed
Eastmoney ETF snapshot, and writes a research-only ten-ETF watchlist.

It does not call OpenAI, another hosted model, a broker, or an account API.
The score is an auditable ranking score, not a probability or trade signal.

Examples:
  py -3.13 scripts/generate_local_news_etf_watchlist.py --dry-run
  py -3.13 scripts/generate_local_news_etf_watchlist.py --run-build
  py -3.13 scripts/generate_local_news_etf_watchlist.py --dry-run --as-of-date 2026-07-03
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "research" / "local_news_watchlist_v1.json"
INBOX = ROOT / "data" / "research" / "local_news_etf_watchlist" / "inbox"
RAW_DIR = ROOT / "data" / "research" / "local_news_etf_watchlist" / "raw"
LOG_DIR = ROOT / "logs" / "local_news_etf_watchlist"
SNAPSHOT_DIR = ROOT / "data" / "market" / "eastmoney" / "full_market" / "snapshots"
BUILD_SCRIPT = ROOT / "scripts" / "build_t0_observation_pool.py"
SH = ZoneInfo("Asia/Shanghai")
SCHEMA_VERSION = "local_news_etf_watchlist_v1"
CODE_RE = re.compile(r"^\d{6}$")
FORBIDDEN_NAME_PARTS = ("货币", "现金管理", "理财")
USER_AGENT = "Mozilla/5.0 (compatible; LocalETFNewsWatchlist/1.0; personal-research)"


@dataclass(frozen=True)
class Article:
    topic_id: str
    topic_label: str
    title: str
    published_at: str
    publisher: str
    publisher_url: str
    source_url: str
    sentiment: int
    source_quality: float
    freshness: float
    evidence_weight: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    safe = (
        config.get("schemaVersion") == "local_news_watchlist_config_v1"
        and config.get("recordOnly") is True
        and config.get("tradeGateEnabled") is False
        and config.get("positionSizingEnabled") is False
        and config.get("paperTradingOnly") is True
        and config.get("diagnosticOnly") is True
    )
    if not safe:
        raise ValueError("local-news config must remain record-only and unable to gate or size trades")
    weights = config.get("scoreWeights", {})
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("local-news scoreWeights must sum to 1")
    if int(config.get("selectionCount", 0)) != 10:
        raise ValueError("local-news v1 preregisters exactly ten selections")
    return config


def market_day_context(day: date) -> dict[str, Any]:
    from generate_chatgpt_etf_watchlist import market_day_context as shared_context

    return shared_context(day)


def load_non_money_master() -> tuple[dict[tuple[str, str], dict[str, str]], Path]:
    from generate_chatgpt_etf_watchlist import load_non_money_master as shared_master

    return shared_master()


def latest_completed_snapshot(previous_session: str, snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    candidates: list[Path] = []
    if snapshot_dir.exists():
        for directory in snapshot_dir.iterdir():
            if directory.is_dir() and directory.name <= previous_session:
                candidates.extend(directory.glob("*_etf.csv.gz"))
    if not candidates:
        raise FileNotFoundError(f"no completed ETF snapshot on or before {previous_session}")
    return sorted(candidates, key=lambda path: (path.parent.name, path.name))[-1]


def load_market_rows(
    snapshot: Path,
    master: dict[tuple[str, str], dict[str, str]],
    minimum_amount: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(snapshot, "rt", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            code = str(raw.get("stockCode", "")).strip()
            exchange = "SH" if str(raw.get("market", "")).strip() == "1" else "SZ"
            canonical = master.get((code, exchange))
            if canonical is None:
                continue
            name = canonical["name"]
            if any(part in name for part in FORBIDDEN_NAME_PARTS):
                continue
            try:
                amount = float(raw.get("amount") or 0.0)
                change_pct = float(raw.get("change_pct") or 0.0)
                price = float(raw.get("currentPrice") or 0.0)
            except (TypeError, ValueError):
                continue
            if amount < minimum_amount or price <= 0:
                continue
            rows.append({
                "stockCode": code,
                "exchange": exchange,
                "name": name,
                "amount": amount,
                "changePct": change_pct,
                "currentPrice": price,
            })
    return rows


def rss_url(query: str) -> str:
    encoded = urllib.parse.quote(query + " when:2d")
    return f"https://news.google.com/rss/search?q={encoded}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"


def normalize_title(value: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    return re.sub(r"\s+", " ", text).strip()


def title_key(value: str) -> str:
    text = normalize_title(value).lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def sentiment_score(title: str, positive_terms: list[str], negative_terms: list[str]) -> int:
    positive = sum(1 for term in positive_terms if term and term in title)
    negative = sum(1 for term in negative_terms if term and term in title)
    return int(clamp(positive - negative, -3, 3))


def publisher_quality(publisher: str, quality_config: dict[str, Any]) -> float:
    if any(name.lower() in publisher.lower() for name in quality_config.get("officialNames", [])):
        return float(quality_config.get("official", 1.0))
    if any(name.lower() in publisher.lower() for name in quality_config.get("majorNames", [])):
        return float(quality_config.get("major", 0.82))
    return float(quality_config.get("other", 0.58))


def parse_rss(
    xml_bytes: bytes,
    topic: dict[str, Any],
    *,
    news_start: datetime,
    cutoff: datetime,
    config: dict[str, Any],
) -> list[Article]:
    root = ET.fromstring(xml_bytes)
    positive_terms = [str(value) for value in config.get("positiveTerms", [])]
    negative_terms = [str(value) for value in config.get("negativeTerms", [])]
    excluded_fragments = [str(value) for value in config.get("excludedTitleFragments", [])]
    quality_config = config.get("sourceQuality", {})
    maximum = int(config.get("maximumArticlesPerTopic", 12))
    window_seconds = max(1.0, (cutoff - news_start).total_seconds())
    result: list[Article] = []
    seen: set[str] = set()
    for item in root.findall("./channel/item"):
        title = normalize_title(item.findtext("title") or "")
        if any(fragment and fragment in title for fragment in excluded_fragments):
            continue
        link = str(item.findtext("link") or "").strip()
        source = item.find("source")
        publisher = normalize_title(source.text if source is not None and source.text else "")
        publisher_url = str(source.attrib.get("url", "") if source is not None else "").strip()
        try:
            published = parsedate_to_datetime(str(item.findtext("pubDate") or "")).astimezone(SH)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (news_start <= published <= cutoff):
            continue
        key = title_key(title)
        if not key or key in seen or not link.startswith(("http://", "https://")):
            continue
        seen.add(key)
        age_fraction = clamp((cutoff - published).total_seconds() / window_seconds, 0.0, 1.0)
        freshness = math.exp(-1.6 * age_fraction)
        quality = publisher_quality(publisher, quality_config)
        sentiment = sentiment_score(title, positive_terms, negative_terms)
        evidence_weight = freshness * quality * (0.5 + 0.5 * min(abs(sentiment), 3) / 3.0)
        result.append(Article(
            topic_id=str(topic["id"]),
            topic_label=str(topic["label"]),
            title=title,
            published_at=published.isoformat(timespec="seconds"),
            publisher=publisher or "unknown",
            publisher_url=publisher_url,
            source_url=link,
            sentiment=sentiment,
            source_quality=round(quality, 6),
            freshness=round(freshness, 6),
            evidence_weight=round(evidence_weight, 6),
        ))
    result.sort(key=lambda row: (row.evidence_weight, row.published_at), reverse=True)
    return result[:maximum]


def fetch_topic(
    topic: dict[str, Any],
    *,
    news_start: datetime,
    cutoff: datetime,
    config: dict[str, Any],
) -> tuple[str, list[Article], str]:
    url = rss_url(str(topic["query"]))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml"})
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("rssTimeoutSeconds", 20))) as response:
            body = response.read(2_000_000)
        return str(topic["id"]), parse_rss(
            body, topic, news_start=news_start, cutoff=cutoff, config=config
        ), ""
    except Exception as exc:
        return str(topic["id"]), [], f"{type(exc).__name__}: {exc}"


def fetch_all_topics(
    *,
    news_start: datetime,
    cutoff: datetime,
    config: dict[str, Any],
) -> tuple[dict[str, list[Article]], dict[str, str]]:
    articles: dict[str, list[Article]] = {}
    errors: dict[str, str] = {}
    topics = list(config.get("topics", []))
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(topics)))) as executor:
        futures = [
            executor.submit(fetch_topic, topic, news_start=news_start, cutoff=cutoff, config=config)
            for topic in topics
        ]
        for future in as_completed(futures):
            topic_id, rows, error = future.result()
            articles[topic_id] = rows
            if error:
                errors[topic_id] = error
    return articles, errors


def assign_topic(name: str, topics: list[dict[str, Any]]) -> dict[str, Any] | None:
    lowered = name.lower()
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for order, topic in enumerate(topics):
        hit_lengths = [
            len(str(keyword))
            for keyword in topic.get("etfKeywords", [])
            if str(keyword).lower() in lowered
        ]
        if hit_lengths:
            matches.append((max(hit_lengths), -order, topic))
    return max(matches, default=(0, 0, None), key=lambda row: (row[0], row[1]))[2]


def topic_metrics(rows: list[Article]) -> dict[str, float]:
    if not rows:
        return {
            "newsDirection": 0.0,
            "newsCoverage": 0.0,
            "sourceQuality": 0.0,
            "netEvidence": 0.0,
            "consistency": 0.0,
        }
    signed = sum(row.evidence_weight * (1 if row.sentiment > 0 else -1 if row.sentiment < 0 else 0)
                 for row in rows)
    positive = sum(row.evidence_weight for row in rows if row.sentiment > 0)
    negative = sum(row.evidence_weight for row in rows if row.sentiment < 0)
    directional = positive + negative
    return {
        "newsDirection": clamp(50.0 + 50.0 * math.tanh(signed / 2.5), 0.0, 100.0),
        "newsCoverage": 100.0 * (1.0 - math.exp(-len(rows) / 3.0)),
        "sourceQuality": 100.0 * sum(row.source_quality for row in rows) / len(rows),
        "netEvidence": signed,
        "consistency": abs(positive - negative) / directional if directional else 0.0,
    }


def score_candidates(
    market_rows: list[dict[str, Any]],
    articles_by_topic: dict[str, list[Article]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    topics = list(config.get("topics", []))
    topic_by_id = {str(topic["id"]): topic for topic in topics}
    weights = {key: float(value) for key, value in config["scoreWeights"].items()}
    minimum_amount = float(config["minimumPreviousDayAmount"])
    candidates: list[dict[str, Any]] = []
    for market in market_rows:
        topic = assign_topic(str(market["name"]), topics)
        if topic is None:
            continue
        topic_id = str(topic["id"])
        news_rows = articles_by_topic.get(topic_id, [])
        if not news_rows:
            continue
        metrics = topic_metrics(news_rows)
        amount = float(market["amount"])
        liquidity = 100.0 * clamp(
            (math.log10(amount) - math.log10(minimum_amount))
            / (math.log10(5_000_000_000.0) - math.log10(minimum_amount)),
            0.0,
            1.0,
        )
        change = float(market["changePct"])
        market_confirmation = clamp(50.0 + 12.0 * clamp(change, -3.0, 3.0), 0.0, 100.0)
        crowding_penalty = clamp(max(0.0, abs(change) - 3.0) * 4.0, 0.0, 20.0)
        breakdown = {
            "newsDirection": metrics["newsDirection"],
            "newsCoverage": metrics["newsCoverage"],
            "sourceQuality": metrics["sourceQuality"],
            "liquidity": liquidity,
            "marketConfirmation": market_confirmation,
            "crowdingPenalty": crowding_penalty,
        }
        total = (
            weights["newsDirection"] * breakdown["newsDirection"]
            + weights["newsCoverage"] * breakdown["newsCoverage"]
            + weights["sourceQuality"] * breakdown["sourceQuality"]
            + weights["liquidity"] * breakdown["liquidity"]
            + weights["marketConfirmation"] * breakdown["marketConfirmation"]
            - crowding_penalty
        )
        confidence = clamp(
            0.35 * breakdown["newsCoverage"] / 100.0
            + 0.45 * breakdown["sourceQuality"] / 100.0
            + 0.20 * metrics["consistency"],
            0.0,
            1.0,
        )
        source_rows = sorted(news_rows, key=lambda row: row.evidence_weight, reverse=True)[:3]
        drivers = [
            f"[{'正向' if row.sentiment > 0 else '负向' if row.sentiment < 0 else '中性'}]"
            f"{row.title}（{row.publisher}，{row.published_at}）"
            for row in source_rows
        ]
        risks = ["规则评分尚未通过前向收益验证，仅用于观察池排序，不是买入指令。"]
        negative = next((row for row in source_rows if row.sentiment < 0), None)
        if negative:
            risks.append(f"存在反向新闻：{negative.title}")
        if abs(change) > 3.0:
            risks.append(f"前一交易日涨跌幅为{change:.2f}%，存在拥挤或反转风险。")
        else:
            risks.append("新闻可能已经被前一交易日价格部分反映。")
        direction = (
            "positive" if metrics["netEvidence"] > 0.15
            else "negative" if metrics["netEvidence"] < -0.15
            else "mixed_or_neutral"
        )
        candidates.append({
            "stockCode": market["stockCode"],
            "exchange": market["exchange"],
            "name": market["name"],
            "topic": {"id": topic_id, "label": topic_by_id[topic_id]["label"]},
            "localScore": round(total, 4),
            "scoreBreakdown": {key: round(value, 4) for key, value in breakdown.items()},
            "evidenceConfidence": round(confidence, 6),
            "newsDirection": direction,
            "previousSessionMarket": {
                "amount": round(amount, 3),
                "changePct": round(change, 4),
                "currentPrice": round(float(market["currentPrice"]), 6),
            },
            "reason": (
                f"本地无模型API规则评分，总分{total:.2f}；"
                f"新闻方向{metrics['newsDirection']:.2f}、覆盖{metrics['newsCoverage']:.2f}、"
                f"来源质量{metrics['sourceQuality']:.2f}、流动性{liquidity:.2f}、"
                f"市场确认{market_confirmation:.2f}、拥挤扣分{crowding_penalty:.2f}。"
                "该分数是证据排序，不是上涨概率。"
            ),
            "newsDrivers": drivers,
            "risks": risks,
            "sourceUrls": [row.source_url for row in source_rows],
            "sourceItems": [asdict(row) for row in source_rows],
        })
    candidates.sort(
        key=lambda row: (
            float(row["localScore"]),
            float(row["evidenceConfidence"]),
            float(row["previousSessionMarket"]["amount"]),
        ),
        reverse=True,
    )
    return candidates


def select_diversified(candidates: list[dict[str, Any]], count: int, max_per_topic: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_codes: set[str] = set()
    topic_counts: dict[str, int] = {}
    for row in candidates:
        topic_id = str(row["topic"]["id"])
        if topic_counts.get(topic_id, 0) >= max_per_topic:
            continue
        if row["stockCode"] in selected_codes:
            continue
        selected.append(row)
        selected_codes.add(row["stockCode"])
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        if len(selected) == count:
            return selected
    for row in candidates:
        if row["stockCode"] in selected_codes:
            continue
        selected.append(row)
        selected_codes.add(row["stockCode"])
        if len(selected) == count:
            return selected
    return selected


def validate_payload(
    payload: Any,
    *,
    master: dict[tuple[str, str], dict[str, str]],
    expected_count: int,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict) or payload.get("schemaVersion") != SCHEMA_VERSION:
        return ["invalid_schema_version"]
    if payload.get("paperTradingOnly") is not True or payload.get("diagnosticOnly") is not True:
        errors.append("missing_research_safety_markers")
    if payload.get("tradeGateEnabled") is not False:
        errors.append("trade_gate_must_be_false")
    rows = payload.get("etfs")
    if not isinstance(rows, list):
        return errors + ["etfs_not_array"]
    if len(rows) != expected_count:
        errors.append(f"invalid_etf_count:{len(rows)}!=expected:{expected_count}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"etf_{index}:not_object")
            continue
        code = str(row.get("stockCode", ""))
        exchange = str(row.get("exchange", ""))
        if not CODE_RE.fullmatch(code):
            errors.append(f"etf_{index}:invalid_code")
        if exchange not in {"SH", "SZ"}:
            errors.append(f"etf_{index}:invalid_exchange")
        if code in seen:
            errors.append(f"etf_{index}:duplicate_code")
        seen.add(code)
        if (code, exchange) not in master:
            errors.append(f"etf_{index}:not_in_non_money_master")
        if not str(row.get("name", "")).strip() or any(
                part in str(row.get("name", "")) for part in FORBIDDEN_NAME_PARTS):
            errors.append(f"etf_{index}:invalid_name")
        for field in ("reason", "newsDrivers", "risks", "sourceUrls", "sourceItems", "scoreBreakdown"):
            value = row.get(field)
            if field == "reason" and not str(value or "").strip():
                errors.append(f"etf_{index}:empty_reason")
            elif field in {"newsDrivers", "risks", "sourceUrls", "sourceItems"} and (
                    not isinstance(value, list) or not value):
                errors.append(f"etf_{index}:empty_{field}")
            elif field == "scoreBreakdown" and not isinstance(value, dict):
                errors.append(f"etf_{index}:invalid_score_breakdown")
        if not isinstance(row.get("localScore"), (int, float)):
            errors.append(f"etf_{index}:invalid_local_score")
        if not isinstance(row.get("evidenceConfidence"), (int, float)):
            errors.append(f"etf_{index}:invalid_evidence_confidence")
        if not all(str(url).startswith(("http://", "https://")) for url in row.get("sourceUrls", [])):
            errors.append(f"etf_{index}:invalid_source_url")
    return errors


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_raw_articles(path: Path, articles: dict[str, list[Article]]) -> None:
    rows = [asdict(row) for topic_rows in articles.values() for row in topic_rows]
    rows.sort(key=lambda row: (row["published_at"], row["topic_id"], row["title"]))
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_bytes(path, content.encode("utf-8"))


def run_build_script(path: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--local-news-file", str(path)],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    return {"stdout": process.stdout, "stderr": process.stderr, "exit_code": process.returncode}


def append_log(day: str, report: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"local_news_etf_watchlist_{day}.log"
    lines = ["=" * 72]
    for key in (
        "start_time", "effectiveDate", "is_trading_day", "generator_version",
        "no_model_api", "news_start", "data_cutoff", "snapshot", "raw_article_count",
        "topic_errors", "eligible_etf_count", "candidate_count", "selected_etf_count",
        "json_schema_valid", "saved_file", "raw_file", "dry_run", "build_stdout",
        "build_stderr", "exit_code", "error", "end_time",
    ):
        value = report.get(key, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}={value}")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def run(args: argparse.Namespace, *, now: datetime | None = None) -> tuple[int, dict[str, Any]]:
    actual_now = (now or datetime.now(tz=SH)).astimezone(SH)
    requested_day = date.fromisoformat(args.as_of_date) if args.as_of_date else actual_now.date()
    if args.as_of_date:
        cutoff = datetime.combine(requested_day, time(8, 30), tzinfo=SH)
    else:
        cutoff = actual_now
    generated_at = cutoff.isoformat(timespec="seconds")
    context = market_day_context(requested_day)
    config = load_config(Path(args.config))
    config_hash = canonical_sha256(config)
    target = INBOX / f"{context['effective_date']}.json"
    raw_target = RAW_DIR / requested_day.isoformat() / "articles.jsonl"
    report: dict[str, Any] = {
        "start_time": generated_at,
        "effectiveDate": context["effective_date"],
        "is_trading_day": context["is_trading_day"],
        "generator_version": config["version"],
        "no_model_api": True,
        "news_start": context["news_start"].isoformat(),
        "data_cutoff": cutoff.isoformat(),
        "snapshot": "",
        "raw_article_count": 0,
        "topic_errors": {},
        "eligible_etf_count": 0,
        "candidate_count": 0,
        "selected_etf_count": 0,
        "json_schema_valid": False,
        "saved_file": "",
        "raw_file": "",
        "dry_run": bool(args.dry_run),
        "build_stdout": "",
        "build_stderr": "",
        "exit_code": 1,
        "error": "",
    }
    exit_code = 1
    try:
        master, master_path = load_non_money_master()
        selected: list[dict[str, Any]] = []
        articles: dict[str, list[Article]] = {}
        if context["is_trading_day"]:
            snapshot = latest_completed_snapshot(str(context["previous_session"]))
            report["snapshot"] = str(snapshot)
            market_rows = load_market_rows(snapshot, master, float(config["minimumPreviousDayAmount"]))
            report["eligible_etf_count"] = len(market_rows)
            articles, topic_errors = fetch_all_topics(
                news_start=context["news_start"],
                cutoff=cutoff,
                config=config,
            )
            report["topic_errors"] = topic_errors
            report["raw_article_count"] = sum(len(rows) for rows in articles.values())
            candidates = score_candidates(market_rows, articles, config)
            report["candidate_count"] = len(candidates)
            selected = select_diversified(
                candidates,
                int(config["selectionCount"]),
                int(config["maxPerTopic"]),
            )
            if len(selected) != int(config["selectionCount"]):
                raise RuntimeError(
                    f"fail closed: only {len(selected)} sourced candidates for "
                    f"{config['selectionCount']} required selections"
                )
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "asOfDate": context["as_of_date"],
            "effectiveDate": context["effective_date"],
            "generatedAt": generated_at,
            "paperTradingOnly": True,
            "diagnosticOnly": True,
            "tradeGateEnabled": False,
            "liveReady": False,
            "formalStrategyAllowed": False,
            "generator": {
                "name": "local_public_rss_deterministic_ranker",
                "version": config["version"],
                "noExternalModelApi": True,
                "feed": "Google News public RSS",
                "configSha256": config_hash,
                "dataCutoff": cutoff.isoformat(),
                "newsStart": context["news_start"].isoformat(),
                "previousSessionSnapshot": report["snapshot"],
                "masterFile": str(master_path),
            },
            "etfs": selected,
        }
        errors = validate_payload(
            payload,
            master=master,
            expected_count=int(config["selectionCount"]) if context["is_trading_day"] else 0,
        )
        if errors:
            raise RuntimeError("local schema validation failed: " + "; ".join(errors))
        report["json_schema_valid"] = True
        report["selected_etf_count"] = len(selected)
        if args.dry_run:
            report["saved_file"] = f"DRY_RUN:{target}"
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            print(f"would_save={target}")
        else:
            if args.as_of_date and requested_day != actual_now.date():
                raise RuntimeError("historical --as-of-date writes are forbidden; use --dry-run")
            atomic_write_json(target, payload)
            reloaded = json.loads(target.read_text(encoding="utf-8"))
            reread_errors = validate_payload(
                reloaded,
                master=master,
                expected_count=int(config["selectionCount"]) if context["is_trading_day"] else 0,
            )
            if reread_errors:
                raise RuntimeError("saved JSON failed re-read validation: " + "; ".join(reread_errors))
            report["saved_file"] = str(target)
            if articles:
                write_raw_articles(raw_target, articles)
                report["raw_file"] = str(raw_target)
        if args.run_build and not args.dry_run:
            build = run_build_script(target)
            report["build_stdout"] = build["stdout"]
            report["build_stderr"] = build["stderr"]
            if build["exit_code"] != 0:
                raise RuntimeError(f"build_t0_observation_pool.py exited {build['exit_code']}")
        exit_code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        report["exit_code"] = exit_code
        report["end_time"] = datetime.now(tz=SH).isoformat(timespec="seconds")
        log_path = append_log(requested_day.isoformat(), report)
        report["log_file"] = str(log_path)
        print(json.dumps({
            "workspace_accessible": ROOT.exists(),
            "rss_no_api": True,
            "json_schema_valid": report["json_schema_valid"],
            "selected_etf_count": report["selected_etf_count"],
            "saved_file": report["saved_file"],
            "build_stdout": report["build_stdout"],
            "build_stderr": report["build_stderr"],
            "exit_code": exit_code,
            "log_file": str(log_path),
        }, ensure_ascii=False))
    return exit_code, report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate a local no-model-API ETF news watchlist")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--dry-run", action="store_true", help="search and validate without writing")
    parser.add_argument("--run-build", action="store_true", help="rebuild the observation pool after writing")
    parser.add_argument("--as-of-date", help="YYYY-MM-DD; historical dates are dry-run only")
    args = parser.parse_args()
    code, _ = run(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
