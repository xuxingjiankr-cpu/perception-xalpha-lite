"""Generate the daily ChatGPT ETF news watchlist and rebuild the observation pool.

This is an unattended, research-only input pipeline.  It never calls a broker,
submits an order, or changes strategy/overlay configuration.  On an XSHG
session it asks the OpenAI Responses API (with web search enabled) for exactly
10 non-money A-share ETFs, validates every item against the latest local
Eastmoney ETF master, atomically writes the v1 JSON, then runs the existing
observation-pool builder.

Required environment:
  OPENAI_API_KEY

Optional environment:
  OPENAI_MODEL                         default: gpt-5-mini
  OPENAI_BASE_URL                     default: https://api.openai.com/v1
  CHATGPT_ETF_VERIFY_URLS             default: 1
  CHATGPT_ETF_MAX_ATTEMPTS            default: 3

Examples:
  py -3.13 scripts/generate_chatgpt_etf_watchlist.py --dry-run
  py -3.13 scripts/generate_chatgpt_etf_watchlist.py --run-build
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "research" / "chatgpt_etf_watchlist" / "inbox"
LOG_DIR = ROOT / "logs" / "chatgpt_etf_watchlist"
UNIVERSE_DIR = ROOT / "data" / "market" / "eastmoney" / "universe"
BUILD_SCRIPT = ROOT / "scripts" / "build_t0_observation_pool.py"
SH = ZoneInfo("Asia/Shanghai")
SCHEMA_VERSION = "chatgpt_etf_watchlist_v1"
CODE_RE = re.compile(r"^\d{6}$")
FORBIDDEN_NAME_PARTS = ("货币", "现金管理", "理财")
RESERVED_SOURCE_HOSTS = {"example.com", "example.org", "example.net", "localhost"}


class WatchlistValidationError(RuntimeError):
    def __init__(self, message: str, rejections: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rejections = rejections


ETF_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stockCode", "exchange", "name", "reason", "newsDrivers", "risks", "sourceUrls"],
    "properties": {
        "stockCode": {"type": "string", "pattern": "^[0-9]{6}$"},
        "exchange": {"type": "string", "enum": ["SH", "SZ"]},
        "name": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
        "newsDrivers": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "risks": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "sourceUrls": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 8}},
    },
}


def response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schemaVersion", "asOfDate", "effectiveDate", "generatedAt", "etfs"],
        "properties": {
            "schemaVersion": {"type": "string", "enum": [SCHEMA_VERSION]},
            "asOfDate": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"},
            "effectiveDate": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"},
            "generatedAt": {"type": "string", "minLength": 20},
            "etfs": {"type": "array", "minItems": 10, "maxItems": 10, "items": ETF_ITEM_SCHEMA},
        },
    }


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _exchange_from_market(value: Any) -> str:
    return "SH" if str(value).strip() == "1" else "SZ"


def _forbidden_name(name: str) -> bool:
    return any(part in name for part in FORBIDDEN_NAME_PARTS)


def latest_universe_file(universe_dir: Path = UNIVERSE_DIR) -> Path:
    files = sorted(universe_dir.glob("eastmoney_universe_etf_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no Eastmoney ETF universe under {universe_dir}")
    return files[-1]


def load_non_money_master(universe_dir: Path = UNIVERSE_DIR) -> tuple[dict[tuple[str, str], dict[str, str]], Path]:
    path = latest_universe_file(universe_dir)
    master: dict[tuple[str, str], dict[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid universe JSON at {path}:{line_number}: {exc}") from exc
        code = str(row.get("stockCode", "")).strip()
        exchange = _exchange_from_market(row.get("market"))
        name = str(row.get("name", "")).strip()
        if CODE_RE.fullmatch(code) and name and not _forbidden_name(name):
            master[(code, exchange)] = {"stockCode": code, "exchange": exchange, "name": name}
    if not master:
        raise ValueError(f"empty non-money ETF master: {path}")
    return master, path


def market_day_context(day: date) -> dict[str, Any]:
    try:
        import exchange_calendars as xcals
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("exchange-calendars is required; run: py -3.13 -m pip install exchange-calendars") from exc

    calendar = xcals.get_calendar("XSHG")
    stamp = pd.Timestamp(day.isoformat())
    is_session = bool(calendar.is_session(stamp))
    if is_session:
        effective = stamp
        previous = calendar.previous_session(stamp)
    else:
        effective = calendar.date_to_session(stamp, direction="next")
        previous = calendar.previous_session(effective)
    news_start = datetime.combine(previous.date(), time(15, 0), tzinfo=SH)
    return {
        "is_trading_day": is_session,
        "as_of_date": day.isoformat(),
        "effective_date": effective.date().isoformat(),
        "previous_session": previous.date().isoformat(),
        "news_start": news_start,
    }


def source_url_syntax_ok(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host or "." not in host:
        return False
    if host in RESERVED_SOURCE_HOSTS or any(host.endswith("." + x) for x in RESERVED_SOURCE_HOSTS):
        return False
    return True


def verify_source_url(url: str, client: Any) -> tuple[bool, str]:
    if not source_url_syntax_ok(url):
        return False, "invalid_or_reserved_url"
    try:
        response = client.head(url, follow_redirects=True)
        if response.status_code in {405, 501}:
            response = client.get(url, follow_redirects=True, headers={"Range": "bytes=0-1023"})
        if response.status_code < 500:
            return True, "ok"
        return False, f"source_http_{response.status_code}"
    except Exception as exc:
        return False, f"source_unreachable:{type(exc).__name__}"


def _nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(x, str) and x.strip() for x in value)


def validate_payload(
    payload: Any,
    *,
    master: dict[tuple[str, str], dict[str, str]],
    context: dict[str, Any],
    verify_urls: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return None, [{"reason": "payload_not_object"}]

    required_top = ("schemaVersion", "asOfDate", "effectiveDate", "generatedAt", "etfs")
    for field in required_top:
        if field not in payload:
            errors.append({"reason": "missing_top_level_field", "field": field})
    if errors:
        return None, errors
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        errors.append({"reason": "invalid_schema_version", "value": payload.get("schemaVersion")})
    if payload.get("asOfDate") != context["as_of_date"]:
        errors.append({"reason": "invalid_as_of_date", "value": payload.get("asOfDate")})
    if payload.get("effectiveDate") != context["effective_date"]:
        errors.append({"reason": "invalid_effective_date", "value": payload.get("effectiveDate")})
    try:
        generated = datetime.fromisoformat(str(payload.get("generatedAt")))
        if generated.utcoffset() is None or generated.utcoffset().total_seconds() != 8 * 3600:
            errors.append({"reason": "generated_at_not_plus_08", "value": payload.get("generatedAt")})
    except (TypeError, ValueError):
        errors.append({"reason": "invalid_generated_at", "value": payload.get("generatedAt")})

    rows = payload.get("etfs")
    expected_count = 10 if context["is_trading_day"] else 0
    if not isinstance(rows, list):
        errors.append({"reason": "etfs_not_array"})
        return None, errors
    if len(rows) != expected_count:
        errors.append({"reason": "invalid_etf_count", "expected": expected_count, "actual": len(rows)})

    http_client = None
    if verify_urls and rows:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for source URL verification") from exc
        http_client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0), headers={"User-Agent": "ETFWatchlistValidator/1.0"})

    normalized: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    try:
        for index, raw in enumerate(rows):
            if not isinstance(raw, dict):
                errors.append({"index": index, "reason": "etf_not_object"})
                continue
            code = str(raw.get("stockCode", "")).strip()
            exchange = str(raw.get("exchange", "")).strip().upper()
            name = str(raw.get("name", "")).strip()
            reason = str(raw.get("reason", "")).strip()
            drivers = raw.get("newsDrivers")
            risks = raw.get("risks")
            urls = raw.get("sourceUrls")
            key = (code, exchange)
            item_errors: list[str] = []
            if not CODE_RE.fullmatch(code):
                item_errors.append("invalid_stock_code")
            if exchange not in {"SH", "SZ"}:
                item_errors.append("invalid_exchange")
            if code in seen_codes:
                item_errors.append("duplicate_stock_code")
            if key not in master:
                item_errors.append("not_in_local_non_money_etf_master")
            if not name:
                item_errors.append("empty_name")
            if _forbidden_name(name):
                item_errors.append("forbidden_money_or_cash_management_etf")
            if not reason:
                item_errors.append("empty_reason")
            if not _nonempty_string_list(drivers):
                item_errors.append("empty_news_drivers")
            if not _nonempty_string_list(risks):
                item_errors.append("empty_risks")
            if not _nonempty_string_list(urls):
                item_errors.append("empty_source_urls")
            elif http_client is not None:
                for url in urls:
                    ok, url_reason = verify_source_url(url, http_client)
                    if not ok:
                        item_errors.append(f"invalid_source_url:{url_reason}")
                        break
            elif not all(source_url_syntax_ok(str(url)) for url in urls):
                item_errors.append("invalid_source_url_syntax")
            if item_errors:
                errors.append({"index": index, "stockCode": code, "exchange": exchange,
                               "reason": ",".join(item_errors)})
                continue
            seen_codes.add(code)
            canonical = master[key]
            normalized.append({
                "stockCode": code,
                "exchange": exchange,
                "name": canonical["name"],
                "reason": reason,
                "newsDrivers": [str(x).strip() for x in drivers],
                "risks": [str(x).strip() for x in risks],
                "sourceUrls": [str(x).strip() for x in urls],
            })
    finally:
        if http_client is not None:
            http_client.close()

    if errors:
        return None, errors
    clean = {
        "schemaVersion": SCHEMA_VERSION,
        "asOfDate": context["as_of_date"],
        "effectiveDate": context["effective_date"],
        "generatedAt": str(payload["generatedAt"]),
        "etfs": normalized,
    }
    return clean, []


def build_catalog(master: dict[tuple[str, str], dict[str, str]]) -> str:
    rows = sorted(master.values(), key=lambda x: (x["exchange"], x["stockCode"]))
    return "\n".join(f"{r['stockCode']}|{r['exchange']}|{r['name']}" for r in rows)


def research_prompt(context: dict[str, Any], generated_at: str, catalog: str, prior_errors: list[dict[str, Any]]) -> str:
    repair = ""
    if prior_errors:
        repair = "\n上一次结果未通过本地校验。必须修复这些问题：\n" + json.dumps(prior_errors, ensure_ascii=False)
    return f"""你是A股ETF新闻研究员。当前中国时间为 {generated_at}。
请使用 web search 调研 {context['news_start'].isoformat()} 至当前时间发生的政策、监管公告、产业新闻、
海外市场和商品价格变化，从下面的本地非货币ETF主表中严格选择10只在
{context['effective_date']} 值得观察的ETF。

硬要求：
- 只能从主表选择，代码、SH/SZ交易所和名称必须匹配。
- stockCode不能重复；排除货币、现金管理、理财类ETF。
- 每项必须有具体新闻驱动理由、至少一个newsDrivers、至少一个risks。
- sourceUrls必须是本次实际检索到的真实直接网页URL，禁止example.com、搜索结果页或虚构链接。
- 优先政府、交易所、公司公告和主流财经媒体；新闻时效必须位于指定窗口或仍直接影响今日市场。
- 这只是paper trading观察名单，不是买入指令，不得保证收益。
- 只返回符合给定JSON schema的内容。
{repair}

本地非货币ETF主表（stockCode|exchange|name）：
{catalog}
"""


def _extract_response_text(response: dict[str, Any]) -> str:
    pieces: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                pieces.append(str(content.get("text", "")))
    text = "\n".join(x for x in pieces if x).strip()
    if not text and isinstance(response.get("output_text"), str):
        text = response["output_text"].strip()
    if not text:
        raise RuntimeError("OpenAI response contained no output_text")
    return text


def call_openai(prompt: str, api_key: str, model: str) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required; run: py -3.13 -m pip install httpx") from exc
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base}/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_payload = {
        "model": model,
        "instructions": (
            "Return research-only A-share ETF observations. Use web search, follow the strict JSON schema, "
            "and never present the list as an order or guaranteed recommendation."
        ),
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "chatgpt_etf_watchlist_v1",
                "strict": True,
                "schema": response_json_schema(),
            }
        },
        "max_output_tokens": 7000,
        "store": False,
    }
    last_error = ""
    with httpx.Client(timeout=httpx.Timeout(240.0, connect=20.0)) as client:
        for tool_type in ("web_search", "web_search_preview"):
            payload = {**base_payload, "tools": [{"type": tool_type}]}
            response = client.post(endpoint, headers=headers, json=payload)
            if response.is_success:
                raw = response.json()
                text = _extract_response_text(raw)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"OpenAI output_text was not valid JSON: {exc}") from exc
            body = response.text[:2000]
            last_error = f"OpenAI HTTP {response.status_code}: {body}"
            if response.status_code != 400 or tool_type == "web_search_preview":
                break
    raise RuntimeError(last_error or "OpenAI request failed")


def generate_trading_day_payload(
    context: dict[str, Any],
    master: dict[tuple[str, str], dict[str, str]],
    *,
    api_key: str,
    model: str,
    generated_at: str,
    verify_urls: bool,
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = build_catalog(master)
    all_rejections: list[dict[str, Any]] = []
    prior_errors: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        raw = call_openai(research_prompt(context, generated_at, catalog, prior_errors), api_key, model)
        # Dates are controlled locally; the model selects and researches ETF items only.
        raw["schemaVersion"] = SCHEMA_VERSION
        raw["asOfDate"] = context["as_of_date"]
        raw["effectiveDate"] = context["effective_date"]
        raw["generatedAt"] = generated_at
        clean, errors = validate_payload(raw, master=master, context=context, verify_urls=verify_urls)
        if clean is not None:
            return clean, all_rejections
        tagged = [{**error, "attempt": attempt} for error in errors]
        all_rejections.extend(tagged)
        prior_errors = tagged
    raise WatchlistValidationError(
        f"OpenAI watchlist failed validation after {max_attempts} attempts",
        all_rejections,
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_build_script() -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        cwd=str(ROOT),
        env=env,
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
    path = LOG_DIR / f"chatgpt_etf_watchlist_{day}.log"
    lines = ["=" * 72]
    for key in (
        "start_time", "generatedAt", "effectiveDate", "is_trading_day", "openai_model",
        "openai_api_available", "selected_etf_count", "rejected_etf_count",
        "rejected_etf_reasons", "saved_file", "dry_run", "build_stdout", "build_stderr",
        "exit_code", "error", "end_time",
    ):
        value = report.get(key, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}={value}")
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def run(args: argparse.Namespace, *, now: datetime | None = None) -> tuple[int, dict[str, Any]]:
    now = (now or datetime.now(tz=SH)).astimezone(SH)
    generated_at = now.isoformat(timespec="seconds")
    context = market_day_context(now.date())
    INBOX.mkdir(parents=True, exist_ok=True)
    target = INBOX / f"{context['effective_date']}.json"
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    report: dict[str, Any] = {
        "start_time": generated_at,
        "generatedAt": generated_at,
        "effectiveDate": context["effective_date"],
        "is_trading_day": context["is_trading_day"],
        "openai_model": model,
        "openai_api_available": bool(api_key),
        "json_schema_valid": False,
        "selected_etf_count": 0,
        "rejected_etf_count": 0,
        "rejected_etf_reasons": [],
        "saved_file": "",
        "dry_run": bool(args.dry_run),
        "build_stdout": "",
        "build_stderr": "",
        "exit_code": 1,
        "error": "",
    }
    exit_code = 1
    try:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        master, master_path = load_non_money_master()
        if context["is_trading_day"]:
            max_attempts = max(1, int(os.environ.get("CHATGPT_ETF_MAX_ATTEMPTS", "3")))
            payload, rejected = generate_trading_day_payload(
                context,
                master,
                api_key=api_key,
                model=model,
                generated_at=generated_at,
                verify_urls=_bool_env("CHATGPT_ETF_VERIFY_URLS", True),
                max_attempts=max_attempts,
            )
        else:
            payload = {
                "schemaVersion": SCHEMA_VERSION,
                "asOfDate": context["as_of_date"],
                "effectiveDate": context["effective_date"],
                "generatedAt": generated_at,
                "etfs": [],
            }
            rejected = []
        clean, final_errors = validate_payload(payload, master=master, context=context, verify_urls=False)
        if clean is None:
            raise RuntimeError(f"final schema validation failed: {json.dumps(final_errors, ensure_ascii=False)}")
        report["json_schema_valid"] = True
        report["selected_etf_count"] = len(clean["etfs"])
        report["rejected_etf_count"] = len(rejected)
        report["rejected_etf_reasons"] = rejected
        report["master_file"] = str(master_path)

        if args.dry_run:
            report["saved_file"] = f"DRY_RUN:{target}"
            print(json.dumps(clean, ensure_ascii=False, indent=2))
            print(f"would_save={target}")
        else:
            atomic_write_json(target, clean)
            reloaded = json.loads(target.read_text(encoding="utf-8"))
            checked, reread_errors = validate_payload(reloaded, master=master, context=context, verify_urls=False)
            if checked is None:
                raise RuntimeError(f"saved JSON failed re-read validation: {json.dumps(reread_errors, ensure_ascii=False)}")
            report["saved_file"] = str(target)

        should_build = bool(args.run_build or not args.dry_run)
        if should_build:
            build = run_build_script()
            report["build_stdout"] = build["stdout"]
            report["build_stderr"] = build["stderr"]
            if build["exit_code"] != 0:
                raise RuntimeError(f"build_t0_observation_pool.py exited {build['exit_code']}")
        exit_code = 0
    except Exception as exc:
        if isinstance(exc, WatchlistValidationError):
            report["rejected_etf_count"] = len(exc.rejections)
            report["rejected_etf_reasons"] = exc.rejections
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        report["exit_code"] = exit_code
        report["end_time"] = datetime.now(tz=SH).isoformat(timespec="seconds")
        log_path = append_log(context["as_of_date"], report)
        report["log_file"] = str(log_path)
        print(json.dumps({
            "workspace_accessible": ROOT.exists(),
            "inbox_writable": INBOX.exists() and os.access(INBOX, os.W_OK),
            "openai_api_available": report["openai_api_available"],
            "json_schema_valid": report["json_schema_valid"],
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
    parser = argparse.ArgumentParser(description="Generate the daily ChatGPT ETF news watchlist")
    parser.add_argument("--dry-run", action="store_true", help="validate and print, but do not write the inbox file")
    parser.add_argument("--run-build", action="store_true", help="run the observation-pool builder (default outside dry-run)")
    args = parser.parse_args()
    code, _ = run(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
