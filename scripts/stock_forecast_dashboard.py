#!/usr/bin/env python3
"""Publish and serve a stable, read-only A-share forecast dashboard.

The dashboard consumes a versioned presentation contract instead of importing a model.
Research models may evolve; the browser API remains ``stock_forecast_dashboard_v1``.
This module has no broker, order, position, overlay or production-decision integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import sys
import tempfile
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover - compatibility fallback for old workspaces
    Style = None
    lazy_pinyin = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "stock_forecast_dashboard"
DEFAULT_ASSETS = ROOT / "dashboard" / "stock_forecast"
SCHEMA_VERSION = "stock_forecast_dashboard_v1"
TAIL_DEFINITION = "next executable gross return <= -3%"
PINYIN_BOUNDARIES = (
    1601,
    1637,
    1833,
    2078,
    2274,
    2302,
    2433,
    2594,
    2787,
    3106,
    3212,
    3472,
    3635,
    3722,
    3730,
    3858,
    4027,
    4086,
    4390,
    4558,
    4684,
    4925,
    5249,
)
PINYIN_INITIALS = "ABCDEFGHJKLMNOPQRSTWXYZ"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def name_initials(name: str) -> str:
    """Return deterministic GB2312 pinyin initials without a network dependency."""
    if lazy_pinyin is not None and Style is not None:
        return "".join(
            lazy_pinyin(
                str(name).strip(),
                style=Style.FIRST_LETTER,
                errors=lambda item: [
                    character.upper()
                    for character in item
                    if character.isascii() and character.isalnum()
                ],
            )
        ).upper()
    result: list[str] = []
    for character in str(name).strip():
        if character.isascii() and character.isalnum():
            result.append(character.upper())
            continue
        try:
            encoded = character.encode("gb2312")
        except UnicodeEncodeError:
            continue
        if len(encoded) != 2:
            continue
        code = (encoded[0] - 0xA0) * 100 + (encoded[1] - 0xA0)
        for index in range(len(PINYIN_BOUNDARIES) - 1, -1, -1):
            if code >= PINYIN_BOUNDARIES[index]:
                result.append(PINYIN_INITIALS[index])
                break
    return "".join(result)


def _security_row(row: dict[str, Any]) -> dict[str, Any]:
    security_id = str(row.get("securityId") or "").upper()
    match = re.fullmatch(r"(SH|SZ)\.(\d{6})", security_id)
    if match is None:
        raise ValueError(f"invalid securityId: {security_id}")
    exchange, stock_code = match.groups()
    probability_up = _as_float(row.get("probabilityUp"))
    probability_tail = _as_float(row.get("probabilitySevereLoss"))
    expected = _as_float(row.get("expectedGrossReturn"))
    factor_count = int(row.get("factorCount") or 12)
    imputed_factor_count = int(row.get("imputedFactorCount") or 0)
    for name, value in (
        ("probabilityUp", probability_up),
        ("probabilityTailLoss", probability_tail),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be inside [0,1]")
    name = str(row.get("name") or "").strip()
    return {
        "rank": int(row["rank"]),
        "securityId": security_id,
        "stockCode": stock_code,
        "exchange": exchange,
        "name": name,
        "nameInitials": name_initials(name),
        "factorScore": _as_float(row.get("adaptiveFactorScore")),
        "expectedGrossReturn": expected,
        "probabilityUp": probability_up,
        "probabilityTailLoss": probability_tail,
        "estimateSource": str(row.get("estimateSource") or "unknown"),
        "factorCount": factor_count,
        "imputedFactorCount": imputed_factor_count,
        "factorCompletion": str(row.get("factorCompletion") or "none"),
        "completeTwelveFactorEstimate": factor_count == 12
        and str(row.get("estimateSource") or "").startswith(
            "twelve_rank_multivariate_calibrated"
        ),
        "status": "diagnostic_only_not_an_order",
    }


def build_contract(
    result: dict[str, Any],
    forecast_rows: list[dict[str, Any]],
    source_result_path: Path | None = None,
) -> dict[str, Any]:
    securities = sorted(
        (_security_row(row) for row in forecast_rows), key=lambda row: row["rank"]
    )
    if len(securities) < 10:
        raise ValueError("dashboard requires at least ten forecast rows")
    if len({row["securityId"] for row in securities}) != len(securities):
        raise ValueError("dashboard securities must be unique")
    if [row["rank"] for row in securities] != list(range(1, len(securities) + 1)):
        raise ValueError("dashboard ranks must be contiguous and one-based")
    top10 = securities[:10]
    if [row["rank"] for row in top10] != list(range(1, 11)):
        raise ValueError("dashboard Top10 ranks changed")
    diagnostics = result.get("periodDiagnostics", {})

    expected_values = [
        row["expectedGrossReturn"]
        for row in securities
        if row["expectedGrossReturn"] is not None
    ]
    probability_up_values = [
        row["probabilityUp"] for row in securities if row["probabilityUp"] is not None
    ]
    probability_tail_values = [
        row["probabilityTailLoss"]
        for row in securities
        if row["probabilityTailLoss"] is not None
    ]
    distribution = dict(result.get("latestForecastDistribution", {}))

    def add_distribution(prefix: str, values: list[float]) -> None:
        if not values:
            return
        distribution.setdefault(f"{prefix}Minimum", min(values))
        distribution.setdefault(f"{prefix}Maximum", max(values))
        distribution.setdefault(f"{prefix}Spread", max(values) - min(values))

    add_distribution("expectedReturn", expected_values)
    add_distribution("probabilityUp", probability_up_values)
    add_distribution("probabilityTailLoss", probability_tail_values)
    top10_expected = [
        row["expectedGrossReturn"]
        for row in top10
        if row["expectedGrossReturn"] is not None
    ]
    top10_probability_up = [
        row["probabilityUp"] for row in top10 if row["probabilityUp"] is not None
    ]
    top10_probability_tail = [
        row["probabilityTailLoss"]
        for row in top10
        if row["probabilityTailLoss"] is not None
    ]
    if top10_expected:
        distribution["top10ExpectedReturnSpread"] = max(top10_expected) - min(
            top10_expected
        )
    if top10_probability_up:
        distribution["top10ProbabilityUpSpread"] = max(top10_probability_up) - min(
            top10_probability_up
        )
    if top10_probability_tail:
        distribution["top10ProbabilityTailLossSpread"] = max(
            top10_probability_tail
        ) - min(top10_probability_tail)

    def metric(period: str, head: str, field: str) -> float | None:
        return _as_float(
            diagnostics.get(period, {})
            .get("probability", {})
            .get(head, {})
            .get(field)
        )

    source_sha = (
        hashlib.sha256(source_result_path.read_bytes()).hexdigest()
        if source_result_path is not None and source_result_path.exists()
        else None
    )
    snapshot = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_read_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "signalDate": str(result["signalDate"]),
        "intendedTradingSession": str(result["intendedTradingSession"]),
        "forecastHorizon": "next buyable open to following sellable open, maximum exit delay 5 sessions",
        "tailLossDefinition": TAIL_DEFINITION,
        "roundTripCostAssumption": 0.003,
        "forecastReliability": {
            "status": str(
                result.get("forecastReliability", {}).get(
                    "status", "unknown_not_validated"
                )
            ),
            "validationUpAuc": metric("validation", "grossUp", "auc"),
            "shadowUpAuc": metric("shadow", "grossUp", "auc"),
            "validationTailAuc": metric("validation", "severeLoss", "auc"),
            "shadowTailAuc": metric("shadow", "severeLoss", "auc"),
            "eligibleForTrading": False,
        },
        "forecastDistribution": json_safe(distribution),
        "source": {
            "runId": str(result.get("runId") or ""),
            "modelCodeVersion": str(result.get("codeVersion") or ""),
            "resultPath": str(source_result_path.resolve())
            if source_result_path is not None
            else None,
            "resultSha256": source_sha,
        },
        "top10": top10,
        "securities": securities,
        "securityCount": len(securities),
        "orders": [],
        "automaticTradingChanges": [],
    }
    validate_contract(snapshot)
    return snapshot


def validate_contract(snapshot: dict[str, Any]) -> None:
    if snapshot.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected dashboard contract schema")
    if snapshot.get("status") != "research_only_read_only_not_trading":
        raise ValueError("dashboard must remain read-only research")
    if len(snapshot.get("top10", [])) != 10:
        raise ValueError("dashboard must expose exactly ten Top10 rows")
    if snapshot.get("orders") != [] or snapshot.get("automaticTradingChanges") != []:
        raise ValueError("dashboard cannot contain orders or trading changes")
    if snapshot.get("tailLossDefinition") != TAIL_DEFINITION:
        raise ValueError("tail-loss semantics changed")
    if snapshot.get("forecastReliability", {}).get("eligibleForTrading") is not False:
        raise ValueError("dashboard forecast cannot declare trading eligibility")
    securities = snapshot.get("securities", [])
    if snapshot.get("securityCount") != len(securities):
        raise ValueError("dashboard security count mismatch")
    if len({row.get("securityId") for row in securities}) != len(securities):
        raise ValueError("dashboard security IDs must be unique")


def publish_snapshot(
    result: dict[str, Any],
    forecast_rows: list[dict[str, Any]],
    output_root: Path = DEFAULT_OUTPUT,
    source_result_path: Path | None = None,
) -> dict[str, str]:
    snapshot = build_contract(result, forecast_rows, source_result_path)
    content = json.dumps(json_safe(snapshot), ensure_ascii=False, indent=2) + "\n"
    dated = output_root / "snapshots" / f"{snapshot['intendedTradingSession']}.json"
    latest = output_root / "latest.json"
    atomic_text(dated, content)
    atomic_text(latest, content)
    reparsed = json.loads(latest.read_text(encoding="utf-8"))
    validate_contract(reparsed)
    return {"snapshot": str(dated.resolve()), "latest": str(latest.resolve())}


def publish_result_file(
    result_path: Path, output_root: Path = DEFAULT_OUTPUT
) -> dict[str, str]:
    result_path = result_path.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows_path = result_path.parent / "latest_all_forecasts.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"full forecast table is missing: {rows_path}")
    rows = pd.read_csv(rows_path).to_dict(orient="records")
    return publish_snapshot(result, rows, output_root, result_path)


def search_snapshot(snapshot: dict[str, Any], query: str) -> list[dict[str, Any]]:
    term = str(query or "").strip().upper()
    if not term:
        return []
    digits = re.sub(r"\D", "", term)
    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for row in snapshot.get("securities", []):
        code = str(row["stockCode"])
        security_id = str(row["securityId"]).upper()
        name = str(row.get("name") or "").upper()
        initials = str(row.get("nameInitials") or "").upper()
        if term in {code, security_id, initials} or (
            len(digits) == 6 and code == digits
        ):
            exact.append(row)
        elif (
            term in security_id
            or term in name
            or term in initials
            or (digits and digits in code)
        ):
            partial.append(row)
    return (exact + partial)[:20]


class DashboardHandler(BaseHTTPRequestHandler):
    assets_root = DEFAULT_ASSETS
    data_root = DEFAULT_OUTPUT

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self) -> dict[str, Any]:
        path = self.data_root / "latest.json"
        if not path.exists():
            raise FileNotFoundError("latest forecast has not been published")
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_contract(value)
        return value

    def _asset(self, filename: str) -> None:
        path = (self.assets_root / filename).resolve()
        if self.assets_root.resolve() not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND.value)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                snapshot = self._snapshot()
                self._json(
                    {
                        "ok": True,
                        "schemaVersion": snapshot["schemaVersion"],
                        "signalDate": snapshot["signalDate"],
                        "intendedTradingSession": snapshot["intendedTradingSession"],
                    }
                )
                return
            if parsed.path == "/api/latest":
                snapshot = self._snapshot()
                public = {key: value for key, value in snapshot.items() if key != "securities"}
                self._json(public)
                return
            if parsed.path == "/api/search":
                query = parse_qs(parsed.query).get("q", [""])[0]
                snapshot = self._snapshot()
                self._json(
                    {
                        "query": query,
                        "signalDate": snapshot["signalDate"],
                        "intendedTradingSession": snapshot["intendedTradingSession"],
                        "matches": search_snapshot(snapshot, query),
                    }
                )
                return
            asset = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
            if asset not in {"index.html", "app.js", "styles.css"}:
                self.send_error(HTTPStatus.NOT_FOUND.value)
                return
            self._asset(asset)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": f"invalid dashboard artifact: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(
            f"{self.log_date_time_string()} {self.client_address[0]} {format % args}\n"
        )
        sys.stdout.flush()


def serve(host: str, port: int, open_browser: bool) -> None:
    if not DEFAULT_ASSETS.exists():
        raise FileNotFoundError(f"dashboard assets missing: {DEFAULT_ASSETS}")
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}/"
    print(
        json.dumps(
            {
                "status": "read_only_research_dashboard",
                "url": url,
                "latest": str((DEFAULT_OUTPUT / "latest.json").resolve()),
                "orders": [],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    server.serve_forever()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish", help="publish one completed forecast run")
    publish.add_argument("--result", type=Path, required=True)
    publish.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    server = sub.add_parser("serve", help="serve the local read-only dashboard")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.command == "publish":
        print(
            json.dumps(
                publish_result_file(args.result, args.output_root),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    serve(args.host, args.port, args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
