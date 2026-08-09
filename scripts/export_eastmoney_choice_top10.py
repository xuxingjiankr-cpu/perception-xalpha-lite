"""Export the latest research-only A-share Top10 for Choice text import.

This script never touches Choice private data files, credentials, orders, positions, or
trading configuration.  It writes a stable one-code-per-line text file that can be mapped
to a dedicated Choice self-stock group through the client's supported import workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "eastmoney_choice_top10_watchlist.json"


class ValidationError(ValueError):
    """Raised when an artifact is unsafe to publish as a watchlist."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def latest_source(source_root: Path) -> Path:
    candidates = list(source_root.glob("*/shadow_top10.json"))
    if not candidates:
        raise ValidationError(f"no shadow_top10.json under {source_root}")
    dated: list[tuple[str, Path]] = []
    for path in candidates:
        try:
            payload = load_json(path)
            dated.append((str(payload.get("signalDate", "")), path))
        except (OSError, json.JSONDecodeError):
            continue
    if not dated:
        raise ValidationError("no parseable Top10 source artifact")
    return max(dated, key=lambda item: (item[0], str(item[1])))[1]


def validate_source(payload: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schemaVersion") != config["sourceSchemaVersion"]:
        raise ValidationError("source schemaVersion is not accepted")
    if payload.get("status") not in set(config["acceptedSourceStatuses"]):
        raise ValidationError("source is not explicitly research-only")
    if payload.get("eligibleForTrading") is not False:
        raise ValidationError("source must have eligibleForTrading=false")
    if payload.get("orders") != [] or payload.get("automaticTradingChanges") != []:
        raise ValidationError("source contains an order or automatic trading change")
    rows = payload.get("top10")
    required = int(config["requiredCount"])
    if not isinstance(rows, list) or len(rows) != required:
        raise ValidationError(f"source must contain exactly {required} rows")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for expected_rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or int(row.get("rank", -1)) != expected_rank:
            raise ValidationError("Top10 ranks must be contiguous and ordered")
        security_id = str(row.get("securityId", ""))
        parts = security_id.split(".")
        if len(parts) != 2 or parts[0] not in {"SH", "SZ"}:
            raise ValidationError(f"invalid A-share securityId: {security_id}")
        exchange, code = parts
        if len(code) != 6 or not code.isdigit():
            raise ValidationError(f"invalid six-digit stock code: {security_id}")
        if code in seen:
            raise ValidationError(f"duplicate stock code: {code}")
        seen.add(code)
        validated.append({**row, "exchange": exchange, "stockCode": code})
    return validated


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    fields = [
        "rank",
        "stockCode",
        "exchange",
        "securityId",
        "name",
        "close",
        "factorScore",
        "expectedGrossReturn",
        "probabilityUp",
        "probabilityNetPositive",
        "probabilitySevereLoss",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def export(config_path: Path, source_path: Path | None, dry_run: bool) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schemaVersion") != "eastmoney_choice_top10_watchlist_v1":
        raise ValidationError("invalid exporter config schemaVersion")
    if config.get("orders") != [] or config.get("automaticTradingChanges") != []:
        raise ValidationError("exporter config must remain watchlist-only")
    if config.get("choice", {}).get("directPrivateFileMutationAllowed") is not False:
        raise ValidationError("direct Choice private-file mutation must remain disabled")
    source = source_path or latest_source(resolve_root_path(str(config["sourceRoot"])))
    payload = load_json(source)
    rows = validate_source(payload, config)
    import_path = resolve_root_path(str(config["stableImportFile"]))
    audit_path = resolve_root_path(str(config["auditCsv"]))
    manifest_path = resolve_root_path(str(config["manifest"]))
    code_bytes = ("\n".join(row["stockCode"] for row in rows) + "\n").encode("ascii")
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    manifest = {
        "schemaVersion": "eastmoney_choice_top10_export_v1",
        "status": "research_only_watchlist_only_not_trading",
        "generatedAt": now,
        "source": str(source.resolve()),
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "signalDate": payload["signalDate"],
        "intendedEntryDate": payload.get("intendedEntryDate"),
        "count": len(rows),
        "codes": [row["stockCode"] for row in rows],
        "stableImportFile": str(import_path.resolve()),
        "watchlistName": config["watchlistName"],
        "choiceIntegrationMode": config["choice"]["integrationMode"],
        "choicePrivateFilesModified": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    if not dry_run:
        atomic_bytes(import_path, code_bytes)
        atomic_bytes(audit_path, render_csv(rows))
        atomic_bytes(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        reparsed = load_json(manifest_path)
        if reparsed.get("codes") != manifest["codes"] or import_path.read_bytes() != code_bytes:
            raise RuntimeError("post-write verification failed")
    return manifest


def refresh_source_for_today() -> None:
    """Refresh the frozen shadow Top10 only on an XSHG trading session."""
    import exchange_calendars as xcals
    import pandas as pd

    calendar = xcals.get_calendar("XSHG")
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    if not calendar.is_session(today):
        return
    entry = calendar.next_session(today)
    exit_session = calendar.next_session(entry)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_alpha070_131_shadow_top10.py"),
        "--entry-date",
        str(entry.date()),
        "--exit-date",
        str(exit_session.date()),
        "--expected-signal-date",
        str(today.date()),
    ]
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Top10 source refresh failed with exit code {completed.returncode}")


def append_log(config: dict[str, Any], result: dict[str, Any] | None, error: str | None) -> None:
    root = resolve_root_path(str(config["logRoot"]))
    root.mkdir(parents=True, exist_ok=True)
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    record = {
        "time": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "ok" if error is None else "failed_closed",
        "result": result,
        "error": error,
    }
    with (root / f"eastmoney_choice_top10_{date}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-source", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    try:
        if args.refresh_source:
            refresh_source_for_today()
        result = export(config_path, args.source.resolve() if args.source else None, args.dry_run)
        append_log(config, result, None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # fail closed and retain the prior valid import file
        append_log(config, None, f"{type(exc).__name__}: {exc}")
        print(f"failed_closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
