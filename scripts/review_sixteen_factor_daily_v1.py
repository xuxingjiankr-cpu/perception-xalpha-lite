#!/usr/bin/env python3
"""Build the idempotent daily accountability ledger for the sixteen-factor Top10.

The observation Top10 is preserved exactly as published.  A separate confidence gate
may return zero qualified names; this script never creates an order or changes a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "sixteen_factor_daily_accountability_v1.json"
SCHEMA_VERSION = "sixteen_factor_daily_accountability_result_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "sixteen_factor_daily_accountability_v1":
        raise ValueError("unexpected accountability config schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("accountability ledger must remain research/shadow-only")
    gate = config["confidenceGate"]
    for key in ("minimumProbabilityUp", "maximumProbabilitySevereLoss"):
        value = float(gate[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be inside [0,1]")
    if gate.get("qualifiedListMayBeEmpty") is not True:
        raise ValueError("confidence gate must be allowed to abstain")
    if gate.get("failedGateMayNeverBeRelabeledAsBuy") is not True:
        raise ValueError("failed confidence rows may not be relabeled as buys")
    if any(bool(value) for key, value in config["safety"].items() if key.startswith("may")):
        raise ValueError("all trading mutation permissions must remain false")
    if config.get("orders") != [] or config.get("automaticTradingChanges") != []:
        raise ValueError("accountability config may not contain trading actions")


def validate_source(payload: dict[str, Any], config: dict[str, Any]) -> None:
    if payload.get("schemaVersion") != config["sourceSchemaVersion"]:
        raise ValueError("unexpected source schema")
    if payload.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("source is not research/shadow-only")
    if payload.get("eligibleForTrading") is not False:
        raise ValueError("source must have eligibleForTrading=false")
    if payload.get("orders") != [] or payload.get("automaticTradingChanges") != []:
        raise ValueError("source contains a trading action")
    rows = payload.get("latestTop10")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("source must contain exactly ten observation rows")


def source_results(root: Path, config: dict[str, Any]) -> dict[str, tuple[Path, dict[str, Any]]]:
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in root.glob("*/result.json"):
        try:
            payload = load_json(path)
            validate_source(payload, config)
            signal_date = str(payload["signalDate"])
            current = selected.get(signal_date)
            if current is None or str(payload.get("generatedAt", "")) > str(current[1].get("generatedAt", "")):
                selected[signal_date] = (path, payload)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not selected:
        raise RuntimeError("no valid sixteen-factor result artifacts found")
    return selected


def session_dates(signal_date: str) -> tuple[str, str]:
    calendar = xcals.get_calendar("XSHG")
    signal = pd.Timestamp(signal_date)
    if not calendar.is_session(signal):
        raise ValueError(f"signal date is not an XSHG session: {signal_date}")
    entry = calendar.next_session(signal)
    exit_session = calendar.next_session(entry)
    return entry.date().isoformat(), exit_session.date().isoformat()


def bars_by_date(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    found: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            date = str(row.get("dt", ""))[:10]
            if date in wanted:
                found[date] = row
    return found


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def qualification(row: dict[str, Any], reliability: str, config: dict[str, Any]) -> tuple[bool, list[str]]:
    gate = config["confidenceGate"]
    failures: list[str] = []
    probability_up = finite(row.get("probabilityUp"))
    expected = finite(row.get("expectedGrossReturn"))
    probability_tail = finite(row.get("probabilitySevereLoss"))
    if probability_up is None or probability_up < float(gate["minimumProbabilityUp"]):
        failures.append("probability_up_below_50pct")
    if expected is None or expected < float(gate["minimumExpectedGrossReturn"]):
        failures.append("expected_gross_return_below_round_trip_cost")
    if probability_tail is None or probability_tail > float(gate["maximumProbabilitySevereLoss"]):
        failures.append("tail_probability_above_limit")
    if reliability != str(gate["requiredForecastReliabilityStatus"]):
        failures.append("historical_directional_reliability_gate_failed")
    return not failures, failures


def review_one(path: Path, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    signal_date = str(payload["signalDate"])
    entry_date, exit_date = session_dates(signal_date)
    if str(payload.get("intendedTradingSession")) != entry_date:
        raise ValueError("source intended session disagrees with XSHG calendar")
    reliability = str(payload.get("forecastReliability", {}).get("status", "unknown"))
    raw_root = resolve(config["rawBarsRoot"])
    adjusted_root = resolve(config["adjustedBarsRoot"])
    rows: list[dict[str, Any]] = []
    for item in payload["latestTop10"]:
        security_id = str(item["securityId"])
        file_name = security_id.replace(".", "_") + ".jsonl"
        wanted = {entry_date, exit_date}
        adjusted = bars_by_date(adjusted_root / file_name, wanted)
        raw = bars_by_date(raw_root / file_name, wanted)
        entry = adjusted.get(entry_date) or raw.get(entry_date)
        exit_bar = adjusted.get(exit_date) or raw.get(exit_date)
        entry_open = finite(entry.get("open")) if entry else None
        entry_close = finite(entry.get("close")) if entry else None
        exit_open = finite(exit_bar.get("open")) if exit_bar else None
        provisional = (
            entry_close / entry_open - 1.0
            if entry_open and entry_close and entry_open > 0.0
            else None
        )
        final_gross = (
            exit_open / entry_open - 1.0
            if entry_open and exit_open and entry_open > 0.0
            else None
        )
        qualified, failures = qualification(item, reliability, config)
        rows.append(
            {
                "rank": int(item["rank"]),
                "securityId": security_id,
                "name": str(item.get("name", "")),
                "forecast": {
                    "expectedGrossReturn": finite(item.get("expectedGrossReturn")),
                    "probabilityUp": finite(item.get("probabilityUp")),
                    "probabilitySevereLoss": finite(item.get("probabilitySevereLoss")),
                },
                "confidenceGate": {
                    "qualified": qualified,
                    "failedReasons": failures,
                    "status": "qualified_shadow_candidate" if qualified else "observation_only_not_qualified",
                },
                "outcome": {
                    "entryOpen": entry_open,
                    "entryClose": entry_close,
                    "exitOpen": exit_open,
                    "provisionalEntrySessionReturn": provisional,
                    "finalGrossReturn": final_gross,
                    "finalNetReturn": (
                        final_gross - float(config["outcome"]["roundTripCost"])
                        if final_gross is not None
                        else None
                    ),
                    "grossUp": final_gross > 0.0 if final_gross is not None else None,
                    "severeLoss": (
                        final_gross <= float(config["outcome"]["severeLossThreshold"])
                        if final_gross is not None
                        else None
                    ),
                },
            }
        )
    completed = [row for row in rows if row["outcome"]["finalGrossReturn"] is not None]
    provisional_rows = [
        row for row in rows if row["outcome"]["provisionalEntrySessionReturn"] is not None
    ]
    return {
        "signalDate": signal_date,
        "entryDate": entry_date,
        "exitDate": exit_date,
        "sourceResult": str(path.resolve()),
        "sourceGeneratedAt": payload.get("generatedAt"),
        "forecastReliability": reliability,
        "observationCount": len(rows),
        "qualifiedCount": sum(int(row["confidenceGate"]["qualified"]) for row in rows),
        "outcomeStatus": "final" if len(completed) == 10 else "provisional" if provisional_rows else "pending",
        "provisional": {
            "observations": len(provisional_rows),
            "winRate": (
                sum(row["outcome"]["provisionalEntrySessionReturn"] > 0.0 for row in provisional_rows)
                / len(provisional_rows)
                if provisional_rows
                else None
            ),
            "meanReturn": (
                sum(row["outcome"]["provisionalEntrySessionReturn"] for row in provisional_rows)
                / len(provisional_rows)
                if provisional_rows
                else None
            ),
        },
        "final": {
            "observations": len(completed),
            "winRate": (
                sum(bool(row["outcome"]["grossUp"]) for row in completed) / len(completed)
                if completed
                else None
            ),
            "meanGrossReturn": (
                sum(row["outcome"]["finalGrossReturn"] for row in completed) / len(completed)
                if completed
                else None
            ),
            "meanNetReturn": (
                sum(row["outcome"]["finalNetReturn"] for row in completed) / len(completed)
                if completed
                else None
            ),
        },
        "top10": rows,
        "orders": [],
        "automaticTradingChanges": [],
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [
        row
        for record in records
        for row in record["top10"]
        if row["outcome"]["finalGrossReturn"] is not None
    ]
    qualified_completed = [row for row in completed_rows if row["confidenceGate"]["qualified"]]

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"observations": 0, "winRate": None, "meanGrossReturn": None, "meanNetReturn": None}
        return {
            "observations": len(rows),
            "winRate": sum(bool(row["outcome"]["grossUp"]) for row in rows) / len(rows),
            "meanGrossReturn": sum(row["outcome"]["finalGrossReturn"] for row in rows) / len(rows),
            "meanNetReturn": sum(row["outcome"]["finalNetReturn"] for row in rows) / len(rows),
        }

    return {
        "completedSignalDays": sum(record["outcomeStatus"] == "final" for record in records),
        "observationTop10": stats(completed_rows),
        "qualifiedShadowCandidates": stats(qualified_completed),
        "warning": "The qualified subset is a safety gate, not evidence of alpha; fresh-forward outcomes are required.",
    }


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def markdown(result: dict[str, Any]) -> str:
    latest = result["latest"]
    lines = [
        "# Sixteen-factor daily accountability",
        "",
        "> Research/shadow-only. The Top10 is an observation ranking, not an order.",
        "",
        f"- signal date: `{latest['signalDate']}`",
        f"- intended entry: `{latest['entryDate']}`; final label: `{latest['exitDate']}` open",
        f"- forecast reliability: `{latest['forecastReliability']}`",
        f"- qualified shadow candidates: `{latest['qualifiedCount']}/10`",
        f"- current outcome status: `{latest['outcomeStatus']}`",
        f"- provisional entry-session win rate: `{pct(latest['provisional']['winRate'])}`; mean: `{pct(latest['provisional']['meanReturn'])}`",
        f"- final next-open win rate: `{pct(latest['final']['winRate'])}`; mean gross: `{pct(latest['final']['meanGrossReturn'])}`",
        "",
        "| rank | security | name | P(up) | expected | P(tail) | qualified | provisional | final |",
        "|---:|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in latest["top10"]:
        forecast = row["forecast"]
        outcome = row["outcome"]
        lines.append(
            f"| {row['rank']} | {row['securityId']} | {row['name']} | "
            f"{pct(forecast['probabilityUp'])} | {pct(forecast['expectedGrossReturn'])} | "
            f"{pct(forecast['probabilitySevereLoss'])} | "
            f"{'yes' if row['confidenceGate']['qualified'] else 'no'} | "
            f"{pct(outcome['provisionalEntrySessionReturn'])} | {pct(outcome['finalGrossReturn'])} |"
        )
    aggregate_result = result["aggregate"]
    lines.extend(
        [
            "",
            "## Matured ledger",
            "",
            f"- completed signal days: `{aggregate_result['completedSignalDays']}`",
            f"- observation Top10: n=`{aggregate_result['observationTop10']['observations']}`, win=`{pct(aggregate_result['observationTop10']['winRate'])}`, mean gross=`{pct(aggregate_result['observationTop10']['meanGrossReturn'])}`",
            f"- qualified shadow subset: n=`{aggregate_result['qualifiedShadowCandidates']['observations']}`, win=`{pct(aggregate_result['qualifiedShadowCandidates']['winRate'])}`, mean gross=`{pct(aggregate_result['qualifiedShadowCandidates']['meanGrossReturn'])}`",
            "",
            "A single bad day is recorded but never used for same-day refitting. Low-confidence days are allowed to produce zero qualified names instead of manufacturing conviction.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    sources = source_results(resolve(config["sourceRoot"]), config)
    records = [review_one(*sources[date], config) for date in sorted(sources)]
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "confidenceGate": config["confidenceGate"],
        "records": records,
        "latest": records[-1],
        "aggregate": aggregate(records),
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    output = resolve(config["outputRoot"])
    ledger = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    rendered = markdown(result)
    atomic_text(output / "daily_ledger.jsonl", ledger)
    atomic_text(output / "latest_daily_review.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    atomic_text(output / "latest_daily_review.md", rendered)
    atomic_text(output / "daily" / f"{records[-1]['signalDate']}.md", rendered)
    return result


def main() -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"):
        __import__("sys").stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config.resolve())
    latest = result["latest"]
    print(
        json.dumps(
            {
                "signalDate": latest["signalDate"],
                "entryDate": latest["entryDate"],
                "outcomeStatus": latest["outcomeStatus"],
                "qualifiedCount": latest["qualifiedCount"],
                "provisionalWinRate": latest["provisional"]["winRate"],
                "provisionalMeanReturn": latest["provisional"]["meanReturn"],
                "orders": [],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
