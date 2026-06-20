"""Audit end-of-day liquidity look-ahead and define a point-in-time research gate."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEAKY = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
RAW = ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
OUT_JSON = ROOT / "outputs" / "edge_research" / "liquidity_lookahead_audit.json"
OUT_MD = ROOT / "outputs" / "edge_research" / "liquidity_lookahead_audit.md"


def point_in_time_liquidity_gate(
    *, cumulative_amount: float, session_fraction: float,
    previous_day_adv: float | None, full_day_floor: float = 50_000_000.0,
) -> tuple[bool, str, float]:
    """Use only information available at the decision timestamp."""
    fraction = max(0.05, min(1.0, float(session_fraction)))
    scaled_floor = float(full_day_floor) * fraction
    if previous_day_adv is not None and float(previous_day_adv) >= full_day_floor:
        return True, "previous_day_adv", scaled_floor
    if float(cumulative_amount) >= scaled_floor:
        return True, "current_cumulative_amount_scaled_by_elapsed_session", scaled_floor
    return False, "below_point_in_time_liquidity_floor", scaled_floor


def leaky_keys(path: Path) -> set[tuple[str, str]]:
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            keys.add((str(row.get("stockCode", "")), str(row.get("timestamp", ""))[:10]))
    return keys


def raw_day_returns(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[(str(row["stockCode"]), str(row["trade_date"]))].append(row)
    out = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["timestamp"])
        if len(rows) < 40:
            continue
        prices = [float(row["close"]) for row in rows]
        out[key] = {
            "open_close": prices[-1] / prices[0] - 1.0,
            "amplitude": max(prices) / min(prices) - 1.0,
            "final_amount": float(rows[-1].get("cumulative_amount") or 0.0),
        }
    return out


def mean_pct(values: list[float]) -> float | None:
    return round(statistics.fmean(values) * 100, 4) if values else None


def build_report() -> dict[str, Any]:
    selected_keys = leaky_keys(LEAKY)
    raw = raw_day_returns(RAW)
    selected = [row for key, row in raw.items() if key in selected_keys]
    omitted = [row for key, row in raw.items() if key not in selected_keys]
    all_rows = list(raw.values())
    selected_mean = mean_pct([row["open_close"] for row in selected])
    all_mean = mean_pct([row["open_close"] for row in all_rows])
    return {
        "schemaVersion": "liquidity_lookahead_audit_v1",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only", "liveReady": False, "formalStrategyAllowed": False,
        "findings": {
            "select_t0_universe_live": {
                "lookahead": False,
                "reason": "uses the current snapshot cumulative amount and elapsed-session fraction only",
            },
            "fetch_yahoo_5m_quotes": {
                "lookahead": True,
                "reason": "fullamt[date] is computed from the complete day before the date's opening rows are emitted",
            },
            "convert_june_to_quotes": {
                "lookahead": True,
                "reason": "full-day amount selects the universe before all intraday decision rows are emitted",
            },
            "build_t0_replay_quotes_from_minute_data": {
                "lookahead": False,
                "reason": "liquidity threshold is evaluated from each timestamp's cumulative amount/session fraction",
            },
        },
        "safeResearchPath": {
            "rawCollector": "scripts/fetch_t0_research_quotes.py",
            "gate": "previous-day ADV >= full-day floor OR current cumulative amount >= floor * elapsed-session fraction",
            "futureDayTurnoverUsed": False,
        },
        "overlapDiagnostic": {
            "rawETFDaySamples": len(all_rows), "selectedByLeakyFullDayGate": len(selected),
            "omittedByLeakyFullDayGate": len(omitted),
            "allRawMeanOpenClosePct": all_mean,
            "leakySelectedMeanOpenClosePct": selected_mean,
            "selectionMeanDifferencePct": round(selected_mean - all_mean, 4) if selected_mean is not None and all_mean is not None else None,
            "interpretation": "descriptive selection effect only; it is not a strategy-PnL correction",
        },
        "baselineImpact": {
            "direction": "potentially_optimistic_and_not_point_in_time_valid",
            "exactCorrectionAvailable": False,
            "reason": "the ungated raw 60-day archive does not cover the original full ~500-ETF universe, so -0.11%/+1.61% cannot be honestly restated",
            "decision": "treat prior baseline levels as method-contaminated diagnostics; rebuild before future opening/intraday claims",
        },
    }


def main() -> int:
    report = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlap = report["overlapDiagnostic"]
    lines = ["# Liquidity Look-ahead Audit", "",
             "| Path | Look-ahead | Finding |", "|---|---|---|",]
    for name, row in report["findings"].items():
        lines.append(f"| {name} | {row['lookahead']} | {row['reason']} |")
    lines += ["", "## Measured overlap effect", "",
              f"- Raw ETF-days: {overlap['rawETFDaySamples']}; selected by leaky gate: {overlap['selectedByLeakyFullDayGate']}.",
              f"- Mean open→close: all raw {overlap['allRawMeanOpenClosePct']}%, leaky-selected {overlap['leakySelectedMeanOpenClosePct']}%, difference {overlap['selectionMeanDifferencePct']}%.",
              "- This is not a valid correction to strategy PnL; the original full universe was not retained ungated.",
              "", "## Decision", "",
              f"- {report['baselineImpact']['decision']}.",
              "- Live selector is unchanged; only future research must use the point-in-time path."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
