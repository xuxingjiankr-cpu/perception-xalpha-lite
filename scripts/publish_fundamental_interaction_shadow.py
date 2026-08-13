#!/usr/bin/env python3
"""Merge a matching PIT-fundamental interaction result into the read-only dashboard.

The merger is fail-closed on signal-date or security mismatches.  It adds optional
shadow forecast fields but never changes rank, factor score, the selected Top10, or any
trading artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stock_forecast_dashboard as dashboard  # noqa: E402


DEFAULT_DASHBOARD = ROOT / "outputs" / "stock_forecast_dashboard" / "latest.json"
DEFAULT_INTERACTION_ROOT = (
    ROOT
    / "outputs"
    / "edge_research"
    / "twelve_factor_fundamental_price_interactions_v1"
)
EXPECTED_INTERACTION_SCHEMA = "twelve_factor_fundamental_price_interactions_result_v1"
SHADOW_STATUS = "shadow_only_rejected_historical_gate_not_ranking"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def discover_matching_result(root: Path, signal_date: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("run_*/result.json"):
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            value.get("schemaVersion") == EXPECTED_INTERACTION_SCHEMA
            and str(value.get("signalDate")) == signal_date
        ):
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(
            f"no fundamental-interaction result matches signalDate {signal_date}"
        )
    return max(candidates)[1]


def interaction_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if result.get("schemaVersion") != EXPECTED_INTERACTION_SCHEMA:
        raise ValueError("unexpected fundamental-interaction result schema")
    if result.get("eligibleForTrading") is not False or result.get("orders") != []:
        raise ValueError("interaction artifact must remain non-trading with empty orders")
    rows = result.get("latestFixedTop10Comparison", [])
    if len(rows) != 10:
        raise ValueError("interaction artifact must contain exactly ten fixed Top10 rows")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        security_id = str(row.get("securityId") or "").upper()
        if not security_id or security_id in output:
            raise ValueError("interaction securities must be unique and non-empty")
        forecasts = row.get("forecasts", {})
        baseline = forecasts.get("baseline", {})
        candidate = forecasts.get("all_interactions", {})
        baseline_values = {
            "expectedGrossReturn": _finite(
                baseline.get("expectedGrossReturn"), "baseline expected return"
            ),
            "probabilityUp": _finite(baseline.get("probabilityUp"), "baseline P(up)"),
            "probabilityTailLoss": _finite(
                baseline.get("probabilitySevereLoss"), "baseline P(tail)"
            ),
        }
        candidate_values = {
            "expectedGrossReturn": _finite(
                candidate.get("expectedGrossReturn"), "candidate expected return"
            ),
            "probabilityUp": _finite(candidate.get("probabilityUp"), "candidate P(up)"),
            "probabilityTailLoss": _finite(
                candidate.get("probabilitySevereLoss"), "candidate P(tail)"
            ),
        }
        for key in ("probabilityUp", "probabilityTailLoss"):
            if not 0.0 <= baseline_values[key] <= 1.0:
                raise ValueError(f"baseline {key} outside [0,1]")
            if not 0.0 <= candidate_values[key] <= 1.0:
                raise ValueError(f"candidate {key} outside [0,1]")
        output[security_id] = {
            "status": SHADOW_STATUS,
            **candidate_values,
            "interactionRunBaseline": baseline_values,
            "deltaExpectedGrossReturn": candidate_values["expectedGrossReturn"]
            - baseline_values["expectedGrossReturn"],
            "deltaProbabilityUp": candidate_values["probabilityUp"]
            - baseline_values["probabilityUp"],
            "deltaProbabilityTailLoss": candidate_values["probabilityTailLoss"]
            - baseline_values["probabilityTailLoss"],
            "interactionRanks": {
                str(key): _finite(value, f"interaction rank {key}")
                for key, value in row.get("interactionRanks", {}).items()
            },
            "changesRanking": False,
            "eligibleForTrading": False,
        }
    return output


def merge_snapshot(
    snapshot: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    dashboard.validate_contract(snapshot)
    signal_date = str(snapshot["signalDate"])
    if str(result.get("signalDate")) != signal_date:
        raise ValueError(
            f"signal-date mismatch: dashboard={signal_date} interaction={result.get('signalDate')}"
        )
    mapping = interaction_map(result)
    top10_ids = [str(row["securityId"]).upper() for row in snapshot["top10"]]
    if set(mapping) != set(top10_ids):
        raise ValueError("interaction Top10 differs from dashboard Top10; refusing merge")
    merged = json.loads(json.dumps(snapshot, ensure_ascii=False))
    for collection in ("top10", "securities"):
        for row in merged[collection]:
            shadow = mapping.get(str(row["securityId"]).upper())
            if shadow is not None:
                annotated = json.loads(json.dumps(shadow, ensure_ascii=False))
                annotated["deltaExpectedGrossReturn"] = annotated[
                    "expectedGrossReturn"
                ] - _finite(row.get("expectedGrossReturn"), "dashboard expected return")
                annotated["deltaProbabilityUp"] = annotated["probabilityUp"] - _finite(
                    row.get("probabilityUp"), "dashboard P(up)"
                )
                annotated["deltaProbabilityTailLoss"] = annotated[
                    "probabilityTailLoss"
                ] - _finite(row.get("probabilityTailLoss"), "dashboard P(tail)")
                row["fundamentalInteractionShadow"] = annotated
    periods = result.get("periods", {})

    def metric(period: str, scope: str, model: str, head: str, field: str) -> float | None:
        value = (
            periods.get(period, {})
            .get(model, {})
            .get(scope, {})
            .get(head, {})
            .get(field)
        )
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    merged["fundamentalInteractionShadow"] = {
        "status": SHADOW_STATUS,
        "modelCodeVersion": str(result.get("codeVersion") or ""),
        "runId": str(result.get("runId") or ""),
        "signalDate": signal_date,
        "availableTop10Rows": len(mapping),
        "rankingChanged": False,
        "historicalHypothesisPass": bool(
            result.get("acceptance", {}).get("historicalHypothesisPass", False)
        ),
        "eligibleForTrading": False,
        "validationTop10UpAuc": metric(
            "validation", "fixedTop10", "all_interactions", "grossUp", "auc"
        ),
        "shadowTop10UpAuc": metric(
            "shadow", "fixedTop10", "all_interactions", "grossUp", "auc"
        ),
        "validationTop10TailAuc": metric(
            "validation", "fixedTop10", "all_interactions", "severeLoss", "auc"
        ),
        "shadowTop10TailAuc": metric(
            "shadow", "fixedTop10", "all_interactions", "severeLoss", "auc"
        ),
        "sourceResultPath": str(result_path.resolve()),
        "sourceResultSha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "mergedAt": datetime.now().astimezone().isoformat(),
    }
    merged["orders"] = []
    merged["automaticTradingChanges"] = []
    dashboard.validate_contract(merged)
    return merged


def publish(
    dashboard_path: Path,
    interaction_result_path: Path | None = None,
) -> dict[str, Any]:
    dashboard_path = dashboard_path.resolve()
    snapshot = load_json(dashboard_path)
    dashboard.validate_contract(snapshot)
    result_path = (
        interaction_result_path.resolve()
        if interaction_result_path is not None
        else discover_matching_result(
            DEFAULT_INTERACTION_ROOT, str(snapshot["signalDate"])
        ).resolve()
    )
    result = load_json(result_path)
    merged = merge_snapshot(snapshot, result, result_path)
    content = json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
    dashboard.atomic_text(dashboard_path, content)
    dated = dashboard_path.parent / "snapshots" / f"{merged['intendedTradingSession']}.json"
    dashboard.atomic_text(dated, content)
    reparsed = load_json(dashboard_path)
    dashboard.validate_contract(reparsed)
    return {
        "dashboard": str(dashboard_path),
        "datedSnapshot": str(dated),
        "interactionResult": str(result_path),
        "signalDate": merged["signalDate"],
        "intendedTradingSession": merged["intendedTradingSession"],
        "availableTop10Rows": merged["fundamentalInteractionShadow"][
            "availableTop10Rows"
        ],
        "rankingChanged": False,
        "eligibleForTrading": False,
        "orders": [],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--interaction-result", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(args.dashboard, args.interaction_result),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
