"""Allocate an immutable Decision Scoring research iteration number.

This only updates the research version registry. It never changes live weights, strategy
configuration, overlays or order gating.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT


REGISTRY = ROOT / "configs" / "research" / "decision_score_version_registry.json"
VALID_STATUSES = {"planned", "completed", "rejected", "diagnostic_only", "forward_shadow"}


def allocate_iteration(registry: dict[str, Any], *, summary: str, change_type: str,
                       status: str, commit: str, versions: dict[str, str] | None = None,
                       timestamp: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    number = int(registry.get("nextIterationNumber", 1))
    iteration_id = f"DSI-{number:04d}"
    active = dict(registry.get("active", {}))
    for key, value in (versions or {}).items():
        if value:
            active[key] = value
    active["iterationId"] = iteration_id
    entry = {
        "iterationId": iteration_id,
        "date": (timestamp or datetime.now().astimezone().isoformat())[:10],
        "commit": commit,
        "status": status,
        "changeType": change_type,
        "summary": summary,
        "versions": {key: value for key, value in (versions or {}).items() if value},
    }
    updated = json.loads(json.dumps(registry))
    updated["active"] = active
    updated["nextIterationNumber"] = number + 1
    updated.setdefault("history", []).append(entry)
    return updated, entry


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Allocate next Decision Scoring iteration ID.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--change-type", required=True,
                        choices=["scorer", "weights", "outcome", "pipeline", "shadow", "audit"])
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--scorer-version", default="")
    parser.add_argument("--weights-version", default="")
    parser.add_argument("--outcome-version", default="")
    parser.add_argument("--pipeline-version", default="")
    parser.add_argument("--shadow-version", default="")
    parser.add_argument("--calibration-version", default="")
    parser.add_argument("--bayesian-version", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    versions = {
        "scorerLogicVersion": args.scorer_version,
        "weightsVersion": args.weights_version,
        "outcomeModelVersion": args.outcome_version,
        "pipelineVersion": args.pipeline_version,
        "shadowCandidateVersion": args.shadow_version,
        "calibrationVersion": args.calibration_version,
        "bayesianModelVersion": args.bayesian_version,
    }
    updated, entry = allocate_iteration(
        registry, summary=args.summary, change_type=args.change_type,
        status=args.status, commit=git_commit(), versions=versions,
    )
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    if not args.dry_run:
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        temporary = REGISTRY.with_suffix(".tmp")
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(REGISTRY)
        print(f"updated: {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
