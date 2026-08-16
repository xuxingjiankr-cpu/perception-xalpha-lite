"""Command-line interface for the point-in-time data doctor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .doctor import audit_research_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed readiness audit for factor-research CSV inputs"
    )
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--fundamentals", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="return exit code 3 when the report contains unresolved warnings",
    )
    args = parser.parse_args()

    fundamentals = pd.read_csv(args.fundamentals) if args.fundamentals else None
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    report = audit_research_data(pd.read_csv(args.prices), fundamentals, config)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if report["status"] == "blocked":
        return 2
    if args.strict_warnings and report["summary"]["warnings"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
