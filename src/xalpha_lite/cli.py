"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .discovery import run_discovery, write_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only point-in-time factor discovery")
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--fundamentals", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/example.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_discovery(pd.read_csv(args.prices), pd.read_csv(args.fundamentals), config)
    write_result(result, args.output)
    print(json.dumps({key: result[key] for key in ("status", "candidate_count", "stage2_count", "historically_validated_count", "pbo")}, ensure_ascii=False, indent=2))
    print(f"saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

