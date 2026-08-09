"""Command line interface for the dependence-aware Evidence Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evidence import BootstrapDesign, evidence_report, write_evidence_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress-test a searched candidate family against a frozen benchmark"
    )
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2_000)
    parser.add_argument("--expected-block-length", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=20_260_809)
    args = parser.parse_args()

    frame = pd.read_csv(args.performance)
    if args.date_column not in frame or args.benchmark not in frame:
        raise ValueError("date and benchmark columns must exist in the input")
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="raise")
    frame = frame.set_index(args.date_column).sort_index()
    benchmark = pd.to_numeric(frame.pop(args.benchmark), errors="coerce")
    if frame.empty:
        raise ValueError("at least one candidate column is required")
    differentials = frame.apply(pd.to_numeric, errors="coerce").sub(benchmark, axis=0)
    report = evidence_report(
        differentials,
        BootstrapDesign(
            repetitions=args.repetitions,
            expected_block_length=args.expected_block_length,
            confidence=args.confidence,
            alpha=args.alpha,
            random_seed=args.random_seed,
        ),
    )
    write_evidence_report(report, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
