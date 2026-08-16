"""Run the complete research loop on deterministic synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .discovery import run_discovery, write_result
from .doctor import audit_research_data
from .synthetic import demo_config, make_synthetic_data


PROJECT_URL = "https://github.com/xuxingjiankr-cpu/perception-xalpha-lite"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zero-setup, research-only demonstration of the full XAlpha loop"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("xalpha-demo-output"))
    parser.add_argument("--days", type=int, default=780)
    parser.add_argument("--symbols", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20_260_816)
    args = parser.parse_args()

    prices, fundamentals = make_synthetic_data(args.days, args.symbols, args.seed)
    config = demo_config(args.seed)
    readiness = audit_research_data(prices, fundamentals, config)
    if readiness["status"] == "blocked":
        raise RuntimeError("the bundled synthetic data failed its own readiness contract")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output / "prices.csv", index=False)
    fundamentals.to_csv(output / "fundamentals.csv", index=False)
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "readiness.json").write_text(
        json.dumps(readiness, indent=2) + "\n", encoding="utf-8"
    )
    result = run_discovery(prices, fundamentals, config)
    write_result(result, output / "result.json")
    print(
        json.dumps(
            {
                "status": result["status"],
                "data_readiness": readiness["status"],
                "candidate_count": result["candidate_count"],
                "stage2_count": result["stage2_count"],
                "validation_survivor_count": result["validation_survivor_count"],
                "hard_stop": result["hard_stop"],
                "output_dir": str(output.resolve()),
                "orders": [],
                "note": "Synthetic output demonstrates mechanics, never investment evidence.",
                "support_if_useful": f"Star the project: {PROJECT_URL}",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
