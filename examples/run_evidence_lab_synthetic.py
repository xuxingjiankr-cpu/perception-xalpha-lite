"""Synthetic-only demonstration of the dependence-aware Evidence Lab."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from xalpha_lite import BootstrapDesign, evidence_report


def make_synthetic_differentials() -> pd.DataFrame:
    rng = np.random.default_rng(20260809)
    observations = 750
    innovations = rng.normal(scale=0.01, size=(observations, 6))
    values = np.zeros_like(innovations)
    for position in range(1, observations):
        values[position] = 0.30 * values[position - 1] + innovations[position]
    values[:, 0] += 0.0025
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2023-01-02", periods=observations),
        columns=[f"candidate_{index:02d}" for index in range(1, 7)],
    )


if __name__ == "__main__":
    artifact = evidence_report(
        make_synthetic_differentials(),
        BootstrapDesign(
            repetitions=999,
            expected_block_length=10,
            random_seed=20260809,
        ),
    )
    summary = {
        "status": artifact["status"],
        "sample_audit": artifact["sample_audit"],
        "white_reality_check": artifact["white_reality_check"],
        "candidate_adjustments": artifact["multiple_testing"]["candidates"],
        "orders": artifact["orders"],
    }
    print(json.dumps(summary, indent=2))
