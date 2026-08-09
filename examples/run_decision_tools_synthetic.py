"""Synthetic-only example for the optional decision research toolkit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from xalpha_lite.decision import (
    calibration_rows,
    chronological_research_partitions,
    fit_logistic_probability_model,
    fit_pairwise_weight_ensemble,
    probability_metrics,
)


def make_synthetic_book() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=900)
    symbols = [f"S{index:03d}" for index in range(100)]
    factors = {
        f"feature_{index:02d}": pd.DataFrame(
            rng.uniform(size=(len(dates), len(symbols))),
            index=dates,
            columns=symbols,
        )
        for index in range(1, 5)
    }
    outcomes = (
        0.02 * factors["feature_01"]
        - 0.01 * factors["feature_02"]
        + pd.DataFrame(
            rng.normal(scale=0.02, size=factors["feature_01"].shape),
            index=dates,
            columns=symbols,
        )
    )
    return factors, outcomes


def main() -> None:
    factors, outcomes = make_synthetic_book()
    partitions = chronological_research_partitions(
        outcomes.index,
        calibration_days=60,
        audit_days=60,
        validation_days=80,
        shadow_days=80,
        purge_days=10,
        minimum_fit_days=500,
    )
    ensemble = fit_pairwise_weight_ensemble(
        factors,
        outcomes,
        partitions["fit"],
        replicas=3,
        block_length=21,
        subsample_fraction=0.8,
        minimum_weight=0.02,
        maximum_weight=0.60,
        top_count=10,
        boundary_count=20,
        minimum_cross_section=50,
    )

    x, y, sample_weight, names = calibration_rows(
        factors, outcomes, partitions["calibration"]
    )
    loss_model = fit_logistic_probability_model(
        x,
        y <= 0.0,
        feature_names=names,
        sample_weight=sample_weight,
    )
    x_audit, y_audit, _, _ = calibration_rows(
        factors, outcomes, partitions["audit"]
    )
    diagnostics = probability_metrics(
        y_audit <= 0.0, loss_model.predict_proba(x_audit)
    )
    print(
        json.dumps(
            {
                "status": "synthetic_demo_research_only_not_trading",
                "factor_names": ensemble["factor_names"],
                "weights": ensemble["weight_map"],
                "weight_replica_std": dict(
                    zip(
                        ensemble["factor_names"],
                        ensemble["replica_std"],
                        strict=True,
                    )
                ),
                "audit_probability_metrics": {
                    key: diagnostics[key]
                    for key in ("brier", "log_loss", "auc", "ece")
                },
                "orders": [],
                "automatic_trading_changes": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
