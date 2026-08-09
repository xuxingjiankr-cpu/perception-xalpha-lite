from __future__ import annotations

import numpy as np
import pandas as pd

from xalpha_lite.evidence import (
    BootstrapDesign,
    benjamini_hochberg_qvalues,
    benjamini_yekutieli_qvalues,
    evidence_report,
    stationary_bootstrap_indices,
)


def _dependent_candidate_family() -> pd.DataFrame:
    rng = np.random.default_rng(20260809)
    observations = 900
    innovations = rng.normal(scale=0.01, size=(observations, 4))
    values = np.zeros_like(innovations)
    for position in range(1, observations):
        values[position] = 0.35 * values[position - 1] + innovations[position]
    values[:, 0] += 0.0035
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2022-01-03", periods=observations),
        columns=["planted", "noise_a", "noise_b", "noise_c"],
    )


def test_stationary_bootstrap_is_deterministic_and_preserves_blocks() -> None:
    first = stationary_bootstrap_indices(100, 199, 12.0, 9)
    second = stationary_bootstrap_indices(100, 199, 12.0, 9)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (199, 100)
    assert ((first >= 0) & (first < 100)).all()
    continuations = first[:, 1:] == (first[:, :-1] + 1) % 100
    assert 0.80 < continuations.mean() < 0.95


def test_multiple_testing_tools_find_only_strong_planted_evidence() -> None:
    report = evidence_report(
        _dependent_candidate_family(),
        BootstrapDesign(
            repetitions=999,
            expected_block_length=12,
            alpha=0.05,
            random_seed=17,
        ),
    )
    assert report["white_reality_check"]["global_null_rejected"] is True
    candidates = report["multiple_testing"]["candidates"]
    assert candidates["planted"]["familywise_rejected"] is True
    assert candidates["planted"]["dependence_robust_fdr_rejected"] is True
    assert all(
        candidates[name]["familywise_rejected"] is False
        for name in ("noise_a", "noise_b", "noise_c")
    )
    assert report["orders"] == []
    assert report["automatic_trading_changes"] == []


def test_zero_family_fails_closed_without_spurious_evidence() -> None:
    frame = pd.DataFrame(
        np.zeros((100, 3)),
        index=pd.bdate_range("2025-01-01", periods=100),
        columns=["a", "b", "c"],
    )
    report = evidence_report(
        frame,
        BootstrapDesign(repetitions=199, expected_block_length=5),
    )
    assert report["white_reality_check"]["global_null_rejected"] is False
    assert all(
        row["familywise_rejected"] is False
        for row in report["multiple_testing"]["candidates"].values()
    )


def test_fdr_adjustments_are_ordered_and_dependency_correction_is_stricter() -> None:
    p_values = {"a": 0.001, "b": 0.02, "c": 0.3, "d": 0.9}
    bh = benjamini_hochberg_qvalues(p_values)
    by = benjamini_yekutieli_qvalues(p_values)
    assert all(0 <= value <= 1 for value in bh.values())
    assert all(by[name] >= bh[name] for name in p_values)
    assert bh["a"] <= bh["b"] <= bh["c"] <= bh["d"]


def test_joint_missing_rows_are_reported_not_silently_imputed() -> None:
    frame = _dependent_candidate_family().iloc[:100].copy()
    frame.iloc[5, 0] = np.nan
    report = evidence_report(
        frame,
        BootstrapDesign(repetitions=199, expected_block_length=5),
    )
    assert report["sample_audit"]["input_observations"] == 100
    assert report["sample_audit"]["joint_observations"] == 99
    assert report["sample_audit"]["dropped_non_joint_observations"] == 1
