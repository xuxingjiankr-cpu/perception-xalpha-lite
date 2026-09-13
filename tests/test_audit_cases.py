from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from xalpha_lite.audit_cases import (
    build_cases, disclosure_case, noise_case, price_basis_case, select_vwap_for_case, write_cases,
)
from xalpha_lite.discovery import deflated_sharpe_ratio, pbo
from xalpha_lite.pit import align_point_in_time_fundamentals


def test_noise_example_calls_real_diagnostics_and_uses_identical_evaluation_support():
    case, tables = noise_case()
    returns = tables["noise_returns.csv"]
    train, test = returns.iloc[:504], returns.iloc[504:]
    assert train.index.max() < test.index.min()
    sharpes = returns.mean() / returns.std(ddof=1)
    expected = deflated_sharpe_ratio(returns[sharpes.idxmax()], sharpes.tolist(), 64)
    assert case["dsr_statistic"] == expected["probability"]
    assert case["pbo"] == pbo(returns, 8)["pbo"]
    assert case["declared_trials"] == returns.shape[1] == 64
    winner = (train.mean() / train.std(ddof=1)).idxmax()
    assert case["train_selected_variant"] == winner
    assert case["train_selected_evaluation_bps"] == pytest.approx(test[winner].mean() * 1e4, abs=1e-6)
    hindsight = (test.mean() / test.std(ddof=1)).idxmax()
    assert case["hindsight_evaluation_bps"] == pytest.approx(test[hindsight].mean() * 1e4, abs=1e-6)
    # Do not require a random realization to lose or every seed to reject: that would be fishing.


def test_future_returns_do_not_change_the_train_selected_variant():
    case, tables = noise_case()
    changed = tables["noise_returns.csv"].copy()
    changed.iloc[504:] = changed.iloc[504:] * -100
    training = changed.iloc[:504]
    assert (training.mean() / training.std(ddof=1)).idxmax() == case["train_selected_variant"]


def test_disclosure_and_restatement_become_available_only_after_their_timestamps():
    case, tables = disclosure_case()
    rows = tables["pit_alignment.csv"]
    assert case["first_disclosed_value_available"] == "2020-02-17"
    assert case["restatement_available"] == "2020-03-04"
    assert rows.loc["2020-02-14", "disclosure_alignment"] == 1
    assert rows.loc["2020-02-17", "disclosure_alignment"] == 4
    assert rows.loc["2020-03-03", "disclosure_alignment"] == 4
    assert rows.loc["2020-03-04", "disclosure_alignment"] == 2.5
    assert case["mismatched_rows"] == rows.unsafe_row.sum() == 26


def test_appending_restatement_cannot_rewrite_past_features():
    _, tables = disclosure_case()
    statements, sessions = tables["disclosures.csv"], tables["pit_alignment.csv"].index
    original = align_point_in_time_fundamentals(statements.iloc[:2], sessions)["eps"]
    revised = align_point_in_time_fundamentals(statements, sessions)["eps"]
    pd.testing.assert_frame_equal(original.loc[:"2020-03-03"], revised.loc[:"2020-03-03"])


def test_missing_disclosure_date_is_not_imputed_from_report_date():
    _, tables = disclosure_case()
    statements = tables["disclosures.csv"].copy()
    statements.loc[1, "notice_date"] = None
    with pytest.raises(ValueError, match="notice_date is mandatory"):
        align_point_in_time_fundamentals(statements, tables["pit_alignment.csv"].index)


def test_price_basis_preserves_archive_and_changes_no_support():
    case, tables = price_basis_case()
    frame = tables["price_basis_inputs.csv"]
    before = frame.copy(deep=True)
    chosen = select_vwap_for_case(frame)
    pd.testing.assert_series_equal(chosen, frame.archive_vwap)
    pd.testing.assert_frame_equal(before, frame)
    comparison = tables["price_basis_comparison.csv"]
    pd.testing.assert_index_equal(frame.index, comparison.index)
    assert np.isfinite(comparison).all().all()
    assert case["rank_changes"] == 4
    assert case["max_archive_to_cash_ratio"] > 5


def test_raw_cash_fallback_is_rejected_against_adjusted_close():
    _, tables = price_basis_case()
    frame = tables["price_basis_inputs.csv"].drop(columns="archive_vwap")
    with pytest.raises(ValueError, match="mixed_price_basis"):
        select_vwap_for_case(frame)


def test_unadjusted_fallback_with_consistent_metadata_is_accepted():
    _, tables = price_basis_case()
    frame = tables["price_basis_inputs.csv"].drop(columns="archive_vwap").copy()
    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].div(frame.adjustment_multiplier, axis=0)
    frame["close_basis"] = "raw"
    pd.testing.assert_series_equal(select_vwap_for_case(frame), frame.amount / frame.volume)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0., 1e9])
def test_invalid_archive_is_not_silently_replaced(invalid):
    _, tables = price_basis_case()
    frame = tables["price_basis_inputs.csv"].copy()
    frame.iloc[0, frame.columns.get_loc("archive_vwap")] = invalid
    with pytest.raises(ValueError, match="invalid_vwap"):
        select_vwap_for_case(frame)


def test_false_basis_metadata_is_rejected_even_if_numeric_value_looks_plausible():
    _, tables = price_basis_case()
    frame = tables["price_basis_inputs.csv"].copy()
    frame["archive_vwap_basis"] = "raw"
    with pytest.raises(ValueError, match="mixed_price_basis"):
        select_vwap_for_case(frame)


def test_reproducible_artifacts_are_synthetic_and_hash_every_input(tmp_path):
    report = write_cases(tmp_path / "first")
    assert report == write_cases(tmp_path / "second") == build_cases()[0]
    assert report["status"] == "synthetic_demonstration_only"
    assert report["orders"] == report["automatic_trading_changes"] == []
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert len(manifest["artifact_sha256"]) == 7
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest() == digest

