from __future__ import annotations

import pandas as pd

from xalpha_lite.doctor import audit_research_data
from xalpha_lite.synthetic import demo_config, make_synthetic_data


def test_clean_synthetic_contract_is_ready_and_never_emits_orders() -> None:
    prices, fundamentals = make_synthetic_data(days=100, symbols=4)
    config = demo_config()
    config.update(
        minimum_train_days=10,
        validation_days=5,
        shadow_days=5,
        purge_days=2,
        prediction_horizon_days=2,
    )
    report = audit_research_data(prices, fundamentals, config)
    assert report["status"] == "ready_with_warnings"
    assert report["summary"]["blocking_checks"] == 0
    assert report["orders"] == []
    assert report["input_digests"]["prices_sha256"]


def test_doctor_fails_closed_on_duplicate_bars_and_missing_notice_dates() -> None:
    prices, fundamentals = make_synthetic_data(days=100, symbols=4)
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    fundamentals.loc[fundamentals.index[0], "notice_date"] = None
    config = demo_config()
    config.update(
        minimum_train_days=10,
        validation_days=5,
        shadow_days=5,
        purge_days=2,
        prediction_horizon_days=2,
    )
    report = audit_research_data(prices, fundamentals, config)
    failed = {row["check_id"] for row in report["checks"] if row["status"] == "fail"}
    assert report["status"] == "blocked"
    assert {"price_keys", "disclosure_timing"}.issubset(failed)


def test_future_named_column_is_visible_but_not_mislabeled_as_proven_leakage() -> None:
    prices, fundamentals = make_synthetic_data(days=30, symbols=4)
    prices["future_return_label"] = 0.0
    report = audit_research_data(prices, fundamentals, None)
    check = next(row for row in report["checks"] if row["check_id"] == "future_named_columns")
    assert check["status"] == "warn"
    assert check["metrics"]["suspicious_columns"] == ["future_return_label"]
    assert "not proof" in check["message"]
