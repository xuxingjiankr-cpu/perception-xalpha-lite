from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xalpha_lite.discovery import build_panel, neutral_factor_weights, run_discovery
from xalpha_lite.dsl import evaluate, validate
from xalpha_lite.paper_mechanisms import (
    bocpd_change_probability,
    causal_dmd_residual,
    causal_two_state_hmm_filter,
    early_warning_features,
    hoeffding_lower_bound,
    triple_barrier_labels,
)
from xalpha_lite.pit import align_point_in_time_fundamentals


def test_pit_uses_next_session_after_later_disclosure_date() -> None:
    sessions = pd.bdate_range("2026-01-01", "2026-01-12")
    statements = pd.DataFrame(
        [{"symbol": "A", "report_date": "2025-12-31", "notice_date": "2026-01-03", "update_date": "2026-01-06", "roe": 12.0}]
    )
    result = align_point_in_time_fundamentals(statements, sessions)["roe"]["A"]
    assert result.loc[:"2026-01-06"].isna().all()
    assert result.loc[pd.Timestamp("2026-01-07")] == 12.0


def test_dsl_is_prefix_causal_and_rejects_unknown_fields() -> None:
    dates = pd.bdate_range("2025-01-01", periods=80)
    panel = {"returns": pd.DataFrame({"A": np.arange(80) / 1000}, index=dates)}
    expression = {"rolling": "mean", "arg": {"field": "returns"}, "window": 20}
    validate(expression, {"returns"}, {20})
    full = evaluate(expression, panel)
    prefix = evaluate(expression, {"returns": panel["returns"].iloc[:50]})
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    try:
        validate({"field": "future_return"}, {"returns"}, {20})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown future field was accepted")


def test_paper_primitives_are_causal_and_bounded() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    series = pd.Series(np.sin(np.arange(100) / 10), index=dates)
    hmm = causal_two_state_hmm_filter(series, (-0.2, 0.2), (0.5, 0.5), np.array([[0.95, 0.05], [0.05, 0.95]]))
    assert np.allclose(hmm.sum(axis=1), 1.0)
    assert early_warning_features(series, 20).index.equals(series.index)
    assert bocpd_change_probability(series).between(0, 1).all()
    features = pd.DataFrame({"x": series, "y": series.shift(1).fillna(0)})
    full = causal_dmd_residual(features, 20)
    prefix = causal_dmd_residual(features.iloc[:70], 20)
    pd.testing.assert_frame_equal(full.iloc[:70], prefix)
    assert 0 <= hoeffding_lower_bound(60, 100) <= 0.6
    labels = triple_barrier_labels(100 * (1 + series / 10), 0.02, 0.02, 10)
    assert len(labels) == len(series)


def test_discovery_never_returns_orders() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    from run_synthetic import make_data

    prices, fundamentals = make_data(days=650, symbols=20)
    config = json.loads((root / "configs/example.json").read_text(encoding="utf-8"))
    config.update(
        {
            "minimum_train_days": 300,
            "validation_days": 100,
            "shadow_days": 100,
            "candidate_budget": 24,
            "stage2_budget": 4,
            "placebo_repetitions": 7,
            "pbo_blocks": 4,
        }
    )
    result = run_discovery(prices, fundamentals, config)
    assert result["status"] == "diagnostic_only_research_only_not_trading"
    assert result["orders"] == []
    assert result["automatic_trading_changes"] == []
    assert result["candidate_count"] > 0
    assert any(row["generation"] > 0 for row in result["mechanism_tree"])
    assert all(
        value is None
        or abs(value)
        < config["maximum_factor_behavior_correlation"]
        for value in result["train_factor_correlation"].values()
    )
    assert result["shadow_metrics_disclosed"] is False
    assert "shadow_commitment_sha256" in result
    assert all("shadow" not in row for row in result["candidates"])


def test_neutral_book_is_zero_investment_and_not_market_beta() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    from run_synthetic import make_data

    prices, fundamentals = make_data(days=120, symbols=20)
    config = json.loads((root / "configs/example.json").read_text(encoding="utf-8"))
    config["minimum_daily_amount"] = 0
    panel = build_panel(prices, fundamentals, config)
    weights = neutral_factor_weights(panel["returns"].rolling(5).sum(), panel, config)
    active = weights.abs().sum(axis=1) > 0
    assert np.allclose(weights.loc[active].sum(axis=1), 0.0, atol=1e-10)
    assert (weights.loc[active].abs().sum(axis=1) <= 1.0 + 1e-10).all()
    equal_weight_market = panel["returns"].mean(axis=1)
    factor_return = (weights * panel["returns"].shift(-1)).sum(axis=1)
    correlation = factor_return.loc[active].corr(equal_weight_market.loc[active])
    assert abs(correlation) < 0.5


def test_transaction_cost_is_applied_to_factor_returns() -> None:
    from xalpha_lite.discovery import _costed_from_desired_array, factor_portfolio

    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    from run_synthetic import make_data

    prices, fundamentals = make_data(days=120, symbols=20)
    config = json.loads((root / "configs/example.json").read_text(encoding="utf-8"))
    config["minimum_daily_amount"] = 0
    panel = build_panel(prices, fundamentals, config)
    signal = panel["returns"].rolling(5).sum()
    one_day = panel["open"].shift(-2) / panel["open"].shift(-1) - 1
    zero_cost = factor_portfolio(signal, one_day, panel, config, 0.0)
    costed = factor_portfolio(signal, one_day, panel, config, 0.003)
    expected_drag = costed["turnover"] * 0.003 / 2
    pd.testing.assert_series_equal(
        zero_cost["gross"] - costed["costed"], expected_drag, check_names=False
    )
    array_costed = _costed_from_desired_array(
        costed["desired_weights"].to_numpy(dtype=float),
        one_day.to_numpy(dtype=float),
        int(config["prediction_horizon_days"]),
        0.003,
    )
    np.testing.assert_allclose(array_costed, costed["costed"].to_numpy(), atol=1e-12)


def test_future_columns_and_missing_disclosure_fail_closed() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    from run_synthetic import make_data

    prices, fundamentals = make_data(days=120, symbols=20)
    prices["future_return"] = prices.groupby("symbol")["close"].shift(-1) / prices["close"] - 1
    config = json.loads((root / "configs/example.json").read_text(encoding="utf-8"))
    panel = build_panel(prices, fundamentals, config)
    assert "future_return" not in panel
    broken = fundamentals.copy()
    broken.loc[broken.index[0], "notice_date"] = None
    try:
        build_panel(prices, broken, config)
    except ValueError as error:
        assert "notice_date is mandatory" in str(error)
    else:
        raise AssertionError("a fundamental row without notice_date was accepted")
