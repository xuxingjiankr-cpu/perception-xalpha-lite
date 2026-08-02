from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xalpha_lite.discovery import run_discovery
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
    config.update({"minimum_train_days": 300, "validation_days": 100, "shadow_days": 100})
    result = run_discovery(prices, fundamentals, config)
    assert result["status"] == "diagnostic_only_research_only_not_trading"
    assert result["orders"] == []
    assert result["automatic_trading_changes"] == []
    assert result["candidate_count"] > 0

