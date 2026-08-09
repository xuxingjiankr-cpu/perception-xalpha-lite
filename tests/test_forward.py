from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xalpha_lite.book import long_only_book, long_only_weights, select_names
from xalpha_lite.discovery import factor_portfolio
from xalpha_lite.forward import (
    append_prediction,
    build_prediction,
    freeze_spec,
    load_spec,
    read_log,
    score_log,
    spec_digest,
    validate_spec,
)
from xalpha_lite.universe import point_in_time_eligibility, sealed_bar_limits

SYMBOLS = [f"S{index:02d}" for index in range(20)]
CONFIG = {
    "prediction_horizon_days": 5,
    "round_trip_cost": 0.003,
    "book_size": 5,
    "minimum_daily_amount": 0.0,
    "maximum_amount_participation": 0.01,
}


def synthetic_panel(days: int = 160, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    steps = rng.normal(0.0, 0.02, size=(days, len(SYMBOLS)))
    close = pd.DataFrame(10.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=SYMBOLS)
    volume = pd.DataFrame(
        rng.uniform(1e5, 5e5, size=(days, len(SYMBOLS))), index=dates, columns=SYMBOLS
    )
    panel = {
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": volume,
        "amount": close * volume,
    }
    panel["returns"] = close.pct_change(fill_method=None)
    panel["vwap"] = close
    panel["market_cap"] = close * 1e8
    panel["industry"] = pd.DataFrame(
        np.tile(np.array(["a", "b", "c", "d"])[None, :].repeat(5, axis=1), (days, 1)),
        index=dates,
        columns=SYMBOLS,
    )
    for flag in ("is_st", "is_suspended", "is_delisted", "limit_up", "limit_down"):
        panel[flag] = pd.DataFrame(False, index=dates, columns=SYMBOLS)
    return panel


def spec_fixture() -> dict[str, object]:
    return {
        "name": "unit_test_spec",
        "rationale": "exercises the frozen-record contract",
        "factors": {"reversal_20": {"unary": "neg", "arg": {"rolling": "mean", "arg": {"field": "returns"}, "window": 20}}},
        "combination": "equal_weight_rank_average",
        "book_size": 5,
        "holding_days": 5,
        "round_trip_cost": 0.003,
        "universe": {"minimum_prior_observations": 30},
    }


def test_sealed_bar_is_detected_and_a_halt_is_not_called_a_limit() -> None:
    panel = synthetic_panel()
    dates = panel["close"].index
    locked, halted = dates[40], dates[41]
    for field in ("high", "low", "close"):
        panel[field].loc[locked, "S03"] = float(panel["close"].loc[dates[39], "S03"]) * 1.1
        panel[field].loc[halted, "S04"] = float(panel["close"].loc[dates[40], "S04"]) * 1.1
    panel["volume"].loc[halted, "S04"] = 0.0

    limit_up, limit_down = sealed_bar_limits(panel)
    assert bool(limit_up.loc[locked, "S03"])
    assert not bool(limit_down.loc[locked, "S03"])
    assert not bool(limit_up.loc[halted, "S04"]), "zero-volume seal is a halt, not a limit"
    assert int(limit_up.sum().sum()) == 1


def test_eligibility_is_decided_only_from_prior_sessions() -> None:
    panel = synthetic_panel()
    rules = {"trailing_amount_window": 20, "minimum_trailing_median_amount": 1e6, "minimum_prior_observations": 30}
    full = point_in_time_eligibility(panel, **rules)
    cut = 120
    truncated = point_in_time_eligibility(
        {key: value.iloc[:cut] for key, value in panel.items()}, **rules
    )
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)


def test_eligibility_excludes_special_treatment_and_abnormal_status() -> None:
    panel = synthetic_panel()
    dates = panel["close"].index
    panel["is_st"].loc[dates[50], "S01"] = True
    panel["trade_status"] = pd.DataFrame(1.0, index=dates, columns=SYMBOLS)
    panel["trade_status"].loc[dates[50], "S02"] = 0.0
    eligible = point_in_time_eligibility(panel, minimum_prior_observations=30)
    assert not bool(eligible.loc[dates[50], "S01"])
    assert not bool(eligible.loc[dates[50], "S02"])
    assert bool(eligible.loc[dates[50], "S03"])


def test_select_names_is_deterministic_under_ties_and_column_order() -> None:
    scores = pd.Series({"B": 1.0, "A": 1.0, "D": 0.5, "C": 1.0})
    assert select_names(scores, 3) == ["A", "B", "C"]
    assert select_names(scores.iloc[::-1], 3) == ["A", "B", "C"]


def test_long_only_book_is_fully_invested_and_holds_exactly_book_size() -> None:
    panel = synthetic_panel()
    signal = panel["returns"].rolling(10).mean()
    weights = long_only_weights(signal, panel, CONFIG)
    active = weights.loc[weights.abs().sum(axis=1) > 0]
    assert len(active) > 50
    assert np.allclose(active.sum(axis=1), 1.0)
    assert set(active.gt(0).sum(axis=1).unique()) == {CONFIG["book_size"]}


def test_long_only_and_neutral_books_are_charged_by_the_same_engine() -> None:
    panel = synthetic_panel()
    signal = panel["returns"].rolling(10).mean()
    one_day = panel["returns"].shift(-1)
    for builder in (
        lambda cost: long_only_book(signal, one_day, panel, CONFIG, round_trip_cost=cost),
        lambda cost: factor_portfolio(signal, one_day, panel, CONFIG, round_trip_cost=cost),
    ):
        free, charged = builder(0.0), builder(0.003)
        implied = (free["costed"] - charged["costed"]) / free["turnover"].replace(0.0, np.nan)
        assert np.allclose(implied.dropna(), 0.003 / 2.0)


def test_freeze_refuses_to_overwrite_and_load_detects_tampering(tmp_path) -> None:
    path = tmp_path / "spec.json"
    frozen = freeze_spec(spec_fixture(), path)
    assert frozen["spec_sha256"] == spec_digest(frozen)
    assert load_spec(path)["name"] == "unit_test_spec"

    with pytest.raises(FileExistsError):
        freeze_spec(spec_fixture(), path)

    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["book_size"] = 3
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="modified after freezing"):
        load_spec(path)


def test_validate_spec_rejects_a_factor_the_dsl_cannot_evaluate() -> None:
    spec = spec_fixture()
    spec["factors"] = {"peek": {"field": "future_return"}}
    with pytest.raises(ValueError, match="peek"):
        validate_spec(spec, {"returns", "close"}, {20})


def test_prediction_log_is_idempotent_and_carries_no_orders(tmp_path) -> None:
    panel = synthetic_panel()
    spec = freeze_spec(spec_fixture(), tmp_path / "spec.json")
    eligible = point_in_time_eligibility(panel, **spec["universe"])
    entry = build_prediction(spec, panel, eligible)
    assert entry["orders"] == [] and len(entry["picks"]) == spec["book_size"]

    log = tmp_path / "log.jsonl"
    assert append_prediction(entry, log) is True
    assert append_prediction(entry, log) is False
    assert len(read_log(log)) == 1


def test_score_never_counts_an_entry_still_inside_its_holding_window(tmp_path) -> None:
    panel = synthetic_panel()
    spec = freeze_spec(spec_fixture(), tmp_path / "spec.json")
    eligible = point_in_time_eligibility(panel, **spec["universe"])
    sessions = list(panel["close"].index)
    log = tmp_path / "log.jsonl"

    matured_date, pending_date = sessions[100], sessions[-3]
    for date in (matured_date, pending_date):
        append_prediction(build_prediction(spec, panel, eligible, as_of=date), log)

    report = score_log(spec, panel, log, eligible)
    assert report["matured_entries"] == 1
    assert report["pending_entries"] == 1
    assert report["orders"] == []
    assert report["verdict"] == "insufficient_forward_sample"


def test_score_drops_legs_that_were_sealed_at_the_limit_on_entry(tmp_path) -> None:
    panel = synthetic_panel()
    spec = freeze_spec(spec_fixture(), tmp_path / "spec.json")
    eligible = point_in_time_eligibility(panel, **spec["universe"])
    sessions = list(panel["close"].index)
    as_of = sessions[100]
    log = tmp_path / "log.jsonl"
    entry = build_prediction(spec, panel, eligible, as_of=as_of)
    append_prediction(entry, log)

    baseline = score_log(spec, panel, log, eligible)
    assert baseline["dropped_unbuyable_legs"] == 0

    entry_bar, locked = sessions[101], entry["picks"][0]["symbol"]
    price = float(panel["close"].loc[sessions[100], locked]) * 1.1
    for field in ("high", "low", "close"):
        panel[field].loc[entry_bar, locked] = price

    sealed = score_log(spec, panel, log, eligible)
    assert sealed["dropped_unbuyable_legs"] == 1
    assert sealed["matured_entries"] == 1
