#!/usr/bin/env python3
"""Focused arithmetic and safety tests for the holding-horizon cost frontier V1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_horizon_cost_frontier_v1 as frontier  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def fixture_config() -> dict:
    return json.loads(
        (
            ROOT / "configs" / "research" / "horizon_cost_frontier_v1.json"
        ).read_text(encoding="utf-8")
    )


def synthetic_books() -> tuple[pd.DatetimeIndex, list[str], pd.Series]:
    dates = pd.bdate_range("2025-01-02", periods=40)
    factors = ["f_a", "f_b", "f_c", "f_d"]
    prior = pd.Series([0.4, 0.3, 0.2, 0.1], index=factors)
    return dates, factors, prior


def synthetic_selection(gross: float, days: int, width: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2025-01-02", periods=days)
    columns = [f"S{i}" for i in range(width)]
    outcome = pd.DataFrame(gross, index=dates, columns=columns)
    mask = pd.DataFrame(True, index=dates, columns=columns)
    return mask, outcome, dates


def main() -> int:
    config = fixture_config()
    check(
        "config is permanently research-only",
        config["status"] == "research_only_shadow_only_not_trading",
    )
    check(
        "all mutation permissions are false",
        all(
            not value
            for key, value in config["safety"].items()
            if key.startswith("may")
        ),
    )
    check(
        "a historical sweep can never promote",
        config["preregisteredHypothesis"]["historicalRunCanPromote"] is False,
    )
    frontier.validate_config(config)
    check("frozen precision base config validates", True)

    dates, factors, prior = synthetic_books()
    books = frontier.weighting_books(dates, factors, prior)
    equal = books["equal_weight"]
    check(
        "equal-weight book gives every factor the same weight",
        float(equal.iloc[0].nunique()) == 1.0,
    )
    check(
        "equal-weight book sums to one",
        abs(float(equal.iloc[0].sum()) - 1.0) < 1e-12,
    )
    check(
        "frozen book reproduces the prior",
        np.allclose(books["frozen_prior"].iloc[0].to_numpy(), prior.to_numpy()),
    )
    single = books["single/f_b"]
    check(
        "single-factor book isolates exactly one factor",
        float(single.iloc[0]["f_b"]) == 1.0
        and float(single.iloc[0].drop("f_b").sum()) == 0.0,
    )
    check(
        "every factor gets its own single-factor book",
        sum(1 for name in books if name.startswith("single/")) == len(factors),
    )

    cost = 0.003
    mask, outcome, sel_dates = synthetic_selection(0.005, 30, 10)
    one = frontier.summarise_book(mask, outcome, sel_dates, 1, cost, -0.03)
    check(
        "net per pick subtracts the whole round trip",
        abs(one["meanNetPerPick"] - (0.005 - cost)) < 1e-12,
    )
    check(
        "at one session net per day equals net per pick",
        abs(one["meanNetPerHoldingDay"] - one["meanNetPerPick"]) < 1e-12,
    )
    ten = frontier.summarise_book(mask, outcome, sel_dates, 10, cost, -0.03)
    check(
        "net per day divides the same round trip across the horizon",
        abs(ten["meanNetPerHoldingDay"] - (0.005 - cost) / 10.0) < 1e-12,
    )
    # What actually falls with the horizon is the per-day COST DRAG, not the net.
    # Holding a CONSTANT gross for longer is strictly worse per day, which is exactly
    # why the sweep has to measure how fast gross really grows with the horizon.
    drag_one = 0.005 / 1.0 - one["meanNetPerHoldingDay"]
    drag_ten = 0.005 / 10.0 - ten["meanNetPerHoldingDay"]
    check(
        "the per-day cost drag falls as one over the horizon",
        abs(drag_one - cost) < 1e-12 and abs(drag_ten - cost / 10.0) < 1e-12,
    )
    check(
        "a constant gross held longer is worse per day, not better",
        ten["meanNetPerHoldingDay"] < one["meanNetPerHoldingDay"],
    )
    mask_acc, outcome_acc, dates_acc = synthetic_selection(0.005 * 10, 30, 10)
    accumulating = frontier.summarise_book(
        mask_acc, outcome_acc, dates_acc, 10, cost, -0.03
    )
    check(
        "only an edge that accumulates with the horizon beats one session per day",
        accumulating["meanNetPerHoldingDay"] > one["meanNetPerHoldingDay"],
    )
    loser = frontier.summarise_book(mask, outcome, sel_dates, 1, 0.01, -0.03)
    check(
        "a gross edge below cost reports a negative net",
        loser["meanNetPerPick"] < 0.0,
    )
    empty = frontier.summarise_book(
        mask.iloc[:0], outcome, pd.DatetimeIndex([]), 1, cost, -0.03
    )
    check("an empty selection reports no edge rather than zero", empty["picks"] == 0
          and empty["meanNetPerPick"] is None)

    long_dates = pd.bdate_range("2025-01-02", periods=200)
    short_window = precision.contained_signal_dates(long_dates, 1, 5)
    long_window = precision.contained_signal_dates(long_dates, 20, 5)
    check(
        "a longer horizon shrinks the contained signal window",
        len(long_window) < len(short_window),
    )
    check(
        "the contained window never reaches the panel edge",
        long_window[-1] < long_dates[-1],
    )

    check(
        "fewer than three day clusters refuses a t statistic",
        frontier.day_clustered_t(pd.Series([0.01, 0.02])) is None,
    )
    check(
        "a constant daily series refuses a t statistic",
        frontier.day_clustered_t(pd.Series([0.01] * 10)) is None,
    )

    def headline(book: str, horizon: int, net: float) -> dict:
        return {"book": book, "holdingTradingDays": horizon, "meanNetPerPick": net}

    losing = {
        "periods": {
            "validation": {"headline": [headline("frozen_prior", 5, -0.001)]},
            "shadow": {"headline": [headline("frozen_prior", 5, -0.002)]},
        }
    }
    check(
        "nothing clearing cost is rejected",
        frontier.build_verdict(losing)["decision"]
        == "reject_all_horizons_no_configuration_clears_cost",
    )
    one_sided = {
        "periods": {
            "validation": {"headline": [headline("frozen_prior", 5, 0.001)]},
            "shadow": {"headline": [headline("frozen_prior", 5, -0.001)]},
        }
    }
    check(
        "clearing cost on validation alone is still rejected",
        frontier.build_verdict(one_sided)["historicalHypothesisPass"] is False,
    )
    both = {
        "periods": {
            "validation": {"headline": [headline("frozen_prior", 5, 0.001)]},
            "shadow": {"headline": [headline("frozen_prior", 5, 0.002)]},
        }
    }
    passed = frontier.build_verdict(both)
    check(
        "clearing cost on both windows is retained for fresh forward only",
        passed["decision"] == "retain_horizon_candidates_for_fresh_forward_only",
    )
    check(
        "even a pass is never eligible for trading",
        passed["eligibleForTrading"] is False and passed["freshForwardRequired"] is True,
    )
    ex_post = {
        "periods": {
            "validation": {"headline": [headline("single/f_a", 5, 0.010)]},
            "shadow": {"headline": [headline("single/f_a", 5, 0.010)]},
        }
    }
    check(
        "an ex-post single factor cannot carry the verdict",
        frontier.build_verdict(ex_post)["historicalHypothesisPass"] is False,
    )

    source = (
        ROOT / "scripts" / "research_horizon_cost_frontier_v1.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (
        "submitorder",
        "cancelorder",
        "build_decision(",
        "latest_strategy_overlay",
    )
    check("source has no trading mutation path", not any(term in source for term in forbidden))
    print("ALL HORIZON COST FRONTIER V1 TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
