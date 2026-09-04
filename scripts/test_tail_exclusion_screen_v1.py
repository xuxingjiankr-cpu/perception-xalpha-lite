#!/usr/bin/env python3
"""Focused tests for the tail-loss exclusion screen.

The horizon frontier was misleading until the measurement was made market
neutral, so the invariant that matters most here is that a market-wide move
cannot move any reported number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_tail_exclusion_screen_v1 as screen  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def fixture_config() -> dict:
    return json.loads(
        (
            ROOT / "configs" / "research" / "tail_exclusion_screen_v1.json"
        ).read_text(encoding="utf-8")
    )


def synthetic_cross_section(seed: int = 20260905):
    """Bottom-ranked names really are more likely to suffer a severe loss."""
    dates = pd.bdate_range("2025-01-02", periods=160)
    symbols = [f"S{i:03d}" for i in range(50)]
    rng = np.random.default_rng(seed)
    # A stable per-symbol score, so bucket membership does not churn.
    base_score = pd.Series(np.linspace(1.0, 0.0, len(symbols)), index=symbols)
    score = pd.DataFrame(
        np.tile(base_score.to_numpy(), (len(dates), 1)), index=dates, columns=symbols
    )
    severe_probability = pd.Series(
        np.where(base_score.to_numpy() <= 0.1, 0.35, 0.05), index=symbols
    )
    draws = rng.random((len(dates), len(symbols)))
    outcome = pd.DataFrame(
        np.where(draws < severe_probability.to_numpy(), -0.06, 0.01),
        index=dates,
        columns=symbols,
    )
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    return score, outcome, eligible, dates


def main() -> int:
    config = fixture_config()
    check(
        "config is permanently research-only",
        config["status"] == "research_only_shadow_only_not_trading",
    )
    check(
        "all mutation permissions are false",
        all(not v for k, v in config["safety"].items() if k.startswith("may")),
    )
    check(
        "a historical screen can never promote",
        config["preregisteredHypothesis"]["historicalRunCanPromote"] is False,
    )
    screen.validate_config(config)
    check("the frozen precision base config validates", True)

    lowered = json.loads(json.dumps(config))
    lowered["evaluation"]["minimumDayClusteredT"] = 1.0
    try:
        screen.validate_config(lowered)
        raised = False
    except ValueError:
        raised = True
    check("the significance bar cannot be lowered below t=2", raised)

    not_neutral = json.loads(json.dumps(config))
    not_neutral["evaluation"]["excessMeasuredAgainstSameDayUniverse"] = False
    try:
        screen.validate_config(not_neutral)
        raised = False
    except ValueError:
        raised = True
    check("the screen cannot be measured against anything but the same day", raised)

    score, outcome, eligible, dates = synthetic_cross_section()
    rows = screen.bucket_diagnostics(score, outcome, eligible, dates, -0.03, 10)
    check("every bucket is reported", len(rows) == 10)
    worst = rows[-1]
    best = rows[0]
    check(
        "the worst bucket carries a higher severe-loss rate than the universe",
        worst["excessSevereLossRate"] > 0.05,
    )
    check(
        "the worst bucket's excess is strongly significant when the signal is real",
        worst["excessSevereLossT"] > 5.0,
    )
    check(
        "the best bucket carries a lower severe-loss rate than the universe",
        best["excessSevereLossRate"] < 0.0,
    )
    check(
        "buckets are ordered worst-last",
        worst["severeLossRate"] > best["severeLossRate"],
    )

    # The lesson from the horizon frontier: a market-wide move must not move any
    # reported excess. Adding a different constant to every stock on each day
    # leaves the day-neutral return excess exactly unchanged.
    market = pd.Series(
        np.linspace(-0.05, 0.05, len(dates)), index=dates
    )
    shocked = outcome.add(market, axis=0)
    shocked_rows = screen.bucket_diagnostics(score, shocked, eligible, dates, -0.03, 10)
    check(
        "a market-wide move leaves the return excess exactly unchanged",
        all(
            abs(a["meanExcessReturn"] - b["meanExcessReturn"]) < 1e-12
            for a, b in zip(rows, shocked_rows)
        ),
    )

    flat = pd.DataFrame(0.01, index=dates, columns=score.columns)
    flat_rows = screen.bucket_diagnostics(score, flat, eligible, dates, -0.03, 10)
    check(
        "with no dispersion every bucket reports zero excess",
        all(abs(row["meanExcessReturn"]) < 1e-12 for row in flat_rows),
    )
    check(
        "with no severe losses the excess severe-loss rate is zero",
        all(abs(row["excessSevereLossRate"]) < 1e-12 for row in flat_rows),
    )

    rng = np.random.default_rng(7)
    noise = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(len(dates), len(score.columns))),
        index=dates,
        columns=score.columns,
    )
    noise_rows = screen.bucket_diagnostics(score, noise, eligible, dates, -0.03, 10)
    check(
        "an unrelated score produces no significant worst-bucket excess",
        abs(noise_rows[-1]["excessSevereLossT"]) < 3.0,
    )

    def cell(period: str, bucket: int, rate: float, t: float) -> dict:
        return {
            "period": period,
            "book": "frozen_prior",
            "holdingTradingDays": 1,
            "bucket": bucket,
            "excessSevereLossRate": rate,
            "excessSevereLossT": t,
            "meanExcessReturn": -0.001,
        }

    both = {"cells": [cell("validation", 10, 0.05, 4.0), cell("shadow", 10, 0.04, 3.0)]}
    check(
        "a worst bucket that is stable across both windows is retained",
        screen.build_verdict(both, config)["decision"]
        == "retain_tail_exclusion_screen_for_fresh_forward_only",
    )
    check(
        "even a pass is never eligible for trading",
        screen.build_verdict(both, config)["eligibleForTrading"] is False,
    )
    one_window = {"cells": [cell("validation", 10, 0.05, 4.0), cell("shadow", 10, 0.04, 1.2)]}
    check(
        "failing the t bar on shadow alone is rejected",
        screen.build_verdict(one_window, config)["historicalHypothesisPass"] is False,
    )
    too_small = {"cells": [cell("validation", 10, 0.002, 4.0), cell("shadow", 10, 0.004, 3.0)]}
    check(
        "a statistically clean but immaterial excess is rejected",
        screen.build_verdict(too_small, config)["historicalHypothesisPass"] is False,
    )
    wrong_bucket = {"cells": [cell("validation", 1, 0.05, 4.0), cell("shadow", 1, 0.04, 3.0)]}
    check(
        "only the worst bucket can carry the verdict",
        screen.build_verdict(wrong_bucket, config)["historicalHypothesisPass"] is False,
    )
    missing = {"cells": [cell("validation", 10, 0.05, 4.0)]}
    check(
        "a missing shadow cell cannot pass by default",
        screen.build_verdict(missing, config)["historicalHypothesisPass"] is False,
    )

    source = (
        ROOT / "scripts" / "research_tail_exclusion_screen_v1.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = ("submitorder", "cancelorder", "build_decision(", "latest_strategy_overlay")
    check(
        "source has no trading mutation path",
        not any(term in source for term in forbidden),
    )
    print("ALL TAIL EXCLUSION SCREEN TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
