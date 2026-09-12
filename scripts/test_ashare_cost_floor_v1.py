#!/usr/bin/env python3
"""Tests for the A-share round-trip cost floor.

The claim that makes this study valid is that Corwin-Schultz reads only the
high/low RATIO, so a backward-adjusted panel gives the same answer as an
unadjusted one. That invariance is pinned here, along with the arithmetic of the
exact tick floor, because a cost number that is quietly wrong would silently
rewrite every rejection in RESEARCH_LOG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_ashare_cost_floor_v1 as cost  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def fixture_config() -> dict:
    return json.loads(
        (ROOT / "configs" / "research" / "ashare_cost_floor_v1.json").read_text(
            encoding="utf-8"
        )
    )


def synthetic_panel(days: int = 120, seed: int = 20260913):
    dates = pd.bdate_range("2025-01-02", periods=days)
    symbols = ["CHEAP", "MID", "DEAR"]
    rng = np.random.default_rng(seed)
    levels = {"CHEAP": 5.0, "MID": 20.0, "DEAR": 200.0}
    close = pd.DataFrame(
        {s: levels[s] * (1.0 + rng.normal(0.0, 0.01, days)).cumprod() for s in symbols},
        index=dates,
    )
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=symbols)
    amount = close * volume
    return {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": volume,
        "amount": amount,
        "returns": close.pct_change(fill_method=None),
        "eligible": pd.DataFrame(True, index=dates, columns=symbols),
    }, dates, symbols


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
        "a historical cost study can never promote",
        config["preregisteredHypothesis"]["historicalRunCanPromote"] is False,
    )
    check(
        "fee parameters stay marked as inputs, not findings",
        config["fees"]["confirmBeforeUse"] is True
        and config["preregisteredHypothesis"]["feeParametersAreInputsNotFindings"] is True,
    )
    cost.validate_config(config)
    check("the frozen precision base config validates", True)

    merged = json.loads(json.dumps(config))
    merged["evaluation"]["reportFloorAndEstimateSeparately"] = False
    try:
        cost.validate_config(merged)
        raised = False
    except ValueError:
        raised = True
    check("the exact floor and the estimate may not be merged", raised)

    adjusted_price = json.loads(json.dumps(config))
    adjusted_price["price"]["actualPriceSource"] = "adjusted_close"
    try:
        cost.validate_config(adjusted_price)
        raised = False
    except ValueError:
        raised = True
    check("an adjusted price basis for the tick floor is refused", raised)

    panel, dates, symbols = synthetic_panel()

    price = cost.actual_traded_price(panel)
    check(
        "actual price is exchange cash over shares",
        np.allclose(
            price.to_numpy(), (panel["amount"] / panel["volume"]).to_numpy(), equal_nan=True
        ),
    )
    # The trap this study exists to avoid: an adjusted close is not real yuan.
    adjusted = panel["close"] * 7.678
    check(
        "a backward-adjusted price level would give a different tick floor",
        not np.allclose(
            cost.relative_tick_floor_bps(price, 0.01).to_numpy(),
            cost.relative_tick_floor_bps(adjusted, 0.01).to_numpy(),
            equal_nan=True,
        ),
    )

    # Regression: DataFrame has no .between, and the run path clips prices.
    wild = pd.DataFrame(
        {"OK": [10.0, 20.0], "DUST": [0.001, 0.002], "ABSURD": [9e6, 9e6]},
        index=pd.bdate_range("2025-01-02", periods=2),
    )
    clipped = cost.clip_to_plausible_price(wild, 0.5, 5000.0)
    check(
        "an implausible amount/volume ratio is dropped, not given a tick floor",
        clipped["OK"].notna().all()
        and clipped["DUST"].isna().all()
        and clipped["ABSURD"].isna().all(),
    )

    flat = pd.DataFrame(
        {"A": [10.0] * 5, "B": [20.0] * 5, "C": [200.0] * 5},
        index=pd.bdate_range("2025-01-02", periods=5),
    )
    floor = cost.relative_tick_floor_bps(flat, 0.01)
    check("a 10 CNY name has a 10 bps round-trip tick floor", abs(floor["A"].iloc[0] - 10.0) < 1e-9)
    check("a 20 CNY name has a 5 bps floor", abs(floor["B"].iloc[0] - 5.0) < 1e-9)
    check("a 200 CNY name has a 0.5 bps floor", abs(floor["C"].iloc[0] - 0.5) < 1e-9)
    check(
        "the floor falls as one over price",
        floor["A"].iloc[0] > floor["B"].iloc[0] > floor["C"].iloc[0],
    )

    # The invariance that licenses using the adjusted panel at all.
    window = 20
    plain = cost.corwin_schultz_bps(panel["high"], panel["low"], window)
    factors = pd.Series({"CHEAP": 3.0, "MID": 7.678, "DEAR": 0.25})
    rescaled = cost.corwin_schultz_bps(
        panel["high"].mul(factors, axis=1), panel["low"].mul(factors, axis=1), window
    )
    check(
        "Corwin-Schultz is invariant to a per-symbol price rescaling",
        np.allclose(plain.to_numpy(), rescaled.to_numpy(), equal_nan=True),
    )
    check(
        "and that invariance is what makes the adjusted panel a valid input",
        plain.notna().to_numpy().sum() > 0,
    )

    sealed_high = panel["close"].copy()
    sealed = cost.corwin_schultz_bps(sealed_high, sealed_high, window)
    tail = sealed.iloc[window:]
    check(
        "a zero-range bar implies a zero spread estimate",
        float(np.nanmax(np.abs(tail.to_numpy()))) < 1e-6,
    )
    check(
        "negative raw estimates are floored at zero, never reported as negative cost",
        float(np.nanmin(plain.to_numpy())) >= 0.0,
    )

    spread = pd.DataFrame({"A": [8.0]}, index=[0])
    total = cost.total_round_trip_bps(spread, 2.5, 5.0)
    check(
        "total is spread plus both commissions plus one stamp",
        abs(float(total["A"].iloc[0]) - (8.0 + 5.0 + 5.0)) < 1e-9,
    )
    check(
        "a zero stamp and zero commission leaves the spread alone",
        abs(float(cost.total_round_trip_bps(spread, 0.0, 0.0)["A"].iloc[0]) - 8.0) < 1e-9,
    )

    def sel(total_floor: float, net_floor: float, net_assumed: float) -> dict:
        return {
            "picks": 100,
            "totalUsingExactFloorBps": total_floor,
            "netUsingExactFloorBps": net_floor,
            "netUsingAssumedBps": net_assumed,
        }

    flips = {
        "assumedRoundTripBps": 30.0,
        "selection": {
            "validation": sel(14.0, 4.0, -12.0),
            "shadow": sel(15.0, 3.0, -11.0),
        },
    }
    verdict = cost.build_verdict(flips)
    check(
        "an overstated assumption that flips a sign is reported as such",
        verdict["decision"] == "flat_cost_assumption_overstated_and_a_conclusion_flips",
    )
    check("even a flip is never eligible for trading", verdict["eligibleForTrading"] is False)

    no_flip = {
        "assumedRoundTripBps": 30.0,
        "selection": {"validation": sel(14.0, -6.0, -22.0)},
    }
    check(
        "an overstated assumption that changes nothing is reported honestly",
        cost.build_verdict(no_flip)["decision"]
        == "flat_cost_assumption_overstated_but_no_conclusion_flips",
    )
    understated = {
        "assumedRoundTripBps": 30.0,
        "selection": {"validation": sel(45.0, -35.0, -20.0)},
    }
    check(
        "a measured cost above the assumption is not called overstatement",
        cost.build_verdict(understated)["decision"]
        == "flat_cost_assumption_is_not_overstated",
    )

    source = (
        ROOT / "scripts" / "research_ashare_cost_floor_v1.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = ("submitorder", "cancelorder", "build_decision(", "latest_strategy_overlay")
    check(
        "source has no trading mutation path",
        not any(term in source for term in forbidden),
    )
    print("ALL A-SHARE COST FLOOR TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
