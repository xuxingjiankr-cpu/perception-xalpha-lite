#!/usr/bin/env python3
"""Focused causal and safety tests for state-conditioned Top10 V3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_twelve_factor_state_conditioned_top10_v3 as state_v3  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def fixture_configs() -> tuple[dict, dict]:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "twelve_factor_state_conditioned_top10_v3.json"
        ).read_text(encoding="utf-8")
    )
    base = state_v3.validate_config(config)
    return config, base


def synthetic_panel() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=80)
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]
    rng = np.random.default_rng(20260829)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.012, size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    close = 10.0 * (1.0 + returns).cumprod()
    amount = pd.DataFrame(
        rng.lognormal(17.0, 0.4, size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    return {"returns": returns, "close": close, "amount": amount, "eligible": eligible}


def synthetic_state_evidence() -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.Series
]:
    dates = pd.bdate_range("2025-01-02", periods=120)
    factors = ["f_good", "f_flat", "f_bad"]
    columns = [
        "market_return_5",
        "market_return_20",
        "market_volatility_20",
        "market_downside_semivariance_20",
        "market_breadth_20",
        "market_return_dispersion_1",
        "market_liquidity_shock_20",
    ]
    regime = np.where(np.arange(len(dates)) % 3 == 0, 0.0, 4.0)
    regime[-1] = 0.0
    states = pd.DataFrame(
        np.column_stack([regime + offset * 0.01 for offset in range(len(columns))]),
        index=dates,
        columns=columns,
    )
    base = pd.DataFrame(index=dates, columns=factors, dtype=float)
    frames = {name: base.copy() for name in state_v3.feedback_v2.METRICS}
    state_good = regime == 0.0
    for date_position, is_good_state in enumerate(state_good):
        if is_good_state:
            values = {
                "top10GrossReturn": [0.025, 0.0, -0.015],
                "top10WinRate": [0.75, 0.5, 0.35],
                "extremeWinnerRate": [0.3, 0.08, 0.01],
                "severeLossAvoidance": [1.0, 0.93, 0.8],
            }
        else:
            values = {
                "top10GrossReturn": [-0.01, 0.0, 0.015],
                "top10WinRate": [0.4, 0.5, 0.65],
                "extremeWinnerRate": [0.02, 0.08, 0.2],
                "severeLossAvoidance": [0.82, 0.93, 0.98],
            }
        for metric, row in values.items():
            frames[metric].iloc[date_position, :] = row
    prior = pd.Series({factor: 0.3333333333333333 for factor in factors})
    return frames, states, prior


def main() -> int:
    config, base_config = fixture_configs()
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

    panel = synthetic_panel()
    original = state_v3.market_state_features(panel)
    changed_panel = {key: value.copy() for key, value in panel.items()}
    changed_panel["returns"].iloc[-10:, :] = 0.25
    changed_panel["close"].iloc[-10:, :] *= 3.0
    changed_panel["amount"].iloc[-10:, :] *= 100.0
    changed = state_v3.market_state_features(changed_panel)
    check(
        "future panel mutations cannot change earlier state features",
        np.allclose(
            original.iloc[:-10].to_numpy(dtype=float),
            changed.iloc[:-10].to_numpy(dtype=float),
            equal_nan=True,
            atol=1e-12,
        ),
    )

    frames, states, prior = synthetic_state_evidence()
    weights, audit, _ = state_v3.state_conditioned_weight_path(
        frames, states, prior, base_config, config
    )
    latest = weights.iloc[-1]
    check("similar-state winner receives more weight", latest["f_good"] > prior["f_good"])
    check("similar-state loser receives less weight", latest["f_bad"] < prior["f_bad"])
    check("all factor directions remain positive", bool(weights.gt(0.0).all().all()))
    check(
        "weights remain on the simplex",
        float((weights.sum(axis=1) - 1.0).abs().max()) < 1e-10,
    )
    check(
        "weights obey frozen relative bounds",
        bool(
            weights.ge(prior * config["weighting"]["minimumRelativeWeight"] - 1e-12)
            .all()
            .all()
            and weights.le(
                prior * config["weighting"]["maximumRelativeWeight"] + 1e-12
            )
            .all()
            .all()
        ),
    )
    check(
        "daily L1 changes obey the frozen cap",
        float(weights.diff().abs().sum(axis=1).max())
        <= config["weighting"]["maximumOneUpdateL1Turnover"] + 1e-12,
    )

    unresolved = {key: value.copy() for key, value in frames.items()}
    for frame in unresolved.values():
        frame.iloc[-7:, :] = frame.iloc[-7:, :].to_numpy()[:, ::-1]
    unresolved_weights, _, _ = state_v3.state_conditioned_weight_path(
        unresolved, states, prior, base_config, config
    )
    check(
        "latest seven unresolved outcome rows cannot change current weights",
        np.allclose(weights.iloc[-1], unresolved_weights.iloc[-1], atol=1e-12),
    )

    resolved = {key: value.copy() for key, value in frames.items()}
    for frame in resolved.values():
        frame.iloc[-25:-7, :] = frame.iloc[-25:-7, :].to_numpy()[:, ::-1]
    resolved_weights, _, _ = state_v3.state_conditioned_weight_path(
        resolved, states, prior, base_config, config
    )
    check(
        "resolved similar-state evidence may change current weights",
        not np.allclose(weights.iloc[-1], resolved_weights.iloc[-1], atol=1e-12),
    )
    latest_audit = audit.iloc[-1]
    resolved_end = pd.Timestamp(latest_audit["resolvedHistoryEnd"])
    resolved_position = states.index.get_loc(resolved_end)
    check(
        "state matching ends at least seven sessions before the signal",
        resolved_position <= len(states.index) - 1 - 7,
    )
    check(
        "insufficient early history fails back to frozen prior",
        np.allclose(weights.iloc[0], prior.to_numpy(dtype=float), atol=1e-12)
        and bool(audit.iloc[0]["usedFrozenPriorFallback"]),
    )
    source = (
        ROOT / "scripts" / "research_twelve_factor_state_conditioned_top10_v3.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (
        "submitorder",
        "cancelorder",
        "build_decision(",
        "latest_strategy_overlay",
    )
    check("source has no trading mutation path", not any(term in source for term in forbidden))
    print("ALL STATE-CONDITIONED TOP10 V3 TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
