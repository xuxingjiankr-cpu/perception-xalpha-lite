"""Manufacture an edge out of nothing, then measure exactly how much of it was never there.

Every factor in this example is pure noise. Ground truth is known and it is zero: no factor
has any relationship to future returns, by construction. Yet selecting the "best" ones after
seeing the outcome window produces a confident, tradeable-looking return series.

The point is not that hindsight helps. It is *how much* it helps, and that the result is
indistinguishable from a discovery unless the selection step itself is audited. The gap this
prints is the quantity every backtest reports as alpha when factor choice is not preregistered.

Run: python examples/selection_artifact.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260807
DAYS = 1500
SYMBOLS = 200
FACTORS = 400
DECILE = 0.10
TOP_K = 10
SPLIT = 1000  # first SPLIT days are "trailing"; the rest is the outcome window


def build_world(rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """A market with no predictable structure and factors with no information about it."""
    dates = pd.bdate_range("2020-01-01", periods=DAYS)
    names = [f"S{index:04d}" for index in range(SYMBOLS)]
    forward = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(DAYS, SYMBOLS)), index=dates, columns=names
    )
    factors = {
        f"noise_{index:03d}": pd.DataFrame(
            rng.normal(size=(DAYS, SYMBOLS)), index=dates, columns=names
        )
        for index in range(FACTORS)
    }
    return forward, factors


def decile_excess(signal: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    """Daily excess return of an equal-weight top-decile book over the equal-weight universe."""
    ranks = signal.rank(axis=1, pct=True)
    held = ranks.ge(1.0 - DECILE)
    weights = held.div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return (weights * forward).sum(axis=1) - forward.mean(axis=1)


def annualised_ir(series: pd.Series) -> float:
    deviation = series.std(ddof=1)
    return float(series.mean() / deviation * np.sqrt(244)) if deviation else float("nan")


def main() -> None:
    rng = np.random.default_rng(SEED)
    forward, factors = build_world(rng)
    excess = {name: decile_excess(signal, forward) for name, signal in factors.items()}

    trailing = {name: series.iloc[:SPLIT] for name, series in excess.items()}
    outcome = {name: series.iloc[SPLIT:] for name, series in excess.items()}

    # (a) hindsight: rank on the very window the result is then reported from
    by_hindsight = sorted(outcome, key=lambda name: -outcome[name].mean())[:TOP_K]
    # (b) honest: rank on trailing data only, report the untouched outcome window
    by_trailing = sorted(trailing, key=lambda name: -annualised_ir(trailing[name]))[:TOP_K]

    hindsight = pd.concat([outcome[name] for name in by_hindsight], axis=1).mean(axis=1)
    honest = pd.concat([outcome[name] for name in by_trailing], axis=1).mean(axis=1)

    print(f"{FACTORS} factors, {SYMBOLS} symbols, {DAYS} sessions. True edge of every factor: ZERO.")
    print(f"Reported on the same {len(hindsight)}-day outcome window, top {TOP_K} equal-weighted.\n")
    print(f"{'selection rule':<34}{'mean bps/day':>14}{'annualised IR':>16}")
    print(f"{'chosen on the outcome window':<34}{hindsight.mean() * 1e4:>14.2f}{annualised_ir(hindsight):>16.2f}")
    print(f"{'chosen on trailing data only':<34}{honest.mean() * 1e4:>14.2f}{annualised_ir(honest):>16.2f}")
    print(f"\nartifact = {(hindsight.mean() - honest.mean()) * 1e4:.2f} bps/day of pure selection.")
    print(
        "\nNothing was learned and nothing was predicted. The first row is what a backtest\n"
        "reports when the factor set is chosen after the fact -- which is why this framework\n"
        "preregisters the selection boundary, counts every trial, and deflates the result\n"
        "instead of ranking candidates on the window it will later quote."
    )


if __name__ == "__main__":
    main()
