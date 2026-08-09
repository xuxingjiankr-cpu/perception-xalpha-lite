"""Long-only top-N book construction, sharing the cost engine with the neutral book.

``neutral_factor_weights`` builds a dollar-neutral, size- and industry-residualised book:
the right construction for asking whether a signal carries information. It is not the
construction most people actually trade, and a factor can look different in the two. This
module adds the concrete alternative — hold the N best names, equally weighted — and routes
it through the *same* tranche, turnover and cost accounting, so a difference between the two
books is a difference in construction and never a difference in how they were charged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .discovery import _returns_from_desired_weights, _tradability


def selectable_mask(
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Names that can actually be bought on the session following the signal."""
    buyable, _ = _tradability(panel, config)
    mask = buyable.astype(bool)
    if eligible is not None:
        mask &= eligible.reindex_like(mask).fillna(False).astype(bool)
    return mask


def select_names(scores: pd.Series, book_size: int) -> list[str]:
    """Deterministic top-N selection: highest score first, symbol name breaks ties.

    Determinism matters more here than anywhere else in the library. A forward record whose
    picks depend on column order is not a record of anything.
    """
    clean = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    ordered = sorted(clean.items(), key=lambda item: (-float(item[1]), str(item[0])))
    return [str(symbol) for symbol, _ in ordered[: int(book_size)]]


def long_only_weights(
    signal: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Equal-weight ``1/N`` on the top ``book_size`` selectable names each session."""
    book_size = int(config.get("book_size", 10))
    mask = selectable_mask(panel, config, eligible)
    masked = signal.where(mask.reindex_like(signal).fillna(False))
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date in signal.index:
        names = select_names(masked.loc[date], book_size)
        if len(names) < book_size:
            continue
        weights.loc[date, names] = 1.0 / float(len(names))
    return weights


def long_only_book(
    signal: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
    round_trip_cost: float | None = None,
    eligible: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Top-N book returns, charged by the shared tranche and turnover accounting."""
    desired = long_only_weights(signal, panel, config, eligible)
    cost = float(config["round_trip_cost"] if round_trip_cost is None else round_trip_cost)
    result = _returns_from_desired_weights(desired, one_day, panel, config, cost)
    result["desired_weights"] = desired
    return result


def universe_benchmark(
    one_day: pd.DataFrame, eligible: pd.DataFrame | None = None
) -> pd.Series:
    """Equal-weight return of the eligible universe — what the book must beat to be worth it.

    A long-only book carries whatever the market did. Reporting it against cash credits the
    construction for beta; reporting it against the universe it selected from does not.
    """
    values = one_day if eligible is None else one_day.where(
        eligible.reindex_like(one_day).fillna(False)
    )
    return values.mean(axis=1)
