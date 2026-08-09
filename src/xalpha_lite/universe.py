"""Point-in-time universe membership and price-limit detection.

Two failure modes this module exists to prevent, both measured on a real equity panel:

*Sealed bars.* On a market with daily price limits, a session that opens, trades and closes
at one single price is a locked board — the quote exists but nobody could take the other
side. Priced as an ordinary fill it contributes +6.05% per leg against +0.38% for legs that
were actually executable. ``build_panel`` accepts ``limit_up``/``limit_down`` columns and
fills them ``False`` when they are absent, so a panel loaded from plain OHLCV silently
backtests trades that could not happen. :func:`sealed_bar_limits` derives them from the bars.

*Survivorship in the membership rule.* Deciding today's eligible universe from a liquidity
statistic that includes today leaks the present into the past. Every rule here is computed
from data strictly before the session it admits.
"""

from __future__ import annotations

import pandas as pd


def sealed_bar_limits(
    panel: dict[str, pd.DataFrame], *, require_volume: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer limit-up and limit-down sessions from the bars themselves.

    A sealed bar is ``high == low``: the whole session printed at one price. Direction comes
    from that price against the previous close — above is a locked bid, below a locked offer.
    Boards differ (10%, 20%, 30%), so no percentage threshold is assumed; the seal is the
    evidence. With ``require_volume`` a zero-volume session is treated as a halt rather than
    a limit, since nothing traded at all.

    Returns ``(limit_up, limit_down)`` aligned to ``panel["close"]``.
    """
    close = panel["close"]
    high, low = panel["high"].reindex_like(close), panel["low"].reindex_like(close)
    sealed = high.notna() & low.notna() & high.eq(low)
    if require_volume and "volume" in panel:
        sealed &= panel["volume"].reindex_like(close).fillna(0.0).gt(0)
    previous = close.shift(1)
    limit_up = (sealed & close.gt(previous)).fillna(False)
    limit_down = (sealed & close.lt(previous)).fillna(False)
    return limit_up.astype(bool), limit_down.astype(bool)


def apply_sealed_bar_limits(
    panel: dict[str, pd.DataFrame], *, overwrite: bool = False
) -> dict[str, pd.DataFrame]:
    """Fill ``limit_up``/``limit_down`` from the bars when the loader did not supply them.

    ``build_panel`` defaults both flags to ``False`` when the source has no limit columns,
    which makes every locked board look tradeable. Call this on any panel built from plain
    OHLCV. Existing flags are kept unless ``overwrite`` is set, so an authoritative exchange
    feed always wins over inference.
    """
    inferred_up, inferred_down = sealed_bar_limits(panel)
    for field, inferred in (("limit_up", inferred_up), ("limit_down", inferred_down)):
        existing = panel.get(field)
        if overwrite or existing is None or not bool(existing.to_numpy().any()):
            panel[field] = inferred
    return panel


def point_in_time_eligibility(
    panel: dict[str, pd.DataFrame],
    *,
    trailing_amount_window: int = 60,
    minimum_trailing_median_amount: float = 0.0,
    minimum_prior_observations: int = 0,
    exclude_st: bool = True,
    require_normal_trade_status: bool = True,
) -> pd.DataFrame:
    """Boolean ``date x symbol`` membership decided only from prior sessions.

    A name is eligible on session ``t`` when, using data up to ``t-1``, its trailing median
    turnover clears the floor and it has enough history to have a trailing statistic at all;
    and, on ``t`` itself, it actually traded and carries no special-treatment or abnormal
    status flag. The liquidity and seasoning tests are shifted; the status tests are not,
    because they are observable at the open.

    Recognised optional panel fields: ``is_st``, ``is_suspended``, ``is_delisted``,
    ``trade_status`` (1 = normal). Missing fields are treated as unrestrictive.
    """
    close, amount = panel["close"], panel["amount"]
    observed = close.notna() & close.gt(0.0)
    eligible = observed.copy()

    if minimum_trailing_median_amount > 0.0:
        window = int(trailing_amount_window)
        trailing = amount.rolling(window, min_periods=max(2, window // 3)).median().shift(1)
        eligible &= trailing.ge(float(minimum_trailing_median_amount))
    if minimum_prior_observations > 0:
        eligible &= observed.cumsum().shift(1).ge(int(minimum_prior_observations))

    if "volume" in panel:
        eligible &= panel["volume"].reindex_like(close).fillna(0.0).gt(0.0)
    eligible &= amount.reindex_like(close).fillna(0.0).gt(0.0)

    if exclude_st and "is_st" in panel:
        eligible &= ~panel["is_st"].reindex_like(close).fillna(False).astype(bool)
    for field in ("is_suspended", "is_delisted"):
        if field in panel:
            eligible &= ~panel[field].reindex_like(close).fillna(False).astype(bool)
    if require_normal_trade_status and "trade_status" in panel:
        status = pd.to_numeric(
            panel["trade_status"].reindex_like(close).stack(future_stack=True), errors="coerce"
        ).unstack()
        eligible &= status.fillna(1.0).eq(1.0)

    return eligible.fillna(False).astype(bool)


def eligibility_summary(eligible: pd.DataFrame) -> dict[str, object]:
    """Per-session membership counts — the churn is itself a finding worth reporting."""
    counts = eligible.sum(axis=1)
    entered = (eligible & ~eligible.shift(1, fill_value=False)).sum(axis=1)
    left = (~eligible & eligible.shift(1, fill_value=False)).sum(axis=1)
    return {
        "sessions": int(len(counts)),
        "median_members": None if not len(counts) else int(counts.median()),
        "minimum_members": None if not len(counts) else int(counts.min()),
        "maximum_members": None if not len(counts) else int(counts.max()),
        "mean_daily_entries": None if not len(entered) else round(float(entered.mean()), 3),
        "mean_daily_exits": None if not len(left) else round(float(left.mean()), 3),
    }
