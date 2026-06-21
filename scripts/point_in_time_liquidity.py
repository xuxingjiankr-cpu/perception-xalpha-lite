"""Small shared helpers for point-in-time intraday liquidity gates."""

from __future__ import annotations

from typing import Any


SESSION_MINUTES = 240.0


def session_minutes_elapsed(minute: int) -> float:
    if minute <= 9 * 60 + 30:
        return 0.0
    if minute <= 11 * 60 + 30:
        return float(minute - (9 * 60 + 30))
    if minute <= 13 * 60:
        return 120.0
    if minute <= 15 * 60:
        return 120.0 + float(minute - 13 * 60)
    return SESSION_MINUTES


def value_at_or_before(values: dict[int, Any], minute: int) -> float | None:
    chosen: float | None = None
    chosen_minute: int | None = None
    for observed_minute, value in values.items():
        if observed_minute <= minute and (chosen_minute is None or observed_minute > chosen_minute):
            try:
                chosen = float(value)
                chosen_minute = observed_minute
            except (TypeError, ValueError):
                continue
    return chosen


def point_in_time_liquidity_gate(amount_by_min: dict[int, Any], minute: int, full_day_floor: float,
                                 previous_day_amount: float | None = None) -> tuple[bool, str, float]:
    """Use only information known at `minute`; never inspect the same day's final amount."""
    if previous_day_amount is not None and float(previous_day_amount) >= float(full_day_floor):
        return True, "previous_day", float(full_day_floor)
    fraction = max(0.05, min(1.0, session_minutes_elapsed(minute) / SESSION_MINUTES))
    threshold = float(full_day_floor) * fraction
    amount_so_far = value_at_or_before(amount_by_min, minute) or 0.0
    return amount_so_far >= threshold, "point_in_time", threshold
