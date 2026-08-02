"""Conservative point-in-time alignment for public financial statements."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


META_COLUMNS = {"symbol", "report_date", "notice_date", "update_date"}


def _available_session(
    notice: pd.Timestamp, update: pd.Timestamp, sessions: pd.DatetimeIndex
) -> pd.Timestamp | None:
    available = max(notice.normalize(), update.normalize())
    position = sessions.searchsorted(available, side="right")
    return None if position >= len(sessions) else sessions[position]


def align_point_in_time_fundamentals(
    statements: pd.DataFrame,
    sessions: Iterable[pd.Timestamp],
    symbols: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Align values after disclosure, never from the report-period end date.

    Required long-table columns are ``symbol``, ``report_date``, ``notice_date``
    and ``update_date``. Numeric columns become ``date x symbol`` matrices.
    Rows missing a notice date are rejected. A value first appears on the market
    session strictly after ``max(notice_date, update_date)``.
    """
    required = META_COLUMNS
    if not required.issubset(statements.columns):
        raise ValueError(f"missing columns: {sorted(required - set(statements.columns))}")
    frame = statements.copy()
    for column in ("report_date", "notice_date", "update_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if frame["notice_date"].isna().any():
        raise ValueError("notice_date is mandatory; report_date is not an availability date")
    frame["update_date"] = frame["update_date"].fillna(frame["notice_date"])
    frame = frame.dropna(subset=["report_date", "notice_date", "symbol"])
    market_index = pd.DatetimeIndex(pd.to_datetime(list(sessions))).sort_values().unique()
    symbol_source = frame["symbol"].astype(str) if symbols is None else symbols
    selected_symbols = sorted(set(map(str, symbol_source)))
    value_columns = [
        column
        for column in frame.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    output: dict[str, pd.DataFrame] = {}
    for value_column in value_columns:
        values: dict[str, pd.Series] = {}
        for symbol, group in frame.groupby(frame["symbol"].astype(str)):
            points: dict[pd.Timestamp, float] = {}
            ordered = group.sort_values(["notice_date", "update_date", "report_date"])
            for row in ordered.itertuples(index=False):
                value = pd.to_numeric(getattr(row, value_column), errors="coerce")
                if pd.isna(value):
                    continue
                date = _available_session(row.notice_date, row.update_date, market_index)
                if date is not None:
                    points[date] = float(value)
            values[symbol] = pd.Series(points, dtype=float).reindex(market_index).ffill()
        output[value_column] = pd.DataFrame(values, index=market_index).reindex(
            index=market_index, columns=selected_symbols
        )
        output[value_column] = output[value_column].replace([np.inf, -np.inf], np.nan)
    return output
