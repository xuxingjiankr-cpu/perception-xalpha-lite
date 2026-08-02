"""Strict loader for the current-master all-A-share research universe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _series_frame(path: Path) -> tuple[pd.DataFrame | None, dict[str, int]]:
    rows = read_jsonl(path)
    audit = {"sourceRows": len(rows), "invalidRows": 0, "duplicateDates": 0}
    if not rows:
        return None, audit
    frame = pd.DataFrame(rows)
    required = {"dt", "open", "high", "low", "close", "vol", "amount"}
    if not required.issubset(frame.columns):
        audit["invalidRows"] = len(frame)
        return None, audit
    frame["date"] = pd.to_datetime(
        frame["dt"].astype(str).str[:10], errors="coerce"
    )
    for column in ("open", "high", "low", "close", "vol", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["date"].notna()
        & frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
        & frame["vol"].ge(0)
        & frame["amount"].ge(0)
    )
    audit["invalidRows"] = int((~valid).sum())
    frame = frame.loc[valid].copy()
    audit["duplicateDates"] = int(frame["date"].duplicated().sum())
    frame = (
        frame.drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )
    return frame, audit


def point_in_time_eligibility(
    panel: dict[str, pd.DataFrame], universe_config: dict[str, Any]
) -> pd.DataFrame:
    """Per-date membership decided from trailing information only.

    The symbol-level filters above use whole-history statistics (median amount, suspension
    and missing-bar fractions) and then apply the verdict to that symbol's ENTIRE history.
    That is lookahead: a stock that only becomes liquid in 2025 is admitted to 2020, and one
    that dries up late is admitted to the years it was liquid on the strength of data that
    had not happened yet. Those filters remain as DATA-VALIDITY gates; tradability is decided
    here, per date, from a trailing window:

      * trailing median amount over `pointInTimeAmountWindow` sessions (shifted, so the
        deciding day's own turnover cannot admit it);
      * at least `pointInTimeMinimumHistory` prior observations, which also stands in for a
        listing-seasoning rule we cannot get from the master;
      * the bar itself traded (positive volume and amount), i.e. not suspended.
    """
    amount = panel["amount"]
    window = int(universe_config.get("pointInTimeAmountWindow", 60))
    seasoning = int(universe_config.get("pointInTimeMinimumHistory", 120))
    floor = float(universe_config.get("pointInTimeMinimumAmount",
                                       universe_config.get("minimumMedianDailyAmountCny", 0.0)))
    trailing_amount = amount.rolling(window, min_periods=max(5, window // 3)).median().shift(1)
    observed = panel["close"].notna() & panel["close"].gt(0)
    seasoned = observed.cumsum().shift(1).ge(seasoning)
    traded = panel["volume"].fillna(0.0).gt(0.0) & amount.fillna(0.0).gt(0.0)
    return (trailing_amount.ge(floor) & seasoned & observed & traded).fillna(False)


def build_panel(
    universe_config: dict[str, Any],
    data_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    master_path = ROOT / universe_config["masterPath"]
    bars_root = ROOT / universe_config["barsRoot"]
    master_rows = read_jsonl(master_path)
    master = {
        str(row.get("securityId")): row
        for row in master_rows
        if row.get("securityId")
    }
    minimum_obs = int(data_config["minimumObservationsPerSymbol"])
    minimum_amount = float(data_config["minimumMedianDailyAmountCny"])
    maximum_suspension_fraction = float(
        universe_config["maximumSuspensionFraction"]
    )
    maximum_missing_fraction = float(universe_config["maximumMissingBarFraction"])
    allowed_exchanges = set(universe_config["exchanges"])
    fields: dict[str, dict[str, pd.Series]] = {
        key: {} for key in ("open", "high", "low", "close", "volume", "amount")
    }
    rejection_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    invalid_rows = 0
    duplicate_dates = 0
    accepted_metadata: dict[str, dict[str, Any]] = {}
    matched_files = 0

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for security_id, security in sorted(master.items()):
        exchange = str(security.get("exchange"))
        code = str(security.get("stockCode") or "").zfill(6)
        if exchange not in allowed_exchanges:
            reject("exchange_not_selected")
            continue
        path = bars_root / f"{exchange}_{code}.jsonl"
        if not path.exists():
            reject("bar_file_missing")
            continue
        matched_files += 1
        frame, file_audit = _series_frame(path)
        invalid_rows += file_audit["invalidRows"]
        duplicate_dates += file_audit["duplicateDates"]
        if frame is None or len(frame) < minimum_obs:
            reject("insufficient_observations")
            continue
        median_amount = float(frame["amount"].median())
        if not np.isfinite(median_amount) or median_amount < minimum_amount:
            reject("insufficient_median_amount")
            continue
        active = frame["vol"].gt(0) & frame["amount"].gt(0)
        suspension_fraction = float((~active).mean())
        if suspension_fraction > maximum_suspension_fraction:
            reject("excess_suspension_fraction")
            continue
        expected = len(pd.bdate_range(frame.index.min(), frame.index.max()))
        missing_fraction = 1.0 - len(frame) / max(1, expected)
        if missing_fraction > maximum_missing_fraction:
            reject("excess_missing_bar_fraction")
            continue
        returns = frame["close"].pct_change(fill_method=None)
        extreme_fraction = float(
            returns.abs()
            .gt(float(universe_config["maximumAbsoluteRawDailyReturn"]))
            .mean()
        )
        if extreme_fraction > float(
            universe_config["maximumExtremeReturnFraction"]
        ):
            reject("raw_price_corporate_action_or_anomaly")
            continue
        for field in fields:
            source = "vol" if field == "volume" else field
            fields[field][security_id] = frame[source]
        first = read_jsonl(path)[:1]
        amount_source = (
            str(first[0].get("amountSource")) if first else "unknown"
        )
        source_counts[amount_source] = source_counts.get(amount_source, 0) + 1
        accepted_metadata[security_id] = {
            "exchange": exchange,
            "board": security.get("board"),
            "name": security.get("name"),
            "observations": len(frame),
            "medianDailyAmountCny": round(median_amount, 2),
            "suspensionFraction": round(suspension_fraction, 8),
            "missingBarFraction": round(max(0.0, missing_fraction), 8),
            "amountSource": amount_source,
        }

    panel = {
        field: pd.DataFrame(values).sort_index()
        for field, values in fields.items()
    }
    close = panel["close"]
    panel["vwap"] = (
        panel["amount"] / panel["volume"].replace(0.0, np.nan)
    ).combine_first(close)
    panel["returns"] = close.pct_change(fill_method=None)
    panel["eligible"] = point_in_time_eligibility(panel, universe_config)
    current_coverage = matched_files / len(master) if master else 0.0
    minimum_coverage = float(
        universe_config["minimumMasterCoverageForHistoricalValidation"]
    )
    exchange_counts: dict[str, int] = {}
    board_counts: dict[str, int] = {}
    for metadata in accepted_metadata.values():
        exchange = str(metadata["exchange"])
        board = str(metadata["board"])
        exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
        board_counts[board] = board_counts.get(board, 0) + 1
    audit = {
        "schemaVersion": "ashare_research_panel_audit_v1",
        "status": "diagnostic_only_research_only",
        "universeKind": "current_discoverable_all_a_shares",
        "masterPath": str(master_path),
        "barsRoot": str(bars_root),
        "masterCount": len(master),
        "matchedBarFiles": matched_files,
        "masterCoverage": round(current_coverage, 8),
        "minimumRequiredCoverage": minimum_coverage,
        "acceptedSymbols": len(accepted_metadata),
        "acceptedByExchange": exchange_counts,
        "acceptedByBoard": board_counts,
        "amountSourceCounts": source_counts,
        "rejectionCounts": rejection_counts,
        "invalidRows": invalid_rows,
        "duplicateDates": duplicate_dates,
        "historicalValidationEligible": bool(
            current_coverage >= minimum_coverage
            and len(accepted_metadata)
            >= int(universe_config["minimumEligibleSymbols"])
            and all(exchange_counts.get(exchange, 0) > 0 for exchange in allowed_exchanges)
        ),
        "pointInTimeMembership": False,
        "survivorshipWarning": (
            "The master represents currently discoverable securities. Delisted "
            "stocks and historical ST membership are incomplete."
        ),
        "noForwardFill": True,
        "rawPricesUnadjusted": True,
        "orders": [],
        "automaticTradingChanges": [],
    }
    return panel, audit
