"""Point-in-time fundamental features for all-A-share offline research.

The source files are appendable per-security snapshots produced by
``collect_ashare_fundamentals.py``.  A statement becomes observable only on the
first market date *after* the later of NOTICE_DATE and UPDATE_DATE.  This is
deliberately conservative: a close-to-close research signal must not see a
statement on (or before) its public release date.

This module has no trading imports and only augments an in-memory research panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_FIELDS = {
    "fund_eps_ytd": "epsYtd",
    "fund_book_value_per_share": "bookValuePerShare",
    "fund_revenue_yoy": "revenueYoyPct",
    "fund_net_profit_yoy": "netProfitYoyPct",
    "fund_roe": "roePct",
    "fund_roic": "roicPct",
    "fund_gross_margin": "grossMarginPct",
    "fund_net_margin": "netMarginPct",
    "fund_debt_asset_ratio": "debtAssetRatioPct",
    "fund_current_ratio": "currentRatio",
    "fund_quick_ratio": "quickRatio",
    "fund_cash_to_revenue": "operatingCashToRevenue",
    "fund_cash_to_profit": "operatingCashToNetProfit",
    "fund_receivable_turnover_days": "receivableTurnoverDays",
    "fund_inventory_turnover_days": "inventoryTurnoverDays",
    "fund_total_asset_turnover": "totalAssetTurnover",
}

DERIVED_FIELDS = (
    "fund_book_to_price",
    "fund_annualized_earnings_yield",
    "fund_quality_composite",
    "fund_growth_composite",
    "fund_balance_sheet_safety",
    "fund_accrual_quality",
)

ALL_FIELDS = tuple(RAW_FIELDS) + DERIVED_FIELDS


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


def _safe_date(value: Any) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def _annualization_multiplier(report_date: pd.Timestamp) -> float:
    month = int(report_date.month)
    if month <= 3:
        return 4.0
    if month <= 6:
        return 2.0
    if month <= 9:
        return 4.0 / 3.0
    return 1.0


def _point_in_time_series(
    rows: list[dict[str, Any]],
    value_key: str,
    market_index: pd.DatetimeIndex,
) -> pd.Series:
    observations: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        notice = _safe_date(row.get("noticeDate"))
        update = _safe_date(row.get("updateDate"))
        report = _safe_date(row.get("reportDate"))
        if pd.isna(notice) or pd.isna(report):
            continue
        available = notice if pd.isna(update) else max(notice, update)
        position = market_index.searchsorted(available.normalize(), side="right")
        if position >= len(market_index):
            continue
        value = pd.to_numeric(row.get(value_key), errors="coerce")
        if pd.isna(value):
            continue
        observations.append((market_index[position], float(value)))
    if not observations:
        return pd.Series(index=market_index, dtype=float)
    # Later API versions win only from their own conservative availability date.
    sparse = pd.Series(
        {date: value for date, value in sorted(observations, key=lambda item: item[0])},
        dtype=float,
    )
    return sparse.reindex(market_index).ffill()


def _annualized_eps_series(
    rows: list[dict[str, Any]], market_index: pd.DatetimeIndex
) -> pd.Series:
    annualized: list[dict[str, Any]] = []
    for row in rows:
        report = _safe_date(row.get("reportDate"))
        eps = pd.to_numeric(row.get("epsYtd"), errors="coerce")
        if pd.isna(report) or pd.isna(eps):
            continue
        item = dict(row)
        item["annualizedEps"] = float(eps) * _annualization_multiplier(report)
        annualized.append(item)
    return _point_in_time_series(annualized, "annualizedEps", market_index)


def _cross_section_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).clip(-5.0, 5.0)


def attach_point_in_time_fundamentals(
    panel: dict[str, pd.DataFrame],
    fundamental_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return a copy of ``panel`` augmented with conservative PIT fundamentals."""
    close = panel["close"]
    root = ROOT / fundamental_config["root"]
    output = {key: value.copy() for key, value in panel.items()}
    raw_values: dict[str, dict[str, pd.Series]] = {
        field: {} for field in RAW_FIELDS
    }
    annualized_eps: dict[str, pd.Series] = {}
    available_symbols = 0
    valid_statement_rows = 0
    missing_notice_rows = 0
    for security_id in close.columns:
        exchange, code = str(security_id).split(".", 1)
        rows = read_jsonl(root / f"{exchange}_{code}.jsonl")
        if not rows:
            continue
        available_symbols += 1
        valid_statement_rows += sum(bool(row.get("noticeDate")) for row in rows)
        missing_notice_rows += sum(not bool(row.get("noticeDate")) for row in rows)
        for field, source in RAW_FIELDS.items():
            raw_values[field][security_id] = _point_in_time_series(
                rows, source, close.index
            )
        annualized_eps[security_id] = _annualized_eps_series(rows, close.index)

    for field, values in raw_values.items():
        output[field] = pd.DataFrame(values, index=close.index).reindex(
            index=close.index, columns=close.columns
        )
    annualized_eps_frame = pd.DataFrame(
        annualized_eps, index=close.index
    ).reindex(index=close.index, columns=close.columns)
    bps = output["fund_book_value_per_share"].where(
        output["fund_book_value_per_share"].gt(0.0)
    )
    output["fund_book_to_price"] = bps / close.replace(0.0, np.nan)
    output["fund_annualized_earnings_yield"] = (
        annualized_eps_frame / close.replace(0.0, np.nan)
    )
    quality_parts = [
        _cross_section_zscore(output["fund_roe"]),
        _cross_section_zscore(output["fund_roic"]),
        _cross_section_zscore(output["fund_gross_margin"]),
        _cross_section_zscore(output["fund_cash_to_profit"]),
    ]
    output["fund_quality_composite"] = sum(quality_parts) / len(quality_parts)
    output["fund_growth_composite"] = (
        _cross_section_zscore(output["fund_revenue_yoy"])
        + _cross_section_zscore(output["fund_net_profit_yoy"])
    ) / 2.0
    output["fund_balance_sheet_safety"] = (
        -_cross_section_zscore(output["fund_debt_asset_ratio"])
        + _cross_section_zscore(output["fund_current_ratio"])
        + _cross_section_zscore(output["fund_quick_ratio"])
    ) / 3.0
    output["fund_accrual_quality"] = _cross_section_zscore(
        output["fund_cash_to_profit"]
    ) - _cross_section_zscore(output["fund_net_margin"])

    minimum_symbol_coverage = float(
        fundamental_config.get("minimumSymbolCoverage", 0.6)
    )
    minimum_latest_feature_coverage = float(
        fundamental_config.get("minimumLatestFeatureCoverage", 0.35)
    )
    symbol_coverage = available_symbols / max(1, len(close.columns))
    latest_coverages = {
        field: round(float(output[field].iloc[-1].notna().mean()), 8)
        for field in ALL_FIELDS
    }
    core_fields = fundamental_config.get(
        "coreFields",
        ["fund_roe", "fund_book_to_price", "fund_revenue_yoy"],
    )
    core_coverage = min((latest_coverages.get(field, 0.0) for field in core_fields), default=0.0)
    eligible = bool(
        symbol_coverage >= minimum_symbol_coverage
        and core_coverage >= minimum_latest_feature_coverage
        and missing_notice_rows == 0
    )
    audit = {
        "schemaVersion": "ashare_fundamental_pit_audit_v1",
        "status": "diagnostic_only_research_only",
        "root": str(root),
        "pricePanelSymbols": len(close.columns),
        "symbolsWithFundamentals": available_symbols,
        "symbolCoverage": round(symbol_coverage, 8),
        "minimumSymbolCoverage": minimum_symbol_coverage,
        "validStatementRows": valid_statement_rows,
        "missingNoticeDateRows": missing_notice_rows,
        "latestFeatureCoverage": latest_coverages,
        "minimumLatestCoreFeatureCoverage": minimum_latest_feature_coverage,
        "historicalValidationEligible": eligible,
        "availabilityRule": "first market date strictly after max(noticeDate, updateDate)",
        "restatementLimitation": (
            "The public endpoint exposes the currently retrievable statement version; "
            "historical restatement vintages may be incomplete. Using UPDATE_DATE is "
            "conservative but cannot reconstruct unavailable old vintages."
        ),
        "pointInTimeMembership": False,
        "survivorshipWarning": (
            "Current discoverable master omits some delisted securities and historical "
            "ST membership; fundamental results retain survivorship bias."
        ),
        "orders": [],
        "automaticTradingChanges": [],
    }
    return output, audit

