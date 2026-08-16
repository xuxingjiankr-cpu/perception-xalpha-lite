"""Fail-closed readiness checks for point-in-time factor research inputs."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
NEUTRALIZATION_COLUMNS = {"industry", "market_cap"}
STATUS_COLUMNS = {"is_st", "is_suspended", "is_delisted"}
LIMIT_COLUMNS = {"limit_up", "limit_down"}
FUNDAMENTAL_COLUMNS = {"symbol", "report_date", "notice_date", "update_date"}
SUSPICIOUS_TOKENS = (
    "future",
    "forward",
    "target",
    "label",
    "next_return",
    "lead_return",
    "realized_return",
    "outcome",
)


def _digest(frame: pd.DataFrame | None) -> str | None:
    if frame is None:
        return None
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def audit_research_data(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit data before discovery and return a machine-readable readiness report.

    The doctor establishes whether the public pipeline can consume the supplied tables without
    violating its explicit contracts. It cannot prove that a vendor preserved every historical
    revision, that a security master is survivorship-free, or that adjusted prices match the
    intended corporate-action convention. Those remain named warnings rather than silent claims.
    """
    config = dict(config or {})
    checks: list[dict[str, Any]] = []

    def record(
        check_id: str,
        status: str,
        severity: str,
        message: str,
        **metrics: Any,
    ) -> None:
        row: dict[str, Any] = {
            "check_id": check_id,
            "status": status,
            "severity": severity,
            "message": message,
        }
        if metrics:
            row["metrics"] = metrics
        checks.append(row)

    price_columns = set(map(str, prices.columns))
    missing_prices = sorted(PRICE_COLUMNS - price_columns)
    record(
        "price_schema",
        "fail" if missing_prices else "pass",
        "blocking",
        "Daily OHLCV and turnover fields must be explicit." if missing_prices
        else "Required daily price fields are present.",
        missing_columns=missing_prices,
    )

    sessions = 0
    symbols = 0
    first_date: str | None = None
    last_date: str | None = None
    if {"date", "symbol"}.issubset(price_columns):
        dates = pd.to_datetime(prices["date"], errors="coerce")
        invalid_dates = int(dates.isna().sum())
        empty_symbols = int(prices["symbol"].isna().sum())
        duplicate_rows = int(
            pd.DataFrame({"date": dates, "symbol": prices["symbol"]})
            .duplicated(["date", "symbol"])
            .sum()
        )
        valid_dates = dates.dropna()
        sessions = int(valid_dates.nunique())
        symbols = int(prices["symbol"].dropna().astype(str).nunique())
        if not valid_dates.empty:
            first_date = str(valid_dates.min().date())
            last_date = str(valid_dates.max().date())
        record(
            "price_keys",
            "fail" if invalid_dates or empty_symbols or duplicate_rows else "pass",
            "blocking",
            "A date-symbol key must be unique and parseable.",
            invalid_dates=invalid_dates,
            empty_symbols=empty_symbols,
            duplicate_date_symbol_rows=duplicate_rows,
        )

    if PRICE_COLUMNS.issubset(price_columns):
        numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
        numeric = prices[numeric_columns].apply(pd.to_numeric, errors="coerce")
        missing_numeric = int(numeric.isna().any(axis=1).sum())
        complete = numeric.dropna()
        invalid_ohlc = 0
        nonpositive_prices = 0
        negative_activity = 0
        if not complete.empty:
            upper = complete[["open", "close", "low"]].max(axis=1)
            lower = complete[["open", "close", "high"]].min(axis=1)
            invalid_ohlc = int(
                (complete["high"].lt(upper) | complete["low"].gt(lower)).sum()
            )
            nonpositive_prices = int((complete[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
            negative_activity = int((complete[["volume", "amount"]] < 0).any(axis=1).sum())
        missing_ratio = missing_numeric / max(1, len(prices))
        blocking_quality = invalid_ohlc + nonpositive_prices + negative_activity > 0 or missing_ratio > 0.10
        record(
            "bar_integrity",
            "fail" if blocking_quality else ("warn" if missing_numeric else "pass"),
            "blocking" if blocking_quality else ("warning" if missing_numeric else "information"),
            "OHLC ordering, positive prices and non-negative activity are checked row by row.",
            incomplete_rows=missing_numeric,
            incomplete_ratio=round(missing_ratio, 6),
            invalid_ohlc_rows=invalid_ohlc,
            nonpositive_price_rows=nonpositive_prices,
            negative_volume_or_amount_rows=negative_activity,
        )

    required_sessions = None
    split_keys = {
        "minimum_train_days",
        "validation_days",
        "shadow_days",
        "purge_days",
    }
    if split_keys.issubset(config):
        required_sessions = (
            int(config["minimum_train_days"])
            + int(config["validation_days"])
            + int(config["shadow_days"])
            + 2 * int(config["purge_days"])
        )
        record(
            "sample_length",
            "fail" if sessions < required_sessions else "pass",
            "blocking",
            "The chronological split must fit before any candidate is evaluated.",
            observed_sessions=sessions,
            required_sessions=required_sessions,
        )
    elif sessions:
        record(
            "sample_length",
            "warn" if sessions < 300 else "pass",
            "warning" if sessions < 300 else "information",
            "No split config was supplied; 300 sessions is only a diagnostic floor, not validation.",
            observed_sessions=sessions,
        )

    require_neutralization = bool(config.get("require_neutralization_data", True))
    missing_neutralization = sorted(NEUTRALIZATION_COLUMNS - price_columns)
    record(
        "neutralization_support",
        "fail" if require_neutralization and missing_neutralization else "pass",
        "blocking" if require_neutralization else "information",
        "Industry and contemporaneous market capitalization support the declared neutral book.",
        required=require_neutralization,
        missing_columns=missing_neutralization,
    )

    missing_status = sorted(STATUS_COLUMNS - price_columns)
    record(
        "security_status_support",
        "fail" if missing_status else "pass",
        "blocking",
        "ST, suspension and delisting flags are required for an executable-support research mask.",
        missing_columns=missing_status,
    )
    missing_limits = sorted(LIMIT_COLUMNS - price_columns)
    record(
        "price_limit_support",
        "fail" if missing_limits else "pass",
        "blocking",
        "Limit-up and limit-down fields are required by discovery; plain quotes must be enriched first.",
        missing_columns=missing_limits,
        inference_utility="xalpha_lite.universe.sealed_bar_limits",
    )

    suspicious = sorted(
        column for column in price_columns
        if any(token in column.lower() for token in SUSPICIOUS_TOKENS)
    )
    record(
        "future_named_columns",
        "warn" if suspicious else "pass",
        "warning" if suspicious else "information",
        "Future-named columns are not proof of leakage, but must remain outside the feature DSL.",
        suspicious_columns=suspicious,
    )

    fundamental_rows = 0 if fundamentals is None else int(len(fundamentals))
    fundamental_features = 0
    if fundamentals is None or fundamentals.empty:
        record(
            "fundamental_availability",
            "warn",
            "warning",
            "No fundamental table was supplied; the run is restricted to market-data mechanisms.",
            rows=fundamental_rows,
        )
    else:
        fundamental_columns = set(map(str, fundamentals.columns))
        missing_fundamentals = sorted(FUNDAMENTAL_COLUMNS - fundamental_columns)
        record(
            "fundamental_schema",
            "fail" if missing_fundamentals else "pass",
            "blocking",
            "Disclosure metadata is part of the value, not optional metadata.",
            missing_columns=missing_fundamentals,
        )
        if FUNDAMENTAL_COLUMNS.issubset(fundamental_columns):
            report_date = pd.to_datetime(fundamentals["report_date"], errors="coerce")
            notice_date = pd.to_datetime(fundamentals["notice_date"], errors="coerce")
            update_date = pd.to_datetime(fundamentals["update_date"], errors="coerce")
            missing_report = int(report_date.isna().sum())
            missing_notice = int(notice_date.isna().sum())
            notice_before_period_end = int((notice_date < report_date).fillna(False).sum())
            update_before_notice = int((update_date < notice_date).fillna(False).sum())
            duplicate_disclosures = int(
                pd.DataFrame(
                    {
                        "symbol": fundamentals["symbol"],
                        "report_date": report_date,
                        "notice_date": notice_date,
                        "update_date": update_date,
                    }
                ).duplicated().sum()
            )
            disclosure_failure = missing_report + missing_notice + notice_before_period_end > 0
            record(
                "disclosure_timing",
                "fail" if disclosure_failure else ("warn" if update_before_notice else "pass"),
                "blocking" if disclosure_failure else ("warning" if update_before_notice else "information"),
                "Values become available only after disclosure, never at report-period end.",
                missing_report_dates=missing_report,
                missing_notice_dates=missing_notice,
                notice_before_report_period_end=notice_before_period_end,
                update_before_notice=update_before_notice,
                duplicate_disclosure_rows=duplicate_disclosures,
            )
        fundamental_features = sum(
            column not in FUNDAMENTAL_COLUMNS
            and pd.api.types.is_numeric_dtype(fundamentals[column])
            for column in fundamentals.columns
        )
        record(
            "fundamental_feature_payload",
            "warn" if fundamental_features == 0 else "pass",
            "warning" if fundamental_features == 0 else "information",
            "Only numeric payload fields are aligned into the research panel.",
            numeric_feature_columns=int(fundamental_features),
        )

    for check_id, message in (
        (
            "survivorship_provenance",
            "A current security master cannot prove historical membership; provide PIT listing and delisting provenance.",
        ),
        (
            "corporate_action_provenance",
            "CSV bars do not encode their adjustment convention; pin vendor, adjustment mode and revision policy externally.",
        ),
    ):
        record(check_id, "warn", "warning", message)

    blocking = [row for row in checks if row["status"] == "fail"]
    warnings = [row for row in checks if row["status"] == "warn"]
    status = "blocked" if blocking else ("ready_with_warnings" if warnings else "ready")
    return {
        "schema_version": "xalpha_data_doctor_v1",
        "status": status,
        "research_status": "research_only_not_trading",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "price_rows": int(len(prices)),
            "sessions": sessions,
            "symbols": symbols,
            "first_date": first_date,
            "last_date": last_date,
            "fundamental_rows": fundamental_rows,
            "fundamental_numeric_features": int(fundamental_features),
            "required_sessions": required_sessions,
            "blocking_checks": len(blocking),
            "warnings": len(warnings),
        },
        "input_digests": {
            "prices_sha256": _digest(prices),
            "fundamentals_sha256": _digest(fundamentals),
        },
        "checks": checks,
        "limitations": [
            "A passing schema audit is not evidence of alpha or profitability.",
            "Vendor revision history and survivorship require external provenance.",
            "Suspicious column names are warnings; causal safety is enforced separately by the DSL.",
        ],
        "orders": [],
    }


__all__ = ["audit_research_data"]
