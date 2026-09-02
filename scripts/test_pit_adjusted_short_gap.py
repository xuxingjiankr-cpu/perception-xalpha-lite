"""Focused causal and fail-closed tests for the PIT short-gap adapter."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import append_ashare_pit_adjusted_mootdx_short_gap as gap


def row(dt: str, close: float, source: str = "mootdx_tdx") -> dict:
    return {
        "dt": dt,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "vol": 100.0,
        "amount": 1000.0,
        "exchange": "SH",
        "stockCode": "600000",
        "securityId": "SH.600000",
        "source": source,
        "isST": 0,
    }


def pit_row(dt: str, raw_close: float, scale: float) -> dict:
    value = row(dt, raw_close * scale, gap.OFFICIAL_SOURCE)
    value.update({"adjustment": gap.ADJUSTMENT, "tradeStatus": 1, "vol": 10000.0})
    return value


def main() -> int:
    raw = [row("2026-08-28", 10.0), row("2026-08-31", 10.1), row("2026-09-01", 10.2)]
    pit = [pit_row("2026-08-28", 10.0, 2.0), pit_row("2026-08-31", 10.1, 2.0)]
    appended, reason, error = gap.plan_symbol_append(
        pit,
        raw,
        "2026-09-01",
        maximum_gap_sessions=2,
        minimum_official_sessions=2,
        maximum_relative_error=1e-10,
        maximum_absolute_raw_return=0.25,
    )
    assert reason is None and len(appended) == 1, (reason, appended)
    assert abs(appended[0]["close"] - 20.4) < 1e-10
    assert appended[0]["dt"] == "2026-09-01"
    assert appended[0]["pointInTimeStatusSource"].startswith("last_baostock")
    assert appended[0]["researchOnly"] is True and appended[0]["tradeStatus"] == 1

    # A future raw row cannot affect the target-date append.
    future = raw + [row("2026-09-02", 99.0)]
    causal, causal_reason, _ = gap.plan_symbol_append(
        pit,
        future,
        "2026-09-01",
        maximum_gap_sessions=2,
        minimum_official_sessions=2,
        maximum_relative_error=1e-10,
        maximum_absolute_raw_return=0.25,
    )
    assert causal_reason is None and causal == appended

    # An unstable vendor/raw scale and a suspicious gap both fail closed.
    unstable = [pit[0], pit_row("2026-08-31", 10.1, 2.1)]
    _, unstable_reason, _ = gap.plan_symbol_append(
        unstable,
        raw,
        "2026-09-01",
        maximum_gap_sessions=2,
        minimum_official_sessions=2,
        maximum_relative_error=1e-4,
        maximum_absolute_raw_return=0.25,
    )
    assert unstable_reason == "unstable_recent_adjustment_scale"

    jump = raw[:2] + [row("2026-09-01", 14.0)]
    _, jump_reason, _ = gap.plan_symbol_append(
        pit,
        jump,
        "2026-09-01",
        maximum_gap_sessions=2,
        minimum_official_sessions=2,
        maximum_relative_error=1e-10,
        maximum_absolute_raw_return=0.25,
    )
    assert jump_reason == "suspected_corporate_action_or_abnormal_raw_return"

    source = (ROOT / "scripts" / "append_ashare_pit_adjusted_mootdx_short_gap.py").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in ("submit_order", "build_decision(", "skillclient", "latest_strategy_overlay"):
        assert forbidden not in source
    print("short-gap adapter tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
