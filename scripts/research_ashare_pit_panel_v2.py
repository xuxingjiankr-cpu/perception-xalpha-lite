"""Opt-in historical research loader; NEVER changes the legacy/frozen loader.

Whole-file liquidity, missingness, length and status cannot remove earlier rows.
Only date-local source validation and lagged rolling membership are admissible.
Historical vendor vintages are not reconstructed by this repair.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import research_ashare_universe as legacy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "baostock_query_history_k_data_plus"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_validity(frame):
    """Unknown/carried-forward status is not equivalent to a verified status."""
    def column(name, default=""):
        return frame[name] if name in frame else pd.Series(default, index=frame.index)
    return (
        column("source").eq(SOURCE)
        & column("adjustment").str.startswith("backward_adjusted_baostock")
        & column("pointInTimeStatus", False).eq(True)
        & column("pointInTimeStatusSource").fillna("").eq("")
        & frame["is_st"].isin([0, 1])
        & frame["trade_status"].isin([0, 1])
    ).fillna(False)


def eligibility(panel, spec):
    """Signal at t close: liquidity/history gates use sessions strictly before t."""
    close, amount = panel["close"], panel["amount"]
    observed = close.notna()
    member = panel["membership"].fillna(False).astype(bool)
    active = panel["trade_status"].eq(1) & panel["volume"].gt(0) & amount.gt(0)
    w = spec["liquidityWindow"]
    prior_member_count = member.shift(1, fill_value=False).rolling(w, min_periods=w).sum()
    prior_observed = observed.shift(1, fill_value=False).rolling(w, min_periods=w).sum()
    missing = 1 - prior_observed / prior_member_count.replace(0, np.nan)
    suspension = ((observed & ~active).shift(1, fill_value=False).rolling(w, min_periods=w).sum()
                  / prior_observed.replace(0, np.nan))
    median = amount.where(active).rolling(w, min_periods=spec["minimumLiquidityObservations"]).median().shift(1)
    seasoned = observed.cumsum().shift(1).ge(spec["minimumPriorObservations"])
    return (member & observed & active & panel["is_st"].eq(0) & seasoned
            & median.ge(spec["minimumAmountCny"])
            & missing.le(spec["maximumMissingFraction"])
            & suspension.le(spec["maximumSuspensionFraction"])).fillna(False)


def build_panel(universe, spec, sessions=None):
    """Fresh numeric content, independent output/cache contract, SH/SZ PIT master."""
    if sessions is None:
        import exchange_calendars as xcals
        sessions = xcals.get_calendar("XSHG", start=spec["startDate"], end=spec["endDate"]).sessions
        if sessions.tz is not None:
            sessions = sessions.tz_localize(None)
    sessions = pd.DatetimeIndex(sessions)
    master_path = ROOT / universe["masterPath"]
    bars = ROOT / universe["barsRoot"]
    master = sorted(legacy.read_jsonl(master_path), key=lambda row: row["securityId"])
    fields = {k: {} for k in ["open", "high", "low", "close", "volume", "amount", "vwap",
                              "preclose", "is_st", "trade_status", "membership"]}
    files, skipped, trusted_counts, masked = [], {}, pd.Series(0, index=sessions), 0
    invalid_rows = duplicates = 0
    for index, row in enumerate(master):
        if row.get("exchange") not in universe["exchanges"]:
            continue
        sid = row["securityId"]
        listing = pd.to_datetime(row.get("listingDate"), errors="coerce")
        if row.get("pointInTimeMembership") is not True or pd.isna(listing):
            skipped[sid] = "missing_pit_listing_metadata"
            continue
        path = bars / f"{row['exchange']}_{row['stockCode']}.jsonl"
        if not path.exists():
            skipped[sid] = "missing_bar_file"
            continue
        before = path.stat()
        frame, audit = legacy._series_frame(path)
        checksum = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"input_changed_while_reading:{path}")
        files.append({"path": str(path.relative_to(ROOT)), "sha256": checksum})
        invalid_rows += audit["invalidRows"]
        duplicates += audit["duplicateDates"]
        if frame is None or frame.empty:
            skipped[sid] = "no_valid_rows"
            continue
        valid = local_validity(frame)
        masked += int((~valid).sum())
        # Retain the symbol even with ZERO trusted rows. Appending bad future data
        # cannot retrospectively remove its previously valid history.
        f = frame.reindex(sessions)
        ok = valid.reindex(sessions, fill_value=False)
        trusted_counts += ok.astype(int)
        for key in fields:
            if key == "membership":
                member = pd.Series(sessions >= listing, index=sessions)
                end = pd.to_datetime(row.get("delistingDate"), errors="coerce")
                if pd.notna(end):
                    member &= sessions <= end
                fields[key][sid] = member
            else:
                source = "vol" if key == "volume" else key
                values = pd.to_numeric(f.get(source, pd.Series(np.nan, index=sessions)), errors="coerce")
                fields[key][sid] = values.where(ok).astype("float32")
        if index % 500 == 0:
            print(f"pit_v2_loaded {index + 1}/{len(master)}", flush=True)
    panel = {k: pd.DataFrame(v, index=sessions).sort_index(axis=1) for k, v in fields.items()}
    if panel["close"].empty:
        raise RuntimeError("no_price_panel")
    panel["returns"] = panel["close"].pct_change(fill_method=None)
    panel["eligible"] = eligibility(panel, spec)
    daily = panel["eligible"].sum(axis=1)
    if daily.max() < spec["minimumCrossSection"]:
        raise RuntimeError(f"insufficient_point_in_time_cross_section:{int(daily.max())}")
    audit = {
        "schemaVersion": "ashare_pit_panel_v2", "researchOnly": True,
        "wholeFileFiltersAffectMembership": False, "legacyLoaderModified": False,
        "unbiasedHistoricalValidationEligible": False,
        "dataRange": [str(sessions[0].date()), str(sessions[-1].date())],
        "symbols": len(panel["close"].columns), "skippedFiles": skipped,
        "maskedUnverifiedSourceOrStatusRows": masked, "invalidOhlcvRows": invalid_rows,
        "duplicateDateRows": duplicates, "masterSha256": digest(master_path),
        "calendarSha256": hashlib.sha256(sessions.asi8.tobytes()).hexdigest(),
        "sourceFiles": files,
        "daily": [{"date": str(d.date()), "trustedRows": int(trusted_counts[d]),
                   "eligible": int(daily[d])} for d in sessions],
        "limitations": [
            "Current vendor file revisions, missing/delisted coverage and historical master vintages remain limitations.",
            "Carried-forward ST/trade status and substituted proxy price sources are masked, not silently accepted.",
            "No BJ PIT coverage; research covers SH/SZ only, not all A-share exchanges.",
            "Adjusted prices are not a cash-dividend total-return index; no live quote freshness claim.",
            "Legacy factor VWAP construction is retained for definition comparability; raw amount/volume versus adjusted OHLC may be scale-inconsistent."]}
    return panel, audit
