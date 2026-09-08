"""Opt-in research price-basis contract. Never call the legacy VWAP builder.

An adjusted transaction VWAP can only be reconstructed from a date-local pair
of official adjusted and unadjusted bars. An OHLC4 proxy is NOT a transaction
VWAP. Missing, substituted or incompatible inputs stay missing.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "baostock_query_history_k_data_plus"
PRICE_RTOL = 1e-5  # Fixed engineering tolerance, not estimated from returns.
FLOW_RTOL = 1e-8
STRICT_HISTORY_SESSIONS = 252  # Conservative, frozen; not selected on outcomes.
OHLC = ("open", "high", "low", "close")


class ContractError(ValueError):
    pass


def identity(row):
    sid, dt = row.get("securityId", ""), row.get("dt", "")
    if not re.fullmatch(r"(?:SH|SZ)\.\d{6}", sid):
        raise ContractError("invalid_security_id")
    try:
        if date.fromisoformat(dt).isoformat() != dt:
            raise ValueError(dt)
    except (TypeError, ValueError):
        raise ContractError("invalid_session_date") from None
    return dt, sid


def official(row, flag):
    identity(row)
    if (row.get("source") != SOURCE or row.get("adjustflag") != flag
            or row.get("pointInTimeStatus") is not True
            or row.get("pointInTimeStatusSource") not in (None, "")):
        raise ContractError("unverified_source_adjustment_or_status")
    if (row.get("volumeUnit") != "shares" or row.get("amountUnit") != "CNY"
            or row.get("amountSource") != "exchange_reported_via_baostock"):
        raise ContractError("unknown_flow_units_or_source")
    for k in (*OHLC, "vol", "amount"):
        v = row.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ContractError("invalid_or_inactive_numeric_bar")
    if not (row["low"] <= min(row["open"], row["close"])
            <= max(row["open"], row["close"]) <= row["high"]):
        raise ContractError("invalid_ohlc_order")
    if type(row.get("tradeStatus")) is not int or row["tradeStatus"] != 1:
        raise ContractError("inactive_or_unknown_trade_status")
    if type(row.get("isST")) is not int or row["isST"] not in (0, 1):
        raise ContractError("unknown_st_status")
    if flag == "1" and row.get("adjustment") != "backward_adjusted_baostock_pctchg_method":
        raise ContractError("unknown_adjusted_price_basis")
    if flag == "3" and row.get("adjustment") != "none_raw_baostock":
        raise ContractError("unknown_raw_price_basis")


def paired_vwap(adjusted, raw):
    """Same-date multiplicative reconstruction, not a vendor adjustment factor.

    No scale estimation from subsequent dates, forward fill, OHLC4 fallback,
    cross-provider splice, or value taken from the incoming 'vwap' field.
    """
    official(adjusted, "1")
    if raw is None:
        raise ContractError("missing_official_raw_companion")
    official(raw, "3")
    if identity(adjusted) != identity(raw):
        raise ContractError("raw_adjusted_identity_mismatch")
    for key in ("vol", "amount"):
        if not math.isclose(adjusted[key], raw[key], rel_tol=FLOW_RTOL, abs_tol=0):
            raise ContractError("raw_adjusted_flow_mismatch")
    if any(adjusted[k] != raw[k] for k in ("isST", "tradeStatus")):
        raise ContractError("raw_adjusted_status_mismatch")
    scale = adjusted["close"] / raw["close"]
    if any(not math.isclose(adjusted[k] / raw[k], scale, rel_tol=PRICE_RTOL, abs_tol=0) for k in OHLC):
        raise ContractError("nonmultiplicative_ohlc_scale")
    raw_vwap = raw["amount"] / raw["vol"]
    if not raw["low"] * (1 - PRICE_RTOL) <= raw_vwap <= raw["high"] * (1 + PRICE_RTOL):
        raise ContractError("transaction_vwap_outside_raw_range")
    return raw_vwap * scale


def index_rows(rows):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise ContractError(f"duplicate_bar:{key}")
        result[key] = row
    return result


def build_factor_inputs(adjusted_rows, raw_rows, sessions):
    """Explicit input allowlist; future labels and arbitrary extras cannot escape.

    Caller supplies the complete exchange-session grid, not just observed dates.
    Unverified adjusted rows mask ALL inputs; missing raw bars mask VWAP only.
    Eligibility/seasoning still belongs to the PIT membership loader, not here.
    """
    adj, raw = index_rows(adjusted_rows), index_rows(raw_rows)
    sessions = pd.DatetimeIndex(sessions)
    if not sessions.is_unique or not sessions.is_monotonic_increasing or sessions.tz is not None:
        raise ContractError("invalid_session_grid")
    cols = sorted({sid for _, sid in adj})
    fields = (*OHLC, "volume", "amount", "vwap")
    panel = {k: pd.DataFrame(np.nan, index=sessions, columns=cols) for k in fields}
    errors = Counter()
    for key, row in adj.items():
        dt, sid = pd.Timestamp(key[0]), key[1]
        if dt not in sessions:
            continue
        try:
            official(row, "1")
        except ContractError as exc:
            errors[str(exc)] += 1
            continue
        for k in (*OHLC, "volume", "amount"):
            panel[k].at[dt, sid] = row["vol" if k == "volume" else k]
        try:
            panel["vwap"].at[dt, sid] = paired_vwap(row, raw.get(key))
        except ContractError as exc:
            errors[str(exc)] += 1
    panel["returns"] = panel["close"].pct_change(fill_method=None)
    return panel, {"researchOnly": True, "rejected": dict(errors), "proxyFallbackAllowed": False}


def compute_rank_book(adjusted_rows, raw_rows, sessions, eligible, frozen_factors):
    """Real vendored compute() entry, fixed directions/weights, complete support.

    This intentionally does not call precision.build_factor_inputs(), which
    would overwrite the verified adjusted VWAP with raw amount / volume.
    """
    panel, audit = build_factor_inputs(adjusted_rows, raw_rows, sessions)
    close = panel["close"]
    if not eligible.index.equals(close.index) or not eligible.columns.equals(close.columns):
        raise ContractError("eligibility_axis_mismatch")
    keys = [f["factorKey"] for f in frozen_factors]
    weights = np.asarray([f["weight"] for f in frozen_factors], dtype=float)
    if not keys or len(set(keys)) != len(keys) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ContractError("invalid_frozen_factor_book")
    vendor = str(ROOT / "scripts/vendor/vibe_factors")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    values = {}
    valid = eligible.eq(True) & close.notna()
    # Require verified VWAP even when a formula happens not to need it today;
    # model membership must not change with which factor received a zero weight.
    complete = valid & panel["vwap"].notna()
    input_observed = pd.DataFrame(True, index=close.index, columns=close.columns)
    for field in (*OHLC, "volume", "amount", "vwap"):
        input_observed &= panel[field].notna()
    complete &= input_observed.rolling(STRICT_HISTORY_SESSIONS, min_periods=STRICT_HISTORY_SESSIONS).sum().eq(STRICT_HISTORY_SESSIONS)
    for item in frozen_factors:
        key, direction = item["factorKey"], item["direction"]
        if not re.fullmatch(r"(?:gtja191|alpha101|qlib158|academic)/[a-z0-9_]+", key) or direction not in (-1, 1):
            raise ContractError("invalid_frozen_factor_reference")
        module = importlib.import_module("src.factors.zoo." + key.replace("/", "."))
        value = module.compute(panel).reindex_like(close).replace([np.inf, -np.inf], np.nan)
        values[key] = value * direction
        complete &= value.notna()
    # All factor percentiles use the SAME complete decision-time cross-section.
    ranks = {k: value.where(complete).rank(axis=1, pct=True) for k, value in values.items()}
    score = sum(ranks[k] * w for k, w in zip(keys, weights / weights.sum())).where(complete)
    audit.update(factorCount=len(keys), completeObservations=int(complete.to_numpy().sum()),
                 missingFactorImputationAllowed=False, orders=[], mayPromote=False,
                 factorDefinitions="existing_implementations_not_certified_canonical_formulas",
                 requiredContiguousInputSessions=STRICT_HISTORY_SESSIONS,
                 internalFactorMissingDataSemanticsCertified=False)
    return ranks, score, audit


def read_rows(path):
    """Strict JSONL ingestion and content hashing in one pass (no file writes)."""
    path = Path(path)
    before = path.stat()
    h, rows, seen = hashlib.sha256(), [], set()
    with path.open("rb") as stream:
        for n, line in enumerate(stream, 1):
            h.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = identity(row)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ContractError(f"invalid_jsonl:{path.name}:{n}:{exc}") from None
            if key in seen:
                raise ContractError(f"duplicate_bar:{path.name}:{n}")
            if path.stem != key[1].replace(".", "_"):
                raise ContractError(f"filename_identity_mismatch:{path.name}:{n}")
            seen.add(key)
            rows.append(row)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"input_changed_during_read:{path.name}")
    return rows, h.hexdigest()


def audit_prices(adjusted_root, raw_root, start, end, minimum_cross_section):
    """Streaming per-symbol audit: no backfill, no cache, no model fit."""
    files, daily, failures, errors = [], {}, [], Counter()
    count = proxy = outside = official_count = usable = 0
    paths = sorted(Path(adjusted_root).glob("*.jsonl"))
    if not paths:
        raise ContractError("missing_adjusted_bar_files")
    for n, path in enumerate(paths, 1):
        try:
            rows, checksum = read_rows(path)
            raw_path = Path(raw_root) / path.name
            raw_rows, raw_hash = read_rows(raw_path) if raw_path.exists() else ([], None)
            raw = index_rows(raw_rows)
            files.append({"path": str(path), "sha256": checksum,
                          "rawPath": str(raw_path), "rawSha256": raw_hash})
            for row in rows:
                if not start <= row["dt"] <= end:
                    continue
                count += 1
                dt = row["dt"]
                d = daily.setdefault(dt, Counter())
                d["rows"] += 1
                proxy += int("proxy" in str(row.get("vwapSource", "")))
                try:
                    official(row, "1")
                    official_count += 1
                    d["officialActiveRows"] += 1
                    legacy = row["amount"] / row["vol"]
                    outside += int(not row["low"] * (1 - PRICE_RTOL) <= legacy <= row["high"] * (1 + PRICE_RTOL))
                    paired_vwap(row, raw.get(identity(row)))
                    usable += 1
                    d["pairedVwapRows"] += 1
                except ContractError as exc:
                    errors[str(exc)] += 1
        except ContractError as exc:
            failures.append(str(exc))
        if n % 500 == 0:
            print(f"strict_price_audit {n}/{len(paths)}", flush=True)
    return {"researchOnly": True, "files": files, "filesScanned": len(paths),
            "rows": count, "officialActiveRows": official_count, "storedProxyRows": proxy,
            "legacyRawVwapOutsideAdjustedRangeRows": outside, "verifiedPairedVwapRows": usable,
            "rejections": dict(errors), "fileErrors": failures,
            "daysWithMinimumPairedCrossSection": sum(v["pairedVwapRows"] >= minimum_cross_section for v in daily.values()),
            "daily": [{"date": k, **dict(v)} for k, v in sorted(daily.items())],
            "dataRange": [min(daily), max(daily)] if daily else [],
            "modelTrainingReady": False,  # Price audit alone cannot approve PIT/events/execution.
            "limitations": ["SH/SZ files only; no BJ point-in-time membership coverage.",
                            "Raw and adjusted matching does not reconstruct historical vendor revisions.",
                            "Transaction VWAP is an input feature, not an achievable fill price.",
                            "No claim that fixing this contract improves returns or proves an alpha."]}
