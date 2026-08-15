#!/usr/bin/env python3
"""Frozen event-first opening-auction forward shadow study.

This module records two preregistered research arms and their matched controls.
It has no broker, order, position, sizing, risk-gate, overlay, production-config,
observation-pool, or build_decision integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_ashare_fundamentals as fundamentals  # noqa: E402
import research_ashare_universe as ashare  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_pit_fundamental_catalyst_v5 as catalyst  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "stock_event_auction_forward_v1.json"
)
SCHEMA_VERSION = "stock_event_auction_forward_result_v1"
CODE_VERSION = "stock_event_auction_forward_v1_20260816"
ARM_NAMES = ("event_liquidity", "event_auction_confirmed")
COHORT_NAMES = (
    "selected",
    "samePoolPermutation",
    "nonEventLiquidityMatched",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(
            json_safe(row),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    atomic_text(path, text)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_file(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != str(expected).lower():
        raise RuntimeError(
            f"frozen source hash mismatch: {path} {actual} != {expected}"
        )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "stock_event_auction_forward_v1":
        raise ValueError("unexpected event-auction forward schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("event-auction forward study must remain research-only")
    prospective = config["prospective"]
    if any(
        prospective.get(key) is not False
        for key in (
            "parametersMayAdapt",
            "historicalOutcomesMayTune",
            "forwardOutcomesMayTuneThisVersion",
        )
    ):
        raise ValueError("all V1 parameters and thresholds must remain frozen")
    event = config["eventPool"]
    if event.get("availabilityRule") != (
        "first_market_date_strictly_after_max_notice_update"
    ):
        raise ValueError("filing availability rule changed")
    if event.get("signalEventDate") != (
        "immediately_prior_available_market_session"
    ):
        raise ValueError("event signal must predate the trade session")
    if event.get("neverForceSelections") is not True:
        raise ValueError("zero-selection days must remain allowed")
    if int(event.get("maximumSelectionsPerArm", 0)) != 10:
        raise ValueError("V1 is preregistered with a Top10 cap")
    auction = config["auction"]
    if auction["armBMarketGate"].get("model") != (
        "none_fixed_mechanism_thresholds"
    ):
        raise ValueError("V1 market gate cannot fit a model")
    if auction["armBStockConfirmation"].get("model") != (
        "none_fixed_mechanism_threshold"
    ):
        raise ValueError("V1 stock confirmation cannot fit a model")
    controls = config["controls"]
    if not all(
        controls.get(key) is True
        for key in (
            "sameDateSameSupportSameCountDeterministicPermutation",
            "sameDateSameAuctionStateNonEventLiquidityMatched",
            "compareEachArmToOwnSupportControl",
        )
    ):
        raise ValueError("both matched controls are mandatory")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(
        bool(value)
        for key, value in config["safety"].items()
        if key.startswith("may")
    ):
        raise ValueError("all trading and production permissions must remain false")
    source = config["sources"]
    verify_file(resolve(source["catalystConfig"]), source["catalystConfigSha256"])
    verify_file(resolve(source["catalystCode"]), source["catalystCodeSha256"])
    verify_file(
        resolve(source["auctionFeatureConfig"]),
        source["auctionFeatureConfigSha256"],
    )


def shanghai_today() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat()


def output_root(config: dict[str, Any]) -> Path:
    return resolve(config["output"]["root"])


def load_price_panel(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    source = config["sources"]
    catalyst_config = load_json(resolve(source["catalystConfig"]))
    catalyst.validate_config(catalyst_config)
    base_path = catalyst.verify_frozen_file(
        catalyst_config, "baseResearchConfig", "baseResearchConfigFileSha256"
    )
    base_config = load_json(base_path)
    perception.validate_config(base_config)
    _, cog_config = perception.load_base_configs(base_config)
    panel, audit = ashare.build_panel(
        base_config["assetUniverse"], cog_config["data"]
    )
    if not audit.get("unbiasedHistoricalValidationEligible", False):
        raise RuntimeError("PIT adjusted price panel failed its unbiased-data gate")
    return panel, audit, catalyst_config


def prior_session(
    index: pd.DatetimeIndex, trade_date: str
) -> pd.Timestamp:
    target = pd.Timestamp(trade_date)
    earlier = index[index < target]
    if earlier.empty:
        raise RuntimeError("no prior market session before requested trade date")
    result = pd.Timestamp(earlier.max()).normalize()
    if (target - result).days > 4:
        raise RuntimeError(
            f"daily panel is stale: prior session {result.date()} for {trade_date}"
        )
    return result


def event_candidates_for_signal_date(
    panel: dict[str, pd.DataFrame],
    catalyst_config: dict[str, Any],
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = panel["close"].columns
    market_index = panel["close"].index[
        panel["close"].index <= signal_date
    ]
    event_config = catalyst_config["fundamentalEvents"]
    fundamental_root = resolve(event_config["root"])
    records: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    files_found = 0
    for security_id in map(str, columns):
        exchange, code = security_id.split(".", 1)
        rows = fundamentals.read_jsonl(
            fundamental_root / f"{exchange}_{code}.jsonl"
        )
        if not rows:
            totals["missing_symbol_file"] += 1
            continue
        files_found += 1
        issuer_records, audit = catalyst.causal_event_records_for_symbol(
            rows, market_index, security_id, event_config
        )
        totals.update(audit)
        for row in issuer_records:
            if pd.Timestamp(row["eventDate"]).normalize() == signal_date:
                records.append(row)
    coverage = files_found / max(len(columns), 1)
    frame = pd.DataFrame.from_records(records)
    threshold = float(event_config["positiveCatalystThreshold"])
    if not frame.empty:
        frame = frame[frame["changeScoreRaw"].gt(threshold)].copy()
        frame = frame.drop_duplicates("securityId", keep="last")
        frame = frame.set_index("securityId", drop=False).sort_index()
    audit = {
        "signalEventDate": signal_date.date().isoformat(),
        "symbolsRequested": len(columns),
        "fundamentalFilesFound": files_found,
        "fundamentalFileCoverage": coverage,
        "positiveEvents": int(len(frame)),
        "processingCounts": dict(totals),
        "availabilityRule": event_config["availabilityRule"],
        "futureOutcomeFieldsRead": 0,
    }
    return frame, audit
def _time_bound(trade_date: str, value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{trade_date}T{value}+08:00")


def read_snapshot_candidate(
    path: Path, trade_date: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        frame = pd.read_csv(
            path,
            compression="gzip",
            dtype={"stockCode": "string"},
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(f"cannot read auction snapshot {path}: {exc}") from exc
    required = {
        "collected_at",
        "trade_date",
        "source_quote_time",
        "stockCode",
        "name",
        "open",
        "prevClose",
        "bidPrice1",
        "askPrice1",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"auction snapshot missing columns: {missing}")
    start = _time_bound(trade_date, config["auction"]["validSourceTimeStart"])
    end = _time_bound(trade_date, config["auction"]["validSourceTimeEnd"])
    collected = pd.to_datetime(frame["collected_at"], errors="coerce", utc=True)
    collected = collected.dt.tz_convert("Asia/Shanghai")
    source_time = pd.to_datetime(
        frame["source_quote_time"], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    if collected.notna().sum() == 0:
        raise RuntimeError("auction snapshot has no valid collection timestamp")
    collected_max = collected.max()
    if not (start <= collected_max <= end):
        raise RuntimeError(
            f"auction snapshot collected outside frozen window: {collected_max}"
        )
    valid_time = (
        source_time.ge(start)
        & source_time.le(end)
        & frame["trade_date"].astype(str).eq(trade_date)
    )
    frame = frame.loc[valid_time].copy()
    frame["stockCode"] = frame["stockCode"].astype(str).str.zfill(6)
    for column in ("open", "prevClose", "bidPrice1", "askPrice1"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("stockCode", keep="last")
    minimum = int(config["auction"]["minimumAllMarketRows"])
    if len(frame) < minimum:
        raise RuntimeError(
            f"auction snapshot has {len(frame)} valid rows; requires {minimum}"
        )
    audit = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "validRows": int(len(frame)),
        "collectionTime": collected_max.isoformat(),
        "minimumRows": minimum,
        "sourceWindow": [start.isoformat(), end.isoformat()],
        "qualityPass": True,
    }
    return frame, audit


def load_auction_snapshot(
    trade_date: str,
    config: dict[str, Any],
    explicit_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if explicit_path is not None:
        return read_snapshot_candidate(explicit_path.resolve(), trade_date, config)
    directory = resolve(config["sources"]["snapshotRoot"]) / trade_date
    paths = sorted(directory.glob("*ashare.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"no A-share snapshot files for {trade_date}")
    valid: list[tuple[pd.Timestamp, pd.DataFrame, dict[str, Any]]] = []
    failures: list[str] = []
    for path in paths:
        try:
            frame, audit = read_snapshot_candidate(path, trade_date, config)
            valid.append((pd.Timestamp(audit["collectionTime"]), frame, audit))
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    if not valid:
        raise RuntimeError("no valid frozen-window snapshot; " + " | ".join(failures))
    _, frame, audit = max(valid, key=lambda item: item[0])
    audit["rejectedSnapshotFiles"] = failures
    return frame, audit


def deterministic_permutation(
    support: pd.DataFrame,
    count: int,
    seed: int,
    trade_date: str,
    arm: str,
) -> list[str]:
    securities = sorted(map(str, support.index))
    if count <= 0 or not securities:
        return []
    material = f"{seed}|{trade_date}|{arm}|same_pool".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = np.random.default_rng(derived)
    indices = generator.choice(
        len(securities), size=min(count, len(securities)), replace=False
    )
    return [securities[int(index)] for index in indices]


def select_by_liquidity(support: pd.DataFrame, maximum: int) -> list[str]:
    if support.empty:
        return []
    ordered = support.assign(_security=support.index.astype(str)).sort_values(
        ["adv20", "_security"], ascending=[False, True], kind="stable"
    )
    return list(map(str, ordered.head(maximum).index))


def nearest_log_adv_control(
    selected: list[str], non_event_support: pd.DataFrame
) -> list[str]:
    if not selected or non_event_support.empty:
        return []
    available = non_event_support.copy()
    available = available[available["adv20"].gt(0.0)].copy()
    chosen: list[str] = []
    for security_id in selected:
        if available.empty:
            break
        target = float(non_event_support.attrs["selectedAdv"][security_id])
        distance = (np.log(available["adv20"]) - math.log(target)).abs()
        key = pd.DataFrame(
            {"distance": distance, "security": available.index.astype(str)},
            index=available.index,
        ).sort_values(["distance", "security"], kind="stable")
        match = str(key.index[0])
        chosen.append(match)
        available = available.drop(index=match)
    return chosen


def selection_row(
    security_id: str,
    rank: int,
    combined: pd.DataFrame,
    signal_date: pd.Timestamp,
    event_member: bool,
) -> dict[str, Any]:
    row = combined.loc[security_id]
    return {
        "rank": rank,
        "securityId": security_id,
        "stockCode": security_id.split(".", 1)[1],
        "name": str(row.get("name") or ""),
        "signalEventDate": signal_date.date().isoformat() if event_member else None,
        "eventMember": event_member,
        "eventChangeScore": (
            float(row["changeScoreRaw"])
            if event_member and pd.notna(row.get("changeScoreRaw"))
            else None
        ),
        "eventAvailableGroups": (
            int(row["availableGroups"])
            if event_member and pd.notna(row.get("availableGroups"))
            else None
        ),
        "prior20SessionMeanAmountCny": float(row["adv20"]),
        "auctionOpen": float(row["open"]),
        "auctionPreviousClose": float(row["prevClose"]),
        "auctionGap": float(row["openingGap"]),
        "status": "research_only_shadow_candidate_not_an_order",
    }


def build_arm_payloads(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    snapshot: pd.DataFrame,
    signal_date: pd.Timestamp,
    trade_date: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = list(map(str, panel["close"].columns))
    code_map: dict[str, str] = {}
    for security_id in columns:
        code = security_id.split(".", 1)[1]
        if code in code_map:
            raise RuntimeError(f"ambiguous SH/SZ stock code in panel: {code}")
        code_map[code] = security_id
    snap = snapshot.copy()
    snap["securityId"] = snap["stockCode"].map(code_map)
    snap = snap.dropna(subset=["securityId"]).set_index("securityId", drop=False)
    prior_eligible = panel["eligible"].loc[signal_date].reindex(columns).fillna(False)
    lookback = int(config["eventPool"]["liquidityLookbackSessions"])
    history = panel["amount"].loc[:signal_date].tail(lookback)
    adv = history.mean(axis=0, skipna=True).where(history.notna().sum(axis=0) >= lookback)
    market = pd.DataFrame(index=pd.Index(columns, name="securityId"))
    market["adv20"] = adv.reindex(columns)
    market["priorEligible"] = prior_eligible.reindex(columns).astype(bool)
    market = market.join(
        snap[["name", "open", "prevClose", "bidPrice1", "askPrice1"]], how="left"
    )
    market["openingGap"] = market["open"].div(
        market["prevClose"].replace(0.0, np.nan)
    ) - 1.0
    common_execution = (
        market["priorEligible"]
        & market["adv20"].gt(0.0)
        & market["open"].gt(0.0)
        & market["prevClose"].gt(0.0)
        & market["askPrice1"].gt(0.0)
        & market["openingGap"].notna()
    )
    observed = market[
        market["priorEligible"]
        & market["open"].gt(0.0)
        & market["prevClose"].gt(0.0)
        & market["openingGap"].notna()
    ]
    minimum_rows = int(config["auction"]["minimumAllMarketRows"])
    if len(observed) < minimum_rows:
        raise RuntimeError(
            f"only {len(observed)} panel-matched auction rows; requires {minimum_rows}"
        )
    breadth = float(observed["openingGap"].gt(0.0).mean())
    median_gap = float(observed["openingGap"].median())
    gate_cfg = config["auction"]["armBMarketGate"]
    market_gate = bool(
        breadth > float(gate_cfg["minimumPositiveGapBreadthExclusive"])
        and median_gap >= float(gate_cfg["minimumMedianOpeningGapInclusive"])
    )
    market_gate_audit = {
        "observedRows": int(len(observed)),
        "positiveGapBreadth": breadth,
        "medianOpeningGap": median_gap,
        "passed": market_gate,
        "modelFitted": False,
        "thresholdsFrozen": True,
    }
    event_ids = set(map(str, events.index)) if not events.empty else set()
    combined = market.join(
        events[
            ["changeScoreRaw", "availableGroups"]
        ] if not events.empty else pd.DataFrame(
            columns=["changeScoreRaw", "availableGroups"], dtype=float
        ),
        how="left",
    )
    event_common = combined.loc[
        combined.index.isin(event_ids) & common_execution
    ].copy()
    support_by_arm = {
        "event_liquidity": event_common,
        "event_auction_confirmed": (
            event_common[event_common["openingGap"].gt(0.0)].copy()
            if market_gate
            else event_common.iloc[0:0].copy()
        ),
    }
    non_event_common = combined.loc[
        (~combined.index.isin(event_ids)) & common_execution
    ].copy()
    non_event_by_arm = {
        "event_liquidity": non_event_common,
        "event_auction_confirmed": (
            non_event_common[non_event_common["openingGap"].gt(0.0)].copy()
            if market_gate
            else non_event_common.iloc[0:0].copy()
        ),
    }
    maximum = int(config["eventPool"]["maximumSelectionsPerArm"])
    seed = int(config["controls"]["seed"])
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        support = support_by_arm[arm]
        selected = select_by_liquidity(support, maximum)
        same_pool = deterministic_permutation(
            support, len(selected), seed, trade_date, arm
        )
        non_event = non_event_by_arm[arm]
        non_event.attrs["selectedAdv"] = {
            security_id: float(combined.loc[security_id, "adv20"])
            for security_id in selected
        }
        matched = nearest_log_adv_control(selected, non_event)
        arms[arm] = {
            "supportCount": int(len(support)),
            "selectionCount": len(selected),
            "zeroSelectionAllowed": True,
            "selectionReason": (
                "selected_without_forcing_quota"
                if selected
                else (
                    "market_gate_failed"
                    if arm == "event_auction_confirmed" and not market_gate
                    else "no_executable_positive_event_candidates"
                )
            ),
            "selected": [
                selection_row(item, rank, combined, signal_date, True)
                for rank, item in enumerate(selected, 1)
            ],
            "samePoolPermutation": [
                selection_row(item, rank, combined, signal_date, True)
                for rank, item in enumerate(same_pool, 1)
            ],
            "nonEventLiquidityMatched": [
                selection_row(item, rank, combined, signal_date, False)
                for rank, item in enumerate(matched, 1)
            ],
        }
        if len(matched) != len(selected):
            raise RuntimeError(f"{arm} could not construct a full non-event control")
    return arms, market_gate_audit


def rebuild_prediction_ledger(config: dict[str, Any]) -> None:
    root = output_root(config)
    rows = [load_json(path) for path in sorted((root / config["output"]["dailyDirectory"]).glob("*.json"))]
    atomic_jsonl(root / config["output"]["predictionLedger"], rows)


def rebuild_outcome_ledger(config: dict[str, Any]) -> None:
    root = output_root(config)
    rows = [load_json(path) for path in sorted((root / config["output"]["outcomeDirectory"]).glob("*.json"))]
    atomic_jsonl(root / config["output"]["outcomeLedger"], rows)


def score(
    config_path: Path,
    trade_date: str,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    if pd.Timestamp(trade_date) <= pd.Timestamp(config["prospective"]["strictlyAfter"]):
        raise RuntimeError("trade date is not strictly after the preregistered cutoff")
    root = output_root(config)
    daily_path = root / config["output"]["dailyDirectory"] / f"{trade_date}.json"
    config_hash = file_sha256(config_path)
    if daily_path.exists():
        existing = load_json(daily_path)
        if existing.get("configSha256") != config_hash:
            raise RuntimeError("existing immutable daily record has a different config hash")
        rebuild_prediction_ledger(config)
        return existing
    panel, panel_audit, catalyst_config = load_price_panel(config)
    signal_date = prior_session(panel["close"].index, trade_date)
    events, event_audit = event_candidates_for_signal_date(
        panel, catalyst_config, signal_date
    )
    minimum_coverage = float(config["eventPool"]["minimumFundamentalFileCoverage"])
    if event_audit["fundamentalFileCoverage"] < minimum_coverage:
        raise RuntimeError(
            "fundamental file coverage below frozen threshold: "
            f"{event_audit['fundamentalFileCoverage']:.4f} < {minimum_coverage:.4f}"
        )
    snapshot, snapshot_audit = load_auction_snapshot(
        trade_date, config, snapshot_path
    )
    arms, market_gate = build_arm_payloads(
        panel, events, snapshot, signal_date, trade_date, config
    )
    artifact = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": config["status"],
        "generatedAt": datetime.now().astimezone().isoformat(),
        "tradeDate": trade_date,
        "signalEventDate": signal_date.date().isoformat(),
        "configSha256": config_hash,
        "prospectiveStrictlyAfter": config["prospective"]["strictlyAfter"],
        "eventAudit": event_audit,
        "snapshotAudit": snapshot_audit,
        "marketGate": market_gate,
        "panelAudit": {
            "universeKind": panel_audit.get("universeKind"),
            "symbols": int(panel["close"].shape[1]),
            "lastAvailableSession": panel["close"].index.max().date().isoformat(),
            "unbiasedHistoricalValidationEligible": panel_audit.get(
                "unbiasedHistoricalValidationEligible"
            ),
        },
        "arms": arms,
        "eligibleForTrading": False,
        "orderInstructions": [],
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(daily_path, artifact)
    rebuild_prediction_ledger(config)
    return artifact


def outcome_rows(
    selections: list[dict[str, Any]],
    panel: dict[str, pd.DataFrame],
    trade_date: str,
    severe_loss: float,
) -> list[dict[str, Any]]:
    date = pd.Timestamp(trade_date)
    output: list[dict[str, Any]] = []
    for selection in selections:
        security_id = str(selection["securityId"])
        resolved = bool(
            date in panel["open"].index
            and security_id in panel["open"].columns
            and bool(panel["eligible"].loc[date, security_id])
        )
        open_price = (
            float(panel["open"].loc[date, security_id]) if resolved else math.nan
        )
        close_price = (
            float(panel["close"].loc[date, security_id]) if resolved else math.nan
        )
        resolved = bool(
            resolved
            and np.isfinite(open_price)
            and np.isfinite(close_price)
            and open_price > 0.0
        )
        gross = close_price / open_price - 1.0 if resolved else None
        output.append(
            {
                "securityId": security_id,
                "rank": int(selection["rank"]),
                "resolved": resolved,
                "open": open_price if resolved else None,
                "close": close_price if resolved else None,
                "grossOpenToCloseReturn": gross,
                "up": bool(gross > 0.0) if gross is not None else None,
                "severeLoss": bool(gross <= severe_loss) if gross is not None else None,
            }
        )
    return output


def cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in rows if row["resolved"]]
    values = np.asarray(
        [row["grossOpenToCloseReturn"] for row in resolved], dtype=float
    )
    return {
        "selectedRows": len(rows),
        "resolvedRows": len(resolved),
        "resolvedFraction": len(resolved) / len(rows) if rows else None,
        "grossUpRate": float(np.mean(values > 0.0)) if len(values) else None,
        "meanGrossReturn": float(np.mean(values)) if len(values) else None,
        "severeLossRate": (
            float(np.mean([row["severeLoss"] for row in resolved]))
            if resolved
            else None
        ),
    }


def settle(config_path: Path, trade_date: str) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    root = output_root(config)
    prediction_path = root / config["output"]["dailyDirectory"] / f"{trade_date}.json"
    if not prediction_path.exists():
        raise FileNotFoundError(f"no immutable morning prediction for {trade_date}")
    outcome_path = root / config["output"]["outcomeDirectory"] / f"{trade_date}.json"
    config_hash = file_sha256(config_path)
    if outcome_path.exists():
        existing = load_json(outcome_path)
        if existing.get("configSha256") != config_hash:
            raise RuntimeError("existing immutable outcome has a different config hash")
        rebuild_outcome_ledger(config)
        report(config_path)
        return existing
    prediction = load_json(prediction_path)
    if prediction.get("configSha256") != config_hash:
        raise RuntimeError("morning prediction does not match frozen config")
    panel, _audit, _catalyst_config = load_price_panel(config)
    date = pd.Timestamp(trade_date)
    if date not in panel["close"].index:
        raise RuntimeError(f"post-close bar for {trade_date} is not yet available")
    severe = float(config["outcome"]["severeLossThreshold"])
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        arms[arm] = {}
        for cohort in COHORT_NAMES:
            rows = outcome_rows(
                prediction["arms"][arm][cohort], panel, trade_date, severe
            )
            arms[arm][cohort] = {
                "rows": rows,
                "summary": cohort_summary(rows),
            }
    artifact = {
        "schemaVersion": "stock_event_auction_forward_outcome_v1",
        "codeVersion": CODE_VERSION,
        "status": config["status"],
        "generatedAt": datetime.now().astimezone().isoformat(),
        "tradeDate": trade_date,
        "configSha256": config_hash,
        "predictionSha256": file_sha256(prediction_path),
        "arms": arms,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(outcome_path, artifact)
    rebuild_outcome_ledger(config)
    report(config_path)
    return artifact


def aggregate_cohort(
    outcomes: list[dict[str, Any]], arm: str, cohort: str
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    active_days = 0
    resolved_active_days = 0
    daily_majority: list[bool] = []
    daily_returns: list[float] = []
    for artifact in outcomes:
        rows = artifact["arms"][arm][cohort]["rows"]
        if rows:
            active_days += 1
        resolved = [row for row in rows if row["resolved"]]
        if rows and len(resolved) == len(rows):
            resolved_active_days += 1
            daily_majority.append(
                float(np.mean([row["up"] for row in resolved])) > 0.5
            )
            daily_returns.append(
                float(np.mean([row["grossOpenToCloseReturn"] for row in resolved]))
            )
        all_rows.extend(resolved)
    values = np.asarray(
        [row["grossOpenToCloseReturn"] for row in all_rows], dtype=float
    )
    return {
        "predictionDays": len(outcomes),
        "activeDays": active_days,
        "resolvedActiveDays": resolved_active_days,
        "activeDayCoverage": active_days / len(outcomes) if outcomes else 0.0,
        "selectedRows": len(all_rows),
        "averageSelectionsPerActiveDay": (
            len(all_rows) / active_days if active_days else 0.0
        ),
        "stockLevelGrossUpRate": (
            float(np.mean(values > 0.0)) if len(values) else None
        ),
        "dailyMajorityUpRate": (
            float(np.mean(daily_majority)) if daily_majority else None
        ),
        "meanGrossOpenToCloseReturn": (
            float(np.mean(values)) if len(values) else None
        ),
        "meanDailyBasketGrossReturn": (
            float(np.mean(daily_returns)) if daily_returns else None
        ),
        "severeLossRate": (
            float(
                np.mean(
                    [
                        row["severeLoss"]
                        for row in all_rows
                    ]
                )
            )
            if all_rows
            else None
        ),
    }


def _better(candidate: Any, control: Any, strict: bool) -> bool:
    if candidate is None or control is None:
        return False
    return bool(candidate > control if strict else candidate <= control)


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Event-first opening-auction forward study V1",
        "",
        "> Research-only and shadow-only. No trading integration.",
        "",
        f"- state: `{result['state']}`",
        f"- prediction days: `{result['predictionDays']}`",
        f"- prospective cutoff: `{result['prospectiveStrictlyAfter']}`",
        "",
        "| arm / cohort | active days | stocks | up rate | mean gross | tail <= -3% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        for cohort in COHORT_NAMES:
            metric = result["arms"][arm][cohort]
            up = metric["stockLevelGrossUpRate"]
            mean = metric["meanGrossOpenToCloseReturn"]
            tail = metric["severeLossRate"]
            lines.append(
                f"| {arm} / {cohort} | {metric['resolvedActiveDays']} | "
                f"{metric['selectedRows']} | "
                f"{up:.2%} | {mean:.4%} | {tail:.2%} |"
                if up is not None and mean is not None and tail is not None
                else f"| {arm} / {cohort} | {metric['resolvedActiveDays']} | "
                f"{metric['selectedRows']} | n/a | n/a | n/a |"
            )
    lines += ["", "## Frozen verdict checks", ""]
    for arm in ARM_NAMES:
        checks = result["checks"][arm]
        lines.append(f"### {arm}")
        lines.extend(
            f"- {key}: {'PASS' if value else 'FAIL'}"
            for key, value in checks.items()
        )
        lines.append("")
    lines += [
        "No result in this report can modify a trading file, place an order, or "
        "promote itself.",
        "",
    ]
    return "\n".join(lines)


def report(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    root = output_root(config)
    daily_dir = root / config["output"]["dailyDirectory"]
    outcome_dir = root / config["output"]["outcomeDirectory"]
    predictions = [load_json(path) for path in sorted(daily_dir.glob("*.json"))]
    outcomes = [load_json(path) for path in sorted(outcome_dir.glob("*.json"))]
    minimum = int(config["prospective"]["minimumIndependentActiveDaysForVerdict"])
    arms: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for arm in ARM_NAMES:
        arms[arm] = {
            cohort: aggregate_cohort(outcomes, arm, cohort)
            for cohort in COHORT_NAMES
        }
        candidate = arms[arm]["selected"]
        control = arms[arm]["nonEventLiquidityMatched"]
        checks[arm] = {
            "minimumIndependentResolvedActiveDays": candidate[
                "resolvedActiveDays"
            ]
            >= minimum,
            "higherStockWinRateThanNonEventControl": _better(
                candidate["stockLevelGrossUpRate"],
                control["stockLevelGrossUpRate"],
                True,
            ),
            "higherMeanGrossThanNonEventControl": _better(
                candidate["meanGrossOpenToCloseReturn"],
                control["meanGrossOpenToCloseReturn"],
                True,
            ),
            "noHigherSevereLossThanNonEventControl": _better(
                candidate["severeLossRate"], control["severeLossRate"], False
            ),
        }
    enough = all(
        checks[arm]["minimumIndependentResolvedActiveDays"] for arm in ARM_NAMES
    )
    supported = enough and all(all(value for value in checks[arm].values()) for arm in ARM_NAMES)
    state = (
        "forward_hypothesis_supported_research_only"
        if supported
        else (
            "forward_hypothesis_rejected"
            if enough
            else "diagnostic_only_insufficient_forward_days"
        )
    )
    result = {
        "schemaVersion": "stock_event_auction_forward_report_v1",
        "codeVersion": CODE_VERSION,
        "status": config["status"],
        "generatedAt": datetime.now().astimezone().isoformat(),
        "state": state,
        "configSha256": file_sha256(config_path),
        "prospectiveStrictlyAfter": config["prospective"]["strictlyAfter"],
        "predictionDays": len(predictions),
        "settledDays": len(outcomes),
        "minimumIndependentActiveDaysForVerdict": minimum,
        "arms": arms,
        "checks": checks,
        "eligibleForTrading": False,
        "automaticPromotionAllowed": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_json(root / config["output"]["reportJson"], result)
    atomic_text(root / config["output"]["reportMarkdown"], render_report(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("score", "settle", "report"), required=True)
    parser.add_argument("--trade-date", default=shanghai_today())
    parser.add_argument("--snapshot-file", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == "score":
        result = score(config_path, args.trade_date, args.snapshot_file)
    elif args.mode == "settle":
        result = settle(config_path, args.trade_date)
    else:
        result = report(config_path)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
