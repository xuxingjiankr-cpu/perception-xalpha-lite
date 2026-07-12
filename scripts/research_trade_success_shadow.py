"""Research-only diagnostics for improving ETF trade-selection success.

This module evaluates five frozen hypotheses without touching the paper agent:

* replace the weakest sellable holding with the top candidate when capacity is full;
* maintain a causal, prior-day-only weak-ETF cooldown candidate list;
* audit whether the live candidate rank is actually monotone with later returns;
* condition outcomes on the already-recorded market regime and symbol group;
* diagnose 5/10/20-minute weakness after actual BUYs (candidate rows are secondary).

Future returns are labels only.  They never participate in candidate/holding selection,
cooldown state, ranking, or any trading path.  All artifacts are diagnostic-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

import decision_probability as dp
from run_etf_paper_trading_agent import ROOT, as_float


DEFAULT_CONFIG = ROOT / "configs" / "research" / "trade_success_shadow_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "trade_success_shadow_v1":
        raise ValueError("unexpected trade-success shadow schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("trade-success study must remain research/shadow-only")
    data = config["data"]
    if not str(data.get("agentName") or ""):
        raise ValueError("a frozen paper agentName is required")
    if int(data["purgeTradingDays"]) < 1:
        raise ValueError("at least one complete trading day must be purged")
    if int(data["embargoTradingDays"]) < 1:
        raise ValueError("at least one complete trading day of embargo is required")
    if int(data["maxLabelHorizonMinutes"]) < max(config["entryWeakness"]["horizonsMinutes"]):
        raise ValueError("declared max label horizon is too short")
    if config["validation"].get("allThresholdsFrozen") is not True:
        raise ValueError("all thresholds must be preregistered and frozen")
    if config["validation"].get("counterfactualRowsMayPromote") is not False:
        raise ValueError("counterfactual candidate rows may never promote a rule")
    if config["validation"].get("purgedWalkForwardRequiredForEvidence") is not True:
        raise ValueError("purged walk-forward must be required for every evidence gate")
    cooldown = config["cooldown"]
    if int(cooldown["promotionMinimumPriorDays"]) < int(cooldown["shadowMinimumPriorDays"]):
        raise ValueError("cooldown promotion history cannot be shorter than shadow history")
    if int(cooldown["cooldownTradingDays"]) != 1:
        raise ValueError("v1 preregisters exactly a one-trading-day cooldown")
    if int(config["rankingAudit"]["minimumCandidatesPerSnapshot"]) < 2:
        raise ValueError("rank audit requires at least two candidates per snapshot")
    if config["entryWeakness"].get("unknownFillTimeMayUseSubmissionTimeForMarkoutOnly") is not True:
        raise ValueError("unknown fill-time fallback contract must be explicit")
    if any(
        config[name].get("enabled") is not True
        for name in [
            "capacityRebalance",
            "cooldown",
            "rankingAudit",
            "regimeAudit",
            "entryWeakness",
        ]
    ):
        raise ValueError("all five preregistered shadow modules must remain enabled")
    safety = config["safety"]
    if safety.get("offlineOnly") is not True or safety.get("recordOnly") is not True:
        raise ValueError("offline, record-only safety flags are required")
    forbidden = [
        "brokerCallsAllowed",
        "onlineInferenceAllowed",
        "liveConfigWritesAllowed",
        "overlayWritesAllowed",
        "positionSizingAllowed",
        "orderSubmissionAllowed",
        "riskGateChangesAllowed",
        "buildDecisionIntegrationAllowed",
        "buySellGateIntegrationAllowed",
        "promotionAllowed",
    ]
    if any(safety.get(key) is not False for key in forbidden):
        raise ValueError("all production mutation permissions must be false")
    integration = config["paperIntegration"]
    if any(
        integration.get(key) is not False
        for key in [
            "allowed",
            "mayGenerateIndependentOrders",
            "mayChangeSellPath",
            "mayChangePositionSizing",
            "mayBypassTripleLock",
            "automaticPromotionAllowed",
        ]
    ):
        raise ValueError("paper integration is forbidden for this study")


def _parse_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def load_records(config: dict[str, Any], as_of_date: str) -> tuple[list[dict[str, Any]], list[Path]]:
    root = resolve(config["data"]["decisionScoresRoot"])
    effective = str(config["data"]["effectiveFrom"])
    selected_paths: list[Path] = []
    deduplicated: dict[str, dict[str, Any]] = {}
    fallback_index = 0
    for path in sorted(root.glob("decision_scores_*.jsonl")):
        rows = list(_parse_json_lines(path))
        included = False
        for row in rows:
            trade_date = str(row.get("date") or "")
            if not (effective <= trade_date <= as_of_date):
                continue
            included = True
            key = str(row.get("decision_id") or f"missing_{fallback_index}")
            fallback_index += 1
            deduplicated[key] = row
        if included:
            selected_paths.append(path)
    records = sorted(
        deduplicated.values(),
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("timestamp") or ""),
            str(row.get("decision_id") or ""),
        ),
    )
    return records, selected_paths


def _mode(values: list[str], default: str = "unknown") -> str:
    clean = [value for value in values if value]
    if not clean:
        return default
    counts = Counter(clean)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def wilson_upper(wins: int, observations: int, z: float = 1.96) -> float | None:
    if observations <= 0:
        return None
    p = wins / observations
    denominator = 1.0 + z * z / observations
    center = p + z * z / (2.0 * observations)
    radius = z * math.sqrt(p * (1.0 - p) / observations + z * z / (4.0 * observations**2))
    return (center + radius) / denominator


def build_cooldown_panel(
    daily_panel: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Create causal weak-ETF flags from prior completed dates only."""
    cfg = config["cooldown"]
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_panel:
        by_date[str(row["trade_date"])].append(row)
    for trade_date in sorted(by_date):
        todays: list[dict[str, Any]] = []
        for row in sorted(by_date[trade_date], key=lambda item: str(item["stockCode"])):
            code = str(row["stockCode"])
            lookback = int(cfg["lookbackIndependentEtfDays"])
            all_prior = history.get(code, [])
            prior = all_prior[-lookback:]
            prior_returns = [float(item["mean_net_return"]) for item in prior]
            prior_wins = sum(value > 0.0 for value in prior_returns)
            prior_losses = sum(value <= 0.0 for value in prior_returns)
            prior_days = len(prior_returns)
            prior_raw = sum(int(item["raw_rows"]) for item in prior)
            prior_mean = float(mean(prior_returns)) if prior_returns else None
            prior_win_rate = prior_wins / prior_days if prior_days else None
            upper = wilson_upper(prior_wins, prior_days)
            flag = bool(
                prior_days >= int(cfg["shadowMinimumPriorDays"])
                and prior_raw >= int(cfg["minimumPriorRawRows"])
                and prior_losses >= int(cfg["minimumLossDaysInLookback"])
                and prior_mean is not None
                and prior_mean <= float(cfg["maximumPriorMeanNetReturn"])
                and prior_win_rate is not None
                and prior_win_rate <= float(cfg["maximumPriorWinRate"])
                and upper is not None
                and upper <= float(cfg["maximumWilsonUpperWinRate"])
            )
            out = dict(row)
            out.update(
                {
                    "prior_days": prior_days,
                    "prior_total_days": len(all_prior),
                    "prior_raw_rows": prior_raw,
                    "prior_mean_net_return": prior_mean,
                    "prior_win_rate": prior_win_rate,
                    "prior_loss_days": prior_losses,
                    "prior_wilson_upper_win_rate": upper,
                    "shadow_cooldown_candidate": flag,
                    "promotion_history_ready": len(all_prior)
                    >= int(cfg["promotionMinimumPriorDays"]),
                    "production_block_allowed": False,
                }
            )
            todays.append(out)
            result.append(out)
        # Only after every symbol for the day has been scored may that day's labels
        # become history. This prevents cross-sectional same-day leakage.
        for row in todays:
            history[str(row["stockCode"])].append(row)
    return result


def current_cooldown_candidates(
    daily_panel: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    cfg = config["cooldown"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_panel:
        grouped[str(row["stockCode"])].append(row)
    result: list[dict[str, Any]] = []
    for code, rows in sorted(grouped.items()):
        lookback = int(cfg["lookbackIndependentEtfDays"])
        rows = sorted(rows, key=lambda row: str(row["trade_date"]))[-lookback:]
        values = [float(row["mean_net_return"]) for row in rows]
        wins = sum(value > 0.0 for value in values)
        losses = sum(value <= 0.0 for value in values)
        upper = wilson_upper(wins, len(values))
        flag = bool(
            len(values) >= int(cfg["shadowMinimumPriorDays"])
            and sum(int(row["raw_rows"]) for row in rows) >= int(cfg["minimumPriorRawRows"])
            and losses >= int(cfg["minimumLossDaysInLookback"])
            and mean(values) <= float(cfg["maximumPriorMeanNetReturn"])
            and wins / len(values) <= float(cfg["maximumPriorWinRate"])
            and upper is not None
            and upper <= float(cfg["maximumWilsonUpperWinRate"])
        )
        if flag:
            result.append(
                {
                    "stockCode": code,
                    "prior_days": len(values),
                    "prior_total_days": len(grouped[code]),
                    "prior_raw_rows": sum(int(row["raw_rows"]) for row in rows),
                    "prior_mean_net_return": float(mean(values)),
                    "prior_win_rate": wins / len(values),
                    "prior_loss_days": losses,
                    "prior_wilson_upper_win_rate": upper,
                    "shadow_only": True,
                    "promotion_history_ready": len(grouped[code])
                    >= int(cfg["promotionMinimumPriorDays"]),
                    "production_block_allowed": False,
                }
            )
    return result


def _average_rankdata(values: list[float]) -> np.ndarray:
    """Return one-based average ranks, preserving ties without scipy dependency."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def purged_day_splits(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expanding splits grouped by complete trade_date, with purge and embargo."""
    days = sorted({str(row.get("trade_date") or row.get("date") or "") for row in rows} - {""})
    initial = int(config["data"]["initialTrainDays"])
    purge = int(config["data"]["purgeTradingDays"])
    embargo = int(config["data"]["embargoTradingDays"])
    folds: list[dict[str, Any]] = []
    test_index = initial + purge
    while test_index < len(days):
        train_end = test_index - purge
        train = days[:train_end]
        test = [days[test_index]]
        if train:
            folds.append(
                {
                    "fold": len(folds) + 1,
                    "train_dates": train,
                    "purge_dates": days[train_end:test_index],
                    "test_dates": test,
                    "embargo_dates": days[test_index + 1 : test_index + 1 + embargo],
                    "thresholds_fitted_on": "preregistered_config_not_refit",
                }
            )
        test_index += 1 + embargo
    return folds


def _market_minute(timestamp: str) -> int | None:
    try:
        hour, minute = [int(value) for value in timestamp[:5].split(":")]
        return hour * 60 + minute
    except Exception:
        return None


def day_cluster_bootstrap(
    rows: list[dict[str, Any]], value_field: str, config: dict[str, Any]
) -> dict[str, Any]:
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is not None:
            by_day[str(row.get("trade_date") or row.get("date"))].append(float(value))
    daily = [float(mean(values)) for _, values in sorted(by_day.items()) if values]
    if not daily:
        return {
            "observations": 0,
            "independent_days": 0,
            "mean": None,
            "median": None,
            "win_rate": None,
            "ci95": [None, None],
        }
    validation = config["validation"]
    resamples = int(validation["bootstrapTradingDayResamples"])
    rng = random.Random(int(validation["bootstrapSeed"]) + sum(ord(char) for char in value_field))
    bootstrap = [mean(rng.choices(daily, k=len(daily))) for _ in range(resamples)] if len(daily) > 1 else daily
    return {
        "observations": sum(len(values) for values in by_day.values()),
        "independent_days": len(daily),
        "mean": float(mean(daily)),
        "median": float(median(daily)),
        "win_rate": sum(value > 0.0 for value in daily) / len(daily),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def summarize_entry_weakness(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    horizons = [int(value) for value in config["entryWeakness"]["horizonsMinutes"]]
    by_type: dict[str, Any] = {}
    for sample_type in ("actual_buy", "rank1_candidate_reference"):
        members = [row for row in rows if row["sample_type"] == sample_type]
        readiness_members = [
            row for row in members if sample_type != "actual_buy" or row.get("readiness_eligible") is True
        ]
        horizon_stats = {
            str(horizon): day_cluster_bootstrap(members, f"return_{horizon}m_net", config)
            for horizon in horizons
        }
        labeled = [row for row in members if row.get("immediate_weakness_label") is not None]
        near = [row for row in labeled if row["near_prior_observed_high"]]
        away = [row for row in labeled if not row["near_prior_observed_high"]]
        by_type[sample_type] = {
            "rows": len(members),
            "independent_days": len({str(row["trade_date"]) for row in members}),
            "readiness_eligible_rows": len(readiness_members),
            "readiness_eligible_days": len(
                {str(row["trade_date"]) for row in readiness_members}
            ),
            "horizon_net_returns": horizon_stats,
            "weakness_rate": mean([row["immediate_weakness_label"] for row in labeled]) if labeled else None,
            "near_prior_high_weakness_rate": mean([row["immediate_weakness_label"] for row in near]) if near else None,
            "away_from_prior_high_weakness_rate": mean([row["immediate_weakness_label"] for row in away]) if away else None,
        }
    actual = by_type["actual_buy"]
    ready = bool(
        actual["readiness_eligible_rows"] >= int(config["entryWeakness"]["minimumExecutedBuys"])
        and actual["readiness_eligible_days"] >= int(config["entryWeakness"]["minimumIndependentDays"])
    )
    return {
        "groups": by_type,
        "actual_execution_evidence_ready": ready,
        "candidate_reference_may_promote": False,
        "entry_gate_change_allowed": False,
    }


def load_agent_runs(
    config: dict[str, Any], as_of_date: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    path = resolve(config["data"]["agentRunsPath"])
    if not path.exists():
        return [], []
    effective = str(config["data"]["effectiveFrom"])
    rows: list[dict[str, Any]] = []
    for raw in _parse_json_lines(path):
        decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else raw
        trade_date = str(decision.get("trade_date") or raw.get("trade_date") or "")
        if effective <= trade_date <= as_of_date:
            rows.append(decision)
    rows.sort(
        key=lambda row: (
            str(row.get("trade_date") or ""),
            str((row.get("system_time") or {}).get("exchange_local_time") or row.get("timestamp") or ""),
        )
    )
    return rows, [path]


def _quote_market_clock(quote: dict[str, Any]) -> tuple[str | None, int | None]:
    raw = str(quote.get("source_quote_time") or "")
    if len(raw) >= 16:
        return raw[:10], _market_minute(raw[11:16])
    timestamp = str(quote.get("timestamp") or "")
    if len(timestamp) >= 16:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(ZoneInfo("Asia/Shanghai"))
            return parsed.date().isoformat(), parsed.hour * 60 + parsed.minute
        except Exception:
            return timestamp[:10], _market_minute(timestamp[11:16])
    return None, None


def load_executable_quote_index(
    config: dict[str, Any], as_of_date: str
) -> tuple[dict[str, dict[str, dict[int, dict[str, Any]]]], list[Path]]:
    root = resolve(config["data"]["minuteQuotesRoot"])
    effective = str(config["data"]["effectiveFrom"])
    index: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    paths: list[Path] = []
    for path in sorted(root.glob("minute_quotes_*.jsonl")):
        file_date = path.stem.replace("minute_quotes_", "")
        if not (effective <= file_date <= as_of_date):
            continue
        paths.append(path)
        for quote in _parse_json_lines(path):
            code = str(quote.get("stockCode") or "").zfill(6)
            trade_date, minute = _quote_market_clock(quote)
            price = as_float(quote.get("currentPrice"), 0.0)
            if not code or not trade_date or minute is None or price <= 0.0:
                continue
            # Same source minute can be collected repeatedly. Last observation wins,
            # but no later minute is ever backfilled into an earlier key.
            index[code][trade_date][minute] = quote
    return {code: dict(days) for code, days in index.items()}, paths


def _quote_at_or_after(
    daymap: dict[int, dict[str, Any]], target_minute: int, maximum_lag: int
) -> tuple[int, dict[str, Any]] | None:
    candidates = [minute for minute in sorted(daymap) if minute >= target_minute]
    if not candidates or candidates[0] - target_minute > maximum_lag:
        return None
    minute = candidates[0]
    return minute, daymap[minute]


def _quote_prices(quote: dict[str, Any]) -> tuple[float, float, float]:
    current = as_float(quote.get("currentPrice"), 0.0)
    bid = as_float(quote.get("bidPrice1"), current)
    ask = as_float(quote.get("askPrice1"), current)
    if bid <= 0.0:
        bid = current
    if ask <= 0.0:
        ask = current
    midpoint = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else current
    return bid, ask, midpoint


def executable_forward_path(
    quote_index: dict[str, Any],
    code: str,
    trade_date: str,
    event_minute: int,
    horizon_minutes: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    daymap = quote_index.get(str(code).zfill(6), {}).get(trade_date)
    if not isinstance(daymap, dict) or not daymap:
        return {"available": False, "reason": "no_quote_day"}
    maximum_lag = int(config["data"]["maximumQuoteLagMinutes"])
    entry = _quote_at_or_after(daymap, event_minute + 1, maximum_lag)
    if entry is None:
        return {"available": False, "reason": "no_next_executable_quote"}
    entry_minute, entry_quote = entry
    target = _quote_at_or_after(daymap, entry_minute + horizon_minutes, maximum_lag)
    if target is None:
        return {"available": False, "reason": "target_quote_missing_or_stale"}
    exit_minute, exit_quote = target
    entry_bid, entry_ask, entry_mid = _quote_prices(entry_quote)
    exit_bid, _, exit_mid = _quote_prices(exit_quote)
    if min(entry_bid, entry_ask, entry_mid, exit_bid, exit_mid) <= 0.0:
        return {"available": False, "reason": "non_positive_executable_price"}
    return {
        "available": True,
        "entry_minute": entry_minute,
        "exit_minute": exit_minute,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "entry_mid": entry_mid,
        "exit_bid": exit_bid,
        "exit_mid": exit_mid,
        "long_executable_return": exit_bid / entry_ask - 1.0,
        "hold_mark_to_bid_return": exit_bid / entry_mid - 1.0,
        "quote_lag_minutes": (entry_minute - event_minute) + (exit_minute - entry_minute - horizon_minutes),
    }


def _exchange_event_minute(run: dict[str, Any]) -> int | None:
    exchange_time = str((run.get("system_time") or {}).get("exchange_local_time") or "")
    if len(exchange_time) >= 16:
        try:
            parsed = datetime.fromisoformat(exchange_time.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(ZoneInfo("Asia/Shanghai"))
            return parsed.hour * 60 + parsed.minute
        except Exception:
            minute = _market_minute(exchange_time[11:16])
            if minute is not None:
                return minute
    ranked = run.get("ranked") if isinstance(run.get("ranked"), list) else []
    source_minutes = [
        _quote_market_clock(quote)[1]
        for quote in ranked
        if isinstance(quote, dict) and _quote_market_clock(quote)[1] is not None
    ]
    return max(source_minutes) if source_minutes else None


def _full_market_context(run: dict[str, Any]) -> dict[str, Any]:
    checks = run.get("risk_checks") if isinstance(run.get("risk_checks"), list) else []
    item = next((check for check in checks if check.get("name") == "full_market_entry_guard"), {})
    detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
    if detail.get("block_new_buy"):
        regime = "risk_off"
    elif detail.get("status") == "available":
        regime = str(detail.get("mode") or "selective_or_neutral")
    else:
        regime = "unavailable"
    return {
        "full_market_regime": regime,
        "full_market_status": str(detail.get("status") or "unknown"),
        "full_market_up_frac": detail.get("up_frac"),
        "full_market_median_change_pct": detail.get("median_change_pct"),
        "full_market_down_gt_1pct_frac": detail.get("down_gt_1pct_frac"),
        "full_market_block_new_buy": bool(detail.get("block_new_buy")),
    }


def build_point_in_time_candidate_events(
    runs: list[dict[str, Any]], quote_index: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct clean candidate snapshots from the agent's complete ranked list."""
    spacing = int(config["data"]["nonOverlappingEventSpacingMinutes"])
    horizons = sorted({5, 10, 20, int(config["rankingAudit"]["primaryHorizonMinutes"])})
    last_minute_by_date: dict[str, int] = {}
    candidate_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    seen_snapshot_minute: set[tuple[str, int]] = set()
    for run in runs:
        trade_date = str(run.get("trade_date") or "")
        event_minute = _exchange_event_minute(run)
        ranked = run.get("ranked") if isinstance(run.get("ranked"), list) else []
        if not trade_date or event_minute is None or not ranked:
            continue
        if (trade_date, event_minute) in seen_snapshot_minute:
            continue
        if trade_date in last_minute_by_date and event_minute - last_minute_by_date[trade_date] < spacing:
            continue
        seen_snapshot_minute.add((trade_date, event_minute))
        last_minute_by_date[trade_date] = event_minute
        held_positions = run.get("positions_t0") if isinstance(run.get("positions_t0"), dict) else {}
        held_codes = {str(code).zfill(6) for code in held_positions}
        source_minutes = [
            _quote_market_clock(quote)[1]
            for quote in ranked
            if isinstance(quote, dict) and _quote_market_clock(quote)[1] is not None
        ]
        max_source_minute = max(source_minutes) if source_minutes else None
        source_time_causal = max_source_minute is None or max_source_minute <= event_minute
        market = _full_market_context(run)
        frontier = run.get("frontier_competition_policy") if isinstance(run.get("frontier_competition_policy"), dict) else {}
        sector = run.get("sector_diversification") if isinstance(run.get("sector_diversification"), dict) else {}
        blocked_sector_codes = {
            str(item.get("stockCode") or "").zfill(6)
            for item in (sector.get("blocked_candidates") or [])
            if isinstance(item, dict) and item.get("stockCode")
        }
        snapshot_id = f"{trade_date}|{event_minute:04d}"
        snapshot = {
            "snapshot_id": snapshot_id,
            "trade_date": trade_date,
            "event_minute_cn": event_minute,
            "state_reason": str((run.get("state_machine") or {}).get("reason") or "unknown"),
            "held_codes": sorted(held_codes),
            "positions_t0": held_positions,
            "sellable_by_code": run.get("t0_sellable_by_code") if isinstance(run.get("t0_sellable_by_code"), dict) else {},
            "sell_score_by_code": run.get("sell_score_by_code") if isinstance(run.get("sell_score_by_code"), dict) else {},
            "pending_order_count": len(run.get("pending_t0_orders") or []),
            "planned_order_count": len(run.get("orders") or []),
            "frontier_applied": bool(frontier.get("appliedToRanking")),
            "frontier_mode": str(frontier.get("mode") or "unknown"),
            "max_source_quote_minute_cn": max_source_minute,
            "source_time_causal": source_time_causal,
            "sector_blocked_codes": sorted(blocked_sector_codes),
            **market,
        }
        snapshots.append(snapshot)
        if not source_time_causal:
            continue
        available_rank = 0
        for original_rank, quote in enumerate(ranked, start=1):
            if not isinstance(quote, dict):
                continue
            code = str(quote.get("stockCode") or "").zfill(6)
            if not code or code in held_codes or code in blocked_sector_codes:
                continue
            available_rank += 1
            execution = quote.get("execution_quality") if isinstance(quote.get("execution_quality"), dict) else {}
            shield = quote.get("safe_policy_shield") if isinstance(quote.get("safe_policy_shield"), dict) else {}
            entry_eligible_value = quote.get("pre_capacity_entry_eligible")
            row: dict[str, Any] = {
                "snapshot_id": snapshot_id,
                "trade_date": trade_date,
                "event_minute_cn": event_minute,
                "stockCode": code,
                "exchange": quote.get("exchange"),
                "name": quote.get("name"),
                "symbol_group": dp.classify_symbol_group(code, quote.get("name")),
                "original_rank": original_rank,
                "eligible_rank": available_rank,
                "candidate_count_full": len(ranked),
                "eligible_candidate_count": None,
                "rank_percentile": (original_rank - 1) / max(1, len(ranked) - 1),
                "held_at_t": False,
                "current_price_at_t": quote.get("currentPrice"),
                "spread_pct_at_t": quote.get("spread_pct"),
                "momentum_at_t": quote.get("momentum"),
                "acceleration_at_t": quote.get("acceleration"),
                "bid_pressure_at_t": quote.get("bid_pressure_3m_pct"),
                "recorded_execution_pass": execution.get("passed") is True,
                "recorded_safety_shield_pass": shield.get("passed") is True,
                "recorded_partial_entry_pass": execution.get("passed") is True and shield.get("passed") is True,
                "pre_capacity_entry_gate_recorded": entry_eligible_value is not None,
                "pre_capacity_entry_gate_pass": entry_eligible_value is True,
                "frontier_applied": snapshot["frontier_applied"],
                "frontier_mode": snapshot["frontier_mode"],
                **market,
            }
            for horizon in horizons:
                path = executable_forward_path(quote_index, code, trade_date, event_minute, horizon, config)
                row[f"return_{horizon}m_executable_gross"] = path.get("long_executable_return")
                row[f"return_{horizon}m_net"] = (
                    float(path["long_executable_return"]) - float(config["data"]["roundTripCost"])
                    if path.get("long_executable_return") is not None
                    else None
                )
                row[f"quote_path_{horizon}m_available"] = bool(path.get("available"))
            candidate_rows.append(row)
        if available_rank > 0:
            for row in candidate_rows[-available_rank:]:
                row["eligible_candidate_count"] = available_rank
    return candidate_rows, snapshots


def build_daily_candidate_panel_from_events(
    candidate_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    primary = int(config["rankingAudit"]["primaryHorizonMinutes"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if int(row["eligible_rank"]) > int(config["capacityRebalance"]["candidateMaximumRank"]):
            continue
        if row.get(f"return_{primary}m_net") is None:
            continue
        grouped[(str(row["trade_date"]), str(row["stockCode"]))].append(row)
    result: list[dict[str, Any]] = []
    for (trade_date, code), rows in sorted(grouped.items()):
        values = [float(row[f"return_{primary}m_net"]) for row in rows]
        result.append(
            {
                "trade_date": trade_date,
                "stockCode": code,
                "symbol_group": _mode([str(row["symbol_group"]) for row in rows]),
                "market_regime": _mode([str(row["full_market_regime"]) for row in rows]),
                "raw_rows": len(rows),
                "actual_buy_rows": 0,
                "candidate_rows": len(rows),
                "mean_net_return": float(mean(values)),
                "median_net_return": float(median(values)),
                "win": int(mean(values) > 0.0),
                "mean_candidate_rank": float(mean([int(row["eligible_rank"]) for row in rows])),
                "first_timestamp": f"{min(int(row['event_minute_cn']) for row in rows) // 60:02d}:"
                f"{min(int(row['event_minute_cn']) for row in rows) % 60:02d}:00",
            }
        )
    return result


def build_rank_events_from_candidates(
    candidate_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    primary = int(config["rankingAudit"]["primaryHorizonMinutes"])
    field = f"return_{primary}m_net"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["snapshot_id"])].append(row)
    result: list[dict[str, Any]] = []
    minimum_candidates = int(config["rankingAudit"]["minimumCandidatesPerSnapshot"])
    for snapshot_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["eligible_rank"]))
        if len(ordered) < minimum_candidates:
            continue
        top = ordered[0]
        comparators = ordered[1:3]
        if top.get(field) is None or any(row.get(field) is None for row in comparators):
            continue
        top_value = float(top[field])
        comparator_mean = float(mean([float(row[field]) for row in comparators]))
        labeled = [row for row in ordered if row.get(field) is not None]
        x = _average_rankdata([-float(row["eligible_rank"]) for row in labeled])
        y = _average_rankdata([float(row[field]) for row in labeled])
        spearman = (
            float(np.corrcoef(x, y)[0, 1])
            if len(labeled) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12
            else None
        )
        result.append(
            {
                "event_key": snapshot_id,
                "trade_date": str(top["trade_date"]),
                "event_minute_cn": int(top["event_minute_cn"]),
                "candidate_count": len(ordered),
                "top_code": str(top["stockCode"]),
                "comparison_codes": ",".join(str(row["stockCode"]) for row in comparators),
                "top_net_return": top_value,
                "rank2_rank3_mean_net_return": comparator_mean,
                "top_minus_rank2_rank3": top_value - comparator_mean,
                "spearman_better_rank_vs_net_return": spearman,
                "full_market_regime": str(top["full_market_regime"]),
                "frontier_applied": bool(top["frontier_applied"]),
                "selection_uses_future_label": False,
            }
        )
    return result


def rank_bucket_summary_from_candidates(
    candidate_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    primary = int(config["rankingAudit"]["primaryHorizonMinutes"])
    field = f"return_{primary}m_net"
    buckets = config["rankingAudit"]["fixedRankBuckets"]
    by_day_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in candidate_rows:
        value = row.get(field)
        if value is None:
            continue
        rank = int(row["eligible_rank"])
        label = next((f"{low}-{high}" for low, high in buckets if int(low) <= rank <= int(high)), None)
        if label:
            by_day_bucket[(str(row["trade_date"]), label)].append(float(value))
    grouped: dict[str, list[float]] = defaultdict(list)
    for (_, label), values in by_day_bucket.items():
        grouped[label].append(float(mean(values)))
    result: list[dict[str, Any]] = []
    for low, high in buckets:
        label = f"{low}-{high}"
        values = grouped.get(label, [])
        result.append(
            {
                "rank_bucket": label,
                "independent_days": len(values),
                "day_balanced_mean_net_return": float(mean(values)) if values else None,
                "day_balanced_win_rate": sum(value > 0.0 for value in values) / len(values) if values else None,
            }
        )
    return result


def summarize_regime_candidate_events(
    candidate_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Group each event by its own point-in-time regime, never an end-of-day mode."""
    primary = int(config["rankingAudit"]["primaryHorizonMinutes"])
    field = f"return_{primary}m_net"
    by_group_symbol_day: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in candidate_rows:
        value = row.get(field)
        if value is None or int(row["eligible_rank"]) > int(
            config["capacityRebalance"]["candidateMaximumRank"]
        ):
            continue
        key = (
            str(row["full_market_regime"]),
            str(row["symbol_group"]),
            str(row["trade_date"]),
            str(row["stockCode"]),
        )
        by_group_symbol_day[key].append(float(value))
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (regime, symbol_group, trade_date, _), values in by_group_symbol_day.items():
        grouped[(regime, symbol_group)].append((trade_date, float(mean(values))))
    cfg = config["regimeAudit"]
    result: list[dict[str, Any]] = []
    for (regime, symbol_group), observations in sorted(grouped.items()):
        values = [value for _, value in observations]
        days = len({trade_date for trade_date, _ in observations})
        result.append(
            {
                "market_regime": regime,
                "symbol_group": symbol_group,
                "symbol_days": len(observations),
                "independent_days": days,
                "mean_net_return": float(mean(values)),
                "median_net_return": float(median(values)),
                "win_rate": sum(value > 0.0 for value in values) / len(values),
                "rule_ready": bool(
                    days >= int(cfg["minimumIndependentDaysPerGroup"])
                    and len(observations) >= int(cfg["minimumSymbolDaysPerGroup"])
                ),
                "trade_gate_allowed": False,
                "regime_assignment": "point_in_time_event",
            }
        )
    return result


def build_capacity_events_from_snapshots(
    snapshots: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    quote_index: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = config["capacityRebalance"]
    primary_horizon = int(config["rankingAudit"]["primaryHorizonMinutes"])
    candidates_by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_snapshot[str(row["snapshot_id"])].append(row)
    result: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if snapshot["state_reason"] != cfg["capacityReason"]:
            continue
        if snapshot["pending_order_count"] or snapshot["planned_order_count"]:
            continue
        candidates = sorted(
            candidates_by_snapshot.get(str(snapshot["snapshot_id"]), []),
            key=lambda row: int(row["eligible_rank"]),
        )
        candidates = [
            row
            for row in candidates
            if int(row["eligible_rank"]) <= int(cfg["candidateMaximumRank"])
            and (
                not cfg.get("candidateMustPassRecordedExecutionAndSafetyChecks")
                or row["recorded_partial_entry_pass"]
            )
        ]
        if not candidates:
            continue
        candidate = candidates[0]
        sellable = {
            str(code).zfill(6): as_float(value, 0.0)
            for code, value in snapshot["sellable_by_code"].items()
            if as_float(value, 0.0) > 0.0
        }
        sell_scores = {
            str(code).zfill(6): as_float(value, 0.0)
            for code, value in snapshot["sell_score_by_code"].items()
            if str(code).zfill(6) in sellable
        }
        sell_scores = {
            code: value
            for code, value in sell_scores.items()
            if value >= float(cfg["incumbentMinimumSellScore"])
        }
        if not sell_scores:
            continue
        held_code = max(sell_scores, key=lambda code: (sell_scores[code], code))
        held_path = executable_forward_path(
            quote_index,
            held_code,
            str(snapshot["trade_date"]),
            int(snapshot["event_minute_cn"]),
            primary_horizon,
            config,
        )
        candidate_path = executable_forward_path(
            quote_index,
            str(candidate["stockCode"]),
            str(snapshot["trade_date"]),
            int(snapshot["event_minute_cn"]),
            primary_horizon,
            config,
        )
        executable = bool(held_path.get("available") and candidate_path.get("available"))
        if executable:
            sell_old_ratio = float(held_path["entry_bid"]) / float(held_path["entry_mid"])
            candidate_growth = float(candidate_path["exit_bid"]) / float(candidate_path["entry_ask"])
            replacement_return = sell_old_ratio * candidate_growth - 1.0 - float(
                config["data"]["incrementalSwapCost"]
            )
            baseline_return = float(held_path["hold_mark_to_bid_return"])
            incremental = replacement_return - baseline_return
        else:
            replacement_return = baseline_return = incremental = None
        pre_capacity_gate_complete = bool(candidate["pre_capacity_entry_gate_recorded"])
        evidence_eligible = bool(
            executable
            and (
                not cfg.get("candidatePreCapacityEntryGateMustBeRecorded")
                or (
                    pre_capacity_gate_complete
                    and candidate["pre_capacity_entry_gate_pass"]
                )
            )
        )
        result.append(
            {
                "event_key": str(snapshot["snapshot_id"]),
                "trade_date": str(snapshot["trade_date"]),
                "event_minute_cn": int(snapshot["event_minute_cn"]),
                "candidate_code": str(candidate["stockCode"]),
                "candidate_rank": int(candidate["eligible_rank"]),
                "held_code": held_code,
                "held_sell_score": float(sell_scores[held_code]),
                "candidate_partial_entry_pass": bool(candidate["recorded_partial_entry_pass"]),
                "candidate_pre_capacity_entry_gate_recorded": pre_capacity_gate_complete,
                "candidate_pre_capacity_entry_gate_pass": bool(candidate["pre_capacity_entry_gate_pass"]),
                "executable_path_available": executable,
                "evidence_eligible": evidence_eligible,
                "replacement_return": replacement_return,
                "baseline_continue_holding_return": baseline_return,
                "incremental_net_return": incremental,
                "shadow_swap_won": int(incremental > 0.0) if incremental is not None else None,
                "full_market_regime": str(snapshot["full_market_regime"]),
                "selection_uses_future_label": False,
                "shadow_only": True,
                "order_allowed": False,
            }
        )
    return result


def build_cooldown_replacement_events(
    candidate_rows: list[dict[str, Any]], cooldown_panel: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    primary = int(config["rankingAudit"]["primaryHorizonMinutes"])
    field = f"return_{primary}m_net"
    flag_map = {
        (str(row["trade_date"]), str(row["stockCode"])): row
        for row in cooldown_panel
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["snapshot_id"])].append(row)
    result: list[dict[str, Any]] = []
    for snapshot_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["eligible_rank"]))
        if not ordered:
            continue
        top = ordered[0]
        key = (str(top["trade_date"]), str(top["stockCode"]))
        top_flag = flag_map.get(key) or {}
        if not top_flag.get("shadow_cooldown_candidate", False) or top.get(field) is None:
            continue
        replacement = next(
            (
                row
                for row in ordered[1:]
                if not (flag_map.get((str(row["trade_date"]), str(row["stockCode"]))) or {}).get(
                    "shadow_cooldown_candidate", False
                )
                and row.get(field) is not None
            ),
            None,
        )
        if replacement is None:
            continue
        result.append(
            {
                "event_key": snapshot_id,
                "trade_date": str(top["trade_date"]),
                "flagged_code": str(top["stockCode"]),
                "replacement_code": str(replacement["stockCode"]),
                "flagged_rank": int(top["eligible_rank"]),
                "replacement_rank": int(replacement["eligible_rank"]),
                "flagged_recorded_partial_entry_pass": bool(
                    top.get("recorded_partial_entry_pass")
                ),
                "replacement_recorded_partial_entry_pass": bool(
                    replacement.get("recorded_partial_entry_pass")
                ),
                "entry_gate_evidence_complete": bool(
                    top.get("recorded_partial_entry_pass")
                    and replacement.get("recorded_partial_entry_pass")
                ),
                "flagged_net_return": float(top[field]),
                "replacement_net_return": float(replacement[field]),
                "replacement_minus_flagged": float(replacement[field]) - float(top[field]),
                "flagged_prior_total_days": int(top_flag.get("prior_total_days") or 0),
                "promotion_history_ready": bool(top_flag.get("promotion_history_ready")),
                "coverage_preserved": True,
                "selection_uses_current_day_label": False,
            }
        )
    return result


def summarize_cooldown_replacements(
    events: list[dict[str, Any]], cooldown_panel: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    cfg = config["cooldown"]
    paired = day_cluster_bootstrap(events, "replacement_minus_flagged", config)
    distinct = len({str(row["flagged_code"]) for row in events})
    ready = bool(
        paired["independent_days"] >= int(cfg["minimumEvaluationDays"])
        and paired["observations"] >= int(cfg["minimumFlaggedSymbolDays"])
        and distinct >= int(cfg["minimumDistinctEtfs"])
        and paired["ci95"][0] is not None
        and paired["ci95"][0] > 0.0
    )
    flagged_symbol_days = sum(bool(row["shadow_cooldown_candidate"]) for row in cooldown_panel)
    return {
        "flagged_symbol_days": flagged_symbol_days,
        "total_symbol_days": len(cooldown_panel),
        "replacement_events": len(events),
        "distinct_flagged_etfs": distinct,
        "retained_coverage_if_applied": 1.0 if events else None,
        "same_snapshot_replacement_minus_flagged": paired,
        "shadow_evidence_ready": ready,
        "mechanical_trade_reduction_risk": False if events else True,
        "production_block_allowed": False,
        "reason": (
            "equal-coverage next-rank replacements improved returns, but actual-fill OOS is still required"
            if ready
            else "insufficient equal-coverage replacement evidence; cooldown remains a shadow hypothesis"
        ),
    }


def load_confirmed_buy_markouts(
    config: dict[str, Any], as_of_date: str, quote_index: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[Path]]:
    path = resolve(config["data"]["orderLifecyclePath"])
    if not path.exists():
        return [], []
    effective = str(config["data"]["effectiveFrom"])
    cost = float(config["data"]["roundTripCost"])
    primary = int(config["entryWeakness"]["primaryHorizonMinutes"])
    rows: list[dict[str, Any]] = []
    for order in _parse_json_lines(path):
        trade_date = str(order.get("trade_date") or "")
        if not (effective <= trade_date <= as_of_date):
            continue
        if str(order.get("agent_name") or "") != str(config["data"]["agentName"]):
            continue
        if str(order.get("direction") or "").lower() != "buy" or order.get("fill_confirmed") is not True:
            continue
        code = str(order.get("stockCode") or "").zfill(6)
        submitted = str(order.get("submitted_at") or "")
        fill_time_known = bool(order.get("fill_time_known")) and bool(order.get("filledTime"))
        event_time = str(order.get("filledTime")) if fill_time_known else submitted
        if not fill_time_known and not config["entryWeakness"].get(
            "unknownFillTimeMayUseSubmissionTimeForMarkoutOnly", False
        ):
            continue
        event_date, event_minute = _quote_market_clock({"timestamp": event_time})
        if event_date and event_date != trade_date:
            trade_date = event_date
        if event_minute is None:
            continue
        filled_price = as_float(order.get("filledPrice"), 0.0)
        row: dict[str, Any] = {
            "event_key": str(order.get("orderId") or f"{trade_date}|{code}|{submitted}"),
            "trade_date": trade_date,
            "timestamp": event_time,
            "stockCode": code,
            "sample_type": "actual_buy",
            "fill_evidence": order.get("fill_evidence"),
            "fill_time_known": fill_time_known,
            "markout_time_basis": "filledTime" if fill_time_known else "submitted_at_diagnostic_only",
            "markout_reference": order.get("markout_reference"),
            "entry_reason": order.get("reason"),
            "entry_price": filled_price,
            "features_use_future_label": False,
        }
        daymap = quote_index.get(code, {}).get(trade_date, {})
        prior = [quote for minute, quote in sorted(daymap.items()) if minute <= event_minute]
        prior_high = max([as_float(quote.get("currentPrice"), 0.0) for quote in prior], default=0.0)
        distance = filled_price / prior_high - 1.0 if filled_price > 0.0 and prior_high > 0.0 else None
        row["distance_to_prior_observed_high"] = distance
        row["near_prior_observed_high"] = bool(
            distance is not None and distance >= float(config["entryWeakness"]["nearPriorHighDistancePct"])
        )
        for horizon in config["entryWeakness"]["horizonsMinutes"]:
            horizon = int(horizon)
            markout = order.get(f"markout_{horizon}m_bps")
            if markout is not None:
                gross = float(markout) / 10000.0
            elif filled_price > 0.0:
                target = _quote_at_or_after(
                    daymap,
                    event_minute + horizon,
                    int(config["data"]["maximumQuoteLagMinutes"]),
                )
                gross = _quote_prices(target[1])[0] / filled_price - 1.0 if target else None
            else:
                gross = None
            row[f"return_{horizon}m_gross"] = gross
            row[f"return_{horizon}m_net"] = gross - cost if gross is not None else None
        primary_return = row.get(f"return_{primary}m_net")
        row["immediate_weakness_label"] = (
            int(float(primary_return) <= float(config["entryWeakness"]["weaknessNetReturnThreshold"]))
            if primary_return is not None
            else None
        )
        close_quote = daymap[max(daymap)] if daymap else None
        target = _quote_at_or_after(
            daymap,
            event_minute + primary,
            int(config["data"]["maximumQuoteLagMinutes"]),
        ) if daymap else None
        if filled_price > 0.0 and close_quote and target:
            exit_primary = _quote_prices(target[1])[0] / filled_price - 1.0
            hold_close = _quote_prices(close_quote)[0] / filled_price - 1.0
            row["primary_exit_vs_hold_close_improvement"] = exit_primary - hold_close
        else:
            row["primary_exit_vs_hold_close_improvement"] = None
        row["readiness_eligible"] = bool(
            fill_time_known
            and filled_price > 0.0
            and row.get(f"return_{primary}m_net") is not None
        )
        rows.append(row)
    return rows, [path]


def _file_fingerprint(paths: list[Path], config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3%}" if percent else f"{float(value):.4f}"


def render_report(result: dict[str, Any]) -> str:
    sample = result["sample"]
    capacity = result["capacityRebalance"]
    cooldown = result["cooldown"]
    ranking = result["rankingAudit"]
    entry = result["entryWeakness"]
    lines = [
        "# ETF Trade Success Shadow Research",
        "",
        "Status: `research_only / shadow_only / no trading integration`",
        "",
        f"- run_id: `{result['runId']}`",
        f"- data: {result['dataRange']['start']} to {result['dataRange']['end']}",
        f"- independent trading days: {sample['independentTradingDays']}",
        f"- confirmed BUY markouts: {sample['confirmedBuyMarkouts']}; executable candidate 10m outcomes: {sample['candidate10mExecutableOutcomes']}",
        "- Candidate outcomes are point-in-time shadow labels, not broker-confirmed trades.",
        "",
        "## Capacity replacement",
        "",
        f"- evidence-eligible swaps: {capacity['summary']['observations']} across {capacity['summary']['independent_days']} days",
        f"- partial-gate diagnostic events: {capacity['partialGateDiagnosticSummary']['observations']} (not eligible for promotion)",
        f"- incremental net mean: {_fmt(capacity['summary']['mean'], True)}; win rate: {_fmt(capacity['summary']['win_rate'], True)}",
        f"- day-cluster 95% CI: [{_fmt(capacity['summary']['ci95'][0], True)}, {_fmt(capacity['summary']['ci95'][1], True)}]",
        f"- purged walk-forward test events: {capacity['purgedWalkForwardSummary']['observations']}",
        f"- evidence ready: `{capacity['evidenceReady']}`; production allowed: `false`",
        "- This policy adds a sell+buy pair; it cannot look better merely by reducing trade count.",
        "",
        "## Weak-ETF cooldown",
        "",
        f"- flagged symbol-days: {cooldown['summary']['flagged_symbol_days']}; current shadow candidates: {len(cooldown['currentCandidates'])}",
        f"- retained coverage if applied: {_fmt(cooldown['summary']['retained_coverage_if_applied'], True)}",
        f"- equal-coverage next-rank replacement minus flagged: {_fmt(cooldown['summary']['same_snapshot_replacement_minus_flagged']['mean'], True)}",
        f"- purged walk-forward replacement events: {cooldown['purgedWalkForwardSummary']['replacement_events']}",
        f"- evidence ready: `{cooldown['evidenceReady']}`; production block allowed: `false`",
        "",
        "## Candidate rank audit",
        "",
        f"- labeled snapshots: {ranking['topMinusBottom']['observations']} across {ranking['topMinusBottom']['independent_days']} days",
        f"- rank-1 minus rank-2/3 mean: {_fmt(ranking['topMinusBottom']['mean'], True)}",
        f"- day-cluster 95% CI: [{_fmt(ranking['topMinusBottom']['ci95'][0], True)}, {_fmt(ranking['topMinusBottom']['ci95'][1], True)}]",
        f"- purged walk-forward paired snapshots: {ranking['purgedWalkForward']['observations']}",
        f"- entry-gate-complete paired snapshots available: {ranking['entryGateEvidenceEligiblePairedSnapshots']}",
        f"- monotonic evidence ready: `{ranking['evidenceReady']}`; rank change allowed: `false`",
        "",
        "| rank bucket | days | day-balanced net | win rate |",
        "|---|---:|---:|---:|",
    ]
    for row in ranking["fixedBuckets"]:
        lines.append(
            f"| {row['rank_bucket']} | {row['independent_days']} | "
            f"{_fmt(row['day_balanced_mean_net_return'], True)} | {_fmt(row['day_balanced_win_rate'], True)} |"
        )
    actual = entry["groups"]["actual_buy"]
    lines.extend(
        [
            "",
            "## Immediate post-entry weakness",
            "",
            f"- actual BUY rows: {actual['rows']} across {actual['independent_days']} days",
            f"- exact-time readiness rows: {actual['readiness_eligible_rows']} across {actual['readiness_eligible_days']} days",
            f"- actual purged-walk-forward evidence ready: `{entry['evidenceReady']}`",
            f"- 10-minute weakness rate: {_fmt(actual['weakness_rate'], True)}",
            "- Rank-1 candidate diagnostics are reported separately and cannot promote an entry rule.",
            "",
            "## Regime conditioning",
            "",
            "| regime | symbol group | symbol-days | days | mean net | win rate | rule ready |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["regimeAudit"]:
        lines.append(
            f"| {row['market_regime']} | {row['symbol_group']} | {row['symbol_days']} | "
            f"{row['independent_days']} | {_fmt(row['mean_net_return'], True)} | "
            f"{_fmt(row['win_rate'], True)} | {row['rule_ready']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- overall status: `{result['verdict']['status']}`",
            f"- reason: {result['verdict']['reason']}",
            "- No BUY/SELL gate, position size, order, risk gate, overlay, or execution lock was changed.",
            "- Promotion requires at least 20 independent forward days plus actual executed-trade validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_content_fingerprint(
    result: dict[str, Any], tables: dict[str, list[dict[str, Any]]]
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(render_report(result).encode("utf-8"))
    for name, rows in sorted(tables.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_csv_text(rows).encode("utf-8"))
    return digest.hexdigest()


def write_artifacts(
    result: dict[str, Any], tables: dict[str, list[dict[str, Any]]], output_root: Path, run_id: str
) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    content_fingerprint = _artifact_content_fingerprint(result, tables)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("inputFingerprint") != result["inputFingerprint"]:
            raise ValueError("sealed run_id already exists with a different input fingerprint")
        if existing.get("contentFingerprint") != content_fingerprint:
            raise ValueError("sealed run_id already exists with different artifact content")
    _atomic_text(
        output_dir / "trade_success_shadow_result.json",
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_text(output_dir / "trade_success_shadow_report.md", render_report(result))
    for name, rows in sorted(tables.items()):
        _atomic_text(output_dir / f"{name}.csv", _csv_text(rows))
    manifest = {
        "schemaVersion": "trade_success_shadow_manifest_v1",
        "runId": run_id,
        "inputFingerprint": result["inputFingerprint"],
        "contentFingerprint": content_fingerprint,
        "status": "research_only",
        "sealed": True,
        "artifacts": sorted(path.name for path in output_dir.iterdir() if path.name != "manifest.json"),
    }
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return output_dir


def run(config: dict[str, Any], as_of_date: str, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    validate_config(config)
    # Decision-score rows are retained for contamination audit only. Clean selection
    # events come from full agent-run snapshots and confirmed fills from the lifecycle.
    records, score_paths = load_records(config, as_of_date)
    runs, run_paths = load_agent_runs(config, as_of_date)
    quote_index, quote_paths = load_executable_quote_index(config, as_of_date)
    candidate_events, snapshots = build_point_in_time_candidate_events(runs, quote_index, config)
    daily_panel = build_daily_candidate_panel_from_events(candidate_events, config)
    cooldown_panel = build_cooldown_panel(daily_panel, config)
    cooldown_current = current_cooldown_candidates(daily_panel, config)
    cooldown_replacements = build_cooldown_replacement_events(candidate_events, cooldown_panel, config)
    capacity_events = build_capacity_events_from_snapshots(
        snapshots, candidate_events, quote_index, config
    )
    capacity_eligible = [row for row in capacity_events if row["evidence_eligible"]]
    rank_events = build_rank_events_from_candidates(candidate_events, config)
    rank_entry_gate_events = build_rank_events_from_candidates(
        [row for row in candidate_events if row.get("recorded_partial_entry_pass") is True],
        config,
    )
    weakness_rows, lifecycle_paths = load_confirmed_buy_markouts(config, as_of_date, quote_index)
    trading_days = sorted(
        {
            str(row["trade_date"])
            for row in snapshots
            if row.get("source_time_causal") is True
        }
    )
    folds = purged_day_splits(
        [{"trade_date": trade_date} for trade_date in trading_days], config
    )
    walk_forward_test_dates = {
        trade_date for fold in folds for trade_date in fold["test_dates"]
    }
    capacity_walk_forward = [
        row for row in capacity_eligible if str(row["trade_date"]) in walk_forward_test_dates
    ]
    rank_walk_forward = [
        row
        for row in rank_entry_gate_events
        if str(row["trade_date"]) in walk_forward_test_dates
    ]
    cooldown_walk_forward = [
        row
        for row in cooldown_replacements
        if str(row["trade_date"]) in walk_forward_test_dates
        and row.get("promotion_history_ready") is True
        and row.get("entry_gate_evidence_complete") is True
    ]
    weakness_walk_forward = [
        row for row in weakness_rows if str(row["trade_date"]) in walk_forward_test_dates
    ]

    source_paths = [Path(__file__).resolve(), (ROOT / "scripts" / "decision_scoring.py").resolve()]
    fingerprint_context = {
        "schemaVersion": "trade_success_shadow_fingerprint_v1",
        "asOfDate": as_of_date,
        "config": config,
    }
    fingerprint = _file_fingerprint(
        score_paths + run_paths + quote_paths + lifecycle_paths + source_paths,
        fingerprint_context,
    )
    run_id = run_id or f"trade_success_shadow_{as_of_date.replace('-', '')}_{fingerprint[:10]}"
    actual_buy_outcomes = len(weakness_rows)
    primary_horizon = int(config["rankingAudit"]["primaryHorizonMinutes"])
    candidate_outcomes = sum(
        row.get(f"return_{primary_horizon}m_net") is not None for row in candidate_events
    )
    capacity_summary = day_cluster_bootstrap(capacity_eligible, "incremental_net_return", config)
    capacity_walk_forward_summary = day_cluster_bootstrap(
        capacity_walk_forward, "incremental_net_return", config
    )
    capacity_diagnostic = day_cluster_bootstrap(capacity_events, "incremental_net_return", config)
    capacity_cfg = config["capacityRebalance"]
    capacity_ready = bool(
        capacity_walk_forward_summary["independent_days"] >= int(capacity_cfg["minimumIndependentDays"])
        and capacity_walk_forward_summary["observations"] >= int(capacity_cfg["minimumEvents"])
        and len({str(row["candidate_code"]) for row in capacity_walk_forward})
        >= int(capacity_cfg["minimumDistinctCandidateEtfs"])
        and capacity_walk_forward_summary["mean"] is not None
        and capacity_walk_forward_summary["mean"] > float(capacity_cfg["minimumMeanIncrementalNetReturn"])
        and capacity_walk_forward_summary["win_rate"] >= float(capacity_cfg["minimumWinRate"])
        and capacity_walk_forward_summary["ci95"][0] is not None
        and capacity_walk_forward_summary["ci95"][0]
        > float(capacity_cfg["bootstrapCiLowerMustExceed"])
    )
    rank_summary = day_cluster_bootstrap(rank_events, "top_minus_rank2_rank3", config)
    rank_walk_forward_summary = day_cluster_bootstrap(
        rank_walk_forward, "top_minus_rank2_rank3", config
    )
    rank_cfg = config["rankingAudit"]
    by_week: dict[str, list[float]] = defaultdict(list)
    for row in rank_walk_forward:
        try:
            parsed = date.fromisoformat(str(row["trade_date"]))
            week = f"{parsed.isocalendar().year}-W{parsed.isocalendar().week:02d}"
        except Exception:
            week = "unknown"
        by_week[week].append(float(row["top_minus_rank2_rank3"]))
    weekly_signs = {week: float(mean(values)) for week, values in sorted(by_week.items())}
    last_four_weeks = list(sorted(weekly_signs))[-4:]
    positive_weeks = sum(weekly_signs[week] > 0.0 for week in last_four_weeks)
    rank_ready = bool(
        rank_walk_forward_summary["independent_days"] >= int(rank_cfg["minimumIndependentDays"])
        and rank_walk_forward_summary["observations"] >= int(rank_cfg["minimumPairedSnapshots"])
        and rank_walk_forward_summary["mean"] is not None
        and rank_walk_forward_summary["mean"] >= float(rank_cfg["minimumEconomicDifference"])
        and rank_walk_forward_summary["ci95"][0] is not None
        and rank_walk_forward_summary["ci95"][0] > 0.0
        and len(last_four_weeks) == 4
        and positive_weeks >= int(rank_cfg["minimumWeeksWithCorrectSignOutOfFour"])
    )
    cooldown_summary = summarize_cooldown_replacements(
        cooldown_replacements, cooldown_panel, config
    )
    cooldown_walk_forward_summary = summarize_cooldown_replacements(
        cooldown_walk_forward, cooldown_panel, config
    )
    entry_summary = summarize_entry_weakness(weakness_rows, config)
    entry_walk_forward_summary = summarize_entry_weakness(weakness_walk_forward, config)
    regime_summary = summarize_regime_candidate_events(candidate_events, config)
    any_ready = any(
        [
            capacity_ready,
            rank_ready,
            cooldown_walk_forward_summary["shadow_evidence_ready"],
            entry_walk_forward_summary["actual_execution_evidence_ready"],
        ]
    )
    score_buy_rows = [row for row in records if str(row.get("decision_type")) == "BUY"]
    score_candidate_rows = [
        row for row in records if str(row.get("decision_type")) == "BUY_CANDIDATE"
    ]
    score_by_snapshot_code: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in records:
        score_by_snapshot_code[
            (str(row.get("date")), str(row.get("timestamp")), str(row.get("etf_code")))
        ].add(str(row.get("decision_type")))
    overlap_count = sum(
        "HOLD" in types and "BUY_CANDIDATE" in types for types in score_by_snapshot_code.values()
    )
    policy_cohorts = Counter(
        f"frontier={row['frontier_mode']}|applied={row['frontier_applied']}" for row in snapshots
    )
    result: dict[str, Any] = {
        "schemaVersion": "trade_success_shadow_result_v1",
        "runId": run_id,
        "scheduledLogicalAt": f"{as_of_date}T15:30:00+08:00",
        "scheduledLogicalAtBasis": "16:30 Asia/Seoul daily research-suite schedule",
        "status": "research_only",
        "shadowOnly": True,
        "inputFingerprint": fingerprint,
        "dataRange": {
            "start": trading_days[0] if trading_days else None,
            "end": trading_days[-1] if trading_days else None,
            "asOfDate": as_of_date,
        },
        "sample": {
            "decisionScoreAuditRows": len(records),
            "agentRuns": len(runs),
            "pointInTimeSnapshots": len(snapshots),
            "pointInTimeCandidateEvents": len(candidate_events),
            "independentTradingDays": len(trading_days),
            "confirmedBuyMarkouts": actual_buy_outcomes,
            "candidate10mExecutableOutcomes": candidate_outcomes,
            "dailySymbolRows": len(daily_panel),
        },
        "dataQualityAudit": {
            "decisionScoreBuyRows": len(score_buy_rows),
            "decisionScoreBuyRowsWithConfirmedExecution": sum(
                row.get("was_executed") is True and str(row.get("fill_status") or "").lower() in {"filled", "confirmed"}
                for row in score_buy_rows
            ),
            "decisionScoreSyntheticPlannedBuyRows": sum(
                row.get("was_executed") is not True for row in score_buy_rows
            ),
            "candidateRowsWithNullEligibility": sum(
                row.get("candidate_eligible") is None for row in score_candidate_rows
            ),
            "candidateHoldSameSnapshotCodeOverlaps": overlap_count,
            "decisionConfigHashCount": len(
                {str(row.get("config_sha256")) for row in records if row.get("config_sha256")}
            ),
            "decisionPolicyHashCount": len(
                {str(row.get("policy_sha256")) for row in records if row.get("policy_sha256")}
            ),
            "decisionRowsMissingPolicyHash": sum(not row.get("policy_sha256") for row in records),
            "policyCohorts": dict(policy_cohorts),
            "fullMarketRegimeCounts": dict(Counter(row["full_market_regime"] for row in snapshots)),
            "sourceTimeNonCausalSnapshots": sum(
                row.get("source_time_causal") is not True for row in snapshots
            ),
            "capacityCandidatesMissingPreCapacityEntryGate": sum(
                not row["candidate_pre_capacity_entry_gate_recorded"] for row in capacity_events
            ),
        },
        "labelDefinitions": {
            "candidateOutcome": "next executable ask to target-horizon bid, maximum six-minute quote lag; label only",
            "actualBuyOutcome": "confirmed lifecycle fill price with post-submission markout; unknown fill time is disclosed",
            "capacityIncremental": "sell-old-at-bid and buy-candidate-at-ask replacement return minus continue-holding return and fixed incremental cost",
            "immediateWeakness": (
                f"{config['entryWeakness']['primaryHorizonMinutes']}m net return <= "
                f"{config['entryWeakness']['weaknessNetReturnThreshold']}"
            ),
        },
        "walkForward": {
            "groupUnit": "complete_trade_date_all_symbols",
            "purgeTradingDays": int(config["data"]["purgeTradingDays"]),
            "embargoTradingDays": int(config["data"]["embargoTradingDays"]),
            "folds": folds,
            "testDates": sorted(walk_forward_test_dates),
            "thresholdRefitAllowed": False,
            "evidenceMetricsUseTestDatesOnly": True,
        },
        "capacityRebalance": {
            "policy": capacity_cfg["policy"],
            "selectedEvents": len(capacity_events),
            "evidenceEligibleEvents": len(capacity_eligible),
            "summary": capacity_summary,
            "purgedWalkForwardSummary": capacity_walk_forward_summary,
            "partialGateDiagnosticSummary": capacity_diagnostic,
            "evidenceReady": capacity_ready,
            "mechanicalTradeReduction": False,
            "incrementalOrderLegsPerEvent": 2,
            "paperIntegrationAllowed": False,
        },
        "cooldown": {
            "summary": cooldown_summary,
            "purgedWalkForwardSummary": cooldown_walk_forward_summary,
            "evidenceReady": cooldown_walk_forward_summary["shadow_evidence_ready"],
            "currentCandidates": cooldown_current,
            "parametersFrozen": True,
            "paperIntegrationAllowed": False,
        },
        "rankingAudit": {
            "topMinusBottom": rank_summary,
            "purgedWalkForward": rank_walk_forward_summary,
            "entryGateEvidenceEligiblePairedSnapshots": len(rank_entry_gate_events),
            "primaryComparison": rank_cfg["primaryComparison"],
            "fixedBuckets": rank_bucket_summary_from_candidates(candidate_events, config),
            "weeklyMeanDifferences": weekly_signs,
            "lastFourWalkForwardWeeks": last_four_weeks,
            "positiveWeeks": positive_weeks,
            "evidenceReady": rank_ready,
            "rankingChangeAllowed": False,
        },
        "regimeAudit": regime_summary,
        "entryWeakness": {
            **entry_summary,
            "purgedWalkForward": entry_walk_forward_summary,
            "evidenceReady": entry_walk_forward_summary[
                "actual_execution_evidence_ready"
            ],
        },
        "lookaheadAudit": {
            "selectionFeaturesPastOnly": True,
            "futureReturnsStoredAsLabelsOnly": True,
            "cooldownUsesStrictlyPriorCompletedDates": True,
            "capacitySelectionIndependentOfFutureReturn": True,
            "rankSelectionIndependentOfFutureReturn": True,
            "sameDayCrossSectionKeptTogether": True,
            "exchangeDecisionTimeDefinesEventMinute": True,
            "sourceQuoteTimesMustNotExceedDecisionMinute": True,
            "targetQuoteMaximumLagMinutes": int(config["data"]["maximumQuoteLagMinutes"]),
            "overlappingSnapshotsThinnedMinutes": int(config["data"]["nonOverlappingEventSpacingMinutes"]),
        },
        "mechanicalReductionAudit": {
            "capacityAddsTwoOrderLegs": True,
            "rankingDoesNotChangeCoverage": True,
            "cooldownRetainedCoverage": cooldown_summary["retained_coverage_if_applied"],
            "cooldownComparedWithNextRankAtEqualCoverage": True,
            "counterfactualRowsCannotPromote": True,
        },
        "verdict": {
            "status": "diagnostic_only" if not any_ready else "shadow_evidence_only",
            "reason": (
                "At least one diagnostic reached its frozen shadow threshold, but actual executed-trade OOS is still required."
                if any_ready
                else "Independent days and broker-confirmed BUY outcomes are insufficient; keep every proposed adjustment out of trading."
            ),
            "productionChangeAllowed": False,
            "paperConfigChangeAllowed": False,
            "buySellGateChangeAllowed": False,
            "positionSizingChangeAllowed": False,
            "automaticPromotionAllowed": False,
        },
    }
    tables = {
        "point_in_time_candidate_events": candidate_events,
        "capacity_rebalance_events": capacity_events,
        "cooldown_symbol_days": cooldown_panel,
        "cooldown_equal_coverage_replacements": cooldown_replacements,
        "current_shadow_cooldown_candidates": cooldown_current,
        "confirmed_buy_entry_weakness_events": weakness_rows,
        "rank_events": rank_events,
        "regime_groups": regime_summary,
    }
    output_dir = write_artifacts(result, tables, resolve(config["output"]["root"]), run_id)
    return result, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    result, output_dir = run(config, args.as_of_date, args.run_id)
    print(render_report(result))
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
