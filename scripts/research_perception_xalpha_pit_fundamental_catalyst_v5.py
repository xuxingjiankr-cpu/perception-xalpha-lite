#!/usr/bin/env python3
"""Causal PIT fundamental-catalyst research for the all-A-share panel.

This study deliberately separates *newly disclosed information* from the slow level
of a financial statement.  A filing is first visible on the first market date strictly
after max(NOTICE_DATE, UPDATE_DATE), the signal is formed after that session closes,
and a hypothetical entry is attempted at the next buyable open.  The primary policy
uses a fixed 75% catalyst-change rank plus 25% of the already frozen twelve-factor
price/volume timing rank.  Outcomes can only reject the preregistered hypothesis.

The module is permanently research/shadow-only.  It cannot call a broker, create an
order, modify a trading configuration, or feed a production decision function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_ashare_fundamentals as fundamentals  # noqa: E402
import research_ashare_universe as ashare  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402


SCHEMA_VERSION = "perception_xalpha_pit_fundamental_catalyst_result_v5"
CODE_VERSION = "perception_xalpha_pit_fundamental_catalyst_v5_20260807"
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "perception_xalpha_pit_fundamental_catalyst_v5.json"
)

GROUP_FIELDS = {
    "growth": ("revenueYoyPct", "netProfitYoyPct"),
    "quality": ("roePct", "grossMarginPct", "netMarginPct"),
    "cash": ("operatingCashToNetProfit",),
    "safety": ("debtAssetRatioPct",),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_pit_fundamental_catalyst_v5":
        raise ValueError("unexpected PIT fundamental catalyst schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("fundamental catalyst research must remain research-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every mutation and trading permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("primaryPolicy") != "catalyst_plus_frozen_timing":
        raise ValueError("the preregistered primary policy cannot change")
    if hypothesis.get("validationAndShadowMayOnlyReject") is not True:
        raise ValueError("external windows must remain reject-only")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical research cannot promote")
    if int(hypothesis.get("countsAsNewResearchTrials", 0)) != 4:
        raise ValueError("the four declared policies must pay four research trials")
    event = config["fundamentalEvents"]
    if event.get("availabilityRule") != (
        "first_market_date_strictly_after_max_notice_update"
    ):
        raise ValueError("fundamental availability must remain strictly causal")
    if event.get("sameAvailabilityDateRule") != "latest_report_date_only":
        raise ValueError("same-day historical statement dumps must be collapsed")
    if event.get("lateRestatementRule") != "skip_non_advancing_report_date":
        raise ValueError("unreconstructable late restatements must fail closed")
    if event.get("requiresPriorDisclosedReport") is not True:
        raise ValueError("a change signal requires a prior disclosed report")
    if event.get("requiresGrowthComponent") is not True:
        raise ValueError("this catalyst hypothesis requires a growth component")
    weights = {str(key): float(value) for key, value in event["groupWeights"].items()}
    if set(weights) != set(GROUP_FIELDS) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("fundamental group weights must cover four groups and sum to one")
    for key in ("changeScales", "levelScales"):
        scales = event[key]
        required = {field for fields in GROUP_FIELDS.values() for field in fields}
        if set(scales) != required or any(float(value) <= 0.0 for value in scales.values()):
            raise ValueError(f"{key} must contain positive frozen scales for every field")
    selection = config["selection"]
    if int(selection["topCount"]) != 10:
        raise ValueError("the primary book must remain Top10")
    if not selection.get("allowCash") or not selection.get("neverForceSelections"):
        raise ValueError("the study must allow an empty book")
    if abs(
        float(selection["catalystWeight"])
        + float(selection["frozenTimingWeight"])
        - 1.0
    ) > 1e-9:
        raise ValueError("primary score weights must sum to one")
    if int(selection["permutationPlacebos"]) < 100:
        raise ValueError("a single placebo draw is not a valid control")
    data = config["data"]
    if int(data["holdingTradingDays"]) != 5:
        raise ValueError("the preregistered holding horizon is five sessions")
    if data.get("periodLocalOutcomeContainment") is not True:
        raise ValueError("every outcome must remain within its evaluation period")
    if not math.isclose(float(data["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("A-share round-trip cost must remain 30 bps")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("research outputs must remain isolated under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must always be empty")


def verify_frozen_file(config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(config[path_key])
    actual = file_sha256(path)
    expected = str(config[hash_key]).lower()
    if actual != expected:
        raise ValueError(f"frozen input changed: {path_key} {actual} != {expected}")
    return path


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _group_values(
    current: dict[str, Any],
    previous: dict[str, Any],
    scales: dict[str, float],
    change: bool,
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for group, fields in GROUP_FIELDS.items():
        values: list[float] = []
        for field in fields:
            now = _finite(current.get(field))
            before = _finite(previous.get(field))
            if now is None or (change and before is None):
                continue
            raw = (now - before) if change else now
            # Lower leverage is better; all other fields are oriented upward.
            if field == "debtAssetRatioPct":
                raw = -raw
            values.append(math.tanh(raw / float(scales[field])))
        output[group] = _mean(values)
    return output


def _weighted_group_score(
    groups: dict[str, float | None], weights: dict[str, float]
) -> float | None:
    present = {key: value for key, value in groups.items() if value is not None}
    denominator = sum(float(weights[key]) for key in present)
    if denominator <= 0.0:
        return None
    return sum(float(weights[key]) * float(value) for key, value in present.items()) / denominator


def causal_event_records_for_symbol(
    rows: list[dict[str, Any]],
    market_index: pd.DatetimeIndex,
    security_id: str,
    event_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create issuer-change events without allowing a future filing into an old row."""
    audit: Counter[str] = Counter()
    by_market_date: dict[pd.Timestamp, dict[str, Any]] = {}
    for raw in rows:
        notice = _date(raw.get("noticeDate"))
        update = _date(raw.get("updateDate"))
        report = _date(raw.get("reportDate"))
        if notice is None or report is None:
            audit["missing_required_date"] += 1
            continue
        available = notice if update is None else max(notice, update)
        position = int(market_index.searchsorted(available, side="right"))
        if position >= len(market_index):
            audit["available_after_panel"] += 1
            continue
        market_date = pd.Timestamp(market_index[position])
        candidate = dict(raw)
        candidate["_reportDate"] = report
        candidate["_availableRaw"] = available
        incumbent = by_market_date.get(market_date)
        if incumbent is None or report > incumbent["_reportDate"]:
            if incumbent is not None:
                audit["same_day_older_report_collapsed"] += 1
            by_market_date[market_date] = candidate
        else:
            audit["same_day_older_report_collapsed"] += 1

    records: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    weights = {
        str(key): float(value) for key, value in event_config["groupWeights"].items()
    }
    change_scales = {
        str(key): float(value) for key, value in event_config["changeScales"].items()
    }
    level_scales = {
        str(key): float(value) for key, value in event_config["levelScales"].items()
    }
    for market_date, current in sorted(by_market_date.items()):
        if previous is None:
            previous = current
            audit["first_report_no_prior"] += 1
            continue
        if current["_reportDate"] <= previous["_reportDate"]:
            audit["non_advancing_report_date_skipped"] += 1
            continue
        change_groups = _group_values(current, previous, change_scales, change=True)
        level_groups = _group_values(current, previous, level_scales, change=False)
        available_groups = sum(value is not None for value in change_groups.values())
        growth_available = change_groups.get("growth") is not None
        change_score = _weighted_group_score(change_groups, weights)
        level_score = _weighted_group_score(level_groups, weights)
        if not growth_available:
            audit["missing_growth_group"] += 1
        elif available_groups < int(event_config["minimumAvailableGroups"]):
            audit["insufficient_component_groups"] += 1
        elif change_score is None or level_score is None:
            audit["score_not_finite"] += 1
        else:
            records.append(
                {
                    "eventDate": market_date,
                    "securityId": security_id,
                    "reportDate": current["_reportDate"],
                    "previousReportDate": previous["_reportDate"],
                    "reportType": str(current.get("reportType") or "unknown"),
                    "changeScoreRaw": float(change_score),
                    "levelScoreRaw": float(level_score),
                    "availableGroups": int(available_groups),
                    **{
                        f"change_{key}": value
                        for key, value in change_groups.items()
                    },
                }
            )
            audit["accepted_events"] += 1
        # Only an advancing, causally available report becomes the comparison base.
        previous = current
    return records, dict(audit)


def build_event_frames(
    market_index: pd.DatetimeIndex,
    columns: pd.Index,
    event_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = ROOT / str(event_config["root"])
    all_records: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    files_found = 0
    for position, security_id in enumerate(map(str, columns), start=1):
        exchange, code = security_id.split(".", 1)
        path = root / f"{exchange}_{code}.jsonl"
        rows = fundamentals.read_jsonl(path)
        if not rows:
            totals["missing_symbol_file"] += 1
            continue
        files_found += 1
        records, audit = causal_event_records_for_symbol(
            rows, market_index, security_id, event_config
        )
        all_records.extend(records)
        totals.update(audit)
        if position % 500 == 0:
            print(
                f"fundamental_events {position}/{len(columns)} accepted={len(all_records)}",
                flush=True,
            )
    table = pd.DataFrame(all_records)
    empty = pd.DataFrame(index=market_index, columns=columns, dtype=float)
    if table.empty:
        return empty, empty.copy(), pd.DataFrame(), {
            "filesFound": files_found,
            "records": 0,
            "rejections": dict(totals),
        }
    # There must be at most one issuer event on a safe market date after collapsing.
    duplicates = int(table.duplicated(["eventDate", "securityId"]).sum())
    if duplicates:
        raise RuntimeError("event table contains duplicate issuer-date rows")
    change = (
        table.pivot(index="eventDate", columns="securityId", values="changeScoreRaw")
        .reindex(index=market_index, columns=columns)
    )
    level = (
        table.pivot(index="eventDate", columns="securityId", values="levelScoreRaw")
        .reindex(index=market_index, columns=columns)
    )
    audit = {
        "schemaVersion": "pit_fundamental_event_audit_v1",
        "status": "research_only",
        "filesFound": files_found,
        "symbolsRequested": len(columns),
        "symbolFileCoverage": round(files_found / max(1, len(columns)), 8),
        "acceptedEvents": len(table),
        "positiveCatalystEvents": int(
            table["changeScoreRaw"].gt(
                float(event_config["positiveCatalystThreshold"])
            ).sum()
        ),
        "eventDateStart": str(pd.Timestamp(table["eventDate"].min()).date()),
        "eventDateEnd": str(pd.Timestamp(table["eventDate"].max()).date()),
        "uniqueEventDates": int(table["eventDate"].nunique()),
        "reportTypeCounts": {
            str(key): int(value)
            for key, value in table["reportType"].value_counts().items()
        },
        "processingCounts": dict(totals),
        "availabilityRule": event_config["availabilityRule"],
        "sameAvailabilityDateRule": event_config["sameAvailabilityDateRule"],
        "lateRestatementRule": event_config["lateRestatementRule"],
        "futureRowsWrittenBack": 0,
    }
    return change, level, table, audit


def daily_rank(signal: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    return signal.where(candidate).rank(axis=1, pct=True, method="average")


def select_top(
    score: pd.DataFrame, candidate: pd.DataFrame, top_count: int
) -> pd.DataFrame:
    ranks = score.where(candidate).rank(axis=1, ascending=False, method="first")
    return ranks.le(top_count).fillna(False)


def daily_net_series(
    selected: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cost: float,
) -> pd.Series:
    gross = returns.reindex(index=dates).where(
        selected.reindex(index=dates).fillna(False)
    ).mean(axis=1, skipna=True)
    return gross.dropna() - float(cost)


def daily_ic_stats(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    candidate: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lag: int,
) -> dict[str, Any]:
    ic = (
        score.reindex(index=dates)
        .where(candidate.reindex(index=dates).fillna(False))
        .corrwith(returns.reindex(index=dates), axis=1, method="spearman")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    hac = autonomous.newey_west_t(ic.to_numpy(dtype=float), lag) if len(ic) >= 20 else None
    return {
        "days": len(ic),
        "meanSpearmanIc": round(float(ic.mean()), 8) if len(ic) else None,
        "medianSpearmanIc": round(float(ic.median()), 8) if len(ic) else None,
        "positiveIcRate": round(float(ic.gt(0.0).mean()), 8) if len(ic) else None,
        "hacT": round(float(hac), 4) if hac is not None else None,
        "hacLag": lag,
    }


def paired_policy_delta(
    primary: pd.Series, counter: pd.Series, lag: int
) -> dict[str, Any]:
    joined = pd.concat([primary.rename("primary"), counter.rename("counter")], axis=1).dropna()
    delta = joined["primary"] - joined["counter"]
    hac = autonomous.newey_west_t(delta.to_numpy(dtype=float), lag) if len(delta) >= 20 else None
    return {
        "pairedDays": len(delta),
        "meanDailyNetDelta": round(float(delta.mean()), 8) if len(delta) else None,
        "positiveDeltaRate": round(float(delta.gt(0.0).mean()), 8) if len(delta) else None,
        "hacT": round(float(hac), 4) if hac is not None else None,
        "hacLag": lag,
    }


def permutation_placebo(
    candidate: pd.DataFrame,
    selected: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    observed_daily_net: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    draws = int(config["selection"]["permutationPlacebos"])
    seed = int(config["selection"]["permutationSeed"])
    cost = float(config["data"]["roundTripCost"])
    rng = np.random.default_rng(seed)
    pools: list[np.ndarray] = []
    counts: list[int] = []
    used_dates: list[pd.Timestamp] = []
    for date in dates:
        available = (
            candidate.loc[date]
            & returns.loc[date].notna()
        )
        values = returns.loc[date, available].to_numpy(dtype=float)
        count = int((selected.loc[date] & returns.loc[date].notna()).sum())
        if count <= 0 or len(values) < count:
            continue
        pools.append(values)
        counts.append(count)
        used_dates.append(pd.Timestamp(date))
    if not pools:
        return {"draws": draws, "days": 0, "empiricalPValue": None}
    placebo_means: list[float] = []
    placebo_win_rates: list[float] = []
    for _ in range(draws):
        daily = np.asarray(
            [float(rng.choice(pool, size=count, replace=False).mean()) - cost
             for pool, count in zip(pools, counts)],
            dtype=float,
        )
        placebo_means.append(float(daily.mean()))
        placebo_win_rates.append(float((daily > 0.0).mean()))
    observed = observed_daily_net.reindex(used_dates).dropna()
    observed_mean = float(observed.mean()) if len(observed) else float("nan")
    exceed = sum(value >= observed_mean for value in placebo_means)
    return {
        "draws": draws,
        "days": len(pools),
        "seed": seed,
        "observedDailyMeanNetReturn": (
            round(observed_mean, 8) if math.isfinite(observed_mean) else None
        ),
        "placeboMeanOfDailyMeanNetReturn": round(float(np.mean(placebo_means)), 8),
        "placeboP90DailyMeanNetReturn": round(float(np.quantile(placebo_means, 0.90)), 8),
        "placeboMeanDailyNetWinRate": round(float(np.mean(placebo_win_rates)), 8),
        "empiricalPValue": round((exceed + 1) / (draws + 1), 8),
        "sameCandidatePoolAndDailySelectionCount": True,
    }


def split_dates(
    index: pd.DatetimeIndex, source: dict[str, Any]
) -> dict[str, pd.DatetimeIndex]:
    audit = source["splitAudit"]
    ranges = {
        "train": audit["train"][:2],
        "validation": audit["validation"][:2],
        "shadow": audit["shadowQuarantine"][:2],
    }
    return {
        name: pd.DatetimeIndex(
            index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
        )
        for name, (start, end) in ranges.items()
    }


def period_event_audit(
    table: pd.DataFrame, dates: pd.DatetimeIndex, threshold: float
) -> dict[str, Any]:
    if table.empty or not len(dates):
        return {"events": 0, "positiveEvents": 0, "eventDays": 0}
    period = table[table["eventDate"].isin(dates)]
    return {
        "events": len(period),
        "positiveEvents": int(period["changeScoreRaw"].gt(threshold).sum()),
        "eventDays": int(period["eventDate"].nunique()),
        "medianEventsPerEventDay": (
            round(float(period.groupby("eventDate").size().median()), 4)
            if len(period)
            else None
        ),
    }


def month_metrics(
    daily: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if daily.empty:
        return rows
    for month, values in daily.groupby(daily.index.to_period("M")):
        rows.append(
            {
                "month": str(month),
                "days": len(values),
                "netWinRate": round(float(values.gt(0.0).mean()), 8),
                "meanNetReturn": round(float(values.mean()), 8),
            }
        )
    return rows


def build_verdict(
    metrics: dict[str, dict[str, dict[str, Any]]],
    placebos: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = config["preregisteredHypothesis"]["primaryPolicy"]
    evaluation = config["evaluation"]
    checks: dict[str, Any] = {}
    for period in ("validation", "shadow"):
        row = metrics[period][primary]["performance"]
        level = metrics[period]["fundamental_level_counter"]["performance"]
        timing = metrics[period]["frozen_timing_counter"]["performance"]
        permutation_p = placebos[period].get("empiricalPValue")
        severe = row.get("stockSevereLossRate")
        timing_severe = timing.get("stockSevereLossRate")
        values = {
            "minimumSignalDays": row["signalDays"]
            >= int(evaluation["minimumExternalSignalDays"]),
            "resolvedFraction": (row.get("resolvedFraction") or 0.0)
            >= float(evaluation["minimumResolvedFraction"]),
            "dailyNetWinRate": (row.get("dailyNetWinRate") or 0.0)
            > float(evaluation["minimumDailyNetWinRate"]),
            "dailyMeanNetReturn": (row.get("dailyMeanNetReturn") or -math.inf)
            > float(evaluation["minimumDailyMeanNetReturn"]),
            "beatsLevelCounter": row.get("dailyMeanNetReturn") is not None
            and level.get("dailyMeanNetReturn") is not None
            and row["dailyMeanNetReturn"] > level["dailyMeanNetReturn"],
            "beatsTimingCounter": row.get("dailyMeanNetReturn") is not None
            and timing.get("dailyMeanNetReturn") is not None
            and row["dailyMeanNetReturn"] > timing["dailyMeanNetReturn"],
            "severeLossNotHigherThanTiming": severe is not None
            and timing_severe is not None
            and severe <= timing_severe,
            "permutationPValue": permutation_p is not None
            and permutation_p <= float(evaluation["maximumPermutationPValue"]),
        }
        values["passed"] = all(values.values())
        checks[period] = values
    historical_pass = all(checks[period]["passed"] for period in checks)
    return {
        "decision": (
            "keep_as_separately_preregistered_fresh_forward_hypothesis_only"
            if historical_pass
            else "reject_for_trading_keep_research_diagnostics"
        ),
        "historicalHypothesisPass": historical_pass,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
        "checks": checks,
        "validationAndShadowWereRejectOnly": True,
        "automaticTradingChanges": [],
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# PIT fundamental catalyst V5",
        "",
        "> **research-only / shadow-only / not a trade signal.**",
        "",
        f"- run_id: `{result['runId']}`",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- universe: `{result['dataAudit']['symbols']}` PIT SH/SZ stocks",
        f"- event records: `{result['eventAudit']['acceptedEvents']}` "
        f"(positive `{result['eventAudit']['positiveCatalystEvents']}`)",
        "- signal / execution: safe post-disclosure close -> next buyable open -> "
        "five-session sellable open",
        f"- verdict: `{result['verdict']['decision']}`",
        "- orders: `[]`",
        "",
        "## Policy comparison",
        "",
        "| period | policy | days | daily net win | daily mean net | stock severe loss | IC | permutation p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("train", "validation", "shadow"):
        for policy, row in result["periodMetrics"][period].items():
            perf = row["performance"]
            ic = row["ic"]
            placebo = result["permutationPlacebos"].get(period, {}) if policy == result["primaryPolicy"] else {}
            def pct(value: Any) -> str:
                return "n/a" if value is None else f"{100 * float(value):.3f}%"
            lines.append(
                f"| {period} | {policy} | {perf['signalDays']} | "
                f"{pct(perf['dailyNetWinRate'])} | {pct(perf['dailyMeanNetReturn'])} | "
                f"{pct(perf['stockSevereLossRate'])} | "
                f"{ic['meanSpearmanIc'] if ic['meanSpearmanIc'] is not None else 'n/a'} | "
                f"{placebo.get('empiricalPValue', 'n/a')} |"
            )
    lines.extend(
        [
            "",
            "## Causal and interpretation audit",
            "",
            "- Filing availability is strictly after `max(noticeDate, updateDate)`; "
            "labels never enter the feature table.",
            "- Multiple historical reports exposed on one date collapse to the latest "
            "report date; non-advancing late restatements are skipped.",
            "- All four policies use the same positive-event candidate set and Top10 cap, "
            "so any difference is not created by reducing the number of selections.",
            "- The permutation control keeps the same candidate pool and daily selection "
            "count. A low p-value is required before calling the ranking informative.",
            "- Liquidity-bucket neutralisation is causal but is not a substitute for "
            "historical industry neutralisation.",
            "- Validation and shadow dates were previously viewed. Even a pass would only "
            "justify a fresh-forward preregistration, never trading.",
            "",
            "## Safety boundary",
            "",
            "No trading config, overlay, order, position, risk gate, sizing rule, broker "
            "path or production decision function was read or changed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    base_path = verify_frozen_file(
        config, "baseResearchConfig", "baseResearchConfigFileSha256"
    )
    timing_path = verify_frozen_file(
        config, "frozenTimingConfig", "frozenTimingConfigFileSha256"
    )
    split_path = verify_frozen_file(
        config, "frozenSplitSummary", "frozenSplitSummaryFileSha256"
    )
    base_config = load_json(base_path)
    perception.validate_config(base_config)
    _, cog_config = perception.load_base_configs(base_config)
    # Build only the clean price/status panel here. The event loader below reads the
    # required statement fields sparsely and therefore does not allocate 22 ffilled
    # fundamental matrices merely to calculate seven disclosure deltas.
    panel, panel_audit = ashare.build_panel(base_config["assetUniverse"], cog_config["data"])
    if not panel_audit.get("unbiasedHistoricalValidationEligible", False):
        raise RuntimeError("clean PIT adjusted panel failed the unbiased-data gate")
    close = panel["close"]
    print(f"panel_ready days={len(close.index)} symbols={len(close.columns)}", flush=True)
    timing_config = load_json(timing_path)
    precision.validate_config(timing_config)
    _, timing_source_hash = precision.verify_frozen_source(timing_config)
    split_source = load_json(split_path)
    change_raw, level_raw, event_table, event_audit = build_event_frames(
        close.index, close.columns, config["fundamentalEvents"]
    )
    if event_audit.get("symbolFileCoverage", 0.0) < 0.95:
        raise RuntimeError("fundamental event file coverage is below 95%")
    threshold = float(config["fundamentalEvents"]["positiveCatalystThreshold"])
    positive = change_raw.gt(threshold) & panel["eligible"]
    bins = int(config["fundamentalEvents"]["liquidityNeutraliseBins"])
    change_neutral = autonomous.size_neutralise(change_raw.where(positive), panel, bins)
    level_neutral = autonomous.size_neutralise(level_raw.where(positive), panel, bins)
    timing_score, _, timing_audit = precision.compute_frozen_scores(panel, timing_config)
    returns, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        holding_days=int(config["data"]["holdingTradingDays"]),
        maximum_exit_delay=int(config["data"]["maximumExitDelayTradingDays"]),
    )
    common_candidate = (
        positive
        & execution_eligible
        & change_neutral.notna()
        & level_neutral.notna()
        & timing_score.notna()
    )
    change_rank = daily_rank(change_neutral, common_candidate)
    level_rank = daily_rank(level_neutral, common_candidate)
    timing_rank = daily_rank(timing_score, common_candidate)
    selection = config["selection"]
    combined = (
        change_rank * float(selection["catalystWeight"])
        + timing_rank * float(selection["frozenTimingWeight"])
    )
    scores = {
        "catalyst_plus_frozen_timing": combined,
        "catalyst_change_only": change_rank,
        "fundamental_level_counter": level_rank,
        "frozen_timing_counter": timing_rank,
    }
    selected = {
        name: select_top(score, common_candidate, int(selection["topCount"]))
        for name, score in scores.items()
    }
    # Every arm ranks exactly the same candidate pool and selects the same count per day.
    daily_counts = {
        name: frame.sum(axis=1).astype(int) for name, frame in selected.items()
    }
    reference_count = daily_counts["catalyst_plus_frozen_timing"]
    if any(not values.equals(reference_count) for values in daily_counts.values()):
        raise RuntimeError("policy selection counts differ despite common candidate pool")
    splits = split_dates(close.index, split_source)
    holding = int(config["data"]["holdingTradingDays"])
    delay = int(config["data"]["maximumExitDelayTradingDays"])
    cost = float(config["data"]["roundTripCost"])
    period_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    daily_series: dict[str, dict[str, pd.Series]] = {}
    placebos: dict[str, dict[str, Any]] = {}
    event_periods: dict[str, dict[str, Any]] = {}
    for period, raw_dates in splits.items():
        dates = precision.contained_signal_dates(raw_dates, holding, delay)
        event_periods[period] = period_event_audit(event_table, dates, threshold)
        period_metrics[period] = {}
        daily_series[period] = {}
        for name, score in scores.items():
            performance = precision.summarize_selection(
                selected[name], returns, exit_delay, dates, config, holding
            )
            ic = daily_ic_stats(score, returns, common_candidate, dates, holding - 1)
            daily = daily_net_series(selected[name], returns, dates, cost)
            daily_series[period][name] = daily
            period_metrics[period][name] = {
                "performance": performance,
                "ic": ic,
                "monthly": month_metrics(daily),
            }
        primary = config["preregisteredHypothesis"]["primaryPolicy"]
        placebos[period] = permutation_placebo(
            common_candidate,
            selected[primary],
            returns,
            dates,
            daily_series[period][primary],
            config,
        )
        for counter in config["preregisteredHypothesis"]["counterPolicies"]:
            period_metrics[period][primary].setdefault("pairedDeltas", {})[counter] = (
                paired_policy_delta(
                    daily_series[period][primary],
                    daily_series[period][counter],
                    holding - 1,
                )
            )
    verdict = build_verdict(period_metrics, placebos, config)
    primary = config["preregisteredHypothesis"]["primaryPolicy"]
    generated = datetime.now(timezone.utc)
    resolved_run_id = run_id or (
        "run_" + generated.strftime("%Y%m%dT%H%M%SZ") + "_pit_fundamental_catalyst_v5"
    )
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": resolved_run_id,
        "generatedAt": generated.isoformat(),
        "codeVersion": CODE_VERSION,
        "configSha256": file_sha256(config_path),
        "frozenInputs": {
            "baseResearchConfigSha256": file_sha256(base_path),
            "frozenTimingConfigSha256": file_sha256(timing_path),
            "frozenTimingSourceSha256": timing_source_hash,
            "frozenSplitSummarySha256": file_sha256(split_path),
        },
        "dataAudit": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "days": len(close.index),
            "symbols": len(close.columns),
            "barInterval": "1d",
            "pointInTimeMembership": panel_audit.get("pointInTimeMembership"),
            "adjustedPrices": not bool(panel_audit.get("rawPricesUnadjusted", True)),
            "historicalStatus": panel_audit.get("historicalStatusAvailable"),
            "unbiasedHistoricalValidationEligible": panel_audit.get(
                "unbiasedHistoricalValidationEligible"
            ),
        },
        "splitAudit": {
            name: [str(dates.min().date()), str(dates.max().date()), len(dates)]
            if len(dates)
            else [None, None, 0]
            for name, dates in splits.items()
        },
        "eventAudit": event_audit,
        "eventPeriods": event_periods,
        "primaryPolicy": primary,
        "policyDefinitions": {
            "catalyst_plus_frozen_timing": {
                "catalystWeight": float(selection["catalystWeight"]),
                "frozenTimingWeight": float(selection["frozenTimingWeight"]),
            },
            "catalyst_change_only": "issuer change versus prior causally available report",
            "fundamental_level_counter": "same filing's absolute fundamental level",
            "frozen_timing_counter": "same candidate pool ranked by frozen twelve-factor score",
        },
        "selectionAudit": {
            "sameCandidatePoolEveryPolicy": True,
            "sameDailySelectionCountEveryPolicy": True,
            "topCount": int(selection["topCount"]),
            "candidateDays": int(common_candidate.any(axis=1).sum()),
            "candidateObservations": int(common_candidate.sum().sum()),
            "meanCandidatesPerCandidateDay": round(
                float(common_candidate.sum(axis=1).replace(0, np.nan).mean()), 8
            ),
            "timingFactorCount": timing_audit["factorCount"],
            "liquidityNeutraliseBins": bins,
            "reducedTradingMechanicalImprovement": False,
        },
        "periodMetrics": period_metrics,
        "permutationPlacebos": placebos,
        "verdict": verdict,
        "lookaheadAudit": {
            "featuresPastOnly": True,
            "strictPostDisclosureAvailability": True,
            "labelsStoredOutsideFeatures": True,
            "periodLocalOutcomeContainment": True,
            "futureRowsWrittenBack": 0,
            "restatementVintageLimitation": config["knownLimitations"][0],
        },
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    result = json_safe(result)
    output = ROOT / config["output"]["root"] / resolved_run_id
    atomic_write(
        output / "summary.json",
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write(output / "report.md", render_report(result))
    atomic_write(
        output / "event_audit.json",
        json.dumps(json_safe(event_audit), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    print(json.dumps({
        "runId": resolved_run_id,
        "output": str(output),
        "verdict": verdict["decision"],
        "historicalHypothesisPass": verdict["historicalHypothesisPass"],
        "orders": [],
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(config_path, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
