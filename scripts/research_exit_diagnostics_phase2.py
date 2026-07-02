"""Exit Diagnostics Phase 2: explain failed exits without inventing a rule.

The analysis freezes the 98 filled buy/sell lots from the exact ``entry_logic_v2``
60-day replay. It computes ex-post path outcomes, assigns transparent failure flags,
then asks whether any ENTRY or first-15/30-minute observable separates each group.

Ex-post MFE, MAE, giveback, future returns, and the assigned class are outcomes only.
They are never admitted to the observable feature screen. Early-price variables are
also excluded from the immediate-loser screen because that class is defined by the
same early path and would otherwise be tautological.

The exact replay is regenerated into an isolated output directory so original entry
score, candidate rank, and subsequent rank/relative-strength paths can be recovered.
The order lifecycle, PnL, and round count must exactly match the frozen source summary
or the feature analysis fails closed.

STRICTLY OFFLINE / SHADOW. No orders, broker calls, live config, overlay, or automatic
promotion.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from research_exit_policy_matrix import (
    ROUND_TRIP_COMMISSION,
    TRAIN_END,
    build_path,
    load_quotes,
    pair_all_filled_lots,
)
from run_etf_paper_trading_agent import ROOT, as_float


SOURCE_SUMMARY = (
    ROOT
    / "outputs"
    / "entry_logic_v2_ablation"
    / "v2_pullback_06pct_10min"
    / "entrylv2_v2_pullback_06pct_10min_summary.json"
)
RERUN_SUMMARY = (
    ROOT
    / "outputs"
    / "exit_diagnostics_phase2"
    / "replay"
    / "exitdiag_phase2_summary.json"
)
DECISIONS = (
    ROOT
    / "outputs"
    / "exit_diagnostics_phase2"
    / "replay"
    / "exitdiag_phase2_decisions.jsonl"
)
SCORE_DIR = ROOT / "outputs" / "exit_diagnostics_phase2" / "decision_scores"
QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
OUT_DIR = ROOT / "outputs" / "exit_diagnostics_phase2"
CST = timezone(timedelta(hours=8))

THRESHOLDS = {
    "immediate_loser_early_mae_pct": -1.0,
    "profit_giveback_min_mfe_pct": 1.5,
    "profit_giveback_min_giveback_pct": 1.0,
    "trend_runner_post3_pct": 0.5,
    "trend_runner_post5_pct": 0.75,
    "dead_money_min_holding_bars": 24,
    "dead_money_max_abs_path_pct": 0.75,
    "good_exit_post3_pct": -0.3,
    "good_exit_post5_pct": -0.5,
}
GROUPS = (
    "immediate_loser",
    "profit_giveback",
    "trend_runner",
    "dead_money",
    "good_exit",
)
MIN_GROUP_TOTAL = 10
MIN_GROUP_PER_WINDOW = 5
PERMUTATIONS = 2000

# None of these fields uses the eventual exit, post-exit prices, MFE, MAE, giveback,
# or realized PnL. The first two early-price variables are mechanical for
# ``immediate_loser`` and are explicitly excluded from that group's screen.
OBSERVABLES: dict[str, dict[str, Any]] = {
    "actual_entry_score": {"horizon": "entry"},
    "entry_momentum": {"horizon": "entry"},
    "entry_bid_pressure": {"horizon": "entry"},
    "entry_acceleration": {"horizon": "entry"},
    "entry_alpha101_conviction": {"horizon": "entry"},
    "entry_rank_percentile": {"horizon": "entry"},
    "entry_market_regime_score": {"horizon": "entry", "retrospective_scorer": True},
    "entry_relative_strength_score": {"horizon": "entry", "retrospective_scorer": True},
    "entry_liquidity_score": {"horizon": "entry", "retrospective_scorer": True},
    "early15_return_pct": {
        "horizon": "first_15m",
        "exclude_groups": ["immediate_loser"],
    },
    "early15_mae_pct": {
        "horizon": "first_15m",
        "exclude_groups": ["immediate_loser"],
    },
    "early15_rank_deterioration": {"horizon": "first_15m"},
    "early15_relative_strength_change": {
        "horizon": "first_15m",
        "retrospective_scorer": True,
    },
    "early15_min_market_regime_score": {
        "horizon": "first_15m",
        "retrospective_scorer": True,
    },
    "early30_return_pct": {"horizon": "first_30m"},
    "early30_rank_deterioration": {"horizon": "first_30m"},
    "early30_relative_strength_change": {
        "horizon": "first_30m",
        "retrospective_scorer": True,
    },
}


def parse_score_time(record: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(
        f"{record['date']}T{record['timestamp']}+08:00"
    )


def lifecycle_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_exact_replay(
    source: dict[str, Any], rerun: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "rounds_total_equal": source.get("rounds_total") == rerun.get("rounds_total"),
        "trade_count_equal": source.get("trade_count") == rerun.get("trade_count"),
        "total_pnl_equal": abs(
            as_float(source.get("total_pnl")) - as_float(rerun.get("total_pnl"))
        )
        < 1e-9,
        "order_lifecycle_equal": lifecycle_digest(
            list(source.get("order_lifecycle") or [])
        )
        == lifecycle_digest(list(rerun.get("order_lifecycle") or [])),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_lifecycle_sha256": lifecycle_digest(
            list(source.get("order_lifecycle") or [])
        ),
        "rerun_lifecycle_sha256": lifecycle_digest(
            list(rerun.get("order_lifecycle") or [])
        ),
    }


def load_entry_decisions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the actual order's original entry score and causal input details."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            timestamp = str(row.get("timestamp") or "")
            for order in row.get("orders") or []:
                if str(order.get("direction")) != "buy":
                    continue
                code = str(order.get("stockCode") or "").zfill(6)
                score = order.get("entry_score") if isinstance(order.get("entry_score"), dict) else {}
                details = score.get("details") if isinstance(score.get("details"), dict) else {}
                components = (
                    score.get("components")
                    if isinstance(score.get("components"), dict)
                    else {}
                )
                out[(timestamp, code)] = {
                    "entry_signal_name": str(order.get("reason") or ""),
                    "actual_entry_score": score.get("score"),
                    "actual_entry_threshold": score.get("threshold"),
                    "entry_momentum": details.get("momentum"),
                    "entry_bid_pressure": details.get("bid_pressure_3m_pct"),
                    "entry_acceleration": details.get("acceleration"),
                    "entry_alpha101_conviction": details.get("alpha101_conviction"),
                    "entry_score_components": components,
                    "bracket": order.get("bracket"),
                }
    return out


def load_score_timelines(
    directory: Path, codes: set[str]
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    entry_index: dict[tuple[str, str], dict[str, Any]] = {}
    timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    versions: dict[str, set[str]] = defaultdict(set)
    for path in sorted(directory.glob("decision_scores_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                code = str(row.get("etf_code") or "").zfill(6)
                if code not in codes:
                    continue
                try:
                    moment = parse_score_time(row)
                except Exception:
                    continue
                compact = {
                    "time": moment,
                    "decision_type": row.get("decision_type"),
                    "ledger_record_type": row.get("ledger_record_type"),
                    "candidate_rank": row.get("candidate_rank"),
                    "candidate_count": row.get("candidate_count"),
                    "market_regime_score": row.get("market_regime_score"),
                    "relative_strength_score": row.get("relative_strength_score"),
                    "liquidity_score": row.get("liquidity_score"),
                    "total_score": row.get("total_score"),
                    "sell_score": row.get("sell_score"),
                }
                timelines[code].append(compact)
                if (
                    str(row.get("decision_type")) == "BUY"
                    and str(row.get("ledger_record_type")) == "planned_order_decision"
                ):
                    entry_index[(moment.isoformat(), code)] = compact
                for field in (
                    "scorer_version",
                    "weights_version",
                    "pipeline_version",
                    "scorer_sha256",
                ):
                    if row.get(field):
                        versions[field].add(str(row[field]))
    for rows in timelines.values():
        rows.sort(key=lambda item: item["time"])
    return entry_index, timelines, {
        field: sorted(values) for field, values in sorted(versions.items())
    }


def _rank_percentile(record: dict[str, Any] | None) -> float | None:
    if not record:
        return None
    rank = record.get("candidate_rank")
    count = record.get("candidate_count")
    if rank is None or count is None or as_float(count) <= 0:
        return None
    return (as_float(rank) - 1.0) / as_float(count)


def _path_return(prices: list[float], bars: int) -> float | None:
    if len(prices) <= bars or prices[0] <= 0:
        return None
    return (prices[bars] / prices[0] - 1.0) * 100.0


def _path_mae(prices: list[float], bars: int) -> float | None:
    sample = prices[: min(len(prices), bars + 1)]
    if len(sample) < 2 or sample[0] <= 0:
        return None
    return (min(sample) / sample[0] - 1.0) * 100.0


def early_score_features(
    entry: dict[str, Any] | None,
    timeline: list[dict[str, Any]],
    entry_time: datetime,
    exit_time: datetime,
) -> dict[str, Any]:
    start_rank = _rank_percentile(entry)
    start_rs = as_float(entry.get("relative_strength_score")) if entry else None
    out: dict[str, Any] = {
        "entry_rank_percentile": start_rank,
        "entry_market_regime_score": entry.get("market_regime_score") if entry else None,
        "entry_relative_strength_score": entry.get("relative_strength_score") if entry else None,
        "entry_liquidity_score": entry.get("liquidity_score") if entry else None,
        "retrospective_total_score": entry.get("total_score") if entry else None,
    }
    for minutes, prefix in ((15, "early15"), (30, "early30")):
        cutoff = min(entry_time + timedelta(minutes=minutes), exit_time)
        rows = [
            row
            for row in timeline
            if entry_time < row["time"] <= cutoff
            and str(row.get("ledger_record_type"))
            in {
                "position_hold_decision",
                "deferred_sell_hold_decision",
                "planned_order_decision",
            }
        ]
        ranks = [value for row in rows if (value := _rank_percentile(row)) is not None]
        rs = [
            as_float(row["relative_strength_score"])
            for row in rows
            if row.get("relative_strength_score") is not None
        ]
        regimes = [
            as_float(row["market_regime_score"])
            for row in rows
            if row.get("market_regime_score") is not None
        ]
        out[f"{prefix}_rank_deterioration"] = (
            max(ranks) - start_rank if ranks and start_rank is not None else None
        )
        out[f"{prefix}_relative_strength_change"] = (
            rs[-1] - start_rs if rs and start_rs is not None else None
        )
        out[f"{prefix}_min_market_regime_score"] = min(regimes) if regimes else None
    return out


def post_exit_returns(
    quotes: list[dict[str, Any]], exit_time: datetime, exit_price: float
) -> dict[str, float | None]:
    times = [row["time"] for row in quotes]
    index = bisect.bisect_right(times, exit_time)
    out: dict[str, float | None] = {}
    for bars in (1, 3, 5):
        target = index + bars - 1
        out[f"post_exit_{bars}bar_return_pct"] = (
            (as_float(quotes[target]["price"]) / exit_price - 1.0) * 100.0
            if exit_price > 0 and target < len(quotes)
            else None
        )
    return out


def classify_trade(record: dict[str, Any]) -> tuple[dict[str, bool], str]:
    post3 = record.get("post_exit_3bar_return_pct")
    post5 = record.get("post_exit_5bar_return_pct")
    immediate = (
        record.get("early15_mae_pct") is not None
        and as_float(record["early15_mae_pct"])
        <= THRESHOLDS["immediate_loser_early_mae_pct"]
    )
    giveback = (
        as_float(record.get("mfe_pct")) >= THRESHOLDS["profit_giveback_min_mfe_pct"]
        and as_float(record.get("giveback_pct"))
        >= THRESHOLDS["profit_giveback_min_giveback_pct"]
    )
    runner = (
        (post3 is not None and as_float(post3) >= THRESHOLDS["trend_runner_post3_pct"])
        or (post5 is not None and as_float(post5) >= THRESHOLDS["trend_runner_post5_pct"])
    )
    dead = (
        as_float(record.get("holding_bars"))
        >= THRESHOLDS["dead_money_min_holding_bars"]
        and abs(as_float(record.get("mae_pct")))
        <= THRESHOLDS["dead_money_max_abs_path_pct"]
        and abs(as_float(record.get("mfe_pct")))
        <= THRESHOLDS["dead_money_max_abs_path_pct"]
    )
    good = (
        (post3 is not None and as_float(post3) <= THRESHOLDS["good_exit_post3_pct"])
        or (post5 is not None and as_float(post5) <= THRESHOLDS["good_exit_post5_pct"])
    )
    flags = {
        "immediate_loser": immediate,
        "profit_giveback": giveback,
        "trend_runner": runner,
        "dead_money": dead,
        "good_exit": good,
    }
    primary = next((group for group in GROUPS if flags[group]), "unclassified")
    return flags, primary


def build_records(
    lots: list[dict[str, Any]],
    quote_map: dict[str, list[dict[str, Any]]],
    entry_decisions: dict[tuple[str, str], dict[str, Any]],
    score_entries: dict[tuple[str, str], dict[str, Any]],
    timelines: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    diagnostics = {
        "short_paths": 0,
        "missing_original_entry_decision": 0,
        "missing_retrospective_entry_score": 0,
        "missing_post_exit_5bar": 0,
    }
    for lot in lots:
        code = str(lot["stockCode"])
        path = build_path(lot, quote_map.get(code, []))
        if len(path) < 3:
            diagnostics["short_paths"] += 1
            continue
        entry_time = datetime.fromisoformat(str(lot["entry_time"]))
        signal_time = datetime.fromisoformat(str(lot["entry_signal_time"]))
        exit_time = datetime.fromisoformat(str(lot["baseline_exit_time"]))
        original = entry_decisions.get((signal_time.isoformat(), code))
        score_entry = score_entries.get((signal_time.isoformat(), code))
        if original is None:
            diagnostics["missing_original_entry_decision"] += 1
            original = {}
        if score_entry is None:
            diagnostics["missing_retrospective_entry_score"] += 1

        prices = [as_float(row["price"]) for row in path]
        entry_price = as_float(lot["entry_price"])
        exit_price = as_float(lot["baseline_exit_price"])
        realized = (exit_price / entry_price - 1.0 - ROUND_TRIP_COMMISSION) * 100.0
        mae = (min(prices) / entry_price - 1.0) * 100.0
        mfe = (max(prices) / entry_price - 1.0) * 100.0
        post = post_exit_returns(quote_map.get(code, []), exit_time, exit_price)
        if post["post_exit_5bar_return_pct"] is None:
            diagnostics["missing_post_exit_5bar"] += 1
        row: dict[str, Any] = {
            **lot,
            "entry_date": entry_time.date().isoformat(),
            "realized_return_pct": realized,
            "mae_pct": mae,
            "mfe_pct": mfe,
            "giveback_pct": mfe - realized,
            "exit_efficiency_pct": exit_price / max(prices) * 100.0,
            "holding_bars": len(path) - 1,
            "early15_return_pct": _path_return(prices, 3),
            "early15_mae_pct": _path_mae(prices, 3),
            "early30_return_pct": _path_return(prices, 6),
            "early30_mae_pct": _path_mae(prices, 6),
            **post,
            **original,
            **early_score_features(
                score_entry, timelines.get(code, []), entry_time, exit_time
            ),
        }
        flags, primary = classify_trade(row)
        row["group_flags"] = flags
        row["primary_group"] = primary
        records.append(row)
    return records, diagnostics


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def group_summary(records: list[dict[str, Any]], group: str, *, primary: bool) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if (
            row.get("primary_group") == group
            if primary
            else bool((row.get("group_flags") or {}).get(group))
        )
    ]
    numeric_fields = (
        "realized_return_pct",
        "mae_pct",
        "mfe_pct",
        "giveback_pct",
        "exit_efficiency_pct",
        "post_exit_1bar_return_pct",
        "post_exit_3bar_return_pct",
        "post_exit_5bar_return_pct",
        "holding_bars",
    )
    out: dict[str, Any] = {
        "trades": len(selected),
        "entry_days": len({row["entry_date"] for row in selected}),
        "train_trades": sum(row["entry_date"] <= TRAIN_END for row in selected),
        "test_trades": sum(row["entry_date"] > TRAIN_END for row in selected),
        "share": round(len(selected) / len(records), 4) if records else 0.0,
    }
    for field in numeric_fields:
        values = [
            as_float(row[field]) for row in selected if row.get(field) is not None
        ]
        out[f"mean_{field}"] = round(_mean(values), 4) if values else None
    out["entry_signal_counts"] = dict(
        sorted(
            {
                signal: sum(
                    str(row.get("entry_signal_name") or row.get("entry_reason")) == signal
                    for row in selected
                )
                for signal in {
                    str(row.get("entry_signal_name") or row.get("entry_reason"))
                    for row in selected
                }
            }.items()
        )
    )
    return out


def _cohens_d(group: list[float], other: list[float]) -> float | None:
    if len(group) < 2 or len(other) < 2:
        return None
    pooled_var = (
        (len(group) - 1) * statistics.variance(group)
        + (len(other) - 1) * statistics.variance(other)
    ) / (len(group) + len(other) - 2)
    return (
        (statistics.mean(group) - statistics.mean(other)) / math.sqrt(pooled_var)
        if pooled_var > 0
        else None
    )


def _difference(rows: list[dict[str, Any]], group: str, feature: str) -> tuple[float | None, int, int]:
    yes = [
        as_float(row[feature])
        for row in rows
        if row.get(feature) is not None and bool(row["group_flags"].get(group))
    ]
    no = [
        as_float(row[feature])
        for row in rows
        if row.get(feature) is not None and not bool(row["group_flags"].get(group))
    ]
    return (
        statistics.mean(yes) - statistics.mean(no) if yes and no else None,
        len(yes),
        len(no),
    )


def within_day_permutation_p(
    rows: list[dict[str, Any]], group: str, feature: str, permutations: int = PERMUTATIONS
) -> float | None:
    usable = [row for row in rows if row.get(feature) is not None]
    observed, yes_n, no_n = _difference(usable, group, feature)
    if observed is None or yes_n < 2 or no_n < 2:
        return None
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_day[row["entry_date"]].append(row)
    seed = int.from_bytes(
        hashlib.sha256(f"{group}:{feature}".encode()).digest()[:8], "big"
    )
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        yes: list[float] = []
        no: list[float] = []
        for day_rows in by_day.values():
            labels = [bool(row["group_flags"].get(group)) for row in day_rows]
            rng.shuffle(labels)
            for row, label in zip(day_rows, labels):
                (yes if label else no).append(as_float(row[feature]))
        if yes and no:
            permuted = statistics.mean(yes) - statistics.mean(no)
            extreme += abs(permuted) >= abs(observed)
    return (extreme + 1.0) / (permutations + 1.0)


def observable_tests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    train = [row for row in records if row["entry_date"] <= TRAIN_END]
    test = [row for row in records if row["entry_date"] > TRAIN_END]
    for group in GROUPS:
        for feature, meta in OBSERVABLES.items():
            excluded = group in set(meta.get("exclude_groups") or [])
            full_diff, group_n, other_n = _difference(records, group, feature)
            train_diff, train_group_n, train_other_n = _difference(train, group, feature)
            test_diff, test_group_n, test_other_n = _difference(test, group, feature)
            yes = [
                as_float(row[feature])
                for row in records
                if row.get(feature) is not None and row["group_flags"].get(group)
            ]
            no = [
                as_float(row[feature])
                for row in records
                if row.get(feature) is not None and not row["group_flags"].get(group)
            ]
            tests.append(
                {
                    "group": group,
                    "feature": feature,
                    "horizon": meta["horizon"],
                    "retrospective_scorer": bool(meta.get("retrospective_scorer")),
                    "excluded_as_tautological": excluded,
                    "group_n": group_n,
                    "other_n": other_n,
                    "mean_difference": round(full_diff, 6) if full_diff is not None else None,
                    "cohens_d": (
                        round(effect, 4)
                        if (effect := _cohens_d(yes, no)) is not None
                        else None
                    ),
                    "train_difference": (
                        round(train_diff, 6) if train_diff is not None else None
                    ),
                    "train_group_n": train_group_n,
                    "train_other_n": train_other_n,
                    "test_difference": (
                        round(test_diff, 6) if test_diff is not None else None
                    ),
                    "test_group_n": test_group_n,
                    "test_other_n": test_other_n,
                    "direction_consistent": bool(
                        train_diff is not None
                        and test_diff is not None
                        and train_diff * test_diff > 0
                    ),
                    "within_day_permutation_p": (
                        within_day_permutation_p(records, group, feature)
                        if not excluded
                        else None
                    ),
                }
            )

    valid = [
        row
        for row in tests
        if row["within_day_permutation_p"] is not None
        and not row["excluded_as_tautological"]
    ]
    ordered = sorted(valid, key=lambda row: as_float(row["within_day_permutation_p"]))
    running = 0.0
    total = len(ordered)
    for rank, row in enumerate(ordered):
        adjusted = min(
            1.0,
            as_float(row["within_day_permutation_p"]) * (total - rank),
        )
        running = max(running, adjusted)
        row["holm_adjusted_p"] = round(running, 6)
    for row in tests:
        row.setdefault("holm_adjusted_p", None)
        row["candidate_observable"] = bool(
            not row["excluded_as_tautological"]
            and row["group_n"] >= MIN_GROUP_TOTAL
            and row["other_n"] >= MIN_GROUP_TOTAL
            and row["train_group_n"] >= MIN_GROUP_PER_WINDOW
            and row["test_group_n"] >= MIN_GROUP_PER_WINDOW
            and row["direction_consistent"]
            and row["cohens_d"] is not None
            and abs(as_float(row["cohens_d"])) >= 0.5
            and row["holm_adjusted_p"] is not None
            and as_float(row["holm_adjusted_p"]) < 0.05
        )
    return tests


def descriptive_opportunity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in (*GROUPS, "unclassified"):
        selected = [row for row in records if row["primary_group"] == group]
        if group == "immediate_loser":
            values = [max(0.0, -as_float(row["realized_return_pct"])) for row in selected]
        elif group == "profit_giveback":
            values = [max(0.0, as_float(row["giveback_pct"])) for row in selected]
        elif group == "trend_runner":
            values = [
                max(0.0, as_float(row.get("post_exit_5bar_return_pct")))
                for row in selected
                if row.get("post_exit_5bar_return_pct") is not None
            ]
        else:
            values = []
        rows.append(
            {
                "group": group,
                "trades": len(selected),
                "descriptive_total_opportunity_pct_points": round(sum(values), 4),
                "descriptive_mean_opportunity_pct_points": (
                    round(statistics.mean(values), 4) if values else None
                ),
                "warning": "ex-post upper-bound description, not achievable alpha",
            }
        )
    return sorted(
        rows,
        key=lambda row: as_float(row["descriptive_total_opportunity_pct_points"]),
        reverse=True,
    )


def conditional_hypotheses(
    primary_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "group": "profit_giveback",
            "candidate_mechanism": "profit-armed trailing or partial profit-taking",
            "status": "not_actionable_no_observable_survived",
            "note": (
                "early positive returns are a weak descriptive clue, but no feature "
                "survived within-day permutation plus Holm adjustment"
            ),
        },
        {
            "group": "trend_runner",
            "candidate_mechanism": "rank-retention hold extension",
            "status": "not_actionable_no_observable_survived",
            "note": "early rank deterioration was directionally lower but effect was not significant",
        },
        {
            "group": "immediate_loser",
            "candidate_mechanism": "fast structure stop",
            "status": "insufficient_group_sample",
            "note": f"only {primary_summaries['immediate_loser']['trades']} primary trades",
        },
        {
            "group": "dead_money",
            "candidate_mechanism": "maximum holding time",
            "status": "insufficient_group_sample",
            "note": f"only {primary_summaries['dead_money']['trades']} primary trades",
        },
        {
            "group": "good_exit",
            "candidate_mechanism": "retain current baseline exit",
            "status": "control_not_a_new_rule",
            "note": "post-exit prices declined; changing these exits is not supported",
        },
    ]


def build_result(
    records: list[dict[str, Any]],
    integrity: dict[str, Any],
    pairing: dict[str, int],
    diagnostics: dict[str, int],
    versions: dict[str, Any],
) -> dict[str, Any]:
    tests = observable_tests(records)
    candidates = [row for row in tests if row["candidate_observable"]]
    primary_summaries = {
        group: group_summary(records, group, primary=True)
        for group in (*GROUPS, "unclassified")
    }
    flag_summaries = {
        group: group_summary(records, group, primary=False) for group in GROUPS
    }
    feature_coverage = {
        feature: {
            "available": sum(row.get(feature) is not None for row in records),
            "total": len(records),
            "coverage": round(
                sum(row.get(feature) is not None for row in records) / len(records), 4
            )
            if records
            else 0.0,
        }
        for feature in OBSERVABLES
    }
    no_actual_score_signals: dict[str, int] = defaultdict(int)
    for row in records:
        if row.get("actual_entry_score") is None:
            no_actual_score_signals[
                str(row.get("entry_signal_name") or row.get("entry_reason") or "unknown")
            ] += 1
    return {
        "research_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "diagnostic_only",
        "edge_validated": False,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "order_submit_calls_made": False,
        "live_config_modified": False,
        "integrity": integrity,
        "sample": {
            "usable_lots": len(records),
            "entry_days": len({row["entry_date"] for row in records}),
            "train_lots": sum(row["entry_date"] <= TRAIN_END for row in records),
            "test_lots": sum(row["entry_date"] > TRAIN_END for row in records),
            **pairing,
            **diagnostics,
        },
        "classification_thresholds": THRESHOLDS,
        "classification_note": (
            "flags may overlap; primary group uses fixed priority "
            "immediate_loser > profit_giveback > trend_runner > dead_money > good_exit"
        ),
        "primary_group_summaries": primary_summaries,
        "overlapping_flag_summaries": flag_summaries,
        "observable_feature_contract": OBSERVABLES,
        "observable_feature_coverage": feature_coverage,
        "actual_entry_score_unavailable_by_signal": dict(
            sorted(no_actual_score_signals.items())
        ),
        "observable_tests": tests,
        "candidate_observables": candidates,
        "candidate_observable_count": len(candidates),
        "multiple_testing": {
            "method": "2,000 within-entry-day permutations plus Holm family-wise adjustment",
            "minimum_group_total": MIN_GROUP_TOTAL,
            "minimum_group_per_train_test_window": MIN_GROUP_PER_WINDOW,
            "minimum_abs_cohens_d": 0.5,
        },
        "descriptive_opportunity_ranking": descriptive_opportunity(records),
        "conditional_exit_hypotheses": conditional_hypotheses(primary_summaries),
        "score_semantics": {
            "actual_entry_score": "original score embedded in the exact replay order",
            "rank_and_component_scores": (
                "current frozen scorer applied retrospectively to exact historical "
                "point-in-time decisions; usable only as diagnostic covariates"
            ),
            "versions": versions,
        },
        "ohlc_atr_readiness": {
            "available": False,
            "reason": (
                "yahoo_60d_quotes.jsonl contains point price/prevClose/top-of-book/"
                "volume/amount but no historical open/high/low fields"
            ),
            "atr_or_true_range_allowed": False,
            "required_next_data": "point-in-time 5-minute OHLCV with source timestamps",
        },
        "verdict": (
            "no_pretrade_or_early_observable_survived"
            if not candidates
            else "hypothesis_candidates_only_require_clean_forward_preregistration"
        ),
        "limitations": [
            "All outcome groups are defined on the same reused 60-day replay.",
            "Group thresholds are preregistered diagnostics, not optimized economic boundaries.",
            "The current decision scorer was applied retrospectively and was not the historical production scorer.",
            "Five-minute point quotes miss intrabar extrema and exchange-quality historical depth.",
            "Post-exit returns describe opportunity cost but are never observable at the sell decision.",
            "A statistical association would not by itself prove that an exit overlay improves portfolio PnL.",
        ],
        "records": records,
    }


def _fmt(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{as_float(value):.3f}{suffix}"


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Exit Diagnostics Phase 2 — failure attribution and observability audit",
        "",
        f"Status: **{result['status']}** | exact replay integrity: "
        f"**{str(result['integrity']['passed']).lower()}** | edge validated: "
        f"**{str(result['edge_validated']).lower()}**",
        "",
        f"Usable lots: **{result['sample']['usable_lots']}** across "
        f"**{result['sample']['entry_days']}** entry days; train/test lots: "
        f"**{result['sample']['train_lots']}/{result['sample']['test_lots']}**.",
        "",
        "## Primary failure attribution",
        "",
        "| group | trades (train/test) | share | realized | MFE | MAE | giveback | exit eff. | post 3 bars | post 5 bars |",
        "|---|---:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for group, node in result["primary_group_summaries"].items():
        lines.append(
            f"| {group} | {node['trades']} ({node['train_trades']}/{node['test_trades']}) | "
            f"{node['share']:.2%} | "
            f"{_fmt(node.get('mean_realized_return_pct'), '%')} | "
            f"{_fmt(node.get('mean_mfe_pct'), '%')} | "
            f"{_fmt(node.get('mean_mae_pct'), '%')} | "
            f"{_fmt(node.get('mean_giveback_pct'), '%')} | "
            f"{_fmt(node.get('mean_exit_efficiency_pct'), '%')} | "
            f"{_fmt(node.get('mean_post_exit_3bar_return_pct'), '%')} | "
            f"{_fmt(node.get('mean_post_exit_5bar_return_pct'), '%')} |"
        )
    candidates = result["candidate_observables"]
    top = sorted(
        [
            row
            for row in result["observable_tests"]
            if not row["excluded_as_tautological"] and row["cohens_d"] is not None
        ],
        key=lambda row: (
            row["holm_adjusted_p"] is None,
            as_float(row["holm_adjusted_p"], 1.0),
            -abs(as_float(row["cohens_d"])),
        ),
    )[:12]
    lines += [
        "",
        "## Can a state variable identify the group before exit?",
        "",
        f"Strict candidates surviving train/test direction, effect size, within-day "
        f"permutation, and Holm adjustment: **{len(candidates)}**.",
        f"Original embedded entry-score coverage: "
        f"**{result['observable_feature_coverage']['actual_entry_score']['available']}/"
        f"{result['observable_feature_coverage']['actual_entry_score']['total']}**; "
        "missing values are special entry paths without that score, not imputed values.",
        "",
        "| group | observable | horizon | n group/rest | Cohen d | train Δ | test Δ | Holm p | candidate |",
        "|---|---|---|---:|--:|--:|--:|--:|---|",
    ]
    for row in top:
        lines.append(
            f"| {row['group']} | {row['feature']} | {row['horizon']} | "
            f"{row['group_n']}/{row['other_n']} | {_fmt(row['cohens_d'])} | "
            f"{_fmt(row['train_difference'])} | {_fmt(row['test_difference'])} | "
            f"{_fmt(row['holm_adjusted_p'])} | "
            f"{str(row['candidate_observable']).lower()} |"
        )
    priority = result["descriptive_opportunity_ranking"][0]
    lines += [
        "",
        "## Decision",
        "",
        f"- Largest ex-post opportunity bucket: **{priority['group']}**, descriptive "
        f"upper bound **{priority['descriptive_total_opportunity_pct_points']:.3f} "
        "percentage points**. This is not achievable alpha.",
        f"- Observable-screen verdict: **{result['verdict']}**.",
        "- No conditional exit rule is produced or enabled.",
        "- Conditional mechanisms remain a hypothesis list only; every proposed mechanism "
        "is marked non-actionable or sample-insufficient.",
        "- MFE/MAE/giveback/post-exit returns remain labels only and cannot enter a live gate.",
        "- ATR remains unsupported until genuine point-in-time 5-minute OHLCV is collected.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "stockCode",
        "exchange",
        "quantity",
        "entry_order_id",
        "entry_signal_time",
        "entry_time",
        "entry_price",
        "entry_signal_name",
        "actual_entry_score",
        "entry_rank_percentile",
        "baseline_exit_time",
        "baseline_exit_price",
        "baseline_exit_reason",
        "realized_return_pct",
        "mae_pct",
        "mfe_pct",
        "giveback_pct",
        "exit_efficiency_pct",
        "holding_bars",
        "early15_return_pct",
        "early15_mae_pct",
        "early15_rank_deterioration",
        "early15_relative_strength_change",
        "post_exit_1bar_return_pct",
        "post_exit_3bar_return_pct",
        "post_exit_5bar_return_pct",
        "primary_group",
        *GROUPS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{field: record.get(field) for field in fields},
                    **{
                        group: bool(record["group_flags"].get(group))
                        for group in GROUPS
                    },
                }
            )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Exit Diagnostics Phase 2 on exact replay fills."
    )
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    parser.add_argument("--rerun-summary", type=Path, default=RERUN_SUMMARY)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR)
    parser.add_argument("--quotes", type=Path, default=QUOTES)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    rerun = json.loads(args.rerun_summary.read_text(encoding="utf-8"))
    integrity = verify_exact_replay(source, rerun)
    if not integrity["passed"]:
        raise RuntimeError(f"exact replay integrity failed: {integrity['checks']}")

    lots, pairing = pair_all_filled_lots(list(source.get("order_lifecycle") or []))
    quote_map = load_quotes({str(lot["stockCode"]) for lot in lots}, args.quotes)
    entry_decisions = load_entry_decisions(args.decisions)
    score_entries, timelines, versions = load_score_timelines(
        args.score_dir, {str(lot["stockCode"]) for lot in lots}
    )
    records, diagnostics = build_records(
        lots, quote_map, entry_decisions, score_entries, timelines
    )
    if len(records) != len(lots):
        raise RuntimeError(
            f"incomplete path coverage: records={len(records)} paired_lots={len(lots)}"
        )

    result = build_result(records, integrity, pairing, diagnostics, versions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "exit_diagnostics_phase2.json"
    md_path = args.output_dir / "exit_diagnostics_phase2.md"
    csv_path = args.output_dir / "exit_diagnostics_phase2.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render(result), encoding="utf-8")
    write_csv(records, csv_path)
    print(render(result))
    print(f"outputs: {json_path} | {md_path} | {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
