#!/usr/bin/env python3
"""Is the frozen A-share book usable as a tail-loss exclusion screen?

The horizon frontier established that the book has no day-neutral selection edge
at any holding horizon, so it cannot pick winners.  Its own reliability gate
nonetheless passes severe-loss AUC on both windows while failing up-AUC on
shadow.  This module tests whether that asymmetry survives a market-neutral
measurement: does the worst-ranked slice of the cross-section carry a severe-loss
rate materially above the eligible universe on the same day, in both windows?

Research-only.  It measures no portfolio, creates no orders and cannot promote.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
VENDOR_ROOT = ROOT / "scripts" / "vendor" / "vibe_factors"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402
import research_horizon_cost_frontier_v1 as frontier  # noqa: E402
import panel_cache  # noqa: E402


SCHEMA_VERSION = "tail_exclusion_screen_result_v1"
CODE_VERSION = "tail_exclusion_screen_v1_20260905"
DEFAULT_CONFIG = ROOT / "configs" / "research" / "tail_exclusion_screen_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schemaVersion") != "tail_exclusion_screen_v1":
        raise ValueError("unexpected tail-screen schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("tail screen must remain research-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("all tail-screen mutation permissions must be false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("tail-screen output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    for key in (
        "factorDefinitionsDirectionsAndPriorWeightsFrozen",
        "bucketGridFrozenBeforeEvaluation",
        "horizonGridFrozenBeforeEvaluation",
        "parametersFrozenBeforeHistoricalEvaluation",
        "validationAndShadowAreRejectOnly",
    ):
        if hypothesis.get(key) is not True:
            raise ValueError(f"missing preregistration flag: {key}")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("a historical tail screen cannot promote")
    screen = config["screen"]
    buckets = int(screen["bucketCount"])
    if buckets < 4 or buckets > 20:
        raise ValueError("bucket count must stay within 4..20")
    grid = [int(h) for h in screen["holdingTradingDaysGrid"]]
    if not grid or sorted(grid) != grid or len(set(grid)) != len(grid):
        raise ValueError("holding grid must be a strictly increasing unique list")
    evaluation = config["evaluation"]
    if evaluation.get("excessMeasuredAgainstSameDayUniverse") is not True:
        raise ValueError("the tail screen must be measured market-neutrally")
    if evaluation.get("bothWindowsMustAgree") is not True:
        raise ValueError("validation and shadow must both be required to agree")
    if float(evaluation["minimumDayClusteredT"]) < 2.0:
        raise ValueError("the significance bar may not be lowered below t=2")
    if float(evaluation["worstBucketMinimumExcessSevereLossRate"]) <= 0.0:
        raise ValueError("a positive minimum excess severe-loss rate is required")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("tail-screen orders must remain empty")
    frozen = load_json(ROOT / config["basePrecisionConfig"])
    precision.validate_config(frozen)
    return frozen


def bucket_diagnostics(
    score: pd.DataFrame,
    outcome: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
    severe: float,
    buckets: int,
) -> list[dict[str, Any]]:
    """Severe-loss and return behaviour of each score bucket against the same day.

    Buckets run from 1 (highest score, the slice the Top10 is drawn from) to
    ``buckets`` (lowest score, the slice an exclusion screen would drop).
    """
    eligible = execution_eligible.reindex(index=dates).fillna(False)
    ranked = score.reindex(index=dates).where(eligible)
    outcomes = outcome.reindex(index=dates).where(ranked.notna())
    # Descending, so bucket 1 is the best-ranked slice the Top10 comes from and
    # bucket N is the worst-ranked slice an exclusion screen would drop. Ranking
    # ascending here would silently invert the verdict.
    pct = ranked.rank(axis=1, pct=True, ascending=False)
    severe_flag = outcomes.le(severe).where(outcomes.notna())

    universe_severe_daily = severe_flag.mean(axis=1).dropna()
    universe_return_daily = outcomes.mean(axis=1).dropna()

    rows: list[dict[str, Any]] = []
    for index in range(buckets):
        low = index / buckets
        high = (index + 1) / buckets
        in_bucket = pct.le(high) if index == 0 else (pct.gt(low) & pct.le(high))
        bucket_severe = severe_flag.where(in_bucket)
        bucket_returns = outcomes.where(in_bucket)
        observations = int(bucket_returns.notna().to_numpy().sum())
        if observations == 0:
            rows.append(
                {
                    "bucket": index + 1,
                    "observations": 0,
                    "severeLossRate": None,
                    "universeSevereLossRate": None,
                    "excessSevereLossRate": None,
                    "excessSevereLossT": None,
                    "meanExcessReturn": None,
                }
            )
            continue
        severe_daily = bucket_severe.mean(axis=1).dropna()
        return_daily = bucket_returns.mean(axis=1).dropna()
        severe_excess_daily = (
            severe_daily - universe_severe_daily.reindex(severe_daily.index)
        ).dropna()
        return_excess_daily = (
            return_daily - universe_return_daily.reindex(return_daily.index)
        ).dropna()
        stacked = bucket_severe.stack(future_stack=True).dropna()
        rows.append(
            {
                "bucket": index + 1,
                "observations": observations,
                "severeLossRate": float(stacked.mean()) if len(stacked) else None,
                "universeSevereLossRate": float(universe_severe_daily.mean()),
                "excessSevereLossRate": (
                    float(severe_excess_daily.mean()) if len(severe_excess_daily) else None
                ),
                "excessSevereLossT": frontier.day_clustered_t(severe_excess_daily),
                "meanExcessReturn": (
                    float(return_excess_daily.mean()) if len(return_excess_daily) else None
                ),
            }
        )
    return rows


def build_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The worst bucket must be materially and significantly worse in BOTH windows."""
    evaluation = config["evaluation"]
    minimum_excess = float(evaluation["worstBucketMinimumExcessSevereLossRate"])
    minimum_t = float(evaluation["minimumDayClusteredT"])
    buckets = int(config["screen"]["bucketCount"])
    surviving: list[dict[str, Any]] = []
    for entry in report["cells"]:
        if entry["period"] != "validation" or entry["bucket"] != buckets:
            continue
        shadow = next(
            (
                item
                for item in report["cells"]
                if item["period"] == "shadow"
                and item["book"] == entry["book"]
                and item["holdingTradingDays"] == entry["holdingTradingDays"]
                and item["bucket"] == buckets
            ),
            None,
        )
        if shadow is None:
            continue
        values = (
            entry["excessSevereLossRate"],
            entry["excessSevereLossT"],
            shadow["excessSevereLossRate"],
            shadow["excessSevereLossT"],
        )
        if any(value is None for value in values):
            continue
        if (
            entry["excessSevereLossRate"] >= minimum_excess
            and entry["excessSevereLossT"] >= minimum_t
            and shadow["excessSevereLossRate"] >= minimum_excess
            and shadow["excessSevereLossT"] >= minimum_t
        ):
            surviving.append(
                {
                    "book": entry["book"],
                    "holdingTradingDays": entry["holdingTradingDays"],
                    "validationExcessSevereLossRate": entry["excessSevereLossRate"],
                    "validationT": entry["excessSevereLossT"],
                    "validationExcessReturn": entry["meanExcessReturn"],
                    "shadowExcessSevereLossRate": shadow["excessSevereLossRate"],
                    "shadowT": shadow["excessSevereLossT"],
                    "shadowExcessReturn": shadow["meanExcessReturn"],
                }
            )
    passed = bool(surviving)
    return {
        "decision": (
            "retain_tail_exclusion_screen_for_fresh_forward_only"
            if passed
            else "reject_tail_exclusion_screen_no_stable_worst_bucket"
        ),
        "survivingConfigurations": surviving,
        "historicalHypothesisPass": passed,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
    }


def markdown_report(report: dict[str, Any]) -> str:
    buckets = report["bucketCount"]
    lines = [
        "# Tail-loss exclusion screen V1",
        "",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Severe loss: executable gross return <= `{report['data']['severeLossThreshold']}`",
        f"- Buckets: {buckets} deciles of the book score across the eligible cross-section",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`; orders: `[]`",
        "",
        "Everything is measured against the same-day eligible universe, so none of it is beta.",
        f"Bucket {buckets} is the worst-ranked slice, the one an exclusion screen would drop.",
        "",
    ]
    for book in report["books"]:
        for horizon in report["holdingTradingDaysGrid"]:
            lines.append(f"## {book}, holding {horizon} session(s)")
            lines.append("")
            lines.append(
                "| bucket | period | obs | severe loss | universe | excess | excess t | excess return bps |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
            for period in report["periodOrder"]:
                for entry in report["cells"]:
                    if (
                        entry["book"] != book
                        or entry["holdingTradingDays"] != horizon
                        or entry["period"] != period
                    ):
                        continue
                    if entry["bucket"] not in (1, buckets):
                        continue

                    def fmt(value: float | None, scale: float = 1.0, digits: int = 4) -> str:
                        return "n/a" if value is None else f"{value * scale:.{digits}f}"

                    lines.append(
                        f"| {entry['bucket']} | {period} | {entry['observations']} | "
                        f"{fmt(entry['severeLossRate'])} | {fmt(entry['universeSevereLossRate'])} | "
                        f"{fmt(entry['excessSevereLossRate'])} | "
                        f"{fmt(entry['excessSevereLossT'], 1.0, 2)} | "
                        f"{fmt(entry['meanExcessReturn'], 1e4, 2)} |"
                    )
            lines.append("")
    lines.extend(["## Known limitations", ""])
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def run(config_path: Path, run_id: str | None = None, *, paired_inputs: dict | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    frozen, source, frozen_sha = guarded.load_frozen_config(
        {"basePrecisionConfig": config["basePrecisionConfig"]}
    )
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    if paired_inputs is None:
        panel_key, _ = panel_cache.cache_key(base, cog_config)
        panel, panel_audit = panel_cache.build_configured_panel_cached(base, cog_config)
        ranks, _static, factor_audit = panel_cache.build_rank_book_cached(panel, frozen, panel_key)
    else:
        output = ROOT / paired_inputs["output"]
        if not output.resolve().is_relative_to((ROOT / "outputs/edge_research/vwap_basis_retest_v1").resolve()):
            raise ValueError("paired_retest_output_not_isolated")
        output.mkdir(parents=True, exist_ok=False)
        panel, panel_audit, ranks, factor_audit = (paired_inputs[k] for k in ("panel", "panelAudit", "ranks", "factorAudit"))
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    factors = list(ranks.keys())
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    ).reindex(factors).astype(float)
    prior /= prior.sum()

    close_index = panel["close"].index
    books = {
        "frozen_prior": frontier.weight_frame(close_index, factors, prior.to_dict()),
        "equal_weight": frontier.weight_frame(
            close_index, factors, {k: 1.0 / len(factors) for k in factors}
        ),
    }
    requested = list(config["screen"]["books"])
    if "single_factor_each" in requested:
        for key in factors:
            books[f"single/{key}"] = frontier.weight_frame(
                close_index, factors, {key: 1.0}
            )
    selected_books = [name for name in requested if name in books]
    if "single_factor_each" in requested:
        selected_books += [name for name in books if name.startswith("single/")]
    scores = {
        name: guarded.adaptive_score(ranks, books[name], panel)
        for name in selected_books
    }
    if "realized_volatility_20" in requested:
        # The free alternative. If a plain twenty-session realised volatility ranks
        # tail risk as well as the factor composite, #13 is "volatile stocks have fat
        # tails" - true, already known, and not a finding worth a forward record.
        volatility = (
            panel["returns"].rolling(20, min_periods=10).std().where(panel["eligible"])
        )
        # Higher volatility must map to a WORSE score, so the sign is flipped to match
        # the book's convention that a high score is a good name.
        scores["realized_volatility_20"] = -volatility
        selected_books.append("realized_volatility_20")
    print(f"scored_books n={len(scores)}", flush=True)

    severe = float(config["data"]["severeLossThreshold"])
    max_delay = int(config["data"]["maximumExitDelayTradingDays"])
    buckets = int(config["screen"]["bucketCount"])
    grid = [int(h) for h in config["screen"]["holdingTradingDaysGrid"]]
    splits = precision.split_dates(close_index, source)
    period_order = [name for name in ("train", "validation", "shadow") if name in splits]

    cells: list[dict[str, Any]] = []
    for holding_days in grid:
        outcome, execution_eligible, _delay = precision.executable_horizon_return(
            panel, holding_days, max_delay
        )
        if paired_inputs is not None:
            execution_eligible &= paired_inputs["commonSupport"]
        contained = {
            name: precision.contained_signal_dates(dates, holding_days, max_delay)
            for name, dates in splits.items()
        }
        for name in selected_books:
            for period in period_order:
                for row in bucket_diagnostics(
                    scores[name],
                    outcome,
                    execution_eligible,
                    contained[period],
                    severe,
                    buckets,
                ):
                    row["book"] = name
                    row["holdingTradingDays"] = holding_days
                    row["period"] = period
                    cells.append(row)
        print(f"horizon_done h={holding_days}", flush=True)

    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_"
        + now.strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + precision.digest({"config": config, "code": CODE_VERSION})[:10]
    )
    if paired_inputs is None:
        output = ROOT / config["output"]["root"] / run_id
        output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": now.isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path.resolve()),
        "configSha256": precision.digest(config),
        "frozenPrecisionConfigSha256": frozen_sha,
        "dataRange": [
            close_index.min().date().isoformat(),
            close_index.max().date().isoformat(),
        ],
        "data": config["data"],
        "books": selected_books,
        "bucketCount": buckets,
        "holdingTradingDaysGrid": grid,
        "periodOrder": period_order,
        "panelAudit": panel_audit,
        "splitAudit": source["splitAudit"],
        "factorAudit": factor_audit,
        "cells": cells,
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    report["verdict"] = build_verdict(report, config)
    if paired_inputs is not None:
        report["basisRetest"] = paired_inputs["audit"]
        report["originalStudyGateNotConclusion"] = report["verdict"]
        report["verdict"] = {"decision": "reject_only_pending_paired_basis_audit", "eligibleForTrading": False,
                             "historicalHypothesisPass": False}

    precision.atomic_write(
        output / "summary.json",
        precision.canonical(precision.json_safe(report)) + "\n",
    )
    precision.atomic_write(output / "report.md", markdown_report(report))
    precision.atomic_write(
        output / "buckets.csv",
        pd.DataFrame(cells).to_csv(index=False, lineterminator="\n"),
    )
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    report = run(args.config.resolve(), args.run_id)
    print(
        precision.canonical(
            {
                "runId": report["runId"],
                "decision": report["verdict"]["decision"],
                "eligibleForTrading": False,
                "orders": [],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
