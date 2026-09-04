#!/usr/bin/env python3
"""Holding-horizon by weighting-book cost frontier for the frozen A-share Top10 book.

Reweighting the book moves gross return inside a band that never reaches the round-trip
cost line.  This module varies the one axis that has never been varied -- how long the
position is held -- because the round-trip cost is paid once no matter how long the
holding period is.  Research-only: it cannot trade, promote or alter any configuration.
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
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402


SCHEMA_VERSION = "horizon_cost_frontier_result_v1"
CODE_VERSION = "horizon_cost_frontier_v1_20260904"
DEFAULT_CONFIG = ROOT / "configs" / "research" / "horizon_cost_frontier_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schemaVersion") != "horizon_cost_frontier_v1":
        raise ValueError("unexpected horizon-frontier schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("horizon frontier must remain research-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("all horizon-frontier mutation permissions must be false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("horizon-frontier output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    for key in (
        "factorDefinitionsDirectionsAndPriorWeightsFrozen",
        "horizonGridFrozenBeforeEvaluation",
        "weightingBooksFrozenBeforeEvaluation",
        "parametersFrozenBeforeHistoricalEvaluation",
        "validationAndShadowAreRejectOnly",
    ):
        if hypothesis.get(key) is not True:
            raise ValueError(f"missing preregistration flag: {key}")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("a historical horizon sweep cannot promote")
    grid = [int(h) for h in config["sweep"]["holdingTradingDaysGrid"]]
    if not grid or sorted(grid) != grid or len(set(grid)) != len(grid):
        raise ValueError("holding grid must be a strictly increasing unique list")
    if grid[0] < 1 or grid[-1] > 60:
        raise ValueError("holding grid must stay within 1..60 sessions")
    if float(config["data"]["roundTripCostAssumption"]) <= 0.0:
        raise ValueError("a positive round-trip cost assumption is required")
    if int(config["data"]["topCount"]) != 10:
        raise ValueError("the frozen selection target must remain Top10")
    evaluation = config["evaluation"]
    for key in (
        "perPickAndPerHoldingDayBothRequired",
        "dayClusteredTRequired",
        "noAbstentionAllowed",
        "sameTop10CountRequired",
        "historicalWindowsAlreadyViewed",
    ):
        if evaluation.get(key) is not True:
            raise ValueError(f"evaluation flag must remain true: {key}")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("horizon-frontier orders must remain empty")
    base_path = ROOT / config["basePrecisionConfig"]
    frozen = load_json(base_path)
    precision.validate_config(frozen)
    return frozen


def weight_frame(
    dates: pd.DatetimeIndex, factors: list[str], values: dict[str, float]
) -> pd.DataFrame:
    """Constant per-factor weights broadcast over every signal date."""
    return pd.DataFrame(
        {key: float(values.get(key, 0.0)) for key in factors},
        index=dates,
        dtype=float,
    )


def weighting_books(
    dates: pd.DatetimeIndex, factors: list[str], prior: pd.Series
) -> dict[str, pd.DataFrame]:
    books: dict[str, pd.DataFrame] = {}
    books["frozen_prior"] = weight_frame(dates, factors, prior.to_dict())
    equal = 1.0 / float(len(factors))
    books["equal_weight"] = weight_frame(dates, factors, {k: equal for k in factors})
    for key in factors:
        books[f"single/{key}"] = weight_frame(dates, factors, {key: 1.0})
    return books


def day_clustered_t(daily_mean: pd.Series) -> float | None:
    """t on the mean of the per-day means; each trading day is one cluster."""
    values = daily_mean.dropna().to_numpy(dtype=float)
    if values.size < 3:
        return None
    standard_error = float(values.std(ddof=1)) / float(np.sqrt(values.size))
    # A constant series has no dispersion, but summing equal floats leaves ~1e-18 of
    # rounding noise, so an absolute > 0 guard would report an astronomical t instead
    # of refusing.  Compare against the scale of the data.
    scale = float(np.abs(values).mean())
    if standard_error <= max(1e-15, scale * 1e-12):
        return None
    return float(values.mean() / standard_error)


def summarise_book(
    mask: pd.DataFrame,
    outcome: pd.DataFrame,
    dates: pd.DatetimeIndex,
    holding_days: int,
    cost: float,
    severe: float,
) -> dict[str, Any]:
    selected = outcome.reindex(index=dates).where(mask.reindex(index=dates).fillna(False))
    stacked = selected.stack(future_stack=True).dropna()
    picks = int(len(stacked))
    if picks == 0:
        return {
            "picks": 0,
            "signalDays": 0,
            "meanGrossPerPick": None,
            "meanNetPerPick": None,
            "meanNetPerHoldingDay": None,
            "upRate": None,
            "severeLossRate": None,
            "dayClusteredT": None,
        }
    daily_mean = selected.mean(axis=1, skipna=True).dropna()
    gross = float(stacked.mean())
    net = gross - cost
    return {
        "picks": picks,
        "signalDays": int(len(daily_mean)),
        "meanGrossPerPick": gross,
        "meanNetPerPick": net,
        "meanNetPerHoldingDay": net / float(holding_days),
        "upRate": float(stacked.gt(0.0).mean()),
        "severeLossRate": float(stacked.le(severe).mean()),
        "dayClusteredT": day_clustered_t(daily_mean),
    }


def markdown_report(report: dict[str, Any]) -> str:
    data = report["data"]
    cost_bps = float(data["roundTripCostAssumption"]) * 1e4
    lines = [
        "# Holding-horizon x weighting cost frontier V1",
        "",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Round-trip cost assumption: **{cost_bps:.1f} bps**, charged once per round trip",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`; orders: `[]`",
        "",
        "Net per pick is gross minus the full round-trip cost. Net per holding day divides",
        "that by the holding horizon, which is the only comparable unit across horizons.",
        "",
    ]
    for period in report["periodOrder"]:
        lines.append(f"## {period}")
        lines.append("")
        lines.append(
            "| horizon | book | picks | gross bps | net bps | net bps/day | up rate | severe loss | day-clustered t |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in report["periods"][period]["headline"]:
            gross = row["meanGrossPerPick"]
            net = row["meanNetPerPick"]
            per_day = row["meanNetPerHoldingDay"]
            lines.append(
                f"| {row['holdingTradingDays']} | {row['book']} | {row['picks']} | "
                f"{'n/a' if gross is None else f'{gross * 1e4:.2f}'} | "
                f"{'n/a' if net is None else f'{net * 1e4:.2f}'} | "
                f"{'n/a' if per_day is None else f'{per_day * 1e4:.2f}'} | "
                f"{'n/a' if row['upRate'] is None else f'{row['upRate']:.4f}'} | "
                f"{'n/a' if row['severeLossRate'] is None else f'{row['severeLossRate']:.4f}'} | "
                f"{'n/a' if row['dayClusteredT'] is None else f'{row['dayClusteredT']:.2f}'} |"
            )
        lines.append("")
    lines.extend(["## Single-factor distribution (ex-post, not a recommendation)", ""])
    lines.append("| horizon | best gross bps | median gross bps | worst gross bps | frozen gross bps | equal gross bps |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in report["singleFactorDistribution"]:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value * 1e4:.2f}"
        lines.append(
            f"| {row['holdingTradingDays']} | {fmt(row['best'])} | {fmt(row['median'])} | "
            f"{fmt(row['worst'])} | {fmt(row['frozen'])} | {fmt(row['equal'])} |"
        )
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def build_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Pass only if some horizon clears cost on BOTH validation and shadow."""
    clearing: list[dict[str, Any]] = []
    for row in report["periods"]["validation"]["headline"]:
        if row["book"] not in {"frozen_prior", "equal_weight"}:
            continue
        net = row["meanNetPerPick"]
        if net is None or net <= 0.0:
            continue
        shadow = next(
            (
                item
                for item in report["periods"]["shadow"]["headline"]
                if item["book"] == row["book"]
                and item["holdingTradingDays"] == row["holdingTradingDays"]
            ),
            None,
        )
        if shadow is None:
            continue
        shadow_net = shadow["meanNetPerPick"]
        if shadow_net is None or shadow_net <= 0.0:
            continue
        clearing.append(
            {
                "book": row["book"],
                "holdingTradingDays": row["holdingTradingDays"],
                "validationNetPerPick": net,
                "shadowNetPerPick": shadow_net,
            }
        )
    passed = bool(clearing)
    return {
        "decision": (
            "retain_horizon_candidates_for_fresh_forward_only"
            if passed
            else "reject_all_horizons_no_configuration_clears_cost"
        ),
        "horizonsClearingCostOnBothWindows": clearing,
        "historicalHypothesisPass": passed,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
    }


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    frozen, source, frozen_sha = guarded.load_frozen_config(
        {"basePrecisionConfig": config["basePrecisionConfig"]}
    )
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )

    ranks, _static_score, factor_audit = rolling.compute_rank_book(panel, frozen)
    factors = list(ranks.keys())
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    prior = prior.reindex(factors).astype(float)
    prior /= prior.sum()

    close_index = panel["close"].index
    books = weighting_books(close_index, factors, prior)
    scores = {
        name: guarded.adaptive_score(ranks, weights, panel)
        for name, weights in books.items()
    }
    print(f"scored_books n={len(scores)}", flush=True)

    data = config["data"]
    cost = float(data["roundTripCostAssumption"])
    severe = float(data["severeLossThreshold"])
    top_count = int(data["topCount"])
    max_delay = int(data["maximumExitDelayTradingDays"])
    support = pd.DataFrame(0, index=close_index, columns=panel["close"].columns)
    splits = precision.split_dates(close_index, source)
    period_order = [name for name in ("train", "validation", "shadow") if name in splits]

    periods: dict[str, dict[str, Any]] = {name: {"headline": []} for name in period_order}
    single_distribution: list[dict[str, Any]] = []

    for holding_days in [int(h) for h in config["sweep"]["holdingTradingDaysGrid"]]:
        outcome, execution_eligible, _delay = precision.executable_horizon_return(
            panel, holding_days, max_delay
        )
        contained = {
            name: precision.contained_signal_dates(dates, holding_days, max_delay)
            for name, dates in splits.items()
        }
        masks = {
            name: precision.selection_mask(
                score, support, execution_eligible, 0, top_count
            )
            for name, score in scores.items()
        }
        for period in period_order:
            dates = contained[period]
            for name in books:
                row = summarise_book(
                    masks[name], outcome, dates, holding_days, cost, severe
                )
                row["book"] = name
                row["holdingTradingDays"] = holding_days
                periods[period]["headline"].append(row)
        shadow_rows = [
            row
            for row in periods[period_order[-1]]["headline"]
            if row["holdingTradingDays"] == holding_days
        ]
        singles = [
            row["meanGrossPerPick"]
            for row in shadow_rows
            if row["book"].startswith("single/") and row["meanGrossPerPick"] is not None
        ]
        frozen_row = next(
            (row for row in shadow_rows if row["book"] == "frozen_prior"), None
        )
        equal_row = next(
            (row for row in shadow_rows if row["book"] == "equal_weight"), None
        )
        single_distribution.append(
            {
                "holdingTradingDays": holding_days,
                "best": max(singles) if singles else None,
                "median": float(np.median(singles)) if singles else None,
                "worst": min(singles) if singles else None,
                "frozen": frozen_row["meanGrossPerPick"] if frozen_row else None,
                "equal": equal_row["meanGrossPerPick"] if equal_row else None,
            }
        )
        print(f"horizon_done h={holding_days}", flush=True)

    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_"
        + now.strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + precision.digest({"config": config, "code": CODE_VERSION})[:10]
    )
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
        "data": data,
        "panelAudit": panel_audit,
        "splitAudit": source["splitAudit"],
        "factorAudit": factor_audit,
        "periodOrder": period_order,
        "periods": periods,
        "singleFactorDistribution": single_distribution,
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    report["verdict"] = build_verdict(report)

    precision.atomic_write(
        output / "summary.json",
        precision.canonical(precision.json_safe(report)) + "\n",
    )
    precision.atomic_write(output / "report.md", markdown_report(report))
    rows = [
        {"period": period, **row}
        for period in period_order
        for row in periods[period]["headline"]
    ]
    precision.atomic_write(
        output / "frontier.csv",
        pd.DataFrame(rows).to_csv(index=False, lineterminator="\n"),
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
