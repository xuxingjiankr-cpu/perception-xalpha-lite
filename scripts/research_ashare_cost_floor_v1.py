#!/usr/bin/env python3
"""What does the A-share Top10 book actually cost to trade?

RESEARCH_LOG #12 showed the entire reachable gross band of this book is 7.5 to
13.8 bps per pick against a round-trip cost of 0.003 written into the configs as
a flat constant. That constant was never measured, and it alone decides the sign
of the conclusion. The ETF line measured its own half-spread empirically; the
stock line never did.

This measures it three ways and keeps them separate: an exact tick-size floor, a
Corwin-Schultz high-low estimate, and Amihud as an ordering check. It then prices
the picks the book actually makes, rather than the average name.

Research-only. It creates no orders and cannot promote anything.
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


SCHEMA_VERSION = "ashare_cost_floor_result_v1"
CODE_VERSION = "ashare_cost_floor_v1_20260913"
DEFAULT_CONFIG = ROOT / "configs" / "research" / "ashare_cost_floor_v1.json"
ROOT_TWO = 3.0 - 2.0 * np.sqrt(2.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schemaVersion") != "ashare_cost_floor_v1":
        raise ValueError("unexpected cost-floor schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("the cost study must remain research-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("all cost-study mutation permissions must be false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("cost-study output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    for key in (
        "factorDefinitionsDirectionsAndPriorWeightsFrozen",
        "costEstimatorsFrozenBeforeEvaluation",
        "feeParametersAreInputsNotFindings",
        "parametersFrozenBeforeHistoricalEvaluation",
        "validationAndShadowAreRejectOnly",
    ):
        if hypothesis.get(key) is not True:
            raise ValueError(f"missing preregistration flag: {key}")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("a historical cost study cannot promote")
    price = config["price"]
    if price.get("actualPriceSource") != "exchange_reported_amount_divided_by_volume":
        raise ValueError(
            "relative tick size must be computed from the unadjusted traded price"
        )
    if float(price["tickSizeCny"]) <= 0.0:
        raise ValueError("a positive tick size is required")
    if config["estimators"]["relativeTickFloor"].get("exact") is not True:
        raise ValueError("the tick floor must stay labelled exact")
    if config["estimators"]["corwinSchultz"].get("exact") is not False:
        raise ValueError("Corwin-Schultz must stay labelled an estimate")
    if config["fees"].get("confirmBeforeUse") is not True:
        raise ValueError("fee parameters must stay marked for confirmation")
    if config["evaluation"].get("reportFloorAndEstimateSeparately") is not True:
        raise ValueError("the exact floor and the estimate may not be merged")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("cost-study orders must remain empty")
    frozen = load_json(ROOT / config["basePrecisionConfig"])
    precision.validate_config(frozen)
    return frozen


def actual_traded_price(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unadjusted average traded price: exchange cash over shares.

    The panel is backward adjusted, so adjusted close cannot be used for anything
    denominated in real yuan. amount/volume is the one unadjusted price the archive
    still carries. It is the same quantity that is wrong as a vwap factor input
    against adjusted OHLC, and exactly right here.
    """
    volume = panel["volume"].replace(0.0, np.nan)
    return panel["amount"].div(volume)


def relative_tick_floor_bps(price: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Hard lower bound on round-trip spread cost: one full tick, in bps.

    A round trip crosses the spread twice, so even a one-tick-wide book costs a
    whole tick. Exact given the tick size; never an expected cost.
    """
    return price.rdiv(tick).mul(1e4)


def corwin_schultz_bps(
    high: pd.DataFrame, low: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Corwin-Schultz (2012) two-day high-low proportional spread, in bps.

    Uses only the high/low RATIO. Backward adjustment is multiplicative and so
    leaves that ratio invariant, which is why the adjusted panel is a valid input
    even though its price level is not real yuan.
    """
    safe_low = low.where(low > 0.0)
    single = np.log(high.div(safe_low)) ** 2
    beta = single.add(single.shift(1))
    two_day_high = high.rolling(2).max()
    two_day_low = safe_low.rolling(2).min()
    gamma = np.log(two_day_high.div(two_day_low)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / ROOT_TWO - np.sqrt(
        gamma.div(ROOT_TWO)
    )
    spread = (np.exp(alpha) - 1.0).div(np.exp(alpha) + 1.0).mul(2.0)
    # Negative estimates are noise, not negative spreads; the standard treatment
    # floors them at zero, which biases the estimator UP toward zero for the
    # least liquid names rather than inventing a free trade.
    spread = spread.where(spread > 0.0, 0.0)
    return spread.rolling(window, min_periods=max(5, window // 2)).mean().mul(1e4)


def amihud_illiquidity(
    returns: pd.DataFrame, amount: pd.DataFrame, window: int
) -> pd.DataFrame:
    scaled = amount.replace(0.0, np.nan)
    return returns.abs().div(scaled).rolling(window, min_periods=window // 2).mean()


def total_round_trip_bps(
    spread_bps: pd.DataFrame, commission_bps: float, stamp_bps: float
) -> pd.DataFrame:
    """Spread crossed twice is already in spread_bps; fees are added on top."""
    return spread_bps.add(2.0 * float(commission_bps) + float(stamp_bps))


def weighted_stat(frame: pd.DataFrame, mask: pd.DataFrame) -> float | None:
    values = frame.where(mask).stack(future_stack=True).dropna()
    return float(values.mean()) if len(values) else None


def summarise_selection(
    mask: pd.DataFrame,
    dates: pd.DatetimeIndex,
    outcome: pd.DataFrame,
    tick_bps: pd.DataFrame,
    cs_bps: pd.DataFrame,
    price: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Price the picks the book actually makes, not the average name."""
    selected = mask.reindex(index=dates).fillna(False)
    gross = outcome.reindex(index=dates).where(selected)
    stacked = gross.stack(future_stack=True).dropna()
    if not len(stacked):
        return {"picks": 0}
    fees = config["fees"]
    commission = float(fees["commissionBpsPerSide"])
    stamp = float(fees["stampDutyBpsSellSideOnly"])
    assumed_bps = float(config["evaluation"]["assumedRoundTripCost"]) * 1e4
    gross_bps = float(stacked.mean()) * 1e4
    floor_bps = weighted_stat(tick_bps.reindex(index=dates), selected)
    estimate_bps = weighted_stat(cs_bps.reindex(index=dates), selected)
    row: dict[str, Any] = {
        "picks": int(len(stacked)),
        "medianActualPriceCny": weighted_stat(price.reindex(index=dates), selected),
        "grossBps": gross_bps,
        "assumedRoundTripBps": assumed_bps,
        "tickFloorSpreadBps": floor_bps,
        "corwinSchultzSpreadBps": estimate_bps,
    }
    if floor_bps is not None:
        row["totalUsingExactFloorBps"] = floor_bps + 2.0 * commission + stamp
        row["netUsingExactFloorBps"] = gross_bps - row["totalUsingExactFloorBps"]
    if estimate_bps is not None:
        row["totalUsingEstimateBps"] = estimate_bps + 2.0 * commission + stamp
        row["netUsingEstimateBps"] = gross_bps - row["totalUsingEstimateBps"]
    row["netUsingAssumedBps"] = gross_bps - assumed_bps
    grid = []
    for c in fees["sensitivityGridCommissionBps"]:
        for s in fees["sensitivityGridStampBps"]:
            if floor_bps is None:
                continue
            grid.append(
                {
                    "commissionBpsPerSide": c,
                    "stampBps": s,
                    "totalFloorBps": floor_bps + 2.0 * float(c) + float(s),
                    "netAtFloorBps": gross_bps - (floor_bps + 2.0 * float(c) + float(s)),
                }
            )
    row["feeSensitivity"] = grid
    return row


def liquidity_buckets(
    amount: pd.DataFrame, buckets: int, window: int
) -> pd.DataFrame:
    trailing = amount.rolling(window, min_periods=window // 2).median()
    return trailing.rank(axis=1, pct=True)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# A-share round-trip cost floor V1",
        "",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Assumed cost in the configs: **{report['assumedRoundTripBps']:.1f} bps**",
        f"- Fee inputs (confirm): commission {report['fees']['commissionBpsPerSide']} bps/side,"
        f" stamp {report['fees']['stampDutyBpsSellSideOnly']} bps sell-side",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`; orders: `[]`",
        "",
        "The tick floor is EXACT given the tick size and is a hard lower bound.",
        "Corwin-Schultz is an ESTIMATE with no surviving calibration anchor.",
        "",
        "## Universe by liquidity decile",
        "",
        "| decile | median price CNY | tick floor bps | CS spread bps |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["liquidityDeciles"]:
        def fmt(value: float | None, digits: int = 2) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {row['decile']} | {fmt(row['medianActualPriceCny'])} | "
            f"{fmt(row['tickFloorSpreadBps'])} | {fmt(row['corwinSchultzSpreadBps'])} |"
        )
    lines.extend(["", "## The picks the book actually makes", ""])
    lines.append(
        "| window | picks | median price | gross bps | tick floor | CS est | net @floor | net @est | net @assumed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for period, row in report["selection"].items():
        if not row.get("picks"):
            continue

        def fmt(key: str, digits: int = 2) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {period} | {row['picks']} | {fmt('medianActualPriceCny')} | "
            f"{fmt('grossBps')} | {fmt('tickFloorSpreadBps')} | "
            f"{fmt('corwinSchultzSpreadBps')} | {fmt('netUsingExactFloorBps')} | "
            f"{fmt('netUsingEstimateBps')} | {fmt('netUsingAssumedBps')} |"
        )
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def build_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Does the flat assumption survive contact with a measured floor?"""
    assumed = float(report["assumedRoundTripBps"])
    overstated = []
    for period, row in report["selection"].items():
        total = row.get("totalUsingExactFloorBps")
        if total is None:
            continue
        if total < assumed:
            overstated.append(
                {
                    "period": period,
                    "measuredFloorTotalBps": total,
                    "assumedBps": assumed,
                    "netAtFloorBps": row.get("netUsingExactFloorBps"),
                    "netAtAssumedBps": row.get("netUsingAssumedBps"),
                    "signFlips": bool(
                        row.get("netUsingExactFloorBps") is not None
                        and row.get("netUsingExactFloorBps") > 0.0
                        and row.get("netUsingAssumedBps", 0.0) <= 0.0
                    ),
                }
            )
    flips = [item for item in overstated if item["signFlips"]]
    if flips:
        decision = "flat_cost_assumption_overstated_and_a_conclusion_flips"
    elif overstated:
        decision = "flat_cost_assumption_overstated_but_no_conclusion_flips"
    else:
        decision = "flat_cost_assumption_is_not_overstated"
    return {
        "decision": decision,
        "windowsWhereAssumptionExceedsMeasuredFloor": overstated,
        "historicalHypothesisPass": bool(flips),
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
    panel_key, _ = panel_cache.cache_key(base, cog_config)
    panel, panel_audit = panel_cache.build_configured_panel_cached(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )

    price_cfg = config["price"]
    tick = float(price_cfg["tickSizeCny"])
    price = actual_traded_price(panel)
    price = price.where(
        price.between(
            float(price_cfg["minimumActualPriceCny"]),
            float(price_cfg["maximumActualPriceCny"]),
        )
    )
    tick_bps = relative_tick_floor_bps(price, tick).where(panel["eligible"])
    cs_bps = corwin_schultz_bps(
        panel["high"], panel["low"], int(config["estimators"]["corwinSchultz"]["windowSessions"])
    ).where(panel["eligible"])
    amihud = amihud_illiquidity(
        panel["returns"],
        panel["amount"],
        int(config["estimators"]["amihud"]["windowSessions"]),
    ).where(panel["eligible"])
    print("cost_estimators_ready", flush=True)

    strata = config["strata"]
    buckets = int(strata["liquidityBucketCount"])
    pct = liquidity_buckets(
        panel["amount"].where(panel["eligible"]), buckets, 60
    )
    decile_rows = []
    for index in range(buckets):
        low = index / buckets
        high = (index + 1) / buckets
        in_bucket = pct.le(high) if index == 0 else (pct.gt(low) & pct.le(high))
        decile_rows.append(
            {
                "decile": index + 1,
                "medianActualPriceCny": weighted_stat(price, in_bucket),
                "tickFloorSpreadBps": weighted_stat(tick_bps, in_bucket),
                "corwinSchultzSpreadBps": weighted_stat(cs_bps, in_bucket),
                "amihud": weighted_stat(amihud, in_bucket),
            }
        )

    ranks, _static, factor_audit = panel_cache.build_rank_book_cached(
        panel, frozen, panel_key
    )
    factors = list(ranks.keys())
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    ).reindex(factors).astype(float)
    prior /= prior.sum()
    weights = frontier.weight_frame(panel["close"].index, factors, prior.to_dict())
    score = guarded.adaptive_score(ranks, weights, panel)

    max_delay = 5
    outcome, execution_eligible, _delay = precision.executable_horizon_return(
        panel, 1, max_delay
    )
    support = pd.DataFrame(
        0, index=panel["close"].index, columns=panel["close"].columns
    )
    mask = precision.selection_mask(score, support, execution_eligible, 0, 10)
    splits = precision.split_dates(panel["close"].index, source)
    selection: dict[str, Any] = {}
    for period in ("validation", "shadow"):
        if period not in splits:
            continue
        dates = precision.contained_signal_dates(splits[period], 1, max_delay)
        selection[period] = summarise_selection(
            mask, dates, outcome, tick_bps, cs_bps, price, config
        )
    print("selection_priced", flush=True)

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
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "assumedRoundTripBps": float(config["evaluation"]["assumedRoundTripCost"]) * 1e4,
        "fees": config["fees"],
        "price": price_cfg,
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "splitAudit": source["splitAudit"],
        "liquidityDeciles": decile_rows,
        "selection": selection,
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
    precision.atomic_write(
        output / "liquidity_deciles.csv",
        pd.DataFrame(decile_rows).to_csv(index=False, lineterminator="\n"),
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
