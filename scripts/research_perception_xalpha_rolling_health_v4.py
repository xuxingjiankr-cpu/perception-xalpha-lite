#!/usr/bin/env python3
"""Causal rolling-health research for the frozen twelve-factor A-share book.

V4 changes no factor formula, direction or base weight.  At each signal close it uses only
factor ICs and counterfactual book outcomes that are guaranteed to have completed at least
seven sessions earlier.  A factor with non-positive rolling health is attenuated to zero
but is never flipped.  A separate lagged book-health gate may abstain while the ungated
counterfactual book continues to be measured, allowing the gate to recover.

This module is permanently research/shadow-only and cannot create an order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
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


SCHEMA_VERSION = "perception_xalpha_rolling_health_result_v4"
CODE_VERSION = "perception_xalpha_rolling_health_v4_20260806"
DEFAULT_CONFIG = ROOT / "configs" / "research" / "perception_xalpha_rolling_health_v4.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_rolling_health_v4":
        raise ValueError("unexpected rolling-health V4 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("rolling-health V4 must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every rolling-health V4 mutation permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("rolling-health V4 output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    required_true = (
        "singleAdaptiveRule",
        "parametersFrozenBeforeV4Run",
        "factorDefinitionsDirectionsAndBaseWeightsRemainFrozen",
        "negativeHealthCanOnlyZeroNotFlipDirection",
        "validationAndShadowMayOnlyReject",
    )
    if not all(hypothesis.get(key) is True for key in required_true):
        raise ValueError("rolling-health preregistration is incomplete")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical rolling-health research cannot promote")
    data = config["data"]
    if int(data["holdingTradingDays"]) != 1:
        raise ValueError("V4 freezes the shortest executable one-session hold")
    expected_lag = (
        int(data["holdingTradingDays"])
        + 1
        + int(data["maximumExitDelayTradingDays"])
    )
    if int(data["fullOutcomeAvailabilityLagTradingDays"]) != expected_lag:
        raise ValueError("outcome availability lag must cover entry, hold and exit delay")
    if data.get("periodLocalOutcomeContainment") is not True:
        raise ValueError("outcomes must remain inside their evaluation period")
    health = config["factorHealth"]
    if health.get("mayReverseFactorDirection") is not False:
        raise ValueError("rolling health may not flip a factor direction")
    if int(health["minimumActiveFactors"]) < 2:
        raise ValueError("rolling book requires multiple active factors")
    gate = config["bookHealthGate"]
    if not gate.get("counterfactualBookAlwaysUpdated"):
        raise ValueError("counterfactual book must update while the gate is closed")
    if not gate.get("closedGateCreatesNoReplacementSelections"):
        raise ValueError("a closed gate must not backfill another selection")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("rolling-health output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("rolling-health orders must remain empty")


def load_frozen_precision_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = ROOT / config["basePrecisionConfig"]
    raw = path.read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    precision.validate_config(frozen)
    source, _ = precision.verify_frozen_source(frozen)
    return frozen, source, hashlib.sha256(raw).hexdigest()


def compute_rank_book(
    panel: dict[str, Any],
    frozen: dict[str, Any],
    *, vwap_basis: str = "archive_vwap_v2",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Compute frozen oriented ranks once; float32 storage keeps the full PIT run bounded."""
    close = panel["close"]
    eligible = panel["eligible"]
    inputs = precision.build_factor_inputs(panel, vwap_basis=vwap_basis)
    ranks: dict[str, pd.DataFrame] = {}
    static_numerator = close * 0.0
    static_available = close * 0.0
    audit = []
    for position, item in enumerate(frozen["frozenFactors"], start=1):
        key = str(item["factorKey"])
        zoo, name = key.split("/", 1)
        module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
        raw = module.compute(inputs).reindex_like(close)
        rank = (raw * float(item["direction"])).where(eligible).rank(axis=1, pct=True)
        rank32 = rank.astype(np.float32)
        ranks[key] = rank32
        weight = float(item["weight"])
        static_numerator = static_numerator.add(
            rank32.fillna(0.0).astype(float) * weight, fill_value=0.0
        )
        static_available = static_available.add(
            rank32.notna().astype(float) * weight, fill_value=0.0
        )
        audit.append(
            {
                "factorKey": key,
                "direction": float(item["direction"]),
                "baseWeight": weight,
                "finiteValues": int(np.isfinite(rank32.to_numpy(dtype=np.float32)).sum()),
            }
        )
        print(f"factor_rank_ready {position}/12 {key}", flush=True)
        del raw, rank
    static_score = static_numerator.div(static_available.replace(0.0, np.nan)).where(eligible)
    return ranks, static_score, {"factorCount": len(audit), "factors": audit,
                                "factorInputBasis": inputs["vwap"].attrs["factorInputBasisAudit"]}


def factor_daily_ic(
    ranks: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    minimum_cross_section: int = 50,
) -> pd.DataFrame:
    """Spearman IC as Pearson correlation of same-date percentile ranks."""
    outcome_rank = outcome.rank(axis=1, pct=True)
    usable = outcome.notna().sum(axis=1).ge(minimum_cross_section)
    values: dict[str, pd.Series] = {}
    for key, rank in ranks.items():
        correlation = rank.astype(float).corrwith(outcome_rank, axis=1)
        values[key] = correlation.where(usable)
    return pd.DataFrame(values, index=outcome.index)


def rolling_factor_health(
    daily_ic: pd.DataFrame,
    frozen: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return rolling IC t, [0,1] multiplier and renormalised causal weights."""
    spec = config["factorHealth"]
    lag = int(config["data"]["fullOutcomeAvailabilityLagTradingDays"])
    observed = daily_ic.shift(lag)
    window = int(spec["windowTradingDays"])
    minimum = int(spec["minimumObservations"])
    rolling = observed.rolling(window, min_periods=minimum)
    count = rolling.count()
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    standard_error = std.div(np.sqrt(count).replace(0.0, np.nan))
    t_value = mean.div(standard_error.replace(0.0, np.nan))
    zero_variance_positive = std.eq(0.0) & mean.gt(0.0) & count.ge(minimum)
    t_value = t_value.mask(zero_variance_positive, np.inf)
    low = float(spec["activationTStatistic"])
    high = float(spec["fullStrengthTStatistic"])
    multiplier = ((t_value - low) / (high - low)).clip(lower=0.0, upper=1.0)
    multiplier = multiplier.where(count.ge(minimum), 0.0).fillna(0.0)
    base_weights = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    raw_weight = multiplier.mul(base_weights, axis=1)
    weights = raw_weight.div(raw_weight.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return t_value.replace([np.inf, -np.inf], np.nan), multiplier, weights


def adaptive_factor_score(
    ranks: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    panel: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    close = panel["close"]
    numerator = close * 0.0
    available = close * 0.0
    for key, rank in ranks.items():
        weight = weights[key].reindex(close.index).fillna(0.0)
        numerator = numerator.add(rank.astype(float).mul(weight, axis=0).fillna(0.0), fill_value=0.0)
        available = available.add(rank.notna().astype(float).mul(weight, axis=0), fill_value=0.0)
    active_count = weights.gt(0.0).sum(axis=1)
    score = numerator.div(available.replace(0.0, np.nan)).where(panel["eligible"])
    score = score.where(
        active_count.ge(int(config["factorHealth"]["minimumActiveFactors"])), axis=0
    )
    return score, active_count


def counterfactual_book_net_return(
    selected: pd.DataFrame,
    outcome: pd.DataFrame,
    cost: float,
) -> pd.Series:
    return outcome.where(selected).mean(axis=1, skipna=True) - float(cost)


def rolling_book_gate(
    counterfactual_net: pd.Series,
    active_count: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    spec = config["bookHealthGate"]
    lag = int(config["data"]["fullOutcomeAvailabilityLagTradingDays"])
    observed = counterfactual_net.shift(lag)
    window = int(spec["windowTradingDays"])
    minimum = int(spec["minimumCompletedSignalDays"])
    count = observed.notna().astype(float).rolling(window, min_periods=1).sum()
    wins = observed.gt(0.0).where(observed.notna()).astype(float).rolling(
        window, min_periods=1
    ).sum()
    mean = observed.rolling(window, min_periods=minimum).mean()
    alpha = float(spec["betaPriorAlpha"])
    beta = float(spec["betaPriorBeta"])
    posterior = (wins + alpha).div(count + alpha + beta)
    gate = (
        count.ge(minimum)
        & posterior.ge(float(spec["minimumPosteriorNetWinProbability"]))
        & mean.gt(float(spec["minimumRollingMeanNetReturn"]))
    )
    if spec.get("requiresMinimumActiveFactors"):
        gate &= active_count.ge(int(config["factorHealth"]["minimumActiveFactors"]))
    audit = pd.DataFrame(
        {
            "completedCounterfactualDays": count,
            "rollingMeanNetReturn": mean,
            "posteriorNetWinProbability": posterior,
            "activeFactorCount": active_count,
            "gateOpen": gate.fillna(False),
        }
    )
    return gate.fillna(False), audit


def build_verdict(
    periods: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    gate = config["evaluation"]
    for period in ("validation", "shadow"):
        static = periods[period]["static"]
        gated = periods[period]["adaptive_gated"]
        checks[period] = {
            "minimumSignalDays": gated["signalDays"]
            >= int(gate["minimumSignalDaysPerExternalPeriod"]),
            "minimumCoverage": (gated["selectionCoverage"] or 0.0)
            >= float(gate["minimumGateCoverage"]),
            "precisionImproves": gated["dailyNetWinRate"] is not None
            and static["dailyNetWinRate"] is not None
            and gated["dailyNetWinRate"] > static["dailyNetWinRate"],
            "meanNetReturnImproves": gated["dailyMeanNetReturn"] is not None
            and static["dailyMeanNetReturn"] is not None
            and gated["dailyMeanNetReturn"] > static["dailyMeanNetReturn"],
            "severeLossNotWorse": gated["stockSevereLossRate"] is not None
            and static["stockSevereLossRate"] is not None
            and gated["stockSevereLossRate"] <= static["stockSevereLossRate"],
        }
        checks[period]["passed"] = all(checks[period].values())
    historical_pass = all(item["passed"] for item in checks.values())
    return {
        "decision": (
            "keep_as_fresh_forward_hypothesis_only"
            if historical_pass
            else "reject_rolling_health_keep_diagnostics"
        ),
        "externalChecks": checks,
        "historicalHypothesisPass": historical_pass,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
    }


def latest_health_rows(
    t_values: pd.DataFrame,
    multipliers: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    date = weights.index.max()
    return pd.DataFrame(
        {
            "signalDate": date.date().isoformat(),
            "factorKey": weights.columns,
            "rollingIcT": t_values.loc[date].reindex(weights.columns).to_numpy(dtype=float),
            "healthMultiplier": multipliers.loc[date].reindex(weights.columns).to_numpy(dtype=float),
            "adaptiveWeight": weights.loc[date].to_numpy(dtype=float),
            "status": "diagnostic_only_not_trading",
        }
    ).sort_values(["adaptiveWeight", "factorKey"], ascending=[False, True])


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Rolling Factor Health V4",
        "",
        f"- Status: `{report['status']}`",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        "- Factor formulas/directions/base weights: frozen from V2",
        "- Outcome availability lag: 7 trading sessions",
        "- Adaptive rule: positive 63-day rolling IC t only; no direction flips",
        "- Book gate: 40 completed counterfactual days, Beta(2,2), posterior win >= 52%, mean net > 0",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`",
        "- Orders: `[]`",
        "",
        "## Period comparison",
        "",
        "| Period | Book | Days | Coverage | Names/day | Stock net win | Daily net win | Wilson lower | Mean net | Severe loss | HAC t |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, books in report["periods"].items():
        for name, metrics in books.items():
            lines.append(
                f"| {period} | {name} | {metrics['signalDays']} | {metrics['selectionCoverage']} | "
                f"{metrics['meanNamesPerSignalDay']} | {metrics['stockNetWinRate']} | "
                f"{metrics['dailyNetWinRate']} | {metrics['dailyNetWinWilsonLower']} | "
                f"{metrics['dailyMeanNetReturn']} | {metrics['stockSevereLossRate']} | "
                f"{metrics['dailyNetHacT']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `adaptive_ungated` isolates factor attenuation from abstention.",
            "- `adaptive_gated` may skip days, but coverage and signal counts remain hard gates.",
            "- The gate always learns from the ungated counterfactual book, so it can reopen.",
            "- A historical pass still cannot promote because all historical windows have been viewed.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    frozen, source, frozen_sha = load_frozen_precision_config(config)
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        int(config["data"]["holdingTradingDays"]),
        int(config["data"]["maximumExitDelayTradingDays"]),
    )
    ranks, static_score, factor_audit = compute_rank_book(panel, frozen)
    daily_ic = factor_daily_ic(ranks, outcome)
    t_values, multipliers, adaptive_weights = rolling_factor_health(
        daily_ic, frozen, config
    )
    adaptive_score, active_count = adaptive_factor_score(
        ranks, adaptive_weights, panel, config
    )
    empty_support = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
    top_count = int(config["data"]["topCount"])
    static_mask = precision.selection_mask(
        static_score, empty_support, execution_eligible, 0, top_count
    )
    adaptive_mask = precision.selection_mask(
        adaptive_score, empty_support, execution_eligible, 0, top_count
    )
    counterfactual_net = counterfactual_book_net_return(
        adaptive_mask, outcome, float(config["data"]["roundTripCost"])
    )
    gate_open, gate_audit = rolling_book_gate(counterfactual_net, active_count, config)
    gated_mask = adaptive_mask.mul(
        gate_open.reindex(adaptive_mask.index).fillna(False), axis=0
    ).astype(bool)
    splits = precision.split_dates(panel["close"].index, source)
    contained = {
        name: precision.contained_signal_dates(
            dates,
            int(config["data"]["holdingTradingDays"]),
            int(config["data"]["maximumExitDelayTradingDays"]),
        )
        for name, dates in splits.items()
    }
    periods: dict[str, dict[str, dict[str, Any]]] = {}
    for period, dates in contained.items():
        periods[period] = {
            "static": precision.summarize_selection(
                static_mask, outcome, exit_delay, dates, frozen, 1
            ),
            "adaptive_ungated": precision.summarize_selection(
                adaptive_mask, outcome, exit_delay, dates, frozen, 1
            ),
            "adaptive_gated": precision.summarize_selection(
                gated_mask, outcome, exit_delay, dates, frozen, 1
            ),
        }
    verdict = build_verdict(periods, config)
    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_" + now.strftime("%Y%m%dT%H%M%SZ") + "_" + precision.digest({"config": config, "code": CODE_VERSION})[:10]
    )
    output = ROOT / config["output"]["root"] / run_id
    latest_health = latest_health_rows(t_values, multipliers, adaptive_weights)
    latest_date = adaptive_score.index.max()
    latest_rows = pd.DataFrame(
        {
            "securityId": adaptive_score.columns,
            "adaptiveScore": adaptive_score.loc[latest_date].to_numpy(dtype=float),
        }
    ).dropna().sort_values(["adaptiveScore", "securityId"], ascending=[False, True]).head(top_count)
    latest_rows.insert(0, "signalDate", latest_date.date().isoformat())
    latest_rows["bookHealthGateOpen"] = bool(gate_open.loc[latest_date])
    latest_rows["status"] = "diagnostic_only_not_an_order"
    report = {
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
        "panelAudit": panel_audit,
        "splitAudit": source["splitAudit"],
        "factorAudit": factor_audit,
        "periods": periods,
        "latestGateAudit": precision.json_safe(gate_audit.loc[latest_date].to_dict()),
        "latestActiveFactorCount": int(active_count.loc[latest_date]),
        "verdict": verdict,
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    precision.atomic_write(
        output / "summary.json",
        precision.canonical(precision.json_safe(report)) + "\n",
    )
    precision.atomic_write(output / "report.md", markdown_report(report))
    health_daily = gate_audit.copy()
    for key in adaptive_weights:
        safe = key.replace("/", "_")
        health_daily[f"weight_{safe}"] = adaptive_weights[key]
        health_daily[f"ic_t_{safe}"] = t_values[key]
    health_daily.index.name = "date"
    precision.atomic_write(
        output / "factor_health_daily.csv",
        health_daily.reset_index().to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_factor_health.csv",
        latest_health.to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_diagnostic_candidates.csv",
        latest_rows.to_csv(index=False, lineterminator="\n"),
    )
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    report = run(args.config, args.run_id)
    print(
        precision.canonical(
            {
                "runId": report["runId"],
                "decision": report["verdict"]["decision"],
                "latestGateAudit": report["latestGateAudit"],
                "eligibleForTrading": False,
                "orders": [],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
