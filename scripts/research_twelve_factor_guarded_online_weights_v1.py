#!/usr/bin/env python3
"""Guarded weekly online weights for the frozen twelve-factor A-share book.

The adapter is deliberately low freedom: it observes only outcome histories that have
fully resolved, requires 20/60-session agreement, penalises redundant evidence, shrinks
to the frozen prior, bounds every factor and caps one-update turnover.  It is permanently
research/shadow-only and has no broker, order, overlay or production-decision path.
"""

from __future__ import annotations

import argparse
import hashlib
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

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402


SCHEMA_VERSION = "twelve_factor_guarded_online_weights_result_v1"
CODE_VERSION = "twelve_factor_guarded_online_weights_v1_20260812"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_guarded_online_weights_v1.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_guarded_online_weights_v1":
        raise ValueError("unexpected guarded-online-weight schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("guarded online weights must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every guarded-online-weight mutation permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("guarded online weights must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    required_true = (
        "factorDefinitionsDirectionsAndPriorWeightsFrozen",
        "singleOnlineRule",
        "parametersFrozenBeforeHistoricalEvaluation",
        "validationAndShadowAreRejectOnly",
    )
    if not all(hypothesis.get(key) is True for key in required_true):
        raise ValueError("guarded online weight preregistration is incomplete")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical guarded online weights cannot promote")
    data = config["data"]
    expected_lag = (
        int(data["holdingTradingDays"])
        + 1
        + int(data["maximumExitDelayTradingDays"])
    )
    if int(data["fullOutcomeAvailabilityLagTradingDays"]) != expected_lag:
        raise ValueError("outcome lag must cover entry, hold and delayed exit")
    adapter = config["adapter"]
    if int(adapter["shortWindowTradingDays"]) >= int(adapter["longWindowTradingDays"]):
        raise ValueError("short evidence window must be shorter than long window")
    if int(adapter["minimumShortObservations"]) > int(adapter["shortWindowTradingDays"]):
        raise ValueError("short minimum observations exceed the window")
    if int(adapter["minimumLongObservations"]) > int(adapter["longWindowTradingDays"]):
        raise ValueError("long minimum observations exceed the window")
    if int(adapter["updateEveryTradingDays"]) < 2:
        raise ValueError("weights may not be updated every session")
    if adapter.get("requireShortLongSignAgreement") is not True:
        raise ValueError("dual-horizon sign agreement must remain enabled")
    if adapter.get("negativeEvidenceMayReverseDirection") is not False:
        raise ValueError("online evidence may not reverse a factor direction")
    if adapter.get("minimumPositiveWeight") is not True:
        raise ValueError("every frozen factor must retain a positive weight")
    if adapter.get("failBackToPriorOnInsufficientEvidence") is not True:
        raise ValueError("insufficient evidence must fail back to the frozen prior")
    shrinkage = float(adapter["shrinkageToFrozenPrior"])
    if not 0.0 < shrinkage <= 0.5:
        raise ValueError("shrinkage must keep at least half the frozen prior")
    lower = float(adapter["minimumRelativeWeight"])
    upper = float(adapter["maximumRelativeWeight"])
    if not 0.0 < lower < 1.0 < upper:
        raise ValueError("relative bounds must contain the frozen prior")
    if not 0.0 < float(adapter["maximumOneUpdateL1Turnover"]) <= 0.10:
        raise ValueError("one-update turnover cap must remain small")
    evaluation = config["evaluation"]
    if evaluation.get("sameTop10CountRequired") is not True:
        raise ValueError("static and adaptive books must use the same Top10 count")
    if evaluation.get("noAbstentionAllowed") is not True:
        raise ValueError("the adapter may not improve mechanically by abstaining")
    if evaluation.get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical windows must be marked as already viewed")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def load_frozen_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = ROOT / config["basePrecisionConfig"]
    raw = path.read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    precision.validate_config(frozen)
    source, _ = precision.verify_frozen_source(frozen)
    if len(frozen["frozenFactors"]) != 12:
        raise ValueError("the guarded adapter requires exactly twelve frozen factors")
    return frozen, source, hashlib.sha256(raw).hexdigest()


def _rolling_t(
    frame: pd.DataFrame, window: int, minimum: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rolling_window = frame.rolling(window, min_periods=minimum)
    count = rolling_window.count()
    mean = rolling_window.mean()
    std = rolling_window.std(ddof=1)
    standard_error = std.div(np.sqrt(count).replace(0.0, np.nan))
    t_value = mean.div(standard_error.replace(0.0, np.nan))
    t_value = t_value.mask(std.eq(0.0) & mean.gt(0.0) & count.ge(minimum), np.inf)
    t_value = t_value.mask(std.eq(0.0) & mean.lt(0.0) & count.ge(minimum), -np.inf)
    return t_value, count


def _bounded_simplex(
    raw: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Scale and clip positive raw weights onto a bounded simplex."""
    raw = np.asarray(raw, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if raw.ndim != 1 or not (raw.shape == lower.shape == upper.shape):
        raise ValueError("bounded-simplex vectors must have the same one-dimensional shape")
    if np.any(raw <= 0.0) or np.any(lower <= 0.0) or np.any(upper < lower):
        raise ValueError("bounded-simplex inputs must remain positive and ordered")
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise ValueError("bounded simplex is infeasible")
    lo, hi = 0.0, 1.0
    while np.minimum(raw * hi, upper).clip(min=lower).sum() < 1.0:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        total = np.clip(raw * mid, lower, upper).sum()
        if total < 1.0:
            lo = mid
        else:
            hi = mid
    result = np.clip(raw * ((lo + hi) / 2.0), lower, upper)
    result /= result.sum()
    return result


def guarded_weight_path(
    daily_ic: pd.DataFrame,
    prior_weights: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a causal weekly path from lagged, fully resolved factor ICs."""
    spec = config["adapter"]
    lag = int(config["data"]["fullOutcomeAvailabilityLagTradingDays"])
    columns = list(prior_weights.index)
    prior = prior_weights.reindex(columns).astype(float)
    prior /= prior.sum()
    observed = daily_ic.reindex(columns=columns).shift(lag)
    short_t, short_n = _rolling_t(
        observed,
        int(spec["shortWindowTradingDays"]),
        int(spec["minimumShortObservations"]),
    )
    long_t, long_n = _rolling_t(
        observed,
        int(spec["longWindowTradingDays"]),
        int(spec["minimumLongObservations"]),
    )
    cap = float(spec["evidenceTCap"])
    cadence = int(spec["updateEveryTradingDays"])
    penalty_strength = float(spec["redundancyPenalty"])
    eta = float(spec["exponentLearningRate"])
    shrink = float(spec["shrinkageToFrozenPrior"])
    lower = prior.to_numpy() * float(spec["minimumRelativeWeight"])
    upper = prior.to_numpy() * float(spec["maximumRelativeWeight"])
    turnover_cap = float(spec["maximumOneUpdateL1Turnover"])
    current = prior.to_numpy(copy=True)
    rows: list[np.ndarray] = []
    updates: list[dict[str, Any]] = []
    for position, date in enumerate(daily_ic.index):
        is_update = position % cadence == 0
        used_prior = False
        evidence = np.zeros(len(columns), dtype=float)
        redundancy = np.zeros(len(columns), dtype=float)
        if is_update:
            enough = (
                short_n.loc[date].reindex(columns).ge(int(spec["minimumShortObservations"]))
                & long_n.loc[date].reindex(columns).ge(int(spec["minimumLongObservations"]))
            ).to_numpy(dtype=bool)
            short_values = short_t.loc[date].reindex(columns).to_numpy(dtype=float)
            long_values = long_t.loc[date].reindex(columns).to_numpy(dtype=float)
            finite = np.isfinite(short_values) & np.isfinite(long_values) & enough
            agree = np.sign(short_values) == np.sign(long_values)
            usable = finite & agree & (np.sign(short_values) != 0.0)
            if usable.any():
                signed_min = np.sign(short_values) * np.minimum(
                    np.abs(short_values), np.abs(long_values)
                )
                evidence[usable] = np.clip(signed_min[usable], -cap, cap)
                history = observed.loc[:date].tail(int(spec["longWindowTradingDays"]))
                corr = history.corr(min_periods=int(spec["minimumShortObservations"])).abs()
                for index, key in enumerate(columns):
                    others = corr.loc[key].drop(labels=[key], errors="ignore").dropna()
                    redundancy[index] = float(others.mean()) if not others.empty else 0.0
                adjusted = evidence * (1.0 - penalty_strength * redundancy)
                tilted = prior.to_numpy() * np.exp(eta * adjusted)
                unconstrained = tilted / tilted.sum()
                shrunk = (1.0 - shrink) * prior.to_numpy() + shrink * unconstrained
                target = _bounded_simplex(shrunk, lower, upper)
            else:
                target = prior.to_numpy(copy=True)
                used_prior = True
            proposed_turnover = float(np.abs(target - current).sum())
            if proposed_turnover > turnover_cap:
                target = current + (target - current) * (turnover_cap / proposed_turnover)
            realised_turnover = float(np.abs(target - current).sum())
            current = target
            updates.append(
                {
                    "updateDate": date.date().isoformat(),
                    "usedFrozenPriorFallback": used_prior,
                    "usableFactorCount": int(usable.sum()),
                    "proposedL1Turnover": proposed_turnover,
                    "realisedL1Turnover": realised_turnover,
                    "maximumAbsoluteEvidence": float(np.max(np.abs(evidence))),
                    "meanRedundancy": float(np.mean(redundancy)),
                }
            )
        rows.append(current.copy())
    weights = pd.DataFrame(rows, index=daily_ic.index, columns=columns)
    update_frame = pd.DataFrame(updates)
    return weights, update_frame


def adaptive_score(
    ranks: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    panel: dict[str, Any],
) -> pd.DataFrame:
    close = panel["close"]
    numerator = close * 0.0
    available = close * 0.0
    for key, rank in ranks.items():
        weight = weights[key].reindex(close.index)
        numerator = numerator.add(rank.astype(float).mul(weight, axis=0).fillna(0.0))
        available = available.add(rank.notna().astype(float).mul(weight, axis=0))
    return numerator.div(available.replace(0.0, np.nan)).where(panel["eligible"])


def paired_daily_difference(
    static_mask: pd.DataFrame,
    adaptive_mask: pd.DataFrame,
    outcome: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    static_daily = outcome.reindex(index=dates).where(
        static_mask.reindex(index=dates).fillna(False)
    ).mean(axis=1, skipna=True)
    adaptive_daily = outcome.reindex(index=dates).where(
        adaptive_mask.reindex(index=dates).fillna(False)
    ).mean(axis=1, skipna=True)
    paired = pd.concat([static_daily, adaptive_daily], axis=1).dropna()
    paired.columns = ["static", "adaptive"]
    difference = paired["adaptive"] - paired["static"]
    overlap = (
        (static_mask.reindex(index=dates).fillna(False) & adaptive_mask.reindex(index=dates).fillna(False))
        .sum(axis=1)
        .reindex(paired.index)
    )
    return {
        "pairedDays": int(len(difference)),
        "meanGrossDifference": float(difference.mean()) if len(difference) else None,
        "pairedHacT": (
            float(autonomous.newey_west_t(difference.to_numpy(dtype=float), 0))
            if len(difference) >= 3
            else None
        ),
        "meanTop10Overlap": float(overlap.mean()) if len(overlap) else None,
        "minimumTop10Overlap": int(overlap.min()) if len(overlap) else None,
    }


def build_verdict(
    periods: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    threshold = float(config["evaluation"]["pairedDailyHacTMinimum"])
    for period in ("validation", "shadow"):
        static = periods[period]["static"]
        adaptive = periods[period]["adaptive"]
        paired = periods[period]["pairedDifference"]
        checks[period] = {
            "sameObservationCount": adaptive["intendedObservations"]
            == static["intendedObservations"],
            "meanGrossImproves": adaptive["dailyMeanGrossReturn"]
            > static["dailyMeanGrossReturn"],
            "dailyNetWinNotWorse": adaptive["dailyNetWinRate"]
            >= static["dailyNetWinRate"],
            "severeLossNotWorse": adaptive["stockSevereLossRate"]
            <= static["stockSevereLossRate"],
            "pairedHacTMeetsThreshold": paired["pairedHacT"] is not None
            and paired["pairedHacT"] >= threshold,
        }
        checks[period]["passed"] = all(checks[period].values())
    historical_pass = all(item["passed"] for item in checks.values())
    return {
        "decision": (
            "retain_as_separate_fresh_forward_hypothesis_only"
            if historical_pass
            else "reject_adaptation_and_use_frozen_static_prior"
        ),
        "externalRejectOnlyChecks": checks,
        "historicalHypothesisPass": historical_pass,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Guarded Online Twelve-Factor Weights V1",
        "",
        f"- Status: `{report['status']}`",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        "- Rule: weekly update; 20/60-session IC agreement; 7-session outcome lag",
        "- Guardrails: redundancy penalty, 75%-125% prior bounds, 25% adaptation, 8% L1 update cap",
        "- No abstention and exactly the same Top10 capacity as the static book",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`",
        "- Orders: `[]`",
        "",
        "## Static versus adaptive Top10",
        "",
        "| Period | Book | Days | Gross mean | Net mean | Daily net win | Severe loss | Max drawdown |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period, payload in report["periods"].items():
        for book in ("static", "adaptive"):
            value = payload[book]
            lines.append(
                f"| {period} | {book} | {value['signalDays']} | "
                f"{value['dailyMeanGrossReturn']} | {value['dailyMeanNetReturn']} | "
                f"{value['dailyNetWinRate']} | {value['stockSevereLossRate']} | "
                f"{value['dailyMaximumDrawdown']} |"
            )
        paired = payload["pairedDifference"]
        lines.append(
            f"| {period} | adaptive-static | {paired['pairedDays']} | "
            f"{paired['meanGrossDifference']} | n/a | n/a | n/a | paired HAC t={paired['pairedHacT']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The adapter changes ranking weights, never the number of selected names. Any improvement therefore cannot be manufactured by skipping difficult days.",
            "Historical validation and shadow windows have already been inspected. They can reject this rule but cannot establish live profitability.",
            "If either external window fails, the operational conclusion is the frozen static prior—not another search over update windows.",
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
    frozen, source, frozen_sha = load_frozen_config(config)
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
    ranks, static_score, factor_audit = rolling.compute_rank_book(panel, frozen)
    daily_ic = rolling.factor_daily_ic(ranks, outcome)
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    adaptive_weights, updates = guarded_weight_path(daily_ic, prior, config)
    online_score = adaptive_score(ranks, adaptive_weights, panel)
    empty_support = pd.DataFrame(
        0, index=panel["close"].index, columns=panel["close"].columns
    )
    top_count = int(config["data"]["topCount"])
    static_mask = precision.selection_mask(
        static_score, empty_support, execution_eligible, 0, top_count
    )
    adaptive_mask = precision.selection_mask(
        online_score, empty_support, execution_eligible, 0, top_count
    )
    splits = precision.split_dates(panel["close"].index, source)
    contained = {
        name: precision.contained_signal_dates(
            dates,
            int(config["data"]["holdingTradingDays"]),
            int(config["data"]["maximumExitDelayTradingDays"]),
        )
        for name, dates in splits.items()
    }
    periods: dict[str, dict[str, Any]] = {}
    for period, dates in contained.items():
        periods[period] = {
            "static": precision.summarize_selection(
                static_mask, outcome, exit_delay, dates, frozen, 1
            ),
            "adaptive": precision.summarize_selection(
                adaptive_mask, outcome, exit_delay, dates, frozen, 1
            ),
            "pairedDifference": paired_daily_difference(
                static_mask, adaptive_mask, outcome, dates
            ),
        }
    verdict = build_verdict(periods, config)
    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_"
        + now.strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + precision.digest({"config": config, "code": CODE_VERSION})[:10]
    )
    output = ROOT / config["output"]["root"] / run_id
    output.mkdir(parents=True, exist_ok=True)
    latest_date = online_score.index.max()
    latest_weights = pd.DataFrame(
        {
            "factorKey": adaptive_weights.columns,
            "frozenWeight": prior.reindex(adaptive_weights.columns).to_numpy(dtype=float),
            "adaptiveWeight": adaptive_weights.loc[latest_date].to_numpy(dtype=float),
        }
    )
    latest_weights["relativeToFrozen"] = (
        latest_weights["adaptiveWeight"] / latest_weights["frozenWeight"]
    )
    latest_weights["signalDate"] = latest_date.date().isoformat()
    latest_weights["status"] = "diagnostic_only_not_trading"
    latest_all_scores = pd.DataFrame(
        {
            "securityId": online_score.columns,
            "adaptiveScore": online_score.loc[latest_date].to_numpy(dtype=float),
        }
    ).dropna().sort_values(
        ["adaptiveScore", "securityId"], ascending=[False, True]
    ).reset_index(drop=True)
    latest_all_scores.insert(0, "rank", np.arange(1, len(latest_all_scores) + 1))
    latest_all_scores.insert(0, "signalDate", latest_date.date().isoformat())
    latest_all_scores["status"] = "diagnostic_only_not_an_order"
    latest_top10 = latest_all_scores.head(top_count).copy()
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
        "adapterAudit": {
            "updateCount": int(len(updates)),
            "priorFallbackUpdates": int(updates["usedFrozenPriorFallback"].sum()),
            "maximumRealisedL1Turnover": float(updates["realisedL1Turnover"].max()),
            "meanRealisedL1Turnover": float(updates["realisedL1Turnover"].mean()),
            "minimumWeight": float(adaptive_weights.min().min()),
            "maximumWeight": float(adaptive_weights.max().max()),
            "maximumWeightSumError": float(
                (adaptive_weights.sum(axis=1) - 1.0).abs().max()
            ),
        },
        "periods": periods,
        "latestWeights": precision.json_safe(latest_weights.to_dict(orient="records")),
        "verdict": verdict,
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    precision.atomic_write(
        output / "summary.json",
        precision.canonical(precision.json_safe(report)) + "\n",
    )
    precision.atomic_write(output / "report.md", markdown_report(report))
    weight_output = adaptive_weights.copy()
    weight_output.index.name = "signalDate"
    precision.atomic_write(
        output / "weights_daily.csv",
        weight_output.reset_index().to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "update_diagnostics.csv",
        updates.to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_weights.csv",
        latest_weights.to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_diagnostic_top10.csv",
        latest_top10.to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_diagnostic_all_scores.csv",
        latest_all_scores.to_csv(index=False, lineterminator="\n"),
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
                "eligibleForTrading": False,
                "orders": [],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
