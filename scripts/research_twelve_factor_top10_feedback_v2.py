#!/usr/bin/env python3
"""Causal daily Top10-outcome feedback for the frozen twelve-factor A-share book.

Every factor is treated as a Top10 expert.  Only fully resolved, seven-session-lagged
historical outcomes may alter later weights.  The adapter is permanently research-only,
always selects ten names, and has no broker, order, overlay or production path.
"""

from __future__ import annotations

import argparse
import hashlib
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


SCHEMA_VERSION = "twelve_factor_top10_feedback_result_v2"
CODE_VERSION = "twelve_factor_top10_feedback_v2_20260829"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_top10_feedback_v2.json"
)
METRICS = (
    "top10GrossReturn",
    "top10WinRate",
    "extremeWinnerRate",
    "severeLossAvoidance",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_top10_feedback_v2":
        raise ValueError("unexpected Top10-feedback schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("Top10 feedback must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every Top10-feedback mutation permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("Top10 feedback must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    required_true = (
        "factorDefinitionsDirectionsAndPriorWeightsFrozen",
        "singleOnlineRule",
        "parametersFrozenBeforeHistoricalEvaluation",
        "validationAndShadowAreRejectOnly",
    )
    if not all(hypothesis.get(key) is True for key in required_true):
        raise ValueError("Top10-feedback preregistration is incomplete")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical Top10 feedback cannot promote")
    data = config["data"]
    expected_lag = (
        int(data["holdingTradingDays"])
        + 1
        + int(data["maximumExitDelayTradingDays"])
    )
    if int(data["fullOutcomeAvailabilityLagTradingDays"]) != expected_lag:
        raise ValueError("outcome lag must cover entry, hold and delayed exit")
    if int(data["topCount"]) != 10:
        raise ValueError("the frozen feedback target must remain Top10")
    quantile = float(data["extremeWinnerCrossSectionQuantile"])
    if not (0.9 <= quantile < 1.0):
        raise ValueError("extreme-winner quantile must be in [0.90, 1.0)")
    spec = config["feedback"]
    if int(spec["updateEveryTradingDays"]) != 1:
        raise ValueError("V2 is a frozen daily feedback rule")
    recent = int(spec["recentResolvedSignalDays"])
    long = int(spec["longResolvedSignalDays"])
    if not (3 <= recent < long):
        raise ValueError("recent feedback must be shorter than long feedback")
    if int(spec["minimumRecentObservations"]) > recent:
        raise ValueError("recent minimum exceeds its window")
    if int(spec["minimumLongObservations"]) > long:
        raise ValueError("long minimum exceeds its window")
    metric_weights = spec["metricWeights"]
    if set(metric_weights) != set(METRICS):
        raise ValueError("the four frozen Top10 feedback metrics are required")
    if any(float(metric_weights[key]) <= 0.0 for key in METRICS):
        raise ValueError("every feedback metric must retain positive weight")
    if abs(sum(float(metric_weights[key]) for key in METRICS) - 1.0) > 1e-12:
        raise ValueError("feedback metric weights must sum to one")
    if abs(
        float(spec["recentEvidenceWeight"])
        + float(spec["longEvidenceWeight"])
        - 1.0
    ) > 1e-12:
        raise ValueError("recent and long evidence weights must sum to one")
    if not (0.0 < float(spec["adaptiveAllocation"]) <= 0.5):
        raise ValueError("at least half the allocation must remain the frozen prior")
    lower = float(spec["minimumRelativeWeight"])
    upper = float(spec["maximumRelativeWeight"])
    if not (0.0 < lower < 1.0 < upper):
        raise ValueError("relative bounds must contain the frozen prior")
    if not (0.0 < float(spec["maximumOneUpdateL1Turnover"]) <= 0.05):
        raise ValueError("daily weight turnover must remain small")
    if spec.get("negativeEvidenceMayReverseDirection") is not False:
        raise ValueError("feedback may not reverse factor direction")
    if spec.get("minimumPositiveWeight") is not True:
        raise ValueError("every factor weight must remain positive")
    if spec.get("failBackToPriorOnInsufficientEvidence") is not True:
        raise ValueError("insufficient feedback must use the prior")
    evaluation = config["evaluation"]
    if evaluation.get("sameTop10CountRequired") is not True:
        raise ValueError("static and feedback books must have the same Top10 count")
    if evaluation.get("noAbstentionAllowed") is not True:
        raise ValueError("feedback may not improve by abstaining")
    if evaluation.get("reportMarketDownAndMarketUpSeparately") is not True:
        raise ValueError("weak-market performance must remain visible")
    if evaluation.get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical windows must be marked viewed")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def expert_top10_metrics(
    ranks: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return factor-expert outcome histories and their exact selection masks."""
    top_count = int(config["data"]["topCount"])
    severe = float(config["data"]["severeLossThreshold"])
    winner_q = float(config["data"]["extremeWinnerCrossSectionQuantile"])
    support = pd.DataFrame(0, index=outcome.index, columns=outcome.columns)
    eligible_outcome = outcome.where(execution_eligible)
    winner_threshold = eligible_outcome.quantile(winner_q, axis=1)
    winner_flag = eligible_outcome.ge(winner_threshold, axis=0)
    values = {key: {} for key in METRICS}
    masks = {}
    for factor, rank in ranks.items():
        mask = precision.selection_mask(
            rank, support, execution_eligible, 0, top_count
        )
        masks[factor] = mask
        selected = outcome.where(mask)
        values["top10GrossReturn"][factor] = selected.mean(axis=1, skipna=True)
        values["top10WinRate"][factor] = (
            outcome.gt(0.0).where(mask).mean(axis=1, skipna=True)
        )
        values["extremeWinnerRate"][factor] = (
            winner_flag.where(mask).mean(axis=1, skipna=True)
        )
        values["severeLossAvoidance"][factor] = (
            1.0 - outcome.le(severe).where(mask).mean(axis=1, skipna=True)
        )
    frames = {
        metric: pd.DataFrame(payload).reindex(index=outcome.index)
        for metric, payload in values.items()
    }
    return frames, masks


def _row_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True).sub(0.5)


def feedback_weight_path(
    metric_frames: dict[str, pd.DataFrame],
    prior_weights: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build daily weights from lagged recent and long Top10 expert outcomes."""
    spec = config["feedback"]
    lag = int(config["data"]["fullOutcomeAvailabilityLagTradingDays"])
    factors = list(prior_weights.index)
    prior = prior_weights.reindex(factors).astype(float)
    prior /= prior.sum()
    recent_window = int(spec["recentResolvedSignalDays"])
    long_window = int(spec["longResolvedSignalDays"])
    recent_min = int(spec["minimumRecentObservations"])
    long_min = int(spec["minimumLongObservations"])
    recent_utilities = []
    long_utilities = []
    observed_metrics = {}
    for metric in METRICS:
        observed = metric_frames[metric].reindex(columns=factors).shift(lag)
        observed_metrics[metric] = observed
        recent = observed.rolling(recent_window, min_periods=recent_min).mean()
        long = observed.rolling(long_window, min_periods=long_min).mean()
        weight = float(spec["metricWeights"][metric])
        recent_utilities.append(_row_percentile(recent) * weight)
        long_utilities.append(_row_percentile(long) * weight)
    recent_utility = sum(recent_utilities)
    long_utility = sum(long_utilities)
    evidence = (
        float(spec["recentEvidenceWeight"]) * recent_utility
        + float(spec["longEvidenceWeight"]) * long_utility
    )
    reference = observed_metrics["top10GrossReturn"]
    recent_count = reference.rolling(recent_window, min_periods=1).count()
    long_count = reference.rolling(long_window, min_periods=1).count()
    usable = recent_count.ge(recent_min) & long_count.ge(long_min) & evidence.notna()
    eta = float(spec["exponentLearningRate"])
    adaptive_allocation = float(spec["adaptiveAllocation"])
    lower = prior.to_numpy(dtype=float) * float(spec["minimumRelativeWeight"])
    upper = prior.to_numpy(dtype=float) * float(spec["maximumRelativeWeight"])
    turnover_cap = float(spec["maximumOneUpdateL1Turnover"])
    current = prior.to_numpy(dtype=float, copy=True)
    weight_rows = []
    diagnostics = []
    for date in evidence.index:
        date_evidence = evidence.loc[date].reindex(factors).to_numpy(dtype=float)
        date_usable = usable.loc[date].reindex(factors).to_numpy(dtype=bool)
        used_prior = not bool(date_usable.all())
        if not used_prior:
            tilted = prior.to_numpy(dtype=float) * np.exp(eta * date_evidence)
            tilted /= tilted.sum()
            raw_target = (
                (1.0 - adaptive_allocation) * prior.to_numpy(dtype=float)
                + adaptive_allocation * tilted
            )
            target = guarded._bounded_simplex(raw_target, lower, upper)
        else:
            target = prior.to_numpy(dtype=float, copy=True)
        proposed_turnover = float(np.abs(target - current).sum())
        if proposed_turnover > turnover_cap:
            target = current + (target - current) * (
                turnover_cap / proposed_turnover
            )
        realised_turnover = float(np.abs(target - current).sum())
        current = target
        weight_rows.append(current.copy())
        diagnostics.append(
            {
                "signalDate": date.date().isoformat(),
                "usedFrozenPriorFallback": used_prior,
                "usableFactorCount": int(date_usable.sum()),
                "recentMinimumCount": int(
                    recent_count.loc[date].reindex(factors).min()
                ),
                "longMinimumCount": int(
                    long_count.loc[date].reindex(factors).min()
                ),
                "maximumAbsoluteEvidence": (
                    None
                    if not np.isfinite(date_evidence).any()
                    else float(np.nanmax(np.abs(date_evidence)))
                ),
                "proposedL1Turnover": proposed_turnover,
                "realisedL1Turnover": realised_turnover,
            }
        )
    weights = pd.DataFrame(weight_rows, index=evidence.index, columns=factors)
    evidence_frames = {
        "recentUtility": recent_utility,
        "longUtility": long_utility,
        "combinedEvidence": evidence,
    }
    return weights, pd.DataFrame(diagnostics), evidence_frames


def selection_diagnostics(
    mask: pd.DataFrame,
    outcome: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    selected_mask = mask.reindex(index=dates).fillna(False)
    selected = outcome.reindex(index=dates).where(selected_mask)
    stacked = selected.stack(future_stack=True).dropna()
    eligible = outcome.reindex(index=dates).where(
        execution_eligible.reindex(index=dates).fillna(False)
    )
    threshold = eligible.quantile(
        float(config["data"]["extremeWinnerCrossSectionQuantile"]), axis=1
    )
    all_extreme = eligible.ge(threshold, axis=0)
    selected_extreme = all_extreme.where(selected_mask)
    selected_extreme_values = selected_extreme.stack(future_stack=True).dropna()
    captured = int(selected_extreme_values.sum()) if len(selected_extreme_values) else 0
    available_extreme = int(all_extreme.sum(axis=1).sum()) if len(all_extreme) else 0
    return {
        "stockObservations": int(len(stacked)),
        "stockGrossWinRate": float(stacked.gt(0.0).mean()) if len(stacked) else None,
        "stockMeanGrossReturn": float(stacked.mean()) if len(stacked) else None,
        "stockSevereLossRateDirect": (
            float(stacked.le(float(config["data"]["severeLossThreshold"])).mean())
            if len(stacked)
            else None
        ),
        "selectedExtremeWinnerRate": (
            float(selected_extreme_values.mean())
            if len(selected_extreme_values)
            else None
        ),
        "extremeWinnersCaptured": captured,
        "availableExtremeWinners": available_extreme,
        "extremeWinnerUniverseCaptureRate": (
            captured / available_extreme if available_extreme else None
        ),
    }


def summarize_book(
    mask: pd.DataFrame,
    outcome: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    exit_delay: pd.DataFrame,
    dates: pd.DatetimeIndex,
    frozen: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = precision.summarize_selection(
        mask, outcome, exit_delay, dates, frozen, 1
    )
    summary.update(
        selection_diagnostics(mask, outcome, execution_eligible, dates, config)
    )
    return summary


def market_state_dates(
    outcome: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, pd.DatetimeIndex]:
    market = outcome.reindex(index=dates).where(
        execution_eligible.reindex(index=dates).fillna(False)
    ).mean(axis=1, skipna=True)
    return {
        "marketDown": pd.DatetimeIndex(market.index[market.lt(0.0)]),
        "marketUpOrFlat": pd.DatetimeIndex(market.index[market.ge(0.0)]),
    }


def build_verdict(
    periods: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    threshold = float(config["evaluation"]["pairedDailyHacTMinimum"])
    checks = {}
    for period in ("validation", "shadow"):
        static = periods[period]["static"]
        adaptive = periods[period]["adaptive"]
        paired = periods[period]["pairedDifference"]
        checks[period] = {
            "sameObservationCount": adaptive["intendedObservations"]
            == static["intendedObservations"],
            "meanGrossImproves": adaptive["dailyMeanGrossReturn"]
            > static["dailyMeanGrossReturn"],
            "stockWinRateImproves": adaptive["stockGrossWinRate"]
            > static["stockGrossWinRate"],
            "extremeWinnerRateNotWorse": adaptive["selectedExtremeWinnerRate"]
            >= static["selectedExtremeWinnerRate"],
            "severeLossNotWorse": adaptive["stockSevereLossRateDirect"]
            <= static["stockSevereLossRateDirect"],
            "pairedHacTMeetsThreshold": paired["pairedHacT"] is not None
            and paired["pairedHacT"] >= threshold,
        }
        checks[period]["passed"] = all(checks[period].values())
    historical_pass = all(item["passed"] for item in checks.values())
    return {
        "decision": (
            "retain_as_fresh_forward_challenger_only"
            if historical_pass
            else "reject_feedback_and_keep_frozen_prior"
        ),
        "externalRejectOnlyChecks": checks,
        "historicalHypothesisPass": historical_pass,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Causal Top10 Factor Feedback V2",
        "",
        f"- Status: `{report['status']}`",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        "- Rule: daily update from the latest seven fully resolved factor-expert Top10 outcomes",
        "- Objective: gross return, win rate, top-5% winner capture and severe-loss avoidance",
        "- No abstention: static and adaptive books both select exactly ten stocks",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`",
        "- Orders: `[]`",
        "",
        "## Static versus feedback-adaptive Top10",
        "",
        "| Period | Book | Days | Gross mean | Stock win | Extreme winner rate | Severe loss | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period, payload in report["periods"].items():
        for book in ("static", "adaptive"):
            value = payload[book]
            lines.append(
                f"| {period} | {book} | {value['signalDays']} | "
                f"{value['dailyMeanGrossReturn']} | {value['stockGrossWinRate']} | "
                f"{value['selectedExtremeWinnerRate']} | "
                f"{value['stockSevereLossRateDirect']} | "
                f"{value['dailyMaximumDrawdown']} |"
            )
        paired = payload["pairedDifference"]
        lines.append(
            f"| {period} | adaptive-static | {paired['pairedDays']} | "
            f"{paired['meanGrossDifference']} | n/a | n/a | n/a | paired HAC t="
            f"{paired['pairedHacT']} |"
        )
    lines.extend(["", "## Weak-market diagnostic", ""])
    for period in ("validation", "shadow"):
        weak = report["periods"][period]["marketStates"]["marketDown"]
        lines.append(
            f"- {period}: market-down days={weak['days']}; static win="
            f"{weak['static']['stockGrossWinRate']}; adaptive win="
            f"{weak['adaptive']['stockGrossWinRate']}; static mean="
            f"{weak['static']['stockMeanGrossReturn']}; adaptive mean="
            f"{weak['adaptive']['stockMeanGrossReturn']}."
        )
    lines.extend(
        [
            "",
            "The market-state split is an evaluation label only. It never enters the same-day ranking.",
            "Historical validation and shadow are reject-only and cannot authorise trading.",
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
    frozen, source, frozen_sha = guarded.load_frozen_config(config)
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
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    metrics, _expert_masks = expert_top10_metrics(
        ranks, outcome, execution_eligible, config
    )
    adaptive_weights, feedback_audit, evidence = feedback_weight_path(
        metrics, prior, config
    )
    adaptive_score = guarded.adaptive_score(ranks, adaptive_weights, panel)
    support = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
    top_count = int(config["data"]["topCount"])
    static_mask = precision.selection_mask(
        static_score, support, execution_eligible, 0, top_count
    )
    adaptive_mask = precision.selection_mask(
        adaptive_score, support, execution_eligible, 0, top_count
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
    periods = {}
    for period, dates in contained.items():
        static_summary = summarize_book(
            static_mask,
            outcome,
            execution_eligible,
            exit_delay,
            dates,
            frozen,
            config,
        )
        adaptive_summary = summarize_book(
            adaptive_mask,
            outcome,
            execution_eligible,
            exit_delay,
            dates,
            frozen,
            config,
        )
        state_payload = {}
        for state, state_dates in market_state_dates(
            outcome, execution_eligible, dates
        ).items():
            state_payload[state] = {
                "days": int(len(state_dates)),
                "static": selection_diagnostics(
                    static_mask,
                    outcome,
                    execution_eligible,
                    state_dates,
                    config,
                ),
                "adaptive": selection_diagnostics(
                    adaptive_mask,
                    outcome,
                    execution_eligible,
                    state_dates,
                    config,
                ),
            }
        periods[period] = {
            "static": static_summary,
            "adaptive": adaptive_summary,
            "pairedDifference": guarded.paired_daily_difference(
                static_mask, adaptive_mask, outcome, dates
            ),
            "marketStates": state_payload,
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
    latest_date = adaptive_score.index.max()
    latest_weights = pd.DataFrame(
        {
            "factorKey": adaptive_weights.columns,
            "frozenWeight": prior.reindex(adaptive_weights.columns).to_numpy(dtype=float),
            "adaptiveWeight": adaptive_weights.loc[latest_date].to_numpy(dtype=float),
            "recentUtility": evidence["recentUtility"].loc[latest_date].to_numpy(dtype=float),
            "longUtility": evidence["longUtility"].loc[latest_date].to_numpy(dtype=float),
            "combinedEvidence": evidence["combinedEvidence"].loc[latest_date].to_numpy(dtype=float),
        }
    )
    latest_weights["relativeToFrozen"] = (
        latest_weights["adaptiveWeight"] / latest_weights["frozenWeight"]
    )
    latest_weights["signalDate"] = latest_date.date().isoformat()
    latest_weights["status"] = "diagnostic_only_not_trading"
    latest_scores = pd.DataFrame(
        {
            "securityId": adaptive_score.columns,
            "adaptiveScore": adaptive_score.loc[latest_date].to_numpy(dtype=float),
        }
    ).dropna().sort_values(
        ["adaptiveScore", "securityId"], ascending=[False, True]
    ).reset_index(drop=True)
    latest_scores.insert(0, "rank", np.arange(1, len(latest_scores) + 1))
    latest_scores.insert(0, "signalDate", latest_date.date().isoformat())
    latest_scores["status"] = "diagnostic_only_not_an_order"
    latest_top10 = latest_scores.head(top_count).copy()
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
        "feedbackAudit": {
            "dailyRows": int(len(feedback_audit)),
            "priorFallbackDays": int(
                feedback_audit["usedFrozenPriorFallback"].sum()
            ),
            "maximumRealisedL1Turnover": float(
                feedback_audit["realisedL1Turnover"].max()
            ),
            "meanRealisedL1Turnover": float(
                feedback_audit["realisedL1Turnover"].mean()
            ),
            "minimumWeight": float(adaptive_weights.min().min()),
            "maximumWeight": float(adaptive_weights.max().max()),
            "maximumWeightSumError": float(
                (adaptive_weights.sum(axis=1) - 1.0).abs().max()
            ),
            "latestWeights": precision.json_safe(
                latest_weights.to_dict(orient="records")
            ),
        },
        "periods": periods,
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
        output / "feedback_diagnostics.csv",
        feedback_audit.to_csv(index=False, lineterminator="\n"),
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
        latest_scores.to_csv(index=False, lineterminator="\n"),
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
