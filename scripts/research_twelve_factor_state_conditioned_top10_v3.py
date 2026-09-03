#!/usr/bin/env python3
"""Past-only nearest-market-state factor experts for A-share Top10 research."""

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
import research_twelve_factor_top10_feedback_v2 as feedback_v2  # noqa: E402


SCHEMA_VERSION = "twelve_factor_state_conditioned_top10_result_v3"
CODE_VERSION = "twelve_factor_state_conditioned_top10_v3_20260829"
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "twelve_factor_state_conditioned_top10_v3.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schemaVersion") != "twelve_factor_state_conditioned_top10_v3":
        raise ValueError("unexpected state-conditioned Top10 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("state-conditioned Top10 must remain research-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("all state-conditioned mutation permissions must be false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("state-conditioned output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    for key in (
        "factorDefinitionsDirectionsAndPriorWeightsFrozen",
        "stateFeaturesFrozen",
        "singleNearestStateRule",
        "parametersFrozenBeforeHistoricalEvaluation",
        "validationAndShadowAreRejectOnly",
    ):
        if hypothesis.get(key) is not True:
            raise ValueError(f"missing preregistration flag: {key}")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical state matching cannot promote")
    base_path = ROOT / config["baseFeedbackConfig"]
    raw = base_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config["baseFeedbackConfigSha256"]:
        raise ValueError("base Top10-feedback config hash mismatch")
    base = json.loads(raw.decode("utf-8"))
    feedback_v2.validate_config(base)
    state = config["state"]
    expected_features = [
        "market_return_5",
        "market_return_20",
        "market_volatility_20",
        "market_downside_semivariance_20",
        "market_breadth_20",
        "market_return_dispersion_1",
        "market_liquidity_shock_20",
    ]
    if state.get("featureColumns") != expected_features:
        raise ValueError("frozen market-state features changed")
    lookback = int(state["historicalLookbackTradingDays"])
    neighbors = int(state["nearestResolvedStates"])
    minimum = int(state["minimumNearestResolvedStates"])
    if not (252 <= lookback <= 756 and 10 <= minimum <= neighbors <= 40):
        raise ValueError("nearest-state history or sample size is outside bounds")
    if state.get("robustScale") != "past_candidate_median_and_mad_only":
        raise ValueError("state scaling must remain past-only")
    if abs(
        float(state["stateEvidenceWeight"])
        + float(state["globalEvidenceWeight"])
        - 1.0
    ) > 1e-12:
        raise ValueError("state/global evidence weights must sum to one")
    weights = config["weighting"]
    if set(weights["metricWeights"]) != set(feedback_v2.METRICS):
        raise ValueError("V3 must use the same four Top10 objectives as V2")
    if abs(sum(map(float, weights["metricWeights"].values())) - 1.0) > 1e-12:
        raise ValueError("V3 metric weights must sum to one")
    if not (0.0 < float(weights["adaptiveAllocation"]) <= 0.5):
        raise ValueError("V3 must remain strongly shrunk to the prior")
    if not (0.0 < float(weights["maximumOneUpdateL1Turnover"]) <= 0.05):
        raise ValueError("V3 daily turnover cap is too large")
    evaluation = config["evaluation"]
    if evaluation.get("noAbstentionAllowed") is not True:
        raise ValueError("V3 may not abstain")
    if evaluation.get("sameTop10CountRequired") is not True:
        raise ValueError("V3 must preserve Top10 capacity")
    if evaluation.get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical windows must remain reject-only")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("V3 orders must remain empty")
    return base


def market_state_features(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build state features known by each signal-day close."""
    eligible = panel["eligible"]
    returns = panel["returns"].where(eligible)
    close = panel["close"]
    amount = panel["amount"].where(eligible)
    market_daily = returns.median(axis=1, skipna=True)
    market_return_5 = market_daily.rolling(5, min_periods=3).sum()
    market_return_20 = market_daily.rolling(20, min_periods=10).sum()
    market_volatility_20 = market_daily.rolling(20, min_periods=10).std()
    downside = market_daily.clip(upper=0.0).pow(2)
    market_downside_semivariance_20 = downside.rolling(20, min_periods=10).mean()
    moving_average = close.rolling(20, min_periods=10).mean()
    denominator = eligible.sum(axis=1).replace(0, np.nan)
    market_breadth_20 = (
        (close.gt(moving_average) & eligible).sum(axis=1).div(denominator)
    )
    market_return_dispersion_1 = returns.quantile(0.75, axis=1) - returns.quantile(
        0.25, axis=1
    )
    prior_amount = amount.rolling(20, min_periods=10).median().shift(1)
    amount_ratio = amount.div(prior_amount.replace(0.0, np.nan))
    market_liquidity_shock_20 = np.log(
        amount_ratio.clip(lower=1e-06)
    ).median(axis=1, skipna=True)
    return pd.DataFrame(
        {
            "market_return_5": market_return_5,
            "market_return_20": market_return_20,
            "market_volatility_20": market_volatility_20,
            "market_downside_semivariance_20": market_downside_semivariance_20,
            "market_breadth_20": market_breadth_20,
            "market_return_dispersion_1": market_return_dispersion_1,
            "market_liquidity_shock_20": market_liquidity_shock_20,
        },
        index=close.index,
    ).replace([np.inf, -np.inf], np.nan)


def _rank_utility(values: pd.DataFrame, metric_weights: dict[str, float]) -> pd.Series:
    utility = pd.Series(0.0, index=values.columns)
    for metric in feedback_v2.METRICS:
        ranked = values.loc[metric].rank(method="average", pct=True).sub(0.5)
        utility = utility.add(ranked * float(metric_weights[metric]), fill_value=0.0)
    return utility


def _metric_means(
    metric_frames: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    factors: list[str],
) -> pd.DataFrame:
    rows = {}
    for metric in feedback_v2.METRICS:
        rows[metric] = (
            metric_frames[metric]
            .reindex(index=dates, columns=factors)
            .mean(axis=0, skipna=True)
        )
    return pd.DataFrame(rows).T.reindex(index=feedback_v2.METRICS, columns=factors)


def state_conditioned_weight_path(
    metric_frames: dict[str, pd.DataFrame],
    states: pd.DataFrame,
    prior_weights: pd.Series,
    base_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select factor evidence from nearest fully resolved historical states."""
    state_spec = config["state"]
    weight_spec = config["weighting"]
    lag = int(base_config["data"]["fullOutcomeAvailabilityLagTradingDays"])
    dates = states.index
    factors = list(prior_weights.index)
    prior = prior_weights.reindex(factors).astype(float)
    prior /= prior.sum()
    lower = prior.to_numpy(dtype=float) * float(weight_spec["minimumRelativeWeight"])
    upper = prior.to_numpy(dtype=float) * float(weight_spec["maximumRelativeWeight"])
    lookback = int(state_spec["historicalLookbackTradingDays"])
    neighbor_count = int(state_spec["nearestResolvedStates"])
    minimum_neighbors = int(state_spec["minimumNearestResolvedStates"])
    global_days = int(state_spec["globalResolvedSignalDays"])
    minimum_global = int(state_spec["minimumGlobalResolvedSignalDays"])
    current = prior.to_numpy(dtype=float, copy=True)
    weight_rows = []
    evidence_rows = []
    diagnostics = []
    for position, date in enumerate(dates):
        resolved_position = position - lag
        start = max(0, resolved_position - lookback + 1)
        candidate_dates = dates[start : resolved_position + 1] if resolved_position >= 0 else dates[:0]
        candidate_states = states.reindex(candidate_dates).dropna(how="any")
        current_state = states.loc[date]
        used_prior = True
        chosen = pd.DatetimeIndex([])
        median_distance = None
        evidence = pd.Series(np.nan, index=factors, dtype=float)
        if len(candidate_states) >= minimum_neighbors and current_state.notna().all():
            center = candidate_states.median(axis=0)
            mad = candidate_states.sub(center).abs().median(axis=0) * 1.4826
            fallback = candidate_states.std(axis=0, ddof=1)
            scale = mad.where(mad.gt(1e-12), fallback).where(lambda x: x.gt(1e-12), 1.0)
            distances = (
                candidate_states.sub(current_state, axis=1)
                .div(scale, axis=1)
                .pow(2)
                .mean(axis=1)
                .pow(0.5)
            )
            chosen = pd.DatetimeIndex(
                distances.sort_values(kind="stable").head(neighbor_count).index
            )
            global_dates = pd.DatetimeIndex(candidate_dates[-global_days:])
            state_values = _metric_means(metric_frames, chosen, factors)
            global_values = _metric_means(metric_frames, global_dates, factors)
            state_counts = (
                metric_frames["top10GrossReturn"]
                .reindex(index=chosen, columns=factors)
                .count(axis=0)
            )
            global_counts = (
                metric_frames["top10GrossReturn"]
                .reindex(index=global_dates, columns=factors)
                .count(axis=0)
            )
            if (
                len(chosen) >= minimum_neighbors
                and bool(state_counts.ge(minimum_neighbors).all())
                and bool(global_counts.ge(minimum_global).all())
                and state_values.notna().all(axis=None)
                and global_values.notna().all(axis=None)
            ):
                state_utility = _rank_utility(
                    state_values, weight_spec["metricWeights"]
                )
                global_utility = _rank_utility(
                    global_values, weight_spec["metricWeights"]
                )
                evidence = (
                    float(state_spec["stateEvidenceWeight"]) * state_utility
                    + float(state_spec["globalEvidenceWeight"]) * global_utility
                )
                used_prior = False
                median_distance = float(distances.reindex(chosen).median())
        if used_prior:
            target = prior.to_numpy(dtype=float, copy=True)
        else:
            tilted = prior.to_numpy(dtype=float) * np.exp(
                float(weight_spec["exponentLearningRate"])
                * evidence.to_numpy(dtype=float)
            )
            tilted /= tilted.sum()
            allocation = float(weight_spec["adaptiveAllocation"])
            raw_target = (1.0 - allocation) * prior.to_numpy(dtype=float) + allocation * tilted
            target = guarded._bounded_simplex(raw_target, lower, upper)
        proposed = float(np.abs(target - current).sum())
        cap = float(weight_spec["maximumOneUpdateL1Turnover"])
        if proposed > cap:
            target = current + (target - current) * (cap / proposed)
        realised = float(np.abs(target - current).sum())
        current = target
        weight_rows.append(current.copy())
        evidence_rows.append(evidence.reindex(factors).to_numpy(dtype=float))
        diagnostics.append(
            {
                "signalDate": date.date().isoformat(),
                "usedFrozenPriorFallback": used_prior,
                "resolvedHistoryEnd": (
                    candidate_dates[-1].date().isoformat() if len(candidate_dates) else None
                ),
                "nearestStateCount": int(len(chosen)),
                "nearestStateStart": chosen.min().date().isoformat() if len(chosen) else None,
                "nearestStateEnd": chosen.max().date().isoformat() if len(chosen) else None,
                "medianRobustDistance": median_distance,
                "proposedL1Turnover": proposed,
                "realisedL1Turnover": realised,
            }
        )
    weights = pd.DataFrame(weight_rows, index=dates, columns=factors)
    evidence_frame = pd.DataFrame(evidence_rows, index=dates, columns=factors)
    return weights, pd.DataFrame(diagnostics), evidence_frame


def build_verdict(periods: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config["evaluation"]["pairedDailyHacTMinimum"])
    checks = {}
    for period in ("validation", "shadow"):
        static = periods[period]["static"]
        recent = periods[period]["recentSeven"]
        state = periods[period]["stateConditioned"]
        paired = periods[period]["stateMinusStatic"]
        checks[period] = {
            "sameObservationCount": state["intendedObservations"]
            == static["intendedObservations"]
            == recent["intendedObservations"],
            "meanGrossImprovesVsStatic": state["dailyMeanGrossReturn"]
            > static["dailyMeanGrossReturn"],
            "meanGrossNotWorseThanRecentSeven": state["dailyMeanGrossReturn"]
            >= recent["dailyMeanGrossReturn"],
            "stockWinRateImprovesVsStatic": state["stockGrossWinRate"]
            > static["stockGrossWinRate"],
            "extremeWinnerRateNotWorse": state["selectedExtremeWinnerRate"]
            >= static["selectedExtremeWinnerRate"],
            "severeLossNotWorse": state["stockSevereLossRateDirect"]
            <= static["stockSevereLossRateDirect"],
            "pairedHacTMeetsThreshold": paired["pairedHacT"] is not None
            and paired["pairedHacT"] >= threshold,
        }
        checks[period]["passed"] = all(checks[period].values())
    passed = all(item["passed"] for item in checks.values())
    return {
        "decision": (
            "retain_state_conditioned_v3_as_fresh_forward_challenger_only"
            if passed
            else "reject_state_conditioning_and_keep_frozen_prior"
        ),
        "externalRejectOnlyChecks": checks,
        "historicalHypothesisPass": passed,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# State-Conditioned Top10 Factor Experts V3",
        "",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        "- Comparison: frozen static vs recent-seven V2 vs nearest-state V3",
        "- Every book selects exactly ten stocks; no market-day abstention",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`; orders: `[]`",
        "",
        "| Period | Book | Gross mean | Stock win | Extreme rate | Severe loss | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period, payload in report["periods"].items():
        for book in ("static", "recentSeven", "stateConditioned"):
            value = payload[book]
            lines.append(
                f"| {period} | {book} | {value['dailyMeanGrossReturn']} | "
                f"{value['stockGrossWinRate']} | {value['selectedExtremeWinnerRate']} | "
                f"{value['stockSevereLossRateDirect']} | {value['dailyMaximumDrawdown']} |"
            )
        paired = payload["stateMinusStatic"]
        lines.append(
            f"| {period} | V3-static | {paired['meanGrossDifference']} | n/a | n/a | n/a | HAC t="
            f"{paired['pairedHacT']} |"
        )
    lines.extend(["", "## Weak-market comparison", ""])
    for period in ("validation", "shadow"):
        weak = report["periods"][period]["marketStates"]["marketDown"]
        lines.append(
            f"- {period}, days={weak['days']}: static win={weak['static']['stockGrossWinRate']}; V2 win="
            f"{weak['recentSeven']['stockGrossWinRate']}; V3 win="
            f"{weak['stateConditioned']['stockGrossWinRate']}; V3 mean="
            f"{weak['stateConditioned']['stockMeanGrossReturn']}."
        )
    lines.extend(
        [
            "",
            "Market-down/up is an evaluation label only and never enters the same-day ranking.",
            "Historical windows are reject-only and cannot authorise trading.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    base_feedback = validate_config(config)
    frozen, source, frozen_sha = guarded.load_frozen_config(base_feedback)
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        int(base_feedback["data"]["holdingTradingDays"]),
        int(base_feedback["data"]["maximumExitDelayTradingDays"]),
    )
    ranks, static_score, factor_audit = rolling.compute_rank_book(panel, frozen)
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    metrics, _ = feedback_v2.expert_top10_metrics(
        ranks, outcome, execution_eligible, base_feedback
    )
    recent_weights, _, _ = feedback_v2.feedback_weight_path(
        metrics, prior, base_feedback
    )
    states = market_state_features(panel)
    state_weights, state_audit, state_evidence = state_conditioned_weight_path(
        metrics, states, prior, base_feedback, config
    )
    recent_score = guarded.adaptive_score(ranks, recent_weights, panel)
    state_score = guarded.adaptive_score(ranks, state_weights, panel)
    support = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
    top_count = int(base_feedback["data"]["topCount"])
    masks = {
        "static": precision.selection_mask(
            static_score, support, execution_eligible, 0, top_count
        ),
        "recentSeven": precision.selection_mask(
            recent_score, support, execution_eligible, 0, top_count
        ),
        "stateConditioned": precision.selection_mask(
            state_score, support, execution_eligible, 0, top_count
        ),
    }
    splits = precision.split_dates(panel["close"].index, source)
    contained = {
        name: precision.contained_signal_dates(
            dates,
            int(base_feedback["data"]["holdingTradingDays"]),
            int(base_feedback["data"]["maximumExitDelayTradingDays"]),
        )
        for name, dates in splits.items()
    }
    periods = {}
    for period, dates in contained.items():
        payload = {}
        for book, mask in masks.items():
            payload[book] = feedback_v2.summarize_book(
                mask,
                outcome,
                execution_eligible,
                exit_delay,
                dates,
                frozen,
                base_feedback,
            )
        payload["stateMinusStatic"] = guarded.paired_daily_difference(
            masks["static"], masks["stateConditioned"], outcome, dates
        )
        payload["stateMinusRecentSeven"] = guarded.paired_daily_difference(
            masks["recentSeven"], masks["stateConditioned"], outcome, dates
        )
        states_payload = {}
        for market_state, state_dates in feedback_v2.market_state_dates(
            outcome, execution_eligible, dates
        ).items():
            state_value = {"days": int(len(state_dates))}
            for book, mask in masks.items():
                state_value[book] = feedback_v2.selection_diagnostics(
                    mask, outcome, execution_eligible, state_dates, base_feedback
                )
            states_payload[market_state] = state_value
        payload["marketStates"] = states_payload
        periods[period] = payload
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
    latest_date = state_score.index.max()
    latest_weights = pd.DataFrame(
        {
            "factorKey": state_weights.columns,
            "frozenWeight": prior.reindex(state_weights.columns).to_numpy(dtype=float),
            "recentSevenWeight": recent_weights.loc[latest_date].to_numpy(dtype=float),
            "stateConditionedWeight": state_weights.loc[latest_date].to_numpy(dtype=float),
            "stateEvidence": state_evidence.loc[latest_date].to_numpy(dtype=float),
        }
    )
    latest_weights["relativeToFrozen"] = (
        latest_weights["stateConditionedWeight"] / latest_weights["frozenWeight"]
    )
    latest_weights["signalDate"] = latest_date.date().isoformat()
    latest_scores = pd.DataFrame(
        {
            "securityId": state_score.columns,
            "stateConditionedScore": state_score.loc[latest_date].to_numpy(dtype=float),
        }
    ).dropna().sort_values(
        ["stateConditionedScore", "securityId"], ascending=[False, True]
    ).reset_index(drop=True)
    latest_scores.insert(0, "rank", np.arange(1, len(latest_scores) + 1))
    latest_scores.insert(0, "signalDate", latest_date.date().isoformat())
    latest_scores["status"] = "diagnostic_only_not_an_order"
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": now.isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path.resolve()),
        "configSha256": precision.digest(config),
        "baseFeedbackConfigSha256": config["baseFeedbackConfigSha256"],
        "frozenPrecisionConfigSha256": frozen_sha,
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "panelAudit": panel_audit,
        "splitAudit": source["splitAudit"],
        "factorAudit": factor_audit,
        "stateFeatureAudit": {
            "featureColumns": list(states.columns),
            "firstCompleteDate": states.dropna(how="any").index.min().date().isoformat(),
            "completeRows": int(states.notna().all(axis=1).sum()),
            "priorFallbackDays": int(state_audit["usedFrozenPriorFallback"].sum()),
            "maximumResolvedHistoryLeadViolation": int(
                (
                    pd.to_datetime(state_audit["resolvedHistoryEnd"])
                    >= pd.to_datetime(state_audit["signalDate"])
                ).fillna(False).sum()
            ),
            "maximumWeightSumError": float(
                (state_weights.sum(axis=1) - 1.0).abs().max()
            ),
            "maximumRealisedL1Turnover": float(
                state_audit["realisedL1Turnover"].max()
            ),
        },
        "periods": periods,
        "latestState": precision.json_safe(states.loc[latest_date].to_dict()),
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
    precision.atomic_write(
        output / "state_features.csv",
        states.rename_axis("signalDate").reset_index().to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "state_match_diagnostics.csv",
        state_audit.to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "weights_daily.csv",
        state_weights.rename_axis("signalDate").reset_index().to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_weights.csv",
        latest_weights.to_csv(index=False, lineterminator="\n"),
    )
    precision.atomic_write(
        output / "latest_diagnostic_top10.csv",
        latest_scores.head(top_count).to_csv(index=False, lineterminator="\n"),
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
