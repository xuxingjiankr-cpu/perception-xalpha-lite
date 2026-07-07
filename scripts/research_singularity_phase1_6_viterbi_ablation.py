"""Research-only causal Viterbi ablation for Singularity Phase 1.6.

The experiment decodes the existing frozen GaussianHMM1D state process with an
online Viterbi endpoint algorithm.  It tests whether prefix-only path stability
features add calibrated 5/10-bar turning-risk information beyond the current
HMM filtered probabilities plus EWS baseline.

This module cannot trade, cannot write Phase 1.5 artifacts, and cannot touch
the paper agent, overlays, risk gates or decision_probability_v1.json.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision_probability import probability_metrics
from research_hmm_nn_bl import ROOT
import research_singularity_phase1_6_hmm_physics_features as phase16
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "singularity_phase1_6_viterbi_ablation_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schemaVersion")
        != "singularity_phase1_6_viterbi_ablation_v1"
    ):
        raise ValueError("unexpected Viterbi ablation schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("Viterbi ablation must remain research/shadow-only")
    if config["source"].get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 forward ledger cannot be an input")
    viterbi = config["viterbi"]
    if (
        viterbi.get("mode") != "online_endpoint_decode"
        or viterbi.get("sessionReset") is not True
        or viterbi.get("usesFullSequenceBacktracking") is not False
        or viterbi.get("usesForwardBackwardSmoothing") is not False
        or viterbi.get("usesFutureObservations") is not False
        or viterbi.get("automaticStateOrParameterSearchAllowed") is not False
        or viterbi.get("newHmmFitAllowed") is not False
    ):
        raise ValueError("Viterbi must be causal, fixed, and prefix-only")
    safety = config["safety"]
    forbidden = [
        "brokerCallsAllowed",
        "onlineInferenceAllowed",
        "phase15WritesAllowed",
        "liveConfigWritesAllowed",
        "overlayWritesAllowed",
        "positionSizingAllowed",
        "orderSubmissionAllowed",
        "riskGateChangesAllowed",
        "buildDecisionIntegrationAllowed",
        "buySellGateIntegrationAllowed",
        "forwardTaskIntegrationAllowed",
        "promotionAllowed",
    ]
    if not safety.get("offlineOnly") or not safety.get("recordOnly"):
        raise ValueError("offline record-only safety flags are required")
    if any(safety.get(key) is not False for key in forbidden):
        raise ValueError("all live mutation flags must be false")
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
        raise ValueError("paper integration is not allowed")


def _emission_log_probability(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    variance_floor: float = 1e-8,
) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), -0.05, 0.05)[:, None]
    safe_variances = np.maximum(np.asarray(variances, dtype=float), variance_floor)
    return -0.5 * (
        np.log(2.0 * math.pi * safe_variances)
        + ((x - means) ** 2) / safe_variances
    )


def online_viterbi_decode(
    values: np.ndarray,
    hmm_audit: dict[str, Any],
) -> pd.DataFrame:
    """Return prefix-only Viterbi endpoint diagnostics for one session.

    The row at index t depends only on values[: t + 1].  It intentionally does
    not backtrack using observations after t.
    """

    observations = np.clip(np.asarray(values, dtype=float), -0.05, 0.05)
    means = np.asarray(hmm_audit["means"], dtype=float)
    variances = np.asarray(hmm_audit["variances"], dtype=float)
    start = np.asarray(hmm_audit["startProbability"], dtype=float)
    transition = np.asarray(hmm_audit["transitionMatrix"], dtype=float)
    n_states = len(means)
    emissions = _emission_log_probability(observations, means, variances)
    log_start = np.log(np.maximum(start, 1e-300))
    log_transition = np.log(np.maximum(transition, 1e-300))
    delta = np.empty((len(observations), n_states), dtype=float)
    states = np.empty(len(observations), dtype=int)
    confidence = np.empty(len(observations), dtype=float)
    margin = np.empty(len(observations), dtype=float)
    transition_risk = np.empty(len(observations), dtype=float)
    for index in range(len(observations)):
        if index == 0:
            delta[index] = log_start + emissions[index]
        else:
            delta[index] = emissions[index] + np.max(
                delta[index - 1][:, None] + log_transition,
                axis=0,
            )
        state = int(np.argmax(delta[index]))
        states[index] = state
        normalized = np.exp(delta[index] - logsumexp(delta[index]))
        confidence[index] = float(normalized[state])
        ordered = np.sort(delta[index])
        margin[index] = float(ordered[-1] - ordered[-2]) if n_states > 1 else 0.0
        transition_risk[index] = float(1.0 - transition[state, state])
    age = np.ones(len(states), dtype=float)
    for index in range(1, len(states)):
        age[index] = age[index - 1] + 1.0 if states[index] == states[index - 1] else 1.0
    return pd.DataFrame(
        {
            "viterbi_state": states.astype(float),
            "viterbi_bear_state": (states == 0).astype(float),
            "viterbi_middle_state": (states == 1).astype(float),
            "viterbi_bull_state": (states == 2).astype(float),
            "viterbi_state_age_bars": age,
            "viterbi_path_transition_risk": transition_risk,
            "viterbi_endpoint_confidence": confidence,
            "viterbi_log_margin": margin,
        }
    )


def prefix_invariance_check(hmm_audit: dict[str, Any]) -> bool:
    rng = np.random.default_rng(20260708)
    prefix = rng.normal(0.0, 0.001, size=36)
    future_shock = np.asarray([0.05, -0.05, 0.04, -0.04])
    first = online_viterbi_decode(prefix, hmm_audit)
    with_future = online_viterbi_decode(np.concatenate([prefix, future_shock]), hmm_audit)
    columns = [
        "viterbi_state",
        "viterbi_bear_state",
        "viterbi_middle_state",
        "viterbi_bull_state",
        "viterbi_state_age_bars",
        "viterbi_path_transition_risk",
        "viterbi_endpoint_confidence",
        "viterbi_log_margin",
    ]
    return bool(
        np.allclose(
            first[columns].to_numpy(dtype=float),
            with_future.iloc[: len(prefix)][columns].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
    )


def build_viterbi_frame(
    market_return: pd.Series,
    hmm_audit: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    series = market_return.sort_index()
    dates = pd.Series(series.index.strftime("%Y-%m-%d"), index=series.index)
    for trade_date, values in series.groupby(dates, sort=True):
        decoded = online_viterbi_decode(values.to_numpy(dtype=float), hmm_audit)
        decoded["timestamp"] = values.index
        decoded["trade_date"] = trade_date
        rows.append(decoded)
    frame = pd.concat(rows, ignore_index=True)
    state = frame["viterbi_state"].to_numpy(dtype=int)
    switches = int(np.sum(state[1:] != state[:-1])) if len(state) > 1 else 0
    possible = max(len(state) - 1, 0)
    audit = {
        "rows": int(len(frame)),
        "firstTimestamp": str(frame["timestamp"].min()),
        "lastTimestamp": str(frame["timestamp"].max()),
        "sessionReset": True,
        "prefixOnly": True,
        "fullSequenceBacktrackingUsed": False,
        "forwardBackwardSmoothingUsed": False,
        "prefixInvarianceCheckPassed": prefix_invariance_check(hmm_audit),
        "endpointStateSwitchRate": switches / possible if possible else None,
        "meanEndpointConfidence": float(frame["viterbi_endpoint_confidence"].mean()),
        "meanStateAgeBars": float(frame["viterbi_state_age_bars"].mean()),
    }
    return frame, audit


def variant_features(config: dict[str, Any]) -> dict[str, list[str]]:
    base = list(config["features"]["base"])
    ews = list(config["features"]["ews"])
    hmm = list(config["features"]["hmm"])
    viterbi = list(config["features"]["viterbi"])
    return {
        "constant_prior": [],
        "ews_baseline": base + ews,
        "current_hmm": base + hmm,
        "current_hmm_ews": base + hmm + ews,
        "hmm_viterbi": base + hmm + viterbi,
        "hmm_ews_viterbi": base + hmm + ews + viterbi,
    }


def purge_tail(rows: pd.DataFrame, count: int) -> tuple[pd.DataFrame, int]:
    ordered = rows.sort_values(["stockCode", "timestamp"], kind="stable")
    remove = ordered.groupby("stockCode", sort=False).tail(count).index
    return ordered.drop(index=remove).reset_index(drop=True), int(len(remove))


def winsor_bounds(
    rows: pd.DataFrame,
    columns: list[str],
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.asarray([]), np.asarray([])
    values = rows[columns].to_numpy(dtype=float)
    return (
        np.quantile(values, quantiles[0], axis=0),
        np.quantile(values, quantiles[1], axis=0),
    )


def clipped_values(
    rows: pd.DataFrame,
    columns: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(rows[columns].to_numpy(dtype=float), lower, upper)


def metric_payload(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    metrics = probability_metrics(
        rows["probability"].to_numpy(dtype=float),
        rows["actual"].to_numpy(dtype=int),
        bin_edges=config["models"]["probabilityBinEdges"],
    )
    threshold = float(config["models"]["highRiskThreshold"])
    high = rows["probability"].to_numpy(dtype=float) >= threshold
    metrics.update(
        {
            "highRiskThreshold": threshold,
            "highRiskCount": int(np.sum(high)),
            "highRiskMeanProbability": (
                float(rows.loc[high, "probability"].mean())
                if np.any(high)
                else None
            ),
            "highRiskHitRate": (
                float(rows.loc[high, "actual"].mean())
                if np.any(high)
                else None
            ),
        }
    )
    return metrics


def run_walk_forward(
    table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    feature_map = variant_features(config)
    union = sorted(
        feature
        for variant in config["variants"]
        for feature in feature_map[variant]
    )
    rows_for_all = table.dropna(subset=union).copy()
    start = config["data"]["walkForwardStart"]
    end = config["data"]["walkForwardEnd"]
    block_days = int(config["models"]["walkForwardTestBlockDays"])
    calibration_days = int(config["models"]["calibrationDays"])
    minimum_fit_days = int(config["models"]["minimumBaseFitDays"])
    embargo_days = int(config["models"]["embargoTradingDays"])
    purge_count = int(config["models"]["purgeDecisionRowsPerSymbol"])
    quantiles = config["features"]["winsorizationQuantiles"]
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for horizon in config["data"]["primaryHorizonsBars"]:
        target = f"turning_point_{horizon}"
        rows = rows_for_all.dropna(subset=[target]).copy()
        rows[target] = rows[target].astype(int)
        test_dates = sorted(
            date for date in rows["trade_date"].unique() if start <= date <= end
        )
        for fold, offset in enumerate(range(0, len(test_dates), block_days), 1):
            block = test_dates[offset : offset + block_days]
            train_raw = rows[rows["trade_date"] < block[0]].copy()
            train_dates = sorted(train_raw["trade_date"].unique())
            if len(train_dates) <= calibration_days + embargo_days + minimum_fit_days:
                continue
            test_embargo = set(train_dates[-embargo_days:])
            before_test = train_raw[~train_raw["trade_date"].isin(test_embargo)]
            before_test, test_purged = purge_tail(before_test, purge_count)
            eligible = sorted(before_test["trade_date"].unique())
            calibration_set = set(eligible[-calibration_days:])
            calibration = before_test[before_test["trade_date"].isin(calibration_set)].copy()
            before_calibration = before_test[
                ~before_test["trade_date"].isin(calibration_set)
            ]
            fit_dates = sorted(before_calibration["trade_date"].unique())
            calibration_embargo = set(fit_dates[-embargo_days:])
            fit_raw = before_calibration[
                ~before_calibration["trade_date"].isin(calibration_embargo)
            ]
            fit, calibration_purged = purge_tail(fit_raw, purge_count)
            test = rows[rows["trade_date"].isin(block)].copy()
            if (
                fit[target].nunique() != 2
                or calibration[target].nunique() != 2
                or test.empty
            ):
                continue
            audit = {
                "horizonBars": horizon,
                "fold": fold,
                "fitStart": str(fit["trade_date"].min()),
                "fitEnd": str(fit["trade_date"].max()),
                "calibrationStart": str(calibration["trade_date"].min()),
                "calibrationEnd": str(calibration["trade_date"].max()),
                "testStart": block[0],
                "testEnd": block[-1],
                "testDates": block,
                "testEmbargoDates": sorted(test_embargo),
                "calibrationEmbargoDates": sorted(calibration_embargo),
                "testBoundaryPurgedRows": test_purged,
                "calibrationBoundaryPurgedRows": calibration_purged,
                "sameTradingDateAcrossBoundaries": False,
                "variants": {},
            }
            for variant in config["variants"]:
                columns = feature_map[variant]
                if not columns:
                    raw = np.full(len(test), float(calibration[target].mean()))
                    probability = raw.copy()
                    bounds = None
                else:
                    lower, upper = winsor_bounds(fit, columns, quantiles)
                    fit_values = clipped_values(fit, columns, lower, upper)
                    calibration_values = clipped_values(
                        calibration, columns, lower, upper
                    )
                    test_values = clipped_values(test, columns, lower, upper)
                    model = Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    C=float(config["models"]["logisticC"]),
                                    max_iter=500,
                                    random_state=int(config["models"]["randomSeed"]),
                                ),
                            ),
                        ]
                    )
                    model.fit(fit_values, fit[target].to_numpy(dtype=int))
                    calibration_raw = np.clip(
                        model.predict_proba(calibration_values)[:, 1],
                        1e-6,
                        1 - 1e-6,
                    )
                    calibrator = LogisticRegression(
                        C=float(config["models"]["calibrationC"]),
                        max_iter=500,
                        random_state=int(config["models"]["randomSeed"]),
                    )
                    calibrator.fit(
                        np.log(calibration_raw / (1.0 - calibration_raw)).reshape(-1, 1),
                        calibration[target].to_numpy(dtype=int),
                    )
                    raw = np.clip(
                        model.predict_proba(test_values)[:, 1],
                        1e-6,
                        1 - 1e-6,
                    )
                    probability = calibrator.predict_proba(
                        np.log(raw / (1.0 - raw)).reshape(-1, 1)
                    )[:, 1]
                    bounds = {
                        column: {
                            "lower": float(lower[index]),
                            "upper": float(upper[index]),
                        }
                        for index, column in enumerate(columns)
                    }
                part = test[
                    [
                        "timestamp",
                        "trade_date",
                        "stockCode",
                        "month",
                        "etf_category",
                        "volatility_regime",
                        "intraday_slot",
                        target,
                    ]
                ].rename(columns={target: "actual"})
                part["horizon_bars"] = horizon
                part["fold"] = fold
                part["variant"] = variant
                part["raw_probability"] = raw
                part["probability"] = probability
                predictions.append(part)
                audit["variants"][variant] = {
                    "features": columns,
                    "fitSamples": int(len(fit)),
                    "calibrationSamples": int(len(calibration)),
                    "testSamples": int(len(test)),
                    "fitPositiveRate": float(fit[target].mean()),
                    "calibrationPositiveRate": float(calibration[target].mean()),
                    "testPositiveRate": float(test[target].mean()),
                    "winsorBounds": bounds,
                }
            audits.append(audit)
    if not predictions:
        raise RuntimeError("no Viterbi walk-forward predictions generated")
    return pd.concat(predictions, ignore_index=True), audits


def summarize_predictions(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    overall: dict[str, Any] = {}
    by_fold: dict[str, Any] = {}
    by_group: dict[str, Any] = {}
    for horizon, horizon_rows in predictions.groupby("horizon_bars"):
        key = str(int(horizon))
        overall[key] = {
            variant: metric_payload(values, config)
            for variant, values in horizon_rows.groupby("variant")
        }
        by_fold[key] = {
            str(int(fold)): {
                variant: metric_payload(values, config)
                for variant, values in fold_rows.groupby("variant")
            }
            for fold, fold_rows in horizon_rows.groupby("fold")
        }
        by_group[key] = {}
        for dimension in ["month", "etf_category", "volatility_regime", "intraday_slot"]:
            by_group[key][dimension] = {
                str(group): {
                    variant: metric_payload(values, config)
                    for variant, values in group_rows.groupby("variant")
                }
                for group, group_rows in horizon_rows.groupby(dimension)
            }
    return {"overall": overall, "byFold": by_fold, "byGroup": by_group}


def improvement_count(candidate: dict[str, Any], baseline: dict[str, Any]) -> int:
    return sum(
        [
            candidate["brier"] < baseline["brier"],
            candidate["log_loss"] < baseline["log_loss"],
            candidate["auc"] is not None
            and baseline["auc"] is not None
            and candidate["auc"] > baseline["auc"],
            candidate["ece"] < baseline["ece"],
        ]
    )


def cluster_bootstrap_brier_delta(
    predictions: pd.DataFrame,
    horizon: int,
    candidate: str,
    baseline: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    keys = ["timestamp", "trade_date", "stockCode", "fold"]
    subset = predictions[
        (predictions["horizon_bars"] == horizon)
        & predictions["variant"].isin([candidate, baseline])
    ]
    wide = subset.pivot(
        index=keys,
        columns="variant",
        values=["actual", "probability"],
    )
    actual = wide[("actual", baseline)].to_numpy(dtype=float)
    candidate_loss = (
        wide[("probability", candidate)].to_numpy(dtype=float) - actual
    ) ** 2
    baseline_loss = (
        wide[("probability", baseline)].to_numpy(dtype=float) - actual
    ) ** 2
    frame = pd.DataFrame(
        {
            "trade_date": wide.index.get_level_values("trade_date"),
            "delta": candidate_loss - baseline_loss,
        }
    )
    daily = frame.groupby("trade_date")["delta"].mean().to_numpy()
    rng = np.random.default_rng(int(config["models"]["randomSeed"]))
    samples = np.empty(int(config["models"]["clusterBootstrapReplicates"]))
    for index in range(len(samples)):
        samples[index] = float(
            np.mean(rng.choice(daily, size=len(daily), replace=True))
        )
    confidence = float(config["models"]["clusterBootstrapConfidence"])
    tail = (1.0 - confidence) / 2.0
    return {
        "cluster": "trade_date",
        "independentTradingDays": int(len(daily)),
        "meanBrierDeltaCandidateMinusBaseline": float(np.mean(daily)),
        "confidence": confidence,
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1.0 - tail)),
        "improvementRequiresUpperBelowZero": True,
        "passes": bool(np.quantile(samples, 1.0 - tail) < 0.0),
    }


def evaluate_candidate(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    candidate: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline = config["successGate"]["comparisonBaseline"]
    required = int(config["successGate"]["minimumImprovedMetricCountOfFour"])
    result: dict[str, Any] = {}
    for horizon in config["data"]["primaryHorizonsBars"]:
        key = str(horizon)
        current = metrics["overall"][key][candidate]
        reference = metrics["overall"][key][baseline]
        folds = metrics["byFold"][key]
        improving_folds = sum(
            improvement_count(values[candidate], values[baseline]) >= required
            for values in folds.values()
        )
        months = metrics["byGroup"][key]["month"]
        improving_months = sum(
            values[candidate]["brier"] < values[baseline]["brier"]
            for values in months.values()
        )
        categories = metrics["byGroup"][key]["etf_category"]
        improving_categories = [
            name
            for name, values in categories.items()
            if values[candidate]["brier"] < values[baseline]["brier"]
        ]
        optimism = (
            None
            if current["highRiskHitRate"] is None
            else current["highRiskMeanProbability"] - current["highRiskHitRate"]
        )
        bootstrap = cluster_bootstrap_brier_delta(
            predictions, horizon, candidate, baseline, config
        )
        gates = {
            "majorityMetrics": improvement_count(current, reference) >= required,
            "majorityFolds": improving_folds / len(folds)
            > float(config["successGate"]["minimumImprovingFoldFraction"]),
            "majorityMonths": improving_months / len(months)
            > float(config["successGate"]["minimumImprovingMonthFraction"]),
            "multipleEtfCategories": len(improving_categories)
            >= int(config["successGate"]["minimumImprovingEtfCategories"]),
            "highRiskSamples": current["highRiskCount"]
            >= int(config["successGate"]["minimumHighRiskSamples"]),
            "highRiskNotOverconfident": optimism is not None
            and optimism <= float(config["successGate"]["maximumHighRiskOptimism"]),
            "clusterBootstrapBrier": bootstrap["passes"],
        }
        result[key] = {
            "improvedMetricCountOfFour": improvement_count(current, reference),
            "improvingFolds": int(improving_folds),
            "totalFolds": int(len(folds)),
            "improvingMonths": int(improving_months),
            "totalMonths": int(len(months)),
            "improvingEtfCategories": improving_categories,
            "highRiskCount": current["highRiskCount"],
            "highRiskOptimism": optimism,
            "clusterBootstrap": bootstrap,
            "gates": gates,
            "passes": all(gates.values()),
        }
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Singularity Phase 1.6 Viterbi ablation report",
        "",
        f"- Run ID: `{result['runId']}`",
        f"- Source commit at run: `{result['sourceCommitAtRun']}`",
        f"- Status: `{result['status']}`",
        f"- Research-only: `{result['researchOnly']}`",
        f"- Shadow-only: `{result['shadowOnly']}`",
        "",
        "Online Viterbi here means prefix endpoint decoding only. It does not "
        "use full-sequence backtracking, forward-backward smoothing, or future "
        "observations.",
        "",
        "## Viterbi feature audit",
        "",
    ]
    for key, value in result["viterbiFeatureAudit"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Calibrated purged walk-forward metrics",
            "",
            "| h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, variants in result["metrics"]["overall"].items():
        for variant, metric in variants.items():
            auc = "null" if metric["auc"] is None else f"{metric['auc']:.6f}"
            lines.append(
                f"| {horizon} | {variant} | {metric['count']:,} | "
                f"{metric['brier']:.6f} | {metric['log_loss']:.6f} | "
                f"{auc} | {metric['ece']:.6f} | {metric['highRiskCount']} |"
            )
    lines.extend(["", "## Candidate gates", ""])
    for horizon, values in result["candidateDiagnostics"].items():
        bootstrap = values["clusterBootstrap"]
        lines.append(
            f"- {horizon}-bar: metric wins {values['improvedMetricCountOfFour']}/4; "
            f"folds {values['improvingFolds']}/{values['totalFolds']}; "
            f"months {values['improvingMonths']}/{values['totalMonths']}; "
            f"ETF categories {len(values['improvingEtfCategories'])}; "
            f"high-risk n={values['highRiskCount']}; Brier delta "
            f"{bootstrap['meanBrierDeltaCandidateMinusBaseline']:.8f} "
            f"CI [{bootstrap['lower']:.8f}, {bootstrap['upper']:.8f}]; "
            f"pass=`{values['passes']}`."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Viterbi helps HMM+EWS under fixed gate: `{result['conclusions']['viterbiHelpsHMM']}`",
            f"- Production or forward-task use allowed: `{result['conclusions']['productionOrForwardUseAllowed']}`",
            f"- Recommended action: `{result['conclusions']['recommendedAction']}`",
            "",
            "This result is not authorization to change paper trading. Any trading "
            "use requires a separate preregistration, fresh forward sample, "
            "costed replay on the actual decision population, and explicit "
            "economic-materiality gates.",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    phase16_result = json.loads(
        resolve(config["source"]["phase16ModelResult"]).read_text(encoding="utf-8")
    )
    hmm_audit = phase16_result["hmmAudit"]
    phase2a_config, _, _, base, _, _, _ = phase16.load_context(
        {"source": {"phase2AConfig": config["source"]["phase2AConfig"]}}
    )
    table = pd.read_csv(resolve(config["source"]["phase16FeatureTable"]))
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    viterbi_frame, viterbi_audit = build_viterbi_frame(
        base["market_ret_1"], hmm_audit
    )
    viterbi_frame["timestamp"] = pd.to_datetime(viterbi_frame["timestamp"])
    table = table.merge(
        viterbi_frame.drop(columns=["trade_date"]),
        on="timestamp",
        how="left",
        validate="many_to_one",
    )
    table["viterbi_agrees_with_filtered_state"] = (
        table["viterbi_state"].to_numpy(dtype=float)
        == table["dominant_hmm_state"].to_numpy(dtype=float)
    ).astype(float)
    rows_before = len(table)
    required = sorted(
        feature
        for variant in config["variants"]
        for feature in variant_features(config)[variant]
    )
    complete_rows = int(table.dropna(subset=required).shape[0])
    predictions, folds = run_walk_forward(table, config)
    metrics = summarize_predictions(predictions, config)
    candidate = config["successGate"]["candidate"]
    diagnostics = evaluate_candidate(metrics, predictions, candidate, config)
    result = {
        "schemaVersion": "singularity_phase1_6_viterbi_ablation_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": phase2a.source_commit(),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "data": {
            "walkForwardStart": config["data"]["walkForwardStart"],
            "walkForwardEnd": config["data"]["walkForwardEnd"],
            "barIntervalMinutes": config["data"]["barIntervalMinutes"],
            "horizonsBars": config["data"]["primaryHorizonsBars"],
            "phase16FeatureRows": rows_before,
            "completeSamePopulationRows": complete_rows,
            "completeCoverage": complete_rows / rows_before if rows_before else None,
            "symbols": int(table["stockCode"].nunique()),
            "tradingDays": int(table["trade_date"].nunique()),
        },
        "hmmAuditSource": {
            "fitEnd": hmm_audit["fitEnd"],
            "states": hmm_audit["states"],
            "means": hmm_audit["means"],
            "transitionMatrix": hmm_audit["transitionMatrix"],
            "reusedExistingHMM": True,
            "newHmmFit": False,
        },
        "viterbiFeatureAudit": {
            **viterbi_audit,
            "filteredAgreementRate": float(
                table["viterbi_agrees_with_filtered_state"].mean()
            ),
            "phase2AConfigValidated": phase2a_config["schemaVersion"]
            == "singularity_phase2a_historical_v1",
        },
        "walkForward": {
            "testBlockDays": config["models"]["walkForwardTestBlockDays"],
            "calibrationDays": config["models"]["calibrationDays"],
            "purgeDecisionRowsPerSymbol": config["models"][
                "purgeDecisionRowsPerSymbol"
            ],
            "embargoTradingDays": config["models"]["embargoTradingDays"],
            "folds": folds,
        },
        "metrics": metrics,
        "candidateDiagnostics": diagnostics,
        "conclusions": {
            "viterbiHelpsHMM": any(values["passes"] for values in diagnostics.values()),
            "productionOrForwardUseAllowed": False,
            "recommendedAction": "retain_as_research_shadow_only_if_increment_is_stable_else_drop",
            "reason": (
                "Viterbi is only a deterministic decoding of the existing HMM. "
                "It may stabilize state interpretation but cannot add new market information."
            ),
        },
        "lookaheadAudit": {
            "featuresPastOnly": True,
            "onlineEndpointDecodeOnly": True,
            "fullSequenceBacktrackingUsed": False,
            "forwardBackwardSmoothingUsed": False,
            "futureObservationsUsed": False,
            "prefixInvarianceCheckPassed": viterbi_audit[
                "prefixInvarianceCheckPassed"
            ],
            "labelsSeparateAndFutureUsing": True,
            "winsorizationFitPerFold": True,
            "plattCalibrationPrecedesTest": True,
            "sameTradingDateAcrossBoundaries": False,
            "phase15LedgerRead": False,
        },
        "skipped": [
            "full_sequence_viterbi_backtracking",
            "forward_backward_smoothing",
            "hmm_state_refit",
            "post_jump_hmm",
            "time_decay_hmm",
            "online_inference",
            "live_trading_integration",
        ],
        "phase15Touched": False,
        "paperIntegrationAllowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    phase2a.atomic_json(output_dir / "viterbi_ablation_result.json", result)
    phase2a.atomic_text(
        output_dir / "viterbi_ablation_report.md",
        render_report(result),
    )
    predictions.to_csv(
        output_dir / "walk_forward_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    run_id = args.run_id or (
        "viterbi_ablation_"
        + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output_dir = resolve(config["output"]["root"]) / run_id
    result = run(config, output_dir)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "status": result["status"],
                "viterbi_helps_hmm": result["conclusions"]["viterbiHelpsHMM"],
                "paper_integration_allowed": result["paperIntegrationAllowed"],
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
