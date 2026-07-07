"""Phase 1.7 research-only risk-stack audit.

This script tests a causal risk-context stack:

    BOCPD -> DMD/Koopman diagnostics -> HMM regime -> Hawkes intensity.

The output is explicitly diagnostic.  It cannot place orders, change paper
trading, change exits, modify position sizing, write overlays, or promote a
strategy.  Labels are used only for offline evaluation.
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
from sklearn.metrics import roc_auc_score

from research_hmm_nn_bl import ROOT
import research_singularity_phase1_6_hmm_physics_features as phase16
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = ROOT / "configs" / "research" / "risk_stack_phase1_7_audit_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "risk_stack_phase1_7_audit_v1":
        raise ValueError("unexpected risk-stack Phase 1.7 schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("risk-stack audit must remain research/shadow-only")
    if config["source"].get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 forward ledger cannot be an input")
    if config["source"].get("paperTradingArtifactsAllowedAsInput") is not False:
        raise ValueError("paper trading artifacts cannot be an input")
    if config["bocpd"].get("parameterSearchAllowed") is not False:
        raise ValueError("BOCPD parameter search is forbidden")
    if config["hawkes"].get("parameterSearchAllowed") is not False:
        raise ValueError("Hawkes parameter search is forbidden")
    if config["hawkes"].get("usesCurrentEventInCurrentIntensity") is not False:
        raise ValueError("Hawkes intensity must exclude the current event")
    if config["riskStack"].get("noParameterSearch") is not True:
        raise ValueError("risk-stack score must be fixed")
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


def normal_logpdf(value: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    safe = np.maximum(variance, 1e-12)
    return -0.5 * (np.log(2.0 * math.pi * safe) + ((value - mean) ** 2) / safe)


def bocpd_gaussian_known_variance(
    values: np.ndarray,
    *,
    hazard: float,
    max_run_length: int,
    alert_run_length: int,
    prior_mean: float,
    prior_variance: float,
    observation_variance: float,
) -> pd.DataFrame:
    """Causal Gaussian BOCPD with fixed hazard and known observation variance.

    Each output row t depends only on observations <= t.  Parameters are fixed
    before evaluation; this is a diagnostic hazard, not a tuned model.
    """

    observations = np.asarray(values, dtype=float)
    log_run = np.asarray([0.0])
    means = np.asarray([prior_mean], dtype=float)
    variances = np.asarray([prior_variance], dtype=float)
    change_probabilities: list[float] = []
    run0_probabilities: list[float] = []
    expected_runs: list[float] = []
    for observation in observations:
        predictive_variance = observation_variance + variances
        log_pred = normal_logpdf(observation, means, predictive_variance)
        growth = log_run + math.log(max(1.0 - hazard, 1e-12)) + log_pred
        change = np.asarray([
            phase2a.logsumexp(log_run + math.log(max(hazard, 1e-12)) + log_pred)
            if hasattr(phase2a, "logsumexp")
            else np.logaddexp.reduce(log_run + math.log(max(hazard, 1e-12)) + log_pred)
        ])
        new_log = np.concatenate([change, growth])
        if len(new_log) > max_run_length + 1:
            new_log = new_log[: max_run_length + 1]
            means = means[:max_run_length]
            variances = variances[:max_run_length]
        new_log -= np.logaddexp.reduce(new_log)

        posterior_variance_from_prior = 1.0 / (
            1.0 / prior_variance + 1.0 / observation_variance
        )
        posterior_mean_from_prior = posterior_variance_from_prior * (
            prior_mean / prior_variance + observation / observation_variance
        )
        grown_variance = 1.0 / (1.0 / variances + 1.0 / observation_variance)
        grown_mean = grown_variance * (
            means / variances + observation / observation_variance
        )
        means = np.concatenate([[posterior_mean_from_prior], grown_mean])
        variances = np.concatenate([[posterior_variance_from_prior], grown_variance])
        if len(means) > max_run_length + 1:
            means = means[: max_run_length + 1]
            variances = variances[: max_run_length + 1]
        log_run = new_log
        probabilities = np.exp(log_run)
        run_lengths = np.arange(len(probabilities), dtype=float)
        alert_end = min(int(alert_run_length), len(probabilities) - 1)
        change_probabilities.append(float(np.sum(probabilities[: alert_end + 1])))
        run0_probabilities.append(float(probabilities[0]))
        expected_runs.append(float(np.sum(probabilities * run_lengths)))
    return pd.DataFrame(
        {
            "bocpd_change_probability": change_probabilities,
            "bocpd_run0_probability": run0_probabilities,
            "bocpd_expected_run_length": expected_runs,
        }
    )


def prefix_invariance_check(config: dict[str, Any]) -> bool:
    rng = np.random.default_rng(20260708)
    prefix = rng.normal(0.0, 0.001, size=40)
    suffix = np.asarray([0.05, -0.05, 0.04])
    bocpd_cfg = config["bocpd"]
    first = bocpd_gaussian_known_variance(
        prefix,
        hazard=float(bocpd_cfg["hazard"]),
        max_run_length=int(bocpd_cfg["maxRunLengthBars"]),
        alert_run_length=int(bocpd_cfg["alertRunLengthBars"]),
        prior_mean=0.0,
        prior_variance=25.0 * float(np.var(prefix) + 1e-8),
        observation_variance=float(np.var(prefix) + 1e-8),
    )
    second = bocpd_gaussian_known_variance(
        np.concatenate([prefix, suffix]),
        hazard=float(bocpd_cfg["hazard"]),
        max_run_length=int(bocpd_cfg["maxRunLengthBars"]),
        alert_run_length=int(bocpd_cfg["alertRunLengthBars"]),
        prior_mean=0.0,
        prior_variance=25.0 * float(np.var(prefix) + 1e-8),
        observation_variance=float(np.var(prefix) + 1e-8),
    ).iloc[: len(prefix)]
    return bool(np.allclose(first.to_numpy(dtype=float), second.to_numpy(dtype=float)))


def build_bocpd_frame(
    market_return: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bocpd_cfg = config["bocpd"]
    training_end = str(config["data"]["trainingStatisticsEnd"])
    training = market_return[market_return.index.strftime("%Y-%m-%d") <= training_end]
    observation_variance = max(float(np.var(training.to_numpy(dtype=float))), 1e-10)
    prior_variance = float(bocpd_cfg["priorVarianceScale"]) * observation_variance
    rows: list[pd.DataFrame] = []
    ordered = market_return.sort_index()
    dates = pd.Series(ordered.index.strftime("%Y-%m-%d"), index=ordered.index)
    for trade_date, values in ordered.groupby(dates, sort=True):
        frame = bocpd_gaussian_known_variance(
            values.to_numpy(dtype=float),
            hazard=float(bocpd_cfg["hazard"]),
            max_run_length=int(bocpd_cfg["maxRunLengthBars"]),
            alert_run_length=int(bocpd_cfg["alertRunLengthBars"]),
            prior_mean=float(bocpd_cfg["priorMean"]),
            prior_variance=prior_variance,
            observation_variance=observation_variance,
        )
        frame["timestamp"] = values.index
        frame["trade_date"] = trade_date
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    audit = {
        "rows": int(len(result)),
        "firstTimestamp": str(result["timestamp"].min()),
        "lastTimestamp": str(result["timestamp"].max()),
        "sessionReset": True,
        "hazard": float(bocpd_cfg["hazard"]),
        "maxRunLengthBars": int(bocpd_cfg["maxRunLengthBars"]),
        "alertRunLengthBars": int(bocpd_cfg["alertRunLengthBars"]),
        "observationVarianceFitEnd": training_end,
        "observationVariance": observation_variance,
        "prefixInvarianceCheckPassed": prefix_invariance_check(config),
        "meanChangeProbability": float(result["bocpd_change_probability"].mean()),
        "p90ChangeProbability": float(result["bocpd_change_probability"].quantile(0.9)),
    }
    return result, audit


def fit_thresholds(table: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    train = table[table["trade_date"] <= config["data"]["trainingStatisticsEnd"]]
    event_cfg = config["hawkes"]["eventDefinitions"]
    thresholds = {
        "marketDropReturn": float(
            train["market_ret_1"].quantile(float(event_cfg["marketDropQuantile"]))
        ),
        "bocpdHighHazard": float(
            train["bocpd_change_probability"].quantile(
                float(event_cfg["bocpdHazardQuantile"])
            )
        ),
        "dmdHighResidual": float(
            train.loc[train["dmd_window_valid"] == 1.0, "dmd_residual_zscore"].quantile(
                float(event_cfg["dmdResidualQuantile"])
            )
        ),
        "hmmHighTransitionRisk": float(
            train["regime_transition_risk"].quantile(
                float(event_cfg["hmmTransitionRiskQuantile"])
            )
        ),
    }
    return thresholds


def add_observable_events(table: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = table.copy()
    out["event_market_drop"] = (
        out["market_ret_1"].to_numpy(dtype=float) <= thresholds["marketDropReturn"]
    ).astype(int)
    out["event_bocpd_hazard"] = (
        out["bocpd_change_probability"].to_numpy(dtype=float)
        >= thresholds["bocpdHighHazard"]
    ).astype(int)
    out["event_dmd_instability"] = (
        (out["dmd_window_valid"].to_numpy(dtype=float) == 1.0)
        & (
            out["dmd_residual_zscore"].fillna(-np.inf).to_numpy(dtype=float)
            >= thresholds["dmdHighResidual"]
        )
    ).astype(int)
    out["event_hmm_transition"] = (
        out["regime_transition_risk"].to_numpy(dtype=float)
        >= thresholds["hmmHighTransitionRisk"]
    ).astype(int)
    event_cols = [
        "event_market_drop",
        "event_bocpd_hazard",
        "event_dmd_instability",
        "event_hmm_transition",
    ]
    out["observable_risk_event"] = (out[event_cols].sum(axis=1) > 0).astype(int)
    return out


def add_discrete_hawkes_intensity(
    table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = table.sort_values(["stockCode", "timestamp"], kind="stable").copy()
    train = out[out["trade_date"] <= config["data"]["trainingStatisticsEnd"]]
    base_rate = max(float(train["observable_risk_event"].mean()), 1e-6)
    decay = float(config["hawkes"]["decayBars"])
    excitation = float(config["hawkes"]["excitation"])
    decay_factor = math.exp(-1.0 / max(decay, 1e-9))
    intensities = pd.Series(index=out.index, dtype=float)
    carry_values = pd.Series(index=out.index, dtype=float)
    for _, group in out.groupby("stockCode", sort=False):
        carry = 0.0
        previous_date: str | None = None
        for index, row in group.iterrows():
            trade_date = str(row["trade_date"])
            if previous_date is not None and trade_date != previous_date:
                carry = 0.0
            carry *= decay_factor
            intensities.at[index] = base_rate + excitation * carry
            carry_values.at[index] = carry
            if int(row["observable_risk_event"]) == 1:
                carry += 1.0
            previous_date = trade_date
    out["hawkes_event_intensity"] = intensities
    out["hawkes_excitation_memory"] = carry_values
    audit = {
        "implementation": config["hawkes"]["implementation"],
        "baselineEventRate": base_rate,
        "decayBars": decay,
        "excitation": excitation,
        "usesCurrentEventInCurrentIntensity": False,
        "trainingEvents": int(train["observable_risk_event"].sum()),
        "trainingRows": int(len(train)),
        "overallEvents": int(out["observable_risk_event"].sum()),
        "overallRows": int(len(out)),
        "eventDensity": float(out["observable_risk_event"].mean()),
    }
    return out.sort_index(), audit


def quantile_score(values: pd.Series, training: pd.Series) -> pd.Series:
    clean = training.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or float(clean.quantile(0.95)) == float(clean.quantile(0.05)):
        return pd.Series(np.nan, index=values.index)
    low = float(clean.quantile(0.05))
    high = float(clean.quantile(0.95))
    return ((values.astype(float) - low) / (high - low)).clip(0.0, 1.0)


def add_risk_scores(table: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = table.copy()
    train = out[out["trade_date"] <= config["data"]["trainingStatisticsEnd"]]
    score_specs = {
        "score_bocpd": "bocpd_change_probability",
        "score_dmd": "dmd_residual_zscore",
        "score_hmm": "regime_transition_risk",
        "score_hawkes": "hawkes_event_intensity",
    }
    audit: dict[str, Any] = {}
    for score_name, source_name in score_specs.items():
        out[score_name] = quantile_score(out[source_name], train[source_name])
        audit[score_name] = {
            "source": source_name,
            "trainingNonNull": int(train[source_name].notna().sum()),
            "trainingP05": (
                float(train[source_name].quantile(0.05))
                if train[source_name].notna().any()
                else None
            ),
            "trainingP95": (
                float(train[source_name].quantile(0.95))
                if train[source_name].notna().any()
                else None
            ),
        }
    out["risk_stack_full_score"] = out[["score_bocpd", "score_hmm", "score_hawkes"]].mean(axis=1)
    out["risk_stack_dmd_score"] = out[
        ["score_bocpd", "score_dmd", "score_hmm", "score_hawkes"]
    ].mean(axis=1)
    audit["combinedScores"] = {
        "fullSession": ["score_bocpd", "score_hmm", "score_hawkes"],
        "dmdComplete": ["score_bocpd", "score_dmd", "score_hmm", "score_hawkes"],
        "weights": "equal_weight_no_parameter_search",
    }
    return out, audit


def safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if frame.empty or frame["label"].nunique() != 2:
        return None
    return float(roc_auc_score(frame["label"].astype(int), frame["score"].astype(float)))


def evaluate_score(
    rows: pd.DataFrame,
    *,
    label: str,
    score: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    data = rows.dropna(subset=[label, score]).copy()
    data[label] = data[label].astype(int)
    if data.empty:
        return {"count": 0, "status": "no_rows"}
    top_fraction = float(config["riskStack"]["topBucketFraction"])
    cutoff = float(data[score].quantile(1.0 - top_fraction))
    top = data[data[score] >= cutoff]
    baseline = float(data[label].mean())
    top_hit = float(top[label].mean()) if len(top) else None
    month_stats = []
    for month, group in data.groupby("month"):
        group_top = group[group[score] >= cutoff]
        if group[label].nunique() == 2 and not group_top.empty:
            month_stats.append(
                {
                    "month": str(month),
                    "base": float(group[label].mean()),
                    "top": float(group_top[label].mean()),
                    "improves": bool(group_top[label].mean() > group[label].mean()),
                }
            )
    category_stats = []
    for category, group in data.groupby("etf_category"):
        group_top = group[group[score] >= cutoff]
        if group[label].nunique() == 2 and not group_top.empty:
            category_stats.append(
                {
                    "category": str(category),
                    "base": float(group[label].mean()),
                    "top": float(group_top[label].mean()),
                    "improves": bool(group_top[label].mean() > group[label].mean()),
                }
            )
    return {
        "count": int(len(data)),
        "positiveRate": baseline,
        "auc": safe_auc(data[label], data[score]),
        "topBucketFraction": top_fraction,
        "topBucketCutoff": cutoff,
        "topBucketCount": int(len(top)),
        "topBucketHitRate": top_hit,
        "topBucketLift": (top_hit / baseline if top_hit is not None and baseline else None),
        "selectionFractionIfUsedAsHardGate": float(len(top) / len(data)),
        "mechanicalTradeReductionIfUsedAsHardGate": float(1.0 - len(top) / len(data)),
        "improvingMonths": int(sum(item["improves"] for item in month_stats)),
        "totalEvaluableMonths": int(len(month_stats)),
        "improvingCategories": int(sum(item["improves"] for item in category_stats)),
        "totalEvaluableCategories": int(len(category_stats)),
        "monthDiagnostics": month_stats,
        "categoryDiagnostics": category_stats,
    }


def evaluate_all(table: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    start = config["data"]["walkForwardStart"]
    end = config["data"]["walkForwardEnd"]
    oos = table[(table["trade_date"] >= start) & (table["trade_date"] <= end)].copy()
    score_specs = {
        "bocpd_only": ("risk_stack_full", "score_bocpd"),
        "dmd_only": ("dmd_complete", "score_dmd"),
        "hmm_transition_only": ("risk_stack_full", "score_hmm"),
        "hawkes_only": ("risk_stack_full", "score_hawkes"),
        "risk_stack_full": ("risk_stack_full", "risk_stack_full_score"),
        "risk_stack_dmd_complete": ("dmd_complete", "risk_stack_dmd_score"),
    }
    result: dict[str, Any] = {}
    for label in config["evaluation"]["labels"]:
        label_result: dict[str, Any] = {}
        for name, (population, score) in score_specs.items():
            rows = oos
            if population == "dmd_complete":
                rows = rows[rows["dmd_window_valid"] == 1.0]
            label_result[name] = evaluate_score(
                rows,
                label=label,
                score=score,
                config=config,
            )
        result[label] = label_result
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Risk Stack Phase 1.7 audit",
        "",
        f"- Run ID: `{result['runId']}`",
        f"- Source commit at run: `{result['sourceCommitAtRun']}`",
        f"- Status: `{result['status']}`",
        f"- Research-only: `{result['researchOnly']}`",
        f"- Paper integration allowed: `{result['paperIntegrationAllowed']}`",
        "",
        "This audit tests BOCPD, DMD, HMM and Hawkes as risk context for exits "
        "and position de-risking. It is not a BUY/SELL model.",
        "",
        "## Data audit",
        "",
    ]
    for key, value in result["dataAudit"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Feature audits", ""])
    lines.append(f"- BOCPD prefix-invariant: `{result['bocpdAudit']['prefixInvarianceCheckPassed']}`")
    lines.append(f"- Hawkes events: `{result['hawkesAudit']['overallEvents']}` / `{result['hawkesAudit']['overallRows']}` rows")
    lines.append(f"- Hawkes event density: `{result['hawkesAudit']['eventDensity']:.4f}`")
    lines.extend(
        [
            "",
            "## OOS bucket diagnostics",
            "",
            "| label | score | n | AUC | base hit | top hit | lift | top n | months + | cats + |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, scores in result["evaluation"].items():
        for score_name, metrics in scores.items():
            auc = "null" if metrics.get("auc") is None else f"{metrics['auc']:.4f}"
            top_hit = (
                "null"
                if metrics.get("topBucketHitRate") is None
                else f"{metrics['topBucketHitRate']:.4f}"
            )
            lift = (
                "null"
                if metrics.get("topBucketLift") is None
                else f"{metrics['topBucketLift']:.3f}"
            )
            lines.append(
                f"| {label} | {score_name} | {metrics.get('count', 0):,} | "
                f"{auc} | {metrics.get('positiveRate', 0):.4f} | {top_hit} | "
                f"{lift} | {metrics.get('topBucketCount', 0):,} | "
                f"{metrics.get('improvingMonths', 0)}/{metrics.get('totalEvaluableMonths', 0)} | "
                f"{metrics.get('improvingCategories', 0)}/{metrics.get('totalEvaluableCategories', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- BOCPD feasible: `{result['conclusions']['bocpdFeasible']}`",
            f"- Hawkes event density sufficient for fitting: `{result['conclusions']['hawkesEventDensitySufficient']}`",
            f"- Full risk stack shows stable edge: `{result['conclusions']['fullRiskStackShowsStableEdge']}`",
            f"- DMD-complete stack shows stable edge: `{result['conclusions']['dmdCompleteStackShowsStableEdge']}`",
            f"- Costed replay justified now: `{result['conclusions']['costedReplayJustifiedNow']}`",
            f"- Recommended next step: `{result['conclusions']['recommendedNextStep']}`",
            "",
            "The result cannot be used in paper trading without a separate costed "
            "replay and fresh forward validation.",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    _, _, _, base, _, _, _ = phase16.load_context(
        {"source": {"phase2AConfig": config["source"]["phase2AConfig"]}}
    )
    phase16_result = json.loads(
        resolve(config["source"]["phase16ModelResult"]).read_text(encoding="utf-8")
    )
    table = pd.read_csv(resolve(config["source"]["phase16FeatureTable"]))
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    table["stockCode"] = table["stockCode"].astype(str).str.zfill(6)
    bocpd, bocpd_audit = build_bocpd_frame(base["market_ret_1"], config)
    bocpd["timestamp"] = pd.to_datetime(bocpd["timestamp"])
    table = table.merge(
        bocpd.drop(columns=["trade_date"]),
        on="timestamp",
        how="left",
        validate="many_to_one",
    )
    thresholds = fit_thresholds(table, config)
    table = add_observable_events(table, thresholds)
    table, hawkes_audit = add_discrete_hawkes_intensity(table, config)
    table, score_audit = add_risk_scores(table, config)
    evaluation = evaluate_all(table, config)
    full_5 = evaluation["turning_point_5"]["risk_stack_full"]
    full_10 = evaluation["turning_point_10"]["risk_stack_full"]
    dmd_5 = evaluation["turning_point_5"]["risk_stack_dmd_complete"]
    dmd_10 = evaluation["turning_point_10"]["risk_stack_dmd_complete"]
    hawkes_density = float(hawkes_audit["eventDensity"])
    stable_full = bool(
        full_5["auc"] is not None
        and full_10["auc"] is not None
        and full_5["auc"] > 0.55
        and full_10["auc"] > 0.55
        and full_5["improvingMonths"] > full_5["totalEvaluableMonths"] / 2
        and full_10["improvingMonths"] > full_10["totalEvaluableMonths"] / 2
    )
    stable_dmd = bool(
        dmd_5["auc"] is not None
        and dmd_10["auc"] is not None
        and dmd_5["auc"] > 0.55
        and dmd_10["auc"] > 0.55
        and dmd_5["improvingMonths"] > dmd_5["totalEvaluableMonths"] / 2
        and dmd_10["improvingMonths"] > dmd_10["totalEvaluableMonths"] / 2
    )
    result = {
        "schemaVersion": "risk_stack_phase1_7_audit_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": phase2a.source_commit(),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "paperIntegrationAllowed": False,
        "dataAudit": {
            "phase16FeatureRows": int(len(table)),
            "symbols": int(table["stockCode"].nunique()),
            "tradingDays": int(table["trade_date"].nunique()),
            "walkForwardStart": config["data"]["walkForwardStart"],
            "walkForwardEnd": config["data"]["walkForwardEnd"],
            "trainingStatisticsEnd": config["data"]["trainingStatisticsEnd"],
            "dmdCompleteRows": int((table["dmd_window_valid"] == 1.0).sum()),
            "dmdCompleteCoverage": float((table["dmd_window_valid"] == 1.0).mean()),
            "hmmSourceFitEnd": phase16_result["hmmAudit"]["fitEnd"],
        },
        "literatureMap": config["literatureMap"],
        "thresholds": thresholds,
        "bocpdAudit": bocpd_audit,
        "hawkesAudit": hawkes_audit,
        "scoreAudit": score_audit,
        "evaluation": evaluation,
        "lookaheadAudit": {
            "featuresPastOnly": True,
            "bocpdSessionReset": True,
            "bocpdPrefixInvariant": bocpd_audit["prefixInvarianceCheckPassed"],
            "hawkesCurrentEventExcludedFromCurrentIntensity": True,
            "thresholdsFitThroughTrainingEndOnly": True,
            "labelsSeparateAndFutureUsing": True,
            "phase15LedgerRead": False,
            "paperTradingArtifactsRead": False,
        },
        "mechanicalReductionAudit": {
            "topBucketFraction": config["riskStack"]["topBucketFraction"],
            "hardGateWouldDropAbout": 1.0 - float(config["riskStack"]["topBucketFraction"]),
            "interpretation": (
                "Top-bucket diagnostics identify risky subsets. They do not prove "
                "trading improvement unless costed replay shows drawdown reduction "
                "without excessive opportunity loss."
            ),
        },
        "conclusions": {
            "bocpdFeasible": bocpd_audit["prefixInvarianceCheckPassed"],
            "hawkesEventDensitySufficient": hawkes_density >= 0.03,
            "fullRiskStackShowsStableEdge": stable_full,
            "dmdCompleteStackShowsStableEdge": stable_dmd,
            "costedReplayJustifiedNow": bool(stable_full or stable_dmd),
            "productionOrForwardUseAllowed": False,
            "recommendedNextStep": (
                "costed_exit_position_replay_shadow_only"
                if stable_full or stable_dmd
                else "retain_audit_only_collect_more_forward_events"
            ),
        },
        "skipped": [
            "continuous_time_hawkes_mle",
            "neural_hawkes",
            "online_inference",
            "paper_trading_integration",
            "position_sizing_integration",
            "exit_gate_integration",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    phase2a.atomic_json(output_dir / "risk_stack_phase1_7_result.json", result)
    phase2a.atomic_text(output_dir / "risk_stack_phase1_7_report.md", render_report(result))
    compact_columns = [
        "timestamp",
        "trade_date",
        "stockCode",
        "bocpd_change_probability",
        "regime_transition_risk",
        "dmd_residual_zscore",
        "hawkes_event_intensity",
        "risk_stack_full_score",
        "risk_stack_dmd_score",
        "turning_point_5",
        "turning_point_10",
    ]
    table[compact_columns].to_csv(
        output_dir / "risk_stack_features_compact.csv",
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
        "risk_stack_p17_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output_dir = resolve(config["output"]["root"]) / run_id
    result = run(config, output_dir)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "status": result["status"],
                "full_stack_edge": result["conclusions"]["fullRiskStackShowsStableEdge"],
                "dmd_stack_edge": result["conclusions"]["dmdCompleteStackShowsStableEdge"],
                "costed_replay_justified": result["conclusions"]["costedReplayJustifiedNow"],
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
