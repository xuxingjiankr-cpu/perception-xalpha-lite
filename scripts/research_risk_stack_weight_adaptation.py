"""Phase 1.8 research-only risk-stack weight adaptation.

This script tests whether fixed, pre-registered candidate weights can improve
the Phase 1.7 BOCPD/DMD/HMM/Hawkes risk-context stack under purged
walk-forward evaluation.  It is diagnostic only: no orders, no live config
writes, no overlays, no position sizing, and no BUY/SELL gate integration.

The core guardrail is that every monthly test fold selects weights only from
past training rows, with a purge of at least the maximum label horizon.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_hmm_nn_bl import ROOT
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = ROOT / "configs" / "research" / "risk_stack_weight_adaptation_v1.json"

COMPONENT_TO_RAW = {
    "bocpd": "bocpd_change_probability",
    "dmd": "dmd_residual_zscore",
    "hmm": "regime_transition_risk",
    "hawkes": "hawkes_event_intensity",
}
COMPONENT_TO_SCORE = {key: f"score_{key}" for key in COMPONENT_TO_RAW}


@dataclass(frozen=True)
class Candidate:
    name: str
    weights: dict[str, float]
    order: int

    @property
    def component_count(self) -> int:
        return sum(1 for value in self.weights.values() if abs(value) > 1e-12)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_candidate(candidate: dict[str, Any], *, allow_dmd: bool) -> None:
    if not candidate.get("name"):
        raise ValueError("candidate name is required")
    weights = candidate.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"candidate {candidate.get('name')} has no weights")
    total = 0.0
    for component, weight in weights.items():
        if component not in COMPONENT_TO_RAW:
            raise ValueError(f"unknown component {component}")
        if component == "dmd" and not allow_dmd:
            raise ValueError("full-session candidates cannot use dmd")
        value = float(weight)
        if value < -1e-12:
            raise ValueError("negative weights are not allowed")
        total += value
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"candidate {candidate.get('name')} weights must sum to 1")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "risk_stack_weight_adaptation_v1":
        raise ValueError("unexpected risk-stack weight-adaptation schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("weight adaptation must remain research/shadow-only")
    source = config["source"]
    if source.get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 forward ledger cannot be an input")
    if source.get("paperTradingArtifactsAllowedAsInput") is not False:
        raise ValueError("paper trading artifacts cannot be an input")
    selection = config["weightSelection"]
    if selection.get("parameterSearchOutsideCandidateGridAllowed") is not False:
        raise ValueError("candidate grid is the only allowed search space")
    if selection.get("negativeWeightsAllowed") is not False:
        raise ValueError("negative weights are forbidden")
    if selection.get("weightsMustSumToOne") is not True:
        raise ValueError("weights must sum to one")
    for item in selection["fullSessionCandidates"]:
        validate_candidate(item, allow_dmd=False)
    for item in selection["dmdCompleteCandidates"]:
        validate_candidate(item, allow_dmd=True)
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


def load_candidates(items: list[dict[str, Any]]) -> list[Candidate]:
    return [
        Candidate(
            name=str(item["name"]),
            weights={key: float(value) for key, value in item["weights"].items()},
            order=index,
        )
        for index, item in enumerate(items)
    ]


def load_table(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase17_result_path = resolve(config["source"]["phase17Result"])
    phase17_result = json.loads(phase17_result_path.read_text(encoding="utf-8"))
    if phase17_result.get("paperIntegrationAllowed") is not False:
        raise ValueError("Phase 1.7 source must be paper-integration disabled")
    table = pd.read_csv(resolve(config["source"]["phase17FeatureTable"]))
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    table["trade_date"] = table["trade_date"].astype(str)
    table["month"] = table["trade_date"].str.slice(0, 7)
    table["stockCode"] = table["stockCode"].astype(str).str.zfill(6)
    table["dmd_window_valid"] = table["dmd_residual_zscore"].notna().astype(int)
    for raw in COMPONENT_TO_RAW.values():
        table[raw] = pd.to_numeric(table[raw], errors="coerce")
    for label in config["evaluation"]["labels"]:
        table[label] = pd.to_numeric(table[label], errors="coerce")
    table = table.sort_values(["timestamp", "stockCode"], kind="stable").reset_index(
        drop=True
    )
    return table, phase17_result


def fit_quantile_normalizer(
    train: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    norm = config["normalization"]
    low_q = float(norm["lowerQuantile"])
    high_q = float(norm["upperQuantile"])
    fitted: dict[str, dict[str, float | None]] = {}
    for component, raw in COMPONENT_TO_RAW.items():
        clean = train[raw].replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            fitted[component] = {"low": None, "high": None}
            continue
        low = float(clean.quantile(low_q))
        high = float(clean.quantile(high_q))
        if not math.isfinite(low) or not math.isfinite(high) or abs(high - low) < 1e-12:
            fitted[component] = {"low": None, "high": None}
            continue
        fitted[component] = {"low": low, "high": high}
    return fitted


def apply_quantile_normalizer(
    frame: pd.DataFrame,
    fitted: dict[str, dict[str, float | None]],
) -> pd.DataFrame:
    out = frame.copy()
    for component, raw in COMPONENT_TO_RAW.items():
        score_col = COMPONENT_TO_SCORE[component]
        low = fitted[component]["low"]
        high = fitted[component]["high"]
        if low is None or high is None:
            out[score_col] = np.nan
        else:
            out[score_col] = ((out[raw].astype(float) - low) / (high - low)).clip(
                0.0, 1.0
            )
    return out


def add_candidate_score(frame: pd.DataFrame, candidate: Candidate, column: str) -> pd.DataFrame:
    out = frame.copy()
    score = pd.Series(0.0, index=out.index, dtype=float)
    valid = pd.Series(True, index=out.index)
    for component, weight in candidate.weights.items():
        component_score = out[COMPONENT_TO_SCORE[component]].astype(float)
        score = score + weight * component_score.fillna(0.0)
        valid = valid & component_score.notna()
    out[column] = score.where(valid, np.nan)
    return out


def safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if frame.empty or frame["label"].nunique() != 2:
        return None
    return float(roc_auc_score(frame["label"].astype(int), frame["score"].astype(float)))


def objective_for_candidate(
    train: pd.DataFrame,
    candidate: Candidate,
    labels: list[str],
    config: dict[str, Any],
) -> float | None:
    scored = add_candidate_score(train, candidate, "_candidate_score")
    values: list[float] = []
    min_rows = int(config["weightSelection"]["minTrainRows"])
    min_pos = int(config["weightSelection"]["minTrainPositiveRowsPerHorizon"])
    for label in labels:
        frame = scored[[label, "_candidate_score"]].dropna()
        if len(frame) < min_rows:
            continue
        positives = int(frame[label].sum())
        negatives = int(len(frame) - positives)
        if positives < min_pos or negatives < min_pos:
            continue
        auc = safe_auc(frame[label], frame["_candidate_score"])
        if auc is not None:
            values.append(auc)
    if not values:
        return None
    return float(np.mean(values))


def select_candidate(
    train: pd.DataFrame,
    candidates: list[Candidate],
    labels: list[str],
    config: dict[str, Any],
) -> tuple[Candidate, dict[str, float | None]]:
    objectives: dict[str, float | None] = {}
    ranked: list[tuple[float, int, int, Candidate]] = []
    for candidate in candidates:
        objective = objective_for_candidate(train, candidate, labels, config)
        objectives[candidate.name] = objective
        if objective is None:
            continue
        ranked.append(
            (
                objective,
                -candidate.component_count,
                -candidate.order,
                candidate,
            )
        )
    if not ranked:
        return candidates[0], objectives
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return ranked[0][3], objectives


def fit_calibrator(
    train: pd.DataFrame,
    *,
    label: str,
    score_col: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config["calibration"]
    frame = train[[label, score_col]].dropna().copy()
    if frame.empty:
        return {"fallback": 0.5, "edges": [], "rates": [], "counts": []}
    base_rate = float(frame[label].mean())
    bins = int(cfg["bins"])
    min_bin_rows = int(cfg["minBinRows"])
    edges = np.unique(
        np.nanquantile(frame[score_col].to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    )
    if len(edges) < 2:
        return {"fallback": base_rate, "edges": [], "rates": [], "counts": []}
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_ids = np.searchsorted(edges[1:], frame[score_col].to_numpy(dtype=float), side="right")
    rates: list[float] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        mask = bin_ids == index
        count = int(mask.sum())
        counts.append(count)
        if count >= min_bin_rows:
            rates.append(float(frame.loc[mask, label].mean()))
        else:
            rates.append(base_rate)
    return {
        "fallback": base_rate,
        "edges": [float(value) for value in edges],
        "rates": rates,
        "counts": counts,
    }


def apply_calibrator(scores: pd.Series, calibrator: dict[str, Any], config: dict[str, Any]) -> pd.Series:
    clip_low, clip_high = config["calibration"]["probabilityClip"]
    if not calibrator["edges"]:
        return pd.Series(float(calibrator["fallback"]), index=scores.index).clip(
            float(clip_low), float(clip_high)
        )
    edges = np.asarray(calibrator["edges"], dtype=float)
    rates = np.asarray(calibrator["rates"], dtype=float)
    score_values = scores.to_numpy(dtype=float)
    bin_ids = np.searchsorted(edges[1:], score_values, side="right")
    probabilities = np.full(len(scores), float(calibrator["fallback"]), dtype=float)
    valid = ~np.isnan(score_values)
    probabilities[valid] = rates[np.clip(bin_ids[valid], 0, len(rates) - 1)]
    return pd.Series(probabilities, index=scores.index).clip(float(clip_low), float(clip_high))


def ece_score(labels: pd.Series, probabilities: pd.Series, bins: int) -> float | None:
    frame = pd.DataFrame({"label": labels, "prob": probabilities}).dropna()
    if frame.empty:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.searchsorted(edges[1:], frame["prob"].to_numpy(dtype=float), side="right")
    total = len(frame)
    ece = 0.0
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        bucket_frame = frame.loc[mask]
        ece += len(bucket_frame) / total * abs(
            float(bucket_frame["prob"].mean()) - float(bucket_frame["label"].mean())
        )
    return float(ece)


def evaluate_prediction_frame(
    prediction: pd.DataFrame,
    *,
    label: str,
    score_col: str,
    probability_col: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = prediction[[label, score_col, probability_col, "month"]].dropna().copy()
    if frame.empty:
        return {"count": 0, "status": "no_rows"}
    frame[label] = frame[label].astype(int)
    positive_rate = float(frame[label].mean())
    auc = safe_auc(frame[label], frame[score_col])
    probs = frame[probability_col].astype(float).clip(1e-9, 1.0 - 1e-9)
    y = frame[label].astype(float)
    brier = float(np.mean((probs - y) ** 2))
    logloss = float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    ece = ece_score(frame[label], probs, int(config["calibration"]["bins"]))
    top_fraction = float(config["weightSelection"]["topBucketFraction"])
    cutoff = float(frame[score_col].quantile(1.0 - top_fraction))
    top = frame[frame[score_col] >= cutoff]
    top_hit = float(top[label].mean()) if len(top) else None
    month_stats = []
    for month, group in frame.groupby("month", sort=True):
        group_top = group[group[score_col] >= cutoff]
        if group_top.empty:
            continue
        month_stats.append(
            {
                "month": str(month),
                "base": float(group[label].mean()),
                "top": float(group_top[label].mean()),
                "improves": bool(group_top[label].mean() > group[label].mean()),
            }
        )
    return {
        "count": int(len(frame)),
        "positiveRate": positive_rate,
        "auc": auc,
        "brier": brier,
        "logloss": logloss,
        "ece": ece,
        "topBucketFraction": top_fraction,
        "topBucketCutoff": cutoff,
        "topBucketCount": int(len(top)),
        "topBucketHitRate": top_hit,
        "topBucketLift": (top_hit / positive_rate if top_hit is not None and positive_rate else None),
        "selectionFractionIfUsedAsHardGate": float(len(top) / len(frame)),
        "mechanicalTradeReductionIfUsedAsHardGate": float(1.0 - len(top) / len(frame)),
        "improvingMonths": int(sum(item["improves"] for item in month_stats)),
        "totalEvaluableMonths": int(len(month_stats)),
        "monthDiagnostics": month_stats,
    }


def fold_months(table: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    start = str(config["data"]["walkForwardStart"])
    end = str(config["data"]["walkForwardEnd"])
    oos = table[(table["trade_date"] >= start) & (table["trade_date"] <= end)]
    return sorted(oos["month"].unique().tolist())


def evaluate_walk_forward(
    table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = list(config["evaluation"]["labels"])
    full_candidates = load_candidates(config["weightSelection"]["fullSessionCandidates"])
    dmd_candidates = load_candidates(config["weightSelection"]["dmdCompleteCandidates"])
    equal_full = full_candidates[0]
    equal_dmd = dmd_candidates[0]
    horizon_minutes = int(config["data"]["purgeBars"]) * int(config["data"]["barIntervalMinutes"])
    prediction_frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    fold_records: list[dict[str, Any]] = []

    for month in fold_months(table, config):
        test_mask = (
            (table["month"] == month)
            & (table["trade_date"] >= config["data"]["walkForwardStart"])
            & (table["trade_date"] <= config["data"]["walkForwardEnd"])
        )
        test_raw = table[test_mask].copy()
        if test_raw.empty:
            continue
        test_start = pd.Timestamp(test_raw["timestamp"].min())
        train_cutoff = test_start - timedelta(minutes=horizon_minutes)
        train_raw = table[table["timestamp"] < train_cutoff].copy()
        normalizer = fit_quantile_normalizer(train_raw, config)
        train_norm = apply_quantile_normalizer(train_raw, normalizer)
        test_norm = apply_quantile_normalizer(test_raw, normalizer)

        populations = {
            "full_session": {
                "train": train_norm,
                "test": test_norm,
                "candidates": full_candidates,
                "equal": equal_full,
            },
            "dmd_complete": {
                "train": train_norm[train_norm["dmd_window_valid"] == 1],
                "test": test_norm[test_norm["dmd_window_valid"] == 1],
                "candidates": dmd_candidates,
                "equal": equal_dmd,
            },
        }
        for population, payload in populations.items():
            train = payload["train"]
            test = payload["test"]
            if test.empty:
                continue
            selected, objectives = select_candidate(
                train,
                payload["candidates"],
                labels,
                config,
            )
            fold_records.append(
                {
                    "month": month,
                    "population": population,
                    "testRows": int(len(test)),
                    "trainRows": int(len(train)),
                    "trainCutoff": train_cutoff.isoformat(),
                    "selectedCandidate": selected.name,
                    "selectedWeights": selected.weights,
                    "trainObjectives": objectives,
                    "normalizer": normalizer,
                }
            )
            modes = {
                "equal": payload["equal"],
                "adaptive": selected,
            }
            if population == "full_session":
                modes["hmm_only"] = next(
                    candidate for candidate in full_candidates if candidate.name == "hmm_only"
                )
            else:
                modes["hmm_only"] = next(
                    candidate for candidate in dmd_candidates if candidate.name == "hmm_only"
                )
                modes["dmd_only"] = next(
                    candidate for candidate in dmd_candidates if candidate.name == "dmd_only"
                )
            for mode_name, candidate in modes.items():
                scored_train = add_candidate_score(train, candidate, "_score")
                scored_test = add_candidate_score(test, candidate, "_score")
                for label in labels:
                    calibrator = fit_calibrator(
                        scored_train,
                        label=label,
                        score_col="_score",
                        config=config,
                    )
                    prediction = scored_test[
                        [
                            "timestamp",
                            "trade_date",
                            "month",
                            "stockCode",
                            label,
                            "_score",
                        ]
                    ].copy()
                    prediction["_probability"] = apply_calibrator(
                        prediction["_score"],
                        calibrator,
                        config,
                    )
                    prediction["candidate"] = candidate.name
                    prediction["mode"] = mode_name
                    prediction["population"] = population
                    prediction["labelName"] = label
                    prediction_frames.setdefault((population, mode_name), []).append(
                        prediction
                    )

    evaluation: dict[str, Any] = {}
    for (population, mode_name), frames in sorted(prediction_frames.items()):
        combined = pd.concat(frames, ignore_index=True)
        label_result: dict[str, Any] = {}
        for label in labels:
            rows = combined[combined["labelName"] == label].rename(columns={label: "_label"})
            label_result[label] = evaluate_prediction_frame(
                rows,
                label="_label",
                score_col="_score",
                probability_col="_probability",
                config=config,
            )
        evaluation.setdefault(population, {})[mode_name] = label_result
    return evaluation, fold_records


def metric_delta(
    evaluation: dict[str, Any],
    *,
    population: str,
    candidate: str,
    baseline: str,
    label: str,
    metric: str,
) -> float | None:
    lhs = evaluation.get(population, {}).get(candidate, {}).get(label, {}).get(metric)
    rhs = evaluation.get(population, {}).get(baseline, {}).get(label, {}).get(metric)
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def summarize_selection(fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(
        [
            {
                "month": row["month"],
                "population": row["population"],
                "selectedCandidate": row["selectedCandidate"],
                "trainRows": row["trainRows"],
                "testRows": row["testRows"],
            }
            for row in fold_records
        ]
    )
    summary: dict[str, Any] = {}
    for population, group in frame.groupby("population", sort=True):
        counts = group["selectedCandidate"].value_counts().to_dict()
        summary[str(population)] = {
            "folds": int(len(group)),
            "selectedCandidateCounts": {str(k): int(v) for k, v in counts.items()},
            "dominantCandidate": str(group["selectedCandidate"].mode().iloc[0]),
        }
    return summary


def build_conclusions(evaluation: dict[str, Any], fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    selection = summarize_selection(fold_records)
    full_5_auc_delta = metric_delta(
        evaluation,
        population="full_session",
        candidate="adaptive",
        baseline="equal",
        label="turning_point_5",
        metric="auc",
    )
    full_10_auc_delta = metric_delta(
        evaluation,
        population="full_session",
        candidate="adaptive",
        baseline="equal",
        label="turning_point_10",
        metric="auc",
    )
    full_5_brier_delta = metric_delta(
        evaluation,
        population="full_session",
        candidate="adaptive",
        baseline="equal",
        label="turning_point_5",
        metric="brier",
    )
    full_10_brier_delta = metric_delta(
        evaluation,
        population="full_session",
        candidate="adaptive",
        baseline="equal",
        label="turning_point_10",
        metric="brier",
    )
    adaptive_beats_equal_auc = bool(
        full_5_auc_delta is not None
        and full_10_auc_delta is not None
        and full_5_auc_delta > 0.0
        and full_10_auc_delta > 0.0
    )
    adaptive_beats_equal_brier = bool(
        full_5_brier_delta is not None
        and full_10_brier_delta is not None
        and full_5_brier_delta < 0.0
        and full_10_brier_delta < 0.0
    )
    dominant_full = selection.get("full_session", {}).get("dominantCandidate")
    multi_model_validated = bool(dominant_full not in {None, "hmm_only", "bocpd_only", "hawkes_only"})
    return {
        "adaptiveFullImprovesEqualAucBothHorizons": adaptive_beats_equal_auc,
        "adaptiveFullImprovesEqualBrierBothHorizons": adaptive_beats_equal_brier,
        "dominantFullSessionCandidate": dominant_full,
        "multiModelBlendValidated": multi_model_validated,
        "interpretation": (
            "Weight adaptation is useful only if the selected weights improve "
            "walk-forward metrics without collapsing to a single reused component. "
            "If the selected candidate is HMM-only, the result is evidence to "
            "down-weight weak physics layers, not evidence for a validated "
            "multi-model stack."
        ),
        "costedReplayJustifiedNow": bool(
            adaptive_beats_equal_auc and adaptive_beats_equal_brier and multi_model_validated
        ),
        "paperIntegrationAllowed": False,
        "recommendedNextStep": (
            "costed_shadow_replay_only_if_multimodel_forward_holds"
            if adaptive_beats_equal_auc and adaptive_beats_equal_brier and multi_model_validated
            else "keep_hmm_as_primary_shadow_risk_context_and_collect_forward_data"
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Risk Stack Phase 1.8 weight adaptation",
        "",
        f"- Run ID: `{result['runId']}`",
        f"- Source commit at run: `{result['sourceCommitAtRun']}`",
        f"- Status: `{result['status']}`",
        f"- Research-only: `{result['researchOnly']}`",
        f"- Paper integration allowed: `{result['paperIntegrationAllowed']}`",
        "",
        "This run tests train-only walk-forward weights for the Phase 1.7 "
        "BOCPD/DMD/HMM/Hawkes risk-context stack. It does not create a trading rule.",
        "",
        "## Data audit",
        "",
    ]
    for key, value in result["dataAudit"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Walk-forward selection",
            "",
            "| population | folds | dominant | candidate counts |",
            "|---|---:|---|---|",
        ]
    )
    for population, summary in result["selectionSummary"].items():
        lines.append(
            f"| {population} | {summary['folds']} | {summary['dominantCandidate']} | "
            f"`{summary['selectedCandidateCounts']}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate OOS metrics",
            "",
            "| population | mode | label | n | AUC | Brier | LogLoss | ECE | top hit | lift |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for population, modes in result["evaluation"].items():
        for mode_name, labels in modes.items():
            for label, metrics in labels.items():
                auc = "null" if metrics.get("auc") is None else f"{metrics['auc']:.4f}"
                ece = "null" if metrics.get("ece") is None else f"{metrics['ece']:.4f}"
                top = (
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
                    f"| {population} | {mode_name} | {label} | {metrics.get('count', 0):,} | "
                    f"{auc} | {metrics.get('brier', 0):.5f} | "
                    f"{metrics.get('logloss', 0):.5f} | {ece} | {top} | {lift} |"
                )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Adaptive full-session weights improve equal AUC on both horizons: `{result['conclusions']['adaptiveFullImprovesEqualAucBothHorizons']}`",
            f"- Adaptive full-session weights improve equal Brier on both horizons: `{result['conclusions']['adaptiveFullImprovesEqualBrierBothHorizons']}`",
            f"- Dominant full-session candidate: `{result['conclusions']['dominantFullSessionCandidate']}`",
            f"- Multi-model blend validated: `{result['conclusions']['multiModelBlendValidated']}`",
            f"- Costed replay justified now: `{result['conclusions']['costedReplayJustifiedNow']}`",
            f"- Recommended next step: `{result['conclusions']['recommendedNextStep']}`",
            "",
            "If adaptation collapses to HMM-only, it means the other components should "
            "be down-weighted in research diagnostics. It is not approval to connect "
            "the score to trading.",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    table, phase17_result = load_table(config)
    evaluation, fold_records = evaluate_walk_forward(table, config)
    selection_summary = summarize_selection(fold_records)
    result = {
        "schemaVersion": "risk_stack_weight_adaptation_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": phase2a.source_commit(),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "paperIntegrationAllowed": False,
        "sourcePhase17RunId": config["source"]["phase17RunId"],
        "dataAudit": {
            "rows": int(len(table)),
            "symbols": int(table["stockCode"].nunique()),
            "tradingDays": int(table["trade_date"].nunique()),
            "walkForwardStart": config["data"]["walkForwardStart"],
            "walkForwardEnd": config["data"]["walkForwardEnd"],
            "initialTrainingEnd": config["data"]["initialTrainingEnd"],
            "folds": int(len(fold_records)),
            "dmdCompleteRows": int((table["dmd_window_valid"] == 1).sum()),
            "phase17PaperIntegrationAllowed": phase17_result["paperIntegrationAllowed"],
        },
        "methodAudit": {
            "candidateGridOnly": True,
            "negativeWeightsAllowed": False,
            "foldNormalizationFitOnTrainingOnly": True,
            "foldCalibrationFitOnTrainingOnly": True,
            "purgeBars": int(config["data"]["purgeBars"]),
            "maxLabelHorizonBars": int(config["data"]["maxLabelHorizonBars"]),
            "phase15LedgerRead": False,
            "paperTradingArtifactsRead": False,
            "labelsFutureUsingButOfflineOnly": True,
            "featuresPastOnlyInheritedFromPhase17": True,
        },
        "selectionSummary": selection_summary,
        "foldSelections": fold_records,
        "evaluation": evaluation,
        "conclusions": build_conclusions(evaluation, fold_records),
        "skipped": [
            "continuous_weight_optimization",
            "negative_weights",
            "full_sample_refit",
            "paper_trading_integration",
            "build_decision_integration",
            "position_sizing_integration",
            "online_inference",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    phase2a.atomic_json(output_dir / "risk_stack_weight_adaptation_result.json", result)
    phase2a.atomic_text(
        output_dir / "risk_stack_weight_adaptation_report.md",
        render_report(result),
    )
    selection_rows = []
    for record in fold_records:
        selection_rows.append(
            {
                "month": record["month"],
                "population": record["population"],
                "selectedCandidate": record["selectedCandidate"],
                "selectedWeights": json.dumps(record["selectedWeights"], ensure_ascii=False),
                "trainRows": record["trainRows"],
                "testRows": record["testRows"],
                "trainCutoff": record["trainCutoff"],
            }
        )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "fold_weight_selection.csv",
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
        "risk_stack_weight_adapt_"
        + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output_dir = resolve(config["output"]["root"]) / run_id
    result = run(config, output_dir)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "status": result["status"],
                "dominant_full_candidate": result["conclusions"][
                    "dominantFullSessionCandidate"
                ],
                "multi_model_blend_validated": result["conclusions"][
                    "multiModelBlendValidated"
                ],
                "costed_replay_justified": result["conclusions"][
                    "costedReplayJustifiedNow"
                ],
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
