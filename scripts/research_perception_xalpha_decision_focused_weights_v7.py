#!/usr/bin/env python3
"""Decision-focused bounded weights for the frozen twelve-factor A-share book.

V6 fitted stock-wise targets.  V7 instead learns a same-day pairwise boundary: names
that truly belonged in the realized Top10 must rank above the next twenty names.  Weight
fitting uses base-fit dates only.  An independent calibration block maps the twelve ranks
to ordinary/severe-loss probabilities and maps five block-subsample weight replicas to
return forecasts.  Audit, validation and shadow can only reject.

This module is permanently research/shadow-only and never imports a broker, order,
production decision, position sizing, overlay or risk-gate path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
VENDOR_ROOT = ROOT / "scripts" / "vendor" / "vibe_factors"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_twelve_factor_utility_weights_v6 as v6  # noqa: E402


SCHEMA_VERSION = "perception_xalpha_decision_focused_weights_result_v7"
CODE_VERSION = "perception_xalpha_decision_focused_weights_v7_20260809"
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "perception_xalpha_decision_focused_weights_v7.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_decision_focused_weights_v7":
        raise ValueError("unexpected decision-focused V7 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("decision-focused V7 must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every V7 mutation permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("V7 output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    for key in (
        "frozenTwelveFactorsAndDirections",
        "decisionLossFitsBaseDatesOnly",
        "calibrationDatesAreIndependent",
        "auditValidationShadowAreRejectOnly",
    ):
        if hypothesis.get(key) is not True:
            raise ValueError(f"missing V7 causal lock: {key}")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical V7 cannot promote")
    training = config["decisionTraining"]
    lower = float(training["minimumFactorWeight"])
    upper = float(training["maximumFactorWeight"])
    if not (0.0 < lower < 1.0 / 12.0 < upper <= 0.25):
        raise ValueError("V7 bounds must contain equal weight and remain conservative")
    if lower * 12.0 > 1.0 or upper * 12.0 < 1.0:
        raise ValueError("V7 bounds cannot form a simplex")
    if int(training["trueTopCount"]) != 10:
        raise ValueError("V7 decision boundary must remain Top10")
    if int(training["blockSubsampleReplicas"]) != 5:
        raise ValueError("V7 replica count changed")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("V7 output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("V7 orders must remain empty")


def v6_prior_weights(
    result: dict[str, Any], factors: list[dict[str, Any]]
) -> np.ndarray:
    lookup = {str(row["factorKey"]): float(row["finalWeight"]) for row in result["weights"]}
    weights = np.asarray([lookup[row["factorKey"]] for row in factors], dtype=float)
    if len(weights) != 12 or abs(float(weights.sum()) - 1.0) > 1e-8:
        raise ValueError("V6 prior is not a twelve-factor simplex")
    return weights


def pairwise_training_arrays(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build daily-equal Top10 versus near-boundary pair differences."""
    spec = config["decisionTraining"]
    keys = list(ranks)
    top_count = int(spec["trueTopCount"])
    negative_count = int(spec["nearBoundaryNegativeCount"])
    cost = float(load_json(ROOT / config["baseV6Config"])["data"]["roundTripCost"])
    severe = float(load_json(ROOT / config["baseV6Config"])["data"]["severeLossThreshold"])
    minimum = int(spec["minimumCrossSectionRows"])
    differences: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    used_days = 0
    used_rows = 0
    for date in dates:
        y = returns.loc[date].to_numpy(dtype=float)
        x = np.column_stack(
            [ranks[key].loc[date].to_numpy(dtype=float) for key in keys]
        )
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(valid.sum()) < max(minimum, top_count + negative_count):
            continue
        xv = x[valid]
        yv = y[valid]
        realized_utility = (
            yv
            - float(spec["realizedUtilityPenaltyNetLoss"]) * (yv <= cost)
            - float(spec["realizedUtilityPenaltySevereLoss"]) * (yv <= severe)
        )
        order = np.argsort(realized_utility, kind="stable")[::-1]
        positive = xv[order[:top_count]]
        negative = xv[order[top_count : top_count + negative_count]]
        day_diff = (positive[:, None, :] - negative[None, :, :]).reshape(
            -1, len(keys)
        )
        differences.append(day_diff)
        sample_weights.append(np.full(len(day_diff), 1.0 / len(day_diff)))
        used_days += 1
        used_rows += int(valid.sum())
    if used_days == 0:
        raise RuntimeError("no V7 dates met the pairwise cross-section requirement")
    return (
        np.concatenate(differences, axis=0),
        np.concatenate(sample_weights, axis=0) / used_days,
        {
            "usedTradingDays": used_days,
            "usedRows": used_rows,
            "pairCount": int(sum(len(row) for row in differences)),
            "pairsPerUsedDay": top_count * negative_count,
        },
    )


def fit_pairwise_weights(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    prior: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    differences, sample_weight, audit = pairwise_training_arrays(
        ranks, returns, dates, config
    )
    spec = config["decisionTraining"]
    lower = float(spec["minimumFactorWeight"])
    upper = float(spec["maximumFactorWeight"])
    l2 = float(spec["l2ShrinkageToV6"])

    def objective(weight: np.ndarray) -> float:
        margin = differences @ weight
        logistic = float(np.sum(np.logaddexp(0.0, -margin) * sample_weight))
        return logistic + l2 * float(np.square(weight - prior).sum())

    def gradient(weight: np.ndarray) -> np.ndarray:
        margin = differences @ weight
        inverse = np.exp(-np.logaddexp(0.0, margin))
        return -(differences.T @ (inverse * sample_weight)) + 2.0 * l2 * (weight - prior)

    result = minimize(
        objective,
        prior,
        jac=gradient,
        method="SLSQP",
        bounds=[(lower, upper)] * len(prior),
        constraints=[{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"V7 pairwise fit failed closed: {result.message}")
    weight = np.asarray(result.x, dtype=float)
    if (
        abs(float(weight.sum()) - 1.0) > 1e-8
        or np.any(weight < lower - 1e-9)
        or np.any(weight > upper + 1e-9)
    ):
        raise RuntimeError("V7 pairwise optimizer violated its simplex")
    audit.update(
        {
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "success": bool(result.success),
        }
    )
    return weight, audit


def block_subsample_dates(
    dates: pd.DatetimeIndex, config: dict[str, Any], replica: int
) -> pd.DatetimeIndex:
    spec = config["decisionTraining"]
    block = int(spec["blockLengthTradingDays"])
    blocks = [dates[start : start + block] for start in range(0, len(dates), block)]
    keep = max(1, int(math.ceil(len(blocks) * float(spec["blockSubsampleFraction"]))))
    rng = np.random.default_rng(int(spec["randomSeed"]) + replica)
    selected = sorted(rng.choice(len(blocks), size=keep, replace=False).tolist())
    return pd.DatetimeIndex(np.concatenate([blocks[index].to_numpy() for index in selected]))


def calibration_rows(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    keys = list(ranks)
    feature_rows: list[np.ndarray] = []
    outcome_rows: list[np.ndarray] = []
    day_sizes: list[int] = []
    for date in dates:
        y = returns.loc[date].to_numpy(dtype=float)
        x = np.column_stack(
            [ranks[key].loc[date].to_numpy(dtype=float) for key in keys]
        )
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if not valid.any():
            continue
        feature_rows.append(x[valid])
        outcome_rows.append(y[valid])
        day_sizes.append(int(valid.sum()))
    if not feature_rows:
        raise RuntimeError("V7 calibration rows are empty")
    mean_size = float(np.mean(day_sizes))
    weights = np.concatenate(
        [np.full(size, mean_size / size, dtype=float) for size in day_sizes]
    )
    return (
        np.concatenate(feature_rows),
        np.concatenate(outcome_rows),
        weights,
        {"rows": int(sum(day_sizes)), "days": len(day_sizes), "meanRowsPerDay": mean_size},
    )


def fit_calibrators(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    all_weights: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    x, y, sample_weight, audit = calibration_rows(ranks, returns, dates)
    if len(y) < int(config["calibration"]["minimumCalibrationRows"]):
        raise RuntimeError("V7 independent calibration block is too small")
    alpha = float(config["calibration"]["expectedReturnRidgeAlpha"])
    return_models: list[Ridge] = []
    for weight in all_weights:
        score = x @ weight
        model = Ridge(alpha=alpha)
        model.fit(score.reshape(-1, 1), y, sample_weight=sample_weight)
        return_models.append(model)
    v6_config = load_json(ROOT / config["baseV6Config"])
    cost = float(v6_config["data"]["roundTripCost"])
    severe = float(v6_config["data"]["severeLossThreshold"])
    classifier_options = {
        "C": float(config["calibration"]["logisticC"]),
        "solver": "lbfgs",
        "max_iter": 500,
    }
    net_model = LogisticRegression(**classifier_options)
    severe_model = LogisticRegression(**classifier_options)
    net_model.fit(x, (y <= cost).astype(int), sample_weight=sample_weight)
    severe_model.fit(x, (y <= severe).astype(int), sample_weight=sample_weight)
    audit.update(
        {
            "netLossPrior": float(np.average(y <= cost, weights=sample_weight)),
            "severeLossPrior": float(np.average(y <= severe, weights=sample_weight)),
        }
    )
    return {
        "returnModels": return_models,
        "netLossModel": net_model,
        "severeLossModel": severe_model,
        "audit": audit,
    }


def predict_frames(
    ranks: dict[str, pd.DataFrame],
    all_weights: list[np.ndarray],
    calibrators: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    keys = list(ranks)
    template = ranks[keys[0]]
    shape = template.shape
    expected = np.full(shape, np.nan, dtype=float)
    uncertainty = np.full(shape, np.nan, dtype=float)
    net_loss = np.full(shape, np.nan, dtype=float)
    severe_loss = np.full(shape, np.nan, dtype=float)
    for row_position, date in enumerate(template.index):
        x = np.column_stack(
            [ranks[key].loc[date].to_numpy(dtype=float) for key in keys]
        )
        valid = np.isfinite(x).all(axis=1)
        if not valid.any():
            continue
        xv = x[valid]
        replica_predictions = np.column_stack(
            [
                model.predict((xv @ weight).reshape(-1, 1))
                for weight, model in zip(
                    all_weights, calibrators["returnModels"], strict=True
                )
            ]
        )
        expected[row_position, valid] = replica_predictions.mean(axis=1)
        uncertainty[row_position, valid] = replica_predictions.std(axis=1, ddof=1)
        net_loss[row_position, valid] = calibrators["netLossModel"].predict_proba(xv)[:, 1]
        severe_loss[row_position, valid] = calibrators["severeLossModel"].predict_proba(xv)[:, 1]
    expected_frame = pd.DataFrame(expected, index=template.index, columns=template.columns)
    uncertainty_frame = pd.DataFrame(
        uncertainty, index=template.index, columns=template.columns
    )
    net_frame = pd.DataFrame(net_loss, index=template.index, columns=template.columns)
    severe_frame = pd.DataFrame(
        severe_loss, index=template.index, columns=template.columns
    )
    calibration = config["calibration"]
    utility = (
        expected_frame
        - float(calibration["uncertaintyPenalty"]) * uncertainty_frame
        - float(calibration["utilityPenaltyNetLoss"]) * net_frame
        - float(calibration["utilityPenaltySevereLoss"]) * severe_frame
    )
    return {
        "expectedReturn": expected_frame,
        "predictionUncertainty": uncertainty_frame,
        "netLossProbability": net_frame,
        "severeLossProbability": severe_frame,
        "utility": utility,
    }


def period_metrics(
    returns: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    v6_config: dict[str, Any],
) -> dict[str, Any]:
    return v6.period_evaluation(returns, predictions, dates, v6_config)


def baseline_metrics(
    returns: pd.DataFrame,
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    v6_config: dict[str, Any],
) -> dict[str, Any]:
    return v6.outcome_metrics(
        returns,
        score,
        dates,
        10,
        float(v6_config["data"]["roundTripCost"]),
        float(v6_config["data"]["severeLossThreshold"]),
    )


def verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["evaluation"]
    audit = report["periods"]["audit"]
    prior = report["calibrationAudit"]["constantPriorNetLoss"]
    checks: dict[str, bool] = {
        "auditNetLossAuc": audit["probability"]["netLoss"]["auc"] is not None
        and audit["probability"]["netLoss"]["auc"]
        >= float(gate["minimumAuditNetLossAuc"]),
        "auditNetLossBrier": audit["probability"]["netLoss"]["brier"] is not None
        and audit["probability"]["netLoss"]["brier"] < prior["brier"],
    }
    external: dict[str, Any] = {}
    for period in ("validation", "shadow"):
        candidate = report["periods"][period]["top"]["10"]
        baseline = report["v6Baseline"][period]
        row = {
            "grossPositive": candidate["meanGrossReturn"] is not None
            and candidate["meanGrossReturn"] > 0.0,
            "grossImprovesV6": candidate["meanGrossReturn"] is not None
            and baseline["meanGrossReturn"] is not None
            and candidate["meanGrossReturn"] > baseline["meanGrossReturn"],
            "nonPositiveNotWorse": candidate["nonPositiveRate"] is not None
            and baseline["nonPositiveRate"] is not None
            and candidate["nonPositiveRate"] <= baseline["nonPositiveRate"],
            "netLossAucAboveRandom": report["periods"][period]["probability"]["netLoss"]["auc"]
            is not None
            and report["periods"][period]["probability"]["netLoss"]["auc"] > 0.5,
        }
        row["passed"] = all(row.values())
        external[period] = row
    stable = all(checks.values()) and all(row["passed"] for row in external.values())
    return {
        "status": "research_only_not_eligible_for_trading",
        "auditChecks": checks,
        "externalRejectOnly": external,
        "stableHistoricalHypothesis": stable,
        "decision": (
            "fresh_forward_preregistration_still_required"
            if stable
            else "reject_for_trading_keep_diagnostics"
        ),
        "historicalRunCanPromote": False,
    }


def latest_rows(
    predictions: dict[str, pd.DataFrame],
    ranks: dict[str, pd.DataFrame],
    names: dict[str, str],
) -> pd.DataFrame:
    date = predictions["utility"].index.max()
    frame = pd.DataFrame(
        {
            "signalDate": date.date().isoformat(),
            "securityId": predictions["utility"].columns,
            "name": [names.get(str(item), "") for item in predictions["utility"].columns],
            "expectedExecutableReturn": predictions["expectedReturn"].loc[date].to_numpy(dtype=float),
            "predictionUncertainty": predictions["predictionUncertainty"].loc[date].to_numpy(dtype=float),
            "netLossProbability": predictions["netLossProbability"].loc[date].to_numpy(dtype=float),
            "severeLossProbability": predictions["severeLossProbability"].loc[date].to_numpy(dtype=float),
            "selectionUtility": predictions["utility"].loc[date].to_numpy(dtype=float),
        }
    )
    for key, value in ranks.items():
        frame[f"factor_{key.replace('/', '__')}"] = value.loc[date].to_numpy(dtype=float)
    return frame.dropna(subset=["selectionUtility"]).sort_values(
        ["selectionUtility", "expectedExecutableReturn", "netLossProbability"],
        ascending=[False, False, True],
    )


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Decision-focused twelve-factor weights V7",
        "",
        "**Research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Latest signal date: `{report['latestSignalDate']}`",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Orders: `[]`",
        "",
        "## Weight comparison",
        "",
        "| Factor | V6 | V7 decision-focused | Replica min | Replica max |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["weights"]:
        lines.append(
            f"| {row['factorKey']} | {row['v6Weight']:.4%} | {row['v7Weight']:.4%} | "
            f"{row['replicaMinimum']:.4%} | {row['replicaMaximum']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Exact-window Top10 comparison",
            "",
            "| Period | Book | Mean gross | Mean net | Net win | Non-positive | Severe loss |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for period in ("audit", "validation", "shadow"):
        for book, metrics in (
            ("V6", report["v6Baseline"][period]),
            ("V7", report["periods"][period]["top"]["10"]),
        ):
            lines.append(
                f"| {period} | {book} | {metrics.get('meanGrossReturn')} | "
                f"{metrics.get('meanNetReturn')} | {metrics.get('netWinRate')} | "
                f"{metrics.get('nonPositiveRate')} | {metrics.get('severeLossRate')} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- V7 trains the Top10 boundary directly instead of minimizing stock-wise squared error.",
            "- Five 21-session block subsamples estimate instability in learned factor weights.",
            "- Ordinary and severe loss probabilities use all twelve ranks on an independent calibration block.",
            "- Validation and shadow are already viewed and can only reject; even a pass requires fresh forward data.",
            "- Latest Top10/Top3/Top1 files are diagnostics, never orders.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    v6_config = load_json(ROOT / config["baseV6Config"])
    v6.validate_config(v6_config)
    v6_result = load_json(ROOT / config["baseV6Result"])
    factors, _, source_sha = v6.source_factors(v6_config)
    base = load_json(ROOT / v6_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    ranks, factor_audit = v6.compute_factor_ranks(panel, factors)
    returns, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        int(v6_config["data"]["holdingTradingDays"]),
        int(v6_config["data"]["maximumExitDelayTradingDays"]),
    )
    returns = returns.where(execution_eligible)
    split = autonomous.make_split(panel["close"].index, cog_config)
    train_dates = pd.DatetimeIndex(split.train[split.train].index)
    validation_dates = pd.DatetimeIndex(split.validation[split.validation].index)
    shadow_dates = pd.DatetimeIndex(split.shadow[split.shadow].index)
    partitions = v6.training_partitions(train_dates, v6_config)
    prior = v6_prior_weights(v6_result, factors)

    final_weight, fit_audit = fit_pairwise_weights(
        ranks, returns, partitions["baseFit"], prior, config
    )
    print("decision_weight_fit_complete=true", flush=True)
    replicas: list[np.ndarray] = []
    replica_audit: list[dict[str, Any]] = []
    for replica in range(int(config["decisionTraining"]["blockSubsampleReplicas"])):
        replica_dates = block_subsample_dates(partitions["baseFit"], config, replica)
        weight, audit = fit_pairwise_weights(
            ranks, returns, replica_dates, prior, config
        )
        replicas.append(weight)
        replica_audit.append(
            {
                "replica": replica + 1,
                "dateCount": len(replica_dates),
                "dateRange": [
                    replica_dates.min().date().isoformat(),
                    replica_dates.max().date().isoformat(),
                ],
                "fit": audit,
            }
        )
        print(f"weight_replica_complete {replica + 1}/{len(range(5))}", flush=True)

    fold_rows: list[dict[str, Any]] = []
    for index, (fold_train, fold_test) in enumerate(
        v6.expanding_folds(partitions["baseFit"], v6_config), start=1
    ):
        fold_weight, _ = fit_pairwise_weights(
            ranks, returns, fold_train, prior, config
        )
        fold_score = v6.weighted_score(ranks, fold_weight)
        fold_rows.append(
            {
                "fold": index,
                "trainRange": [fold_train.min().date().isoformat(), fold_train.max().date().isoformat()],
                "testRange": [fold_test.min().date().isoformat(), fold_test.max().date().isoformat()],
                "metrics": baseline_metrics(returns, fold_score, fold_test, v6_config),
            }
        )
        print(f"inner_fold_complete {index}/4", flush=True)

    all_weights = [final_weight, *replicas]
    calibrators = fit_calibrators(
        ranks, returns, partitions["calibration"], all_weights, config
    )
    predictions = predict_frames(ranks, all_weights, calibrators, config)
    print("independent_calibration_and_prediction_complete=true", flush=True)

    periods = {
        "audit": period_metrics(returns, predictions, partitions["audit"], v6_config),
        "validation": period_metrics(returns, predictions, validation_dates, v6_config),
        "shadow": period_metrics(returns, predictions, shadow_dates, v6_config),
    }
    v6_score = v6.weighted_score(ranks, prior)
    v6_baseline = {
        "audit": baseline_metrics(returns, v6_score, partitions["audit"], v6_config),
        "validation": baseline_metrics(returns, v6_score, validation_dates, v6_config),
        "shadow": baseline_metrics(returns, v6_score, shadow_dates, v6_config),
    }
    calibration_y = calibration_rows(
        ranks, returns, partitions["audit"]
    )[1]
    cost = float(v6_config["data"]["roundTripCost"])
    constant_prior = v6.baseline_prior_metrics(
        (calibration_y <= cost).astype(float),
        float(calibrators["audit"]["netLossPrior"]),
    )
    replica_matrix = np.vstack(replicas)
    weight_rows = [
        {
            "factorKey": item["factorKey"],
            "direction": item["direction"],
            "v6Weight": float(prior[position]),
            "v7Weight": float(final_weight[position]),
            "replicaMinimum": float(replica_matrix[:, position].min()),
            "replicaMaximum": float(replica_matrix[:, position].max()),
            "replicaStd": float(replica_matrix[:, position].std(ddof=1)),
        }
        for position, item in enumerate(factors)
    ]
    current_run = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": current_run,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sourceSummarySha256": source_sha,
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "latestSignalDate": panel["close"].index.max().date().isoformat(),
        "symbolCount": int(panel["close"].shape[1]),
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "trainingPartitions": {
            key: [value.min().date().isoformat(), value.max().date().isoformat(), len(value)]
            for key, value in partitions.items()
        },
        "decisionFit": fit_audit,
        "weightReplicas": replica_audit,
        "innerWalkForward": fold_rows,
        "calibrationAudit": {
            **calibrators["audit"],
            "constantPriorNetLoss": constant_prior,
        },
        "weights": weight_rows,
        "periods": periods,
        "v6Baseline": v6_baseline,
        "exitDelayDistribution": {
            str(key): int(value)
            for key, value in exit_delay.stack(future_stack=True).value_counts().sort_index().items()
        },
        "orders": [],
        "automaticTradingChanges": [],
        "knownLimitations": config["knownLimitations"],
    }
    report["verdict"] = verdict(report, config)
    latest = latest_rows(predictions, ranks, v6.name_map(base))
    output = ROOT / config["output"]["root"] / current_run
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output / "summary.json",
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(output / "report.md", markdown_report(report))
    pd.DataFrame(weight_rows).to_csv(output / "trained_weights.csv", index=False)
    for count in (10, 3, 1):
        latest.head(count).to_csv(output / f"latest_diagnostic_top{count}.csv", index=False)
    latest.head(0).assign(
        qualificationStatus="empty_because_historical_research_cannot_promote"
    ).to_csv(output / "qualified_selections.csv", index=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runId": current_run,
                "latestSignalDate": report["latestSignalDate"],
                "verdict": report["verdict"]["decision"],
                "orders": [],
            },
            ensure_ascii=False,
        )
    )
    return report


def self_test() -> None:
    config = load_json(DEFAULT_CONFIG)
    validate_config(config)
    dates = pd.bdate_range("2022-01-03", periods=100)
    columns = [f"S{index:03d}" for index in range(120)]
    rng = np.random.default_rng(7)
    useful = pd.DataFrame(rng.uniform(size=(100, 120)), index=dates, columns=columns)
    ranks = {"useful": useful}
    for index in range(11):
        ranks[f"noise_{index}"] = pd.DataFrame(
            rng.uniform(size=useful.shape), index=dates, columns=columns
        )
    returns = useful * 0.05 + pd.DataFrame(
        rng.normal(scale=0.002, size=useful.shape), index=dates, columns=columns
    )
    prior = np.full(12, 1.0 / 12.0)
    weight, audit = fit_pairwise_weights(ranks, returns, dates, prior, config)
    assert abs(float(weight.sum()) - 1.0) < 1e-8
    assert weight[0] > np.median(weight[1:])
    assert audit["pairCount"] == len(dates) * 200
    print("self_test_passed=true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args.config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
