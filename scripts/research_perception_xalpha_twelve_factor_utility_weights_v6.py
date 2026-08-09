#!/usr/bin/env python3
"""Train bounded non-equal weights for the frozen twelve-factor A-share book.

This module is permanently research/shadow-only.  Factor definitions and directions come
from the train-only V2 factor-zoo artifact.  Weights are fitted on an earlier base-fit block;
an independent calibration block maps the frozen score to expected return and loss
probabilities; reliability audit, validation and shadow blocks can only reject the result.

No broker, order, production-decision, position-sizing or risk-gate path is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
VENDOR_ROOT = ROOT / "scripts" / "vendor" / "vibe_factors"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402


SCHEMA_VERSION = "perception_xalpha_twelve_factor_utility_weights_result_v6"
CODE_VERSION = "perception_xalpha_twelve_factor_utility_weights_v6_20260809"
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "perception_xalpha_twelve_factor_utility_weights_v6.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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
    if config.get("schemaVersion") != "perception_xalpha_twelve_factor_utility_weights_v6":
        raise ValueError("unexpected twelve-factor utility-weight schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("utility-weight research must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every utility-weight mutation permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("utility-weight output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    required_true = (
        "factorDefinitionsAndDirectionsFrozen",
        "onlyTrainDatesMayFitWeights",
        "calibrationDatesIndependentOfWeightFit",
        "auditValidationAndShadowAreRejectOnly",
    )
    if not all(hypothesis.get(key) is True for key in required_true):
        raise ValueError("causal preregistration flags must remain true")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical weight research cannot promote")
    data = config["data"]
    if int(data["holdingTradingDays"]) != 1:
        raise ValueError("V6 is preregistered for the next executable session only")
    if list(map(int, data["topCounts"])) != [1, 3, 10]:
        raise ValueError("V6 must report the frozen Top1/Top3/Top10 grid")
    training = config["weightTraining"]
    heads = training["objectiveHeads"]
    if set(heads) != {"returnPercentile", "netNonLoss", "severeNonLoss"}:
        raise ValueError("the three utility heads changed")
    if abs(sum(map(float, heads.values())) - 1.0) > 1e-12:
        raise ValueError("utility-head weights must sum to one")
    lower = float(training["minimumFactorWeight"])
    upper = float(training["maximumFactorWeight"])
    if not (0.0 < lower < 1.0 / 12.0 < upper < 1.0):
        raise ValueError("factor bounds do not contain the twelve-factor equal weight")
    if lower * 12.0 > 1.0 or upper * 12.0 < 1.0:
        raise ValueError("factor bounds cannot form a simplex")
    purge = int(training["purgeTradingDays"])
    maximum_lookahead = int(data["holdingTradingDays"]) + int(
        data["maximumExitDelayTradingDays"]
    )
    if purge < maximum_lookahead:
        raise ValueError("purge must cover the complete executable exit delay")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("V6 output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("V6 orders must remain empty")


def source_factors(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    path = ROOT / config["frozenSourceSummary"]
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8"))
    if source.get("schemaVersion") != "perception_xalpha_nextday_factor_zoo_result_v2":
        raise ValueError("frozen source is not the expected V2 result")
    discovery = source["factorDiscovery"]
    factors = [
        {
            "factorKey": str(item["factorKey"]),
            "direction": float(item["direction"]),
        }
        for item in discovery["selectedFactors"]
    ]
    if len(factors) != 12 or len({item["factorKey"] for item in factors}) != 12:
        raise ValueError("the frozen source must contain twelve unique factors")
    if discovery["selectedFactorPrefixCausality"]["passed"] != 12:
        raise ValueError("not every frozen factor passed the source causality audit")
    return factors, source, hashlib.sha256(raw).hexdigest()


def build_factor_inputs(panel: dict[str, Any]) -> dict[str, Any]:
    inputs = dict(panel)
    close = panel["close"]
    inputs["returns"] = close.pct_change(fill_method=None)
    volume = panel["volume"].replace(0.0, np.nan)
    inputs["vwap"] = panel["amount"].div(volume).combine_first(close)
    return inputs


def compute_factor_ranks(
    panel: dict[str, Any], factors: list[dict[str, Any]]
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    close = panel["close"]
    eligible = panel["eligible"]
    inputs = build_factor_inputs(panel)
    output: dict[str, pd.DataFrame] = {}
    audit: list[dict[str, Any]] = []
    for position, item in enumerate(factors, start=1):
        key = item["factorKey"]
        zoo, name = key.split("/", 1)
        module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
        raw = module.compute(inputs).reindex_like(close)
        oriented = raw * float(item["direction"])
        ranked = oriented.where(eligible).rank(axis=1, pct=True).astype(np.float32)
        output[key] = ranked
        audit.append(
            {
                "factorKey": key,
                "direction": float(item["direction"]),
                "finiteValues": int(np.isfinite(ranked.to_numpy()).sum()),
                "pastOnlySourceAudit": "passed_in_frozen_v2_source",
            }
        )
        print(f"factor_rank_ready {position}/12 {key}", flush=True)
        del raw, oriented, ranked
    return output, audit


def training_partitions(
    train_dates: pd.DatetimeIndex, config: dict[str, Any]
) -> dict[str, pd.DatetimeIndex]:
    training = config["weightTraining"]
    calibration_days = int(training["calibrationTradingDays"])
    audit_days = int(training["reliabilityAuditTradingDays"])
    purge = int(training["purgeTradingDays"])
    needed = (
        int(training["minimumBaseFitTradingDays"])
        + calibration_days
        + audit_days
        + 2 * purge
    )
    if len(train_dates) < needed:
        raise RuntimeError(f"insufficient train dates for frozen partitions: {len(train_dates)} < {needed}")
    audit = train_dates[-audit_days:]
    calibration_end = len(train_dates) - audit_days - purge
    calibration = train_dates[calibration_end - calibration_days : calibration_end]
    base_end = calibration_end - calibration_days - purge
    base_fit = train_dates[:base_end]
    if len(base_fit) < int(training["minimumBaseFitTradingDays"]):
        raise RuntimeError("base-fit partition is too short")
    return {"baseFit": base_fit, "calibration": calibration, "audit": audit}


def _head_cross_products(
    ranks: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    dates: pd.DatetimeIndex,
    minimum_rows: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    keys = list(ranks)
    features = len(keys)
    gram = np.zeros((features, features), dtype=float)
    moment = np.zeros(features, dtype=float)
    used_days = 0
    used_rows = 0
    for date in dates:
        y = target.loc[date].to_numpy(dtype=float)
        x = np.column_stack([ranks[key].loc[date].to_numpy(dtype=float) for key in keys])
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        count = int(valid.sum())
        if count < minimum_rows:
            continue
        xv = x[valid]
        yv = y[valid]
        gram += (xv.T @ xv) / count
        moment += (xv.T @ yv) / count
        used_days += 1
        used_rows += count
    if used_days == 0:
        raise RuntimeError("no dates met the minimum cross-section for weight fitting")
    return gram / used_days, moment / used_days, used_days, used_rows


def fit_bounded_head_weights(
    ranks: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lower: float,
    upper: float,
    l2: float,
    minimum_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one non-negative simplex head with equal weight per trading day."""
    gram, moment, used_days, used_rows = _head_cross_products(
        ranks, target, dates, minimum_rows
    )
    size = len(ranks)
    prior = np.full(size, 1.0 / size)

    def objective(weight: np.ndarray) -> float:
        residual = float(weight @ gram @ weight - 2.0 * moment @ weight)
        shrinkage = float(l2 * np.square(weight - prior).sum())
        return residual + shrinkage

    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(lower, upper)] * size,
        constraints=[{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"bounded head-weight fit failed closed: {result.message}")
    weight = np.asarray(result.x, dtype=float)
    if (
        abs(float(weight.sum()) - 1.0) > 1e-8
        or np.any(weight < lower - 1e-9)
        or np.any(weight > upper + 1e-9)
    ):
        raise RuntimeError("bounded head-weight optimizer violated constraints")
    return weight, {
        "usedTradingDays": used_days,
        "usedRows": used_rows,
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "success": bool(result.success),
    }


def combine_head_weights(
    fitted: dict[str, np.ndarray], objective_heads: dict[str, float]
) -> np.ndarray:
    if set(fitted) != set(objective_heads):
        raise ValueError("fitted heads do not match preregistered objective heads")
    combined = sum(
        np.asarray(fitted[name], dtype=float) * float(objective_heads[name])
        for name in objective_heads
    )
    combined = np.asarray(combined, dtype=float)
    if abs(float(combined.sum()) - 1.0) > 1e-10:
        raise RuntimeError("combined factor weights do not sum to one")
    return combined


def weighted_score(
    ranks: dict[str, pd.DataFrame], weights: np.ndarray
) -> pd.DataFrame:
    keys = list(ranks)
    numerator = ranks[keys[0]] * 0.0
    available = ranks[keys[0]] * 0.0
    for key, weight in zip(keys, weights, strict=True):
        frame = ranks[key]
        numerator = numerator.add(frame.fillna(0.0) * float(weight), fill_value=0.0)
        available = available.add(frame.notna().astype(float) * float(weight), fill_value=0.0)
    return numerator.div(available.replace(0.0, np.nan))


def return_percentile_target(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.rank(axis=1, pct=True).where(returns.notna())


def expanding_folds(
    base_dates: pd.DatetimeIndex, config: dict[str, Any]
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    training = config["weightTraining"]
    count = int(training["expandingInnerFolds"])
    test_days = int(training["foldTestTradingDays"])
    purge = int(training["purgeTradingDays"])
    minimum = int(training["minimumBaseFitTradingDays"])
    latest_test_start = len(base_dates) - count * test_days
    if latest_test_start - purge < minimum:
        raise RuntimeError("base-fit window cannot form the preregistered expanding folds")
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for index in range(count):
        test_start = latest_test_start + index * test_days
        train_end = test_start - purge
        folds.append((base_dates[:train_end], base_dates[test_start : test_start + test_days]))
    return folds


def top_mask(score: pd.DataFrame, top_count: int) -> pd.DataFrame:
    return score.rank(axis=1, ascending=False, method="first").le(top_count).fillna(False)


def outcome_metrics(
    returns: pd.DataFrame,
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    top_count: int,
    cost: float,
    severe: float,
) -> dict[str, Any]:
    selected = returns.reindex(index=dates).where(top_mask(score.reindex(index=dates), top_count))
    stacked = selected.stack(future_stack=True).dropna().astype(float)
    daily = selected.mean(axis=1, skipna=True).dropna()
    if stacked.empty or daily.empty:
        return {
            "observations": 0,
            "signalDays": 0,
            "meanGrossReturn": None,
            "meanNetReturn": None,
            "netWinRate": None,
            "nonPositiveRate": None,
            "severeLossRate": None,
            "dailyCvar10": None,
        }
    tail = max(1, int(math.ceil(len(daily) * 0.10)))
    return {
        "observations": int(len(stacked)),
        "signalDays": int(len(daily)),
        "meanNamesPerDay": round(float(len(stacked) / len(daily)), 6),
        "meanGrossReturn": round(float(stacked.mean()), 8),
        "meanNetReturn": round(float(stacked.mean() - cost), 8),
        "netWinRate": round(float(stacked.gt(cost).mean()), 8),
        "nonPositiveRate": round(float(stacked.le(0.0).mean()), 8),
        "severeLossRate": round(float(stacked.le(severe).mean()), 8),
        "dailyCvar10": round(float(daily.nsmallest(tail).mean() - cost), 8),
    }


def fit_all_heads(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    training = config["weightTraining"]
    cost = float(config["data"]["roundTripCost"])
    severe = float(config["data"]["severeLossThreshold"])
    targets = {
        "returnPercentile": return_percentile_target(returns),
        "netNonLoss": returns.gt(cost).astype(float).where(returns.notna()),
        "severeNonLoss": returns.gt(severe).astype(float).where(returns.notna()),
    }
    fitted: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    for name, target in targets.items():
        fitted[name], audit[name] = fit_bounded_head_weights(
            ranks,
            target,
            dates,
            float(training["minimumFactorWeight"]),
            float(training["maximumFactorWeight"]),
            float(training["l2ShrinkageToEqualWeight"]),
            int(config["data"]["minimumCrossSectionRows"]),
        )
    return fitted, audit


def stack_score_outcome(
    score: pd.DataFrame, returns: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    score_part = score.reindex(index=dates).stack(future_stack=True).rename("score")
    return_part = returns.reindex(index=dates).stack(future_stack=True).rename("return")
    rows = pd.concat([score_part, return_part], axis=1).dropna().reset_index()
    rows = rows.rename(columns={rows.columns[0]: "date", rows.columns[1]: "securityId"})
    counts = rows.groupby("date")["securityId"].transform("count").clip(lower=1)
    rows["sampleWeight"] = 1.0 / counts
    return rows


def fit_calibrators(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = stack_score_outcome(score, returns, dates)
    if len(rows) < int(config["calibration"]["minimumCalibrationRows"]):
        raise RuntimeError("independent calibration block has too few rows")
    x = rows[["score"]].to_numpy(dtype=float)
    y = rows["return"].to_numpy(dtype=float)
    sample_weight = rows["sampleWeight"].to_numpy(dtype=float)
    return_model = Ridge(alpha=float(config["calibration"]["expectedReturnRidgeAlpha"]))
    return_model.fit(x, y, sample_weight=sample_weight)
    cost = float(config["data"]["roundTripCost"])
    severe = float(config["data"]["severeLossThreshold"])
    net_loss = y <= cost
    severe_loss = y <= severe
    net_model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    severe_model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    net_model.fit(rows["score"], net_loss.astype(float), sample_weight=sample_weight)
    severe_model.fit(rows["score"], severe_loss.astype(float), sample_weight=sample_weight)
    return {
        "returnModel": return_model,
        "netLossModel": net_model,
        "severeLossModel": severe_model,
        "rows": len(rows),
        "days": int(rows["date"].nunique()),
        "returnPrior": float(np.average(y, weights=sample_weight)),
        "netLossPrior": float(np.average(net_loss, weights=sample_weight)),
        "severeLossPrior": float(np.average(severe_loss, weights=sample_weight)),
    }


def predict_frames(score: pd.DataFrame, calibrators: dict[str, Any], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    values = score.to_numpy(dtype=float)
    finite = np.isfinite(values)
    flat = values[finite]
    expected = np.full_like(values, np.nan, dtype=float)
    net_loss = np.full_like(values, np.nan, dtype=float)
    severe_loss = np.full_like(values, np.nan, dtype=float)
    if len(flat):
        expected[finite] = calibrators["returnModel"].predict(flat.reshape(-1, 1))
        net_loss[finite] = calibrators["netLossModel"].predict(flat)
        severe_loss[finite] = calibrators["severeLossModel"].predict(flat)
    expected_frame = pd.DataFrame(expected, index=score.index, columns=score.columns)
    net_frame = pd.DataFrame(net_loss, index=score.index, columns=score.columns).clip(0.0, 1.0)
    severe_frame = pd.DataFrame(severe_loss, index=score.index, columns=score.columns).clip(0.0, 1.0)
    utility = (
        expected_frame
        - float(config["calibration"]["utilityPenaltyNetLoss"]) * net_frame
        - float(config["calibration"]["utilityPenaltySevereLoss"]) * severe_frame
    )
    return {
        "expectedReturn": expected_frame,
        "netLossProbability": net_frame,
        "severeLossProbability": severe_frame,
        "utility": utility,
    }


def probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(probability)
    y = actual[valid].astype(int)
    p = np.clip(probability[valid].astype(float), 1e-8, 1.0 - 1e-8)
    if len(y) == 0:
        return {"n": 0, "events": 0, "brier": None, "logLoss": None, "auc": None, "ece": None}
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        member = bins == index
        if member.any():
            ece += float(member.mean()) * abs(float(y[member].mean()) - float(p[member].mean()))
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "eventRate": round(float(y.mean()), 8),
        "meanProbability": round(float(p.mean()), 8),
        "brier": round(float(brier_score_loss(y, p)), 8),
        "logLoss": round(float(log_loss(y, p, labels=[0, 1])), 8),
        "auc": round(auc, 8) if auc is not None else None,
        "ece": round(float(ece), 8),
    }


def period_evaluation(
    returns: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = stack_score_outcome(predictions["utility"], returns, dates)
    index = pd.MultiIndex.from_frame(rows[["date", "securityId"]])
    net_prob = predictions["netLossProbability"].stack(future_stack=True).reindex(index).to_numpy(dtype=float)
    severe_prob = predictions["severeLossProbability"].stack(future_stack=True).reindex(index).to_numpy(dtype=float)
    actual = rows["return"].to_numpy(dtype=float)
    cost = float(config["data"]["roundTripCost"])
    severe = float(config["data"]["severeLossThreshold"])
    output = {
        "rows": len(rows),
        "probability": {
            "netLoss": probability_metrics((actual <= cost).astype(float), net_prob),
            "severeLoss": probability_metrics((actual <= severe).astype(float), severe_prob),
        },
        "top": {},
    }
    for top_count in map(int, config["data"]["topCounts"]):
        output["top"][str(top_count)] = outcome_metrics(
            returns,
            predictions["utility"],
            dates,
            top_count,
            cost,
            severe,
        )
    return output


def baseline_prior_metrics(actual: np.ndarray, prior: float) -> dict[str, Any]:
    probability = np.full(len(actual), float(prior), dtype=float)
    return probability_metrics(actual.astype(float), probability)


def reliability_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    audit = report["periods"]["audit"]
    gate = config["evaluation"]
    checks: dict[str, bool] = {}
    for head in ("netLoss", "severeLoss"):
        metrics = audit["probability"][head]
        prior = report["calibrationAudit"]["constantPriorMetrics"][head]
        checks[f"{head}Auc"] = metrics["auc"] is not None and metrics["auc"] >= float(gate["minimumAuditAuc"])
        checks[f"{head}Brier"] = metrics["brier"] is not None and prior["brier"] is not None and metrics["brier"] < prior["brier"]
    top10 = audit["top"]["10"]
    checks["auditTop10NetReturn"] = top10["meanNetReturn"] is not None and top10["meanNetReturn"] > 0.0
    checks["auditTop10NetWin"] = top10["netWinRate"] is not None and top10["netWinRate"] >= float(gate["requireAuditTop10NetWinRate"])
    audit_pass = all(checks.values())
    external = {}
    for name in ("validation", "shadow"):
        row = report["periods"][name]["top"]["10"]
        external[name] = {
            "meanNetPositive": row["meanNetReturn"] is not None and row["meanNetReturn"] > 0.0,
            "netWinAtLeast55": row["netWinRate"] is not None and row["netWinRate"] >= 0.55,
        }
        external[name]["passed"] = all(external[name].values())
    stable = bool(audit_pass and all(item["passed"] for item in external.values()))
    return {
        "status": "research_only_not_eligible_for_trading",
        "auditChecks": checks,
        "auditPassed": audit_pass,
        "externalRejectOnly": external,
        "stableHistoricalHypothesis": stable,
        "decision": (
            "fresh_forward_preregistration_still_required"
            if stable
            else "reject_for_trading_keep_diagnostics"
        ),
        "historicalRunCanPromote": False,
    }


def _repair_name(value: Any) -> str:
    text = str(value or "")
    try:
        if any(marker in text for marker in ("Ã", "æ", "ç", "è", "å", "é", "ä")):
            return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def name_map(base: dict[str, Any]) -> dict[str, str]:
    path = ROOT / base["assetUniverse"]["masterPath"]
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        output[str(row.get("securityId"))] = _repair_name(row.get("name"))
    return output


def latest_rows(
    score: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    ranks: dict[str, pd.DataFrame],
    names: dict[str, str],
) -> pd.DataFrame:
    date = score.index.max()
    frame = pd.DataFrame(
        {
            "signalDate": date.date().isoformat(),
            "securityId": score.columns,
            "name": [names.get(str(item), "") for item in score.columns],
            "factorScore": score.loc[date].to_numpy(dtype=float),
            "expectedExecutableReturn": predictions["expectedReturn"].loc[date].to_numpy(dtype=float),
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
        "# Twelve-factor return/loss utility weights V6",
        "",
        "**Research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Latest signal date: `{report['latestSignalDate']}`",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Orders: `[]`",
        "",
        "## Frozen factor weights learned on base-fit dates only",
        "",
        "| Factor | Direction | Return head | Net-nonloss head | Severe-nonloss head | Final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["weights"]:
        lines.append(
            f"| {item['factorKey']} | {item['direction']} | {item['returnPercentile']:.4%} | "
            f"{item['netNonLoss']:.4%} | {item['severeNonLoss']:.4%} | {item['finalWeight']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Top10 diagnostics",
            "",
            "| Period | Days | Mean gross | Mean net | Net win | Non-positive | Severe loss | CVaR10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("innerWalkForward", "audit", "validation", "shadow"):
        if name == "innerWalkForward":
            metrics = report["innerWalkForwardAggregate"]
        else:
            metrics = report["periods"][name]["top"]["10"]
        lines.append(
            f"| {name} | {metrics.get('signalDays')} | {metrics.get('meanGrossReturn')} | "
            f"{metrics.get('meanNetReturn')} | {metrics.get('netWinRate')} | "
            f"{metrics.get('nonPositiveRate')} | {metrics.get('severeLossRate')} | "
            f"{metrics.get('dailyCvar10')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Final weights combine 55% expected-return rank, 30% net-nonloss and 15% severe-nonloss heads.",
            "- Every factor is bounded to 2%-20%; no factor is silently dropped or allowed to dominate.",
            "- Weight fitting, probability calibration and reliability audit use disjoint chronological blocks with a six-session purge.",
            "- Validation and shadow are already viewed and can only reject; even a pass still requires fresh forward evidence.",
            "- Top10/Top3/Top1 files are diagnostics, never orders.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    factors, source, source_sha = source_factors(config)
    base = load_json(ROOT / config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    ranks, factor_audit = compute_factor_ranks(panel, factors)
    returns, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        int(config["data"]["holdingTradingDays"]),
        int(config["data"]["maximumExitDelayTradingDays"]),
    )
    returns = returns.where(execution_eligible)
    split = autonomous.make_split(panel["close"].index, cog_config)
    train_dates = pd.DatetimeIndex(split.train[split.train].index)
    validation_dates = pd.DatetimeIndex(split.validation[split.validation].index)
    shadow_dates = pd.DatetimeIndex(split.shadow[split.shadow].index)
    partitions = training_partitions(train_dates, config)
    fitted_heads, fit_audit = fit_all_heads(
        ranks, returns, partitions["baseFit"], config
    )
    final_weights = combine_head_weights(
        fitted_heads, config["weightTraining"]["objectiveHeads"]
    )
    score = weighted_score(ranks, final_weights).where(panel["eligible"])
    print("full_base_weight_fit_complete=true", flush=True)

    folds = expanding_folds(partitions["baseFit"], config)
    fold_rows: list[dict[str, Any]] = []
    fold_weight_rows: list[np.ndarray] = []
    for index, (fold_train, fold_test) in enumerate(folds, start=1):
        fold_heads, _ = fit_all_heads(ranks, returns, fold_train, config)
        fold_weights = combine_head_weights(
            fold_heads, config["weightTraining"]["objectiveHeads"]
        )
        fold_score = weighted_score(ranks, fold_weights)
        metrics = outcome_metrics(
            returns,
            fold_score,
            fold_test,
            int(config["data"]["primaryTopCount"]),
            float(config["data"]["roundTripCost"]),
            float(config["data"]["severeLossThreshold"]),
        )
        fold_rows.append(
            {
                "fold": index,
                "trainRange": [fold_train.min().date().isoformat(), fold_train.max().date().isoformat()],
                "testRange": [fold_test.min().date().isoformat(), fold_test.max().date().isoformat()],
                "metrics": metrics,
            }
        )
        fold_weight_rows.append(fold_weights)
        print(f"inner_fold_complete {index}/{len(folds)}", flush=True)

    calibrators = fit_calibrators(score, returns, partitions["calibration"], config)
    predictions = predict_frames(score, calibrators, config)
    periods = {
        "audit": period_evaluation(returns, predictions, partitions["audit"], config),
        "validation": period_evaluation(returns, predictions, validation_dates, config),
        "shadow": period_evaluation(returns, predictions, shadow_dates, config),
    }
    audit_rows = stack_score_outcome(score, returns, partitions["audit"])
    audit_actual = audit_rows["return"].to_numpy(dtype=float)
    calibration_audit = {
        "fitRows": calibrators["rows"],
        "fitDays": calibrators["days"],
        "constantPriorMetrics": {
            "netLoss": baseline_prior_metrics(
                (audit_actual <= float(config["data"]["roundTripCost"])).astype(float),
                calibrators["netLossPrior"],
            ),
            "severeLoss": baseline_prior_metrics(
                (audit_actual <= float(config["data"]["severeLossThreshold"])).astype(float),
                calibrators["severeLossPrior"],
            ),
        },
    }
    keys = list(ranks)
    weights = []
    for position, item in enumerate(factors):
        weights.append(
            {
                "factorKey": item["factorKey"],
                "direction": item["direction"],
                "returnPercentile": float(fitted_heads["returnPercentile"][position]),
                "netNonLoss": float(fitted_heads["netNonLoss"][position]),
                "severeNonLoss": float(fitted_heads["severeNonLoss"][position]),
                "finalWeight": float(final_weights[position]),
                "innerFoldMinimum": float(np.min([row[position] for row in fold_weight_rows])),
                "innerFoldMaximum": float(np.max([row[position] for row in fold_weight_rows])),
            }
        )
    fold_metrics = [row["metrics"] for row in fold_rows]
    total_obs = sum(int(row["observations"]) for row in fold_metrics)
    total_days = sum(int(row["signalDays"]) for row in fold_metrics)
    inner_aggregate = {
        "observations": total_obs,
        "signalDays": total_days,
        "meanGrossReturn": (
            round(sum(row["meanGrossReturn"] * row["observations"] for row in fold_metrics) / total_obs, 8)
            if total_obs
            else None
        ),
        "meanNetReturn": (
            round(sum(row["meanNetReturn"] * row["observations"] for row in fold_metrics) / total_obs, 8)
            if total_obs
            else None
        ),
        "netWinRate": (
            round(sum(row["netWinRate"] * row["observations"] for row in fold_metrics) / total_obs, 8)
            if total_obs
            else None
        ),
        "nonPositiveRate": (
            round(sum(row["nonPositiveRate"] * row["observations"] for row in fold_metrics) / total_obs, 8)
            if total_obs
            else None
        ),
        "severeLossRate": (
            round(sum(row["severeLossRate"] * row["observations"] for row in fold_metrics) / total_obs, 8)
            if total_obs
            else None
        ),
        "dailyCvar10": None,
    }
    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_" + now.strftime("%Y%m%dT%H%M%SZ") + "_" + digest(config)[:10]
    )
    latest = latest_rows(score, predictions, ranks, name_map(base))
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": now.isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": digest(config),
        "sourceSummarySha256": source_sha,
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "latestSignalDate": latest.iloc[0]["signalDate"] if not latest.empty else None,
        "splitAudit": split.audit,
        "trainingPartitions": {
            name: [dates.min().date().isoformat(), dates.max().date().isoformat(), len(dates)]
            for name, dates in partitions.items()
        },
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "fitAudit": fit_audit,
        "weights": weights,
        "innerWalkForward": fold_rows,
        "innerWalkForwardAggregate": inner_aggregate,
        "calibrationAudit": calibration_audit,
        "periods": periods,
        "exitDelay": {
            "maximumTradingDays": int(config["data"]["maximumExitDelayTradingDays"]),
            "resolvedRows": int(exit_delay.notna().to_numpy().sum()),
        },
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    report["verdict"] = reliability_verdict(report, config)
    output = ROOT / config["output"]["root"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(
        output / "summary.json",
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write(output / "report.md", markdown_report(report))
    pd.DataFrame(weights).to_csv(output / "trained_weights.csv", index=False, encoding="utf-8-sig")
    for count in (10, 3, 1):
        latest.head(count).to_csv(
            output / f"latest_diagnostic_top{count}.csv", index=False, encoding="utf-8-sig"
        )
    latest.iloc[0:0].to_csv(output / "qualified_selections.csv", index=False, encoding="utf-8-sig")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    validate_config(load_json(DEFAULT_CONFIG))
    fitted = {
        "returnPercentile": np.array([0.7, 0.3]),
        "netNonLoss": np.array([0.2, 0.8]),
        "severeNonLoss": np.array([0.5, 0.5]),
    }
    combined = combine_head_weights(
        fitted, {"returnPercentile": 0.55, "netNonLoss": 0.30, "severeNonLoss": 0.15}
    )
    assert np.allclose(combined, [0.52, 0.48])
    print("self_test_passed=true")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    report = run(args.config, args.run_id)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runId": report["runId"],
                "latestSignalDate": report["latestSignalDate"],
                "verdict": report["verdict"]["decision"],
                "orders": report["orders"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
