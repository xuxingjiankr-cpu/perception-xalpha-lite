#!/usr/bin/env python3
"""Frozen nonlinear and learning-to-rank audit for the twelve-factor Top10.

The script is historical research only.  It does not publish the dashboard,
produce orders, mutate a trading configuration, or promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_guarded_weight_top10_forecast_v1 as forecast  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_rolling_health_v4 as rolling  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as discrimination  # noqa: E402
import overfitting_guard  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_rank_discrimination_v1.json"
)
SCHEMA_VERSION = "twelve_factor_rank_discrimination_result_v1"
CODE_VERSION = "twelve_factor_rank_discrimination_v1_20260812"

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names, but LGBM.* was fitted"
)
warnings.filterwarnings("ignore", message="Found 'eval_at' in params.*")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_rank_discrimination_v1":
        raise ValueError("unexpected rank-discrimination schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("rank discrimination must remain research-only")
    for path_key, hash_key in (
        ("sourceForecastConfig", "sourceForecastConfigSha256"),
        ("partitionTemplate", "partitionTemplateSha256"),
    ):
        if sha256(ROOT / config[path_key]) != str(config[hash_key]).lower():
            raise ValueError(f"frozen dependency changed: {path_key}")
    data = config["data"]
    if int(data["factorCount"]) != 12:
        raise ValueError("the audit must use exactly twelve factors")
    if int(data["maximumImputedFactorsForEligibility"]) > 2:
        raise ValueError("factor completion support was widened")
    models = config["models"]
    if models["candidates"] != ["linear", "nonlinear", "lambdarank", "hybrid"]:
        raise ValueError("candidate family or ordering changed")
    if models.get("hyperparameterSearchAllowed") is not False:
        raise ValueError("historical hyperparameter search is forbidden")
    if models["calibration"].get("mechanicalProbabilityStretchingAllowed") is not False:
        raise ValueError("mechanical probability stretching is forbidden")
    evaluation = config["evaluation"]
    if evaluation.get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical reuse must be explicit")
    if int(evaluation["trialCount"]) != len(models["candidates"]):
        raise ValueError("trial count must include every candidate")
    if not 0 <= int(evaluation["pairedDailyGrossHacLag"]) <= 10:
        raise ValueError("paired daily HAC lag must remain bounded")
    if not 0.0 <= float(evaluation["pboMaximumForForwardStudy"]) < 0.5:
        raise ValueError("PBO gate must be stricter than chance")
    if not 0.0 < float(evaluation["deflatedSharpeAlpha"]) <= 0.10:
        raise ValueError("DSR alpha must remain conservative")
    if evaluation.get("supportMustBeIdentical") is not True:
        raise ValueError("all candidates must share identical rows")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")
    if any(
        bool(value)
        for key, value in config.get("safety", {}).items()
        if key.startswith("may")
    ):
        raise ValueError("all trading permissions must remain false")


def relevance_labels(values: np.ndarray, levels: int) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    return np.clip((ranks * levels - 1e-12).astype(int), 0, levels - 1)


def stack_rows(
    ranks: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    baseline_score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    maximum_rows_per_day: int,
    seed: int,
    minimum_cross_section: int,
    relevance_levels: int,
) -> dict[str, Any]:
    keys = list(ranks)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    relevance: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[int] = []
    used_dates: list[pd.Timestamp] = []
    baseline_scores: list[np.ndarray] = []
    for date in dates:
        x = np.column_stack([ranks[key].loc[date].to_numpy(float) for key in keys])
        y = outcome.loc[date].to_numpy(float)
        baseline = baseline_score.loc[date].to_numpy(float)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1) & np.isfinite(baseline)
        xv, yv, baseline_valid = x[valid], y[valid], baseline[valid]
        if len(yv) < minimum_cross_section:
            continue
        chosen = discrimination.deterministic_indices(
            len(yv), maximum_rows_per_day, seed, date
        )
        xv, yv = xv[chosen], yv[chosen]
        baseline_valid = baseline_valid[chosen]
        xs.append(xv)
        ys.append(yv)
        relevance.append(relevance_labels(yv, relevance_levels))
        weights.append(np.full(len(yv), 1.0 / len(yv), dtype=float))
        groups.append(len(yv))
        used_dates.append(date)
        baseline_scores.append(baseline_valid)
    if not xs:
        raise RuntimeError("no common rows for rank-discrimination fit")
    return {
        "x": np.vstack(xs),
        "y": np.concatenate(ys),
        "relevance": np.concatenate(relevance),
        "weights": np.concatenate(weights),
        "groups": groups,
        "dates": used_dates,
        "baselineScore": np.concatenate(baseline_scores),
    }


@dataclass
class ScoreModels:
    nonlinear: lgb.LGBMRegressor
    ranker: lgb.LGBMRanker
    nonlinear_mean: float
    nonlinear_std: float
    ranker_mean: float
    ranker_std: float
    hybrid_nonlinear_weight: float
    hybrid_ranker_weight: float

    def predict(
        self, x: np.ndarray, baseline_score: np.ndarray
    ) -> dict[str, np.ndarray]:
        linear = np.asarray(baseline_score, dtype=float)
        nonlinear = self.nonlinear.predict(x)
        ranker = self.ranker.predict(x)
        hybrid = (
            self.hybrid_nonlinear_weight
            * (nonlinear - self.nonlinear_mean)
            / self.nonlinear_std
            + self.hybrid_ranker_weight
            * (ranker - self.ranker_mean)
            / self.ranker_std
        )
        return {
            "linear": np.asarray(linear, dtype=float),
            "nonlinear": np.asarray(nonlinear, dtype=float),
            "lambdarank": np.asarray(ranker, dtype=float),
            "hybrid": np.asarray(hybrid, dtype=float),
        }


def fit_score_models(table: dict[str, Any], config: dict[str, Any]) -> tuple[ScoreModels, dict[str, Any]]:
    spec = config["models"]
    seed = int(config["data"]["deterministicSeed"])
    x = table["x"]
    y = table["y"]
    weights = table["weights"]
    tree = spec["nonlinear"]
    nonlinear = lgb.LGBMRegressor(
        objective="regression_l2",
        num_leaves=int(tree["numLeaves"]),
        learning_rate=float(tree["learningRate"]),
        n_estimators=int(tree["nEstimators"]),
        min_child_samples=int(tree["minChildSamples"]),
        reg_lambda=float(tree["regLambda"]),
        feature_fraction=float(tree["featureFraction"]),
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
    )
    nonlinear.fit(x, y, sample_weight=weights)
    rank = spec["lambdarank"]
    ranker = lgb.LGBMRanker(
        objective=str(rank["objective"]),
        label_gain=list(map(int, rank["labelGain"])),
        eval_at=list(map(int, rank["evalAt"])),
        num_leaves=int(rank["numLeaves"]),
        learning_rate=float(rank["learningRate"]),
        n_estimators=int(rank["nEstimators"]),
        min_child_samples=int(rank["minChildSamples"]),
        reg_lambda=float(rank["regLambda"]),
        feature_fraction=float(rank["featureFraction"]),
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
    )
    ranker.fit(
        x,
        table["relevance"],
        group=table["groups"],
        sample_weight=weights,
    )
    nonlinear_fit = np.asarray(nonlinear.predict(x), dtype=float)
    ranker_fit = np.asarray(ranker.predict(x), dtype=float)

    def scale(values: np.ndarray) -> tuple[float, float]:
        mean = float(np.average(values, weights=weights))
        variance = float(np.average(np.square(values - mean), weights=weights))
        std = math.sqrt(max(variance, 0.0))
        if std < 1e-12:
            raise RuntimeError("candidate score collapsed during base fit")
        return mean, std

    nonlinear_mean, nonlinear_std = scale(nonlinear_fit)
    ranker_mean, ranker_std = scale(ranker_fit)
    hybrid = spec["hybrid"]
    models = ScoreModels(
        nonlinear=nonlinear,
        ranker=ranker,
        nonlinear_mean=nonlinear_mean,
        nonlinear_std=nonlinear_std,
        ranker_mean=ranker_mean,
        ranker_std=ranker_std,
        hybrid_nonlinear_weight=float(hybrid["nonlinearWeight"]),
        hybrid_ranker_weight=float(hybrid["lambdaRankWeight"]),
    )
    audit = {
        "rows": int(len(y)),
        "days": int(len(table["groups"])),
        "groupsAreContiguousTradingDates": True,
        "equalWeightPerTradingDay": True,
        "hyperparameterSearchPerformed": False,
        "linearBaseline": str(spec["linear"]["definition"]),
        "nonlinearFeatureImportances": nonlinear.feature_importances_.tolist(),
        "lambdaRankFeatureImportances": ranker.feature_importances_.tolist(),
    }
    return models, audit


@dataclass
class ScoreCalibrator:
    mean: float
    std: float
    expected_return: Ridge
    gross_up: LogisticRegression
    severe_loss: LogisticRegression

    def predict(self, score: np.ndarray) -> dict[str, np.ndarray]:
        z = ((np.asarray(score, dtype=float) - self.mean) / self.std).reshape(-1, 1)
        return {
            "expectedReturn": self.expected_return.predict(z),
            "grossUp": self.gross_up.predict_proba(z)[:, 1],
            "severeLoss": self.severe_loss.predict_proba(z)[:, 1],
        }


def fit_calibrators(
    models: ScoreModels,
    table: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, ScoreCalibrator], dict[str, Any]]:
    scores = models.predict(table["x"], table["baselineScore"])
    y = table["y"]
    weights = table["weights"]
    spec = config["models"]["calibration"]
    result: dict[str, ScoreCalibrator] = {}
    audit: dict[str, Any] = {}
    for name, values in scores.items():
        mean = float(np.average(values, weights=weights))
        variance = float(np.average(np.square(values - mean), weights=weights))
        std = math.sqrt(max(variance, 0.0))
        if std < 1e-12:
            raise RuntimeError(f"{name} score collapsed in calibration")
        z = ((values - mean) / std).reshape(-1, 1)
        expected = Ridge(alpha=float(spec["returnRidgeAlpha"]))
        expected.fit(z, y, sample_weight=weights)

        def logistic(target: np.ndarray) -> LogisticRegression:
            model = LogisticRegression(
                C=float(spec["probabilityLogisticC"]),
                solver="lbfgs",
                max_iter=1000,
                random_state=int(config["data"]["deterministicSeed"]),
            )
            model.fit(z, target.astype(int), sample_weight=weights)
            return model

        up = logistic(y > 0.0)
        tail = logistic(y <= float(config["data"]["severeLossThreshold"]))
        result[name] = ScoreCalibrator(mean, std, expected, up, tail)
        audit[name] = {
            "scoreMean": mean,
            "scoreStd": std,
            "expectedReturnCoefficient": float(np.ravel(expected.coef_)[0]),
            "grossUpCoefficient": float(np.ravel(up.coef_)[0]),
            "severeLossCoefficient": float(np.ravel(tail.coef_)[0]),
        }
    return result, audit


def probability_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for value in range(bins):
        member = bucket == value
        if member.any():
            ece += float(member.mean()) * abs(float(p[member].mean()) - float(y[member].mean()))
    return {
        "n": int(len(y)),
        "eventRate": float(y.mean()),
        "meanProbability": float(p.mean()),
        "minimumProbability": float(p.min()),
        "maximumProbability": float(p.max()),
        "probabilitySpread": float(p.max() - p.min()),
        "brier": float(brier_score_loss(y, p)),
        "logLoss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ece": float(ece),
    }


def date_arrays(
    ranks: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    baseline_score: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.column_stack([frame.loc[date].to_numpy(float) for frame in ranks.values()])
    y = outcome.loc[date].to_numpy(float)
    baseline = baseline_score.loc[date].to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1) & np.isfinite(baseline)
    return x[valid], y[valid], baseline[valid]


def evaluate_period(
    models: ScoreModels,
    calibrators: dict[str, ScoreCalibrator],
    ranks: dict[str, pd.DataFrame],
    outcome: pd.DataFrame,
    baseline_score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = list(config["models"]["candidates"])
    accumulator: dict[str, dict[str, list[Any]]] = {
        name: {
            "y": [],
            "pUp": [],
            "pTail": [],
            "expected": [],
            "dailyGross": [],
            "dailyStockWin": [],
            "dailyTail": [],
            "dailyRankIc": [],
            "dailyNdcg10": [],
            "dailyTop10PUpSpread": [],
        }
        for name in candidates
    }
    used_dates: list[pd.Timestamp] = []
    minimum = int(config["data"]["minimumCrossSection"])
    for date in dates:
        x, y, baseline = date_arrays(ranks, outcome, baseline_score, date)
        if len(y) < minimum:
            continue
        used_dates.append(date)
        score_map = models.predict(x, baseline)
        relevance = relevance_labels(y, 100).astype(float)
        for name in candidates:
            score = score_map[name]
            predictions = calibrators[name].predict(score)
            target = accumulator[name]
            target["y"].extend(y.tolist())
            target["pUp"].extend(predictions["grossUp"].tolist())
            target["pTail"].extend(predictions["severeLoss"].tolist())
            target["expected"].extend(predictions["expectedReturn"].tolist())
            top = np.argsort(score, kind="stable")[-10:]
            top_return = y[top]
            target["dailyGross"].append(float(top_return.mean()))
            target["dailyStockWin"].append(float((top_return > 0.0).mean()))
            target["dailyTail"].append(
                float(
                    (top_return <= float(config["data"]["severeLossThreshold"])).mean()
                )
            )
            target["dailyRankIc"].append(
                float(pd.Series(score).corr(pd.Series(y), method="spearman"))
            )
            target["dailyNdcg10"].append(
                float(ndcg_score(relevance.reshape(1, -1), score.reshape(1, -1), k=10))
            )
            target["dailyTop10PUpSpread"].append(
                float(np.ptp(predictions["grossUp"][top]))
            )
    metrics: dict[str, Any] = {}
    daily = pd.DataFrame(index=pd.DatetimeIndex(used_dates))
    for name in candidates:
        values = accumulator[name]
        y = np.asarray(values["y"], dtype=float)
        gross = np.asarray(values["dailyGross"], dtype=float)
        daily[name] = gross
        metrics[name] = {
            "tradingDays": int(len(gross)),
            "stockObservations": int(len(y)),
            "grossUp": probability_metrics(y > 0.0, np.asarray(values["pUp"])),
            "severeLoss": probability_metrics(
                y <= float(config["data"]["severeLossThreshold"]),
                np.asarray(values["pTail"]),
            ),
            "top10MeanGrossReturn": float(gross.mean()),
            "top10MeanNetReturn": float(
                gross.mean() - float(config["data"]["roundTripCost"])
            ),
            "top10DailyWinRate": float((gross > 0.0).mean()),
            "top10StockWinRate": float(np.mean(values["dailyStockWin"])),
            "top10SevereLossRate": float(np.mean(values["dailyTail"])),
            "meanDailyRankIc": float(np.nanmean(values["dailyRankIc"])),
            "meanDailyNdcgAt10": float(np.nanmean(values["dailyNdcg10"])),
            "meanDailyTop10ProbabilityUpSpread": float(
                np.mean(values["dailyTop10PUpSpread"])
            ),
            "expectedReturnMae": float(
                np.mean(np.abs(np.asarray(values["expected"]) - y))
            ),
        }
    return metrics, daily


def newey_west_t(values: np.ndarray, lag: int) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return None
    centered = values - values.mean()
    n = len(centered)
    long_run_variance = float(np.dot(centered, centered) / n)
    for offset in range(1, min(int(lag), n - 1) + 1):
        weight = 1.0 - offset / (int(lag) + 1.0)
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0.0 or not math.isfinite(long_run_variance):
        return None
    return float(values.mean() / math.sqrt(long_run_variance / n))


def paired_hac_t(
    candidate: pd.Series, baseline: pd.Series, lag: int
) -> float | None:
    difference = (candidate - baseline).dropna().to_numpy(float)
    return newey_west_t(difference, lag)


def acceptance(
    periods: dict[str, Any], daily: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    minimum = int(config["evaluation"]["minimumMetricsImprovedPerPeriod"])
    t_minimum = float(config["evaluation"]["pairedDailyGrossHacTMinimum"])
    hac_lag = int(config["evaluation"]["pairedDailyGrossHacLag"])
    for candidate in ("nonlinear", "lambdarank", "hybrid"):
        checks: dict[str, Any] = {}
        for period in ("validation", "shadow"):
            baseline = periods[period]["linear"]
            model = periods[period][candidate]
            improvements = {
                "top10MeanGrossReturn": model["top10MeanGrossReturn"]
                > baseline["top10MeanGrossReturn"],
                "top10StockWinRate": model["top10StockWinRate"]
                > baseline["top10StockWinRate"],
                "meanDailyRankIc": model["meanDailyRankIc"]
                > baseline["meanDailyRankIc"],
                "grossUpAuc": model["grossUp"]["auc"] > baseline["grossUp"]["auc"],
                "grossUpBrier": model["grossUp"]["brier"] < baseline["grossUp"]["brier"],
            }
            hac_t = paired_hac_t(
                daily[period][candidate], daily[period]["linear"], hac_lag
            )
            checks[period] = {
                "improvements": improvements,
                "metricsImproved": int(sum(improvements.values())),
                "pairedDailyGrossHacT": hac_t,
                "passed": int(sum(improvements.values())) >= minimum
                and improvements["top10MeanGrossReturn"]
                and hac_t is not None
                and hac_t >= t_minimum,
            }
        result[candidate] = {
            "periods": checks,
            "historicalGatePassed": all(item["passed"] for item in checks.values()),
            "eligibleForTrading": False,
        }
    return result


def selection_exposure(
    periods: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    candidates = list(config["models"]["candidates"])
    output: dict[str, Any] = {}
    for prior, current in (("audit", "validation"), ("validation", "shadow")):
        trailing_choice = max(
            candidates, key=lambda name: periods[prior][name]["top10MeanNetReturn"]
        )
        hindsight_choice = max(
            candidates, key=lambda name: periods[current][name]["top10MeanNetReturn"]
        )
        trailing = periods[current][trailing_choice]["top10MeanNetReturn"]
        hindsight = periods[current][hindsight_choice]["top10MeanNetReturn"]
        output[current] = {
            "selectionPeriod": prior,
            "trailingChoice": trailing_choice,
            "trailingSelectedNetReturn": trailing,
            "hindsightChoice": hindsight_choice,
            "hindsightNetReturn": hindsight,
            "overfittingExposure": hindsight - trailing,
        }
    return output


def deflated_sharpe_difference(
    candidate: pd.Series,
    baseline: pd.Series,
    n_trials: int,
    alpha: float,
) -> dict[str, Any]:
    difference = (candidate - baseline).dropna().to_numpy(float)
    result: dict[str, Any] = {
        "method": "approx_deflated_sharpe_on_daily_return_difference",
        "nDays": int(len(difference)),
        "nTrials": int(n_trials),
        "alpha": float(alpha),
        "significant": False,
    }
    if len(difference) < 8:
        result["reason"] = "insufficient_shared_days"
        return result
    mean = float(difference.mean())
    std = float(difference.std(ddof=1))
    if std <= 1e-12:
        result.update(
            {
                "reason": "zero_variance",
                "meanDifference": mean,
                "stdDifference": std,
                "pValue": 0.0 if mean > 0.0 else 1.0,
                "significant": mean > 0.0,
            }
        )
        return result
    daily_sharpe = mean / std
    t_stat = daily_sharpe * math.sqrt(len(difference))
    selection_bias_t = NormalDist().inv_cdf(1.0 - 1.0 / max(2, n_trials))
    deflated_t = t_stat - selection_bias_t
    p_value = 1.0 - NormalDist().cdf(deflated_t)
    result.update(
        {
            "reason": "ok",
            "meanDifference": mean,
            "stdDifference": std,
            "dailySharpeDifference": daily_sharpe,
            "tStatistic": t_stat,
            "selectionBiasT": selection_bias_t,
            "deflatedT": deflated_t,
            "pValue": p_value,
            "significant": bool(mean > 0.0 and p_value < alpha),
        }
    )
    return result


def overfitting_audit(
    daily: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    candidates = list(config["models"]["candidates"])
    combined = pd.concat([daily["validation"], daily["shadow"]], axis=0)
    pbo = overfitting_guard.combinatorial_symmetric_pbo(
        [combined[name].astype(float).tolist() for name in candidates],
        n_blocks=8,
    )
    trials = int(config["evaluation"]["trialCount"])
    alpha = float(config["evaluation"]["deflatedSharpeAlpha"])
    dsr = {
        name: deflated_sharpe_difference(
            daily["shadow"][name], daily["shadow"]["linear"], trials, alpha
        )
        for name in candidates
        if name != "linear"
    }
    return {
        "pbo": pbo,
        "pboPassed": pbo.get("pbo") is not None
        and float(pbo["pbo"])
        <= float(config["evaluation"]["pboMaximumForForwardStudy"]),
        "deflatedSharpeOnShadowDifference": dsr,
        "allCandidatesCounted": True,
    }


def apply_overfitting_gates(
    gates: dict[str, Any], audit: dict[str, Any]
) -> list[str]:
    passing: list[str] = []
    for candidate, item in gates.items():
        pre = bool(item["historicalGatePassed"])
        dsr = bool(
            audit["deflatedSharpeOnShadowDifference"][candidate]["significant"]
        )
        item["preOverfittingGatePassed"] = pre
        item["pboPassed"] = bool(audit["pboPassed"])
        item["deflatedSharpePassed"] = dsr
        item["historicalGatePassed"] = pre and bool(audit["pboPassed"]) and dsr
        if item["historicalGatePassed"]:
            passing.append(candidate)
    return passing


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Twelve-Factor Rank Discrimination Audit",
        "",
        f"- Run: `{result['runId']}`",
        f"- Data: `{result['dataRange'][0]}` to `{result['dataRange'][1]}`",
        "- Status: `research_only / historical_windows_already_viewed`",
        "- Same 12 PIT factor ranks and identical rows for every model",
        "- No probability stretching; no dashboard or trading integration",
        "",
        "| Period | Model | Top10 gross | Top10 net@30bp | Stock win | Rank IC | Up AUC | Up Brier | P(up) spread |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, candidates in result["periods"].items():
        for name, item in candidates.items():
            lines.append(
                f"| {period} | {name} | {item['top10MeanGrossReturn']:.3%} | "
                f"{item['top10MeanNetReturn']:.3%} | {item['top10StockWinRate']:.2%} | "
                f"{item['meanDailyRankIc']:.4f} | {item['grossUp']['auc']:.4f} | "
                f"{item['grossUp']['brier']:.4f} | "
                f"{item['meanDailyTop10ProbabilityUpSpread']:.3%} |"
            )
    lines.extend(
        [
            "",
            f"Historical winner: `{result['verdict']['historicalWinner']}`",
            f"PBO: `{result.get('overfittingAudit', {}).get('pbo', {}).get('pbo')}`",
            f"Merits preregistered forward study: `{str(result['verdict']['meritsFreshForwardStudy']).lower()}`",
            "Eligible for trading: `false`",
            "",
            "A positive historical gate only permits a separately preregistered fresh-forward study. These windows have already been viewed and cannot authorize deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    effective_run_id = run_id or datetime.now().astimezone().strftime(
        "run_%Y%m%dT%H%M%S%z"
    )
    source_config = load_json(ROOT / config["sourceForecastConfig"])
    forecast.validate_config(source_config)
    guarded_config = load_json(ROOT / source_config["guardedWeightsConfig"])
    import research_twelve_factor_guarded_online_weights_v1 as guarded

    frozen, _source, _source_hash = guarded.load_frozen_config(guarded_config)
    partition_config = load_json(ROOT / config["partitionTemplate"])
    base = load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    outcome, execution_eligible, _delay = precision.executable_horizon_return(
        panel, 1, 5
    )
    ranks, _score, factor_audit = rolling.compute_rank_book(panel, frozen)
    completed, missing_count, support, completion_audit = forecast.complete_rank_book(
        ranks,
        panel["eligible"],
        int(config["data"]["maximumImputedFactorsForEligibility"]),
        float(config["data"]["neutralCompletionRank"]),
    )
    outcome = outcome.where(execution_eligible & support)
    daily_ic = rolling.factor_daily_ic(ranks, outcome)
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    )
    adaptive_weights, weight_updates = guarded.guarded_weight_path(
        daily_ic, prior, guarded_config
    )
    baseline_score = guarded.adaptive_score(
        completed, adaptive_weights, panel
    ).where(support)
    partitions = discrimination.fixed_partitions(panel["close"].index, partition_config)
    levels = int(config["models"]["lambdarank"]["relevanceLevels"])
    base_rows = stack_rows(
        completed,
        outcome,
        baseline_score,
        partitions["baseFit"],
        int(config["data"]["maximumBaseFitRowsPerDay"]),
        int(config["data"]["deterministicSeed"]),
        int(config["data"]["minimumCrossSection"]),
        levels,
    )
    models, model_audit = fit_score_models(base_rows, config)
    print(f"models_ready rows={len(base_rows['y'])}", flush=True)
    calibration_rows = stack_rows(
        completed,
        outcome,
        baseline_score,
        partitions["calibration"],
        int(config["data"]["maximumCalibrationRowsPerDay"]),
        int(config["data"]["deterministicSeed"]),
        int(config["data"]["minimumCrossSection"]),
        levels,
    )
    calibrators, calibration_audit = fit_calibrators(models, calibration_rows, config)
    periods: dict[str, Any] = {}
    daily: dict[str, pd.DataFrame] = {}
    for name in config["evaluation"]["periods"]:
        periods[name], daily[name] = evaluate_period(
            models,
            calibrators,
            completed,
            outcome,
            baseline_score,
            partitions[name],
            config,
        )
        print(f"period_ready {name} days={len(daily[name])}", flush=True)
    checkpoint_root = (
        ROOT
        / config["output"]["root"]
        / effective_run_id
    )
    precision.atomic_write(
        checkpoint_root / "evaluation_checkpoint.json",
        json.dumps(safe(periods), ensure_ascii=False, indent=2) + "\n",
    )
    precision.atomic_write(
        checkpoint_root / "daily_top10_gross_checkpoint.csv",
        pd.concat(daily, axis=1).to_csv(lineterminator="\n"),
    )
    gates = acceptance(periods, daily, config)
    overfitting = overfitting_audit(daily, config)
    passing = apply_overfitting_gates(gates, overfitting)
    winner = max(
        config["models"]["candidates"],
        key=lambda name: periods["shadow"][name]["top10MeanGrossReturn"],
    )
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_historical_windows_already_viewed",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "runId": effective_run_id,
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "factorCompletionAudit": completion_audit,
        "guardedWeightUpdateCount": int(len(weight_updates)),
        "latestFactorMissingCountDistribution": safe(
            missing_count.loc[missing_count.index.max()].value_counts().sort_index().to_dict()
        ),
        "modelFitAudit": model_audit,
        "calibrationAudit": calibration_audit,
        "periods": periods,
        "candidateAcceptance": gates,
        "overfittingAudit": overfitting,
        "selectionExposure": selection_exposure(periods, config),
        "verdict": {
            "historicalWinner": winner,
            "historicalGatePassingCandidates": passing,
            "meritsFreshForwardStudy": bool(passing),
            "historicalResultCanPromote": False,
            "eligibleForTrading": False,
            "reason": (
                "historical_gate_passed_but_requires_separate_fresh_forward_preregistration"
                if passing
                else "no_candidate_passed_locked_validation_and_shadow_gates"
            ),
        },
        "trialsCounted": int(config["evaluation"]["trialCount"]),
        "orders": [],
        "automaticTradingChanges": [],
    }
    root = ROOT / config["output"]["root"] / result["runId"]
    precision.atomic_write(
        root / "result.json", json.dumps(safe(result), ensure_ascii=False, indent=2) + "\n"
    )
    precision.atomic_write(root / "report.md", markdown(result))
    return result


def finalize_existing(
    run_root: Path, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    result_path = run_root / "result.json"
    result = load_json(result_path)
    combined = pd.read_csv(
        run_root / "daily_top10_gross_checkpoint.csv",
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )
    daily = {
        period: combined[period].astype(float)
        for period in config["evaluation"]["periods"]
    }
    audit = overfitting_audit(daily, config)
    result["overfittingAudit"] = audit
    passing = apply_overfitting_gates(result["candidateAcceptance"], audit)
    result["verdict"]["historicalGatePassingCandidates"] = passing
    result["verdict"]["meritsFreshForwardStudy"] = bool(passing)
    result["verdict"]["reason"] = (
        "historical_gate_passed_but_requires_separate_fresh_forward_preregistration"
        if passing
        else "no_candidate_passed_locked_validation_shadow_and_overfitting_gates"
    )
    precision.atomic_write(
        result_path, json.dumps(safe(result), ensure_ascii=False, indent=2) + "\n"
    )
    precision.atomic_write(run_root / "report.md", markdown(result))
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--finalize-existing", type=Path)
    args = parser.parse_args()
    result = (
        finalize_existing(args.finalize_existing.resolve(), args.config.resolve())
        if args.finalize_existing is not None
        else run(args.config.resolve(), args.run_id)
    )
    print(
        json.dumps(
            {
                "runId": result["runId"],
                "dataRange": result["dataRange"],
                "verdict": result["verdict"],
                "selectionExposure": result["selectionExposure"],
                "orders": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
