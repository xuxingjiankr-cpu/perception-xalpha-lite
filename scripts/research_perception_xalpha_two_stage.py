"""Causal two-stage A-share selector built on a frozen factor quartet.

The first stage only creates a daily top-50 candidate pool.  A frozen nonlinear
classifier/regressor pair then estimates absolute ten-session win probability and
expected return.  Everything in this module is historical research output: it has no
broker import, creates no orders and cannot mutate a trading configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_cogalpha_etf as core  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "perception_xalpha_two_stage_selector_v1.json"
)
SCHEMA_VERSION = "perception_xalpha_two_stage_selector_v1"
CODE_VERSION = "perception_xalpha_two_stage_v1.0"


@dataclass
class FrozenModel:
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    probability_calibrator: IsotonicRegression
    return_calibrator: Ridge
    feature_columns: list[str]
    prior_probability: float
    prior_return: float
    return_clip: tuple[float, float]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unexpected two-stage selector schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("two-stage selector must remain research-only")
    safety = config.get("safety", {})
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading mutation permissions must remain false")
    data = config["data"]
    model = config["model"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the preregistered target horizon is ten sessions")
    if int(model["purgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("purge must cover the label horizon")
    if int(data["candidatePoolSize"]) != 50:
        raise ValueError("candidate pool size is frozen at fifty")
    if int(data["maximumSelectionsPerDay"]) != 10:
        raise ValueError("maximum daily selections are frozen at ten")
    factors = config.get("frozenFactorDefinitions", [])
    if len(factors) != 4:
        raise ValueError("exactly four frozen factors are required")
    if len({row["factorId"] for row in factors}) != 4:
        raise ValueError("factor identifiers must be unique")
    if len({row["name"] for row in factors}) != 4:
        raise ValueError("factor names must be unique")
    frozen_weights = config.get("frozenFactorWeights")
    if frozen_weights is not None:
        names = {str(row["name"]) for row in factors}
        if set(frozen_weights) != names:
            raise ValueError("frozen factor weights must match factor definitions")
        values = np.asarray(list(map(float, frozen_weights.values())), dtype=float)
        if np.any(values <= 0.0) or not np.isclose(values.sum(), 1.0):
            raise ValueError("frozen factor weights must be positive and sum to one")
    policy = config["selectionPolicy"]
    probability = float(policy["minimumCalibratedWinProbability"])
    if not 0.5 < probability < 1.0:
        raise ValueError("win-probability threshold must exceed chance")
    if policy.get("allowCash") is not True or policy.get("neverForceTenSelections") is not True:
        raise ValueError("the selector must be allowed to hold cash")
    quantiles = list(map(float, model["returnTargetWinsorQuantiles"]))
    if len(quantiles) != 2 or not 0.0 < quantiles[0] < quantiles[1] < 1.0:
        raise ValueError("invalid return winsorisation quantiles")


def build_factor_rank_frames(
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Evaluate the four frozen DSL programs and orient them using train-frozen signs."""
    bins = int(config["data"]["factorLiquidityNeutralisationBins"])
    output: dict[str, pd.DataFrame] = {}
    for definition in config["frozenFactorDefinitions"]:
        raw = core.evaluate_expression(definition["expression"], panel).replace(
            [np.inf, -np.inf], np.nan
        )
        raw *= float(definition["direction"])
        neutral = autonomous.size_neutralise(raw, panel, bins)
        output[str(definition["name"])] = neutral.rank(axis=1, pct=True)
    return output


def build_past_only_feature_frames(
    panel: dict[str, pd.DataFrame],
    factor_ranks: dict[str, pd.DataFrame],
    factor_weights: dict[str, float] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Create features available by the signal-day close; no negative shift is allowed."""
    eligible = panel["eligible"]
    returns = panel["returns"]
    close = panel["close"]
    amount = panel["amount"]
    factor_frames = [factor_ranks[name] for name in factor_ranks]
    if factor_weights is None:
        composite = sum(factor_frames) / float(len(factor_frames))
    else:
        if set(factor_weights) != set(factor_ranks):
            raise ValueError("factor weights do not match factor-rank frames")
        composite = sum(
            factor_ranks[name] * float(factor_weights[name])
            for name in factor_ranks
        )
    factor_stack = np.stack([frame.to_numpy(dtype=float) for frame in factor_frames])
    finite = np.isfinite(factor_stack)
    count = finite.sum(axis=0)
    safe = np.where(finite, factor_stack, 0.0)
    mean = safe.sum(axis=0) / np.maximum(count, 1)
    variance = (
        np.where(finite, (factor_stack - mean) ** 2, 0.0).sum(axis=0)
        / np.maximum(count, 1)
    )
    minimum_values = np.where(finite, factor_stack, np.inf).min(axis=0)
    minimum_values[count == 0] = np.nan
    dispersion_values = np.sqrt(variance)
    dispersion_values[count == 0] = np.nan
    minimum = pd.DataFrame(
        minimum_values, index=close.index, columns=close.columns
    )
    dispersion = pd.DataFrame(
        dispersion_values, index=close.index, columns=close.columns
    )
    features: dict[str, pd.DataFrame] = {
        f"factor_{name}": frame for name, frame in factor_ranks.items()
    }
    features.update(
        {
            "factor_composite": composite,
            "factor_minimum": minimum,
            "factor_dispersion": dispersion,
            "stock_return_5": returns.rolling(5, min_periods=3).sum(),
            "stock_return_20": returns.rolling(20, min_periods=10).sum(),
            "stock_return_60": returns.rolling(60, min_periods=30).sum(),
            "stock_volatility_20": returns.rolling(20, min_periods=10).std(),
            "stock_volatility_60": returns.rolling(60, min_periods=30).std(),
            "stock_drawdown_60": close.div(
                close.rolling(60, min_periods=30).max().replace(0.0, np.nan)
            )
            - 1.0,
            "stock_amount_ratio_5_20": amount.rolling(5, min_periods=3)
            .mean()
            .div(amount.rolling(20, min_periods=10).mean().replace(0.0, np.nan)),
            "stock_liquidity_percentile": amount.rolling(20, min_periods=10)
            .median()
            .shift(1)
            .rank(axis=1, pct=True),
            "stock_open_gap": panel["open"].div(close.shift(1).replace(0.0, np.nan))
            - 1.0,
            "stock_intraday_return": close.div(panel["open"].replace(0.0, np.nan))
            - 1.0,
        }
    )
    for name in features:
        features[name] = features[name].where(eligible).replace(
            [np.inf, -np.inf], np.nan
        )
    market_daily = returns.where(eligible).median(axis=1)
    moving_average = close.rolling(20, min_periods=10).mean()
    denominator = eligible.sum(axis=1).replace(0, np.nan)
    market = {
        "market_return_20": market_daily.rolling(20, min_periods=10).sum(),
        "market_return_60": market_daily.rolling(60, min_periods=30).sum(),
        "market_volatility_20": market_daily.rolling(20, min_periods=10).std(),
        "market_breadth_20": (close.gt(moving_average) & eligible)
        .sum(axis=1)
        .div(denominator),
    }
    return features, market


def candidate_pool_mask(
    composite: pd.DataFrame,
    eligible: pd.DataFrame,
    pool_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = composite.where(eligible).rank(axis=1, ascending=False, method="first")
    return ranked.le(pool_size), ranked


def build_candidate_table(
    panel: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
    market_features: dict[str, pd.Series],
    target: pd.DataFrame,
    pool_size: int,
) -> pd.DataFrame:
    composite = features["factor_composite"]
    pool, candidate_rank = candidate_pool_mask(
        composite, panel["eligible"], pool_size
    )
    index = candidate_rank.where(pool).stack().index
    table = pd.DataFrame(index=index)
    table.index.names = ["date", "securityId"]
    table["candidate_rank"] = candidate_rank.where(pool).stack().reindex(index)
    for name, frame in features.items():
        table[name] = frame.where(pool).stack().reindex(index)
    dates = table.index.get_level_values("date")
    for name, series in market_features.items():
        table[name] = series.reindex(dates).to_numpy(dtype=float)
    table["target_return_10d"] = target.where(pool).stack().reindex(index)
    benchmark = target.where(panel["eligible"]).mean(axis=1)
    table["benchmark_return_10d"] = benchmark.reindex(dates).to_numpy(dtype=float)
    table["label_positive_10d"] = np.where(
        table["target_return_10d"].notna(),
        table["target_return_10d"].gt(0.0).astype(float),
        np.nan,
    )
    return table.reset_index()


def model_feature_columns(config: dict[str, Any]) -> list[str]:
    feature_set = config["featureSet"]
    return [
        *feature_set["factorFeatures"],
        *feature_set["stockFeatures"],
        *feature_set["marketFeatures"],
    ]


def chronological_fit_calibration_dates(
    available_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(available_dates)))
    calibration_days = int(config["model"]["calibrationTradingDays"])
    purge = int(config["model"]["purgeTradingDays"])
    minimum = int(config["model"]["minimumBaseFitTradingDays"])
    if len(dates) < minimum + purge + calibration_days:
        raise RuntimeError("insufficient dates for frozen fit/calibration split")
    calibration = dates[-calibration_days:]
    fit_end_position = len(dates) - calibration_days - purge - 1
    fit = dates[: fit_end_position + 1]
    if len(fit) < minimum:
        raise RuntimeError("base fit period is shorter than preregistered minimum")
    audit = {
        "fitDateRange": [fit[0].date().isoformat(), fit[-1].date().isoformat()],
        "fitTradingDays": len(fit),
        "purgeTradingDays": purge,
        "calibrationDateRange": [
            calibration[0].date().isoformat(),
            calibration[-1].date().isoformat(),
        ],
        "calibrationTradingDays": len(calibration),
        "strictChronology": bool(fit[-1] < calibration[0]),
    }
    return fit, calibration, audit


def _classifier(config: dict[str, Any]) -> HistGradientBoostingClassifier:
    spec = config["model"]["classifier"]
    return HistGradientBoostingClassifier(
        learning_rate=float(spec["learningRate"]),
        max_iter=int(spec["maxIter"]),
        max_leaf_nodes=int(spec["maxLeafNodes"]),
        min_samples_leaf=int(spec["minSamplesLeaf"]),
        l2_regularization=float(spec["l2Regularization"]),
        random_state=int(spec["randomState"]),
    )


def _regressor(config: dict[str, Any]) -> HistGradientBoostingRegressor:
    spec = config["model"]["regressor"]
    return HistGradientBoostingRegressor(
        loss=str(spec["loss"]),
        learning_rate=float(spec["learningRate"]),
        max_iter=int(spec["maxIter"]),
        max_leaf_nodes=int(spec["maxLeafNodes"]),
        min_samples_leaf=int(spec["minSamplesLeaf"]),
        l2_regularization=float(spec["l2Regularization"]),
        random_state=int(spec["randomState"]),
    )


def fit_frozen_model(
    table: pd.DataFrame,
    available_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> FrozenModel:
    feature_columns = model_feature_columns(config)
    fit_dates, calibration_dates, audit = chronological_fit_calibration_dates(
        available_dates, config
    )
    labelled = table[table["target_return_10d"].notna()].copy()
    fit_rows = labelled[labelled["date"].isin(fit_dates)]
    calibration_rows = labelled[labelled["date"].isin(calibration_dates)]
    if fit_rows.empty or calibration_rows.empty:
        raise RuntimeError("fit or calibration table is empty")
    x_fit = fit_rows[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_class = fit_rows["label_positive_10d"].astype(int)
    y_return = fit_rows["target_return_10d"].astype(float)
    quantiles = list(map(float, config["model"]["returnTargetWinsorQuantiles"]))
    lower, upper = map(float, y_return.quantile(quantiles).tolist())
    classifier = _classifier(config)
    regressor = _regressor(config)
    classifier.fit(x_fit, y_class)
    regressor.fit(x_fit, y_return.clip(lower, upper))
    x_calibration = calibration_rows[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    raw_probability = classifier.predict_proba(x_calibration)[:, 1]
    probability_calibrator = IsotonicRegression(
        y_min=0.001, y_max=0.999, out_of_bounds="clip"
    )
    probability_calibrator.fit(
        raw_probability, calibration_rows["label_positive_10d"].astype(int)
    )
    raw_return = regressor.predict(x_calibration)
    return_calibrator = Ridge(alpha=float(config["model"]["returnCalibrationAlpha"]))
    return_calibrator.fit(
        (raw_return * 100.0).reshape(-1, 1),
        calibration_rows["target_return_10d"].clip(lower, upper),
    )
    audit.update(
        {
            "fitRows": len(fit_rows),
            "calibrationRows": len(calibration_rows),
            "featureCount": len(feature_columns),
            "returnTargetClip": [lower, upper],
            "fitMaximumLabelDate": fit_rows["date"].max().date().isoformat(),
            "calibrationMinimumLabelDate": calibration_rows["date"]
            .min()
            .date()
            .isoformat(),
        }
    )
    return FrozenModel(
        classifier=classifier,
        regressor=regressor,
        probability_calibrator=probability_calibrator,
        return_calibrator=return_calibrator,
        feature_columns=feature_columns,
        prior_probability=float(y_class.mean()),
        prior_return=float(y_return.mean()),
        return_clip=(lower, upper),
        audit=audit,
    )


def predict(model: FrozenModel, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    x = output[model.feature_columns].replace([np.inf, -np.inf], np.nan)
    raw_probability = model.classifier.predict_proba(x)[:, 1]
    raw_return = model.regressor.predict(x)
    output["raw_win_probability"] = raw_probability
    output["calibrated_win_probability"] = model.probability_calibrator.predict(
        raw_probability
    )
    output["calibrated_non_positive_probability"] = (
        1.0 - output["calibrated_win_probability"]
    )
    output["raw_expected_return"] = raw_return
    output["calibrated_expected_return"] = model.return_calibrator.predict(
        (raw_return * 100.0).reshape(-1, 1)
    )
    output["prior_win_probability"] = model.prior_probability
    output["prior_expected_return"] = model.prior_return
    return output


def walk_forward_train_predictions(
    table: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    model_cfg = config["model"]
    minimum_history = (
        int(model_cfg["minimumBaseFitTradingDays"])
        + int(model_cfg["calibrationTradingDays"])
        + int(model_cfg["purgeTradingDays"])
    )
    dates = pd.DatetimeIndex(sorted(pd.unique(train_dates)))
    evaluation_dates = dates[minimum_history:]
    folds = int(model_cfg["walkForwardFolds"])
    if len(evaluation_dates) < folds:
        return pd.DataFrame(), []
    blocks = [pd.DatetimeIndex(block) for block in np.array_split(evaluation_dates, folds)]
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for number, block in enumerate(blocks, start=1):
        if block.empty:
            continue
        history = dates[dates < block[0]]
        model = fit_frozen_model(table, history, config)
        rows = table[table["date"].isin(block) & table["target_return_10d"].notna()]
        if rows.empty:
            continue
        predicted = predict(model, rows)
        predicted["walk_forward_fold"] = number
        predictions.append(predicted)
        audits.append(
            {
                "fold": number,
                "predictionDateRange": [
                    block[0].date().isoformat(),
                    block[-1].date().isoformat(),
                ],
                "predictionRows": len(predicted),
                "model": model.audit,
            }
        )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        audits,
    )


def ece_score(labels: pd.Series, probabilities: pd.Series, bins: int) -> float | None:
    valid = labels.notna() & probabilities.notna()
    y = labels.loc[valid].astype(float)
    p = probabilities.loc[valid].astype(float).clip(0.0, 1.0)
    if y.empty:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    total = len(y)
    error = 0.0
    for number in range(bins):
        member = bucket == number
        if not np.any(member):
            continue
        error += float(np.sum(member)) / total * abs(
            float(y.to_numpy()[member].mean()) - float(p.to_numpy()[member].mean())
        )
    return error


def probability_metrics(
    rows: pd.DataFrame,
    probability_column: str,
    bins: int,
) -> dict[str, Any]:
    valid = rows[rows["label_positive_10d"].notna() & rows[probability_column].notna()]
    if valid.empty:
        return {"n": 0, "brier": None, "logLoss": None, "auc": None, "ece": None}
    y = valid["label_positive_10d"].astype(int)
    p = valid[probability_column].astype(float).clip(0.001, 0.999)
    auc = float(roc_auc_score(y, p)) if y.nunique() == 2 else None
    return {
        "n": len(valid),
        "positiveRate": round(float(y.mean()), 8),
        "meanProbability": round(float(p.mean()), 8),
        "brier": round(float(brier_score_loss(y, p)), 8),
        "logLoss": round(float(log_loss(y, p, labels=[0, 1])), 8),
        "auc": round(auc, 8) if auc is not None else None,
        "ece": round(float(ece_score(y, p, bins) or 0.0), 8),
    }


def probability_buckets(rows: pd.DataFrame) -> list[dict[str, Any]]:
    valid = rows[
        rows["label_positive_10d"].notna()
        & rows["calibrated_win_probability"].notna()
    ].copy()
    if valid.empty:
        return []
    valid["bucketLower"] = (
        np.floor(valid["calibrated_win_probability"].clip(0.0, 0.999999) * 10.0)
        / 10.0
    )
    output = []
    for lower, group in valid.groupby("bucketLower", sort=True):
        output.append(
            {
                "lower": round(float(lower), 1),
                "upper": round(float(lower + 0.1), 1),
                "n": len(group),
                "meanProbability": round(
                    float(group["calibrated_win_probability"].mean()), 6
                ),
                "hitRate": round(float(group["label_positive_10d"].mean()), 6),
                "meanReturn": round(float(group["target_return_10d"].mean()), 8),
            }
        )
    return output


def select_rows(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    policy = config["selectionPolicy"]
    allowed = rows[
        rows["calibrated_win_probability"].ge(
            float(policy["minimumCalibratedWinProbability"])
        )
        & rows["calibrated_expected_return"].ge(
            float(policy["minimumCalibratedExpectedReturn"])
        )
    ].copy()
    if allowed.empty:
        return allowed
    allowed = allowed.sort_values(
        ["date", str(policy["rankingField"]), "calibrated_win_probability"],
        ascending=[True, False, False],
    )
    return allowed.groupby("date", sort=False).head(
        int(config["data"]["maximumSelectionsPerDay"])
    )


def baseline_rows(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return rows[
        rows["candidate_rank"].le(int(config["data"]["maximumSelectionsPerDay"]))
    ].copy()


def selected_mask(
    selected: pd.DataFrame,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    output = pd.DataFrame(False, index=index, columns=columns)
    for row in selected[["date", "securityId"]].itertuples(index=False):
        if row.date in output.index and row.securityId in output.columns:
            output.at[row.date, row.securityId] = True
    return output


def _maximum_drawdown(returns: pd.Series) -> float | None:
    values = returns.fillna(0.0)
    if values.empty:
        return None
    equity = (1.0 + values).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def portfolio_metrics(
    selected: pd.DataFrame,
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = selected[selected["date"].isin(dates)].copy()
    mask = selected_mask(selected, target.index, target.columns)
    count = mask.sum(axis=1)
    basket = target.where(mask).mean(axis=1).reindex(dates).dropna()
    benchmark = target.where(eligible).mean(axis=1).reindex(basket.index)
    sleeve = mask.astype(float).div(count.replace(0, np.nan), axis=0).fillna(0.0)
    holding = int(config["data"]["holdingTradingDays"])
    weights = sleeve.rolling(holding, min_periods=1).sum() / float(holding)
    exposure = weights.sum(axis=1).clip(0.0, 1.0)
    daily = (weights * one_day).sum(axis=1, min_count=1).fillna(0.0).reindex(dates)
    cash = 1.0 - exposure
    turnover = (
        weights.diff().abs().sum(axis=1) + cash.diff().abs()
    ) / 2.0
    turnover = turnover.reindex(dates).fillna(0.0)
    costed = daily - turnover * float(config["data"]["roundTripCost"])

    def annualized(values: pd.Series) -> float:
        return float((1.0 + values).prod() ** (252.0 / max(1, len(values))) - 1.0)

    individual = selected["target_return_10d"].dropna()
    return {
        "calendarDays": len(dates),
        "newSignalDays": int(selected["date"].nunique()),
        "selectedStockObservations": len(individual),
        "averageSelectionsOnSignalDay": round(
            float(selected.groupby("date").size().mean()) if len(selected) else 0.0, 4
        ),
        "averageExposure": round(float(exposure.reindex(dates).mean()), 8),
        "basketTenDayMeanReturn": round(float(basket.mean()), 8) if len(basket) else None,
        "basketTenDayMedianReturn": round(float(basket.median()), 8)
        if len(basket)
        else None,
        "basketTenDayWinRate": round(float((basket > 0.0).mean()), 8)
        if len(basket)
        else None,
        "basketTenDayExcessMean": round(float((basket - benchmark).mean()), 8)
        if len(basket)
        else None,
        "individualTenDayMeanReturn": round(float(individual.mean()), 8)
        if len(individual)
        else None,
        "individualTenDayWinRate": round(float((individual > 0.0).mean()), 8)
        if len(individual)
        else None,
        "grossCumulativeReturn": round(float((1.0 + daily).prod() - 1.0), 8),
        "grossAnnualizedReturn": round(annualized(daily), 8),
        "grossMaximumDrawdown": round(float(_maximum_drawdown(daily) or 0.0), 8),
        "costedCumulativeReturn": round(float((1.0 + costed).prod() - 1.0), 8),
        "costedAnnualizedReturn": round(annualized(costed), 8),
        "costedMaximumDrawdown": round(
            float(_maximum_drawdown(costed) or 0.0), 8
        ),
        "averageDailyTurnover": round(float(turnover.mean()), 8),
    }


def period_report(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    eligible: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = predictions[predictions["date"].isin(dates)].copy()
    selected = select_rows(rows, config)
    baseline = baseline_rows(rows, config)
    bins = int(config["evaluation"]["eceBins"])
    return {
        "candidateRows": len(rows),
        "modelProbability": probability_metrics(
            rows, "calibrated_win_probability", bins
        ),
        "constantPriorProbability": probability_metrics(
            rows, "prior_win_probability", bins
        ),
        "probabilityBuckets": probability_buckets(rows),
        "secondStagePortfolio": portfolio_metrics(
            selected, target, one_day, eligible, dates, config
        ),
        "frozenFactorTop10Baseline": portfolio_metrics(
            baseline, target, one_day, eligible, dates, config
        ),
    }


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    stable = True
    for period in ("validation", "shadow"):
        second = report["periods"][period]["secondStagePortfolio"]
        baseline = report["periods"][period]["frozenFactorTop10Baseline"]
        has_comparable_selections = bool(
            second.get("selectedStockObservations", 0) > 0
            and second.get("basketTenDayMeanReturn") is not None
            and baseline.get("basketTenDayMeanReturn") is not None
            and second.get("basketTenDayWinRate") is not None
            and baseline.get("basketTenDayWinRate") is not None
        )
        return_delta = (
            second["basketTenDayMeanReturn"] - baseline["basketTenDayMeanReturn"]
            if has_comparable_selections
            else None
        )
        win_delta = (
            second["basketTenDayWinRate"] - baseline["basketTenDayWinRate"]
            if has_comparable_selections
            else None
        )
        probability = report["periods"][period]["modelProbability"]
        prior = report["periods"][period]["constantPriorProbability"]
        probability_better = bool(
            probability.get("brier") is not None
            and prior.get("brier") is not None
            and probability["brier"] < prior["brier"]
            and probability.get("auc") is not None
            and probability["auc"] > 0.5
        )
        improves = bool(
            has_comparable_selections
            and return_delta is not None
            and win_delta is not None
            and return_delta > 0.0
            and win_delta > 0.0
            and probability_better
        )
        comparisons[period] = {
            "hasComparableSelections": has_comparable_selections,
            "basketMeanReturnDelta": (
                round(return_delta, 8) if return_delta is not None else None
            ),
            "basketWinRateDelta": (
                round(win_delta, 8) if win_delta is not None else None
            ),
            "probabilityBetterThanPrior": probability_better,
            "improvesReturnAndWinRate": improves,
        }
        stable = stable and improves
    return {
        "status": "research_only_not_eligible_for_trading",
        "stableHistoricalIncrement": stable,
        "comparisons": comparisons,
        "decision": (
            "retain_as_research_hypothesis_only"
            if stable
            else "reject_for_trading_keep_diagnostics"
        ),
        "promotionAllowed": False,
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Perception-XAlpha Two-Stage Selector",
        "",
        "**Status: research-only / shadow-only / not a trading signal.**",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Data: {report['dataRange'][0]} to {report['dataRange'][1]}",
        "- First stage: frozen four-factor top 50.",
        "- Second stage: calibrated ten-day win probability and expected return.",
        "- Orders: always empty.",
        "",
        "## Historical comparison",
        "",
        "| Period | Model Brier | Model AUC | Selected days | 10d mean | 10d win | Baseline 10d mean | Baseline win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("trainWalkForward", "validation", "shadow"):
        block = report["periods"].get(period)
        if not block:
            continue
        probability = block["modelProbability"]
        model = block["secondStagePortfolio"]
        baseline = block["frozenFactorTop10Baseline"]
        lines.append(
            "| {period} | {brier} | {auc} | {days} | {mean} | {win} | {base_mean} | {base_win} |".format(
                period=period,
                brier=probability.get("brier"),
                auc=probability.get("auc"),
                days=model["newSignalDays"],
                mean=model["basketTenDayMeanReturn"],
                win=model["basketTenDayWinRate"],
                base_mean=baseline["basketTenDayMeanReturn"],
                base_win=baseline["basketTenDayWinRate"],
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Decision: `{report['verdict']['decision']}`",
            f"- Stable historical increment: `{report['verdict']['stableHistoricalIncrement']}`",
            "- This historical run cannot promote or connect to trading.",
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
    base = load_json(ROOT / config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    if int(cog_config["data"]["predictionHorizonTradingDays"]) != int(
        config["data"]["holdingTradingDays"]
    ):
        raise ValueError("base label horizon and selector horizon differ")
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    factor_ranks = build_factor_rank_frames(panel, config)
    features, market_features = build_past_only_feature_frames(
        panel, factor_ranks, config.get("frozenFactorWeights")
    )
    target, one_day = autonomous.target_frames(panel, cog_config)
    split = autonomous.make_split(panel["close"].index, cog_config)
    table = build_candidate_table(
        panel,
        features,
        market_features,
        target,
        int(config["data"]["candidatePoolSize"]),
    )
    train_dates = pd.DatetimeIndex(split.train[split.train].index)
    validation_dates = pd.DatetimeIndex(split.validation[split.validation].index)
    shadow_dates = pd.DatetimeIndex(split.shadow[split.shadow].index)
    train_walk_forward, fold_audits = walk_forward_train_predictions(
        table, train_dates, config
    )
    frozen_model = fit_frozen_model(table, train_dates, config)
    external_rows = table[
        table["date"].isin(validation_dates.union(shadow_dates))
        | table["date"].eq(table["date"].max())
    ]
    external_predictions = predict(frozen_model, external_rows)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = run_id or (
        "run_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + digest({"config": config, "code": CODE_VERSION})[:10]
    )
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": created_at,
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": digest(config),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "candidateRows": len(table),
        "featureColumns": model_feature_columns(config),
        "modelAudit": frozen_model.audit,
        "walkForwardAudits": fold_audits,
        "periods": {},
        "knownLimitations": config["knownLimitations"],
        "skippedFeatures": config["featureSet"]["skippedFeatures"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    if not train_walk_forward.empty:
        walk_dates = pd.DatetimeIndex(
            sorted(pd.unique(train_walk_forward["date"]))
        )
        report["periods"]["trainWalkForward"] = period_report(
            train_walk_forward,
            walk_dates,
            target,
            one_day,
            panel["eligible"],
            config,
        )
    report["periods"]["validation"] = period_report(
        external_predictions,
        validation_dates,
        target,
        one_day,
        panel["eligible"],
        config,
    )
    report["periods"]["shadow"] = period_report(
        external_predictions,
        shadow_dates,
        target,
        one_day,
        panel["eligible"],
        config,
    )
    report["verdict"] = verdict(report)
    latest_date = table["date"].max()
    latest = external_predictions[external_predictions["date"].eq(latest_date)].copy()
    latest_selected = select_rows(latest, config)
    selected_ids = set(latest_selected["securityId"])
    latest = latest.sort_values(
        ["calibrated_expected_return", "calibrated_win_probability"],
        ascending=False,
    )
    latest_output_columns = [
        "date",
        "securityId",
        "candidate_rank",
        "factor_composite",
        "calibrated_win_probability",
        "calibrated_non_positive_probability",
        "calibrated_expected_return",
    ]
    latest_ranking = latest[latest_output_columns].copy()
    latest_ranking["selectedByFrozenPolicy"] = latest_ranking["securityId"].isin(
        selected_ids
    )
    output_directory = ROOT / config["output"]["root"] / run_id
    output_directory.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        output_directory / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(output_directory / "report.md", markdown_report(report))
    latest_ranking.to_csv(output_directory / "latest_ranking.csv", index=False)
    model_manifest = {
        "schemaVersion": "perception_xalpha_two_stage_model_manifest_v1",
        "status": "research_only_not_online_inference",
        "runId": run_id,
        "featureColumns": frozen_model.feature_columns,
        "modelAudit": frozen_model.audit,
        "selectionPolicy": config["selectionPolicy"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    atomic_write_text(
        output_directory / "model_manifest.json",
        json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args.config.resolve(), args.run_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
