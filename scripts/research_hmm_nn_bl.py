"""Frozen HMM + neural forecast + Black-Litterman intraday research.

This module is deliberately offline and cannot submit orders.  It forms a
training-only universe, fits all parameters through 2026-05-20, freezes them,
and evaluates six non-overlapping 30-minute decisions per day on the untouched
2026-05-21 through 2026-06-18 window.

The design fixes the defects found in the Monteiro reference implementation:
one shared neural model is fitted once on a proper forward-return target; HMM
parameters are fitted only on training sequences; filtering is causal; and the
BL layer is separately ablated instead of receiving credit for an equal-weight
energy exposure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from overfitting_guard import combinatorial_symmetric_pbo
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
    reality_check_spa,
)


ROOT = Path(__file__).resolve().parents[1]
CN = timezone(timedelta(hours=8))
DEFAULT_CONFIG = ROOT / "configs" / "research" / "hmm_nn_bl_preregistered.json"
DEFAULT_QUOTES = ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "hmm_nn_bl"

FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "vol_6",
    "market_ret_1",
    "market_ret_3",
    "breadth",
    "relative_strength",
    "amount_rank",
    "hmm_bear_probability",
    "hmm_bull_probability",
    "hmm_expected_return",
    "session_fraction",
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


@dataclass
class GaussianHMM1D:
    n_states: int = 3
    n_iter: int = 40
    variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        self.start_probability = np.full(self.n_states, 1.0 / self.n_states)
        self.transition = np.full((self.n_states, self.n_states), 0.025)
        np.fill_diagonal(self.transition, 0.95)
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        self.means = np.linspace(-0.001, 0.001, self.n_states)
        self.variances = np.full(self.n_states, 1e-6)

    def _emission_log_probability(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)[:, None]
        variances = np.maximum(self.variances, self.variance_floor)
        return -0.5 * (
            np.log(2.0 * math.pi * variances)
            + ((x - self.means) ** 2) / variances
        )

    def _forward_backward(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        emissions = self._emission_log_probability(values)
        log_start = np.log(np.maximum(self.start_probability, 1e-300))
        log_transition = np.log(np.maximum(self.transition, 1e-300))
        n = len(values)
        alpha = np.empty((n, self.n_states))
        alpha[0] = log_start + emissions[0]
        for index in range(1, n):
            alpha[index] = emissions[index] + logsumexp(
                alpha[index - 1][:, None] + log_transition, axis=0
            )
        beta = np.zeros((n, self.n_states))
        for index in range(n - 2, -1, -1):
            beta[index] = logsumexp(
                log_transition + emissions[index + 1][None, :] + beta[index + 1][None, :],
                axis=1,
            )
        log_likelihood = logsumexp(alpha[-1])
        gamma = np.exp(alpha + beta - log_likelihood)
        xi_sum = np.zeros((self.n_states, self.n_states))
        for index in range(n - 1):
            log_xi = (
                alpha[index][:, None]
                + log_transition
                + emissions[index + 1][None, :]
                + beta[index + 1][None, :]
                - log_likelihood
            )
            xi_sum += np.exp(log_xi)
        return gamma, xi_sum, alpha

    def fit(self, sequences: Iterable[np.ndarray]) -> "GaussianHMM1D":
        clean = [
            np.clip(np.asarray(sequence, dtype=float), -0.05, 0.05)
            for sequence in sequences
            if len(sequence) >= 3 and np.isfinite(sequence).all()
        ]
        if not clean:
            raise ValueError("HMM needs at least one finite sequence")
        pooled = np.concatenate(clean)
        quantiles = np.linspace(0.15, 0.85, self.n_states)
        self.means = np.quantile(pooled, quantiles)
        global_variance = max(float(np.var(pooled)), self.variance_floor * 10)
        self.variances = np.full(self.n_states, global_variance)

        for _ in range(self.n_iter):
            start_sum = np.zeros(self.n_states)
            transition_sum = np.zeros((self.n_states, self.n_states))
            gamma_sum = np.zeros(self.n_states)
            weighted_sum = np.zeros(self.n_states)
            weighted_square_sum = np.zeros(self.n_states)
            for sequence in clean:
                gamma, xi_sum, _ = self._forward_backward(sequence)
                start_sum += gamma[0]
                transition_sum += xi_sum
                gamma_sum += gamma.sum(axis=0)
                weighted_sum += (gamma * sequence[:, None]).sum(axis=0)
                weighted_square_sum += (gamma * (sequence[:, None] ** 2)).sum(axis=0)
            self.start_probability = (start_sum + 1e-3) / (
                start_sum.sum() + 1e-3 * self.n_states
            )
            transition_sum += 1e-3
            self.transition = transition_sum / transition_sum.sum(axis=1, keepdims=True)
            safe_gamma = np.maximum(gamma_sum, 1e-12)
            self.means = weighted_sum / safe_gamma
            second_moment = weighted_square_sum / safe_gamma
            self.variances = np.maximum(
                second_moment - self.means**2, self.variance_floor
            )

        order = np.argsort(self.means)
        self.means = self.means[order]
        self.variances = self.variances[order]
        self.start_probability = self.start_probability[order]
        self.transition = self.transition[np.ix_(order, order)]
        return self

    def filter_probabilities(self, values: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(values, dtype=float), -0.05, 0.05)
        emissions = self._emission_log_probability(x)
        probabilities = np.empty((len(x), self.n_states))
        prior = self.start_probability.copy()
        for index in range(len(x)):
            likelihood = np.exp(emissions[index] - np.max(emissions[index]))
            posterior = prior * likelihood
            posterior /= max(float(posterior.sum()), 1e-300)
            probabilities[index] = posterior
            prior = posterior @ self.transition
        return probabilities


def select_training_universe(
    path: Path, train_end: str, universe_size: int
) -> tuple[list[str], dict[str, Any]]:
    daily_last: dict[tuple[str, str], float] = {}
    names: dict[str, str] = {}
    asset_classes: dict[str, str] = {}
    train_dates: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trade_date = str(row.get("trade_date") or str(row.get("timestamp"))[:10])
            if trade_date > train_end:
                continue
            code = str(row.get("stockCode", "")).zfill(6)
            amount = float(row.get("cumulative_amount") or 0.0)
            daily_last[(trade_date, code)] = max(
                amount, daily_last.get((trade_date, code), 0.0)
            )
            names[code] = str(row.get("name") or code)
            asset_classes[code] = str(row.get("asset_class") or "unknown")
            train_dates.add(trade_date)
    required_days = max(1, math.ceil(len(train_dates) * 0.80))
    by_code: dict[str, list[float]] = defaultdict(list)
    for (_, code), amount in daily_last.items():
        by_code[code].append(amount)
    ranked = [
        (
            code,
            len(amounts),
            float(np.mean(amounts)),
        )
        for code, amounts in by_code.items()
        if len(amounts) >= required_days
    ]
    ranked.sort(key=lambda item: (item[2], item[1], item[0]), reverse=True)
    selected = [item[0] for item in ranked[:universe_size]]
    if len(selected) < min(5, universe_size):
        raise ValueError(f"insufficient training universe: {len(selected)}")
    audit = {
        "trainingDates": len(train_dates),
        "minimumCoverageDays": required_days,
        "selected": [
            {
                "stockCode": code,
                "name": names.get(code, code),
                "assetClass": asset_classes.get(code),
                "trainingDays": days,
                "averageDailyCumulativeAmount": round(amount, 2),
            }
            for code, days, amount in ranked[:universe_size]
        ],
        "selectionUsesOos": False,
    }
    return selected, audit


def load_panel(path: Path, selected: list[str]) -> pd.DataFrame:
    selected_set = set(selected)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            code = str(row.get("stockCode", "")).zfill(6)
            if code not in selected_set:
                continue
            rows.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]),
                    "trade_date": str(row.get("trade_date") or str(row["timestamp"])[:10]),
                    "stockCode": code,
                    "close": float(row["close"]),
                    "cumulative_amount": float(row.get("cumulative_amount") or 0.0),
                }
            )
    if not rows:
        raise ValueError("selected universe has no quote rows")
    return pd.DataFrame.from_records(rows).sort_values(
        ["timestamp", "stockCode"], kind="stable"
    )


def _within_day_ratio(values: pd.DataFrame, periods: int) -> pd.DataFrame:
    dates = pd.Series(values.index.date, index=values.index)
    return values.groupby(dates).transform(lambda frame: frame / frame.shift(periods) - 1.0)


def _within_day_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    dates = pd.Series(returns.index.date, index=returns.index)
    return returns.groupby(dates).transform(
        lambda frame: frame.rolling(window, min_periods=window).std()
    )


def build_feature_frames(
    panel: pd.DataFrame, hmm: GaussianHMM1D, train_end: str
) -> dict[str, Any]:
    prices = panel.pivot(index="timestamp", columns="stockCode", values="close").sort_index()
    amounts = panel.pivot(
        index="timestamp", columns="stockCode", values="cumulative_amount"
    ).reindex(prices.index)
    dates = pd.Series(prices.index.strftime("%Y-%m-%d"), index=prices.index)
    returns_1 = _within_day_ratio(prices, 1)
    returns_3 = _within_day_ratio(prices, 3)
    returns_6 = _within_day_ratio(prices, 6)
    volatility_6 = _within_day_volatility(returns_1, 6)
    market_return_1 = returns_1.median(axis=1, skipna=True).fillna(0.0)
    market_return_3 = returns_3.median(axis=1, skipna=True).fillna(0.0)
    breadth = (returns_1 > 0).sum(axis=1) / returns_1.notna().sum(axis=1).clip(lower=1)
    relative_strength = returns_6.rank(axis=1, pct=True, method="average")
    amount_rank = amounts.rank(axis=1, pct=True, method="average")

    training_sequences = [
        market_return_1[dates == trade_date].to_numpy(dtype=float)
        for trade_date in sorted(set(dates[dates <= train_end]))
    ]
    hmm.fit(training_sequences)
    probabilities = pd.DataFrame(
        index=prices.index,
        columns=[f"hmm_{state}" for state in range(hmm.n_states)],
        dtype=float,
    )
    for trade_date in sorted(set(dates)):
        mask = dates == trade_date
        probabilities.loc[mask, :] = hmm.filter_probabilities(
            market_return_1[mask].to_numpy(dtype=float)
        )
    expected = probabilities.to_numpy(dtype=float) @ hmm.means
    session_minutes = prices.index.hour * 60 + prices.index.minute
    session_fraction = pd.Series(
        np.clip((session_minutes - 570) / 330.0, 0.0, 1.0), index=prices.index
    )
    return {
        "prices": prices,
        "amounts": amounts,
        "dates": dates,
        "returns_1": returns_1,
        "ret_1": returns_1,
        "ret_3": returns_3,
        "ret_6": returns_6,
        "vol_6": volatility_6,
        "market_ret_1": market_return_1,
        "market_ret_3": market_return_3,
        "breadth": breadth,
        "relative_strength": relative_strength,
        "amount_rank": amount_rank,
        "hmm_bear_probability": probabilities.iloc[:, 0],
        "hmm_bull_probability": probabilities.iloc[:, -1],
        "hmm_expected_return": pd.Series(expected, index=prices.index),
        "session_fraction": session_fraction,
    }


def build_samples(
    frames: dict[str, Any],
    decision_times: set[str],
    holding_bars: int,
) -> pd.DataFrame:
    prices: pd.DataFrame = frames["prices"]
    records: list[dict[str, Any]] = []
    for trade_date, day_prices in prices.groupby(prices.index.strftime("%Y-%m-%d")):
        timestamps = list(day_prices.index)
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
        for timestamp in timestamps:
            if timestamp.strftime("%H:%M") not in decision_times:
                continue
            decision_index = timestamp_to_index[timestamp]
            entry_index = decision_index + 1
            exit_index = entry_index + holding_bars
            if exit_index >= len(timestamps):
                continue
            entry_time, exit_time = timestamps[entry_index], timestamps[exit_index]
            if (entry_time - timestamp).total_seconds() > 10 * 60:
                continue
            for code in prices.columns:
                entry_price = prices.at[entry_time, code]
                exit_price = prices.at[exit_time, code]
                if not np.isfinite(entry_price) or not np.isfinite(exit_price):
                    continue
                item: dict[str, Any] = {
                    "trade_date": trade_date,
                    "timestamp": timestamp,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "stockCode": code,
                    "gross_forward_return": float(exit_price / entry_price - 1.0),
                }
                valid = True
                for feature in FEATURES:
                    source = frames[feature]
                    if isinstance(source, pd.DataFrame):
                        value = source.at[timestamp, code]
                    else:
                        value = source.at[timestamp]
                    if not np.isfinite(value):
                        valid = False
                        break
                    item[feature] = float(value)
                if valid:
                    records.append(item)
    return pd.DataFrame.from_records(records)


def make_neural_model(config: dict[str, Any]) -> Pipeline:
    nn = config["neuralNetwork"]
    estimator = MLPRegressor(
        hidden_layer_sizes=tuple(int(x) for x in nn["hiddenLayers"]),
        activation=str(nn["activation"]),
        solver=str(nn["solver"]),
        alpha=float(nn["alpha"]),
        max_iter=int(nn["maxIterations"]),
        random_state=int(nn["randomSeed"]),
    )
    return Pipeline([("scale", StandardScaler()), ("mlp", estimator)])


def black_litterman_posterior(
    covariance: np.ndarray,
    views: np.ndarray,
    *,
    tau: float,
    risk_aversion: float,
    view_confidence: float,
    residual_variance: float,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    views = np.asarray(views, dtype=float)
    n = len(views)
    if covariance.shape != (n, n):
        raise ValueError("covariance/view shape mismatch")
    covariance = (covariance + covariance.T) / 2.0
    market_weights = np.full(n, 1.0 / n)
    equilibrium = risk_aversion * covariance @ market_weights
    tau_covariance = max(tau, 1e-8) * covariance
    omega_floor = max(residual_variance, 1e-10)
    omega = np.diag(
        np.maximum(
            np.diag(tau_covariance) * (1.0 - view_confidence) / max(view_confidence, 1e-6),
            omega_floor,
        )
    )
    inv_tau = np.linalg.pinv(tau_covariance)
    inv_omega = np.linalg.pinv(omega)
    posterior_covariance = np.linalg.pinv(inv_tau + inv_omega)
    return posterior_covariance @ (
        inv_tau @ equilibrium + inv_omega @ views
    )


def optimize_long_only(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    round_trip_cost: float,
    risk_aversion: float,
    max_weight: float,
    max_assets: int,
) -> np.ndarray:
    mu = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(mu)
    output = np.zeros(n)
    eligible = np.where(mu > round_trip_cost)[0]
    if len(eligible) == 0:
        return output
    eligible = eligible[np.argsort(mu[eligible])[::-1][:max_assets]]
    sub_mu = mu[eligible] - round_trip_cost
    sub_cov = covariance[np.ix_(eligible, eligible)]

    def objective(weights: np.ndarray) -> float:
        return float(
            -(sub_mu @ weights - 0.5 * risk_aversion * weights @ sub_cov @ weights)
        )

    x0 = np.full(len(eligible), min(max_weight, 1.0 / len(eligible)))
    if x0.sum() > 1.0:
        x0 /= x0.sum()
    solution = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(eligible),
        constraints=[{"type": "ineq", "fun": lambda weights: 1.0 - weights.sum()}],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    weights = solution.x if solution.success else x0
    weights = np.clip(weights, 0.0, max_weight)
    if weights.sum() > 1.0:
        weights /= weights.sum()
    output[eligible] = weights
    return output


def covariance_asof(
    returns: pd.DataFrame,
    timestamp: pd.Timestamp,
    columns: list[str],
    lookback: int,
    ridge: float,
    holding_bars: int,
) -> np.ndarray:
    history = returns.loc[returns.index <= timestamp, columns].tail(lookback)
    covariance = history.cov(min_periods=max(10, min(40, len(history) // 3))).to_numpy()
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(covariance).copy()
    positive = diagonal[diagonal > 0]
    fallback = float(np.median(positive)) if len(positive) else 1e-6
    for index, value in enumerate(diagonal):
        if value <= 0:
            covariance[index, index] = fallback
    covariance *= holding_bars
    covariance += np.eye(len(columns)) * ridge
    return covariance


def _equal_top_weights(
    scores: np.ndarray,
    *,
    threshold: float,
    max_assets: int,
    max_weight: float,
    exposure: float = 1.0,
) -> np.ndarray:
    output = np.zeros(len(scores))
    eligible = np.where(scores > threshold)[0]
    if len(eligible) == 0 or exposure <= 0:
        return output
    eligible = eligible[np.argsort(scores[eligible])[::-1][:max_assets]]
    each = min(max_weight, exposure / len(eligible))
    output[eligible] = each
    return output


def evaluate_variants(
    samples: pd.DataFrame,
    frames: dict[str, Any],
    predictions: np.ndarray,
    config: dict[str, Any],
    residual_variance: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    execution = config["execution"]
    bl = config["blackLitterman"]
    cost = float(execution["roundTripCostBps"]) / 10_000.0
    max_assets = int(execution["maxAssets"])
    max_weight = float(execution["maxWeightPerAsset"])
    samples = samples.copy()
    samples["prediction"] = predictions
    variants = {name: [] for name in config["variants"]}
    allocation_rows: list[dict[str, Any]] = []

    for timestamp, group in samples.groupby("timestamp", sort=True):
        group = group.sort_values("stockCode")
        codes = group["stockCode"].tolist()
        actual = group["gross_forward_return"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        momentum = group["ret_6"].to_numpy(dtype=float)
        bull_probability = float(group["hmm_bull_probability"].iloc[0])
        bear_probability = float(group["hmm_bear_probability"].iloc[0])
        regime_exposure = float(np.clip(0.25 + 0.75 * bull_probability - 0.25 * bear_probability, 0.0, 1.0))
        covariance = covariance_asof(
            frames["returns_1"],
            timestamp,
            codes,
            int(bl["covarianceLookbackBars"]),
            float(bl["covarianceRidge"]),
            int(execution["holdingBars"]),
        )
        posterior = black_litterman_posterior(
            covariance,
            prediction,
            tau=float(bl["tau"]),
            risk_aversion=float(bl["riskAversion"]),
            view_confidence=float(bl["viewConfidence"]),
            residual_variance=residual_variance,
        )
        weights = {
            "momentum_top5": _equal_top_weights(
                momentum, threshold=0.0, max_assets=max_assets, max_weight=max_weight
            ),
            "nn_top5": _equal_top_weights(
                prediction, threshold=cost, max_assets=max_assets, max_weight=max_weight
            ),
            "hmm_nn_top5": _equal_top_weights(
                prediction,
                threshold=cost,
                max_assets=max_assets,
                max_weight=max_weight,
                exposure=regime_exposure,
            ),
            "nn_bl": optimize_long_only(
                posterior,
                covariance,
                round_trip_cost=cost,
                risk_aversion=float(bl["riskAversion"]),
                max_weight=max_weight,
                max_assets=max_assets,
            ),
        }
        weights["hmm_nn_bl"] = weights["nn_bl"] * regime_exposure
        for name, vector in weights.items():
            invested = float(vector.sum())
            net_return = float(vector @ actual - invested * cost)
            variants[name].append(
                {
                    "trade_date": str(group["trade_date"].iloc[0]),
                    "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "net_return": net_return,
                    "gross_return": float(vector @ actual),
                    "exposure": invested,
                    "positions": int(np.sum(vector > 1e-8)),
                }
            )
            for index in np.where(vector > 1e-8)[0]:
                allocation_rows.append(
                    {
                        "variant": name,
                        "trade_date": str(group["trade_date"].iloc[0]),
                        "timestamp": pd.Timestamp(timestamp).isoformat(),
                        "stockCode": codes[index],
                        "weight": round(float(vector[index]), 8),
                        "prediction": round(float(prediction[index]), 8),
                        "bl_posterior": round(float(posterior[index]), 8),
                        "gross_forward_return": round(float(actual[index]), 8),
                        "hmm_bull_probability": round(bull_probability, 8),
                        "hmm_bear_probability": round(bear_probability, 8),
                    }
                )
    return variants, allocation_rows


def daily_returns(intervals: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in intervals:
        grouped[row["trade_date"]].append(float(row["net_return"]))
    return {
        day: float(np.prod([1.0 + value for value in values]) - 1.0)
        for day, values in sorted(grouped.items())
    }


def performance(
    intervals: list[dict[str, Any]], daily: dict[str, float]
) -> dict[str, Any]:
    values = np.asarray(list(daily.values()), dtype=float)
    if len(values) == 0:
        return {"days": 0}
    equity = np.cumprod(1.0 + values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    total_exposure = sum(float(row["exposure"]) for row in intervals)
    return {
        "days": int(len(values)),
        "intervals": len(intervals),
        "positionDecisions": int(sum(int(row["positions"]) for row in intervals)),
        "averageIntervalExposure": round(
            total_exposure / len(intervals) if intervals else 0.0, 6
        ),
        "netTotalReturn": round(float(equity[-1] - 1.0), 8),
        "dailyMean": round(float(np.mean(values)), 8),
        "dailyStd": round(std, 8),
        "dailySharpe": round(float(np.mean(values) / std * math.sqrt(252.0)), 4)
        if std > 0
        else None,
        "winDayRate": round(float(np.mean(values > 0)), 4),
        "worstDay": round(float(np.min(values)), 8),
        "maxDrawdown": round(float(np.min(drawdown)), 8),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HMM + Neural Network + Black-Litterman OOS Research",
        "",
        "Status: `diagnostic_only / offline / no live gating`",
        "",
        "All model parameters and the ETF universe were frozen using data through "
        f"{report['split']['trainEnd']}. The primary window "
        f"{report['split']['oosStart']} to {report['split']['oosEnd']} was not used for fitting.",
        "",
        "## OOS results after 12 bps round-trip cost",
        "",
        "| Variant | Net return | Sharpe | Daily std | Worst day | Max DD | Avg exposure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["oosMetrics"].items():
        sharpe = metrics.get("dailySharpe")
        lines.append(
            f"| {name} | {metrics.get('netTotalReturn', 0):.2%} | "
            f"{sharpe if sharpe is not None else 'n/a'} | "
            f"{metrics.get('dailyStd', 0):.2%} | {metrics.get('worstDay', 0):.2%} | "
            f"{metrics.get('maxDrawdown', 0):.2%} | "
            f"{metrics.get('averageIntervalExposure', 0):.1%} |"
        )
    gates = report["evidence"]
    lines.extend(
        [
            "",
            "## Evidence gates",
            "",
            f"- Best HMM/NN/BL variant: `{report['bestModelVariant']}`.",
            f"- PBO across preregistered variants: `{gates['pbo'].get('pbo')}`.",
            f"- DM vs momentum: significant=`{gates['dm'].get('significant')}`, "
            f"mean daily difference=`{gates['dm'].get('mean_diff')}`.",
            f"- DSR diagnostic: significant=`{gates['dsr'].get('significant')}`, "
            f"p=`{gates['dsr'].get('p_value')}`.",
            f"- SPA: reject no-superior-model=`{gates['spa'].get('reject')}`, "
            f"p=`{gates['spa'].get('p_value')}`.",
            "",
            "## Locked verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "Even a passing historical result remains a forward-shadow candidate. This script "
            "does not modify the paper agent, live config, strategy overlay, execution locks, "
            "orders, or position sizing.",
            "",
            "## Method limitations",
            "",
            "- Only 60 trading days are available; the untouched OOS window is 21 days.",
            "- Yahoo bars have no historical bid/ask or order-book depth; 12 bps is a fixed cost stress.",
            "- Current official product membership can retain survivorship bias.",
            "- The neural model is one frozen architecture, not evidence that neural networks are universally superior.",
            "- Black-Litterman stabilizes views and weights; it cannot create alpha when forecasts have none.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    quotes = Path(args.quotes)
    output_dir = Path(args.output_dir)
    data_cfg = config["data"]
    execution = config["execution"]

    selected, universe_audit = select_training_universe(
        quotes, str(data_cfg["trainEnd"]), int(data_cfg["universeSize"])
    )
    panel = load_panel(quotes, selected)
    hmm_cfg = config["hmm"]
    hmm = GaussianHMM1D(
        n_states=int(hmm_cfg["states"]),
        n_iter=int(hmm_cfg["iterations"]),
        variance_floor=float(hmm_cfg["varianceFloor"]),
    )
    frames = build_feature_frames(panel, hmm, str(data_cfg["trainEnd"]))
    samples = build_samples(
        frames,
        set(execution["decisionTimes"]),
        int(execution["holdingBars"]),
    )
    train = samples[
        (samples["trade_date"] >= str(data_cfg["trainStart"]))
        & (samples["trade_date"] <= str(data_cfg["trainEnd"]))
    ].copy()
    oos = samples[
        (samples["trade_date"] >= str(data_cfg["oosStart"]))
        & (samples["trade_date"] <= str(data_cfg["oosEnd"]))
    ].copy()
    if train.empty or oos.empty:
        raise RuntimeError(f"empty train/oos samples: train={len(train)} oos={len(oos)}")

    model = make_neural_model(config)
    model.fit(train[FEATURES].to_numpy(), train["gross_forward_return"].to_numpy() * 10_000.0)
    train_prediction = model.predict(train[FEATURES].to_numpy()) / 10_000.0
    oos_prediction = model.predict(oos[FEATURES].to_numpy()) / 10_000.0
    residual_variance = max(
        float(np.var(train["gross_forward_return"].to_numpy() - train_prediction)),
        1e-10,
    )

    train_variants, _ = evaluate_variants(
        train, frames, train_prediction, config, residual_variance
    )
    oos_variants, allocation_rows = evaluate_variants(
        oos, frames, oos_prediction, config, residual_variance
    )
    train_daily = {name: daily_returns(rows) for name, rows in train_variants.items()}
    oos_daily = {name: daily_returns(rows) for name, rows in oos_variants.items()}
    train_metrics = {
        name: performance(train_variants[name], train_daily[name])
        for name in config["variants"]
    }
    oos_metrics = {
        name: performance(oos_variants[name], oos_daily[name])
        for name in config["variants"]
    }

    model_variants = [name for name in config["variants"] if name != "momentum_top5"]
    best_model = max(
        model_variants,
        key=lambda name: (
            oos_metrics[name].get("dailySharpe")
            if oos_metrics[name].get("dailySharpe") is not None
            else -999.0
        ),
    )
    shared_dates = sorted(
        set.intersection(*(set(oos_daily[name]) for name in config["variants"]))
    )
    pbo = combinatorial_symmetric_pbo(
        [[oos_daily[name][day] for day in shared_dates] for name in config["variants"]],
        n_blocks=8,
    )
    baseline_days = oos_daily["momentum_top5"]
    best_days = oos_daily[best_model]
    dm = diebold_mariano_hln(baseline_days, best_days, alpha=0.05)
    dsr = deflated_sharpe_diagnostic(
        baseline_days, best_days, n_trials=len(config["variants"]), alpha=0.10
    )
    baseline_losses = [-baseline_days[day] for day in shared_dates]
    alternatives = {
        name: [-oos_daily[name][day] for day in shared_dates]
        for name in model_variants
    }
    spa = reality_check_spa(
        baseline_losses, alternatives, alpha=0.05, n_boot=1000, seed=20260628
    )
    gates_cfg = config["evidenceGates"]
    gate_checks = {
        "minimumOosDays": len(shared_dates) >= int(gates_cfg["minimumOosDays"]),
        "positiveNetReturn": oos_metrics[best_model]["netTotalReturn"] > 0.0,
        "positiveSharpe": (oos_metrics[best_model].get("dailySharpe") or -999.0) > 0.0,
        "netReturnImproved": oos_metrics[best_model]["netTotalReturn"]
        > oos_metrics["momentum_top5"]["netTotalReturn"],
        "sharpeImproved": (
            oos_metrics[best_model].get("dailySharpe") or -999.0
        )
        > (oos_metrics["momentum_top5"].get("dailySharpe") or -999.0),
        "dmSignificant": bool(dm.get("significant")),
        "dsrSignificant": bool(dsr.get("significant")),
        "pboPass": pbo.get("pbo") is not None
        and float(pbo["pbo"]) <= float(gates_cfg["maximumPbo"]),
        "spaPass": bool(spa.get("reject")),
    }
    historical_pass = all(gate_checks.values())
    verdict = (
        "historical_oos_pass_forward_shadow_required"
        if historical_pass
        else "no_validated_incremental_edge"
    )
    verdict_reason = (
        "The preregistered model cleared the historical gates, but the same 60-day dataset "
        "has already supported prior research. It may only proceed to a new-date forward shadow."
        if historical_pass
        else "At least one preregistered OOS evidence gate failed. The model must not alter live "
        "entry gating, exits, or sizing."
    )
    report = {
        "schemaVersion": "hmm_nn_bl_research_result_v1",
        "generatedAt": datetime.now(CN).isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "split": {
            "trainStart": data_cfg["trainStart"],
            "trainEnd": data_cfg["trainEnd"],
            "oosStart": data_cfg["oosStart"],
            "oosEnd": data_cfg["oosEnd"],
            "trainSamples": len(train),
            "oosSamples": len(oos),
            "oosDays": len(shared_dates),
        },
        "universe": universe_audit,
        "hmm": {
            "stateMeans": [round(float(value), 10) for value in hmm.means],
            "stateStd": [round(math.sqrt(float(value)), 10) for value in hmm.variances],
            "transition": np.round(hmm.transition, 8).tolist(),
        },
        "neuralNetwork": {
            "features": FEATURES,
            "trainResidualStd": round(math.sqrt(residual_variance), 8),
            "architectureFrozen": True,
        },
        "trainMetricsDiagnosticOnly": train_metrics,
        "oosMetrics": oos_metrics,
        "bestModelVariant": best_model,
        "evidence": {
            "gateChecks": gate_checks,
            "pbo": pbo,
            "dm": dm,
            "dsr": dsr,
            "spa": spa,
        },
        "verdict": verdict,
        "verdictReason": verdict_reason,
        "liveChanges": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "hmm_nn_bl_oos_result.json", report)
    daily_rows = [
        {"trade_date": day, **{name: oos_daily[name][day] for name in config["variants"]}}
        for day in shared_dates
    ]
    write_csv(output_dir / "hmm_nn_bl_oos_daily.csv", daily_rows)
    write_csv(output_dir / "hmm_nn_bl_oos_allocations.csv", allocation_rows)
    (output_dir / "hmm_nn_bl_oos_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(
        {
            "status": report["status"],
            "verdict": verdict,
            "bestModelVariant": best_model,
            "oosDays": len(shared_dates),
            "oosMetrics": oos_metrics,
            "output": str(output_dir / "hmm_nn_bl_oos_report.md"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
