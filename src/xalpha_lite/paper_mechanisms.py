"""Minimal causal building blocks inspired by papers used in the parent project.

These functions are transparent research primitives, not trading recommendations.
Every time-series output at t uses observations no later than t unless a function is
explicitly a label generator (``triple_barrier_labels``).
"""

from __future__ import annotations

from itertools import product
from statistics import NormalDist
import numpy as np
import pandas as pd


def early_warning_features(series: pd.Series, window: int = 20) -> pd.DataFrame:
    """Generic EWS: variance, lag-1 autocorrelation and absolute skew."""
    values = pd.to_numeric(series, errors="coerce")
    rolling = values.rolling(window, min_periods=window)
    return pd.DataFrame(
        {
            "ews_variance": rolling.var(),
            "ews_autocorr1": rolling.apply(
                lambda x: pd.Series(x).autocorr(lag=1), raw=False
            ),
            "ews_abs_skew": rolling.skew().abs(),
        },
        index=series.index,
    )


def causal_two_state_hmm_filter(
    observations: pd.Series,
    means: tuple[float, float],
    standard_deviations: tuple[float, float],
    transition: np.ndarray,
    initial: tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """Forward-only two-state Gaussian HMM probabilities with frozen parameters."""
    matrix = np.asarray(transition, dtype=float)
    if matrix.shape != (2, 2) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("transition must be a 2x2 row-stochastic matrix")
    probability = np.asarray(initial, dtype=float)
    output: list[np.ndarray] = []
    for value in pd.to_numeric(observations, errors="coerce"):
        predicted = probability @ matrix
        if np.isfinite(value):
            likelihood = np.asarray(
                [
                    np.exp(-0.5 * ((value - mean) / max(sd, 1e-12)) ** 2)
                    / max(sd, 1e-12)
                    for mean, sd in zip(means, standard_deviations)
                ]
            )
            posterior = predicted * likelihood
            probability = posterior / max(posterior.sum(), 1e-300)
        else:
            probability = predicted
        output.append(probability.copy())
    return pd.DataFrame(
        output,
        index=observations.index,
        columns=["hmm_state0_probability", "hmm_state1_probability"],
    )


def bocpd_change_probability(
    observations: pd.Series,
    hazard: float = 1 / 100,
    prior_mean: float = 0.0,
    prior_precision: float = 1.0,
    observation_precision: float = 100.0,
    maximum_run_length: int = 250,
) -> pd.Series:
    """Adams-MacKay BOCPD with a Normal known-variance model.

    The returned value is P(run length = 0 | observations through t).
    """
    if not 0.0 < hazard < 1.0:
        raise ValueError("hazard must be between zero and one")
    probabilities = np.array([1.0])
    means = np.array([prior_mean], dtype=float)
    precisions = np.array([prior_precision], dtype=float)
    changes: list[float] = []
    for value in pd.to_numeric(observations, errors="coerce"):
        if not np.isfinite(value):
            changes.append(float(hazard))
            continue
        predictive_variance = 1.0 / observation_precision + 1.0 / precisions
        likelihood = np.exp(-0.5 * (value - means) ** 2 / predictive_variance) / np.sqrt(
            2.0 * np.pi * predictive_variance
        )
        growth = probabilities * likelihood * (1.0 - hazard)
        change = float(np.sum(probabilities * likelihood * hazard))
        next_probabilities = np.concatenate([[change], growth])[: maximum_run_length + 1]
        total = next_probabilities.sum()
        probabilities = next_probabilities / max(total, 1e-300)
        posterior_precision = precisions + observation_precision
        posterior_mean = (
            precisions * means + observation_precision * value
        ) / posterior_precision
        means = np.concatenate([[prior_mean], posterior_mean])[: len(probabilities)]
        precisions = np.concatenate([[prior_precision], posterior_precision])[: len(probabilities)]
        changes.append(float(probabilities[0]))
    return pd.Series(changes, index=observations.index, name="bocpd_change_probability")


def causal_dmd_residual(
    features: pd.DataFrame, window: int = 60, ridge: float = 1e-6
) -> pd.DataFrame:
    """Fixed-window linear DMD one-step residual, fitted only through t-1."""
    values = features.astype(float).to_numpy()
    residuals = np.full_like(values, np.nan, dtype=float)
    for position in range(window + 1, len(values)):
        history = values[position - window - 1 : position]
        if not np.isfinite(history).all():
            continue
        x, y = history[:-1].T, history[1:].T
        operator = y @ x.T @ np.linalg.pinv(x @ x.T + ridge * np.eye(x.shape[0]))
        prediction = operator @ values[position - 1]
        residuals[position] = values[position] - prediction
    return pd.DataFrame(
        residuals,
        index=features.index,
        columns=[f"dmd_residual_{column}" for column in features.columns],
    )


def simplified_lppls_features(
    price: pd.Series,
    windows: tuple[int, ...] = (60, 120),
    m_grid: tuple[float, ...] = (0.3, 0.5, 0.7),
    omega_grid: tuple[float, ...] = (6.0, 9.0, 12.0),
    tc_multipliers: tuple[float, ...] = (1.05, 1.15, 1.30),
) -> pd.DataFrame:
    """Fixed-grid classic LPPLS feasibility/stability features.

    This intentionally returns residual and tc dispersion, not a single magic crash date.
    Every scan window ends at t and the grid is fixed before evaluation.
    """
    log_price = np.log(pd.to_numeric(price, errors="coerce").where(price > 0))
    residual_out, tc_dispersion_out = [], []
    for end in range(len(log_price)):
        fits: list[tuple[float, float]] = []
        for window, m, omega, multiplier in product(
            windows, m_grid, omega_grid, tc_multipliers
        ):
            if end + 1 < window:
                continue
            y = log_price.iloc[end - window + 1 : end + 1].to_numpy()
            if not np.isfinite(y).all():
                continue
            t = np.arange(window, dtype=float)
            tc = (window - 1) + multiplier * window
            tau = np.maximum(tc - t, 1e-6)
            power = tau**m
            design = np.column_stack(
                [np.ones(window), power, power * np.cos(omega * np.log(tau)), power * np.sin(omega * np.log(tau))]
            )
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            rmse = float(np.sqrt(np.mean((y - design @ coefficients) ** 2)))
            fits.append((rmse, tc - (window - 1)))
        if not fits:
            residual_out.append(np.nan)
            tc_dispersion_out.append(np.nan)
            continue
        fits.sort(key=lambda item: item[0])
        best = fits[: max(3, len(fits) // 10)]
        residual_out.append(float(np.median([item[0] for item in best])))
        tc_dispersion_out.append(float(np.std([item[1] for item in best])))
    return pd.DataFrame(
        {
            "lppls_residual": residual_out,
            "lppls_tc_dispersion": tc_dispersion_out,
        },
        index=price.index,
    )


def black_litterman_posterior(
    prior_returns: np.ndarray,
    covariance: np.ndarray,
    pick_matrix: np.ndarray,
    views: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman posterior mean and covariance."""
    pi, sigma, p, q, uncertainty = map(
        lambda value: np.asarray(value, dtype=float),
        (prior_returns, covariance, pick_matrix, views, omega),
    )
    tau_sigma_inverse = np.linalg.pinv(tau * sigma)
    omega_inverse = np.linalg.pinv(uncertainty)
    posterior_covariance = np.linalg.pinv(
        tau_sigma_inverse + p.T @ omega_inverse @ p
    )
    posterior_mean = posterior_covariance @ (
        tau_sigma_inverse @ pi + p.T @ omega_inverse @ q
    )
    return posterior_mean, sigma + posterior_covariance


def hoeffding_lower_bound(successes: int, trials: int, delta: float = 0.05) -> float:
    """One-sided finite-sample lower confidence bound for a bounded hit rate."""
    if trials <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("trials must be positive and delta must be in (0,1)")
    empirical = successes / trials
    radius = np.sqrt(np.log(1.0 / delta) / (2.0 * trials))
    return float(max(0.0, empirical - radius))


def triple_barrier_labels(
    close: pd.Series,
    take_profit: float,
    stop_loss: float,
    maximum_holding_bars: int,
) -> pd.DataFrame:
    """Offline labels: first profit, loss or time barrier after each observation.

    This function intentionally uses future data and must never be used as a feature.
    """
    prices = pd.to_numeric(close, errors="coerce")
    records = []
    for start, entry in enumerate(prices):
        label, exit_position, realized = np.nan, np.nan, np.nan
        if np.isfinite(entry):
            for position in range(start + 1, min(len(prices), start + maximum_holding_bars + 1)):
                value = prices.iloc[position]
                if not np.isfinite(value):
                    continue
                realized = value / entry - 1.0
                if realized >= take_profit:
                    label, exit_position = 1.0, position
                    break
                if realized <= -stop_loss:
                    label, exit_position = -1.0, position
                    break
            else:
                position = min(len(prices) - 1, start + maximum_holding_bars)
                value = prices.iloc[position]
                if position > start and np.isfinite(value):
                    realized = value / entry - 1.0
                    label, exit_position = 0.0, position
        records.append((label, exit_position, realized))
    return pd.DataFrame(
        records,
        index=close.index,
        columns=["barrier_label", "exit_position", "realized_return"],
    )


def normal_quantile(probability: float) -> float:
    return NormalDist().inv_cdf(probability)
