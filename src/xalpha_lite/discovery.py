"""Bounded factor synthesis, neutral evaluation and anti-overfitting guards."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .dsl import canonical, evaluate, fields_used, validate
from .pit import align_point_in_time_fundamentals
from . import search_protocol as search


WINDOWS = {3, 5, 10, 20, 60, 120, 252}


@dataclass(frozen=True)
class Split:
    train: pd.Series
    validation: pd.Series
    shadow: pd.Series
    audit: dict[str, Any]


def _f(name: str) -> dict[str, Any]:
    return {"field": name}


def _roll(operation: str, argument: dict[str, Any], window: int) -> dict[str, Any]:
    return {"rolling": operation, "arg": argument, "window": window}


def _u(operation: str, argument: dict[str, Any]) -> dict[str, Any]:
    return {"unary": operation, "arg": argument}


def _b(operation: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"binary": operation, "left": left, "right": right}


def _pivot(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="symbol", values=field).sort_index()


def build_panel(
    prices: pd.DataFrame, statements: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    controls = {"industry", "market_cap"}
    if bool(config.get("require_neutralization_data", True)):
        required |= controls
    if not required.issubset(prices.columns):
        raise ValueError(f"prices missing: {sorted(required - set(prices.columns))}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"])
    numeric = ("open", "high", "low", "close", "volume", "amount", "market_cap")
    for field in numeric:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    panel: dict[str, pd.DataFrame] = {
        field: _pivot(frame, field)
        for field in ("open", "high", "low", "close", "volume", "amount")
    }
    close = panel["close"]
    panel["returns"] = close.pct_change(fill_method=None)
    # A monotone session counter, broadcast across symbols. Constant cross-sectionally, so it
    # carries no ranking information on its own; its purpose is to let the DSL express trend
    # quality — corr(close, session_index, w) is the R of price against time, not its direction.
    panel["session_index"] = pd.DataFrame(
        np.repeat(np.arange(len(close), dtype=float)[:, None], len(close.columns), axis=1),
        index=close.index,
        columns=close.columns,
    )
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    market = panel["returns"].median(axis=1)
    panel["market_return"] = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], len(close.columns), axis=1),
        index=close.index,
        columns=close.columns,
    )
    if "market_cap" in frame:
        panel["market_cap"] = _pivot(frame, "market_cap").reindex_like(close)
    if "industry" in frame:
        panel["industry"] = _pivot(frame, "industry").reindex_like(close)
    if "trade_status" in frame:
        # Numeric rather than boolean (1 = normal), and read by point_in_time_eligibility.
        panel["trade_status"] = _pivot(frame, "trade_status").reindex_like(close)
    for field in ("is_st", "is_suspended", "is_delisted", "limit_up", "limit_down"):
        if field in frame:
            values = frame[field]
            if values.dtype == object:
                frame[field] = values.astype(str).str.lower().isin({"1", "true", "yes"})
            # where(), not fillna(): filling an object-dtype pivot downcasts, which pandas
            # deprecates and will change under us.
            flags = _pivot(frame, field).reindex_like(close)
            panel[field] = flags.where(flags.notna(), False).astype(bool)
        else:
            panel[field] = pd.DataFrame(False, index=close.index, columns=close.columns)
    fundamentals = align_point_in_time_fundamentals(statements, close.index, close.columns)
    panel.update(fundamentals)
    if "bps" in panel:
        panel["book_to_price"] = panel["bps"].where(panel["bps"] > 0) / close
    if "eps_ytd" in panel:
        panel["earnings_yield_proxy"] = panel["eps_ytd"] / close

    def cs_z(field: str) -> pd.DataFrame:
        value = panel[field]
        return value.sub(value.mean(axis=1), axis=0).div(
            value.std(axis=1).replace(0.0, np.nan), axis=0
        ).clip(-5, 5)

    available = set(panel)
    if {"roe", "roic", "gross_margin", "cash_to_profit"}.issubset(available):
        panel["quality_composite"] = sum(
            cs_z(field) for field in ("roe", "roic", "gross_margin", "cash_to_profit")
        ) / 4.0
    if {"revenue_yoy", "profit_yoy"}.issubset(available):
        panel["growth_composite"] = (cs_z("revenue_yoy") + cs_z("profit_yoy")) / 2.0
    if {"debt_ratio", "current_ratio", "quick_ratio"}.issubset(available):
        panel["safety_composite"] = (
            -cs_z("debt_ratio") + cs_z("current_ratio") + cs_z("quick_ratio")
        ) / 3.0
    return panel


def _candidate(
    name: str,
    domain: str,
    phenomenon: str,
    mechanism: str,
    expression: dict[str, Any],
    counter_expressions: list[dict[str, Any]] | None = None,
    parents: list[str] | None = None,
    generation: int = 0,
    operator: str = "seed",
    search_paradigm: str | None = None,
) -> dict[str, Any]:
    fingerprint = hashlib.sha256(canonical(expression).encode()).hexdigest()
    factor_id = f"factor_{fingerprint[:20]}"
    birth = {
        "factor_id": factor_id,
        "expression_sha256": fingerprint,
        "parents": parents or [],
        "generation": generation,
        "operator": operator,
        "phenomenon": phenomenon,
        "mechanism": mechanism,
    }
    if search_paradigm is not None:
        birth["search_paradigm"] = search_paradigm
    candidate = {
        "factor_id": factor_id,
        "name": name,
        "domain": domain,
        "phenomenon": phenomenon,
        "mechanism": mechanism,
        "expression": expression,
        "counter_expressions": counter_expressions or [],
        "inputs": sorted(fields_used(expression)),
        "birth_certificate": {
            **birth,
            "certificate_sha256": hashlib.sha256(
                json.dumps(birth, sort_keys=True).encode()
            ).hexdigest(),
        },
    }
    if search_paradigm is not None:
        candidate["search_paradigm"] = search_paradigm
    return candidate


def seed_library(panel: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Small economic seed set from which the bounded generator grows expressions."""
    ret, close, amount = _f("returns"), _f("close"), _f("amount")
    momentum20 = _roll("sum", ret, 20)
    reversal5 = _u("neg", _roll("sum", ret, 5))
    volatility20 = _roll("std", ret, 20)
    low_beta = _u(
        "neg", {"corr": True, "left": ret, "right": _f("market_return"), "window": 60}
    )
    rows = [
        _candidate("momentum_20", "technical", "information_diffusion", "underreaction", momentum20, [_roll("sum", ret, 60)]),
        _candidate("reversal_5", "technical", "liquidity_reversal", "temporary_price_pressure", reversal5, [_u("neg", _roll("sum", ret, 20))]),
        _candidate("low_beta_60", "technical", "risk_budgeting", "leverage_constraint", low_beta, [_u("neg", {"corr": True, "left": ret, "right": _f("market_return"), "window": 20})]),
        _candidate("low_volatility_20", "technical", "volatility_feedback", "risk_budgeting", _u("neg", volatility20), [_u("neg", _roll("std", ret, 60))]),
        _candidate("liquidity_20", "technical", "order_splitting", "persistent_flow", _roll("mean", _u("signed_log1p", amount), 20), [_roll("mean", _u("signed_log1p", amount), 60)]),
        _candidate("range_position", "technical", "crowding_unwind", "breakout_or_crowding", _b("div", close, _roll("max", close, 120)), [_b("div", close, _roll("max", close, 60))]),
    ]
    fundamental_specs = [
        ("book_to_price", "value_repricing", "valuation_dispersion", "book_to_price"),
        ("earnings_yield_proxy", "value_repricing", "earnings_anchor", "earnings_yield_proxy"),
        ("quality", "fundamental_diffusion", "persistent_profitability", "quality_composite"),
        ("growth", "fundamental_diffusion", "slow_information_diffusion", "growth_composite"),
        ("financial_safety", "risk_budgeting", "distress_avoidance", "safety_composite"),
        ("cash_conversion", "fundamental_diffusion", "accrual_mispricing", "cash_to_profit"),
    ]
    for name, phenomenon, mechanism, field in fundamental_specs:
        if field in panel:
            rows.append(
                _candidate(
                    name,
                    "fundamental",
                    phenomenon,
                    mechanism,
                    _f(field),
                    [{"lag": 252, "arg": _f(field)}],
                )
            )
    if "quality_composite" in panel:
        quality = _f("quality_composite")
        rows += [
            _candidate("quality_low_beta", "hybrid", "conditional_quality", "quality_under_risk_constraints", _b("mul", quality, low_beta), [quality, low_beta]),
            _candidate("quality_momentum", "hybrid", "conditional_quality", "quality_information_diffusion", _b("mul", quality, momentum20), [quality, momentum20]),
            _candidate("quality_per_volatility", "hybrid", "conditional_quality", "risk_adjusted_quality", _b("div", quality, volatility20), [quality, _u("neg", volatility20)]),
        ]
    if "book_to_price" in panel:
        value = _f("book_to_price")
        rows += [
            _candidate("value_reversal", "hybrid", "conditional_value", "forced_sale_recovery", _b("mul", value, reversal5), [value, reversal5]),
            _candidate("value_momentum", "hybrid", "conditional_value", "value_catalyst", _b("mul", value, momentum20), [value, momentum20]),
        ]
    if "growth_composite" in panel:
        growth = _f("growth_composite")
        rows.append(_candidate("growth_momentum", "hybrid", "conditional_growth", "growth_diffusion", _b("mul", growth, momentum20), [growth, momentum20]))
    return rows


def detect_train_phenomena(panel: dict[str, pd.DataFrame], split: Split) -> dict[str, Any]:
    returns = panel["returns"].loc[split.train]
    dispersion = returns.std(axis=1)
    market = returns.median(axis=1)
    return {
        "data_scope": "train_only",
        "cross_section_dispersion_median": round(float(dispersion.median()), 8),
        "market_autocorrelation_1": round(float(market.autocorr(1)), 8),
        "fundamental_field_count": sum(field in panel for field in ("quality_composite", "growth_composite", "book_to_price", "safety_composite")),
        "questions": [
            "Does slow information diffusion survive industry/size neutralisation?",
            "Does forced-sale reversal beat scale-matched and permutation controls?",
            "Do fundamental signals add beyond their technical or fundamental legs?",
        ],
    }


def make_split(index: pd.DatetimeIndex, config: dict[str, Any]) -> Split:
    validation, shadow, purge = int(config["validation_days"]), int(config["shadow_days"]), int(config["purge_days"])
    minimum_train = int(config["minimum_train_days"])
    if purge < int(config["prediction_horizon_days"]):
        raise ValueError("purge must cover the label horizon")
    required = minimum_train + validation + shadow + 2 * purge
    if len(index) < required:
        raise ValueError(f"need at least {required} dates, found {len(index)}")
    shadow_start = len(index) - shadow
    validation_end = shadow_start - purge
    validation_start = validation_end - validation
    train_end = validation_start - purge
    positions = np.arange(len(index))
    train = pd.Series(positions < train_end, index=index)
    validation_mask = pd.Series((positions >= validation_start) & (positions < validation_end), index=index)
    shadow_mask = pd.Series(positions >= shadow_start, index=index)
    return Split(
        train,
        validation_mask,
        shadow_mask,
        {
            "train": [str(index[0].date()), str(index[train_end - 1].date())],
            "validation": [str(index[validation_start].date()), str(index[validation_end - 1].date())],
            "shadow": "SHA256_COMMITMENT_ONLY",
            "purge_days": purge,
            "label_horizon_days": int(config["prediction_horizon_days"]),
        },
    )


def _rank_ic(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    return signal.rank(axis=1, pct=True).corrwith(target.rank(axis=1, pct=True), axis=1)


def newey_west_t(values: pd.Series, lags: int) -> float | None:
    x = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(x) < max(20, lags + 3):
        return None
    centred = x - x.mean()
    long_variance = float(np.dot(centred, centred) / len(x))
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centred[lag:], centred[:-lag]) / len(x))
        long_variance += 2 * (1 - lag / (lags + 1)) * covariance
    standard_error = math.sqrt(max(long_variance, 1e-20) / len(x))
    return round(float(x.mean() / standard_error), 4)


def _tradability(panel: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum_amount = float(config.get("minimum_daily_amount", 0.0))
    common = (
        ~panel["is_st"].astype(bool)
        & ~panel["is_suspended"].astype(bool)
        & ~panel["is_delisted"].astype(bool)
        & panel["volume"].gt(0)
        & panel["amount"].ge(minimum_amount)
    )
    # Signal at close t enters on t+1, so next-session feasibility is the relevant mask.
    buyable = (common & ~panel["limit_up"].astype(bool)).shift(-1, fill_value=False)
    shortable_proxy = (common & ~panel["limit_down"].astype(bool)).shift(
        -1, fill_value=False
    )
    return buyable.astype(bool), shortable_proxy.astype(bool)


def neutral_factor_weights(
    signal: pd.DataFrame, panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> pd.DataFrame:
    ranks = signal.rank(axis=1, pct=True)
    centred = ranks.sub(ranks.mean(axis=1), axis=0)
    size = np.log(panel["market_cap"].where(panel["market_cap"] > 0))
    industry = panel["industry"]
    buyable, shortable = _tradability(panel, config)
    output = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date in signal.index:
        row = centred.loc[date]
        valid = (
            row.notna()
            & size.loc[date].notna()
            & industry.loc[date].notna()
            & buyable.loc[date]
            & shortable.loc[date]
        )
        if int(valid.sum()) < 10:
            continue
        names = row.index[valid]
        y = row.loc[names].to_numpy(dtype=float)
        log_size = size.loc[date, names].to_numpy(dtype=float)
        log_size = (log_size - np.nanmean(log_size)) / max(np.nanstd(log_size), 1e-12)
        dummies = pd.get_dummies(industry.loc[date, names].astype(str), drop_first=True, dtype=float)
        design = np.column_stack([np.ones(len(names)), log_size, dummies.to_numpy()])
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        gross = float(np.abs(residual).sum())
        if gross > 0:
            output.loc[date, names] = residual / gross
    return output


def _returns_from_desired_weights(
    desired: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
    round_trip_cost: float,
) -> dict[str, Any]:
    horizon = int(config["prediction_horizon_days"])
    # Equal overlapping tranches: each close-t signal remains active for h sessions.
    effective = desired.rolling(horizon, min_periods=1).mean()
    turnover = effective.diff().abs().sum(axis=1).fillna(effective.abs().sum(axis=1))
    gross = (effective * one_day).sum(axis=1)
    costed = gross - turnover * round_trip_cost / 2.0
    changes = effective.diff().abs()
    participation = float(config.get("maximum_amount_participation", 0.01))
    capacity = (panel["amount"] * participation / changes.replace(0.0, np.nan)).min(axis=1)
    return {"gross": gross, "costed": costed, "turnover": turnover, "weights": effective, "capacity": capacity}


def _costed_from_desired_array(
    weights: np.ndarray,
    one_day_values: np.ndarray,
    horizon: int,
    round_trip_cost: float,
) -> np.ndarray:
    """NumPy-equivalent holding-tranche and cost calculation used by placebos."""
    cumulative = np.vstack(
        [np.zeros((1, weights.shape[1])), np.cumsum(weights, axis=0)]
    )
    starts = np.maximum(0, np.arange(len(weights)) - horizon + 1)
    totals = cumulative[np.arange(1, len(weights) + 1)] - cumulative[starts]
    counts = np.minimum(np.arange(1, len(weights) + 1), horizon)[:, None]
    effective = totals / counts
    changes = np.vstack([np.abs(effective[0]), np.abs(np.diff(effective, axis=0))])
    safe_returns = np.nan_to_num(one_day_values, nan=0.0, posinf=0.0, neginf=0.0)
    gross = np.sum(effective * safe_returns, axis=1)
    turnover = np.sum(changes, axis=1)
    return gross - turnover * round_trip_cost / 2.0


def factor_portfolio(
    signal: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
    round_trip_cost: float | None = None,
) -> dict[str, Any]:
    desired = neutral_factor_weights(signal, panel, config)
    result = _returns_from_desired_weights(
        desired,
        one_day,
        panel,
        config,
        float(config["round_trip_cost"] if round_trip_cost is None else round_trip_cost),
    )
    result["desired_weights"] = desired
    return result


def _stats(values: pd.Series, mask: pd.Series) -> dict[str, Any]:
    x = values.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
    std = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    equity = (1.0 + x).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    maximum_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    annual_return = float(x.mean() * 244) if len(x) else np.nan
    return {
        "n": int(len(x)),
        "mean": None if not len(x) else round(float(x.mean()), 8),
        "ir_ann": None if not np.isfinite(std) or std == 0 else round(float(x.mean() / std * np.sqrt(244)), 4),
        "hit_rate": None if not len(x) else round(float((x > 0).mean()), 4),
        "max_drawdown": None if not np.isfinite(maximum_drawdown) else round(maximum_drawdown, 6),
        "calmar": None if not np.isfinite(maximum_drawdown) or maximum_drawdown >= 0 else round(annual_return / abs(maximum_drawdown), 4),
    }


def permutation_test(
    primary: dict[str, Any],
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    validation_mask: pd.Series,
    config: dict[str, Any],
    factor_seed: int,
) -> dict[str, Any]:
    repetitions = int(config["placebo_repetitions"])
    observed = float(primary["costed"].loc[validation_mask].mean())
    validation_dates = validation_mask.index[validation_mask]
    first = max(
        0,
        primary["desired_weights"].index.get_loc(validation_dates[0])
        - int(config["prediction_horizon_days"]),
    )
    last = primary["desired_weights"].index.get_loc(validation_dates[-1])
    active_dates = primary["desired_weights"].index[first : last + 1]
    columns = primary["desired_weights"].columns
    base_weights = primary["desired_weights"].loc[active_dates].to_numpy(dtype=float)
    one_day_values = one_day.loc[active_dates, columns].to_numpy(dtype=float)
    log_size = np.log(panel["market_cap"].where(panel["market_cap"] > 0))
    industry = panel["industry"]
    grouped_locations: list[list[np.ndarray]] = []
    for date in active_dates:
        size_bucket = pd.qcut(
            log_size.loc[date].rank(method="first"),
            5,
            labels=False,
            duplicates="drop",
        )
        groups = pd.DataFrame({"industry": industry.loc[date], "size": size_bucket})
        grouped_locations.append(
            [
                locations
                for members in groups.groupby(
                    ["industry", "size"], dropna=True
                ).groups.values()
                if len(
                    locations := columns.get_indexer(list(members))
                )
                > 1
            ]
        )
    horizon = int(config["prediction_horizon_days"])
    validation_positions = np.array(
        [active_dates.get_loc(date) for date in validation_dates], dtype=int
    )

    placebo_means: list[float] = []
    placebo_returns: list[pd.Series] = []
    for iteration in range(repetitions):
        desired = base_weights.copy()
        rng = np.random.default_rng(factor_seed + iteration * 100_003)
        for position, groups in enumerate(grouped_locations):
            source = base_weights[position]
            for locations in groups:
                desired[position, locations] = rng.permutation(source[locations])
        values = _costed_from_desired_array(
            desired,
            one_day_values,
            horizon,
            float(config["round_trip_cost"]),
        )[validation_positions]
        placebo_means.append(float(np.mean(values)))
        if iteration < 20:
            placebo_returns.append(
                pd.Series(
                    values,
                    index=validation_dates,
                    name=f"placebo_{iteration:02d}",
                )
            )
    p_value = (1 + sum(value >= observed for value in placebo_means)) / (repetitions + 1)
    return {
        "repetitions": repetitions,
        "empirical_p_value": round(float(p_value), 6),
        "observed_validation_costed_mean": round(observed, 8),
        "placebo_mean_quantiles": {
            str(q): round(float(np.quantile(placebo_means, q)), 8) for q in (0.05, 0.5, 0.95)
        },
        "sample_returns": placebo_returns,
    }


def pbo(candidate_returns: pd.DataFrame, blocks: int = 8, focus: str | None = None) -> dict[str, Any]:
    data = candidate_returns.dropna(how="all")
    if data.shape[1] < 2 or len(data) < blocks * 5:
        return {"pbo": None, "splits": 0, "focus_selection_splits": 0}
    partitions = np.array_split(np.arange(len(data)), blocks)
    failures, focus_selections = [], 0
    for chosen in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([partitions[index] for index in chosen])
        outside = np.concatenate([partitions[index] for index in range(blocks) if index not in chosen])
        is_ir = data.iloc[inside].mean() / data.iloc[inside].std().replace(0.0, np.nan)
        winner = is_ir.idxmax()
        if focus is not None and winner != focus:
            continue
        focus_selections += int(focus is not None)
        oos_ir = data.iloc[outside].mean() / data.iloc[outside].std().replace(0.0, np.nan)
        percentile = float(oos_ir.rank(pct=True).get(winner, np.nan))
        if np.isfinite(percentile):
            failures.append(percentile <= 0.5)
    return {
        "pbo": round(float(np.mean(failures)), 4) if failures else None,
        "splits": len(failures),
        "focus_selection_splits": focus_selections,
    }


def deflated_sharpe_ratio(
    returns: pd.Series, trial_daily_sharpes: list[float], declared_trials: int
) -> dict[str, Any]:
    x = returns.dropna().to_numpy(dtype=float)
    trials = max(len(trial_daily_sharpes), int(declared_trials), 2)
    if len(x) < 30 or np.std(x, ddof=1) == 0:
        return {"probability": None, "trials": trials}
    daily_sr = float(np.mean(x) / np.std(x, ddof=1))
    trial_scale = float(np.std(trial_daily_sharpes, ddof=1)) if len(trial_daily_sharpes) > 1 else 0.0
    gamma, normal = 0.5772156649, NormalDist()
    expected_max = trial_scale * (
        (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
        + gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
    )
    skew = float(pd.Series(x).skew())
    kurtosis = float(pd.Series(x).kurtosis() + 3.0)
    denominator = math.sqrt(max(1e-12, 1 - skew * daily_sr + (kurtosis - 1) * daily_sr**2 / 4))
    z = (daily_sr - expected_max) * math.sqrt(len(x) - 1) / denominator
    return {
        "probability": round(float(normal.cdf(z)), 6),
        "observed_sharpe_ann": round(daily_sr * math.sqrt(244), 4),
        "expected_max_trial_sharpe_ann": round(expected_max * math.sqrt(244), 4),
        "trial_sharpe_std_ann": round(trial_scale * math.sqrt(244), 4),
        "trials": trials,
    }


def _train_score(
    candidate: dict[str, Any], panel: dict[str, pd.DataFrame], target: pd.DataFrame, split: Split
) -> tuple[float, float, pd.DataFrame, pd.DataFrame] | None:
    try:
        validate(candidate["expression"], set(panel), WINDOWS)
        signal = evaluate(candidate["expression"], panel).replace([np.inf, -np.inf], np.nan)
    except (KeyError, ValueError):
        return None
    ic = _rank_ic(signal, target).loc[split.train].dropna()
    if len(ic) < 60:
        return None
    direction = 1.0 if ic.mean() >= 0 else -1.0
    return abs(float(ic.mean())), direction, signal * direction, signal


def purged_walk_forward(
    raw_signal: pd.DataFrame,
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    split: Split,
    config: dict[str, Any],
    fitted_direction: float,
    fitted_portfolio: dict[str, Any],
) -> dict[str, Any]:
    """Fit direction on each fold's past and evaluate costed returns on its future."""
    train_dates = split.train.index[split.train]
    purge = int(config["purge_days"])
    folds = int(config.get("walk_forward_folds", 3))
    minimum_history = max(60, int(config["prediction_horizon_days"]) * 3)
    if len(train_dates) <= minimum_history + purge + folds:
        return {"folds": [], "positive_folds": 0, "eligible_folds": 0}
    test_blocks = np.array_split(train_dates[minimum_history + purge :], folds)
    portfolio_cache: dict[float, dict[str, Any]] = {fitted_direction: fitted_portfolio}
    reports: list[dict[str, Any]] = []
    for block in test_blocks:
        if len(block) == 0:
            continue
        first_position = train_dates.get_loc(block[0])
        fit_dates = train_dates[: max(0, first_position - purge)]
        ic = _rank_ic(raw_signal, target).loc[fit_dates].dropna()
        if len(ic) < 60:
            continue
        direction = 1.0 if ic.mean() >= 0 else -1.0
        if direction not in portfolio_cache:
            portfolio_cache[direction] = factor_portfolio(
                raw_signal * direction, one_day, panel, config
            )
        mask = pd.Series(False, index=split.train.index)
        mask.loc[block] = True
        metrics = _stats(portfolio_cache[direction]["costed"], mask)
        reports.append(
            {
                "test_start": str(pd.Timestamp(block[0]).date()),
                "test_end": str(pd.Timestamp(block[-1]).date()),
                "fit_observations": int(len(ic)),
                "direction": int(direction),
                "costed": metrics,
            }
        )
    return {
        "folds": reports,
        "positive_folds": sum(
            row["costed"]["mean"] is not None and row["costed"]["mean"] > 0
            for row in reports
        ),
        "eligible_folds": len(reports),
        "purge_days": purge,
        "direction_refit_past_only": True,
    }


def _generated_child(
    left: dict[str, Any],
    right: dict[str, Any],
    operator: str,
    window: int,
    attempt: int,
    search_paradigm: str | None = None,
) -> dict[str, Any]:
    if operator == "temporal_zscore":
        expression = {"zscore": True, "arg": left["expression"], "window": window}
        parent_ids = [left["factor_id"]]
    elif operator == "rolling_refinement":
        expression = _roll("mean", left["expression"], window)
        parent_ids = [left["factor_id"]]
    elif operator == "causal_lag_mutation":
        expression = {"lag": window, "arg": left["expression"]}
        parent_ids = [left["factor_id"]]
    elif operator == "mechanism_crossover_mul":
        expression = _b("mul", left["expression"], right["expression"])
        parent_ids = [left["factor_id"], right["factor_id"]]
    elif operator == "mechanism_crossover_add":
        expression = _b("add", left["expression"], right["expression"])
        parent_ids = [left["factor_id"], right["factor_id"]]
    else:
        raise ValueError(f"unknown bounded synthesis operator: {operator}")
    generation = 1 + max(
        int(left["birth_certificate"]["generation"]),
        int(right["birth_certificate"]["generation"]),
    )
    return _candidate(
        f"generated_{operator}_{attempt:03d}",
        "generated_hybrid" if len(parent_ids) > 1 else left["domain"],
        left["phenomenon"],
        f"{left['mechanism']} | {operator}",
        expression,
        (
            [left["expression"], right["expression"]]
            if len(parent_ids) > 1
            else left["counter_expressions"]
        ),
        parent_ids,
        generation,
        operator,
        search_paradigm,
    )


def bounded_generate_audited(
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    split: Split,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[float, float, pd.DataFrame, pd.DataFrame]],
    dict[str, Any],
]:
    budget = int(config["candidate_budget"])
    candidates = seed_library(panel)
    scored: dict[str, tuple[float, float, pd.DataFrame, pd.DataFrame]] = {}
    seen = {row["factor_id"] for row in candidates}

    def score_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            result = _train_score(row, panel, target, split)
            if result is not None:
                scored[row["factor_id"]] = result

    score_rows(candidates)
    protocol = config.get("search_protocol")
    if protocol is not None:
        search.validate_search_protocol(protocol, budget)
        benchmark = search.empty_benchmark()
        generation_audit: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate["search_paradigm"] = "seed"
            search.record_candidate(
                benchmark,
                "seed",
                unique=True,
                fast_screen_passed=candidate["factor_id"] in scored,
            )
        attempt = 0
        for scheduled_generation in range(1, len(protocol["generation_schedule"]) + 1):
            if len(candidates) >= budget:
                break
            paradigm = search.paradigm_for_generation(scheduled_generation, protocol)
            frontier, comparisons = search.select_frontier(
                candidates, scored, paradigm, protocol
            )
            generation_row = {
                "scheduled_generation": scheduled_generation,
                "search_paradigm": paradigm,
                "frontier": [candidate["factor_id"] for candidate in frontier],
                "pairwise_train_comparisons": comparisons,
                "selection_evidence": "train_only",
                "validation_feedback_used": False,
                "shadow_feedback_used": False,
                "generated_children": 0,
            }
            generation_audit.append(generation_row)
            if not frontier:
                break
            children_per_parent = int(
                protocol["arms"][paradigm]["children_per_parent"]
            )
            for parent_index, left in enumerate(frontier):
                for child_index in range(children_per_parent):
                    if len(candidates) >= budget or attempt >= budget * 30:
                        break
                    right = frontier[(parent_index + child_index + 1) % len(frontier)]
                    operator = search.choose_operator(
                        protocol,
                        paradigm,
                        scheduled_generation,
                        parent_index,
                        child_index,
                        int(config["random_seed"]),
                    )
                    window = (5, 10, 20, 60, 120)[
                        (attempt + scheduled_generation + child_index) % 5
                    ]
                    child = _generated_child(
                        left,
                        right,
                        operator,
                        window,
                        attempt,
                        paradigm,
                    )
                    attempt += 1
                    generation_row["generated_children"] += 1
                    if child["factor_id"] in seen:
                        search.record_candidate(
                            benchmark,
                            paradigm,
                            unique=False,
                            fast_screen_passed=False,
                        )
                        continue
                    try:
                        validate(child["expression"], set(panel), WINDOWS)
                    except ValueError:
                        search.record_candidate(
                            benchmark,
                            paradigm,
                            unique=False,
                            fast_screen_passed=False,
                        )
                        continue
                    seen.add(child["factor_id"])
                    candidates.append(child)
                    score_rows([child])
                    child_result = scored.get(child["factor_id"])
                    comparable_parent_scores = [
                        scored[parent_id][0]
                        for parent_id in child["birth_certificate"]["parents"]
                        if parent_id in scored
                    ]
                    search.record_candidate(
                        benchmark,
                        paradigm,
                        unique=True,
                        fast_screen_passed=child_result is not None,
                        child_score=None if child_result is None else child_result[0],
                        parent_score=(
                            max(comparable_parent_scores)
                            if comparable_parent_scores
                            else None
                        ),
                    )
                if len(candidates) >= budget or attempt >= budget * 30:
                    break
        return candidates[:budget], scored, {
            "enabled": True,
            "schema_version": protocol["schema_version"],
            "generation_schedule": list(protocol["generation_schedule"]),
            "proposal_role": protocol["proposal_role"],
            "absolute_factor_judgement_allowed": False,
            "final_judge": protocol["final_judge"],
            "generation_audit": generation_audit,
            "_benchmark": benchmark,
        }

    attempt = 0
    while len(candidates) < budget and attempt < budget * 30 and scored:
        parents = sorted(
            (row for row in candidates if row["factor_id"] in scored),
            key=lambda row: scored[row["factor_id"]][0],
            reverse=True,
        )[: max(2, int(config.get("parent_pool_size", 8)))]
        left = parents[attempt % len(parents)]
        right = parents[(attempt * 3 + 1) % len(parents)]
        operation = attempt % 5
        window = (5, 10, 20, 60, 120)[attempt % 5]
        operator = (
            "temporal_zscore",
            "rolling_refinement",
            "causal_lag_mutation",
            "mechanism_crossover_mul",
            "mechanism_crossover_add",
        )[operation]
        child = _generated_child(
            left, right, operator, window, attempt
        )
        attempt += 1
        if child["factor_id"] in seen:
            continue
        try:
            validate(child["expression"], set(panel), WINDOWS)
        except ValueError:
            continue
        seen.add(child["factor_id"])
        candidates.append(child)
        score_rows([child])
    return candidates[:budget], scored, {"enabled": False}


def bounded_generate(
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    split: Split,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[float, float, pd.DataFrame, pd.DataFrame]],
]:
    """Backward-compatible candidate generator without the optional audit payload."""
    candidates, scored, _ = bounded_generate_audited(panel, target, split, config)
    return candidates, scored


def _signal_correlation(left: pd.DataFrame, right: pd.DataFrame, mask: pd.Series) -> float | None:
    values = left.loc[mask].corrwith(right.loc[mask], axis=1).dropna()
    return None if values.empty else round(float(values.mean()), 6)


def _ic_decay(signal: pd.DataFrame, panel: dict[str, pd.DataFrame], mask: pd.Series) -> dict[str, Any]:
    output = {}
    for horizon in (1, 5, 10, 20):
        target = panel["open"].shift(-(horizon + 1)) / panel["open"].shift(-1) - 1.0
        ic = _rank_ic(signal, target).loc[mask].dropna()
        output[str(horizon)] = {
            "mean": None if ic.empty else round(float(ic.mean()), 8),
            "newey_west_t": newey_west_t(ic, horizon),
        }
    return output


def _quantile_monotonicity(
    signal: pd.DataFrame, target: pd.DataFrame, mask: pd.Series
) -> dict[str, Any]:
    ranks = signal.rank(axis=1, pct=True).loc[mask]
    values = target.loc[mask]
    means = []
    for index in range(5):
        selected = values.where((ranks > index / 5) & (ranks <= (index + 1) / 5))
        means.append(float(selected.stack().mean()))
    monotonicity = pd.Series(means).corr(pd.Series(range(5)), method="spearman")
    return {"quintile_means": [round(value, 8) for value in means], "spearman": round(float(monotonicity), 6)}


def _hash_frame(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _run_manifest(
    prices: pd.DataFrame, statements: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    return {
        "schema_version": "xalpha_lite_run_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "price_data_sha256": _hash_frame(prices),
        "fundamental_data_sha256": _hash_frame(statements),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def run_discovery(
    prices: pd.DataFrame,
    statements: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if config.get("status") != "research_only_not_trading":
        raise ValueError("the public lite package is permanently research-only")
    panel = build_panel(prices, statements, config)
    split = make_split(panel["close"].index, config)
    horizon = int(config["prediction_horizon_days"])
    target = panel["open"].shift(-(horizon + 1)) / panel["open"].shift(-1) - 1.0
    one_day = panel["open"].shift(-2) / panel["open"].shift(-1) - 1.0
    research_plan = detect_train_phenomena(panel, split)
    candidates, train_scored, search_audit = bounded_generate_audited(
        panel, target, split, config
    )
    signal_cache = {factor_id: values[2] for factor_id, values in train_scored.items()}
    ranked = sorted(
        (row for row in candidates if row["factor_id"] in train_scored),
        key=lambda row: train_scored[row["factor_id"]][0],
        reverse=True,
    )
    stage2: list[dict[str, Any]] = []
    correlation_pruned: list[dict[str, Any]] = []
    maximum_behaviour_correlation = float(
        config["maximum_factor_behavior_correlation"]
    )
    for candidate in ranked:
        conflict = None
        conflict_correlation = None
        for selected in stage2:
            correlation = _signal_correlation(
                signal_cache[candidate["factor_id"]],
                signal_cache[selected["factor_id"]],
                split.train,
            )
            if correlation is not None and abs(correlation) >= maximum_behaviour_correlation:
                conflict = selected["factor_id"]
                conflict_correlation = correlation
                break
        if conflict is not None:
            correlation_pruned.append(
                {
                    "factor_id": candidate["factor_id"],
                    "conflicts_with": conflict,
                    "train_signal_correlation": conflict_correlation,
                }
            )
            continue
        stage2.append(candidate)
        if len(stage2) >= int(config["stage2_budget"]):
            break
    evaluated_rows: list[dict[str, Any]] = []
    validation_returns: dict[str, pd.Series] = {}
    validation_sharpes: list[float] = []
    shadow_commitment_rows: list[dict[str, Any]] = []
    for candidate in stage2:
        factor_id = candidate["factor_id"]
        train_ic, direction, signal, raw_signal = train_scored[factor_id]
        primary = factor_portfolio(signal, one_day, panel, config)
        walk_forward = purged_walk_forward(
            raw_signal,
            target,
            one_day,
            panel,
            split,
            config,
            direction,
            primary,
        )
        counter_results = []
        counter_signals = []
        counter_train_metrics = []
        for expression in candidate["counter_expressions"]:
            raw_counter = evaluate(expression, panel).replace(
                [np.inf, -np.inf], np.nan
            )
            counter_ic = _rank_ic(raw_counter, target).loc[split.train].dropna()
            if len(counter_ic) < 60:
                continue
            counter_direction = 1.0 if counter_ic.mean() >= 0 else -1.0
            counter = raw_counter * counter_direction
            counter_signals.append(counter)
            counter_results.append(factor_portfolio(counter, one_day, panel, config))
            counter_train_metrics.append(
                {
                    "direction_learned_on_train": int(counter_direction),
                    "train_rank_ic": round(abs(float(counter_ic.mean())), 8),
                }
            )
        counter_ir_values = [
            _stats(result["costed"], split.validation)["ir_ann"] for result in counter_results
        ]
        strongest_counter_index = max(
            range(len(counter_results)),
            key=lambda index: -np.inf if counter_ir_values[index] is None else counter_ir_values[index],
            default=None,
        )
        strongest_counter = None if strongest_counter_index is None else counter_results[strongest_counter_index]
        counter_metrics = (
            {
                "ir_ann": None,
                "mean": None,
                "direction_learned_on_train": None,
                "train_rank_ic": None,
            }
            if strongest_counter is None
            else {
                **_stats(strongest_counter["costed"], split.validation),
                **counter_train_metrics[strongest_counter_index],
            }
        )
        counter_correlation = (
            None
            if strongest_counter_index is None
            else _signal_correlation(signal, counter_signals[strongest_counter_index], split.validation)
        )
        placebo = permutation_test(
            primary,
            one_day,
            panel,
            split.validation,
            config,
            int(config["random_seed"]) ^ int(factor_id[-8:], 16),
        )
        relative_returns = {"primary": primary["costed"]}
        if strongest_counter is not None:
            relative_returns["counter"] = strongest_counter["costed"]
        for series in placebo.pop("sample_returns"):
            relative_returns[series.name] = series
        factor_pbo = pbo(
            pd.DataFrame(relative_returns).loc[split.validation],
            int(config["pbo_blocks"]),
            focus="primary",
        )
        primary_validation = _stats(primary["costed"], split.validation)
        gross_validation = _stats(primary["gross"], split.validation)
        validation_returns[factor_id] = primary["costed"]
        x = primary["costed"].loc[split.validation].dropna()
        if len(x) > 1 and x.std() > 0:
            validation_sharpes.append(float(x.mean() / x.std()))
        cost_curve = {
            str(bps): _stats(
                _returns_from_desired_weights(
                    primary["desired_weights"],
                    one_day,
                    panel,
                    config,
                    bps / 10_000,
                )["costed"],
                split.validation,
            )
            for bps in (0, 10, 30, 50)
        }
        capacity = primary["capacity"].loc[split.validation].replace([np.inf, -np.inf], np.nan).dropna()
        row = {
            **{key: value for key, value in candidate.items() if key != "counter_expressions"},
            "direction_learned_on_train": int(direction),
            "train_rank_ic": round(train_ic, 8),
            "validation": {
                "gross": gross_validation,
                "costed": primary_validation,
                "ic_decay": _ic_decay(signal, panel, split.validation),
                "quantile_monotonicity": _quantile_monotonicity(signal, target, split.validation),
                "turnover_mean": round(float(primary["turnover"].loc[split.validation].mean()), 6),
                "capacity_median_cny": None if capacity.empty else round(float(capacity.median()), 2),
                "cost_sensitivity_bps": cost_curve,
            },
            "counter": {**counter_metrics, "signal_correlation": counter_correlation},
            "placebo_distribution": placebo,
            "factor_relative_pbo": factor_pbo,
            "purged_walk_forward": walk_forward,
        }
        evaluated_rows.append(row)
        shadow_commitment_rows.append(
            {
                "factor_id": factor_id,
                "shadow_costed": _stats(primary["costed"], split.shadow),
                "shadow_gross": _stats(primary["gross"], split.shadow),
            }
        )

    global_pbo = pbo(
        pd.DataFrame(validation_returns).loc[split.validation], int(config["pbo_blocks"])
    )
    pbo_pass = global_pbo["pbo"] is not None and global_pbo["pbo"] <= float(config["pbo_maximum"])
    declared_trials = int(config["previous_research_trials"]) + len(candidates)
    for row in evaluated_rows:
        primary = row["validation"]["costed"]
        row["dsr"] = deflated_sharpe_ratio(
            validation_returns[row["factor_id"]].loc[split.validation],
            validation_sharpes,
            declared_trials,
        )
        counter_mean = row["counter"]["mean"]
        guards = {
            "train_rank_ic_pass": row["train_rank_ic"] >= float(config["minimum_train_rank_ic"]),
            "positive_after_cost": primary["mean"] is not None and primary["mean"] > 0 and primary["ir_ann"] is not None and primary["ir_ann"] > 0,
            "counter_distinct": row["counter"]["signal_correlation"] is None or abs(row["counter"]["signal_correlation"]) < float(config["maximum_counter_correlation"]),
            "counter_beaten_after_cost": counter_mean is not None and primary["mean"] is not None and primary["mean"] > counter_mean,
            "permutation_p_value_pass": row["placebo_distribution"]["empirical_p_value"] <= float(config["permutation_alpha"]),
            "factor_relative_pbo_pass": row["factor_relative_pbo"]["pbo"] is not None and row["factor_relative_pbo"]["pbo"] <= float(config["pbo_maximum"]),
            "project_pbo_pass": pbo_pass,
            "dsr_pass": row["dsr"].get("probability") is not None and row["dsr"]["probability"] >= float(config["dsr_probability_minimum"]),
            "walk_forward_pass": row["purged_walk_forward"]["eligible_folds"] >= int(config["minimum_positive_walk_forward_folds"]) and row["purged_walk_forward"]["positive_folds"] >= int(config["minimum_positive_walk_forward_folds"]),
        }
        row["guards"] = guards
        row["validation_survivor"] = all(guards.values())

    shadow_text = json.dumps(shadow_commitment_rows, sort_keys=True, separators=(",", ":"))
    shadow_hash = hashlib.sha256(shadow_text.encode()).hexdigest()
    # Pairwise train-only behaviour correlations for duplicate/orthogonalisation audit.
    correlations: dict[str, float | None] = {}
    for left, right in combinations([row["factor_id"] for row in evaluated_rows], 2):
        correlations[f"{left}|{right}"] = _signal_correlation(
            signal_cache[left], signal_cache[right], split.train
        )
    hard_stop = None if pbo_pass else "GLOBAL_VALIDATION_PBO_FAILED_NO_SURVIVORS_ALLOWED"
    if hard_stop:
        for row in evaluated_rows:
            row["validation_survivor"] = False
    if search_audit["enabled"]:
        benchmark = search_audit.pop("_benchmark")
        for row in evaluated_rows:
            search.record_stage2(
                benchmark,
                str(row.get("search_paradigm", "seed")),
                bool(row["validation_survivor"]),
            )
        search_audit["benchmark"] = search.finalise_benchmark(benchmark)
    result = {
        "schema_version": "perception_xalpha_lite_result_v2",
        "status": "diagnostic_only_research_only_not_trading",
        "run_manifest": _run_manifest(prices, statements, config),
        "research_plan": research_plan,
        "split": split.audit,
        "candidate_count": len(candidates),
        "stage2_count": len(evaluated_rows),
        "stage2_correlation_pruning": {
            "maximum_absolute_train_correlation": maximum_behaviour_correlation,
            "pruned": correlation_pruned,
        },
        "validation_survivor_count": sum(row["validation_survivor"] for row in evaluated_rows),
        "hard_stop": hard_stop,
        "project_validation_pbo": global_pbo,
        "shadow_commitment_sha256": shadow_hash,
        "shadow_metrics_disclosed": False,
        "trial_ledger": {
            "previous_research_trials": int(config["previous_research_trials"]),
            "current_candidates": len(candidates),
            "declared_total_trials": declared_trials,
        },
        "search_protocol": search_audit,
        "train_factor_correlation": correlations,
        "mechanism_tree": [row["birth_certificate"] for row in candidates],
        "candidates": evaluated_rows,
        "orders": [],
        "automatic_trading_changes": [],
        "warning": (
            "A zero-investment factor portfolio measures research alpha, not an executable "
            "A-share short book. Historical survivors remain forward-shadow hypotheses."
        ),
    }
    return result


def write_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
