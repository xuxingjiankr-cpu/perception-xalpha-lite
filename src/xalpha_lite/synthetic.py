"""Deterministic synthetic inputs for a zero-setup research demonstration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_synthetic_data(
    days: int = 780,
    symbols: int = 30,
    seed: int = 20_260_816,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a toy market that satisfies the public point-in-time data contract."""
    if days < 20 or symbols < 4:
        raise ValueError("the synthetic market needs at least 20 sessions and 4 symbols")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=days)
    names = [f"S{index:04d}" for index in range(symbols)]
    latent_quality = rng.normal(size=symbols)
    market = rng.normal(0.0002, 0.009, size=days)
    returns = market[:, None] + rng.normal(0, 0.014, size=(days, symbols))
    returns += 0.00015 * latent_quality[None, :]
    close = 20 * np.exp(np.cumsum(returns, axis=0))
    open_price = close * np.exp(rng.normal(0, 0.002, size=close.shape))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.01, size=close.shape))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.01, size=close.shape))
    volume = rng.lognormal(15, 0.5, size=close.shape)
    amount = volume * (open_price + high + low + close) / 4
    rows: list[dict[str, Any]] = []
    for date_position, date in enumerate(dates):
        for symbol_position, symbol in enumerate(names):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "symbol": symbol,
                    "open": open_price[date_position, symbol_position],
                    "high": high[date_position, symbol_position],
                    "low": low[date_position, symbol_position],
                    "close": close[date_position, symbol_position],
                    "volume": volume[date_position, symbol_position],
                    "amount": amount[date_position, symbol_position],
                    "industry": f"industry_{symbol_position % 5}",
                    "market_cap": (5e9 + symbol_position * 2e8)
                    * close[date_position, symbol_position]
                    / close[0, symbol_position],
                    "is_st": False,
                    "is_suspended": False,
                    "is_delisted": False,
                    "limit_up": False,
                    "limit_down": False,
                }
            )
    statements: list[dict[str, Any]] = []
    for report_position in range(60, days, 63):
        report_date = dates[report_position]
        notice_date = dates[min(days - 2, report_position + 30)]
        for symbol_position, symbol in enumerate(names):
            quality = latent_quality[symbol_position] + rng.normal(0, 0.2)
            statements.append(
                {
                    "symbol": symbol,
                    "report_date": report_date.date().isoformat(),
                    "notice_date": notice_date.date().isoformat(),
                    "update_date": notice_date.date().isoformat(),
                    "eps_ytd": max(0.01, 0.5 + 0.2 * quality),
                    "bps": max(0.1, 4 + quality),
                    "roe": 10 + 3 * quality,
                    "roic": 8 + 2 * quality,
                    "gross_margin": 25 + 4 * quality,
                    "cash_to_profit": 1 + 0.15 * quality,
                    "revenue_yoy": 8 + 4 * quality,
                    "profit_yoy": 7 + 5 * quality,
                    "debt_ratio": 45 - 5 * quality,
                    "current_ratio": 1.5 + 0.2 * quality,
                    "quick_ratio": 1.1 + 0.15 * quality,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(statements)


def demo_config(seed: int = 20_260_816) -> dict[str, Any]:
    """A compact full-pipeline configuration whose outputs remain explicitly synthetic."""
    return {
        "status": "research_only_not_trading",
        "prediction_horizon_days": 10,
        "purge_days": 10,
        "validation_days": 100,
        "shadow_days": 100,
        "minimum_train_days": 504,
        "round_trip_cost": 0.003,
        "require_neutralization_data": True,
        "minimum_daily_amount": 1_000_000,
        "maximum_amount_participation": 0.01,
        "candidate_budget": 20,
        "stage2_budget": 6,
        "parent_pool_size": 6,
        "minimum_train_rank_ic": 0.005,
        "walk_forward_folds": 3,
        "minimum_positive_walk_forward_folds": 2,
        "pbo_blocks": 6,
        "pbo_maximum": 0.2,
        "placebo_repetitions": 40,
        "permutation_alpha": 0.05,
        "maximum_counter_correlation": 0.9,
        "maximum_factor_behavior_correlation": 0.85,
        "dsr_probability_minimum": 0.95,
        "previous_research_trials": 0,
        "random_seed": seed,
    }


__all__ = ["demo_config", "make_synthetic_data"]
