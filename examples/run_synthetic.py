"""Generate a deterministic toy market and run the full research pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xalpha_lite.discovery import run_discovery, write_result


def make_data(days: int = 900, symbols: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260803)
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
    records = []
    for date_position, date in enumerate(dates):
        for symbol_position, symbol in enumerate(names):
            records.append(
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
    statements = []
    for report_position in range(60, days, 63):
        report_date = dates[report_position]
        notice_position = min(days - 2, report_position + 30)
        notice_date = dates[notice_position]
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
    return pd.DataFrame(records), pd.DataFrame(statements)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/example.json").read_text(encoding="utf-8"))
    prices, fundamentals = make_data()
    result = run_discovery(prices, fundamentals, config)
    output = root / "outputs/synthetic_result.json"
    write_result(result, output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "candidate_count",
                    "validation_survivor_count",
                    "project_validation_pbo",
                )
            },
            indent=2,
        )
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
