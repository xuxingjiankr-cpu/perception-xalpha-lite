"""Bounded factor synthesis, causal validation and anti-overfitting guards."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .dsl import canonical, evaluate, fields_used, validate
from .pit import align_point_in_time_fundamentals


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


def build_panel(prices: pd.DataFrame, statements: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices missing: {sorted(required - set(prices.columns))}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"])
    panel = {
        field: frame.pivot(index="date", columns="symbol", values=field).sort_index()
        for field in ("open", "high", "low", "close", "volume", "amount")
    }
    close = panel["close"]
    panel["returns"] = close.pct_change(fill_method=None)
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    market = panel["returns"].median(axis=1)
    panel["market_return"] = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], len(close.columns), axis=1),
        index=close.index,
        columns=close.columns,
    )
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


def candidate_library(panel: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Economically named hypotheses, not a blind parameter grid."""
    ret, close, amount = _f("returns"), _f("close"), _f("amount")
    momentum20 = _roll("sum", ret, 20)
    reversal5 = _u("neg", _roll("sum", ret, 5))
    volatility20 = _roll("std", ret, 20)
    low_beta = _u(
        "neg", {"corr": True, "left": ret, "right": _f("market_return"), "window": 60}
    )
    technical = [
        ("T01", "momentum_20", "technical", momentum20),
        ("T02", "reversal_5", "technical", reversal5),
        ("T03", "low_beta_60", "technical", low_beta),
        ("T04", "low_volatility_20", "technical", _u("neg", volatility20)),
        ("T05", "liquidity_20", "technical", _roll("mean", _u("signed_log1p", amount), 20)),
        ("T06", "price_range_position", "technical", _b("div", close, _roll("max", close, 120))),
    ]
    fundamental_map = [
        ("F01", "book_to_price", "fundamental", "book_to_price"),
        ("F02", "earnings_yield_proxy", "fundamental", "earnings_yield_proxy"),
        ("F03", "quality", "fundamental", "quality_composite"),
        ("F04", "growth", "fundamental", "growth_composite"),
        ("F05", "financial_safety", "fundamental", "safety_composite"),
        ("F06", "cash_conversion", "fundamental", "cash_to_profit"),
        ("F07", "negative_leverage", "fundamental", "debt_ratio"),
    ]
    fundamental = [
        (identifier, name, domain, _u("neg", _f(field)) if name == "negative_leverage" else _f(field))
        for identifier, name, domain, field in fundamental_map
        if field in panel
    ]
    hybrids: list[tuple[str, str, str, dict[str, Any]]] = []
    if "quality_composite" in panel:
        quality = _f("quality_composite")
        hybrids += [
            ("H01", "quality_low_beta", "hybrid", _b("mul", quality, low_beta)),
            ("H02", "quality_momentum", "hybrid", _b("mul", quality, momentum20)),
            ("H03", "quality_per_volatility", "hybrid", _b("div", quality, volatility20)),
        ]
    if "book_to_price" in panel:
        value = _f("book_to_price")
        hybrids += [
            ("H04", "value_reversal", "hybrid", _b("mul", value, reversal5)),
            ("H05", "value_momentum", "hybrid", _b("mul", value, momentum20)),
        ]
    if "growth_composite" in panel:
        hybrids.append(("H06", "growth_momentum", "hybrid", _b("mul", _f("growth_composite"), momentum20)))
    if "safety_composite" in panel:
        hybrids.append(("H07", "safety_low_beta", "hybrid", _b("mul", _f("safety_composite"), low_beta)))
    rows = []
    for identifier, name, domain, expression in technical + fundamental + hybrids:
        validate(expression, set(panel), WINDOWS)
        fingerprint = hashlib.sha256(canonical(expression).encode()).hexdigest()[:16]
        rows.append(
            {
                "factor_id": f"{identifier}_{fingerprint}",
                "name": name,
                "domain": domain,
                "expression": expression,
                "inputs": sorted(fields_used(expression)),
            }
        )
    return rows


def make_split(index: pd.DatetimeIndex, config: dict[str, Any]) -> Split:
    validation, shadow, purge = (
        int(config["validation_days"]), int(config["shadow_days"]), int(config["purge_days"])
    )
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
    train = pd.Series(np.arange(len(index)) < train_end, index=index)
    val = pd.Series((np.arange(len(index)) >= validation_start) & (np.arange(len(index)) < validation_end), index=index)
    shadow_mask = pd.Series(np.arange(len(index)) >= shadow_start, index=index)
    return Split(
        train,
        val,
        shadow_mask,
        {
            "train": [str(index[0].date()), str(index[train_end - 1].date())],
            "validation": [str(index[validation_start].date()), str(index[validation_end - 1].date())],
            "shadow": [str(index[shadow_start].date()), str(index[-1].date())],
            "purge_days": purge,
        },
    )


def _rank_ic(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    return signal.rank(axis=1, pct=True).corrwith(target.rank(axis=1, pct=True), axis=1)


def _portfolio_returns(
    signal: pd.DataFrame, one_day: pd.DataFrame, cost: float
) -> tuple[pd.Series, pd.Series]:
    ranks = signal.rank(axis=1, pct=True)
    weights = ranks.div(ranks.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    gross = (weights * one_day).sum(axis=1)
    return gross, gross - turnover * cost


def _stats(values: pd.Series, mask: pd.Series) -> dict[str, Any]:
    x = values.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
    std = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    return {
        "n": int(len(x)),
        "mean": None if not len(x) else round(float(x.mean()), 8),
        "ir_ann": None if not np.isfinite(std) or std == 0 else round(float(x.mean() / std * np.sqrt(244)), 4),
        "hit_rate": None if not len(x) else round(float((x > 0).mean()), 4),
    }


def _deterministic_placebo(signal: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = signal.copy()
    for position, date in enumerate(output.index):
        rng = np.random.default_rng(seed + position)
        output.loc[date] = rng.permutation(output.loc[date].to_numpy())
    return output


def _walk_forward_positive_folds(
    signal: pd.DataFrame, target: pd.DataFrame, one_day: pd.DataFrame, train_end: int, cost: float
) -> tuple[int, int]:
    boundaries = np.linspace(max(252, train_end // 3), train_end, 4, dtype=int)
    positive = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        historical = _rank_ic(signal.iloc[:start], target.iloc[:start]).dropna()
        direction = 1.0 if historical.mean() >= 0 else -1.0
        gross, _ = _portfolio_returns(direction * signal.iloc[start:end], one_day.iloc[start:end], cost)
        positive += int(gross.mean() > 0)
    return positive, 3


def pbo(candidate_returns: pd.DataFrame, blocks: int = 8) -> dict[str, Any]:
    data = candidate_returns.dropna(how="all")
    if data.shape[1] < 2 or len(data) < blocks * 5:
        return {"pbo": None, "splits": 0}
    partitions = np.array_split(np.arange(len(data)), blocks)
    logits = []
    for chosen in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([partitions[index] for index in chosen])
        outside = np.concatenate([partitions[index] for index in range(blocks) if index not in chosen])
        is_ir = data.iloc[inside].mean() / data.iloc[inside].std().replace(0.0, np.nan)
        winner = is_ir.idxmax()
        oos_ir = data.iloc[outside].mean() / data.iloc[outside].std().replace(0.0, np.nan)
        percentile = float(oos_ir.rank(pct=True).get(winner, np.nan))
        if np.isfinite(percentile):
            logits.append(math.log(percentile / max(1e-12, 1.0 - percentile)))
    return {"pbo": round(float(np.mean(np.asarray(logits) <= 0)), 4) if logits else None, "splits": len(logits)}


def deflated_sharpe_ratio(returns: pd.Series, trials: int) -> dict[str, Any]:
    x = returns.dropna().to_numpy(dtype=float)
    if len(x) < 30 or np.std(x, ddof=1) == 0:
        return {"probability": None, "observed_sharpe_ann": None}
    daily_sr = float(np.mean(x) / np.std(x, ddof=1))
    trials = max(1, int(trials))
    gamma = 0.5772156649
    normal = NormalDist()
    expected_max = (
        (1 - gamma) * normal.inv_cdf(1 - 1 / max(2, trials))
        + gamma * normal.inv_cdf(1 - 1 / max(2, trials * math.e))
    ) / math.sqrt(len(x))
    skew = float(pd.Series(x).skew())
    kurtosis = float(pd.Series(x).kurtosis() + 3.0)
    denominator = math.sqrt(max(1e-12, 1 - skew * daily_sr + (kurtosis - 1) * daily_sr**2 / 4))
    z = (daily_sr - expected_max) * math.sqrt(len(x) - 1) / denominator
    return {
        "probability": round(float(normal.cdf(z)), 6),
        "observed_sharpe_ann": round(daily_sr * math.sqrt(244), 4),
        "expected_max_noise_sharpe_ann": round(expected_max * math.sqrt(244), 4),
        "trials": trials,
    }


def run_discovery(
    prices: pd.DataFrame,
    statements: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if config.get("status") != "research_only_not_trading":
        raise ValueError("the public lite package is permanently research-only")
    panel = build_panel(prices, statements)
    split = make_split(panel["close"].index, config)
    horizon = int(config["prediction_horizon_days"])
    target = panel["open"].shift(-(horizon + 1)) / panel["open"].shift(-1) - 1.0
    one_day = panel["open"].shift(-2) / panel["open"].shift(-1) - 1.0
    candidates = candidate_library(panel)[: int(config["candidate_budget"])]
    rows, return_series = [], {}
    train_end = int(split.train.sum())
    for candidate in candidates:
        signal = evaluate(candidate["expression"], panel).replace([np.inf, -np.inf], np.nan)
        train_ic = _rank_ic(signal, target).loc[split.train].dropna()
        if train_ic.empty:
            continue
        direction = 1.0 if train_ic.mean() >= 0 else -1.0
        signal *= direction
        gross, costed = _portfolio_returns(signal, one_day, float(config["round_trip_cost"]))
        counter_gross, _ = _portfolio_returns(signal.shift(20), one_day, float(config["round_trip_cost"]))
        placebo_gross, _ = _portfolio_returns(
            _deterministic_placebo(signal, int(config["random_seed"])), one_day, float(config["round_trip_cost"])
        )
        positive, folds = _walk_forward_positive_folds(
            signal, target, one_day, train_end, float(config["round_trip_cost"])
        )
        row = {
            **candidate,
            "direction_learned_on_train": int(direction),
            "train_rank_ic": round(float(train_ic.mean() * direction), 8),
            "periods": {
                "train_gross": _stats(gross, split.train),
                "validation_gross": _stats(gross, split.validation),
                "validation_costed": _stats(costed, split.validation),
                "shadow_gross": _stats(gross, split.shadow),
            },
            "counter_validation": _stats(counter_gross, split.validation),
            "placebo_validation": _stats(placebo_gross, split.validation),
            "walk_forward": {"positive_folds": positive, "folds": folds},
        }
        rows.append(row)
        return_series[candidate["factor_id"]] = gross
    rows.sort(key=lambda row: row["train_rank_ic"], reverse=True)
    selected = rows[: int(config["stage2_budget"])]
    returns = pd.DataFrame(return_series)
    pbo_result = pbo(returns.loc[split.train], int(config["pbo_blocks"]))
    for row in selected:
        primary = row["periods"]["validation_gross"]["ir_ann"]
        counter = row["counter_validation"]["ir_ann"]
        placebo = row["placebo_validation"]["ir_ann"]
        row["guards"] = {
            "train_rank_ic_pass": row["train_rank_ic"] >= float(config["minimum_train_rank_ic"]),
            "walk_forward_pass": row["walk_forward"]["positive_folds"] >= int(config["minimum_positive_walk_forward_folds"]),
            "counter_beaten": primary is not None and counter is not None and primary > counter,
            "placebo_beaten": primary is not None and placebo is not None and primary > placebo,
            "project_pbo_pass": pbo_result["pbo"] is not None and pbo_result["pbo"] <= float(config["pbo_maximum"]),
        }
        row["historically_validated"] = all(row["guards"].values())
    best = selected[0] if selected else None
    dsr = (
        deflated_sharpe_ratio(
            returns[best["factor_id"]].loc[split.validation],
            int(config["prior_trials"]) + len(rows),
        )
        if best
        else {}
    )
    return {
        "schema_version": "perception_xalpha_lite_result_v1",
        "status": "diagnostic_only_research_only_not_trading",
        "split": split.audit,
        "candidate_count": len(rows),
        "stage2_count": len(selected),
        "historically_validated_count": sum(row["historically_validated"] for row in selected),
        "pbo": pbo_result,
        "dsr_best_train_selected": dsr,
        "candidates": selected,
        "orders": [],
        "automatic_trading_changes": [],
        "warning": "Historical evidence is not a trade signal. Use fresh forward shadow data before any deployment discussion.",
    }


def write_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)

