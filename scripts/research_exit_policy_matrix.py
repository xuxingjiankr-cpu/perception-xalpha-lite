"""Causal exit-policy matrix on the paper agent's actual 60-day replay entries.

Entries, quantities, and the current baseline exits are frozen from the existing
``entry_logic_v2`` replay. Candidate policies may exit earlier, but never later than a
recorded baseline exit. A condition observed on five-minute bar t fills at the next
bar's bid, avoiding same-bar trigger fills and ex-post-high selling.

The fixed, preregistered policy library covers:

* fixed stop-loss and take-profit boundaries;
* peak trailing stops;
* maximum holding bars;
* causal moving-average crosses;
* three fixed triple-barrier combinations.

ATR is deliberately not approximated: the Yahoo replay contains point quotes rather
than historical OHLC bars, so true range is unavailable. Signal-reversal exits are
also excluded because historical point-in-time component-score paths were not stored.

The 2026-05-21..2026-06-18 window has already been reused by this project. Results are
therefore diagnostic even when a numerical screen looks favorable. At least 50
strictly unseen OOS trades and genuinely new forward days are required for a verdict.

STRICTLY OFFLINE / SHADOW. No orders, broker calls, live config, or overlay writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import overfitting_guard as og
from research_exit_timing_actual_trades import (
    DEFAULT_REPLAY_SUMMARY,
    QUOTES,
    ROUND_TRIP_COMMISSION,
    TRAIN_END,
    build_path,
    load_quotes,
    parse_time,
    sell_fill_price,
)
from run_etf_paper_trading_agent import ROOT, as_float


OUT_DIR = ROOT / "outputs" / "exit_policy_matrix"
MIN_TOTAL_TRADES = 50
MIN_UNSEEN_TEST_TRADES = 50
MIN_FORWARD_DAYS = 20
WALK_FORWARD_TEST_DAYS = 5
WALK_FORWARD_EMBARGO_DAYS = 1

POLICIES: tuple[dict[str, Any], ...] = (
    {"name": "current_baseline", "family": "baseline", "kind": "baseline"},
    *(
        {"name": f"fixed_stop_{pct:g}pct", "family": "fixed_stop", "kind": "stop", "pct": pct / 100}
        for pct in (1, 2, 3, 5)
    ),
    *(
        {"name": f"fixed_take_{pct:g}pct", "family": "fixed_take", "kind": "take", "pct": pct / 100}
        for pct in (2, 3, 5, 8)
    ),
    *(
        {"name": f"trailing_{pct:g}pct", "family": "trailing", "kind": "trail", "pct": pct / 100}
        for pct in (1, 2, 3, 5)
    ),
    {
        "name": "profit_arm2_trail2",
        "family": "profit_protection",
        "kind": "armed_trail",
        "arm": 0.02,
        "trail": 0.02,
    },
    *(
        {"name": f"time_exit_{bars}bars", "family": "time_exit", "kind": "time", "bars": bars}
        for bars in (6, 12, 24, 48)
    ),
    *(
        {"name": f"ma_exit_{window}bars", "family": "ma_exit", "kind": "ma", "window": window}
        for window in (5, 10, 20)
    ),
    {
        "name": "triple_tp2_sl1_t12",
        "family": "triple_barrier",
        "kind": "triple",
        "take": 0.02,
        "stop": 0.01,
        "bars": 12,
    },
    {
        "name": "triple_tp3_sl2_t24",
        "family": "triple_barrier",
        "kind": "triple",
        "take": 0.03,
        "stop": 0.02,
        "bars": 24,
    },
    {
        "name": "triple_tp5_sl3_t48",
        "family": "triple_barrier",
        "kind": "triple",
        "take": 0.05,
        "stop": 0.03,
        "bars": 48,
    },
    {
        "name": "triple_tp3_sl1p5_t5d",
        "family": "triple_barrier",
        "kind": "triple",
        "take": 0.03,
        "stop": 0.015,
        "trading_days": 5,
    },
)
POLICY_BY_NAME = {row["name"]: row for row in POLICIES}
POLICY_NAMES = tuple(POLICY_BY_NAME)
NON_BASELINE = tuple(name for name in POLICY_NAMES if name != "current_baseline")

LITERATURE = (
    {
        "title": "When Do Stop-Loss Rules Stop Losses?",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338",
        "finding": "simple stops reduce expected return under a random walk but can add value under momentum",
    },
    {
        "title": "Trading Securities Using Trailing Stops",
        "url": "https://doi.org/10.1287/mnsc.41.6.1096",
        "finding": "trailing distance changes gain distribution, variance, and holding duration",
    },
    {
        "title": "Risk Reduction Using Trailing Stop-Loss Rules",
        "url": "https://doi.org/10.1111/irfi.12328",
        "finding": "trailing stops may lower mean return while reducing downside risk, conditional on costs",
    },
    {
        "title": "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit",
        "url": "https://arxiv.org/abs/1411.5062",
        "finding": "optimal exits depend on the assumed price process and transaction costs",
    },
)


def pair_all_filled_lots(
    order_lifecycle: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """FIFO-pair every confirmed buy with its recorded sell, including hard exits."""
    queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    lots: list[dict[str, Any]] = []
    diagnostics = {
        "filled_buys": 0,
        "filled_sells": 0,
        "paired_lots": 0,
        "hard_exit_lots": 0,
        "timing_exit_lots": 0,
        "unmatched_sell_quantity": 0,
    }
    filled = [
        row
        for row in order_lifecycle
        if str(row.get("status")) == "filled"
        and row.get("fill_time")
        and as_float(row.get("filled_qty"), 0.0) > 0
        and str(row.get("side")) in {"buy", "sell"}
    ]
    filled.sort(key=lambda row: parse_time(str(row["fill_time"])))
    for row in filled:
        code = str(row.get("stockCode") or "").zfill(6)
        quantity = int(as_float(row.get("filled_qty"), 0.0))
        if row["side"] == "buy":
            diagnostics["filled_buys"] += 1
            queues[code].append(
                {
                    "remaining": quantity,
                    "entry_time": str(row["fill_time"]),
                    "entry_price": as_float(row.get("fill_price"), 0.0),
                    "entry_reason": str(row.get("reason") or ""),
                }
            )
            continue

        diagnostics["filled_sells"] += 1
        remaining = quantity
        while remaining > 0 and queues[code]:
            entry = queues[code][0]
            take = min(remaining, int(entry["remaining"]))
            exit_reason = str(row.get("reason") or "")
            lots.append(
                {
                    "stockCode": code,
                    "quantity": take,
                    "entry_time": entry["entry_time"],
                    "entry_price": entry["entry_price"],
                    "entry_reason": entry["entry_reason"],
                    "baseline_exit_time": str(row["fill_time"]),
                    "baseline_exit_price": as_float(row.get("fill_price"), 0.0),
                    "baseline_exit_reason": exit_reason,
                }
            )
            diagnostics["paired_lots"] += 1
            if exit_reason == "unified_sell_score_exit":
                diagnostics["timing_exit_lots"] += 1
            else:
                diagnostics["hard_exit_lots"] += 1
            entry["remaining"] -= take
            remaining -= take
            if entry["remaining"] <= 0:
                queues[code].popleft()
        diagnostics["unmatched_sell_quantity"] += remaining
    return lots, diagnostics


def _ma_cross(prices: list[float], index: int, window: int) -> bool:
    if index < window:
        return False
    current_ma = statistics.mean(prices[index - window + 1 : index + 1])
    previous_ma = statistics.mean(prices[index - window : index])
    return prices[index] < current_ma and prices[index - 1] >= previous_ma


def _five_day_time_barrier(path: list[dict[str, Any]], index: int, days: int) -> bool:
    """Expire near the close of the Nth distinct A-share session, counting entry day."""
    distinct_dates = {
        path[position]["time"].date() for position in range(index + 1)
    }
    moment = path[index]["time"]
    return len(distinct_dates) >= days and (moment.hour, moment.minute) >= (14, 50)


def simulate_policy(
    path: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    """Observe at t and fill at t+1; never delay the recorded baseline exit."""
    if len(path) < 2:
        raise ValueError("path requires entry and baseline exit")
    kind = str(policy["kind"])
    if kind == "baseline":
        return {
            "exit_index": len(path) - 1,
            "exit_price": as_float(path[-1]["price"]),
            "triggered": False,
            "reason": "current_recorded_exit",
        }

    prices = [as_float(row["price"]) for row in path]
    entry = prices[0]
    peak = entry
    for index in range(1, len(path) - 1):
        current = prices[index]
        peak = max(peak, current)
        trigger = False
        reason = ""
        if kind == "stop":
            trigger = current <= entry * (1.0 - as_float(policy["pct"]))
            reason = "fixed_stop"
        elif kind == "take":
            trigger = current >= entry * (1.0 + as_float(policy["pct"]))
            reason = "fixed_take"
        elif kind == "trail":
            trigger = peak > entry and current <= peak * (1.0 - as_float(policy["pct"]))
            reason = "peak_trailing_stop"
        elif kind == "armed_trail":
            trigger = (
                peak >= entry * (1.0 + as_float(policy["arm"]))
                and current <= peak * (1.0 - as_float(policy["trail"]))
            )
            reason = "profit_armed_trailing_stop"
        elif kind == "time":
            trigger = index >= int(policy["bars"])
            reason = "maximum_holding_bars"
        elif kind == "ma":
            trigger = _ma_cross(prices, index, int(policy["window"]))
            reason = "moving_average_cross"
        elif kind == "triple":
            if current <= entry * (1.0 - as_float(policy["stop"])):
                trigger, reason = True, "triple_stop"
            elif current >= entry * (1.0 + as_float(policy["take"])):
                trigger, reason = True, "triple_take"
            elif policy.get("bars") is not None and index >= int(policy["bars"]):
                trigger, reason = True, "triple_time"
            elif policy.get("trading_days") is not None and _five_day_time_barrier(
                path, index, int(policy["trading_days"])
            ):
                trigger, reason = True, "triple_trading_day_time"
        else:
            raise ValueError(f"unknown policy kind: {kind}")

        if trigger:
            fill_index = index + 1
            return {
                "exit_index": fill_index,
                "exit_price": sell_fill_price(path[fill_index]),
                "triggered": True,
                "reason": reason,
            }
    return {
        "exit_index": len(path) - 1,
        "exit_price": as_float(path[-1]["price"]),
        "triggered": False,
        "reason": "fallback_to_recorded_exit",
    }


def evaluate_lot(
    lot: dict[str, Any], path: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    entry = as_float(lot["entry_price"])
    full_prices = [as_float(row["price"]) for row in path]
    full_low, full_high = min(full_prices), max(full_prices)
    evaluated: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        result = simulate_policy(path, policy)
        index = int(result["exit_index"])
        exit_price = as_float(result["exit_price"])
        held_prices = full_prices[: index + 1]
        causal_peak = max(held_prices)
        causal_low = min(held_prices)
        result.update(
            {
                "family": policy["family"],
                "net_return_pct": (exit_price / entry - 1.0 - ROUND_TRIP_COMMISSION) * 100.0,
                "mae_pct": (causal_low / entry - 1.0) * 100.0,
                "mfe_pct": (causal_peak / entry - 1.0) * 100.0,
                "exit_percentile": (
                    (exit_price - full_low) / (full_high - full_low) * 100.0
                    if full_high > full_low
                    else 50.0
                ),
                "sell_efficiency_pct": exit_price / full_high * 100.0,
                "shortfall_to_full_high_bps": (exit_price / full_high - 1.0) * 10_000.0,
                "giveback_from_causal_peak_bps": (exit_price / causal_peak - 1.0) * 10_000.0,
                "holding_bars": index,
                "exit_time": path[index]["time"].isoformat(),
                "post_exit_1bar_bps": (
                    full_prices[min(index + 1, len(path) - 1)] / exit_price - 1.0
                )
                * 10_000.0,
                "post_exit_3bar_bps": (
                    full_prices[min(index + 3, len(path) - 1)] / exit_price - 1.0
                )
                * 10_000.0,
                "post_exit_5bar_bps": (
                    full_prices[min(index + 5, len(path) - 1)] / exit_price - 1.0
                )
                * 10_000.0,
            }
        )
        evaluated[str(policy["name"])] = result
    return evaluated


def split_records(records: list[dict[str, Any]], window: str) -> list[dict[str, Any]]:
    if window == "train":
        return [row for row in records if str(row["entry_time"])[:10] <= TRAIN_END]
    if window == "test":
        return [row for row in records if str(row["entry_time"])[:10] > TRAIN_END]
    return records


def _max_drawdown(returns_pct: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns_pct:
        equity *= 1.0 + value / 100.0
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst * 100.0


def daily_returns(
    records: list[dict[str, Any]], policy_name: str, *, date_field: str = "entry_time"
) -> dict[str, float]:
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in records:
        day = str(row[date_field])[:10]
        by_day[day].append(as_float(row["policies"][policy_name]["net_return_pct"]))
    return {day: statistics.mean(values) for day, values in sorted(by_day.items())}


def policy_stats(records: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    rows = [row["policies"][policy_name] for row in records]
    if not rows:
        return {"trades": 0}
    returns = [as_float(row["net_return_pct"]) for row in rows]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    ordered = sorted(returns)
    daily = list(daily_returns(records, policy_name).values())
    daily_std = statistics.stdev(daily) if len(daily) > 1 else 0.0
    daily_sharpe = statistics.mean(daily) / daily_std * math.sqrt(252.0) if daily_std > 0 else 0.0
    max_drawdown = _max_drawdown(daily)
    annualized_mean = statistics.mean(daily) * 252.0 if daily else 0.0
    return {
        "trades": len(rows),
        "independent_entry_days": len(daily),
        "mean_net_return_pct": round(statistics.mean(returns), 4),
        "median_net_return_pct": round(statistics.median(returns), 4),
        "win_rate": round(len(wins) / len(returns), 4),
        "profit_loss_ratio": round(
            statistics.mean(wins) / abs(statistics.mean(losses)), 4
        )
        if wins and losses
        else None,
        "expectancy_pct": round(statistics.mean(returns), 4),
        "p5_net_return_pct": round(ordered[max(0, math.ceil(len(ordered) * 0.05) - 1)], 4),
        "p95_net_return_pct": round(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)], 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "daily_sharpe": round(daily_sharpe, 3),
        "calmar_proxy": round(annualized_mean / abs(max_drawdown), 3)
        if max_drawdown < 0
        else None,
        "avg_mae_pct": round(statistics.mean(as_float(row["mae_pct"]) for row in rows), 4),
        "avg_mfe_pct": round(statistics.mean(as_float(row["mfe_pct"]) for row in rows), 4),
        "mean_exit_percentile": round(
            statistics.mean(as_float(row["exit_percentile"]) for row in rows), 2
        ),
        "mean_sell_efficiency_pct": round(
            statistics.mean(as_float(row["sell_efficiency_pct"]) for row in rows), 2
        ),
        "median_sell_efficiency_pct": round(
            statistics.median(as_float(row["sell_efficiency_pct"]) for row in rows), 2
        ),
        "sell_efficiency_ge_95_rate": round(
            sum(as_float(row["sell_efficiency_pct"]) >= 95.0 for row in rows) / len(rows),
            4,
        ),
        "sell_efficiency_ge_90_rate": round(
            sum(as_float(row["sell_efficiency_pct"]) >= 90.0 for row in rows) / len(rows),
            4,
        ),
        "mean_high_shortfall_bps": round(
            statistics.mean(as_float(row["shortfall_to_full_high_bps"]) for row in rows), 2
        ),
        "mean_holding_bars": round(
            statistics.mean(as_float(row["holding_bars"]) for row in rows), 2
        ),
        "trigger_rate": round(sum(bool(row["triggered"]) for row in rows) / len(rows), 4),
        "post_exit_1bar_bps": round(
            statistics.mean(as_float(row["post_exit_1bar_bps"]) for row in rows), 2
        ),
        "post_exit_3bar_bps": round(
            statistics.mean(as_float(row["post_exit_3bar_bps"]) for row in rows), 2
        ),
        "post_exit_5bar_bps": round(
            statistics.mean(as_float(row["post_exit_5bar_bps"]) for row in rows), 2
        ),
    }


def paired_day_comparison(
    records: list[dict[str, Any]], candidate: str
) -> dict[str, Any]:
    base = daily_returns(records, "current_baseline")
    alt = daily_returns(records, candidate)
    days = sorted(set(base) & set(alt))
    differences = [alt[day] - base[day] for day in days]
    if not differences:
        return {"days": 0}
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    mean = statistics.mean(differences)
    return {
        "days": len(days),
        "mean_improvement_pct": round(mean, 4),
        "day_clustered_t": round(mean / std * math.sqrt(len(differences)), 3)
        if std > 0
        else None,
        "days_improved": sum(value > 0 for value in differences),
    }


def select_train_winners(
    train: list[dict[str, Any]], train_stats: dict[str, dict[str, Any]]
) -> dict[str, str]:
    del train
    families = sorted({str(policy["family"]) for policy in POLICIES if policy["family"] != "baseline"})
    winners: dict[str, str] = {}
    for family in families:
        names = [str(policy["name"]) for policy in POLICIES if policy["family"] == family]
        winners[family] = max(
            names,
            key=lambda name: train_stats[name].get("mean_net_return_pct", -math.inf),
        )
    winners["global_nonbaseline"] = max(
        NON_BASELINE,
        key=lambda name: train_stats[name].get("mean_net_return_pct", -math.inf),
    )
    return winners


def walk_forward(records: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({str(row["entry_time"])[:10] for row in records})
    folds: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(dates):
        test_dates = dates[cursor : cursor + WALK_FORWARD_TEST_DAYS]
        train_end_index = cursor - WALK_FORWARD_EMBARGO_DAYS
        train_dates = dates[: max(0, train_end_index)]
        train_rows = [row for row in records if str(row["entry_time"])[:10] in train_dates]
        test_rows = [row for row in records if str(row["entry_time"])[:10] in test_dates]
        if (
            len(train_rows) >= MIN_TOTAL_TRADES
            and len(train_dates) >= MIN_FORWARD_DAYS
            and test_rows
        ):
            train_means = {
                name: statistics.mean(
                    as_float(row["policies"][name]["net_return_pct"]) for row in train_rows
                )
                for name in NON_BASELINE
            }
            selected = max(NON_BASELINE, key=lambda name: train_means[name])
            selected_returns = [
                as_float(row["policies"][selected]["net_return_pct"]) for row in test_rows
            ]
            baseline_returns = [
                as_float(row["policies"]["current_baseline"]["net_return_pct"])
                for row in test_rows
            ]
            folds.append(
                {
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "train_trades": len(train_rows),
                    "embargo_days": WALK_FORWARD_EMBARGO_DAYS,
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "test_trades": len(test_rows),
                    "selected_policy": selected,
                    "selected_test_mean_pct": round(statistics.mean(selected_returns), 4),
                    "baseline_test_mean_pct": round(statistics.mean(baseline_returns), 4),
                    "improvement_pct": round(
                        statistics.mean(
                            selected - baseline
                            for selected, baseline in zip(selected_returns, baseline_returns)
                        ),
                        4,
                    ),
                }
            )
        cursor += WALK_FORWARD_TEST_DAYS
    improvements = [as_float(fold["improvement_pct"]) for fold in folds]
    return {
        "method": "anchored expanding train, one entry-day embargo, five entry-day test blocks",
        "minimum_train_trades": MIN_TOTAL_TRADES,
        "folds": folds,
        "fold_count": len(folds),
        "mean_fold_improvement_pct": round(statistics.mean(improvements), 4)
        if improvements
        else None,
        "positive_folds": sum(value > 0 for value in improvements),
    }


def build_result(
    records: list[dict[str, Any]],
    diagnostics: dict[str, int],
    summary_path: Path,
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for window in ("full", "train", "test"):
        sample = split_records(records, window)
        stats = {name: policy_stats(sample, name) for name in POLICY_NAMES}
        windows[window] = {
            "trades": len(sample),
            "entry_days": len({str(row["entry_time"])[:10] for row in sample}),
            "policies": {
                name: {
                    "stats": stats[name],
                    "vs_baseline": paired_day_comparison(sample, name),
                }
                for name in POLICY_NAMES
            },
        }

    train_stats = {
        name: windows["train"]["policies"][name]["stats"] for name in POLICY_NAMES
    }
    winners = select_train_winners(split_records(records, "train"), train_stats)
    selected = winners["global_nonbaseline"]
    entry_days = sorted(daily_returns(records, "current_baseline"))
    performance_matrix = [
        [daily_returns(records, name)[day] for day in entry_days] for name in POLICY_NAMES
    ]
    pbo = og.combinatorial_symmetric_pbo(performance_matrix, n_blocks=8)

    test_daily = list(daily_returns(split_records(records, "test"), selected).values())
    selected_sharpe = (
        statistics.mean(test_daily) / statistics.pstdev(test_daily)
        if len(test_daily) > 1 and statistics.pstdev(test_daily) > 0
        else 0.0
    )
    dsr = og.deflated_significance_note(
        n_trials=len(NON_BASELINE),
        observed_sharpe=selected_sharpe,
        n_obs=len(test_daily),
    )
    selected_test = windows["test"]["policies"][selected]
    baseline_test = windows["test"]["policies"]["current_baseline"]
    gates = {
        "total_trade_population_at_least_50": len(records) >= MIN_TOTAL_TRADES,
        "strict_unseen_test_trades_at_least_50": windows["test"]["trades"]
        >= MIN_UNSEEN_TEST_TRADES,
        "test_mean_return_above_baseline": selected_test["stats"].get(
            "mean_net_return_pct", -math.inf
        )
        > baseline_test["stats"].get("mean_net_return_pct", math.inf),
        "test_drawdown_better_than_baseline": selected_test["stats"].get(
            "max_drawdown_pct", -math.inf
        )
        > baseline_test["stats"].get("max_drawdown_pct", math.inf),
        "test_day_t_at_least_1p96": as_float(
            selected_test["vs_baseline"].get("day_clustered_t"), -math.inf
        )
        >= 1.96,
        "pbo_below_0p25": pbo.get("pbo") is not None
        and as_float(pbo.get("pbo")) < 0.25,
        "deflated_sharpe_exceeds_noise": dsr.get("flag") == "exceeds_noise_max",
        "clean_new_forward_window": False,
    }
    return {
        "research_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "diagnostic_only",
        "readiness": "reused_oos_not_clean",
        "edge_validated": False,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "paper_trading_only": True,
        "order_submit_calls_made": False,
        "live_config_modified": False,
        "source_quotes": str(QUOTES),
        "source_replay_summary": str(summary_path),
        "split": f"entry date <= {TRAIN_END} is train; later entries are reused test",
        "sample": {
            **diagnostics,
            "usable_lots": len(records),
            "full_entry_days": windows["full"]["entry_days"],
            "train_trades": windows["train"]["trades"],
            "test_trades": windows["test"]["trades"],
            "train_entry_days": windows["train"]["entry_days"],
            "test_entry_days": windows["test"]["entry_days"],
        },
        "methodology": {
            "entries": "all actual filled buys from the frozen entry_logic_v2 replay",
            "baseline": "the recorded confirmed sell, including hard exits",
            "candidate_constraint": "candidate may exit earlier but can never delay the recorded baseline/hard exit",
            "execution": "trigger on five-minute bar t, fill at next bar bid",
            "cost": f"next bid/recorded fill less {ROUND_TRIP_COMMISSION:.4%} round-trip commission",
            "future_high_usage": "MAE/MFE and high shortfall are outcome metrics only",
            "policy_library": list(POLICIES),
            "unavailable": {
                "atr": "not tested: replay lacks historical OHLC true range",
                "partial_atr_trailing": (
                    "not tested: partial exits plus a 2-ATR remainder require true OHLC ATR "
                    "and portfolio cash/quantity resimulation"
                ),
                "signal_reversal": "not tested: point-in-time component-score paths were not stored",
            },
        },
        "windows": windows,
        "train_selected_winners": winners,
        "global_selected_nonbaseline": selected,
        "walk_forward": walk_forward(records),
        "multiple_testing": {"pbo": pbo, "deflated_sharpe": dsr},
        "promotion_gates": gates,
        "all_gates_passed": all(gates.values()),
        "verdict": "no_promotable_edge_reused_oos_and_insufficient_unseen_trades",
        "limitations": [
            "The nominal test window has already informed earlier strategy research.",
            "The replay entry configuration was selected using the same 60-day history.",
            "Five-minute point quotes miss intrabar barrier touches and true OHLC ATR.",
            "Candidate exits are evaluated lot-by-lot; freed cash and replacement entries are not resimulated.",
            "Daily equal-weight trade returns are diagnostics, not a capital-accurate portfolio equity curve.",
            "Yahoo bid/spread fields are synthetic proxies rather than historical exchange depth.",
        ],
        "literature": list(LITERATURE),
        "records": records,
    }


def report_markdown(result: dict[str, Any]) -> str:
    test = result["windows"]["test"]["policies"]
    winners = result["train_selected_winners"]
    display = ["current_baseline", *dict.fromkeys(winners.values())]
    lines = [
        "# Exit Policy Matrix — fixed entries, causal next-bar exits",
        "",
        f"Status: **{result['status']}** | readiness: **{result['readiness']}** | "
        f"edge validated: **{str(result['edge_validated']).lower()}**",
        "",
        f"Usable lots: **{result['sample']['usable_lots']}**; train/test trades: "
        f"**{result['sample']['train_trades']}/{result['sample']['test_trades']}**; "
        f"train/test entry days: **{result['sample']['train_entry_days']}/"
        f"{result['sample']['test_entry_days']}**.",
        "",
        "## Train-selected family winners evaluated on the reused test window",
        "",
        "| policy | family | trades | mean net | win | MDD | Sharpe | sell eff. | >=95% | MAE | MFE | trigger | day Δ / t |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in display:
        node = test[name]
        stats = node["stats"]
        comparison = node["vs_baseline"]
        lines.append(
            f"| {name} | {POLICY_BY_NAME[name]['family']} | {stats.get('trades', 0)} | "
            f"{stats.get('mean_net_return_pct', 0):+.4f}% | "
            f"{stats.get('win_rate', 0):.2%} | "
            f"{stats.get('max_drawdown_pct', 0):+.3f}% | {stats.get('daily_sharpe')} | "
            f"{stats.get('mean_sell_efficiency_pct', 0):.2f}% | "
            f"{stats.get('sell_efficiency_ge_95_rate', 0):.2%} | "
            f"{stats.get('avg_mae_pct', 0):+.3f}% | {stats.get('avg_mfe_pct', 0):+.3f}% | "
            f"{stats.get('trigger_rate', 0):.2%} | "
            f"{comparison.get('mean_improvement_pct', 0):+.4f}% / "
            f"{comparison.get('day_clustered_t')} |"
        )

    selected = result["global_selected_nonbaseline"]
    selected_test = test[selected]
    baseline_test = test["current_baseline"]
    walk = result["walk_forward"]
    failed = [name for name, passed in result["promotion_gates"].items() if not passed]
    lines += [
        "",
        "## Decision",
        "",
        f"- Global non-baseline winner selected on TRAIN only: **{selected}**.",
        f"- Its reused-test mean difference versus baseline: "
        f"**{selected_test['vs_baseline'].get('mean_improvement_pct', 0):+.4f}% per entry day**, "
        f"t = **{selected_test['vs_baseline'].get('day_clustered_t')}**.",
        f"- Reused-test sell efficiency: **{selected_test['stats'].get('mean_sell_efficiency_pct')}%** "
        f"versus baseline **{baseline_test['stats'].get('mean_sell_efficiency_pct')}%**; "
        f"right-tail p95: **{selected_test['stats'].get('p95_net_return_pct')}%** versus "
        f"**{baseline_test['stats'].get('p95_net_return_pct')}%**.",
        f"- PBO: **{result['multiple_testing']['pbo'].get('pbo')}**; "
        f"deflated-Sharpe screen: **{result['multiple_testing']['deflated_sharpe'].get('flag')}**.",
        f"- Anchored walk-forward folds: **{walk['fold_count']}**; mean fold difference: "
        f"**{walk['mean_fold_improvement_pct']}%**; positive folds: **{walk['positive_folds']}**.",
        f"- Failed promotion gates: **{', '.join(failed)}**.",
        f"- Verdict: **{result['verdict']}**.",
        "- No live strategy, overlay, order path, or execution lock was changed.",
        "",
        "## Method exclusions",
        "",
        f"- ATR: {result['methodology']['unavailable']['atr']}.",
        f"- Partial profit-taking plus ATR remainder: "
        f"{result['methodology']['unavailable']['partial_atr_trailing']}.",
        f"- Signal reversal: {result['methodology']['unavailable']['signal_reversal']}.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limitations"])
    lines += ["", "## Sources", ""]
    lines.extend(
        f"- [{item['title']}]({item['url']}): {item['finding']}."
        for item in result["literature"]
    )
    lines.append("")
    return "\n".join(lines)


def write_policy_csv(result: dict[str, Any], path: Path) -> None:
    fields = [
        "window",
        "policy",
        "family",
        "trades",
        "independent_entry_days",
        "mean_net_return_pct",
        "median_net_return_pct",
        "win_rate",
        "profit_loss_ratio",
        "max_drawdown_pct",
        "daily_sharpe",
        "calmar_proxy",
        "avg_mae_pct",
        "avg_mfe_pct",
        "mean_exit_percentile",
        "mean_sell_efficiency_pct",
        "median_sell_efficiency_pct",
        "sell_efficiency_ge_95_rate",
        "sell_efficiency_ge_90_rate",
        "mean_high_shortfall_bps",
        "trigger_rate",
        "mean_improvement_pct",
        "day_clustered_t",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window in ("full", "train", "test"):
            for name in POLICY_NAMES:
                node = result["windows"][window]["policies"][name]
                writer.writerow(
                    {
                        "window": window,
                        "policy": name,
                        "family": POLICY_BY_NAME[name]["family"],
                        **{key: node["stats"].get(key) for key in fields if key in node["stats"]},
                        "mean_improvement_pct": node["vs_baseline"].get("mean_improvement_pct"),
                        "day_clustered_t": node["vs_baseline"].get("day_clustered_t"),
                    }
                )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Causal fixed-entry exit policy matrix on actual replay trades."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_REPLAY_SUMMARY)
    parser.add_argument("--quotes", type=Path, default=QUOTES)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    lots, diagnostics = pair_all_filled_lots(list(summary.get("order_lifecycle") or []))
    quote_map = load_quotes({lot["stockCode"] for lot in lots}, args.quotes)
    records: list[dict[str, Any]] = []
    skipped_short_paths = 0
    for lot in lots:
        path = build_path(lot, quote_map.get(lot["stockCode"], []))
        if len(path) < 3:
            skipped_short_paths += 1
            continue
        records.append({**lot, "policies": evaluate_lot(lot, path)})
    diagnostics["short_paths_skipped"] = skipped_short_paths
    if not records:
        raise RuntimeError("no usable filled replay paths")

    result = build_result(records, diagnostics, args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "exit_policy_matrix_60d.json"
    md_path = args.output_dir / "exit_policy_matrix_60d.md"
    csv_path = args.output_dir / "exit_policy_matrix_60d.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(report_markdown(result), encoding="utf-8")
    write_policy_csv(result, csv_path)

    print(report_markdown(result))
    print(f"outputs: {json_path} | {md_path} | {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
