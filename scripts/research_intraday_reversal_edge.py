"""Literature-driven ETF liquidity-reversal research, strictly offline.

The preregistered tests cover:
1. cross-sectional 30-minute reversal after a relative price shock;
2. the same signal conditioned on point-in-time Amihud price impact;
3. end-of-day reversal among intraday losers.

Universe formation uses training-period data only. Signals are observed at t,
entry is the next five-minute close, and all results include explicit round-trip
cost. No broker, live config, strategy overlay or paper-agent state is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from overfitting_guard import combinatorial_symmetric_pbo
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
    reality_check_spa,
)


ROOT = Path(__file__).resolve().parents[1]
CN = timezone(timedelta(hours=8))
DEFAULT_CONFIG = ROOT / "configs" / "research" / "intraday_reversal_preregistered.json"
DEFAULT_QUOTES = (
    ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "intraday_reversal"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def select_training_universe(
    quotes_path: Path,
    *,
    train_end: str,
    allowed_asset_classes: set[str],
    universe_size: int,
    minimum_coverage: float,
) -> tuple[list[str], dict[str, Any]]:
    daily_last_amount: dict[tuple[str, str], float] = {}
    names: dict[str, str] = {}
    classes: dict[str, str] = {}
    training_dates: set[str] = set()
    with quotes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trade_date = str(row.get("trade_date") or str(row["timestamp"])[:10])
            asset_class = str(row.get("asset_class") or "")
            if trade_date > train_end or asset_class not in allowed_asset_classes:
                continue
            code = str(row["stockCode"]).zfill(6)
            amount = max(0.0, float(row.get("cumulative_amount") or 0.0))
            key = (trade_date, code)
            daily_last_amount[key] = max(amount, daily_last_amount.get(key, 0.0))
            names[code] = str(row.get("name") or code)
            classes[code] = asset_class
            training_dates.add(trade_date)
    required_days = max(1, math.ceil(len(training_dates) * minimum_coverage))
    by_code: dict[str, list[float]] = defaultdict(list)
    for (_, code), amount in daily_last_amount.items():
        by_code[code].append(amount)
    ranked = [
        (code, len(amounts), float(np.mean(amounts)))
        for code, amounts in by_code.items()
        if len(amounts) >= required_days
    ]
    ranked.sort(key=lambda item: (item[2], item[1], item[0]), reverse=True)
    selected = [code for code, _, _ in ranked[:universe_size]]
    if len(selected) < max(10, universe_size // 2):
        raise RuntimeError(f"only {len(selected)} instruments passed training-only universe")
    return selected, {
        "trainingDates": len(training_dates),
        "minimumCoverageDays": required_days,
        "selectionUsesOos": False,
        "selected": [
            {
                "stockCode": code,
                "name": names.get(code, code),
                "assetClass": classes.get(code),
                "trainingDays": days,
                "averageDailyCumulativeAmount": round(amount, 2),
            }
            for code, days, amount in ranked[:universe_size]
        ],
    }


def load_selected_panel(quotes_path: Path, selected: list[str]) -> dict[str, pd.DataFrame]:
    selected_set = set(selected)
    records: list[dict[str, Any]] = []
    with quotes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            code = str(row.get("stockCode", "")).zfill(6)
            if code not in selected_set:
                continue
            close = float(row["close"])
            records.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]),
                    "stockCode": code,
                    "close": close,
                    "prev_close": float(row["prev_close"])
                    if row.get("prev_close") is not None
                    else np.nan,
                    "bar_amount": max(0.0, float(row.get("bar_volume") or 0.0)) * close,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("selected universe has no rows")
    prices = frame.pivot(index="timestamp", columns="stockCode", values="close").sort_index()
    bar_amount = frame.pivot(
        index="timestamp", columns="stockCode", values="bar_amount"
    ).reindex(prices.index)
    prev_close = frame.pivot(
        index="timestamp", columns="stockCode", values="prev_close"
    ).reindex(prices.index)
    return {"prices": prices, "bar_amount": bar_amount, "prev_close": prev_close}


def within_day_ratio(values: pd.DataFrame, periods: int) -> pd.DataFrame:
    dates = pd.Series(values.index.strftime("%Y-%m-%d"), index=values.index)
    return values.groupby(dates).transform(
        lambda day: day / day.shift(periods) - 1.0
    )


def within_day_rolling_sum(values: pd.DataFrame, window: int) -> pd.DataFrame:
    dates = pd.Series(values.index.strftime("%Y-%m-%d"), index=values.index)
    return values.groupby(dates).transform(
        lambda day: day.rolling(window, min_periods=window).sum()
    )


def feature_panels(
    panel: dict[str, pd.DataFrame], lookback_bars: int, amount_window_bars: int
) -> dict[str, pd.DataFrame]:
    prices = panel["prices"]
    past_return = within_day_ratio(prices, lookback_bars)
    cross_median = past_return.median(axis=1, skipna=True)
    residual_return = past_return.sub(cross_median, axis=0)
    rolling_amount = within_day_rolling_sum(panel["bar_amount"], amount_window_bars)
    amihud = residual_return.abs() / (rolling_amount / 100_000_000.0).clip(lower=1e-6)
    amihud_rank = amihud.rank(axis=1, pct=True, method="average")
    return {
        "prices": prices,
        "prev_close": panel["prev_close"],
        "past_return": past_return,
        "residual_return": residual_return,
        "amihud_rank": amihud_rank,
    }


def select_tail(
    signal: pd.Series,
    *,
    fraction: float,
    side: str,
    max_assets: int,
    threshold: float | None = None,
    eligibility: pd.Series | None = None,
) -> list[str]:
    finite = signal[np.isfinite(signal)]
    if eligibility is not None:
        finite = finite[eligibility.reindex(finite.index).fillna(False)]
    if threshold is not None:
        finite = finite[finite <= threshold] if side == "bottom" else finite[finite >= threshold]
    count = min(max_assets, max(1, int(math.ceil(len(signal.dropna()) * fraction))))
    if finite.empty:
        return []
    ordered = finite.sort_values(ascending=side == "bottom")
    return [str(code) for code in ordered.index[:count]]


def future_trade_return(
    day_prices: pd.DataFrame,
    decision_index: int,
    code: str,
    *,
    holding_bars: int,
    maximum_entry_delay_minutes: int = 10,
) -> float | None:
    entry_index = decision_index + 1
    exit_index = entry_index + holding_bars
    if exit_index >= len(day_prices.index):
        return None
    decision_time = day_prices.index[decision_index]
    entry_time = day_prices.index[entry_index]
    exit_time = day_prices.index[exit_index]
    if (entry_time - decision_time).total_seconds() > maximum_entry_delay_minutes * 60:
        return None
    if (exit_time - entry_time).total_seconds() > (holding_bars + 2) * 5 * 60:
        return None
    entry = day_prices.iloc[entry_index].get(code)
    exit_price = day_prices.iloc[exit_index].get(code)
    if not np.isfinite(entry) or not np.isfinite(exit_price) or float(entry) <= 0:
        return None
    return float(exit_price / entry - 1.0)


def end_of_day_trade_return(
    day_prices: pd.DataFrame, decision_index: int, code: str, exit_time_text: str
) -> float | None:
    entry_index = decision_index + 1
    if entry_index >= len(day_prices.index):
        return None
    exit_matches = [
        index
        for index, timestamp in enumerate(day_prices.index)
        if timestamp.strftime("%H:%M") == exit_time_text
    ]
    if not exit_matches:
        return None
    exit_index = exit_matches[-1]
    if exit_index <= entry_index:
        return None
    decision_time = day_prices.index[decision_index]
    entry_time = day_prices.index[entry_index]
    if (entry_time - decision_time).total_seconds() > 10 * 60:
        return None
    entry = day_prices.iloc[entry_index].get(code)
    exit_price = day_prices.iloc[exit_index].get(code)
    if not np.isfinite(entry) or not np.isfinite(exit_price) or float(entry) <= 0:
        return None
    return float(exit_price / entry - 1.0)


def portfolio_interval(
    returns: dict[str, float | None],
    selected: list[str],
    *,
    max_weight: float,
    round_trip_cost: float,
) -> dict[str, Any]:
    usable = [(code, returns.get(code)) for code in selected if returns.get(code) is not None]
    weights = {code: max_weight for code, _ in usable}
    exposure = min(1.0, sum(weights.values()))
    if sum(weights.values()) > 1.0:
        scale = 1.0 / sum(weights.values())
        weights = {code: weight * scale for code, weight in weights.items()}
    gross = sum(weights[code] * float(value) for code, value in usable)
    net = gross - exposure * round_trip_cost
    return {
        "gross_return": gross,
        "net_return": net,
        "exposure": exposure,
        "positions": len(usable),
        "codes": [code for code, _ in usable],
    }


def generate_intervals(
    features: dict[str, pd.DataFrame],
    config: dict[str, Any],
    cost_bps: float,
) -> dict[str, list[dict[str, Any]]]:
    execution = config["execution"]
    signal_cfg = config["signals"]
    prices = features["prices"]
    variants = {name: [] for name in signal_cfg["variants"]}
    cost = cost_bps / 10_000.0
    fraction = float(signal_cfg["crossSectionFraction"])
    max_assets = int(execution["maxAssets"])
    max_weight = float(execution["maxWeightPerAsset"])
    shock_threshold = -float(signal_cfg["costFilteredShockMultiple"]) * cost
    intraday_times = set(execution["intradayDecisionTimes"])
    eod_time = str(execution["endOfDayDecisionTime"])
    eod_predictor_time = str(execution["endOfDayPredictorTime"])

    for trade_date, day_prices in prices.groupby(prices.index.strftime("%Y-%m-%d")):
        if len(day_prices) < 20:
            continue
        for decision_index, timestamp in enumerate(day_prices.index):
            time_text = timestamp.strftime("%H:%M")
            if time_text in intraday_times:
                residual = features["residual_return"].loc[timestamp]
                amihud_rank = features["amihud_rank"].loc[timestamp]
                selections = {
                    "momentum_30m_control": select_tail(
                        residual, fraction=fraction, side="top", max_assets=max_assets
                    ),
                    "reversal_30m": select_tail(
                        residual, fraction=fraction, side="bottom", max_assets=max_assets
                    ),
                    "reversal_30m_cost_filtered": select_tail(
                        residual,
                        fraction=fraction,
                        side="bottom",
                        max_assets=max_assets,
                        threshold=shock_threshold,
                    ),
                    "reversal_30m_high_amihud": select_tail(
                        residual,
                        fraction=fraction,
                        side="bottom",
                        max_assets=max_assets,
                        eligibility=amihud_rank >= float(signal_cfg["amihudHighRankMinimum"]),
                    ),
                }
                all_codes = sorted(set(code for codes in selections.values() for code in codes))
                future = {
                    code: future_trade_return(
                        day_prices,
                        decision_index,
                        code,
                        holding_bars=int(execution["intradayHoldingBars"]),
                    )
                    for code in all_codes
                }
                for name, codes in selections.items():
                    interval = portfolio_interval(
                        future,
                        codes,
                        max_weight=max_weight,
                        round_trip_cost=cost,
                    )
                    variants[name].append(
                        {
                            "trade_date": trade_date,
                            "timestamp": timestamp.isoformat(),
                            **interval,
                        }
                    )

            if time_text == eod_time:
                predictor_matches = [
                    index
                    for index, candidate_time in enumerate(day_prices.index)
                    if candidate_time.strftime("%H:%M") == eod_predictor_time
                ]
                if not predictor_matches:
                    continue
                predictor_time = day_prices.index[predictor_matches[-1]]
                previous_close = features["prev_close"].loc[predictor_time]
                return_of_day = day_prices.loc[predictor_time] / previous_close - 1.0
                finite_return_of_day = return_of_day[np.isfinite(return_of_day)]
                if len(finite_return_of_day) < 10:
                    continue
                residual = return_of_day - finite_return_of_day.median()
                selections = {
                    "eod_momentum_control": select_tail(
                        residual, fraction=fraction, side="top", max_assets=max_assets
                    ),
                    "eod_reversal": select_tail(
                        residual, fraction=fraction, side="bottom", max_assets=max_assets
                    ),
                    "eod_reversal_cost_filtered": select_tail(
                        residual,
                        fraction=fraction,
                        side="bottom",
                        max_assets=max_assets,
                        threshold=shock_threshold,
                    ),
                }
                all_codes = sorted(set(code for codes in selections.values() for code in codes))
                future = {
                    code: end_of_day_trade_return(
                        day_prices,
                        decision_index,
                        code,
                        str(execution["endOfDayExitTime"]),
                    )
                    for code in all_codes
                }
                for name, codes in selections.items():
                    interval = portfolio_interval(
                        future,
                        codes,
                        max_weight=max_weight,
                        round_trip_cost=cost,
                    )
                    variants[name].append(
                        {
                            "trade_date": trade_date,
                            "timestamp": timestamp.isoformat(),
                            **interval,
                        }
                    )
    return variants


def daily_returns(intervals: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in intervals:
        grouped[str(row["trade_date"])].append(float(row["net_return"]))
    return {
        day: float(np.prod([1.0 + value for value in values]) - 1.0)
        for day, values in sorted(grouped.items())
    }


def performance(intervals: list[dict[str, Any]], daily: dict[str, float]) -> dict[str, Any]:
    values = np.asarray(list(daily.values()), dtype=float)
    active_intervals = [row for row in intervals if float(row["exposure"]) > 0]
    if len(values) == 0:
        return {"days": 0}
    equity = np.cumprod(1.0 + values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "days": int(len(values)),
        "intervals": len(intervals),
        "activeIntervals": len(active_intervals),
        "positionDecisions": int(sum(int(row["positions"]) for row in intervals)),
        "averageExposure": round(
            float(np.mean([float(row["exposure"]) for row in intervals])), 6
        ),
        "grossTotalReturn": round(
            float(
                np.prod(
                    [
                        1.0 + float(row["gross_return"])
                        for row in intervals
                    ]
                )
                - 1.0
            ),
            8,
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
    primary = report["costStress"]["12.0"]["oosMetrics"]
    lines = [
        "# Intraday ETF Liquidity-Reversal OOS Validation",
        "",
        "Status: `diagnostic_only / offline / no live gating`",
        "",
        f"- Training/universe formation: {report['split']['trainStart']} to {report['split']['trainEnd']}",
        f"- Untouched OOS: {report['split']['oosStart']} to {report['split']['oosEnd']}",
        "- Entry: next five-minute close; primary cost: 12 bps round trip.",
        "",
        "## Primary OOS results",
        "",
        "| Variant | Net return | Sharpe | Win days | Worst day | Max DD | Exposure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in primary.items():
        sharpe = metrics.get("dailySharpe")
        lines.append(
            f"| {name} | {metrics.get('netTotalReturn', 0):.2%} | "
            f"{sharpe if sharpe is not None else 'n/a'} | "
            f"{metrics.get('winDayRate', 0):.1%} | {metrics.get('worstDay', 0):.2%} | "
            f"{metrics.get('maxDrawdown', 0):.2%} | {metrics.get('averageExposure', 0):.1%} |"
        )
    evidence = report["evidence"]
    lines.extend(
        [
            "",
            "## Statistical audit",
            "",
            f"- Best reversal candidate: `{report['bestReversalCandidate']}`.",
            f"- PBO: `{evidence['pbo'].get('pbo')}`.",
            f"- DM versus cash: significant=`{evidence['dm'].get('significant')}`, "
            f"mean daily return=`{evidence['dm'].get('mean_diff')}`.",
            f"- DSR: significant=`{evidence['dsr'].get('significant')}`, "
            f"p=`{evidence['dsr'].get('p_value')}`.",
            f"- SPA across all preregistered signals: reject=`{evidence['spa'].get('reject')}`, "
            f"p=`{evidence['spa'].get('p_value')}`.",
            "",
            "## Locked verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "## Research still needed",
            "",
        ]
    )
    for item in report["deferredResearch"]:
        lines.append(f"- **{item['topic']}** — blocked by: {item['blockedBy']}.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The OOS window contains only 21 trading days.",
            "- Yahoo bars lack historical bid/ask and depth; fixed costs cannot model adverse selection.",
            "- Current product membership may retain survivorship bias.",
            "- These studies originate largely in stocks or U.S. ETFs; transfer to A-share-listed T+0 ETFs is an empirical question, not an assumption.",
            "- No result in this report changes live configuration, locks, orders, exits or sizing.",
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
    signal_cfg = config["signals"]

    selected, universe_audit = select_training_universe(
        quotes,
        train_end=str(data_cfg["trainEnd"]),
        allowed_asset_classes=set(data_cfg["allowedAssetClasses"]),
        universe_size=int(data_cfg["universeSize"]),
        minimum_coverage=float(data_cfg["minimumTrainingDayCoverage"]),
    )
    panel = load_selected_panel(quotes, selected)
    features = feature_panels(
        panel,
        int(signal_cfg["shortHorizonLookbackBars"]),
        int(signal_cfg["amihudAmountWindowBars"]),
    )

    cost_results: dict[str, Any] = {}
    daily_by_cost: dict[str, dict[str, dict[str, float]]] = {}
    for cost_bps in execution["costStressBps"]:
        intervals = generate_intervals(features, config, float(cost_bps))
        train_intervals = {
            name: [
                row
                for row in rows
                if str(data_cfg["trainStart"]) <= row["trade_date"] <= str(data_cfg["trainEnd"])
            ]
            for name, rows in intervals.items()
        }
        oos_intervals = {
            name: [
                row
                for row in rows
                if str(data_cfg["oosStart"]) <= row["trade_date"] <= str(data_cfg["oosEnd"])
            ]
            for name, rows in intervals.items()
        }
        train_daily = {name: daily_returns(rows) for name, rows in train_intervals.items()}
        oos_daily = {name: daily_returns(rows) for name, rows in oos_intervals.items()}
        key = str(float(cost_bps))
        daily_by_cost[key] = oos_daily
        cost_results[key] = {
            "trainMetricsDiagnosticOnly": {
                name: performance(train_intervals[name], train_daily[name])
                for name in signal_cfg["variants"]
            },
            "oosMetrics": {
                name: performance(oos_intervals[name], oos_daily[name])
                for name in signal_cfg["variants"]
            },
        }

    primary_key = str(float(execution["primaryRoundTripCostBps"]))
    primary_metrics = cost_results[primary_key]["oosMetrics"]
    reversal_names = [
        name for name in signal_cfg["variants"] if "reversal" in name
    ]
    best_reversal = max(
        reversal_names,
        key=lambda name: (
            primary_metrics[name].get("dailySharpe")
            if primary_metrics[name].get("dailySharpe") is not None
            else -999.0
        ),
    )
    primary_daily = daily_by_cost[primary_key]
    shared_dates = sorted(
        set.intersection(*(set(primary_daily[name]) for name in signal_cfg["variants"]))
    )
    pbo = combinatorial_symmetric_pbo(
        [
            [primary_daily[name][day] for day in shared_dates]
            for name in signal_cfg["variants"]
        ],
        n_blocks=8,
    )
    cash_days = {day: 0.0 for day in shared_dates}
    best_days = primary_daily[best_reversal]
    dm = diebold_mariano_hln(cash_days, best_days, alpha=0.05)
    dsr = deflated_sharpe_diagnostic(
        cash_days,
        best_days,
        n_trials=len(signal_cfg["variants"]),
        alpha=0.10,
    )
    spa = reality_check_spa(
        [0.0 for _ in shared_dates],
        {
            name: [-primary_daily[name][day] for day in shared_dates]
            for name in signal_cfg["variants"]
        },
        alpha=0.05,
        n_boot=1000,
        seed=20260628,
    )
    stress_metrics = cost_results["20.0"]["oosMetrics"][best_reversal]
    gate_cfg = config["evidenceGates"]
    gate_checks = {
        "minimumOosDays": len(shared_dates) >= int(gate_cfg["minimumOosDays"]),
        "positiveNetReturnAt12Bps": primary_metrics[best_reversal]["netTotalReturn"] > 0,
        "positiveSharpeAt12Bps": (primary_metrics[best_reversal].get("dailySharpe") or -999) > 0,
        "positiveNetReturnAt20Bps": stress_metrics["netTotalReturn"] > 0,
        "dmSignificant": bool(dm.get("significant")),
        "dsrSignificant": bool(dsr.get("significant")),
        "pboPass": pbo.get("pbo") is not None
        and float(pbo["pbo"]) <= float(gate_cfg["maximumPbo"]),
        "spaPass": bool(spa.get("reject")),
    }
    passed = all(gate_checks.values())
    verdict = (
        "historical_oos_pass_forward_shadow_required"
        if passed
        else "no_validated_incremental_edge"
    )
    verdict_reason = (
        "The candidate cleared the preregistered historical gates but still requires new-date "
        "forward shadow evidence before any trading proposal."
        if passed
        else "At least one absolute-return, cost-stress or multiple-testing gate failed; none "
        "of these signals may alter live trading."
    )
    report = {
        "schemaVersion": "intraday_reversal_research_result_v1",
        "generatedAt": datetime.now(CN).isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "split": {
            "trainStart": data_cfg["trainStart"],
            "trainEnd": data_cfg["trainEnd"],
            "oosStart": data_cfg["oosStart"],
            "oosEnd": data_cfg["oosEnd"],
            "oosDays": len(shared_dates),
        },
        "universe": universe_audit,
        "costStress": cost_results,
        "bestReversalCandidate": best_reversal,
        "evidence": {
            "gateChecks": gate_checks,
            "pbo": pbo,
            "dm": dm,
            "dsr": dsr,
            "spa": spa,
        },
        "verdict": verdict,
        "verdictReason": verdict_reason,
        "deferredResearch": config["deferredResearch"],
        "liveChanges": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "intraday_reversal_oos_result.json", report)
    daily_rows = [
        {
            "trade_date": day,
            **{name: primary_daily[name][day] for name in signal_cfg["variants"]},
        }
        for day in shared_dates
    ]
    write_csv(output_dir / "intraday_reversal_oos_daily.csv", daily_rows)
    (output_dir / "intraday_reversal_oos_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "bestReversalCandidate": best_reversal,
                "oosDays": len(shared_dates),
                "primaryOosMetrics": primary_metrics,
                "gateChecks": gate_checks,
                "output": str(output_dir / "intraday_reversal_oos_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
