"""Preregistered two-timescale ETF timing research.

The slow signal selects liquid confirmed T+0 ETFs with positive relative
strength.  The fast signal waits for a pullback from a high that was already
known at the decision time, then requires a causal-VWAP recovery before entering
on the next observed five-minute close.  The full candidate exits at the known
high, an equal-distance stop, or a fixed time exit.

This is offline research.  It cannot import a broker client, write live config,
write a strategy overlay, size a live position, or submit an order.
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
from research_intraday_reversal_edge import (
    load_selected_panel,
    select_training_universe,
)
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
    reality_check_spa,
)


ROOT = Path(__file__).resolve().parents[1]
CN = timezone(timedelta(hours=8))
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "trend_pullback_recovery_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "trend_pullback_recovery"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def causal_vwap(
    prices: pd.Series,
    bar_amount: pd.Series,
    end_index: int,
) -> float | None:
    """VWAP through end_index only; never uses final daily volume."""
    px = prices.iloc[: end_index + 1].astype(float)
    amount = bar_amount.iloc[: end_index + 1].astype(float)
    volume = amount / px.replace(0.0, np.nan)
    valid = np.isfinite(px) & np.isfinite(amount) & np.isfinite(volume)
    total_volume = float(volume[valid].sum())
    if total_volume <= 0:
        return None
    return float(amount[valid].sum() / total_volume)


def fixed_horizon_return(
    prices: pd.Series,
    entry_index: int,
    holding_bars: int,
) -> tuple[float, int] | None:
    exit_index = entry_index + holding_bars
    if entry_index < 0 or exit_index >= len(prices):
        return None
    entry = float(prices.iloc[entry_index])
    exit_price = float(prices.iloc[exit_index])
    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
        return None
    return float(exit_price / entry - 1.0), exit_index


def find_pullback_entry(
    prices: pd.Series,
    bar_amount: pd.Series,
    *,
    decision_index: int,
    reference_high: float,
    minimum_pullback: float,
    maximum_wait_bars: int,
    require_non_negative_last_bar: bool,
    require_above_vwap: bool,
) -> dict[str, Any] | None:
    """Wait for a real dip and recovery, then enter one full bar later."""
    pullback_seen = False
    last_trigger_index = min(
        len(prices) - 2, decision_index + maximum_wait_bars
    )
    for trigger_index in range(decision_index + 1, last_trigger_index + 1):
        current = float(prices.iloc[trigger_index])
        previous = float(prices.iloc[trigger_index - 1])
        if not np.isfinite(current) or not np.isfinite(previous) or current <= 0:
            continue
        pullback_seen = pullback_seen or (
            current <= reference_high * (1.0 - minimum_pullback)
        )
        if not pullback_seen:
            continue
        if require_non_negative_last_bar and current < previous:
            continue
        vwap = causal_vwap(prices, bar_amount, trigger_index)
        if require_above_vwap and (vwap is None or current < vwap):
            continue
        entry_index = trigger_index + 1
        entry = float(prices.iloc[entry_index])
        if not np.isfinite(entry) or entry <= 0:
            return None
        return {
            "trigger_index": trigger_index,
            "entry_index": entry_index,
            "trigger_price": current,
            "entry_price": entry,
            "causal_vwap_at_trigger": vwap,
        }
    return None


def recovery_exit(
    prices: pd.Series,
    *,
    entry_index: int,
    reference_high: float,
    holding_bars: int,
) -> dict[str, Any] | None:
    """Known target/equal-distance stop; a trigger fills on the next bar."""
    final_index = entry_index + holding_bars
    if final_index >= len(prices):
        return None
    entry = float(prices.iloc[entry_index])
    target_distance = reference_high - entry
    if not np.isfinite(entry) or entry <= 0 or target_distance <= 0:
        fixed = fixed_horizon_return(prices, entry_index, holding_bars)
        if fixed is None:
            return None
        gross, exit_index = fixed
        return {
            "gross_return": gross,
            "exit_index": exit_index,
            "exit_reason": "time_exit_no_positive_target_distance",
            "target": None,
            "stop": None,
        }
    target = reference_high
    stop = entry - target_distance
    for observation_index in range(entry_index + 1, final_index):
        observed = float(prices.iloc[observation_index])
        if not np.isfinite(observed):
            continue
        reason = None
        if observed >= target:
            reason = "known_high_recovery"
        elif observed <= stop:
            reason = "equal_distance_stop"
        if reason is None:
            continue
        exit_index = observation_index + 1
        exit_price = float(prices.iloc[exit_index])
        if not np.isfinite(exit_price) or exit_price <= 0:
            return None
        return {
            "gross_return": float(exit_price / entry - 1.0),
            "exit_index": exit_index,
            "exit_reason": reason,
            "target": target,
            "stop": stop,
        }
    exit_price = float(prices.iloc[final_index])
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None
    return {
        "gross_return": float(exit_price / entry - 1.0),
        "exit_index": final_index,
        "exit_reason": "time_exit",
        "target": target,
        "stop": stop,
    }


def select_codes(
    day_prices: pd.DataFrame,
    day_amount: pd.DataFrame,
    decision_index: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = config["selection"]
    lookback = int(selection["trendLookbackBars"])
    if decision_index < lookback:
        return []
    current = day_prices.iloc[decision_index]
    previous = day_prices.iloc[decision_index - lookback]
    trend = current / previous - 1.0
    eligible: list[dict[str, Any]] = []
    for code, value in trend.items():
        if not np.isfinite(value) or float(value) < float(
            selection["minimumTrendReturn"]
        ):
            continue
        code_prices = day_prices[code]
        code_amount = day_amount[code]
        vwap = causal_vwap(code_prices, code_amount, decision_index)
        price = float(current.get(code, np.nan))
        if not np.isfinite(price) or price <= 0:
            continue
        if selection["requirePriceAtOrAboveCausalVwap"] and (
            vwap is None or price < vwap
        ):
            continue
        eligible.append(
            {
                "stockCode": str(code),
                "trend_return": float(value),
                "decision_price": price,
                "causal_vwap": vwap,
            }
        )
    eligible.sort(key=lambda row: (-row["trend_return"], row["stockCode"]))
    count = min(
        int(selection["maximumSelectionsPerDecision"]),
        max(1, math.ceil(len(eligible) * float(selection["topFraction"]))),
    )
    return eligible[:count]


def generate_opportunities(
    prices: pd.DataFrame,
    bar_amount: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    selection = config["selection"]
    entry_cfg = config["entry"]
    exit_cfg = config["exit"]
    decision_times = set(selection["decisionTimes"])
    rows: list[dict[str, Any]] = []
    date_keys = prices.index.strftime("%Y-%m-%d")
    for trade_date in sorted(set(date_keys)):
        mask = date_keys == trade_date
        day_prices = prices.loc[mask]
        day_amount = bar_amount.loc[mask]
        for decision_index, timestamp in enumerate(day_prices.index):
            if timestamp.strftime("%H:%M") not in decision_times:
                continue
            selected = select_codes(
                day_prices, day_amount, decision_index, config
            )
            for rank, item in enumerate(selected, start=1):
                code = item["stockCode"]
                code_prices = day_prices[code]
                code_amount = day_amount[code]
                immediate_entry_index = decision_index + 1
                immediate = fixed_horizon_return(
                    code_prices,
                    immediate_entry_index,
                    int(exit_cfg["holdingBars"]),
                )
                if immediate is None:
                    continue
                high_start = max(
                    0, decision_index - int(entry_cfg["referenceHighBars"]) + 1
                )
                reference_high = float(
                    code_prices.iloc[high_start : decision_index + 1].max()
                )
                pullback = find_pullback_entry(
                    code_prices,
                    code_amount,
                    decision_index=decision_index,
                    reference_high=reference_high,
                    minimum_pullback=float(
                        entry_cfg["minimumPullbackFromKnownHigh"]
                    ),
                    maximum_wait_bars=int(entry_cfg["maximumWaitBars"]),
                    require_non_negative_last_bar=bool(
                        entry_cfg["requireNonNegativeLastBar"]
                    ),
                    require_above_vwap=bool(
                        entry_cfg["requirePriceAtOrAboveCausalVwap"]
                    ),
                )
                pullback_hold = None
                recovery = None
                if pullback is not None:
                    pullback_hold = fixed_horizon_return(
                        code_prices,
                        int(pullback["entry_index"]),
                        int(exit_cfg["holdingBars"]),
                    )
                    recovery = recovery_exit(
                        code_prices,
                        entry_index=int(pullback["entry_index"]),
                        reference_high=reference_high,
                        holding_bars=int(exit_cfg["holdingBars"]),
                    )
                rows.append(
                    {
                        "trade_date": trade_date,
                        "decision_time": timestamp.isoformat(),
                        "rank": rank,
                        **item,
                        "reference_high": reference_high,
                        "immediate_entry_time": code_prices.index[
                            immediate_entry_index
                        ].isoformat(),
                        "immediate_hold_gross_return": immediate[0],
                        "pullback_filled": pullback is not None,
                        "pullback_trigger_time": (
                            code_prices.index[pullback["trigger_index"]].isoformat()
                            if pullback
                            else None
                        ),
                        "pullback_entry_time": (
                            code_prices.index[pullback["entry_index"]].isoformat()
                            if pullback
                            else None
                        ),
                        "pullback_entry_price": (
                            pullback["entry_price"] if pullback else None
                        ),
                        "pullback_hold_gross_return": (
                            pullback_hold[0] if pullback_hold else None
                        ),
                        "pullback_recovery_gross_return": (
                            recovery["gross_return"] if recovery else None
                        ),
                        "pullback_recovery_exit_reason": (
                            recovery["exit_reason"] if recovery else None
                        ),
                    }
                )
    return rows


RETURN_FIELDS = {
    "immediate_hold_control": "immediate_hold_gross_return",
    "pullback_hold_ablation": "pullback_hold_gross_return",
    "pullback_recovery_candidate": "pullback_recovery_gross_return",
}


def daily_portfolios(
    rows: list[dict[str, Any]],
    *,
    cost_bps: float,
    max_weight: float,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    by_interval: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_interval[(row["trade_date"], row["decision_time"])].append(row)
    daily: dict[str, dict[str, float]] = {
        variant: {} for variant in RETURN_FIELDS
    }
    trades = {variant: 0 for variant in RETURN_FIELDS}
    cost = cost_bps / 10_000.0
    for (trade_date, _), selected in sorted(by_interval.items()):
        weight = min(max_weight, 1.0 / len(selected))
        for variant, field in RETURN_FIELDS.items():
            interval_return = 0.0
            for row in selected:
                gross = row.get(field)
                if gross is None:
                    continue
                interval_return += weight * (float(gross) - cost)
                trades[variant] += 1
            prior = daily[variant].get(trade_date, 0.0)
            daily[variant][trade_date] = (
                (1.0 + prior) * (1.0 + interval_return) - 1.0
            )
    return daily, trades


def metrics(daily: dict[str, float], trades: int) -> dict[str, Any]:
    values = np.asarray([daily[day] for day in sorted(daily)], dtype=float)
    if not len(values):
        return {"days": 0, "trades": trades}
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(values.mean() / std * math.sqrt(252)) if std > 0 else None
    return {
        "days": int(len(values)),
        "trades": int(trades),
        "totalReturn": float(equity[-1] - 1.0),
        "meanDailyReturn": float(values.mean()),
        "dailyStd": std,
        "dailySharpe": sharpe,
        "winDayRate": float(np.mean(values > 0)),
        "worstDay": float(values.min()),
        "maxDrawdown": float(np.min(equity / peaks - 1.0)),
    }


def paired_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    seed: int = 20260628,
    samples: int = 20_000,
) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(candidate))
    diff = np.asarray(
        [candidate[day] - baseline[day] for day in shared], dtype=float
    )
    if not len(diff):
        return {"days": 0, "meanDifference": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(samples, len(diff)))].mean(axis=1)
    return {
        "days": len(diff),
        "meanDifference": float(diff.mean()),
        "ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "probabilityDifferenceLeZero": float(np.mean(boot <= 0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render(report: dict[str, Any]) -> str:
    primary = report["costStress"]["12"]
    lines = [
        "# Trend–Pullback–Recovery Historical Replay",
        "",
        "Status: `diagnostic_only / reused OOS / no live change`",
        "",
        f"- Train: {report['split']['trainStart']} to {report['split']['trainEnd']}",
        f"- Nominal OOS: {report['split']['oosStart']} to {report['split']['oosEnd']}",
        f"- Training-only universe: {report['universe']['selectedCount']} ETFs.",
        f"- OOS selections: {report['selectionCounts']['oos']}; pullback fills: "
        f"{report['selectionCounts']['oosPullbackFills']} "
        f"({report['selectionCounts']['oosPullbackFillRate']:.1%}).",
        "",
        "## Nominal OOS at 12 bps",
        "",
        "| Variant | Return | Sharpe | Daily std | Worst day | Max DD | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in RETURN_FIELDS:
        m = primary["oosMetrics"][variant]
        lines.append(
            f"| {variant} | {m['totalReturn']:.2%} | "
            f"{m['dailySharpe'] if m['dailySharpe'] is not None else 'n/a'} | "
            f"{m['dailyStd']:.2%} | {m['worstDay']:.2%} | "
            f"{m['maxDrawdown']:.2%} | {m['trades']} |"
        )
    lines += [
        "",
        "## Evidence",
        "",
        f"- Candidate minus immediate-control bootstrap 95% CI: "
        f"{report['pairedBootstrap12bps']['ci95']}.",
        f"- DM significant: `{report['dm']['significant']}`.",
        f"- DSR significant: `{report['dsr']['significant']}`.",
        f"- SPA reject: `{report['spa']['reject']}`.",
        f"- Gate checks: `{report['gateChecks']}`.",
        "",
        "## Verdict",
        "",
        f"`{report['verdict']}`",
        "",
        report["interpretation"]["oosReuseDisclosure"],
        "",
        "A historical pass could justify record-only forward shadowing, never direct "
        "paper/live gating. A failure rejects this fixed implementation without tuning.",
        "",
        "No live configuration, overlay, order path, position sizing or execution lock was changed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    quotes = ROOT / config["data"]["quotes"]
    selected, universe = select_training_universe(
        quotes,
        train_end=config["data"]["trainEnd"],
        allowed_asset_classes=set(config["data"]["allowedAssetClasses"]),
        universe_size=int(config["data"]["trainingUniverseSize"]),
        minimum_coverage=float(config["data"]["minimumTrainingCoverageRatio"]),
    )
    panel = load_selected_panel(quotes, selected)
    rows = generate_opportunities(
        panel["prices"], panel["bar_amount"], config
    )
    train_rows = [
        row
        for row in rows
        if config["data"]["trainStart"]
        <= row["trade_date"]
        <= config["data"]["trainEnd"]
    ]
    oos_rows = [
        row
        for row in rows
        if config["data"]["oosStart"]
        <= row["trade_date"]
        <= config["data"]["oosEnd"]
    ]

    cost_stress: dict[str, Any] = {}
    daily_by_cost: dict[str, dict[str, dict[str, float]]] = {}
    trades_by_cost: dict[str, dict[str, int]] = {}
    for bps in config["portfolio"]["costStressBps"]:
        key = str(int(bps))
        train_daily, train_trades = daily_portfolios(
            train_rows,
            cost_bps=float(bps),
            max_weight=float(config["portfolio"]["maximumWeightPerSelection"]),
        )
        oos_daily, oos_trades = daily_portfolios(
            oos_rows,
            cost_bps=float(bps),
            max_weight=float(config["portfolio"]["maximumWeightPerSelection"]),
        )
        daily_by_cost[key] = oos_daily
        trades_by_cost[key] = oos_trades
        cost_stress[key] = {
            "trainMetrics": {
                variant: metrics(train_daily[variant], train_trades[variant])
                for variant in RETURN_FIELDS
            },
            "oosMetrics": {
                variant: metrics(oos_daily[variant], oos_trades[variant])
                for variant in RETURN_FIELDS
            },
        }

    primary_key = str(
        int(config["portfolio"]["primaryRoundTripCostBps"])
    )
    primary_daily = daily_by_cost[primary_key]
    primary_metrics = cost_stress[primary_key]["oosMetrics"]
    baseline = primary_daily["immediate_hold_control"]
    candidate = primary_daily["pullback_recovery_candidate"]
    bootstrap = paired_bootstrap(baseline, candidate)
    dm = diebold_mariano_hln(baseline, candidate)
    dsr = deflated_sharpe_diagnostic(
        baseline, candidate, n_trials=len(RETURN_FIELDS)
    )
    shared_days = sorted(
        set.intersection(
            *[set(primary_daily[variant]) for variant in RETURN_FIELDS]
        )
    )
    spa = reality_check_spa(
        [-primary_daily["immediate_hold_control"][day] for day in shared_days],
        {
            variant: [-primary_daily[variant][day] for day in shared_days]
            for variant in RETURN_FIELDS
            if variant != "immediate_hold_control"
        },
    )
    all_dates = sorted({row["trade_date"] for row in rows})
    pbo_matrix = []
    for variant in RETURN_FIELDS:
        daily_all, _ = daily_portfolios(
            rows,
            cost_bps=float(config["portfolio"]["primaryRoundTripCostBps"]),
            max_weight=float(config["portfolio"]["maximumWeightPerSelection"]),
        )
        pbo_matrix.append(
            [daily_all[variant].get(day, 0.0) for day in all_dates]
        )
    pbo = combinatorial_symmetric_pbo(pbo_matrix, n_blocks=8)

    oos_candidate = primary_metrics["pullback_recovery_candidate"]
    oos_baseline = primary_metrics["immediate_hold_control"]
    stress20 = cost_stress["20"]["oosMetrics"][
        "pullback_recovery_candidate"
    ]
    fill_count = sum(bool(row["pullback_filled"]) for row in oos_rows)
    gate_checks = {
        "minimumOosDays": oos_candidate["days"]
        >= int(config["evidenceGates"]["minimumOosDays"]),
        "minimumOosCandidateTrades": oos_candidate["trades"]
        >= int(config["evidenceGates"]["minimumOosCandidateTrades"]),
        "positiveCandidateNetAt12Bps": oos_candidate["totalReturn"] > 0,
        "positiveCandidateNetAt20Bps": stress20["totalReturn"] > 0,
        "positivePairedImprovementAt12Bps": (
            bootstrap["meanDifference"] is not None
            and bootstrap["meanDifference"] > 0
        ),
        "pairedBootstrapLowerPositive": (
            bootstrap["ci95"][0] is not None and bootstrap["ci95"][0] > 0
        ),
        "noWorseDailyStd": oos_candidate["dailyStd"]
        <= oos_baseline["dailyStd"],
        "noWorseWorstDay": oos_candidate["worstDay"]
        >= oos_baseline["worstDay"],
        "dmSignificant": bool(dm["significant"]),
        "dsrSignificant": bool(dsr["significant"]),
        "spaReject": bool(spa["reject"]),
    }
    historical_pass = all(gate_checks.values())
    verdict = (
        "historical_plausibility_only_forward_shadow_permitted"
        if historical_pass
        else "fixed_trend_pullback_recovery_not_validated"
    )
    report = {
        "schemaVersion": "trend_pullback_recovery_result_v1",
        "generatedAt": datetime.now(CN).isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "split": {
            key: config["data"][key]
            for key in ("trainStart", "trainEnd", "oosStart", "oosEnd")
        },
        "universe": {
            **universe,
            "selectedCount": len(selected),
        },
        "selectionCounts": {
            "train": len(train_rows),
            "oos": len(oos_rows),
            "oosPullbackFills": fill_count,
            "oosPullbackFillRate": fill_count / len(oos_rows)
            if oos_rows
            else 0.0,
        },
        "costStress": cost_stress,
        "pairedBootstrap12bps": bootstrap,
        "dm": dm,
        "dsr": dsr,
        "spa": spa,
        "pbo": pbo,
        "gateChecks": gate_checks,
        "historicalPass": historical_pass,
        "verdict": verdict,
        "interpretation": config["interpretation"],
        "safety": config["safety"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "trend_pullback_recovery_result.json", report
    )
    write_csv(
        args.output_dir / "trend_pullback_recovery_opportunities.csv", rows
    )
    (args.output_dir / "trend_pullback_recovery_report.md").write_text(
        render(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "oosSelections": len(oos_rows),
                "oosPullbackFills": fill_count,
                "primary12bps": primary_metrics,
                "gateChecks": gate_checks,
                "output": str(
                    args.output_dir / "trend_pullback_recovery_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
