"""Historical OOS test of long-only underreaction among same-index T0 ETF clones.

The hypothesis and every threshold are fixed in
``configs/research/same_index_underreaction_preregistered.json``. Universe
liquidity is selected on the training period only. A decision uses completed
bars, enters on the next observed five-minute bar and exits six bars later.

This script is offline and cannot submit orders or modify live configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
    reality_check_spa,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "same_index_underreaction_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "same_index_underreaction"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_benchmark_map(
    master_path: Path,
    allowed_asset_classes: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    code_to_benchmark: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for line in master_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        code = str(row.get("code") or "").zfill(6)
        benchmark = str(row.get("benchmark_code") or row.get("benchmark_index") or "")
        if (
            row.get("t0_confirmed")
            and not row.get("is_money_like")
            and str(row.get("asset_class")) in allowed_asset_classes
            and benchmark
        ):
            code_to_benchmark[code] = benchmark
            metadata[code] = row
    return code_to_benchmark, metadata


def training_only_groups(
    quotes_path: Path,
    code_to_benchmark: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    data_cfg = config["data"]
    train_start = str(data_cfg["trainStart"])
    train_end = str(data_cfg["trainEnd"])
    daily_amount: dict[tuple[str, str], float] = {}
    dates: set[str] = set()
    for line in quotes_path.open("r", encoding="utf-8"):
        row = json.loads(line)
        trade_date = str(row.get("trade_date") or str(row.get("timestamp"))[:10])
        code = str(row.get("stockCode") or "").zfill(6)
        if not (train_start <= trade_date <= train_end) or code not in code_to_benchmark:
            continue
        amount = float(row.get("cumulative_amount") or row.get("amount") or 0.0)
        daily_amount[(trade_date, code)] = max(
            amount, daily_amount.get((trade_date, code), 0.0)
        )
        dates.add(trade_date)
    required_days = max(
        1,
        math.ceil(
            len(dates) * float(data_cfg["minimumTrainingCoverageRatio"])
        ),
    )
    by_code: dict[str, list[float]] = defaultdict(list)
    for (_, code), amount in daily_amount.items():
        by_code[code].append(amount)
    eligible = {
        code: {
            "days": len(amounts),
            "average_amount": float(np.mean(amounts)),
        }
        for code, amounts in by_code.items()
        if len(amounts) >= required_days
        and float(np.mean(amounts))
        >= float(data_cfg["minimumTrainingAverageDailyAmount"])
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for code in eligible:
        grouped[code_to_benchmark[code]].append(code)
    maximum = int(data_cfg["maximumMembersPerBenchmark"])
    minimum = int(data_cfg["minimumMembersPerBenchmark"])
    selected: dict[str, list[str]] = {}
    for benchmark, codes in grouped.items():
        ranked = sorted(
            codes,
            key=lambda code: (eligible[code]["average_amount"], code),
            reverse=True,
        )[:maximum]
        if len(ranked) >= minimum:
            selected[benchmark] = ranked
    return dict(sorted(selected.items())), {
        "selectionUsesOos": False,
        "trainingDates": len(dates),
        "minimumCoverageDays": required_days,
        "benchmarks": len(selected),
        "codes": sum(len(codes) for codes in selected.values()),
        "groups": selected,
        "liquidity": {
            code: {
                "trainingDays": values["days"],
                "averageTrainingDailyAmount": round(values["average_amount"], 2),
            }
            for code, values in eligible.items()
            if any(code in codes for codes in selected.values())
        },
    }


def load_price_panel(quotes_path: Path, codes: set[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for line in quotes_path.open("r", encoding="utf-8"):
        row = json.loads(line)
        code = str(row.get("stockCode") or "").zfill(6)
        if code not in codes:
            continue
        price = row.get("close", row.get("currentPrice"))
        if price is None:
            continue
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]),
                "stockCode": code,
                "price": float(price),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("no same-index quote rows loaded")
    return (
        frame.pivot(index="timestamp", columns="stockCode", values="price")
        .sort_index()
    )


def interval_return(
    day_prices: pd.DataFrame,
    decision_index: int,
    code: str,
    holding_bars: int,
    maximum_entry_delay_minutes: int,
) -> float | None:
    entry_index = decision_index + 1
    exit_index = entry_index + holding_bars
    if exit_index >= len(day_prices):
        return None
    decision_time = day_prices.index[decision_index]
    entry_time = day_prices.index[entry_index]
    exit_time = day_prices.index[exit_index]
    if (entry_time - decision_time).total_seconds() > maximum_entry_delay_minutes * 60:
        return None
    if (exit_time - entry_time).total_seconds() > (holding_bars + 1) * 5 * 60:
        return None
    entry = day_prices.iloc[entry_index].get(code)
    exit_price = day_prices.iloc[exit_index].get(code)
    if not np.isfinite(entry) or not np.isfinite(exit_price) or float(entry) <= 0:
        return None
    return float(exit_price / entry - 1.0)


def generate_signals(
    prices: pd.DataFrame,
    groups: dict[str, list[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    signal_cfg = config["signal"]
    execution = config["execution"]
    formation = int(signal_cfg["formationBars"])
    holding = int(execution["holdingBars"])
    decision_times = set(signal_cfg["decisionTimes"])
    maximum_signals = int(signal_cfg["maximumBenchmarkSignalsPerDecision"])
    rows: list[dict[str, Any]] = []
    for trade_date, day_prices in prices.groupby(prices.index.strftime("%Y-%m-%d")):
        for decision_index, timestamp in enumerate(day_prices.index):
            if timestamp.strftime("%H:%M") not in decision_times:
                continue
            if decision_index < formation:
                continue
            candidates: list[dict[str, Any]] = []
            current = day_prices.iloc[decision_index]
            previous = day_prices.iloc[decision_index - formation]
            returns = current / previous - 1.0
            for benchmark, members in groups.items():
                group_returns = returns.reindex(members).dropna()
                if len(group_returns) < int(config["data"]["minimumMembersPerBenchmark"]):
                    continue
                group_momentum = float(group_returns.median())
                if group_momentum < float(signal_cfg["groupMomentumMinimum"]):
                    continue
                residuals = group_returns - group_momentum
                laggard = str(residuals.idxmin())
                laggard_residual = float(residuals.loc[laggard])
                if laggard_residual > float(signal_cfg["laggardResidualMaximum"]):
                    continue
                candidates.append(
                    {
                        "benchmark": benchmark,
                        "laggard": laggard,
                        "members": list(group_returns.index),
                        "group_momentum": group_momentum,
                        "laggard_residual": laggard_residual,
                    }
                )
            candidates.sort(key=lambda row: (row["laggard_residual"], row["benchmark"]))
            for signal in candidates[:maximum_signals]:
                selected_return = interval_return(
                    day_prices,
                    decision_index,
                    signal["laggard"],
                    holding,
                    int(execution["maximumEntryDelayMinutes"]),
                )
                peer_returns = [
                    interval_return(
                        day_prices,
                        decision_index,
                        code,
                        holding,
                        int(execution["maximumEntryDelayMinutes"]),
                    )
                    for code in signal["members"]
                    if code != signal["laggard"]
                ]
                peer_returns = [
                    value for value in peer_returns if value is not None
                ]
                if selected_return is None or not peer_returns:
                    continue
                rows.append(
                    {
                        "trade_date": trade_date,
                        "decision_time": timestamp.isoformat(),
                        "entry_time": day_prices.index[decision_index + 1].isoformat(),
                        "benchmark": signal["benchmark"],
                        "laggard": signal["laggard"],
                        "members": signal["members"],
                        "group_momentum": signal["group_momentum"],
                        "laggard_residual": signal["laggard_residual"],
                        "candidate_gross_return": selected_return,
                        "peer_control_gross_return": float(np.median(peer_returns)),
                        "paired_gross_edge": selected_return
                        - float(np.median(peer_returns)),
                    }
                )
    return rows


def daily_portfolios(
    signals: list[dict[str, Any]],
    *,
    cost_bps: float,
    max_weight: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[dict[str, Any]]]:
    by_day_time: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in signals:
        by_day_time[(row["trade_date"], row["decision_time"])].append(row)
    candidate_daily: dict[str, float] = defaultdict(float)
    control_daily: dict[str, float] = defaultdict(float)
    edge_daily: dict[str, float] = defaultdict(float)
    intervals: list[dict[str, Any]] = []
    cost = cost_bps / 10_000.0
    for (trade_date, decision_time), rows in sorted(by_day_time.items()):
        weight = min(max_weight, 1.0 / len(rows))
        exposure = weight * len(rows)
        candidate_gross = sum(
            weight * row["candidate_gross_return"] for row in rows
        )
        control_gross = sum(
            weight * row["peer_control_gross_return"] for row in rows
        )
        candidate_net = candidate_gross - exposure * cost
        control_net = control_gross - exposure * cost
        paired_edge = candidate_net - control_net
        candidate_daily[trade_date] += candidate_net
        control_daily[trade_date] += control_net
        edge_daily[trade_date] += paired_edge
        intervals.append(
            {
                "trade_date": trade_date,
                "decision_time": decision_time,
                "signals": len(rows),
                "exposure": exposure,
                "candidate_net_return": candidate_net,
                "control_net_return": control_net,
                "paired_net_edge": paired_edge,
            }
        )
    return dict(candidate_daily), dict(control_daily), dict(edge_daily), intervals


def metrics(daily: dict[str, float], trades: int) -> dict[str, Any]:
    values = np.asarray([daily[day] for day in sorted(daily)], dtype=float)
    if not len(values):
        return {
            "days": 0,
            "trades": trades,
            "totalReturn": 0.0,
            "dailySharpe": None,
            "winDayRate": None,
            "worstDay": None,
            "maxDrawdown": None,
        }
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    sharpe = (
        float(values.mean() / values.std(ddof=1) * math.sqrt(252))
        if len(values) > 1 and values.std(ddof=1) > 0
        else None
    )
    return {
        "days": len(values),
        "trades": trades,
        "totalReturn": float(equity[-1] - 1.0),
        "meanDailyReturn": float(values.mean()),
        "dailyStd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "dailySharpe": sharpe,
        "winDayRate": float(np.mean(values > 0)),
        "worstDay": float(values.min()),
        "maxDrawdown": float(np.min(equity / peaks - 1.0)),
    }


def paired_bootstrap(
    daily_edge: dict[str, float],
    *,
    samples: int = 5000,
    seed: int = 20260628,
) -> dict[str, Any]:
    values = [daily_edge[day] for day in sorted(daily_edge)]
    if len(values) < 2:
        return {"days": len(values), "lower95": None, "upper95": None}
    rng = random.Random(seed)
    means = [
        statistics_mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    ]
    return {
        "days": len(values),
        "mean": float(np.mean(values)),
        "lower95": float(np.quantile(means, 0.025)),
        "upper95": float(np.quantile(means, 0.975)),
    }


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Same-Index ETF Underreaction OOS Replay",
        "",
        "Status: `diagnostic_only / fixed hypothesis / no live change`",
        "",
        f"- Train: {report['split']['trainStart']} to {report['split']['trainEnd']}",
        f"- OOS: {report['split']['oosStart']} to {report['split']['oosEnd']}",
        f"- Training-only universe: {report['universe']['benchmarks']} benchmark groups, "
        f"{report['universe']['codes']} ETFs.",
        "- Signal: positive 15-minute group move plus lagging clone; enter next bar, hold 30 minutes.",
        "",
        "| Cost | Candidate OOS return | Peer control | Paired edge | Sharpe | Trades |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cost, result in report["costStress"].items():
        candidate = result["candidate"]
        control = result["peerControl"]
        edge = result["pairedEdge"]
        lines.append(
            f"| {cost} bps | {candidate['totalReturn']:.2%} | "
            f"{control['totalReturn']:.2%} | {edge['totalReturn']:.2%} | "
            f"{candidate['dailySharpe'] if candidate['dailySharpe'] is not None else 'n/a'} | "
            f"{candidate['trades']} |"
        )
    evidence = report["evidence"]
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Paired daily bootstrap 95% CI: "
            f"[{evidence['pairedBootstrap']['lower95']}, "
            f"{evidence['pairedBootstrap']['upper95']}].",
            f"- DM vs cash: significant=`{evidence['dmVsCash'].get('significant')}`.",
            f"- DM vs peer control: significant=`{evidence['dmVsPeer'].get('significant')}`.",
            f"- DSR: significant=`{evidence['dsr'].get('significant')}`.",
            f"- SPA: reject=`{evidence['spa'].get('reject')}`.",
            "",
            "## Verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    lines.append("This result cannot alter live entry, exits, sizing, overlays or execution locks.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quotes_path = ROOT / config["data"]["quotes"]
    master_path = ROOT / config["data"]["master"]
    code_to_benchmark, _ = load_benchmark_map(
        master_path, set(config["data"]["allowedAssetClasses"])
    )
    groups, universe_audit = training_only_groups(
        quotes_path, code_to_benchmark, config
    )
    prices = load_price_panel(
        quotes_path, {code for members in groups.values() for code in members}
    )
    signals = generate_signals(prices, groups, config)
    data_cfg = config["data"]
    train_signals = [
        row
        for row in signals
        if data_cfg["trainStart"] <= row["trade_date"] <= data_cfg["trainEnd"]
    ]
    oos_signals = [
        row
        for row in signals
        if data_cfg["oosStart"] <= row["trade_date"] <= data_cfg["oosEnd"]
    ]
    execution = config["execution"]
    cost_stress: dict[str, Any] = {}
    primary_daily: tuple[
        dict[str, float], dict[str, float], dict[str, float]
    ] | None = None
    for cost in execution["costStressBps"]:
        candidate, control, edge, _ = daily_portfolios(
            oos_signals,
            cost_bps=float(cost),
            max_weight=float(execution["maxWeightPerSignal"]),
        )
        cost_stress[str(cost)] = {
            "candidate": metrics(candidate, len(oos_signals)),
            "peerControl": metrics(control, len(oos_signals)),
            "pairedEdge": metrics(edge, len(oos_signals)),
        }
        if float(cost) == float(execution["primaryRoundTripCostBps"]):
            primary_daily = (candidate, control, edge)
    if primary_daily is None:
        raise RuntimeError("primary cost missing from cost stress")
    candidate_daily, control_daily, edge_daily = primary_daily
    shared_dates = sorted(set(candidate_daily) & set(control_daily))
    cash = {day: 0.0 for day in shared_dates}
    candidate_common = {day: candidate_daily[day] for day in shared_dates}
    control_common = {day: control_daily[day] for day in shared_dates}
    edge_common = {day: edge_daily[day] for day in shared_dates}
    dm_cash = diebold_mariano_hln(cash, candidate_common, alpha=0.05)
    dm_peer = diebold_mariano_hln(control_common, candidate_common, alpha=0.05)
    dsr = deflated_sharpe_diagnostic(
        cash, candidate_common, n_trials=2, alpha=0.10
    )
    spa = reality_check_spa(
        [0.0 for _ in shared_dates],
        {
            "candidate": [-candidate_common[day] for day in shared_dates],
            "peer_control": [-control_common[day] for day in shared_dates],
        },
        alpha=0.05,
        n_boot=1000,
        seed=20260628,
    )
    bootstrap = paired_bootstrap(edge_common)
    primary = cost_stress[str(execution["primaryRoundTripCostBps"])]
    stress = cost_stress["20"]
    gates = config["evidenceGates"]
    gate_checks = {
        "minimumOosDays": len(shared_dates) >= int(gates["minimumOosDays"]),
        "minimumOosTrades": len(oos_signals) >= int(gates["minimumOosTrades"]),
        "positiveCandidateNetAt12Bps": primary["candidate"]["totalReturn"] > 0,
        "positiveCandidateNetAt20Bps": stress["candidate"]["totalReturn"] > 0,
        "positivePairedEdge": primary["pairedEdge"]["totalReturn"] > 0,
        "pairedBootstrapLowerPositive": bootstrap.get("lower95") is not None
        and float(bootstrap["lower95"]) > 0,
        "dmVsCashSignificant": bool(dm_cash.get("significant")),
        "dmVsPeerSignificant": bool(dm_peer.get("significant")),
        "dsrSignificant": bool(dsr.get("significant")),
        "spaReject": bool(spa.get("reject")),
    }
    passed = all(gate_checks.values())
    verdict = (
        "historical_oos_plausible_forward_shadow_only"
        if passed
        else "no_validated_same_index_underreaction_edge"
    )
    report = {
        "schemaVersion": "same_index_underreaction_result_v1",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "config": str(config_path),
        "split": {
            "trainStart": data_cfg["trainStart"],
            "trainEnd": data_cfg["trainEnd"],
            "oosStart": data_cfg["oosStart"],
            "oosEnd": data_cfg["oosEnd"],
            "oosDaysWithSignals": len(shared_dates),
        },
        "universe": universe_audit,
        "signals": {
            "train": len(train_signals),
            "oos": len(oos_signals),
        },
        "costStress": cost_stress,
        "evidence": {
            "gateChecks": gate_checks,
            "pairedBootstrap": bootstrap,
            "dmVsCash": dm_cash,
            "dmVsPeer": dm_peer,
            "dsr": dsr,
            "spa": spa,
        },
        "verdict": verdict,
        "verdictReason": (
            "All fixed historical gates passed, but current benchmark membership and "
            "synthetic historical spreads require a new-date shadow test."
            if passed
            else "The fixed long-only clone-laggard rule failed at least one return, "
            "cost, paired-control or statistical gate and must not affect trading."
        ),
        "limitations": config["knownLimitations"],
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "same_index_underreaction_result.json", report)
    write_csv(output_dir / "same_index_underreaction_signals.csv", signals)
    write_csv(
        output_dir / "same_index_underreaction_oos_daily.csv",
        [
            {
                "trade_date": day,
                "candidate_net_return": candidate_common[day],
                "peer_control_net_return": control_common[day],
                "paired_net_edge": edge_common[day],
            }
            for day in shared_dates
        ],
    )
    (output_dir / "same_index_underreaction_report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "universe": {
                    "benchmarks": universe_audit["benchmarks"],
                    "codes": universe_audit["codes"],
                },
                "signals": report["signals"],
                "primary12bps": primary,
                "gateChecks": gate_checks,
                "output": str(
                    output_dir / "same_index_underreaction_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
