"""Own-ETF overnight return -> next-session intraday return research.

This differs from the existing overseas-lead study: the predictor is each ETF's
own close-to-open return, ranked cross-sectionally at 09:30. Entry uses the next
five-minute observation and exit uses the final regularly available Yahoo bar.
All universe formation and dispersion thresholds are frozen before OOS.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from overfitting_guard import combinatorial_symmetric_pbo
from research_intraday_reversal_edge import (
    CN,
    ROOT,
    atomic_json,
    load_selected_panel,
    performance,
    portfolio_interval,
    select_tail,
    select_training_universe,
    write_csv,
)
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
    reality_check_spa,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "overnight_cross_section_preregistered.json"
)
DEFAULT_QUOTES = (
    ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "overnight_cross_section"


def build_daily_observations(panel: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    prices = panel["prices"]
    previous_close = panel["prev_close"]
    observations: list[dict[str, Any]] = []
    for trade_date, day_prices in prices.groupby(prices.index.strftime("%Y-%m-%d")):
        by_time = {timestamp.strftime("%H:%M"): timestamp for timestamp in day_prices.index}
        if not {"09:30", "09:35", "14:55"}.issubset(by_time):
            continue
        signal_time = by_time["09:30"]
        entry_time = by_time["09:35"]
        exit_time = by_time["14:55"]
        signal_price = prices.loc[signal_time]
        prior = previous_close.loc[signal_time]
        overnight = signal_price / prior - 1.0
        finite_overnight = overnight[np.isfinite(overnight)]
        if len(finite_overnight) < 10:
            continue
        residual = overnight - finite_overnight.median()
        entry = prices.loc[entry_time]
        exit_price = prices.loc[exit_time]
        forward = exit_price / entry - 1.0
        finite_residual = residual[np.isfinite(residual)]
        dispersion = float(finite_residual.std(ddof=1)) if len(finite_residual) > 1 else np.nan
        observations.append(
            {
                "trade_date": trade_date,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "residual": residual,
                "forward": forward,
                "dispersion": dispersion,
            }
        )
    return observations


def training_dispersion_threshold(
    observations: list[dict[str, Any]], train_start: str, train_end: str
) -> float:
    values = [
        float(row["dispersion"])
        for row in observations
        if train_start <= row["trade_date"] <= train_end
        and np.isfinite(row["dispersion"])
    ]
    if len(values) < 10:
        raise RuntimeError("insufficient training days for dispersion threshold")
    return float(np.median(values))


def generate_intervals(
    observations: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    cost_bps: float,
    dispersion_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    execution = config["execution"]
    signal_cfg = config["signals"]
    variants = {name: [] for name in signal_cfg["variants"]}
    cost = cost_bps / 10_000.0
    shock_threshold = -float(signal_cfg["costFilteredShockMultiple"]) * cost
    fraction = float(signal_cfg["crossSectionFraction"])
    max_assets = int(execution["maxAssets"])
    max_weight = float(execution["maxWeightPerAsset"])
    for row in observations:
        residual: pd.Series = row["residual"]
        forward: pd.Series = row["forward"]
        reversal = select_tail(
            residual, fraction=fraction, side="bottom", max_assets=max_assets
        )
        selections = {
            "overnight_momentum_control": select_tail(
                residual, fraction=fraction, side="top", max_assets=max_assets
            ),
            "overnight_reversal": reversal,
            "overnight_reversal_cost_filtered": select_tail(
                residual,
                fraction=fraction,
                side="bottom",
                max_assets=max_assets,
                threshold=shock_threshold,
            ),
            "overnight_reversal_high_dispersion": reversal
            if float(row["dispersion"]) > dispersion_threshold
            else [],
        }
        forward_map = {
            str(code): float(value) if np.isfinite(value) else None
            for code, value in forward.items()
        }
        for name, selected in selections.items():
            result = portfolio_interval(
                forward_map,
                selected,
                max_weight=max_weight,
                round_trip_cost=cost,
            )
            variants[name].append(
                {
                    "trade_date": row["trade_date"],
                    "timestamp": pd.Timestamp(row["signal_time"]).isoformat(),
                    "dispersion": row["dispersion"],
                    **result,
                }
            )
    return variants


def daily_returns(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {str(row["trade_date"]): float(row["net_return"]) for row in rows}


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["costStress"]["12.0"]["oosMetrics"]
    lines = [
        "# Own-ETF Overnight–Intraday Cross-Section OOS Validation",
        "",
        "Status: `diagnostic_only / offline / no live gating`",
        "",
        f"- Training/universe formation: {report['split']['trainStart']} to {report['split']['trainEnd']}",
        f"- Untouched OOS: {report['split']['oosStart']} to {report['split']['oosEnd']}",
        "- Signal 09:30, entry 09:35, exit 14:55; primary round-trip cost 12 bps.",
        "",
        "| Variant | Net return | Sharpe | Win days | Worst day | Max DD | Exposure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row.get('netTotalReturn', 0):.2%} | "
            f"{row.get('dailySharpe') if row.get('dailySharpe') is not None else 'n/a'} | "
            f"{row.get('winDayRate', 0):.1%} | {row.get('worstDay', 0):.2%} | "
            f"{row.get('maxDrawdown', 0):.2%} | {row.get('averageExposure', 0):.1%} |"
        )
    evidence = report["evidence"]
    lines.extend(
        [
            "",
            "## Evidence gates",
            "",
            f"- Best reversal variant: `{report['bestCandidate']}`.",
            f"- Training-only dispersion threshold: `{report['trainingDispersionThreshold']:.6f}`.",
            f"- PBO: `{evidence['pbo'].get('pbo')}`.",
            f"- DM vs cash: significant=`{evidence['dm'].get('significant')}`.",
            f"- DSR: significant=`{evidence['dsr'].get('significant')}`.",
            f"- SPA: reject=`{evidence['spa'].get('reject')}`.",
            "",
            "## Locked verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "The result cannot alter live entry, exits, sizing, overlays or execution locks.",
            "",
            "## Limitations",
            "",
            "- Only 21 OOS trading days.",
            "- 09:30 is the first Yahoo five-minute close, not the auction clearing price.",
            "- Historical bid/ask and market impact are unavailable; fixed costs are a proxy.",
            "- Current ETF membership can retain survivorship bias.",
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
    observations = build_daily_observations(panel)
    dispersion_threshold = training_dispersion_threshold(
        observations, str(data_cfg["trainStart"]), str(data_cfg["trainEnd"])
    )

    cost_results: dict[str, Any] = {}
    oos_daily_by_cost: dict[str, dict[str, dict[str, float]]] = {}
    for cost_bps in execution["costStressBps"]:
        intervals = generate_intervals(
            observations,
            config,
            cost_bps=float(cost_bps),
            dispersion_threshold=dispersion_threshold,
        )
        train_rows = {
            name: [
                row for row in rows
                if str(data_cfg["trainStart"]) <= row["trade_date"] <= str(data_cfg["trainEnd"])
            ]
            for name, rows in intervals.items()
        }
        oos_rows = {
            name: [
                row for row in rows
                if str(data_cfg["oosStart"]) <= row["trade_date"] <= str(data_cfg["oosEnd"])
            ]
            for name, rows in intervals.items()
        }
        train_daily = {name: daily_returns(rows) for name, rows in train_rows.items()}
        oos_daily = {name: daily_returns(rows) for name, rows in oos_rows.items()}
        key = str(float(cost_bps))
        oos_daily_by_cost[key] = oos_daily
        cost_results[key] = {
            "trainMetricsDiagnosticOnly": {
                name: performance(train_rows[name], train_daily[name])
                for name in signal_cfg["variants"]
            },
            "oosMetrics": {
                name: performance(oos_rows[name], oos_daily[name])
                for name in signal_cfg["variants"]
            },
        }

    primary_key = str(float(execution["primaryRoundTripCostBps"]))
    primary_metrics = cost_results[primary_key]["oosMetrics"]
    candidates = [name for name in signal_cfg["variants"] if "reversal" in name]
    best = max(
        candidates,
        key=lambda name: primary_metrics[name].get("dailySharpe")
        if primary_metrics[name].get("dailySharpe") is not None
        else -999.0,
    )
    primary_daily = oos_daily_by_cost[primary_key]
    shared_dates = sorted(
        set.intersection(*(set(primary_daily[name]) for name in signal_cfg["variants"]))
    )
    pbo = combinatorial_symmetric_pbo(
        [[primary_daily[name][day] for day in shared_dates] for name in signal_cfg["variants"]],
        n_blocks=8,
    )
    cash = {day: 0.0 for day in shared_dates}
    dm = diebold_mariano_hln(cash, primary_daily[best], alpha=0.05)
    dsr = deflated_sharpe_diagnostic(
        cash, primary_daily[best], n_trials=len(signal_cfg["variants"]), alpha=0.10
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
    stress = cost_results["20.0"]["oosMetrics"][best]
    gate_cfg = config["evidenceGates"]
    gate_checks = {
        "minimumOosDays": len(shared_dates) >= int(gate_cfg["minimumOosDays"]),
        "positiveNetReturnAt12Bps": primary_metrics[best]["netTotalReturn"] > 0,
        "positiveSharpeAt12Bps": (primary_metrics[best].get("dailySharpe") or -999) > 0,
        "positiveNetReturnAt20Bps": stress["netTotalReturn"] > 0,
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
    report = {
        "schemaVersion": "overnight_cross_section_research_result_v1",
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
        "trainingDispersionThreshold": dispersion_threshold,
        "costStress": cost_results,
        "bestCandidate": best,
        "evidence": {
            "gateChecks": gate_checks,
            "pbo": pbo,
            "dm": dm,
            "dsr": dsr,
            "spa": spa,
        },
        "verdict": verdict,
        "verdictReason": (
            "Historical gates passed, but new-date forward shadow evidence remains mandatory."
            if passed
            else "At least one absolute-return, cost-stress or multiple-testing gate failed; "
            "the signal must not affect trading."
        ),
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "overnight_cross_section_oos_result.json", report)
    write_csv(
        output_dir / "overnight_cross_section_oos_daily.csv",
        [
            {
                "trade_date": day,
                **{name: primary_daily[name][day] for name in signal_cfg["variants"]},
            }
            for day in shared_dates
        ],
    )
    (output_dir / "overnight_cross_section_oos_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(
        {
            "status": report["status"],
            "verdict": verdict,
            "bestCandidate": best,
            "oosDays": len(shared_dates),
            "primaryOosMetrics": primary_metrics,
            "gateChecks": gate_checks,
            "output": str(output_dir / "overnight_cross_section_oos_report.md"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
