"""Frozen Laplace-copula test of causal same-index ETF price chains.

The model is fitted only on the registered training window. At each registered
decision time it uses completed bars, enters a selected laggard on the next
observed five-minute bar and exits after the fixed holding period.

This script is offline research. It cannot submit orders, edit live configs or
write a strategy overlay.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from research_same_index_underreaction import (
    ROOT,
    atomic_json,
    daily_portfolios,
    interval_return,
    load_benchmark_map,
    load_price_panel,
    metrics,
    paired_bootstrap,
    training_only_groups,
    write_csv,
)
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
    reality_check_spa,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "laplace_copula_price_chain_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "laplace_copula_price_chain"
STANDARD_NORMAL = NormalDist()


def laplace_fit(values: np.ndarray) -> tuple[float, float]:
    """Return the Laplace MLE (median, mean absolute deviation from median)."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        raise ValueError("cannot fit an empty Laplace sample")
    location = float(np.median(clean))
    scale = float(np.mean(np.abs(clean - location)))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Laplace scale is zero")
    return location, scale


def laplace_cdf(value: float, location: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("Laplace scale must be positive")
    if value < location:
        return 0.5 * math.exp((value - location) / scale)
    return 1.0 - 0.5 * math.exp(-(value - location) / scale)


def normal_score(
    value: float,
    location: float,
    scale: float,
    *,
    clip: float,
) -> float:
    probability = min(1.0 - clip, max(clip, laplace_cdf(value, location, scale)))
    return float(STANDARD_NORMAL.inv_cdf(probability))


def conditional_lower_tail(
    leader_return: float,
    lagger_return: float,
    model: dict[str, float],
    *,
    clip: float,
) -> float:
    """P(Z_lagger <= observed | Z_leader) under a frozen Gaussian copula."""
    leader_z = normal_score(
        leader_return,
        model["leaderLocation"],
        model["leaderScale"],
        clip=clip,
    )
    lagger_z = normal_score(
        lagger_return,
        model["laggerLocation"],
        model["laggerScale"],
        clip=clip,
    )
    rho = max(-0.999, min(0.999, float(model["rho"])))
    conditional_z = (lagger_z - rho * leader_z) / math.sqrt(1.0 - rho * rho)
    return float(STANDARD_NORMAL.cdf(conditional_z))


def _formation_returns(
    day_prices: pd.DataFrame,
    decision_index: int,
    formation_bars: int,
) -> pd.Series:
    current = day_prices.iloc[decision_index]
    previous = day_prices.iloc[decision_index - formation_bars]
    return current / previous - 1.0


def collect_training_pair_returns(
    prices: pd.DataFrame,
    groups: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    formation = int(model_cfg["formationBars"])
    decision_times = set(model_cfg["decisionTimes"])
    observations: dict[tuple[str, str, str], list[tuple[float, float]]] = (
        defaultdict(list)
    )
    for trade_date, day_prices in prices.groupby(prices.index.strftime("%Y-%m-%d")):
        if not (data_cfg["trainStart"] <= trade_date <= data_cfg["trainEnd"]):
            continue
        for decision_index, timestamp in enumerate(day_prices.index):
            if (
                timestamp.strftime("%H:%M") not in decision_times
                or decision_index < formation
            ):
                continue
            returns = _formation_returns(day_prices, decision_index, formation)
            for benchmark, members in groups.items():
                joint = returns.reindex(members).dropna()
                for leader in joint.index:
                    for lagger in joint.index:
                        if leader == lagger:
                            continue
                        observations[(benchmark, str(leader), str(lagger))].append(
                            (float(joint.loc[leader]), float(joint.loc[lagger]))
                        )
    return {
        key: (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
        )
        for key, rows in observations.items()
    }


def _laplace_minus_gaussian_log_likelihood(values: np.ndarray) -> float:
    location, scale = laplace_fit(values)
    laplace_ll = float(
        np.sum(-math.log(2.0 * scale) - np.abs(values - location) / scale)
    )
    gaussian_location = float(np.mean(values))
    gaussian_scale = float(np.std(values, ddof=0))
    if gaussian_scale <= 1e-12:
        return float("-inf")
    gaussian_ll = float(
        np.sum(
            -math.log(gaussian_scale * math.sqrt(2.0 * math.pi))
            - 0.5 * ((values - gaussian_location) / gaussian_scale) ** 2
        )
    )
    return laplace_ll - gaussian_ll


def fit_frozen_pair_models(
    pair_returns: dict[
        tuple[str, str, str], tuple[np.ndarray, np.ndarray]
    ],
    config: dict[str, Any],
) -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[str, Any]]:
    model_cfg = config["model"]
    minimum = int(model_cfg["minimumTrainingPairObservations"])
    minimum_rho = float(model_cfg["minimumFrozenGaussianCopulaRho"])
    clip = float(model_cfg["cdfClip"])
    models: dict[tuple[str, str, str], dict[str, float]] = {}
    marginal_deltas: list[float] = []
    rejected = defaultdict(int)
    for key, (leader_values, lagger_values) in pair_returns.items():
        if len(leader_values) < minimum:
            rejected["insufficient_training_observations"] += 1
            continue
        try:
            leader_location, leader_scale = laplace_fit(leader_values)
            lagger_location, lagger_scale = laplace_fit(lagger_values)
        except ValueError:
            rejected["degenerate_laplace_scale"] += 1
            continue
        leader_z = np.asarray(
            [
                normal_score(
                    value, leader_location, leader_scale, clip=clip
                )
                for value in leader_values
            ],
            dtype=float,
        )
        lagger_z = np.asarray(
            [
                normal_score(
                    value, lagger_location, lagger_scale, clip=clip
                )
                for value in lagger_values
            ],
            dtype=float,
        )
        rho = float(np.corrcoef(leader_z, lagger_z)[0, 1])
        if not math.isfinite(rho) or rho < minimum_rho:
            rejected["copula_rho_below_floor"] += 1
            continue
        models[key] = {
            "observations": float(len(leader_values)),
            "leaderLocation": leader_location,
            "leaderScale": leader_scale,
            "laggerLocation": lagger_location,
            "laggerScale": lagger_scale,
            "rho": min(0.999, rho),
        }
        marginal_deltas.extend(
            [
                _laplace_minus_gaussian_log_likelihood(leader_values),
                _laplace_minus_gaussian_log_likelihood(lagger_values),
            ]
        )
    finite_deltas = [value for value in marginal_deltas if math.isfinite(value)]
    audit = {
        "candidateOrderedPairs": len(pair_returns),
        "acceptedOrderedPairs": len(models),
        "rejected": dict(rejected),
        "laplaceMarginalFit": {
            "comparisons": len(finite_deltas),
            "fractionHigherLikelihoodThanGaussian": (
                float(np.mean(np.asarray(finite_deltas) > 0))
                if finite_deltas
                else None
            ),
            "medianLogLikelihoodAdvantage": (
                float(np.median(finite_deltas)) if finite_deltas else None
            ),
        },
    }
    return models, audit


def generate_price_chain_signals(
    prices: pd.DataFrame,
    groups: dict[str, list[str]],
    models: dict[tuple[str, str, str], dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model_cfg = config["model"]
    signal_cfg = config["signal"]
    execution = config["execution"]
    formation = int(model_cfg["formationBars"])
    holding = int(execution["holdingBars"])
    decision_times = set(model_cfg["decisionTimes"])
    maximum_signals = int(signal_cfg["maximumBenchmarkSignalsPerDecision"])
    clip = float(model_cfg["cdfClip"])
    rows: list[dict[str, Any]] = []
    for trade_date, day_prices in prices.groupby(prices.index.strftime("%Y-%m-%d")):
        for decision_index, timestamp in enumerate(day_prices.index):
            if (
                timestamp.strftime("%H:%M") not in decision_times
                or decision_index < formation
            ):
                continue
            formation_returns = _formation_returns(
                day_prices, decision_index, formation
            )
            candidates: list[dict[str, Any]] = []
            for benchmark, members in groups.items():
                group_returns = formation_returns.reindex(members).dropna()
                best: dict[str, Any] | None = None
                for leader in group_returns.index:
                    leader_return = float(group_returns.loc[leader])
                    if leader_return < float(
                        signal_cfg["leaderMomentumMinimum"]
                    ):
                        continue
                    for lagger in group_returns.index:
                        if leader == lagger:
                            continue
                        lagger_return = float(group_returns.loc[lagger])
                        gap = leader_return - lagger_return
                        if gap < float(
                            signal_cfg["leaderLaggerReturnGapMinimum"]
                        ):
                            continue
                        model = models.get(
                            (benchmark, str(leader), str(lagger))
                        )
                        if model is None:
                            continue
                        tail = conditional_lower_tail(
                            leader_return,
                            lagger_return,
                            model,
                            clip=clip,
                        )
                        if tail > float(
                            signal_cfg["conditionalLowerTailMaximum"]
                        ):
                            continue
                        proposal = {
                            "benchmark": benchmark,
                            "leader": str(leader),
                            "laggard": str(lagger),
                            "members": list(group_returns.index),
                            "leader_return": leader_return,
                            "laggard_return": lagger_return,
                            "leader_laggard_gap": gap,
                            "conditional_lower_tail": tail,
                            "frozen_rho": model["rho"],
                        }
                        if best is None or (
                            proposal["conditional_lower_tail"],
                            proposal["laggard"],
                        ) < (
                            best["conditional_lower_tail"],
                            best["laggard"],
                        ):
                            best = proposal
                if best is not None:
                    candidates.append(best)
            candidates.sort(
                key=lambda row: (
                    row["conditional_lower_tail"],
                    row["benchmark"],
                )
            )
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
                peer_control = float(np.median(peer_returns))
                rows.append(
                    {
                        "trade_date": trade_date,
                        "decision_time": timestamp.isoformat(),
                        "entry_time": day_prices.index[
                            decision_index + 1
                        ].isoformat(),
                        **signal,
                        "candidate_gross_return": selected_return,
                        "peer_control_gross_return": peer_control,
                        "paired_gross_edge": selected_return - peer_control,
                    }
                )
    return rows


def _fill_dates(values: dict[str, float], dates: list[str]) -> dict[str, float]:
    return {day: float(values.get(day, 0.0)) for day in dates}


def markdown(report: dict[str, Any]) -> str:
    fit = report["modelAudit"]["laplaceMarginalFit"]
    lines = [
        "# Laplace-Copula Same-Index ETF Price-Chain Replay",
        "",
        "Status: `diagnostic_only / reused historical OOS / no live change`",
        "",
        f"- Train fit: {report['split']['trainStart']} to {report['split']['trainEnd']}.",
        f"- Exploratory test: {report['split']['oosStart']} to {report['split']['oosEnd']}.",
        f"- Universe: {report['universe']['benchmarks']} benchmark groups, "
        f"{report['universe']['codes']} ETFs.",
        f"- Frozen ordered pair models: {report['modelAudit']['acceptedOrderedPairs']}.",
        f"- Laplace likelihood beat Gaussian in "
        f"{fit['fractionHigherLikelihoodThanGaussian']:.1%} of fitted marginal comparisons."
        if fit["fractionHigherLikelihoodThanGaussian"] is not None
        else "- Marginal fit comparison unavailable.",
        "- Signal: leader up at least 0.20% over 15 minutes; lagger at least "
        "0.15% behind and below the frozen 5% conditional tail; next-bar entry, "
        "30-minute hold.",
        "",
        "| Cost | Candidate return | Peer control | Paired edge | Sharpe | Trades |",
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
            f"- DM vs peer: significant=`{evidence['dmVsPeer'].get('significant')}`.",
            f"- DSR: significant=`{evidence['dsr'].get('significant')}`.",
            f"- SPA: reject=`{evidence['spa'].get('reject')}`.",
            f"- Fresh unseen OOS gate: `False` (the test dates were already reused).",
            "",
            "## Verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "A superior Laplace marginal fit only describes heavy tails; it is not "
            "evidence that the conditional event predicts a profitable catch-up.",
            "",
            "## Paper applicability audit",
            "",
            "- The 2018 Laplace-transform cash-flow article computes discounted "
            "present value and supplies no price-prediction mechanism.",
            "- The 2022 paper motivates Laplace marginals plus a Gaussian copula, "
            "but reports one illustrative trade and ignores costs; this replay is "
            "therefore an independent long-only hypothesis test.",
            "- arXiv:2607.01638 concerns Laplace-Beltrami PDEs for liquid crystals "
            "and has no defensible mapping to financial price chains.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "This result cannot alter live entries, exits, sizing, overlays or "
            "execution locks.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    quotes_path = ROOT / data_cfg["quotes"]
    master_path = ROOT / data_cfg["master"]
    code_to_benchmark, _ = load_benchmark_map(
        master_path, set(data_cfg["allowedAssetClasses"])
    )
    groups, universe_audit = training_only_groups(
        quotes_path, code_to_benchmark, config
    )
    prices = load_price_panel(
        quotes_path, {code for members in groups.values() for code in members}
    )
    pair_returns = collect_training_pair_returns(prices, groups, config)
    models, model_audit = fit_frozen_pair_models(pair_returns, config)
    signals = generate_price_chain_signals(prices, groups, models, config)
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
    oos_dates = sorted(
        {
            timestamp.strftime("%Y-%m-%d")
            for timestamp in prices.index
            if data_cfg["oosStart"]
            <= timestamp.strftime("%Y-%m-%d")
            <= data_cfg["oosEnd"]
        }
    )
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
        candidate = _fill_dates(candidate, oos_dates)
        control = _fill_dates(control, oos_dates)
        edge = _fill_dates(edge, oos_dates)
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
    cash = {day: 0.0 for day in oos_dates}
    dm_cash = diebold_mariano_hln(cash, candidate_daily, alpha=0.05)
    dm_peer = diebold_mariano_hln(control_daily, candidate_daily, alpha=0.05)
    dsr = deflated_sharpe_diagnostic(
        cash, candidate_daily, n_trials=2, alpha=0.10
    )
    spa = reality_check_spa(
        [0.0 for _ in oos_dates],
        {
            "candidate": [-candidate_daily[day] for day in oos_dates],
            "peer_control": [-control_daily[day] for day in oos_dates],
        },
        alpha=0.05,
        n_boot=1000,
        seed=20260704,
    )
    bootstrap = paired_bootstrap(
        edge_daily, samples=5000, seed=20260704
    )
    primary = cost_stress[str(execution["primaryRoundTripCostBps"])]
    stress = cost_stress["20"]
    gates = config["evidenceGates"]
    gate_checks = {
        "minimumOosDays": len(oos_dates) >= int(gates["minimumOosDays"]),
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
        "freshUnseenOosForPromotion": not bool(
            data_cfg["oosWindowPreviouslyReused"]
        ),
    }
    numerical_checks = {
        key: value
        for key, value in gate_checks.items()
        if key != "freshUnseenOosForPromotion"
    }
    historical_plausible = all(numerical_checks.values())
    verdict = (
        "historical_pattern_only_requires_fresh_forward_validation"
        if historical_plausible
        else "no_validated_laplace_copula_price_chain_edge"
    )
    report = {
        "schemaVersion": "laplace_copula_price_chain_result_v1",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "config": str(config_path),
        "split": {
            "trainStart": data_cfg["trainStart"],
            "trainEnd": data_cfg["trainEnd"],
            "oosStart": data_cfg["oosStart"],
            "oosEnd": data_cfg["oosEnd"],
            "oosDays": len(oos_dates),
            "oosWindowPreviouslyReused": bool(
                data_cfg["oosWindowPreviouslyReused"]
            ),
        },
        "universe": universe_audit,
        "modelAudit": model_audit,
        "signals": {"train": len(train_signals), "oos": len(oos_signals)},
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
            "The frozen historical numerical gates passed, but the reused test "
            "window makes promotion invalid; only new forward dates can confirm it."
            if historical_plausible
            else "The frozen conditional-tail rule failed at least one return, "
            "cost, sample, paired-control or statistical gate."
        ),
        "limitations": config["knownLimitations"],
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "laplace_copula_price_chain_result.json", report)
    write_csv(output_dir / "laplace_copula_price_chain_signals.csv", signals)
    write_csv(
        output_dir / "laplace_copula_price_chain_oos_daily.csv",
        [
            {
                "trade_date": day,
                "candidate_net_return": candidate_daily[day],
                "peer_control_net_return": control_daily[day],
                "paired_net_edge": edge_daily[day],
            }
            for day in oos_dates
        ],
    )
    (output_dir / "laplace_copula_price_chain_report.md").write_text(
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
                "modelAudit": model_audit,
                "signals": report["signals"],
                "primary12bps": primary,
                "gateChecks": gate_checks,
                "output": str(
                    output_dir / "laplace_copula_price_chain_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
