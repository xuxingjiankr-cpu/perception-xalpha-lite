"""Refresh the research-only forward probability ledger and diagnostics.

Consumes only forecasts already recorded at decision time and outcomes attached after
the fixed horizon.  It never fits a calibrator, learns likelihood ratios, modifies a
strategy config, enables a gate, or sizes a position.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import decision_probability as dp
from run_etf_paper_trading_agent import ROOT, as_float


OUTPUT_DIR = ROOT / "outputs" / "decision_probability_research"
DIAGNOSTIC_JSON = OUTPUT_DIR / "forward_probability_diagnostics.json"
PRIOR_JSON = OUTPUT_DIR / "bayesian_prior_registry.json"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_buy_forecasts(records: list[dict[str, Any]], effective: str) -> list[dict[str, Any]]:
    return [row for row in records
            if str(row.get("date")) >= effective
            and str(row.get("signal_direction") or row.get("decision_type")) == "BUY"
            and row.get("calibrated_probability", row.get("posterior_prob")) is not None
            and row.get("probability_outcome") is not None
            and row.get("outcome_horizon_complete") is True]


def metric_for(rows: list[dict[str, Any]], probability_field: str,
               bin_edges: list[float]) -> dict[str, Any]:
    probabilities = [as_float(row.get(probability_field), 0.5) for row in rows]
    outcomes = [int(row.get("probability_outcome")) for row in rows]
    return dp.probability_metrics(probabilities, outcomes, bin_edges=bin_edges)


def select_latest_days(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    dates = sorted({str(row.get("date")) for row in rows})
    accepted = set(dates[-count:])
    return [row for row in rows if str(row.get("date")) in accepted]


def segment_summary(rows: list[dict[str, Any]], alpha: float, beta: float,
                    confidence_k: float) -> dict[str, Any]:
    count = len(rows)
    wins = sum(int(row.get("probability_outcome")) for row in rows)
    days = len({str(row.get("date")) for row in rows})
    return {
        "count": count,
        "wins": wins,
        "losses": count - wins,
        "independentDays": days,
        "posteriorProbability": (wins + alpha) / (count + alpha + beta),
        "confidence": count / (count + confidence_k) if count else 0.0,
    }


def build_prior_registry(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    prior_cfg = config.get("bayesianPrior", {})
    alpha = as_float(prior_cfg.get("alpha"), 2.0)
    beta = as_float(prior_cfg.get("beta"), 2.0)
    confidence_k = as_float(prior_cfg.get("confidenceK"), 50.0)
    dimensions = [str(value) for value in prior_cfg.get("dimensions", [])]
    by_dimension: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(dimension) or "unknown")].append(row)
        by_dimension[dimension] = {
            key: segment_summary(values, alpha, beta, confidence_k)
            for key, values in sorted(grouped.items())
        }
    joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        joint[dp.prior_registry_key(row)].append(row)
    return {
        "schemaVersion": "decision_probability_bayesian_prior_registry_v1",
        "iterationId": config.get("iterationId"),
        "bayesianModelVersion": config.get("bayesianModelVersion"),
        "generatedAt": datetime.now().astimezone().isoformat(),
        "effectiveThrough": max((str(row.get("date")) for row in rows), default=None),
        "researchOnly": True,
        "allowedForTradeGate": False,
        "alpha": alpha,
        "beta": beta,
        "confidenceK": confidence_k,
        "overall": segment_summary(rows, alpha, beta, confidence_k),
        "byDimension": by_dimension,
        "jointSegments": {
            key: segment_summary(values, alpha, beta, confidence_k)
            for key, values in sorted(joint.items())
        },
        "configuredLikelihoodRatios": prior_cfg.get("configuredLikelihoodRatios", []),
        "autoLearnedLikelihoodRatios": [],
    }


def ledger_completeness(records: list[dict[str, Any]], effective: str,
                        max_candidates: int) -> dict[str, Any]:
    forward = [row for row in records if str(row.get("date")) >= effective]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in forward:
        groups[(str(row.get("date")), str(row.get("timestamp")))].append(row)
    expected = recorded = complete_groups = 0
    for rows in groups.values():
        final = next((row for row in rows if row.get("ledger_record_type") == "final_decision"), None)
        if not final:
            continue
        candidate_count = int(as_float(final.get("candidate_count"), 0))
        group_expected = min(candidate_count, max_candidates)
        if final.get("decision_type") == "BUY" and final.get("was_executed") and group_expected:
            group_expected -= 1
        group_recorded = sum(1 for row in rows if row.get("ledger_record_type") == "no_trade_buy_candidate")
        expected += group_expected
        recorded += group_recorded
        complete_groups += int(group_recorded == group_expected)
    return {
        "records": len(forward),
        "snapshotGroups": len(groups),
        "completeSnapshotGroups": complete_groups,
        "finalDecisions": sum(row.get("ledger_record_type") == "final_decision" for row in forward),
        "noTradeBuyCandidates": sum(row.get("ledger_record_type") == "no_trade_buy_candidate" for row in forward),
        "expectedNoTradeCandidates": expected,
        "recordedNoTradeCandidates": recorded,
        "candidateCoverage": recorded / expected if expected else None,
        "completedProbabilityOutcomes": sum(row.get("probability_outcome") is not None for row in forward),
        "incompleteProbabilityHorizons": sum(
            row.get("posterior_prob") is not None and row.get("probability_outcome") is None for row in forward
        ),
    }


def build_diagnostics(records: list[dict[str, Any]], model: dict[str, Any],
                      config: dict[str, Any]) -> dict[str, Any]:
    effective = str(model.get("effectiveFrom"))
    calibration = config.get("calibration", {})
    edges = [float(value) for value in calibration.get("binEdges", [0, .2, .4, .6, .8, 1])]
    rows = completed_buy_forecasts(records, effective)
    executed = [row for row in rows if row.get("was_executed") and row.get("decision_type") == "BUY"]
    latest_days = select_latest_days(rows, int(calibration.get("rollingTradingDays", 20)))
    latest_signals = rows[-int(calibration.get("rollingSignalCount", 50)):] if rows else []
    metrics = metric_for(rows, "calibrated_probability", edges)
    raw_metrics = metric_for(rows, "raw_probability", edges)
    bayes_metrics = metric_for(rows, "bayes_posterior_prob", edges)
    neutral = dp.probability_metrics([0.5] * len(rows),
                                     [int(row.get("probability_outcome")) for row in rows],
                                     bin_edges=edges)
    rolling_days_metrics = metric_for(latest_days, "calibrated_probability", edges)
    rolling_signal_metrics = metric_for(latest_signals, "calibrated_probability", edges)
    days = len({str(row.get("date")) for row in rows})
    min_auc = as_float(calibration.get("minimumStableAuc"), 0.60)
    max_ece = as_float(calibration.get("maximumEce"), 0.05)
    proper_score_pass = bool(
        metrics.get("brier") is not None and neutral.get("brier") is not None
        and metrics["brier"] < neutral["brier"]
        and metrics["log_loss"] < neutral["log_loss"]
    )
    stable_auc_pass = bool(
        metrics.get("auc") is not None and metrics["auc"] >= min_auc
        and rolling_days_metrics.get("auc") is not None and rolling_days_metrics["auc"] >= min_auc
        and rolling_signal_metrics.get("auc") is not None and rolling_signal_metrics["auc"] >= min_auc
    )
    calibration_pass = metrics.get("ece") is not None and metrics["ece"] <= max_ece
    tier_results = []
    for tier in config.get("readinessTiers", []):
        sample_pass = (len(rows) >= int(tier.get("minimumBuyOutcomes", 0))
                       and days >= int(tier.get("minimumIndependentTradingDays", 0)))
        statistical_pass = sample_pass and proper_score_pass and calibration_pass and stable_auc_pass
        if tier.get("name") == "position_candidate":
            statistical_pass = False  # EV-bin stability is not yet established.
        tier_results.append({**tier, "samplePass": sample_pass,
                             "statisticalPass": statistical_pass})
    candidate_cfg = config.get("candidateLedger", {})
    return {
        "schemaVersion": "decision_probability_forward_diagnostics_v1",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "iterationId": config.get("iterationId"),
        "effectiveFrom": effective,
        "researchOnly": True,
        "tradeGateEnabled": False,
        "positionSizingEnabled": False,
        "sample": {"completedBuyForecasts": len(rows), "independentTradingDays": days,
                   "executedBuyForecasts": len(executed),
                   "noTradeCandidateForecasts": len(rows) - len(executed)},
        "ledger": ledger_completeness(
            records, effective, int(candidate_cfg.get("maxRankedCandidatesPerSnapshot", 30)),
        ),
        "calibration": metrics,
        "rawProbabilityHistogram": raw_metrics,
        "bayesianCalibration": bayes_metrics,
        "neutralHalf": neutral,
        "rolling20TradingDays": rolling_days_metrics,
        "rolling50Signals": rolling_signal_metrics,
        "gates": {"properScoresBeatNeutralHalf": proper_score_pass,
                  "ecePass": calibration_pass, "stableAucPass": stable_auc_pass,
                  "maximumEce": max_ece, "minimumStableAuc": min_auc,
                  "recommendationOnly": True, "promotionAllowed": False},
        "readinessTiers": tier_results,
        "daily": daily_metrics(rows, edges),
    }


def daily_metrics(rows: list[dict[str, Any]], edges: list[float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("date"))].append(row)
    result = []
    for date, values in sorted(grouped.items()):
        result.append({"date": date, **metric_for(values, "calibrated_probability", edges)})
    return result


def format_metric(value: Any, percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2%}" if percent else f"{float(value):.6f}"


def render_ledger(diagnostic: dict[str, Any]) -> str:
    sample, ledger = diagnostic["sample"], diagnostic["ledger"]
    lines = ["# Forward Shadow Probability Ledger Summary", "",
             "Status: `research_only / trade_invalid_probability`", "",
             f"- effective_from: {diagnostic['effectiveFrom']}",
             f"- completed BUY-direction forecasts: {sample['completedBuyForecasts']} across {sample['independentTradingDays']} days",
             f"- executed BUY / no-trade counterfactual: {sample['executedBuyForecasts']} / {sample['noTradeCandidateForecasts']}",
             f"- final decisions / no-trade candidates recorded: {ledger['finalDecisions']} / {ledger['noTradeBuyCandidates']}",
             f"- candidate coverage: {format_metric(ledger['candidateCoverage'], True)}",
             f"- incomplete fixed horizons: {ledger['incompleteProbabilityHorizons']}", "",
             "No candidate row is treated as a fill. Candidate outcomes are explicitly next-snapshot-to-close counterfactual BUY returns.",
             "No probability field can gate or size an order.", "", "## Daily diagnostics", "",
             "| date | n | mean p | actual rate | Brier | LogLoss | AUC | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in diagnostic["daily"][-20:]:
        lines.append(f"| {row['date']} | {row['count']} | {format_metric(row['mean_predicted'], True)} | "
                     f"{format_metric(row['actual_rate'], True)} | {format_metric(row['brier'])} | "
                     f"{format_metric(row['log_loss'])} | {format_metric(row['auc'])} | {format_metric(row['ece'])} |")
    return "\n".join(lines) + "\n"


def render_bins(diagnostic: dict[str, Any]) -> str:
    metrics = diagnostic["calibration"]
    raw = diagnostic["rawProbabilityHistogram"]
    lines = ["# Calibration Bins — DCAL-1.0.1", "",
             "No Platt/isotonic refit is applied. These are forward diagnostics only.", "",
             f"- Brier: {format_metric(metrics['brier'])}",
             f"- LogLoss: {format_metric(metrics['log_loss'])}",
             f"- AUC: {format_metric(metrics['auc'])}",
             f"- ECE / MCE: {format_metric(metrics['ece'])} / {format_metric(metrics['mce'])}", "",
             "| probability_bin | count | avgPredProb | actualWinRate | calibrationError | Brier | LogLoss |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics["bins"]:
        lines.append(f"| {row['low']:.2f}—{row['high']:.2f} | {row['count']} | "
                     f"{format_metric(row['mean_predicted'], True)} | {format_metric(row['actual_rate'], True)} | "
                     f"{format_metric(row['calibration_error'], True)} | {format_metric(row['brier'])} | "
                     f"{format_metric(row['log_loss'])} |")
    lines.extend(["", "## Raw probability histogram", "",
                  "DCAL-1.0.1 applies no new calibration transform, so raw and calibrated probabilities are currently identical by design.", "",
                  "| probability_bin | count | share |", "|---|---:|---:|"])
    for row in raw["bins"]:
        share = row["count"] / raw["count"] if raw["count"] else 0.0
        lines.append(f"| {row['low']:.2f}—{row['high']:.2f} | {row['count']} | {share:.2%} |")
    return "\n".join(lines) + "\n"


def render_priors(registry: dict[str, Any]) -> str:
    overall = registry["overall"]
    lines = ["# Bayesian Prior Registry — DBAYES-1.0.1", "",
             "Status: `research_only`; allowed_for_trade_gate: `false`", "",
             f"Beta prior: alpha={registry['alpha']}, beta={registry['beta']}; confidence=n/(n+{registry['confidenceK']:.0f}).",
             f"Overall: n={overall['count']}, days={overall['independentDays']}, posterior={overall['posteriorProbability']:.2%}, confidence={overall['confidence']:.2%}.",
             "Configured LR values are config-only; none are learned automatically.", ""]
    for dimension, groups in registry["byDimension"].items():
        lines.extend([f"## {dimension}", "", "| group | n | days | posterior | confidence |",
                      "|---|---:|---:|---:|---:|"])
        for name, value in groups.items():
            lines.append(f"| {name} | {value['count']} | {value['independentDays']} | "
                         f"{value['posteriorProbability']:.2%} | {value['confidence']:.2%} |")
        lines.append("")
    return "\n".join(lines)


def render_readiness(diagnostic: dict[str, Any]) -> str:
    gates = diagnostic["gates"]
    metrics = diagnostic["calibration"]
    neutral = diagnostic["neutralHalf"]
    lines = ["# Probability Readiness Check", "",
             "Final status: `research-valid start / trade-invalid probability`", "",
             f"- Brier vs neutral 50%: {format_metric(metrics['brier'])} vs {format_metric(neutral['brier'])}",
             f"- LogLoss vs neutral 50%: {format_metric(metrics['log_loss'])} vs {format_metric(neutral['log_loss'])}",
             f"- ECE <= {gates['maximumEce']:.2%}: `{gates['ecePass']}`",
             f"- full + rolling AUC >= {gates['minimumStableAuc']:.2f}: `{gates['stableAucPass']}`",
             f"- both proper scores beat neutral: `{gates['properScoresBeatNeutralHalf']}`", "",
             "## Full and rolling diagnostics", "",
             "| window | n | Brier | LogLoss | AUC | ECE | MCE |",
             "|---|---:|---:|---:|---:|---:|---:|",
             f"| full forward | {metrics['count']} | {format_metric(metrics['brier'])} | {format_metric(metrics['log_loss'])} | {format_metric(metrics['auc'])} | {format_metric(metrics['ece'])} | {format_metric(metrics['mce'])} |",
             f"| latest 20 trading days | {diagnostic['rolling20TradingDays']['count']} | {format_metric(diagnostic['rolling20TradingDays']['brier'])} | {format_metric(diagnostic['rolling20TradingDays']['log_loss'])} | {format_metric(diagnostic['rolling20TradingDays']['auc'])} | {format_metric(diagnostic['rolling20TradingDays']['ece'])} | {format_metric(diagnostic['rolling20TradingDays']['mce'])} |",
             f"| latest 50 signals | {diagnostic['rolling50Signals']['count']} | {format_metric(diagnostic['rolling50Signals']['brier'])} | {format_metric(diagnostic['rolling50Signals']['log_loss'])} | {format_metric(diagnostic['rolling50Signals']['auc'])} | {format_metric(diagnostic['rolling50Signals']['ece'])} | {format_metric(diagnostic['rolling50Signals']['mce'])} |",
             "",
             "| tier | minimum outcomes | minimum days | sample pass | statistical pass | action |",
             "|---|---:|---:|---:|---:|---|"]
    actions = {"shadow_warmup": "report only", "research_validated": "consider simulated filtering",
               "gate_candidate": "recommendation report only", "position_candidate": "requires stable EV bins"}
    for tier in diagnostic["readinessTiers"]:
        lines.append(f"| {tier['name']} | {tier['minimumBuyOutcomes']} | {tier['minimumIndependentTradingDays']} | "
                     f"{str(tier['samplePass']).lower()} | {str(tier['statisticalPass']).lower()} | "
                     f"{actions.get(tier['name'], 'report only')} |")
    lines.extend(["", "Even if every statistical gate passes, this pipeline cannot enable live filtering or sizing.", ""])
    return "\n".join(lines)


def refresh_reports(records: list[dict[str, Any]]) -> dict[str, Any]:
    model = dp.load_shadow_model()
    config = dp.load_research_config()
    if not model or not config:
        raise RuntimeError("safe probability model/research config missing")
    diagnostics = build_diagnostics(records, model, config)
    forward_rows = completed_buy_forecasts(records, str(model.get("effectiveFrom")))
    registry = build_prior_registry(forward_rows, config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(DIAGNOSTIC_JSON, diagnostics)
    atomic_json(PRIOR_JSON, registry)
    (OUTPUT_DIR / "forward_shadow_ledger_summary.md").write_text(render_ledger(diagnostics), encoding="utf-8")
    (OUTPUT_DIR / "calibration_bins.md").write_text(render_bins(diagnostics), encoding="utf-8")
    (OUTPUT_DIR / "bayesian_prior_registry.md").write_text(render_priors(registry), encoding="utf-8")
    (OUTPUT_DIR / "probability_readiness_check.md").write_text(render_readiness(diagnostics), encoding="utf-8")
    return diagnostics


def main() -> int:
    import run_decision_score_report as score_report
    diagnostics = refresh_reports(score_report.load_records(None))
    print(render_readiness(diagnostics))
    print(f"reports: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
