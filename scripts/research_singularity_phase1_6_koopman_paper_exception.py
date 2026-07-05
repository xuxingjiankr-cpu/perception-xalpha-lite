"""Evaluate the user-authorized Koopman paper-only coverage exception.

Coverage, 10-bar evaluation and high-risk sample count may be waived. Causal
same-sample comparison, majority stability and the trading-date clustered
Brier confidence interval cannot be waived. The script never edits the paper
agent or its configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from research_hmm_nn_bl import ROOT
import research_singularity_phase1_6_hmm_physics_model as physics
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "singularity_phase1_6_koopman_paper_exception_v1.json"
)
PAPER_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
AGENT_SOURCE = ROOT / "scripts" / "run_t0_intraday_agent.py"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schemaVersion")
        != "singularity_phase1_6_koopman_paper_exception_v1"
    ):
        raise ValueError("unexpected Koopman exception schema")
    if (
        config.get("status") != "research_only"
        or config.get("diagnosticOnly") is not True
        or config.get("userAuthorizedPaperException") is not True
    ):
        raise ValueError("exception audit must remain explicit research")
    exemptions = config["paperOnlyExemptions"]
    if not all(
        exemptions.get(key) is True
        for key in [
            "fullSessionCoverageMinimumWaived",
            "tenBarEvaluationWaived",
            "highRiskMinimumSampleGateWaived",
        ]
    ):
        raise ValueError("paper exception contract changed")
    integration = config["paperIntegration"]
    if (
        integration.get("allowedOnlyIfAllNonWaivableGatesPass") is not True
        or integration.get("policyTypeIfEligible") != "risk_veto_only"
        or integration.get("mayGenerateIndependentBuyOrSell") is not False
        or integration.get("mayChangeSellPath") is not False
        or integration.get("mayBypassTripleLock") is not False
        or integration.get("automaticPromotionAllowed") is not False
    ):
        raise ValueError("paper integration exception is too broad")
    if config["source"].get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 ledger cannot be an input")


def render_report(result: dict[str, Any]) -> str:
    metric = result["metrics"]
    diagnostic = result["diagnostic"]
    bootstrap = diagnostic["clusterBootstrap"]
    return "\n".join(
        [
            "# Koopman paper-only exception audit",
            "",
            f"- Run ID: `{result['runId']}`",
            f"- Status: `{result['status']}`",
            f"- Same-sample rows: {metric['baseline']['count']:,}",
            f"- Coverage exception used: `{result['exemptionsApplied']['coverage']}`",
            f"- 10-bar exception used: `{result['exemptionsApplied']['tenBar']}`",
            f"- High-risk sample exception used: `{result['exemptionsApplied']['highRiskSamples']}`",
            "",
            "| variant | Brier | LogLoss | AUC | ECE | high-risk n |",
            "|---|---:|---:|---:|---:|---:|",
            f"| HMM+EWS+LPPLS | {metric['baseline']['brier']:.6f} | "
            f"{metric['baseline']['log_loss']:.6f} | "
            f"{metric['baseline']['auc']:.6f} | "
            f"{metric['baseline']['ece']:.6f} | "
            f"{metric['baseline']['highRiskCount']} |",
            f"| +Koopman/DMD | {metric['candidate']['brier']:.6f} | "
            f"{metric['candidate']['log_loss']:.6f} | "
            f"{metric['candidate']['auc']:.6f} | "
            f"{metric['candidate']['ece']:.6f} | "
            f"{metric['candidate']['highRiskCount']} |",
            "",
            f"- Improved metrics: {diagnostic['improvedMetricCountOfFour']}/4",
            f"- Improving folds: {diagnostic['improvingFolds']}/{diagnostic['totalFolds']}",
            f"- Improving months: {diagnostic['improvingMonths']}/{diagnostic['totalMonths']}",
            f"- Improving ETF categories: {len(diagnostic['improvingEtfCategories'])}",
            f"- Brier delta candidate-baseline: "
            f"{bootstrap['meanBrierDeltaCandidateMinusBaseline']:.8f}",
            f"- Cluster bootstrap CI: [{bootstrap['lower']:.8f}, "
            f"{bootstrap['upper']:.8f}]",
            "",
            f"- Non-waivable gates passed: `{result['nonWaivableGatesPassed']}`",
            f"- Paper risk-veto integration allowed: `{result['paperIntegrationAllowed']}`",
            f"- Main paper config modified: `{result['paperConfigModified']}`",
            "",
            result["conclusion"],
            "",
        ]
    )


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paper_hash_before = sha256(PAPER_CONFIG)
    agent_hash_before = sha256(AGENT_SOURCE)
    model_config = json.loads(
        resolve(config["source"]["modelConfig"]).read_text(encoding="utf-8")
    )
    physics.validate_config(model_config)
    feature_table = pd.read_csv(
        resolve(config["source"]["featureTable"]),
        parse_dates=["timestamp"],
    )
    evaluation_config = copy.deepcopy(model_config)
    population = config["comparison"]["population"]
    evaluation_config["populations"][population] = {
        "variants": [
            config["comparison"]["baseline"],
            config["comparison"]["candidate"],
        ],
        "requiresDmd": True,
        "cannotBeExtrapolatedToFullSession": True,
    }
    predictions, fold_audits = physics.run_population_walk_forward(
        feature_table,
        population,
        evaluation_config["populations"][population]["variants"],
        evaluation_config,
    )
    metrics = physics.summarize_predictions(
        predictions, evaluation_config
    )
    horizon = str(config["comparison"]["primaryHorizonBars"])
    baseline_name = config["comparison"]["baseline"]
    candidate_name = config["comparison"]["candidate"]
    baseline = metrics["overall"][population][horizon][baseline_name]
    candidate = metrics["overall"][population][horizon][candidate_name]
    fold_metrics = metrics["byFold"][population][horizon]
    minimum_metrics = int(
        config["nonWaivableGates"][
            "minimumImprovedMetricCountOfFour"
        ]
    )
    improving_folds = sum(
        physics.improvement_count(
            values[candidate_name], values[baseline_name]
        )
        >= minimum_metrics
        for values in fold_metrics.values()
    )
    months = metrics["byGroup"][population][horizon]["month"]
    improving_months = sum(
        values[candidate_name]["brier"]
        < values[baseline_name]["brier"]
        for values in months.values()
    )
    categories = metrics["byGroup"][population][horizon]["etf_category"]
    improving_categories = [
        name
        for name, values in categories.items()
        if values[candidate_name]["brier"]
        < values[baseline_name]["brier"]
    ]
    bootstrap = physics.cluster_bootstrap_brier_delta(
        predictions,
        population,
        int(horizon),
        candidate_name,
        baseline_name,
        evaluation_config,
    )
    gates = {
        "minimumImprovedMetrics": physics.improvement_count(
            candidate, baseline
        )
        >= minimum_metrics,
        "majorityFolds": improving_folds / len(fold_metrics)
        > float(
            config["nonWaivableGates"][
                "minimumImprovingFoldFraction"
            ]
        ),
        "majorityMonths": improving_months / len(months)
        > float(
            config["nonWaivableGates"][
                "minimumImprovingMonthFraction"
            ]
        ),
        "multipleEtfCategories": len(improving_categories)
        >= int(
            config["nonWaivableGates"][
                "minimumImprovingEtfCategories"
            ]
        ),
        "clusterBootstrapBrier": bootstrap["passes"],
        "sameForecastPopulation": baseline["count"] == candidate["count"],
        "causalPastOnlyFeatures": True,
    }
    allowed = all(gates.values())
    paper_hash_after = sha256(PAPER_CONFIG)
    agent_hash_after = sha256(AGENT_SOURCE)
    result = {
        "schemaVersion": "singularity_phase1_6_koopman_exception_result_v1",
        "runId": output_dir.name,
        "generatedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "status": "diagnostic_only",
        "researchOnly": True,
        "userAuthorizedPaperException": True,
        "comparison": config["comparison"],
        "exemptionsApplied": {
            "coverage": True,
            "tenBar": True,
            "highRiskSamples": True,
        },
        "metrics": {
            "baseline": baseline,
            "candidate": candidate,
        },
        "diagnostic": {
            "improvedMetricCountOfFour": physics.improvement_count(
                candidate, baseline
            ),
            "improvingFolds": improving_folds,
            "totalFolds": len(fold_metrics),
            "improvingMonths": improving_months,
            "totalMonths": len(months),
            "improvingEtfCategories": improving_categories,
            "clusterBootstrap": bootstrap,
            "gates": gates,
        },
        "folds": fold_audits,
        "nonWaivableGatesPassed": allowed,
        "paperIntegrationAllowed": allowed,
        "paperConfigHashBefore": paper_hash_before,
        "paperConfigHashAfter": paper_hash_after,
        "agentSourceHashBefore": agent_hash_before,
        "agentSourceHashAfter": agent_hash_after,
        "paperConfigModified": paper_hash_before != paper_hash_after,
        "agentSourceModified": agent_hash_before != agent_hash_after,
        "conclusion": (
            "Koopman passes the user-authorized paper exception's remaining "
            "predictive gates; a separate risk-veto integration commit is "
            "permitted."
            if allowed
            else "Even after waiving coverage, 10-bar and high-risk sample "
            "gates, Koopman does not pass the non-waivable predictive gates "
            "and is not integrated into paper trading."
        ),
        "phase15Touched": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    phase2a.atomic_json(
        output_dir / "koopman_paper_exception_result.json", result
    )
    phase2a.atomic_text(
        output_dir / "koopman_paper_exception_report.md",
        render_report(result),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    run_id = args.run_id or (
        "koopman_exception_"
        + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output = resolve(config["output"]["root"]) / run_id
    result = run(config, output)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "non_waivable_gates_passed": result[
                    "nonWaivableGatesPassed"
                ],
                "paper_integration_allowed": result[
                    "paperIntegrationAllowed"
                ],
                "paper_config_modified": result["paperConfigModified"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
