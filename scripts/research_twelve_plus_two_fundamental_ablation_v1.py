#!/usr/bin/env python3
"""Frozen twelve-factor plus two PIT fundamental-factor ablation.

All policies rank the same complete-support price Top100 and select the same Top10 count.
The historical windows have already been observed, so this study can reject but cannot
authorize a model or trading change.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_ashare_universe as ashare  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_financial_statement_factor_discovery_v1 as discovery  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_pit_fundamental_catalyst_v5 as catalyst  # noqa: E402


SCHEMA_VERSION = "twelve_plus_two_fundamental_ablation_result_v1"
CODE_VERSION = "twelve_plus_two_fundamental_ablation_v1_20260812"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_plus_two_fundamental_ablation_v1.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_plus_two_fundamental_ablation_v1":
        raise ValueError("unexpected twelve-plus-two schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("ablation must remain research-only")
    if config["fundamentalFactors"] != [
        "inventory_days__level",
        "cash_to_revenue__level",
    ]:
        raise ValueError("the two post-discovery factors must remain frozen")
    if config["data"].get("availabilityRule") != (
        "first_market_date_strictly_after_max_notice_update"
    ):
        raise ValueError("fundamental availability must remain PIT-safe")
    if int(config["data"]["holdingTradingDays"]) != 1:
        raise ValueError("the ablation is frozen to next-session outcomes")
    if not math.isclose(float(config["data"]["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("A-share cost stress must remain 30 bps")
    if config["selection"].get("requireBothFundamentalsForEveryArm") is not True:
        raise ValueError("all arms must use identical complete support")
    if int(config["selection"]["firstStageCandidateCount"]) != 100:
        raise ValueError("first stage must remain Top100")
    if int(config["selection"]["topCount"]) != 10:
        raise ValueError("final selection must remain Top10")
    required_policies = {
        "price_12_baseline",
        "price_12_plus_inventory",
        "price_12_plus_cash_to_revenue",
        "price_12_plus_both",
    }
    if set(config["policies"]) != required_policies:
        raise ValueError("the four preregistered policies changed")
    for name, policy in config["policies"].items():
        weights = [
            float(policy["priceWeight"]),
            float(policy["inventoryDaysWeight"]),
            float(policy["cashToRevenueWeight"]),
        ]
        if any(value < 0.0 for value in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=1e-12
        ):
            raise ValueError(f"invalid fixed policy weights: {name}")
    evaluation = config["evaluation"]
    if int(evaluation["priorDiscoveryTrials"]) != 64:
        raise ValueError("the inspected source-discovery trials must be charged")
    if int(evaluation["newPolicyTrials"]) != 3:
        raise ValueError("the three extensions must be charged")
    if evaluation.get("historicalWindowsAlreadyViewed") is not True:
        raise ValueError("historical reuse must be explicit")
    if evaluation.get("historicalRunCanPromote") is not False:
        raise ValueError("historical ablation cannot promote")
    safety = config["safety"]
    if any(bool(value) for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all mutation and trading permissions must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output must remain diagnostic_only")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def verify_frozen(config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(config[path_key])
    if discovery.file_sha256(path) != str(config[hash_key]).lower():
        raise ValueError(f"frozen input changed: {path_key}")
    return path


def _normal_two_sided_p(t_value: float | None) -> float | None:
    if t_value is None or not math.isfinite(float(t_value)):
        return None
    return math.erfc(abs(float(t_value)) / math.sqrt(2.0))


def policy_score(
    policy: dict[str, Any],
    price_rank: pd.DataFrame,
    inventory_rank: pd.DataFrame,
    cash_rank: pd.DataFrame,
) -> pd.DataFrame:
    return (
        price_rank * float(policy["priceWeight"])
        + inventory_rank * float(policy["inventoryDaysWeight"])
        + cash_rank * float(policy["cashToRevenueWeight"])
    )


def evaluate_period(
    selections: dict[str, pd.DataFrame],
    scores: dict[str, pd.DataFrame],
    common: pd.DataFrame,
    returns: pd.DataFrame,
    exit_delay: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    holding = int(config["data"]["holdingTradingDays"])
    cost = float(config["data"]["roundTripCost"])
    baseline_name = str(config["evaluation"]["baselinePolicy"])
    baseline_daily = catalyst.daily_net_series(
        selections[baseline_name], returns, dates, cost
    )
    rows: dict[str, Any] = {}
    for name, selected in selections.items():
        performance = precision.summarize_selection(
            selected, returns, exit_delay, dates, config, holding
        )
        daily = catalyst.daily_net_series(selected, returns, dates, cost)
        paired = catalyst.paired_policy_delta(daily, baseline_daily, 0)
        rows[name] = {
            "performance": performance,
            "scoreIc": catalyst.daily_ic_stats(
                scores[name], returns, common, dates, 0
            ),
            "pairedDeltaVsBaseline": paired,
            "pairedNormalApproxP": _normal_two_sided_p(paired.get("hacT")),
        }
    return rows


def period_checks(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, bool]:
    c = candidate["performance"]
    b = baseline["performance"]
    return {
        "grossUpRateImproved": float(c.get("stockGrossWinRate") or 0.0)
        > float(b.get("stockGrossWinRate") or 0.0),
        "stockMeanGrossReturnImproved": float(c.get("stockMeanGrossReturn") or -math.inf)
        > float(b.get("stockMeanGrossReturn") or -math.inf),
        "dailyMeanGrossReturnImproved": float(c.get("dailyMeanGrossReturn") or -math.inf)
        > float(b.get("dailyMeanGrossReturn") or -math.inf),
        "severeLossNotIncreased": float(c.get("stockSevereLossRate") or math.inf)
        <= float(b.get("stockSevereLossRate") or math.inf),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Frozen 12 + two fundamental factors ablation V1",
        "",
        "> **Research-only / shadow-only / not a trading signal.**",
        "",
        f"- run_id: `{result['runId']}`",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- universe: `{result['dataAudit']['symbols']}` PIT SH/SZ stocks",
        f"- statement events: `{result['eventAudit']['statementEvents']}`",
        f"- verdict: `{result['verdict']['decision']}`",
        "- all arms: same complete-support price Top100, same daily Top10 count",
        "- orders: `[]`",
        "",
        "## Policy comparison",
        "",
        "| period | policy | stock up | mean gross | daily gross | severe loss | IC | paired delta | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ("train", "validation", "shadow"):
        for policy, row in result["periodMetrics"][period].items():
            perf = row["performance"]
            lines.append(
                f"| {period} | {policy} | "
                f"{100 * float(perf.get('stockGrossWinRate') or 0):.3f}% | "
                f"{100 * float(perf.get('stockMeanGrossReturn') or 0):.4f}% | "
                f"{100 * float(perf.get('dailyMeanGrossReturn') or 0):.4f}% | "
                f"{100 * float(perf.get('stockSevereLossRate') or 0):.3f}% | "
                f"{row['scoreIc'].get('meanSpearmanIc')} | "
                f"{100 * float(row['pairedDeltaVsBaseline'].get('meanDailyNetDelta') or 0):.4f}% | "
                f"{row.get('pairedNormalApproxP')} |"
            )
    lines.extend([
        "",
        "## Audit boundary",
        "",
        "- Original twelve-factor weights are unchanged internally and scaled as one frozen price meta-score.",
        "- Adding one factor gives it one of thirteen slots; adding both gives each one of fourteen slots.",
        "- Both fundamental values are required for every arm, including baseline, so missingness cannot manufacture improvement.",
        "- The two factors were chosen after inspecting a 64-candidate discovery run. All 64 prior trials plus three new policies are charged.",
        "- Validation and shadow windows were previously viewed. This run can reject but cannot confirm or promote.",
        "- No trading configuration, Top10 artifact, order, position, overlay, risk gate or execution lock was changed.",
        "",
    ])
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    base_path = verify_frozen(
        config, "baseResearchConfig", "baseResearchConfigFileSha256"
    )
    timing_path = verify_frozen(
        config, "frozenTimingConfig", "frozenTimingConfigFileSha256"
    )
    split_path = verify_frozen(
        config, "frozenSplitSummary", "frozenSplitSummaryFileSha256"
    )
    source_path = verify_frozen(
        config, "sourceDiscoveryConfig", "sourceDiscoveryConfigFileSha256"
    )
    source_config = load_json(source_path)
    discovery.validate_config(source_config)
    all_definitions = discovery.generate_candidates(source_config)
    by_name = {item["name"]: item for item in all_definitions}
    definitions = [by_name[name] for name in config["fundamentalFactors"]]

    base_config = load_json(base_path)
    perception.validate_config(base_config)
    _, cog_config = perception.load_base_configs(base_config)
    panel, panel_audit = ashare.build_panel(
        base_config["assetUniverse"], cog_config["data"]
    )
    if not panel_audit.get("unbiasedHistoricalValidationEligible", False):
        raise RuntimeError("clean PIT adjusted panel failed its unbiased-data gate")
    close = panel["close"]
    timing_config = load_json(timing_path)
    precision.validate_config(timing_config)
    timing_score, _, timing_audit = precision.compute_frozen_scores(panel, timing_config)
    if int(timing_audit["factorCount"]) != 12:
        raise RuntimeError("frozen price model is not the expected twelve-factor model")
    returns, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        holding_days=int(config["data"]["holdingTradingDays"]),
        maximum_exit_delay=int(config["data"]["maximumExitDelayTradingDays"]),
    )
    price_candidate = panel["eligible"] & execution_eligible & timing_score.notna()
    price_top100 = catalyst.select_top(
        timing_score,
        price_candidate,
        int(config["selection"]["firstStageCandidateCount"]),
    )
    event_table, event_audit = discovery.build_event_table(
        close.index, close.columns, source_config, definitions
    )
    frames: dict[str, pd.DataFrame] = {}
    for definition in definitions:
        raw = discovery.mechanism.factor_frame(
            event_table,
            definition["factorId"],
            close.index,
            close.columns,
            int(definition["maximumAgeTradingDays"]),
        )
        frames[definition["name"]] = autonomous.size_neutralise(
            raw,
            panel,
            int(config["selection"]["liquidityNeutraliseBins"]),
        )
    inventory = frames["inventory_days__level"]
    cash = frames["cash_to_revenue__level"]
    common = (
        price_top100
        & execution_eligible
        & timing_score.notna()
        & inventory.notna()
        & cash.notna()
    )
    price_rank = catalyst.daily_rank(timing_score, common)
    inventory_rank = catalyst.daily_rank(inventory, common)
    cash_rank = catalyst.daily_rank(cash, common)
    scores = {
        name: policy_score(policy, price_rank, inventory_rank, cash_rank)
        for name, policy in config["policies"].items()
    }
    selections = {
        name: catalyst.select_top(
            score, common, int(config["selection"]["topCount"])
        )
        for name, score in scores.items()
    }
    reference = selections["price_12_baseline"].sum(axis=1)
    if any(not selected.sum(axis=1).equals(reference) for selected in selections.values()):
        raise RuntimeError("policy daily selection counts differ")

    split_source = load_json(split_path)
    raw_splits = catalyst.split_dates(close.index, split_source)
    splits = {
        name: precision.contained_signal_dates(
            dates,
            int(config["data"]["holdingTradingDays"]),
            int(config["data"]["maximumExitDelayTradingDays"]),
        )
        for name, dates in raw_splits.items()
    }
    period_metrics = {
        period: evaluate_period(
            selections, scores, common, returns, exit_delay, dates, config
        )
        for period, dates in splits.items()
    }
    primary = str(config["evaluation"]["primaryPolicy"])
    baseline = str(config["evaluation"]["baselinePolicy"])
    checks = {
        period: period_checks(
            period_metrics[period][primary], period_metrics[period][baseline]
        )
        for period in config["evaluation"]["externalPeriods"]
    }
    total_trials = int(config["evaluation"]["priorDiscoveryTrials"]) + int(
        config["evaluation"]["newPolicyTrials"]
    )
    corrected_threshold = float(config["evaluation"]["familyWiseAlpha"]) / total_trials
    validation_p = period_metrics["validation"][primary]["pairedNormalApproxP"]
    multiple_testing_pass = (
        validation_p is not None and float(validation_p) <= corrected_threshold
    )
    historical_pass = (
        all(all(values.values()) for values in checks.values())
        and multiple_testing_pass
    )
    generated = datetime.now(timezone.utc)
    resolved_run_id = run_id or "run_" + generated.strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": resolved_run_id,
        "generatedAt": generated.isoformat(),
        "codeVersion": CODE_VERSION,
        "configSha256": discovery.file_sha256(config_path),
        "dataAudit": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "days": len(close.index),
            "symbols": len(close.columns),
            "pointInTimeMembership": panel_audit.get("pointInTimeMembership"),
            "adjustedPrices": not bool(panel_audit.get("rawPricesUnadjusted", True)),
            "unbiasedHistoricalValidationEligible": panel_audit.get(
                "unbiasedHistoricalValidationEligible"
            ),
        },
        "eventAudit": event_audit,
        "splitAudit": {
            name: [str(dates.min().date()), str(dates.max().date()), len(dates)]
            if len(dates)
            else [None, None, 0]
            for name, dates in splits.items()
        },
        "factorDefinitions": definitions,
        "policyDefinitions": config["policies"],
        "supportAudit": {
            "commonCandidateDays": int(common.any(axis=1).sum()),
            "commonCandidateObservations": int(common.sum().sum()),
            "sameCandidatePoolEveryPolicy": True,
            "sameDailyTop10CountEveryPolicy": True,
            "reducedTradingMechanicalImprovement": False,
        },
        "periodMetrics": period_metrics,
        "verdict": {
            "checks": checks,
            "priorDiscoveryTrials": int(config["evaluation"]["priorDiscoveryTrials"]),
            "newPolicyTrials": int(config["evaluation"]["newPolicyTrials"]),
            "totalTrialsCharged": total_trials,
            "correctedPThreshold": corrected_threshold,
            "validationPairedP": validation_p,
            "multipleTestingPassed": multiple_testing_pass,
            "historicalPass": historical_pass,
            "eligibleForTrading": False,
            "historicalRunCanPromote": False,
            "decision": (
                "fresh_forward_preregistration_only"
                if historical_pass
                else "reject_twelve_plus_two_increment"
            ),
        },
        "orders": [],
        "automaticTradingChanges": [],
    }
    output_dir = ROOT / str(config["output"]["root"]) / resolved_run_id
    safe = catalyst.json_safe(result)
    discovery.atomic_write(
        output_dir / "result.json",
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n",
    )
    discovery.atomic_write(output_dir / "report.md", render_report(safe) + "\n")
    manifest = {
        "schemaVersion": "twelve_plus_two_fundamental_ablation_manifest_v1",
        "status": safe["status"],
        "runId": resolved_run_id,
        "codeVersion": CODE_VERSION,
        "configSha256": discovery.file_sha256(config_path),
        "sourceDiscoveryConfigSha256": discovery.file_sha256(source_path),
        "orders": [],
    }
    discovery.atomic_write(
        output_dir / "run_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    result = run(args.config, args.run_id)
    print(
        json.dumps(
            {
                "runId": result["runId"],
                "decision": result["verdict"]["decision"],
                "historicalPass": result["verdict"]["historicalPass"],
                "eligibleForTrading": False,
                "orders": [],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
