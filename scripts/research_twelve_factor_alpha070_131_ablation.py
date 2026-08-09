"""Fixed Top10 ablation for the existing 12 factors and GTJA Alpha131.

Alpha070 already belongs to the frozen twelve-factor baseline.  This study therefore
never duplicates it.  It evaluates the original weighted 12-factor score, a mechanical
Alpha131 insertion, and three preregistered symmetric Alpha070/Alpha131 weight levels.
Historical outcomes can reject but cannot choose a weight or connect to trading.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

import research_fundamental_top10_discrimination as top10  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_twelve_factor_utility_weights_v6 as v6  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402


SCHEMA_VERSION = "twelve_factor_alpha070_131_ablation_result_v3"
CODE_VERSION = "twelve_factor_alpha070_131_ablation_v3_20260809"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_alpha070_131_ablation_v3.json"
)
TOP10_CONFIG = (
    ROOT / "configs" / "research" / "fundamental_top10_discrimination_v2.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify(config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(config[path_key])
    actual = file_sha256(path)
    if actual != str(config[hash_key]).lower():
        raise ValueError(f"frozen dependency changed: {path_key}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_alpha070_131_ablation_v3":
        raise ValueError("unexpected Alpha070/131 ablation schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("ablation must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every mutation and trading permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("outputStatus must remain diagnostic_only")
    contract = config["factorContract"]
    if int(contract["baselineCount"]) != 12:
        raise ValueError("baseline must contain twelve factors")
    if not contract.get("alpha070AlreadyInBaseline"):
        raise ValueError("Alpha070 must be recognised as an existing factor")
    if not contract.get("duplicateAlpha070Forbidden"):
        raise ValueError("Alpha070 duplication must fail closed")
    if contract.get("alpha070Key") != "gtja191/alpha_070":
        raise ValueError("unexpected Alpha070 definition")
    if contract.get("alpha131Key") != "gtja191/alpha_131":
        raise ValueError("unexpected Alpha131 definition")
    expected_policies = {
        "frozen_weighted_12_baseline",
        "weighted_12_plus_alpha131_equal_insertion",
        "alpha070_alpha131_each_05pct",
        "alpha070_alpha131_each_10pct",
        "alpha070_alpha131_each_15pct",
    }
    if set(config["policies"]) != expected_policies:
        raise ValueError("the five preregistered policies cannot change")
    if config["evaluation"].get("weightLadderWasFixedBeforeOutcomes") != [0.05, 0.10, 0.15]:
        raise ValueError("weight ladder cannot change")
    if int(config["evaluation"]["topCount"]) != 10:
        raise ValueError("evaluation must remain exactly Top10")
    if int(config["evaluation"]["holdingTradingDays"]) != 1:
        raise ValueError("outcome must remain next-session executable")
    if not math.isclose(float(config["evaluation"]["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("round-trip cost must remain 30 bps")
    if float(config["incrementalAcceptance"]["minimumPairedHacTForMeanNetImprovement"]) < 2.0:
        raise ValueError("paired improvement significance gate cannot be weakened")
    if config["incrementalAcceptance"].get("historicalRunCanPromote"):
        raise ValueError("historical results cannot promote")
    if config["forward"].get("historicalWinnerMayBeForwardSelected"):
        raise ValueError("historical winner may not select a forward policy")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must always be empty")


def frozen_factors_and_weights(
    config: dict[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    v6_path = verify(
        config, "frozenTwelveFactorConfig", "frozenTwelveFactorConfigFileSha256"
    )
    verify(config, "frozenTwelveFactorSource", "frozenTwelveFactorSourceFileSha256")
    weight_path = verify(
        config, "frozenTwelveFactorWeights", "frozenTwelveFactorWeightsFileSha256"
    )
    verify(config, "alpha070Module", "alpha070ModuleFileSha256")
    verify(config, "alpha131Module", "alpha131ModuleFileSha256")
    verify(config, "top10EvaluationCode", "top10EvaluationCodeFileSha256")
    v6_config = load_json(v6_path)
    v6.validate_config(v6_config)
    factors, _source, _sha = v6.source_factors(v6_config)
    keys = [str(item["factorKey"]) for item in factors]
    alpha070 = str(config["factorContract"]["alpha070Key"])
    alpha131 = str(config["factorContract"]["alpha131Key"])
    if keys.count(alpha070) != 1 or alpha131 in keys:
        raise ValueError("baseline must contain Alpha070 once and exclude Alpha131")
    weight_result = load_json(weight_path)
    lookup = {
        str(row["factorKey"]): float(row["finalWeight"])
        for row in weight_result["weights"]
    }
    if set(lookup) != set(keys):
        raise ValueError("frozen weight result does not match the twelve factors")
    weights = np.asarray([lookup[key] for key in keys], dtype=float)
    if abs(float(weights.sum()) - 1.0) > 1e-8:
        raise ValueError("frozen twelve-factor weights do not sum to one")
    return factors, weights, v6_config


def policy_weights(
    factor_keys: list[str], baseline: np.ndarray, alpha070: str, alpha131: str
) -> dict[str, dict[str, float]]:
    base = {key: float(weight) for key, weight in zip(factor_keys, baseline, strict=True)}
    policies: dict[str, dict[str, float]] = {
        "frozen_weighted_12_baseline": dict(base),
        "weighted_12_plus_alpha131_equal_insertion": {
            **{key: value * 12.0 / 13.0 for key, value in base.items()},
            alpha131: 1.0 / 13.0,
        },
    }
    other_keys = [key for key in factor_keys if key != alpha070]
    other_sum = sum(base[key] for key in other_keys)
    for level, name in (
        (0.05, "alpha070_alpha131_each_05pct"),
        (0.10, "alpha070_alpha131_each_10pct"),
        (0.15, "alpha070_alpha131_each_15pct"),
    ):
        remaining = 1.0 - 2.0 * level
        policy = {key: base[key] / other_sum * remaining for key in other_keys}
        policy[alpha070] = level
        policy[alpha131] = level
        policies[name] = policy
    for name, weights in policies.items():
        if abs(sum(weights.values()) - 1.0) > 1e-8:
            raise RuntimeError(f"policy weights do not sum to one: {name}")
    return policies


def weighted_score(
    ranks: dict[str, pd.DataFrame], weights: dict[str, float], eligible: pd.DataFrame
) -> pd.DataFrame:
    common = eligible.copy().fillna(False)
    for key in weights:
        common &= ranks[key].notna()
    score = sum(ranks[key].where(common) * weight for key, weight in weights.items())
    return score.where(common).rank(axis=1, pct=True, method="average").astype("float32")


def paired_increment(
    candidate: pd.DataFrame, baseline: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    columns = [
        "topNetMean",
        "topGrossUpProbability",
        "topNetPositiveProbability",
        "topNetLossProbability",
        "returnLiftVsMatched",
    ]
    joined = candidate[columns].join(
        baseline[columns], how="inner", lsuffix="_candidate", rsuffix="_baseline"
    ).dropna()
    deltas = {
        "meanNetReturn": joined["topNetMean_candidate"] - joined["topNetMean_baseline"],
        "grossUpProbability": joined["topGrossUpProbability_candidate"]
        - joined["topGrossUpProbability_baseline"],
        "netPositiveProbability": joined["topNetPositiveProbability_candidate"]
        - joined["topNetPositiveProbability_baseline"],
        "netLossProbability": joined["topNetLossProbability_candidate"]
        - joined["topNetLossProbability_baseline"],
        "matchedReturnLift": joined["returnLiftVsMatched_candidate"]
        - joined["returnLiftVsMatched_baseline"],
    }
    lag = 5
    mean_delta = deltas["meanNetReturn"].to_numpy(float)
    hac = autonomous.newey_west_t(mean_delta, lag) if len(mean_delta) >= 20 else None
    output = {
        "pairedDays": len(joined),
        "meanNetReturnDelta": round(float(deltas["meanNetReturn"].mean()), 8),
        "grossUpProbabilityDelta": round(float(deltas["grossUpProbability"].mean()), 8),
        "netPositiveProbabilityDelta": round(float(deltas["netPositiveProbability"].mean()), 8),
        "netLossProbabilityDelta": round(float(deltas["netLossProbability"].mean()), 8),
        "matchedReturnLiftDelta": round(float(deltas["matchedReturnLift"].mean()), 8),
        "meanNetReturnDeltaHacT": round(float(hac), 4) if hac is not None else None,
        "inferenceUnit": "trading_day",
    }
    threshold = float(
        config["incrementalAcceptance"]["minimumPairedHacTForMeanNetImprovement"]
    )
    checks = {
        "grossUpProbabilityImproved": output["grossUpProbabilityDelta"] > 0.0,
        "netPositiveProbabilityImproved": output["netPositiveProbabilityDelta"] > 0.0,
        "netLossProbabilityReduced": output["netLossProbabilityDelta"] < 0.0,
        "meanNetReturnImproved": output["meanNetReturnDelta"] > 0.0,
        "matchedReturnLiftImproved": output["matchedReturnLiftDelta"] > 0.0,
        "meanNetReturnImprovementSignificant": (
            output["meanNetReturnDeltaHacT"] is not None
            and output["meanNetReturnDeltaHacT"] >= threshold
        ),
    }
    output["checks"] = checks
    output["allIncrementalChecksPassed"] = all(checks.values())
    return output


def render_report(result: dict[str, Any]) -> str:
    pct = lambda value: "n/a" if value is None else f"{100 * float(value):.3f}%"
    lines = [
        "# Twelve-factor Alpha070/Alpha131 Top10 ablation V3",
        "",
        "> **Research-only. Historical outcomes cannot choose a weight or connect to trading.**",
        "",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- symbols: `{result['dataAudit']['symbols']}`",
        "- Alpha070: already present once in the twelve-factor baseline",
        "- Alpha131: new positive-orientation candidate",
        "- execution: close t -> buyable open t+1 -> sellable open t+2 -> 30 bps",
        "",
        "## Exact Top10 results",
        "",
        "| policy | 070 weight | 131 weight | gross up | net positive | net loss | mean net | median net | lift vs matched |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["policies"].items():
        metrics = row["metrics"]
        weights = row["weights"]
        lines.append(
            f"| {name} | {pct(weights.get('gtja191/alpha_070'))} | "
            f"{pct(weights.get('gtja191/alpha_131', 0.0))} | "
            f"{pct(metrics.get('top10GrossUpProbability'))} | "
            f"{pct(metrics.get('top10NetPositiveProbability'))} | "
            f"{pct(metrics.get('top10NetLossProbability'))} | "
            f"{pct(metrics.get('top10MeanNetReturn'))} | "
            f"{pct(metrics.get('top10MedianNetReturn'))} | "
            f"{pct(metrics.get('meanNetReturnLiftVsMatchedControl'))} |"
        )
    lines.extend(
        [
            "",
            "## Increment versus the same twelve-factor baseline",
            "",
            "| policy | up delta | net-win delta | loss delta | mean-net delta | matched-lift delta | HAC t | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, row in result["policies"].items():
        if name == "frozen_weighted_12_baseline":
            continue
        delta = row["incrementVsBaseline"]
        lines.append(
            f"| {name} | {pct(delta['grossUpProbabilityDelta'])} | "
            f"{pct(delta['netPositiveProbabilityDelta'])} | "
            f"{pct(delta['netLossProbabilityDelta'])} | "
            f"{pct(delta['meanNetReturnDelta'])} | "
            f"{pct(delta['matchedReturnLiftDelta'])} | "
            f"{delta['meanNetReturnDeltaHacT']} | {delta['allIncrementalChecksPassed']} |"
        )
    lines.extend(
        [
            "",
            "The historically best weight is not selected for forward use. A useful result "
            "must improve every preregistered probability/return criterion and reach paired "
            "day-clustered HAC t >= 2.0. Validated factors remain `0`; orders remain `[]`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    factors, frozen_weights, v6_config = frozen_factors_and_weights(config)
    base = load_json(ROOT / str(v6_config["baseResearchConfig"]))
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    ranks, factor_audit = v6.compute_factor_ranks(panel, factors)
    alpha131_key = str(config["factorContract"]["alpha131Key"])
    alpha131_zoo, alpha131_name = alpha131_key.split("/", 1)
    alpha131_module = importlib.import_module(
        f"src.factors.zoo.{alpha131_zoo}.{alpha131_name}"
    )
    inputs = v6.build_factor_inputs(panel)
    alpha131_raw = alpha131_module.compute(inputs).reindex_like(panel["close"])
    ranks[alpha131_key] = (
        alpha131_raw
        * float(config["factorContract"]["alpha131Direction"])
    ).where(panel["eligible"]).rank(axis=1, pct=True).astype("float32")
    factor_audit.append(
        {
            "factorKey": alpha131_key,
            "direction": float(config["factorContract"]["alpha131Direction"]),
            "finiteValues": int(np.isfinite(ranks[alpha131_key].to_numpy()).sum()),
            "pastOnlySourceAudit": "static_formula_and_prefix_causality_required",
        }
    )
    del alpha131_raw, inputs
    keys = [str(item["factorKey"]) for item in factors]
    policies = policy_weights(
        keys,
        frozen_weights,
        str(config["factorContract"]["alpha070Key"]),
        alpha131_key,
    )
    common = panel["eligible"].copy().fillna(False)
    for rank in ranks.values():
        common &= rank.notna()
    evaluation_panel = dict(panel)
    evaluation_panel["eligible"] = common
    top10_config = load_json(TOP10_CONFIG)
    top10.validate_config(top10_config)
    outputs: dict[str, Any] = {}
    daily_by_policy: dict[str, pd.DataFrame] = {}
    for name, weights in policies.items():
        print(f"policy {name}", flush=True)
        score = weighted_score(ranks, weights, common)
        daily, metrics = top10.evaluate_daily_top10(
            evaluation_panel, score, top10_config
        )
        outputs[name] = {"weights": weights, "metrics": metrics}
        daily_by_policy[name] = daily
    baseline_name = "frozen_weighted_12_baseline"
    for name in outputs:
        if name == baseline_name:
            outputs[name]["incrementVsBaseline"] = None
            continue
        outputs[name]["incrementVsBaseline"] = paired_increment(
            daily_by_policy[name], daily_by_policy[baseline_name], config
        )
    passed = [
        name
        for name, row in outputs.items()
        if row.get("incrementVsBaseline", {}).get("allIncrementalChecksPassed")
    ]
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "dataAudit": {
            "start": str(panel["close"].index.min().date()),
            "end": str(panel["close"].index.max().date()),
            "sessions": len(panel["close"].index),
            "symbols": len(panel["close"].columns),
            "commonSupportMeanStocks": round(float(common.sum(axis=1).mean()), 2),
            "panel": panel_audit,
        },
        "factorAudit": factor_audit,
        "alpha070AlreadyPresentExactlyOnce": True,
        "policies": outputs,
        "incrementallyPassedPolicies": passed,
        "historicalWinnerMayBeSelected": False,
        "validatedFactorCount": 0,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
        "knownLimitations": config["knownLimitations"],
    }
    identifier = run_id or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z")
    result["runId"] = identifier
    root = ROOT / str(config["output"]["root"]) / identifier
    atomic_write(
        root / "result.json",
        json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(root / "report.md", render_report(result))
    return {
        "runId": identifier,
        "report": str(root / "report.md"),
        "incrementallyPassedPolicies": passed,
        "eligibleForTrading": False,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    response = run(args.config.resolve(), args.run_id)
    print(json.dumps(json_safe(response), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
