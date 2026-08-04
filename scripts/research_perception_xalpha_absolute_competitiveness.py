"""Fail-closed zero-to-ten absolute-competitiveness audit for the joint factor book.

This is an offline research wrapper.  It keeps every one of the seven factors non-zero,
checks absolute ten-session outcomes (not merely relative IC), and emits no qualified names
unless historical, current-market, and point-in-time exposure-data gates all pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_cogalpha_etf as core  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_joint_factor_weights as joint  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/research/perception_xalpha_absolute_competitiveness_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_absolute_competitiveness_v1":
        raise ValueError("unexpected absolute-competitiveness schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("absolute-competitiveness audit must remain research-only")
    safety = config.get("safety", {})
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading mutation permissions must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    if hypothesis.get("sourceWeightsUseTrainOnly") is not True:
        raise ValueError("source weights must be train-only")
    if hypothesis.get("validationAndShadowMayNotTuneWeightsOrThresholds") is not True:
        raise ValueError("evaluation blocks cannot tune this audit")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical audit cannot promote")
    if hypothesis.get("profitGuaranteeClaimAllowed") is not False:
        raise ValueError("profit guarantees are forbidden")
    weights = config["weights"]
    lower, upper = float(weights["minimumEach"]), float(weights["maximumEach"])
    if not 0.0 < lower < upper < 1.0 or 7 * lower > 1.0 or 7 * upper < 1.0:
        raise ValueError("invalid seven-factor bounds")
    selection = config["selection"]
    if selection.get("allowZeroSelections") is not True or selection.get("neverFillQuota") is not True:
        raise ValueError("the absolute selector must be allowed to return zero")
    if selection.get("approvedOnlyWhenAllGatesPass") is not True:
        raise ValueError("all gates must be conjunctive")
    exposure = config["pointInTimeExposureData"]
    if exposure.get("requiredForAbsoluteQualification") is not True:
        raise ValueError("PIT exposure data must remain a qualification prerequisite")
    if exposure.get("missingPolicy") != "fail_closed_do_not_fake_with_current_classification_or_turnover":
        raise ValueError("missing PIT exposure data must fail closed")


def exposure_data_audit(config: dict[str, Any]) -> dict[str, Any]:
    exposure = config["pointInTimeExposureData"]
    paths: dict[str, str | None] = {
        "industry": exposure.get("industryClassificationPath"),
        "floatMarketCap": exposure.get("floatMarketCapPath"),
    }
    checks = {
        key: bool(value and (ROOT / value).exists()) for key, value in paths.items()
    }
    return {
        "paths": paths,
        "available": checks,
        "pass": all(checks.values()),
        "missingPolicy": exposure["missingPolicy"],
    }


def slot_metrics(
    composite: pd.DataFrame,
    target: pd.DataFrame,
    split: autonomous.Split,
    maximum_slots: int,
    hac_lag: int,
) -> dict[str, list[dict[str, Any]]]:
    rank = composite.rank(axis=1, ascending=False, method="first")
    output: dict[str, list[dict[str, Any]]] = {}
    for period, mask in {
        "train": split.train,
        "validation": split.validation,
        "shadow": split.shadow,
    }.items():
        dates = mask[mask].index
        rows: list[dict[str, Any]] = []
        for slot in range(1, maximum_slots + 1):
            returns = target.where(rank.eq(slot)).mean(axis=1).reindex(dates)
            rows.append({"slot": slot, **joint.series_metrics(returns, hac_lag)})
        output[period] = rows
    return output


def evidence_checks(
    metrics: dict[str, Any], slots: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["absoluteEvidenceGate"]
    periods: dict[str, Any] = {}
    for period in gate["periodsRequired"]:
        values = metrics["periods"][period]
        gross = values["top10GrossReturn"]
        net = values["top10NetReturn"]
        independent = values["independentTenDayEvents"]
        checks = {
            "grossMeanPositive": float(gross["mean"] or -99.0)
            > float(gate["minimumTop10GrossMeanExclusive"]),
            "netMeanPositive": float(net["mean"] or -99.0)
            > float(gate["minimumTop10NetMeanExclusive"]),
            "winRate": float(gross["positiveRate"] or 0.0)
            >= float(gate["minimumTop10WinRateInclusive"]),
            "independentEvents": int(independent["n"] or 0)
            >= int(gate["minimumIndependentEvents"]),
            "independentMean": float(independent["mean"] or -99.0)
            > float(gate["minimumIndependentEventMeanExclusive"]),
            "independentWinRate": float(independent["positiveRate"] or 0.0)
            >= float(gate["minimumIndependentEventWinRateInclusive"]),
        }
        if gate.get("requireEveryRankSlotPositiveMean"):
            checks["everySlotPositiveMean"] = all(
                float(row["mean"] or -99.0) > 0.0 for row in slots[period]
            )
        if gate.get("requireEveryRankSlotWinRateAboveChance"):
            checks["everySlotWinRateAboveChance"] = all(
                float(row["positiveRate"] or 0.0) > 0.5 for row in slots[period]
            )
        periods[period] = {"checks": checks, "pass": all(checks.values())}
    return {"periods": periods, "pass": all(row["pass"] for row in periods.values())}


def current_market_state(panel: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    eligible = panel["eligible"]
    returns = panel["returns"]
    close = panel["close"]
    market_daily = returns.where(eligible).median(axis=1)
    market_return_20 = float(market_daily.rolling(20, min_periods=10).sum().iloc[-1])
    moving_average = close.rolling(20, min_periods=10).mean()
    denominator = int(eligible.iloc[-1].sum())
    breadth = float(
        ((close.iloc[-1] > moving_average.iloc[-1]) & eligible.iloc[-1]).sum()
        / max(1, denominator)
    )
    gate = config["currentMarketGate"]
    checks = {
        "marketReturn20": market_return_20
        > float(gate["marketReturn20MinimumExclusive"]),
        "marketBreadth20": breadth >= float(gate["marketBreadth20MinimumInclusive"]),
    }
    return {
        "asOfDate": close.index[-1].date().isoformat(),
        "marketReturn20": round(market_return_20, 8),
        "marketBreadth20": round(breadth, 8),
        "checks": checks,
        "pass": all(checks.values()),
    }


def master_names(base: dict[str, Any]) -> dict[str, str]:
    path = ROOT / base["assetUniverse"]["masterPath"]
    names: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("securityId"):
            names[str(row["securityId"])] = str(row.get("name") or "")
    return names


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 绝对竞争力 0–10 只筛选",
        "",
        "> **research-only / shadow-only / not a trading signal.** 不满足全部绝对证据门时输出0只。",
        "",
        f"- run_id: `{result['runId']}`",
        f"- as_of: {result['asOfDate']}",
        f"- qualified_count: **{len(result['qualifiedResearchSelections'])}**",
        f"- all_gates_pass: `{result['allGatesPass']}`",
        f"- blocking_reasons: `{result['blockingReasons']}`",
        "",
        "## 七因子有界权重",
        "",
        "| 因子 | 权重 |",
        "|---|---:|",
    ]
    for name, value in result["boundedWeights"].items():
        lines.append(f"| {name} | {100 * value:.2f}% |")
    lines.extend(["", "## 研究候选（只有全部门通过才会进入合格名单）", "", "|排名|代码|名称|分数|", "|---:|---|---|---:|"])
    for row in result["candidateTop10"]:
        lines.append(f"|{row['rank']}|{row['securityId']}|{row['name']}|{100 * row['score']:.2f}|")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 候选排名不等于合格投资标的。",
            "- 任何阻断条件存在时，qualifiedResearchSelections 保持为空。",
            "- 不生成订单、不修改交易配置、不自动晋级。",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    joint_config_path = ROOT / config["baseJointConfig"]
    joint_config = load_json(joint_config_path)
    joint.validate_config(joint_config)
    source = load_json(ROOT / config["sourceJointResult"])
    if source.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("source joint result is not research-only")
    definitions = joint.factor_definitions(joint_config)
    names = [row["key"] for row in definitions]
    source_weights = np.array([source["recommendedWeights"][name] for name in names], dtype=float)
    bounded = joint.project_bounded_simplex(
        source_weights,
        float(config["weights"]["minimumEach"]),
        float(config["weights"]["maximumEach"]),
    )
    bounded_weights = {
        name: float(value) for name, value in zip(names, bounded, strict=True)
    }
    base = load_json(ROOT / joint_config["baseResearchConfig"])
    _perception_config, cog_config = perception.load_base_configs(base)
    print("loading PIT-adjusted panel ...", flush=True)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    target, _one_day = autonomous.target_frames(panel, cog_config)
    split = autonomous.make_split(panel["close"].index, cog_config)
    bins = int(joint_config["data"]["factorLiquidityNeutralisationBins"])
    frames: dict[str, pd.DataFrame] = {}
    available = panel["eligible"].copy()
    for index, definition in enumerate(definitions, start=1):
        print(f"factor {index}/7: {definition['key']}", flush=True)
        raw = core.evaluate_expression(definition["expression"], panel).replace(
            [np.inf, -np.inf], np.nan
        )
        neutral = autonomous.size_neutralise(
            raw * float(definition["direction"]), panel, bins
        )
        rank = neutral.rank(axis=1, pct=True).where(panel["eligible"]).astype(np.float32)
        frames[definition["key"]] = rank
        available &= rank.notna()
        del raw, neutral
    metrics, latest = joint.evaluate_scheme(
        "bounded_all_seven_absolute_candidate",
        frames,
        bounded_weights,
        available,
        target,
        split,
        int(config["data"]["maximumSelectionsPerDay"]),
        float(config["data"]["roundTripCost"]),
        9,
        int(config["data"]["holdingTradingDays"]),
    )
    composite = joint.weighted_composite(frames, bounded_weights, available)
    slots = slot_metrics(composite, target, split, 10, 9)
    evidence = evidence_checks(metrics, slots, config)
    market = current_market_state(panel, config)
    exposure = exposure_data_audit(config)
    all_gates = bool(evidence["pass"] and market["pass"] and exposure["pass"])
    blocking: list[str] = []
    if not evidence["pass"]:
        blocking.append("ABSOLUTE_VALIDATION_SHADOW_EVIDENCE_FAILED")
    if not market["pass"]:
        blocking.append("CURRENT_MARKET_GATE_CLOSED")
    if not exposure["pass"]:
        blocking.append("PIT_INDUSTRY_OR_FLOAT_MARKET_CAP_MISSING")
    name_map = master_names(base)
    candidate_rows: list[dict[str, Any]] = []
    for row in latest.sort_values("rank").to_dict(orient="records"):
        security_id = str(row["securityId"])
        candidate_rows.append(
            {
                "rank": int(row["rank"]),
                "securityId": security_id,
                "name": name_map.get(security_id, ""),
                "score": round(float(row["score"]), 8),
            }
        )
    qualified = candidate_rows if all_gates else []
    run_id = run_id or f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    result = {
        "schemaVersion": config["schemaVersion"],
        "status": "research_only_shadow_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "runId": run_id,
        "asOfDate": panel["close"].index[-1].date().isoformat(),
        "dataRange": [
            panel["close"].index[0].date().isoformat(),
            panel["close"].index[-1].date().isoformat(),
        ],
        "split": split.audit,
        "panelAudit": panel_audit,
        "sourceWeights": source["recommendedWeights"],
        "boundedWeights": bounded_weights,
        "weightRule": config["weights"],
        "absoluteEvidence": evidence,
        "periodMetrics": metrics["periods"],
        "slotMetrics": slots,
        "currentMarketGate": market,
        "pointInTimeExposureData": exposure,
        "allGatesPass": all_gates,
        "blockingReasons": blocking,
        "candidateTop10": candidate_rows,
        "qualifiedResearchSelections": qualified,
        "orders": [],
        "automaticTradingChanges": [],
        "promotionAllowed": False,
        "verdict": "zero_selections_fail_closed" if not qualified else "research_candidates_only_not_promotable",
    }
    output = ROOT / config["output"]["root"] / run_id
    atomic_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_text(output / "report.md", markdown(result))
    pd.DataFrame(candidate_rows).to_csv(
        output / "candidate_top10.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(qualified).to_csv(
        output / "qualified_selections.csv", index=False, encoding="utf-8-sig"
    )
    print(f"qualified={len(qualified)} blockers={blocking} saved={output}", flush=True)
    return result


def self_test() -> None:
    config = load_json(DEFAULT_CONFIG)
    validate_config(config)
    joint_config = load_json(ROOT / config["baseJointConfig"])
    definitions = joint.factor_definitions(joint_config)
    source = load_json(ROOT / config["sourceJointResult"])
    values = np.array(
        [source["recommendedWeights"][row["key"]] for row in definitions], dtype=float
    )
    projected = joint.project_bounded_simplex(values, 0.05, 0.20)
    assert math.isclose(float(projected.sum()), 1.0, abs_tol=1e-8)
    assert float(projected.min()) >= 0.05 - 1e-10
    assert float(projected.max()) <= 0.20 + 1e-10
    assert exposure_data_audit(config)["pass"] is False
    print("self_test=ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args.config.resolve(), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
