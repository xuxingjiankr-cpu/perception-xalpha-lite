"""Robust train-only weighting of the union of the frozen V2 and V10 factors.

The script deliberately does not search validation or shadow outcomes.  It evaluates the
seven unique expressions on the same PIT-adjusted A-share panel, liquidity-neutralises each
factor, estimates non-negative maximum-IC-information-ratio weights inside the training
period, and reports validation/shadow diagnostics without permitting promotion.
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
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_cogalpha_etf as core  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/research/perception_xalpha_joint_factor_weights_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_joint_factor_weights_v1":
        raise ValueError("unexpected joint-factor-weight schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("joint weights must remain research-only")
    safety = config.get("safety", {})
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all research mutation permissions must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    data = config["data"]
    optimization = config["optimization"]
    if int(data["holdingTradingDays"]) != 10:
        raise ValueError("the frozen factor target must remain ten trading days")
    if int(optimization["purgeTradingDays"]) < int(data["holdingTradingDays"]):
        raise ValueError("purge must cover the label horizon")
    lower = float(optimization["minimumWeight"])
    upper = float(optimization["maximumWeight"])
    if lower < 0.0 or not lower < upper < 1.0 or 7 * upper < 1.0:
        raise ValueError("invalid bounded-simplex limits")
    if optimization.get("validationOrShadowMayFitWeights") is not False:
        raise ValueError("validation/shadow fitting must remain disabled")
    nested = config["nestedGroupOptimization"]
    if nested.get("selectionData") != "train_only":
        raise ValueError("nested group weights must use train only")
    if nested.get("validationOrShadowMayFitWeights") is not False:
        raise ValueError("nested group validation/shadow fitting must remain disabled")
    nested_lower = float(nested["minimumIncludedWeight"])
    nested_upper = float(nested["maximumIncludedWeight"])
    if nested_lower <= 0.0 or not nested_lower < nested_upper < 1.0:
        raise ValueError("invalid nested group weight limits")
    groups = list(nested["groups"])
    if [len(group["factors"]) for group in groups] != [4, 5, 6, 7]:
        raise ValueError("nested groups must contain four through seven factors")
    previous: set[str] = set()
    for group in groups:
        factors = list(group["factors"])
        current = set(factors)
        if len(current) != len(factors) or not previous.issubset(current):
            raise ValueError("nested groups must add unique factors monotonically")
        if len(current) * nested_lower > 1.0 or len(current) * nested_upper < 1.0:
            raise ValueError("nested group bounded simplex is infeasible")
        previous = current


def canonical_expression(expression: dict[str, Any]) -> str:
    return json.dumps(expression, sort_keys=True, separators=(",", ":"))


def factor_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    old = load_json(ROOT / config["oldFactorConfig"])["frozenFactorDefinitions"]
    new = load_json(ROOT / config["newFactorArtifact"])["selectedFactors"]
    unique: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in old:
        fingerprint = canonical_expression(row["expression"])
        unique[fingerprint] = {
            "key": str(row["name"]),
            "factorId": str(row["factorId"]),
            "expression": row["expression"],
            "direction": float(row["direction"]),
            "sources": ["old"],
            "aliases": [str(row["name"])],
        }
        order.append(fingerprint)
    for row in new:
        fingerprint = canonical_expression(row["expression"])
        alias = str(row["archetypeId"])
        if fingerprint in unique:
            existing = unique[fingerprint]
            if not math.isclose(
                float(existing["direction"]),
                float(row["directionFromTrain"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("shared expression has conflicting frozen directions")
            existing["sources"].append("new")
            existing["aliases"].append(alias)
            existing["key"] = "growth_quality_momentum_shared"
            continue
        unique[fingerprint] = {
            "key": alias,
            "factorId": str(row["factorId"]),
            "expression": row["expression"],
            "direction": float(row["directionFromTrain"]),
            "sources": ["new"],
            "aliases": [alias],
        }
        order.append(fingerprint)
    output = [unique[item] for item in order]
    if len(output) != 7:
        raise ValueError(f"expected seven unique factors after de-duplication, found {len(output)}")
    if sum("old" in row["sources"] and "new" in row["sources"] for row in output) != 1:
        raise ValueError("expected exactly one shared expression")
    return output


def project_bounded_simplex(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    start = np.full(len(values), 1.0 / len(values), dtype=float)
    result = minimize(
        lambda weight: float(np.square(weight - values).sum()),
        start,
        method="SLSQP",
        bounds=[(lower, upper)] * len(values),
        constraints=[{"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)}],
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"bounded-simplex projection failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def maximum_ic_ir_weights(
    daily_ic: pd.DataFrame, lower: float, upper: float
) -> np.ndarray:
    clean = daily_ic.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(60, 5 * clean.shape[1]):
        raise RuntimeError("insufficient complete IC days for weight fitting")
    values = clean.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    covariance = LedoitWolf().fit(values).covariance_
    covariance = covariance + np.eye(len(mean)) * 1e-12
    start = np.full(len(mean), 1.0 / len(mean), dtype=float)

    def objective(weight: np.ndarray) -> float:
        variance = float(weight @ covariance @ weight)
        return -float(weight @ mean) / math.sqrt(max(variance, 1e-18))

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(lower, upper)] * len(mean),
        constraints=[{"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)}],
        options={"maxiter": 2_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"maximum-IC-IR fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def expanding_weight_fits(
    train_ic: pd.DataFrame,
    minimum_fit: int,
    folds: int,
    purge: int,
    lower: float,
    upper: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    clean = train_ic.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < minimum_fit + folds * 20 + purge:
        raise RuntimeError("training history is too short for the frozen inner folds")
    boundaries = np.linspace(minimum_fit + purge, len(clean), folds + 1, dtype=int)
    records: list[dict[str, Any]] = []
    fitted: list[np.ndarray] = []
    oof_values: list[float] = []
    for fold in range(folds):
        evaluation_start = int(boundaries[fold])
        evaluation_end = int(boundaries[fold + 1])
        fit_end = evaluation_start - purge
        weight = maximum_ic_ir_weights(clean.iloc[:fit_end], lower, upper)
        proxy = clean.iloc[evaluation_start:evaluation_end].to_numpy(dtype=float) @ weight
        fitted.append(weight)
        oof_values.extend(proxy.tolist())
        records.append(
            {
                "fold": fold + 1,
                "fit": [
                    clean.index[0].date().isoformat(),
                    clean.index[fit_end - 1].date().isoformat(),
                    fit_end,
                ],
                "purgeTradingDays": purge,
                "evaluate": [
                    clean.index[evaluation_start].date().isoformat(),
                    clean.index[evaluation_end - 1].date().isoformat(),
                    evaluation_end - evaluation_start,
                ],
                "linearIcProxyMean": round(float(np.mean(proxy)), 8),
                "weights": {
                    column: round(float(value), 8)
                    for column, value in zip(clean.columns, weight, strict=True)
                },
            }
        )
    full = maximum_ic_ir_weights(clean, lower, upper)
    robust_centre = np.median(np.vstack([*fitted, full]), axis=0)
    recommended = project_bounded_simplex(robust_centre, lower, upper)
    return records, full, recommended


def weighted_composite(
    frames: dict[str, pd.DataFrame], weights: dict[str, float], available: pd.DataFrame
) -> pd.DataFrame:
    output: pd.DataFrame | None = None
    for name, weight in weights.items():
        if weight <= 0.0:
            continue
        contribution = frames[name] * float(weight)
        output = contribution if output is None else output + contribution
    if output is None:
        raise ValueError("empty factor combination")
    return output.where(available)


def series_metrics(series: pd.Series, hac_lag: int) -> dict[str, Any]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return {"n": int(len(clean)), "mean": None, "median": None, "std": None, "tHac": None}
    hac = autonomous.newey_west_t(clean.to_numpy(dtype=float), hac_lag)
    return {
        "n": int(len(clean)),
        "mean": round(float(clean.mean()), 8),
        "median": round(float(clean.median()), 8),
        "std": round(float(clean.std(ddof=1)), 8),
        "positiveRate": round(float((clean > 0.0).mean()), 8),
        "tHac": round(float(hac), 4) if hac is not None else None,
        "hacLag": hac_lag,
    }


def evaluate_scheme(
    name: str,
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
    factor_available: pd.DataFrame,
    target: pd.DataFrame,
    split: autonomous.Split,
    top_count: int,
    round_trip_cost: float,
    hac_lag: int,
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    composite = weighted_composite(frames, weights, factor_available)
    daily_rank_ic = composite.corrwith(target, axis=1, method="spearman")
    descending_rank = composite.rank(axis=1, ascending=False, method="first")
    selected = descending_rank.le(top_count)
    selected_return = target.where(selected)
    basket_gross = selected_return.mean(axis=1)
    basket_net = basket_gross - round_trip_cost
    benchmark = target.where(factor_available).mean(axis=1)
    excess = basket_gross - benchmark
    fills = selected_return.notna().sum(axis=1)
    masks = {
        "train": split.train,
        "validation": split.validation,
        "shadow": split.shadow,
    }
    periods: dict[str, Any] = {}
    for period, mask in masks.items():
        dates = mask[mask].index
        gross_slice = basket_gross.reindex(dates).dropna()
        independent = gross_slice.iloc[::horizon]
        individual = selected_return.reindex(dates).stack(future_stack=True).dropna()
        rank_slots: list[dict[str, Any]] = []
        for slot in range(1, top_count + 1):
            slot_return = (
                target.where(descending_rank.eq(slot))
                .stack(future_stack=True)
                .groupby(level=0)
                .first()
                .reindex(dates)
                .dropna()
            )
            slot_metrics = series_metrics(slot_return, hac_lag)
            slot_metrics["rank"] = slot
            slot_metrics["independent"] = series_metrics(
                slot_return.iloc[::horizon], 0
            )
            rank_slots.append(slot_metrics)
        periods[period] = {
            "rankIc": series_metrics(daily_rank_ic.reindex(dates), hac_lag),
            "top10GrossReturn": series_metrics(basket_gross.reindex(dates), hac_lag),
            "top10NetReturn": series_metrics(basket_net.reindex(dates), hac_lag),
            "top10ExcessReturn": series_metrics(excess.reindex(dates), hac_lag),
            "independentTenDayEvents": series_metrics(independent, 0),
            "individualSelections": {
                "n": int(len(individual)),
                "mean": round(float(individual.mean()), 8) if len(individual) else None,
                "winRate": round(float((individual > 0.0).mean()), 8) if len(individual) else None,
            },
            "averageRealisedFills": round(float(fills.reindex(dates).replace(0, np.nan).mean()), 4),
            "rankSlotGrossReturn": rank_slots,
        }
    latest_date = composite.index[-1]
    latest = (
        pd.DataFrame({"score": composite.loc[latest_date]})
        .dropna()
        .sort_values("score", ascending=False)
        .head(top_count)
    )
    latest["rank"] = np.arange(1, len(latest) + 1)
    latest["scheme"] = name
    latest["asOfDate"] = latest_date.date().isoformat()
    latest.index.name = "securityId"
    return {"weights": weights, "periods": periods}, latest.reset_index()


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# V2 + V10 联合因子稳健权重",
        "",
        "> **research-only / shadow-only / not a trading signal.** 权重仅由训练段拟合；验证和 shadow 未参与求权重。",
        "",
        f"- run_id: `{result['runId']}`",
        f"- 数据区间: {result['dataRange'][0]} ~ {result['dataRange'][1]}",
        f"- 切分: `{result['split']}`",
        f"- 独立因子数: {len(result['factorDefinitions'])}（一项旧/新重复表达式已合并）",
        "",
        "## 推荐的研究权重",
        "",
        "| 因子 | 权重 | 来源 |",
        "|---|---:|---|",
    ]
    weights = result["recommendedWeights"]
    for row in result["factorDefinitions"]:
        lines.append(
            f"| {row['key']} | {weights[row['key']] * 100:.2f}% | {'+'.join(row['sources'])} |"
        )
    lines.extend(
        [
            "",
            "## 同样本比较",
            "",
            "| 组合 | 区间 | Rank IC | IC HAC t | Top10均值 | Top10胜率 | 扣30bp均值 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scheme, payload in result["schemes"].items():
        for period in ("train", "validation", "shadow"):
            metrics = payload["periods"][period]
            ic = metrics["rankIc"]
            gross = metrics["top10GrossReturn"]
            net = metrics["top10NetReturn"]
            lines.append(
                f"| {scheme} | {period} | {100 * float(ic['mean'] or 0):.3f}% | "
                f"{ic['tHac']} | {100 * float(gross['mean'] or 0):.3f}% | "
                f"{100 * float(gross['positiveRate'] or 0):.1f}% | {100 * float(net['mean'] or 0):.3f}% |"
            )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 这是同一历史研究窗口上的再组合；即便 validation/shadow 改善，也不是全新的最终 OOS。",
            "- 原始 V10 四因子均至少失败一项稳健性门，联合权重不能洗掉这些单因子失败。",
            "- Top10 的10日标签每日重叠；报告使用 HAC，并另外列出每10日抽样的独立事件。",
            "- 不写交易配置、不生成订单、不自动晋级。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    definitions = factor_definitions(config)
    base = load_json(ROOT / config["baseResearchConfig"])
    _perception_config, cog_config = perception.load_base_configs(base)
    if int(cog_config["data"]["predictionHorizonTradingDays"]) != int(
        config["data"]["holdingTradingDays"]
    ):
        raise ValueError("base panel and joint-weight target horizons differ")
    print("loading PIT-adjusted panel ...", flush=True)
    panel, data_audit = perception.build_configured_panel(base, cog_config)
    target, _one_day = autonomous.target_frames(panel, cog_config)
    split = autonomous.make_split(panel["close"].index, cog_config)
    print(
        f"panel={panel['close'].shape[1]} symbols x {panel['close'].shape[0]} dates; split={split.audit}",
        flush=True,
    )
    bins = int(config["data"]["factorLiquidityNeutralisationBins"])
    frames: dict[str, pd.DataFrame] = {}
    factor_available = panel["eligible"].copy()
    for index, definition in enumerate(definitions, start=1):
        print(f"factor {index}/7: {definition['key']}", flush=True)
        raw = core.evaluate_expression(definition["expression"], panel).replace(
            [np.inf, -np.inf], np.nan
        )
        raw = raw * float(definition["direction"])
        neutral = autonomous.size_neutralise(raw, panel, bins)
        rank = neutral.rank(axis=1, pct=True).where(panel["eligible"]).astype(np.float32)
        frames[definition["key"]] = rank
        factor_available &= rank.notna()
        del raw, neutral
    target_common = target.where(factor_available)
    daily_ic = pd.DataFrame(
        {
            name: frame.corrwith(target_common, axis=1, method="spearman")
            for name, frame in frames.items()
        }
    )
    train_dates = split.train[split.train].index
    train_ic = daily_ic.reindex(train_dates)
    optimization = config["optimization"]
    fold_records, full_weights, recommended = expanding_weight_fits(
        train_ic,
        int(optimization["minimumFitTradingDays"]),
        int(optimization["innerFolds"]),
        int(optimization["purgeTradingDays"]),
        float(optimization["minimumWeight"]),
        float(optimization["maximumWeight"]),
    )
    names = list(frames)
    recommended_weights = {
        name: float(value) for name, value in zip(names, recommended, strict=True)
    }
    full_weight_map = {
        name: float(value) for name, value in zip(names, full_weights, strict=True)
    }
    old_names = [row["key"] for row in definitions if "old" in row["sources"]]
    new_names = [row["key"] for row in definitions if "new" in row["sources"]]
    new_plus_reversal_names = [*new_names, "multi_period_reversal"]
    if len(set(new_plus_reversal_names)) != 5:
        raise ValueError(
            "new-four plus multi-period reversal must contain five unique factors"
        )

    def equal(names_in_scheme: list[str]) -> dict[str, float]:
        return {name: (1.0 / len(names_in_scheme) if name in names_in_scheme else 0.0) for name in names}

    schemes = {
        "old_four_equal": equal(old_names),
        "new_four_equal": equal(new_names),
        "new_four_plus_multi_period_reversal_equal": equal(
            new_plus_reversal_names
        ),
        "union_seven_equal": equal(names),
        "union_seven_optimized": recommended_weights,
    }
    nested_config = config["nestedGroupOptimization"]
    nested_group_fits: dict[str, Any] = {}
    for group in nested_config["groups"]:
        group_name = str(group["name"])
        group_factors = list(group["factors"])
        missing = sorted(set(group_factors).difference(names))
        if missing:
            raise ValueError(f"nested group {group_name} has unknown factors: {missing}")
        fold_rows, group_full, group_recommended = expanding_weight_fits(
            train_ic[group_factors],
            int(nested_config["minimumFitTradingDays"]),
            int(nested_config["innerFolds"]),
            int(nested_config["purgeTradingDays"]),
            float(nested_config["minimumIncludedWeight"]),
            float(nested_config["maximumIncludedWeight"]),
        )
        group_weights = {name: 0.0 for name in names}
        for factor, value in zip(
            group_factors, group_recommended, strict=True
        ):
            group_weights[factor] = float(value)
        schemes[group_name] = group_weights
        nested_group_fits[group_name] = {
            "factors": group_factors,
            "selectionData": "train_only",
            "innerFolds": fold_rows,
            "fullTrainWeightsDiagnostic": {
                factor: float(value)
                for factor, value in zip(group_factors, group_full, strict=True)
            },
            "recommendedWeights": {
                factor: group_weights[factor] for factor in group_factors
            },
            "weightBoundsForIncludedFactors": [
                float(nested_config["minimumIncludedWeight"]),
                float(nested_config["maximumIncludedWeight"]),
            ],
        }
    evaluated: dict[str, Any] = {}
    latest_rows: list[pd.DataFrame] = []
    for scheme, weights in schemes.items():
        print(f"evaluate {scheme}", flush=True)
        metrics, latest = evaluate_scheme(
            scheme,
            frames,
            weights,
            factor_available,
            target,
            split,
            int(config["data"]["topCount"]),
            float(config["data"]["roundTripCost"]),
            int(config["evaluation"]["hacLagTradingDays"]),
            int(config["data"]["holdingTradingDays"]),
        )
        evaluated[scheme] = metrics
        latest_rows.append(latest)
    run_id = run_id or f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output = ROOT / config["output"]["root"] / run_id
    factor_rows = [
        {
            "key": row["key"],
            "factorId": row["factorId"],
            "direction": row["direction"],
            "sources": row["sources"],
            "aliases": row["aliases"],
            "expression": row["expression"],
        }
        for row in definitions
    ]
    result = {
        "schemaVersion": config["schemaVersion"],
        "status": "research_only_shadow_only_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "runId": run_id,
        "dataRange": [
            panel["close"].index[0].date().isoformat(),
            panel["close"].index[-1].date().isoformat(),
        ],
        "dataAudit": data_audit,
        "split": split.audit,
        "factorDefinitions": factor_rows,
        "commonFactorAvailability": {
            "rows": int(factor_available.to_numpy().sum()),
            "latestEligible": int(factor_available.iloc[-1].sum()),
        },
        "optimization": {
            "selectionData": "train_only",
            "method": optimization["method"],
            "innerFolds": fold_records,
            "fullTrainWeightsDiagnostic": full_weight_map,
            "recommendedAggregation": optimization["finalAggregation"],
            "weightBounds": [
                float(optimization["minimumWeight"]),
                float(optimization["maximumWeight"]),
            ],
        },
        "recommendedWeights": recommended_weights,
        "nestedGroupOptimization": nested_group_fits,
        "schemes": evaluated,
        "orders": [],
        "automaticTradingChanges": [],
        "promotionAllowed": False,
        "verdict": "research_only_reused_history_requires_fresh_forward_validation",
    }
    atomic_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_text(output / "report.md", markdown_report(result))
    pd.DataFrame(
        [
            {
                "factor": row["key"],
                "sources": "+".join(row["sources"]),
                "weight": recommended_weights[row["key"]],
                "weightPct": recommended_weights[row["key"]] * 100.0,
            }
            for row in factor_rows
        ]
    ).to_csv(output / "recommended_weights.csv", index=False, encoding="utf-8-sig")
    latest_frame = pd.concat(latest_rows, ignore_index=True)
    latest_frame.to_csv(output / "latest_rankings.csv", index=False, encoding="utf-8-sig")
    latest_estimates = latest_frame.copy()
    shadow_slot_lookup = {
        (scheme, int(row["rank"])): row
        for scheme, payload in evaluated.items()
        for row in payload["periods"]["shadow"]["rankSlotGrossReturn"]
    }
    latest_estimates["historicalShadowExpectedReturn10d"] = [
        shadow_slot_lookup[(str(row.scheme), int(row.rank))]["mean"]
        for row in latest_estimates.itertuples(index=False)
    ]
    latest_estimates["historicalShadowNonPositiveProbability10d"] = [
        (
            1.0 - float(shadow_slot_lookup[(str(row.scheme), int(row.rank))]["positiveRate"])
            if shadow_slot_lookup[(str(row.scheme), int(row.rank))].get("positiveRate")
            is not None
            else np.nan
        )
        for row in latest_estimates.itertuples(index=False)
    ]
    latest_estimates["historicalShadowObservations"] = [
        int(shadow_slot_lookup[(str(row.scheme), int(row.rank))]["n"])
        for row in latest_estimates.itertuples(index=False)
    ]
    latest_estimates["estimateKind"] = (
        "same_rank_shadow_history_not_stock_specific_probability_model"
    )
    latest_estimates.to_csv(
        output / "latest_rank_return_estimates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "group": group_name,
                "factor": factor,
                "included": factor in payload["factors"],
                "weight": evaluated[group_name]["weights"].get(factor, 0.0),
                "weightPct": evaluated[group_name]["weights"].get(factor, 0.0)
                * 100.0,
            }
            for group_name, payload in nested_group_fits.items()
            for factor in names
        ]
    ).to_csv(output / "nested_group_weights.csv", index=False, encoding="utf-8-sig")
    print(f"saved={output}", flush=True)
    return result


def self_test() -> None:
    rng = np.random.default_rng(20260805)
    values = rng.normal(0.0, 0.02, size=(700, 7))
    values[:, 0] += 0.012
    values[:, 1] += 0.006
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(7)])
    weight = maximum_ic_ir_weights(frame, 0.0, 0.35)
    assert math.isclose(float(weight.sum()), 1.0, abs_tol=1e-8)
    assert float(weight.min()) >= -1e-10
    assert float(weight.max()) <= 0.35 + 1e-10
    projected = project_bounded_simplex(np.array([1.0, 0, 0, 0, 0, 0, 0]), 0.0, 0.35)
    assert math.isclose(float(projected.sum()), 1.0, abs_tol=1e-8)
    assert float(projected.max()) <= 0.35 + 1e-10
    validate_config(load_json(DEFAULT_CONFIG))
    assert len(factor_definitions(load_json(DEFAULT_CONFIG))) == 7
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
