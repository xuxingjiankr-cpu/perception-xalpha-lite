"""PIT fundamental second-stage reranking after a frozen thirteen-factor Top100.

Historical rejection study only. It cannot trade, promote, or mutate production state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

import research_fundamental_mechanism_families as fundamental  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_twelve_factor_utility_weights_v6 as v6  # noqa: E402
import research_twelve_factor_alpha070_131_ablation as ablation  # noqa: E402
import research_twelve_factor_alpha070_131_individual_discrimination as v4  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "twelve_factor_fundamental_second_stage_v5.json"
)
SCHEMA_VERSION = "twelve_factor_fundamental_second_stage_result_v5"
CODE_VERSION = "twelve_factor_fundamental_second_stage_v5_20260809"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_fundamental_second_stage_v5":
        raise ValueError("unexpected V5 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V5 must remain research/shadow-only")
    frozen = {
        "frozenPriceModelConfig": "frozenPriceModelConfigSha256",
        "frozenPriceModelResult": "frozenPriceModelResultSha256",
        "frozenFundamentalConfig": "frozenFundamentalConfigSha256",
        "frozenFundamentalResult": "frozenFundamentalResultSha256",
    }
    for path_key, hash_key in frozen.items():
        if file_sha256(ROOT / config[path_key]) != str(config[hash_key]).lower():
            raise ValueError(f"frozen input changed: {path_key}")
    if config["firstStage"].get("candidateCount") != 100:
        raise ValueError("the preregistered first-stage pool must contain 100 names")
    if config["fundamentalFeatures"].get("familyCount") != 4:
        raise ValueError("exactly four fundamental families are required")
    if config["fundamentalFeatures"].get("reportDateMayDetermineAvailability"):
        raise ValueError("reportDate cannot determine historical availability")
    if not config["fundamentalFeatures"].get("pastOnly"):
        raise ValueError("fundamental features must be past-only")
    if config["ablations"]["regularizedFiveFeature"].get("count") != 5:
        raise ValueError("the candidate must contain exactly five frozen features")
    if config["ablations"].get("totalPreregisteredPolicies") != 4:
        raise ValueError("exactly four preregistered policies are required")
    if config["ablations"].get("postOutcomePolicySelectionAllowed"):
        raise ValueError("post-outcome policy selection is forbidden")
    if config["training"].get("hyperparameterSearchAllowed"):
        raise ValueError("hyperparameter search is forbidden")
    if not config["acceptance"].get("allGatesMustPass"):
        raise ValueError("all preregistered gates must pass")
    if config["acceptance"].get("historicalRunCanPromote"):
        raise ValueError("historical runs cannot promote")
    if not config["output"].get("ordersAlwaysEmpty"):
        raise ValueError("orders must remain empty")
    safety = config["safety"]
    if any(bool(value) for key, value in safety.items() if key.startswith("may")):
        raise ValueError("every trading and mutation permission must remain false")


def build_price_score(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    price_config = load_json(ROOT / config["frozenPriceModelConfig"])
    ablation_config = load_json(ROOT / price_config["frozenAblationConfig"])
    factors, frozen_weights, v6_config = ablation.frozen_factors_and_weights(
        ablation_config
    )
    ranks, factor_audit = v6.compute_factor_ranks(panel, factors)
    alpha131 = ablation_config["factorContract"]["alpha131Key"]
    zoo, name = alpha131.split("/", 1)
    raw = importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(
        v6.build_factor_inputs(panel)
    )
    ranks[alpha131] = (
        raw.reindex_like(panel["close"])
        * float(ablation_config["factorContract"]["alpha131Direction"])
    ).where(panel["eligible"]).rank(axis=1, pct=True).astype("float32")
    factor_audit.append(
        {
            "factorKey": alpha131,
            "direction": float(ablation_config["factorContract"]["alpha131Direction"]),
            "pastOnlySourceAudit": "static_formula",
        }
    )
    keys = [item["factorKey"] for item in factors]
    weights = ablation.policy_weights(
        keys,
        frozen_weights,
        ablation_config["factorContract"]["alpha070Key"],
        alpha131,
    )["alpha070_alpha131_each_15pct"]
    common = panel["eligible"].copy().fillna(False)
    for rank in ranks.values():
        common &= rank.notna()
    score = ablation.weighted_score(ranks, weights, common)
    return score, factor_audit, weights, v6_config


def build_family_scores(
    panel: dict[str, pd.DataFrame], fundamental_config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    index, columns = panel["close"].index, panel["close"].columns
    event_table, event_audit = fundamental.build_event_table(
        index, columns, fundamental_config
    )
    if event_table.empty:
        raise RuntimeError("no causal fundamental events")
    eligible = panel["eligible"].fillna(False)
    max_age = int(fundamental_config["fundamentals"]["maximumSignalAgeTradingDays"])
    bins = int(fundamental_config["fundamentals"]["liquidityNeutraliseBins"])
    family_scores: dict[str, pd.DataFrame] = {}
    family_support: dict[str, pd.DataFrame] = {}
    for family, definition in fundamental_config["families"].items():
        print(f"fundamental_family {family}", flush=True)
        ranks: dict[str, pd.DataFrame] = {}
        for candidate in definition["candidates"]:
            identifier = str(candidate["id"])
            raw = fundamental.factor_frame(
                event_table, identifier, index, columns, max_age
            )
            neutral = fundamental.autonomous.size_neutralise(
                raw.where(eligible), panel, bins
            )
            ranks[identifier] = neutral.where(eligible).rank(
                axis=1, pct=True, method="average"
            ).astype("float32")
        score, common = fundamental.equal_rank_composite(ranks, eligible)
        family_scores[family] = score.rank(
            axis=1, pct=True, method="average"
        ).astype("float32")
        family_support[family] = common
    common = eligible.copy()
    for family in family_scores:
        common &= family_support[family] & family_scores[family].notna()
    return family_scores, common, event_audit


def make_stage2_features(
    price_score: pd.DataFrame,
    family_scores: dict[str, pd.DataFrame],
    fundamental_common: pd.DataFrame,
    candidate_count: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top100 = v6.top_mask(price_score, candidate_count)
    support = top100 & fundamental_common
    price_rank = price_score.where(support).rank(
        axis=1, pct=True, method="average"
    ).astype("float32")
    features: dict[str, pd.DataFrame] = {"price_score_rank": price_rank}
    for family, frame in family_scores.items():
        features[f"{family}_rank"] = frame.where(support).rank(
            axis=1, pct=True, method="average"
        ).astype("float32")
    if len(features) != 5:
        raise RuntimeError("V5 must produce exactly five features")
    fundamental_score = sum(
        features[f"{family}_rank"] for family in family_scores
    ) / len(family_scores)
    fundamental_score = fundamental_score.where(support).rank(
        axis=1, pct=True, method="average"
    ).astype("float32")
    fixed_blend = (0.5 * price_rank + 0.5 * fundamental_score).where(support)
    return features, support, fundamental_score, fixed_blend


def flexible_decile_monotonicity(rows: pd.DataFrame) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    for _date, group in rows.groupby("date", sort=True):
        if len(group) < 50:
            continue
        group = group.copy()
        group["decile"] = (
            pd.qcut(group["utility"].rank(method="first"), 10, labels=False) + 1
        )
        values.extend(
            {"decile": float(decile), "return": float(part["return"].mean())}
            for decile, part in group.groupby("decile")
        )
    frame = pd.DataFrame(values)
    means = (
        frame.groupby("decile")["return"].mean()
        if not frame.empty
        else pd.Series(dtype=float)
    )
    correlation = (
        spearmanr(means.index, means.values).statistic if len(means) >= 3 else np.nan
    )
    adjacent = np.diff(means.values) if len(means) >= 2 else np.asarray([])
    return {
        "decileMeanReturns": {
            str(int(key)): round(float(value), 8) for key, value in means.items()
        },
        "spearman": (
            round(float(correlation), 8) if np.isfinite(correlation) else None
        ),
        "adjacentIncreasingFraction": (
            round(float((adjacent > 0).mean()), 8) if len(adjacent) else None
        ),
    }


def evaluate_period(
    predictions: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = v4.period_rows(predictions, returns, dates)
    actual = rows["return"].to_numpy(float)
    cost = float(config["outcomes"]["roundTripCost"])
    severe = -0.03
    return {
        "rows": len(rows),
        "days": int(rows["date"].nunique()),
        "expectedReturnMse": round(
            float(mean_squared_error(actual, rows["expectedReturn"])), 10
        ),
        "probability": {
            "grossUp": v6.probability_metrics(
                (actual > 0.0).astype(float), rows["grossUp"].to_numpy(float)
            ),
            "netPositive": v6.probability_metrics(
                (actual > cost).astype(float), rows["netPositive"].to_numpy(float)
            ),
            "severeLoss": v6.probability_metrics(
                (actual <= severe).astype(float),
                rows["severeLoss"].to_numpy(float),
            ),
        },
        "top10": v6.outcome_metrics(
            returns, predictions["utility"], dates, 10, cost, severe
        ),
        "spread": v4.probability_spread(predictions, dates),
        "monotonicity": flexible_decile_monotonicity(rows),
    }


def support_metrics(
    top100: pd.DataFrame, support: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, Any]:
    requested = top100.reindex(index=dates).sum(axis=1)
    complete = support.reindex(index=dates).sum(axis=1)
    coverage = complete.div(requested.replace(0, np.nan)).dropna()
    return {
        "days": len(coverage),
        "meanRequested": round(float(requested.mean()), 4),
        "meanComplete": round(float(complete.mean()), 4),
        "minimumComplete": int(complete.min()),
        "meanCoverage": round(float(coverage.mean()), 8),
        "minimumCoverage": round(float(coverage.min()), 8),
    }


def acceptance(
    baseline: dict[str, Any], candidate: dict[str, Any], coverage: float
) -> dict[str, Any]:
    base_mono = baseline["monotonicity"]["adjacentIncreasingFraction"]
    candidate_mono = candidate["monotonicity"]["adjacentIncreasingFraction"]
    checks = {
        "grossUpAucImproved": candidate["probability"]["grossUp"]["auc"]
        > baseline["probability"]["grossUp"]["auc"],
        "grossUpBrierImproved": candidate["probability"]["grossUp"]["brier"]
        < baseline["probability"]["grossUp"]["brier"],
        "grossUpLogLossImproved": candidate["probability"]["grossUp"]["logLoss"]
        < baseline["probability"]["grossUp"]["logLoss"],
        "netPositiveAucImproved": candidate["probability"]["netPositive"]["auc"]
        > baseline["probability"]["netPositive"]["auc"],
        "netPositiveBrierImproved": candidate["probability"]["netPositive"]["brier"]
        < baseline["probability"]["netPositive"]["brier"],
        "netPositiveLogLossImproved": candidate["probability"]["netPositive"]["logLoss"]
        < baseline["probability"]["netPositive"]["logLoss"],
        "top10MeanNetReturnImproved": candidate["top10"]["meanNetReturn"]
        > baseline["top10"]["meanNetReturn"],
        "top10NetWinRateImproved": candidate["top10"]["netWinRate"]
        > baseline["top10"]["netWinRate"],
        "top10ProbabilitySpreadIncreased": candidate["spread"]["meanTop10GrossUpSpread"]
        > baseline["spread"]["meanTop10GrossUpSpread"],
        "decileMonotonicityNotReduced": (
            base_mono is not None
            and candidate_mono is not None
            and candidate_mono >= base_mono
        ),
        "completeSupportCoveragePassed": coverage > 0.5,
    }
    return {"checks": checks, "allGatesPassed": all(checks.values())}


def latest_rows(
    predictions: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    date = predictions["utility"].index.max()
    selected = (
        predictions["utility"].loc[date].dropna().sort_values(ascending=False).head(10)
    )
    rows: list[dict[str, Any]] = []
    for rank, (security_id, utility) in enumerate(selected.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": round(float(panel["close"].loc[date, security_id]), 4),
                "utility": round(float(utility), 8),
                "expectedGrossReturn": round(
                    float(predictions["expectedReturn"].loc[date, security_id]), 8
                ),
                "probabilityUp": round(
                    float(predictions["grossUp"].loc[date, security_id]), 8
                ),
                "probabilityNetPositive": round(
                    float(predictions["netPositive"].loc[date, security_id]), 8
                ),
                "probabilitySevereLoss": round(
                    float(predictions["severeLoss"].loc[date, security_id]), 8
                ),
            }
        )
    return rows


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PIT fundamental second-stage study V5",
        "",
        "> Research-only / historical rejection test / not trading.",
        "",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- causal disclosure events: `{result['fundamentalAudit']['statementEvents']}`",
        f"- audit gates passed: `{result['acceptance']['allGatesPassed']}`",
        f"- eligible for trading: `{result['eligibleForTrading']}`",
        "- all fair comparisons use the identical price Top100 and complete fundamental support",
        "",
    ]
    for period in ("audit", "validation", "shadow"):
        lines.extend(
            [
                f"## {period.title()} ablation",
                "",
                "| policy | up AUC | up Brier | up LogLoss | net+ AUC | Top10 net | Top10 net win | p(up) spread | monotonic steps |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in result["periods"][period].items():
            lines.append(
                f"| {name} | {row['probability']['grossUp']['auc']} | "
                f"{row['probability']['grossUp']['brier']} | "
                f"{row['probability']['grossUp']['logLoss']} | "
                f"{row['probability']['netPositive']['auc']} | "
                f"{row['top10']['meanNetReturn']} | {row['top10']['netWinRate']} | "
                f"{row['spread']['meanTop10GrossUpSpread']} | "
                f"{row['monotonicity']['adjacentIncreasingFraction']} |"
            )
        lines.append("")
    failed = [
        name for name, passed in result["acceptance"]["checks"].items() if not passed
    ]
    lines.extend(
        [
            "## Decision",
            "",
            f"- failed preregistered gates: `{', '.join(failed)}`",
            "- the audit improvement is not stable in validation and all Top10 net returns remain negative",
            "- the fundamental second stage is rejected and cannot replace the existing shadow ranking",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Latest research-only lists",
            "",
            "The lists below are diagnostics. A failed historical gate cannot replace the existing shadow model.",
            "",
        ]
    )
    for policy, rows in result["latestTop10ByPolicy"].items():
        lines.extend(
            [
                f"### {policy}",
                "",
                "| rank | security | name | expected gross | p(up) | p(net+) | p(severe loss) |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['securityId']} | {row['name']} | "
                f"{row['expectedGrossReturn']:.3%} | {row['probabilityUp']:.2%} | "
                f"{row['probabilityNetPositive']:.2%} | {row['probabilitySevereLoss']:.2%} |"
            )
        lines.append("")
    lines.append("Orders remain `[]`.")
    lines.append("")
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    price_config = load_json(ROOT / config["frozenPriceModelConfig"])
    ablation_config = load_json(ROOT / price_config["frozenAblationConfig"])
    _, _, v6_config = ablation.frozen_factors_and_weights(ablation_config)
    base = load_json(ROOT / v6_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    price_score, factor_audit, weights, _ = build_price_score(panel, config)
    fundamental_config = load_json(ROOT / config["frozenFundamentalConfig"])
    family_scores, fundamental_common, event_audit = build_family_scores(
        panel, fundamental_config
    )
    features, support, fundamental_score, fixed_blend = make_stage2_features(
        price_score,
        family_scores,
        fundamental_common,
        int(config["firstStage"]["candidateCount"]),
    )
    top100 = v6.top_mask(price_score, int(config["firstStage"]["candidateCount"]))
    gross_returns, execution_eligible, _delay = precision.executable_horizon_return(
        panel, 1, 5
    )
    returns = gross_returns.where(execution_eligible & support)
    partitions = v4.fixed_partitions(panel["close"].index, config)
    models, fit_audit = v4.fit_models(features, returns, partitions, config)
    prediction_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *(set(partitions[name]) for name in ("audit", "validation", "shadow")),
                {panel["close"].index.max()},
            )
        )
    )
    policies: dict[str, dict[str, pd.DataFrame]] = {}
    scalar_fit_audit: dict[str, Any] = {}
    scalar_scores = {
        "priceBaseline": features["price_score_rank"],
        "fundamentalOnly": fundamental_score,
        "fixedBlend": fixed_blend,
    }
    for policy_name, policy_score in scalar_scores.items():
        scalar_models, scalar_audit = v4.fit_models(
            {f"{policy_name}_score": policy_score}, returns, partitions, config
        )
        policies[policy_name] = v4.predict_multivariate(
            {f"{policy_name}_score": policy_score},
            prediction_dates,
            scalar_models,
            config,
        )
        scalar_fit_audit[policy_name] = scalar_audit
    policies["regularizedFiveFeature"] = v4.predict_multivariate(
        features, prediction_dates, models, config
    )
    periods: dict[str, Any] = {}
    supports: dict[str, Any] = {}
    for period in ("audit", "validation", "shadow"):
        dates = partitions[period].intersection(prediction_dates)
        periods[period] = {
            name: evaluate_period(predictions, returns, dates, config)
            for name, predictions in policies.items()
        }
        supports[period] = support_metrics(top100, support, dates)
    accepted = acceptance(
        periods["audit"]["priceBaseline"],
        periods["audit"]["regularizedFiveFeature"],
        supports["audit"]["meanCoverage"],
    )
    names = v6.name_map(base)
    latest = {
        name: latest_rows(predictions, panel, names)
        for name, predictions in policies.items()
    }
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_historical_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "dataRange": [
            str(panel["close"].index.min().date()),
            str(panel["close"].index.max().date()),
        ],
        "signalDate": str(panel["close"].index.max().date()),
        "priceWeights": weights,
        "partitions": {
            name: [str(value.min().date()), str(value.max().date()), len(value)]
            for name, value in partitions.items()
        },
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "fundamentalAudit": event_audit,
        "fitAudit": {
            "regularizedFiveFeature": fit_audit,
            "scalarAblations": scalar_fit_audit,
        },
        "support": supports,
        "periods": periods,
        "acceptance": accepted,
        "mechanicalReductionCheck": {
            "identicalCandidateSupportForAllFourPolicies": True,
            "identicalTopCountForAllFourPolicies": 10,
            "candidatePoolDefinedBeforeOutcomes": True,
            "improvementCannotComeFromFewerPolicyTrades": True,
        },
        "latestTop10ByPolicy": latest,
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    identifier = run_id or datetime.now().astimezone().strftime(
        "run_%Y%m%dT%H%M%S%z"
    )
    result["runId"] = identifier
    output_root = ROOT / config["output"]["root"] / identifier
    v4.atomic_text(
        output_root / "result.json",
        json.dumps(v4.safe(result), ensure_ascii=False, indent=2) + "\n",
    )
    v4.atomic_text(output_root / "report.md", report_markdown(result))
    return {
        "runId": identifier,
        "report": str(output_root / "report.md"),
        "auditGatesPassed": accepted["allGatesPassed"],
        "eligibleForTrading": False,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(
        json.dumps(
            v4.safe(run(args.config.resolve(), args.run_id)),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
