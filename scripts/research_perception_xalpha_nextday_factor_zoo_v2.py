#!/usr/bin/env python3
"""Train-only factor-zoo discovery for the shortest executable A-share horizon.

This module is permanently research/shadow-only.  Factor direction, multiple-testing
correction, stability filtering, de-correlation and weights are decided using the existing
training split only.  Validation and shadow outcomes can reject the result but can never
change the discovered factor book.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT
import research_cogalpha_autonomous as autonomous
import research_perception_xalpha_nextday_explosion as v1


SCHEMA_VERSION = "perception_xalpha_nextday_factor_zoo_result_v2"
CODE_VERSION = "nextday_factor_zoo_v2_20260806"
DEFAULT_CONFIG = ROOT / "configs" / "research" / "perception_xalpha_nextday_factor_zoo_v2.json"
VENDOR_ROOT = ROOT / "scripts" / "vendor" / "vibe_factors"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


@dataclass(frozen=True)
class FactorScreen:
    factor_key: str
    direction: float
    orientation_ic_mean: float
    orientation_ic_hac_t: float | None
    confirmation_ic_mean: float
    confirmation_ic_hac_t: float | None
    confirmation_p_value: float
    confirmation_q_value: float | None
    positive_folds: int
    gross_excess_bps: float
    non_positive_improvement: float
    severe_loss_improvement: float
    turnover: float
    turnover_efficiency: float
    usable_days: int
    selection_score: float | None = None
    selected: bool = False
    rejection_reasons: tuple[str, ...] = ()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively make research diagnostics strict-JSON serialisable."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_nextday_factor_zoo_v2":
        raise ValueError("unexpected V2 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("V2 must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every V2 mutation permission must remain false")
    discovery = config["factorDiscovery"]
    if int(discovery["declaredLibrarySize"]) != 456:
        raise ValueError("the preregistered multiple-testing burden must remain 456")
    if bool(discovery["validationAndShadowCompletelyForbiddenForDiscovery"]) is not True:
        raise ValueError("external outcomes must be forbidden during discovery")
    if abs(sum(map(float, discovery["selectionScoreWeights"].values())) - 1.0) > 1e-9:
        raise ValueError("factor selection-score weights must sum to one")
    generator = config["candidateGenerator"]
    if int(generator["candidatePoolSize"]) != 1000:
        raise ValueError("the preregistered candidate pool must remain 1000")
    if abs(
        float(generator["legacyBurstScoreWeight"])
        + float(generator["discoveredFactorScoreWeight"])
        - 1.0
    ) > 1e-9:
        raise ValueError("candidate score weights must sum to one")
    policy = config["selectionPolicy"]
    if not policy.get("allowCash") or not policy.get("neverForceSelections"):
        raise ValueError("V2 selection must allow an empty book")
    if int(policy["maximumSelectionsPerDay"]) != 10:
        raise ValueError("V2 may expose at most ten shadow rows")
    if abs(sum(map(float, policy["integratedRankingWeights"].values())) - 1.0) > 1e-9:
        raise ValueError("integrated ranking weights must sum to one")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("V2 output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("V2 orders must remain empty")


def enumerate_zoo(config: dict[str, Any]) -> list[tuple[str, str]]:
    factors: list[tuple[str, str]] = []
    for zoo in config["factorDiscovery"]["libraries"]:
        root = VENDOR_ROOT / "src" / "factors" / "zoo" / str(zoo)
        for path in sorted(root.glob("*.py")):
            if path.stem != "__init__":
                factors.append((str(zoo), path.stem))
    if len(factors) != int(config["factorDiscovery"]["declaredLibrarySize"]):
        raise RuntimeError(
            f"declared factor burden differs from code: {len(factors)} != "
            f"{config['factorDiscovery']['declaredLibrarySize']}"
        )
    return factors


def screening_columns(columns: pd.Index, size: int) -> list[str]:
    ordered = sorted(map(str, columns))
    if len(ordered) <= size:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, size, dtype=int)
    return [ordered[index] for index in positions]


def factor_panel(panel: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key, value in panel.items():
        if isinstance(value, pd.DataFrame):
            subset[key] = value.reindex(columns=columns)
        else:
            subset[key] = value
    close = subset["close"]
    subset["returns"] = close.pct_change(fill_method=None)
    if "vwap" not in subset:
        volume = subset["volume"].replace(0.0, np.nan)
        subset["vwap"] = subset["amount"].div(volume).combine_first(close)
    return subset


def _normal_one_sided_p(statistic: float | None) -> float:
    if statistic is None or not math.isfinite(statistic):
        return 1.0
    return float(0.5 * math.erfc(float(statistic) / math.sqrt(2.0)))


def benjamini_hochberg(
    p_values: dict[str, float],
    declared_trials: int,
) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    raw: dict[str, float] = {}
    for rank, (name, value) in enumerate(ordered, start=1):
        raw[name] = min(1.0, max(0.0, float(value)) * declared_trials / rank)
    adjusted: dict[str, float] = {}
    running = 1.0
    for name, _ in reversed(ordered):
        running = min(running, raw[name])
        adjusted[name] = running
    return adjusted


def _safe_hac(values: pd.Series, lag: int) -> float | None:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) < max(20, lag + 3):
        return None
    value = autonomous.newey_west_t(clean, lag)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _daily_top_book(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
    top_fraction: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    signal = signal.reindex(index=dates)
    target = target.reindex(index=dates)
    allowed = eligible.reindex(index=dates).fillna(False) & target.notna() & signal.notna()
    ranks = signal.where(allowed).rank(axis=1, pct=True)
    top = ranks.ge(1.0 - top_fraction) & allowed
    top_mean = target.where(top).mean(axis=1)
    benchmark = target.where(allowed).mean(axis=1)
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return top_mean, benchmark, weights


def screen_one_factor(
    factor_key: str,
    raw_signal: pd.DataFrame,
    target: pd.DataFrame,
    eligible: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[FactorScreen | None, pd.Series]:
    spec = config["factorDiscovery"]
    minimum_cross_section = int(spec["minimumDailyCrossSection"])
    signal = (
        raw_signal.reindex(index=target.index, columns=target.columns)
        .where(eligible)
        .replace([np.inf, -np.inf], np.nan)
    )
    usable = (
        signal.reindex(index=train_dates).notna()
        & target.reindex(index=train_dates).notna()
        & eligible.reindex(index=train_dates).fillna(False)
    ).sum(axis=1).ge(minimum_cross_section)
    usable_dates = pd.DatetimeIndex(usable[usable].index)
    if len(usable_dates) < 300:
        return None, pd.Series(dtype=float)
    daily_ic = signal.reindex(index=usable_dates).corrwith(
        target.reindex(index=usable_dates), axis=1, method="spearman"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily_ic) < 300:
        return None, pd.Series(dtype=float)
    cut = max(100, int(len(daily_ic) * float(spec["orientationFractionOfTrain"])))
    if len(daily_ic) - cut < 100:
        return None, pd.Series(dtype=float)
    orientation = daily_ic.iloc[:cut]
    confirmation_raw = daily_ic.iloc[cut:]
    direction = 1.0 if float(orientation.mean()) >= 0.0 else -1.0
    orientation_t_raw = _safe_hac(orientation, int(spec["hacLagTradingDays"]))
    orientation_t = abs(orientation_t_raw) if orientation_t_raw is not None else None
    confirmation_ic = confirmation_raw * direction
    confirmation_t = _safe_hac(confirmation_ic, int(spec["hacLagTradingDays"]))
    confirmation_dates = pd.DatetimeIndex(confirmation_ic.index)
    folds = np.array_split(np.arange(len(confirmation_ic)), int(spec["confirmationFolds"]))
    positive_folds = sum(
        bool(len(indices) and float(confirmation_ic.iloc[indices].mean()) > 0.0)
        for indices in folds
    )
    oriented = signal * direction
    top_mean, benchmark, weights = _daily_top_book(
        oriented,
        target,
        eligible,
        confirmation_dates,
        float(spec["topFraction"]),
    )
    gross_excess = (top_mean - benchmark).dropna()
    loss = target.le(0.0).where(target.notna()).astype(float)
    severe = target.le(float(config["data"]["severeLossThreshold"])).where(target.notna()).astype(float)
    top_loss, bench_loss, _ = _daily_top_book(
        oriented, loss, eligible & target.notna(), confirmation_dates, float(spec["topFraction"])
    )
    top_severe, bench_severe, _ = _daily_top_book(
        oriented, severe, eligible & target.notna(), confirmation_dates, float(spec["topFraction"])
    )
    turnover = float((weights.diff().abs().sum(axis=1) / 2.0).mean())
    gross_bps = float(gross_excess.mean()) * 1e4 if len(gross_excess) else float("nan")
    loss_improvement = float((bench_loss - top_loss).mean())
    severe_improvement = float((bench_severe - top_severe).mean())
    efficiency = gross_bps / max(turnover * 1e4, 1e-9)
    result = FactorScreen(
        factor_key=factor_key,
        direction=direction,
        orientation_ic_mean=float(orientation.mean() * direction),
        orientation_ic_hac_t=orientation_t,
        confirmation_ic_mean=float(confirmation_ic.mean()),
        confirmation_ic_hac_t=confirmation_t,
        confirmation_p_value=_normal_one_sided_p(confirmation_t),
        confirmation_q_value=None,
        positive_folds=int(positive_folds),
        gross_excess_bps=gross_bps,
        non_positive_improvement=loss_improvement,
        severe_loss_improvement=severe_improvement,
        turnover=turnover,
        turnover_efficiency=efficiency,
        usable_days=len(daily_ic),
    )
    return result, confirmation_ic


def _replace_screen(item: FactorScreen, **changes: Any) -> FactorScreen:
    values = dict(item.__dict__)
    values.update(changes)
    return FactorScreen(**values)


def discover_factors(
    panel: dict[str, Any],
    outcomes: dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[list[FactorScreen], list[FactorScreen], dict[str, pd.Series], dict[str, int]]:
    spec = config["factorDiscovery"]
    columns = screening_columns(panel["close"].columns, int(spec["screeningUniverseSize"]))
    subset = factor_panel(panel, columns)
    eligible = subset["eligible"].fillna(False)
    target = outcomes["target_executable_return"].reindex(columns=columns)
    screens: list[FactorScreen] = []
    ic_series: dict[str, pd.Series] = {}
    failures: dict[str, int] = {}
    declared = enumerate_zoo(config)
    for index, (zoo, name) in enumerate(declared, start=1):
        key = f"{zoo}/{name}"
        try:
            module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
            raw = module.compute(subset)
            if not isinstance(raw, pd.DataFrame):
                raise TypeError("factor compute did not return a DataFrame")
            item, series = screen_one_factor(
                key, raw, target, eligible, train_dates, config
            )
            if item is None:
                failures["insufficient_usable_data"] = failures.get("insufficient_usable_data", 0) + 1
                continue
            screens.append(item)
            ic_series[key] = series
        except Exception as exc:  # Factor zoo is adversarial input; fail one candidate closed.
            reason = type(exc).__name__
            failures[reason] = failures.get(reason, 0) + 1
        if index % 50 == 0 or index == len(declared):
            print(
                f"factor_screen_progress={index}/{len(declared)} usable={len(screens)}",
                flush=True,
            )
    q_values = benjamini_hochberg(
        {item.factor_key: item.confirmation_p_value for item in screens},
        int(spec["declaredLibrarySize"]),
    )
    gated: list[FactorScreen] = []
    for item in screens:
        q_value = q_values.get(item.factor_key, 1.0)
        reasons: list[str] = []
        if item.orientation_ic_hac_t is None or item.orientation_ic_hac_t < float(
            spec["minimumOrientationAbsoluteHacT"]
        ):
            reasons.append("orientation_hac_t")
        if item.confirmation_ic_hac_t is None or item.confirmation_ic_hac_t < float(
            spec["minimumConfirmationHacT"]
        ):
            reasons.append("confirmation_hac_t")
        if item.positive_folds < int(spec["minimumPositiveConfirmationFolds"]):
            reasons.append("fold_instability")
        if q_value > float(spec["maximumBenjaminiHochbergQ"]):
            reasons.append("fdr")
        if item.gross_excess_bps <= float(spec["minimumConfirmationGrossExcessBps"]):
            reasons.append("gross_excess")
        if item.non_positive_improvement <= float(
            spec["minimumConfirmationNonPositiveImprovement"]
        ):
            reasons.append("non_positive_not_improved")
        if item.severe_loss_improvement < float(
            spec["minimumConfirmationSevereLossImprovement"]
        ):
            reasons.append("severe_loss_worse")
        if item.turnover > float(spec["maximumDailyTurnover"]):
            reasons.append("turnover")
        updated = _replace_screen(
            item,
            confirmation_q_value=float(q_value),
            rejection_reasons=tuple(reasons),
        )
        gated.append(updated)
    survivors = [item for item in gated if not item.rejection_reasons]
    if survivors:
        frame = pd.DataFrame(
            {
                "factor_key": [item.factor_key for item in survivors],
                "confirmation_ic_t_rank": [item.confirmation_ic_hac_t for item in survivors],
                "gross_excess_rank": [item.gross_excess_bps for item in survivors],
                "non_positive_improvement_rank": [item.non_positive_improvement for item in survivors],
                "severe_loss_improvement_rank": [item.severe_loss_improvement for item in survivors],
                "turnover_efficiency_rank": [item.turnover_efficiency for item in survivors],
            }
        ).set_index("factor_key")
        weights = spec["selectionScoreWeights"]
        score = pd.Series(0.0, index=frame.index)
        for metric, weight in weights.items():
            score = score.add(frame[metric].rank(pct=True) * float(weight), fill_value=0.0)
        by_key = {item.factor_key: item for item in gated}
        gated = [
            _replace_screen(item, selection_score=float(score[item.factor_key]))
            if item.factor_key in score
            else item
            for item in gated
        ]
        candidates = sorted(
            (item for item in gated if item.selection_score is not None),
            key=lambda item: (-float(item.selection_score), item.factor_key),
        )
        retained: list[FactorScreen] = []
        correlation_limit = float(spec["maximumPairwiseDailyIcCorrelation"])
        for item in candidates:
            too_close = False
            for prior in retained:
                pair = pd.concat(
                    [ic_series[item.factor_key], ic_series[prior.factor_key]], axis=1
                ).dropna()
                correlation = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) >= 20 else 0.0
                if math.isfinite(correlation) and abs(correlation) > correlation_limit:
                    too_close = True
                    break
            if too_close:
                continue
            retained.append(item)
            if len(retained) >= int(spec["maximumSelectedFactors"]):
                break
        selected_keys = {item.factor_key for item in retained}
        gated = [
            _replace_screen(item, selected=item.factor_key in selected_keys)
            for item in gated
        ]
    selected = sorted(
        (item for item in gated if item.selected),
        key=lambda item: (-float(item.selection_score or 0.0), item.factor_key),
    )
    return gated, selected, ic_series, failures


def _feature_name(factor_key: str, config: dict[str, Any]) -> str:
    prefix = str(config["featureSet"]["discoveredColumnsPrefix"])
    return prefix + re.sub(r"[^a-zA-Z0-9]+", "_", factor_key).strip("_").lower()


def compute_selected_factor_features(
    panel: dict[str, Any],
    selected: list[FactorScreen],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, float]]:
    features: dict[str, pd.DataFrame] = {}
    scores: list[tuple[pd.DataFrame, float]] = []
    raw_weights = np.asarray(
        [max(float(item.selection_score or 0.0), 1e-9) for item in selected], dtype=float
    )
    raw_weights = raw_weights / raw_weights.sum() if len(raw_weights) else raw_weights
    weights: dict[str, float] = {}
    for item, weight in zip(selected, raw_weights):
        zoo, name = item.factor_key.split("/", 1)
        module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
        raw = module.compute(panel)
        oriented = raw.reindex_like(panel["close"]) * float(item.direction)
        rank = oriented.where(panel["eligible"]).rank(axis=1, pct=True)
        feature_name = _feature_name(item.factor_key, config)
        features[feature_name] = rank
        scores.append((rank, float(weight)))
        weights[item.factor_key] = float(weight)
    if not scores:
        empty = panel["close"] * np.nan
        return features, empty, weights
    composite = panel["close"] * 0.0
    available = panel["close"] * 0.0
    for rank, weight in scores:
        composite = composite.add(rank.fillna(0.0) * weight, fill_value=0.0)
        available = available.add(rank.notna().astype(float) * weight, fill_value=0.0)
    composite = composite.div(available.replace(0.0, np.nan))
    return features, composite.where(panel["eligible"]), weights


def audit_selected_prefix_causality(
    panel: dict[str, Any],
    selected: list[FactorScreen],
    train_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[list[FactorScreen], dict[str, Any]]:
    """Reject a selected factor if an unseen suffix changes its earlier values."""
    if not selected:
        return [], {"checked": 0, "passed": 0, "failed": []}
    columns = screening_columns(panel["close"].columns, 80)
    complete = factor_panel(panel, columns)
    cutoff = pd.Timestamp(train_dates.max())
    prefix: dict[str, Any] = {}
    for key, value in complete.items():
        prefix[key] = value.loc[:cutoff] if isinstance(value, pd.DataFrame) else value
    comparison_dates = pd.DatetimeIndex(
        complete["close"].index[complete["close"].index <= cutoff][-10:]
    )
    passed: list[FactorScreen] = []
    failed: list[dict[str, Any]] = []
    for item in selected:
        zoo, name = item.factor_key.split("/", 1)
        try:
            module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
            full_values = module.compute(complete).reindex(
                index=comparison_dates, columns=columns
            )
            prefix_values = module.compute(prefix).reindex(
                index=comparison_dates, columns=columns
            )
            left = full_values.to_numpy(dtype=float)
            right = prefix_values.to_numpy(dtype=float)
            equal = bool(np.allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True))
            maximum_difference = float(
                np.nanmax(np.abs(left - right))
            ) if np.isfinite(left - right).any() else 0.0
        except Exception as exc:
            equal = False
            maximum_difference = None
            failed.append(
                {
                    "factorKey": item.factor_key,
                    "reason": type(exc).__name__,
                    "maximumDifference": maximum_difference,
                }
            )
            continue
        if equal:
            passed.append(item)
        else:
            failed.append(
                {
                    "factorKey": item.factor_key,
                    "reason": "prefix_changed_by_unseen_suffix",
                    "maximumDifference": maximum_difference,
                }
            )
    return passed, {"checked": len(selected), "passed": len(passed), "failed": failed}


def legacy_candidate_score(
    features: dict[str, pd.DataFrame],
    panel: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    score = panel["close"] * 0.0
    available = panel["close"] * 0.0
    for name, weight in config["legacyBurstWeights"].items():
        ranked = features[name].where(panel["eligible"]).rank(axis=1, pct=True)
        score = score.add(ranked.fillna(0.0) * float(weight), fill_value=0.0)
        available = available.add(ranked.notna().astype(float) * float(weight), fill_value=0.0)
    return score.div(available.replace(0.0, np.nan)).where(panel["eligible"])


def combined_candidate_score(
    legacy: pd.DataFrame,
    discovered: pd.DataFrame,
    panel: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generator = config["candidateGenerator"]
    score = (
        legacy.rank(axis=1, pct=True) * float(generator["legacyBurstScoreWeight"])
        + discovered.rank(axis=1, pct=True)
        * float(generator["discoveredFactorScoreWeight"])
    ).where(panel["eligible"])
    rank = score.rank(axis=1, ascending=False, method="first")
    mask = rank.le(int(generator["candidatePoolSize"]))
    return score, rank, mask


def all_heads_ready(model: v1.ExplosionModel) -> bool:
    return bool(
        model.return_enabled
        and model.tail_enabled
        and all(head.enabled for head in model.probability_heads.values())
    )


def integrated_rank(rows: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    policy = config["selectionPolicy"]
    definitions = {
        "expected_return": ("effective_expected_executable_return", True),
        "strong_gain_probability": ("effective_probability_strong_gain", True),
        "limit_touch_probability": ("effective_probability_limit_touch", True),
        "inverse_non_positive_probability": ("effective_probability_non_positive", False),
        "inverse_net_loss_probability": ("effective_probability_net_loss", False),
        "inverse_severe_loss_probability": ("effective_probability_severe_loss", False),
    }
    result = pd.Series(0.0, index=rows.index)
    for key, weight in policy["integratedRankingWeights"].items():
        column, higher_is_better = definitions[key]
        percentile = rows.groupby("date")[column].rank(
            pct=True, ascending=higher_is_better
        )
        result = result.add(percentile * float(weight), fill_value=0.0)
    return result


def select_rows(
    rows: pd.DataFrame,
    config: dict[str, Any],
    model_ready: bool,
) -> pd.DataFrame:
    if rows.empty or not model_ready:
        return rows.iloc[0:0].copy()
    policy = config["selectionPolicy"]
    selected = rows[
        rows["effective_expected_executable_return"].ge(
            float(policy["minimumExpectedExecutableReturn"])
        )
        & rows["effective_probability_non_positive"].le(
            float(policy["maximumNonPositiveProbability"])
        )
        & rows["effective_probability_net_loss"].le(
            float(policy["maximumNetLossProbability"])
        )
        & rows["effective_probability_strong_gain"].ge(
            float(policy["minimumStrongGainProbability"])
        )
        & rows["effective_probability_limit_touch"].ge(
            float(policy["minimumLimitTouchProbability"])
        )
        & rows["effective_probability_severe_loss"].le(
            float(policy["maximumSevereLossProbability"])
        )
        & rows["effective_predicted_tenth_percentile_return"].ge(
            float(policy["minimumPredictedTenthPercentileReturn"])
        )
    ].copy()
    if selected.empty:
        return selected
    selected["integrated_rank_score"] = integrated_rank(selected, config)
    selected = selected.sort_values(
        ["date", "integrated_rank_score", "effective_expected_executable_return"],
        ascending=[True, False, False],
    )
    return selected.groupby("date", sort=False).head(
        int(policy["maximumSelectionsPerDay"])
    )


def diagnostic_rows(rows: pd.DataFrame, config: dict[str, Any], count: int = 10) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    output = rows.copy()
    output["integrated_rank_score"] = integrated_rank(output, config)
    return (
        output.sort_values(
            ["date", "integrated_rank_score", "effective_expected_executable_return"],
            ascending=[True, False, False],
        )
        .groupby("date", sort=False)
        .head(count)
    )


def outcome_summary(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    valid = rows[rows["target_executable_return"].notna()].copy()
    if valid.empty:
        return {
            "observations": 0,
            "signalDays": 0,
            "meanGrossReturn": None,
            "meanNetReturn": None,
            "winRate": None,
            "nonPositiveRate": None,
            "netLossRate": None,
            "strongGainRate": None,
            "limitTouchRate": None,
            "severeLossRate": None,
            "cvar10": None,
        }
    values = valid["target_executable_return"].astype(float)
    tail_count = max(1, int(math.ceil(len(values) * 0.10)))
    return {
        "observations": len(valid),
        "signalDays": int(valid["date"].nunique()),
        "meanGrossReturn": round(float(values.mean()), 8),
        "meanNetReturn": round(float(values.mean() - config["data"]["roundTripCost"]), 8),
        "medianGrossReturn": round(float(values.median()), 8),
        "winRate": round(float(values.gt(0.0).mean()), 8),
        "nonPositiveRate": round(float(values.le(0.0).mean()), 8),
        "netLossRate": round(
            float(values.le(float(config["data"]["roundTripCost"])).mean()), 8
        ),
        "strongGainRate": round(float(valid["label_strong_gain"].mean()), 8),
        "limitTouchRate": round(float(valid["label_limit_touch"].mean()), 8),
        "severeLossRate": round(float(valid["label_severe_loss"].mean()), 8),
        "cvar10": round(float(values.nsmallest(tail_count).mean()), 8),
    }


def period_report(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
    model_ready: bool,
) -> dict[str, Any]:
    rows = predictions[
        predictions["date"].isin(dates)
        & predictions["target_executable_return"].notna()
    ].copy()
    selected = select_rows(rows, config, model_ready)
    diagnostic = diagnostic_rows(rows, config, 10)
    mapping = {
        "strong_gain": "label_strong_gain",
        "limit_touch": "label_limit_touch",
        "non_positive": "label_non_positive",
        "net_loss": "label_net_loss",
        "severe_loss": "label_severe_loss",
    }
    heads: dict[str, Any] = {}
    for name, label in mapping.items():
        heads[name] = {
            "effective": v1.probability_metrics(
                rows[label], rows[f"effective_probability_{name}"]
            ),
            "raw": v1.probability_metrics(
                rows[label], rows[f"raw_probability_{name}"]
            ),
        }
    return {
        "candidateRows": len(rows),
        "probabilityHeads": heads,
        "candidatePoolOutcome": outcome_summary(rows, config),
        "diagnosticTop10Outcome": outcome_summary(diagnostic, config),
        "policySelectionOutcome": outcome_summary(selected, config),
    }


def build_verdict(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["evaluation"]
    periods: dict[str, Any] = {}
    for name in ("validation", "shadow"):
        selected = report["periods"][name]["policySelectionOutcome"]
        checks = {
            "minimumSignalDays": selected["signalDays"] >= int(gate["minimumSignalDaysPerPeriod"]),
            "minimumObservations": selected["observations"]
            >= int(gate["minimumSelectedObservationsPerPeriod"]),
            "grossReturn": selected["meanGrossReturn"] is not None
            and selected["meanGrossReturn"] >= float(gate["minimumGrossMeanExecutableReturn"]),
            "netReturn": selected["meanNetReturn"] is not None
            and selected["meanNetReturn"] >= float(gate["minimumNetMeanExecutableReturn"]),
            "winRate": selected["winRate"] is not None
            and selected["winRate"] >= float(gate["minimumBasketWinRate"]),
            "nonPositiveRate": selected["nonPositiveRate"] is not None
            and selected["nonPositiveRate"] <= float(gate["maximumObservedNonPositiveRate"]),
            "severeLossRate": selected["severeLossRate"] is not None
            and selected["severeLossRate"] <= float(gate["maximumObservedSevereLossRate"]),
        }
        if gate.get("mustImproveGrossReturnAndNonPositiveRateVsV1Diagnostic"):
            reference = report["referenceV1"]["periods"].get(name, {})
            checks["grossReturnImprovesVsV1"] = (
                selected["meanGrossReturn"] is not None
                and reference.get("meanGrossReturn") is not None
                and selected["meanGrossReturn"] > reference["meanGrossReturn"]
            )
            checks["nonPositiveRateImprovesVsV1"] = (
                selected["nonPositiveRate"] is not None
                and reference.get("nonPositiveRate") is not None
                and selected["nonPositiveRate"] < reference["nonPositiveRate"]
            )
        periods[name] = {"checks": checks, "passed": all(checks.values())}
    stable = bool(
        report["modelReady"]
        and report["candidateRecall"]["gatePassed"]
        and all(item["passed"] for item in periods.values())
    )
    return {
        "status": "research_only_not_eligible_for_trading",
        "periods": periods,
        "stableHistoricalHypothesis": stable,
        "decision": (
            "fresh_forward_preregistration_still_required"
            if stable
            else "reject_for_trading_keep_diagnostics"
        ),
        "historicalRunCanPromote": False,
    }


def _screen_dict(item: FactorScreen) -> dict[str, Any]:
    def rounded(value: float | None, digits: int) -> float | None:
        if value is None or not math.isfinite(float(value)):
            return None
        return round(float(value), digits)

    return {
        "factorKey": item.factor_key,
        "direction": item.direction,
        "orientationIcMean": rounded(item.orientation_ic_mean, 8),
        "orientationIcHacT": rounded(item.orientation_ic_hac_t, 4),
        "confirmationIcMean": rounded(item.confirmation_ic_mean, 8),
        "confirmationIcHacT": rounded(item.confirmation_ic_hac_t, 4),
        "confirmationPValue": rounded(item.confirmation_p_value, 10),
        "confirmationQValue": rounded(item.confirmation_q_value, 10),
        "positiveFolds": item.positive_folds,
        "grossExcessBps": rounded(item.gross_excess_bps, 4),
        "nonPositiveImprovement": rounded(item.non_positive_improvement, 8),
        "severeLossImprovement": rounded(item.severe_loss_improvement, 8),
        "turnover": rounded(item.turnover, 8),
        "turnoverEfficiency": rounded(item.turnover_efficiency, 8),
        "usableDays": item.usable_days,
        "selectionScore": rounded(item.selection_score, 8),
        "selected": item.selected,
        "rejectionReasons": list(item.rejection_reasons),
    }


def _screen_from_dict(item: dict[str, Any]) -> FactorScreen:
    return FactorScreen(
        factor_key=str(item["factorKey"]),
        direction=float(item["direction"]),
        orientation_ic_mean=float(item["orientationIcMean"]),
        orientation_ic_hac_t=float(item["orientationIcHacT"])
        if item.get("orientationIcHacT") is not None
        else None,
        confirmation_ic_mean=float(item["confirmationIcMean"]),
        confirmation_ic_hac_t=float(item["confirmationIcHacT"])
        if item.get("confirmationIcHacT") is not None
        else None,
        confirmation_p_value=float(item["confirmationPValue"]),
        confirmation_q_value=float(item["confirmationQValue"])
        if item.get("confirmationQValue") is not None
        else None,
        positive_folds=int(item["positiveFolds"]),
        gross_excess_bps=float(item["grossExcessBps"])
        if item.get("grossExcessBps") is not None
        else float("nan"),
        non_positive_improvement=float(item["nonPositiveImprovement"])
        if item.get("nonPositiveImprovement") is not None
        else float("nan"),
        severe_loss_improvement=float(item["severeLossImprovement"])
        if item.get("severeLossImprovement") is not None
        else float("nan"),
        turnover=float(item["turnover"])
        if item.get("turnover") is not None
        else float("nan"),
        turnover_efficiency=float(item["turnoverEfficiency"])
        if item.get("turnoverEfficiency") is not None
        else float("nan"),
        usable_days=int(item["usableDays"]),
        selection_score=float(item["selectionScore"])
        if item.get("selectionScore") is not None
        else None,
        selected=bool(item.get("selected", False)),
        rejection_reasons=tuple(map(str, item.get("rejectionReasons", []))),
    )


def screen_checkpoint_path(
    panel: dict[str, Any], config: dict[str, Any]
) -> Path:
    checkpoint_key = _digest(
        {
            "config": config,
            "dataEnd": panel["close"].index.max().date().isoformat(),
            "symbols": panel["close"].shape[1],
        }
    )[:16]
    return (
        ROOT
        / config["output"]["root"]
        / "_screen_checkpoints"
        / f"screen_{checkpoint_key}.json"
    )


def load_v1_reference(config: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / config["evaluation"].get(
        "referenceV1Summary",
        "outputs/edge_research/perception_xalpha_nextday_explosion_v1/"
        "run_20260806_nextday_explosion_v1/summary.json",
    )
    if not path.exists():
        return {"available": False, "path": str(path), "periods": {}}
    payload = load_json(path)
    periods: dict[str, Any] = {}
    for name in ("validation", "shadow"):
        item = payload.get("periods", {}).get(name, {}).get(
            "rawDiagnosticTop3Outcome", {}
        )
        periods[name] = {
            "meanGrossReturn": item.get("meanGrossReturn"),
            "nonPositiveRate": item.get("nonPositiveRate"),
            "observations": item.get("observations"),
        }
    return {"available": True, "path": str(path), "periods": periods}


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Next-session factor-zoo V2",
        "",
        f"- Status: `{report['status']}`",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        f"- Zoo: {report['factorDiscovery']['usable']} usable / {report['factorDiscovery']['declared']} declared",
        f"- Train-only survivors before de-correlation: {report['factorDiscovery']['gateSurvivors']}",
        f"- Selected decorrelated factors: {report['factorDiscovery']['selectedCount']}",
        f"- Model ready: `{report['modelReady']}`",
        f"- Candidate recall gate: `{report['candidateRecall']['gatePassed']}`",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Orders: `[]`",
        "",
        "## Selected train-only factors",
        "",
        "| Factor | Direction | IC t | Gross excess (bps) | Loss improvement | Severe improvement | Turnover |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["factorDiscovery"]["selectedFactors"]:
        lines.append(
            f"| {item['factorKey']} | {item['direction']} | {item['confirmationIcHacT']} | "
            f"{item['grossExcessBps']} | {item['nonPositiveImprovement']} | "
            f"{item['severeLossImprovement']} | {item['turnover']} |"
        )
    lines.extend(
        [
            "",
            "## External diagnostics",
            "",
            "| Period | Selected obs | Gross mean | Net mean | Win | Non-positive | Severe loss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("validation", "shadow"):
        item = report["periods"][name]["policySelectionOutcome"]
        lines.append(
            f"| {name} | {item['observations']} | {item['meanGrossReturn']} | "
            f"{item['meanNetReturn']} | {item['winRate']} | {item['nonPositiveRate']} | "
            f"{item['severeLossRate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Validation and shadow are reject-only because those windows have been viewed before.",
            "- Top10/Top3/Top1 are empty unless every reliability and risk gate passes.",
            "- A probability estimate is not a guarantee of profit or a limit event.",
            "- This run cannot alter trading, sizing, positions, risk gates or execution locks.",
        ]
    )
    return "\n".join(lines) + "\n"


def _name_map(base: dict[str, Any]) -> dict[str, str]:
    path = ROOT / base["assetUniverse"]["masterPath"]
    names: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        names[str(row.get("securityId"))] = str(row.get("name") or "")
    return names


def run(
    config_path: Path,
    run_id: str | None = None,
    reuse_screen_checkpoint: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    base = load_json(ROOT / config["baseResearchConfig"])
    _, cog_config = v1.perception.load_base_configs(base)
    panel, panel_audit = v1.perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    factor_config = load_json(ROOT / config["baseFactorConfig"])
    v1.two_stage.validate_config(factor_config)
    feature_proxy = {"featureSet": {"columns": config["featureSet"]["baseColumns"]}}
    features, market = v1.build_past_only_features(panel, factor_config, feature_proxy)
    outcomes = v1.build_outcome_frames(panel, config)
    target = outcomes["target_executable_return"]
    outcomes["label_net_loss"] = target.le(float(config["data"]["roundTripCost"])).where(
        target.notna()
    )
    split = autonomous.make_split(panel["close"].index, cog_config)
    train_dates = pd.DatetimeIndex(split.train[split.train].index)
    validation_dates = pd.DatetimeIndex(split.validation[split.validation].index)
    shadow_dates = pd.DatetimeIndex(split.shadow[split.shadow].index)
    checkpoint_path = screen_checkpoint_path(panel, config)
    if reuse_screen_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"requested exact screen checkpoint is missing: {checkpoint_path}"
            )
        checkpoint = load_json(checkpoint_path)
        if checkpoint.get("configSha256") != _digest(config):
            raise RuntimeError("screen checkpoint config hash mismatch")
        if checkpoint.get("dataEnd") != panel["close"].index.max().date().isoformat():
            raise RuntimeError("screen checkpoint data range mismatch")
        screens = [_screen_from_dict(item) for item in checkpoint["screens"]]
        selected = [_screen_from_dict(item) for item in checkpoint["selectedFactors"]]
        causality_audit = checkpoint["causalityAudit"]
        failures = {"checkpointReused": 1}
        print(
            f"factor_screen_checkpoint_reused=true usable={len(screens)} selected={len(selected)}",
            flush=True,
        )
    else:
        screens, selected, _, failures = discover_factors(
            panel, outcomes, train_dates, config
        )
        print(
            f"factor_screen_complete usable={len(screens)} pre_correlation_selected={len(selected)}",
            flush=True,
        )
        selected, causality_audit = audit_selected_prefix_causality(
            panel, selected, train_dates, config
        )
        print(
            f"causality_audit_complete selected={len(selected)} failed={len(causality_audit['failed'])}",
            flush=True,
        )
        causal_keys = {item.factor_key for item in selected}
        screens = [
            _replace_screen(item, selected=item.factor_key in causal_keys)
            for item in screens
        ]
    selected_dicts = [_screen_dict(item) for item in selected]
    print(
        "selected_factor_keys=" + ",".join(item.factor_key for item in selected),
        flush=True,
    )
    atomic_write(
        checkpoint_path,
        json.dumps(
            json_safe(
                {
                    "status": "research_only_train_only_checkpoint",
                    "configSha256": _digest(config),
                    "dataEnd": panel["close"].index.max().date().isoformat(),
                    "selectedFactors": selected_dicts,
                    "screens": [_screen_dict(item) for item in screens],
                    "causalityAudit": causality_audit,
                    "orders": [],
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    selected_features, discovered_score, factor_weights = compute_selected_factor_features(
        panel, selected, config
    )
    features.update(selected_features)
    legacy_score = legacy_candidate_score(features, panel, config)
    candidate_score, candidate_rank, candidate_mask = combined_candidate_score(
        legacy_score, discovered_score, panel, config
    )
    recall = v1.candidate_recall(candidate_mask, outcomes, train_dates, config)
    dynamic_config = copy.deepcopy(config)
    dynamic_columns = list(config["featureSet"]["baseColumns"]) + list(selected_features)
    dynamic_config["featureSet"]["columns"] = dynamic_columns
    table = v1.build_candidate_table(
        panel,
        features,
        market,
        outcomes,
        candidate_score,
        candidate_rank,
        candidate_mask,
        dynamic_config,
    )
    model: v1.ExplosionModel | None = None
    model_audit: dict[str, Any] = {"status": "not_fit_no_selected_factors"}
    if selected:
        print("nested_model_fit_started=true", flush=True)
        model = v1.fit_model(table, train_dates, dynamic_config)
        model_audit = model.audit
        print("nested_model_fit_complete=true", flush=True)
    model_ready = bool(
        model is not None
        and all_heads_ready(model)
        and recall["gatePassed"]
        and len(selected) > 0
    )
    external = table[
        table["date"].isin(validation_dates.union(shadow_dates))
        | table["date"].eq(table["date"].max())
    ].copy()
    predictions = v1.score_rows(model, external) if model is not None else external.copy()
    if model is None:
        for name in config["model"]["probabilityHeads"]:
            predictions[f"raw_probability_{name}"] = np.nan
            predictions[f"effective_probability_{name}"] = np.nan
        predictions["raw_expected_executable_return"] = np.nan
        predictions["calibrated_expected_executable_return"] = np.nan
        predictions["effective_expected_executable_return"] = np.nan
        predictions["raw_predicted_tenth_percentile_return"] = np.nan
        predictions["effective_predicted_tenth_percentile_return"] = np.nan
    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_" + now.strftime("%Y%m%dT%H%M%SZ") + "_" + _digest(config)[:10]
    )
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": now.isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": _digest(config),
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "splitAudit": split.audit,
        "panelAudit": panel_audit,
        "factorDiscovery": {
            "declared": int(config["factorDiscovery"]["declaredLibrarySize"]),
            "usable": len(screens),
            "gateSurvivors": sum(not item.rejection_reasons for item in screens),
            "selectedCount": len(selected),
            "selectedFactors": selected_dicts,
            "selectedTrainOnlyWeights": factor_weights,
            "selectedFactorPrefixCausality": causality_audit,
            "failures": failures,
            "validationRowsReadDuringDiscovery": 0,
            "shadowRowsReadDuringDiscovery": 0,
        },
        "candidateRows": len(table),
        "candidateRecall": recall,
        "featureColumns": dynamic_columns,
        "modelAudit": model_audit,
        "modelReady": model_ready,
        "referenceV1": load_v1_reference(config),
        "periods": {
            "validation": period_report(
                predictions, validation_dates, dynamic_config, model_ready
            ),
            "shadow": period_report(
                predictions, shadow_dates, dynamic_config, model_ready
            ),
        },
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    report["verdict"] = build_verdict(report, config)
    output_root = ROOT / config["output"]["root"] / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    screen_rows = [_screen_dict(item) for item in screens]
    atomic_write(
        output_root / "factor_screen.json",
        json.dumps(
            json_safe({
                "status": "research_only_train_only_factor_screen",
                "declaredTrials": int(config["factorDiscovery"]["declaredLibrarySize"]),
                "factors": screen_rows,
                "orders": [],
            }),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    pd.DataFrame(screen_rows).to_csv(output_root / "factor_screen.csv", index=False)
    latest_date = table["date"].max()
    latest = predictions[predictions["date"].eq(latest_date)].copy()
    latest_diagnostic = diagnostic_rows(latest, dynamic_config, 10)
    latest_selected = select_rows(latest, dynamic_config, model_ready)
    names = _name_map(base)
    for frame in (latest_diagnostic, latest_selected):
        frame.insert(2, "name", frame["securityId"].map(names).fillna(""))
    columns = [
        "date",
        "securityId",
        "name",
        "candidate_rank",
        "candidate_score",
        "integrated_rank_score",
        "calibrated_expected_executable_return",
        "effective_expected_executable_return",
        "effective_probability_strong_gain",
        "effective_probability_limit_touch",
        "effective_probability_non_positive",
        "effective_probability_net_loss",
        "effective_probability_severe_loss",
        "effective_predicted_tenth_percentile_return",
    ]
    for frame, filename in (
        (latest_diagnostic, "latest_diagnostic_top10.csv"),
        (latest_selected, "latest_selected_shadow.csv"),
    ):
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
        frame[columns].to_csv(output_root / filename, index=False)
    atomic_write(
        output_root / "summary.json",
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write(output_root / "report.md", markdown_report(report))
    atomic_write(
        output_root / "model_manifest.json",
        json.dumps(
            json_safe({
                "schemaVersion": "perception_xalpha_nextday_factor_zoo_model_manifest_v2",
                "status": "research_only_not_online_inference",
                "runId": run_id,
                "modelReady": model_ready,
                "selectedFactors": selected_dicts,
                "selectionPolicy": config["selectionPolicy"],
                "orders": [],
                "automaticTradingChanges": [],
            }),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reuse-screen-checkpoint",
        action="store_true",
        help="Reuse only an exact config/data-hashed train-only factor-screen checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(
        args.config.resolve(),
        args.run_id,
        reuse_screen_checkpoint=args.reuse_screen_checkpoint,
    )
    print(
        json.dumps(
            {
                "runId": report["runId"],
                "usableFactors": report["factorDiscovery"]["usable"],
                "selectedFactors": report["factorDiscovery"]["selectedCount"],
                "modelReady": report["modelReady"],
                "decision": report["verdict"]["decision"],
                "orders": [],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
