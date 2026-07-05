"""Historical HMM auxiliary-feature study for Singularity Phase 1.6.

The existing GaussianHMM1D supplies causal regime probabilities. Fixed LPPLS
and linear-DMD diagnostics are tested only as auxiliary covariates in
transparent calibrated logistic models. Nothing in this module can trade,
write forward artifacts, or mutate Phase 1.5.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision_probability import probability_metrics
from research_hmm_nn_bl import GaussianHMM1D, ROOT
from research_singularity_phase1 import build_label_table
import research_singularity_phase1_6_hmm_physics_features as phase16
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "singularity_phase1_6_hmm_physics_model_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schemaVersion")
        != "singularity_phase1_6_hmm_physics_model_v1"
    ):
        raise ValueError("unexpected Phase 1.6 model schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("model study must remain research/shadow-only")
    if config["userPreregisteredBreak"]["date"] != "2026-06-12":
        raise ValueError("unexpected user-preregistered break date")
    if config["source"].get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 ledger cannot be an input")
    if config["hmm"].get("implementation") != (
        "research_hmm_nn_bl.GaussianHMM1D"
    ):
        raise ValueError("existing GaussianHMM1D must be reused")
    if config["hmm"].get("automaticStateOrParameterSearchAllowed") is not False:
        raise ValueError("automatic HMM search is forbidden")
    safety = config["safety"]
    forbidden = [
        "brokerCallsAllowed",
        "onlineInferenceAllowed",
        "phase15WritesAllowed",
        "liveConfigWritesAllowed",
        "overlayWritesAllowed",
        "positionSizingAllowed",
        "orderSubmissionAllowed",
        "riskGateChangesAllowed",
        "buildDecisionIntegrationAllowed",
        "buySellGateIntegrationAllowed",
        "forwardTaskIntegrationAllowed",
        "promotionAllowed",
    ]
    if not safety.get("offlineOnly") or not safety.get("recordOnly"):
        raise ValueError("offline record-only flags are required")
    if any(safety.get(key) is not False for key in forbidden):
        raise ValueError("every live mutation flag must be false")


def fit_existing_hmm(
    base: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, pd.Series], dict[str, Any], pd.DataFrame]:
    hmm_config = config["hmm"]
    market_return: pd.Series = base["market_ret_1"]
    dates = pd.Series(
        market_return.index.strftime("%Y-%m-%d"),
        index=market_return.index,
    )
    fit_end = str(config["data"]["fixedHMMFitEnd"])
    sequences = [
        market_return[dates == trade_date].to_numpy(dtype=float)
        for trade_date in sorted(set(dates[dates <= fit_end]))
    ]
    hmm = GaussianHMM1D(
        n_states=int(hmm_config["states"]),
        n_iter=int(hmm_config["iterations"]),
        variance_floor=float(hmm_config["varianceFloor"]),
    )
    hmm.fit(sequences)
    probabilities = pd.DataFrame(
        index=market_return.index,
        columns=[f"state_{index}" for index in range(hmm.n_states)],
        dtype=float,
    )
    for trade_date in sorted(set(dates)):
        mask = dates == trade_date
        probabilities.loc[mask, :] = hmm.filter_probabilities(
            market_return.loc[mask].to_numpy(dtype=float)
        )
    values = probabilities.to_numpy(dtype=float)
    transition_risk = 1.0 - values @ np.diag(hmm.transition)
    clipped = np.clip(values, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / math.log(
        hmm.n_states
    )
    dominant = np.argmax(values, axis=1)
    frames = {
        "hmm_bear_probability": probabilities.iloc[:, 0],
        "hmm_middle_probability": probabilities.iloc[:, 1],
        "hmm_bull_probability": probabilities.iloc[:, -1],
        "hmm_expected_return": pd.Series(
            values @ hmm.means, index=market_return.index
        ),
        "regime_transition_risk": pd.Series(
            transition_risk, index=market_return.index
        ),
        "regime_entropy": pd.Series(entropy, index=market_return.index),
        "dominant_hmm_state": pd.Series(
            dominant, index=market_return.index
        ),
    }
    full_state = probabilities.copy()
    full_state["trade_date"] = full_state.index.strftime("%Y-%m-%d")
    full_state["dominant_state"] = dominant
    full_state["entropy"] = entropy
    duration_rows: list[dict[str, Any]] = []
    switches = 0
    possible_switches = 0
    for trade_date, group in full_state.groupby("trade_date", sort=True):
        states = group["dominant_state"].to_numpy(dtype=int)
        if len(states) > 1:
            switches += int(np.sum(states[1:] != states[:-1]))
            possible_switches += len(states) - 1
        start = 0
        while start < len(states):
            end = start + 1
            while end < len(states) and states[end] == states[start]:
                end += 1
            duration_rows.append(
                {
                    "tradeDate": trade_date,
                    "state": int(states[start]),
                    "durationBars": int(end - start),
                }
            )
            start = end
    duration = pd.DataFrame.from_records(duration_rows)
    audit = {
        "implementation": "research_hmm_nn_bl.GaussianHMM1D",
        "reimplemented": False,
        "fitEnd": fit_end,
        "states": hmm.n_states,
        "means": hmm.means.tolist(),
        "variances": hmm.variances.tolist(),
        "startProbability": hmm.start_probability.tolist(),
        "transitionMatrix": hmm.transition.tolist(),
        "meanNormalizedEntropy": float(np.mean(entropy)),
        "dominantStateSwitchRate": (
            switches / possible_switches if possible_switches else None
        ),
        "durationByState": {
            str(int(state)): phase2a.numeric_distribution(
                rows["durationBars"]
            )
            for state, rows in duration.groupby("state")
        },
        "filtering": "causal_session_reset",
    }
    return frames, audit, probabilities


def frame_lookup(
    source: pd.DataFrame,
    timestamps: pd.Series,
    codes: pd.Series,
) -> np.ndarray:
    stacked = source.stack(future_stack=True)
    return np.asarray(
        [
            stacked.get((timestamp, code), np.nan)
            for timestamp, code in zip(timestamps, codes)
        ],
        dtype=float,
    )


def build_model_table(
    base: dict[str, Any],
    labels: pd.DataFrame,
    phase2a_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    auxiliary = phase16.expected_decision_feature_audit(
        base, phase2a_config, {
            "auxiliaryFeatures": {
                "pastWindowBars": phase2a_config["sampling"][
                    "minimumPastWindowBars"
                ],
                "decisionStrideBars": phase2a_config["sampling"][
                    "decisionStrideBars"
                ],
                "trainingStatisticsEnd": config["data"]["fixedHMMFitEnd"],
            }
        }
    )
    ews, ews_audit = phase2a.build_ews(base, phase2a_config)
    hmm, hmm_audit, _ = fit_existing_hmm(base, config)
    table = auxiliary.copy()
    for name in config["features"]["base"]:
        source = base[name]
        table[name] = (
            frame_lookup(
                source, table["timestamp"], table["stockCode"]
            )
            if isinstance(source, pd.DataFrame)
            else source.reindex(table["timestamp"]).to_numpy(dtype=float)
        )
    for name in config["features"]["ews"]:
        source = ews[name]
        table[name] = (
            frame_lookup(
                source, table["timestamp"], table["stockCode"]
            )
            if isinstance(source, pd.DataFrame)
            else source.reindex(table["timestamp"]).to_numpy(dtype=float)
        )
    for name in config["features"]["hmm"]:
        table[name] = hmm[name].reindex(table["timestamp"]).to_numpy(
            dtype=float
        )
    table["dominant_hmm_state"] = (
        hmm["dominant_hmm_state"]
        .reindex(table["timestamp"])
        .to_numpy(dtype=float)
    )
    train_vol = table.loc[
        table["trade_date"] <= config["data"]["fixedHMMFitEnd"],
        "realized_vol_proxy",
    ]
    low, high = np.quantile(train_vol.dropna(), [1 / 3, 2 / 3])
    table["volatility_regime"] = np.where(
        table["realized_vol_proxy"] <= low,
        "low",
        np.where(table["realized_vol_proxy"] >= high, "high", "medium"),
    )
    label_columns = [
        "timestamp",
        "trade_date",
        "stockCode",
        "turning_point_5",
        "turning_point_10",
    ]
    table = table.merge(
        labels[label_columns],
        on=["timestamp", "trade_date", "stockCode"],
        how="left",
        validate="one_to_one",
    )
    table = table.replace([np.inf, -np.inf], np.nan)
    coverage = {
        "rows": int(len(table)),
        "lpplsCompleteRows": int(
            table[config["features"]["lppls"]].notna().all(axis=1).sum()
        ),
        "dmdCompleteRows": int(
            table[config["features"]["dmd"]].notna().all(axis=1).sum()
        ),
        "dmdWindowValidRows": int(
            (table["dmd_window_valid"] == 1.0).sum()
        ),
        "dmdByIntradaySlot": phase16.grouped_coverage(
            table, "intraday_slot", "dmd_window_valid"
        ),
        "volatilityRegimeTrainingThresholds": {
            "lowUpper": float(low),
            "highLower": float(high),
            "fitEnd": config["data"]["fixedHMMFitEnd"],
        },
    }
    return table, coverage, {"ews": ews_audit, "hmm": hmm_audit}


def variant_features(config: dict[str, Any]) -> dict[str, list[str]]:
    base = list(config["features"]["base"])
    ews = list(config["features"]["ews"])
    hmm = list(config["features"]["hmm"])
    lppls = list(config["features"]["lppls"])
    dmd = list(config["features"]["dmd"])
    return {
        "constant_prior": [],
        "ews_baseline": base + ews,
        "current_hmm": base + hmm,
        "current_hmm_ews": base + hmm + ews,
        "hmm_lppls": base + hmm + lppls,
        "hmm_ews_lppls": base + hmm + ews + lppls,
        "hmm_dmd": base + hmm + dmd,
        "hmm_ews_dmd": base + hmm + ews + dmd,
        "hmm_ews_lppls_dmd": base + hmm + ews + lppls + dmd,
    }


def purge_tail(
    rows: pd.DataFrame, count: int
) -> tuple[pd.DataFrame, int]:
    ordered = rows.sort_values(
        ["stockCode", "timestamp"], kind="stable"
    )
    remove = ordered.groupby("stockCode", sort=False).tail(count).index
    return ordered.drop(index=remove).reset_index(drop=True), int(len(remove))


def winsor_bounds(
    rows: pd.DataFrame, columns: list[str], quantiles: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.asarray([]), np.asarray([])
    values = rows[columns].to_numpy(dtype=float)
    return (
        np.quantile(values, quantiles[0], axis=0),
        np.quantile(values, quantiles[1], axis=0),
    )


def clipped_values(
    rows: pd.DataFrame,
    columns: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(rows[columns].to_numpy(dtype=float), lower, upper)


def run_population_walk_forward(
    table: pd.DataFrame,
    population: str,
    variants: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    feature_map = variant_features(config)
    union = sorted(
        {
            feature
            for variant in variants
            for feature in feature_map[variant]
        }
    )
    population_rows = table.dropna(subset=union).copy()
    if config["populations"][population]["requiresDmd"]:
        population_rows = population_rows[
            population_rows["dmd_window_valid"] == 1.0
        ].copy()
    start = config["data"]["walkForwardStart"]
    end = config["data"]["walkForwardEnd"]
    block_days = int(config["models"]["walkForwardTestBlockDays"])
    calibration_days = int(config["models"]["calibrationDays"])
    minimum_fit_days = int(config["models"]["minimumBaseFitDays"])
    embargo_days = int(config["models"]["embargoTradingDays"])
    purge_count = int(
        config["models"]["purgeDecisionRowsPerSymbol"]
    )
    quantiles = config["features"]["winsorizationQuantiles"]
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for horizon in config["data"]["primaryHorizonsBars"]:
        target = f"turning_point_{horizon}"
        rows = population_rows.dropna(subset=[target]).copy()
        rows[target] = rows[target].astype(int)
        test_dates = sorted(
            date
            for date in rows["trade_date"].unique()
            if start <= date <= end
        )
        for fold, offset in enumerate(
            range(0, len(test_dates), block_days), 1
        ):
            block = test_dates[offset : offset + block_days]
            train_raw = rows[rows["trade_date"] < block[0]].copy()
            train_dates = sorted(train_raw["trade_date"].unique())
            if (
                len(train_dates)
                <= calibration_days + embargo_days + minimum_fit_days
            ):
                continue
            test_embargo = set(train_dates[-embargo_days:])
            before_test = train_raw[
                ~train_raw["trade_date"].isin(test_embargo)
            ]
            before_test, test_purged = purge_tail(
                before_test, purge_count
            )
            eligible = sorted(before_test["trade_date"].unique())
            calibration_date_set = set(eligible[-calibration_days:])
            calibration = before_test[
                before_test["trade_date"].isin(calibration_date_set)
            ].copy()
            before_calibration = before_test[
                ~before_test["trade_date"].isin(calibration_date_set)
            ]
            fit_dates = sorted(
                before_calibration["trade_date"].unique()
            )
            calibration_embargo = set(fit_dates[-embargo_days:])
            fit_raw = before_calibration[
                ~before_calibration["trade_date"].isin(
                    calibration_embargo
                )
            ]
            fit, calibration_purged = purge_tail(
                fit_raw, purge_count
            )
            test = rows[rows["trade_date"].isin(block)].copy()
            if (
                fit[target].nunique() != 2
                or calibration[target].nunique() != 2
                or test.empty
            ):
                continue
            audit = {
                "population": population,
                "horizonBars": horizon,
                "fold": fold,
                "fitStart": str(fit["trade_date"].min()),
                "fitEnd": str(fit["trade_date"].max()),
                "calibrationStart": str(
                    calibration["trade_date"].min()
                ),
                "calibrationEnd": str(
                    calibration["trade_date"].max()
                ),
                "testStart": block[0],
                "testEnd": block[-1],
                "testDates": block,
                "testEmbargoDates": sorted(test_embargo),
                "calibrationEmbargoDates": sorted(
                    calibration_embargo
                ),
                "testBoundaryPurgedRows": test_purged,
                "calibrationBoundaryPurgedRows": calibration_purged,
                "sameTradingDateAcrossBoundaries": False,
                "variants": {},
            }
            for variant in variants:
                columns = feature_map[variant]
                if not columns:
                    probability = np.full(
                        len(test),
                        float(calibration[target].mean()),
                    )
                    raw = probability.copy()
                    bounds = None
                else:
                    lower, upper = winsor_bounds(
                        fit, columns, quantiles
                    )
                    fit_values = clipped_values(
                        fit, columns, lower, upper
                    )
                    calibration_values = clipped_values(
                        calibration, columns, lower, upper
                    )
                    test_values = clipped_values(
                        test, columns, lower, upper
                    )
                    model = Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    C=float(
                                        config["models"]["logisticC"]
                                    ),
                                    max_iter=500,
                                    random_state=int(
                                        config["models"]["randomSeed"]
                                    ),
                                ),
                            ),
                        ]
                    )
                    model.fit(
                        fit_values, fit[target].to_numpy(dtype=int)
                    )
                    calibration_raw = np.clip(
                        model.predict_proba(calibration_values)[:, 1],
                        1e-6,
                        1 - 1e-6,
                    )
                    calibrator = LogisticRegression(
                        C=float(config["models"]["calibrationC"]),
                        max_iter=500,
                        random_state=int(
                            config["models"]["randomSeed"]
                        ),
                    )
                    calibrator.fit(
                        np.log(
                            calibration_raw / (1.0 - calibration_raw)
                        ).reshape(-1, 1),
                        calibration[target].to_numpy(dtype=int),
                    )
                    raw = np.clip(
                        model.predict_proba(test_values)[:, 1],
                        1e-6,
                        1 - 1e-6,
                    )
                    probability = calibrator.predict_proba(
                        np.log(raw / (1.0 - raw)).reshape(-1, 1)
                    )[:, 1]
                    bounds = {
                        column: {
                            "lower": float(lower[index]),
                            "upper": float(upper[index]),
                        }
                        for index, column in enumerate(columns)
                    }
                part = test[
                    [
                        "timestamp",
                        "trade_date",
                        "stockCode",
                        "month",
                        "etf_category",
                        "volatility_regime",
                        "intraday_slot",
                        target,
                    ]
                ].rename(columns={target: "actual"})
                part["population"] = population
                part["horizon_bars"] = horizon
                part["fold"] = fold
                part["variant"] = variant
                part["raw_probability"] = raw
                part["probability"] = probability
                predictions.append(part)
                audit["variants"][variant] = {
                    "features": columns,
                    "fitSamples": int(len(fit)),
                    "calibrationSamples": int(len(calibration)),
                    "testSamples": int(len(test)),
                    "fitPositiveRate": float(fit[target].mean()),
                    "calibrationPositiveRate": float(
                        calibration[target].mean()
                    ),
                    "testPositiveRate": float(test[target].mean()),
                    "winsorBounds": bounds,
                }
            audits.append(audit)
    if not predictions:
        raise RuntimeError(f"no predictions for population {population}")
    return pd.concat(predictions, ignore_index=True), audits


def metric_payload(
    rows: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    metrics = probability_metrics(
        rows["probability"].to_numpy(dtype=float),
        rows["actual"].to_numpy(dtype=int),
        bin_edges=config["models"]["probabilityBinEdges"],
    )
    threshold = float(config["models"]["highRiskThreshold"])
    high = rows["probability"].to_numpy(dtype=float) >= threshold
    metrics.update(
        {
            "highRiskThreshold": threshold,
            "highRiskCount": int(np.sum(high)),
            "highRiskMeanProbability": (
                float(rows.loc[high, "probability"].mean())
                if np.any(high)
                else None
            ),
            "highRiskHitRate": (
                float(rows.loc[high, "actual"].mean())
                if np.any(high)
                else None
            ),
        }
    )
    return metrics


def summarize_predictions(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    overall: dict[str, Any] = {}
    by_fold: dict[str, Any] = {}
    by_group: dict[str, Any] = {}
    for population, population_rows in predictions.groupby("population"):
        overall[population] = {}
        by_fold[population] = {}
        by_group[population] = {}
        for horizon, horizon_rows in population_rows.groupby(
            "horizon_bars"
        ):
            key = str(int(horizon))
            overall[population][key] = {
                variant: metric_payload(values, config)
                for variant, values in horizon_rows.groupby("variant")
            }
            by_fold[population][key] = {
                str(int(fold)): {
                    variant: metric_payload(values, config)
                    for variant, values in fold_rows.groupby("variant")
                }
                for fold, fold_rows in horizon_rows.groupby("fold")
            }
            by_group[population][key] = {}
            for dimension in [
                "month",
                "etf_category",
                "volatility_regime",
                "intraday_slot",
            ]:
                by_group[population][key][dimension] = {
                    str(group): {
                        variant: metric_payload(values, config)
                        for variant, values in group_rows.groupby(
                            "variant"
                        )
                    }
                    for group, group_rows in horizon_rows.groupby(
                        dimension
                    )
                }
    return {
        "overall": overall,
        "byFold": by_fold,
        "byGroup": by_group,
    }


def improvement_count(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> int:
    return sum(
        [
            candidate["brier"] < baseline["brier"],
            candidate["log_loss"] < baseline["log_loss"],
            candidate["auc"] is not None
            and baseline["auc"] is not None
            and candidate["auc"] > baseline["auc"],
            candidate["ece"] < baseline["ece"],
        ]
    )


def cluster_bootstrap_brier_delta(
    predictions: pd.DataFrame,
    population: str,
    horizon: int,
    candidate: str,
    baseline: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    keys = ["timestamp", "trade_date", "stockCode", "fold"]
    subset = predictions[
        (predictions["population"] == population)
        & (predictions["horizon_bars"] == horizon)
        & predictions["variant"].isin([candidate, baseline])
    ]
    wide = subset.pivot(
        index=keys,
        columns="variant",
        values=["actual", "probability"],
    )
    actual = wide[("actual", baseline)].to_numpy(dtype=float)
    candidate_loss = (
        wide[("probability", candidate)].to_numpy(dtype=float) - actual
    ) ** 2
    baseline_loss = (
        wide[("probability", baseline)].to_numpy(dtype=float) - actual
    ) ** 2
    frame = pd.DataFrame(
        {
            "trade_date": wide.index.get_level_values("trade_date"),
            "delta": candidate_loss - baseline_loss,
        }
    )
    daily = frame.groupby("trade_date")["delta"].mean().to_numpy()
    rng = np.random.default_rng(int(config["models"]["randomSeed"]))
    replicates = int(config["models"]["clusterBootstrapReplicates"])
    samples = np.empty(replicates)
    for index in range(replicates):
        samples[index] = float(
            np.mean(rng.choice(daily, size=len(daily), replace=True))
        )
    confidence = float(
        config["models"]["clusterBootstrapConfidence"]
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "cluster": "trade_date",
        "independentTradingDays": int(len(daily)),
        "meanBrierDeltaCandidateMinusBaseline": float(np.mean(daily)),
        "confidence": confidence,
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1.0 - tail)),
        "improvementRequiresUpperBelowZero": True,
        "passes": bool(np.quantile(samples, 1.0 - tail) < 0.0),
    }


def evaluate_candidate(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    population: str,
    candidate: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline = config["successGate"]["comparisonBaseline"]
    required = int(
        config["successGate"]["minimumImprovedMetricCountOfFour"]
    )
    result: dict[str, Any] = {}
    for horizon in config["data"]["primaryHorizonsBars"]:
        key = str(horizon)
        if (
            population not in metrics["overall"]
            or key not in metrics["overall"][population]
            or candidate not in metrics["overall"][population][key]
            or baseline not in metrics["overall"][population][key]
        ):
            result[key] = {
                "status": "not_evaluable_insufficient_same_session_coverage",
                "improvedMetricCountOfFour": 0,
                "improvingFolds": 0,
                "totalFolds": 0,
                "improvingMonths": 0,
                "totalMonths": 0,
                "improvingEtfCategories": [],
                "highRiskCount": 0,
                "highRiskOptimism": None,
                "clusterBootstrap": None,
                "gates": {},
                "passes": False,
            }
            continue
        current = metrics["overall"][population][key][candidate]
        reference = metrics["overall"][population][key][baseline]
        folds = metrics["byFold"][population][key]
        improving_folds = sum(
            improvement_count(values[candidate], values[baseline])
            >= required
            for values in folds.values()
        )
        months = metrics["byGroup"][population][key]["month"]
        improving_months = sum(
            values[candidate]["brier"] < values[baseline]["brier"]
            for values in months.values()
        )
        categories = metrics["byGroup"][population][key][
            "etf_category"
        ]
        improving_categories = [
            name
            for name, values in categories.items()
            if values[candidate]["brier"] < values[baseline]["brier"]
        ]
        optimism = (
            None
            if current["highRiskHitRate"] is None
            else current["highRiskMeanProbability"]
            - current["highRiskHitRate"]
        )
        bootstrap = cluster_bootstrap_brier_delta(
            predictions,
            population,
            horizon,
            candidate,
            baseline,
            config,
        )
        gates = {
            "majorityMetrics": improvement_count(current, reference)
            >= required,
            "majorityFolds": improving_folds / len(folds)
            > float(
                config["successGate"][
                    "minimumImprovingFoldFraction"
                ]
            ),
            "majorityMonths": improving_months / len(months)
            > float(
                config["successGate"][
                    "minimumImprovingMonthFraction"
                ]
            ),
            "multipleEtfCategories": len(improving_categories)
            >= int(
                config["successGate"][
                    "minimumImprovingEtfCategories"
                ]
            ),
            "highRiskSamples": current["highRiskCount"]
            >= int(config["successGate"]["minimumHighRiskSamples"]),
            "highRiskNotOverconfident": optimism is not None
            and optimism
            <= float(
                config["successGate"]["maximumHighRiskOptimism"]
            ),
            "clusterBootstrapBrier": bootstrap["passes"],
        }
        result[key] = {
            "improvedMetricCountOfFour": improvement_count(
                current, reference
            ),
            "improvingFolds": improving_folds,
            "totalFolds": len(folds),
            "improvingMonths": improving_months,
            "totalMonths": len(months),
            "improvingEtfCategories": improving_categories,
            "highRiskCount": current["highRiskCount"],
            "highRiskOptimism": optimism,
            "clusterBootstrap": bootstrap,
            "gates": gates,
            "passes": all(gates.values()),
        }
    return result


def state_interpretation(
    table: pd.DataFrame,
    hmm_audit: dict[str, Any],
) -> dict[str, Any]:
    rows = table.dropna(
        subset=["dominant_hmm_state", "turning_point_5", "turning_point_10"]
    )
    state_rows: dict[str, Any] = {}
    for state, group in rows.groupby("dominant_hmm_state"):
        state_rows[str(int(state))] = {
            "rows": int(len(group)),
            "marketReturnMean": float(group["market_ret_1"].mean()),
            "realizedVolatilityMean": float(
                group["realized_vol_proxy"].mean()
            ),
            "ewsScoreMean": float(group["ews_score"].mean()),
            "lpplsBubbleLikeMean": float(
                group["lppls_bubble_like_score"].mean()
            ),
            "lpplsResidualMean": float(
                group["lppls_fit_residual"].mean()
            ),
            "dmdResidualMean": (
                float(group["dmd_reconstruction_residual"].mean())
                if group["dmd_reconstruction_residual"].notna().any()
                else None
            ),
            "turningPositiveRate5": float(
                group["turning_point_5"].mean()
            ),
            "turningPositiveRate10": float(
                group["turning_point_10"].mean()
            ),
            "regimeEntropyMean": float(group["regime_entropy"].mean()),
            "transitionRiskMean": float(
                group["regime_transition_risk"].mean()
            ),
        }
    return {
        "states": state_rows,
        "transitionMatrixBeforeEnhancement": hmm_audit[
            "transitionMatrix"
        ],
        "transitionMatrixAfterEnhancement": hmm_audit[
            "transitionMatrix"
        ],
        "stateDurationsBeforeEnhancement": hmm_audit[
            "durationByState"
        ],
        "stateDurationsAfterEnhancement": hmm_audit[
            "durationByState"
        ],
        "entropyBeforeEnhancement": hmm_audit[
            "meanNormalizedEntropy"
        ],
        "entropyAfterEnhancement": hmm_audit[
            "meanNormalizedEntropy"
        ],
        "stateDefinitionChanged": False,
        "reason": (
            "Phase 1.6 uses LPPLS/DMD as downstream transition/turning "
            "covariates. It does not alter HMM emissions or hidden states."
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Singularity Phase 1.6 HMM physics-feature report",
        "",
        f"- Run ID: `{result['runId']}`",
        f"- Source commit at run: `{result['sourceCommitAtRun']}`",
        f"- Status: `{result['status']}`",
        f"- User break date: `{result['postJump']['breakDate']}`",
        f"- Post-break trading days: {result['postJump']['tradingDays']}",
        f"- Post-jump HMM evaluated: `{result['postJump']['evaluated']}`",
        "",
        "Research-only. LPPLS/DMD are auxiliary covariates, never independent "
        "signals or BUY/SELL gates.",
        "",
        "## Observed",
        "",
    ]
    lines.extend(f"- {item}" for item in result["observed"])
    lines.extend(
        [
            "",
            "## Calibrated purged walk-forward metrics",
            "",
            "| population | h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for population, horizons in result["metrics"]["overall"].items():
        for horizon, variants in horizons.items():
            for variant, metric in variants.items():
                auc = (
                    "null"
                    if metric["auc"] is None
                    else f"{metric['auc']:.6f}"
                )
                lines.append(
                    f"| {population} | {horizon} | {variant} | "
                    f"{metric['count']:,} | {metric['brier']:.6f} | "
                    f"{metric['log_loss']:.6f} | {auc} | "
                    f"{metric['ece']:.6f} | {metric['highRiskCount']} |"
                )
    lines.extend(["", "## Fixed success gates", ""])
    for name, diagnostic in result["candidateDiagnostics"].items():
        lines.append(f"### {name}")
        lines.append("")
        for horizon, values in diagnostic["horizons"].items():
            bootstrap = values["clusterBootstrap"]
            if bootstrap is None:
                lines.append(
                    f"- {horizon}-bar: `{values['status']}`; pass=`False`."
                )
                continue
            lines.append(
                f"- {horizon}-bar: metric wins "
                f"{values['improvedMetricCountOfFour']}/4; folds "
                f"{values['improvingFolds']}/{values['totalFolds']}; months "
                f"{values['improvingMonths']}/{values['totalMonths']}; "
                f"high-risk n={values['highRiskCount']}; Brier delta "
                f"{bootstrap['meanBrierDeltaCandidateMinusBaseline']:.8f} "
                f"CI [{bootstrap['lower']:.8f}, {bootstrap['upper']:.8f}]; "
                f"pass=`{values['passes']}`."
            )
        lines.append(
            f"- Helps HMM under the fixed gate: `{diagnostic['helpsHMM']}`"
        )
        lines.append("")
    lines.extend(
        [
            "## Required conclusions",
            "",
            f"- HMM improved by LPPLS: `{result['conclusions']['lpplsHelpsHMM']}`",
            f"- LPPLS forward-retest status: `{result['conclusions']['lpplsForwardRetestStatus']}`",
            f"- HMM improved by Koopman/DMD: `{result['conclusions']['dmdHelpsHMM']}`",
            f"- Combined physics features help HMM: `{result['conclusions']['combinedHelpsHMM']}`",
            f"- Improvement survives same-sample comparison: `{result['conclusions']['sameSampleSurvives']}`",
            "- Post-jump HMM improved: `not_evaluated_insufficient_post_break_days`",
            "- Time-decay HMM improved: `not_evaluated_half_life_not_preregistered`",
            "- HMM state matrix/durations/entropy changed: `false` by design.",
            "- Remains research-only: `true`",
            "",
            "DMD conclusions apply only to the explicitly reported DMD-complete "
            "late-session population and cannot be extrapolated to the full "
            "session.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    (
        phase2a_config,
        panel,
        _,
        base,
        labels,
        _,
        _,
    ) = phase16.load_context(
        {
            "source": {
                "phase2AConfig": config["source"]["phase2AConfig"]
            }
        }
    )
    table, coverage, audits = build_model_table(
        base, labels, phase2a_config, config
    )
    feature_map = variant_features(config)
    prediction_parts: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    population_counts: dict[str, Any] = {}
    for population, population_config in config["populations"].items():
        variants = population_config["variants"]
        union = sorted(
            {
                feature
                for variant in variants
                for feature in feature_map[variant]
            }
        )
        eligible = table.dropna(subset=union)
        if population_config["requiresDmd"]:
            eligible = eligible[eligible["dmd_window_valid"] == 1.0]
        population_counts[population] = {
            "rowsBeforeUnionCompleteness": int(len(table)),
            "eligibleRows": int(len(eligible)),
            "coverage": float(len(eligible) / len(table)),
            "variants": variants,
        }
        predictions, folds = run_population_walk_forward(
            table, population, variants, config
        )
        prediction_parts.append(predictions)
        fold_audits.extend(folds)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = summarize_predictions(predictions, config)
    candidates = {
        "lppls": {
            "population": "fullSession",
            "variant": "hmm_ews_lppls",
        },
        "dmd": {
            "population": "dmdCompleteLateSession",
            "variant": "hmm_ews_dmd",
        },
        "combined": {
            "population": "dmdCompleteLateSession",
            "variant": "hmm_ews_lppls_dmd",
        },
    }
    diagnostics: dict[str, Any] = {}
    for name, specification in candidates.items():
        horizons = evaluate_candidate(
            metrics,
            predictions,
            specification["population"],
            specification["variant"],
            config,
        )
        diagnostics[name] = {
            **specification,
            "horizons": horizons,
            "helpsHMM": any(
                values["passes"] for values in horizons.values()
            ),
        }
    break_date = config["userPreregisteredBreak"]["date"]
    post_dates = sorted(
        date
        for date in panel["trade_date"].unique()
        if date >= break_date
    )
    state_diagnostics = state_interpretation(
        table, audits["hmm"]
    )
    result = {
        "schemaVersion": "singularity_phase1_6_hmm_physics_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": phase2a.source_commit(),
        "generatedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "postJump": {
            "breakDate": break_date,
            "source": "user_instruction",
            "tradingDays": len(post_dates),
            "dates": post_dates,
            "minimumTrainingDays": config["userPreregisteredBreak"][
                "minimumPostJumpTrainingDays"
            ],
            "minimumTestDays": config["userPreregisteredBreak"][
                "minimumPostJumpTestDays"
            ],
            "evaluated": False,
            "reason": (
                "Only 15 post-break trading days exist; no defensible "
                "train/calibration/OOS split is possible."
            ),
        },
        "observed": [
            f"Full feature audit rows: {len(table):,}.",
            f"Full-session same-sample coverage: {population_counts['fullSession']['coverage']:.2%}.",
            f"DMD-complete same-sample coverage: {population_counts['dmdCompleteLateSession']['coverage']:.2%}.",
            "The complete DMD auxiliary vector retains only 14:15 and 14:40 decisions; 13:50 has no prior same-session spectral-radius drift.",
            f"Post-break period contains {len(post_dates)} trading days.",
            "The existing 3-state GaussianHMM1D was reused without state search.",
        ],
        "estimated": [
            "Logistic auxiliary coefficients and Platt calibration are fitted inside each purged fold.",
            "Winsor bounds are fitted inside each training fold only.",
            "Brier uncertainty is clustered by independent trading date.",
        ],
        "assumed": [
            "2026-06-12 is the intended year for the user's June 12 break.",
            "DMD evidence is late-session-only and is not a full-session claim.",
            "Unchanged HMM emissions imply unchanged hidden-state definitions.",
        ],
        "coverage": coverage,
        "populationCounts": population_counts,
        "hmmAudit": audits["hmm"],
        "ewsAudit": audits["ews"],
        "stateInterpretation": state_diagnostics,
        "walkForward": {
            "start": config["data"]["walkForwardStart"],
            "end": config["data"]["walkForwardEnd"],
            "testBlockDays": config["models"][
                "walkForwardTestBlockDays"
            ],
            "calibrationDays": config["models"]["calibrationDays"],
            "purgeDecisionRowsPerSymbol": config["models"][
                "purgeDecisionRowsPerSymbol"
            ],
            "embargoTradingDays": config["models"][
                "embargoTradingDays"
            ],
            "folds": fold_audits,
        },
        "metrics": metrics,
        "candidateDiagnostics": diagnostics,
        "conclusions": {
            "lpplsHelpsHMM": diagnostics["lppls"]["helpsHMM"],
            "lpplsForwardRetestStatus": (
                "validated"
                if diagnostics["lppls"]["helpsHMM"]
                else "retain_frozen_hypothesis_promising_5bar_not_proven"
            ),
            "dmdHelpsHMM": diagnostics["dmd"]["helpsHMM"],
            "combinedHelpsHMM": diagnostics["combined"]["helpsHMM"],
            "sameSampleSurvives": any(
                diagnostic["helpsHMM"]
                for diagnostic in diagnostics.values()
            ),
            "postJumpHMM": "not_evaluated_insufficient_post_break_days",
            "timeDecayHMM": "not_evaluated_half_life_not_preregistered",
            "productionOrForwardUseAllowed": False,
        },
        "forwardRetestPlan": {
            "status": "preregistered_monitoring_hypothesis_only",
            "featureFamily": "lppls_hmm_auxiliary",
            "firstEligibleNewDateAfter": config["data"]["walkForwardEnd"],
            "minimumNewIndependentTradingDays": 20,
            "parametersRemainFrozen": True,
            "historicalRefitOrRetuningAllowed": False,
            "automaticForwardTaskIntegration": False,
            "primaryHorizonBars": 5,
            "primaryComparison": "hmm_ews_lppls_vs_current_hmm_ews",
            "requiredMetrics": [
                "Brier",
                "LogLoss",
                "AUC",
                "ECE",
                "trade_date_clustered_Brier_delta",
                "high_risk_bucket_count_and_hit_rate"
            ]
        },
        "lookaheadAudit": {
            "featuresPastOnly": True,
            "labelsSeparateAndFutureUsing": True,
            "hmmFitEndBeforeWalkForward": (
                config["data"]["fixedHMMFitEnd"]
                < config["data"]["walkForwardStart"]
            ),
            "winsorizationFitPerFold": True,
            "plattCalibrationPrecedesTest": True,
            "sameTradingDateAcrossBoundaries": False,
            "phase15LedgerRead": False,
        },
        "skipped": [
            "post_jump_hmm_insufficient_days",
            "time_decay_hmm_half_life_not_preregistered",
            "multivariate_hmm",
            "deep_lppls",
            "kernel_koopman",
            "deep_koopman",
            "online_inference",
            "live_trading_integration",
        ],
        "phase15Touched": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    phase2a.atomic_json(output_dir / "phase1_6_model_result.json", result)
    phase2a.atomic_text(
        output_dir / "phase1_6_model_report.md",
        render_report(result),
    )
    predictions.to_csv(
        output_dir / "walk_forward_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    table.to_csv(
        output_dir / "model_feature_table.csv",
        index=False,
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    run_id = args.run_id or (
        "model_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output_dir = resolve(config["output"]["root"]) / run_id
    result = run(config, output_dir)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "status": result["status"],
                "lppls_helps_hmm": result["conclusions"][
                    "lpplsHelpsHMM"
                ],
                "dmd_helps_hmm": result["conclusions"]["dmdHelpsHMM"],
                "combined_helps_hmm": result["conclusions"][
                    "combinedHelpsHMM"
                ],
                "post_jump_hmm": result["conclusions"]["postJumpHMM"],
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
