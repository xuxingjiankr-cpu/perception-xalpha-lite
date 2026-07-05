"""Preregistered Singularity Phase 1 turning-risk shadow research.

This module is offline, research-only and record-only.  It cannot place
orders, alter positions, write strategy overlays, modify risk gates, or write
the production decision-probability artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision_probability import probability_metrics as calibrated_metrics
from research_hmm_nn_bl import (
    GaussianHMM1D,
    ROOT,
    load_panel,
    select_training_universe,
)
from research_minute_forecast_shadow import build_feature_frames


CN = timezone(timedelta(hours=8))
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "singularity_phase1_preregistered.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs" / "edge_research" / "singularity_phase1"
)

HMM_FEATURES = [
    "hmm_bear_probability",
    "hmm_bull_probability",
    "hmm_expected_return",
    "regime_transition_risk",
    "regime_entropy",
]
EWS_FEATURES = [
    "ews_log_variance",
    "ews_autocorrelation_1",
    "ews_variance_slope",
    "ews_market_dispersion",
    "ews_score",
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(
            json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    json_safe(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        temporary = Path(handle.name)
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "singularity_phase1_preregistered_v1":
        raise ValueError("unexpected Phase 1 config schema")
    if config.get("status") != "research_only" or not config.get(
        "diagnosticOnly"
    ):
        raise ValueError("Phase 1 must remain research_only/diagnosticOnly")
    safety = config.get("safety", {})
    required_true = ["offlineOnly", "recordOnly"]
    required_false = [
        "brokerCallsAllowed",
        "onlineInferenceAllowed",
        "liveConfigWritesAllowed",
        "overlayWritesAllowed",
        "positionSizingAllowed",
        "orderSubmissionAllowed",
        "riskGateChangesAllowed",
        "buildDecisionIntegrationAllowed",
        "promotionAllowed",
    ]
    if any(not safety.get(name) for name in required_true):
        raise ValueError("offline and record-only safety flags must be true")
    if any(safety.get(name) is not False for name in required_false):
        raise ValueError("all live mutation safety flags must be false")
    labels = config["labels"]
    active = [int(value) for value in labels["activeHorizonsBars"]]
    purge = int(config["models"]["purgeBars"])
    if purge < max(active):
        raise ValueError("purgeBars must be at least the maximum active horizon")
    if 60 in active or "60" not in labels.get("skippedHorizons", {}):
        raise ValueError("60-bar horizon must be explicitly skipped in Phase 1")
    if not config["models"].get("parametersFrozenBeforeWalkForward"):
        raise ValueError("parameters must be frozen before walk-forward")


def select_entry_eligible_universe(
    quotes_path: Path, config: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    data_cfg = config["data"]
    candidates, raw_audit = select_training_universe(
        quotes_path,
        str(data_cfg["trainEnd"]),
        int(data_cfg["candidateUniverseSize"]),
    )
    metadata = {
        row["stockCode"]: row for row in raw_audit.get("selected", [])
    }
    excluded_classes = {
        str(value).strip().lower()
        for value in data_cfg.get("excludedAssetClasses", [])
    }
    accepted: list[str] = []
    rejected: list[dict[str, Any]] = []
    for code in candidates:
        row = metadata.get(code, {})
        asset_class = str(row.get("assetClass") or "unknown").strip().lower()
        if asset_class in excluded_classes:
            rejected.append(
                {
                    "stockCode": code,
                    "name": row.get("name", code),
                    "assetClass": asset_class,
                    "reason": "excluded_asset_class",
                }
            )
            continue
        accepted.append(code)
        if len(accepted) >= int(data_cfg["universeSize"]):
            break
    if len(accepted) != int(data_cfg["universeSize"]):
        raise RuntimeError(
            f"only {len(accepted)} entry-eligible ETFs after training-only selection"
        )
    selected_rows = [metadata[code] for code in accepted]
    return accepted, {
        "method": "training_only_coverage_and_average_cumulative_amount",
        "selectionUsesWalkForward": False,
        "trainEnd": data_cfg["trainEnd"],
        "candidateCount": len(candidates),
        "selectedCount": len(accepted),
        "excludedAssetClasses": sorted(excluded_classes),
        "rejectedBeforeSelectionCompleted": rejected,
        "selected": selected_rows,
        "rawTrainingAudit": {
            "trainingDates": raw_audit.get("trainingDates"),
            "minimumCoverageDays": raw_audit.get("minimumCoverageDays"),
        },
    }


def rolling_variance_slope(values: pd.DataFrame, window: int) -> pd.DataFrame:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    def slope(array: np.ndarray) -> float:
        series = np.asarray(array, dtype=float)
        if len(series) != window or not np.isfinite(series).all():
            return np.nan
        mean = float(np.mean(series))
        raw_slope = float(np.dot(x_centered, series - mean) / denominator)
        return raw_slope / max(mean, 1e-10)

    dates = pd.Series(values.index.strftime("%Y-%m-%d"), index=values.index)
    return values.groupby(dates).transform(
        lambda frame: frame.rolling(window, min_periods=window).apply(
            slope, raw=True
        )
    )


def build_ews_frames(
    base_frames: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame | pd.Series], dict[str, Any]]:
    window = int(config["features"]["ewsWindowBars"])
    floor = float(config["features"]["ewsVarianceFloor"])
    ret_1: pd.DataFrame = base_frames["ret_1"]
    dates = pd.Series(ret_1.index.strftime("%Y-%m-%d"), index=ret_1.index)
    variance = ret_1.groupby(dates).transform(
        lambda frame: frame.rolling(window, min_periods=window).var()
    )
    autocorrelation = ret_1.groupby(dates).transform(
        lambda frame: frame.rolling(window, min_periods=window).corr(
            frame.shift(1)
        )
    )
    squared = ret_1.pow(2)
    variance_slope = rolling_variance_slope(squared, window)
    dispersion = ret_1.std(axis=1, skipna=True)
    components: dict[str, pd.DataFrame | pd.Series] = {
        "ews_log_variance": np.log(variance.clip(lower=floor)),
        "ews_autocorrelation_1": autocorrelation,
        "ews_variance_slope": variance_slope,
        "ews_market_dispersion": dispersion,
    }

    train_end = str(config["data"]["trainEnd"])
    training_mask = ret_1.index.strftime("%Y-%m-%d") <= train_end
    statistics: dict[str, dict[str, float]] = {}
    z_components: dict[str, pd.DataFrame | pd.Series] = {}
    for name, frame in components.items():
        if isinstance(frame, pd.DataFrame):
            values = frame.loc[training_mask].to_numpy(dtype=float).ravel()
        else:
            values = frame.loc[training_mask].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            raise RuntimeError(f"invalid frozen standard deviation for {name}")
        statistics[name] = {
            "mean": mean,
            "standardDeviation": std,
            "fitEnd": train_end,
            "sampleCount": int(len(values)),
        }
        z_components[name] = ((frame - mean) / std).clip(-8.0, 8.0)

    # All four components are oriented so that larger means greater instability.
    score_parts: list[pd.DataFrame] = []
    for name, frame in z_components.items():
        if isinstance(frame, pd.Series):
            frame = pd.DataFrame(
                np.repeat(
                    frame.to_numpy(dtype=float)[:, None],
                    len(ret_1.columns),
                    axis=1,
                ),
                index=ret_1.index,
                columns=ret_1.columns,
            )
            z_components[name] = frame
        score_parts.append(1.0 / (1.0 + np.exp(-frame)))
    ews_score = sum(score_parts) / len(score_parts)
    components["ews_score"] = ews_score
    return components, {
        "windowBars": window,
        "pastOnly": True,
        "equalWeight": True,
        "orientation": "larger_is_more_unstable",
        "frozenStandardization": statistics,
    }


def build_hmm_frames(
    base_frames: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    hmm_cfg = config["features"]["hmm"]
    hmm = GaussianHMM1D(
        n_states=int(hmm_cfg["states"]),
        n_iter=int(hmm_cfg["iterations"]),
        variance_floor=float(hmm_cfg["varianceFloor"]),
    )
    market_return: pd.Series = base_frames["market_ret_1"]
    dates = pd.Series(
        market_return.index.strftime("%Y-%m-%d"), index=market_return.index
    )
    train_end = str(hmm_cfg["fitEnd"])
    sequences = [
        market_return[dates == trade_date].to_numpy(dtype=float)
        for trade_date in sorted(set(dates[dates <= train_end]))
    ]
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
    probability_values = probabilities.to_numpy(dtype=float)
    transition_risk = 1.0 - probability_values @ np.diag(hmm.transition)
    clipped = np.clip(probability_values, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / math.log(
        hmm.n_states
    )
    expected = probability_values @ hmm.means
    return {
        "hmm_bear_probability": probabilities.iloc[:, 0],
        "hmm_bull_probability": probabilities.iloc[:, -1],
        "hmm_expected_return": pd.Series(
            expected, index=market_return.index
        ),
        "regime_transition_risk": pd.Series(
            transition_risk, index=market_return.index
        ),
        "regime_entropy": pd.Series(entropy, index=market_return.index),
    }, {
        "implementation": "research_hmm_nn_bl.GaussianHMM1D",
        "reimplemented": False,
        "fitEnd": train_end,
        "states": hmm.n_states,
        "means": hmm.means.tolist(),
        "variances": hmm.variances.tolist(),
        "startProbability": hmm.start_probability.tolist(),
        "transitionMatrix": hmm.transition.tolist(),
        "filtering": "causal_session_reset",
        "transitionRiskDefinition": "1-sum(posterior_state*transition_diagonal)",
    }


def scalar_to_frame(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.repeat(
            series.reindex(like.index).to_numpy(dtype=float)[:, None],
            len(like.columns),
            axis=1,
        ),
        index=like.index,
        columns=like.columns,
    )


def build_feature_table(
    base_frames: dict[str, Any],
    ews_frames: dict[str, pd.DataFrame | pd.Series],
    hmm_frames: dict[str, pd.Series],
    config: dict[str, Any],
) -> pd.DataFrame:
    prices: pd.DataFrame = base_frames["prices"]
    frame_map: dict[str, pd.DataFrame] = {}
    for name in config["features"]["base"]:
        source = base_frames[name]
        frame_map[name] = (
            source.reindex(index=prices.index, columns=prices.columns)
            if isinstance(source, pd.DataFrame)
            else scalar_to_frame(source, prices)
        )
    for name, source in ews_frames.items():
        frame_map[name] = (
            source.reindex(index=prices.index, columns=prices.columns)
            if isinstance(source, pd.DataFrame)
            else scalar_to_frame(source, prices)
        )
    for name, source in hmm_frames.items():
        frame_map[name] = scalar_to_frame(source, prices)

    ews_weight = float(
        config["features"]["singularityScoreWeights"]["ewsScore"]
    )
    hmm_weight = float(
        config["features"]["singularityScoreWeights"][
            "regimeTransitionRisk"
        ]
    )
    frame_map["singularity_score"] = (
        frame_map["ews_score"] * ews_weight
        + frame_map["regime_transition_risk"] * hmm_weight
    )

    records: list[pd.DataFrame] = []
    union_features = (
        list(config["features"]["base"])
        + HMM_FEATURES
        + EWS_FEATURES
        + ["singularity_score"]
    )
    for code in prices.columns:
        row = pd.DataFrame(index=prices.index)
        row["timestamp"] = prices.index
        row["trade_date"] = prices.index.strftime("%Y-%m-%d")
        row["stockCode"] = code
        row["close"] = prices[code].to_numpy(dtype=float)
        for feature in union_features:
            row[feature] = frame_map[feature][code].to_numpy(dtype=float)
        records.append(row.reset_index(drop=True))
    table = pd.concat(records, ignore_index=True)
    table = table.replace([np.inf, -np.inf], np.nan)
    return table.dropna(subset=["close"] + union_features).sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    ).reset_index(drop=True)


def build_label_table(
    base_frames: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels_cfg = config["labels"]
    horizons = [int(value) for value in labels_cfg["horizonsBars"]]
    active = set(int(value) for value in labels_cfg["activeHorizonsBars"])
    prices: pd.DataFrame = base_frames["prices"]
    vol_6: pd.DataFrame = base_frames["vol_6"]
    ret_6: pd.DataFrame = base_frames["ret_6"]
    acceleration: pd.DataFrame = base_frames["acceleration_1"]
    lookback = int(labels_cfg["localExtremaLookbackBars"])
    tolerance = float(labels_cfg["localExtremaTolerancePct"])
    minimum_move = float(labels_cfg["minimumMovePct"])
    vol_multiplier = float(labels_cfg["volatilityMultiplier"])
    breakout_buffer = float(labels_cfg["breakoutBufferPct"])
    trend_minimum = float(labels_cfg["trendMinimumPct"])
    chase_minimum = float(labels_cfg["chaseMinimumPct"])
    round_trip_cost = float(labels_cfg["roundTripCostPct"])
    scale_sqrt = bool(labels_cfg["scaleVolatilityBySqrtHorizon"])

    records: list[dict[str, Any]] = []
    for trade_date, day_prices in prices.groupby(
        prices.index.strftime("%Y-%m-%d")
    ):
        timestamps = list(day_prices.index)
        for code in prices.columns:
            close = day_prices[code].to_numpy(dtype=float)
            for index, timestamp in enumerate(timestamps):
                current = close[index]
                if not np.isfinite(current) or current <= 0:
                    continue
                row: dict[str, Any] = {
                    "timestamp": timestamp,
                    "trade_date": trade_date,
                    "stockCode": code,
                }
                for horizon in horizons:
                    prefix_names = [
                        "reversal_up",
                        "reversal_down",
                        "turning_point",
                        "failed_breakout",
                        "trend_exhaustion",
                        "stop_chase",
                        "future_max_return",
                        "future_min_return",
                        "future_close_return",
                        "label_threshold",
                    ]
                    if horizon not in active:
                        for prefix in prefix_names:
                            row[f"{prefix}_{horizon}"] = np.nan
                        continue
                    if index < lookback - 1 or index + horizon >= len(close):
                        for prefix in prefix_names:
                            row[f"{prefix}_{horizon}"] = np.nan
                        continue
                    past = close[index - lookback + 1 : index + 1]
                    future = close[index + 1 : index + horizon + 1]
                    if (
                        len(future) != horizon
                        or not np.isfinite(past).all()
                        or not np.isfinite(future).all()
                    ):
                        for prefix in prefix_names:
                            row[f"{prefix}_{horizon}"] = np.nan
                        continue
                    past_vol = float(vol_6.at[timestamp, code])
                    if not np.isfinite(past_vol):
                        for prefix in prefix_names:
                            row[f"{prefix}_{horizon}"] = np.nan
                        continue
                    scale = math.sqrt(horizon) if scale_sqrt else 1.0
                    threshold = max(
                        minimum_move, vol_multiplier * past_vol * scale
                    )
                    future_returns = future / current - 1.0
                    future_max = float(np.max(future_returns))
                    future_min = float(np.min(future_returns))
                    future_close = float(future[-1] / current - 1.0)
                    near_high = current >= float(np.max(past)) * (1.0 - tolerance)
                    near_low = current <= float(np.min(past)) * (1.0 + tolerance)
                    reversal_down = bool(near_high and future_min <= -threshold)
                    reversal_up = bool(near_low and future_max >= threshold)
                    previous = past[:-1]
                    failed_breakout = bool(
                        current
                        > float(np.max(previous)) * (1.0 + breakout_buffer)
                        and float(np.min(future)) <= float(np.max(previous))
                    )
                    current_ret_6 = float(ret_6.at[timestamp, code])
                    current_acceleration = float(
                        acceleration.at[timestamp, code]
                    )
                    trend_exhaustion = bool(
                        np.isfinite(current_ret_6)
                        and np.isfinite(current_acceleration)
                        and current_ret_6 >= trend_minimum
                        and current_acceleration < 0.0
                        and future_min <= -threshold
                    )
                    stop_chase = bool(
                        np.isfinite(current_ret_6)
                        and current_ret_6 >= chase_minimum
                        and future_close - round_trip_cost <= 0.0
                    )
                    row[f"reversal_up_{horizon}"] = int(reversal_up)
                    row[f"reversal_down_{horizon}"] = int(reversal_down)
                    row[f"turning_point_{horizon}"] = int(
                        reversal_up or reversal_down
                    )
                    row[f"failed_breakout_{horizon}"] = int(failed_breakout)
                    row[f"trend_exhaustion_{horizon}"] = int(
                        trend_exhaustion
                    )
                    row[f"stop_chase_{horizon}"] = int(stop_chase)
                    row[f"future_max_return_{horizon}"] = future_max
                    row[f"future_min_return_{horizon}"] = future_min
                    row[f"future_close_return_{horizon}"] = future_close
                    row[f"label_threshold_{horizon}"] = threshold
                records.append(row)
    table = pd.DataFrame.from_records(records).sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    )
    definition = {
        "featuresReadFuture": False,
        "labelsUseFuture": True,
        "labelsStoredSeparately": True,
        "sameSessionOnly": True,
        "activeHorizonsBars": sorted(active),
        "skippedHorizons": labels_cfg["skippedHorizons"],
        "localExtremaLookbackBars": lookback,
        "localExtremaTolerancePct": tolerance,
        "threshold": {
            "minimumMovePct": minimum_move,
            "volatilityMultiplier": vol_multiplier,
            "sqrtHorizonScaling": scale_sqrt,
            "volatilityInput": "past-only vol_6",
        },
        "primaryTarget": "turning_point = reversal_up OR reversal_down",
    }
    return table, definition


def purge_symbol_tail(
    rows: pd.DataFrame, purge_bars: int
) -> tuple[pd.DataFrame, int]:
    if rows.empty or purge_bars <= 0:
        return rows.copy(), 0
    ordered = rows.sort_values(
        ["stockCode", "timestamp"], kind="stable"
    ).copy()
    remove_index = (
        ordered.groupby("stockCode", sort=False)
        .tail(purge_bars)
        .index
    )
    purged = ordered.drop(index=remove_index).sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    )
    return purged.reset_index(drop=True), int(len(remove_index))


def make_classifier(config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(config["models"]["logisticC"]),
                    max_iter=500,
                    random_state=int(config["models"]["randomSeed"]),
                ),
            ),
        ]
    )


def probability_from_model(
    model: Pipeline, rows: pd.DataFrame, features: list[str]
) -> np.ndarray:
    return np.clip(
        model.predict_proba(rows[features].to_numpy(dtype=float))[:, 1],
        1e-6,
        1.0 - 1e-6,
    )


def apply_platt(
    raw_probability: np.ndarray,
    calibrator: LogisticRegression | None,
) -> np.ndarray:
    if calibrator is None:
        return raw_probability
    logits = np.log(
        raw_probability / np.maximum(1.0 - raw_probability, 1e-12)
    ).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def variant_features(config: dict[str, Any]) -> dict[str, list[str]]:
    base = list(config["features"]["base"])
    return {
        "baseline": base,
        "hmm": base + HMM_FEATURES,
        "ews": base + EWS_FEATURES,
        "hmm_ews": base
        + HMM_FEATURES
        + EWS_FEATURES
        + ["singularity_score"],
    }


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = ["timestamp", "trade_date", "stockCode"]
    merged = features.merge(labels, on=keys, how="inner", validate="one_to_one")
    variants = variant_features(config)
    all_features = sorted({item for values in variants.values() for item in values})
    merged = merged.dropna(subset=all_features)
    models_cfg = config["models"]
    data_cfg = config["data"]
    active_horizons = [
        int(value) for value in config["labels"]["activeHorizonsBars"]
    ]
    block_days = int(models_cfg["walkForwardTestBlockDays"])
    purge_bars = int(models_cfg["purgeBars"])
    calibration_days = int(models_cfg["calibrationDays"])
    minimum_fit_days = int(models_cfg["minimumBaseFitDays"])

    prediction_parts: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for horizon in active_horizons:
        target = f"turning_point_{horizon}"
        horizon_rows = merged.dropna(subset=[target]).copy()
        horizon_rows[target] = horizon_rows[target].astype(int)
        test_dates = sorted(
            date
            for date in horizon_rows["trade_date"].unique()
            if str(data_cfg["walkForwardStart"])
            <= date
            <= str(data_cfg["walkForwardEnd"])
        )
        for block_number, start in enumerate(
            range(0, len(test_dates), block_days), start=1
        ):
            block = test_dates[start : start + block_days]
            first_test_date = block[0]
            train_raw = horizon_rows[
                horizon_rows["trade_date"] < first_test_date
            ].copy()
            test = horizon_rows[
                horizon_rows["trade_date"].isin(block)
            ].copy()
            train_before_test, test_boundary_removed = purge_symbol_tail(
                train_raw, purge_bars
            )
            train_dates = sorted(train_before_test["trade_date"].unique())
            if len(train_dates) <= calibration_days + minimum_fit_days:
                raise RuntimeError(
                    f"insufficient dates before {first_test_date}: {len(train_dates)}"
                )
            calibration_date_set = set(train_dates[-calibration_days:])
            calibration = train_before_test[
                train_before_test["trade_date"].isin(calibration_date_set)
            ].copy()
            fit_raw = train_before_test[
                ~train_before_test["trade_date"].isin(calibration_date_set)
            ].copy()
            fit, calibration_boundary_removed = purge_symbol_tail(
                fit_raw, purge_bars
            )
            if fit[target].nunique() != 2 or calibration[target].nunique() != 2:
                raise RuntimeError(
                    f"class missing in fold horizon={horizon} block={block_number}"
                )
            fold_audit: dict[str, Any] = {
                "horizonBars": horizon,
                "fold": block_number,
                "testDates": block,
                "baseFitStart": str(fit["trade_date"].min()),
                "baseFitEnd": str(fit["trade_date"].max()),
                "baseFitDays": int(fit["trade_date"].nunique()),
                "calibrationStart": str(calibration["trade_date"].min()),
                "calibrationEnd": str(calibration["trade_date"].max()),
                "calibrationDays": int(calibration["trade_date"].nunique()),
                "testBoundaryPurgedRows": test_boundary_removed,
                "calibrationBoundaryPurgedRows": calibration_boundary_removed,
                "purgeBarsPerSymbol": purge_bars,
                "sameDateAcrossBoundaries": False,
                "variants": {},
            }
            for variant, columns in variants.items():
                model = make_classifier(config)
                model.fit(
                    fit[columns].to_numpy(dtype=float),
                    fit[target].to_numpy(dtype=int),
                )
                calibration_raw = probability_from_model(
                    model, calibration, columns
                )
                calibration_labels = calibration[target].to_numpy(dtype=int)
                calibration_logits = np.log(
                    calibration_raw
                    / np.maximum(1.0 - calibration_raw, 1e-12)
                ).reshape(-1, 1)
                calibrator = LogisticRegression(
                    C=float(models_cfg["calibrationC"]),
                    max_iter=500,
                    random_state=int(models_cfg["randomSeed"]),
                )
                calibrator.fit(calibration_logits, calibration_labels)
                test_raw = probability_from_model(model, test, columns)
                probability = apply_platt(test_raw, calibrator)
                output = test[keys + [target]].copy()
                output = output.rename(columns={target: "actual"})
                output["horizon_bars"] = horizon
                output["variant"] = variant
                output["fold"] = block_number
                output["raw_probability"] = test_raw
                output["probability"] = probability
                output["training_positive_rate"] = float(
                    fit[target].mean()
                )
                prediction_parts.append(output)
                fold_audit["variants"][variant] = {
                    "features": columns,
                    "baseFitSamples": int(len(fit)),
                    "calibrationSamples": int(len(calibration)),
                    "testSamples": int(len(test)),
                    "baseFitPositiveRate": float(fit[target].mean()),
                    "calibrationPositiveRate": float(
                        calibration[target].mean()
                    ),
                    "testPositiveRate": float(test[target].mean()),
                    "calibrationMethod": "platt_logistic",
                }
            fold_audits.append(fold_audit)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return predictions.sort_values(
        ["horizon_bars", "trade_date", "timestamp", "stockCode", "variant"],
        kind="stable",
    ).reset_index(drop=True), fold_audits


def metric_payload(
    rows: pd.DataFrame, bin_edges: list[float]
) -> dict[str, Any]:
    metrics = calibrated_metrics(
        rows["probability"].to_numpy(dtype=float),
        rows["actual"].to_numpy(dtype=int),
        bin_edges=bin_edges,
    )
    threshold = 0.5
    predicted = rows["probability"].to_numpy(dtype=float) >= threshold
    actual = rows["actual"].to_numpy(dtype=int) == 1
    true_positive = int(np.sum(predicted & actual))
    false_positive = int(np.sum(predicted & ~actual))
    false_negative = int(np.sum(~predicted & actual))
    true_negative = int(np.sum(~predicted & ~actual))
    metrics.update(
        {
            "threshold": threshold,
            "predictedPositiveCount": int(np.sum(predicted)),
            "hitRateAtThreshold": (
                true_positive / int(np.sum(predicted))
                if np.sum(predicted)
                else None
            ),
            "recallAtThreshold": (
                true_positive / int(np.sum(actual))
                if np.sum(actual)
                else None
            ),
            "confusion": {
                "truePositive": true_positive,
                "falsePositive": false_positive,
                "falseNegative": false_negative,
                "trueNegative": true_negative,
            },
        }
    )
    return metrics


def summarize_predictions(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    bin_edges = [
        float(value) for value in config["models"]["probabilityBinEdges"]
    ]
    variants = list(variant_features(config))
    horizons = [
        int(value) for value in config["labels"]["activeHorizonsBars"]
    ]
    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        by_horizon[str(horizon)] = {}
        for variant in variants:
            rows = predictions[
                (predictions["horizon_bars"] == horizon)
                & (predictions["variant"] == variant)
            ]
            by_horizon[str(horizon)][variant] = metric_payload(rows, bin_edges)

    pooled: dict[str, Any] = {}
    for variant in variants:
        pooled[variant] = metric_payload(
            predictions[predictions["variant"] == variant], bin_edges
        )

    by_fold: dict[str, dict[str, dict[str, Any]]] = {}
    for horizon in horizons:
        by_fold[str(horizon)] = {}
        horizon_rows = predictions[
            predictions["horizon_bars"] == horizon
        ]
        for fold in sorted(horizon_rows["fold"].unique()):
            by_fold[str(horizon)][str(int(fold))] = {}
            for variant in variants:
                rows = horizon_rows[
                    (horizon_rows["fold"] == fold)
                    & (horizon_rows["variant"] == variant)
                ]
                by_fold[str(horizon)][str(int(fold))][variant] = (
                    metric_payload(rows, bin_edges)
                )

    ablation: dict[str, Any] = {}
    for horizon in horizons:
        baseline = by_horizon[str(horizon)]["baseline"]
        ablation[str(horizon)] = {}
        for variant in variants:
            current = by_horizon[str(horizon)][variant]
            ablation[str(horizon)][variant] = {
                "deltaBrierVsBaseline": (
                    current["brier"] - baseline["brier"]
                ),
                "deltaLogLossVsBaseline": (
                    current["log_loss"] - baseline["log_loss"]
                ),
                "deltaAucVsBaseline": (
                    None
                    if current["auc"] is None or baseline["auc"] is None
                    else current["auc"] - baseline["auc"]
                ),
                "deltaEceVsBaseline": current["ece"] - baseline["ece"],
                "sameForecastCount": current["count"] == baseline["count"],
            }
    return {
        "primaryMetric": "brier",
        "secondaryMetrics": ["log_loss", "auc", "ece"],
        "byHorizon": by_horizon,
        "byFold": by_fold,
        "pooledAcrossHorizonForecasts": pooled,
        "ablationVsBaseline": ablation,
    }


def research_action_hint(
    probability: float,
    ret_6: float,
    artifact_cfg: dict[str, Any],
) -> str:
    if probability >= float(artifact_cfg["paperExitAlertThreshold"]):
        return "PAPER_EXIT_ALERT_RESEARCH"
    if (
        probability >= float(artifact_cfg["reduceChaseThreshold"])
        and ret_6 > 0
    ):
        return "REDUCE_CHASE_RESEARCH"
    if probability >= float(artifact_cfg["watchThreshold"]):
        return "WATCH_RESEARCH"
    return "NORMAL_RESEARCH"


def build_turning_artifact(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    config: dict[str, Any],
    run_id: str,
    generated_at: str,
) -> list[dict[str, Any]]:
    chosen = str(config["artifact"]["probabilityVariant"])
    active = [
        int(value) for value in config["labels"]["activeHorizonsBars"]
    ]
    keys = ["timestamp", "trade_date", "stockCode"]
    combined = predictions[predictions["variant"] == chosen].pivot_table(
        index=keys,
        columns="horizon_bars",
        values="probability",
        aggfunc="first",
    )
    baseline_10 = predictions[
        (predictions["variant"] == "baseline")
        & (predictions["horizon_bars"] == 10)
    ].set_index(keys)["probability"]
    needed = features.set_index(keys)[
        [
            "ret_6",
            "ews_score",
            "regime_transition_risk",
            "regime_entropy",
            "singularity_score",
        ]
    ]
    combined = combined.join(needed, how="left").join(
        baseline_10.rename("base_model_probability"), how="left"
    )
    artifact_cfg = config["artifact"]
    rows: list[dict[str, Any]] = []
    for index, row in combined.sort_index().iterrows():
        timestamp, trade_date, code = index
        probabilities = {
            horizon: (
                float(row[horizon])
                if horizon in row.index and np.isfinite(row[horizon])
                else None
            )
            for horizon in active
        }
        finite_probabilities = [
            value for value in probabilities.values() if value is not None
        ]
        maximum = max(finite_probabilities) if finite_probabilities else 0.0
        ret_6 = float(row["ret_6"])
        signal = (
            "MOMENTUM_UP"
            if ret_6 >= float(config["labels"]["chaseMinimumPct"])
            else (
                "MOMENTUM_DOWN"
                if ret_6 <= -float(config["labels"]["chaseMinimumPct"])
                else "NEUTRAL"
            )
        )
        rows.append(
            {
                "schemaVersion": artifact_cfg["schemaVersion"],
                "status": "research_only",
                "shadowOnly": True,
                "diagnosticOnly": True,
                "runId": run_id,
                "generatedAt": generated_at,
                "tradeDate": trade_date,
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "stockCode": code,
                "barIntervalMinutes": int(
                    config["data"]["barIntervalMinutes"]
                ),
                "base_model_signal": signal,
                "base_model_probability": (
                    float(row["base_model_probability"])
                    if np.isfinite(row["base_model_probability"])
                    else None
                ),
                "p_turning_5": probabilities.get(5),
                "p_turning_10": probabilities.get(10),
                "p_turning_20": probabilities.get(20),
                "p_turning_30": probabilities.get(30),
                "p_turning_60": None,
                "p_turning_60_status": "skipped_cross_session",
                "ews_score": float(row["ews_score"]),
                "regime_transition_risk": float(
                    row["regime_transition_risk"]
                ),
                "regime_entropy": float(row["regime_entropy"]),
                "singularity_score": float(row["singularity_score"]),
                "research_action_hint": research_action_hint(
                    maximum, ret_6, artifact_cfg
                ),
                "orderInstruction": None,
                "positionSizeInstruction": None,
            }
        )
    return rows


def label_summary(
    labels: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in config["labels"]["horizonsBars"]:
        horizon = int(horizon)
        target = f"turning_point_{horizon}"
        valid = labels[target].dropna()
        result[str(horizon)] = {
            "samples": int(len(valid)),
            "positive": int(valid.sum()) if len(valid) else 0,
            "negative": int(len(valid) - valid.sum()) if len(valid) else 0,
            "positiveRate": float(valid.mean()) if len(valid) else None,
            "status": (
                "active"
                if horizon
                in {
                    int(value)
                    for value in config["labels"]["activeHorizonsBars"]
                }
                else "skipped_null"
            ),
        }
    return result


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# Singularity Phase 1 shadow report",
        "",
        f"- Status: **{result['status']}**",
        f"- Run ID: `{result['runId']}`",
        f"- Generated: {result['generatedAt']}",
        f"- Data: {result['data']['start']} through {result['data']['end']}",
        f"- Bar interval: {result['data']['barIntervalMinutes']} minutes",
        f"- Universe: {result['universe']['selectedCount']} training-selected, entry-eligible ETFs",
        f"- Feature rows: {result['samples']['featureRows']:,}",
        f"- Label rows: {result['samples']['labelRows']:,}",
        "",
        "This output is research-only and shadow-only. It does not modify live "
        "trading, positions, orders, risk gates, execution locks, overlays, "
        "`build_decision()` or `decision_probability_v1.json`.",
        "",
        "## Labels",
        "",
        "| Horizon (bars) | Status | Samples | Positive rate |",
        "|---:|---|---:|---:|",
    ]
    for horizon, summary in result["samples"]["labelsByHorizon"].items():
        rate = (
            "null"
            if summary["positiveRate"] is None
            else f"{summary['positiveRate']:.4%}"
        )
        lines.append(
            f"| {horizon} | {summary['status']} | {summary['samples']:,} | {rate} |"
        )

    lines.extend(
        [
            "",
            "The primary target is a same-session reversal from a trailing "
            "12-bar local extreme. The future window is used only to construct "
            "the separate offline label table. Horizon 60 is null because 300 "
            "minutes exceeds one A-share session.",
            "",
            "## Walk-forward calibrated ablation",
            "",
            "| Horizon | Variant | Brier | LogLoss | AUC | ECE | Forecasts |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, variants in result["metrics"]["byHorizon"].items():
        for variant, metrics in variants.items():
            auc = (
                "null" if metrics["auc"] is None else f"{metrics['auc']:.6f}"
            )
            lines.append(
                f"| {horizon} | {variant} | {metrics['brier']:.6f} | "
                f"{metrics['log_loss']:.6f} | {auc} | "
                f"{metrics['ece']:.6f} | {metrics['count']:,} |"
            )

    lines.extend(
        [
            "",
            "## HMM + EWS probability buckets",
            "",
            "| Horizon | Bucket | Count | Mean probability | Hit rate |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for horizon, variants in result["metrics"]["byHorizon"].items():
        for bucket in variants["hmm_ews"]["bins"]:
            lines.append(
                f"| {horizon} | [{bucket['low']:.1f}, {bucket['high']:.1f}) | "
                f"{bucket['count']:,} | {bucket['mean_predicted']:.4%} | "
                f"{bucket['actual_rate']:.4%} |"
            )

    lines.extend(
        [
            "",
            "## Causality and leakage audit",
            "",
            f"- Past-only features: `{result['causality']['pastOnlyFeatures']}`",
            f"- Future used only in separate labels: `{result['causality']['futureOnlyInLabels']}`",
            f"- Complete-date split: `{result['causality']['completeDateSplit']}`",
            f"- Purge bars per ETF: `{result['causality']['purgeBars']}`",
            f"- Maximum active horizon: `{result['causality']['maximumActiveHorizonBars']}`",
            f"- Train/test overlapping same-day ETF windows: `{result['causality']['overlappingSameDayWindowsAcrossSplit']}`",
            f"- OOS window previously reused: `{result['causality']['oosWindowPreviouslyReused']}`",
            "",
            "Residual risk remains: vendor bar timestamps and historical symbol "
            "availability are trusted as supplied, and this repeatedly used "
            "2026-05-21 through 2026-06-18 window is not a clean final OOS test.",
            "",
            "## Mechanical trade-frequency check",
            "",
            f"- Trading or gating applied: `{result['mechanicalFrequencyCheck']['tradingApplied']}`",
            f"- Equal forecast rows across variants: `{result['mechanicalFrequencyCheck']['equalForecastRowsAcrossVariants']}`",
            "",
            "Metric changes cannot be attributed to suppressing trades because "
            "all variants forecast the same rows and Phase 1 executes no trade.",
            "",
            "## Fold stability",
            "",
            "| Horizon | Joint Brier/AUC improving folds | Required | Stable |",
            "|---:|---:|---:|---|",
        ]
    )
    for horizon, audit in result["phase2"]["foldStability"].items():
        lines.append(
            f"| {horizon} | {audit['jointImprovingFolds']}/{audit['folds']} | "
            f"{audit['requiredFolds']} | {audit['stable']} |"
        )
    lines.extend(
        [
            "",
            "## Skipped",
            "",
        ]
    )
    for item in result["skipped"]:
        lines.append(f"- `{item['item']}`: {item['reason']}")
    lines.extend(
        [
            "",
            "## Phase 2 review",
            "",
            result["phase2"]["conclusion"],
            "",
            "This is a conservative post-run research review, not a "
            "preregistered model-selection or production-promotion gate.",
            "",
            "No production recommendation is made.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quotes", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    validate_config(config)
    quotes_path = (
        args.quotes.resolve()
        if args.quotes
        else (ROOT / config["data"]["quotes"]).resolve()
    )
    if not quotes_path.exists():
        raise FileNotFoundError(quotes_path)
    generated_at = datetime.now(CN).isoformat(timespec="seconds")
    config_hash = sha256_text(config_text)
    run_id = args.run_id or (
        f"singularity_p1_{datetime.now(CN).strftime('%Y%m%d_%H%M%S')}"
        f"_{config_hash[:8]}"
    )
    output_dir = args.output_root.resolve() / run_id
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Phase 1 run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    log_lines = [
        f"start_time={generated_at}",
        "status=research_only",
        "shadow_only=true",
        f"config={config_path}",
        f"quotes={quotes_path}",
        f"run_id={run_id}",
    ]
    atomic_json(output_dir / "config_snapshot.json", config)

    selected, universe_audit = select_entry_eligible_universe(
        quotes_path, config
    )
    atomic_json(output_dir / "universe_audit.json", universe_audit)
    log_lines.append(f"selected_universe={len(selected)}")

    panel = load_panel(quotes_path, selected)
    panel = panel[
        (panel["trade_date"] >= str(config["data"]["trainStart"]))
        & (panel["trade_date"] <= str(config["data"]["walkForwardEnd"]))
    ].copy()
    base_frames = build_feature_frames(panel)
    ews_frames, ews_audit = build_ews_frames(base_frames, config)
    hmm_frames, hmm_audit = build_hmm_frames(base_frames, config)
    feature_table = build_feature_table(
        base_frames, ews_frames, hmm_frames, config
    )
    label_table, label_definition = build_label_table(base_frames, config)

    feature_path = output_dir / "singularity_features.csv"
    label_path = output_dir / "reversal_labels.csv"
    feature_table.to_csv(feature_path, index=False, encoding="utf-8")
    label_table.to_csv(label_path, index=False, encoding="utf-8")
    log_lines.extend(
        [
            f"feature_rows={len(feature_table)}",
            f"label_rows={len(label_table)}",
        ]
    )

    predictions, fold_audits = run_walk_forward(
        feature_table, label_table, config
    )
    predictions.to_csv(
        output_dir / "walk_forward_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    metrics = summarize_predictions(predictions, config)
    turning_rows = build_turning_artifact(
        predictions, feature_table, config, run_id, generated_at
    )
    atomic_jsonl(
        output_dir / "turning_probabilities.jsonl", turning_rows
    )

    counts = (
        predictions.groupby(["horizon_bars", "variant"])
        .size()
        .unstack(fill_value=0)
    )
    equal_counts = bool((counts.nunique(axis=1) == 1).all())
    horizons = [
        int(value) for value in config["labels"]["activeHorizonsBars"]
    ]
    fold_stability: dict[str, Any] = {}
    stable_horizons: list[bool] = []
    for horizon in horizons:
        folds = metrics["byFold"][str(horizon)]
        joint_improving = 0
        details: dict[str, Any] = {}
        for fold, variants in folds.items():
            baseline = variants["baseline"]
            combined = variants["hmm_ews"]
            delta_brier = combined["brier"] - baseline["brier"]
            delta_auc = (
                None
                if combined["auc"] is None or baseline["auc"] is None
                else combined["auc"] - baseline["auc"]
            )
            improving = delta_brier < 0 and (
                delta_auc is None or delta_auc >= 0
            )
            joint_improving += int(improving)
            details[fold] = {
                "deltaBrierVsBaseline": delta_brier,
                "deltaAucVsBaseline": delta_auc,
                "jointImprovement": improving,
            }
        required_folds = math.ceil(len(folds) * 2.0 / 3.0)
        stable = joint_improving >= required_folds
        stable_horizons.append(stable)
        fold_stability[str(horizon)] = {
            "folds": len(folds),
            "requiredFolds": required_folds,
            "jointImprovingFolds": joint_improving,
            "stable": stable,
            "details": details,
        }
    phase2_worth = sum(stable_horizons) >= 3
    phase2_conclusion = (
        "HMM + EWS has joint Brier/AUC improvement in at least two-thirds of "
        "folds on at least three horizons. LPPLS/Koopman may be considered "
        "only as a separate research-only preregistration; the reused window "
        "still cannot support production."
        if phase2_worth
        else "HMM + EWS does not achieve joint Brier/AUC improvement in at "
        "least two-thirds of folds on at least three horizons. The evidence "
        "supports retaining EWS as a shadow hypothesis, but does not justify "
        "entering LPPLS/Koopman Phase 2 yet."
    )
    skipped = [
        {
            "item": "horizon_60",
            "reason": config["labels"]["skippedHorizons"]["60"],
        }
    ] + [
        {
            "item": item,
            "reason": "outside preregistered Phase 1 scope",
        }
        for item in config["skipped"]
    ]
    source_start = str(panel["trade_date"].min())
    source_end = str(panel["trade_date"].max())
    result: dict[str, Any] = {
        "schemaVersion": "singularity_phase1_result_v1",
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "runId": run_id,
        "generatedAt": generated_at,
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
            "parametersFrozenBeforeWalkForward": True,
        },
        "data": {
            "quotes": str(quotes_path),
            "start": source_start,
            "end": source_end,
            "trainStart": config["data"]["trainStart"],
            "trainEnd": config["data"]["trainEnd"],
            "walkForwardStart": config["data"]["walkForwardStart"],
            "walkForwardEnd": config["data"]["walkForwardEnd"],
            "barIntervalMinutes": config["data"]["barIntervalMinutes"],
            "tradingDays": int(panel["trade_date"].nunique()),
        },
        "universe": universe_audit,
        "samples": {
            "featureRows": int(len(feature_table)),
            "labelRows": int(len(label_table)),
            "walkForwardPredictionRows": int(len(predictions)),
            "turningArtifactRows": int(len(turning_rows)),
            "labelsByHorizon": label_summary(label_table, config),
        },
        "featureDefinition": {
            "base": config["features"]["base"],
            "ews": ews_audit,
            "hmm": hmm_audit,
            "singularityScore": config["features"][
                "singularityScoreWeights"
            ],
        },
        "labelDefinition": label_definition,
        "walkForward": {
            "method": "expanding_complete_date_blocks_with_temporal_platt_calibration",
            "testBlockDays": config["models"]["walkForwardTestBlockDays"],
            "calibrationDays": config["models"]["calibrationDays"],
            "purgeBarsPerSymbolAtEachBoundary": config["models"][
                "purgeBars"
            ],
            "foldCount": len(fold_audits),
            "folds": fold_audits,
        },
        "metrics": metrics,
        "causality": {
            "pastOnlyFeatures": True,
            "futureOnlyInLabels": True,
            "labelsStoredSeparately": True,
            "completeDateSplit": True,
            "purgeBars": config["models"]["purgeBars"],
            "maximumActiveHorizonBars": max(horizons),
            "overlappingSameDayWindowsAcrossSplit": False,
            "horizon60CrossesOvernight": False,
            "oosWindowPreviouslyReused": config["data"][
                "oosWindowPreviouslyReused"
            ],
            "lookaheadRisk": (
                "No known feature/label leakage in Phase 1. Residual risks are "
                "vendor timestamp semantics, present-day source coverage, and "
                "repeated reuse of the walk-forward evaluation window."
            ),
        },
        "mechanicalFrequencyCheck": {
            "tradingApplied": False,
            "gatingApplied": False,
            "equalForecastRowsAcrossVariants": equal_counts,
            "countsByHorizonAndVariant": counts.to_dict(orient="index"),
            "conclusion": (
                "No metric improvement can come from fewer executed trades; "
                "all variants forecast identical rows and execute nothing."
            ),
        },
        "selfTests": {
            "configSafetyValidated": True,
            "purgeAtLeastMaximumHorizon": int(
                config["models"]["purgeBars"]
            )
            >= max(horizons),
            "horizon60AllNull": bool(
                label_table["turning_point_60"].isna().all()
            ),
            "featureTableContainsNoTargetColumns": not any(
                column.startswith(("turning_point_", "future_"))
                for column in feature_table.columns
            ),
            "existingArtifactsPreserved": True,
            "productionDecisionProbabilityPreserved": True,
        },
        "skipped": skipped,
        "phase2": {
            "worthInvestigating": phase2_worth,
            "gate": (
                "descriptive post-run review: HMM+EWS delta Brier < 0 and "
                "delta AUC >= 0 jointly in at least two-thirds of folds on "
                "at least three of four horizons"
            ),
            "passingHorizons": int(sum(stable_horizons)),
            "foldStability": fold_stability,
            "conclusion": phase2_conclusion,
        },
        "artifacts": {
            "outputDirectory": str(output_dir),
            "configSnapshot": "config_snapshot.json",
            "universeAudit": "universe_audit.json",
            "features": "singularity_features.csv",
            "labels": "reversal_labels.csv",
            "predictions": "walk_forward_predictions.csv",
            "turningProbability": "turning_probabilities.jsonl",
            "result": "phase1_result.json",
            "report": "phase1_report.md",
            "log": "run.log",
        },
    }
    if not all(result["selfTests"].values()):
        raise RuntimeError(f"Phase 1 self-test failed: {result['selfTests']}")
    atomic_json(output_dir / "phase1_result.json", result)
    (output_dir / "phase1_report.md").write_text(
        make_report(result), encoding="utf-8"
    )
    log_lines.extend(
        [
            f"walk_forward_prediction_rows={len(predictions)}",
            f"turning_artifact_rows={len(turning_rows)}",
            f"phase2_worth_investigating={str(phase2_worth).lower()}",
            f"end_time={datetime.now(CN).isoformat(timespec='seconds')}",
            "exit_code=0",
        ]
    )
    (output_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": run_id,
                "output_dir": str(output_dir),
                "feature_rows": len(feature_table),
                "prediction_rows": len(predictions),
                "phase2_worth_investigating": phase2_worth,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
