"""Post-close Singularity Phase 1.5 forward shadow monitor.

The runner loads a hash-pinned frozen model, reconstructs causal five-minute
features from archived Eastmoney full-market snapshots, records probabilities
and outcomes in separate files, and refreshes research-only diagnostics.

It cannot fit a model, submit an order, size a position, modify a strategy
config, write an overlay, or participate in build_decision/BUY/SELL gates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from decision_probability import probability_metrics
from research_hmm_nn_bl import GaussianHMM1D, ROOT
from research_minute_forecast_shadow import build_feature_frames
from research_singularity_phase1 import (
    EWS_FEATURES,
    HMM_FEATURES,
    build_feature_table,
    build_label_table,
    rolling_variance_slope,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "singularity_phase1_5_forward.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_config_bundle(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schemaVersion") != "singularity_phase1_5_forward_v1":
        raise ValueError("unexpected Phase 1.5 config schema")
    if (
        config.get("status") != "research_only"
        or not config.get("shadowOnly")
        or not config.get("diagnosticOnly")
    ):
        raise ValueError("Phase 1.5 must remain research/shadow/diagnostic only")
    safety = config["safety"]
    if not safety.get("offlinePostCloseOnly") or not safety.get("recordOnly"):
        raise ValueError("post-close record-only flags must remain enabled")
    forbidden_true = [
        "brokerCallsAllowed",
        "onlineInferenceAllowed",
        "liveConfigWritesAllowed",
        "overlayWritesAllowed",
        "positionSizingAllowed",
        "orderSubmissionAllowed",
        "riskGateChangesAllowed",
        "buildDecisionIntegrationAllowed",
        "buySellGateIntegrationAllowed",
        "promotionAllowed",
    ]
    if any(safety.get(name) is not False for name in forbidden_true):
        raise ValueError("all mutation/trading safety flags must remain false")
    phase1_path = resolve(config["phase1"]["config"])
    if json_sha256(phase1_path) != config["phase1"]["configSha256"]:
        raise RuntimeError("sealed Phase 1 config hash mismatch")
    model_path = resolve(config["frozenModel"]["path"])
    actual_model_hash = json_sha256(model_path)
    if actual_model_hash != config["frozenModel"]["sha256"]:
        raise RuntimeError(
            f"frozen model hash mismatch: {actual_model_hash}"
        )
    bundle = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        bundle.get("schemaVersion")
        != "singularity_phase1_5_frozen_model_v1"
        or bundle.get("status") != "research_only"
        or bundle.get("runtimeRefitAllowed") is not False
        or bundle.get("parameterUpdatesAllowed") is not False
        or bundle.get("phase1ConfigSha256")
        != config["phase1"]["configSha256"]
        or bundle.get("historicalCutoff")
        != config["phase1"]["historicalCutoff"]
        or bundle.get("prospectiveAfter")
        != config["forward"]["prospectiveAfter"]
        or bundle.get("hashMode") != "canonical_json_sha256"
    ):
        raise RuntimeError("frozen model safety/version contract mismatch")
    return config, bundle, actual_model_hash


def minute_of_day(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def in_configured_session(timestamp: pd.Timestamp, input_cfg: dict[str, Any]) -> bool:
    value = timestamp.hour * 60 + timestamp.minute
    return (
        minute_of_day(input_cfg["morningStart"])
        <= value
        < minute_of_day(input_cfg["morningEndExclusive"])
    ) or (
        minute_of_day(input_cfg["afternoonStart"])
        <= value
        < minute_of_day(input_cfg["afternoonEndExclusive"])
    )


def load_snapshot_panel(
    snapshot_dir: Path,
    trade_date: str,
    universe: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_cfg = config["input"]
    files = sorted(snapshot_dir.glob(input_cfg["filePattern"]))
    if not files:
        raise FileNotFoundError(f"no snapshot files in {snapshot_dir}")
    selected = set(universe)
    latest: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    corrupt_files: list[str] = []
    source_rows = 0
    for path in files:
        try:
            with gzip.open(
                path, "rt", encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    code = str(row.get("stockCode") or "").zfill(6)
                    if code not in selected:
                        continue
                    stamp_text = str(row.get(input_cfg["timestampField"]) or "")
                    if not stamp_text:
                        continue
                    try:
                        source_time = pd.Timestamp(stamp_text)
                        price = float(row.get(input_cfg["priceField"]) or 0.0)
                        amount = float(
                            row.get(input_cfg["cumulativeAmountField"]) or 0.0
                        )
                    except (TypeError, ValueError):
                        continue
                    if (
                        source_time.strftime("%Y-%m-%d") != trade_date
                        or not in_configured_session(source_time, input_cfg)
                        or not np.isfinite(price)
                        or price <= 0
                    ):
                        continue
                    source_rows += 1
                    bucket = source_time.floor(
                        f"{int(config['forward']['barIntervalMinutes'])}min"
                    )
                    key = (bucket, code)
                    if key not in latest or source_time > latest[key]["source_time"]:
                        latest[key] = {
                            "timestamp": bucket,
                            "trade_date": trade_date,
                            "stockCode": code,
                            "close": price,
                            "cumulative_amount": max(amount, 0.0),
                            "source_time": source_time,
                        }
        except (EOFError, OSError, UnicodeError, csv.Error) as exc:
            corrupt_files.append(f"{path.name}: {exc}")
    if not latest:
        raise RuntimeError(f"no selected-universe rows in {snapshot_dir}")
    rows = list(latest.values())
    panel = pd.DataFrame.from_records(rows).sort_values(
        ["timestamp", "stockCode"], kind="stable"
    )
    counts = panel.groupby("timestamp")["stockCode"].nunique().sort_index()
    coverage = counts / max(len(universe), 1)
    last_source = max(row["source_time"] for row in rows)
    last_required = pd.Timestamp(
        f"{trade_date}T{input_cfg['minimumLastSourceTime']}:00+08:00"
    )
    audit = {
        "snapshotDirectory": str(snapshot_dir.resolve()),
        "snapshotFiles": len(files),
        "corruptFiles": corrupt_files,
        "rawSelectedRows": source_rows,
        "normalizedRows": int(len(panel)),
        "fiveMinuteBars": int(len(counts)),
        "universeSize": len(universe),
        "medianUniverseCoverageFraction": float(coverage.median()),
        "minimumUniverseCoverageFraction": float(coverage.min()),
        "maximumUniverseCoverageFraction": float(coverage.max()),
        "lastSourceTime": last_source.isoformat(),
        "panelSha256": sha256_text(
            panel[
                [
                    "timestamp",
                    "stockCode",
                    "close",
                    "cumulative_amount",
                    "source_time",
                ]
            ].to_csv(index=False)
        ),
    }
    gates = {
        "minimumFiveMinuteBars": int(len(counts))
        >= int(input_cfg["minimumFiveMinuteBars"]),
        "minimumMedianUniverseCoverage": float(coverage.median())
        >= float(input_cfg["minimumMedianUniverseCoverageFraction"]),
        "minimumLastSourceTime": last_source >= last_required,
    }
    audit["qualityGates"] = gates
    audit["qualityPass"] = all(gates.values())
    if not audit["qualityPass"]:
        raise RuntimeError(f"forward snapshot quality failed: {gates}")
    return panel.drop(columns=["source_time"]), audit


def build_frozen_ews_frames(
    base_frames: dict[str, Any],
    phase1_config: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, pd.DataFrame | pd.Series]:
    window = int(phase1_config["features"]["ewsWindowBars"])
    floor = float(phase1_config["features"]["ewsVarianceFloor"])
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
    components: dict[str, pd.DataFrame | pd.Series] = {
        "ews_log_variance": np.log(variance.clip(lower=floor)),
        "ews_autocorrelation_1": autocorrelation,
        "ews_variance_slope": rolling_variance_slope(ret_1.pow(2), window),
        "ews_market_dispersion": ret_1.std(axis=1, skipna=True),
    }
    statistics = bundle["frozenEwsStandardization"]
    score_frames: list[pd.DataFrame] = []
    for name, frame in components.items():
        stats = statistics[name]
        z = (
            (frame - float(stats["mean"]))
            / float(stats["standardDeviation"])
        ).clip(-8.0, 8.0)
        if isinstance(z, pd.Series):
            z = pd.DataFrame(
                np.repeat(
                    z.to_numpy(dtype=float)[:, None],
                    len(ret_1.columns),
                    axis=1,
                ),
                index=ret_1.index,
                columns=ret_1.columns,
            )
        score_frames.append(1.0 / (1.0 + np.exp(-z)))
    components["ews_score"] = sum(score_frames) / len(score_frames)
    return components


def build_frozen_hmm_frames(
    base_frames: dict[str, Any], bundle: dict[str, Any]
) -> dict[str, pd.Series]:
    frozen = bundle["frozenHmm"]
    hmm = GaussianHMM1D(
        n_states=int(frozen["states"]),
        n_iter=0,
        variance_floor=float(
            bundle["featureConfiguration"]["hmm"]["varianceFloor"]
        ),
    )
    hmm.means = np.asarray(frozen["means"], dtype=float)
    hmm.variances = np.asarray(frozen["variances"], dtype=float)
    hmm.start_probability = np.asarray(
        frozen["startProbability"], dtype=float
    )
    hmm.transition = np.asarray(frozen["transitionMatrix"], dtype=float)
    market_return: pd.Series = base_frames["market_ret_1"]
    dates = pd.Series(
        market_return.index.strftime("%Y-%m-%d"), index=market_return.index
    )
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
    expected = values @ hmm.means
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
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def score_frozen_model(
    rows: pd.DataFrame, model: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(model["features"])
    values = rows[columns].to_numpy(dtype=float)
    mean = np.asarray(model["scalerMean"], dtype=float)
    scale = np.asarray(model["scalerScale"], dtype=float)
    coefficient = np.asarray(model["logisticCoefficient"], dtype=float)
    if not (
        values.shape[1] == len(mean) == len(scale) == len(coefficient)
    ):
        raise RuntimeError("frozen model dimension mismatch")
    standardized = (values - mean) / scale
    logit = (
        standardized @ coefficient + float(model["logisticIntercept"])
    )
    raw = np.clip(sigmoid(logit), 1e-6, 1.0 - 1e-6)
    calibrated = sigmoid(
        float(model["plattCoefficient"])
        * np.log(raw / (1.0 - raw))
        + float(model["plattIntercept"])
    )
    return raw, calibrated


def score_all_variants(
    feature_table: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    keys = ["timestamp", "trade_date", "stockCode"]
    parts: list[pd.DataFrame] = []
    for horizon, horizon_bundle in bundle["modelsByHorizon"].items():
        for variant, model in horizon_bundle["models"].items():
            raw, probability = score_frozen_model(feature_table, model)
            part = feature_table[keys].copy()
            part["horizon_bars"] = int(horizon)
            part["variant"] = variant
            part["raw_probability"] = raw
            part["probability"] = probability
            parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["trade_date", "timestamp", "stockCode", "horizon_bars", "variant"],
        kind="stable",
    )


def build_probability_artifact(
    predictions: pd.DataFrame,
    feature_table: pd.DataFrame,
    config: dict[str, Any],
    model_hash: str,
) -> list[dict[str, Any]]:
    keys = ["timestamp", "trade_date", "stockCode"]
    lookup = {
        (row.timestamp, row.stockCode, int(row.horizon_bars), row.variant): float(
            row.probability
        )
        for row in predictions.itertuples(index=False)
    }
    primary_variant = "hmm_ews"
    active = (
        list(config["forward"]["primaryHorizonsBars"])
        + list(config["forward"]["observationalHorizonsBars"])
    )
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for feature in feature_table.itertuples(index=False):
        timestamp = pd.Timestamp(feature.timestamp)
        nested: dict[str, dict[str, float]] = {}
        for variant in config["evaluation"]["variants"]:
            nested[variant] = {
                str(horizon): lookup[
                    (timestamp, feature.stockCode, int(horizon), variant)
                ]
                for horizon in active
            }
        rows.append(
            {
                "schemaVersion": "singularity_phase1_5_turning_probability_v1",
                "status": "research_only",
                "shadowOnly": True,
                "diagnosticOnly": True,
                "version": config["version"],
                "modelSha256": model_hash,
                "generatedAt": generated,
                "tradeDate": feature.trade_date,
                "timestamp": timestamp.isoformat(),
                "featureAvailableAt": (
                    timestamp
                    + pd.Timedelta(
                        minutes=int(config["forward"]["barIntervalMinutes"])
                    )
                ).isoformat(),
                "stockCode": feature.stockCode,
                "probabilities": nested,
                "p_turning_5": nested[primary_variant].get("5"),
                "p_turning_10": nested[primary_variant].get("10"),
                "p_turning_20": nested[primary_variant].get("20"),
                "p_turning_30": nested[primary_variant].get("30"),
                "p_turning_60": None,
                "p_turning_60_status": "skipped_cross_session",
                "ews_score": float(feature.ews_score),
                "regime_transition_risk": float(
                    feature.regime_transition_risk
                ),
                "regime_entropy": float(feature.regime_entropy),
                "singularity_score": float(feature.singularity_score),
                "orderInstruction": None,
                "positionSizeInstruction": None,
                "gateInstruction": None,
            }
        )
    return rows


def build_outcomes(
    label_table: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active = (
        list(config["forward"]["primaryHorizonsBars"])
        + list(config["forward"]["observationalHorizonsBars"])
    )
    for label in label_table.itertuples(index=False):
        source = label._asdict()
        for horizon in active:
            target = source[f"turning_point_{horizon}"]
            if pd.isna(target):
                continue
            rows.append(
                {
                    "schemaVersion": "singularity_phase1_5_turning_outcome_v1",
                    "status": "research_only_offline_label",
                    "version": config["version"],
                    "tradeDate": source["trade_date"],
                    "timestamp": pd.Timestamp(source["timestamp"]).isoformat(),
                    "stockCode": source["stockCode"],
                    "horizonBars": int(horizon),
                    "turningPoint": int(target),
                    "reversalUp": int(source[f"reversal_up_{horizon}"]),
                    "reversalDown": int(source[f"reversal_down_{horizon}"]),
                    "failedBreakout": int(
                        source[f"failed_breakout_{horizon}"]
                    ),
                    "trendExhaustion": int(
                        source[f"trend_exhaustion_{horizon}"]
                    ),
                    "stopChase": int(source[f"stop_chase_{horizon}"]),
                    "futureMaxReturn": float(
                        source[f"future_max_return_{horizon}"]
                    ),
                    "futureMinReturn": float(
                        source[f"future_min_return_{horizon}"]
                    ),
                    "futureCloseReturn": float(
                        source[f"future_close_return_{horizon}"]
                    ),
                    "labelThreshold": float(
                        source[f"label_threshold_{horizon}"]
                    ),
                    "usedForFeatureConstruction": False,
                }
            )
    return rows


def metric_payload(
    rows: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    edges = [float(value) for value in config["evaluation"]["probabilityBinEdges"]]
    metrics = probability_metrics(
        rows["probability"].to_numpy(dtype=float),
        rows["actual"].to_numpy(dtype=int),
        bin_edges=edges,
    )
    threshold = float(config["evaluation"]["highRiskThreshold"])
    predicted = rows["probability"].to_numpy(dtype=float) >= threshold
    actual = rows["actual"].to_numpy(dtype=int) == 1
    negatives = ~actual
    false_positive = int(np.sum(predicted & negatives))
    metrics.update(
        {
            "highRiskThreshold": threshold,
            "highRiskCount": int(np.sum(predicted)),
            "highRiskMeanProbability": (
                float(
                    np.mean(
                        rows["probability"].to_numpy(dtype=float)[predicted]
                    )
                )
                if np.any(predicted)
                else None
            ),
            "highRiskHitRate": (
                float(np.mean(actual[predicted])) if np.any(predicted) else None
            ),
            "falsePositiveCount": false_positive,
            "falsePositiveRate": (
                false_positive / int(np.sum(negatives))
                if np.sum(negatives)
                else None
            ),
        }
    )
    return metrics


def build_evaluation_long(
    predictions: pd.DataFrame,
    label_table: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["timestamp", "trade_date", "stockCode"]
    feature_subset = feature_table[keys + ["ret_6"]]
    parts: list[pd.DataFrame] = []
    for horizon in sorted(predictions["horizon_bars"].unique()):
        label_columns = keys + [
            f"turning_point_{horizon}",
            f"reversal_down_{horizon}",
            f"stop_chase_{horizon}",
            f"future_max_return_{horizon}",
        ]
        labels = label_table[label_columns].rename(
            columns={
                f"turning_point_{horizon}": "actual",
                f"reversal_down_{horizon}": "reversal_down",
                f"stop_chase_{horizon}": "stop_chase",
                f"future_max_return_{horizon}": "future_max_return",
            }
        )
        current = predictions[
            predictions["horizon_bars"] == horizon
        ].merge(labels, on=keys, how="inner")
        current = current.merge(feature_subset, on=keys, how="left")
        current = current.dropna(
            subset=[
                "actual",
                "reversal_down",
                "stop_chase",
                "future_max_return",
                "ret_6",
            ]
        )
        current["actual"] = current["actual"].astype(int)
        current["reversal_down"] = current["reversal_down"].astype(int)
        current["stop_chase"] = current["stop_chase"].astype(int)
        parts.append(current)
    return pd.concat(parts, ignore_index=True)


def counterfactual_diagnostics(
    rows: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    threshold = float(config["evaluation"]["highRiskThreshold"])
    chase_minimum = float(
        json.loads(resolve(config["phase1"]["config"]).read_text(encoding="utf-8"))[
            "labels"
        ]["chaseMinimumPct"]
    )
    result: dict[str, Any] = {}
    configured_horizons = (
        list(config["forward"]["primaryHorizonsBars"])
        + list(config["forward"]["observationalHorizonsBars"])
    )
    for horizon in configured_horizons:
        result[str(int(horizon))] = {}
        horizon_rows = rows[rows["horizon_bars"] == horizon]
        for variant in config["evaluation"]["variants"]:
            current = horizon_rows[horizon_rows["variant"] == variant].copy()
            high_risk = current["probability"] >= threshold
            chase = current["ret_6"] >= chase_minimum
            remaining_chase = chase & ~high_risk
            flagged_reversal_down = high_risk & (
                current["reversal_down"] == 1
            )
            efficiency = 1.0 / (
                1.0
                + np.maximum(
                    current.loc[
                        flagged_reversal_down, "future_max_return"
                    ].to_numpy(dtype=float),
                    0.0,
                )
            )
            result[str(int(horizon))][variant] = {
                "chaseCandidates": int(np.sum(chase)),
                "chaseFailureRateAll": (
                    float(current.loc[chase, "stop_chase"].mean())
                    if np.any(chase)
                    else None
                ),
                "highRiskChaseCount": int(np.sum(chase & high_risk)),
                "highRiskChaseFailureRate": (
                    float(current.loc[chase & high_risk, "stop_chase"].mean())
                    if np.any(chase & high_risk)
                    else None
                ),
                "remainingChaseCountIfExcluded": int(
                    np.sum(remaining_chase)
                ),
                "remainingChaseFailureRateIfExcluded": (
                    float(
                        current.loc[remaining_chase, "stop_chase"].mean()
                    )
                    if np.any(remaining_chase)
                    else None
                ),
                "counterfactualOnlyNoGateApplied": True,
                "flaggedReversalDownCount": int(
                    np.sum(flagged_reversal_down)
                ),
                "sellEfficiencyProxyMean": (
                    float(np.mean(efficiency)) if len(efficiency) else None
                ),
                "sellEfficiencyDefinition": (
                    "decision_price / max(decision_price, future_horizon_high)"
                ),
                "actualSellExecutionEvaluated": False,
            }
    return result


def summarize_daily(
    evaluation: pd.DataFrame,
    data_audit: dict[str, Any],
    config: dict[str, Any],
    model_hash: str,
    smoke_test: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    count_matrix: dict[str, Any] = {}
    configured_horizons = (
        list(config["forward"]["primaryHorizonsBars"])
        + list(config["forward"]["observationalHorizonsBars"])
    )
    for horizon in configured_horizons:
        metrics[str(int(horizon))] = {}
        count_matrix[str(int(horizon))] = {}
        for variant in config["evaluation"]["variants"]:
            rows = evaluation[
                (evaluation["horizon_bars"] == horizon)
                & (evaluation["variant"] == variant)
            ]
            metrics[str(int(horizon))][variant] = metric_payload(rows, config)
            count_matrix[str(int(horizon))][variant] = int(len(rows))
    equal_population = all(
        len(set(counts.values())) == 1 for counts in count_matrix.values()
    )
    return {
        "schemaVersion": "singularity_phase1_5_daily_result_v1",
        "status": "research_only_smoke_test" if smoke_test else "research_only",
        "shadowOnly": True,
        "diagnosticOnly": True,
        "version": config["version"],
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tradeDate": str(evaluation["trade_date"].iloc[0]),
        "modelSha256": model_hash,
        "smokeTestExcludedFromForwardLedger": smoke_test,
        "dataAudit": data_audit,
        "metricsByHorizonAndVariant": metrics,
        "counterfactualDiagnostics": counterfactual_diagnostics(
            evaluation, config
        ),
        "mechanicalFrequencyCheck": {
            "tradingApplied": False,
            "gateApplied": False,
            "equalForecastPopulationAcrossVariants": equal_population,
            "outcomeCounts": count_matrix,
        },
        "horizonPriority": {
            "primary": config["forward"]["primaryHorizonsBars"],
            "observationalOnly": config["forward"][
                "observationalHorizonsBars"
            ],
            "skippedNull": config["forward"]["skippedHorizonsBars"],
        },
        "productionImpact": {
            "orders": False,
            "positions": False,
            "buildDecision": False,
            "buySellGates": False,
            "riskGates": False,
        },
    }


def format_value(value: Any, digits: int = 6) -> str:
    return "null" if value is None else f"{float(value):.{digits}f}"


def render_daily_report(result: dict[str, Any]) -> str:
    lines = [
        "# Singularity Phase 1.5 daily shadow report",
        "",
        f"- Date: `{result['tradeDate']}`",
        f"- Status: `{result['status']}`",
        f"- Model: `{result['version']}` / `{result['modelSha256'][:12]}`",
        f"- Five-minute bars: {result['dataAudit']['fiveMinuteBars']}",
        f"- Median universe coverage: {result['dataAudit']['medianUniverseCoverageFraction']:.2%}",
        "",
        "Post-close reconstruction only. No order, position, risk gate, "
        "BUY/SELL gate or build_decision integration.",
        "",
        "## Primary horizons",
        "",
        "| h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n | FPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in result["horizonPriority"]["primary"]:
        for variant, metrics in result["metricsByHorizonAndVariant"][
            str(horizon)
        ].items():
            lines.append(
                f"| {horizon} | {variant} | {metrics['count']} | "
                f"{format_value(metrics['brier'])} | "
                f"{format_value(metrics['log_loss'])} | "
                f"{format_value(metrics['auc'])} | "
                f"{format_value(metrics['ece'])} | "
                f"{metrics['highRiskCount']} | "
                f"{format_value(metrics['falsePositiveRate'])} |"
            )
    lines.extend(
        [
            "",
            "20/30-bar outputs are observation-only. Horizon 60 remains null.",
            "Chase and sell-efficiency fields are counterfactual diagnostics, "
            "not proposed trading rules.",
            "",
        ]
    )
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_ledger(
    output_root: Path, result: dict[str, Any], config: dict[str, Any]
) -> None:
    ledger_path = output_root / config["output"]["ledgerFile"]
    rows = read_jsonl(ledger_path)
    stable = {
        "schemaVersion": "singularity_phase1_5_forward_day_v1",
        "tradeDate": result["tradeDate"],
        "version": result["version"],
        "modelSha256": result["modelSha256"],
        "status": result["status"],
        "dataAudit": result["dataAudit"],
        "mechanicalFrequencyCheck": result["mechanicalFrequencyCheck"],
        "metricsByHorizonAndVariant": result["metricsByHorizonAndVariant"],
        "counterfactualDiagnostics": result["counterfactualDiagnostics"],
        "researchOnly": True,
        "shadowOnly": True,
    }
    existing = next(
        (row for row in rows if row["tradeDate"] == result["tradeDate"]), None
    )
    if existing is not None and existing != stable:
        raise RuntimeError(
            f"refusing to revise sealed forward day {result['tradeDate']}"
        )
    rows = [row for row in rows if row["tradeDate"] != result["tradeDate"]]
    rows.append(stable)
    atomic_jsonl(
        ledger_path, sorted(rows, key=lambda row: row["tradeDate"])
    )


def load_cumulative_long(
    output_root: Path, config: dict[str, Any]
) -> pd.DataFrame:
    daily_root = output_root / config["output"]["dailyDirectory"]
    records: list[dict[str, Any]] = []
    if not daily_root.exists():
        return pd.DataFrame()
    for day_dir in sorted(path for path in daily_root.iterdir() if path.is_dir()):
        probability_path = day_dir / config["output"]["probabilityFile"]
        outcome_path = day_dir / config["output"]["outcomeFile"]
        if not probability_path.exists() or not outcome_path.exists():
            continue
        probabilities = {
            (row["timestamp"], row["stockCode"]): row
            for row in read_jsonl(probability_path)
        }
        for outcome in read_jsonl(outcome_path):
            forecast = probabilities.get(
                (outcome["timestamp"], outcome["stockCode"])
            )
            if forecast is None:
                continue
            horizon = str(outcome["horizonBars"])
            for variant in config["evaluation"]["variants"]:
                records.append(
                    {
                        "trade_date": outcome["tradeDate"],
                        "timestamp": outcome["timestamp"],
                        "stockCode": outcome["stockCode"],
                        "horizon_bars": int(horizon),
                        "variant": variant,
                        "probability": float(
                            forecast["probabilities"][variant][horizon]
                        ),
                        "actual": int(outcome["turningPoint"]),
                    }
                )
    return pd.DataFrame.from_records(records)


def window_comparisons(
    rows: pd.DataFrame,
    horizon: int,
    candidate: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = rows[rows["horizon_bars"] == horizon].copy()
    selected["iso_week"] = pd.to_datetime(
        selected["trade_date"]
    ).dt.strftime("%G-W%V")
    result: dict[str, Any] = {}
    for window_name, group_column in (
        ("daily", "trade_date"),
        ("weekly", "iso_week"),
    ):
        comparisons: list[dict[str, Any]] = []
        for window, values in selected.groupby(group_column):
            baseline = values[values["variant"] == "baseline"]
            current = values[values["variant"] == candidate]
            if len(baseline) == 0 or len(current) == 0:
                continue
            base_metric = metric_payload(baseline, config)
            cand_metric = metric_payload(current, config)
            comparisons.append(
                {
                    "window": str(window),
                    "deltaBrier": cand_metric["brier"]
                    - base_metric["brier"],
                    "deltaLogLoss": cand_metric["log_loss"]
                    - base_metric["log_loss"],
                    "deltaAuc": (
                        None
                        if cand_metric["auc"] is None
                        or base_metric["auc"] is None
                        else cand_metric["auc"] - base_metric["auc"]
                    ),
                }
            )
        eligible_auc = [
            row for row in comparisons if row["deltaAuc"] is not None
        ]
        result[window_name] = {
            "windows": len(comparisons),
            "brierImprovementFraction": (
                sum(row["deltaBrier"] < 0 for row in comparisons)
                / len(comparisons)
                if comparisons
                else None
            ),
            "logLossImprovementFraction": (
                sum(row["deltaLogLoss"] < 0 for row in comparisons)
                / len(comparisons)
                if comparisons
                else None
            ),
            "aucImprovementFraction": (
                sum(row["deltaAuc"] > 0 for row in eligible_auc)
                / len(eligible_auc)
                if eligible_auc
                else None
            ),
            "details": comparisons,
        }
    return result


def aggregate_counterfactual_monitoring(
    output_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    ledger = read_jsonl(output_root / config["output"]["ledgerFile"])
    result: dict[str, Any] = {}
    horizons = (
        list(config["forward"]["primaryHorizonsBars"])
        + list(config["forward"]["observationalHorizonsBars"])
    )
    for horizon in horizons:
        result[str(horizon)] = {}
        for variant in config["evaluation"]["variants"]:
            nodes = [
                row["counterfactualDiagnostics"][str(horizon)][variant]
                for row in ledger
                if str(horizon) in row.get("counterfactualDiagnostics", {})
            ]

            def weighted_rate(count_key: str, rate_key: str) -> tuple[int, float | None]:
                count = sum(int(node.get(count_key) or 0) for node in nodes)
                events = sum(
                    int(node.get(count_key) or 0) * float(node.get(rate_key))
                    for node in nodes
                    if node.get(rate_key) is not None
                )
                return count, (events / count if count else None)

            chase_count, chase_rate = weighted_rate(
                "chaseCandidates", "chaseFailureRateAll"
            )
            high_count, high_rate = weighted_rate(
                "highRiskChaseCount", "highRiskChaseFailureRate"
            )
            remaining_count, remaining_rate = weighted_rate(
                "remainingChaseCountIfExcluded",
                "remainingChaseFailureRateIfExcluded",
            )
            sell_count, sell_efficiency = weighted_rate(
                "flaggedReversalDownCount", "sellEfficiencyProxyMean"
            )
            result[str(horizon)][variant] = {
                "chaseCandidates": chase_count,
                "chaseFailureRateAll": chase_rate,
                "highRiskChaseCount": high_count,
                "highRiskChaseFailureRate": high_rate,
                "remainingChaseCountIfExcluded": remaining_count,
                "remainingChaseFailureRateIfExcluded": remaining_rate,
                "remainingMinusAllFailureRate": (
                    None
                    if chase_rate is None or remaining_rate is None
                    else remaining_rate - chase_rate
                ),
                "flaggedReversalDownCount": sell_count,
                "sellEfficiencyProxyMean": sell_efficiency,
                "actualTradingRuleApplied": False,
            }
        baseline_efficiency = result[str(horizon)]["baseline"][
            "sellEfficiencyProxyMean"
        ]
        for variant in config["evaluation"]["variants"]:
            current = result[str(horizon)][variant]["sellEfficiencyProxyMean"]
            result[str(horizon)][variant]["sellEfficiencyDeltaVsBaseline"] = (
                None
                if current is None or baseline_efficiency is None
                else current - baseline_efficiency
            )
    return result


def build_forward_validation(
    output_root: Path,
    config: dict[str, Any],
    model_hash: str,
) -> dict[str, Any]:
    rows = load_cumulative_long(output_root, config)
    minimum_days = int(config["forward"]["minimumIndependentTradingDays"])
    days = (
        sorted(rows["trade_date"].unique()) if not rows.empty else []
    )
    report: dict[str, Any] = {
        "schemaVersion": "singularity_phase1_5_forward_validation_v1",
        "status": (
            "insufficient_forward_days"
            if len(days) < minimum_days
            else "forward_review_available"
        ),
        "researchOnly": True,
        "shadowOnly": True,
        "version": config["version"],
        "modelSha256": model_hash,
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prospectiveAfter": config["forward"]["prospectiveAfter"],
        "independentTradingDays": len(days),
        "minimumIndependentTradingDays": minimum_days,
        "daysRemaining": max(0, minimum_days - len(days)),
        "primaryHorizons": config["forward"]["primaryHorizonsBars"],
        "observationalHorizons": config["forward"][
            "observationalHorizonsBars"
        ],
        "horizon60": "skipped_null",
        "metrics": {},
        "comparisons": {},
        "counterfactualMonitoring": aggregate_counterfactual_monitoring(
            output_root, config
        ),
        "phase2DiscussionAllowed": False,
        "automaticPromotionAllowed": False,
        "invariantAttestation": "manual_full_suite_and_T88_required",
    }
    if rows.empty:
        report["conclusion"] = (
            "Waiting for the first eligible post-2026-07-05 session."
        )
        return report
    configured_horizons = (
        list(config["forward"]["primaryHorizonsBars"])
        + list(config["forward"]["observationalHorizonsBars"])
    )
    for horizon in configured_horizons:
        report["metrics"][str(int(horizon))] = {}
        for variant in config["evaluation"]["variants"]:
            current = rows[
                (rows["horizon_bars"] == horizon)
                & (rows["variant"] == variant)
            ]
            report["metrics"][str(int(horizon))][variant] = metric_payload(
                current, config
            )
    majority = float(config["evaluation"]["majorityFraction"])
    min_high_risk = int(
        config["evaluation"]["minimumHighRiskSamplesPerPrimaryHorizon"]
    )
    max_optimism = float(config["evaluation"]["maximumHighRiskOptimism"])
    candidate_passes: dict[str, Any] = {}
    for candidate in config["evaluation"]["primaryCandidateVariants"]:
        candidate_passes[candidate] = {}
        for horizon in config["forward"]["primaryHorizonsBars"]:
            comparison = window_comparisons(
                rows, int(horizon), candidate, config
            )
            metrics = report["metrics"][str(horizon)][candidate]
            high_count = int(metrics["highRiskCount"])
            high_optimism = (
                None
                if metrics["highRiskHitRate"] is None
                else metrics["highRiskMeanProbability"]
                - metrics["highRiskHitRate"]
            )

            def metric_majority(name: str) -> bool:
                return any(
                    window[name] is not None and window[name] > majority
                    for window in comparison.values()
                )

            gates = {
                "brierMajority": metric_majority(
                    "brierImprovementFraction"
                ),
                "logLossMajority": metric_majority(
                    "logLossImprovementFraction"
                ),
                "aucMajority": metric_majority("aucImprovementFraction"),
                "highRiskSamples": high_count >= min_high_risk,
                "highRiskNotOverOptimistic": (
                    high_optimism is not None
                    and high_optimism <= max_optimism
                ),
            }
            candidate_passes[candidate][str(horizon)] = {
                "windowComparison": comparison,
                "highRiskCount": high_count,
                "highRiskOptimismApproximation": high_optimism,
                "gates": gates,
                "statisticalPass": all(gates.values()),
            }
    report["comparisons"] = candidate_passes
    population_equal = all(
        len(
            {
                report["metrics"][str(horizon)][variant]["count"]
                for variant in config["evaluation"]["variants"]
            }
        )
        == 1
        for horizon in (
            config["forward"]["primaryHorizonsBars"]
            + config["forward"]["observationalHorizonsBars"]
        )
    )
    report["mechanicalFrequencyCheck"] = {
        "equalForecastPopulationAcrossVariants": population_equal,
        "tradingApplied": False,
        "gateApplied": False,
    }
    candidates_with_both_primary = [
        candidate
        for candidate, horizons in candidate_passes.items()
        if all(
            horizons[str(horizon)]["statisticalPass"]
            for horizon in config["forward"]["primaryHorizonsBars"]
        )
    ]
    report["phase2PreconditionsExceptManualInvariants"] = bool(
        len(days) >= minimum_days
        and population_equal
        and candidates_with_both_primary
    )
    report["phase2DiscussionAllowed"] = False
    report["conclusion"] = (
        "Insufficient independent forward days; no Phase 2 discussion."
        if len(days) < minimum_days
        else "Statistical monitoring is available, but Phase 2 remains blocked "
        "until every preregistered condition and a fresh invariant/T88 "
        "attestation pass in a separate human review."
    )
    return report


def render_forward_validation(report: dict[str, Any]) -> str:
    lines = [
        "# Singularity Phase 1.5 forward validation",
        "",
        f"- Status: `{report['status']}`",
        f"- Independent forward days: {report['independentTradingDays']}/{report['minimumIndependentTradingDays']}",
        f"- Days remaining: {report['daysRemaining']}",
        f"- Phase 2 discussion allowed: `{report['phase2DiscussionAllowed']}`",
        "",
        "Research-only, shadow-only, no online inference or trading integration.",
    ]
    if report.get("metrics"):
        lines.extend(
            [
                "",
                "## Primary-horizon cumulative metrics",
                "",
                "| h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n | FPR |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for horizon in report["primaryHorizons"]:
            for variant, metrics in report["metrics"][str(horizon)].items():
                lines.append(
                    f"| {horizon} | {variant} | {metrics['count']} | "
                    f"{format_value(metrics['brier'])} | "
                    f"{format_value(metrics['log_loss'])} | "
                    f"{format_value(metrics['auc'])} | "
                    f"{format_value(metrics['ece'])} | "
                    f"{metrics['highRiskCount']} | "
                    f"{format_value(metrics['falsePositiveRate'])} |"
                )
        lines.extend(
            [
                "",
                "## Primary-horizon window stability",
                "",
                "| h | candidate | daily Brier/LogLoss/AUC improve | weekly Brier/LogLoss/AUC improve |",
                "|---:|---|---|---|",
            ]
        )
        for candidate, horizons in report["comparisons"].items():
            for horizon in report["primaryHorizons"]:
                windows = horizons[str(horizon)]["windowComparison"]
                daily = windows["daily"]
                weekly = windows["weekly"]
                lines.append(
                    f"| {horizon} | {candidate} | "
                    f"{format_value(daily['brierImprovementFraction'], 3)}/"
                    f"{format_value(daily['logLossImprovementFraction'], 3)}/"
                    f"{format_value(daily['aucImprovementFraction'], 3)} | "
                    f"{format_value(weekly['brierImprovementFraction'], 3)}/"
                    f"{format_value(weekly['logLossImprovementFraction'], 3)}/"
                    f"{format_value(weekly['aucImprovementFraction'], 3)} |"
                )
        lines.extend(
            [
                "",
                "## Counterfactual monitoring",
                "",
                "| h | variant | chase n | all fail rate | remaining n | remaining fail rate | flagged reversal-down n | sell-efficiency proxy |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for horizon in report["primaryHorizons"]:
            for variant, diagnostics in report[
                "counterfactualMonitoring"
            ][str(horizon)].items():
                lines.append(
                    f"| {horizon} | {variant} | "
                    f"{diagnostics['chaseCandidates']} | "
                    f"{format_value(diagnostics['chaseFailureRateAll'])} | "
                    f"{diagnostics['remainingChaseCountIfExcluded']} | "
                    f"{format_value(diagnostics['remainingChaseFailureRateIfExcluded'])} | "
                    f"{diagnostics['flaggedReversalDownCount']} | "
                    f"{format_value(diagnostics['sellEfficiencyProxyMean'])} |"
                )
    lines.extend(
        [
            "",
            report["conclusion"],
            "",
            "20/30 bars remain observation-only; 60 bars remain null. "
            "No report can automatically promote a model.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_validation_outputs(
    output_root: Path,
    config: dict[str, Any],
    model_hash: str,
) -> dict[str, Any]:
    report = build_forward_validation(output_root, config, model_hash)
    atomic_json(
        output_root / config["output"]["validationJson"], report
    )
    atomic_text(
        output_root / config["output"]["validationMarkdown"],
        render_forward_validation(report),
    )
    return report


def latest_eligible_snapshot_date(
    snapshot_root: Path, prospective_after: str
) -> str | None:
    dates = sorted(
        path.name
        for path in snapshot_root.iterdir()
        if path.is_dir()
        and len(path.name) == 10
        and path.name > prospective_after
    ) if snapshot_root.exists() else []
    return dates[-1] if dates else None


def process_day(
    trade_date: str,
    config: dict[str, Any],
    bundle: dict[str, Any],
    model_hash: str,
    output_root: Path,
    smoke_test: bool,
) -> dict[str, Any]:
    prospective_after = str(config["forward"]["prospectiveAfter"])
    if trade_date <= prospective_after and not smoke_test:
        raise ValueError(
            f"official forward date must be after {prospective_after}"
        )
    snapshot_dir = resolve(config["input"]["snapshotRoot"]) / trade_date
    panel, data_audit = load_snapshot_panel(
        snapshot_dir, trade_date, list(bundle["universe"]), config
    )
    phase1_config = {
        "data": {"trainEnd": bundle["historicalCutoff"]},
        "features": bundle["featureConfiguration"],
        "labels": bundle["labelConfiguration"],
        "models": bundle["modelConfiguration"],
    }
    base_frames = build_feature_frames(panel)
    ews_frames = build_frozen_ews_frames(
        base_frames, phase1_config, bundle
    )
    hmm_frames = build_frozen_hmm_frames(base_frames, bundle)
    feature_table = build_feature_table(
        base_frames, ews_frames, hmm_frames, phase1_config
    )
    label_table, _ = build_label_table(base_frames, phase1_config)
    if feature_table.empty or label_table.empty:
        raise RuntimeError("forward feature or label table is empty")
    predictions = score_all_variants(feature_table, bundle)
    probability_rows = build_probability_artifact(
        predictions, feature_table, config, model_hash
    )
    outcome_rows = build_outcomes(label_table, config)
    evaluation = build_evaluation_long(
        predictions, label_table, feature_table
    )
    result = summarize_daily(
        evaluation, data_audit, config, model_hash, smoke_test
    )
    base_dir = output_root / (
        config["output"]["smokeDirectory"]
        if smoke_test
        else config["output"]["dailyDirectory"]
    )
    day_dir = base_dir / trade_date
    existing_result = day_dir / config["output"]["dailyResultFile"]
    if existing_result.exists():
        existing = json.loads(existing_result.read_text(encoding="utf-8"))
        if (
            existing.get("modelSha256") != model_hash
            or existing.get("dataAudit", {}).get("panelSha256")
            != data_audit["panelSha256"]
        ):
            raise RuntimeError(
                f"refusing to revise existing sealed day: {trade_date}"
            )
    atomic_jsonl(
        day_dir / config["output"]["probabilityFile"], probability_rows
    )
    atomic_jsonl(day_dir / config["output"]["outcomeFile"], outcome_rows)
    atomic_json(existing_result, result)
    atomic_text(
        day_dir / config["output"]["dailyReportFile"],
        render_daily_report(result),
    )
    if not smoke_test:
        write_ledger(output_root, result, config)
        refresh_validation_outputs(
            output_root, config, model_hash
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--date", default="")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="allow a known historical date but exclude it from the forward ledger",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, bundle, model_hash = load_config_bundle(args.config.resolve())
    output_root = resolve(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_root = resolve(config["input"]["snapshotRoot"])
    trade_date = args.date or latest_eligible_snapshot_date(
        snapshot_root, str(config["forward"]["prospectiveAfter"])
    )
    if trade_date is None:
        report = refresh_validation_outputs(
            output_root, config, model_hash
        )
        status = {
            "schemaVersion": "singularity_phase1_5_status_v1",
            "status": "waiting_for_first_forward_session",
            "researchOnly": True,
            "shadowOnly": True,
            "prospectiveAfter": config["forward"]["prospectiveAfter"],
            "firstEligibleDate": config["forward"]["firstEligibleDate"],
            "modelSha256": model_hash,
            "independentTradingDays": report["independentTradingDays"],
        }
        atomic_json(
            output_root / config["output"]["latestStatus"], status
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    result = process_day(
        trade_date,
        config,
        bundle,
        model_hash,
        output_root,
        bool(args.smoke_test),
    )
    status = {
        "schemaVersion": "singularity_phase1_5_status_v1",
        "status": result["status"],
        "tradeDate": trade_date,
        "researchOnly": True,
        "shadowOnly": True,
        "smokeTest": bool(args.smoke_test),
        "modelSha256": model_hash,
        "outputRoot": str(output_root),
    }
    atomic_json(output_root / config["output"]["latestStatus"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
