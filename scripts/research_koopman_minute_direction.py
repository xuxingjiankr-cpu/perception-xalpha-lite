"""Preregistered one-minute Koopman/DMD direction feasibility study.

The fixed linear delay-DMD operator is re-estimated after every completed
one-minute bar once a same-session 24-bar warm-up exists.  It forecasts the
next one-minute state and is evaluated as a directional covariate against a
causal price/market baseline with purged, calibrated walk-forward tests.

This file is historical research only.  It never imports the trading agent,
submits orders, writes overlays, or changes the frozen Phase 1.5 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research_hmm_nn_bl import ROOT
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "koopman_minute_direction_v1.json"
)
PAPER_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
AGENT_SOURCE = ROOT / "scripts" / "run_t0_intraday_agent.py"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "koopman_minute_direction_v1":
        raise ValueError("unexpected Koopman minute-direction schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("minute direction study must remain research-only")
    directive = config["userDirective"]
    if (
        directive.get("koopmanRole") != "per_minute_direction_judgment"
        or directive.get("lateSessionConfirmationOnly") is not False
    ):
        raise ValueError("Koopman role no longer matches the user directive")
    koopman = config["koopman"]
    if (
        koopman.get("implementation")
        != "fixed_window_linear_delay_dmd_direction"
        or koopman.get("decisionStrideBars") != 1
        or koopman.get("parameterSearchAllowed") is not False
        or koopman.get("kernelEnabled") is not False
        or koopman.get("deepEnabled") is not False
        or koopman.get("crossSessionWindowsAllowed") is not False
    ):
        raise ValueError("Koopman minute-direction contract changed")
    if config["labels"].get("horizonBars") != 1:
        raise ValueError("the preregistered target must remain next-minute")
    safety = config["safety"]
    if not all(
        safety.get(key) is False
        for key in [
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
    ):
        raise ValueError("research safety boundary was weakened")
    integration = config["paperIntegration"]
    if (
        integration.get("mayGenerateIndependentOrders") is not False
        or integration.get("mayChangeSellPath") is not False
        or integration.get("mayChangePositionSizing") is not False
        or integration.get("mayBypassTripleLock") is not False
    ):
        raise ValueError("paper integration contract is too broad")


def read_symbol(path: Path, code: str, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            trade_date = str(row["dt"])[:10]
            if not (
                config["data"]["historicalStart"]
                <= trade_date
                <= config["data"]["historicalEnd"]
            ):
                continue
            rows.append(
                {
                    "timestamp": pd.Timestamp(row["dt"]),
                    "trade_date": trade_date,
                    "stockCode": code,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("vol", 0.0)),
                    "amount": float(row.get("amount", 0.0)),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError(f"empty one-minute source: {path}")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError(f"duplicate one-minute timestamps: {code}")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_panel(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = resolve(config["data"]["root"])
    category = {
        row["stockCode"]: row["category"] for row in config["universe"]
    }
    frames: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    expected = int(config["data"]["expectedBarsPerFullSession"])
    for code in category:
        path = root / f"{code}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = read_symbol(path, code, config)
        frame["etf_category"] = category[code]
        per_day = frame.groupby("trade_date").size()
        audit[code] = {
            "rows": int(len(frame)),
            "tradingDays": int(per_day.size),
            "first": str(frame["trade_date"].min()),
            "last": str(frame["trade_date"].max()),
            "fullSessionDays": int((per_day == expected).sum()),
            "minimumBarsPerDay": int(per_day.min()),
            "maximumBarsPerDay": int(per_day.max()),
            "duplicateTimestamps": 0,
            "invalidOhlcRows": int(
                (
                    (frame["close"] <= 0)
                    | (frame["high"] < frame["low"])
                    | (frame["high"] < frame[["open", "close"]].max(axis=1))
                    | (frame["low"] > frame[["open", "close"]].min(axis=1))
                ).sum()
            ),
        }
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    ).reset_index(drop=True)
    return panel, audit


def add_causal_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    group_keys = ["stockCode", "trade_date"]
    output["ret_1"] = output.groupby(group_keys, sort=False)["close"].transform(
        lambda values: np.log(values).diff()
    )
    output["ret_3"] = output.groupby(group_keys, sort=False)["ret_1"].transform(
        lambda values: values.rolling(3, min_periods=3).sum()
    )
    output["vol_6"] = output.groupby(group_keys, sort=False)["ret_1"].transform(
        lambda values: values.rolling(6, min_periods=6).std(ddof=0)
    )
    market = output.groupby("timestamp", sort=False)["ret_1"].median()
    output["market_ret_1"] = output["timestamp"].map(market)
    output["relative_strength"] = (
        output["ret_1"] - output["market_ret_1"]
    )
    output["session_bar"] = output.groupby(group_keys, sort=False).cumcount()
    counts = output.groupby(group_keys, sort=False)["close"].transform("size")
    output["session_fraction"] = output["session_bar"] / np.maximum(
        counts - 1, 1
    )
    output["month"] = output["trade_date"].str[:7]
    return output


def fixed_standardization(
    panel: pd.DataFrame, config: dict[str, Any]
) -> dict[str, dict[str, float | str]]:
    cutoff = str(config["koopman"]["standardizationFitEnd"])
    fit = panel[panel["trade_date"] <= cutoff]
    stats: dict[str, dict[str, float | str]] = {}
    for name in config["koopman"]["variables"]:
        values = fit[name].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std <= 1e-12:
            raise RuntimeError(f"invalid Koopman scale for {name}")
        stats[name] = {
            "mean": mean,
            "standardDeviation": std,
            "fitEnd": cutoff,
        }
    return stats


def dmd_direction(
    standardized_variables: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float] | None:
    delay = int(config["koopman"]["delayDimension"])
    if (
        len(standardized_variables) < delay + 4
        or not np.isfinite(standardized_variables).all()
    ):
        return None
    states = np.asarray(
        [
            standardized_variables[index - delay + 1 : index + 1][
                ::-1
            ].ravel()
            for index in range(delay - 1, len(standardized_variables))
        ],
        dtype=float,
    )
    if len(states) < 5:
        return None
    x_train = states[:-1].T
    y_train = states[1:].T
    ridge = float(config["koopman"]["ridge"])
    gram = x_train @ x_train.T + ridge * np.eye(x_train.shape[0])
    cross = y_train @ x_train.T
    try:
        operator = np.linalg.solve(gram.T, cross.T).T
    except np.linalg.LinAlgError:
        return None
    fitted = operator @ x_train
    training_error = float(
        np.linalg.norm(y_train - fitted)
        / max(np.linalg.norm(y_train), 1e-12)
    )
    forecast = operator @ states[-1]
    eigenvalues = np.linalg.eigvals(operator)
    radius = float(np.max(np.abs(eigenvalues)))
    if not (
        np.isfinite(forecast).all()
        and math.isfinite(training_error)
        and math.isfinite(radius)
    ):
        return None
    return {
        "forecast_ret_z": float(forecast[0]),
        "forecast_market_z": float(forecast[1]),
        "forecast_relative_z": float(forecast[2]),
        "koopman_training_error": training_error,
        "koopman_spectral_radius": radius,
        "koopman_instability": max(0.0, radius - 1.0),
    }


def build_tables(
    panel: pd.DataFrame,
    stats: dict[str, dict[str, float | str]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variables = list(config["koopman"]["variables"])
    window = int(config["koopman"]["windowBars"])
    stride = int(config["koopman"]["decisionStrideBars"])
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    potential = 0
    for (code, trade_date), day in panel.groupby(
        ["stockCode", "trade_date"], sort=False
    ):
        day = day.sort_values("timestamp", kind="stable").reset_index(drop=True)
        values = np.column_stack(
            [
                (
                    day[name].to_numpy(dtype=float)
                    - float(stats[name]["mean"])
                )
                / float(stats[name]["standardDeviation"])
                for name in variables
            ]
        )
        for index in range(window - 1, len(day) - 1, stride):
            potential += 1
            history = values[index - window + 1 : index + 1]
            direction = dmd_direction(history, config)
            if direction is None:
                continue
            row = day.iloc[index]
            next_row = day.iloc[index + 1]
            next_return = float(
                math.log(float(next_row["close"]) / float(row["close"]))
            )
            keys = {
                "timestamp": pd.Timestamp(row["timestamp"]),
                "trade_date": str(trade_date),
                "stockCode": str(code),
            }
            feature_rows.append(
                {
                    **keys,
                    "month": str(trade_date)[:7],
                    "etf_category": str(row["etf_category"]),
                    "intraday_minute": pd.Timestamp(row["timestamp"]).strftime(
                        "%H:%M"
                    ),
                    "ret_1": float(row["ret_1"]),
                    "ret_3": float(row["ret_3"]),
                    "market_ret_1": float(row["market_ret_1"]),
                    "relative_strength": float(row["relative_strength"]),
                    "vol_6": float(row["vol_6"]),
                    "session_fraction": float(row["session_fraction"]),
                    "koopman_forecast_return": (
                        direction["forecast_ret_z"]
                        * float(stats["ret_1"]["standardDeviation"])
                        + float(stats["ret_1"]["mean"])
                    ),
                    "koopman_forecast_market_return": (
                        direction["forecast_market_z"]
                        * float(
                            stats["market_ret_1"]["standardDeviation"]
                        )
                        + float(stats["market_ret_1"]["mean"])
                    ),
                    "koopman_forecast_relative_strength": (
                        direction["forecast_relative_z"]
                        * float(
                            stats["relative_strength"][
                                "standardDeviation"
                            ]
                        )
                        + float(stats["relative_strength"]["mean"])
                    ),
                    "koopman_training_error": direction[
                        "koopman_training_error"
                    ],
                    "koopman_spectral_radius": direction[
                        "koopman_spectral_radius"
                    ],
                    "koopman_instability": direction[
                        "koopman_instability"
                    ],
                }
            )
            label_rows.append(
                {
                    **keys,
                    "next_timestamp": pd.Timestamp(next_row["timestamp"]),
                    "next_minute_return": next_return,
                    "next_minute_direction": (
                        1 if next_return > 0 else (0 if next_return < 0 else None)
                    ),
                }
            )
    features = pd.DataFrame.from_records(feature_rows)
    labels = pd.DataFrame.from_records(label_rows)
    if features.empty or labels.empty:
        raise RuntimeError("no minute-direction rows were generated")
    features = features.sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    ).reset_index(drop=True)
    labels = labels.sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    ).reset_index(drop=True)
    return features, labels, {
        "potentialDecisionRowsAfterWarmup": potential,
        "successfulKoopmanRows": int(len(features)),
        "coverage": float(len(features) / potential) if potential else 0.0,
        "firstDecisionMinute": str(features["intraday_minute"].min()),
        "lastDecisionMinute": str(features["intraday_minute"].max()),
        "neutralLabelRows": int(
            labels["next_minute_direction"].isna().sum()
        ),
        "directionalLabelRows": int(
            labels["next_minute_direction"].notna().sum()
        ),
    }


def purge_tail(
    rows: pd.DataFrame, bars: int
) -> tuple[pd.DataFrame, int]:
    if rows.empty or bars <= 0:
        return rows.copy(), 0
    ordered = rows.sort_values(
        ["stockCode", "trade_date", "timestamp"], kind="stable"
    )
    remove = (
        ordered.groupby(["stockCode", "trade_date"], sort=False)
        .tail(bars)
        .index
    )
    return ordered.drop(index=remove).reset_index(drop=True), int(len(remove))


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


def calibrated_probability(
    model: Pipeline,
    calibrator: LogisticRegression,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.clip(model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return raw, calibrator.predict_proba(logits)[:, 1]


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = ["timestamp", "trade_date", "stockCode"]
    rows = features.merge(
        labels[keys + ["next_minute_return", "next_minute_direction"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    rows = rows.dropna(subset=["next_minute_direction"]).copy()
    rows["actual"] = rows["next_minute_direction"].astype(int)
    variants = config["models"]["variants"]
    required = sorted(
        {column for columns in variants.values() for column in columns}
    )
    rows = rows.dropna(subset=required)
    test_dates = sorted(
        date
        for date in rows["trade_date"].unique()
        if config["data"]["walkForwardStart"]
        <= date
        <= config["data"]["walkForwardEnd"]
    )
    block_days = int(config["models"]["walkForwardTestBlockDays"])
    calibration_days = int(config["models"]["calibrationDays"])
    embargo_days = int(config["models"]["embargoTradingDays"])
    purge_bars = int(config["models"]["purgeBarsPerSymbol"])
    minimum_fit_days = int(config["models"]["minimumBaseFitDays"])
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold, offset in enumerate(range(0, len(test_dates), block_days), 1):
        block = test_dates[offset : offset + block_days]
        train_raw = rows[rows["trade_date"] < block[0]].copy()
        train_dates = sorted(train_raw["trade_date"].unique())
        if len(train_dates) <= calibration_days + embargo_days + minimum_fit_days:
            continue
        test_embargo = (
            set(train_dates[-embargo_days:]) if embargo_days else set()
        )
        before_test = train_raw[
            ~train_raw["trade_date"].isin(test_embargo)
        ]
        before_test, test_purged = purge_tail(before_test, purge_bars)
        available_dates = sorted(before_test["trade_date"].unique())
        calibration_set = set(available_dates[-calibration_days:])
        calibration = before_test[
            before_test["trade_date"].isin(calibration_set)
        ].copy()
        before_calibration = before_test[
            ~before_test["trade_date"].isin(calibration_set)
        ].copy()
        fit_dates = sorted(before_calibration["trade_date"].unique())
        calibration_embargo = (
            set(fit_dates[-embargo_days:]) if embargo_days else set()
        )
        fit_raw = before_calibration[
            ~before_calibration["trade_date"].isin(calibration_embargo)
        ]
        fit, calibration_purged = purge_tail(fit_raw, purge_bars)
        test = rows[rows["trade_date"].isin(block)].copy()
        if (
            fit["actual"].nunique() != 2
            or calibration["actual"].nunique() != 2
            or test.empty
        ):
            continue
        audit = {
            "fold": fold,
            "testDates": block,
            "fitStart": str(fit["trade_date"].min()),
            "fitEnd": str(fit["trade_date"].max()),
            "calibrationStart": str(calibration["trade_date"].min()),
            "calibrationEnd": str(calibration["trade_date"].max()),
            "testEmbargoDates": sorted(test_embargo),
            "calibrationEmbargoDates": sorted(calibration_embargo),
            "purgeBarsPerSymbol": purge_bars,
            "testBoundaryPurgedRows": test_purged,
            "calibrationBoundaryPurgedRows": calibration_purged,
            "sameTradingDateAcrossBoundaries": False,
            "variants": {},
        }
        for variant, columns in variants.items():
            model = make_classifier(config)
            model.fit(
                fit[columns].to_numpy(dtype=float),
                fit["actual"].to_numpy(dtype=int),
            )
            calibration_raw = np.clip(
                model.predict_proba(
                    calibration[columns].to_numpy(dtype=float)
                )[:, 1],
                1e-6,
                1 - 1e-6,
            )
            calibrator = LogisticRegression(
                C=float(config["models"]["calibrationC"]),
                max_iter=500,
                random_state=int(config["models"]["randomSeed"]),
            )
            calibrator.fit(
                np.log(
                    calibration_raw / (1.0 - calibration_raw)
                ).reshape(-1, 1),
                calibration["actual"].to_numpy(dtype=int),
            )
            raw, probability = calibrated_probability(
                model, calibrator, test[columns].to_numpy(dtype=float)
            )
            part = test[
                [
                    "timestamp",
                    "trade_date",
                    "stockCode",
                    "month",
                    "etf_category",
                    "intraday_minute",
                    "actual",
                    "next_minute_return",
                    "koopman_forecast_return",
                ]
            ].copy()
            part["fold"] = fold
            part["variant"] = variant
            part["raw_probability"] = raw
            part["probability"] = probability
            parts.append(part)
            audit["variants"][variant] = {
                "features": columns,
                "fitSamples": int(len(fit)),
                "calibrationSamples": int(len(calibration)),
                "testSamples": int(len(test)),
                "fitPositiveRate": float(fit["actual"].mean()),
                "calibrationPositiveRate": float(
                    calibration["actual"].mean()
                ),
                "testPositiveRate": float(test["actual"].mean()),
            }
        audits.append(audit)
    if not parts:
        raise RuntimeError("minute-direction walk-forward produced no folds")
    predictions = pd.concat(parts, ignore_index=True)
    return predictions.sort_values(
        ["trade_date", "timestamp", "stockCode", "variant"],
        kind="stable",
    ).reset_index(drop=True), audits


def metric_payload(
    rows: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    probability = np.clip(
        rows["probability"].to_numpy(dtype=float), 1e-12, 1 - 1e-12
    )
    actual = rows["actual"].to_numpy(dtype=int)
    edges = [float(value) for value in config["models"]["probabilityBinEdges"]]
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        mask = (probability >= low) & (
            (probability < high)
            | ((index == len(edges) - 2) & (probability <= high))
        )
        if not np.any(mask):
            continue
        mean_probability = float(np.mean(probability[mask]))
        hit_rate = float(np.mean(actual[mask]))
        ece += float(np.mean(mask)) * abs(mean_probability - hit_rate)
        bins.append(
            {
                "low": low,
                "high": high,
                "count": int(np.sum(mask)),
                "meanProbability": mean_probability,
                "hitRate": hit_rate,
            }
        )
    return {
        "count": int(len(rows)),
        "brier": float(np.mean((probability - actual) ** 2)),
        "log_loss": float(
            -np.mean(
                actual * np.log(probability)
                + (1 - actual) * np.log(1 - probability)
            )
        ),
        "auc": (
            float(roc_auc_score(actual, probability))
            if len(np.unique(actual)) == 2
            else None
        ),
        "ece": float(ece),
        "meanProbability": float(np.mean(probability)),
        "actualUpRate": float(np.mean(actual)),
        "directionalAccuracyAtHalf": float(
            np.mean((probability >= 0.5) == actual)
        ),
        "bins": bins,
    }


def summarize(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    variants = list(config["models"]["variants"])
    overall = {
        variant: metric_payload(
            predictions[predictions["variant"] == variant], config
        )
        for variant in variants
    }
    by_fold = {
        str(int(fold)): {
            variant: metric_payload(
                values[values["variant"] == variant], config
            )
            for variant in variants
        }
        for fold, values in predictions.groupby("fold")
    }
    by_group: dict[str, Any] = {}
    for dimension in ["month", "etf_category"]:
        by_group[dimension] = {
            str(group): {
                variant: metric_payload(
                    values[values["variant"] == variant], config
                )
                for variant in variants
            }
            for group, values in predictions.groupby(dimension)
        }
    return {"overall": overall, "byFold": by_fold, "byGroup": by_group}


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


def cluster_bootstrap_brier(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    baseline = config["successGate"]["comparisonBaseline"]
    candidate = config["successGate"]["candidate"]
    keys = ["timestamp", "trade_date", "stockCode", "fold"]
    subset = predictions[
        predictions["variant"].isin([baseline, candidate])
    ]
    wide = subset.pivot(
        index=keys,
        columns="variant",
        values=["actual", "probability"],
    )
    actual = wide[("actual", baseline)].to_numpy(dtype=float)
    delta = (
        wide[("probability", candidate)].to_numpy(dtype=float) - actual
    ) ** 2 - (
        wide[("probability", baseline)].to_numpy(dtype=float) - actual
    ) ** 2
    daily = (
        pd.DataFrame(
            {
                "trade_date": wide.index.get_level_values("trade_date"),
                "delta": delta,
            }
        )
        .groupby("trade_date")["delta"]
        .mean()
        .to_numpy()
    )
    rng = np.random.default_rng(int(config["models"]["randomSeed"]))
    replicates = int(config["models"]["clusterBootstrapReplicates"])
    samples = np.asarray(
        [
            np.mean(rng.choice(daily, size=len(daily), replace=True))
            for _ in range(replicates)
        ],
        dtype=float,
    )
    confidence = float(config["models"]["clusterBootstrapConfidence"])
    tail = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, tail))
    upper = float(np.quantile(samples, 1.0 - tail))
    return {
        "cluster": "trade_date",
        "independentTradingDays": int(len(daily)),
        "meanBrierDeltaCandidateMinusBaseline": float(np.mean(daily)),
        "confidence": confidence,
        "lower": lower,
        "upper": upper,
        "passes": upper < 0.0,
    }


def evaluate_gate(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline_name = config["successGate"]["comparisonBaseline"]
    candidate_name = config["successGate"]["candidate"]
    baseline = metrics["overall"][baseline_name]
    candidate = metrics["overall"][candidate_name]
    minimum = int(
        config["successGate"]["minimumImprovedMetricCountOfFour"]
    )
    improving_folds = sum(
        improvement_count(values[candidate_name], values[baseline_name])
        >= minimum
        for values in metrics["byFold"].values()
    )
    improving_months = sum(
        improvement_count(values[candidate_name], values[baseline_name])
        >= minimum
        for values in metrics["byGroup"]["month"].values()
    )
    improving_categories = [
        name
        for name, values in metrics["byGroup"]["etf_category"].items()
        if improvement_count(values[candidate_name], values[baseline_name])
        >= minimum
    ]
    bootstrap = cluster_bootstrap_brier(predictions, config)
    dates = predictions["trade_date"].nunique()
    fold_count = len(metrics["byFold"])
    month_count = len(metrics["byGroup"]["month"])
    candidate_count = candidate["count"]
    baseline_count = baseline["count"]
    gates = {
        "minimumOosTradingDays": dates
        >= int(config["successGate"]["minimumOosTradingDays"]),
        "minimumImprovedMetrics": improvement_count(candidate, baseline)
        >= minimum,
        "majorityFolds": improving_folds / fold_count
        > float(
            config["successGate"][
                "minimumImprovingFoldFractionExclusive"
            ]
        ),
        "majorityMonths": improving_months / month_count
        > float(
            config["successGate"][
                "minimumImprovingMonthFractionExclusive"
            ]
        ),
        "multipleEtfCategories": len(improving_categories)
        >= int(
            config["successGate"]["minimumImprovingEtfCategories"]
        ),
        "clusterBootstrapBrier": bootstrap["passes"],
        "sameForecastPopulation": candidate_count == baseline_count,
        "causalPastOnlyFeatures": True,
    }
    return {
        "improvedMetricCountOfFour": improvement_count(
            candidate, baseline
        ),
        "improvingFolds": improving_folds,
        "totalFolds": fold_count,
        "improvingMonths": improving_months,
        "totalMonths": month_count,
        "improvingEtfCategories": improving_categories,
        "clusterBootstrap": bootstrap,
        "gates": gates,
        "passes": all(gates.values()),
    }


def raw_direction_diagnostics(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    candidate = config["successGate"]["candidate"]
    rows = predictions[predictions["variant"] == candidate].copy()
    forecast = rows["koopman_forecast_return"].to_numpy(dtype=float)
    actual_return = rows["next_minute_return"].to_numpy(dtype=float)
    actual_direction = rows["actual"].to_numpy(dtype=int)
    forecast_direction = forecast > 0
    return {
        "rows": int(len(rows)),
        "rawForecastSignAccuracy": float(
            np.mean(forecast_direction == actual_direction)
        ),
        "rawForecastReturnCorrelation": float(
            np.corrcoef(forecast, actual_return)[0, 1]
        ),
        "forecastPositiveFraction": float(np.mean(forecast_direction)),
        "actualPositiveFraction": float(np.mean(actual_direction)),
    }


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False, suffix=".tmp.gz"
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_report(result: dict[str, Any]) -> str:
    baseline_name = result["comparison"]["baseline"]
    candidate_name = result["comparison"]["candidate"]
    baseline = result["metrics"]["overall"][baseline_name]
    candidate = result["metrics"]["overall"][candidate_name]
    gate = result["successGate"]
    bootstrap = gate["clusterBootstrap"]
    audit = result["featureAudit"]
    return "\n".join(
        [
            "# Koopman one-minute direction study",
            "",
            f"- Run ID: `{result['runId']}`",
            f"- Status: `{result['status']}`",
            f"- Data: {result['dataRange']['start']} through "
            f"{result['dataRange']['end']}",
            f"- Universe / trading days / raw rows: "
            f"{result['universeSize']} / {result['tradingDays']} / "
            f"{result['rawRows']:,}",
            f"- Koopman direction rows: "
            f"{audit['successfulKoopmanRows']:,} "
            f"({audit['coverage']:.2%} of post-warm-up opportunities)",
            f"- Intraday availability: {audit['firstDecisionMinute']} through "
            f"{audit['lastDecisionMinute']}, recomputed every minute",
            "",
            "The target is next-minute up/down direction. Exact zero returns "
            "are retained in the separate label table as neutral and excluded "
            "from binary fitting. Features use only completed same-session bars.",
            "",
            "| variant | n | Brier | LogLoss | AUC | ECE | accuracy@0.5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| {baseline_name} | {baseline['count']:,} | "
            f"{baseline['brier']:.6f} | {baseline['log_loss']:.6f} | "
            f"{baseline['auc']:.6f} | {baseline['ece']:.6f} | "
            f"{baseline['directionalAccuracyAtHalf']:.4%} |",
            f"| {candidate_name} | {candidate['count']:,} | "
            f"{candidate['brier']:.6f} | {candidate['log_loss']:.6f} | "
            f"{candidate['auc']:.6f} | {candidate['ece']:.6f} | "
            f"{candidate['directionalAccuracyAtHalf']:.4%} |",
            "",
            f"- Improved metrics: {gate['improvedMetricCountOfFour']}/4",
            f"- Improving folds: {gate['improvingFolds']}/"
            f"{gate['totalFolds']}",
            f"- Improving months: {gate['improvingMonths']}/"
            f"{gate['totalMonths']}",
            f"- Improving ETF categories: "
            f"{len(gate['improvingEtfCategories'])}",
            f"- Brier delta candidate-baseline: "
            f"{bootstrap['meanBrierDeltaCandidateMinusBaseline']:.8f}",
            f"- Trading-date cluster CI: [{bootstrap['lower']:.8f}, "
            f"{bootstrap['upper']:.8f}]",
            f"- All preregistered gates passed: `{gate['passes']}`",
            f"- Paper or online integration allowed: "
            f"`{result['paperIntegrationAllowed']}`",
            "",
            result["conclusion"],
            "",
            "This is a directional context model, not a turning-probability "
            "model and not an order generator. No trading configuration, SELL "
            "path, sizing rule, risk gate, overlay, or execution lock changed.",
            "",
        ]
    )


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paper_hash_before = sha256(PAPER_CONFIG)
    agent_hash_before = sha256(AGENT_SOURCE)
    panel, source_audit = load_panel(config)
    panel = add_causal_inputs(panel)
    stats = fixed_standardization(panel, config)
    features, labels, feature_audit = build_tables(panel, stats, config)
    predictions, fold_audits = run_walk_forward(
        features, labels, config
    )
    metrics = summarize(predictions, config)
    gate = evaluate_gate(predictions, metrics, config)
    paper_hash_after = sha256(PAPER_CONFIG)
    agent_hash_after = sha256(AGENT_SOURCE)
    result = {
        "schemaVersion": "koopman_minute_direction_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": source_commit(),
        "generatedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "dataRange": {
            "start": str(panel["trade_date"].min()),
            "end": str(panel["trade_date"].max()),
        },
        "universeSize": int(panel["stockCode"].nunique()),
        "tradingDays": int(panel["trade_date"].nunique()),
        "rawRows": int(len(panel)),
        "sourceAudit": source_audit,
        "standardization": stats,
        "featureAudit": feature_audit,
        "labelDefinition": config["labels"],
        "comparison": {
            "baseline": config["successGate"]["comparisonBaseline"],
            "candidate": config["successGate"]["candidate"],
        },
        "walkForwardFolds": fold_audits,
        "metrics": metrics,
        "rawKoopmanDirection": raw_direction_diagnostics(
            predictions, config
        ),
        "successGate": gate,
        "paperIntegrationAllowed": False,
        "paperIntegrationReason": (
            "Historical gates pass, but the preregistration explicitly "
            "requires a separate shadow-first integration review."
            if gate["passes"]
            else "The preregistered historical evidence gate failed."
        ),
        "paperConfigHashBefore": paper_hash_before,
        "paperConfigHashAfter": paper_hash_after,
        "agentSourceHashBefore": agent_hash_before,
        "agentSourceHashAfter": agent_hash_after,
        "paperConfigModified": paper_hash_before != paper_hash_after,
        "agentSourceModified": agent_hash_before != agent_hash_after,
        "conclusion": (
            "The fixed per-minute Koopman direction covariates pass the "
            "historical evidence gate. They remain research-only pending a "
            "separate immutable shadow artifact and fresh forward validation."
            if gate["passes"]
            else "The fixed per-minute Koopman direction covariates do not "
            "show sufficiently stable incremental information over the causal "
            "baseline and are not connected to paper trading."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    atomic_csv_gzip(
        output_dir / "minute_direction_features.csv.gz", features
    )
    atomic_csv_gzip(
        output_dir / "minute_direction_labels.csv.gz", labels
    )
    atomic_csv_gzip(
        output_dir / "walk_forward_predictions.csv.gz", predictions
    )
    phase2a.atomic_json(
        output_dir / "koopman_minute_direction_result.json", result
    )
    phase2a.atomic_text(
        output_dir / "koopman_minute_direction_report.md",
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
        "koopman_minute_direction_"
        + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output_dir = resolve(config["output"]["root"]) / run_id
    result = run(config, output_dir)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "oos_days": result["successGate"]["clusterBootstrap"][
                    "independentTradingDays"
                ],
                "historical_gates_passed": result["successGate"]["passes"],
                "paper_integration_allowed": result[
                    "paperIntegrationAllowed"
                ],
                "paper_config_modified": result["paperConfigModified"],
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
