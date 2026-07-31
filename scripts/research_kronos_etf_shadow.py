"""Preregistered zero-shot Kronos ETF K-line study.

Offline/shadow-only.  This file has no broker, order, position, overlay,
production-probability, risk-gate or build_decision imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "kronos_etf_shadow_preregistered.json"
)
DEFAULT_KRONOS_REPO = ROOT / "_external" / "Kronos"
DEFAULT_DEPS = ROOT / "_external" / "kronos_deps"
DEFAULT_CUDA_DEPS = ROOT / "_external" / "kronos_cuda_deps"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
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
                json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        temporary = Path(handle.name)
    temporary.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "kronos_etf_shadow_preregistered_v1":
        raise ValueError("unexpected Kronos shadow config schema")
    if config.get("status") != "research_only" or not config.get("diagnosticOnly"):
        raise ValueError("Kronos study must remain research_only/diagnosticOnly")
    safety = config.get("safety", {})
    if not safety.get("offlineOnly") or not safety.get("recordOnly"):
        raise ValueError("offlineOnly and recordOnly must be true")
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
    if any(safety.get(key) is not False for key in forbidden_true):
        raise ValueError("all production mutation/integration flags must be false")
    forecast = config["forecast"]
    if not forecast.get("zeroShot") or not forecast.get("weightsFrozen"):
        raise ValueError("Kronos weights must be frozen and zero-shot")
    if forecast.get("fineTuningAllowed") is not False:
        raise ValueError("fine-tuning is prohibited in this study")
    if int(forecast["lookbackBars"]) > int(forecast["maxContext"]):
        raise ValueError("lookback exceeds Kronos maximum context")
    horizons = [int(v) for v in forecast["horizonsBars"]]
    if not horizons or min(horizons) < 1 or horizons != sorted(set(horizons)):
        raise ValueError("horizons must be sorted unique positive bars")
    execution = config["executionAssumptions"]
    if not execution.get("decisionUsesCompletedBarOnly"):
        raise ValueError("decision must use a completed bar")
    if execution.get("entry") != "next_bar_open":
        raise ValueError("same-bar execution is prohibited")
    if execution.get("sameBarFillAllowed") is not False:
        raise ValueError("same-bar fill must be false")
    data = config["data"]
    dates = [
        data["modelFitStart"],
        data["modelFitEnd"],
        data["calibrationStart"],
        data["calibrationEnd"],
        data["oosStart"],
        data["oosEnd"],
    ]
    if not (
        dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]
    ):
        raise ValueError("fit/calibration/OOS dates must be disjoint and ordered")
    if not config["calibration"].get("parametersFrozenBeforeOos"):
        raise ValueError("calibration parameters must freeze before OOS")
    if config["evaluation"].get("promotionAllowed") is not False:
        raise ValueError("promotion must be prohibited")


def load_symbol(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
                rows.append(
                    {
                        "timestamp": pd.Timestamp(raw["dt"]),
                        "open": float(raw["open"]),
                        "high": float(raw["high"]),
                        "low": float(raw["low"]),
                        "close": float(raw["close"]),
                        "volume": float(raw.get("vol") or 0.0),
                        "amount": float(raw.get("amount") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["time"] = frame["timestamp"].dt.strftime("%H:%M")
    return frame.reset_index(drop=True)


def load_panel(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    directory = ROOT / config["data"]["barsDirectory"]
    panel: dict[str, pd.DataFrame] = {}
    for code in config["data"]["symbols"]:
        path = directory / f"{code}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing preregistered symbol: {path}")
        panel[code] = load_symbol(path)
    return panel


def audit_panel(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    all_dates: set[str] = set()
    invalid_total = 0
    incomplete_total = 0
    minimum = int(config["data"]["minimumCompleteBarsPerDay"])
    for code, frame in panel.items():
        invalid = (
            (frame["open"] <= 0)
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["low"] <= 0)
        )
        counts = frame.groupby("date").size()
        incomplete = int((counts < minimum).sum())
        invalid_total += int(invalid.sum())
        incomplete_total += incomplete
        dates = set(frame["date"])
        all_dates.update(dates)
        rows.append(
            {
                "stockCode": code,
                "rows": len(frame),
                "start": frame["date"].min() if len(frame) else None,
                "end": frame["date"].max() if len(frame) else None,
                "tradingDays": len(dates),
                "duplicateTimestampsAfterDedup": int(
                    frame["timestamp"].duplicated().sum()
                ),
                "invalidOhlcRows": int(invalid.sum()),
                "incompleteDays": incomplete,
                "medianBarsPerDay": float(counts.median()) if len(counts) else None,
            }
        )
    return {
        "schemaVersion": "kronos_etf_data_audit_v1",
        "status": "research_only",
        "source": config["data"]["barsDirectory"],
        "symbols": rows,
        "symbolCount": len(panel),
        "unionTradingDays": len(all_dates),
        "invalidOhlcRows": invalid_total,
        "incompleteSymbolDays": incomplete_total,
        "modelPretrainingBoundaryNote": (
            "Kronos paper reports pretraining through 2024-06; evaluation begins "
            "2024-07 and untouched OOS begins 2025-07."
        ),
        "adjustmentRisk": (
            "mootdx files do not carry an explicit adjustment flag; split-like "
            "discontinuities remain a data-quality limitation."
        ),
        "timezone": "Asia/Shanghai wall-clock timestamps without embedded offset",
    }


def causal_feature_vector(context: pd.DataFrame, session_fraction: float) -> list[float]:
    """Past-only OHLCVA baseline features at the completed decision bar."""
    close = context["close"].to_numpy(dtype=float)
    high = context["high"].to_numpy(dtype=float)
    low = context["low"].to_numpy(dtype=float)
    volume = np.log1p(context["volume"].to_numpy(dtype=float))

    def ret(n: int) -> float:
        return float(close[-1] / close[-1 - n] - 1.0) if len(close) > n else 0.0

    returns = np.diff(np.log(np.maximum(close, 1e-12)))
    range_6 = float(
        np.mean((high[-6:] - low[-6:]) / np.maximum(close[-6:], 1e-12))
    )
    vol_12 = float(np.std(returns[-12:], ddof=0)) if len(returns) >= 12 else 0.0
    recent = volume[-20:]
    volume_z = float(
        (recent[-1] - recent.mean()) / (recent.std(ddof=0) + 1e-8)
    )
    return [
        ret(1),
        ret(3),
        ret(6),
        ret(12),
        range_6,
        vol_12,
        volume_z,
        float(session_fraction),
    ]


def build_contexts(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Construct prefix-only model inputs and separate future outcomes."""
    lookback = int(config["forecast"]["lookbackBars"])
    max_h = max(int(v) for v in config["forecast"]["horizonsBars"])
    decision_times = set(config["data"]["decisionTimes"])
    start = config["data"]["modelFitStart"]
    end = config["data"]["oosEnd"]
    cost = float(config["executionAssumptions"]["roundTripCostPct"])
    contexts: list[dict[str, Any]] = []
    for code, frame in panel.items():
        for i in range(lookback - 1, len(frame) - max_h):
            row = frame.iloc[i]
            date = str(row["date"])
            if date < start or date > end or row["time"] not in decision_times:
                continue
            future = frame.iloc[i + 1 : i + max_h + 1]
            if len(future) != max_h or any(future["date"] != date):
                continue
            prefix = frame.iloc[i - lookback + 1 : i + 1][
                ["timestamp", "open", "high", "low", "close", "volume", "amount"]
            ].copy()
            if len(prefix) != lookback:
                continue
            entry = float(future.iloc[0]["open"])
            outcomes: dict[int, dict[str, Any]] = {}
            for horizon in config["forecast"]["horizonsBars"]:
                h = int(horizon)
                gross = float(future.iloc[h - 1]["close"]) / entry - 1.0
                net = gross - cost
                outcomes[h] = {
                    "grossReturn": gross,
                    "netReturn": net,
                    "positiveNet": int(net > 0),
                    "exitTimestamp": future.iloc[h - 1]["timestamp"],
                }
            session_rows = frame[frame["date"] == date]
            session_index = int((session_rows["timestamp"] <= row["timestamp"]).sum())
            contexts.append(
                {
                    "stockCode": code,
                    "decisionTimestamp": row["timestamp"],
                    "date": date,
                    "prefix": prefix,
                    "futureTimestamps": list(future["timestamp"]),
                    "features": causal_feature_vector(
                        prefix, session_index / max(len(session_rows), 1)
                    ),
                    "outcomes": outcomes,
                }
            )
    return sorted(contexts, key=lambda r: (r["decisionTimestamp"], r["stockCode"]))


def split_name(date: str, config: dict[str, Any]) -> str | None:
    data = config["data"]
    if data["modelFitStart"] <= date <= data["modelFitEnd"]:
        return "fit"
    if data["calibrationStart"] <= date <= data["calibrationEnd"]:
        return "calibration"
    if data["oosStart"] <= date <= data["oosEnd"]:
        return "oos"
    return None


def fit_baselines(
    contexts: list[dict[str, Any]], config: dict[str, Any]
) -> dict[int, Pipeline]:
    models: dict[int, Pipeline] = {}
    fit_rows = [r for r in contexts if split_name(r["date"], config) == "fit"]
    for horizon in config["forecast"]["horizonsBars"]:
        h = int(horizon)
        x = np.asarray([r["features"] for r in fit_rows], dtype=float)
        y = np.asarray([r["outcomes"][h]["positiveNet"] for r in fit_rows], dtype=int)
        if len(x) < int(config["baseline"]["minimumFitRows"]) or len(set(y)) < 2:
            raise RuntimeError(f"insufficient baseline fit rows for horizon {h}")
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=float(config["baseline"]["logisticC"]),
                        max_iter=1000,
                        random_state=int(config["forecast"]["randomSeed"]),
                    ),
                ),
            ]
        )
        model.fit(x, y)
        models[h] = model
    return models


def load_kronos_predictor(
    config: dict[str, Any], repo: Path, device: str | None, cache: Path
) -> Any:
    if DEFAULT_CUDA_DEPS.exists():
        sys.path.insert(0, str(DEFAULT_CUDA_DEPS))
    if DEFAULT_DEPS.exists():
        sys.path.insert(0, str(DEFAULT_DEPS))
    if not (repo / "model" / "__init__.py").exists():
        raise FileNotFoundError(
            f"Kronos source not found at {repo}; clone the official repository there"
        )
    sys.path.insert(0, str(repo))
    os.environ.setdefault("HF_HOME", str(cache))
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    tokenizer = KronosTokenizer.from_pretrained(config["forecast"]["tokenizerRepo"])
    model = Kronos.from_pretrained(config["forecast"]["modelRepo"])
    return KronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=int(config["forecast"]["maxContext"]),
    )


def forecast_context(
    predictor: Any, context: dict[str, Any], config: dict[str, Any], paths: int
) -> dict[int, float]:
    prefix = context["prefix"]
    x_df = prefix[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = pd.Series(prefix["timestamp"].to_numpy())
    max_h = max(int(v) for v in config["forecast"]["horizonsBars"])
    y_ts = pd.Series(context["futureTimestamps"][:max_h])
    cost = float(config["executionAssumptions"]["roundTripCostPct"])
    returns: dict[int, list[float]] = defaultdict(list)
    import torch

    context_seed = int(
        hashlib.sha256(
            f"{context['stockCode']}|{context['decisionTimestamp']}".encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    seed = int(config["forecast"]["randomSeed"]) + context_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Duplicate the same causal context in one GPU batch. Each batch row samples
    # independently, preserving path dispersion without N serial model calls.
    predicted_paths = predictor.predict_batch(
        df_list=[x_df] * paths,
        x_timestamp_list=[x_ts] * paths,
        y_timestamp_list=[y_ts] * paths,
        pred_len=max_h,
        T=float(config["forecast"]["temperature"]),
        top_p=float(config["forecast"]["topP"]),
        sample_count=1,
        verbose=False,
    )
    for predicted in predicted_paths:
        entry = float(predicted.iloc[0]["open"])
        for horizon in config["forecast"]["horizonsBars"]:
            h = int(horizon)
            net = float(predicted.iloc[h - 1]["close"]) / entry - 1.0 - cost
            returns[h].append(net)
    return {
        h: float(np.mean(np.asarray(values) > 0))
        for h, values in returns.items()
    }


def clipped_logit(probability: np.ndarray, clip: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, edges: list[float]
) -> float:
    total = len(y)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1.0 else p <= right)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean() - p[mask].mean()))
    return float(value) if total else float("nan")


def metric_block(
    y: np.ndarray, p: np.ndarray, returns: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    edges = [float(v) for v in config["evaluation"]["probabilityBinEdges"]]
    return {
        "rows": len(y),
        "positiveRate": float(y.mean()) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "logLoss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "ece": expected_calibration_error(y, p, edges) if len(y) else None,
        "probabilityReturnCorrelation": (
            float(np.corrcoef(p, returns)[0, 1])
            if len(y) > 2 and np.std(p) > 0 and np.std(returns) > 0
            else None
        ),
    }


def buckets(
    rows: list[dict[str, Any]], horizon: int, key: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    edges = [float(v) for v in config["evaluation"]["probabilityBinEdges"]]
    result = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [
            r
            for r in rows
            if left <= r[key][horizon] < right
            or (right == 1.0 and r[key][horizon] == 1.0)
        ]
        if not selected:
            continue
        result.append(
            {
                "left": left,
                "right": right,
                "rows": len(selected),
                "meanProbability": float(np.mean([r[key][horizon] for r in selected])),
                "hitRate": float(
                    np.mean([r["outcomes"][horizon]["positiveNet"] for r in selected])
                ),
                "meanNetReturn": float(
                    np.mean([r["outcomes"][horizon]["netReturn"] for r in selected])
                ),
            }
        )
    return result


def evaluate(
    contexts: list[dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_key = {
        (r["date"], r["stockCode"], r["decisionTimestamp"]): r
        for r in forecast_rows
    }
    baseline_models = fit_baselines(contexts, config)
    selected = []
    for row in contexts:
        if split_name(row["date"], config) not in {"calibration", "oos"}:
            continue
        key = (row["date"], row["stockCode"], row["decisionTimestamp"].isoformat())
        forecast = by_key.get(key)
        if forecast is None:
            continue
        features = np.asarray(row["features"], dtype=float).reshape(1, -1)
        selected.append(
            {
                **row,
                "baselineProbability": {
                    h: float(baseline_models[h].predict_proba(features)[0, 1])
                    for h in baseline_models
                },
                "kronosRawProbability": {
                    int(k): float(v)
                    for k, v in forecast["kronosRawProbability"].items()
                },
            }
        )
    metrics: dict[str, Any] = {"status": "diagnostic_only", "horizons": {}}
    bucket_output: dict[str, Any] = {"status": "diagnostic_only", "horizons": {}}
    clip = float(config["calibration"]["clipProbability"])
    for horizon in config["forecast"]["horizonsBars"]:
        h = int(horizon)
        calibration = [
            r for r in selected if split_name(r["date"], config) == "calibration"
        ]
        oos = [r for r in selected if split_name(r["date"], config) == "oos"]
        if len(calibration) < int(config["calibration"]["minimumRowsPerHorizon"]):
            metrics["horizons"][str(h)] = {
                "status": "insufficient_calibration_rows",
                "calibrationRows": len(calibration),
                "oosRows": len(oos),
            }
            continue
        raw_cal = np.asarray([r["kronosRawProbability"][h] for r in calibration])
        y_cal = np.asarray([r["outcomes"][h]["positiveNet"] for r in calibration])
        calibrator = LogisticRegression(C=1.0, random_state=20260731)
        calibrator.fit(clipped_logit(raw_cal, clip), y_cal)
        for rows in (calibration, oos):
            raw = np.asarray([r["kronosRawProbability"][h] for r in rows])
            calibrated = calibrator.predict_proba(clipped_logit(raw, clip))[:, 1]
            for row, probability in zip(rows, calibrated):
                row.setdefault("kronosProbability", {})[h] = float(probability)
        base_cal = np.asarray([r["baselineProbability"][h] for r in calibration])
        kronos_cal = np.asarray([r["kronosProbability"][h] for r in calibration])
        weights = [float(v) for v in config["calibration"]["ensembleWeightGrid"]]
        best_weight = min(
            weights,
            key=lambda w: float(np.mean(((1 - w) * base_cal + w * kronos_cal - y_cal) ** 2)),
        )
        for rows in (calibration, oos):
            for row in rows:
                row.setdefault("ensembleProbability", {})[h] = float(
                    (1 - best_weight) * row["baselineProbability"][h]
                    + best_weight * row["kronosProbability"][h]
                )
        horizon_metrics: dict[str, Any] = {
            "status": "evaluated",
            "calibrationRows": len(calibration),
            "oosRows": len(oos),
            "frozenKronosWeight": best_weight,
            "variants": {},
        }
        y = np.asarray([r["outcomes"][h]["positiveNet"] for r in oos])
        realized = np.asarray([r["outcomes"][h]["netReturn"] for r in oos])
        for name, key in [
            ("baseline", "baselineProbability"),
            ("kronos_raw", "kronosRawProbability"),
            ("kronos", "kronosProbability"),
            ("baseline_kronos", "ensembleProbability"),
        ]:
            p = np.asarray([r[key][h] for r in oos])
            variant_metrics = metric_block(y, p, realized, config)
            variant_metrics["thresholds"] = {}
            for threshold in config["evaluation"]["probabilityThresholds"]:
                threshold = float(threshold)
                mask = p >= threshold
                variant_metrics["thresholds"][str(threshold)] = {
                    "rows": int(mask.sum()),
                    "hitRate": float(y[mask].mean()) if mask.any() else None,
                    "meanNetReturn": (
                        float(realized[mask].mean()) if mask.any() else None
                    ),
                }
            horizon_metrics["variants"][name] = variant_metrics
        metrics["horizons"][str(h)] = horizon_metrics
        bucket_output["horizons"][str(h)] = {
            name: buckets(oos, h, key, config)
            for name, key in [
                ("baseline", "baselineProbability"),
                ("kronos", "kronosProbability"),
                ("baseline_kronos", "ensembleProbability"),
            ]
        }
    metrics["samePopulationAcrossVariants"] = True
    metrics["productionEligible"] = False
    primary_gate: dict[str, Any] = {}
    gate_pass = True
    for horizon in config["evaluation"]["primaryHorizonsBars"]:
        h = str(int(horizon))
        block = metrics["horizons"].get(h, {})
        if block.get("status") != "evaluated":
            primary_gate[h] = {"pass": False, "reason": "not_evaluated"}
            gate_pass = False
            continue
        baseline = block["variants"]["baseline"]
        ensemble = block["variants"]["baseline_kronos"]
        improvements = {
            "brier": ensemble["brier"] < baseline["brier"],
            "logLoss": ensemble["logLoss"] < baseline["logLoss"],
            "auc": ensemble["auc"] > baseline["auc"],
            "ece": ensemble["ece"] < baseline["ece"],
        }
        improved_count = sum(improvements.values())
        high_rows = ensemble["thresholds"]["0.65"]["rows"]
        passed = (
            improved_count >= 3
            and high_rows
            >= int(config["evaluation"]["minimumHighProbabilityRows"])
        )
        primary_gate[h] = {
            "pass": passed,
            "improvements": improvements,
            "improvedMetricCount": improved_count,
            "requiredImprovedMetricCount": 3,
            "highProbabilityRows": high_rows,
            "requiredHighProbabilityRows": int(
                config["evaluation"]["minimumHighProbabilityRows"]
            ),
        }
        gate_pass = gate_pass and passed
    oos_dates = {
        row["date"]
        for row in selected
        if split_name(row["date"], config) == "oos"
    }
    enough_days = (
        len(oos_dates) >= int(config["evaluation"]["minimumOosDaysForConclusion"])
    )
    metrics["preregisteredGate"] = {
        "pass": bool(gate_pass and enough_days),
        "oosDays": len(oos_dates),
        "minimumOosDays": int(
            config["evaluation"]["minimumOosDaysForConclusion"]
        ),
        "primaryHorizons": primary_gate,
        "sameRowsUsedForProbabilityMetrics": True,
        "notMechanicalFewerRowsForProbabilityMetrics": True,
    }
    metrics["researchVerdict"] = (
        "incremental_shadow_hypothesis_survives"
        if metrics["preregisteredGate"]["pass"]
        else "no_incremental_edge_under_preregistered_gate"
    )
    metrics["skippedAfterPrimaryGate"] = [
        "production integration (always prohibited)",
        "fine-tuning",
        "parameter search",
        "month/symbol promotion analysis when the primary gate fails",
    ]
    metrics["note"] = (
        "Metric improvement is research evidence only; thresholded trade counts "
        "are reported separately and cannot authorize integration."
    )
    return metrics, bucket_output


def report_text(
    run_id: str,
    audit: dict[str, Any],
    context_count: int,
    inference_count: int,
    metrics: dict[str, Any] | None,
    mode: str,
) -> str:
    lines = [
        "# Kronos ETF shadow report",
        "",
        f"- run_id: `{run_id}`",
        "- status: `research_only / diagnostic_only / shadow_only`",
        f"- mode: `{mode}`",
        f"- symbols: {audit['symbolCount']}",
        f"- union trading days: {audit['unionTradingDays']}",
        f"- eligible causal contexts: {context_count}",
        f"- inferred contexts: {inference_count}",
        f"- invalid OHLC rows: {audit['invalidOhlcRows']}",
        "- production integration: prohibited",
        "",
    ]
    if metrics:
        lines.extend(
            [
                f"- research verdict: `{metrics.get('researchVerdict')}`",
                f"- preregistered gate: `{metrics.get('preregisteredGate', {}).get('pass')}`",
                "",
                "## Frozen OOS metrics",
                "",
            ]
        )
        for horizon, block in metrics.get("horizons", {}).items():
            lines.append(f"### {horizon} bars")
            if block.get("status") != "evaluated":
                lines.append(f"- status: `{block.get('status')}`")
                continue
            lines.append(f"- OOS rows: {block['oosRows']}")
            lines.append(f"- frozen Kronos weight: {block['frozenKronosWeight']}")
            for name, values in block["variants"].items():
                lines.append(
                    f"- {name}: Brier={values['brier']}, "
                    f"LogLoss={values['logLoss']}, AUC={values['auc']}, "
                    f"ECE={values['ece']}"
                )
            lines.append("")
    else:
        lines.extend(
            [
                "No edge conclusion is produced by audit/smoke mode.",
                "A complete frozen calibration and OOS run is required.",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-contexts", type=int, default=4)
    parser.add_argument("--smoke-paths", type=int, default=2)
    parser.add_argument("--kronos-repo", type=Path, default=DEFAULT_KRONOS_REPO)
    parser.add_argument("--model-cache", type=Path, default=ROOT / "_external" / "hf")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--forecast-input",
        type=Path,
        default=None,
        help="Re-evaluate an existing research-only forecast_rows.jsonl without inference.",
    )
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_text = args.config.read_text(encoding="utf-8")
    config = json.loads(config_text)
    validate_config(config)
    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_root = ROOT / config["artifact"]["outputRoot"] / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    panel = load_panel(config)
    audit = audit_panel(panel, config)
    contexts = build_contexts(panel, config)
    atomic_json(output_root / "data_audit.json", audit)
    split_counts = defaultdict(int)
    for row in contexts:
        split_counts[split_name(row["date"], config) or "excluded"] += 1
    manifest = {
        "schemaVersion": "kronos_etf_shadow_run_v1",
        "runId": run_id,
        "status": "research_only",
        "diagnosticOnly": True,
        "shadowOnly": True,
        "configSha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "createdAt": datetime.now().astimezone().isoformat(),
        "contextCount": len(contexts),
        "splitCounts": dict(split_counts),
        "mode": "audit_only" if args.audit_only else ("smoke" if args.smoke else "full"),
        "productionArtifactsWritten": [],
    }
    if args.audit_only:
        atomic_json(output_root / "run_manifest.json", manifest)
        (output_root / "report.md").write_text(
            report_text(run_id, audit, len(contexts), 0, None, "audit_only"),
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"output: {output_root}")
        return 0

    if args.forecast_input is not None:
        forecast_rows = [
            json.loads(line)
            for line in args.forecast_input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics, buckets_output = evaluate(contexts, forecast_rows, config)
        atomic_json(output_root / "metrics.json", metrics)
        atomic_json(output_root / "probability_buckets.json", buckets_output)
        manifest["mode"] = "frozen_reanalysis_no_inference"
        manifest["forecastInput"] = str(args.forecast_input.resolve())
        manifest["inferredContexts"] = len(forecast_rows)
        manifest["productionEligible"] = False
        atomic_json(output_root / "run_manifest.json", manifest)
        (output_root / "report.md").write_text(
            report_text(
                run_id,
                audit,
                len(contexts),
                len(forecast_rows),
                metrics,
                "frozen_reanalysis_no_inference",
            ),
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"output: {output_root}")
        return 0

    inferable = [
        row
        for row in contexts
        if split_name(row["date"], config) in {"calibration", "oos"}
    ]
    paths = int(config["forecast"]["samplePaths"])
    if args.smoke:
        inferable = inferable[: max(1, args.max_contexts)]
        paths = min(paths, max(1, args.smoke_paths))
        manifest["smokeOverrides"] = {
            "maxContexts": len(inferable),
            "samplePaths": paths,
            "nonConclusive": True,
        }
    predictor = load_kronos_predictor(
        config, args.kronos_repo.resolve(), args.device, args.model_cache.resolve()
    )
    forecast_rows = []
    for index, row in enumerate(inferable, 1):
        raw = forecast_context(predictor, row, config, paths)
        forecast_rows.append(
            {
                "schemaVersion": "kronos_etf_path_probability_research_v1",
                "status": "research_only",
                "stockCode": row["stockCode"],
                "date": row["date"],
                "decisionTimestamp": row["decisionTimestamp"].isoformat(),
                "kronosRawProbability": raw,
                "samplePaths": paths,
                "outcomes": row["outcomes"],
            }
        )
        print(f"forecast {index}/{len(inferable)} {row['stockCode']} {row['date']}")
    atomic_jsonl(output_root / "forecast_rows.jsonl", forecast_rows)
    metrics = None
    buckets_output = None
    if not args.smoke:
        metrics, buckets_output = evaluate(contexts, forecast_rows, config)
        atomic_json(output_root / "metrics.json", metrics)
        atomic_json(output_root / "probability_buckets.json", buckets_output)
    manifest["inferredContexts"] = len(forecast_rows)
    manifest["model"] = config["forecast"]["modelRepo"]
    manifest["device"] = getattr(predictor, "device", args.device)
    manifest["productionEligible"] = False
    atomic_json(output_root / "run_manifest.json", manifest)
    (output_root / "report.md").write_text(
        report_text(
            run_id,
            audit,
            len(contexts),
            len(forecast_rows),
            metrics,
            "smoke_non_conclusive" if args.smoke else "frozen_historical",
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
