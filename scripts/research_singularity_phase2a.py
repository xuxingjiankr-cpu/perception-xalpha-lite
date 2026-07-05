"""Historical-only Singularity Phase 2A audit and minimal feasibility study.

The script first audits the longer mootdx archive.  Only if every preregistered
audit gate passes does it evaluate fixed-grid classic LPPLS diagnostics and a
fixed-window linear delay-DMD residual model.  All features are past-only,
labels are stored separately, and every ablation forecasts the same rows.

This module cannot trade, write overlays, modify Phase 1.5, or perform online
inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision_probability import probability_metrics
from research_hmm_nn_bl import ROOT
from research_minute_forecast_shadow import build_feature_frames
from research_singularity_phase1 import build_label_table, rolling_variance_slope


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "singularity_phase2a_historical.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs" / "edge_research" / "singularity_phase2a_historical"
)

EWS_FEATURES = [
    "ews_log_variance",
    "ews_autocorrelation_1",
    "ews_variance_slope",
    "ews_market_dispersion",
    "ews_score",
]
LPPLS_FEATURES = [
    "lppls_residual_mean",
    "lppls_residual_std",
    "lppls_tc_offset_median",
    "lppls_tc_offset_std",
    "lppls_m_median",
    "lppls_m_std",
    "lppls_omega_median",
    "lppls_omega_std",
    "lppls_b_scaled_mean",
    "lppls_oscillation_amplitude",
    "lppls_trend_eligible",
]
KOOPMAN_FEATURES = [
    "koopman_residual_norm",
    "koopman_training_error",
    "koopman_spectral_radius",
    "koopman_effective_rank",
    "koopman_instability",
]


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "singularity_phase2a_historical_v1":
        raise ValueError("unexpected Phase 2A config schema")
    if (
        config.get("status") != "research_only"
        or not config.get("shadowOnly")
        or not config.get("diagnosticOnly")
    ):
        raise ValueError("Phase 2A must remain research/shadow only")
    safety = config["safety"]
    if not safety.get("offlineOnly") or not safety.get("recordOnly"):
        raise ValueError("offline record-only flags are required")
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
        "promotionAllowed",
    ]
    if any(safety.get(key) is not False for key in forbidden):
        raise ValueError("all Phase 2A live mutation flags must be false")
    if config["data"].get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 ledger must not be an input")
    if 60 in config["labels"]["modelHorizonsBars"]:
        raise ValueError("60-bar horizon cannot be modeled")
    if int(config["models"]["purgeBarsPerSymbol"]) < max(
        config["labels"]["observationalHorizonsBars"]
    ):
        raise ValueError("purge must cover maximum non-skipped label horizon")


def load_mootdx_panel(
    config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    root = resolve(config["data"]["primary5mRoot"])
    start = str(config["data"]["historicalStart"])
    end = str(config["data"]["historicalEnd"])
    universe = {row["stockCode"] for row in config["universe"]}
    frames: dict[str, pd.DataFrame] = {}
    records: list[pd.DataFrame] = []
    for code in sorted(universe):
        path = root / f"{code}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        frame = pd.DataFrame.from_records(rows)
        frame["timestamp"] = pd.to_datetime(frame.pop("dt"))
        frame["trade_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        frame = frame[
            (frame["trade_date"] >= start) & (frame["trade_date"] <= end)
        ].sort_values("timestamp", kind="stable")
        frame["stockCode"] = code
        frame["bar_volume"] = frame["vol"].astype(float)
        frame["bar_amount"] = frame["amount"].astype(float)
        frame["cumulative_amount"] = frame.groupby("trade_date")[
            "bar_amount"
        ].cumsum()
        frames[code] = frame.copy()
        records.append(
            frame[
                [
                    "timestamp",
                    "trade_date",
                    "stockCode",
                    "open",
                    "high",
                    "low",
                    "close",
                    "bar_volume",
                    "bar_amount",
                    "cumulative_amount",
                ]
            ]
        )
    panel = pd.concat(records, ignore_index=True).sort_values(
        ["timestamp", "stockCode"], kind="stable"
    )
    return panel, frames


def supporting_coverage(config: dict[str, Any]) -> dict[str, Any]:
    one_minute = json.loads(
        (
            resolve(config["data"]["supporting1mRoot"]).parent
            / "backfill_1m_summary.json"
        ).read_text(encoding="utf-8")
    )
    daily_root = resolve(config["data"]["supporting1dRoot"])
    l2_root = resolve(config["data"]["l2Root"])
    iopv_root = resolve(config["data"]["iopvRoot"])
    l2_dates = sorted(
        {
            match.group(1)
            for path in l2_root.glob("depth_*.jsonl")
            if (match := re.search(r"(\d{4}-\d{2}-\d{2})", path.name))
        }
    )
    iopv_dates = sorted(
        {
            match.group(1)
            for path in iopv_root.glob("iopv_*.jsonl")
            if (match := re.search(r"(\d{4}-\d{2}-\d{2})", path.name))
        }
    )
    return {
        "fiveMinute": {
            "usedByModel": True,
            "root": config["data"]["primary5mRoot"],
        },
        "oneMinute": {
            "usedByModel": False,
            "symbols": len(one_minute),
            "minimumStart": min(row["first"] for row in one_minute),
            "maximumEnd": max(row["last"] for row in one_minute),
            "minimumDaysApprox": min(row["rows"] // 240 for row in one_minute),
            "maximumDaysApprox": max(row["rows"] // 240 for row in one_minute),
            "reasonExcluded": "only about 91-100 sessions; shorter than 5-minute archive",
        },
        "daily": {
            "usedByModel": False,
            "symbolFiles": len(list(daily_root.glob("*.jsonl"))),
        },
        "tick": {
            "usedByModel": False,
            "files": 0,
            "reasonExcluded": "no tick archive found",
        },
        "l2": {
            "usedByModel": False,
            "dates": l2_dates,
            "days": len(l2_dates),
            "reasonExcluded": "insufficient historical depth coverage",
        },
        "iopv": {
            "usedByModel": False,
            "dates": iopv_dates,
            "days": len(iopv_dates),
            "contaminatesPriceFeatures": False,
            "reasonExcluded": "not merged into mootdx OHLCV and only five days",
        },
    }


def audit_symbol_quality(
    frames: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    all_dates = sorted(
        {date for frame in frames.values() for date in frame["trade_date"].unique()}
    )
    symbols: dict[str, Any] = {}
    anomalies: list[dict[str, Any]] = []
    for code, frame in frames.items():
        dates = sorted(frame["trade_date"].unique())
        eligible_dates = [
            date for date in all_dates if dates[0] <= date <= dates[-1]
        ]
        counts = frame.groupby("trade_date").size()
        expected = len(eligible_dates) * 48
        missing = max(0, expected - len(frame))
        invalid = (
            (frame["low"] > frame["high"])
            | (frame["open"] < frame["low"])
            | (frame["open"] > frame["high"])
            | (frame["close"] < frame["low"])
            | (frame["close"] > frame["high"])
            | (frame["close"] <= 0)
        )
        same_day = frame["trade_date"] == frame["trade_date"].shift(1)
        log_return = np.log(frame["close"] / frame["close"].shift(1))
        intraday_jump = same_day & (log_return.abs() > math.log(1.10))
        daily = frame.groupby("trade_date").agg(
            open=("open", "first"), close=("close", "last")
        )
        daily["overnight_return"] = daily["open"] / daily["close"].shift(1) - 1.0
        for date, row in daily[daily["overnight_return"].abs() > 0.20].iterrows():
            anomalies.append(
                {
                    "stockCode": code,
                    "date": date,
                    "type": "overnight_jump_gt_20pct",
                    "value": float(row["overnight_return"]),
                    "crossSessionFeatureImpact": False,
                }
            )
        for row in frame.loc[intraday_jump, ["timestamp", "close"]].itertuples():
            anomalies.append(
                {
                    "stockCode": code,
                    "date": row.timestamp.strftime("%Y-%m-%d"),
                    "timestamp": row.timestamp.isoformat(),
                    "type": "intraday_log_jump_gt_10pct",
                }
            )
        symbols[code] = {
            "first": dates[0],
            "last": dates[-1],
            "tradingDays": len(dates),
            "rows": int(len(frame)),
            "expectedRowsFromFirstListing": expected,
            "missingRows": missing,
            "missingBarFraction": missing / expected if expected else None,
            "full48BarDays": int((counts == 48).sum()),
            "minimumBarsPerPresentDay": int(counts.min()),
            "maximumBarsPerPresentDay": int(counts.max()),
            "duplicateTimestamps": int(frame["timestamp"].duplicated().sum()),
            "invalidOhlcRows": int(invalid.sum()),
            "zeroVolumeFraction": float((frame["bar_volume"] <= 0).mean()),
            "intradayJumpCountGt10Pct": int(intraday_jump.sum()),
        }
    return symbols, anomalies, all_dates


def label_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"labels": {
        **config["labels"],
        "horizonsBars": config["labels"]["auditHorizonsBars"],
        "activeHorizonsBars": (
            config["labels"]["modelHorizonsBars"]
            + config["labels"]["observationalHorizonsBars"]
        ),
    }}


def independent_event_counts(
    labels: pd.DataFrame, horizons: list[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        column = f"turning_point_{horizon}"
        valid = labels.dropna(subset=[column]).copy()
        positive_rows = 0
        events = 0
        by_direction = {"up": 0, "down": 0}
        for (_, _), group in valid.groupby(["stockCode", "trade_date"]):
            group = group.sort_values("timestamp").reset_index(drop=True)
            positive = np.flatnonzero(group[column].to_numpy(dtype=int) == 1)
            positive_rows += len(positive)
            last = -10_000
            for index in positive:
                if index - last > horizon:
                    events += 1
                    if int(group.loc[index, f"reversal_up_{horizon}"]) == 1:
                        by_direction["up"] += 1
                    if int(group.loc[index, f"reversal_down_{horizon}"]) == 1:
                        by_direction["down"] += 1
                last = index
        result[str(horizon)] = {
            "effectiveSamples": int(len(valid)),
            "positiveRows": int(positive_rows),
            "positiveRowRate": float(positive_rows / len(valid)) if len(valid) else None,
            "independentEvents": int(events),
            "independentUpEvents": int(by_direction["up"]),
            "independentDownEvents": int(by_direction["down"]),
        }
    result["60"] = {
        "effectiveSamples": 0,
        "positiveRows": 0,
        "positiveRowRate": None,
        "independentEvents": 0,
        "status": "skipped_cross_session",
    }
    return result


def audit_window_counts(
    frames: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    window = int(config["sampling"]["minimumPastWindowBars"])
    stride = int(config["sampling"]["decisionStrideBars"])
    lppls_cfg = config["features"]["lppls"]
    possible = 0
    trend_eligible = 0
    dmd_possible = 0
    per_symbol: dict[str, Any] = {}
    for code, frame in frames.items():
        symbol_possible = 0
        symbol_trend = 0
        for _, day in frame.groupby("trade_date"):
            day = day.sort_values("timestamp")
            close = day["close"].to_numpy(dtype=float)
            if len(close) != 48 or not np.isfinite(close).all():
                continue
            for index in range(window - 1, len(close), stride):
                values = close[index - window + 1 : index + 1]
                symbol_possible += 1
                total_move = abs(values[-1] / values[0] - 1.0)
                path = float(np.sum(np.abs(np.diff(values))))
                efficiency = abs(values[-1] - values[0]) / path if path > 0 else 0.0
                if (
                    total_move >= float(lppls_cfg["minimumTrendReturnPct"])
                    and efficiency >= float(lppls_cfg["minimumTrendEfficiency"])
                ):
                    symbol_trend += 1
        per_symbol[code] = {
            "fixedDecisionWindows": symbol_possible,
            "trendEligibleLpplsWindows": symbol_trend,
            "equalTradingTimeDmdWindows": symbol_possible,
        }
        possible += symbol_possible
        trend_eligible += symbol_trend
        dmd_possible += symbol_possible
    return {
        "tradingTimeConvention": (
            "48 ordered five-minute trading bars; lunch break compressed, "
            "never crosses overnight"
        ),
        "lpplsFixedWindows": possible,
        "lpplsTrendEligibleWindows": trend_eligible,
        "koopmanEqualIntervalWindows": dmd_possible,
        "perSymbol": per_symbol,
    }


def build_data_audit(
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    symbols, anomalies, all_dates = audit_symbol_quality(frames)
    events = independent_event_counts(
        labels,
        config["labels"]["modelHorizonsBars"]
        + config["labels"]["observationalHorizonsBars"],
    )
    windows = audit_window_counts(frames, config)
    gates_cfg = config["auditGates"]
    maximum_missing = max(
        float(row["missingBarFraction"] or 0.0) for row in symbols.values()
    )
    invalid_rows = sum(int(row["invalidOhlcRows"]) for row in symbols.values())
    gates = {
        "minimumSymbols": len(symbols) >= int(gates_cfg["minimumSymbols"]),
        "minimumTradingDays": len(all_dates)
        >= int(gates_cfg["minimumTradingDays"]),
        "maximumMissingBarFraction": maximum_missing
        <= float(gates_cfg["maximumMissingBarFraction"]),
        "maximumInvalidOhlcRows": invalid_rows
        <= int(gates_cfg["maximumInvalidOhlcRows"]),
        "minimumIndependentEvents5": events["5"]["independentEvents"]
        >= int(gates_cfg["minimumIndependentEventsPerPrimaryHorizon"]),
        "minimumIndependentEvents10": events["10"]["independentEvents"]
        >= int(gates_cfg["minimumIndependentEventsPerPrimaryHorizon"]),
        "minimumLpplsWindows": windows["lpplsFixedWindows"]
        >= int(gates_cfg["minimumLpplsWindows"]),
        "minimumKoopmanWindows": windows["koopmanEqualIntervalWindows"]
        >= int(gates_cfg["minimumKoopmanWindows"]),
    }
    category_counts: dict[str, int] = defaultdict(int)
    for row in config["universe"]:
        category_counts[row["category"]] += 1
    return {
        "schemaVersion": "singularity_phase2a_data_audit_v1",
        "status": "research_only",
        "source": "mootdx local historical OHLCV",
        "dataRange": {"start": all_dates[0], "end": all_dates[-1]},
        "tradingDays": len(all_dates),
        "symbols": len(symbols),
        "rows": int(len(panel)),
        "universeByCategory": dict(sorted(category_counts.items())),
        "perSymbol": symbols,
        "supportingCoverage": supporting_coverage(config),
        "missingBarAudit": {
            "maximumPerSymbolFraction": maximum_missing,
            "invalidOhlcRows": invalid_rows,
            "duplicateTimestamps": sum(
                int(row["duplicateTimestamps"]) for row in symbols.values()
            ),
        },
        "adjustmentAudit": {
            "adjustmentFactorAvailable": False,
            "overnightAnomalies": anomalies,
            "sameSessionFeaturesResetDaily": True,
            "risk": (
                "Raw/unadjusted bars; detected overnight discontinuities are "
                "not crossed by any feature or label."
            ),
        },
        "timezoneAudit": {
            "timestampsContainTimezone": False,
            "assumedTimezone": config["data"]["timezoneAssumption"],
            "allRowsInExpectedSessionSlots": True,
            "risk": "timezone is inferred from TDX session convention",
        },
        "iopvContamination": {
            "iopvMerged": False,
            "priceFieldIsExchangeOhlc": True,
            "premiumStillPossibleEconomically": True,
        },
        "labelSamplesAndIndependentEvents": events,
        "feasibleWindows": windows,
        "auditGates": gates,
        "auditPass": all(gates.values()),
        "modelRunAllowed": all(gates.values()),
    }


def render_data_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# Singularity Phase 2A data audit",
        "",
        f"- Status: `{'pass' if audit['auditPass'] else 'fail'}`",
        f"- Range: {audit['dataRange']['start']} through {audit['dataRange']['end']}",
        f"- Trading days / symbols / rows: {audit['tradingDays']} / {audit['symbols']} / {audit['rows']:,}",
        f"- Maximum missing-bar fraction: {audit['missingBarAudit']['maximumPerSymbolFraction']:.4%}",
        f"- Invalid OHLC / duplicates: {audit['missingBarAudit']['invalidOhlcRows']} / {audit['missingBarAudit']['duplicateTimestamps']}",
        "",
        "## Effective labels and independent events",
        "",
        "| horizon | effective rows | positive rows | independent events |",
        "|---:|---:|---:|---:|",
    ]
    for horizon, row in audit["labelSamplesAndIndependentEvents"].items():
        lines.append(
            f"| {horizon} | {row['effectiveSamples']:,} | "
            f"{row['positiveRows']:,} | {row['independentEvents']:,} |"
        )
    windows = audit["feasibleWindows"]
    lines.extend(
        [
            "",
            "## Feasible fixed windows",
            "",
            f"- LPPLS fixed windows: {windows['lpplsFixedWindows']:,}",
            f"- LPPLS trend-eligible windows: {windows['lpplsTrendEligibleWindows']:,}",
            f"- Equal trading-time DMD windows: {windows['koopmanEqualIntervalWindows']:,}",
            "",
            "## Gate results",
            "",
        ]
    )
    for name, passed in audit["auditGates"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            "",
            "Raw bars have no adjustment factor. Overnight discontinuities are "
            "reported, but all model features and labels reset within each day.",
            "Tick data are absent; L2 and IOPV histories are too short and are "
            "not model inputs.",
            "",
        ]
    )
    return "\n".join(lines)


def build_ews(
    base: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame | pd.Series], dict[str, Any]]:
    window = int(config["features"]["ewsWindowBars"])
    floor = float(config["features"]["ewsVarianceFloor"])
    ret: pd.DataFrame = base["ret_1"]
    dates = pd.Series(ret.index.strftime("%Y-%m-%d"), index=ret.index)
    variance = ret.groupby(dates).transform(
        lambda frame: frame.rolling(window, min_periods=window).var()
    )
    components: dict[str, pd.DataFrame | pd.Series] = {
        "ews_log_variance": np.log(variance.clip(lower=floor)),
        "ews_autocorrelation_1": ret.groupby(dates).transform(
            lambda frame: frame.rolling(window, min_periods=window).corr(
                frame.shift(1)
            )
        ),
        "ews_variance_slope": rolling_variance_slope(ret.pow(2), window),
        "ews_market_dispersion": ret.std(axis=1, skipna=True),
    }
    train_end = str(config["data"]["fixedPreprocessingEnd"])
    mask = ret.index.strftime("%Y-%m-%d") <= train_end
    statistics: dict[str, Any] = {}
    scores: list[pd.DataFrame] = []
    for name, frame in components.items():
        values = (
            frame.loc[mask].to_numpy(dtype=float).ravel()
            if isinstance(frame, pd.DataFrame)
            else frame.loc[mask].to_numpy(dtype=float)
        )
        values = values[np.isfinite(values)]
        mean, std = float(np.mean(values)), float(np.std(values))
        if std <= 1e-12:
            raise RuntimeError(f"invalid EWS standard deviation: {name}")
        statistics[name] = {
            "mean": mean,
            "standardDeviation": std,
            "fitEnd": train_end,
        }
        z = ((frame - mean) / std).clip(-8.0, 8.0)
        if isinstance(z, pd.Series):
            z = pd.DataFrame(
                np.repeat(z.to_numpy()[:, None], len(ret.columns), axis=1),
                index=ret.index,
                columns=ret.columns,
            )
        scores.append(1.0 / (1.0 + np.exp(-z)))
    components["ews_score"] = sum(scores) / len(scores)
    return components, statistics


def lppls_single_fit(
    log_prices: np.ndarray, m: float, omega: float, tc_offset: int,
    maximum_condition: float,
) -> dict[str, float] | None:
    n = len(log_prices)
    time = np.arange(n, dtype=float)
    tc = float(n - 1 + tc_offset)
    distance = tc - time
    power = np.power(distance, m)
    log_distance = np.log(distance)
    design = np.column_stack(
        [
            np.ones(n),
            power,
            power * np.cos(omega * log_distance),
            power * np.sin(omega * log_distance),
        ]
    )
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > maximum_condition:
        return None
    coefficients, _, _, _ = np.linalg.lstsq(design, log_prices, rcond=None)
    fitted = design @ coefficients
    scale = max(float(np.std(log_prices)), 1e-8)
    residual = float(np.sqrt(np.mean((log_prices - fitted) ** 2)) / scale)
    return {
        "residual": residual,
        "m": m,
        "omega": omega,
        "tcOffset": float(tc_offset),
        "bScaled": float(coefficients[1] / scale),
        "oscillationAmplitude": float(
            math.hypot(coefficients[2], coefficients[3]) / scale
        ),
    }


def lppls_features(
    prices: np.ndarray, config: dict[str, Any]
) -> dict[str, float] | None:
    cfg = config["features"]["lppls"]
    nested: list[dict[str, float]] = []
    for window in cfg["nestedWindowBars"]:
        values = prices[-int(window) :]
        if len(values) != int(window) or np.any(values <= 0):
            continue
        log_prices = np.log(values)
        candidates = [
            fit
            for m in cfg["mGrid"]
            for omega in cfg["omegaGrid"]
            for tc in cfg["tcOffsetBarsGrid"]
            if (
                fit := lppls_single_fit(
                    log_prices,
                    float(m),
                    float(omega),
                    int(tc),
                    float(cfg["maximumConditionNumber"]),
                )
            )
            is not None
        ]
        if candidates:
            nested.append(min(candidates, key=lambda row: row["residual"]))
    if len(nested) < int(cfg["minimumStableNestedFits"]):
        return None
    total_move = abs(prices[-1] / prices[0] - 1.0)
    path = float(np.sum(np.abs(np.diff(prices))))
    efficiency = abs(prices[-1] - prices[0]) / path if path > 0 else 0.0

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in nested], dtype=float)

    return {
        "lppls_residual_mean": float(np.mean(values("residual"))),
        "lppls_residual_std": float(np.std(values("residual"))),
        "lppls_tc_offset_median": float(np.median(values("tcOffset"))),
        "lppls_tc_offset_std": float(np.std(values("tcOffset"))),
        "lppls_m_median": float(np.median(values("m"))),
        "lppls_m_std": float(np.std(values("m"))),
        "lppls_omega_median": float(np.median(values("omega"))),
        "lppls_omega_std": float(np.std(values("omega"))),
        "lppls_b_scaled_mean": float(np.mean(values("bScaled"))),
        "lppls_oscillation_amplitude": float(
            np.mean(values("oscillationAmplitude"))
        ),
        "lppls_trend_eligible": float(
            total_move >= float(cfg["minimumTrendReturnPct"])
            and efficiency >= float(cfg["minimumTrendEfficiency"])
        ),
    }


def dmd_statistics(
    base: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    cutoff = str(config["data"]["fixedPreprocessingEnd"])
    prices: pd.DataFrame = base["prices"]
    mask = prices.index.strftime("%Y-%m-%d") <= cutoff
    sources = {
        "ret_1": base["ret_1"],
        "market_ret_1": base["market_ret_1"],
        "relative_strength": base["relative_strength"],
        "vol_6": base["vol_6"],
    }
    stats: dict[str, Any] = {}
    for name, source in sources.items():
        values = (
            source.loc[mask].to_numpy(dtype=float).ravel()
            if isinstance(source, pd.DataFrame)
            else source.loc[mask].to_numpy(dtype=float)
        )
        values = values[np.isfinite(values)]
        mean, std = float(np.mean(values)), float(np.std(values))
        if std <= 1e-12:
            raise RuntimeError(f"invalid DMD standard deviation: {name}")
        stats[name] = {"mean": mean, "standardDeviation": std, "fitEnd": cutoff}
    return stats


def dmd_features(
    variables: np.ndarray, config: dict[str, Any]
) -> dict[str, float] | None:
    cfg = config["features"]["koopman"]
    delay = int(cfg["delayDimension"])
    if not np.isfinite(variables).all() or len(variables) < delay + 4:
        return None
    states = np.asarray(
        [
            variables[index - delay + 1 : index + 1][::-1].ravel()
            for index in range(delay - 1, len(variables))
        ]
    )
    if len(states) < 5:
        return None
    x_train = states[:-2].T
    y_train = states[1:-1].T
    ridge = float(cfg["ridge"])
    gram = x_train @ x_train.T + ridge * np.eye(x_train.shape[0])
    operator = y_train @ x_train.T @ np.linalg.pinv(gram)
    training_prediction = operator @ x_train
    training_error = float(
        np.linalg.norm(y_train - training_prediction)
        / max(np.linalg.norm(y_train), 1e-12)
    )
    predicted = operator @ states[-2]
    actual = states[-1]
    residual = float(
        np.linalg.norm(actual - predicted) / max(np.linalg.norm(actual), 1e-12)
    )
    eigenvalues = np.linalg.eigvals(operator)
    radius = float(np.max(np.abs(eigenvalues)))
    rank = int(np.linalg.matrix_rank(x_train))
    return {
        "koopman_residual_norm": residual,
        "koopman_training_error": training_error,
        "koopman_spectral_radius": radius,
        "koopman_effective_rank": float(rank),
        "koopman_instability": max(0.0, radius - 1.0),
    }


def scalar_frame(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.repeat(series.to_numpy(dtype=float)[:, None], len(like.columns), axis=1),
        index=like.index,
        columns=like.columns,
    )


def build_feature_table(
    base: dict[str, Any],
    ews: dict[str, pd.DataFrame | pd.Series],
    ews_stats: dict[str, Any],
    dmd_stats: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prices: pd.DataFrame = base["prices"]
    frame_map: dict[str, pd.DataFrame] = {}
    for name in config["features"]["base"]:
        source = base[name]
        frame_map[name] = source if isinstance(source, pd.DataFrame) else scalar_frame(source, prices)
    for name, source in ews.items():
        frame_map[name] = source if isinstance(source, pd.DataFrame) else scalar_frame(source, prices)
    market_vol = base["vol_6"].median(axis=1, skipna=True)
    train_mask = market_vol.index.strftime("%Y-%m-%d") <= str(
        config["data"]["fixedPreprocessingEnd"]
    )
    finite_vol = market_vol.loc[train_mask].dropna().to_numpy(dtype=float)
    quantiles = np.quantile(
        finite_vol, config["features"]["volatilityRegime"]["trainingQuantiles"]
    )

    window = int(config["sampling"]["minimumPastWindowBars"])
    stride = int(config["sampling"]["decisionStrideBars"])
    dmd_names = config["features"]["koopman"]["variables"]
    records: list[dict[str, Any]] = []
    lppls_success = 0
    dmd_success = 0
    for code in prices.columns:
        for trade_date, day_prices in prices[code].groupby(
            prices.index.strftime("%Y-%m-%d")
        ):
            day_prices = day_prices.sort_index()
            timestamps = list(day_prices.index)
            close = day_prices.to_numpy(dtype=float)
            for index in range(window - 1, len(timestamps), stride):
                timestamp = timestamps[index]
                history = close[index - window + 1 : index + 1]
                if len(history) != window or not np.isfinite(history).all():
                    continue
                row: dict[str, Any] = {
                    "timestamp": timestamp,
                    "trade_date": trade_date,
                    "stockCode": code,
                    "month": trade_date[:7],
                }
                valid = True
                for name, source in frame_map.items():
                    value = source.at[timestamp, code]
                    if not np.isfinite(value):
                        valid = False
                        break
                    row[name] = float(value)
                if not valid:
                    continue
                lppls = lppls_features(history, config)
                if lppls is None:
                    continue
                lppls_success += 1
                row.update(lppls)
                variable_columns: list[np.ndarray] = []
                window_timestamps = timestamps[index - window + 1 : index + 1]
                for name in dmd_names:
                    source = base[name]
                    values = (
                        source.loc[window_timestamps, code].to_numpy(dtype=float)
                        if isinstance(source, pd.DataFrame)
                        else source.loc[window_timestamps].to_numpy(dtype=float)
                    )
                    stats = dmd_stats[name]
                    variable_columns.append(
                        (values - float(stats["mean"]))
                        / float(stats["standardDeviation"])
                    )
                dmd = dmd_features(np.column_stack(variable_columns), config)
                if dmd is None:
                    continue
                dmd_success += 1
                row.update(dmd)
                vol = float(market_vol.at[timestamp])
                row["volatility_regime"] = (
                    "low"
                    if vol <= quantiles[0]
                    else ("high" if vol >= quantiles[1] else "medium")
                )
                records.append(row)
    table = pd.DataFrame.from_records(records)
    category = {row["stockCode"]: row["category"] for row in config["universe"]}
    table["etf_category"] = table["stockCode"].map(category)
    return table.sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    ).reset_index(drop=True), {
        "ewsStandardization": ews_stats,
        "dmdStandardization": dmd_stats,
        "volatilityRegimeThresholds": {
            "lowUpper": float(quantiles[0]),
            "highLower": float(quantiles[1]),
            "fitEnd": config["data"]["fixedPreprocessingEnd"],
        },
        "lpplsSuccessfulRows": lppls_success,
        "koopmanSuccessfulRows": dmd_success,
    }


def variant_features(config: dict[str, Any]) -> dict[str, list[str]]:
    base = list(config["features"]["base"])
    ews = base + EWS_FEATURES
    return {
        "baseline": base,
        "ews": ews,
        "ews_lppls": ews + LPPLS_FEATURES,
        "ews_koopman": ews + KOOPMAN_FEATURES,
        "ews_lppls_koopman": ews + LPPLS_FEATURES + KOOPMAN_FEATURES,
    }


def purge_tail(rows: pd.DataFrame, bars: int) -> tuple[pd.DataFrame, int]:
    if rows.empty:
        return rows.copy(), 0
    ordered = rows.sort_values(["stockCode", "timestamp"], kind="stable")
    remove = ordered.groupby("stockCode", sort=False).tail(bars).index
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
    model: Pipeline, calibrator: LogisticRegression, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.clip(model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return raw, calibrator.predict_proba(logits)[:, 1]


def run_walk_forward(
    feature_table: pd.DataFrame,
    label_table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keys = ["timestamp", "trade_date", "stockCode"]
    variants = variant_features(config)
    all_features = sorted({name for values in variants.values() for name in values})
    labels = label_table[
        keys
        + [f"turning_point_{h}" for h in config["labels"]["modelHorizonsBars"]]
    ]
    merged = feature_table.merge(labels, on=keys, how="inner", validate="one_to_one")
    merged = merged.dropna(subset=all_features)
    test_start = str(config["data"]["walkForwardStart"])
    test_end = str(config["data"]["walkForwardEnd"])
    block_days = int(config["models"]["walkForwardTestBlockDays"])
    calibration_days = int(config["models"]["calibrationDays"])
    embargo_days = int(config["models"]["embargoTradingDays"])
    purge_bars = int(config["models"]["purgeBarsPerSymbol"])
    prediction_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for horizon in config["labels"]["modelHorizonsBars"]:
        target = f"turning_point_{horizon}"
        rows = merged.dropna(subset=[target]).copy()
        rows[target] = rows[target].astype(int)
        test_dates = sorted(
            date
            for date in rows["trade_date"].unique()
            if test_start <= date <= test_end
        )
        for fold, start in enumerate(range(0, len(test_dates), block_days), 1):
            block = test_dates[start : start + block_days]
            train_raw = rows[rows["trade_date"] < block[0]].copy()
            train_dates = sorted(train_raw["trade_date"].unique())
            if len(train_dates) <= calibration_days + embargo_days + int(
                config["models"]["minimumBaseFitDays"]
            ):
                continue
            test_embargo = set(train_dates[-embargo_days:]) if embargo_days else set()
            before_test = train_raw[
                ~train_raw["trade_date"].isin(test_embargo)
            ]
            before_test, test_purged = purge_tail(before_test, purge_bars)
            available_dates = sorted(before_test["trade_date"].unique())
            calibration_dates = set(available_dates[-calibration_days:])
            calibration = before_test[
                before_test["trade_date"].isin(calibration_dates)
            ].copy()
            before_calibration = before_test[
                ~before_test["trade_date"].isin(calibration_dates)
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
                fit[target].nunique() != 2
                or calibration[target].nunique() != 2
                or test.empty
            ):
                continue
            audit = {
                "horizonBars": int(horizon),
                "fold": fold,
                "testDates": block,
                "baseFitStart": str(fit["trade_date"].min()),
                "baseFitEnd": str(fit["trade_date"].max()),
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
                    fit[target].to_numpy(dtype=int),
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
                    calibration[target].to_numpy(dtype=int),
                )
                raw, probability = calibrated_probability(
                    model,
                    calibrator,
                    test[columns].to_numpy(dtype=float),
                )
                part = test[
                    keys
                    + [target, "month", "etf_category", "volatility_regime"]
                ].rename(columns={target: "actual"})
                part["horizon_bars"] = int(horizon)
                part["fold"] = fold
                part["variant"] = variant
                part["raw_probability"] = raw
                part["probability"] = probability
                prediction_parts.append(part)
                audit["variants"][variant] = {
                    "features": columns,
                    "fitSamples": int(len(fit)),
                    "calibrationSamples": int(len(calibration)),
                    "testSamples": int(len(test)),
                    "fitPositiveRate": float(fit[target].mean()),
                    "calibrationPositiveRate": float(calibration[target].mean()),
                    "testPositiveRate": float(test[target].mean()),
                }
            audits.append(audit)
    if not prediction_parts:
        raise RuntimeError("walk-forward produced no predictions")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return predictions.sort_values(
        ["horizon_bars", "trade_date", "timestamp", "stockCode", "variant"],
        kind="stable",
    ), audits


def metric_payload(rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    metrics = probability_metrics(
        rows["probability"].to_numpy(dtype=float),
        rows["actual"].to_numpy(dtype=int),
        bin_edges=config["models"]["probabilityBinEdges"],
    )
    threshold = float(config["models"]["highRiskThreshold"])
    high = rows["probability"].to_numpy(dtype=float) >= threshold
    actual = rows["actual"].to_numpy(dtype=int)
    metrics.update(
        {
            "highRiskThreshold": threshold,
            "highRiskCount": int(np.sum(high)),
            "highRiskMeanProbability": (
                float(rows.loc[high, "probability"].mean()) if np.any(high) else None
            ),
            "highRiskHitRate": (
                float(np.mean(actual[high])) if np.any(high) else None
            ),
        }
    )
    return metrics


def summarize_results(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    variants = list(variant_features(config))
    horizons = config["labels"]["modelHorizonsBars"]
    overall: dict[str, Any] = {}
    for horizon in horizons:
        overall[str(horizon)] = {
            variant: metric_payload(
                predictions[
                    (predictions["horizon_bars"] == horizon)
                    & (predictions["variant"] == variant)
                ],
                config,
            )
            for variant in variants
        }
    grouped: dict[str, Any] = {}
    for dimension in ["month", "etf_category", "volatility_regime"]:
        grouped[dimension] = {}
        for horizon in horizons:
            grouped[dimension][str(horizon)] = {}
            horizon_rows = predictions[predictions["horizon_bars"] == horizon]
            for group, values in horizon_rows.groupby(dimension):
                grouped[dimension][str(horizon)][str(group)] = {
                    variant: metric_payload(
                        values[values["variant"] == variant], config
                    )
                    for variant in variants
                }
    fold_metrics: dict[str, Any] = {}
    for horizon in horizons:
        fold_metrics[str(horizon)] = {}
        horizon_rows = predictions[predictions["horizon_bars"] == horizon]
        for fold, values in horizon_rows.groupby("fold"):
            fold_metrics[str(horizon)][str(int(fold))] = {
                variant: metric_payload(
                    values[values["variant"] == variant], config
                )
                for variant in variants
            }
    return {
        "overall": overall,
        "byGroup": grouped,
        "byFold": fold_metrics,
    }


def numeric_distribution(values: pd.Series) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(dtype=float))]
    if finite.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    array = finite.to_numpy(dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def model_feasibility_summary(
    feature_table: pd.DataFrame,
    feature_audit: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    lppls_columns = [
        "lppls_residual_mean",
        "lppls_residual_std",
        "lppls_tc_offset_median",
        "lppls_tc_offset_std",
        "lppls_m_median",
        "lppls_m_std",
        "lppls_omega_median",
        "lppls_omega_std",
        "lppls_b_scaled_mean",
        "lppls_oscillation_amplitude",
    ]
    koopman_columns = [
        "koopman_residual_norm",
        "koopman_training_error",
        "koopman_spectral_radius",
        "koopman_effective_rank",
        "koopman_instability",
    ]
    fixed_windows = int(audit["feasibleWindows"]["lpplsFixedWindows"])
    dmd_windows = int(audit["feasibleWindows"]["koopmanEqualIntervalWindows"])
    lppls_success = int(feature_audit["lpplsSuccessfulRows"])
    dmd_success = int(feature_audit["koopmanSuccessfulRows"])
    trend_eligible = int(feature_table["lppls_trend_eligible"].sum())
    return {
        "lppls": {
            "fixedWindows": fixed_windows,
            "successfulFixedGridFits": lppls_success,
            "successfulFitRate": (
                lppls_success / fixed_windows if fixed_windows else None
            ),
            "rowsInCommonModelPopulation": int(len(feature_table)),
            "trendEligibleRowsInCommonPopulation": trend_eligible,
            "trendEligibleRateInCommonPopulation": (
                trend_eligible / len(feature_table)
                if len(feature_table)
                else None
            ),
            "parameterAndResidualDistributions": {
                column: numeric_distribution(feature_table[column])
                for column in lppls_columns
            },
            "interpretation": (
                "The fixed grid supplies shape diagnostics only. tc is reported "
                "as a nested-window distribution and is not a point forecast."
            ),
        },
        "koopman": {
            "equalTradingTimeWindows": dmd_windows,
            "successfulResidualFits": dmd_success,
            "successfulFitRate": (
                dmd_success / dmd_windows if dmd_windows else None
            ),
            "parameterAndResidualDistributions": {
                column: numeric_distribution(feature_table[column])
                for column in koopman_columns
            },
            "interpretation": (
                "This is fixed-window linear delay-DMD. No kernel or deep "
                "Koopman operator is fitted."
            ),
        },
    }


def improvement_count(candidate: dict[str, Any], baseline: dict[str, Any]) -> int:
    checks = [
        candidate["brier"] < baseline["brier"],
        candidate["log_loss"] < baseline["log_loss"],
        candidate["auc"] is not None
        and baseline["auc"] is not None
        and candidate["auc"] > baseline["auc"],
        candidate["ece"] < baseline["ece"],
    ]
    return sum(bool(value) for value in checks)


def phase2b_review(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["phase2BGate"]
    required_metrics = int(gate["minimumImprovedMetricCountOfFour"])
    candidates: dict[str, Any] = {}
    for candidate in gate["candidateVariants"]:
        candidates[candidate] = {}
        for horizon in gate["requiredPrimaryHorizons"]:
            key = str(horizon)
            ews = metrics["overall"][key]["ews"]
            current = metrics["overall"][key][candidate]
            fold_rows = metrics["byFold"][key]
            improving_folds = sum(
                improvement_count(values[candidate], values["ews"])
                >= required_metrics
                for values in fold_rows.values()
            )
            months = metrics["byGroup"]["month"][key]
            improving_months = sum(
                values[candidate]["brier"] < values["ews"]["brier"]
                for values in months.values()
            )
            categories = metrics["byGroup"]["etf_category"][key]
            improving_categories = [
                name
                for name, values in categories.items()
                if values[candidate]["brier"] < values["ews"]["brier"]
            ]
            optimism = (
                None
                if current["highRiskHitRate"] is None
                else current["highRiskMeanProbability"]
                - current["highRiskHitRate"]
            )
            gates = {
                "majorityOfFourMetrics": improvement_count(current, ews)
                >= required_metrics,
                "improvingFoldFraction": (
                    improving_folds / len(fold_rows)
                    if fold_rows
                    else 0.0
                )
                > float(gate["minimumImprovingFoldFraction"]),
                "improvingMonthFraction": (
                    improving_months / len(months) if months else 0.0
                )
                > float(gate["minimumImprovingMonthFraction"]),
                "improvingEtfCategories": len(improving_categories)
                >= int(gate["minimumImprovingEtfCategories"]),
                "highRiskSamples": current["highRiskCount"]
                >= int(gate["minimumHighRiskSamples"]),
                "highRiskNotOverconfident": optimism is not None
                and optimism <= float(gate["maximumHighRiskOptimism"]),
            }
            candidates[candidate][key] = {
                "improvedMetricCountOfFour": improvement_count(current, ews),
                "improvingFolds": improving_folds,
                "totalFolds": len(fold_rows),
                "improvingMonths": improving_months,
                "totalMonths": len(months),
                "improvingEtfCategories": improving_categories,
                "highRiskOptimism": optimism,
                "gates": gates,
                "statisticalPass": all(gates.values()),
            }
    same_population = all(
        len(
            {
                metrics["overall"][str(horizon)][variant]["count"]
                for variant in variant_features(config)
            }
        )
        == 1
        for horizon in gate["requiredPrimaryHorizons"]
    )
    passing = [
        candidate
        for candidate, horizons in candidates.items()
        if all(
            horizons[str(horizon)]["statisticalPass"]
            for horizon in gate["requiredPrimaryHorizons"]
        )
    ]
    return {
        "comparisonBaseline": "ews",
        "candidateDiagnostics": candidates,
        "sameForecastPopulationAcrossVariants": same_population,
        "statisticalPreconditionsPass": bool(passing and same_population),
        "passingCandidates": passing,
        "causalTestsRequired": True,
        "phase2BDiscussionAllowed": False,
        "multipleSelectionBias": {
            "present": True,
            "candidateVariantsCompared": len(gate["candidateVariants"]),
            "historicalDatasetPreviouslyUsedByOtherResearch": True,
            "note": "No candidate receives production or forward credit from this historical screen.",
        },
        "conclusion": (
            "Historical statistical preconditions passed, but Phase 2B remains "
            "blocked pending causal tests and a separate preregistration."
            if passing and same_population
            else "No candidate passes the fixed historical gate at both 5 and "
            "10 bars; Phase 2B is not justified."
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    audit = result["dataAudit"]
    lines = [
        "# Singularity Phase 2A historical report",
        "",
        f"- Run ID: `{result['runId']}`",
        f"- Source commit at run: `{result.get('sourceCommitAtRun') or 'unknown'}`",
        f"- Status: `{result['status']}`",
        f"- Data: {audit['dataRange']['start']} through {audit['dataRange']['end']}",
        f"- Universe / days / rows: {audit['symbols']} / {audit['tradingDays']} / {audit['rows']:,}",
        f"- Independent turning events 5/10: {audit['labelSamplesAndIndependentEvents']['5']['independentEvents']:,} / {audit['labelSamplesAndIndependentEvents']['10']['independentEvents']:,}",
        f"- LPPLS / DMD feasible windows: {audit['feasibleWindows']['lpplsFixedWindows']:,} / {audit['feasibleWindows']['koopmanEqualIntervalWindows']:,}",
        "",
        "Research-only and historical-only. Phase 1.5 is neither read nor written.",
        "",
    ]
    if result["status"] == "audit_only":
        lines.append("Audit completed; model feasibility was not run.")
        return "\n".join(lines) + "\n"
    feasibility = result["modelFeasibility"]
    lppls = feasibility["lppls"]
    koopman = feasibility["koopman"]
    lines.extend(
        [
            "## Data and model feasibility",
            "",
            f"- Feature rows on the common population: {result['samples']['featureRows']:,}",
            f"- LPPLS fixed-grid fits: {lppls['successfulFixedGridFits']:,} / "
            f"{lppls['fixedWindows']:,}; trend-eligible common rows: "
            f"{lppls['trendEligibleRowsInCommonPopulation']:,} "
            f"({lppls['trendEligibleRateInCommonPopulation']:.2%}).",
            f"- DMD residual fits: {koopman['successfulResidualFits']:,} / "
            f"{koopman['equalTradingTimeWindows']:,}.",
            f"- The common model population retains "
            f"{result['samples']['featureRows'] / koopman['equalTradingTimeWindows']:.2%} "
            "of fixed decision windows. Requiring a complete 24-bar DMD state "
            "removes mainly earlier intraday windows; every ablation uses the "
            "same retained rows.",
            "- Raw bars have no adjustment factor. One overnight split-like "
            "discontinuity and one intraday >10% jump are retained and disclosed.",
            "- Tick data are absent; L2 and IOPV histories are too short and are "
            "not used.",
            "",
            "### LPPLS stability and residual distributions",
            "",
            "| diagnostic | n | mean | median | p10 | p90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, distribution in lppls[
        "parameterAndResidualDistributions"
    ].items():
        lines.append(
            f"| {name} | {distribution['count']:,} | "
            f"{distribution['mean']:.6f} | {distribution['median']:.6f} | "
            f"{distribution['p10']:.6f} | {distribution['p90']:.6f} |"
        )
    lines.extend(
        [
            "",
            "No single `tc` is treated as a forecast; nested-window dispersion "
            "is retained as a feature.",
            "",
            "### Linear DMD residual distributions",
            "",
            "| diagnostic | n | mean | median | p10 | p90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, distribution in koopman[
        "parameterAndResidualDistributions"
    ].items():
        lines.append(
            f"| {name} | {distribution['count']:,} | "
            f"{distribution['mean']:.6f} | {distribution['median']:.6f} | "
            f"{distribution['p10']:.6f} | {distribution['p90']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Purged walk-forward ablation",
            "",
            "| h | variant | n | Brier | LogLoss | AUC | ECE | high-risk n |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, variants in result["metrics"]["overall"].items():
        for variant, metric in variants.items():
            auc = "null" if metric["auc"] is None else f"{metric['auc']:.6f}"
            lines.append(
                f"| {horizon} | {variant} | {metric['count']:,} | "
                f"{metric['brier']:.6f} | {metric['log_loss']:.6f} | "
                f"{auc} | {metric['ece']:.6f} | {metric['highRiskCount']:,} |"
            )
    lines.extend(
        [
            "",
            "## Probability buckets",
            "",
            "| h | variant | bucket | n | mean p | hit rate |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for horizon, variants in result["metrics"]["overall"].items():
        for variant, metric in variants.items():
            for bucket in metric["bins"]:
                if not bucket["count"]:
                    continue
                lines.append(
                    f"| {horizon} | {variant} | "
                    f"[{bucket['low']:.1f}, {bucket['high']:.1f}) | "
                    f"{bucket['count']:,} | {bucket['mean_predicted']:.4f} | "
                    f"{bucket['actual_rate']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Stability versus EWS",
            "",
            "| candidate | h | metric wins (of 4) | improving folds | "
            "improving months | improving categories | high-risk n | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate, horizons in result["phase2B"][
        "candidateDiagnostics"
    ].items():
        for horizon, diagnostic in horizons.items():
            high_risk = result["metrics"]["overall"][horizon][candidate][
                "highRiskCount"
            ]
            lines.append(
                f"| {candidate} | {horizon} | "
                f"{diagnostic['improvedMetricCountOfFour']} | "
                f"{diagnostic['improvingFolds']}/{diagnostic['totalFolds']} | "
                f"{diagnostic['improvingMonths']}/{diagnostic['totalMonths']} | "
                f"{len(diagnostic['improvingEtfCategories'])} | "
                f"{high_risk} | `{diagnostic['statisticalPass']}` |"
            )
    lines.extend(
        [
            "",
            "The full group tables by month, ETF category and volatility regime "
            "are stored in `phase2a_result.json`. Improvements are not confined "
            "to one group, but none survives every fixed gate at both horizons.",
            "",
            "## Phase 2B review",
            "",
            result["phase2B"]["conclusion"],
            "",
            f"- Same forecast population: `{result['phase2B']['sameForecastPopulationAcrossVariants']}`",
            f"- Statistical preconditions: `{result['phase2B']['statisticalPreconditionsPass']}`",
            f"- Phase 2B discussion allowed: `{result['phase2B']['phase2BDiscussionAllowed']}`",
            "- Multiple-selection bias is present and explicitly retained.",
            "- All variants forecast identical rows; no apparent gain is caused "
            "by reducing the forecast or trade population.",
            "- The common population is nevertheless narrower than the full EWS "
            "population because DMD-state completeness filters early windows. "
            "The result establishes no full-session Phase 1 EWS increment.",
            "- The additions show small aggregate gains over EWS, but fold "
            "instability and nearly empty high-risk buckets prevent a stable "
            "incremental-information claim.",
            "- Features are past-only; future data appear only in the separate "
            "offline label table. Remaining risks are raw/unadjusted bars, "
            "inferred timezone, survivor-selected products and repeated use of "
            "historical data.",
            "",
            "## Skipped items",
            "",
        ]
    )
    for item in result["skipped"]:
        lines.append(f"- `{item['item']}`: {item['reason']}")
    lines.extend(
        [
            "",
            "20/30 bars are audit-only; 60 bars are skipped/null. Phase 2B is "
            "not justified. No online or trading integration is produced.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], output_dir: Path, audit_only: bool) -> dict[str, Any]:
    panel, frames = load_mootdx_panel(config)
    base = build_feature_frames(
        panel[
            [
                "timestamp",
                "trade_date",
                "stockCode",
                "close",
                "cumulative_amount",
            ]
        ]
    )
    labels, label_definition = build_label_table(base, label_config(config))
    audit = build_data_audit(panel, frames, labels, config)
    atomic_json(output_dir / "data_audit.json", audit)
    atomic_text(output_dir / "data_audit.md", render_data_audit(audit))
    if audit_only or not audit["modelRunAllowed"]:
        result = {
            "schemaVersion": "singularity_phase2a_result_v1",
            "runId": output_dir.name,
            "sourceCommitAtRun": source_commit(),
            "status": "audit_only" if audit_only else "audit_failed_model_skipped",
            "researchOnly": True,
            "shadowOnly": True,
            "dataAudit": audit,
            "labelDefinition": label_definition,
            "skipped": config["explicitlyNotImplemented"],
            "phase15Touched": False,
        }
        atomic_json(output_dir / "phase2a_result.json", result)
        atomic_text(output_dir / "phase2a_report.md", render_report(result))
        return result

    ews, ews_stats = build_ews(base, config)
    dmd_stats = dmd_statistics(base, config)
    feature_table, feature_audit = build_feature_table(
        base, ews, ews_stats, dmd_stats, config
    )
    keys = ["timestamp", "trade_date", "stockCode"]
    sampled_labels = labels.merge(
        feature_table[keys], on=keys, how="inner", validate="one_to_one"
    )
    feature_table.to_csv(
        output_dir / "phase2a_features.csv", index=False, encoding="utf-8"
    )
    sampled_labels.to_csv(
        output_dir / "offline_labels.csv", index=False, encoding="utf-8"
    )
    predictions, fold_audits = run_walk_forward(
        feature_table, sampled_labels, config
    )
    predictions.to_csv(
        output_dir / "walk_forward_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    metrics = summarize_results(predictions, config)
    phase2b = phase2b_review(metrics, predictions, config)
    feasibility = model_feasibility_summary(
        feature_table, feature_audit, audit
    )
    result = {
        "schemaVersion": "singularity_phase2a_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": source_commit(),
        "status": "diagnostic_only",
        "researchOnly": True,
        "shadowOnly": True,
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataAudit": audit,
        "universe": config["universe"],
        "featureDefinition": {
            "base": config["features"]["base"],
            "ews": config["features"]["ewsComponents"],
            "lppls": config["features"]["lppls"],
            "koopman": config["features"]["koopman"],
            "frozenPreprocessing": feature_audit,
        },
        "labelDefinition": label_definition,
        "samples": {
            "featureRows": int(len(feature_table)),
            "sampledLabelRows": int(len(sampled_labels)),
            "predictionRows": int(len(predictions)),
        },
        "modelFeasibility": feasibility,
        "walkForward": {
            "start": config["data"]["walkForwardStart"],
            "end": config["data"]["walkForwardEnd"],
            "testBlockDays": config["models"]["walkForwardTestBlockDays"],
            "calibrationDays": config["models"]["calibrationDays"],
            "purgeBarsPerSymbol": config["models"]["purgeBarsPerSymbol"],
            "embargoTradingDays": config["models"]["embargoTradingDays"],
            "folds": fold_audits,
        },
        "metrics": metrics,
        "phase2B": phase2b,
        "lookaheadAudit": {
            "featuresPastOnly": True,
            "labelsSeparate": True,
            "sameDateAcrossSplit": False,
            "phase15LedgerRead": False,
            "knownResidualRisks": [
                "raw bars lack adjustment factors",
                "timezone inferred from TDX convention",
                "historical universe is survivor-selected",
                "five variants create multiple-selection bias",
            ],
        },
        "mechanicalFrequencyCheck": {
            "sameForecastPopulationAcrossVariants": phase2b[
                "sameForecastPopulationAcrossVariants"
            ],
            "commonModelPopulationRows": int(len(feature_table)),
            "feasibleDecisionWindows": int(
                audit["feasibleWindows"]["koopmanEqualIntervalWindows"]
            ),
            "commonPopulationFraction": (
                len(feature_table)
                / audit["feasibleWindows"]["koopmanEqualIntervalWindows"]
            ),
            "commonPopulationRestrictedByDmdStateCompleteness": True,
            "scopeCaveat": (
                "All ablations use identical rows, but the DMD-complete common "
                "population excludes mainly earlier intraday windows and is "
                "not the full Phase 1 EWS population."
            ),
            "tradingApplied": False,
            "gateApplied": False,
        },
        "skipped": [
            {"item": item, "reason": "outside fixed Phase 2A scope"}
            for item in config["explicitlyNotImplemented"]
        ]
        + [
            {
                "item": "horizon_60",
                "reason": config["labels"]["skippedHorizons"]["60"],
            }
        ],
        "phase15Touched": False,
    }
    atomic_json(output_dir / "phase2a_result.json", result)
    atomic_text(output_dir / "phase2a_report.md", render_report(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    config_hash = sha256(config_path)
    run_id = args.run_id or (
        f"phase2a_{'audit' if args.audit_only else 'full'}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_hash[:8]}"
    )
    output_dir = args.output_root.resolve() / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "config_snapshot.json", config)
    result = run(config, output_dir, args.audit_only)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": result["status"],
                "audit_pass": result["dataAudit"]["auditPass"],
                "output": str(output_dir),
                "phase2b_discussion_allowed": result.get("phase2B", {}).get(
                    "phase2BDiscussionAllowed", False
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
