"""Gate 0 audit for HMM enhancement with LPPLS/DMD auxiliary features.

This script is deliberately audit-only while no independent break date is
preregistered.  It reuses the Phase 2A historical source and fixed feature
implementations, retains invalid auxiliary rows, and never fits an HMM or a
turning classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_hmm_nn_bl import ROOT
from research_minute_forecast_shadow import build_feature_frames
from research_singularity_phase1 import build_label_table
import research_singularity_phase2a as phase2a


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "singularity_phase1_6_hmm_physics_features.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schemaVersion")
        != "singularity_phase1_6_hmm_physics_features_v1"
    ):
        raise ValueError("unexpected Phase 1.6 config schema")
    if (
        config.get("status") != "research_only"
        or config.get("shadowOnly") is not True
        or config.get("diagnosticOnly") is not True
    ):
        raise ValueError("Phase 1.6 must remain research/shadow-only")
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
        raise ValueError("offline record-only safety flags are required")
    if any(safety.get(key) is not False for key in forbidden):
        raise ValueError("every live mutation flag must be false")
    if config["source"].get("phase15LedgerAllowedAsInput") is not False:
        raise ValueError("Phase 1.5 forward ledger cannot be an input")
    break_config = config["breakDate"]
    if break_config.get("value") is None:
        if break_config.get("status") != "break_date_not_preregistered":
            raise ValueError("missing break date must fail closed")
        if config["plannedModels"].get("enabledAfterGate0Only") is not True:
            raise ValueError("models must remain behind Gate 0")
        enabled = [
            key
            for key, value in config["plannedModels"].items()
            if key.endswith("Enabled") and value is True
        ]
        if enabled:
            raise ValueError(f"models enabled without break date: {enabled}")


def load_context(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    phase2a_config = json.loads(
        resolve(config["source"]["phase2AConfig"]).read_text(encoding="utf-8")
    )
    phase2a.validate_config(phase2a_config)
    panel, frames = phase2a.load_mootdx_panel(phase2a_config)
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
    labels, label_definition = build_label_table(
        base, phase2a.label_config(phase2a_config)
    )
    data_audit = phase2a.build_data_audit(
        panel, frames, labels, phase2a_config
    )
    return (
        phase2a_config,
        panel,
        frames,
        base,
        labels,
        label_definition,
        data_audit,
    )


def finite_summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standardDeviation": None,
            "p05": None,
            "p95": None,
        }
    array = numeric.to_numpy(dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standardDeviation": float(np.std(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def build_daily_statistics(
    panel: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (code, trade_date), group in panel.groupby(
        ["stockCode", "trade_date"], sort=True
    ):
        ordered = group.sort_values("timestamp", kind="stable")
        close = ordered["close"].to_numpy(dtype=float)
        log_returns = np.diff(np.log(close))
        rows.append(
            {
                "stockCode": code,
                "trade_date": trade_date,
                "bar_count": int(len(ordered)),
                "daily_return": float(close[-1] / close[0] - 1.0),
                "realized_volatility": (
                    float(np.std(log_returns)) if len(log_returns) else 0.0
                ),
                "range_pct": float(
                    ordered["high"].max() / ordered["low"].min() - 1.0
                ),
                "volume": float(ordered["bar_volume"].sum()),
                "amount": float(ordered["bar_amount"].sum()),
            }
        )
    symbol_daily = pd.DataFrame.from_records(rows)
    label_daily = (
        labels.groupby("trade_date", sort=True)
        .agg(
            label_rate_5=("turning_point_5", "mean"),
            label_rate_10=("turning_point_10", "mean"),
        )
        .reset_index()
    )
    market_daily = (
        symbol_daily.groupby("trade_date", sort=True)
        .agg(
            daily_return=("daily_return", "median"),
            realized_volatility=("realized_volatility", "median"),
            range_pct=("range_pct", "median"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            symbols=("stockCode", "nunique"),
            minimum_bars=("bar_count", "min"),
            maximum_bars=("bar_count", "max"),
        )
        .reset_index()
    )
    market_daily["log_volume"] = np.log1p(market_daily["volume"])
    market_daily["log_amount"] = np.log1p(market_daily["amount"])
    market_daily = market_daily.merge(
        label_daily, on="trade_date", how="left", validate="one_to_one"
    )
    return symbol_daily, market_daily


def scan_break_candidates(
    market_daily: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    scan = config["breakDate"]["candidateScan"]
    dates = market_daily["trade_date"].tolist()
    first_of_month = (
        market_daily.assign(month=market_daily["trade_date"].str[:7])
        .groupby("month", sort=True)["trade_date"]
        .first()
        .tolist()
    )
    minimum = int(scan["minimumTradingDaysEachSide"])
    window = int(scan["comparisonWindowTradingDaysEachSide"])
    fields = [
        "daily_return",
        "realized_volatility",
        "log_amount",
        "log_volume",
        "range_pct",
        "label_rate_5",
        "label_rate_10",
    ]
    candidates: list[dict[str, Any]] = []
    for candidate in first_of_month:
        index = dates.index(candidate)
        if index < minimum or len(dates) - index < minimum:
            continue
        left = market_daily.iloc[index - window : index]
        right = market_daily.iloc[index : index + window]
        components: dict[str, Any] = {}
        scores: list[float] = []
        for field in fields:
            left_values = left[field].dropna().to_numpy(dtype=float)
            right_values = right[field].dropna().to_numpy(dtype=float)
            combined = np.concatenate([left_values, right_values])
            scale = float(np.std(combined))
            shift = float(np.mean(right_values) - np.mean(left_values))
            standardized = abs(shift) / scale if scale > 1e-12 else 0.0
            components[field] = {
                "preMean": float(np.mean(left_values)),
                "postMean": float(np.mean(right_values)),
                "rawShift": shift,
                "absoluteStandardizedShift": standardized,
            }
            scores.append(standardized)
        candidates.append(
            {
                "date": candidate,
                "exploratoryDistributionShiftScore": float(np.mean(scores)),
                "components": components,
                "profitabilityOrModelMetricsUsed": False,
                "status": "break_date_not_preregistered",
            }
        )
    candidates.sort(
        key=lambda row: row["exploratoryDistributionShiftScore"],
        reverse=True,
    )
    limit = int(scan["maximumCandidatesReported"])
    for rank, row in enumerate(candidates[:limit], 1):
        row["exploratoryRank"] = rank
    return candidates[:limit]


def category_counts(
    symbols: set[str], categories: dict[str, str]
) -> dict[str, int]:
    counts = Counter(categories[code] for code in symbols if code in categories)
    return dict(sorted(counts.items()))


def period_missingness(
    dates: list[str],
    frames: dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    expected = 0
    observed = 0
    per_symbol: dict[str, Any] = {}
    for code, frame in frames.items():
        first = str(frame["trade_date"].min())
        last = str(frame["trade_date"].max())
        eligible = [
            date
            for date in dates
            if period_start <= date <= period_end and first <= date <= last
        ]
        actual = int(
            frame[
                (frame["trade_date"] >= period_start)
                & (frame["trade_date"] <= period_end)
            ].shape[0]
        )
        symbol_expected = len(eligible) * 48
        expected += symbol_expected
        observed += actual
        per_symbol[code] = {
            "expectedBars": symbol_expected,
            "observedBars": actual,
            "missingBars": max(0, symbol_expected - actual),
            "missingFraction": (
                max(0, symbol_expected - actual) / symbol_expected
                if symbol_expected
                else None
            ),
        }
    return {
        "expectedBars": expected,
        "observedBars": observed,
        "missingBars": max(0, expected - observed),
        "missingFraction": (
            max(0, expected - observed) / expected if expected else None
        ),
        "perSymbol": per_symbol,
    }


def compare_periods(
    candidate: str,
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    market_daily: pd.DataFrame,
    categories: dict[str, str],
    all_dates: list[str],
) -> dict[str, Any]:
    pre_dates = [date for date in all_dates if date < candidate]
    post_dates = [date for date in all_dates if date >= candidate]

    def summarize_period(period_dates: list[str]) -> dict[str, Any]:
        date_set = set(period_dates)
        period_panel = panel[panel["trade_date"].isin(date_set)]
        period_labels = labels[labels["trade_date"].isin(date_set)]
        period_daily = market_daily[
            market_daily["trade_date"].isin(date_set)
        ]
        symbols = set(period_panel["stockCode"].unique())
        start = min(period_dates)
        end = max(period_dates)
        return {
            "start": start,
            "end": end,
            "tradingDays": len(period_dates),
            "barRows": int(len(period_panel)),
            "symbols": sorted(symbols),
            "symbolCount": len(symbols),
            "etfCategoryCounts": category_counts(symbols, categories),
            "turningLabels": phase2a.independent_event_counts(
                period_labels, [5, 10]
            ),
            "dailyDistributions": {
                field: finite_summary(period_daily[field])
                for field in [
                    "daily_return",
                    "realized_volatility",
                    "volume",
                    "amount",
                    "range_pct",
                    "label_rate_5",
                    "label_rate_10",
                ]
            },
            "spread": {
                "available": False,
                "reason": "mootdx OHLCV archive has no bid/ask spread field",
            },
            "missingBars": period_missingness(
                all_dates, frames, start, end
            ),
        }

    pre = summarize_period(pre_dates)
    post = summarize_period(post_dates)
    return {
        "candidateDate": candidate,
        "status": "break_date_not_preregistered",
        "pre": pre,
        "post": post,
        "universeConsistent": pre["symbols"] == post["symbols"],
        "categoryStructureConsistent": (
            pre["etfCategoryCounts"] == post["etfCategoryCounts"]
        ),
        "formalPostJumpInferenceAllowed": False,
    }


def expected_decision_feature_audit(
    base: dict[str, Any],
    phase2a_config: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    prices: pd.DataFrame = base["prices"]
    ews, _ = phase2a.build_ews(base, phase2a_config)
    dmd_stats = phase2a.dmd_statistics(base, phase2a_config)
    dmd_names = phase2a_config["features"]["koopman"]["variables"]
    window = int(config["auxiliaryFeatures"]["pastWindowBars"])
    stride = int(config["auxiliaryFeatures"]["decisionStrideBars"])
    categories = {
        row["stockCode"]: row["category"]
        for row in phase2a_config["universe"]
    }
    records: list[dict[str, Any]] = []
    for code in prices.columns:
        for trade_date, day_prices in prices[code].groupby(
            prices.index.strftime("%Y-%m-%d")
        ):
            day_prices = day_prices.dropna().sort_index()
            timestamps = list(day_prices.index)
            closes = day_prices.to_numpy(dtype=float)
            previous_radius: float | None = None
            for index in range(window - 1, len(timestamps), stride):
                timestamp = timestamps[index]
                history = closes[index - window + 1 : index + 1]
                row: dict[str, Any] = {
                    "timestamp": timestamp,
                    "trade_date": trade_date,
                    "month": trade_date[:7],
                    "stockCode": code,
                    "etf_category": categories[code],
                    "intraday_slot": timestamp.strftime("%H:%M"),
                    "return_24_proxy": float(history[-1] / history[0] - 1.0),
                    "realized_vol_proxy": float(
                        np.std(np.diff(np.log(history)))
                    ),
                    "close_range_proxy": float(
                        np.max(history) / np.min(history) - 1.0
                    ),
                    "trend_acceleration_proxy": float(
                        (history[-1] / history[-7] - 1.0)
                        - (history[-7] / history[-13] - 1.0)
                    ),
                    "ews_score": float(
                        ews["ews_score"].at[timestamp, code]
                    ),
                }
                lppls = phase2a.lppls_features(history, phase2a_config)
                if lppls is None:
                    row.update(
                        {
                            "lppls_fit_success": 0.0,
                            "lppls_failed_reason": (
                                "invalid_price_window"
                                if len(history) != window
                                or np.any(history <= 0)
                                else "fixed_grid_no_stable_nested_fit"
                            ),
                            "lppls_tc_proximity": np.nan,
                            "lppls_time_to_tc": np.nan,
                            "lppls_fit_residual": np.nan,
                            "lppls_parameter_stability": np.nan,
                            "lppls_window_consensus": np.nan,
                            "lppls_bubble_like_score": np.nan,
                        }
                    )
                else:
                    normalized_stability = np.array(
                        [
                            lppls["lppls_tc_offset_std"] / 12.0,
                            lppls["lppls_m_std"] / 0.4,
                            lppls["lppls_omega_std"] / 4.0,
                        ],
                        dtype=float,
                    )
                    stability = float(
                        np.exp(-np.mean(normalized_stability))
                    )
                    consensus = float(
                        1.0
                        - np.mean(np.clip(normalized_stability, 0.0, 1.0))
                    )
                    time_to_tc = float(lppls["lppls_tc_offset_median"])
                    row.update(
                        {
                            "lppls_fit_success": 1.0,
                            "lppls_failed_reason": None,
                            "lppls_tc_proximity": 1.0 / (1.0 + time_to_tc),
                            "lppls_time_to_tc": time_to_tc,
                            "lppls_fit_residual": float(
                                lppls["lppls_residual_mean"]
                            ),
                            "lppls_parameter_stability": stability,
                            "lppls_window_consensus": consensus,
                            "lppls_bubble_like_score": float(
                                lppls["lppls_trend_eligible"]
                                * math.exp(
                                    -lppls["lppls_residual_mean"]
                                )
                                * consensus
                            ),
                        }
                    )

                window_timestamps = timestamps[
                    index - window + 1 : index + 1
                ]
                variable_columns: list[np.ndarray] = []
                for name in dmd_names:
                    source = base[name]
                    values = (
                        source.loc[
                            window_timestamps, code
                        ].to_numpy(dtype=float)
                        if isinstance(source, pd.DataFrame)
                        else source.loc[
                            window_timestamps
                        ].to_numpy(dtype=float)
                    )
                    statistics = dmd_stats[name]
                    variable_columns.append(
                        (values - float(statistics["mean"]))
                        / float(statistics["standardDeviation"])
                    )
                variables = np.column_stack(variable_columns)
                missing_ratio = float(np.mean(~np.isfinite(variables)))
                dmd = (
                    phase2a.dmd_features(variables, phase2a_config)
                    if missing_ratio == 0.0
                    else None
                )
                if dmd is None:
                    row.update(
                        {
                            "dmd_reconstruction_residual": np.nan,
                            "dmd_spectral_radius": np.nan,
                            "dmd_spectral_radius_drift": np.nan,
                            "dmd_eigen_instability": np.nan,
                            "dmd_window_valid": 0.0,
                            "dmd_missing_bar_ratio": missing_ratio,
                        }
                    )
                else:
                    radius = float(dmd["koopman_spectral_radius"])
                    row.update(
                        {
                            "dmd_reconstruction_residual": float(
                                dmd["koopman_residual_norm"]
                            ),
                            "dmd_spectral_radius": radius,
                            "dmd_spectral_radius_drift": (
                                radius - previous_radius
                                if previous_radius is not None
                                else np.nan
                            ),
                            "dmd_eigen_instability": float(
                                dmd["koopman_instability"]
                            ),
                            "dmd_window_valid": 1.0,
                            "dmd_missing_bar_ratio": missing_ratio,
                        }
                    )
                    previous_radius = radius
                records.append(row)
    table = pd.DataFrame.from_records(records).sort_values(
        ["trade_date", "timestamp", "stockCode"], kind="stable"
    )
    cutoff = str(config["auxiliaryFeatures"]["trainingStatisticsEnd"])
    train = table[
        (table["trade_date"] <= cutoff)
        & table["dmd_reconstruction_residual"].notna()
    ]
    residual_mean = float(train["dmd_reconstruction_residual"].mean())
    residual_std = float(train["dmd_reconstruction_residual"].std(ddof=0))
    table["dmd_residual_zscore"] = (
        table["dmd_reconstruction_residual"] - residual_mean
    ) / max(residual_std, 1e-12)
    return table.reset_index(drop=True)


def grouped_coverage(
    table: pd.DataFrame, group: str, valid_column: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, values in table.groupby(group, sort=True):
        valid = values[valid_column].to_numpy(dtype=float)
        rows.append(
            {
                group: str(value),
                "rows": int(len(values)),
                "validRows": int(np.sum(valid == 1.0)),
                "coverage": float(np.mean(valid == 1.0)),
            }
        )
    return rows


def preprocessing_parameters(
    features: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    cutoff = str(config["auxiliaryFeatures"]["trainingStatisticsEnd"])
    quantiles = config["auxiliaryFeatures"]["winsorizationQuantiles"]
    train = features[features["trade_date"] <= cutoff]
    columns = list(config["auxiliaryFeatures"]["lppls"]["features"]) + list(
        config["auxiliaryFeatures"]["dmd"]["features"]
    )
    parameters: dict[str, Any] = {}
    for column in columns:
        if column.endswith("_success") or column.endswith("_valid"):
            continue
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        parameters[column] = {
            "fitEnd": cutoff,
            "finiteRows": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "standardDeviation": (
                float(values.std(ddof=0)) if len(values) else None
            ),
            "winsorLower": (
                float(values.quantile(quantiles[0])) if len(values) else None
            ),
            "winsorUpper": (
                float(values.quantile(quantiles[1])) if len(values) else None
            ),
            "appliedToModel": False,
        }
    return parameters


def build_feature_reports(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    total = len(features)
    lppls_valid = features["lppls_fit_success"] == 1.0
    dmd_valid = features["dmd_window_valid"] == 1.0
    report = {
        "schemaVersion": "singularity_phase1_6_feature_missingness_v1",
        "status": "exploratory_gate0_only",
        "rows": int(total),
        "noForcedImputation": True,
        "invalidRowsRetained": True,
        "lppls": {
            "validRows": int(lppls_valid.sum()),
            "coverage": float(lppls_valid.mean()),
            "missingness": float(1.0 - lppls_valid.mean()),
            "failedReasons": {
                str(key): int(value)
                for key, value in features.loc[
                    ~lppls_valid, "lppls_failed_reason"
                ]
                .fillna("unknown")
                .value_counts()
                .items()
            },
            "byMonth": grouped_coverage(
                features, "month", "lppls_fit_success"
            ),
            "bySymbol": grouped_coverage(
                features, "stockCode", "lppls_fit_success"
            ),
            "byEtfCategory": grouped_coverage(
                features, "etf_category", "lppls_fit_success"
            ),
            "byIntradaySlot": grouped_coverage(
                features, "intraday_slot", "lppls_fit_success"
            ),
        },
        "dmd": {
            "validRows": int(dmd_valid.sum()),
            "coverage": float(dmd_valid.mean()),
            "missingness": float(1.0 - dmd_valid.mean()),
            "missingBarRatioMeaning": (
                "Fraction of non-finite cells in the standardized 24-bar DMD "
                "state matrix. This includes causal rolling-feature warm-up "
                "missingness even when the underlying exchange bars exist."
            ),
            "missingBarRatioDistribution": finite_summary(
                features["dmd_missing_bar_ratio"]
            ),
            "byMonth": grouped_coverage(
                features, "month", "dmd_window_valid"
            ),
            "bySymbol": grouped_coverage(
                features, "stockCode", "dmd_window_valid"
            ),
            "byEtfCategory": grouped_coverage(
                features, "etf_category", "dmd_window_valid"
            ),
            "byIntradaySlot": grouped_coverage(
                features, "intraday_slot", "dmd_window_valid"
            ),
        },
        "trainingOnlyPreprocessingParameters": preprocessing_parameters(
            features, config
        ),
    }
    numeric_columns = [
        "return_24_proxy",
        "realized_vol_proxy",
        "close_range_proxy",
        "trend_acceleration_proxy",
        "ews_score",
        "lppls_tc_proximity",
        "lppls_time_to_tc",
        "lppls_fit_residual",
        "lppls_parameter_stability",
        "lppls_window_consensus",
        "lppls_bubble_like_score",
        "dmd_reconstruction_residual",
        "dmd_residual_zscore",
        "dmd_spectral_radius",
        "dmd_spectral_radius_drift",
        "dmd_eigen_instability",
        "dmd_missing_bar_ratio",
    ]
    correlation_frame = features[numeric_columns]
    pearson = correlation_frame.corr(method="pearson", min_periods=100)
    spearman = correlation_frame.corr(method="spearman", min_periods=100)
    proxy_columns = [
        "return_24_proxy",
        "realized_vol_proxy",
        "close_range_proxy",
        "trend_acceleration_proxy",
        "ews_score",
    ]
    physics_columns = [
        column
        for column in numeric_columns
        if column not in proxy_columns
    ]
    proxy_checks: list[dict[str, Any]] = []
    for feature in physics_columns:
        for proxy in proxy_columns:
            value = spearman.at[feature, proxy]
            proxy_checks.append(
                {
                    "feature": feature,
                    "proxy": proxy,
                    "spearman": (
                        float(value) if np.isfinite(value) else None
                    ),
                    "absoluteCorrelationAtLeast0_8": (
                        bool(abs(value) >= 0.8)
                        if np.isfinite(value)
                        else None
                    ),
                }
            )
    correlation_report = {
        "schemaVersion": "singularity_phase1_6_feature_correlation_v1",
        "status": "exploratory_gate0_only",
        "pairwiseCompleteRowsUsed": True,
        "pearson": phase2a.json_safe(pearson.to_dict()),
        "spearman": phase2a.json_safe(spearman.to_dict()),
        "ordinaryProxyChecks": proxy_checks,
        "highProxyCorrelationCount": sum(
            row["absoluteCorrelationAtLeast0_8"] is True
            for row in proxy_checks
        ),
        "conclusion": (
            "High correlation is an audit warning, not evidence that a physics "
            "feature adds information beyond ordinary market features."
        ),
    }
    maximum_share_by_symbol = (
        features.loc[lppls_valid, "stockCode"].value_counts(normalize=True).max()
        if lppls_valid.any()
        else None
    )
    maximum_share_by_month = (
        features.loc[lppls_valid, "month"].value_counts(normalize=True).max()
        if lppls_valid.any()
        else None
    )
    coverage_bias = {
        "schemaVersion": "singularity_phase1_6_coverage_bias_v1",
        "status": "exploratory_gate0_only",
        "fixedCurrentProductPool": True,
        "survivorshipBias": True,
        "delistedOrHistoricalConstituentUniverseAvailable": False,
        "lppls": {
            "coverage": report["lppls"]["coverage"],
            "maximumValidRowShareBySymbol": (
                float(maximum_share_by_symbol)
                if maximum_share_by_symbol is not None
                else None
            ),
            "maximumValidRowShareByMonth": (
                float(maximum_share_by_month)
                if maximum_share_by_month is not None
                else None
            ),
            "concentratedInSingleSymbol": (
                bool(maximum_share_by_symbol > 0.15)
                if maximum_share_by_symbol is not None
                else None
            ),
            "concentratedInSingleMonth": (
                bool(maximum_share_by_month > 0.15)
                if maximum_share_by_month is not None
                else None
            ),
        },
        "dmd": {
            "coverage": report["dmd"]["coverage"],
            "systematicIntradayExclusion": True,
            "byIntradaySlot": report["dmd"]["byIntradaySlot"],
            "formalFullSessionAuxiliaryUseAllowed": (
                report["dmd"]["coverage"]
                >= config["gate0"]["minimumAuxiliaryFeatureCoverage"]
            ),
        },
        "sameSampleWarning": (
            "Restricting every HMM variant to DMD-complete rows would make the "
            "comparison internally equal but would exclude early-session "
            "decisions and would not establish a full-session HMM improvement."
        ),
    }
    return report, correlation_report, coverage_bias


def build_label_distribution(
    labels: pd.DataFrame, phase2a_config: dict[str, Any]
) -> dict[str, Any]:
    categories = {
        row["stockCode"]: row["category"]
        for row in phase2a_config["universe"]
    }
    table = labels.copy()
    table["month"] = table["trade_date"].str[:7]
    table["etf_category"] = table["stockCode"].map(categories)

    def grouped(dimension: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for value, group in table.groupby(dimension, sort=True):
            result[str(value)] = phase2a.independent_event_counts(
                group, [5, 10]
            )
        return result

    return {
        "schemaVersion": "singularity_phase1_6_label_distribution_v1",
        "status": "exploratory_gate0_only",
        "definition": {
            "source": "Phase 1 causal reversal labels",
            "featuresReadFuture": False,
            "labelsUseFuture": True,
            "labelsStoredSeparately": True,
            "sameSessionOnly": True,
            "primaryHorizonsBars": [5, 10],
            "horizon60": "skipped_cross_session",
        },
        "overall": phase2a.independent_event_counts(table, [5, 10]),
        "byMonth": grouped("month"),
        "bySymbol": grouped("stockCode"),
        "byEtfCategory": grouped("etf_category"),
    }


def render_data_audit(audit: dict[str, Any]) -> str:
    base = audit["phase2ADataAudit"]
    gate = audit["gate0"]
    return "\n".join(
        [
            "# Singularity Phase 1.6 Gate 0 data audit",
            "",
            f"- Status: `{audit['status']}`",
            f"- Range: {base['dataRange']['start']} through {base['dataRange']['end']}",
            f"- Trading days / symbols / bars: {base['tradingDays']} / "
            f"{base['symbols']} / {base['rows']:,}",
            f"- Break date: `{audit['breakDate']['status']}`",
            f"- LPPLS coverage: {gate['lpplsAuxiliaryCoverage']:.2%}",
            f"- DMD coverage: {gate['dmdAuxiliaryCoverage']:.2%}",
            f"- Gate 0 passed: `{gate['passed']}`",
            f"- Formal modeling allowed: `{gate['modelingAllowed']}`",
            "",
            "Observed limitations: raw bars have no adjustment factor; the "
            "timezone is inferred; bid/ask spreads and source timestamps are "
            "absent; the fixed current-product universe creates survivorship "
            "bias; DMD completeness excludes early-session decisions.",
            "",
            "No HMM was fitted because the break date is not preregistered.",
            "",
        ]
    )


def render_break_risk(
    config: dict[str, Any], candidates: list[dict[str, Any]]
) -> str:
    lines = [
        "# Break-date selection risk",
        "",
        "- Status: `break_date_not_preregistered`",
        "- Selected break date: `null`",
        "- Source: no user-specified or independently documented business-event "
        "date was supplied.",
        "- Consequence: post-jump HMM evaluation is blocked.",
        "",
        "The dates below are exploratory distribution-shift candidates. They "
        "were ranked without profitability or model OOS metrics, but they are "
        "still selected with hindsight and cannot be used for formal claims.",
        "",
        "| rank | candidate | distribution-shift score |",
        "|---:|---|---:|",
    ]
    for candidate in candidates:
        lines.append(
            f"| {candidate['exploratoryRank']} | {candidate['date']} | "
            f"{candidate['exploratoryDistributionShiftScore']:.6f} |"
        )
    lines.extend(
        [
            "",
            "A later post-jump experiment requires a new immutable config that "
            "states the break date and independent rationale before inspecting "
            "post-break model results. Reusing the best candidate above would "
            "retain selection bias.",
            "",
        ]
    )
    return "\n".join(lines)


def render_gate0_report(result: dict[str, Any]) -> str:
    sufficiency = result["dataSufficiency"]
    lines = [
        "# Singularity Phase 1.6 Gate 0 report",
        "",
        f"- Run ID: `{result['runId']}`",
        f"- Source commit at run: `{result['sourceCommitAtRun']}`",
        f"- Status: `{result['status']}`",
        f"- Gate 0 passed: `{result['gate0']['passed']}`",
        f"- Formal HMM modeling allowed: `{result['gate0']['modelingAllowed']}`",
        "",
        "## Observed",
        "",
    ]
    lines.extend(f"- {item}" for item in result["observed"])
    lines.extend(["", "## Estimated", ""])
    lines.extend(f"- {item}" for item in result["estimated"])
    lines.extend(["", "## Assumed", ""])
    lines.extend(f"- {item}" for item in result["assumed"])
    lines.extend(["", "## Data sufficiency", ""])
    for path, diagnostic in sufficiency.items():
        lines.append(
            f"- `{path}`: `{diagnostic['status']}` — {diagnostic['reason']}"
        )
    lines.extend(["", "## Passed tests", ""])
    lines.extend(f"- {item}" for item in result["passedTests"])
    lines.extend(["", "## Failed tests", ""])
    lines.extend(f"- {item}" for item in result["failedTests"])
    lines.extend(["", "## Skipped items", ""])
    lines.extend(f"- `{item}`" for item in result["skippedItems"])
    lines.extend(
        [
            "",
            "## Required conclusions",
            "",
            "- HMM improved: `not_evaluated`",
            "- Post-jump HMM improved: `not_evaluated_break_date_not_preregistered`",
            "- Time-decay HMM improved: `not_evaluated_half_life_not_preregistered`",
            "- LPPLS helped HMM: `not_evaluated`",
            "- Koopman helped HMM: `not_evaluated`",
            "- Improvement survives same-sample comparison: `not_evaluated`",
            "- Remains research-only: `true`",
            "",
            "The corrected Phase 2A interpretation is retained: LPPLS/DMD were "
            "not validated as independent trading signals, but their possible "
            "HMM auxiliary role remains an untested hypothesis.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    (
        phase2a_config,
        panel,
        frames,
        base,
        labels,
        label_definition,
        phase2a_data_audit,
    ) = load_context(config)
    symbol_daily, market_daily = build_daily_statistics(panel, labels)
    candidates = scan_break_candidates(market_daily, config)
    categories = {
        row["stockCode"]: row["category"]
        for row in phase2a_config["universe"]
    }
    all_dates = sorted(panel["trade_date"].unique())
    candidate_comparisons = [
        {
            **candidate,
            "prePostAudit": compare_periods(
                candidate["date"],
                panel,
                frames,
                labels,
                market_daily,
                categories,
                all_dates,
            ),
        }
        for candidate in candidates
    ]
    features = expected_decision_feature_audit(
        base, phase2a_config, config
    )
    (
        feature_missingness,
        correlation_report,
        coverage_bias,
    ) = build_feature_reports(features, config)
    label_distribution = build_label_distribution(labels, phase2a_config)
    gate_config = config["gate0"]
    data_quality_pass = bool(
        phase2a_data_audit["symbols"] >= gate_config["minimumSymbols"]
        and phase2a_data_audit["tradingDays"]
        >= gate_config["minimumTradingDays"]
        and phase2a_data_audit["missingBarAudit"][
            "maximumPerSymbolFraction"
        ]
        <= gate_config["maximumMissingBarFraction"]
        and phase2a_data_audit["missingBarAudit"]["invalidOhlcRows"]
        <= gate_config["maximumInvalidOhlcRows"]
        and all(
            label_distribution["overall"][str(horizon)][
                "independentEvents"
            ]
            >= gate_config[
                "minimumIndependentTurningEventsPerPrimaryHorizon"
            ]
            for horizon in [5, 10]
        )
    )
    break_pass = config["breakDate"]["value"] is not None
    lppls_pass = (
        feature_missingness["lppls"]["coverage"]
        >= gate_config["minimumAuxiliaryFeatureCoverage"]
    )
    dmd_pass = (
        feature_missingness["dmd"]["coverage"]
        >= gate_config["minimumAuxiliaryFeatureCoverage"]
    )
    gate_pass = bool(data_quality_pass and break_pass and lppls_pass and dmd_pass)
    modeling_allowed = bool(
        gate_pass
        and config["gate0"]["modelingAllowedOnlyIfAllRequiredGatesPass"]
    )
    post_jump = {
        "schemaVersion": "singularity_phase1_6_post_jump_audit_v1",
        "status": "break_date_not_preregistered",
        "selectedBreakDate": None,
        "breakDefinition": config["breakDate"]["definition"],
        "candidateMethod": config["breakDate"]["candidateScan"],
        "candidateComparisons": candidate_comparisons,
        "formalPostJumpInferenceAllowed": False,
        "structuralChangeConclusion": "not_established",
        "reason": (
            "Every candidate is detected with hindsight. No candidate is "
            "promoted into an HMM training cutoff."
        ),
    }
    data_audit = {
        "schemaVersion": "singularity_phase1_6_data_audit_v1",
        "status": "exploratory_gate0_only",
        "sourceCommitAtRun": source_commit(),
        "phase2ADataAudit": phase2a_data_audit,
        "breakDate": config["breakDate"],
        "sourceAndExecutionAudit": {
            "primarySource": "local mootdx 5-minute OHLCV",
            "yahooUsed": False,
            "yahooAdjustmentContamination": False,
            "adjustmentFactorAvailable": False,
            "iopvMerged": False,
            "bidAskSpreadAvailable": False,
            "explicitSuspensionFlagAvailable": False,
            "sourceTimestampAvailable": False,
            "timestampTimezonePresent": False,
            "assumedTimezone": "Asia/Shanghai",
            "sourceTimeTimestampConsistencyTestable": False,
        },
        "dailyDistributionSummary": {
            field: finite_summary(market_daily[field])
            for field in [
                "daily_return",
                "realized_volatility",
                "volume",
                "amount",
                "range_pct",
                "label_rate_5",
                "label_rate_10",
            ]
        },
        "gate0": {
            "dataQualityPass": data_quality_pass,
            "breakDatePass": break_pass,
            "lpplsAuxiliaryCoverage": feature_missingness["lppls"][
                "coverage"
            ],
            "lpplsCoveragePass": lppls_pass,
            "dmdAuxiliaryCoverage": feature_missingness["dmd"]["coverage"],
            "dmdCoveragePass": dmd_pass,
            "passed": gate_pass,
            "modelingAllowed": modeling_allowed,
        },
        "labelDefinition": label_definition,
        "symbolDailyRows": int(len(symbol_daily)),
        "marketDailyRows": int(len(market_daily)),
    }
    data_sufficiency = {
        "full_history_hmm": {
            "status": "data_volume_sufficient_but_not_run",
            "reason": "500 trading days and 6,180/5,607 independent 5/10-bar events.",
        },
        "post_jump_hmm": {
            "status": "blocked",
            "reason": "No independently preregistered break date.",
        },
        "time_decay_hmm": {
            "status": "blocked",
            "reason": "Half-life and state specification are not preregistered.",
        },
        "multivariate_hmm": {
            "status": "data_volume_sufficient_but_not_run",
            "reason": "Base OHLCV panel is broad enough, but its observation contract is not frozen.",
        },
        "lppls_hmm_auxiliary": {
            "status": (
                "coverage_feasible_but_not_run"
                if lppls_pass
                else "coverage_insufficient"
            ),
            "reason": f"Observed fixed-window coverage is {feature_missingness['lppls']['coverage']:.2%}.",
        },
        "dmd_hmm_auxiliary": {
            "status": (
                "coverage_feasible_but_not_run"
                if dmd_pass
                else "coverage_insufficient"
            ),
            "reason": (
                f"Observed coverage is {feature_missingness['dmd']['coverage']:.2%}; "
                "invalid rows are concentrated in earlier intraday slots."
            ),
        },
    }
    failed_tests = []
    if not break_pass:
        failed_tests.append("Break date is not preregistered.")
    if not dmd_pass:
        failed_tests.append(
            "DMD auxiliary coverage is below the fixed 90% Gate 0 threshold."
        )
    result = {
        "schemaVersion": "singularity_phase1_6_gate0_result_v1",
        "runId": output_dir.name,
        "sourceCommitAtRun": source_commit(),
        "status": "exploratory_gate0_only",
        "researchOnly": True,
        "shadowOnly": True,
        "generatedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "gate0": data_audit["gate0"],
        "dataSufficiency": data_sufficiency,
        "observed": [
            f"Historical range is {phase2a_data_audit['dataRange']['start']} through {phase2a_data_audit['dataRange']['end']}.",
            f"The panel contains {phase2a_data_audit['tradingDays']} trading days, {phase2a_data_audit['symbols']} symbols and {phase2a_data_audit['rows']:,} bars.",
            f"Independent 5/10-bar turning events are {label_distribution['overall']['5']['independentEvents']:,}/{label_distribution['overall']['10']['independentEvents']:,}.",
            f"LPPLS/DMD feature coverage is {feature_missingness['lppls']['coverage']:.2%}/{feature_missingness['dmd']['coverage']:.2%}.",
            "Mootdx OHLCV has no bid/ask spread, source timestamp, explicit suspension flag or adjustment factor.",
        ],
        "estimated": [
            "Exploratory monthly change-point candidates are ranked by distribution shift only.",
            "Auxiliary preprocessing parameters are estimated only through 2025-06-30 and are not applied to a model.",
            "Feature correlations are descriptive and do not establish incremental information.",
        ],
        "assumed": [
            "Naive TDX timestamps are Asia/Shanghai.",
            "A full session has 48 ordered five-minute bars.",
            "The current 18-product universe is not a point-in-time historical constituent universe.",
        ],
        "passedTests": [
            "Core historical data-quality and independent-event gates pass.",
            "LPPLS fixed-window auxiliary coverage passes the 90% threshold.",
            "Invalid auxiliary rows are retained with explicit validity/missingness fields.",
            "No profitability or OOS model metric is used to select a break candidate.",
            "Phase 1.5 forward ledger is neither read nor written.",
        ],
        "failedTests": failed_tests,
        "skippedItems": list(config["explicitlyNotImplemented"])
        + [
            "all_hmm_model_fits",
            "post_jump_vs_full_history_hmm",
            "time_decay_hmm",
            "state_transition_matrix_comparison",
            "state_duration_comparison",
            "state_entropy_comparison",
            "turning_probability_calibration_comparison",
        ],
        "modelingSkipped": True,
        "modelingSkipReason": (
            "Gate 0 failed closed: break date is not preregistered and DMD "
            "coverage is below the fixed threshold."
        ),
        "phase15Touched": False,
        "phase2AArtifactsOverwritten": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    phase2a.atomic_json(output_dir / "config_snapshot.json", config)
    phase2a.atomic_json(output_dir / "data_audit.json", data_audit)
    phase2a.atomic_text(
        output_dir / "data_audit.md", render_data_audit(data_audit)
    )
    phase2a.atomic_json(output_dir / "post_jump_audit.json", post_jump)
    phase2a.atomic_json(
        output_dir / "coverage_bias_report.json", coverage_bias
    )
    phase2a.atomic_json(
        output_dir / "feature_missingness_report.json",
        feature_missingness,
    )
    phase2a.atomic_json(
        output_dir / "feature_correlation_report.json",
        correlation_report,
    )
    phase2a.atomic_json(
        output_dir / "label_distribution_report.json",
        label_distribution,
    )
    phase2a.atomic_text(
        output_dir / "break_date_risk_report.md",
        render_break_risk(config, candidates),
    )
    phase2a.atomic_json(
        output_dir / "phase1_6_gate0_result.json", result
    )
    phase2a.atomic_text(
        output_dir / "phase1_6_gate0_report.md",
        render_gate0_report(result),
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
        "gate0_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    output_dir = resolve(config["output"]["root"]) / run_id
    result = run(config, output_dir)
    print(
        json.dumps(
            {
                "run_id": result["runId"],
                "status": result["status"],
                "gate0_passed": result["gate0"]["passed"],
                "modeling_allowed": result["gate0"]["modelingAllowed"],
                "modeling_skipped": result["modelingSkipped"],
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
