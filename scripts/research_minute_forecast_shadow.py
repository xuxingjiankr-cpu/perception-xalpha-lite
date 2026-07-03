"""Frozen five-minute ETF probability and return forecast shadow replay.

The model uses only completed bars at the forecast timestamp, assumes entry on
the next observed bar and evaluates fixed 5/15/30-minute holding horizons after
12 bps round-trip cost. L2 and IOPV are audited for forward readiness but are
excluded until enough complete forward days exist.

This script is offline, record-only research and cannot submit orders or modify
live configuration, overlays, execution locks or position sizing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research_hmm_nn_bl import (
    ROOT,
    _within_day_ratio,
    _within_day_volatility,
    load_panel,
    select_training_universe,
)
from run_t0_strategy_evolution import (
    deflated_sharpe_diagnostic,
    diebold_mariano_hln,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "minute_forecast_shadow_v1.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "minute_forecast_shadow"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def build_feature_frames(panel: pd.DataFrame) -> dict[str, Any]:
    prices = panel.pivot(
        index="timestamp", columns="stockCode", values="close"
    ).sort_index()
    amounts = panel.pivot(
        index="timestamp", columns="stockCode", values="cumulative_amount"
    ).reindex(prices.index)
    ret_1 = _within_day_ratio(prices, 1)
    ret_3 = _within_day_ratio(prices, 3)
    ret_6 = _within_day_ratio(prices, 6)
    dates = pd.Series(prices.index.date, index=prices.index)
    acceleration = ret_1.groupby(dates).transform(
        lambda frame: frame - frame.shift(1)
    )
    vol_6 = _within_day_volatility(ret_1, 6)
    market_ret_1 = ret_1.median(axis=1, skipna=True).fillna(0.0)
    market_ret_3 = ret_3.median(axis=1, skipna=True).fillna(0.0)
    breadth = (ret_1 > 0).sum(axis=1) / ret_1.notna().sum(axis=1).clip(
        lower=1
    )
    relative_strength = ret_6.rank(axis=1, pct=True, method="average")
    amount_rank = amounts.rank(axis=1, pct=True, method="average")
    session_minutes = prices.index.hour * 60 + prices.index.minute
    session_fraction = pd.Series(
        np.clip((session_minutes - 570) / 330.0, 0.0, 1.0),
        index=prices.index,
    )
    return {
        "prices": prices,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_6": ret_6,
        "acceleration_1": acceleration,
        "vol_6": vol_6,
        "market_ret_1": market_ret_1,
        "market_ret_3": market_ret_3,
        "breadth": breadth,
        "relative_strength": relative_strength,
        "amount_rank": amount_rank,
        "session_fraction": session_fraction,
    }


def build_samples(
    frames: dict[str, Any],
    features: list[str],
    horizon_bars: int,
) -> pd.DataFrame:
    """Build causal features and next-bar entry labels for every completed bar."""
    prices: pd.DataFrame = frames["prices"]
    records: list[dict[str, Any]] = []
    for trade_date, day_prices in prices.groupby(
        prices.index.strftime("%Y-%m-%d")
    ):
        timestamps = list(day_prices.index)
        for decision_index, timestamp in enumerate(timestamps):
            entry_index = decision_index + 1
            exit_index = entry_index + horizon_bars
            if exit_index >= len(timestamps):
                continue
            entry_time = timestamps[entry_index]
            exit_time = timestamps[exit_index]
            if (entry_time - timestamp).total_seconds() > 10 * 60:
                continue
            if (
                exit_time - entry_time
            ).total_seconds() > (horizon_bars + 1) * 5 * 60:
                continue
            for code in prices.columns:
                entry_price = prices.at[entry_time, code]
                exit_price = prices.at[exit_time, code]
                if (
                    not np.isfinite(entry_price)
                    or not np.isfinite(exit_price)
                    or float(entry_price) <= 0
                ):
                    continue
                row: dict[str, Any] = {
                    "trade_date": trade_date,
                    "timestamp": timestamp,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "stockCode": code,
                    "gross_forward_return": float(
                        exit_price / entry_price - 1.0
                    ),
                }
                valid = True
                for feature in features:
                    source = frames[feature]
                    value = (
                        source.at[timestamp, code]
                        if isinstance(source, pd.DataFrame)
                        else source.at[timestamp]
                    )
                    if not np.isfinite(value):
                        valid = False
                        break
                    row[feature] = float(value)
                if valid:
                    records.append(row)
    return pd.DataFrame.from_records(records)


def make_classifier(config: dict[str, Any]) -> Pipeline:
    model_cfg = config["models"]
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(model_cfg["logisticC"]),
                    max_iter=500,
                    random_state=int(model_cfg["randomSeed"]),
                ),
            ),
        ]
    )


def make_regressor(config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(config["models"]["ridgeAlpha"]))),
        ]
    )


def fit_temporal_calibrated_classifier(
    train: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    config: dict[str, Any],
) -> tuple[Pipeline, LogisticRegression, dict[str, Any]]:
    """Fit the base classifier before a disjoint trailing calibration window."""
    dates = sorted(train["trade_date"].unique())
    calibration_days = int(config["models"]["calibrationDays"])
    if len(dates) <= calibration_days + 5:
        raise RuntimeError(
            f"insufficient train dates for temporal calibration: {len(dates)}"
        )
    calibration_dates = set(dates[-calibration_days:])
    fit_mask = ~train["trade_date"].isin(calibration_dates).to_numpy()
    calibration_mask = ~fit_mask
    base = make_classifier(config)
    base.fit(
        train.loc[fit_mask, features].to_numpy(dtype=float),
        labels[fit_mask],
    )
    raw_probability = np.clip(
        base.predict_proba(
            train.loc[calibration_mask, features].to_numpy(dtype=float)
        )[:, 1],
        1e-6,
        1.0 - 1e-6,
    )
    raw_logit = np.log(raw_probability / (1.0 - raw_probability)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=float(config["models"]["calibrationC"]),
        max_iter=500,
        random_state=int(config["models"]["randomSeed"]),
    )
    calibrator.fit(raw_logit, labels[calibration_mask])
    audit = {
        "baseFitStart": dates[0],
        "baseFitEnd": dates[-calibration_days - 1],
        "baseFitDays": len(dates) - calibration_days,
        "calibrationStart": dates[-calibration_days],
        "calibrationEnd": dates[-1],
        "calibrationDays": calibration_days,
        "baseFitSamples": int(np.sum(fit_mask)),
        "calibrationSamples": int(np.sum(calibration_mask)),
    }
    return base, calibrator, audit


def calibrated_probability(
    base: Pipeline,
    calibrator: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    raw_probability = np.clip(
        base.predict_proba(features)[:, 1], 1e-6, 1.0 - 1e-6
    )
    raw_logit = np.log(raw_probability / (1.0 - raw_probability)).reshape(-1, 1)
    return calibrator.predict_proba(raw_logit)[:, 1]


def probability_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    training_prior: float,
) -> dict[str, Any]:
    labels = np.asarray(actual, dtype=int)
    forecast = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    prior = np.full(len(labels), np.clip(training_prior, 1e-6, 1 - 1e-6))
    auc = (
        float(roc_auc_score(labels, forecast))
        if len(np.unique(labels)) == 2
        else None
    )
    return {
        "samples": int(len(labels)),
        "actualSuccessRate": float(np.mean(labels)),
        "trainingPrior": float(training_prior),
        "brier": float(brier_score_loss(labels, forecast)),
        "priorBrier": float(brier_score_loss(labels, prior)),
        "logLoss": float(log_loss(labels, forecast, labels=[0, 1])),
        "priorLogLoss": float(log_loss(labels, prior, labels=[0, 1])),
        "auc": auc,
    }


def daily_auc_metrics(
    trade_dates: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    rows: list[float] = []
    dates = np.asarray(trade_dates)
    labels = np.asarray(actual, dtype=int)
    forecast = np.asarray(probability, dtype=float)
    for trade_date in sorted(set(dates)):
        mask = dates == trade_date
        if len(np.unique(labels[mask])) != 2:
            continue
        rows.append(float(roc_auc_score(labels[mask], forecast[mask])))
    return {
        "days": len(rows),
        "mean": float(np.mean(rows)) if rows else None,
        "median": float(np.median(rows)) if rows else None,
        "minimum": float(np.min(rows)) if rows else None,
        "maximum": float(np.max(rows)) if rows else None,
    }


def calibration_bins(
    actual: np.ndarray,
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(actual, dtype=int)
    forecast = np.asarray(probability, dtype=float)
    bins: list[dict[str, Any]] = []
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        mask = (forecast >= lower) & (
            forecast <= upper if upper >= 1.0 else forecast < upper
        )
        if not np.any(mask):
            continue
        bins.append(
            {
                "lower": round(float(lower), 1),
                "upper": round(float(upper), 1),
                "count": int(np.sum(mask)),
                "meanForecast": float(np.mean(forecast[mask])),
                "actualRate": float(np.mean(labels[mask])),
            }
        )
    return bins


def evaluate_policy(
    samples: pd.DataFrame,
    *,
    horizon_bars: int,
    cost: float,
    probability_threshold: float,
    expected_net_threshold: float,
    downside_net_threshold: float,
    max_assets: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    selected_rows: list[dict[str, Any]] = []
    daily: dict[str, float] = defaultdict(float)
    tranche_exposure = min(1.0, 1.0 / max(1, horizon_bars))
    for timestamp, group in samples.groupby("timestamp", sort=True):
        eligible = group[
            (group["probability"] >= probability_threshold)
            & (group["expected_net_return"] >= expected_net_threshold)
            & (group["downside_net_quantile"] >= downside_net_threshold)
        ].copy()
        if eligible.empty:
            continue
        eligible = eligible.sort_values(
            ["probability", "expected_net_return", "stockCode"],
            ascending=[False, False, True],
        ).head(max_assets)
        weight = tranche_exposure / len(eligible)
        interval_return = 0.0
        for _, row in eligible.iterrows():
            net_return = float(row["gross_forward_return"]) - cost
            interval_return += weight * net_return
            selected_rows.append(
                {
                    "horizon_bars": horizon_bars,
                    "trade_date": row["trade_date"],
                    "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "entry_time": pd.Timestamp(row["entry_time"]).isoformat(),
                    "exit_time": pd.Timestamp(row["exit_time"]).isoformat(),
                    "stockCode": row["stockCode"],
                    "weight": weight,
                    "probability": float(row["probability"]),
                    "expected_net_return": float(
                        row["expected_net_return"]
                    ),
                    "downside_net_quantile": float(
                        row["downside_net_quantile"]
                    ),
                    "actual_net_return": net_return,
                }
            )
        daily[str(eligible["trade_date"].iloc[0])] += interval_return
    all_dates = sorted(samples["trade_date"].unique())
    daily_complete = {day: float(daily.get(day, 0.0)) for day in all_dates}
    daily_values = np.asarray(list(daily_complete.values()), dtype=float)
    trade_returns = np.asarray(
        [row["actual_net_return"] for row in selected_rows], dtype=float
    )
    equity = np.cumprod(1.0 + daily_values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    daily_std = (
        float(np.std(daily_values, ddof=1)) if len(daily_values) > 1 else 0.0
    )
    metrics = {
        "days": len(all_dates),
        "selectedTrades": len(selected_rows),
        "activeDays": int(np.sum(daily_values != 0)),
        "averageNetTrade": (
            float(np.mean(trade_returns)) if len(trade_returns) else None
        ),
        "medianNetTrade": (
            float(np.median(trade_returns)) if len(trade_returns) else None
        ),
        "winTradeRate": (
            float(np.mean(trade_returns > 0)) if len(trade_returns) else None
        ),
        "tradeNetQ05": (
            float(np.quantile(trade_returns, 0.05))
            if len(trade_returns)
            else None
        ),
        "netPortfolioReturn": float(equity[-1] - 1.0),
        "dailyStd": daily_std,
        "dailySharpe": (
            float(np.mean(daily_values) / daily_std * math.sqrt(252))
            if daily_std > 0
            else None
        ),
        "worstDay": float(np.min(daily_values)),
        "maxDrawdown": float(np.min(drawdown)),
        "trancheExposure": tranche_exposure,
    }
    return metrics, selected_rows, daily_complete


def evaluate_momentum_control(
    samples: pd.DataFrame,
    *,
    horizon_bars: int,
    cost: float,
    max_assets: int,
) -> dict[str, float]:
    daily: dict[str, float] = defaultdict(float)
    tranche = min(1.0, 1.0 / max(1, horizon_bars))
    for _, group in samples.groupby("timestamp", sort=True):
        eligible = group[group["ret_6"] > 0].sort_values(
            ["ret_6", "stockCode"], ascending=[False, True]
        ).head(max_assets)
        if eligible.empty:
            continue
        weight = tranche / len(eligible)
        daily[str(eligible["trade_date"].iloc[0])] += float(
            weight
            * np.sum(
                eligible["gross_forward_return"].to_numpy(dtype=float) - cost
            )
        )
    dates = sorted(samples["trade_date"].unique())
    return {day: float(daily.get(day, 0.0)) for day in dates}


def daily_performance(daily: dict[str, float]) -> dict[str, Any]:
    values = np.asarray([daily[day] for day in sorted(daily)], dtype=float)
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "days": len(values),
        "netPortfolioReturn": float(equity[-1] - 1.0),
        "dailyStd": std,
        "dailySharpe": (
            float(np.mean(values) / std * math.sqrt(252)) if std > 0 else None
        ),
        "worstDay": float(np.min(values)),
        "maxDrawdown": float(np.min(equity / peaks - 1.0)),
    }


def forward_feature_readiness(config: dict[str, Any]) -> dict[str, Any]:
    readiness = config["forwardFeatureReadiness"]

    def dates_for(pattern: str, prefix: str) -> list[str]:
        dates: list[str] = []
        for raw_path in glob.glob(str(ROOT / pattern)):
            path = Path(raw_path)
            if path.stat().st_size < 1_000_000:
                continue
            stem = path.stem
            if stem.startswith(prefix):
                dates.append(stem[len(prefix) :])
        return sorted(set(dates))

    l2_dates = dates_for(readiness["l2Pattern"], "depth_")
    iopv_dates = dates_for(readiness["iopvPattern"], "iopv_")
    complete = sorted(set(l2_dates) & set(iopv_dates))
    minimum = int(readiness["minimumCompleteDaysBeforeModelUse"])
    return {
        "l2Days": l2_dates,
        "iopvDays": iopv_dates,
        "completeJointDays": complete,
        "completeDayCount": len(complete),
        "minimumRequired": minimum,
        "readyForModelFitting": len(complete) >= minimum,
        "includedInCurrentModel": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Five-Minute ETF Forecast Shadow Replay",
        "",
        "Status: `diagnostic_only / frozen model / reused OOS / no live change`",
        "",
        f"- Train: {report['split']['trainStart']} to {report['split']['trainEnd']}.",
        f"- Test: {report['split']['oosStart']} to {report['split']['oosEnd']}.",
        f"- Frozen training-only universe: {report['universe']['count']} ETFs.",
        "- Every completed five-minute bar is scored; entry is the next observed "
        "bar and round-trip cost is 12 bps.",
        "",
        "| Horizon | OOS samples | AUC | Brier / prior | Selected | Avg net trade | "
        "Policy return | Sharpe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, result in report["horizons"].items():
        forecast = result["forecast"]
        policy = result["policy"]
        lines.append(
            f"| {int(horizon) * 5}m | {forecast['samples']} | "
            f"{forecast['auc'] if forecast['auc'] is not None else 'n/a'} | "
            f"{forecast['brier']:.4f} / {forecast['priorBrier']:.4f} | "
            f"{policy['selectedTrades']} | "
            f"{policy['averageNetTrade'] if policy['averageNetTrade'] is not None else 'n/a'} | "
            f"{policy['netPortfolioReturn']:.2%} | "
            f"{policy['dailySharpe'] if policy['dailySharpe'] is not None else 'n/a'} |"
        )
    readiness = report["forwardFeatureReadiness"]
    lines.extend(
        [
            "",
            "## Forward feature readiness",
            "",
            f"- Complete joint L2 + IOPV days: {readiness['completeDayCount']} / "
            f"{readiness['minimumRequired']}.",
            "- L2 and IOPV are excluded from model fitting until the fixed day "
            "minimum is reached.",
            "",
            "## Verdict",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdictReason"],
            "",
            "A model must beat the training-prior probability forecast and produce "
            "positive cost-adjusted returns. Losing less than momentum or holding "
            "cash is not an edge.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "This result cannot alter live entries, exits, sizing, overlays, orders "
            "or execution locks.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    quotes = ROOT / data_cfg["quotes"]
    selected, universe_audit = select_training_universe(
        quotes,
        str(data_cfg["trainEnd"]),
        int(data_cfg["universeSize"]),
    )
    panel = load_panel(quotes, selected)
    frames = build_feature_frames(panel)
    features = list(config["features"])
    cost = float(config["targets"]["roundTripCostBps"]) / 10_000.0
    output_dir = Path(args.output_dir)
    horizons: dict[str, Any] = {}
    all_selected: list[dict[str, Any]] = []
    all_daily_rows: list[dict[str, Any]] = []
    all_gates: dict[str, Any] = {}
    for horizon in config["targets"]["horizonBars"]:
        horizon = int(horizon)
        samples = build_samples(frames, features, horizon)
        train = samples[
            (samples["trade_date"] >= data_cfg["trainStart"])
            & (samples["trade_date"] <= data_cfg["trainEnd"])
        ].copy()
        oos = samples[
            (samples["trade_date"] >= data_cfg["oosStart"])
            & (samples["trade_date"] <= data_cfg["oosEnd"])
        ].copy()
        if train.empty or oos.empty:
            raise RuntimeError(
                f"empty samples for horizon {horizon}: "
                f"train={len(train)}, oos={len(oos)}"
            )
        train_labels = (
            train["gross_forward_return"].to_numpy(dtype=float) > cost
        ).astype(int)
        oos_labels = (
            oos["gross_forward_return"].to_numpy(dtype=float) > cost
        ).astype(int)
        classifier, calibrator, calibration_audit = (
            fit_temporal_calibrated_classifier(
                train, features, train_labels, config
            )
        )
        regressor = make_regressor(config)
        regressor.fit(
            train[features].to_numpy(dtype=float),
            train["gross_forward_return"].to_numpy(dtype=float) * 10_000.0,
        )
        train_expected = (
            regressor.predict(train[features].to_numpy(dtype=float)) / 10_000.0
        )
        oos_probability = calibrated_probability(
            classifier,
            calibrator,
            oos[features].to_numpy(dtype=float),
        )
        oos_expected = (
            regressor.predict(oos[features].to_numpy(dtype=float)) / 10_000.0
        )
        residual_q = float(
            np.quantile(
                train["gross_forward_return"].to_numpy(dtype=float)
                - train_expected,
                float(config["models"]["downsideQuantile"]),
            )
        )
        oos["probability"] = oos_probability
        oos["expected_net_return"] = oos_expected - cost
        oos["downside_net_quantile"] = oos_expected + residual_q - cost
        forecast_metrics = probability_metrics(
            oos_labels, oos_probability, float(np.mean(train_labels))
        )
        forecast_metrics["dailyAuc"] = daily_auc_metrics(
            oos["trade_date"].to_numpy(), oos_labels, oos_probability
        )
        forecast_metrics["expectedReturnSpearman"] = float(
            pd.Series(oos_expected).corr(
                pd.Series(oos["gross_forward_return"].to_numpy(dtype=float)),
                method="spearman",
            )
        )
        policy_cfg = config["shadowPolicy"]
        policy_metrics, selected_rows, policy_daily = evaluate_policy(
            oos,
            horizon_bars=horizon,
            cost=cost,
            probability_threshold=float(
                policy_cfg["minimumProbability"]
            ),
            expected_net_threshold=float(
                policy_cfg["minimumExpectedNetReturn"]
            ),
            downside_net_threshold=float(
                policy_cfg["minimumDownsideNetQuantile"]
            ),
            max_assets=int(policy_cfg["maximumAssetsPerTimestamp"]),
        )
        momentum_daily = evaluate_momentum_control(
            oos,
            horizon_bars=horizon,
            cost=cost,
            max_assets=int(policy_cfg["maximumAssetsPerTimestamp"]),
        )
        momentum_metrics = daily_performance(momentum_daily)
        dm = diebold_mariano_hln(
            momentum_daily, policy_daily, alpha=0.05
        )
        cash = {day: 0.0 for day in policy_daily}
        dsr = deflated_sharpe_diagnostic(
            cash, policy_daily, n_trials=3, alpha=0.10
        )
        gates_cfg = config["evidenceGates"]
        gates = {
            "minimumOosDays": policy_metrics["days"]
            >= int(gates_cfg["minimumOosDays"]),
            "minimumSelectedTrades": policy_metrics["selectedTrades"]
            >= int(gates_cfg["minimumSelectedTrades"]),
            "brierBeatsTrainingPrior": forecast_metrics["brier"]
            < forecast_metrics["priorBrier"]
            * float(gates_cfg["maximumBrierVsTrainingPrior"]),
            "logLossBeatsTrainingPrior": forecast_metrics["logLoss"]
            < forecast_metrics["priorLogLoss"]
            * float(gates_cfg["maximumLogLossVsTrainingPrior"]),
            "minimumAuc": forecast_metrics["auc"] is not None
            and forecast_metrics["auc"] >= float(gates_cfg["minimumAuc"]),
            "positiveAverageNetTrade": policy_metrics["averageNetTrade"]
            is not None
            and policy_metrics["averageNetTrade"] > 0,
            "positiveNetPortfolioReturn": policy_metrics[
                "netPortfolioReturn"
            ]
            > 0,
            "positiveDailySharpe": (
                policy_metrics["dailySharpe"] is not None
                and policy_metrics["dailySharpe"] > 0
            ),
            "freshUnseenForwardWindow": not bool(
                data_cfg["oosWindowPreviouslyReused"]
            ),
        }
        horizons[str(horizon)] = {
            "minutes": horizon * 5,
            "trainSamples": len(train),
            "forecast": forecast_metrics,
            "calibrationAudit": calibration_audit,
            "calibration": calibration_bins(oos_labels, oos_probability),
            "trainingResidualQ05": residual_q,
            "policy": policy_metrics,
            "momentumControl": momentum_metrics,
            "evidence": {
                "gates": gates,
                "dmVsMomentum": dm,
                "dsrVsCash": dsr,
            },
        }
        all_gates[str(horizon)] = gates
        all_selected.extend(selected_rows)
        for day in sorted(policy_daily):
            all_daily_rows.append(
                {
                    "horizon_bars": horizon,
                    "trade_date": day,
                    "policy_net_return": policy_daily[day],
                    "momentum_net_return": momentum_daily[day],
                }
            )
    readiness = forward_feature_readiness(config)
    historical_passes = [
        horizon
        for horizon, gates in all_gates.items()
        if all(
            value
            for key, value in gates.items()
            if key != "freshUnseenForwardWindow"
        )
    ]
    verdict = (
        "historical_candidate_forward_shadow_only"
        if historical_passes
        else "no_validated_minute_forecast_edge"
    )
    report = {
        "schemaVersion": "minute_forecast_shadow_result_v1",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "config": str(config_path),
        "split": {
            "trainStart": data_cfg["trainStart"],
            "trainEnd": data_cfg["trainEnd"],
            "oosStart": data_cfg["oosStart"],
            "oosEnd": data_cfg["oosEnd"],
            "oosWindowPreviouslyReused": bool(
                data_cfg["oosWindowPreviouslyReused"]
            ),
        },
        "universe": {
            "count": len(selected),
            "selectionUsesOos": universe_audit["selectionUsesOos"],
            "codes": selected,
        },
        "features": features,
        "horizons": horizons,
        "forwardFeatureReadiness": readiness,
        "historicalPassingHorizons": historical_passes,
        "verdict": verdict,
        "verdictReason": (
            "At least one fixed horizon passed the historical numerical gates, "
            "but the reused OOS window prohibits deployment; new forward data is required."
            if historical_passes
            else "No fixed horizon simultaneously beat the probability prior and "
            "produced positive cost-adjusted shadow returns."
        ),
        "limitations": config["knownLimitations"],
        "liveChanges": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "minute_forecast_shadow_result.json", report)
    write_csv(
        output_dir / "minute_forecast_shadow_selected.csv", all_selected
    )
    write_csv(output_dir / "minute_forecast_shadow_daily.csv", all_daily_rows)
    (output_dir / "minute_forecast_shadow_report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": verdict,
                "universe": len(selected),
                "historicalPassingHorizons": historical_passes,
                "forwardFeatureReadiness": readiness,
                "horizons": {
                    key: {
                        "forecast": value["forecast"],
                        "policy": value["policy"],
                        "gates": value["evidence"]["gates"],
                    }
                    for key, value in horizons.items()
                },
                "output": str(
                    output_dir / "minute_forecast_shadow_report.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
