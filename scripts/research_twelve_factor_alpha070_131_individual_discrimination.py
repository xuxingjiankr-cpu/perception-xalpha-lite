"""Regularized multivariate individual-stock discrimination for the 13-factor book.

This is a historical, research-only rejection test.  It cannot trade or promote itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_twelve_factor_utility_weights_v6 as v6  # noqa: E402
import research_twelve_factor_alpha070_131_ablation as ablation  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "twelve_factor_alpha070_131_individual_discrimination_v4.json"
)
SCHEMA_VERSION = "twelve_factor_alpha070_131_individual_discrimination_result_v4"
CODE_VERSION = "twelve_factor_alpha070_131_individual_discrimination_v4_20260809"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "twelve_factor_alpha070_131_individual_discrimination_v4":
        raise ValueError("unexpected individual discrimination schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("study must remain research/shadow-only")
    if file_sha256(ROOT / config["frozenAblationConfig"]) != config["frozenAblationConfigSha256"]:
        raise ValueError("frozen ablation config changed")
    if file_sha256(ROOT / config["frozenAblationResult"]) != config["frozenAblationResultSha256"]:
        raise ValueError("frozen ablation result changed")
    if config["features"].get("count") != 13 or not config["features"].get("pastOnly"):
        raise ValueError("exactly thirteen past-only ranks are required")
    if config["training"].get("hyperparameterSearchAllowed"):
        raise ValueError("hyperparameter search is forbidden")
    if config["ranking"].get("temperatureScalingForbidden") is not True:
        raise ValueError("mechanical probability stretching is forbidden")
    if config["acceptance"].get("allGatesMustPass") is not True:
        raise ValueError("every preregistered gate must pass")
    if config["acceptance"].get("historicalRunCanPromote") is not False:
        raise ValueError("historical run cannot promote")
    safety = config["safety"]
    if any(bool(value) for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading and mutation permissions must remain false")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must remain empty")


def date_range(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))])


def fixed_partitions(index: pd.DatetimeIndex, config: dict[str, Any]) -> dict[str, pd.DatetimeIndex]:
    isolation = config["timeIsolation"]
    output = {
        "baseFit": date_range(index, *isolation["baseFit"]),
        "calibration": date_range(index, *isolation["calibration"]),
        "audit": date_range(index, *isolation["audit"]),
        "validation": date_range(index, *isolation["validation"]),
        "shadow": pd.DatetimeIndex(index[index >= pd.Timestamp(isolation["shadow"][0])]),
    }
    order = ["baseFit", "calibration", "audit", "validation", "shadow"]
    if any(output[name].empty for name in order):
        raise RuntimeError("one or more frozen partitions are empty")
    if not all(output[left].max() < output[right].min() for left, right in zip(order, order[1:])):
        raise RuntimeError("frozen partitions overlap or are out of order")
    return output


def deterministic_indices(count: int, limit: int, seed: int, date: pd.Timestamp) -> np.ndarray:
    if limit <= 0 or count <= limit:
        return np.arange(count)
    date_seed = int(date.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed ^ date_seed)
    return np.sort(rng.choice(count, size=limit, replace=False))


def stack_features(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    maximum_rows_per_day: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    keys = list(ranks)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used_days = 0
    for date in dates:
        y = returns.loc[date].to_numpy(float)
        x = np.column_stack([ranks[key].loc[date].to_numpy(float) for key in keys])
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        xv, yv = x[valid], y[valid]
        if not len(yv):
            continue
        chosen = deterministic_indices(len(yv), maximum_rows_per_day, seed, date)
        xv, yv = xv[chosen], yv[chosen]
        xs.append(xv)
        ys.append(yv)
        weights.append(np.full(len(yv), 1.0 / len(yv), dtype=float))
        used_days += 1
    if not xs:
        raise RuntimeError("no complete feature rows")
    return np.vstack(xs), np.concatenate(ys), np.concatenate(weights), used_days


@dataclass
class Models:
    return_base: Ridge
    return_calibrator: Ridge
    classifiers: dict[str, LogisticRegression]
    probability_calibrators: dict[str, LogisticRegression]


def classifier(config: dict[str, Any], calibration: bool = False) -> LogisticRegression:
    params = (
        config["training"]["probabilityCalibration"]
        if calibration
        else config["training"]["probabilityEstimator"]
    )
    return LogisticRegression(
        C=float(params["C"]),
        solver="lbfgs",
        max_iter=(1000 if calibration else int(params["maxIter"])),
        class_weight=None,
        random_state=int(config["training"]["deterministicSeed"]),
    )


def targets(y: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "grossUp": y > 0.0,
        "netPositive": y > float(config["outcomes"]["roundTripCost"]),
        "severeLoss": y <= -0.03,
    }


def fit_models(
    ranks: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    partitions: dict[str, pd.DatetimeIndex],
    config: dict[str, Any],
) -> tuple[Models, dict[str, Any]]:
    training = config["training"]
    seed = int(training["deterministicSeed"])
    x_base, y_base, w_base, base_days = stack_features(
        ranks,
        returns,
        partitions["baseFit"],
        int(training["maximumBaseFitRowsPerDay"]),
        seed,
    )
    return_base = Ridge(alpha=float(training["returnEstimator"]["alpha"]))
    return_base.fit(x_base, y_base, sample_weight=w_base)
    classifiers: dict[str, LogisticRegression] = {}
    for name, target in targets(y_base, config).items():
        model = classifier(config)
        model.fit(x_base, target.astype(int), sample_weight=w_base)
        classifiers[name] = model

    x_cal, y_cal, w_cal, cal_days = stack_features(
        ranks, returns, partitions["calibration"], 0, seed
    )
    raw_return = return_base.predict(x_cal).reshape(-1, 1)
    return_calibrator = Ridge(alpha=float(training["returnCalibration"]["alpha"]))
    return_calibrator.fit(raw_return, y_cal, sample_weight=w_cal)
    probability_calibrators: dict[str, LogisticRegression] = {}
    cal_targets = targets(y_cal, config)
    for name, model in classifiers.items():
        raw_logit = model.decision_function(x_cal).reshape(-1, 1)
        calibrator = classifier(config, calibration=True)
        calibrator.fit(raw_logit, cal_targets[name].astype(int), sample_weight=w_cal)
        probability_calibrators[name] = calibrator
    audit = {
        "baseRows": len(y_base),
        "baseDays": base_days,
        "calibrationRows": len(y_cal),
        "calibrationDays": cal_days,
        "returnCoefficients": return_base.coef_.tolist(),
        "classifierCoefficients": {
            name: model.coef_[0].tolist() for name, model in classifiers.items()
        },
    }
    return Models(return_base, return_calibrator, classifiers, probability_calibrators), audit


def empty_frame(index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)


def predict_multivariate(
    ranks: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    models: Models,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    keys = list(ranks)
    columns = next(iter(ranks.values())).columns
    output = {
        name: empty_frame(dates, columns)
        for name in ("expectedReturn", "grossUp", "netPositive", "severeLoss", "utility")
    }
    for date in dates:
        x = np.column_stack([ranks[key].loc[date].to_numpy(float) for key in keys])
        valid = np.isfinite(x).all(axis=1)
        if not valid.any():
            continue
        xv = x[valid]
        raw_return = models.return_base.predict(xv).reshape(-1, 1)
        expected = models.return_calibrator.predict(raw_return)
        probabilities: dict[str, np.ndarray] = {}
        for name, model in models.classifiers.items():
            raw_logit = model.decision_function(xv).reshape(-1, 1)
            probabilities[name] = models.probability_calibrators[name].predict_proba(raw_logit)[:, 1]
        utility = expected - 0.01 * (1.0 - probabilities["netPositive"]) - 0.02 * probabilities["severeLoss"]
        for name, values in {
            "expectedReturn": expected,
            "grossUp": probabilities["grossUp"],
            "netPositive": probabilities["netPositive"],
            "severeLoss": probabilities["severeLoss"],
            "utility": utility,
        }.items():
            output[name].loc[date, valid] = values
    return output


def scalar_predictions(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    calibration_dates: pd.DatetimeIndex,
    prediction_dates: pd.DatetimeIndex,
    v6_config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    calibrators = v6.fit_calibrators(score, returns, calibration_dates, v6_config)
    predicted = v6.predict_frames(score.reindex(index=prediction_dates), calibrators, v6_config)
    rows = v6.stack_score_outcome(score, returns, calibration_dates)
    up = IsotonicRegression(increasing=True, out_of_bounds="clip")
    up.fit(
        rows["score"].to_numpy(float),
        rows["return"].gt(0.0).astype(float),
        sample_weight=rows["sampleWeight"].to_numpy(float),
    )
    values = score.reindex(index=prediction_dates).to_numpy(float)
    finite = np.isfinite(values)
    gross_up = np.full_like(values, np.nan, dtype=float)
    gross_up[finite] = up.predict(values[finite])
    return {
        "expectedReturn": predicted["expectedReturn"],
        "grossUp": pd.DataFrame(gross_up, index=prediction_dates, columns=score.columns),
        "netPositive": 1.0 - predicted["netLossProbability"],
        "severeLoss": predicted["severeLossProbability"],
        "utility": predicted["utility"],
    }


def period_rows(
    predictions: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts = {
        name: frame.reindex(index=dates).stack(future_stack=True)
        for name, frame in predictions.items()
    }
    parts["return"] = returns.reindex(index=dates).stack(future_stack=True)
    return pd.concat(parts, axis=1).dropna().reset_index(names=["date", "securityId"])


def decile_monotonicity(rows: pd.DataFrame) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    for _date, group in rows.groupby("date", sort=True):
        if len(group) < 100:
            continue
        group = group.copy()
        group["decile"] = pd.qcut(group["utility"].rank(method="first"), 10, labels=False) + 1
        values.extend(
            {"decile": float(decile), "return": float(part["return"].mean())}
            for decile, part in group.groupby("decile")
        )
    frame = pd.DataFrame(values)
    means = frame.groupby("decile")["return"].mean() if not frame.empty else pd.Series(dtype=float)
    correlation = spearmanr(means.index, means.values).statistic if len(means) >= 3 else np.nan
    adjacent = np.diff(means.values) if len(means) >= 2 else np.asarray([])
    return {
        "decileMeanReturns": {str(int(key)): round(float(value), 8) for key, value in means.items()},
        "spearman": round(float(correlation), 8) if np.isfinite(correlation) else None,
        "adjacentIncreasingFraction": round(float((adjacent > 0).mean()), 8) if len(adjacent) else None,
    }


def probability_spread(predictions: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> dict[str, Any]:
    top = v6.top_mask(predictions["utility"].reindex(index=dates), 10)
    spreads = (predictions["grossUp"].reindex(index=dates).where(top).max(axis=1) - predictions["grossUp"].reindex(index=dates).where(top).min(axis=1)).dropna()
    return {
        "days": len(spreads),
        "meanTop10GrossUpSpread": round(float(spreads.mean()), 8) if len(spreads) else None,
        "medianTop10GrossUpSpread": round(float(spreads.median()), 8) if len(spreads) else None,
    }


def evaluate_period(
    predictions: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = period_rows(predictions, returns, dates)
    actual = rows["return"].to_numpy(float)
    cost = float(config["outcomes"]["roundTripCost"])
    severe = -0.03
    output = {
        "rows": len(rows),
        "days": int(rows["date"].nunique()),
        "expectedReturnMse": round(float(mean_squared_error(actual, rows["expectedReturn"])), 10),
        "probability": {
            "grossUp": v6.probability_metrics((actual > 0.0).astype(float), rows["grossUp"].to_numpy(float)),
            "netPositive": v6.probability_metrics((actual > cost).astype(float), rows["netPositive"].to_numpy(float)),
            "severeLoss": v6.probability_metrics((actual <= severe).astype(float), rows["severeLoss"].to_numpy(float)),
        },
        "top10": v6.outcome_metrics(returns, predictions["utility"], dates, 10, cost, severe),
        "spread": probability_spread(predictions, dates),
        "monotonicity": decile_monotonicity(rows),
    }
    return output


def acceptance(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "grossUpAucImproved": candidate["probability"]["grossUp"]["auc"] > baseline["probability"]["grossUp"]["auc"],
        "grossUpBrierImproved": candidate["probability"]["grossUp"]["brier"] < baseline["probability"]["grossUp"]["brier"],
        "grossUpLogLossImproved": candidate["probability"]["grossUp"]["logLoss"] < baseline["probability"]["grossUp"]["logLoss"],
        "netPositiveAucImproved": candidate["probability"]["netPositive"]["auc"] > baseline["probability"]["netPositive"]["auc"],
        "netPositiveBrierImproved": candidate["probability"]["netPositive"]["brier"] < baseline["probability"]["netPositive"]["brier"],
        "netPositiveLogLossImproved": candidate["probability"]["netPositive"]["logLoss"] < baseline["probability"]["netPositive"]["logLoss"],
        "top10MeanNetReturnImproved": candidate["top10"]["meanNetReturn"] > baseline["top10"]["meanNetReturn"],
        "top10NetWinRateImproved": candidate["top10"]["netWinRate"] > baseline["top10"]["netWinRate"],
        "top10ProbabilitySpreadIncreased": candidate["spread"]["meanTop10GrossUpSpread"] > baseline["spread"]["meanTop10GrossUpSpread"],
        "decileMonotonicityNotReduced": candidate["monotonicity"]["adjacentIncreasingFraction"] >= baseline["monotonicity"]["adjacentIncreasingFraction"],
    }
    return {"checks": checks, "allGatesPassed": all(checks.values())}


def latest_rows(
    predictions: dict[str, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    date = predictions["utility"].index.max()
    utility = predictions["utility"].loc[date].dropna().sort_values(ascending=False).head(10)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, value) in enumerate(utility.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": round(float(panel["close"].loc[date, security_id]), 4),
                "utility": round(float(value), 8),
                "expectedGrossReturn": round(float(predictions["expectedReturn"].loc[date, security_id]), 8),
                "probabilityUp": round(float(predictions["grossUp"].loc[date, security_id]), 8),
                "probabilityNetPositive": round(float(predictions["netPositive"].loc[date, security_id]), 8),
                "probabilitySevereLoss": round(float(predictions["severeLoss"].loc[date, security_id]), 8),
            }
        )
    return rows


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Thirteen-factor individual discrimination V4",
        "",
        "> Research-only / post-selection historical hypothesis / not trading.",
        "",
        f"- data: `{result['dataRange'][0]}..{result['dataRange'][1]}`",
        f"- audit gates passed: `{result['acceptance']['allGatesPassed']}`",
        f"- eligible for trading: `{result['eligibleForTrading']}`",
        "",
        "## Audit comparison",
        "",
        "| model | up AUC | up Brier | up LogLoss | net+ AUC | Top10 net | Top10 net win | Top10 p(up) spread | monotonic steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("scalarBaseline", "multivariate"):
        row = result["periods"]["audit"][name]
        lines.append(
            f"| {name} | {row['probability']['grossUp']['auc']} | {row['probability']['grossUp']['brier']} | "
            f"{row['probability']['grossUp']['logLoss']} | {row['probability']['netPositive']['auc']} | "
            f"{row['top10']['meanNetReturn']} | {row['top10']['netWinRate']} | "
            f"{row['spread']['meanTop10GrossUpSpread']} | {row['monotonicity']['adjacentIncreasingFraction']} |"
        )
    lines.extend(["", "## Latest exploratory Top10", "", "| rank | security | name | expected gross | p(up) | p(net+) | p(severe loss) |", "|---:|---|---|---:|---:|---:|---:|"])
    for row in result["latestTop10"]:
        lines.append(
            f"| {row['rank']} | {row['securityId']} | {row['name']} | {row['expectedGrossReturn']:.3%} | "
            f"{row['probabilityUp']:.2%} | {row['probabilityNetPositive']:.2%} | {row['probabilitySevereLoss']:.2%} |"
        )
    lines.extend(["", "Probability dispersion is accepted only together with better audit accuracy and Top10 outcomes. Orders remain `[]`.", ""])
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    ablation_config = load_json(ROOT / config["frozenAblationConfig"])
    factors, frozen_weights, v6_config = ablation.frozen_factors_and_weights(ablation_config)
    base = load_json(ROOT / v6_config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}", flush=True)
    ranks, factor_audit = v6.compute_factor_ranks(panel, factors)
    alpha131 = ablation_config["factorContract"]["alpha131Key"]
    zoo, name = alpha131.split("/", 1)
    raw = importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(v6.build_factor_inputs(panel))
    ranks[alpha131] = (raw.reindex_like(panel["close"]) * float(ablation_config["factorContract"]["alpha131Direction"])).where(panel["eligible"]).rank(axis=1, pct=True).astype("float32")
    factor_audit.append({"factorKey": alpha131, "direction": 1.0, "pastOnlySourceAudit": "static_formula"})
    keys = [item["factorKey"] for item in factors]
    weights = ablation.policy_weights(keys, frozen_weights, ablation_config["factorContract"]["alpha070Key"], alpha131)[config["policy"]["name"]]
    common = panel["eligible"].copy().fillna(False)
    for value in ranks.values():
        common &= value.notna()
    ranks = {key: value.where(common) for key, value in ranks.items()}
    score = ablation.weighted_score(ranks, weights, common)
    returns, execution_eligible, _delay = precision.executable_horizon_return(panel, 1, 5)
    returns = returns.where(execution_eligible & common)
    partitions = fixed_partitions(panel["close"].index, config)
    models, fit_audit = fit_models(ranks, returns, partitions, config)
    prediction_dates = pd.DatetimeIndex(sorted(set().union(*[set(value) for key, value in partitions.items() if key in {"audit", "validation", "shadow"}], {panel["close"].index.max()})))
    multi = predict_multivariate(ranks, prediction_dates, models, config)
    scalar = scalar_predictions(score, returns, partitions["calibration"], prediction_dates, v6_config)
    periods: dict[str, Any] = {}
    for name in ("audit", "validation", "shadow"):
        dates = partitions[name].intersection(prediction_dates)
        periods[name] = {
            "scalarBaseline": evaluate_period(scalar, returns, dates, config),
            "multivariate": evaluate_period(multi, returns, dates, config),
        }
    accepted = acceptance(periods["audit"]["scalarBaseline"], periods["audit"]["multivariate"])
    latest = latest_rows(multi, panel, v6.name_map(base))
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_post_selection_historical_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "dataRange": [str(panel["close"].index.min().date()), str(panel["close"].index.max().date())],
        "signalDate": str(panel["close"].index.max().date()),
        "weights": weights,
        "partitions": {name: [str(value.min().date()), str(value.max().date()), len(value)] for name, value in partitions.items()},
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "fitAudit": fit_audit,
        "periods": periods,
        "acceptance": accepted,
        "latestTop10": latest,
        "historicalRunCanPromote": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    identifier = run_id or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z")
    result["runId"] = identifier
    root = ROOT / config["output"]["root"] / identifier
    atomic_text(root / "result.json", json.dumps(safe(result), ensure_ascii=False, indent=2) + "\n")
    atomic_text(root / "report.md", report_markdown(result))
    return {"runId": identifier, "report": str(root / "report.md"), "auditGatesPassed": accepted["allGatesPassed"], "latestTop10": latest, "eligibleForTrading": False}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(safe(run(args.config.resolve(), args.run_id)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
