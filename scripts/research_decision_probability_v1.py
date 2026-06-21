"""Fit and audit one conservative score-to-probability shadow model.

Preregistered design (DSI-0009): May-only training, fixed June diagnostic test, BUY
decisions, next-snapshot-to-close outcome, 14bps round-trip cost.  The intercept is a
Beta(2,2)-smoothed base rate and the only fitted evidence term is total_score with a
fixed L2 penalty.  No hyperparameter search, isotonic fit, component refit, trade gate,
position sizing or automatic promotion is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import decision_probability as dp
from run_etf_paper_trading_agent import ROOT, as_float


TRAIN_END = "2026-05-31"
TEST_START = "2026-06-01"
EFFECTIVE_FROM = "2026-06-22"
ROUND_TRIP_COST = 0.0014
BETA_ALPHA = 2.0
BETA_BETA = 2.0
SLOPE_L2 = 10.0
CONFIDENCE_K_DAYS = 50
MIN_FORWARD_DAYS = 20
MIN_FORWARD_OUTCOMES = 50
DEFAULT_SCORES = (
    ROOT / "outputs" / "decision_score_pseudo_forward" / "20260506_20260618"
    / "scores_with_yahoo_may_contaminated"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_probability_research"
DEFAULT_MODEL = ROOT / "configs" / "shadow" / "decision_probability_v1.json"


def load_buy_records(score_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(score_dir.glob("decision_scores_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (isinstance(row, dict) and row.get("decision_type") == "BUY"
                    and row.get("realized_return") is not None):
                rows.append(row)
    return rows


def day_balanced_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(str(row.get("date")) for row in rows)
    scale = len(rows) / max(1, len(counts))
    return np.array([scale / counts[str(row.get("date"))] for row in rows], dtype=float)


def fit_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty probability training sample")
    score = np.array([as_float(row.get("total_score")) for row in rows], dtype=float)
    gross = np.array([as_float(row.get("realized_return")) for row in rows], dtype=float)
    outcome = (gross - ROUND_TRIP_COST > 0).astype(float)
    weight = day_balanced_weights(rows)
    total_weight = float(weight.sum())
    prior = float((np.dot(weight, outcome) + BETA_ALPHA)
                  / (total_weight + BETA_ALPHA + BETA_BETA))
    center = float(np.dot(weight, score) / total_weight)
    variance = float(np.dot(weight, (score - center) ** 2) / total_weight)
    scale = max(variance ** 0.5, 1.0)
    standardized = (score - center) / scale

    # Intercept stays anchored to the Beta-smoothed base rate.  Fit exactly one slope.
    slope = 0.0
    prior_logodds = dp.logit(prior)
    for _ in range(100):
        probability = np.array([dp.sigmoid(prior_logodds + slope * x)
                                for x in standardized], dtype=float)
        gradient = float(np.dot(weight * standardized, outcome - probability) - SLOPE_L2 * slope)
        information = float(np.dot(weight * standardized ** 2, probability * (1.0 - probability))
                            + SLOPE_L2)
        step = gradient / max(information, 1e-12)
        slope += step
        if abs(step) < 1e-12:
            break

    successes = outcome == 1
    failures = ~successes
    expected_win = float(np.average(gross[successes], weights=weight[successes])) if successes.any() else 0.0
    failure_mean = float(np.average(gross[failures], weights=weight[failures])) if failures.any() else 0.0
    expected_loss = max(0.0, -failure_mean)
    training_days = len({str(row.get("date")) for row in rows})
    sample_hash_fields = [
        {"date": row.get("date"), "decision_id": row.get("decision_id"),
         "total_score": row.get("total_score"), "realized_return": row.get("realized_return")}
        for row in rows
    ]
    sample_hash = hashlib.sha256(json.dumps(
        sample_hash_fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "schemaVersion": "decision_probability_shadow_v1",
        "iterationId": "DSI-0009",
        "calibrationVersion": "DCAL-1.0.0",
        "bayesianModelVersion": "DBAYES-1.0.0",
        "shadowCandidateVersion": "DSHADOW-2.0.0",
        "recordOnly": True,
        "tradeGateEnabled": False,
        "positionSizingEnabled": False,
        "autoPromotionAllowed": False,
        "effectiveFrom": EFFECTIVE_FROM,
        "decisionType": "BUY",
        "horizon": "next_snapshot_to_same_day_last_quote",
        "successDefinition": "realized_return_minus_0.0014_gt_0",
        "prior": {
            "type": "day_balanced_beta_binomial",
            "alpha": BETA_ALPHA,
            "beta": BETA_BETA,
            "probability": prior,
            "logOdds": prior_logodds,
        },
        "scoreTransform": {
            "feature": "total_score",
            "center": center,
            "scale": scale,
            "regularizedSlope": slope,
            "l2Penalty": SLOPE_L2,
            "interpretation": "slope_times_z_is_discounted_log_likelihood_ratio",
        },
        "payoff": {
            "expectedGrossWin": expected_win,
            "expectedGrossLoss": expected_loss,
            "trainingFailureGrossMean": failure_mean,
            "roundTripCost": ROUND_TRIP_COST,
            "evFormula": "p*expectedGrossWin-(1-p)*expectedGrossLoss-roundTripCost",
        },
        "modelConfidence": training_days / (training_days + CONFIDENCE_K_DAYS),
        "training": {
            "endDate": TRAIN_END,
            "rows": len(rows),
            "independentTradingDays": training_days,
            "sampleSha256": sample_hash,
            "maySourceKnownContaminated": True,
            "scorerPostDatesSample": True,
        },
        "forwardValidation": {
            "minimumIndependentTradingDays": MIN_FORWARD_DAYS,
            "minimumBuyOutcomes": MIN_FORWARD_OUTCOMES,
            "metrics": ["brier", "log_loss", "auc", "ece", "reliability_bins"],
            "refitPolicy": "manual_only_after_forward_gate; never automatic",
        },
        "limitations": [
            "May training source uses known final-day-turnover-contaminated fallback data.",
            "The score design post-dates both May and June historical samples.",
            "Multiple decisions per day are correlated; model confidence uses trading days.",
            "No regime-specific prior or multi-evidence LR is fit at this sample size.",
        ],
    }


def evaluate(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [1 if as_float(row.get("realized_return")) - ROUND_TRIP_COST > 0 else 0
                for row in rows]
    forecasts = [dp.forecast_shadow(total_score=as_float(row.get("total_score")),
                                    decision_type="BUY", date=EFFECTIVE_FROM, model=model)
                 for row in rows]
    probability = [as_float(value.get("posterior_prob"), 0.5) for value in forecasts]
    prior = as_float(model.get("prior", {}).get("probability"), 0.5)
    return {
        "model": dp.probability_metrics(probability, outcomes),
        "constant_training_prior": dp.probability_metrics([prior] * len(rows), outcomes),
        "neutral_half": dp.probability_metrics([0.5] * len(rows), outcomes),
        "days": len({str(row.get("date")) for row in rows}),
    }


def render_report(result: dict[str, Any]) -> str:
    model_metrics = result["june_diagnostic"]["model"]
    prior_metrics = result["june_diagnostic"]["constant_training_prior"]
    neutral_metrics = result["june_diagnostic"]["neutral_half"]
    model = result["model"]
    fmt = lambda value: "n/a" if value is None else f"{value:.6f}"
    lines = [
        "# Decision Probability V1 — May Train / June Diagnostic Test",
        "",
        "Status: `diagnostic_only / forward_shadow_not_promotable`",
        "",
        "## Preregistered model",
        "",
        "- BUY only; outcome = next-snapshot-to-close gross return minus 14bps > 0.",
        "- Beta(2,2)-smoothed, day-balanced base rate; one total-score log-odds slope.",
        "- Fixed L2=10. No component refit, grid search, isotonic fit or threshold search.",
        "- The score likelihood term is one combined evidence term, so correlated sub-scores are not multiplied again.",
        "- `tradeGateEnabled=false`; `positionSizingEnabled=false`; action is always record-only.",
        "",
        "## Model parameters",
        "",
        f"- prior probability: {model['prior']['probability']:.4%}",
        f"- total-score center / scale: {model['scoreTransform']['center']:.4f} / {model['scoreTransform']['scale']:.4f}",
        f"- regularized log-odds slope: {model['scoreTransform']['regularizedSlope']:+.6f}",
        f"- model confidence (independent days n/(n+50)): {model['modelConfidence']:.4%}",
        f"- expected gross win / loss / cost: {model['payoff']['expectedGrossWin']:.4%} / "
        f"{model['payoff']['expectedGrossLoss']:.4%} / {model['payoff']['roundTripCost']:.4%}",
        "",
        "## Fixed June diagnostic",
        "",
        f"- sample: {model_metrics['count']} BUY outcomes / {result['june_diagnostic']['days']} days",
        "",
        "| forecast | Brier ↓ | LogLoss ↓ | AUC ↑ | ECE ↓ | mean p | actual win rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| constant May prior | {fmt(prior_metrics['brier'])} | {fmt(prior_metrics['log_loss'])} | "
        f"{fmt(prior_metrics['auc'])} | {fmt(prior_metrics['ece'])} | "
        f"{fmt(prior_metrics['mean_predicted'])} | {fmt(prior_metrics['actual_rate'])} |",
        f"| neutral 50% reference | {fmt(neutral_metrics['brier'])} | {fmt(neutral_metrics['log_loss'])} | "
        f"{fmt(neutral_metrics['auc'])} | {fmt(neutral_metrics['ece'])} | "
        f"{fmt(neutral_metrics['mean_predicted'])} | {fmt(neutral_metrics['actual_rate'])} |",
        f"| one-slope posterior | {fmt(model_metrics['brier'])} | {fmt(model_metrics['log_loss'])} | "
        f"{fmt(model_metrics['auc'])} | {fmt(model_metrics['ece'])} | "
        f"{fmt(model_metrics['mean_predicted'])} | {fmt(model_metrics['actual_rate'])} |",
        "",
        "### Reliability bins",
        "",
        "| predicted interval | n | mean predicted | actual rate |",
        "|---|---:|---:|---:|",
    ]
    for bucket in model_metrics["bins"]:
        lines.append(f"| [{bucket['low']:.1f}, {bucket['high']:.1f}) | {bucket['count']} | "
                     f"{bucket['mean_predicted']:.4%} | {bucket['actual_rate']:.4%} |")
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- test-day gate (>=20): `{result['gates']['june_test_days_gate']}`",
        f"- calibration beats constant prior on both Brier and LogLoss: `"
        f"{result['gates']['beats_prior_brier_and_logloss']}`",
        f"- calibration beats neutral 50% on both Brier and LogLoss: `"
        f"{result['gates']['beats_neutral_half_brier_and_logloss']}`",
        "- promotion_allowed: `false`",
        "- AUC shows a weak ranking hint, but predicted probabilities are materially too low in June and do not beat the neutral 50% reference on proper scoring rules.",
        "- June has only 13 independent days and was already inspected during score research. It is not a clean confirmation.",
        "- The checked-in parameters are a preregistered forecast for data from 2026-06-22 onward. Refit is manual only after >=50 BUY outcomes and >=20 independent days.",
        "- Regime priors, four evidence-group LRs, EV gating and Bayesian position sizing remain deferred; fitting them now would be small-sample overfit.",
        "",
    ])
    return "\n".join(lines)


def run(score_dir: Path) -> dict[str, Any]:
    records = load_buy_records(score_dir)
    train = [row for row in records if str(row.get("date")) <= TRAIN_END]
    test = [row for row in records if str(row.get("date")) >= TEST_START]
    if not train or not test:
        raise RuntimeError("May training or June test BUY outcomes are missing")
    model = fit_model(train)
    diagnostic = evaluate(model, test)
    mm, pm = diagnostic["model"], diagnostic["constant_training_prior"]
    nm = diagnostic["neutral_half"]
    return {
        "status": "diagnostic_only",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "model": model,
        "data": {"scoreDir": str(score_dir), "trainEnd": TRAIN_END,
                 "testStart": TEST_START, "mayTrainingContaminated": True,
                 "junePointInTime": True, "researcherSelectionBias": True},
        "sample": {"trainRows": len(train),
                   "trainDays": len({str(row.get('date')) for row in train}),
                   "testRows": len(test), "testDays": diagnostic["days"]},
        "june_diagnostic": diagnostic,
        "gates": {
            "june_test_days_gate": diagnostic["days"] >= MIN_FORWARD_DAYS,
            "beats_prior_brier_and_logloss": bool(
                mm["brier"] is not None and pm["brier"] is not None and mm["brier"] < pm["brier"]
                and mm["log_loss"] < pm["log_loss"]
            ),
            "beats_neutral_half_brier_and_logloss": bool(
                mm["brier"] is not None and nm["brier"] is not None and mm["brier"] < nm["brier"]
                and mm["log_loss"] < nm["log_loss"]
            ),
            "promotion_allowed": False,
        },
        "liveChanges": False,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit frozen Bayesian score probability shadow V1.")
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    result = run(Path(args.scores))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(Path(args.model), result["model"])
    atomic_json(output / "decision_probability_v1_diagnostic.json", result)
    report = render_report(result)
    (output / "decision_probability_v1_diagnostic.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
