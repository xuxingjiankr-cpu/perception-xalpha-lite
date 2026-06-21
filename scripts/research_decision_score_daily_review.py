"""Daily forward review of decision-score discrimination (research-only).

The review asks whether semantically high component scores actually outperform low
scores on completed forward BUY-direction outcomes.  It controls market-day confounding
with same-day high-minus-low differences, bootstraps trading days (not individual rows),
and applies Holm correction across the seven component tests.  It never rewrites weights,
strategy parameters, probability models, or execution gates.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import decision_probability as dp
import decision_scoring as ds
from run_etf_paper_trading_agent import ROOT, as_float


OUTPUT_DIR = ROOT / "outputs" / "decision_score_review"
ROUND_TRIP_COST = 0.0014
MIN_REVIEW_DAYS = 20
MIN_GROUP_ROWS = 30
MIN_PAIRED_DAYS = 10
MIN_ADJUSTMENT_DAYS = 40
MIN_ADJUSTMENT_GROUP_ROWS = 50
MIN_ADJUSTMENT_PAIRED_DAYS = 20
EQUIVALENCE_MARGIN = 0.0005  # 5bp; only a narrow CI fully inside this band is "equivalent".
BOOTSTRAPS = 2000


FEATURES = ["total_score", *ds.SCORE_RANGES.keys()]


def thresholds(feature: str) -> tuple[float, float]:
    if feature == "total_score":
        return 65.0, 71.0
    low, high = ds.SCORE_RANGES[feature]
    width = high - low
    return low + width / 3.0, low + 2.0 * width / 3.0


def row_return(row: dict[str, Any]) -> float | None:
    if str(row.get("signal_direction") or row.get("decision_type")) != "BUY":
        return None
    if row.get("probability_outcome") is None or row.get("outcome_horizon_complete") is not True:
        return None
    raw = row.get("counterfactual_return") if row.get("decision_type") == "BUY_CANDIDATE" else row.get("realized_return")
    return as_float(raw) - as_float(row.get("estimated_round_trip_cost"), ROUND_TRIP_COST) if raw is not None else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_paired_days(differences: list[float], feature: str) -> dict[str, Any]:
    if len(differences) < 2:
        return {"n": 0, "ci95": [None, None], "pTwoSided": None}
    seed = int(hashlib.sha256(feature.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    samples = [mean(rng.choices(differences, k=len(differences))) for _ in range(BOOTSTRAPS)]
    non_positive = (sum(value <= 0 for value in samples) + 1) / (len(samples) + 1)
    non_negative = (sum(value >= 0 for value in samples) + 1) / (len(samples) + 1)
    return {
        "n": BOOTSTRAPS,
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "pTwoSided": min(1.0, 2.0 * min(non_positive, non_negative)),
    }


def feature_review(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    low_cut, high_cut = thresholds(feature)
    high = [(row, row_return(row)) for row in rows if as_float(row.get(feature)) >= high_cut]
    low = [(row, row_return(row)) for row in rows if as_float(row.get(feature)) <= low_cut]
    high = [(row, value) for row, value in high if value is not None]
    low = [(row, value) for row, value in low if value is not None]
    high_values = [value for _, value in high]
    low_values = [value for _, value in low]
    by_day_high: dict[str, list[float]] = defaultdict(list)
    by_day_low: dict[str, list[float]] = defaultdict(list)
    for row, value in high:
        by_day_high[str(row.get("date"))].append(value)
    for row, value in low:
        by_day_low[str(row.get("date"))].append(value)
    paired_dates = sorted(set(by_day_high) & set(by_day_low))
    paired = [mean(by_day_high[date]) - mean(by_day_low[date]) for date in paired_dates]
    bootstrap = bootstrap_paired_days(paired, feature)
    high_mean = mean(high_values) if high_values else None
    low_mean = mean(low_values) if low_values else None
    difference = mean(paired) if paired else (
        high_mean - low_mean if high_mean is not None and low_mean is not None else None
    )
    days = len({str(row.get("date")) for row in rows if row_return(row) is not None})
    review_ready = (days >= MIN_REVIEW_DAYS and len(high) >= MIN_GROUP_ROWS
                    and len(low) >= MIN_GROUP_ROWS and len(paired) >= MIN_PAIRED_DAYS)
    adjustment_ready = (days >= MIN_ADJUSTMENT_DAYS and len(high) >= MIN_ADJUSTMENT_GROUP_ROWS
                        and len(low) >= MIN_ADJUSTMENT_GROUP_ROWS
                        and len(paired) >= MIN_ADJUSTMENT_PAIRED_DAYS)
    return {
        "feature": feature,
        "lowCut": low_cut,
        "highCut": high_cut,
        "highCount": len(high),
        "lowCount": len(low),
        "independentDays": days,
        "pairedDays": len(paired),
        "highMeanNetReturn": high_mean,
        "lowMeanNetReturn": low_mean,
        "highWinRate": sum(value > 0 for value in high_values) / len(high_values) if high_values else None,
        "lowWinRate": sum(value > 0 for value in low_values) / len(low_values) if low_values else None,
        "pairedHighMinusLow": difference,
        "bootstrap": bootstrap,
        "reviewReady": review_ready,
        "adjustmentReady": adjustment_ready,
        "highNotStrong": difference is not None and difference <= 0,
        "highGroupNonPositive": high_mean is not None and high_mean <= 0,
        "lowNotWeak": low_mean is not None and low_mean >= 0,
        "holmAdjustedP": None,
        "assessment": "insufficient_forward_sample",
        "suggestedAction": "keep_frozen_and_collect",
    }


def apply_holm(reviews: list[dict[str, Any]]) -> None:
    valid = [(index, review["bootstrap"]["pTwoSided"])
             for index, review in enumerate(reviews)
             if review["bootstrap"]["pTwoSided"] is not None]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        reviews[index]["holmAdjustedP"] = running


def classify_review(review: dict[str, Any]) -> None:
    if not review["reviewReady"]:
        return
    difference = review["pairedHighMinusLow"]
    adjusted_p = review["holmAdjustedP"]
    ci_low, ci_high = review["bootstrap"]["ci95"]
    if not review["adjustmentReady"]:
        review["assessment"] = "review_only_not_adjustment_ready"
        review["suggestedAction"] = "observe_without_weight_change"
    elif adjusted_p is not None and adjusted_p < 0.05 and difference is not None and difference < 0:
        review["assessment"] = "statistically_inverted_high_score_underperforms"
        review["suggestedAction"] = "candidate_downweight_or_redefine_after_separate_oos"
    elif adjusted_p is not None and adjusted_p < 0.05 and difference is not None and difference > 0:
        review["assessment"] = "statistically_positive_discrimination"
        review["suggestedAction"] = "retain_weight_no_automatic_increase"
    elif ci_low is not None and ci_high is not None and ci_low >= -EQUIVALENCE_MARGIN and ci_high <= EQUIVALENCE_MARGIN:
        review["assessment"] = "practically_equivalent_high_and_low_within_5bp"
        review["suggestedAction"] = "candidate_simplification_after_separate_oos"
    else:
        review["assessment"] = "uncertain_no_statistical_adjustment"
        review["suggestedAction"] = "keep_frozen_and_collect"


def build_review(records: list[dict[str, Any]], as_of_date: str) -> dict[str, Any]:
    effective = str((dp.load_shadow_model() or {}).get("effectiveFrom") or "2026-06-22")
    forward = [row for row in records if effective <= str(row.get("date")) <= as_of_date]
    buy_rows = [row for row in forward if row_return(row) is not None]
    reviews = [feature_review(buy_rows, feature) for feature in FEATURES]
    apply_holm(reviews)
    for review in reviews:
        classify_review(review)
    ledger_counts = Counter(str(row.get("ledger_record_type") or "legacy") for row in forward)
    decision_counts = Counter(str(row.get("decision_type") or "UNKNOWN") for row in forward)
    return {
        "schemaVersion": "decision_score_daily_statistical_review_v1",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "asOfDate": as_of_date,
        "effectiveFrom": effective,
        "status": "diagnostic_only",
        "autoWeightChangeAllowed": False,
        "tradeGateChangeAllowed": False,
        "method": {
            "outcome": "BUY-direction next-snapshot-to-close return minus recorded round-trip cost",
            "highLowCuts": "fixed semantic score-range thirds; total score uses <=65 and >=71",
            "marketDayControl": "same-day high-minus-low means",
            "uncertainty": f"{BOOTSTRAPS} trading-day bootstrap resamples",
            "multipleTesting": "Holm family-wise correction across total + seven component scores",
            "equivalenceMargin": EQUIVALENCE_MARGIN,
        },
        "sample": {
            "allDecisionRecords": len(forward),
            "completedBuyDirectionOutcomes": len(buy_rows),
            "independentTradingDays": len({str(row.get("date")) for row in buy_rows}),
            "ledgerRecordTypes": dict(ledger_counts),
            "decisionTypes": dict(decision_counts),
        },
        "reviews": reviews,
        "adjustmentCandidates": [review["feature"] for review in reviews
                                 if review["suggestedAction"].startswith("candidate_")],
        "safetyConclusion": "recommendations_only; weights remain frozen; separate future OOS required",
    }


def fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3%}" if percent else f"{float(value):.6f}"


def render(review: dict[str, Any]) -> str:
    sample = review["sample"]
    lines = ["# Decision Score Daily Statistical Review", "",
             "Status: `diagnostic_only / no automatic weight changes`", "",
             f"- as_of_date: {review['asOfDate']}",
             f"- records: {sample['allDecisionRecords']}; completed BUY-direction outcomes: "
             f"{sample['completedBuyDirectionOutcomes']}; independent days: {sample['independentTradingDays']}",
             f"- decision types: `{json.dumps(sample['decisionTypes'], ensure_ascii=False, sort_keys=True)}`",
             f"- ledger types: `{json.dumps(sample['ledgerRecordTypes'], ensure_ascii=False, sort_keys=True)}`",
             "- High/low comparisons are paired within the same market day; uncertainty resamples trading days.",
             "- Holm correction controls the eight simultaneous score tests.", "",
             "| score part | high n | low n | paired days | high mean | low mean | paired high-low | 95% CI | Holm p | diagnostic flags | assessment |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for item in review["reviews"]:
        ci = item["bootstrap"]["ci95"]
        flags = ", ".join(name for name, enabled in (
            ("high_not_strong", item["highNotStrong"]),
            ("high_non_positive", item["highGroupNonPositive"]),
            ("low_not_weak", item["lowNotWeak"]),
        ) if enabled) or "-"
        lines.append(f"| {item['feature']} | {item['highCount']} | {item['lowCount']} | {item['pairedDays']} | "
                     f"{fmt(item['highMeanNetReturn'], True)} | {fmt(item['lowMeanNetReturn'], True)} | "
                     f"{fmt(item['pairedHighMinusLow'], True)} | [{fmt(ci[0], True)}, {fmt(ci[1], True)}] | "
                     f"{fmt(item['holmAdjustedP'])} | {flags} | {item['assessment']} |")
    lines.extend(["", "## Adjustment audit", ""])
    if review["adjustmentCandidates"]:
        lines.append("Statistical adjustment candidates: " + ", ".join(review["adjustmentCandidates"]))
    else:
        lines.append("No score component is currently eligible for statistical adjustment.")
    lines.extend(["", "Any future adjustment needs >=40 independent days, adequate high/low groups, Holm-adjusted evidence, and a separate OOS validation. Nothing is auto-applied.", ""])
    return "\n".join(lines)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_review(records: list[dict[str, Any]], as_of_date: str) -> tuple[Path, Path, dict[str, Any]]:
    review = build_review(records, as_of_date)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compact = as_of_date.replace("-", "")
    json_path = OUTPUT_DIR / f"daily_score_review_{compact}.json"
    md_path = OUTPUT_DIR / f"daily_score_review_{compact}.md"
    atomic_json(json_path, review)
    md_path.write_text(render(review), encoding="utf-8")
    atomic_json(OUTPUT_DIR / "latest_daily_score_review.json", review)
    (OUTPUT_DIR / "latest_daily_score_review.md").write_text(render(review), encoding="utf-8")
    return json_path, md_path, review


def main() -> int:
    import run_decision_score_report as score_report
    as_of = datetime.now().astimezone().date().isoformat()
    json_path, md_path, review = write_review(score_report.load_records(None), as_of)
    print(render(review))
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
