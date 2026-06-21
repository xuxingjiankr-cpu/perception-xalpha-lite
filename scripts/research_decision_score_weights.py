"""One-shot May-train / June-test research for decision-score components.

May is development-only and known contaminated. June uses point-in-time replay rows, but
the scorer itself was designed after the sample existed, so the result remains diagnostic.
Exactly one day-balanced ridge candidate with a fixed penalty is fitted; there is no grid
search and no live/config output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from run_etf_paper_trading_agent import ROOT, as_float


FEATURES = [
    "market_regime_score",
    "relative_strength_score",
    "liquidity_score",
    "entry_quality_score",
    "execution_score",
    "counterfactual_score",
    "risk_penalty",
]
TRAIN_END = "2026-05-31"
TEST_START = "2026-06-01"
ROUND_TRIP_COST = 0.0014
RIDGE_ALPHA = 10.0
MIN_TEST_DAYS = 20
DEFAULT_SCORES = (
    ROOT / "outputs" / "decision_score_pseudo_forward" / "20260506_20260618"
    / "scores_with_yahoo_may_contaminated"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_score_weight_research"


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
    n_days = max(1, len(counts))
    scale = len(rows) / n_days if rows else 1.0
    return np.array([scale / counts[str(row.get("date"))] for row in rows], dtype=float)


def fit_fixed_ridge(rows: list[dict[str, Any]], *, alpha: float = RIDGE_ALPHA,
                    cost: float = ROUND_TRIP_COST) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty training rows")
    raw = np.array([[as_float(row.get(feature)) for feature in FEATURES] for row in rows], dtype=float)
    target = np.array([as_float(row.get("realized_return")) - cost for row in rows], dtype=float)
    weights = day_balanced_weights(rows)
    weights = weights / weights.sum()
    center = np.sum(raw * weights[:, None], axis=0)
    variance = np.sum(((raw - center) ** 2) * weights[:, None], axis=0)
    scale = np.sqrt(variance)
    active = scale > 1e-12
    safe_scale = np.where(active, scale, 1.0)
    standardized = (raw - center) / safe_scale
    design = np.column_stack([np.ones(len(rows)), standardized])
    root_w = np.sqrt(weights * len(rows))
    weighted_design = design * root_w[:, None]
    weighted_target = target * root_w
    penalty = np.diag([0.0] + [alpha] * len(FEATURES))
    beta = np.linalg.solve(weighted_design.T @ weighted_design + penalty,
                           weighted_design.T @ weighted_target)
    beta[1:][~active] = 0.0
    return {
        "intercept": float(beta[0]),
        "coefficients": {feature: float(beta[i + 1]) for i, feature in enumerate(FEATURES)},
        "center": {feature: float(center[i]) for i, feature in enumerate(FEATURES)},
        "scale": {feature: float(safe_scale[i]) for i, feature in enumerate(FEATURES)},
        "active": {feature: bool(active[i]) for i, feature in enumerate(FEATURES)},
        "alpha": alpha,
        "round_trip_cost": cost,
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    prediction = np.full(len(rows), as_float(model.get("intercept")), dtype=float)
    for feature in FEATURES:
        if not model["active"].get(feature):
            continue
        raw = np.array([as_float(row.get(feature)) for row in rows], dtype=float)
        prediction += (
            (raw - as_float(model["center"].get(feature)))
            / as_float(model["scale"].get(feature), 1.0)
            * as_float(model["coefficients"].get(feature))
        )
    return prediction


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float | None:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    if rank:
        x, y = ranks(x), ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def frozen_tertiles(train_score: np.ndarray, test_score: np.ndarray,
                    test_return: np.ndarray) -> dict[str, Any]:
    low, high = np.quantile(train_score, [1 / 3, 2 / 3])
    bottom = test_return[test_score <= low]
    top = test_return[test_score >= high]
    return {
        "train_low_cut": float(low),
        "train_high_cut": float(high),
        "test_top_count": int(len(top)),
        "test_bottom_count": int(len(bottom)),
        "test_top_mean": float(np.mean(top)) if len(top) else None,
        "test_bottom_mean": float(np.mean(bottom)) if len(bottom) else None,
        "test_top_minus_bottom": float(np.mean(top) - np.mean(bottom)) if len(top) and len(bottom) else None,
    }


def metrics(train_scores: np.ndarray, test_scores: np.ndarray,
            test_returns: np.ndarray) -> dict[str, Any]:
    return {
        "pearson": correlation(test_scores, test_returns),
        "spearman": correlation(test_scores, test_returns, rank=True),
        "tertiles": frozen_tertiles(train_scores, test_scores, test_returns),
    }


def render_markdown(result: dict[str, Any]) -> str:
    frozen = result["test_metrics"]["frozen_total_score"]
    candidate = result["test_metrics"]["ridge_candidate"]
    coeffs = result["candidate_model"]["coefficients"]
    active = result["candidate_model"]["active"]
    shadow = result["shadow_hypothesis"]
    fmt = lambda value: "n/a" if value is None else f"{value:.6f}"
    lines = [
        "# Decision-Score Weight Research: May Train / June Test",
        "",
        "Status: `diagnostic_only / candidate_not_promotable`",
        "",
        "- May training universe uses the known final-day-turnover-contaminated Yahoo fallback.",
        "- June test rows are point-in-time, but the scorer design post-dates the sample.",
        "- BUY decisions only; target is next-snapshot-to-close return minus 14bps.",
        "- One fixed ridge candidate (`alpha=10`), day-balanced. No hyperparameter search.",
        "- No live config, score range, strategy parameter or execution lock was changed.",
        "",
        "## Sample",
        "",
        f"- train: {result['sample']['train_rows']} BUY decisions / {result['sample']['train_days']} days",
        f"- test: {result['sample']['test_rows']} BUY decisions / {result['sample']['test_days']} days",
        f"- formal minimum: {result['gates']['minimum_test_days']} test days",
        "",
        "## June fixed-test comparison",
        "",
        "| model | Pearson | Spearman | top-minus-bottom net return |",
        "|---|---:|---:|---:|",
        f"| frozen total score | {fmt(frozen['pearson'])} | {fmt(frozen['spearman'])} | {fmt(frozen['tertiles']['test_top_minus_bottom'])} |",
        f"| May ridge candidate | {fmt(candidate['pearson'])} | {fmt(candidate['spearman'])} | {fmt(candidate['tertiles']['test_top_minus_bottom'])} |",
        "",
        "## Data-derived shadow hypothesis",
        "",
        f"- Keep the existing frozen component weights unchanged.",
        f"- Shadow-tag BUY decisions with `total_score >= {shadow['total_score_min']}`.",
        f"- June observations: {shadow['test_count']} BUY decisions; cost-adjusted mean "
        f"{shadow['test_mean_net_return']:.4%}.",
        f"- May-low-threshold comparison group mean: {shadow['test_bottom_mean_net_return']:.4%}.",
        "- This is a forward hypothesis only; `trade_gate_enabled=false`.",
        "",
        "## Candidate standardized coefficients",
        "",
        "| component | active in May | coefficient |",
        "|---|---|---:|",
    ]
    for feature in FEATURES:
        lines.append(f"| {feature} | {str(active[feature]).lower()} | {coeffs[feature]:+.8f} |")
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- test_days_gate: `{result['gates']['test_days_gate']}`",
        f"- candidate_improves_both_metrics: `{result['gates']['candidate_improves_both_metrics']}`",
        f"- promotion_allowed: `{result['gates']['promotion_allowed']}`",
        "- These workdays can prioritize hypotheses, but cannot authorize weight changes.",
        "- Keep the existing scorer frozen; compare this one preregistered candidate only on new real-forward days.",
        "",
    ])
    return "\n".join(lines)


def run_research(score_dir: Path) -> dict[str, Any]:
    records = load_buy_records(score_dir)
    train = [row for row in records if str(row.get("date")) <= TRAIN_END]
    test = [row for row in records if str(row.get("date")) >= TEST_START]
    if not train or not test:
        raise RuntimeError("May train or June test BUY records are missing")
    model = fit_fixed_ridge(train)
    train_candidate = predict(model, train)
    test_candidate = predict(model, test)
    train_frozen = np.array([as_float(row.get("total_score")) for row in train])
    test_frozen = np.array([as_float(row.get("total_score")) for row in test])
    test_returns = np.array([as_float(row.get("realized_return")) - ROUND_TRIP_COST for row in test])
    frozen_metrics = metrics(train_frozen, test_frozen, test_returns)
    candidate_metrics = metrics(train_candidate, test_candidate, test_returns)
    test_days = len({str(row.get("date")) for row in test})
    frozen_spread = frozen_metrics["tertiles"]["test_top_minus_bottom"]
    candidate_spread = candidate_metrics["tertiles"]["test_top_minus_bottom"]
    frozen_tertiles = frozen_metrics["tertiles"]
    improves = bool(
        candidate_metrics["pearson"] is not None and frozen_metrics["pearson"] is not None
        and candidate_metrics["pearson"] > frozen_metrics["pearson"]
        and candidate_spread is not None and frozen_spread is not None
        and candidate_spread > frozen_spread
    )
    return {
        "status": "diagnostic_only",
        "candidate_status": "candidate_not_promotable",
        "hypothesis": "May-only day-balanced component weighting improves June cost-adjusted BUY ranking",
        "data": {
            "score_dir": str(score_dir),
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "may_training_contaminated": True,
            "june_point_in_time_rows": True,
            "researcher_selection_bias": True,
        },
        "sample": {
            "train_rows": len(train),
            "train_days": len({str(row.get("date")) for row in train}),
            "test_rows": len(test),
            "test_days": test_days,
        },
        "candidate_model": model,
        "test_metrics": {
            "frozen_total_score": frozen_metrics,
            "ridge_candidate": candidate_metrics,
        },
        "shadow_hypothesis": {
            "name": "frozen_score_high_tertile_buy",
            "total_score_min": frozen_tertiles["train_high_cut"],
            "test_count": frozen_tertiles["test_top_count"],
            "test_mean_net_return": frozen_tertiles["test_top_mean"],
            "test_bottom_count": frozen_tertiles["test_bottom_count"],
            "test_bottom_mean_net_return": frozen_tertiles["test_bottom_mean"],
            "trade_gate_enabled": False,
            "weights_changed": False,
            "status": "unvalidated_forward_shadow",
        },
        "gates": {
            "minimum_test_days": MIN_TEST_DAYS,
            "test_days_gate": test_days >= MIN_TEST_DAYS,
            "candidate_improves_both_metrics": improves,
            "promotion_allowed": False,
            "reason": "historical diagnostic only; fewer than 20 June BUY-outcome days and scorer post-dates sample",
        },
        "live_changes": False,
        "weights_refit_applied": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Research frozen decision-score component weights offline.")
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    score_dir = Path(args.scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_research(score_dir)
    json_path = output_dir / "may_train_june_test_weight_candidate.json"
    md_path = output_dir / "may_train_june_test_weight_candidate.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render_markdown(result)
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
