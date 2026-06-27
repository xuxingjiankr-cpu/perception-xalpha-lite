"""Decision-score effectiveness report (offline, read-only).

Reads outputs/decision_scores/decision_scores_*.jsonl, groups decisions by total-score
bucket and by each sub-score tertile, and reports realized performance so we can see
whether high scores actually do better and which sub-score predicts. Also attributes
mistakes and writes a conclusion -- or `sample_insufficient` when there isn't enough
data for any honest conclusion.

NOT alpha. NOT an auto-trading signal. Diagnostic only.
Run: py -3.13 scripts/run_decision_score_report.py [--date YYYYMMDD]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float
from decision_scoring import classify_mistake
import decision_probability as dp

SCORE_DIR = ROOT / "outputs" / "decision_scores"
SHADOW_CONFIG = ROOT / "configs" / "shadow" / "decision_score_high71_candidate.json"
MIN_OUTCOMES_FOR_CONCLUSION = 30
MIN_OUTCOME_DAYS_FOR_CONCLUSION = 20
SUBSCORES = [
    "market_regime_score", "relative_strength_score", "liquidity_score",
    "entry_quality_score", "execution_score", "counterfactual_score", "risk_penalty",
]
FIXED_TOTAL_SCORE_BUCKETS = [
    ("0-40", 0.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-75", 60.0, 75.0),
    ("75-90", 75.0, 90.0),
    ("90+", 90.0, None),
]


def _outcome(rec: dict[str, Any]) -> float | None:
    decision_type = str(rec.get("decision_type") or "").upper()
    signal_direction = str(rec.get("signal_direction") or "").upper()
    if rec.get("counterfactual_return") is not None and (
        decision_type == "BUY_CANDIDATE" or signal_direction == "BUY"
    ):
        return float(rec["counterfactual_return"])
    if decision_type in ("BUY", "SELL"):
        for k in ("realized_return", "return_1d"):
            if rec.get(k) is not None:
                return float(rec[k])
    return None


def _grp_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    rets = [_outcome(r) for r in recs]
    rets = [x for x in rets if x is not None]
    trades = [r for r in recs if str(r.get("decision_type")) in ("BUY", "SELL") and _outcome(r) is not None]
    out: dict[str, Any] = {"decision_count": len(recs), "trade_count": len(trades),
                           "with_outcome": len(rets)}
    if rets:
        wins = sum(1 for x in rets if x > 0)
        mae = [as_float(r.get("max_adverse_excursion")) for r in recs if r.get("max_adverse_excursion") is not None]
        mfe = [as_float(r.get("max_favorable_excursion")) for r in recs if r.get("max_favorable_excursion") is not None]
        out.update({
            "win_rate": round(wins / len(rets), 3),
            "avg_return": round(sum(rets) / len(rets), 4),
            "median_return": round(median(rets), 4),
            "expectancy": round(sum(rets) / len(rets), 4),
            "avg_MAE": round(sum(mae) / len(mae), 4) if mae else None,
            "avg_MFE": round(sum(mfe) / len(mfe), 4) if mfe else None,
            "max_drawdown": round(min(rets), 4),
        })
    return out


def _tertile_label(values: list[float], v: float) -> str:
    s = sorted(values)
    if len(s) < 3:
        return "all"
    lo, hi = s[len(s) // 3], s[2 * len(s) // 3]
    return "low" if v <= lo else ("high" if v >= hi else "mid")


def _fixed_total_score_bucket(total_score: float) -> str:
    for label, low, high in FIXED_TOTAL_SCORE_BUCKETS:
        if total_score >= low and (high is None or total_score < high):
            return label
    return "out_of_range"


def load_records(date: str | None) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    pat = f"decision_scores_{date}.jsonl" if date else "decision_scores_*.jsonl"
    for p in sorted(SCORE_DIR.glob(pat)):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    recs.append(r)
            except Exception:
                continue
    return recs


def load_shadow_config() -> dict[str, Any] | None:
    if not SHADOW_CONFIG.exists():
        return None
    try:
        value = json.loads(SHADOW_CONFIG.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def shadow_forward_stats(records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    candidate = cfg.get("candidate", {})
    validation = cfg.get("forwardValidation", {})
    threshold = as_float(candidate.get("totalScoreMin"), 71.0)
    effective = str(validation.get("effectiveFrom") or "9999-12-31")
    cost = as_float(cfg.get("diagnosticEvidence", {}).get("roundTripCost"), 0.0014)
    buys = [
        row for row in records
        if str(row.get("decision_type")) == "BUY"
        and str(row.get("date")) >= effective
        and row.get("realized_return") is not None
    ]
    high = [as_float(row.get("realized_return")) - cost for row in buys
            if as_float(row.get("total_score")) >= threshold]
    below = [as_float(row.get("realized_return")) - cost for row in buys
             if as_float(row.get("total_score")) < threshold]
    days = len({str(row.get("date")) for row in buys})
    high_mean = sum(high) / len(high) if high else None
    below_mean = sum(below) / len(below) if below else None
    spread = high_mean - below_mean if high_mean is not None and below_mean is not None else None
    enough = (days >= int(validation.get("minimumTradingDays", 20))
              and len(high) >= int(validation.get("minimumDirectionalOutcomes", 30)))
    passed = bool(enough and high_mean is not None and high_mean > 0
                  and spread is not None and spread > 0)
    return {
        "threshold": threshold,
        "effective_from": effective,
        "days": days,
        "high_count": len(high),
        "below_count": len(below),
        "high_mean_net": high_mean,
        "below_mean_net": below_mean,
        "high_minus_below": spread,
        "sample_ready": enough,
        "shadow_pass": passed,
        "promotion_allowed": False,
    }


def probability_forward_stats(records: list[dict[str, Any]],
                              model: dict[str, Any]) -> dict[str, Any]:
    effective = str(model.get("effectiveFrom") or "9999-12-31")
    rows = [row for row in records
            if str(row.get("date")) >= effective
            and str(row.get("decision_type")) == "BUY"
            and row.get("posterior_prob") is not None
            and row.get("probability_outcome") is not None]
    probabilities = [as_float(row.get("posterior_prob"), 0.5) for row in rows]
    outcomes = [int(row.get("probability_outcome")) for row in rows]
    metrics = dp.probability_metrics(probabilities, outcomes)
    prior = as_float(model.get("prior", {}).get("probability"), 0.5)
    baseline = dp.probability_metrics([prior] * len(rows), outcomes)
    neutral = dp.probability_metrics([0.5] * len(rows), outcomes)
    days = len({str(row.get("date")) for row in rows})
    validation = model.get("forwardValidation", {})
    ready = (days >= int(validation.get("minimumIndependentTradingDays", 20))
             and len(rows) >= int(validation.get("minimumBuyOutcomes", 50)))
    improved = bool(
        ready and metrics.get("brier") is not None and baseline.get("brier") is not None
        and metrics["brier"] < baseline["brier"]
        and metrics["log_loss"] < baseline["log_loss"]
    )
    return {"effective_from": effective, "days": days, "rows": len(rows),
            "metrics": metrics, "constant_prior": baseline, "neutral_half": neutral,
            "sample_ready": ready,
            "beats_prior_brier_and_logloss": improved, "promotion_allowed": False}


def build_report(records: list[dict[str, Any]]) -> str:
    n = len(records)
    with_outcome = sum(1 for r in records if _outcome(r) is not None)
    lines = ["# Decision-Score Effectiveness Report", ""]
    outcome_days = len({str(r.get("date")) for r in records if _outcome(r) is not None})
    lines.append(f"Decisions: {n} | with outcome: {with_outcome} | outcome days: {outcome_days} | "
                 f"generated: {datetime.now().astimezone().isoformat()}")
    lines.append("")
    lines.append("Diagnostic only. NOT alpha, NOT an auto-trading signal, must not drive live sizing.")
    versions = {
        key: sorted({str(row.get(key)) for row in records if row.get(key)})
        for key in ("iteration_id", "scorer_version", "weights_version", "outcome_model_version",
                    "pipeline_version", "calibration_version", "bayesian_model_version")
    }
    if records:
        lines.append("Version provenance: " + "; ".join(
            f"{key}={','.join(value) if value else 'missing'}" for key, value in versions.items()
        ))
        if any(len(value) > 1 for value in versions.values()):
            lines.append("**mixed_version_warning**: report contains more than one semantic version.")
    lines.append("")
    insufficient = (with_outcome < MIN_OUTCOMES_FOR_CONCLUSION or
                    outcome_days < MIN_OUTCOME_DAYS_FOR_CONCLUSION)
    if insufficient:
        lines.append(f"**sample_insufficient**: {with_outcome} directional decisions across "
                     f"{outcome_days} trading days have outcomes (need >= {MIN_OUTCOMES_FOR_CONCLUSION} "
                     f"decisions and >= {MIN_OUTCOME_DAYS_FOR_CONCLUSION} days). No definitive conclusion is drawn below.")
    lines.append("")

    # 1) by fixed total-score ranges requested for forward discrimination checks.
    lines.append("## 1. By fixed total-score range")
    lines.append("| total_score range | decisions | outcomes | trades | win_rate | avg_ret | median | expectancy | avg_MAE | avg_MFE | max_dd |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, _, _ in FIXED_TOTAL_SCORE_BUCKETS:
        group = [r for r in records if _fixed_total_score_bucket(as_float(r.get("total_score"))) == label]
        g = _grp_stats(group)
        lines.append(f"| {label} | {g['decision_count']} | {g['with_outcome']} | {g['trade_count']} | "
                     f"{g.get('win_rate','-')} | {g.get('avg_return','-')} | "
                     f"{g.get('median_return','-')} | {g.get('expectancy','-')} | "
                     f"{g.get('avg_MAE','-')} | {g.get('avg_MFE','-')} | {g.get('max_drawdown','-')} |")
    lines.append("")

    # 2) by legacy semantic score bucket
    lines.append("## 2. By legacy total-score bucket")
    lines.append("| bucket | decisions | trades | win_rate | avg_ret | median | expectancy | avg_MAE | avg_MFE | max_dd |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in ("A", "B", "C", "D", "E"):
        g = _grp_stats([r for r in records if r.get("score_bucket") == b])
        lines.append(f"| {b} | {g['decision_count']} | {g['trade_count']} | {g.get('win_rate','-')} | "
                     f"{g.get('avg_return','-')} | {g.get('median_return','-')} | {g.get('expectancy','-')} | "
                     f"{g.get('avg_MAE','-')} | {g.get('avg_MFE','-')} | {g.get('max_drawdown','-')} |")
    lines.append("")

    # 3) by sub-score tertile
    lines.append("## 3. By sub-score group (high/mid/low tertiles)")
    predictive: dict[str, float | None] = {}
    for sc in SUBSCORES:
        vals = [as_float(r.get(sc)) for r in records if r.get(sc) is not None]
        lines.append(f"### {sc}")
        lines.append("| group | decisions | win_rate | avg_ret |")
        lines.append("|---|---:|---:|---:|")
        avg_by_grp: dict[str, float] = {}
        for grp in ("high", "mid", "low", "all"):
            sub = [r for r in records if r.get(sc) is not None and _tertile_label(vals, as_float(r.get(sc))) == grp]
            if not sub:
                continue
            g = _grp_stats(sub)
            if g.get("avg_return") is not None:
                avg_by_grp[grp] = g["avg_return"]
            lines.append(f"| {grp} | {g['decision_count']} | {g.get('win_rate','-')} | {g.get('avg_return','-')} |")
        # predictive power proxy = avg_ret(high) - avg_ret(low)
        if "high" in avg_by_grp and "low" in avg_by_grp:
            predictive[sc] = round(avg_by_grp["high"] - avg_by_grp["low"], 4)
        lines.append("")

    # 4) mistake attribution
    lines.append("## 4. Mistake attribution")
    counts: dict[str, int] = {}
    for r in records:
        mt = r.get("mistake_type") or (classify_mistake(r) if _outcome(r) is not None else "UNKNOWN")
        counts[mt] = counts.get(mt, 0) + 1
    for k in sorted(counts, key=lambda x: -counts[x]):
        lines.append(f"- {k}: {counts[k]}")
    lines.append("")

    # 5) conclusions
    lines.append("## 5. Scoring-logic conclusions")
    if insufficient:
        lines.append("- sample_insufficient: outcomes too few for predictive claims.")
        lines.append("- Action: keep recording decisions + outcomes until the sample grows.")
    else:
        ranked = sorted((k for k, v in predictive.items() if v is not None),
                        key=lambda k: predictive[k], reverse=True)
        if ranked:
            lines.append(f"- Most predictive sub-score (high-minus-low avg return): "
                         f"{ranked[0]} ({predictive[ranked[0]]:+}).")
            lines.append(f"- Least useful sub-score: {ranked[-1]} ({predictive[ranked[-1]]:+}).")
        a = _grp_stats([r for r in records if r.get("score_bucket") in ("A", "B")])
        e = _grp_stats([r for r in records if r.get("score_bucket") in ("D", "E")])
        if a.get("avg_return") is not None and e.get("avg_return") is not None:
            verdict = "YES" if a["avg_return"] > e["avg_return"] else "NO"
            lines.append(f"- Do high-score (A/B) decisions out-return low (D/E)? **{verdict}** "
                         f"(A/B avg {a['avg_return']} vs D/E avg {e['avg_return']}).")
        lines.append("- Next: down-weight sub-scores with ~0 high-minus-low spread; investigate "
                     "buckets where high score but negative expectancy (scoring-logic error).")
    lines.append("")

    shadow_cfg = load_shadow_config()
    if shadow_cfg:
        shadow = shadow_forward_stats(records, shadow_cfg)
        show = lambda value: "-" if value is None else f"{value:.4%}"
        lines.extend([
            "## 6. Preregistered forward shadow: frozen score >= 71",
            "",
            f"- effective_from: {shadow['effective_from']}",
            f"- forward BUY outcome days: {shadow['days']}",
            f"- high-score outcomes: {shadow['high_count']}; net mean: {show(shadow['high_mean_net'])}",
            f"- below-threshold outcomes: {shadow['below_count']}; net mean: {show(shadow['below_mean_net'])}",
            f"- high-minus-below: {show(shadow['high_minus_below'])}",
            f"- sample_ready: `{shadow['sample_ready']}`",
            f"- shadow_pass: `{shadow['shadow_pass']}`",
            "- trade_gate_enabled: `false`; promotion_allowed: `false`",
            "",
        ])
    probability_model = dp.load_shadow_model()
    if probability_model:
        probability = probability_forward_stats(records, probability_model)
        metrics = probability["metrics"]
        baseline = probability["constant_prior"]
        neutral = probability["neutral_half"]
        show = lambda value: "-" if value is None else f"{value:.6f}"
        lines.extend([
            "## 7. Preregistered probability calibration shadow",
            "",
            f"- effective_from: {probability['effective_from']}",
            f"- forward BUY outcomes: {probability['rows']} across {probability['days']} independent days",
            f"- posterior Brier / LogLoss / AUC / ECE: {show(metrics['brier'])} / "
            f"{show(metrics['log_loss'])} / {show(metrics['auc'])} / {show(metrics['ece'])}",
            f"- constant-prior Brier / LogLoss: {show(baseline['brier'])} / {show(baseline['log_loss'])}",
            f"- neutral-50% Brier / LogLoss: {show(neutral['brier'])} / {show(neutral['log_loss'])}",
            f"- sample_ready: `{probability['sample_ready']}`",
            f"- beats_prior_brier_and_logloss: `{probability['beats_prior_brier_and_logloss']}`",
            "- trade_gate_enabled: `false`; position_sizing_enabled: `false`; promotion_allowed: `false`",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Decision-score effectiveness report (offline).")
    ap.add_argument("--date", default=None, help="YYYYMMDD; default = all available days")
    args = ap.parse_args()
    records = load_records(args.date)
    report = build_report(records)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = args.date or datetime.now().strftime("%Y%m%d")
    out = SCORE_DIR / f"score_effectiveness_report_{stamp}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
