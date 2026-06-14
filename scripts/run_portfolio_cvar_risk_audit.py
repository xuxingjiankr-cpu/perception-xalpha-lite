"""Offline CVaR risk audit for ETF paper-trading replay outputs.

This script estimates portfolio tail risk from existing replay summaries. It is
diagnostic-only: no market data is fetched, no broker API is called, and no
orders are generated. The output is intended to inform risk review, not to make
live trading decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float, write_json


OUT_DIR = ROOT / "outputs" / "portfolio_cvar_risk_audit"
DEFAULT_SUMMARY_GLOB = "outputs/t0_replay/*_summary.json"


def load_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def extract_daily_returns(summary: dict[str, Any]) -> list[dict[str, Any]]:
    per_day = summary.get("per_day")
    if not isinstance(per_day, dict):
        return []
    total_pnl = as_float(summary.get("total_pnl"), 0.0)
    final_assets = as_float(summary.get("final_assets"), 0.0)
    initial_assets = final_assets - total_pnl if final_assets > 0 else 1_000_000.0
    if initial_assets <= 0:
        initial_assets = 1_000_000.0
    rows: list[dict[str, Any]] = []
    for day, vals in sorted(per_day.items()):
        if not isinstance(vals, dict):
            continue
        pnl = as_float(vals.get("pnl"), 0.0)
        rows.append({
            "date": str(day),
            "pnl": pnl,
            "return": pnl / initial_assets,
        })
    return rows


def quantile_nearest(values: list[float], q: float) -> float | None:
    vals = sorted(x for x in values if math.isfinite(x))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def cvar_from_returns(returns: list[float], confidence: float) -> dict[str, Any]:
    losses = sorted([-r for r in returns if math.isfinite(r)], reverse=True)
    if not losses:
        return {"available": False, "confidence": confidence}
    tail_q = 1.0 - confidence
    tail_count = max(1, math.ceil(len(losses) * tail_q))
    tail_losses = losses[:tail_count]
    var_loss = quantile_nearest(losses, confidence)
    cvar_loss = sum(tail_losses) / len(tail_losses)
    return {
        "available": True,
        "confidence": confidence,
        "observation_count": len(losses),
        "tail_count": tail_count,
        "var_loss_pct": var_loss,
        "cvar_loss_pct": cvar_loss,
        "worst_loss_pct": losses[0],
    }


def select_summary(summary_glob: str, preferred: str | None) -> Path | None:
    paths = sorted(ROOT.glob(summary_glob), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if preferred:
        preferred_path = Path(preferred)
        if not preferred_path.is_absolute():
            preferred_path = ROOT / preferred_path
        return preferred_path if preferred_path.exists() else None
    return paths[0] if paths else None


def build_audit(summary_path: Path) -> dict[str, Any]:
    summary = load_json_safe(summary_path) or {}
    rows = extract_daily_returns(summary)
    returns = [as_float(r.get("return")) for r in rows]
    cvar_90 = cvar_from_returns(returns, 0.90)
    cvar_95 = cvar_from_returns(returns, 0.95)
    sample_warning = len(returns) < 20
    negative_days = sum(1 for r in returns if r < 0)
    positive_days = sum(1 for r in returns if r > 0)
    total_return = sum(returns)
    return {
        "source_summary": str(summary_path),
        "source_label": summary.get("label") or summary_path.stem.replace("_summary", ""),
        "observation_count": len(returns),
        "sample_insufficient_for_stable_cvar": sample_warning,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "zero_days": len(returns) - positive_days - negative_days,
        "total_return_approx": total_return,
        "worst_day": min(rows, key=lambda x: as_float(x.get("return")), default=None),
        "best_day": max(rows, key=lambda x: as_float(x.get("return")), default=None),
        "cvar_90": cvar_90,
        "cvar_95": cvar_95,
        "daily_returns": rows,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    audit = report.get("audit", {})
    c90 = audit.get("cvar_90", {})
    c95 = audit.get("cvar_95", {})
    lines = [
        "# Portfolio CVaR Risk Audit",
        "",
        "Research-only paper trading diagnostic. No API calls, no order submissions, and no live-ready claim.",
        "",
        f"- Source summary: {audit.get('source_summary')}",
        f"- Source label: {audit.get('source_label')}",
        f"- Observations: {audit.get('observation_count')}",
        f"- Sample insufficient for stable CVaR: {audit.get('sample_insufficient_for_stable_cvar')}",
        f"- Positive days: {audit.get('positive_days')}",
        f"- Negative days: {audit.get('negative_days')}",
        "",
        "## Tail Risk",
        f"- 90% VaR loss: {as_float(c90.get('var_loss_pct')):.4%}",
        f"- 90% CVaR loss: {as_float(c90.get('cvar_loss_pct')):.4%}",
        f"- 95% VaR loss: {as_float(c95.get('var_loss_pct')):.4%}",
        f"- 95% CVaR loss: {as_float(c95.get('cvar_loss_pct')):.4%}",
        "",
        "## Interpretation",
        "- CVaR is estimated from replay daily returns and is not a guarantee of future loss.",
        "- With fewer than 20 observations, treat the number as a stress diagnostic only.",
        "- This report can justify tighter position sizing, but it does not approve live trading.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CVaR risk audit for paper-trading replay outputs")
    parser.add_argument("--summary", default=None, help="Specific replay summary JSON")
    parser.add_argument("--summary-glob", default=DEFAULT_SUMMARY_GLOB)
    parser.add_argument("--label", default="latest")
    args = parser.parse_args()

    summary_path = select_summary(args.summary_glob, args.summary)
    if summary_path is None:
        raise SystemExit(f"no summary found for {args.summary_glob}")
    audit = build_audit(summary_path)
    report = {
        "label": args.label,
        "source": "arxiv_2606_13618_cvar_risk_budget",
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "api_calls_made": False,
        "order_submit_calls_made": False,
        "audit": audit,
    }
    out_json = OUT_DIR / f"{args.label}_portfolio_cvar_risk_audit.json"
    out_md = OUT_DIR / f"{args.label}_portfolio_cvar_risk_audit.md"
    write_json(out_json, report)
    write_markdown(out_md, report)
    print(json.dumps({
        "status": "written",
        "summary": str(out_json),
        "markdown": str(out_md),
        "source_summary": str(summary_path),
        "observations": audit.get("observation_count"),
        "sample_insufficient": audit.get("sample_insufficient_for_stable_cvar"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
