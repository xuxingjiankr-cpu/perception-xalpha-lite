"""Offline post-selection Sharpe audit for paper-trading research outputs.

The audit is inspired by post-selection Sharpe estimation: when many variants
are replayed and the best observed Sharpe is selected, the selected Sharpe is
usually optimistic. This script computes a simple James-Stein shrinkage estimate
across replay candidates. It is research-only and never calls broker APIs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float, write_json


OUT_DIR = ROOT / "outputs" / "post_selection_sharpe_audit"
DEFAULT_INPUT_GLOBS = [
    "outputs/t0_replay/*_summary.json",
]


def load_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def annualized_sharpe(returns: list[float], periods_per_year: int = 252) -> float | None:
    vals = [float(x) for x in returns if math.isfinite(float(x))]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var) * math.sqrt(periods_per_year)


def extract_daily_returns(summary: dict[str, Any]) -> list[float]:
    per_day = summary.get("per_day")
    if not isinstance(per_day, dict):
        return []
    total_pnl = as_float(summary.get("total_pnl"), 0.0)
    final_assets = as_float(summary.get("final_assets"), 0.0)
    initial_assets = final_assets - total_pnl if final_assets > 0 else 1_000_000.0
    if initial_assets <= 0:
        initial_assets = 1_000_000.0
    returns: list[float] = []
    for _, day in sorted(per_day.items()):
        if not isinstance(day, dict):
            continue
        pnl = as_float(day.get("pnl"), 0.0)
        returns.append(pnl / initial_assets)
    return returns


def discover_candidates(input_globs: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in input_globs:
        paths.extend(ROOT.glob(pattern))
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        summary = load_json_safe(path)
        if not summary:
            continue
        returns = extract_daily_returns(summary)
        sharpe = annualized_sharpe(returns)
        if sharpe is None:
            continue
        candidates.append({
            "name": str(summary.get("label") or path.stem.replace("_summary", "")),
            "path": str(path),
            "observations": len(returns),
            "returns": returns,
            "observed_sharpe": sharpe,
            "total_pnl": as_float(summary.get("total_pnl"), 0.0),
            "final_assets": as_float(summary.get("final_assets"), 0.0),
        })
    return candidates


def james_stein_shrinkage(candidates: list[dict[str, Any]], periods_per_year: int = 252) -> dict[str, Any]:
    k = len(candidates)
    if k == 0:
        return {"available": False, "reason": "no_candidates"}
    daily_snr = [as_float(c.get("observed_sharpe")) / math.sqrt(periods_per_year) for c in candidates]
    grand_mean = sum(daily_snr) / k
    n_eff = int(median([int(c.get("observations", 0)) for c in candidates if int(c.get("observations", 0)) > 0] or [0]))
    spread_sq = sum((x - grand_mean) ** 2 for x in daily_snr)
    if k <= 2 or n_eff <= 1 or spread_sq <= 0:
        shrink_factor = 1.0
        status = "insufficient_cross_section_for_shrinkage"
    else:
        shrink_factor = max(0.0, 1.0 - ((k - 2) / n_eff) / spread_sq)
        status = "james_stein_positive_part"

    audited: list[dict[str, Any]] = []
    for cand, snr in zip(candidates, daily_snr):
        shrunk_snr = grand_mean + shrink_factor * (snr - grand_mean)
        shrunk_sharpe = shrunk_snr * math.sqrt(periods_per_year)
        observed = as_float(cand.get("observed_sharpe"))
        audited.append({
            "name": cand["name"],
            "path": cand["path"],
            "observations": cand["observations"],
            "observed_sharpe": observed,
            "james_stein_shrunk_sharpe": shrunk_sharpe,
            "selection_haircut": observed - shrunk_sharpe,
            "total_pnl": cand["total_pnl"],
            "final_assets": cand["final_assets"],
        })
    audited.sort(key=lambda x: x["observed_sharpe"], reverse=True)
    selected = audited[0] if audited else None
    return {
        "available": True,
        "status": status,
        "candidate_count": k,
        "n_eff": n_eff,
        "grand_mean_daily_snr": grand_mean,
        "shrink_factor": shrink_factor,
        "selected_by_observed_sharpe": selected,
        "candidates": audited,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report.get("selected_by_observed_sharpe") or {}
    lines = [
        "# Post-Selection Sharpe Audit",
        "",
        "Research-only paper trading diagnostic. No broker API calls, no order submissions, and no live-ready claim.",
        "",
        "## Summary",
        f"- Candidate count: {report.get('candidate_count', 0)}",
        f"- Effective observations: {report.get('n_eff', 0)}",
        f"- Shrinkage method: {report.get('status')}",
        f"- James-Stein shrink factor: {as_float(report.get('shrink_factor')):.4f}",
        "",
        "## Selected Candidate",
    ]
    if selected:
        lines.extend([
            f"- Name: {selected.get('name')}",
            f"- Observed Sharpe: {as_float(selected.get('observed_sharpe')):.4f}",
            f"- Shrunk Sharpe: {as_float(selected.get('james_stein_shrunk_sharpe')):.4f}",
            f"- Selection haircut: {as_float(selected.get('selection_haircut')):.4f}",
        ])
    else:
        lines.append("- unavailable")
    lines.extend([
        "",
        "## Candidates",
        "| name | obs | observed Sharpe | shrunk Sharpe | haircut |",
        "|---|---:|---:|---:|---:|",
    ])
    for c in report.get("candidates", []):
        lines.append(
            f"| {c.get('name')} | {c.get('observations')} | "
            f"{as_float(c.get('observed_sharpe')):.4f} | "
            f"{as_float(c.get('james_stein_shrunk_sharpe')):.4f} | "
            f"{as_float(c.get('selection_haircut')):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline post-selection Sharpe audit")
    parser.add_argument("--input-glob", action="append", default=None, help="Glob under repo root; can repeat")
    parser.add_argument("--label", default="latest")
    args = parser.parse_args()

    input_globs = args.input_glob or DEFAULT_INPUT_GLOBS
    candidates = discover_candidates(input_globs)
    audit = james_stein_shrinkage(candidates)
    report = {
        "label": args.label,
        "source": "arxiv_2606_01650_post_selection_sharpe",
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "api_calls_made": False,
        "order_submit_calls_made": False,
        "input_globs": input_globs,
        **audit,
    }
    out_json = OUT_DIR / f"{args.label}_post_selection_sharpe_audit.json"
    out_md = OUT_DIR / f"{args.label}_post_selection_sharpe_audit.md"
    write_json(out_json, report)
    write_markdown(out_md, report)
    print(json.dumps({
        "status": "written",
        "summary": str(out_json),
        "markdown": str(out_md),
        "candidate_count": report.get("candidate_count", 0),
        "selected": (report.get("selected_by_observed_sharpe") or {}).get("name"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
