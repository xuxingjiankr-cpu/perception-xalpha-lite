"""Daily paper-trading self-review.

This script reads local logs only. It does not call trading APIs, does not
submit orders, and does not modify strategy parameters. Recommendations are
diagnostic and require human approval before any code/config change.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def today_cn() -> str:
    return datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_date_row(row: dict[str, Any], date: str) -> bool:
    for key in ("timestamp", "trade_date", "submitTime", "submit_time"):
        value = str(row.get(key) or "")
        if value.startswith(date):
            return True
    return False


def quota_1002_in_obj(obj: Any) -> bool:
    if isinstance(obj, dict):
        raw = obj.get("raw")
        if isinstance(raw, dict) and str(raw.get("code")) == "1002":
            return True
        message = str(obj.get("message") or "")
        if "配额" in message or "quota" in message.lower():
            return True
        return any(quota_1002_in_obj(v) for v in obj.values())
    if isinstance(obj, list):
        return any(quota_1002_in_obj(x) for x in obj)
    return False


def count_risk_failures(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for row in rows:
        checks = row.get("risk_checks")
        if checks is None:
            checks = (row.get("risk_report") or {}).get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, dict) and not check.get("passed"):
                out[str(check.get("name"))] += 1
    return dict(out.most_common())


def summarize(date: str) -> dict[str, Any]:
    t0_runs = [x for x in read_jsonl(ROOT / "outputs/t0_intraday_agent/t0_agent_runs.jsonl") if is_date_row(x, date)]
    lf_runs = [x for x in read_jsonl(ROOT / "outputs/paper_trading_agent/agent_runs.jsonl") if is_date_row(x, date)]
    t0_blotter = [x for x in read_csv(ROOT / "outputs/t0_intraday_agent/t0_order_blotter.csv") if is_date_row(x, date)]
    lf_blotter = [x for x in read_csv(ROOT / "outputs/paper_trading_agent/order_blotter.csv") if is_date_row(x, date)]

    t0_status = Counter(str(x.get("status")) for x in t0_runs)
    lf_status = Counter(str(x.get("status")) for x in lf_runs)
    t0_reasons = Counter(str((x.get("state_machine") or {}).get("reason")) for x in t0_runs)
    lf_reasons = Counter(str(x.get("reason")) for x in lf_runs)

    t0_submitted = [x for x in t0_blotter if str(x.get("event_type")) == "submitted" or str(x.get("submit_ok")).lower() == "true"]
    lf_submitted = [x for x in lf_blotter if str(x.get("event_type")) == "submitted" or str(x.get("submit_ok")).lower() == "true"]

    t0_quote_attempts = 0
    t0_quote_ok = 0
    for run in t0_runs:
        for q in run.get("quotes") or []:
            t0_quote_attempts += 1
            if isinstance(q, dict) and q.get("quote_ok"):
                t0_quote_ok += 1

    quota_events = sum(1 for x in [*t0_runs, *lf_runs] if quota_1002_in_obj(x))
    recommendations: list[dict[str, Any]] = []
    if quota_events:
        recommendations.append({
            "id": "quota_backoff",
            "severity": "high",
            "finding": "API quota exhaustion was observed.",
            "recommendation": "Keep quota_exhausted_backoff enabled; avoid additional polling after first 1002 error.",
            "auto_apply": False,
            "requires_human_approval": True,
        })
    if not t0_submitted:
        recommendations.append({
            "id": "t0_sample_insufficient",
            "severity": "medium",
            "finding": "T0 agent submitted no orders today.",
            "recommendation": "Do not tune T0 entry/exit thresholds from today's data; collect more valid, non-quota-limited samples.",
            "auto_apply": False,
            "requires_human_approval": True,
        })
    if lf_submitted and quota_events:
        recommendations.append({
            "id": "fill_reconciliation_next_day",
            "severity": "medium",
            "finding": "Orders were submitted before quota exhaustion, but final fill/account state could not be confirmed.",
            "recommendation": "Run fill reconciliation before the next trading session and include pending/filled status in the next report.",
            "auto_apply": False,
            "requires_human_approval": True,
        })

    return {
        "date": date,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "self_iteration_policy": {
            "enabled": True,
            "mode": "diagnostic_report_only",
            "auto_apply_changes": False,
            "requires_human_approval": True,
        },
        "summary": {
            "t0_runs": len(t0_runs),
            "lf_runs": len(lf_runs),
            "t0_status": dict(t0_status),
            "lf_status": dict(lf_status),
            "t0_top_reasons": dict(t0_reasons.most_common(10)),
            "lf_top_reasons": dict(lf_reasons.most_common(10)),
            "t0_submitted_orders": len(t0_submitted),
            "lf_submitted_orders": len(lf_submitted),
            "t0_quote_attempts": t0_quote_attempts,
            "t0_quote_ok": t0_quote_ok,
            "quota_1002_runs": quota_events,
        },
        "risk_failures": {
            "t0": count_risk_failures(t0_runs),
            "low_frequency": count_risk_failures(lf_runs),
        },
        "recommendations": recommendations,
    }


def write_markdown(path: Path, review: dict[str, Any]) -> None:
    summary = review["summary"]
    lines = [
        f"# Trading Self-Review {review['date']}",
        "",
        "This is paper trading diagnostics only. No recommendations are auto-applied.",
        "",
        "## Summary",
        "",
        f"- T0 runs: {summary['t0_runs']}",
        f"- Low-frequency runs: {summary['lf_runs']}",
        f"- T0 submitted orders: {summary['t0_submitted_orders']}",
        f"- Low-frequency submitted orders: {summary['lf_submitted_orders']}",
        f"- T0 quote ok / attempts: {summary['t0_quote_ok']} / {summary['t0_quote_attempts']}",
        f"- Quota 1002 affected runs: {summary['quota_1002_runs']}",
        "",
        "## Recommendations",
        "",
    ]
    for rec in review.get("recommendations", []):
        lines.extend([
            f"### {rec['id']} ({rec['severity']})",
            "",
            f"- Finding: {rec['finding']}",
            f"- Recommendation: {rec['recommendation']}",
            f"- Auto apply: {rec['auto_apply']}",
            f"- Requires human approval: {rec['requires_human_approval']}",
            "",
        ])
    if not review.get("recommendations"):
        lines.append("No actionable recommendation from today's logs.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily paper-trading self-review")
    parser.add_argument("--date", default=today_cn())
    args = parser.parse_args()
    out_dir = ROOT / "outputs" / "trading_self_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    review = summarize(args.date)
    json_path = out_dir / f"{args.date.replace('-', '')}_self_review.json"
    md_path = out_dir / f"{args.date.replace('-', '')}_self_review.md"
    json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_markdown(md_path, review)
    print(json.dumps({
        "status": "written",
        "date": args.date,
        "json": str(json_path),
        "markdown": str(md_path),
        "auto_apply_changes": False,
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
