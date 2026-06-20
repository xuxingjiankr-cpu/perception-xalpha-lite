"""Post-close L4 prospective shadow replay and fail-closed verdict pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from overfitting_guard import combinatorial_symmetric_pbo
from run_t0_strategy_evolution import deflated_sharpe_diagnostic, diebold_mariano_hln


ROOT = Path(__file__).resolve().parents[1]
QUOTE_DIR = ROOT / "outputs" / "t0_intraday_agent"
BASELINE = ROOT / "configs" / "shadow" / "l4_forward_baseline.json"
CANDIDATE = ROOT / "configs" / "shadow" / "l4_forward_candidate.json"
LEDGER = ROOT / "outputs" / "edge_research" / "l4_forward_validation.jsonl"
VERDICT_JSON = ROOT / "outputs" / "edge_research" / "l4_forward_verdict.json"
VERDICT_MD = ROOT / "outputs" / "edge_research" / "l4_forward_verdict.md"
PROSPECTIVE_AFTER = "2026-06-18"
EXPECTED_BASELINE_SHA256 = "11aa712f279d9cd5bf319b2cd0ef658cbc97ceae152a5fc8815ae903db5c9c34"
EXPECTED_CANDIDATE_SHA256 = "22b3848c106ca74749fef808d69b67af1d1fbabc68371f7042d8e121309b31c1"
MIN_DAYS = 20
MIN_STD_REDUCTION_PCT = 10.0
MIN_WORST_DAY_IMPROVEMENT = 0.0
MAX_PBO = 0.25
DATE_RE = re.compile(r"minute_quotes_(\d{4}-\d{2}-\d{2})\.jsonl$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(content)
        temp = Path(handle.name)
    temp.replace(path)


def verify_frozen_configs() -> None:
    actual = (sha256(BASELINE), sha256(CANDIDATE))
    expected = (EXPECTED_BASELINE_SHA256, EXPECTED_CANDIDATE_SHA256)
    if actual != expected:
        raise RuntimeError(f"preregistered config hash mismatch: actual={actual} expected={expected}")


def latest_prospective_quote() -> tuple[str, Path] | None:
    found = []
    for path in QUOTE_DIR.glob("minute_quotes_*.jsonl"):
        match = DATE_RE.search(path.name)
        if match and match.group(1) > PROSPECTIVE_AFTER:
            found.append((match.group(1), path))
    return max(found) if found else None


def quote_file_closed(path: Path, trade_date: str) -> bool:
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stamp = str(row.get("timestamp") or "")
            if stamp.startswith(trade_date):
                last = max(last, stamp)
    return bool(last and last[11:16] >= "14:55")


def run_replay(config: Path, quotes: Path, trade_date: str, role: str) -> Path:
    label = f"l4_forward_{trade_date}_{role}"
    summary = ROOT / "outputs" / "t0_replay" / f"{label}_summary.json"
    command = [sys.executable, str(ROOT / "scripts" / "replay_t0_decisions.py"),
               "--config", str(config), "--quotes", str(quotes), "--date", trade_date,
               "--label", label, "--output-detail", "summary"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1200, check=False)
    if completed.returncode != 0 or not summary.exists():
        raise RuntimeError(f"{role} replay failed: {completed.stderr[-1500:]} {completed.stdout[-1500:]}")
    return summary


def day_metrics(summary: dict[str, Any], trade_date: str) -> dict[str, Any]:
    node = summary.get("per_day", {}).get(trade_date, {})
    trades = [trade for trade in summary.get("trades", []) if trade.get("trade_date") == trade_date]
    pnls = [float(trade.get("pnl", 0.0)) for trade in trades]
    gross = float(node.get("gross_pnl", node.get("pnl", 0.0)))
    buy, sell = float(node.get("buy_notional", 0.0)), float(node.get("sell_notional", 0.0))
    return {
        "total_pnl": round(gross, 2),
        "trade_pnl_std": round(statistics.stdev(pnls), 2) if len(pnls) >= 2 else 0.0,
        "worst_trade_pnl": round(min(pnls), 2) if pnls else 0.0,
        "trades": len(pnls), "winning_trades": sum(value > 0 for value in pnls),
        "losing_trades": sum(value < 0 for value in pnls),
        "day_won": gross > 0, "buy_notional": round(buy, 2), "sell_notional": round(sell, 2),
        "net_12bps": round(gross - 0.0006 * (buy + sell), 2),
    }


def read_ledger(path: Path = LEDGER) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(rows, key=lambda row: row["tradeDate"])


def write_ledger(rows: list[dict[str, Any]], path: Path = LEDGER) -> None:
    content = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    atomic_bytes(path, content)


def upsert_day(row: dict[str, Any], path: Path = LEDGER) -> bool:
    rows = read_ledger(path)
    old = next((item for item in rows if item["tradeDate"] == row["tradeDate"]), None)
    def stable(value: dict[str, Any] | None) -> dict[str, Any] | None:
        return {key: item for key, item in value.items() if key != "recordedAt"} if value else value
    if stable(old) == stable(row):
        return False
    rows = [item for item in rows if item["tradeDate"] != row["tradeDate"]] + [row]
    write_ledger(sorted(rows, key=lambda item: item["tradeDate"]), path)
    return True


def series_metrics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"total": 0.0, "mean": None, "std": None, "worst": None, "sharpe": None}
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    return {"total": round(sum(values), 2), "mean": round(statistics.fmean(values), 2),
            "std": round(sd, 2), "worst": round(min(values), 2),
            "sharpe": round(statistics.fmean(values) / sd * math.sqrt(252), 4) if sd > 0 else None}


def build_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row["tradeDate"] > PROSPECTIVE_AFTER]
    n = len(rows)
    base = [float(row["baseline"]["net_12bps"]) for row in rows]
    cand = [float(row["candidate"]["net_12bps"]) for row in rows]
    base_metrics, cand_metrics = series_metrics(base), series_metrics(cand)
    report: dict[str, Any] = {
        "schemaVersion": "l4_forward_verdict_v1", "status": "diagnostic_only",
        "hypothesisId": "l4_selectivity_forward_v1", "forwardDays": n,
        "minimumForwardDays": MIN_DAYS, "prospectiveAfter": PROSPECTIVE_AFTER,
        "liveReady": False, "formalStrategyAllowed": False,
        "baseline": base_metrics, "candidate": cand_metrics,
        "decision": "insufficient_forward_days" if n < MIN_DAYS else "not_evaluated",
        "recommendLive": False,
    }
    if n < MIN_DAYS:
        report["daysRemaining"] = MIN_DAYS - n
        return report
    std_reduction = (1 - cand_metrics["std"] / base_metrics["std"]) * 100 if base_metrics["std"] else 0.0
    worst_improvement = cand_metrics["worst"] - base_metrics["worst"]
    base_days = {row["tradeDate"]: row["baseline"]["net_12bps"] for row in rows}
    cand_days = {row["tradeDate"]: row["candidate"]["net_12bps"] for row in rows}
    dm = diebold_mariano_hln(base_days, cand_days, alpha=0.05)
    dsr = deflated_sharpe_diagnostic(base_days, cand_days, n_trials=1, alpha=0.10)
    pbo = combinatorial_symmetric_pbo([base, cand], n_blocks=8)
    gates = {
        "stdReduction": std_reduction >= MIN_STD_REDUCTION_PCT,
        "worstDayImprovement": worst_improvement > MIN_WORST_DAY_IMPROVEMENT,
        "sharpeNotLower": (cand_metrics["sharpe"] or -1e99) >= (base_metrics["sharpe"] or -1e99),
        "dm": bool(dm.get("significant")), "dsr": bool(dsr.get("significant")),
        "pbo": pbo.get("pbo") is not None and float(pbo["pbo"]) <= MAX_PBO,
    }
    passed = all(gates.values())
    report.update({"status": "forward_validation_complete", "decision": "l4_forward_pass" if passed else "l4_forward_fail",
                   "stdReductionPct": round(std_reduction, 2), "worstDayImprovement": round(worst_improvement, 2),
                   "dm": dm, "dsr": dsr, "pbo": pbo, "gates": gates,
                   "recommendLive": False,
                   "note": "A pass permits a separate review proposal only; this pipeline never changes live trading."})
    return report


def render_verdict(report: dict[str, Any]) -> str:
    lines = ["# L4 Forward Shadow Verdict", "", f"Status: `{report['decision']}`",
             f"Forward days: {report['forwardDays']}/{report['minimumForwardDays']}", "",
             "Pure shadow. No live config, overlay or order path is modified."]
    if report["forwardDays"] < report["minimumForwardDays"]:
        lines += ["", f"Insufficient sample: {report['daysRemaining']} additional post-2026-06-18 trading days required.",
                  "No conclusion and no recommendation to deploy L4."]
    else:
        lines += ["", f"- Daily std reduction: {report['stdReductionPct']}%",
                  f"- Worst-day improvement: {report['worstDayImprovement']}",
                  f"- Gates: `{report['gates']}`", "", "No automatic deployment is permitted."]
    return "\n".join(lines) + "\n"


def publish_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = build_verdict(rows)
    atomic_bytes(VERDICT_JSON, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    atomic_bytes(VERDICT_MD, render_verdict(report).encode("utf-8"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="")
    parser.add_argument("--quotes", default="")
    args = parser.parse_args()
    verify_frozen_configs()
    if not LEDGER.exists():
        write_ledger([])
    if args.quotes:
        quotes = Path(args.quotes)
        trade_date = args.date or (DATE_RE.search(quotes.name).group(1) if DATE_RE.search(quotes.name) else "")
    elif args.date:
        trade_date, quotes = args.date, QUOTE_DIR / f"minute_quotes_{args.date}.jsonl"
    else:
        latest = latest_prospective_quote()
        if latest is None:
            report = publish_verdict(read_ledger())
            print(render_verdict(report))
            return 0
        trade_date, quotes = latest
    if not trade_date or trade_date <= PROSPECTIVE_AFTER:
        raise ValueError(f"date must be prospective after {PROSPECTIVE_AFTER}")
    if not quotes.exists():
        report = publish_verdict(read_ledger())
        print(f"skip: quote file missing: {quotes}")
        print(render_verdict(report))
        return 0
    if not quote_file_closed(quotes, trade_date):
        raise RuntimeError(f"quote file is not closed through 14:55: {quotes}")
    summaries = {role: run_replay(config, quotes, trade_date, role)
                 for role, config in (("baseline", BASELINE), ("candidate", CANDIDATE))}
    row = {"schemaVersion": "l4_forward_day_v1", "tradeDate": trade_date,
           "recordedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
           "quoteFile": str(quotes.resolve()), "quoteSha256": sha256(quotes),
           "baselineConfigSha256": sha256(BASELINE), "candidateConfigSha256": sha256(CANDIDATE),
           "baseline": day_metrics(json.loads(summaries["baseline"].read_text(encoding="utf-8")), trade_date),
           "candidate": day_metrics(json.loads(summaries["candidate"].read_text(encoding="utf-8")), trade_date),
           "paperTradingOnly": True, "diagnosticOnly": True}
    row["deltaCandidateMinusBaseline"] = {
        "total_pnl": round(row["candidate"]["total_pnl"] - row["baseline"]["total_pnl"], 2),
        "trade_pnl_std": round(row["candidate"]["trade_pnl_std"] - row["baseline"]["trade_pnl_std"], 2),
        "worst_trade_pnl": round(row["candidate"]["worst_trade_pnl"] - row["baseline"]["worst_trade_pnl"], 2),
        "net_12bps": round(row["candidate"]["net_12bps"] - row["baseline"]["net_12bps"], 2)}
    changed = upsert_day(row)
    report = publish_verdict(read_ledger())
    print(f"ledger_changed={changed} date={trade_date}")
    print(render_verdict(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
