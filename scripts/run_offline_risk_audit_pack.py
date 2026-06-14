"""Run the offline paper-trading robustness audit pack.

This convenience wrapper runs:
1. T0 missingness stress replay
2. Post-selection Sharpe audit
3. Portfolio CVaR risk audit

All child scripts are offline diagnostics. No broker API calls or order
submissions are made.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, write_json


OUT_DIR = ROOT / "outputs" / "offline_risk_audit_pack"


def run_child(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return {
        "command": [sys.executable, *args],
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline ETF paper-trading risk audit pack")
    parser.add_argument("--label", default="latest")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD filter for missingness replay")
    args = parser.parse_args()

    commands: list[list[str]] = [
        ["scripts/replay_t0_missingness_stress.py", "--label", args.label],
        ["scripts/run_post_selection_sharpe_audit.py", "--label", args.label],
        ["scripts/run_portfolio_cvar_risk_audit.py", "--label", args.label],
    ]
    if args.date:
        commands[0].extend(["--date", args.date])

    results = [run_child(cmd) for cmd in commands]
    summary = {
        "label": args.label,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_recommendation": False,
        "api_calls_made": False,
        "order_submit_calls_made": False,
        "all_ok": all(r["ok"] for r in results),
        "results": results,
    }
    out_json = OUT_DIR / f"{args.label}_offline_risk_audit_pack.json"
    write_json(out_json, summary)
    print(json.dumps({
        "status": "written",
        "summary": str(out_json),
        "all_ok": summary["all_ok"],
        "children": [{"ok": r["ok"], "returncode": r["returncode"], "command": r["command"]} for r in results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
