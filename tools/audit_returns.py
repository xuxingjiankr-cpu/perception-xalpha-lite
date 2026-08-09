"""Audit somebody else's backtest for overfitting, in their own fork.

Fork this repository, replace ``audit/returns.csv`` with your own daily returns, say in
``audit/audit.json`` how many variants you actually tried, and push. GitHub Actions runs the
audit in your fork and writes the verdict to the run summary. Your returns stay in your fork;
nothing is uploaded anywhere.

Three questions get asked, and they are not the same question:

*Was the winner picked by luck?* Combinatorially symmetric cross-validation splits the sample
many ways, picks the best variant on each in-sample half, and checks where it lands out of
sample. If the winner is below median more often than not, selection is doing the work.

*Is the best Sharpe big enough to survive the search that found it?* The deflated Sharpe ratio
discounts an observed Sharpe by the best one you would expect from the same number of trials
on pure noise. The trial count is the number of variants you *tried*, including the ones you
deleted — not the number in this file. Understating it is the most common way to pass.

*Does any variant beat zero once the whole family is accounted for?* White's Reality Check
tests the family-wide null with a stationary bootstrap that preserves serial dependence.

The tool has no opinion about your strategy and cannot see it. It only reports whether the
evidence supports the conclusion you drew.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xalpha_lite.discovery import deflated_sharpe_ratio, pbo  # noqa: E402
from xalpha_lite.evidence import BootstrapDesign, white_reality_check  # noqa: E402

AUDIT = ROOT / "audit"
RETURNS, CONFIG = AUDIT / "returns.csv", AUDIT / "audit.json"


def load() -> tuple[pd.DataFrame, dict]:
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    frame = pd.read_csv(RETURNS)
    date_column = next((c for c in frame.columns if c.lower() in ("date", "dt", "day")), None)
    if date_column:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.empty:
        raise SystemExit("audit/returns.csv has no numeric return columns")
    if float(np.nanmax(np.abs(numeric.to_numpy()))) > 1.5:
        # A column of +2.3 is percent, not a fraction, and would inflate every statistic.
        print("note: values exceed 1.5, reading them as percent and dividing by 100")
        numeric = numeric / 100.0
    return numeric, config


def verdict(pbo_value, dsr_probability, reality_p) -> tuple[str, str]:
    if pbo_value is not None and pbo_value > 0.5:
        return ("SELECTION IS DOING THE WORK",
                "The in-sample winner lands below median out of sample more often than not. "
                "That is the signature of a choice, not of an edge.")
    if dsr_probability is not None and dsr_probability < 0.95:
        return ("CONSISTENT WITH LUCK",
                "The best Sharpe here is not large enough to stand out from the best you would "
                "expect from this many trials on noise.")
    if reality_p is not None and reality_p > 0.10:
        return ("FAMILY-WIDE NULL NOT REJECTED",
                "Taking the whole set of variants into account, no member beats zero at a "
                "conventional level.")
    if pbo_value is None and dsr_probability is None:
        return ("NOT ENOUGH TO TEST", "Too few observations or variants to run the audit.")
    return ("SURVIVES THESE TESTS",
            "Passing is not proof. It means these particular ways of being wrong have been "
            "ruled out on this sample. Only forward data settles it.")


def main() -> int:
    returns, config = load()
    declared = int(config.get("declared_trials", returns.shape[1]))
    blocks = int(config.get("cscv_blocks", 8))

    if declared <= returns.shape[1]:
        print(f"WARNING: declared_trials={declared} but {returns.shape[1]} variants are in the "
              f"file. Every variant you tried and discarded still counts. Understating this is "
              f"the single most common way a backtest passes a test it should fail.")

    sharpes = [float(returns[c].mean() / returns[c].std(ddof=1))
               for c in returns.columns if returns[c].std(ddof=1) > 0]
    best = returns.mean().div(returns.std(ddof=1).replace(0.0, np.nan)).idxmax()

    pbo_result = pbo(returns, blocks=blocks)
    dsr = deflated_sharpe_ratio(returns[best], sharpes, declared)
    reality = (white_reality_check(returns, BootstrapDesign())
               if returns.shape[1] >= 2 and len(returns) >= 60 else {})
    reality_p = reality.get("p_value")

    headline, explanation = verdict(pbo_result.get("pbo"), dsr.get("probability"), reality_p)
    report = {
        "observations": int(len(returns)),
        "variants_submitted": int(returns.shape[1]),
        "declared_trials": declared,
        "best_variant": str(best),
        "best_sharpe_annualised": dsr.get("observed_sharpe_ann"),
        "expected_best_from_noise_annualised": dsr.get("expected_max_trial_sharpe_ann"),
        "deflated_sharpe_probability": dsr.get("probability"),
        "pbo": pbo_result.get("pbo"),
        "cscv_splits": pbo_result.get("splits"),
        "reality_check_p_value": reality_p,
        "verdict": headline,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    rows = [
        ("Observations", report["observations"]),
        ("Variants in file", report["variants_submitted"]),
        ("Trials declared", declared),
        ("Best variant", f"`{best}`"),
        ("Its Sharpe (annualised)", report["best_sharpe_annualised"]),
        ("Best Sharpe expected from noise at this trial count", report["expected_best_from_noise_annualised"]),
        ("Deflated Sharpe probability", report["deflated_sharpe_probability"]),
        ("Probability of backtest overfitting (CSCV)", report["pbo"]),
        ("Reality Check p-value", reality_p),
    ]
    summary = [
        f"## {headline}", "", explanation, "",
        "| | |", "|---|--:|",
        *[f"| {label} | {value if value is not None else '—'} |" for label, value in rows],
        "",
        "> Deflated Sharpe probability is the chance this Sharpe is real given the number of",
        "> trials; below 0.95 is weak. PBO above 0.5 means the in-sample winner is usually a",
        "> below-median performer out of sample. Neither can be fixed by trying harder on the",
        "> same data — only by data that did not exist when the rule was written.",
    ]
    if declared <= returns.shape[1]:
        summary += ["", "**Trial count looks understated.** Variants you tried and threw away "
                    "still count toward the search. Raise `declared_trials` in `audit/audit.json` "
                    "and rerun; if the verdict changes, the earlier one was the wrong answer."]

    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        Path(path).write_text("\n".join(summary) + "\n", encoding="utf-8")
    (AUDIT / "verdict.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
