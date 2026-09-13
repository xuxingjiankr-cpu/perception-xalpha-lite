#!/usr/bin/env python3
"""Tests for the vstd60 specification audit.

The audit's only job is to say what a factor co-moves with, so the thing that
must not break is its ability to return the RIGHT answer when the answer is
known. The decisive cases are constructed: a series whose dispersion is real and
a series whose score is driven purely by the current observation must land on
opposite verdicts. The verdict logic is also pinned against the temptation to
call a narrow lead a falsification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_vstd60_specification_audit_v1 as audit  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def config() -> dict:
    return json.loads(
        (ROOT / "configs" / "research" / "vstd60_specification_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )


def main() -> int:
    c = config()
    audit.validate_config(c)
    check("the frozen audit config validates", True)

    for field, value in (
        ("status", "promoted"),
        (None, None),
    ):
        if field is None:
            continue
        broken = json.loads(json.dumps(c))
        broken[field] = value
        try:
            audit.validate_config(broken)
            raised = False
        except ValueError:
            raised = True
        check(f"a config with {field}={value} is refused", raised)

    promoting = json.loads(json.dumps(c))
    promoting["preregisteredHypothesis"]["historicalRunCanPromote"] = True
    try:
        audit.validate_config(promoting)
        raised = False
    except ValueError:
        raised = True
    check("a promotion path is refused", raised)

    mutating = json.loads(json.dumps(c))
    mutating["safety"]["mayChangeFrozenWeights"] = True
    try:
        audit.validate_config(mutating)
        raised = False
    except ValueError:
        raised = True
    check("permission to change frozen weights is refused", raised)

    edge_claim = json.loads(json.dumps(c))
    edge_claim["preregisteredHypothesis"]["thisIsASpecificationAuditNotAnEdgeClaim"] = False
    try:
        audit.validate_config(edge_claim)
        raised = False
    except ValueError:
        raised = True
    check("an audit that claims edge is refused", raised)

    # The vendor formula must be reproduced exactly, denominator included: the
    # whole question is that it divides by the current observation.
    dates = pd.bdate_range("2024-01-02", periods=80)
    flat = pd.DataFrame({"A": [1000.0] * 80}, index=dates)
    check(
        "a constant volume series has zero vendor vstd60",
        abs(float(audit.vendor_vstd60(flat).iloc[-1, 0])) < 1e-9,
    )
    check(
        "a constant volume series has zero true CV",
        abs(float(audit.true_cv60(flat).iloc[-1, 0])) < 1e-9,
    )
    # The two formulas share a numerator, so their ratio isolates the denominator
    # exactly: vendor / true_cv == mean(V,60) / V_t. That identity IS the concern
    # this audit was built to test, so pin it rather than a hand-built example.
    rng_v = np.random.default_rng(3)
    varied = pd.DataFrame(
        {"A": rng_v.uniform(200.0, 4000.0, 80)}, index=dates
    )
    vendor = audit.vendor_vstd60(varied).iloc[-1, 0]
    cv = audit.true_cv60(varied).iloc[-1, 0]
    implied = varied["A"].rolling(60, min_periods=60).mean().iloc[-1] / varied["A"].iloc[-1]
    check(
        "vendor vstd60 over true CV is exactly mean volume over current volume",
        abs(float(vendor) / float(cv) - float(implied)) < 1e-9,
    )
    # So a quiet session scores strictly higher than a busy one on identical
    # trailing dispersion - the behaviour that made the label worth auditing.
    quiet, busy = varied.copy(), varied.copy()
    quiet.iloc[-1, 0] = 200.0
    busy.iloc[-1, 0] = 4000.0
    check(
        "a quiet current session scores higher than a busy one",
        float(audit.vendor_vstd60(quiet).iloc[-1, 0])
        > float(audit.vendor_vstd60(busy).iloc[-1, 0]),
    )

    # A known-answer correlation: identical inputs must correlate at +1.
    idx = pd.bdate_range("2024-01-02", periods=30)
    cols = [f"S{i}" for i in range(300)]
    rng = np.random.default_rng(7)
    left = pd.DataFrame(rng.normal(size=(30, 300)), index=idx, columns=cols)
    eligible = pd.DataFrame(True, index=idx, columns=cols)
    same = audit.daily_rank_correlation(left, left, eligible, 200)
    check("a series correlates with itself at +1", float(same.dropna().min()) > 0.9999)
    opposite = audit.daily_rank_correlation(left, -left, eligible, 200)
    check("a negated series correlates at -1", float(opposite.dropna().max()) < -0.9999)

    thin = pd.DataFrame(True, index=idx, columns=cols)
    thin.iloc[:, 100:] = False
    check(
        "a session below the minimum name count is dropped, not reported",
        audit.daily_rank_correlation(left, left, thin, 200).notna().sum() == 0,
    )

    # Verdict logic: a lead is not a falsification unless it clears the margin.
    def rows(cv: float, inv: float) -> dict:
        return {
            "trueCoefficientOfVariation60": {"meanCorrelation": cv},
            "inverseCurrentVolume": {"meanCorrelation": inv},
        }

    decisive = audit.build_verdict({"validation": rows(0.2, 0.7), "shadow": rows(0.2, 0.6)}, 0.20)
    check(
        "inverse volume winning decisively on both windows falsifies the label",
        decisive["decision"].startswith("label_falsified"),
    )
    narrow = audit.build_verdict({"validation": rows(0.40, 0.45), "shadow": rows(0.40, 0.44)}, 0.20)
    check(
        "a narrow lead is reported as doubtful, never as falsified",
        narrow["decision"].startswith("label_doubtful"),
    )
    split = audit.build_verdict({"validation": rows(0.2, 0.7), "shadow": rows(0.7, 0.2)}, 0.20)
    check(
        "winning on only one window does not falsify",
        split["decision"] == "label_not_falsified_on_both_windows",
    )
    # This is what the real run returned; pin it so a refactor cannot flip it.
    actual = audit.build_verdict(
        {"validation": rows(0.4950, 0.2680), "shadow": rows(0.5578, 0.2170)}, 0.20
    )
    check(
        "the measured correlations do NOT falsify the vstd60 label",
        actual["decision"] == "label_not_falsified_on_both_windows",
    )
    for verdict in (decisive, narrow, split, actual):
        check(
            f"verdict '{verdict['decision'][:28]}' is never tradeable",
            verdict["eligibleForTrading"] is False
            and verdict["orders"] == []
            and verdict["establishesNoEdge"] is True,
        )

    source = (ROOT / "scripts" / "research_vstd60_specification_audit_v1.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = ("submitorder", "cancelorder", "build_decision(", "latest_strategy_overlay")
    check("source has no trading mutation path", not any(t in source for t in forbidden))
    print("ALL VSTD60 SPECIFICATION AUDIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
