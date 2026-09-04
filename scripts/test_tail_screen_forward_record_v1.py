#!/usr/bin/env python3
"""Integrity tests for the tail-screen fresh-forward record.

A forward record is only worth anything if it cannot be edited after the fact and
cannot score a session whose outcome was already visible when it was written.
Those two properties are what these tests protect.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import run_tail_screen_forward_record_v1 as record  # noqa: E402

from xalpha_lite.forward import append_prediction, freeze_spec, load_spec, read_log


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def synthetic_panel(days: int = 60, width: int = 40, seed: int = 20260905):
    dates = pd.bdate_range("2025-01-02", periods=days)
    symbols = [f"S{i:03d}" for i in range(width)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        10.0 * (1.0 + rng.normal(0.0, 0.01, size=(days, width))).cumprod(axis=0),
        index=dates,
        columns=symbols,
    )
    panel = {
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": pd.DataFrame(1e6, index=dates, columns=symbols),
        "eligible": pd.DataFrame(True, index=dates, columns=symbols),
    }
    return panel, dates, symbols


def main() -> int:
    spec = record.DRAFT_SPEC
    check(
        "the draft spec is permanently research-only",
        spec["status"] == "research_only_shadow_only_not_trading",
    )
    check(
        "the spec forbids orders, sizing, config changes and self-promotion",
        len(spec["cannot"]) == 4
        and any("orders" in item for item in spec["cannot"])
        and any("promote" in item for item in spec["cannot"]),
    )
    check(
        "the spec records the volatility confound rather than hiding it",
        "most volatile" in spec["known_confound"],
    )
    check(
        "the acceptance bar matches the historical gate",
        spec["acceptance"]["minimum_day_clustered_t"] >= 2.0
        and spec["acceptance"]["minimum_excess_severe_loss_rate"] >= 0.01
        and spec["acceptance"]["minimum_resolved_sessions"] >= 60,
    )

    scores = pd.Series(
        np.linspace(1.0, 0.0, 20), index=[f"S{i:03d}" for i in range(20)]
    )
    eligible = pd.Series(True, index=scores.index)
    best, worst = record.bucket_members(scores, eligible, 10)
    check("the best bucket holds the top-scored names", best == ["S000", "S001"])
    check("the worst bucket holds the bottom-scored names", worst == ["S018", "S019"])
    ineligible = eligible.copy()
    ineligible["S019"] = False
    _, worst_filtered = record.bucket_members(scores, ineligible, 10)
    check(
        "an ineligible name cannot enter the recorded slice",
        "S019" not in worst_filtered,
    )
    empty_best, empty_worst = record.bucket_members(
        scores, pd.Series(False, index=scores.index), 10
    )
    check(
        "no eligible names records nothing rather than guessing",
        empty_best == [] and empty_worst == [],
    )

    universe = pd.Series(
        [-0.05, -0.04, 0.02, 0.03, 0.01, -0.01],
        index=["A", "B", "C", "D", "E", "F"],
    )
    measured = record.session_excess(universe, ["A", "B"], -0.03)
    check("a recorded slice of pure losers reports rate 1.0", measured[0] == 1.0)
    check(
        "the universe rate is measured on the same session",
        abs(measured[1] - 2.0 / 6.0) < 1e-12,
    )
    shifted = universe + 0.10
    shifted_measure = record.session_excess(shifted, ["A", "B"], -0.03)
    check(
        "a market-wide move leaves the return excess unchanged",
        abs(shifted_measure[2] - measured[2]) < 1e-12,
    )
    check(
        "names absent from the session are dropped, not counted as safe",
        record.session_excess(universe, ["A", "ZZZ"], -0.03)[0] == 1.0,
    )
    check(
        "a slice with no surviving name reports nothing",
        record.session_excess(universe, ["ZZZ"], -0.03) is None,
    )

    acceptance = spec["acceptance"]
    strong = [
        {"resolvedSessions": 80, "excessSevereLossRate": 0.05, "dayClusteredT": 4.0},
        {"resolvedSessions": 80, "excessSevereLossRate": 0.04, "dayClusteredT": 3.0},
    ]
    check(
        "two strong horizons on enough fresh sessions confirm",
        record.decide(strong, acceptance) == "forward_record_confirms_tail_ranking",
    )
    thin = [
        {"resolvedSessions": 10, "excessSevereLossRate": 0.20, "dayClusteredT": 9.0},
        {"resolvedSessions": 10, "excessSevereLossRate": 0.20, "dayClusteredT": 9.0},
    ]
    check(
        "a huge effect on too few sessions still withholds a verdict",
        record.decide(thin, acceptance) == "insufficient_fresh_sessions_no_verdict_yet",
    )
    one_sided = [
        {"resolvedSessions": 80, "excessSevereLossRate": 0.05, "dayClusteredT": 4.0},
        {"resolvedSessions": 80, "excessSevereLossRate": 0.04, "dayClusteredT": 1.1},
    ]
    check(
        "one horizon failing the t bar rejects the whole record",
        record.decide(one_sided, acceptance) == "forward_record_rejects_tail_ranking",
    )
    check(
        "no results at all is never a pass",
        record.decide([], acceptance) == "insufficient_fresh_sessions_no_verdict_yet",
    )

    workspace = Path(tempfile.mkdtemp(prefix="tail_forward_test_", dir=ROOT / "outputs"))
    try:
        spec_path = workspace / "spec.json"
        frozen = freeze_spec(dict(spec), spec_path)
        check("freezing stamps a digest", bool(frozen["spec_sha256"]))
        try:
            freeze_spec(dict(spec), spec_path)
            overwrote = True
        except FileExistsError:
            overwrote = False
        check("a frozen spec can never be overwritten", not overwrote)
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        payload["acceptance"]["minimum_day_clustered_t"] = 0.5
        spec_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_spec(spec_path)
            tampering_accepted = True
        except Exception:
            tampering_accepted = False
        check("a tampered spec refuses to load", not tampering_accepted)

        log_path = workspace / "predictions.jsonl"
        entry = {
            "data_as_of": "2026-09-04",
            "spec_sha256": frozen["spec_sha256"],
            "worst_bucket": ["S001"],
        }
        check("the first record appends", append_prediction(entry, log_path) is True)
        check(
            "the same session is never recorded twice",
            append_prediction(dict(entry), log_path) is False,
        )
        check("the log holds exactly one entry", len(read_log(log_path)) == 1)

        panel, dates, symbols = synthetic_panel()
        config = {"data": {"maximumExitDelayTradingDays": 5}}
        # Regression: execution_eligible requires t+1 to exist, so it is empty on the
        # newest session. A recorder filtering on it would log nothing, every day.
        _o, execution_eligible, _d = precision.executable_horizon_return(panel, 1, 5)
        check(
            "executable eligibility is empty on the newest session",
            int(execution_eligible.iloc[-1].astype(bool).sum()) == 0,
        )
        newest_best, newest_worst = record.bucket_members(
            pd.Series(np.linspace(1.0, 0.0, len(symbols)), index=symbols),
            panel["eligible"].iloc[-1],
            10,
        )
        check(
            "point-in-time eligibility still records the newest session",
            len(newest_worst) > 0 and len(newest_best) > 0,
        )
        resolvable = precision.contained_signal_dates(dates, 1, 5)
        unresolved_date = dates[-1].date().isoformat()
        check(
            "the last session in the panel is not yet resolvable",
            unresolved_date not in {d.date().isoformat() for d in resolvable},
        )
        entries = [
            {"data_as_of": unresolved_date, "worst_bucket": symbols[:4]},
            {"data_as_of": resolvable[0].date().isoformat(), "worst_bucket": symbols[:4]},
        ]
        scored = record.score_entries(entries, panel, config, 1, -0.03)
        check(
            "an unresolved session is held pending, never scored",
            scored["pendingSessions"] >= 1,
        )
        check(
            "only resolved sessions reach the scorecard",
            scored["resolvedSessions"] <= 1,
        )
        all_pending = record.score_entries(
            [{"data_as_of": unresolved_date, "worst_bucket": symbols[:4]}],
            panel,
            config,
            1,
            -0.03,
        )
        check(
            "a log of nothing but unresolved sessions scores nothing",
            all_pending["resolvedSessions"] == 0
            and all_pending["excessSevereLossRate"] is None,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    source = (
        ROOT / "scripts" / "run_tail_screen_forward_record_v1.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = ("submitorder", "cancelorder", "build_decision(", "latest_strategy_overlay")
    check(
        "source has no trading mutation path",
        not any(term in source for term in forbidden),
    )
    print("ALL TAIL SCREEN FORWARD RECORD TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
