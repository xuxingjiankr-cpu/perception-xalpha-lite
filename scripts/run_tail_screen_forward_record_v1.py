#!/usr/bin/env python3
"""Fresh-forward record for the tail-loss exclusion screen (RESEARCH_LOG #13).

The historical sweep found a strong, monotone, out-of-sample-stable ranking of
severe-loss risk: worst decile +12.5pp excess at t=19.7 on validation and +10.1pp
at t=11.8 on shadow, at holding horizons of one and five sessions.  Every one of
those windows had already been viewed, so that result can justify exactly one
thing - a preregistered record scored on data that did not exist when the rule
was written.

This is that record.  It writes the rule once, appends one immutable prediction
per signal date, and scores only sessions whose outcome window has fully
resolved.  It creates no orders, sizes nothing and cannot promote itself.

    py -3.13 scripts/run_tail_screen_forward_record_v1.py freeze
    py -3.13 scripts/run_tail_screen_forward_record_v1.py log
    py -3.13 scripts/run_tail_screen_forward_record_v1.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
VENDOR_ROOT = ROOT / "scripts" / "vendor" / "vibe_factors"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_twelve_factor_guarded_online_weights_v1 as guarded  # noqa: E402
import research_horizon_cost_frontier_v1 as frontier  # noqa: E402
import research_tail_exclusion_screen_v1 as screen  # noqa: E402
import panel_cache  # noqa: E402

from xalpha_lite.forward import append_prediction, freeze_spec, load_spec, read_log

# The specification and the prediction log live under docs/ because they are TRACKED:
# a frozen, tamper-evident record whose only copy sits in a gitignored directory has
# no independent timestamp, and the digest it carries could be recomputed by whoever
# edited it. The scorecard is derived and stays in outputs/.
RECORD_DIR = ROOT / "docs" / "forward_records"
OUT_DIR = ROOT / "outputs" / "forward_record"
SPEC_PATH = RECORD_DIR / "tail_exclusion_screen_v1.spec.json"
LOG_PATH = RECORD_DIR / "tail_exclusion_screen_v1.predictions.jsonl"
CODE_VERSION = "tail_screen_forward_record_v1_20260905"

DRAFT_SPEC: dict[str, Any] = {
    "name": "tail_exclusion_screen_v1",
    "status": "research_only_shadow_only_not_trading",
    "rationale": (
        "RESEARCH_LOG #13 found the frozen twelve-factor book ranks severe-loss risk "
        "monotonically across all ten score deciles and stably out of sample, while "
        "#12 found it has no day-neutral selection edge at any horizon. Every "
        "historical window behind #13 has been viewed, so this record exists to "
        "settle the tail claim on sessions that had not happened when it was frozen."
    ),
    "book": "frozen_prior",
    "bucket_count": 10,
    "horizons": [1, 5],
    "severe_loss_threshold": -0.03,
    "benchmark": "same_day_eligible_universe",
    "claim": (
        "The worst score decile carries a severe-loss rate at least one percentage "
        "point above the same-day eligible universe, with a day-clustered t of at "
        "least two, at holding horizons of one and five sessions."
    ),
    "acceptance": {
        "minimum_excess_severe_loss_rate": 0.01,
        "minimum_day_clustered_t": 2.0,
        "minimum_resolved_sessions": 60,
        "both_horizons_must_agree": True,
    },
    "known_confound": (
        "The worst decile is also the most volatile slice; on the historical shadow "
        "window it carried the HIGHEST excess return (+6.7 bps at h=1, +32.9 at h=5) "
        "while being negative on validation. This record therefore also tracks the "
        "excess return of the excluded slice, and a tail result that arrives together "
        "with a positive return excess is a risk trade, not an edge."
    ),
    "cannot": [
        "create or modify orders",
        "size or hold any position",
        "alter trading configuration, risk gates or strategy overlays",
        "promote itself on any historical window",
    ],
}


def build_scores() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Panel, per-horizon outcomes and the frozen-book score, all from cache when warm."""
    config = screen.load_json(screen.DEFAULT_CONFIG)
    screen.validate_config(config)
    frozen, source, _sha = guarded.load_frozen_config(
        {"basePrecisionConfig": config["basePrecisionConfig"]}
    )
    base = screen.load_json(ROOT / frozen["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel_key, _ = panel_cache.cache_key(base, cog_config)
    panel, _audit = panel_cache.build_configured_panel_cached(base, cog_config)
    ranks, _static, _factor_audit = panel_cache.build_rank_book_cached(
        panel, frozen, panel_key
    )
    factors = list(ranks.keys())
    prior = pd.Series(
        {item["factorKey"]: float(item["weight"]) for item in frozen["frozenFactors"]}
    ).reindex(factors).astype(float)
    prior /= prior.sum()
    weights = frontier.weight_frame(panel["close"].index, factors, prior.to_dict())
    score = guarded.adaptive_score(ranks, weights, panel)
    return score, panel, config, source


def bucket_members(
    score_row: pd.Series, eligible_row: pd.Series, buckets: int
) -> tuple[list[str], list[str]]:
    """Names in the best and worst score buckets on one session."""
    usable = score_row.where(eligible_row.astype(bool)).dropna()
    if usable.empty:
        return [], []
    pct = usable.rank(pct=True, ascending=False)
    best = sorted(pct[pct.le(1.0 / buckets)].index.astype(str))
    worst = sorted(pct[pct.gt((buckets - 1) / buckets)].index.astype(str))
    return best, worst


def cmd_freeze(_args: argparse.Namespace) -> int:
    frozen = freeze_spec(DRAFT_SPEC, SPEC_PATH)
    print(
        json.dumps(
            {k: frozen[k] for k in ("name", "frozen_at", "spec_sha256")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_log(_args: argparse.Namespace) -> int:
    spec = load_spec(SPEC_PATH)
    score, panel, config, _source = build_scores()
    # Point-in-time eligibility, NOT executable_horizon_return's execution_eligible:
    # that one requires t+1 to exist, so it is False for every name on the newest
    # session - which is precisely the session a forward record has to record.
    # Scoring re-derives executable eligibility later, when the future does exist.
    eligible = panel["eligible"]
    signal_date = score.dropna(how="all").index.max()
    best, worst = bucket_members(
        score.loc[signal_date],
        eligible.loc[signal_date],
        int(spec["bucket_count"]),
    )
    eligible_count = int(eligible.loc[signal_date].astype(bool).sum())
    entry = {
        "schema": "tail_exclusion_forward_prediction_v1",
        "status": spec["status"],
        "spec_sha256": spec["spec_sha256"],
        "code_version": CODE_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "data_as_of": signal_date.date().isoformat(),
        "eligible_names": eligible_count,
        "bucket_count": int(spec["bucket_count"]),
        "worst_bucket": worst,
        "best_bucket": best,
        "orders": [],
        "automatic_trading_changes": [],
    }
    appended = append_prediction(entry, LOG_PATH)
    print(
        json.dumps(
            {
                "data_as_of": entry["data_as_of"],
                "appended": appended,
                "eligible_names": eligible_count,
                "worst_bucket_names": len(worst),
                "best_bucket_names": len(best),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def session_excess(
    universe: pd.Series, names: list[str], severe: float
) -> tuple[float, float, float] | None:
    """One session's tail and return excess for a recorded slice, against that day.

    Both figures are differences taken inside the same session, so a market-wide move
    cannot enter either of them.
    """
    universe = universe.dropna()
    if universe.empty:
        return None
    selected = universe.reindex([n for n in names if n in universe.index]).dropna()
    if selected.empty:
        return None
    return (
        float(selected.le(severe).mean()),
        float(universe.le(severe).mean()),
        float(selected.mean() - universe.mean()),
    )


def decide(results: list[dict[str, Any]], acceptance: dict[str, Any]) -> str:
    """No verdict until every horizon has enough fresh sessions; then both must pass."""
    minimum_sessions = int(acceptance["minimum_resolved_sessions"])
    minimum_excess = float(acceptance["minimum_excess_severe_loss_rate"])
    minimum_t = float(acceptance["minimum_day_clustered_t"])
    if not results:
        return "insufficient_fresh_sessions_no_verdict_yet"
    if any(item["resolvedSessions"] < minimum_sessions for item in results):
        return "insufficient_fresh_sessions_no_verdict_yet"
    passed = all(
        item.get("excessSevereLossRate") is not None
        and item.get("dayClusteredT") is not None
        and item["excessSevereLossRate"] >= minimum_excess
        and item["dayClusteredT"] >= minimum_t
        for item in results
    )
    return (
        "forward_record_confirms_tail_ranking"
        if passed
        else "forward_record_rejects_tail_ranking"
    )


def score_entries(
    entries: list[dict[str, Any]],
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
    horizon: int,
    severe: float,
) -> dict[str, Any]:
    """Score only sessions whose full outcome window has already resolved."""
    max_delay = int(config["data"]["maximumExitDelayTradingDays"])
    outcome, execution_eligible, _delay = precision.executable_horizon_return(
        panel, horizon, max_delay
    )
    sessions = panel["close"].index
    resolvable = precision.contained_signal_dates(sessions, horizon, max_delay)
    resolved_set = {stamp.date().isoformat() for stamp in resolvable}

    worst_daily: list[float] = []
    universe_daily: list[float] = []
    return_excess_daily: list[float] = []
    scored_dates: list[str] = []
    pending = 0
    for entry in entries:
        as_of = entry.get("data_as_of")
        if as_of not in resolved_set:
            pending += 1
            continue
        stamp = pd.Timestamp(as_of)
        if stamp not in outcome.index:
            pending += 1
            continue
        row = outcome.loc[stamp]
        eligible_row = execution_eligible.loc[stamp].astype(bool)
        universe = row.where(eligible_row).dropna()
        if universe.empty:
            continue
        measured = session_excess(universe, entry.get("worst_bucket", []), severe)
        if measured is None:
            continue
        worst_rate, universe_rate, return_excess = measured
        worst_daily.append(worst_rate)
        universe_daily.append(universe_rate)
        return_excess_daily.append(return_excess)
        scored_dates.append(as_of)

    if not worst_daily:
        return {
            "horizon": horizon,
            "resolvedSessions": 0,
            "pendingSessions": pending,
            "excessSevereLossRate": None,
            "dayClusteredT": None,
            "excessReturnBps": None,
        }
    excess = pd.Series(np.array(worst_daily) - np.array(universe_daily))
    return {
        "horizon": horizon,
        "resolvedSessions": len(worst_daily),
        "pendingSessions": pending,
        "firstScoredSession": scored_dates[0],
        "lastScoredSession": scored_dates[-1],
        "worstBucketSevereLossRate": float(np.mean(worst_daily)),
        "universeSevereLossRate": float(np.mean(universe_daily)),
        "excessSevereLossRate": float(excess.mean()),
        "dayClusteredT": frontier.day_clustered_t(excess),
        "excessReturnBps": float(np.mean(return_excess_daily) * 1e4),
    }


def cmd_score(_args: argparse.Namespace) -> int:
    spec = load_spec(SPEC_PATH)
    entries = [
        entry
        for entry in read_log(LOG_PATH)
        if entry.get("spec_sha256") == spec["spec_sha256"]
    ]
    _score, panel, config, _source = build_scores()
    severe = float(spec["severe_loss_threshold"])
    acceptance = spec["acceptance"]
    horizons = [int(h) for h in spec["horizons"]]
    results = [
        score_entries(entries, panel, config, horizon, severe) for horizon in horizons
    ]

    decision = decide(results, acceptance)

    report = {
        "schema": "tail_exclusion_forward_scorecard_v1",
        "status": spec["status"],
        "specSha256": spec["spec_sha256"],
        "codeVersion": CODE_VERSION,
        "scoredAt": datetime.now(timezone.utc).isoformat(),
        "loggedSessions": len(entries),
        "acceptance": acceptance,
        "horizons": results,
        "decision": decision,
        "eligibleForTrading": False,
        "knownConfound": spec["known_confound"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{spec['name']}.scorecard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    sub.add_parser("log").set_defaults(func=cmd_log)
    sub.add_parser("score").set_defaults(func=cmd_score)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
