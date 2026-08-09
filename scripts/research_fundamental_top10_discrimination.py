"""Evaluate a frozen PIT fundamental score as an exact daily Top10 selector.

This module answers the decision question directly: are the ten names selected at close
more likely to rise, and do they earn more after executable T+1 timing and 30 bps cost,
than every other eligible stock and a board/liquidity-matched control?

Research/shadow-only.  Historical outcomes can reject but never refit or promote V2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_fundamental_mechanism_families as mechanism  # noqa: E402


SCHEMA_VERSION = "fundamental_top10_discrimination_result_v2"
CODE_VERSION = "fundamental_top10_discrimination_v2_20260809"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "fundamental_top10_discrimination_v2.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "fundamental_top10_discrimination_v2":
        raise ValueError("unexpected Top10 discrimination schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("Top10 study must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every mutation and trading permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("outputStatus must remain diagnostic_only")
    objective = config["objective"]
    if int(objective["topCount"]) != 10:
        raise ValueError("the decision set must remain exactly Top10")
    if not objective.get("neverReplaceUnexecutableSelections"):
        raise ValueError("future executability may not replace a selected name")
    if objective.get("historicalOutcomesMayChangeScore"):
        raise ValueError("historical outcomes cannot change the frozen score")
    if objective.get("historicalRunCanPromote"):
        raise ValueError("historical results cannot promote")
    execution = config["execution"]
    if int(execution["holdingTradingDays"]) != 1:
        raise ValueError("V2 is the fixed one-session executable target")
    if not math.isclose(float(execution["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("round-trip cost must remain 30 bps")
    gates = config["preregisteredAcceptanceGates"]
    if int(gates["minimumIndependentForwardTradingDays"]) < 60:
        raise ValueError("at least 60 independent forward days are required")
    if not gates.get("allGatesMustPass"):
        raise ValueError("all preregistered gates must pass")
    if float(gates["minimumHacTForProbabilityLift"]) < 2.0:
        raise ValueError("probability inference gate may not be weakened")
    if float(gates["minimumHacTForReturnLift"]) < 2.0:
        raise ValueError("return inference gate may not be weakened")
    calibration = config["calibration"]
    if calibration.get("forwardOutcomesMayRefitFrozenVersion"):
        raise ValueError("forward outcomes cannot refit V2")
    if config["forward"].get("allowParameterChangesInPlace"):
        raise ValueError("forward parameters are immutable")
    if config["forward"].get("allowAutomaticPromotion"):
        raise ValueError("automatic promotion is forbidden")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("outputs must remain isolated under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must always be empty")


def verify_frozen_file(config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(config[path_key])
    actual = file_sha256(path)
    expected = str(config[hash_key]).lower()
    if actual != expected:
        raise ValueError(f"frozen dependency changed: {path_key} {actual} != {expected}")
    return path


def build_frozen_score(
    config: dict[str, Any]
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
    dict[str, Any],
]:
    mechanism_path = verify_frozen_file(
        config, "frozenMechanismConfig", "frozenMechanismConfigFileSha256"
    )
    verify_frozen_file(config, "frozenMechanismCode", "frozenMechanismCodeFileSha256")
    mechanism_config = load_json(mechanism_path)
    mechanism.validate_config(mechanism_config)
    panel, panel_audit = mechanism.load_panel(mechanism_config)
    close = panel["close"]
    eligible = panel["eligible"].fillna(False)
    event_table, event_audit = mechanism.build_event_table(
        close.index, close.columns, mechanism_config
    )
    max_age = int(mechanism_config["fundamentals"]["maximumSignalAgeTradingDays"])
    bins = int(mechanism_config["fundamentals"]["liquidityNeutraliseBins"])
    family_scores: dict[str, pd.DataFrame] = {}
    for family, definition in mechanism_config["families"].items():
        print(f"Top10 score family {family}", flush=True)
        ranks: dict[str, pd.DataFrame] = {}
        for candidate in definition["candidates"]:
            identifier = str(candidate["id"])
            raw = mechanism.factor_frame(
                event_table, identifier, close.index, close.columns, max_age
            )
            neutral = autonomous.size_neutralise(raw.where(eligible), panel, bins)
            ranks[identifier] = neutral.where(eligible).rank(
                axis=1, pct=True, method="average"
            ).astype("float32")
        family_score, common = mechanism.equal_rank_composite(ranks, eligible)
        family_scores[family] = family_score.where(common).rank(
            axis=1, pct=True, method="average"
        ).astype("float32")
        del ranks
    primary, common = mechanism.equal_rank_composite(family_scores, eligible)
    primary = primary.where(common).rank(
        axis=1, pct=True, method="average"
    ).astype("float32")
    return panel, primary, family_scores, panel_audit, event_audit


def select_fixed_top10(
    score: pd.DataFrame, eligible: pd.DataFrame, top_count: int = 10
) -> pd.DataFrame:
    """Select before the outcome; never replace a future unexecutable name."""
    ranks = score.where(eligible).rank(axis=1, ascending=False, method="first")
    return ranks.le(top_count).fillna(False)


def board_map(config: dict[str, Any]) -> pd.Series:
    mechanism_config = load_json(ROOT / str(config["frozenMechanismConfig"]))
    base = load_json(ROOT / str(mechanism_config["baseResearchConfig"]))
    rows = mechanism.ashare.read_jsonl(
        ROOT / str(base["assetUniverse"]["masterPath"])
    )
    return pd.Series(
        {
            str(row.get("securityId")): str(row.get("board") or "unknown")
            for row in rows
            if row.get("securityId")
        },
        dtype="object",
    )


def trailing_liquidity_deciles(
    panel: dict[str, pd.DataFrame], lookback: int, deciles: int
) -> pd.DataFrame:
    trailing = panel["amount"].rolling(
        lookback, min_periods=max(5, lookback // 3)
    ).median().shift(1)
    percentile = trailing.rank(axis=1, pct=True)
    return np.ceil(percentile * deciles).clip(1, deciles).astype("float32")


def matched_control(
    date: pd.Timestamp,
    selected_resolved: pd.Series,
    eligible_rest: pd.Series,
    returns: pd.Series,
    boards: pd.Series,
    liquidity: pd.Series,
    cost: float,
) -> dict[str, float | int | None]:
    selected_ids = selected_resolved.index[selected_resolved]
    if not len(selected_ids):
        return {"count": 0, "meanNet": None, "grossUpProbability": None, "netPositiveProbability": None}
    weights: list[int] = []
    means: list[float] = []
    gross_probabilities: list[float] = []
    net_probabilities: list[float] = []
    total_control = 0
    selected_groups: dict[tuple[str, int], int] = {}
    for security_id in selected_ids:
        bucket = liquidity.get(security_id)
        if pd.isna(bucket):
            continue
        key = (str(boards.get(security_id, "unknown")), int(bucket))
        selected_groups[key] = selected_groups.get(key, 0) + 1
    for (board, bucket), selected_count in selected_groups.items():
        mask = (
            eligible_rest
            & boards.reindex(eligible_rest.index).eq(board)
            & liquidity.reindex(eligible_rest.index).eq(float(bucket))
            & returns.notna()
        )
        values = returns[mask]
        if not len(values):
            continue
        weights.append(selected_count)
        means.append(float((values - cost).mean()))
        gross_probabilities.append(float(values.gt(0.0).mean()))
        net_probabilities.append(float(values.gt(cost).mean()))
        total_control += len(values)
    if not weights:
        return {"count": 0, "meanNet": None, "grossUpProbability": None, "netPositiveProbability": None}
    return {
        "count": total_control,
        "meanNet": float(np.average(means, weights=weights)),
        "grossUpProbability": float(np.average(gross_probabilities, weights=weights)),
        "netPositiveProbability": float(np.average(net_probabilities, weights=weights)),
    }


def evaluate_daily_top10(
    panel: dict[str, pd.DataFrame],
    score: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    execution = config["execution"]
    cost = float(execution["roundTripCost"])
    horizon = int(execution["holdingTradingDays"])
    top_count = int(config["objective"]["topCount"])
    eligible = panel["eligible"].fillna(False)
    selected = select_fixed_top10(score, eligible, top_count)
    label = mechanism.horizon_label(panel, eligible, horizon)
    liquidity = trailing_liquidity_deciles(
        panel,
        int(execution["matchedControlLiquidityLookback"]),
        int(execution["matchedControlLiquidityDeciles"]),
    )
    boards = board_map(config).reindex(score.columns).fillna("unknown")
    mature_dates = score.index[: -(horizon + 1)]
    minimum_universe = int(execution["minimumDailyEligibleUniverse"])
    rows: list[dict[str, Any]] = []
    top_selected_total = 0
    top_resolved_total = 0
    top_gross_wins = 0
    top_net_wins = 0
    top_net_losses = 0
    top_gross_sum = 0.0
    top_net_sum = 0.0
    top_net_values: list[float] = []
    rest_count = 0
    rest_gross_wins = 0
    rest_net_wins = 0
    rest_net_losses = 0
    rest_gross_sum = 0.0
    rest_net_sum = 0.0

    for date in mature_dates:
        selected_row = selected.loc[date]
        chosen = int(selected_row.sum())
        if chosen == 0:
            continue
        label_row = label.loc[date]
        resolved_mask = selected_row & label_row.notna()
        top = label_row[resolved_mask]
        rest_mask = eligible.loc[date] & ~selected_row & label_row.notna()
        rest = label_row[rest_mask]
        if len(rest) < minimum_universe:
            continue
        top_selected_total += chosen
        top_resolved_total += len(top)
        top_gross_wins += int(top.gt(0.0).sum())
        top_net_wins += int(top.gt(cost).sum())
        top_net_losses += int(top.lt(cost).sum())
        top_gross_sum += float(top.sum())
        top_net = top - cost
        top_net_sum += float(top_net.sum())
        top_net_values.extend(map(float, top_net))
        rest_count += len(rest)
        rest_gross_wins += int(rest.gt(0.0).sum())
        rest_net_wins += int(rest.gt(cost).sum())
        rest_net_losses += int(rest.lt(cost).sum())
        rest_gross_sum += float(rest.sum())
        rest_net_sum += float((rest - cost).sum())
        matched = matched_control(
            pd.Timestamp(date),
            resolved_mask,
            rest_mask,
            label_row,
            boards,
            liquidity.loc[date],
            cost,
        )
        # A future unfilled selection occupies cash; it is not replaced by rank 11.
        top_gross_mean = float(top.sum()) / chosen
        top_net_mean = float(top_net.sum()) / chosen
        row = {
            "date": pd.Timestamp(date),
            "selected": chosen,
            "resolved": len(top),
            "resolvedFraction": len(top) / chosen,
            "topGrossMean": top_gross_mean,
            "topNetMean": top_net_mean,
            "topMedianNet": float(top_net.median()) if len(top_net) else None,
            "topGrossUpProbability": float(top.gt(0.0).sum()) / chosen,
            "topNetPositiveProbability": float(top.gt(cost).sum()) / chosen,
            "topNetLossProbability": float(top.lt(cost).sum()) / chosen,
            "restMeanNet": float((rest - cost).mean()),
            "restGrossUpProbability": float(rest.gt(0.0).mean()),
            "restNetPositiveProbability": float(rest.gt(cost).mean()),
            "restNetLossProbability": float(rest.lt(cost).mean()),
            "returnLiftVsRest": top_net_mean - float((rest - cost).mean()),
            "grossUpProbabilityLift": float(top.gt(0.0).sum()) / chosen
            - float(rest.gt(0.0).mean()),
            "netPositiveProbabilityLift": float(top.gt(cost).sum()) / chosen
            - float(rest.gt(cost).mean()),
            "matchedCount": matched["count"],
            "matchedMeanNet": matched["meanNet"],
            "matchedGrossUpProbability": matched["grossUpProbability"],
            "matchedNetPositiveProbability": matched["netPositiveProbability"],
            "returnLiftVsMatched": (
                top_net_mean - float(matched["meanNet"])
                if matched["meanNet"] is not None
                else None
            ),
        }
        rows.append(row)
    daily = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    lag = int(config["metrics"]["hacLagTradingDays"])

    def hac(column: str) -> float | None:
        if daily.empty or column not in daily:
            return None
        values = daily[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        value = autonomous.newey_west_t(values, lag) if len(values) >= 20 else None
        return round(float(value), 4) if value is not None else None

    summary = {
        "tradingDays": len(daily),
        "selectedSlots": top_selected_total,
        "resolvedSelections": top_resolved_total,
        "resolvedSelectionFraction": round(top_resolved_total / max(1, top_selected_total), 8),
        "top10GrossUpProbability": round(top_gross_wins / max(1, top_selected_total), 8),
        "restGrossUpProbability": round(rest_gross_wins / max(1, rest_count), 8),
        "grossUpProbabilityLift": round(
            top_gross_wins / max(1, top_selected_total)
            - rest_gross_wins / max(1, rest_count),
            8,
        ),
        "top10NetPositiveProbability": round(top_net_wins / max(1, top_selected_total), 8),
        "restNetPositiveProbability": round(rest_net_wins / max(1, rest_count), 8),
        "netPositiveProbabilityLift": round(
            top_net_wins / max(1, top_selected_total)
            - rest_net_wins / max(1, rest_count),
            8,
        ),
        "top10NetLossProbability": round(top_net_losses / max(1, top_selected_total), 8),
        "restNetLossProbability": round(rest_net_losses / max(1, rest_count), 8),
        "netLossProbabilityLift": round(
            top_net_losses / max(1, top_selected_total)
            - rest_net_losses / max(1, rest_count),
            8,
        ),
        "top10MeanGrossReturn": round(top_gross_sum / max(1, top_selected_total), 8),
        "top10MeanNetReturn": round(top_net_sum / max(1, top_selected_total), 8),
        "top10MedianNetReturn": round(float(np.median(top_net_values)), 8) if top_net_values else None,
        "restMeanGrossReturn": round(rest_gross_sum / max(1, rest_count), 8),
        "restMeanNetReturn": round(rest_net_sum / max(1, rest_count), 8),
        "meanNetReturnLiftVsRest": round(float(daily["returnLiftVsRest"].mean()), 8) if len(daily) else None,
        "meanNetReturnLiftVsMatchedControl": round(float(daily["returnLiftVsMatched"].mean()), 8) if len(daily) else None,
        "dailyOutperformanceRateVsRest": round(float(daily["returnLiftVsRest"].gt(0.0).mean()), 8) if len(daily) else None,
        "hacT": {
            "top10MeanNetReturn": hac("topNetMean"),
            "grossUpProbabilityLift": hac("grossUpProbabilityLift"),
            "netPositiveProbabilityLift": hac("netPositiveProbabilityLift"),
            "netReturnLiftVsRest": hac("returnLiftVsRest"),
            "netReturnLiftVsMatchedControl": hac("returnLiftVsMatched"),
        },
        "inferenceUnit": "trading_day",
        "unfilledSelectionsCountAsCashAndAreNeverReplaced": True,
    }
    return daily, summary


def gate_verdict(summary: dict[str, Any], config: dict[str, Any], forward: bool) -> dict[str, Any]:
    gates = config["preregisteredAcceptanceGates"]
    hac = summary.get("hacT", {})
    checks = {
        "minimumDays": summary.get("tradingDays", 0)
        >= int(gates["minimumIndependentForwardTradingDays"]),
        "resolvedFraction": (summary.get("resolvedSelectionFraction") or 0.0)
        >= float(gates["minimumResolvedSelectionFraction"]),
        "grossUpProbabilityLift": (summary.get("grossUpProbabilityLift") or -math.inf)
        > float(gates["minimumGrossUpProbabilityLift"]),
        "netPositiveProbabilityLift": (summary.get("netPositiveProbabilityLift") or -math.inf)
        > float(gates["minimumNetPositiveProbabilityLift"]),
        "netLossProbabilityLift": (summary.get("netLossProbabilityLift") or math.inf)
        < float(gates["maximumNetLossProbabilityLift"]),
        "top10MeanNetReturn": (summary.get("top10MeanNetReturn") or -math.inf)
        > float(gates["minimumTop10MeanNetReturn"]),
        "top10MedianNetReturn": (summary.get("top10MedianNetReturn") or -math.inf)
        > float(gates["minimumTop10MedianNetReturn"]),
        "returnLiftVsRest": (summary.get("meanNetReturnLiftVsRest") or -math.inf)
        > float(gates["minimumNetReturnLiftVsRest"]),
        "returnLiftVsMatched": (summary.get("meanNetReturnLiftVsMatchedControl") or -math.inf)
        > float(gates["minimumNetReturnLiftVsMatchedControl"]),
        "dailyOutperformanceRate": (summary.get("dailyOutperformanceRateVsRest") or 0.0)
        > float(gates["minimumDailyOutperformanceRate"]),
        "probabilityHac": (hac.get("grossUpProbabilityLift") or -math.inf)
        >= float(gates["minimumHacTForProbabilityLift"]),
        "top10ReturnHac": (hac.get("top10MeanNetReturn") or -math.inf)
        >= float(gates["minimumHacTForTop10NetReturn"]),
        "returnLiftHac": min(
            hac.get("netReturnLiftVsRest") or -math.inf,
            hac.get("netReturnLiftVsMatchedControl") or -math.inf,
        )
        >= float(gates["minimumHacTForReturnLift"]),
    }
    passed = all(checks.values())
    return {
        "scope": "clean_forward" if forward else "historical_diagnostic_only",
        "checks": checks,
        "allPassed": passed,
        "eligibleForTrading": False,
        "decision": (
            "forward_evidence_passed_requires_separate_human_review"
            if forward and passed
            else "insufficient_or_failed_keep_research_only"
        ),
    }


def frozen_calibration(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    n = int(summary.get("selectedSlots") or 0)
    alpha = float(calibration["betaPriorAlpha"])
    beta = float(calibration["betaPriorBeta"])
    gross_wins = float(summary.get("top10GrossUpProbability") or 0.0) * n
    net_wins = float(summary.get("top10NetPositiveProbability") or 0.0) * n
    prior_n = float(calibration["expectedReturnPriorEffectiveObservations"])
    prior_mean = float(calibration["expectedReturnPriorMean"])
    observed_mean = float(summary.get("top10MeanNetReturn") or 0.0)
    expected = (observed_mean * n + prior_mean * prior_n) / max(1.0, n + prior_n)
    return {
        "schemaVersion": "fundamental_top10_frozen_calibration_v2",
        "status": "historical_diagnostic_prior_for_forward_shadow_only",
        "trainingDataEndsAt": calibration["trainingDataEndsAt"],
        "modelVersion": config["forward"]["modelVersion"],
        "selectedSlots": n,
        "estimatedGrossUpProbability": round((gross_wins + alpha) / (n + alpha + beta), 8),
        "estimatedNetPositiveProbability": round((net_wins + alpha) / (n + alpha + beta), 8),
        "estimatedNetLossProbability": round(1.0 - (net_wins + alpha) / (n + alpha + beta), 8),
        "estimatedMeanNetReturn": round(expected, 8),
        "notValidated": True,
        "mayRefit": False,
    }


def render_report(result: dict[str, Any]) -> str:
    row = result["historicalMetrics"]
    verdict = result["historicalGateAudit"]
    pct = lambda value: "n/a" if value is None else f"{100 * float(value):.3f}%"
    lines = [
        "# Fixed daily Top10 discrimination V2",
        "",
        "> **Research-only / shadow-only / not validated. Historical data can only reject.**",
        "",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- universe: `{result['dataAudit']['symbols']}` PIT SH/SZ stocks",
        f"- evaluated sessions: `{row['tradingDays']}`",
        f"- resolved selections: `{row['resolvedSelections']}/{row['selectedSlots']}` "
        f"({pct(row['resolvedSelectionFraction'])})",
        "- execution: close t score -> buyable open t+1 -> sellable open t+2 -> 30 bps cost",
        "",
        "## Probability discrimination",
        "",
        "| metric | Top10 | other eligible | lift | HAC t |",
        "|---|---:|---:|---:|---:|",
        f"| gross up | {pct(row['top10GrossUpProbability'])} | {pct(row['restGrossUpProbability'])} | "
        f"{pct(row['grossUpProbabilityLift'])} | {row['hacT']['grossUpProbabilityLift']} |",
        f"| net positive | {pct(row['top10NetPositiveProbability'])} | {pct(row['restNetPositiveProbability'])} | "
        f"{pct(row['netPositiveProbabilityLift'])} | {row['hacT']['netPositiveProbabilityLift']} |",
        f"| net loss | {pct(row['top10NetLossProbability'])} | {pct(row['restNetLossProbability'])} | "
        f"{pct(row['netLossProbabilityLift'])} | n/a |",
        "",
        "## Return discrimination",
        "",
        "| metric | value | HAC t |",
        "|---|---:|---:|",
        f"| Top10 mean gross return | {pct(row['top10MeanGrossReturn'])} | n/a |",
        f"| Top10 mean net return | {pct(row['top10MeanNetReturn'])} | {row['hacT']['top10MeanNetReturn']} |",
        f"| Top10 median net return | {pct(row['top10MedianNetReturn'])} | n/a |",
        f"| other eligible mean net return | {pct(row['restMeanNetReturn'])} | n/a |",
        f"| Top10 net lift vs rest | {pct(row['meanNetReturnLiftVsRest'])} | {row['hacT']['netReturnLiftVsRest']} |",
        f"| Top10 net lift vs matched control | {pct(row['meanNetReturnLiftVsMatchedControl'])} | "
        f"{row['hacT']['netReturnLiftVsMatchedControl']} |",
        "",
        "## Gate audit",
        "",
        f"Historical gates passed numerically: `{verdict['allPassed']}`. Even a historical pass cannot promote.",
        "",
    ]
    for name, passed in verdict["checks"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The objective is aggregate forward discrimination, not a guarantee that every "
            "selected stock rises every day. An unfilled selection remains cash and is never "
            "replaced using future executability. Inference is clustered by trading day, so "
            "thousands of same-day stock outcomes do not create false sample size.",
            "",
            "Validated factors remain `0`; orders remain `[]`.",
            "",
        ]
    )
    return "\n".join(lines)


def next_xshg_session(date: pd.Timestamp) -> str:
    try:
        import exchange_calendars as calendars

        value = calendars.get_calendar("XSHG").next_session(date.normalize())
        return str(pd.Timestamp(value).date())
    except Exception:
        return str((date.normalize() + pd.offsets.BDay(1)).date())


def security_names(config: dict[str, Any]) -> dict[str, str]:
    mechanism_config = load_json(ROOT / str(config["frozenMechanismConfig"]))
    base = load_json(ROOT / str(mechanism_config["baseResearchConfig"]))
    rows = mechanism.ashare.read_jsonl(
        ROOT / str(base["assetUniverse"]["masterPath"])
    )
    return {
        str(row.get("securityId")): str(row.get("name") or "")
        for row in rows
        if row.get("securityId")
    }


def write_forward_prediction(
    score: pd.DataFrame,
    family_scores: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    as_of = pd.Timestamp(score.index.max()).normalize()
    cutoff = pd.Timestamp(config["forward"]["cleanEvidenceStartsStrictlyAfter"])
    if as_of <= cutoff:
        return {
            "status": "waiting_for_first_clean_forward_panel_date",
            "latestPanelDate": str(as_of.date()),
            "officialPredictionWritten": False,
        }
    selected = select_fixed_top10(score, eligible, 10).loc[as_of]
    names = security_names(config)
    values = score.loc[as_of, selected].sort_values(ascending=False)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, value) in enumerate(values.items(), start=1):
        exchange, stock_code = str(security_id).split(".", 1)
        rows.append(
            {
                "rank": rank,
                "securityId": str(security_id),
                "exchange": exchange,
                "stockCode": stock_code,
                "name": names.get(str(security_id), ""),
                "score": round(float(value), 8),
                "familyScores": {
                    family: round(float(frame.loc[as_of, security_id]), 8)
                    for family, frame in family_scores.items()
                },
            }
        )
    payload = {
        "schemaVersion": "fundamental_top10_forward_prediction_v2",
        "status": "research_only_shadow_only_unvalidated",
        "modelVersion": config["forward"]["modelVersion"],
        "asOfDate": str(as_of.date()),
        "effectiveDate": next_xshg_session(as_of),
        "generatedAt": datetime.now().astimezone().isoformat(),
        "estimatedTop10GroupProbabilityUp": calibration["estimatedGrossUpProbability"],
        "estimatedTop10GroupProbabilityNetPositive": calibration["estimatedNetPositiveProbability"],
        "estimatedTop10GroupLossProbability": calibration["estimatedNetLossProbability"],
        "estimatedTop10GroupMeanNetReturn": calibration["estimatedMeanNetReturn"],
        "estimateStatus": "frozen_historical_diagnostic_prior_not_validated",
        "selections": rows,
        "tradeInstruction": False,
        "validatedFactorCount": 0,
        "orders": [],
    }
    root = ROOT / str(config["forward"]["outputRoot"])
    path = root / f"predictions_{as_of:%Y-%m-%d}.json"
    if path.exists():
        old, new = load_json(path), dict(payload)
        old.pop("generatedAt", None)
        new.pop("generatedAt", None)
        if old != new:
            raise RuntimeError("immutable Top10 prediction differs for existing date")
    else:
        atomic_write(path, json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")
    ledger = root / "predictions.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []
    dates = {str(json.loads(line).get("asOfDate")) for line in lines if line.strip()}
    if str(as_of.date()) not in dates:
        atomic_write(ledger, "\n".join(lines + [canonical(json_safe(payload))]) + "\n")
    return {
        "status": "forward_prediction_written",
        "latestPanelDate": str(as_of.date()),
        "effectiveDate": payload["effectiveDate"],
        "officialPredictionWritten": True,
        "path": str(path),
        "selected": len(rows),
    }


def run(config_path: Path, mode: str, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    panel, score, families, panel_audit, event_audit = build_frozen_score(config)
    daily, historical = evaluate_daily_top10(panel, score, config)
    calibration_end = pd.Timestamp(config["calibration"]["trainingDataEndsAt"])
    training_daily = daily[daily.index <= calibration_end]
    # The current preregistered source ends on the calibration date. Fail closed if a
    # later historical row would accidentally enter the frozen prior.
    if len(training_daily) != len(daily):
        _, historical_for_calibration = evaluate_daily_top10(
            {key: value.loc[:calibration_end] for key, value in panel.items()},
            score.loc[:calibration_end],
            config,
        )
    else:
        historical_for_calibration = historical
    calibration = frozen_calibration(historical_for_calibration, config)
    output_root = ROOT / str(config["output"]["root"])
    calibration_path = output_root / "frozen_calibration.json"
    calibration_content = json.dumps(json_safe(calibration), ensure_ascii=False, indent=2) + "\n"
    if calibration_path.exists() and load_json(calibration_path) != json_safe(calibration):
        raise RuntimeError("frozen Top10 calibration would change")
    if not calibration_path.exists():
        atomic_write(calibration_path, calibration_content)
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_unvalidated",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "configSha256": digest(config),
        "dataAudit": {
            "start": str(score.index.min().date()),
            "end": str(score.index.max().date()),
            "tradingDays": len(score.index),
            "symbols": len(score.columns),
            "panel": panel_audit,
            "fundamentals": event_audit,
        },
        "historicalMetrics": historical,
        "historicalGateAudit": gate_verdict(historical, config, forward=False),
        "frozenCalibration": calibration,
        "historicalWindowPreviouslyViewed": True,
        "validatedFactorCount": 0,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    response: dict[str, Any] = {"mode": mode}
    if mode in {"historical", "both"}:
        identifier = run_id or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z")
        result["runId"] = identifier
        run_root = output_root / identifier
        atomic_write(
            run_root / "result.json",
            json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write(run_root / "report.md", render_report(result))
        daily.to_csv(run_root / "daily_comparison.csv", encoding="utf-8", index=True)
        response["historical"] = {
            "runId": identifier,
            "report": str(run_root / "report.md"),
            "result": str(run_root / "result.json"),
            "historicalGatesPassed": result["historicalGateAudit"]["allPassed"],
            "eligibleForTrading": False,
        }
    if mode in {"forward", "both"}:
        response["forward"] = write_forward_prediction(
            score, families, panel["eligible"].fillna(False), calibration, config
        )
    return response


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode", choices=("historical", "forward", "both"), default="both"
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    response = run(args.config.resolve(), args.mode, args.run_id)
    print(json.dumps(json_safe(response), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
