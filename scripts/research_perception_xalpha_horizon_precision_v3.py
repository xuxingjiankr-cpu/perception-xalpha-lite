#!/usr/bin/env python3
"""Frozen-factor holding-horizon and consensus precision research for A shares.

The twelve factors, their directions and their weights are copied exactly from the V2
train-only result.  This study does not mine another factor or fit another prediction
model.  It asks two narrower questions:

* does majority agreement improve the probability of a cost-positive outcome; and
* does holding for 2/3/5/10 sessions work better than the original one-session label?

Signals are formed after day-t close, entries are attempted at t+1 open, and exits are
attempted after the configured holding period.  A locked intended exit is carried to the
first sellable open within a small fixed delay instead of being silently dropped.  All
outputs are diagnostic research artifacts and orders are always empty.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
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

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402


SCHEMA_VERSION = "perception_xalpha_horizon_precision_result_v3"
CODE_VERSION = "perception_xalpha_horizon_precision_v3_20260806"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_horizon_precision_v3.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
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
    if config.get("schemaVersion") != "perception_xalpha_horizon_precision_v3":
        raise ValueError("unexpected precision V3 schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("precision V3 must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every precision V3 mutation permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("precision V3 output must remain diagnostic_only")
    hypothesis = config["preregisteredHypothesis"]
    if not hypothesis.get("factorKeysDirectionsAndWeightsFrozen"):
        raise ValueError("factor book must be frozen")
    if not hypothesis.get("validationAndShadowMayOnlyReject"):
        raise ValueError("external windows must remain reject-only")
    if hypothesis.get("historicalRunCanPromote") is not False:
        raise ValueError("historical precision research cannot promote")
    factors = config["frozenFactors"]
    keys = [str(item["factorKey"]) for item in factors]
    if len(factors) != 12 or len(keys) != len(set(keys)):
        raise ValueError("precision V3 requires twelve unique frozen factors")
    if any(float(item["direction"]) not in (-1.0, 1.0) for item in factors):
        raise ValueError("factor direction must be +1 or -1")
    if abs(sum(float(item["weight"]) for item in factors) - 1.0) > 1e-9:
        raise ValueError("frozen factor weights must sum to one")
    horizons = list(map(int, config["data"]["holdingTradingDays"]))
    counts = list(map(int, config["data"]["topCounts"]))
    if horizons != [1, 2, 3, 5, 10] or counts != [1, 3, 10]:
        raise ValueError("the preregistered horizon/count grid cannot change")
    if config["data"].get("periodLocalOutcomeContainment") is not True:
        raise ValueError("every outcome must remain inside its evaluation period")
    expected_trials = len(horizons) * len(counts) * len(config["policies"])
    if int(hypothesis["countsAsAdditionalPolicyTrials"]) != expected_trials:
        raise ValueError("policy trial burden does not match the declared grid")
    if config["selectionRule"].get("developmentChoiceUsesTrainOnly") is not True:
        raise ValueError("development policy choice must use train only")
    if config["selectionRule"].get("externalEvaluationIsRejectOnly") is not True:
        raise ValueError("external evaluation must be reject-only")
    output = str(config["output"]["root"]).replace("\\", "/")
    if not output.startswith("outputs/edge_research/"):
        raise ValueError("precision output must remain under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("precision orders must remain empty")


def verify_frozen_source(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = ROOT / config["frozenSourceSummary"]
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8"))
    if source.get("schemaVersion") != "perception_xalpha_nextday_factor_zoo_result_v2":
        raise ValueError("frozen source is not the expected V2 result")
    discovery = source["factorDiscovery"]
    if discovery["selectedFactorPrefixCausality"]["passed"] != 12:
        raise ValueError("not every source factor passed prefix causality")
    source_factors = {
        item["factorKey"]: float(item["direction"])
        for item in discovery["selectedFactors"]
    }
    source_weights = {
        str(key): float(value)
        for key, value in discovery["selectedTrainOnlyWeights"].items()
    }
    configured = {item["factorKey"]: float(item["direction"]) for item in config["frozenFactors"]}
    configured_weights = {
        item["factorKey"]: float(item["weight"]) for item in config["frozenFactors"]
    }
    if configured != source_factors:
        raise ValueError("configured factor keys/directions differ from the frozen V2 source")
    if any(abs(configured_weights[key] - source_weights[key]) > 1e-12 for key in configured):
        raise ValueError("configured factor weights differ from the frozen V2 source")
    return source, hashlib.sha256(raw).hexdigest()


def build_factor_inputs(
    panel: dict[str, Any], *, vwap_basis: str = "archive_vwap_v2"
) -> dict[str, Any]:
    """Preserve the supplied price-basis-consistent proxy, not cash/share VWAP.

    legacy_amount_volume_v1 exists solely for labelled identical-support research
    comparisons. Neither version calls an OHLC4 proxy true transaction VWAP.
    """
    inputs = dict(panel)
    close = panel["close"]
    inputs["returns"] = close.pct_change(fill_method=None)
    volume = panel["volume"].replace(0.0, np.nan)
    cash = panel["amount"].div(volume)
    if vwap_basis not in ("archive_vwap_v2", "legacy_amount_volume_v1"):
        raise ValueError("unknown_factor_vwap_basis")
    archived = panel.get("vwap", close * np.nan).reindex_like(close)
    if vwap_basis == "archive_vwap_v2":
        inputs["vwap"] = archived.combine_first(cash).combine_first(close)
        archive_mask = archived.notna()
    else:
        inputs["vwap"] = cash.combine_first(close)
        archive_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    cash_mask = ~archive_mask & cash.notna()
    close_mask = ~archive_mask & ~cash_mask & close.notna()
    inputs["vwap"].attrs["factorInputBasisAudit"] = {
        "version": vwap_basis,
        "archivePreservedCells": int(archive_mask.to_numpy().sum()),
        "unadjustedAmountVolumeFallbackCells": int(cash_mask.to_numpy().sum()),
        "closeFallbackCells": int(close_mask.to_numpy().sum()),
        "archiveSemantics": "preserved_as_supplied_not_certified_true_transaction_vwap",
        "fallbackBasisCertified": False,
    }
    return inputs


def compute_frozen_scores(
    panel: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute one past-only composite and an independent-factor agreement count."""
    close = panel["close"]
    eligible = panel["eligible"]
    inputs = build_factor_inputs(panel)
    numerator = close * 0.0
    available_weight = close * 0.0
    support = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=np.int16)
    support_thresholds = {
        float(item["supportRankPercentile"]) for item in config["policies"].values()
    }
    if len(support_thresholds) != 1:
        raise ValueError("all policies must share one frozen support percentile")
    support_threshold = support_thresholds.pop()
    audit: list[dict[str, Any]] = []
    for position, item in enumerate(config["frozenFactors"], start=1):
        key = str(item["factorKey"])
        zoo, name = key.split("/", 1)
        module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
        raw = module.compute(inputs).reindex_like(close)
        oriented = raw * float(item["direction"])
        ranked = oriented.where(eligible).rank(axis=1, pct=True)
        weight = float(item["weight"])
        numerator = numerator.add(ranked.fillna(0.0) * weight, fill_value=0.0)
        available_weight = available_weight.add(
            ranked.notna().astype(float) * weight, fill_value=0.0
        )
        support = support.add(ranked.ge(support_threshold).astype(np.int16), fill_value=0)
        audit.append(
            {
                "factorKey": key,
                "direction": float(item["direction"]),
                "weight": weight,
                "finiteValues": int(np.isfinite(raw.to_numpy(dtype=float)).sum()),
                "pastOnlySourceAudit": "passed_in_frozen_v2_source",
            }
        )
        print(f"factor_ready {position}/12 {key}", flush=True)
        del raw, oriented, ranked
    score = numerator.div(available_weight.replace(0.0, np.nan)).where(eligible)
    return score, support.astype(np.int16), {
        "factorInputBasis": inputs["vwap"].attrs["factorInputBasisAudit"],
        "factorCount": len(audit),
        "supportRankPercentile": support_threshold,
        "factors": audit,
    }


def executable_horizon_return(
    panel: dict[str, pd.DataFrame],
    holding_days: int,
    maximum_exit_delay: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return t+1-open to first sellable open at/after t+1+h, without exit backfill."""
    open_price = panel["open"]
    eligible = panel["eligible"]
    buyable, sellable = autonomous.tradability_frames(panel)
    entry = open_price.shift(-1)
    execution_eligible = (
        eligible
        & eligible.shift(-1).eq(True)
        & buyable.shift(-1).eq(True)
        & entry.notna()
    )
    chosen_exit = open_price * np.nan
    chosen_delay = open_price * np.nan
    unresolved = execution_eligible.copy()
    for delay in range(maximum_exit_delay + 1):
        offset = holding_days + 1 + delay
        candidate = open_price.shift(-offset)
        candidate_sellable = sellable.shift(-offset).eq(True) & candidate.notna()
        take = unresolved & candidate_sellable
        chosen_exit = chosen_exit.where(~take, candidate)
        chosen_delay = chosen_delay.where(~take, float(delay))
        unresolved = unresolved & ~take
    returns = chosen_exit.div(entry.replace(0.0, np.nan)) - 1.0
    returns = returns.where(execution_eligible)
    return returns, execution_eligible, chosen_delay.where(execution_eligible)


def selection_mask(
    score: pd.DataFrame,
    support: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    minimum_support: int,
    top_count: int,
) -> pd.DataFrame:
    """Rank only information available at the t+1 opening decision; never inspect the exit."""
    candidate = execution_eligible & score.notna()
    if minimum_support > 0:
        candidate &= support.ge(minimum_support)
    ranks = score.where(candidate).rank(axis=1, ascending=False, method="first")
    return ranks.le(top_count).fillna(False)


def _wilson_lower(successes: int, count: int, z: float = 1.959963984540054) -> float | None:
    if count <= 0:
        return None
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = proportion + z * z / (2.0 * count)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
    return (centre - radius) / denominator


def summarize_selection(
    selected: pd.DataFrame,
    returns: pd.DataFrame,
    exit_delay: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
    holding_days: int,
) -> dict[str, Any]:
    selected_period = selected.reindex(index=dates).fillna(False)
    returns_period = returns.reindex(index=dates).where(selected_period)
    delay_period = exit_delay.reindex(index=dates).where(selected_period)
    intended_count = int(selected_period.sum().sum())
    resolved = returns_period.stack(future_stack=True).dropna().astype(float)
    resolved_count = len(resolved)
    daily_gross = returns_period.mean(axis=1, skipna=True).dropna()
    cost = float(config["data"]["roundTripCost"])
    daily_net = daily_gross - cost
    daily_wins = int(daily_net.gt(0.0).sum())
    daily_count = len(daily_net)
    stock_wins = int(resolved.gt(cost).sum())
    tail_count = max(1, int(math.ceil(daily_count * 0.10))) if daily_count else 0
    equity = (1.0 + daily_net).cumprod() if daily_count else pd.Series(dtype=float)
    maximum_drawdown = (
        float((equity.div(equity.cummax()) - 1.0).min()) if not equity.empty else None
    )
    hac_t = (
        autonomous.newey_west_t(daily_net.to_numpy(dtype=float), max(0, holding_days - 1))
        if daily_count >= 3
        else None
    )
    delayed = delay_period.stack(future_stack=True).dropna().astype(float)
    signal_days = int(selected_period.any(axis=1).sum())
    return {
        "intendedObservations": intended_count,
        "resolvedObservations": resolved_count,
        "unresolvedObservations": intended_count - resolved_count,
        "resolvedFraction": round(resolved_count / intended_count, 8) if intended_count else None,
        "signalDays": signal_days,
        "availableTradingDays": len(dates),
        "selectionCoverage": round(signal_days / len(dates), 8) if len(dates) else None,
        "meanNamesPerSignalDay": round(intended_count / signal_days, 8) if signal_days else None,
        "stockGrossWinRate": round(float(resolved.gt(0.0).mean()), 8) if resolved_count else None,
        "stockNetWinRate": round(stock_wins / resolved_count, 8) if resolved_count else None,
        "stockNetWinWilsonLower": round(_wilson_lower(stock_wins, resolved_count), 8) if resolved_count else None,
        "stockMeanGrossReturn": round(float(resolved.mean()), 8) if resolved_count else None,
        "stockMeanNetReturn": round(float(resolved.mean() - cost), 8) if resolved_count else None,
        "stockMedianGrossReturn": round(float(resolved.median()), 8) if resolved_count else None,
        "stockSevereLossRate": round(
            float(resolved.le(float(config["data"]["severeLossThreshold"])).mean()), 8
        ) if resolved_count else None,
        "dailyBasketObservations": daily_count,
        "dailyNetWinRate": round(daily_wins / daily_count, 8) if daily_count else None,
        "dailyNetWinWilsonLower": round(_wilson_lower(daily_wins, daily_count), 8) if daily_count else None,
        "dailyMeanGrossReturn": round(float(daily_gross.mean()), 8) if daily_count else None,
        "dailyMeanNetReturn": round(float(daily_net.mean()), 8) if daily_count else None,
        "dailyMedianNetReturn": round(float(daily_net.median()), 8) if daily_count else None,
        "dailyCvar10": round(float(daily_net.nsmallest(tail_count).mean()), 8) if tail_count else None,
        "dailyMaximumDrawdown": round(maximum_drawdown, 8) if maximum_drawdown is not None else None,
        "dailyNetHacT": round(float(hac_t), 4) if hac_t is not None else None,
        "meanExitDelayTradingDays": round(float(delayed.mean()), 8) if len(delayed) else None,
        "delayedExitRate": round(float(delayed.gt(0.0).mean()), 8) if len(delayed) else None,
    }


def split_dates(index: pd.DatetimeIndex, source: dict[str, Any]) -> dict[str, pd.DatetimeIndex]:
    audit = source["splitAudit"]
    ranges = {
        "train": audit["train"][:2],
        "validation": audit["validation"][:2],
        "shadow": audit["shadowQuarantine"][:2],
    }
    output: dict[str, pd.DatetimeIndex] = {}
    for name, (start, end) in ranges.items():
        output[name] = pd.DatetimeIndex(index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))])
    return output


def contained_signal_dates(
    dates: pd.DatetimeIndex,
    holding_days: int,
    maximum_exit_delay: int,
) -> pd.DatetimeIndex:
    """Conservatively keep only signals whose full outcome window ends in-period."""
    lookahead = int(holding_days) + 1 + int(maximum_exit_delay)
    if len(dates) <= lookahead:
        return pd.DatetimeIndex([])
    return dates[:-lookahead]


def choose_development_policy(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any] | None:
    minimum_days = int(config["selectionRule"]["minimumDevelopmentSignalDays"])
    eligible = [
        row for row in rows
        if row["period"] == "train"
        and row["metrics"]["signalDays"] >= minimum_days
        and (row["metrics"]["resolvedFraction"] or 0.0) >= 0.98
    ]
    if not eligible:
        return None
    keys = config["selectionRule"]["sortKeys"]
    metric_names = {
        "daily_net_win_wilson_lower": "dailyNetWinWilsonLower",
        "daily_mean_net_return": "dailyMeanNetReturn",
        "stock_net_win_rate": "stockNetWinRate",
    }
    return max(
        eligible,
        key=lambda row: tuple(
            float(row["metrics"].get(metric_names[key]) or -math.inf) for key in keys
        ),
    )


def build_verdict(
    result_rows: list[dict[str, Any]],
    choice: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    if choice is None:
        return {
            "decision": "insufficient_development_sample",
            "historicalHypothesisPass": False,
            "eligibleForTrading": False,
            "freshForwardRequired": True,
        }
    lookup = {
        (row["period"], row["policy"], row["holdingDays"], row["topCount"]): row["metrics"]
        for row in result_rows
    }
    selected_key = (choice["policy"], choice["holdingDays"], choice["topCount"])
    checks: dict[str, Any] = {}
    for period in ("validation", "shadow"):
        selected = lookup[(period, *selected_key)]
        baseline = lookup[(period, "weighted_composite", 1, 10)]
        checks[period] = {
            "minimumSignalDays": selected["signalDays"]
            >= int(config["data"]["minimumSignalDaysPerPeriod"]),
            "precisionImproves": selected["dailyNetWinRate"] is not None
            and baseline["dailyNetWinRate"] is not None
            and selected["dailyNetWinRate"] > baseline["dailyNetWinRate"],
            "meanNetReturnImproves": selected["dailyMeanNetReturn"] is not None
            and baseline["dailyMeanNetReturn"] is not None
            and selected["dailyMeanNetReturn"] > baseline["dailyMeanNetReturn"],
            "meanNamesNotBelowOne": (selected["meanNamesPerSignalDay"] or 0.0) >= 1.0,
            "resolvedFractionPass": (selected["resolvedFraction"] or 0.0) >= 0.98,
        }
        checks[period]["passed"] = all(checks[period].values())
    historical_pass = all(checks[period]["passed"] for period in checks)
    return {
        "decision": (
            "keep_as_fresh_forward_hypothesis_only"
            if historical_pass
            else "reject_precision_change_keep_diagnostics"
        ),
        "developmentChoice": {
            "policy": choice["policy"],
            "holdingDays": choice["holdingDays"],
            "topCount": choice["topCount"],
            "chosenWithoutExternalOutcomes": True,
        },
        "externalChecks": checks,
        "historicalHypothesisPass": historical_pass,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
        "reasonTradingRemainsDisabled": "external windows were previously viewed and V2 factor discovery already consumed the development window",
    }


def markdown_report(report: dict[str, Any]) -> str:
    choice = report["verdict"].get("developmentChoice")
    lines = [
        "# Horizon Precision V3",
        "",
        f"- Status: `{report['status']}`",
        f"- Run: `{report['runId']}`",
        f"- Data: `{report['dataRange'][0]}` to `{report['dataRange'][1]}`",
        "- Frozen factors: `12` (no factor re-mining)",
        "- Execution: signal at close, buy next open, exit after 1/2/3/5/10 sessions",
        "- Locked exit handling: carry to first sellable open within the fixed delay",
        f"- Development choice: `{choice}`",
        f"- Verdict: `{report['verdict']['decision']}`",
        "- Eligible for trading: `False`",
        "- Orders: `[]`",
        "",
        "## Precision frontier",
        "",
        "| Period | Policy | Hold | Top | Days | Names/day | Stock net win | Daily net win | Wilson lower | Mean net | Severe loss | HAC t |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["precisionFrontier"]:
        metrics = row["metrics"]
        lines.append(
            "| {period} | {policy} | {holding} | {top} | {days} | {names} | {stock} | {daily} | {wilson} | {mean} | {severe} | {hac} |".format(
                period=row["period"],
                policy=row["policy"],
                holding=row["holdingDays"],
                top=row["topCount"],
                days=metrics["signalDays"],
                names=metrics["meanNamesPerSignalDay"],
                stock=metrics["stockNetWinRate"],
                daily=metrics["dailyNetWinRate"],
                wilson=metrics["dailyNetWinWilsonLower"],
                mean=metrics["dailyMeanNetReturn"],
                severe=metrics["stockSevereLossRate"],
                hac=metrics["dailyNetHacT"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Precision means cost-positive return, not raw direction accuracy or whole-universe AUC.",
            "- The development choice uses train rows only; validation and shadow can only reject it.",
            "- Fewer names and skipped days are reported explicitly so abstention cannot masquerade as an improvement.",
            "- Longer holding observations overlap; the reported HAC t uses holding_days-1 lags.",
            "- Even a historical pass requires a separately preregistered fresh-forward study.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["knownLimitations"])
    return "\n".join(lines) + "\n"


def latest_candidates(
    score: pd.DataFrame,
    support: pd.DataFrame,
    panel: dict[str, Any],
    choice: dict[str, Any] | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    if choice is None:
        return pd.DataFrame()
    date = score.index.max()
    minimum_support = int(config["policies"][choice["policy"]]["minimumFactorSupport"])
    eligible = panel["eligible"].loc[date] & score.loc[date].notna()
    if minimum_support:
        eligible &= support.loc[date].ge(minimum_support)
    rows = pd.DataFrame(
        {
            "securityId": score.columns,
            "factorScore": score.loc[date].to_numpy(dtype=float),
            "factorSupport": support.loc[date].to_numpy(dtype=int),
            "eligibleAtSignalClose": eligible.to_numpy(dtype=bool),
        }
    )
    rows = rows[rows["eligibleAtSignalClose"]].sort_values(
        ["factorScore", "factorSupport", "securityId"], ascending=[False, False, True]
    ).head(int(choice["topCount"]))
    rows.insert(0, "signalDate", date.date().isoformat())
    rows["status"] = "diagnostic_only_not_an_order"
    return rows


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    source, source_sha = verify_frozen_source(config)
    base = load_json(ROOT / config["baseResearchConfig"])
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    score, support, factor_audit = compute_frozen_scores(panel, config)
    splits = split_dates(panel["close"].index, source)
    maximum_exit_delay = int(config["data"]["maximumExitDelayTradingDays"])
    result_rows: list[dict[str, Any]] = []
    masks: dict[tuple[str, int, int], pd.DataFrame] = {}
    for holding_days in map(int, config["data"]["holdingTradingDays"]):
        returns, execution_eligible, exit_delay = executable_horizon_return(
            panel, holding_days, maximum_exit_delay
        )
        for policy_name, policy in config["policies"].items():
            for top_count in map(int, config["data"]["topCounts"]):
                mask = selection_mask(
                    score,
                    support,
                    execution_eligible,
                    int(policy["minimumFactorSupport"]),
                    top_count,
                )
                masks[(policy_name, holding_days, top_count)] = mask
                for period, dates in splits.items():
                    evaluation_dates = contained_signal_dates(
                        dates, holding_days, maximum_exit_delay
                    )
                    result_rows.append(
                        {
                            "period": period,
                            "policy": policy_name,
                            "holdingDays": holding_days,
                            "topCount": top_count,
                            "metrics": summarize_selection(
                                mask,
                                returns,
                                exit_delay,
                                evaluation_dates,
                                config,
                                holding_days,
                            ),
                        }
                    )
        print(f"horizon_ready holding_days={holding_days}", flush=True)
    choice = choose_development_policy(result_rows, config)
    verdict = build_verdict(result_rows, choice, config)
    now = datetime.now(timezone.utc)
    run_id = run_id or (
        "run_" + now.strftime("%Y%m%dT%H%M%SZ") + "_" + digest({"config": config, "code": CODE_VERSION})[:10]
    )
    output = ROOT / config["output"]["root"] / run_id
    latest = latest_candidates(score, support, panel, choice, config)
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "research_only_shadow_only_not_trading",
        "runId": run_id,
        "createdAt": now.isoformat(),
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path.resolve()),
        "configSha256": digest(config),
        "frozenSourcePath": str((ROOT / config["frozenSourceSummary"]).resolve()),
        "frozenSourceSha256": source_sha,
        "dataRange": [
            panel["close"].index.min().date().isoformat(),
            panel["close"].index.max().date().isoformat(),
        ],
        "panelAudit": panel_audit,
        "factorAudit": factor_audit,
        "splitAudit": source["splitAudit"],
        "maximumExitDelayTradingDays": maximum_exit_delay,
        "precisionFrontier": result_rows,
        "verdict": verdict,
        "latestDiagnosticCandidateCount": len(latest),
        "knownLimitations": config["knownLimitations"],
        "orders": [],
        "automaticTradingChanges": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "summary.json", canonical(json_safe(report)) + "\n")
    atomic_write(output / "report.md", markdown_report(report))
    flat_rows = []
    for row in result_rows:
        flat_rows.append(
            {
                "period": row["period"],
                "policy": row["policy"],
                "holdingDays": row["holdingDays"],
                "topCount": row["topCount"],
                **row["metrics"],
            }
        )
    atomic_write(
        output / "precision_frontier.csv",
        pd.DataFrame(flat_rows).to_csv(index=False, lineterminator="\n"),
    )
    atomic_write(
        output / "latest_diagnostic_candidates.csv",
        latest.to_csv(index=False, lineterminator="\n"),
    )
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    report = run(args.config, args.run_id)
    print(
        canonical(
            {
                "runId": report["runId"],
                "decision": report["verdict"]["decision"],
                "developmentChoice": report["verdict"].get("developmentChoice"),
                "eligibleForTrading": False,
                "orders": [],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
