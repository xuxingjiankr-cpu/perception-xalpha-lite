#!/usr/bin/env python3
"""Bounded PIT financial-statement factor discovery for next-session A-share ranking.

The generator is mechanism based: it expands an audited field registry into levels,
issuer changes, same-fiscal-period growth and cross-statement confirmation candidates.
Only the training period can select candidates for full evaluation.  Validation and
shadow results are quarantined diagnostics and can never alter trading state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_ashare_fundamentals as fundamentals  # noqa: E402
import research_ashare_universe as ashare  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_fundamental_mechanism_families as mechanism  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_pit_fundamental_catalyst_v5 as catalyst  # noqa: E402


SCHEMA_VERSION = "financial_statement_factor_discovery_result_v1"
CODE_VERSION = "financial_statement_factor_discovery_v1_20260811"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "financial_statement_factor_discovery_v1.json"
)


# Scales only make accounting units comparable inside mechanism composites.  Every
# standalone candidate is ranked cross-sectionally, so no scale is fitted to outcomes.
FIELD_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "roe", "field": "roePct", "family": "profitability", "orientation": 1.0, "levelScale": 15.0, "changeScale": 3.0},
    {"id": "roic", "field": "roicPct", "family": "profitability", "orientation": 1.0, "levelScale": 15.0, "changeScale": 3.0},
    {"id": "gross_margin", "field": "grossMarginPct", "family": "profitability", "orientation": 1.0, "levelScale": 30.0, "changeScale": 5.0},
    {"id": "net_margin", "field": "netMarginPct", "family": "profitability", "orientation": 1.0, "levelScale": 10.0, "changeScale": 3.0},
    {"id": "revenue_growth", "field": "revenueYoyPct", "family": "growth", "orientation": 1.0, "levelScale": 30.0, "changeScale": 20.0},
    {"id": "profit_growth", "field": "netProfitYoyPct", "family": "growth", "orientation": 1.0, "levelScale": 50.0, "changeScale": 30.0},
    {"id": "cash_to_revenue", "field": "operatingCashToRevenue", "family": "cash_flow_quality", "orientation": 1.0, "levelScale": 0.30, "changeScale": 0.15},
    {"id": "cash_to_profit", "field": "operatingCashToNetProfit", "family": "cash_flow_quality", "orientation": 1.0, "levelScale": 1.0, "changeScale": 0.50},
    {"id": "debt_assets", "field": "debtAssetRatioPct", "family": "balance_sheet_safety", "orientation": -1.0, "levelScale": 50.0, "changeScale": 5.0},
    {"id": "current_ratio", "field": "currentRatio", "family": "balance_sheet_safety", "orientation": 1.0, "levelScale": 2.0, "changeScale": 0.50},
    {"id": "quick_ratio", "field": "quickRatio", "family": "balance_sheet_safety", "orientation": 1.0, "levelScale": 2.0, "changeScale": 0.50},
    {"id": "receivable_days", "field": "receivableTurnoverDays", "family": "operating_efficiency", "orientation": -1.0, "levelScale": 180.0, "changeScale": 30.0},
    {"id": "inventory_days", "field": "inventoryTurnoverDays", "family": "operating_efficiency", "orientation": -1.0, "levelScale": 180.0, "changeScale": 30.0},
    {"id": "asset_turnover", "field": "totalAssetTurnover", "family": "operating_efficiency", "orientation": 1.0, "levelScale": 1.0, "changeScale": 0.25},
)

YTD_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "eps_yoy", "field": "epsYtd", "family": "growth", "orientation": 1.0},
    {"id": "revenue_ytd_yoy", "field": "revenueYtd", "family": "growth", "orientation": 1.0},
    {"id": "parent_profit_ytd_yoy", "field": "parentNetProfitYtd", "family": "growth", "orientation": 1.0},
    {"id": "book_value_yoy", "field": "bookValuePerShare", "family": "balance_sheet_safety", "orientation": 1.0},
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _bounded(value: float, scale: float) -> float:
    return math.tanh(float(value) / float(scale))


def _symmetric_yoy(now: float, before: float) -> float | None:
    denominator = abs(now) + abs(before)
    if denominator <= 1e-12:
        return None
    return 2.0 * (now - before) / denominator


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "financial_statement_factor_discovery_v1":
        raise ValueError("unexpected financial-statement discovery schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("financial-statement discovery must remain research-only")
    data = config["data"]
    if data.get("availabilityRule") != "first_market_date_strictly_after_max_notice_update":
        raise ValueError("financial statements must use conservative PIT availability")
    if int(data["holdingTradingDays"]) != 1:
        raise ValueError("V1 is preregistered for the next-session objective")
    if not math.isclose(float(data["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("A-share cost stress must remain 30 bps")
    generation = config["generation"]
    if generation.get("provider") != "deterministic_financial_statement_mechanism_grammar":
        raise ValueError("the generator must remain deterministic and bounded")
    if generation.get("remoteApiAllowed") is not False or generation.get("arbitraryPythonAllowed") is not False:
        raise ValueError("remote generation and arbitrary Python are prohibited")
    if int(generation["maximumGeneratedCandidates"]) > 128:
        raise ValueError("candidate budget is not bounded")
    if int(generation["maximumFullEvaluationCandidates"]) > 32:
        raise ValueError("full-evaluation budget is not bounded")
    if generation.get("validationFeedbackAllowed") is not False or generation.get("shadowFeedbackAllowed") is not False:
        raise ValueError("external outcomes may not feed the generator")
    selection = config["selection"]
    if int(selection["topCount"]) != 10 or int(selection["firstStageCandidateCount"]) != 100:
        raise ValueError("the comparison must remain price Top100 to final Top10")
    if not math.isclose(float(selection["priceWeight"]) + float(selection["fundamentalWeight"]), 1.0, abs_tol=1e-12):
        raise ValueError("fixed blend weights must sum to one")
    if config["evaluation"].get("historicalRunCanPromote") is not False:
        raise ValueError("historical discovery cannot promote")
    safety = config["safety"]
    permissions = [key for key in safety if key.startswith("may")]
    if not permissions or any(bool(safety[key]) for key in permissions):
        raise ValueError("all mutation and trading permissions must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must always remain empty")


def verify_frozen(config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(config[path_key])
    actual = file_sha256(path)
    if actual != str(config[hash_key]).lower():
        raise ValueError(f"frozen input changed: {path_key}")
    return path


def _birth(definition: dict[str, Any]) -> dict[str, Any]:
    item = dict(definition)
    item["factorId"] = "fs_" + digest(definition)[:16]
    item["formulaSha256"] = digest(definition)
    item["status"] = "DRAFT_RESEARCH_ONLY"
    item["pastOnly"] = True
    return item


def generate_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the accounting registry without looking at any return label."""
    level_age = int(config["data"]["levelMaximumAgeTradingDays"])
    change_age = int(config["data"]["changeMaximumAgeTradingDays"])
    candidates: list[dict[str, Any]] = []
    family_transform: dict[tuple[str, str], list[str]] = {}

    for spec in FIELD_SPECS:
        for transform, scale, age in (
            ("level", spec["levelScale"], level_age),
            ("change_from_prior_disclosed_report", spec["changeScale"], change_age),
        ):
            definition = _birth({
                "name": f"{spec['id']}__{transform}",
                "family": spec["family"],
                "kind": "primitive",
                "transform": transform,
                "field": spec["field"],
                "orientation": spec["orientation"],
                "scale": scale,
                "maximumAgeTradingDays": age,
                "economicHypothesis": f"{spec['family']} information diffusion after a causally available filing",
            })
            candidates.append(definition)
            family_transform.setdefault((spec["family"], transform), []).append(definition["factorId"])

    for spec in YTD_SPECS:
        definition = _birth({
            "name": f"{spec['id']}__same_fiscal_period_symmetric_yoy",
            "family": spec["family"],
            "kind": "primitive",
            "transform": "same_fiscal_period_symmetric_yoy",
            "field": spec["field"],
            "orientation": spec["orientation"],
            "scale": 1.0,
            "maximumAgeTradingDays": change_age,
            "economicHypothesis": "same-period accounting growth avoids mixing seasonal cumulative statements",
        })
        candidates.append(definition)
        family_transform.setdefault((spec["family"], definition["transform"]), []).append(definition["factorId"])

    # The grammar creates two robust, non-fitted aggregators for every mechanism block.
    # Mean measures breadth; minimum requires all available statements to confirm.
    for (family, transform), inputs in sorted(family_transform.items()):
        if len(inputs) < 2:
            continue
        for aggregate in ("mean", "minimum"):
            candidates.append(_birth({
                "name": f"{family}__{transform}__{aggregate}",
                "family": family,
                "kind": "composite",
                "transform": "mechanism_breadth" if aggregate == "mean" else "mechanism_confirmation",
                "aggregate": aggregate,
                "inputs": inputs,
                "maximumAgeTradingDays": change_age if transform != "level" else level_age,
                "economicHypothesis": f"multiple {family} line items should agree rather than rely on one noisy ratio",
            }))

    # Cross-statement confirmations are declared by economic mechanism, not historical
    # performance.  The generator expands both breadth and strict-confirmation variants.
    by_name = {item["name"]: item["factorId"] for item in candidates}
    cross_blocks = {
        "growth_cash_confirmation": [
            "growth__change_from_prior_disclosed_report__mean",
            "cash_flow_quality__change_from_prior_disclosed_report__mean",
        ],
        "growth_profitability_confirmation": [
            "growth__change_from_prior_disclosed_report__mean",
            "profitability__change_from_prior_disclosed_report__mean",
        ],
        "quality_safety_confirmation": [
            "profitability__level__mean",
            "balance_sheet_safety__level__mean",
        ],
        "cash_profitability_confirmation": [
            "cash_flow_quality__level__mean",
            "profitability__level__mean",
        ],
        "efficiency_margin_confirmation": [
            "operating_efficiency__change_from_prior_disclosed_report__mean",
            "profitability__change_from_prior_disclosed_report__mean",
        ],
    }
    for name, input_names in cross_blocks.items():
        if not all(value in by_name for value in input_names):
            continue
        inputs = [by_name[value] for value in input_names]
        for aggregate in ("mean", "minimum"):
            candidates.append(_birth({
                "name": f"{name}__{aggregate}",
                "family": "cross_statement_confirmation",
                "kind": "composite",
                "transform": "mechanism_breadth" if aggregate == "mean" else "mechanism_confirmation",
                "aggregate": aggregate,
                "inputs": inputs,
                "maximumAgeTradingDays": change_age,
                "economicHypothesis": "independent financial statements should confirm the same operating improvement",
            }))

    deduplicated = {item["formulaSha256"]: item for item in candidates}
    bounded = list(deduplicated.values())[: int(config["generation"]["maximumGeneratedCandidates"])]
    if len(bounded) != len(candidates):
        raise ValueError("generated candidate library exceeded its preregistered budget")
    return bounded


def _primitive_value(
    definition: dict[str, Any],
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    prior_year: dict[str, Any] | None,
) -> float | None:
    field = str(definition["field"])
    now = _finite(current.get(field))
    raw: float | None = None
    transform = definition["transform"]
    if transform == "level":
        raw = now
    elif transform == "change_from_prior_disclosed_report":
        before = _finite(previous.get(field)) if previous is not None else None
        if now is not None and before is not None:
            raw = now - before
    elif transform == "same_fiscal_period_symmetric_yoy":
        before = _finite(prior_year.get(field)) if prior_year is not None else None
        if now is not None and before is not None:
            raw = _symmetric_yoy(now, before)
    if raw is None or not math.isfinite(raw):
        return None
    return _bounded(float(definition["orientation"]) * raw, float(definition["scale"]))


def candidate_values(
    candidates: list[dict[str, Any]],
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    prior_year: dict[str, Any] | None,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for definition in candidates:
        factor_id = definition["factorId"]
        if definition["kind"] == "primitive":
            values[factor_id] = _primitive_value(definition, current, previous, prior_year)
            continue
        available = [values.get(item) for item in definition["inputs"]]
        finite = [float(item) for item in available if item is not None and math.isfinite(float(item))]
        if len(finite) != len(available) or not finite:
            values[factor_id] = None
        elif definition["aggregate"] == "minimum":
            values[factor_id] = min(finite)
        else:
            values[factor_id] = float(np.mean(finite))
    return values


def causal_records_for_symbol(
    rows: list[dict[str, Any]],
    market_index: pd.DatetimeIndex,
    security_id: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build candidate values without allowing a filing into an earlier feature row."""
    audit: Counter[str] = Counter()
    by_market_date: dict[pd.Timestamp, dict[pd.Timestamp, dict[str, Any]]] = {}
    for raw in rows:
        notice = _date(raw.get("noticeDate"))
        update = _date(raw.get("updateDate"))
        report = _date(raw.get("reportDate"))
        if notice is None or report is None:
            audit["missing_required_date"] += 1
            continue
        available = notice if update is None else max(notice, update)
        position = int(market_index.searchsorted(available, side="right"))
        if position >= len(market_index):
            audit["available_after_panel"] += 1
            continue
        market_date = pd.Timestamp(market_index[position])
        candidate = dict(raw)
        candidate["_reportDate"] = report
        bucket = by_market_date.setdefault(market_date, {})
        if report in bucket:
            audit["same_day_same_report_collapsed"] += 1
        bucket[report] = candidate

    history: dict[pd.Timestamp, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for market_date, bucket in sorted(by_market_date.items()):
        ordered = sorted(bucket.items())
        for contextual_report, contextual in ordered[:-1]:
            history[contextual_report] = contextual
            audit["same_day_older_report_used_as_context"] += 1
        report, current = ordered[-1]
        if previous is not None and report <= previous["_reportDate"]:
            history[report] = current
            audit["non_advancing_report_date_skipped"] += 1
            continue
        prior_year = history.get(pd.Timestamp(report - pd.DateOffset(years=1)).normalize())
        prior_reports = [known for known in history if known < report]
        prior_disclosed = history[max(prior_reports)] if prior_reports else previous
        values = candidate_values(candidates, current, prior_disclosed, prior_year)
        records.append({
            "eventDate": market_date,
            "securityId": security_id,
            "reportDate": report,
            "previousReportDate": prior_disclosed.get("_reportDate") if prior_disclosed else None,
            **values,
        })
        audit["accepted_statement_events"] += 1
        history[report] = current
        previous = current
    return records, dict(audit)


def build_event_table(
    market_index: pd.DatetimeIndex,
    columns: pd.Index,
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = ROOT / str(config["data"]["fundamentalRoot"])
    records: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    files_found = 0
    for position, security_id in enumerate(map(str, columns), start=1):
        exchange, code = security_id.split(".", 1)
        rows = fundamentals.read_jsonl(root / f"{exchange}_{code}.jsonl")
        if not rows:
            totals["missing_symbol_file"] += 1
            continue
        files_found += 1
        issuer, audit = causal_records_for_symbol(rows, market_index, security_id, candidates)
        records.extend(issuer)
        totals.update(audit)
        if position % 500 == 0:
            print(f"financial_statement_events {position}/{len(columns)} records={len(records)}", flush=True)
    table = pd.DataFrame(records)
    if not table.empty and table.duplicated(["eventDate", "securityId"]).any():
        raise RuntimeError("duplicate issuer-date rows after PIT collapse")
    audit = {
        "root": str(root),
        "symbolsRequested": len(columns),
        "filesFound": files_found,
        "symbolFileCoverage": round(files_found / max(1, len(columns)), 8),
        "statementEvents": len(table),
        "eventDateStart": str(pd.Timestamp(table["eventDate"].min()).date()) if len(table) else None,
        "eventDateEnd": str(pd.Timestamp(table["eventDate"].max()).date()) if len(table) else None,
        "processingCounts": dict(totals),
        "availabilityRule": config["data"]["availabilityRule"],
        "reportDateUsedForAvailability": False,
        "futureRowsWrittenBack": 0,
    }
    return table, audit


def _normal_two_sided_p(t_value: float | None) -> float | None:
    if t_value is None or not math.isfinite(float(t_value)):
        return None
    return math.erfc(abs(float(t_value)) / math.sqrt(2.0))


def evaluate_candidate(
    definition: dict[str, Any],
    event_table: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    timing_score: pd.DataFrame,
    first_stage: pd.DataFrame,
    returns: pd.DataFrame,
    execution_eligible: pd.DataFrame,
    exit_delay: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = mechanism.factor_frame(
        event_table,
        definition["factorId"],
        panel["close"].index,
        panel["close"].columns,
        int(definition["maximumAgeTradingDays"]),
    )
    neutral = autonomous.size_neutralise(
        raw,
        panel,
        int(config["selection"]["liquidityNeutraliseBins"]),
    )
    common = first_stage & execution_eligible & neutral.notna() & timing_score.notna()
    fundamental_rank = catalyst.daily_rank(neutral, common)
    price_rank = catalyst.daily_rank(timing_score, common)
    blend = (
        price_rank * float(config["selection"]["priceWeight"])
        + fundamental_rank * float(config["selection"]["fundamentalWeight"])
    )
    top_count = int(config["selection"]["topCount"])
    selected = catalyst.select_top(blend, common, top_count)
    baseline = catalyst.select_top(price_rank, common, top_count)
    if not selected.sum(axis=1).equals(baseline.sum(axis=1)):
        raise RuntimeError("candidate and baseline selection counts differ")
    holding = int(config["data"]["holdingTradingDays"])
    candidate_performance = precision.summarize_selection(
        selected, returns, exit_delay, dates, config, holding
    )
    baseline_performance = precision.summarize_selection(
        baseline, returns, exit_delay, dates, config, holding
    )
    candidate_daily = catalyst.daily_net_series(
        selected, returns, dates, float(config["data"]["roundTripCost"])
    )
    baseline_daily = catalyst.daily_net_series(
        baseline, returns, dates, float(config["data"]["roundTripCost"])
    )
    paired = catalyst.paired_policy_delta(candidate_daily, baseline_daily, max(0, holding - 1))
    return {
        "factorId": definition["factorId"],
        "name": definition["name"],
        "family": definition["family"],
        "support": {
            "candidateDays": int(common.reindex(index=dates).any(axis=1).sum()),
            "candidateObservations": int(common.reindex(index=dates).sum().sum()),
            "sameCandidatePool": True,
            "sameDailySelectionCount": True,
        },
        "rankIc": catalyst.daily_ic_stats(
            fundamental_rank, returns, common, dates, max(0, holding - 1)
        ),
        "blend": candidate_performance,
        "priceBaselineSameSupport": baseline_performance,
        "pairedDeltaVsPrice": paired,
        "pairedNormalApproxP": _normal_two_sided_p(paired.get("hacT")),
    }


def _fast_score(row: dict[str, Any]) -> float:
    ic = row["rankIc"].get("meanSpearmanIc")
    delta = row["pairedDeltaVsPrice"].get("meanDailyNetDelta")
    if ic is None or delta is None:
        return -math.inf
    return float(ic) + 5.0 * float(delta)


def _period_pass(row: dict[str, Any], minimum_days: int) -> dict[str, bool]:
    blend = row["blend"]
    baseline = row["priceBaselineSameSupport"]
    return {
        "minimumSignalDays": int(blend.get("signalDays") or 0) >= minimum_days,
        "positiveRankIc": float(row["rankIc"].get("meanSpearmanIc") or -math.inf) > 0.0,
        "grossUpRateImproved": float(blend.get("stockGrossWinRate") or 0.0) > float(baseline.get("stockGrossWinRate") or 0.0),
        "meanGrossReturnImproved": float(blend.get("stockMeanGrossReturn") or -math.inf) > float(baseline.get("stockMeanGrossReturn") or -math.inf),
        "pairedMeanImproved": float(row["pairedDeltaVsPrice"].get("meanDailyNetDelta") or -math.inf) > 0.0,
    }


def apply_multiple_testing(
    full_rows: list[dict[str, Any]], generated_trials: int, config: dict[str, Any]
) -> None:
    fdr = float(config["evaluation"]["benjaminiHochbergFdr"])
    ranked = sorted(
        [row for row in full_rows if row["validation"].get("pairedNormalApproxP") is not None],
        key=lambda row: float(row["validation"]["pairedNormalApproxP"]),
    )
    for rank, row in enumerate(ranked, start=1):
        p_value = float(row["validation"]["pairedNormalApproxP"])
        threshold = fdr * rank / max(1, generated_trials)
        row["multipleTesting"] = {
            "generatedTrials": generated_trials,
            "rank": rank,
            "validationP": p_value,
            "bhThresholdUsingAllGeneratedTrials": threshold,
            "passed": p_value <= threshold,
        }
    for row in full_rows:
        row.setdefault("multipleTesting", {
            "generatedTrials": generated_trials,
            "rank": None,
            "validationP": None,
            "bhThresholdUsingAllGeneratedTrials": None,
            "passed": False,
        })
        validation_checks = _period_pass(row["validation"], int(config["selection"]["minimumExternalSignalDays"]))
        shadow_checks = _period_pass(row["shadow"], int(config["selection"]["minimumExternalSignalDays"]))
        row["verdict"] = {
            "validation": validation_checks,
            "shadow": shadow_checks,
            "historicalPass": all(validation_checks.values())
            and all(shadow_checks.values())
            and bool(row["multipleTesting"]["passed"]),
            "eligibleForTrading": False,
            "freshForwardStillRequired": True,
        }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Financial-statement factor discovery V1",
        "",
        "> **Research-only / shadow-only / not a trading signal.**",
        "",
        f"- run_id: `{result['runId']}`",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- universe: `{result['dataAudit']['symbols']}` PIT SH/SZ stocks",
        f"- statement events: `{result['eventAudit']['statementEvents']}`",
        f"- generated candidates: `{result['generationAudit']['generatedCandidates']}`",
        f"- full evaluation candidates: `{result['generationAudit']['fullEvaluationCandidates']}`",
        f"- historically passing candidates: `{result['verdict']['historicalPassingCandidates']}`",
        "- objective: safe filing-session close -> next buyable open -> following sellable open",
        "- orders: `[]`",
        "",
        "## Full evaluation",
        "",
        "| candidate | family | period | up rate | baseline | mean gross | baseline | IC | paired delta | BH pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["fullEvaluation"]:
        for period in ("validation", "shadow"):
            metric = row[period]
            blend = metric["blend"]
            base = metric["priceBaselineSameSupport"]
            lines.append(
                f"| {row['name']} | {row['family']} | {period} | "
                f"{100 * float(blend.get('stockGrossWinRate') or 0):.2f}% | "
                f"{100 * float(base.get('stockGrossWinRate') or 0):.2f}% | "
                f"{100 * float(blend.get('stockMeanGrossReturn') or 0):.3f}% | "
                f"{100 * float(base.get('stockMeanGrossReturn') or 0):.3f}% | "
                f"{metric['rankIc'].get('meanSpearmanIc')} | "
                f"{100 * float(metric['pairedDeltaVsPrice'].get('meanDailyNetDelta') or 0):.3f}% | "
                f"{row['multipleTesting']['passed']} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- Candidate formulas were generated without return labels. Only train metrics selected the full-evaluation set.",
        "- Validation and shadow outcomes never feed candidate synthesis, mutation or weights.",
        "- Every candidate is compared with the frozen price score on the identical Top100 support and identical Top10 count.",
        "- Filing availability is the first market session strictly after `max(noticeDate, updateDate)`.",
        "- Historical windows have already been viewed elsewhere. A numerical pass is only a preregistered forward hypothesis.",
        "- Incomplete original restatement vintages and unavailable historical industry membership remain material limitations.",
        "",
        "No trading configuration, observation pool, order, position, risk gate, overlay, or execution lock was read or modified.",
        "",
    ])
    return "\n".join(lines)


def run(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    base_path = verify_frozen(config, "baseResearchConfig", "baseResearchConfigFileSha256")
    timing_path = verify_frozen(config, "frozenTimingConfig", "frozenTimingConfigFileSha256")
    split_path = verify_frozen(config, "frozenSplitSummary", "frozenSplitSummaryFileSha256")
    base_config = load_json(base_path)
    perception.validate_config(base_config)
    _, cog_config = perception.load_base_configs(base_config)
    panel, panel_audit = ashare.build_panel(base_config["assetUniverse"], cog_config["data"])
    if not panel_audit.get("unbiasedHistoricalValidationEligible", False):
        raise RuntimeError("clean PIT adjusted panel failed its unbiased-data gate")
    close = panel["close"]
    timing_config = load_json(timing_path)
    precision.validate_config(timing_config)
    timing_score, _, timing_audit = precision.compute_frozen_scores(panel, timing_config)
    returns, execution_eligible, exit_delay = precision.executable_horizon_return(
        panel,
        holding_days=int(config["data"]["holdingTradingDays"]),
        maximum_exit_delay=int(config["data"]["maximumExitDelayTradingDays"]),
    )
    first_stage_candidates = panel["eligible"] & execution_eligible & timing_score.notna()
    first_stage = catalyst.select_top(
        timing_score,
        first_stage_candidates,
        int(config["selection"]["firstStageCandidateCount"]),
    )
    candidates = generate_candidates(config)
    event_table, event_audit = build_event_table(close.index, close.columns, config, candidates)
    if float(event_audit["symbolFileCoverage"]) < float(config["data"]["minimumSymbolFileCoverage"]):
        raise RuntimeError("fundamental file coverage is below the frozen gate")
    split_source = load_json(split_path)
    splits = catalyst.split_dates(close.index, split_source)

    fast_rows: list[dict[str, Any]] = []
    for position, definition in enumerate(candidates, start=1):
        row = evaluate_candidate(
            definition, event_table, panel, timing_score, first_stage,
            returns, execution_eligible, exit_delay, splits["train"], config,
        )
        row["fastScore"] = _fast_score(row)
        fast_rows.append(row)
        print(f"financial_factor_fast {position}/{len(candidates)} {definition['name']} score={row['fastScore']:.8f}", flush=True)
    eligible_fast = [
        row for row in fast_rows
        if row["rankIc"].get("days", 0) >= int(config["selection"]["minimumTrainIcDays"])
        and math.isfinite(float(row["fastScore"]))
    ]
    selected_ids = {
        row["factorId"]
        for row in sorted(eligible_fast, key=lambda item: float(item["fastScore"]), reverse=True)[
            : int(config["generation"]["maximumFullEvaluationCandidates"])
        ]
    }
    definitions = {item["factorId"]: item for item in candidates}
    full_rows: list[dict[str, Any]] = []
    for position, factor_id in enumerate(selected_ids, start=1):
        definition = definitions[factor_id]
        full = {
            "factorId": factor_id,
            "name": definition["name"],
            "family": definition["family"],
            "trainFastScore": next(row["fastScore"] for row in fast_rows if row["factorId"] == factor_id),
        }
        for period in ("validation", "shadow"):
            full[period] = evaluate_candidate(
                definition, event_table, panel, timing_score, first_stage,
                returns, execution_eligible, exit_delay, splits[period], config,
            )
        full_rows.append(full)
        print(f"financial_factor_full {position}/{len(selected_ids)} {definition['name']}", flush=True)
    apply_multiple_testing(full_rows, len(candidates), config)
    full_rows.sort(key=lambda item: (
        bool(item["verdict"]["historicalPass"]),
        float(item["validation"]["pairedDeltaVsPrice"].get("meanDailyNetDelta") or -math.inf),
    ), reverse=True)

    generated = datetime.now(timezone.utc)
    resolved_run_id = run_id or "run_" + generated.strftime("%Y%m%dT%H%M%SZ")
    passing = [row for row in full_rows if row["verdict"]["historicalPass"]]
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": resolved_run_id,
        "generatedAt": generated.isoformat(),
        "codeVersion": CODE_VERSION,
        "configSha256": file_sha256(config_path),
        "dataAudit": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "days": len(close.index),
            "symbols": len(close.columns),
            "barInterval": "1d",
            "pointInTimeMembership": panel_audit.get("pointInTimeMembership"),
            "adjustedPrices": not bool(panel_audit.get("rawPricesUnadjusted", True)),
            "unbiasedHistoricalValidationEligible": panel_audit.get("unbiasedHistoricalValidationEligible"),
        },
        "eventAudit": event_audit,
        "generationAudit": {
            "provider": config["generation"]["provider"],
            "generatedCandidates": len(candidates),
            "trainEligibleCandidates": len(eligible_fast),
            "fullEvaluationCandidates": len(full_rows),
            "validationFeedbackReturnedToGenerator": False,
            "shadowFeedbackReturnedToGenerator": False,
            "candidateTrialsCharged": len(candidates),
        },
        "splitAudit": {
            name: [str(dates.min().date()), str(dates.max().date()), len(dates)] if len(dates) else [None, None, 0]
            for name, dates in splits.items()
        },
        "modelAudit": {
            "priceFactorCount": timing_audit["factorCount"],
            "firstStageCandidateCount": int(config["selection"]["firstStageCandidateCount"]),
            "topCount": int(config["selection"]["topCount"]),
            "priceWeight": float(config["selection"]["priceWeight"]),
            "fundamentalWeight": float(config["selection"]["fundamentalWeight"]),
            "sameSupportAndTop10Count": True,
            "reducedTradingMechanicalImprovement": False,
        },
        "candidateRegistry": candidates,
        "fastScreen": sorted(fast_rows, key=lambda item: float(item["fastScore"]), reverse=True),
        "fullEvaluation": full_rows,
        "verdict": {
            "historicalPassingCandidates": len(passing),
            "candidateIds": [row["factorId"] for row in passing],
            "eligibleForTrading": False,
            "historicalRunCanPromote": False,
            "freshForwardDaysRequired": int(config["evaluation"]["freshForwardDaysRequired"]),
            "decision": "preregister_forward_hypothesis_only" if passing else "no_validated_incremental_financial_statement_factor",
        },
        "knownLimitations": [
            "Validation and shadow windows were previously observed and can only reject.",
            "The provider may not expose every original historical restatement vintage.",
            "Historical industry membership is unavailable; liquidity bins are not full industry neutralisation.",
            "A daily bar cannot reproduce opening-auction queue priority.",
        ],
        "orders": [],
        "automaticTradingChanges": [],
    }
    output_dir = ROOT / str(config["output"]["root"]) / resolved_run_id
    safe = catalyst.json_safe(result)
    atomic_write(output_dir / "result.json", json.dumps(safe, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output_dir / "candidate_registry.json", json.dumps(catalyst.json_safe(candidates), ensure_ascii=False, indent=2) + "\n")
    atomic_write(output_dir / "report.md", render_report(safe) + "\n")
    manifest = {
        "schemaVersion": "financial_statement_factor_discovery_manifest_v1",
        "status": result["status"],
        "runId": resolved_run_id,
        "codeVersion": CODE_VERSION,
        "configPath": str(config_path),
        "configSha256": file_sha256(config_path),
        "baseConfigSha256": file_sha256(base_path),
        "timingConfigSha256": file_sha256(timing_path),
        "splitSummarySha256": file_sha256(split_path),
        "candidateRegistrySha256": digest(candidates),
        "orders": [],
    }
    atomic_write(output_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    result = run(args.config, args.run_id)
    print(json.dumps({
        "runId": result["runId"],
        "generatedCandidates": result["generationAudit"]["generatedCandidates"],
        "fullEvaluationCandidates": result["generationAudit"]["fullEvaluationCandidates"],
        "historicalPassingCandidates": result["verdict"]["historicalPassingCandidates"],
        "eligibleForTrading": False,
        "orders": [],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
