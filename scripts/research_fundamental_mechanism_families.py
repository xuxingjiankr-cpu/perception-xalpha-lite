"""Preregistered PIT fundamental-family research and frozen forward shadow ledger.

The primary policy is deliberately boring: equal component ranks inside each of four
economic mechanism families, then exactly 25% per family.  Historical performance is
never allowed to choose a primary component or fit a weight.  Hindsight and trailing
selection are both computed, on identical dates, only to disclose selection overfit.

Permanently research/shadow-only.  No order, broker, trading configuration, risk gate,
position sizing, strategy overlay or production decision path is imported or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

import overfitting_guard as overfit  # noqa: E402
import research_ashare_fundamentals as fundamentals  # noqa: E402
import research_ashare_universe as ashare  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402


SCHEMA_VERSION = "fundamental_mechanism_families_result_v1"
CODE_VERSION = "fundamental_mechanism_families_v1_20260809"
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "fundamental_mechanism_families_v1.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def git_commit() -> str | None:
    git = ROOT / ".git"
    if git.is_file():
        line = git.read_text(encoding="utf-8").strip()
        if line.startswith("gitdir:"):
            git = (ROOT / line.split(":", 1)[1].strip()).resolve()
    head = git / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref:"):
        return value
    ref = value.split(":", 1)[1].strip()
    direct = git / ref
    if direct.exists():
        return direct.read_text(encoding="utf-8").strip()
    common = git
    common_dir = git / "commondir"
    if common_dir.exists():
        common = (git / common_dir.read_text(encoding="utf-8").strip()).resolve()
    packed = common / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    return None


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "fundamental_mechanism_families_v1":
        raise ValueError("unexpected fundamental mechanism schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("study must remain research/shadow-only")
    safety = config.get("safety", {})
    mutation_keys = [key for key in safety if key.startswith("may")]
    if not mutation_keys or any(bool(safety[key]) for key in mutation_keys):
        raise ValueError("every mutation and trading permission must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("outputStatus must remain diagnostic_only")
    hypothesis = config["hypothesis"]
    if hypothesis.get("primaryPolicy") != (
        "equal_weight_within_family_then_equal_weight_across_four_families"
    ):
        raise ValueError("primary equal-family policy cannot change")
    forbidden_true = (
        "historicalFactorSelectionAllowedInPrimary",
        "historicalWeightFittingAllowed",
        "priceVolumeInputsAllowed",
        "historicalRunCanPromote",
    )
    if any(bool(hypothesis.get(key)) for key in forbidden_true):
        raise ValueError("selection, fitted weights, price-volume and promotion are forbidden")
    fundamental = config["fundamentals"]
    if fundamental.get("availabilityRule") != (
        "first_market_date_strictly_after_max_notice_update"
    ):
        raise ValueError("notice/update availability must remain strictly causal")
    if fundamental.get("reportDateUsage") != (
        "fiscal_chronology_and_same_period_comparison_only_never_availability"
    ):
        raise ValueError("reportDate may never define information availability")
    if not fundamental.get("requireAllCandidatesWithinFamily"):
        raise ValueError("missing candidates may not be reweighted")
    if not fundamental.get("requireAllFamiliesInPrimary"):
        raise ValueError("missing families may not be reweighted")
    families = config.get("families", {})
    expected = {
        "earnings_innovation",
        "growth_acceleration",
        "quality",
        "cash_flow_quality",
    }
    if set(families) != expected:
        raise ValueError("exactly four frozen mechanism families are required")
    candidate_ids: set[str] = set()
    allowed_transforms = {
        "level",
        "change_from_prior_disclosed_report",
        "same_fiscal_period_symmetric_yoy",
        "cash_margin_minus_net_margin",
    }
    for family, definition in families.items():
        candidates = definition.get("candidates", [])
        if len(candidates) < 2:
            raise ValueError(f"{family} needs at least two frozen candidates")
        for candidate in candidates:
            identifier = str(candidate.get("id") or "")
            if not identifier or identifier in candidate_ids:
                raise ValueError("candidate ids must be non-empty and globally unique")
            candidate_ids.add(identifier)
            if candidate.get("transform") not in allowed_transforms:
                raise ValueError(f"unsupported transform for {identifier}")
            if float(candidate.get("scale", 0.0)) <= 0.0:
                raise ValueError(f"candidate scale must be positive: {identifier}")
            if float(candidate.get("orientation", 0.0)) not in (-1.0, 1.0):
                raise ValueError(f"candidate orientation must be +/-1: {identifier}")
    evaluation = config["evaluation"]
    if int(evaluation["holdingTradingDays"]) != 20:
        raise ValueError("the preregistered horizon is fixed at 20 sessions")
    if not math.isclose(float(evaluation["roundTripCost"]), 0.003, abs_tol=1e-12):
        raise ValueError("round-trip cost must remain 30 bps")
    if not math.isclose(float(evaluation["topFraction"]), 0.10, abs_tol=1e-12):
        raise ValueError("top-decile construction is frozen")
    if evaluation.get("selectionExposureComparison") != "same_trailing_oos_dates":
        raise ValueError("selection exposure must compare identical dates")
    forward = config["forward"]
    if forward.get("useOutcomesForRefit") or forward.get("allowParameterChangesInPlace"):
        raise ValueError("forward outcomes cannot refit the frozen version")
    if int(forward["minimumIndependentForwardTradingDays"]) < 60:
        raise ValueError("at least 60 clean forward sessions are required")
    root = str(config["output"]["root"]).replace("\\", "/")
    if not root.startswith("outputs/edge_research/"):
        raise ValueError("outputs must remain isolated under edge_research")
    if config["output"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders must always be empty")


def verify_frozen_file(config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = ROOT / str(config[path_key])
    actual = file_sha256(path)
    expected = str(config[hash_key]).lower()
    if actual != expected:
        raise ValueError(f"frozen input changed: {path_key} {actual} != {expected}")
    return path


def _date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _symmetric_yoy(current: float, prior: float) -> float:
    denominator = abs(current) + abs(prior)
    return 0.0 if denominator <= 1e-12 else 2.0 * (current - prior) / denominator


def candidate_value(
    definition: dict[str, Any],
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    prior_year: dict[str, Any] | None,
) -> float | None:
    """Compute one frozen factor using only statements disclosed by this event date."""
    transform = str(definition["transform"])
    field = str(definition["field"])
    now = _finite(current.get(field))
    raw: float | None = None
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
    elif transform == "cash_margin_minus_net_margin":
        other = _finite(current.get(str(definition["otherField"])))
        if now is not None and other is not None:
            raw = now - other * float(definition.get("otherScale", 1.0))
    if raw is None or not math.isfinite(raw):
        return None
    return math.tanh(
        float(definition["orientation"]) * raw / float(definition["scale"])
    )


def causal_fundamental_records_for_symbol(
    rows: list[dict[str, Any]],
    market_index: pd.DatetimeIndex,
    security_id: str,
    families: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build PIT feature events; reportDate never controls availability."""
    audit: Counter[str] = Counter()
    by_market_date: dict[pd.Timestamp, dict[str, Any]] = {}
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
        candidate["_availableRaw"] = available
        incumbent = by_market_date.get(market_date)
        if incumbent is None or report > incumbent["_reportDate"]:
            if incumbent is not None:
                audit["same_day_older_report_collapsed"] += 1
            by_market_date[market_date] = candidate
        else:
            audit["same_day_older_report_collapsed"] += 1

    definitions = [
        (family, candidate)
        for family, family_definition in families.items()
        for candidate in family_definition["candidates"]
    ]
    history: dict[pd.Timestamp, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for market_date, current in sorted(by_market_date.items()):
        report = current["_reportDate"]
        if previous is not None and report <= previous["_reportDate"]:
            audit["non_advancing_report_date_skipped"] += 1
            continue
        prior_key = report - pd.DateOffset(years=1)
        prior_year = history.get(pd.Timestamp(prior_key).normalize())
        record: dict[str, Any] = {
            "eventDate": market_date,
            "securityId": security_id,
            "reportDate": report,
            "previousReportDate": (
                previous["_reportDate"] if previous is not None else None
            ),
        }
        available_count = 0
        for family, definition in definitions:
            value = candidate_value(definition, current, previous, prior_year)
            record[str(definition["id"])] = value
            if value is not None:
                available_count += 1
                audit[f"available_{family}_{definition['id']}"] += 1
        record["availableCandidateCount"] = available_count
        records.append(record)
        audit["accepted_statement_events"] += 1
        history[report] = current
        previous = current
    return records, dict(audit)


def build_event_table(
    market_index: pd.DatetimeIndex,
    columns: pd.Index,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = ROOT / str(config["fundamentals"]["root"])
    records: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    files_found = 0
    for position, security_id in enumerate(map(str, columns), start=1):
        exchange, code = security_id.split(".", 1)
        path = root / f"{exchange}_{code}.jsonl"
        rows = fundamentals.read_jsonl(path)
        if not rows:
            totals["missing_symbol_file"] += 1
            continue
        files_found += 1
        issuer_records, audit = causal_fundamental_records_for_symbol(
            rows, market_index, security_id, config["families"]
        )
        records.extend(issuer_records)
        totals.update(audit)
        if position % 500 == 0:
            print(
                f"fundamental PIT {position}/{len(columns)} events={len(records)}",
                flush=True,
            )
    table = pd.DataFrame(records)
    if not table.empty and table.duplicated(["eventDate", "securityId"]).any():
        raise RuntimeError("duplicate issuer-date event after causal collapse")
    audit = {
        "schemaVersion": "fundamental_mechanism_event_audit_v1",
        "status": "research_only",
        "root": str(root),
        "symbolsRequested": len(columns),
        "filesFound": files_found,
        "symbolFileCoverage": round(files_found / max(1, len(columns)), 8),
        "statementEvents": len(table),
        "eventDateStart": (
            str(pd.Timestamp(table["eventDate"].min()).date()) if len(table) else None
        ),
        "eventDateEnd": (
            str(pd.Timestamp(table["eventDate"].max()).date()) if len(table) else None
        ),
        "processingCounts": dict(totals),
        "availabilityRule": config["fundamentals"]["availabilityRule"],
        "reportDateUsedForAvailability": False,
        "futureRowsWrittenBack": 0,
    }
    return table, audit


def factor_frame(
    event_table: pd.DataFrame,
    factor_id: str,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    max_age: int,
) -> pd.DataFrame:
    if event_table.empty or factor_id not in event_table:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    sparse = event_table.pivot(
        index="eventDate", columns="securityId", values=factor_id
    )
    return (
        sparse.reindex(index=index, columns=columns)
        .ffill(limit=max_age)
        .astype("float32")
    )


def equal_rank_composite(
    ranks: dict[str, pd.DataFrame], base_eligible: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight ranks without silently reweighting missing components."""
    if not ranks:
        raise ValueError("at least one rank is required")
    common = base_eligible.copy().fillna(False)
    for rank in ranks.values():
        common &= rank.notna()
    composite = sum(rank.where(common) for rank in ranks.values()) / len(ranks)
    return composite, common


def horizon_label(
    panel: dict[str, pd.DataFrame], eligible: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Buy open[t+1], sell open[t+1+h], with sealed-limit/halt executability."""
    open_, high, low, close = (
        panel["open"],
        panel["high"],
        panel["low"],
        panel["close"],
    )
    forward = open_.shift(-(horizon + 1)) / open_.shift(-1) - 1.0
    sealed = high.eq(low) & high.notna()
    prior = close.shift(1)
    buyable = ~(sealed & close.gt(prior)) & panel["volume"].fillna(0.0).gt(0.0)
    sellable = ~(sealed & close.lt(prior)) & panel["volume"].fillna(0.0).gt(0.0)
    return forward.where(
        buyable.shift(-1) & sellable.shift(-(horizon + 1)) & eligible
    )


def portfolio_series(
    signal: pd.DataFrame,
    label: pd.DataFrame,
    eligible: pd.DataFrame,
    horizon: int,
    top_fraction: float,
    round_trip_cost: float,
    minimum_cross_section: int,
) -> dict[str, pd.Series] | None:
    """Same overlapping-tranche, turnover-costed book as the independent audit."""
    signal = signal.where(eligible).replace([np.inf, -np.inf], np.nan)
    usable = (signal.notna() & label.notna()).sum(axis=1) >= minimum_cross_section
    if int(usable.sum()) < 60:
        return None
    ranks = signal.rank(axis=1, pct=True)
    top = ranks.ge(1.0 - top_fraction) & usable.to_numpy()[:, None]
    weights = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if horizon > 1:
        weights = weights.rolling(horizon).mean().fillna(0.0)
        weights = weights.div(
            weights.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0.0)
    daily_label = label / horizon
    book = (weights * daily_label).sum(axis=1, min_count=1)
    benchmark = daily_label.where(eligible)[usable].mean(axis=1)
    gross_excess = (book - benchmark)[usable].dropna()
    turnover = (weights.diff().abs().sum(axis=1) / 2.0).reindex(gross_excess.index)
    cost = turnover * round_trip_cost
    return {
        "grossExcess": gross_excess,
        "turnover": turnover,
        "cost": cost,
        "net": gross_excess - cost,
    }


def series_stats(parts: dict[str, pd.Series] | None) -> dict[str, Any]:
    if parts is None or len(parts["net"]) < 2:
        return {"n": 0}
    net = parts["net"].dropna()
    gross = parts["grossExcess"].reindex(net.index)
    cost = parts["cost"].reindex(net.index)
    equity = (1.0 + net).cumprod()
    std = float(net.std(ddof=1))
    return {
        "n": len(net),
        "grossMeanBpsPerDay": round(float(gross.mean()) * 1e4, 4),
        "costMeanBpsPerDay": round(float(cost.mean()) * 1e4, 4),
        "netMeanBpsPerDay": round(float(net.mean()) * 1e4, 4),
        "netInformationRatioAnnualized": (
            round(float(net.mean()) / std * math.sqrt(244), 4) if std > 0 else None
        ),
        "netWinRate": round(float(net.gt(0.0).mean()), 6),
        "maxDrawdownPct": round(
            float((equity / equity.cummax() - 1.0).min()) * 100.0, 4
        ),
        "averageOneWayTurnover": round(float(parts["turnover"].mean()), 6),
    }


def parts_from_net(net: pd.Series) -> dict[str, pd.Series]:
    zero = pd.Series(0.0, index=net.index)
    return {"net": net, "grossExcess": net, "cost": zero, "turnover": zero}


def selection_exposure(
    candidate_series: dict[str, pd.Series], config: dict[str, Any]
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    first_year = int(evaluation["firstTrailingSelectionYear"])
    minimum = int(evaluation["minimumTrailingDays"])
    if not candidate_series:
        return {"status": "no_candidates"}

    def information_ratio(values: pd.Series) -> float | None:
        values = values.dropna()
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return float(values.mean() / std * math.sqrt(244)) if std > 0 else None

    full_scores = {
        name: information_ratio(values) for name, values in candidate_series.items()
    }
    finite = [(score, name) for name, score in full_scores.items() if score is not None]
    if not finite:
        return {"status": "no_finite_candidate_ir"}
    hindsight_name = max(finite)[1]
    all_dates = sorted(set().union(*(set(values.index) for values in candidate_series.values())))
    years = sorted({pd.Timestamp(date).year for date in all_dates if pd.Timestamp(date).year >= first_year})
    trailing_parts: list[pd.Series] = []
    hindsight_parts: list[pd.Series] = []
    picks: dict[str, str] = {}
    for year in years:
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        ranked: list[tuple[float, str]] = []
        for name, values in candidate_series.items():
            trailing = values[values.index < start].dropna()
            if len(trailing) < minimum:
                continue
            score = information_ratio(trailing)
            if score is not None:
                ranked.append((score, name))
        if not ranked:
            continue
        chosen = max(ranked)[1]
        picks[str(year)] = chosen
        trailing = candidate_series[chosen]
        hindsight = candidate_series[hindsight_name]
        joined = pd.concat(
            [
                trailing[(trailing.index >= start) & (trailing.index < end)].rename("trailing"),
                hindsight[(hindsight.index >= start) & (hindsight.index < end)].rename("hindsight"),
            ],
            axis=1,
        ).dropna()
        if len(joined):
            trailing_parts.append(joined["trailing"])
            hindsight_parts.append(joined["hindsight"])
    if not trailing_parts:
        return {
            "status": "insufficient_trailing_history",
            "hindsightSelected": hindsight_name,
        }
    trailing = pd.concat(trailing_parts).sort_index()
    hindsight = pd.concat(hindsight_parts).sort_index()
    common = trailing.index.intersection(hindsight.index)
    trailing, hindsight = trailing.reindex(common), hindsight.reindex(common)
    trailing_stats = series_stats(parts_from_net(trailing))
    hindsight_stats = series_stats(parts_from_net(hindsight))
    return {
        "status": "diagnostic_only",
        "sameEvaluationDates": True,
        "evaluationDateStart": str(common.min().date()),
        "evaluationDateEnd": str(common.max().date()),
        "evaluationDays": len(common),
        "hindsightSelected": hindsight_name,
        "trailingPicksByYear": picks,
        "hindsightSelectedNet": hindsight_stats,
        "trailingSelectedNet": trailing_stats,
        "selectionOverfitExposureBpsPerDay": round(
            float(hindsight.mean() - trailing.mean()) * 1e4, 4
        ),
    }


def load_panel(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    base_path = verify_frozen_file(
        config, "baseResearchConfig", "baseResearchConfigFileSha256"
    )
    verify_frozen_file(config, "handoff", "handoffFileSha256")
    base = load_json(base_path)
    cog = load_json(ROOT / str(base["baseCogAlphaConfig"]))
    panel, audit = ashare.build_panel(base["assetUniverse"], cog["data"])
    if not audit.get("unbiasedHistoricalValidationEligible"):
        raise RuntimeError("clean PIT-adjusted panel failed closed")
    return panel, audit


def evaluate(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    panel, panel_audit = load_panel(config)
    close = panel["close"]
    eligible = panel["eligible"].fillna(False)
    event_table, event_audit = build_event_table(close.index, close.columns, config)
    if event_table.empty:
        raise RuntimeError("no causal fundamental events")
    horizon = int(config["evaluation"]["holdingTradingDays"])
    label = horizon_label(panel, eligible, horizon)
    max_age = int(config["fundamentals"]["maximumSignalAgeTradingDays"])
    bins = int(config["fundamentals"]["liquidityNeutraliseBins"])
    top_fraction = float(config["evaluation"]["topFraction"])
    cost = float(config["evaluation"]["roundTripCost"])
    minimum_cross_section = int(config["evaluation"]["minimumCrossSection"])
    family_scores: dict[str, pd.DataFrame] = {}
    family_results: dict[str, Any] = {}
    all_candidate_net: dict[str, pd.Series] = {}

    for family, definition in config["families"].items():
        print(f"family {family}", flush=True)
        ranks: dict[str, pd.DataFrame] = {}
        for candidate in definition["candidates"]:
            identifier = str(candidate["id"])
            raw = factor_frame(event_table, identifier, close.index, close.columns, max_age)
            neutral = autonomous.size_neutralise(raw.where(eligible), panel, bins)
            ranks[identifier] = neutral.where(eligible).rank(
                axis=1, pct=True, method="average"
            ).astype("float32")
        family_score, common = equal_rank_composite(ranks, eligible)
        family_score = family_score.rank(axis=1, pct=True, method="average").astype("float32")
        family_scores[family] = family_score
        candidate_metrics: dict[str, Any] = {}
        candidate_net: dict[str, pd.Series] = {}
        for identifier, rank in ranks.items():
            parts = portfolio_series(
                rank, label, common, horizon, top_fraction, cost, minimum_cross_section
            )
            candidate_metrics[identifier] = series_stats(parts)
            if parts is not None:
                candidate_net[identifier] = parts["net"]
                all_candidate_net[f"{family}/{identifier}"] = parts["net"]
        family_parts = portfolio_series(
            family_score, label, common, horizon, top_fraction, cost, minimum_cross_section
        )
        family_metrics = series_stats(family_parts)
        trial_count = len(definition["candidates"])
        dsr = overfit.deflated_significance_note(
            n_trials=trial_count,
            observed_sharpe=float(family_metrics.get("netInformationRatioAnnualized") or 0.0),
            n_obs=int(family_metrics.get("n") or 1),
        )
        family_results[family] = {
            "economicMechanism": definition["economicMechanism"],
            "trialLedger": {
                "scope": family,
                "trials": trial_count,
                "pooledWithPriceVolumeTrials": False,
            },
            "commonSupportMeanStocks": round(float(common.sum(axis=1).mean()), 2),
            "candidateMetrics": candidate_metrics,
            "equalCandidateFamilyMetrics": family_metrics,
            "selectionExposure": selection_exposure(candidate_net, config),
            "deflatedSharpeDiagnostic": dsr,
        }
        del ranks

    primary, primary_common = equal_rank_composite(family_scores, eligible)
    primary = primary.rank(axis=1, pct=True, method="average").astype("float32")
    primary_parts = portfolio_series(
        primary,
        label,
        primary_common,
        horizon,
        top_fraction,
        cost,
        minimum_cross_section,
    )
    primary_metrics = series_stats(primary_parts)
    primary_dsr = overfit.deflated_significance_note(
        n_trials=1,
        observed_sharpe=float(primary_metrics.get("netInformationRatioAnnualized") or 0.0),
        n_obs=int(primary_metrics.get("n") or 1),
    )
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "codeVersion": CODE_VERSION,
        "status": "research_only_shadow_only_unvalidated",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "gitCommit": git_commit(),
        "configSha256": digest(config),
        "dataAudit": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "tradingDays": len(close.index),
            "symbols": len(close.columns),
            "panel": panel_audit,
            "fundamentals": event_audit,
        },
        "label": {
            "signal": "safe disclosure session close",
            "entry": "next buyable open",
            "exit": f"sellable open {horizon} sessions after entry",
            "holdingTradingDays": horizon,
            "roundTripCost": cost,
        },
        "primaryPolicy": {
            "definition": config["hypothesis"]["primaryPolicy"],
            "withinFamilyWeights": "equal",
            "familyWeights": {family: 0.25 for family in family_scores},
            "commonSupportMeanStocks": round(float(primary_common.sum(axis=1).mean()), 2),
            "metrics": primary_metrics,
            "deflatedSharpeDiagnostic": primary_dsr,
            "trialLedger": {
                "scope": "four_family_fixed_integration",
                "trials": 1,
                "pooledWithComponentOrPriceVolumeTrials": False,
            },
        },
        "families": family_results,
        "allCandidateSelectionExposure": selection_exposure(all_candidate_net, config),
        "overfitExposureRequirementSatisfied": True,
        "historicalWindowPreviouslyViewed": True,
        "validatedFactorCount": 0,
        "eligibleForTrading": False,
        "freshForwardRequired": True,
        "orders": [],
        "automaticTradingChanges": [],
        "knownLimitations": config["knownLimitations"],
    }
    return result, {"primary": primary, **family_scores}


def render_historical_report(result: dict[str, Any]) -> str:
    primary = result["primaryPolicy"]
    lines = [
        "# PIT fundamental mechanism families V1",
        "",
        "> **Research-only / shadow-only / unvalidated. Validated factors: 0.**",
        "",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- universe: `{result['dataAudit']['symbols']}` clean PIT SH/SZ stocks",
        f"- causal statement events: `{result['dataAudit']['fundamentals']['statementEvents']}`",
        f"- horizon / cost: `{result['label']['holdingTradingDays']}` sessions / "
        f"`{10000 * result['label']['roundTripCost']:.0f}` bps round trip",
        "- primary construction: equal candidates inside each family; 25% per family",
        "- historical results cannot promote or change a weight",
        "",
        "## Fixed equal-family result",
        "",
        "| n | gross bps/day | cost bps/day | net bps/day | net IR | max DD |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = primary["metrics"]
    lines.append(
        f"| {metrics.get('n', 0)} | {metrics.get('grossMeanBpsPerDay', 'n/a')} | "
        f"{metrics.get('costMeanBpsPerDay', 'n/a')} | "
        f"{metrics.get('netMeanBpsPerDay', 'n/a')} | "
        f"{metrics.get('netInformationRatioAnnualized', 'n/a')} | "
        f"{metrics.get('maxDrawdownPct', 'n/a')}% |"
    )
    lines.extend(
        [
            "",
            "## Mechanism families and selection exposure",
            "",
            "| family | trials | equal-family net bps/day | hindsight net | trailing net | exposure bps/day | DSR |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for family, row in result["families"].items():
        exposure = row["selectionExposure"]
        hindsight = exposure.get("hindsightSelectedNet", {}).get("netMeanBpsPerDay")
        trailing = exposure.get("trailingSelectedNet", {}).get("netMeanBpsPerDay")
        lines.append(
            f"| {family} | {row['trialLedger']['trials']} | "
            f"{row['equalCandidateFamilyMetrics'].get('netMeanBpsPerDay', 'n/a')} | "
            f"{hindsight if hindsight is not None else 'n/a'} | "
            f"{trailing if trailing is not None else 'n/a'} | "
            f"{exposure.get('selectionOverfitExposureBpsPerDay', 'n/a')} | "
            f"{row['deflatedSharpeDiagnostic'].get('flag', 'n/a')} |"
        )
    all_exposure = result["allCandidateSelectionExposure"]
    lines.extend(
        [
            "",
            "## Required selection-overfit disclosure",
            "",
            f"- hindsight component: `{all_exposure.get('hindsightSelected', 'n/a')}`",
            f"- hindsight net: `{all_exposure.get('hindsightSelectedNet', {}).get('netMeanBpsPerDay', 'n/a')}` bps/day",
            f"- trailing-only net: `{all_exposure.get('trailingSelectedNet', {}).get('netMeanBpsPerDay', 'n/a')}` bps/day",
            f"- overfit exposure: `{all_exposure.get('selectionOverfitExposureBpsPerDay', 'n/a')}` bps/day",
            "- both numbers use exactly the same trailing-OOS dates and 30 bps cost model",
            "",
            "## Causal and safety boundary",
            "",
            "`noticeDate` is mandatory. Information becomes visible only on the first market "
            "session strictly after `max(noticeDate, updateDate)`. `reportDate` only orders "
            "fiscal periods and never controls availability. This run created no order and "
            "changed no trading, sizing, position, overlay, broker, risk-gate or decision file.",
            "",
            "No historical result is clean final OOS evidence. The validated-factor count "
            "remains zero; only post-2026-08-07 frozen forward observations can change the "
            "research assessment after separate review.",
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
        value = date.normalize() + pd.offsets.BDay(1)
        return str(pd.Timestamp(value).date())


def master_names(config: dict[str, Any]) -> dict[str, str]:
    base = load_json(ROOT / str(config["baseResearchConfig"]))
    rows = ashare.read_jsonl(ROOT / str(base["assetUniverse"]["masterPath"]))
    return {
        str(row.get("securityId")): str(row.get("name") or "")
        for row in rows
        if row.get("securityId")
    }


def write_forward_snapshot(
    config: dict[str, Any], scores: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    primary = scores["primary"]
    as_of = pd.Timestamp(primary.index.max()).normalize()
    cutoff = pd.Timestamp(config["forward"]["cleanEvidenceStartsStrictlyAfter"])
    if as_of <= cutoff:
        return {
            "status": "waiting_for_first_clean_forward_panel_date",
            "latestPanelDate": str(as_of.date()),
            "cleanEvidenceStartsStrictlyAfter": str(cutoff.date()),
            "officialSnapshotWritten": False,
        }
    available = primary.loc[as_of].dropna().sort_values(ascending=False)
    top_count = int(config["forward"]["topCount"])
    selected = available.head(top_count)
    names = master_names(config)
    rows: list[dict[str, Any]] = []
    for rank, (security_id, score) in enumerate(selected.items(), start=1):
        exchange, stock_code = str(security_id).split(".", 1)
        row = {
            "rank": rank,
            "securityId": str(security_id),
            "exchange": exchange,
            "stockCode": stock_code,
            "name": names.get(str(security_id), ""),
            "equalFamilyScore": round(float(score), 8),
            "familyScores": {
                family: round(float(frame.loc[as_of, security_id]), 8)
                for family, frame in scores.items()
                if family != "primary" and pd.notna(frame.loc[as_of, security_id])
            },
        }
        rows.append(row)
    snapshot = {
        "schemaVersion": "fundamental_mechanism_forward_prediction_v1",
        "status": "research_only_shadow_only_unvalidated",
        "modelVersion": config["forward"]["frozenModelVersion"],
        "asOfDate": str(as_of.date()),
        "effectiveDate": next_xshg_session(as_of),
        "generatedAt": datetime.now().astimezone().isoformat(),
        "definition": "equal candidates within family; exactly 25% per family",
        "historicalWeightsUsed": False,
        "expectedReturnClaim": None,
        "lossProbabilityClaim": None,
        "tradeInstruction": False,
        "selections": rows,
        "validatedFactorCount": 0,
        "orders": [],
    }
    root = ROOT / str(config["forward"]["outputRoot"])
    daily_path = root / f"predictions_{as_of:%Y-%m-%d}.json"
    payload = json_safe(snapshot)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if daily_path.exists():
        existing = load_json(daily_path)
        comparable_existing = dict(existing)
        comparable_new = dict(payload)
        comparable_existing.pop("generatedAt", None)
        comparable_new.pop("generatedAt", None)
        if comparable_existing != comparable_new:
            raise RuntimeError("immutable forward snapshot differs for existing asOfDate")
    else:
        atomic_write(daily_path, content)
    ledger_path = root / "forward_predictions.jsonl"
    existing_lines = ledger_path.read_text(encoding="utf-8").splitlines() if ledger_path.exists() else []
    existing_dates = {
        str(json.loads(line).get("asOfDate"))
        for line in existing_lines
        if line.strip()
    }
    if str(as_of.date()) not in existing_dates:
        atomic_write(
            ledger_path,
            "\n".join(existing_lines + [canonical(payload)]) + "\n",
        )
    report_path = root / f"report_{as_of:%Y-%m-%d}.md"
    report = "\n".join(
        [
            f"# Fundamental mechanism forward shadow — {as_of:%Y-%m-%d}",
            "",
            "> Research-only, unvalidated, not a trading recommendation.",
            "",
            f"- effective session: `{snapshot['effectiveDate']}`",
            f"- frozen version: `{snapshot['modelVersion']}`",
            f"- selected: `{len(rows)}`",
            "- expected-return and loss-probability claims: `null` until clean forward validation",
            "- orders: `[]`",
            "",
            "| rank | security | name | equal-family score |",
            "|---:|---|---|---:|",
            *[
                f"| {row['rank']} | {row['securityId']} | {row['name']} | {row['equalFamilyScore']:.6f} |"
                for row in rows
            ],
            "",
        ]
    )
    atomic_write(report_path, report)
    return {
        "status": "forward_shadow_written",
        "latestPanelDate": str(as_of.date()),
        "effectiveDate": snapshot["effectiveDate"],
        "officialSnapshotWritten": True,
        "snapshot": str(daily_path),
        "ledger": str(ledger_path),
        "selected": len(rows),
    }


def run(config_path: Path, mode: str, run_id: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    result, scores = evaluate(config)
    output_root = ROOT / str(config["output"]["root"])
    response: dict[str, Any] = {"mode": mode}
    if mode in {"historical", "both"}:
        identifier = run_id or datetime.now().astimezone().strftime("run_%Y%m%dT%H%M%S%z")
        run_root = output_root / identifier
        result["runId"] = identifier
        atomic_write(
            run_root / "result.json",
            json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write(run_root / "report.md", render_historical_report(result))
        response["historical"] = {
            "runId": identifier,
            "result": str(run_root / "result.json"),
            "report": str(run_root / "report.md"),
            "validatedFactorCount": 0,
            "eligibleForTrading": False,
        }
    if mode in {"forward", "both"}:
        response["forward"] = write_forward_snapshot(config, scores)
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
