"""Market-perception-driven autonomous ETF factor research.

This module extends the existing CogAlpha research loop with immutable
Phenomenon Tickets and an append-only experiment registry. It is permanently
offline/research-only and cannot reach trading code or production artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import overfitting_guard as og  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_cogalpha_etf as core  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "research" / "perception_xalpha_mvp.json"
CODE_VERSION = "perception_xalpha_mvp_v1.0"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "perception_xalpha_mvp_v1":
        raise ValueError("unexpected Perception-XAlpha schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("system must remain research/shadow-only")
    safety = config.get("safety", {})
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all mutation and trading permissions must be false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output must remain diagnostic_only")
    generation = config["factorGeneration"]
    if any(
        generation.get(key) is not False
        for key in (
            "arbitraryPythonAllowed",
            "networkGenerationAllowed",
            "subprocessAllowed",
            "dynamicImportAllowed",
        )
    ):
        raise ValueError("unsafe code-generation capability must be disabled")
    expected_families = {
        "price_volume_coupling",
        "lead_lag",
        "conditional_beta",
        "shock_response",
    }
    if set(generation["families"]) != expected_families:
        raise ValueError("the four preregistered factor families must remain fixed")
    if int(generation["maximumCandidatesPerRun"]) > 32:
        raise ValueError("candidate search budget is not bounded")
    perception = config["perception"]
    if float(perception["residualZThreshold"]) < 2.0:
        raise ValueError("phenomenon threshold cannot be weakened below 2 sigma")
    if int(perception["minimumIndependentDays"]) < 3:
        raise ValueError("phenomena require independent trading days")
    validation = config["validation"]
    if validation.get("validationOrShadowFeedbackAllowed") is not False:
        raise ValueError("validation/shadow feedback is prohibited")
    if not validation.get("humanApprovalRequired"):
        raise ValueError("human approval boundary must remain explicit")
    if config["stateMachine"][:3] != [
        "DRAFT",
        "STATIC_VALIDATED",
        "LEAKAGE_VALIDATED",
    ]:
        raise ValueError("causal validation states cannot be bypassed")


def derived_cogalpha_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    base_path = ROOT / config["baseCogAlphaConfig"]
    base = load_json(base_path)
    autonomous.validate_config(base)
    derived = copy.deepcopy(base)
    for field in config["factorGeneration"]["additionalPastOnlyInputs"]:
        if field not in derived["search"]["allowedInputs"]:
            derived["search"]["allowedInputs"].append(field)
    derived["search"]["allowedWindows"] = list(
        config["factorGeneration"]["allowedWindows"]
    )
    derived["search"]["maximumExpressionDepth"] = int(
        config["factorGeneration"]["maximumExpressionDepth"]
    )
    return derived, base_path


def broadcast(series: pd.Series, columns: pd.Index) -> pd.DataFrame:
    values = np.repeat(series.to_numpy(dtype=float)[:, None], len(columns), axis=1)
    return pd.DataFrame(values, index=series.index, columns=columns)


def past_zscore(
    frame: pd.DataFrame, window: int, standard_deviation_floor: float = 1e-8
) -> pd.DataFrame:
    """Current observation minus a baseline formed strictly through t-1."""
    past = frame.shift(1)
    mean = past.rolling(window, min_periods=window).mean()
    std = past.rolling(window, min_periods=window).std(ddof=0)
    std = std.where(std >= standard_deviation_floor)
    return (frame - mean) / std


def enrich_panel(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    enriched = {key: value.copy() for key, value in panel.items()}
    returns = enriched["returns"]
    clean_market_input = returns.where(
        returns.abs().le(
            float(config["perception"]["maximumAbsoluteDailyReturn"])
        )
    )
    market = clean_market_input.median(axis=1, skipna=True)
    market_up = market.clip(lower=0.0)
    market_down = market.clip(upper=0.0)
    market_abs = market.abs()
    window = int(config["perception"]["rollingWindow"])
    market_shock = past_zscore(market.to_frame("market"), window)["market"]
    columns = returns.columns
    enriched["market_return"] = broadcast(market, columns)
    enriched["market_up_return"] = broadcast(market_up, columns)
    enriched["market_down_return"] = broadcast(market_down, columns)
    enriched["market_abs_return"] = broadcast(market_abs, columns)
    enriched["market_shock"] = broadcast(market_shock, columns)
    return enriched


def data_fingerprint(
    panel: dict[str, pd.DataFrame], config_text: str
) -> tuple[str, str]:
    close = panel["close"]
    snapshot = {
        "start": close.index.min().isoformat(),
        "end": close.index.max().isoformat(),
        "rows": len(close),
        "columns": sorted(map(str, close.columns)),
        "lastClose": {
            str(k): None if pd.isna(v) else round(float(v), 8)
            for k, v in close.iloc[-1].items()
        },
    }
    data_hash = digest(snapshot)
    return data_hash, hashlib.sha256(config_text.encode("utf-8")).hexdigest()


def detector_scores(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    perception = config["perception"]
    window = int(perception["rollingWindow"])
    returns = panel["returns"]
    log_volume = np.log1p(panel["volume"].clip(lower=0.0))
    price_range = (panel["high"] - panel["low"]) / panel["close"].replace(0.0, np.nan)
    valid = (
        returns.abs().le(float(perception["maximumAbsoluteDailyReturn"]))
        & price_range.ge(0.0)
        & price_range.le(float(perception["maximumDailyRangePct"]))
        & panel["volume"].gt(0.0)
        & panel["amount"].gt(0.0)
    )
    clean_returns = returns.where(valid)
    market = panel["market_return"]
    fast = int(perception["correlationFastWindow"])
    slow = int(perception["correlationSlowWindow"])
    corr_fast = clean_returns.rolling(fast, min_periods=fast).corr(market)
    corr_slow = clean_returns.rolling(slow, min_periods=slow).corr(market)
    corr_residual = corr_fast - corr_slow
    cap = float(perception["maximumResidualZForRegistry"])
    standardized = past_zscore(
        clean_returns, window, standard_deviation_floor=0.0001
    ).clip(lower=-cap, upper=cap)
    cusum_window = int(perception["cusumWindow"])
    cusum = standardized.rolling(
        cusum_window, min_periods=cusum_window
    ).sum() / math.sqrt(cusum_window)
    return {
        "return_shock": past_zscore(
            clean_returns, window, standard_deviation_floor=0.0001
        ).abs().clip(upper=cap),
        "volume_anomaly": past_zscore(
            log_volume.where(valid), window, standard_deviation_floor=0.05
        ).abs().clip(upper=cap),
        "range_anomaly": past_zscore(
            price_range.where(valid), window, standard_deviation_floor=0.0001
        ).abs().clip(upper=cap),
        "correlation_break": past_zscore(
            corr_residual, slow, standard_deviation_floor=0.01
        ).abs().clip(upper=cap),
        "cusum_shift": cusum.abs().clip(upper=cap),
    }


def data_quality_audit(
    panel: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    perception = config["perception"]
    returns = panel["returns"]
    price_range = (panel["high"] - panel["low"]) / panel["close"].replace(0.0, np.nan)
    invalid_return = returns.abs().gt(
        float(perception["maximumAbsoluteDailyReturn"])
    )
    invalid_range = price_range.lt(0.0) | price_range.gt(
        float(perception["maximumDailyRangePct"])
    )
    invalid_flow = panel["volume"].le(0.0) | panel["amount"].le(0.0)
    total = int(returns.notna().to_numpy().sum())
    excluded = invalid_return | invalid_range | invalid_flow
    return {
        "observedRows": total,
        "excludedRows": int(excluded.to_numpy().sum()),
        "excludedFraction": (
            round(float(excluded.to_numpy().sum()) / max(total, 1), 8)
        ),
        "abnormalReturnRows": int(invalid_return.to_numpy().sum()),
        "abnormalRangeRows": int(invalid_range.to_numpy().sum()),
        "nonPositiveFlowRows": int(invalid_flow.to_numpy().sum()),
        "maximumAbsoluteDailyReturn": float(
            perception["maximumAbsoluteDailyReturn"]
        ),
        "maximumDailyRangePct": float(perception["maximumDailyRangePct"]),
        "excludedBeforePhenomenonDetection": True,
    }


def build_tickets(
    scores: dict[str, pd.DataFrame],
    config: dict[str, Any],
    data_hash: str,
    config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold = float(config["perception"]["residualZThreshold"])
    minimum_days = int(config["perception"]["minimumIndependentDays"])
    minimum_events = int(config["perception"]["minimumEventCount"])
    maximum_assets = int(config["perception"]["maximumAffectedAssets"])
    tickets: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for phenomenon_id in config["perception"]["detectors"]:
        frame = scores[phenomenon_id]
        events = frame.where(frame >= threshold).stack().rename("score")
        if events.empty:
            rejected[phenomenon_id] = "no_events_above_threshold"
            continue
        days = pd.DatetimeIndex(events.index.get_level_values(0)).normalize()
        independent_days = int(days.nunique())
        event_count = int(len(events))
        if independent_days < minimum_days or event_count < minimum_events:
            rejected[phenomenon_id] = "recurrence_gate_failed"
            continue
        asset_counts = (
            events.groupby(level=1).size().sort_values(ascending=False).head(maximum_assets)
        )
        first = pd.Timestamp(events.index.get_level_values(0).min())
        last = pd.Timestamp(events.index.get_level_values(0).max())
        payload = {
            "schemaVersion": "perception_xalpha_phenomenon_ticket_v1",
            "status": "research_only",
            "phenomenonId": phenomenon_id,
            "relationType": "recurring_standardized_residual",
            "detectedAt": last.isoformat(),
            "firstObservedAt": first.isoformat(),
            "baselineValue": threshold,
            "observedValue": round(float(events.max()), 8),
            "medianObservedValue": round(float(events.median()), 8),
            "residualZScore": round(float(events.max()), 8),
            "residualCappedForRegistry": bool(
                float(events.max())
                >= float(config["perception"]["maximumResidualZForRegistry"])
            ),
            "independentDays": independent_days,
            "eventCount": event_count,
            "affectedAssets": [
                {"stockCode": str(code)[-6:], "eventCount": int(count)}
                for code, count in asset_counts.items()
            ],
            "dataSnapshotHash": data_hash,
            "configHash": config_hash,
            "dataQualityPassed": True,
            "immutable": True,
        }
        payload["ticketId"] = "ticket_" + digest(payload)[:20]
        tickets.append(payload)
    return tickets, {
        "accepted": len(tickets),
        "rejected": len(rejected),
        "rejectedReasons": rejected,
    }


def f(name: str) -> dict[str, Any]:
    return {"field": name}


def unary(op: str, arg: dict[str, Any]) -> dict[str, Any]:
    return {"unary": op, "arg": arg}


def binary(
    op: str, left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    return {"binary": op, "left": left, "right": right}


def roll(op: str, arg: dict[str, Any], window: int) -> dict[str, Any]:
    return {"rolling": op, "arg": arg, "window": window}


def factor_templates() -> dict[str, list[dict[str, Any]]]:
    returns = f("returns")
    log_volume = unary("signed_log1p", f("volume"))
    log_amount = unary("signed_log1p", f("amount"))
    lag1 = {"lag": 1, "arg": returns}
    market = f("market_return")
    up = f("market_up_return")
    down = f("market_down_return")

    def beta(market_leg: dict[str, Any]) -> dict[str, Any]:
        numerator = roll("mean", binary("mul", returns, market_leg), 20)
        denominator = roll("mean", binary("mul", market_leg, market_leg), 20)
        return binary("div", numerator, denominator)

    return {
        "price_volume_coupling": [
            binary(
                "div",
                roll("mean", unary("abs", returns), 20),
                roll("mean", log_amount, 20),
            ),
            {"corr": True, "left": returns, "right": log_volume, "window": 20},
        ],
        "lead_lag": [
            {"corr": True, "left": returns, "right": lag1, "window": 20},
            binary("sub", lag1, {"lag": 5, "arg": returns}),
        ],
        "conditional_beta": [
            beta(down),
            binary("sub", beta(up), beta(down)),
        ],
        "shock_response": [
            binary(
                "mul",
                {"lag": 1, "arg": f("market_shock")},
                unary("neg", roll("sum", returns, 3)),
            ),
            {
                "corr": True,
                "left": returns,
                "right": {"lag": 1, "arg": f("market_abs_return")},
                "window": 20,
            },
        ],
    }


FAMILY_AGENT = {
    "price_volume_coupling": "price_volume_coherence",
    "lead_lag": "lag_response",
    "conditional_beta": "volatility_asymmetry",
    "shock_response": "reversal",
}

FAMILY_TICKETS = {
    "price_volume_coupling": {"volume_anomaly", "range_anomaly"},
    "lead_lag": {"correlation_break", "return_shock"},
    "conditional_beta": {"correlation_break", "cusum_shift"},
    "shock_response": {"return_shock", "cusum_shift"},
}


def generate_candidates(
    tickets: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    templates = factor_templates()
    ticket_map = {row["phenomenonId"]: row for row in tickets}
    maximum = int(config["factorGeneration"]["maximumCandidatesPerRun"])
    per_family = int(config["factorGeneration"]["candidatesPerFamily"])
    output = []
    for family in config["factorGeneration"]["families"]:
        related = sorted(FAMILY_TICKETS[family] & set(ticket_map))
        if not related:
            continue
        source_tickets = [ticket_map[key]["ticketId"] for key in related]
        for expression in templates[family][:per_family]:
            hypothesis = copy.deepcopy(
                autonomous.ROLE_HYPOTHESES[FAMILY_AGENT[family]]
            )
            candidate = autonomous.candidate_record(
                FAMILY_AGENT[family],
                expression,
                0,
                source_tickets,
                "concrete",
                "phenomenon_ticket_deterministic",
                rationale=(
                    f"{family} candidate generated from recurring "
                    f"{', '.join(related)} tickets."
                ),
                hypothesis=hypothesis,
            )
            candidate["family"] = family
            candidate["phenomenonTicketIds"] = source_tickets
            candidate["factorId"] = "factor_" + digest(
                {
                    "family": family,
                    "tickets": source_tickets,
                    "expression": expression,
                }
            )[:20]
            candidate["id"] = candidate["factorId"]
            output.append(candidate)
            if len(output) >= maximum:
                return output
    return output


def expression_config(
    cog_config: dict[str, Any],
) -> dict[str, Any]:
    return autonomous.expression_config(cog_config)


def static_lint(
    candidate: dict[str, Any], cog_config: dict[str, Any]
) -> tuple[bool, str | None]:
    try:
        core.validate_expression(candidate["expression"], expression_config(cog_config))
    except Exception as exc:
        return False, f"dsl_rejected:{type(exc).__name__}"
    serialized = canonical(candidate["expression"]).lower()
    forbidden = [
        "__import__",
        "subprocess",
        "system(",
        "eval(",
        "exec(",
        "open(",
        "socket",
        "submitorder",
        "build_decision",
    ]
    if any(token in serialized for token in forbidden):
        return False, "forbidden_token"
    return True, None


def prefix_invariant(
    expression: dict[str, Any], panel: dict[str, pd.DataFrame]
) -> bool:
    cutoff = len(panel["close"]) - 40
    if cutoff < 300:
        return False
    full = core.evaluate_expression(expression, panel).iloc[:cutoff]
    prefix_panel = {key: value.iloc[:cutoff].copy() for key, value in panel.items()}
    prefix = core.evaluate_expression(expression, prefix_panel)
    return bool(
        np.allclose(
            full.to_numpy(dtype=float),
            prefix.to_numpy(dtype=float),
            equal_nan=True,
        )
    )


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    panel: dict[str, pd.DataFrame],
    split: autonomous.Split,
    cog_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[autonomous.Evaluation], list[dict[str, Any]]]:
    target, one_day = autonomous.target_frames(panel, cog_config)
    evaluations: list[autonomous.Evaluation] = []
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        record = {
            "factorId": candidate["factorId"],
            "family": candidate["family"],
            "ticketIds": candidate["phenomenonTicketIds"],
            "stateHistory": ["DRAFT"],
            "expression": candidate["expression"],
            "hypothesis": candidate["hypothesis"],
            "status": "DRAFT",
        }
        safe, reason = static_lint(candidate, cog_config)
        if not safe:
            record.update(status="REJECTED", rejectionReason=reason)
            record["stateHistory"].append("REJECTED")
            records.append(record)
            continue
        record["stateHistory"].append("STATIC_VALIDATED")
        if not prefix_invariant(candidate["expression"], panel):
            record.update(status="REJECTED", rejectionReason="prefix_leakage")
            record["stateHistory"].append("REJECTED")
            records.append(record)
            continue
        record["stateHistory"].append("LEAKAGE_VALIDATED")
        evaluation, reason = autonomous.evaluate_candidate(
            candidate, panel, target, one_day, split, cog_config
        )
        if evaluation is None:
            record.update(status="REJECTED", rejectionReason=reason)
            record["stateHistory"].append("REJECTED")
            records.append(record)
            continue
        train = evaluation.summary["periods"]["train"]
        validation = evaluation.summary["periods"]["validation"]
        train_ok = (
            abs(float(train["rankIc"]["mean"] or 0.0))
            >= float(config["validation"]["minimumAbsoluteTrainRankIc"])
            and float(train["rankIc"]["irAnn"] or -99.0)
            >= float(config["validation"]["minimumTrainRankIcIr"])
            and float(train["costedLongOnly"]["t"] or -99.0)
            >= float(config["validation"]["historicalValidationTMinimum"])
        )
        validation_ok = (
            float(validation["costedLongOnly"]["irAnn"] or -99.0)
            > float(config["validation"]["validationCostedIrMinimum"])
            and float(validation["rankIc"]["mean"] or -99.0) > 0.0
        )
        if train_ok:
            record["stateHistory"].append("HISTORICALLY_VALIDATED")
        record["status"] = (
            "HISTORICALLY_VALIDATED" if train_ok and validation_ok else "REJECTED"
        )
        if record["status"] == "REJECTED":
            record["stateHistory"].append("REJECTED")
            record["rejectionReason"] = (
                "validation_gate_failed" if train_ok else "train_gate_failed"
            )
        record["metrics"] = evaluation.summary["periods"]
        record["trainFitness"] = evaluation.fitness
        record["validationUsedForGeneration"] = False
        record["shadowUsedForGeneration"] = False
        records.append(record)
        evaluations.append(evaluation)
    return evaluations, records


def pair_relationships(
    evaluations: list[autonomous.Evaluation], split: autonomous.Split
) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(evaluations):
        for right in evaluations[left_index + 1 :]:
            daily = left.signal.loc[split.train].corrwith(
                right.signal.loc[split.train], axis=1, method="spearman"
            )
            correlation = float(daily.mean()) if daily.notna().any() else None
            rows.append(
                {
                    "relationshipId": "rel_"
                    + digest(
                        {
                            "left": left.candidate["factorId"],
                            "right": right.candidate["factorId"],
                        }
                    )[:20],
                    "leftFactorId": left.candidate["factorId"],
                    "rightFactorId": right.candidate["factorId"],
                    "relationship": "train_cross_section_rank_correlation",
                    "value": correlation,
                }
            )
    return rows


APPEND_ONLY_TABLES = [
    "phenomenon_tickets",
    "factors",
    "experiments",
    "validations",
    "factor_relationships",
]


def initialize_registry(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS data_snapshots "
        "(snapshot_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    for table in APPEND_ONLY_TABLES:
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(entity_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_update "
            f"BEFORE UPDATE ON {table} BEGIN "
            "SELECT RAISE(ABORT, 'append_only_update_forbidden'); END"
        )
        connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete "
            f"BEFORE DELETE ON {table} BEGIN "
            "SELECT RAISE(ABORT, 'append_only_delete_forbidden'); END"
        )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS run_registry "
        "(run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, data_hash TEXT NOT NULL, "
        "config_hash TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    connection.commit()


def insert_entity(
    connection: sqlite3.Connection, table: str, entity_id: str, payload: dict[str, Any]
) -> None:
    text = canonical(payload)
    connection.execute(
        f"INSERT OR IGNORE INTO {table} "
        "(entity_id, created_at, content_hash, payload) VALUES (?, ?, ?, ?)",
        (
            entity_id,
            datetime.now(timezone.utc).isoformat(),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text,
        ),
    )


def registry_has_run(
    connection: sqlite3.Connection, data_hash: str, config_hash: str
) -> bool:
    row = connection.execute(
        "SELECT 1 FROM run_registry WHERE data_hash=? AND config_hash=? "
        "AND status='complete' LIMIT 1",
        (data_hash, config_hash),
    ).fetchone()
    return row is not None


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Perception-XAlpha MVP research report",
        "",
        f"- run_id: `{result['runId']}`",
        f"- status: `{result['status']}`",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        f"- universe: `{result['dataAudit']['symbols']}` ETFs",
        f"- tickets accepted: `{result['ticketAudit']['accepted']}`",
        f"- factors generated/evaluated: `{result['factorAudit']['generated']}` / `{result['factorAudit']['evaluated']}`",
        f"- historically validated: `{result['factorAudit']['historicallyValidated']}`",
        f"- PBO: `{result['guards']['pbo']}`",
        "- automatic trading changes: `[]`",
        "",
        "## Phenomenon tickets",
        "",
        "| phenomenon | events | independent days | max residual z | affected assets |",
        "|---|---:|---:|---:|---:|",
    ]
    for ticket in result["tickets"]:
        lines.append(
            f"| {ticket['phenomenonId']} | {ticket['eventCount']} | "
            f"{ticket['independentDays']} | {ticket['residualZScore']} | "
            f"{len(ticket['affectedAssets'])} |"
        )
    lines.extend(
        [
            "",
            "## Factor states",
            "",
            "| factor | family | state | train RankIC | validation net IR | rejection |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for factor in result["factors"]:
        metrics = factor.get("metrics", {})
        train_rank = metrics.get("train", {}).get("rankIc", {}).get("mean")
        validation_ir = (
            metrics.get("validation", {}).get("costedLongOnly", {}).get("irAnn")
        )
        lines.append(
            f"| {factor['factorId']} | {factor['family']} | {factor['status']} | "
            f"{train_rank} | {validation_ir} | {factor.get('rejectionReason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Historical validation cannot promote a factor. Validation and shadow metrics "
            "were not returned to the generator. BORN/SHADOW require a separately "
            "preregistered fresh-forward protocol and explicit human approval.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config_path: Path,
    use_state: bool = True,
    force: bool = False,
    maximum_candidates: int | None = None,
) -> tuple[dict[str, Any], Path | None]:
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    validate_config(config)
    cog_config, base_path = derived_cogalpha_config(config)
    panel = enrich_panel(core.build_panel(cog_config), config)
    close = panel["close"]
    if close.empty:
        raise RuntimeError("no eligible daily ETF panel")
    split = autonomous.make_split(close.index, cog_config)
    data_hash, config_hash = data_fingerprint(panel, config_text)
    registry_path = ROOT / config["registry"]["sqlitePath"]
    connection: sqlite3.Connection | None = None
    if use_state:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(registry_path)
        initialize_registry(connection)
        if (
            not force
            and config["registry"]["skipWhenDataFingerprintUnchanged"]
            and registry_has_run(connection, data_hash, config_hash)
        ):
            connection.close()
            return {
                "status": "no_new_data",
                "dataHash": data_hash,
                "configHash": config_hash,
            }, None
    scores = detector_scores(panel, config)
    quality_audit = data_quality_audit(panel, config)
    tickets, ticket_audit = build_tickets(
        scores, config, data_hash, config_hash
    )
    candidates = generate_candidates(tickets, config)
    if maximum_candidates is not None:
        candidates = candidates[: max(1, int(maximum_candidates))]
    evaluations, factor_records = evaluate_candidates(
        candidates, panel, split, cog_config, config
    )
    relationships = pair_relationships(evaluations, split)
    common_dates = sorted(
        set.intersection(
            *(
                set(item.long_net.loc[split.train].dropna().index)
                for item in evaluations
            )
        )
    ) if evaluations else []
    perf_matrix = [
        item.long_net.reindex(common_dates).fillna(0.0).tolist()
        for item in evaluations
    ]
    pbo = (
        og.combinatorial_symmetric_pbo(perf_matrix, n_blocks=8)
        if len(perf_matrix) >= 2 and common_dates
        else {"pbo": None, "reason": "need at least two evaluated factors"}
    )
    validated = [
        row for row in factor_records if row["status"] == "HISTORICALLY_VALIDATED"
    ]
    run_id = (
        f"perception_xalpha_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_"
        f"{data_hash[:10]}"
    )
    result = {
        "schemaVersion": "perception_xalpha_result_v1",
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": run_id,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "codeVersion": CODE_VERSION,
        "dataHash": data_hash,
        "configHash": config_hash,
        "baseCogAlphaConfig": str(base_path.relative_to(ROOT)),
        "dataAudit": {
            "start": close.index.min().date().isoformat(),
            "end": close.index.max().date().isoformat(),
            "days": len(close),
            "symbols": len(close.columns),
            "barInterval": "1d",
            "quality": quality_audit,
        },
        "splitAudit": split.audit,
        "ticketAudit": ticket_audit,
        "factorAudit": {
            "generated": len(candidates),
            "evaluated": len(evaluations),
            "historicallyValidated": len(validated),
            "born": 0,
            "shadow": 0,
            "validationOrShadowFedBack": False,
        },
        "tickets": tickets,
        "factors": factor_records,
        "factorRelationships": relationships,
        "guards": {
            "pbo": pbo,
            "priorFamilyTrials": int(config["validation"]["priorFamilyTrials"]),
            "arbitraryPythonExecuted": False,
            "networkGenerationUsed": False,
            "subprocessUsed": False,
            "sameHistoricalWindowCannotPromote": True,
        },
        "automaticTradingChanges": [],
        "orders": [],
        "verdict": (
            "Historical candidates exist but none can be promoted. A separate fresh-forward "
            "protocol is mandatory."
            if validated
            else "No factor passed the preregistered train plus validation gates."
        ),
    }
    output = ROOT / config["registry"]["outputRoot"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "result.json", result)
    atomic_json(output / "phenomenon_tickets.json", {"tickets": tickets})
    atomic_json(output / "factor_registry_snapshot.json", {"factors": factor_records})
    atomic_json(
        output / "shadow_candidate.json",
        {
            "schemaVersion": "perception_xalpha_shadow_candidate_v1",
            "status": "research_only_not_a_trade_signal",
            "runId": run_id,
            "eligibleFactorIds": [row["factorId"] for row in validated],
            "orders": [],
            "automaticTradingChanges": [],
            "warning": "Historical research only; not connected to observation or trading.",
        },
    )
    (output / "report.md").write_text(render_report(result) + "\n", encoding="utf-8")
    if connection is not None:
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "INSERT OR IGNORE INTO data_snapshots "
            "(snapshot_id, created_at, payload) VALUES (?, ?, ?)",
            (data_hash, now, canonical(result["dataAudit"])),
        )
        for ticket in tickets:
            insert_entity(
                connection,
                "phenomenon_tickets",
                ticket["ticketId"],
                ticket,
            )
        for factor in factor_records:
            insert_entity(connection, "factors", factor["factorId"], factor)
            experiment_id = "experiment_" + digest(
                {"runId": run_id, "factorId": factor["factorId"]}
            )[:20]
            insert_entity(
                connection,
                "experiments",
                experiment_id,
                {
                    "experimentId": experiment_id,
                    "runId": run_id,
                    "factorId": factor["factorId"],
                    "stateHistory": factor["stateHistory"],
                    "trainOnlyGeneration": True,
                },
            )
            if factor.get("metrics"):
                validation_id = "validation_" + digest(
                    {"runId": run_id, "factorId": factor["factorId"]}
                )[:20]
                insert_entity(
                    connection,
                    "validations",
                    validation_id,
                    {
                        "validationId": validation_id,
                        "runId": run_id,
                        "factorId": factor["factorId"],
                        "metrics": factor["metrics"],
                        "usedForGeneration": False,
                    },
                )
        for relationship in relationships:
            insert_entity(
                connection,
                "factor_relationships",
                relationship["relationshipId"],
                relationship,
            )
        connection.execute(
            "INSERT INTO run_registry "
            "(run_id, created_at, data_hash, config_hash, status, payload) "
            "VALUES (?, ?, ?, ?, 'complete', ?)",
            (run_id, now, data_hash, config_hash, canonical(result["factorAudit"])),
        )
        connection.commit()
        connection.close()
    return result, output


def self_test() -> None:
    config = load_json(DEFAULT_CONFIG)
    validate_config(config)
    connection = sqlite3.connect(":memory:")
    initialize_registry(connection)
    insert_entity(connection, "factors", "factor_test", {"value": 1})
    failed_closed = False
    try:
        connection.execute(
            "UPDATE factors SET payload='changed' WHERE entity_id='factor_test'"
        )
    except sqlite3.DatabaseError:
        failed_closed = True
    if not failed_closed:
        raise AssertionError("append-only registry accepted an update")
    connection.close()
    print("Perception-XAlpha self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--maximum-candidates", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    result, output = run(
        args.config,
        use_state=not args.no_state,
        force=args.force,
        maximum_candidates=args.maximum_candidates,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "runId": result.get("runId"),
                "output": str(output) if output else None,
                "tickets": result.get("ticketAudit", {}).get("accepted"),
                "factors": result.get("factorAudit", {}).get("generated"),
                "historicallyValidated": result.get("factorAudit", {}).get(
                    "historicallyValidated"
                ),
                "automaticTradingChanges": result.get("automaticTradingChanges", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
