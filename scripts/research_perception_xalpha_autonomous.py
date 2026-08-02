"""Autonomous, mechanism-first A-share instrument factor discovery.

The engine converts immutable market-phenomenon tickets into falsifiable mechanism
hypotheses, causal DSL programs and Primary/Counter/Placebo bundles. Evolution uses
train-only feedback. Validation and shadow remain quarantined and can never alter
the paper-trading system.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import overfitting_guard as og  # noqa: E402
import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_cogalpha_etf as core  # noqa: E402
import research_perception_xalpha as perception  # noqa: E402

DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "perception_xalpha_autonomous_v2.json"
)
CODE_VERSION = "perception_xalpha_autonomous_v2.3"


REJECTION_REASONS = {
    "STATIC_DSL_REJECTED",
    "PREFIX_LEAKAGE",
    "TOO_MANY_NAN",
    "INSUFFICIENT_CROSS_SECTION",
    "INSUFFICIENT_TRAIN_DAYS",
    "WEAK_TRAIN_RANK_IC",
    "WEAK_TRAIN_RANK_IC_IR",
    "NEGATIVE_COSTED_TRAIN_IR",
    "NEGATIVE_GROSS_TRAIN_IR",
    "DUPLICATE_BEHAVIOR",
    "FULL_EVALUATION_FAILED",
    "COUNTER_NOT_BEATEN",
    "PLACEBO_NOT_BEATEN",
    "PURGED_WALK_FORWARD_FAILED",
    "VALIDATION_RANK_IC_FAILED",
    "VALIDATION_COSTED_IR_FAILED",
    "VALIDATION_GROSS_IR_FAILED",
    "PROJECT_PBO_FAILED",
    "MULTIPLE_TESTING_DSR_FAILED",
}


ARCHETYPES: dict[str, dict[str, Any]] = {
    "order_splitting": {
        "agent": "daily_trend",
        "counterAgent": "reversal",
        "phenomena": {
            "volume_anomaly",
            "return_shock",
            "correlation_break",
        },
        "mechanism": (
            "Large parent orders are split across sessions because available instrument "
            "liquidity is finite."
        ),
        "forcedTrader": "benchmark-sensitive institutions and execution algorithms",
        "persistence": "unfinished parent orders preserve directional pressure",
        "falsifiablePrediction": (
            "past price-volume coherence should predict next-open relative returns "
            "better than a reversal counter or delayed placebo"
        ),
        "failureCondition": "abnormally high market volatility interrupts execution",
    },
    "liquidity_reversal": {
        "agent": "reversal",
        "counterAgent": "daily_trend",
        "phenomena": {
            "return_shock",
            "range_anomaly",
            "volume_anomaly",
        },
        "mechanism": (
            "Urgent liquidity demand temporarily pushes instrument prices away from the "
            "cross-sectional clearing level."
        ),
        "forcedTrader": "urgent sellers and inventory-constrained liquidity providers",
        "persistence": "dealer inventory recovers over several sessions",
        "falsifiablePrediction": (
            "past shock magnitude should predict reversal after costs and outperform "
            "a continuation counter"
        ),
        "failureCondition": "systemic risk shocks prevent inventory normalization",
    },
    "risk_budgeting": {
        "agent": "volatility_regime",
        "counterAgent": "stability",
        "phenomena": {
            "range_anomaly",
            "cusum_shift",
            "correlation_break",
        },
        "mechanism": (
            "Volatility-targeting and drawdown mandates adjust risk with delayed "
            "estimates, creating predictable allocation pressure."
        ),
        "forcedTrader": "risk-control, volatility-targeting and leveraged portfolios",
        "persistence": "risk estimates and mandate adjustments update gradually",
        "falsifiablePrediction": (
            "past volatility-state changes should explain relative returns beyond a "
            "stable-market counter"
        ),
        "failureCondition": "volatility normalizes before mandates rebalance",
    },
    "information_diffusion": {
        "agent": "lag_response",
        "counterAgent": "bar_shape",
        "phenomena": {
            "correlation_break",
            "return_shock",
            "cusum_shift",
        },
        "mechanism": (
            "Related instruments incorporate common information at different speeds because "
            "attention and liquidity differ."
        ),
        "forcedTrader": "attention-constrained and rule-based investors",
        "persistence": "information reaches products and investor groups asynchronously",
        "falsifiablePrediction": (
            "strictly lagged response structure should outperform contemporaneous "
            "bar-shape evidence and a delayed placebo"
        ),
        "failureCondition": "common news is incorporated synchronously at the open",
    },
    "crowding_unwind": {
        "agent": "herding_proxy",
        "counterAgent": "price_volume_coherence",
        "phenomena": {
            "correlation_break",
            "volume_anomaly",
            "cusum_shift",
        },
        "mechanism": (
            "Crowded thematic positions create common price-volume pressure and "
            "nonlinear unwind risk."
        ),
        "forcedTrader": "crowded systematic and thematic portfolios",
        "persistence": "common constraints force staged rather than instantaneous exits",
        "falsifiablePrediction": (
            "past crowding proxies should predict relative payoff asymmetry beyond "
            "ordinary price-volume coherence"
        ),
        "failureCondition": "fresh inflows absorb the attempted unwind",
    },
    "benchmark_rebalancing": {
        "agent": "market_cycle",
        "counterAgent": "reversal",
        "phenomena": {
            "volume_anomaly",
            "correlation_break",
            "range_anomaly",
        },
        "mechanism": (
            "Index and strategic-allocation rebalancing creates predictable, "
            "multi-session demand around constrained liquidity."
        ),
        "forcedTrader": "index trackers and allocation rebalancers",
        "persistence": "mandated trades are distributed to reduce market impact",
        "falsifiablePrediction": (
            "slow cycle and volume structure should beat an idiosyncratic reversal "
            "counter after turnover costs"
        ),
        "failureCondition": "offsetting creation-redemption flow neutralizes demand",
    },
    "volatility_feedback": {
        "agent": "volatility_asymmetry",
        "counterAgent": "daily_trend",
        "phenomena": {
            "return_shock",
            "range_anomaly",
            "cusum_shift",
        },
        "mechanism": (
            "Price declines raise measured risk, forcing further de-risking and "
            "changing subsequent liquidity asymmetrically."
        ),
        "forcedTrader": "loss-sensitive and volatility-controlled portfolios",
        "persistence": "risk constraints bind at heterogeneous thresholds",
        "falsifiablePrediction": (
            "downside volatility asymmetry should add information beyond unconditional "
            "trend and a delayed placebo"
        ),
        "failureCondition": "policy or liquidity intervention breaks feedback",
    },
}


@dataclass
class FastResult:
    candidate: dict[str, Any]
    signal: pd.DataFrame
    long_net: pd.Series
    metrics: dict[str, Any]
    fitness: float
    behavior: dict[str, Any]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def expression_text(expression: dict[str, Any]) -> str:
    """Stable readable rendering of the audited factor DSL."""
    if "field" in expression:
        return str(expression["field"])
    if "unary" in expression:
        return f"{expression['unary']}({expression_text(expression['arg'])})"
    if "binary" in expression:
        return (
            f"{expression['binary']}("
            f"{expression_text(expression['left'])},"
            f"{expression_text(expression['right'])})"
        )
    if "lag" in expression:
        return f"lag({expression_text(expression['arg'])},{int(expression['lag'])})"
    if "rolling" in expression:
        return (
            f"rolling_{expression['rolling']}("
            f"{expression_text(expression['arg'])},{int(expression['window'])})"
        )
    if "corr" in expression:
        return (
            f"rolling_corr({expression_text(expression['left'])},"
            f"{expression_text(expression['right'])},{int(expression['window'])})"
        )
    for operator in ("zscore", "drawdown", "range_position"):
        if operator in expression:
            return (
                f"{operator}({expression_text(expression['arg'])},"
                f"{int(expression['window'])})"
            )
    return canonical(expression)


def append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete UTF-8 JSON line so an active cycle is inspectable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    if config.get("schemaVersion") != "perception_xalpha_autonomous_v2":
        raise ValueError("unexpected autonomous Perception-XAlpha schema")
    if config.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("the autonomous system must remain research/shadow-only")
    safety = config.get("safety", {})
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all trading and mutation permissions must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if set(config["mechanismLibrary"]["archetypes"]) != set(ARCHETYPES):
        raise ValueError("the preregistered mechanism library changed")
    synthesis = config["synthesis"]
    if synthesis.get("provider") != "deterministic_local_grammar":
        raise ValueError("v2 must not depend on a hosted generator")
    if synthesis.get("remoteApiAllowed") is not False:
        raise ValueError("remote factor generation is prohibited")
    if synthesis.get("arbitraryPythonAllowed") is not False:
        raise ValueError("generated Python is prohibited")
    if int(synthesis["maximumPrimaryCandidatesPerCycle"]) > 512:
        raise ValueError("search budget is not bounded")
    if int(synthesis["maximumStage2Bundles"]) > 32:
        raise ValueError("Stage-2 budget is not bounded")
    structured_library = synthesis.get("structuredSeedLibrary")
    if structured_library not in {
        None,
        "a_share_tradeable_v1",
        "a_share_predictive_v2",
        "a_share_fundamental_pit_v1",
    }:
        raise ValueError("unknown structured seed library")
    if int(synthesis.get("maximumStructuredSeedsPerQuestion", 0)) > 16:
        raise ValueError("structured seed budget is not bounded")
    if not math.isclose(
        sum(float(value) for value in synthesis["operatorWeights"].values()),
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("evolution operator weights must sum to one")
    director = config["researchDirector"]
    if (
        director.get("validationFeedbackAllowed") is not False
        or director.get("shadowFeedbackAllowed") is not False
    ):
        raise ValueError("validation/shadow feedback is forbidden")
    full = config["fullEvaluation"]
    objective = config.get("discoveryObjective", {})
    mode = str(objective.get("mode", "costed_tradeability"))
    if mode not in {"costed_tradeability", "gross_predictive"}:
        raise ValueError("unknown discovery objective")
    if mode == "gross_predictive":
        if int(objective.get("targetCredibleFactorsPerCycle", 0)) not in range(1, 11):
            raise ValueError("gross discovery target must be between one and ten")
        if objective.get("costMetricsRemainMandatory") is not True:
            raise ValueError("gross discovery must still report cost stress")
    if full.get("validationOrShadowMetricsReturnedToGenerator") is not False:
        raise ValueError("validation/shadow must be quarantined")
    if full.get("historicalRunCanPromote") is not False:
        raise ValueError("historical research cannot promote")
    if not full.get("humanApprovalRequired"):
        raise ValueError("human approval must remain mandatory")
    universe = config.get("assetUniverse", {"kind": "etf"})
    universe_kind = universe.get("kind", "etf")
    registered_cost = 0.003 if universe_kind == "all_a_shares" else 0.00155
    if not math.isclose(
        float(full["roundTripCost"]), registered_cost, abs_tol=1e-12
    ):
        raise ValueError(
            "registered round-trip cost differs from the frozen universe cost"
        )
    if universe_kind == "all_a_shares":
        if set(universe.get("exchanges", [])) != {"SH", "SZ", "BJ"}:
            raise ValueError("all-A-share research must include SH, SZ and BJ")
        if universe.get("pointInTimeMembershipAvailable") is not False:
            raise ValueError("current-master survivorship limitation must be explicit")
        for key in ("masterPath", "barsRoot"):
            value = str(universe.get(key, "")).replace("\\", "/").lower()
            if not value.startswith("data/market/ashare_research/"):
                raise ValueError("all-A-share inputs must stay in the research data root")
        fundamental = universe.get("fundamentalData")
        if fundamental is not None:
            root = str(fundamental.get("root", "")).replace("\\", "/").lower()
            if not root.startswith("data/market/ashare_research/fundamentals_pit"):
                raise ValueError("fundamental inputs must stay in the research data root")
            fields = list(fundamental.get("fields", []))
            if not fields or any(not str(field).startswith("fund_") for field in fields):
                raise ValueError("fundamental fields must be explicit fund_* research fields")
            if fundamental.get("availabilityRule") != (
                "first_market_date_strictly_after_max_notice_update"
            ):
                raise ValueError("fundamental availability rule must remain conservative")
        output_root = str(config["registry"]["outputRoot"]).replace("\\", "/")
        if "perception_xalpha_all_ashares" not in output_root:
            raise ValueError("all-A-share output root must be independently isolated")
        if int(universe["minimumEligibleSymbols"]) < 1000:
            raise ValueError("all-A-share validation cannot use a small convenience sample")
        if float(
            universe["minimumMasterCoverageForHistoricalValidation"]
        ) < 0.9:
            raise ValueError("all-A-share research must fail closed on partial coverage")
    elif universe_kind != "etf":
        raise ValueError(f"unsupported asset universe: {universe_kind}")
    horizon = int(full["maximumLabelHorizonTradingDays"])
    if int(full["purgeTradingDays"]) < horizon:
        raise ValueError("walk-forward purge must cover the label horizon")
    construction = str(config["fastScreen"].get("bookConstruction", "top_decile"))
    if construction not in {"top_decile", "rank_weighted", "linear_rank_tilt"}:
        raise ValueError("unknown long-only book construction")


def load_base_configs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    perception_config = load_json(ROOT / config["basePerceptionConfig"])
    perception.validate_config(perception_config)
    cog_config = load_json(ROOT / config["baseCogAlphaConfig"])
    autonomous.validate_config(cog_config)
    cog_config = copy.deepcopy(cog_config)
    for field in perception_config["factorGeneration"]["additionalPastOnlyInputs"]:
        if field not in cog_config["search"]["allowedInputs"]:
            cog_config["search"]["allowedInputs"].append(field)
    cog_config["search"]["maximumExpressionDepth"] = int(
        config["synthesis"]["maximumExpressionDepth"]
    )
    cog_config["data"]["roundTripCost"] = float(
        config["fullEvaluation"]["roundTripCost"]
    )
    # One book definition for every stage: inject the construction knobs the screen uses so
    # evaluate_candidate (Primary/Counter/Placebo, purged walk-forward, DSR/PBO) prices the
    # same portfolio. Without this a factor is screened as one strategy and validated as
    # another, and the two verdicts are not comparable.
    screen = config["fastScreen"]
    cog_config["data"]["sizeNeutralise"] = bool(screen.get("sizeNeutralise", False))
    cog_config["data"]["sizeNeutraliseBins"] = int(screen.get("sizeNeutraliseBins", 5))
    cog_config["data"]["bookConstruction"] = str(
        screen.get("bookConstruction", "top_decile")
    )
    cog_config["data"]["holdForPredictionHorizon"] = True
    universe = config.get("assetUniverse", {})
    if universe.get("kind") == "all_a_shares":
        overrides = universe.get("dataOverrides", {})
        for key in (
            "minimumObservationsPerSymbol",
            "minimumMedianDailyAmountCny",
            "minimumCrossSection",
        ):
            if key in overrides:
                cog_config["data"][key] = overrides[key]
        cog_config["data"]["barsRoot"] = universe["barsRoot"]
        cog_config["data"]["survivorshipWarning"] = (
            "Current discoverable SH/SZ/BJ master only; delisted securities and "
            "historical point-in-time ST membership are incomplete."
        )
        fundamental = universe.get("fundamentalData")
        if fundamental:
            for field in fundamental.get("fields", []):
                if field not in cog_config["search"]["allowedInputs"]:
                    cog_config["search"]["allowedInputs"].append(field)
    if int(cog_config["data"]["predictionHorizonTradingDays"]) != int(
        config["fullEvaluation"]["maximumLabelHorizonTradingDays"]
    ):
        raise ValueError("frozen label horizon differs from the base evaluator")
    return perception_config, cog_config


def build_configured_panel(
    config: dict[str, Any],
    cog_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    universe = config.get("assetUniverse", {"kind": "etf"})
    if universe.get("kind", "etf") == "etf":
        panel = core.build_panel(cog_config)
        return panel, {
            "status": "diagnostic_only_research_only",
            "universeKind": "etf",
            "historicalValidationEligible": True,
            "pointInTimeMembership": False,
            "survivorshipWarning": cog_config["data"].get(
                "survivorshipWarning"
            ),
            "orders": [],
            "automaticTradingChanges": [],
        }
    import research_ashare_universe as ashare

    panel, audit = ashare.build_panel(universe, cog_config["data"])
    fundamental = universe.get("fundamentalData")
    if fundamental:
        import research_ashare_fundamentals as fundamentals

        panel, fundamental_audit = fundamentals.attach_point_in_time_fundamentals(
            panel, fundamental
        )
        audit["fundamentalAudit"] = fundamental_audit
        audit["historicalValidationEligible"] = bool(
            audit.get("historicalValidationEligible", False)
            and fundamental_audit.get("historicalValidationEligible", False)
        )
    return panel, audit


APPEND_ONLY_TABLES = [
    "research_cycles",
    "research_plans",
    "mechanism_hypotheses",
    "factor_bundles",
    "experiments",
    "rejection_events",
    "factor_observations",
]


def initialize_registry(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
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
        "CREATE TABLE IF NOT EXISTS run_index "
        "(run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, input_hash TEXT NOT NULL, "
        "status TEXT NOT NULL, output_path TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    connection.commit()


def insert_entity(
    connection: sqlite3.Connection,
    table: str,
    entity_id: str,
    payload: dict[str, Any],
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


def previous_archetype_counts(
    connection: sqlite3.Connection | None,
) -> dict[str, int]:
    if connection is None:
        return {}
    counts: dict[str, int] = {}
    for (payload_text,) in connection.execute(
        "SELECT payload FROM mechanism_hypotheses"
    ):
        try:
            archetype = str(json.loads(payload_text).get("archetypeId", ""))
        except json.JSONDecodeError:
            continue
        if archetype:
            counts[archetype] = counts.get(archetype, 0) + 1
    return counts


def completed_cycle_count(connection: sqlite3.Connection | None) -> int:
    if connection is None:
        return 0
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM run_index WHERE status='complete'"
        ).fetchone()[0]
    )


def previous_factor_ids(connection: sqlite3.Connection | None) -> set[str]:
    if connection is None:
        return set()
    output: set[str] = set()
    for (payload_text,) in connection.execute(
        "SELECT payload FROM factor_observations"
    ):
        try:
            factor_id = str(json.loads(payload_text).get("factorId", ""))
        except json.JSONDecodeError:
            continue
        if factor_id:
            output.add(factor_id)
    return output


def cumulative_factor_catalog(connection: sqlite3.Connection) -> dict[str, Any]:
    """Rebuild a compact catalog from immutable per-cycle factor observations."""
    factors: dict[str, dict[str, Any]] = {}
    rows = connection.execute(
        "SELECT created_at, payload FROM factor_observations ORDER BY created_at"
    )
    for created_at, payload_text in rows:
        try:
            row = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        factor_id = str(row.get("factorId", ""))
        if not factor_id:
            continue
        current = factors.setdefault(
            factor_id,
            {
                "factorId": factor_id,
                "fingerprint": row.get("fingerprint"),
                "archetypeId": row.get("archetypeId"),
                "expression": row.get("expression"),
                "firstSeenAt": created_at,
                "lastSeenAt": created_at,
                "observationCount": 0,
                "candidatePassCount": 0,
                "crediblePassCount": 0,
            },
        )
        current["lastSeenAt"] = created_at
        current["observationCount"] += 1
        current["candidatePassCount"] += int(
            row.get("preMultipleTestingStatus") == "HISTORICALLY_VALIDATED"
        )
        current["crediblePassCount"] += int(
            row.get("finalStatus") == "HISTORICALLY_VALIDATED"
        )
        current["latestRunId"] = row.get("cycleId")
        current["latestStatus"] = row.get("finalStatus")
        current["latestRejectionReasons"] = row.get("rejectionReasons", [])
        current["latestMetrics"] = row.get("metrics")
    return {
        "schemaVersion": "perception_xalpha_cumulative_factor_catalog_v1",
        "status": "research_only_not_trade_signals",
        "uniqueFactorCount": len(factors),
        "factors": sorted(factors.values(), key=lambda row: row["factorId"]),
        "orders": [],
        "automaticTradingChanges": [],
    }


def build_research_plan(
    tickets: list[dict[str, Any]],
    config: dict[str, Any],
    historical_counts: dict[str, int],
    cycle_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    ranked: list[dict[str, Any]] = []
    for ticket in tickets:
        for archetype_id, archetype in ARCHETYPES.items():
            if ticket["phenomenonId"] not in archetype["phenomena"]:
                continue
            recurrence = math.log1p(int(ticket["independentDays"]))
            density = math.log1p(int(ticket["eventCount"]))
            severity = min(float(ticket["residualZScore"]), 20.0) / 4.0
            coverage = math.log1p(len(ticket["affectedAssets"]))
            novelty = 1.0 / (1.0 + historical_counts.get(archetype_id, 0))
            score = recurrence + density + severity + coverage + 3.0 * novelty
            ranked.append(
                {
                    "ticketId": ticket["ticketId"],
                    "phenomenonId": ticket["phenomenonId"],
                    "archetypeId": archetype_id,
                    "priorityScore": round(score, 8),
                    "noveltyWeight": round(novelty, 8),
                }
            )
    ranked.sort(
        key=lambda row: (
            -float(row["priorityScore"]),
            str(row["phenomenonId"]),
            str(row["archetypeId"]),
        )
    )
    maximum = int(config["researchDirector"]["maximumQuestionsPerCycle"])
    per_phenomenon = int(
        config["researchDirector"]["maximumMechanismsPerPhenomenon"]
    )
    selected: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    used_archetypes: set[str] = set()
    for row in ranked:
        phenomenon = str(row["phenomenonId"])
        archetype_id = str(row["archetypeId"])
        if used.get(phenomenon, 0) >= per_phenomenon:
            continue
        if archetype_id in used_archetypes and len(used_archetypes) < maximum:
            continue
        selected.append(row)
        used[phenomenon] = used.get(phenomenon, 0) + 1
        used_archetypes.add(archetype_id)
        if len(selected) >= maximum:
            break
    request = {
        "schemaVersion": "perception_xalpha_research_cycle_request_v1",
        "status": "research_only",
        "cycleId": cycle_id,
        "objective": (
            "Select recurring phenomena whose competing market mechanisms can be "
            "falsified with causal A-share instrument factors."
        ),
        "availableTicketIds": [ticket["ticketId"] for ticket in tickets],
        "historicalFeedbackScope": "train_only_counts_and_rejections",
        "validationFeedbackUsed": False,
        "shadowFeedbackUsed": False,
    }
    plan_rows: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        archetype = ARCHETYPES[row["archetypeId"]]
        hypothesis = {
            "schemaVersion": "perception_xalpha_mechanism_hypothesis_v1",
            "status": "research_only",
            "hypothesisId": "hypothesis_"
            + digest(
                {
                    "ticketId": row["ticketId"],
                    "archetypeId": row["archetypeId"],
                }
            )[:20],
            "cycleId": cycle_id,
            "ticketId": row["ticketId"],
            "phenomenonId": row["phenomenonId"],
            "archetypeId": row["archetypeId"],
            "mechanism": archetype["mechanism"],
            "forcedTrader": archetype["forcedTrader"],
            "persistence": archetype["persistence"],
            "falsifiablePrediction": archetype["falsifiablePrediction"],
            "failureCondition": archetype["failureCondition"],
            "counterArchetypeAgent": archetype["counterAgent"],
            "immutable": True,
        }
        hypotheses.append(hypothesis)
        plan_rows.append(
            {
                "priority": index,
                **row,
                "hypothesisId": hypothesis["hypothesisId"],
                "decision": "synthesize_and_falsify",
            }
        )
    plan = {
        "schemaVersion": "perception_xalpha_research_plan_v1",
        "status": "research_only",
        "cycleId": cycle_id,
        "planId": "plan_" + digest(plan_rows)[:20],
        "questions": plan_rows,
        "selectionPriority": config["researchDirector"]["priority"],
        "validationFeedbackUsed": False,
        "shadowFeedbackUsed": False,
    }
    return request, plan, hypotheses


def make_candidate(
    hypothesis: dict[str, Any],
    expression: dict[str, Any],
    generation: int,
    parents: list[str],
    source: str,
) -> dict[str, Any]:
    archetype_id = str(hypothesis["archetypeId"])
    agent = str(ARCHETYPES[archetype_id]["agent"])
    factor_id = "factor_" + digest(
        {"archetypeId": archetype_id, "expression": expression}
    )[:20]
    candidate = autonomous.candidate_record(
        agent,
        expression,
        generation,
        parents,
        "creative" if generation else "concrete",
        source,
        rationale=str(hypothesis["falsifiablePrediction"]),
        hypothesis={
            "mechanism": str(hypothesis["mechanism"]),
            "forcedTrader": str(hypothesis["forcedTrader"]),
            "persistence": str(hypothesis["persistence"]),
        },
    )
    candidate.update(
        {
            "id": factor_id,
            "factorId": factor_id,
            "fingerprint": core.digest(expression),
            "archetypeId": archetype_id,
            "hypothesisId": hypothesis["hypothesisId"],
            "ticketId": hypothesis["ticketId"],
        }
    )
    return candidate


def structured_tradeable_seeds(
    archetype_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Frozen A-share OHLCV hypotheses, not a parameter sweep.

    Random grammar is useful for novelty but can miss the small set of mechanisms that
    have a plausible path from cross-sectional IC to a long-only, costed portfolio.  These
    expressions are strictly close-t/past-only and use the existing audited DSL.  Direction
    is still learned on train only; names are provenance, never a promise of alpha.
    """

    def field(name: str) -> dict[str, Any]:
        return {"field": name}

    def roll(op: str, arg: dict[str, Any], window: int) -> dict[str, Any]:
        return {"rolling": op, "arg": arg, "window": window}

    def unary(op: str, arg: dict[str, Any]) -> dict[str, Any]:
        return {"unary": op, "arg": arg}

    def binary(
        op: str, left: dict[str, Any], right: dict[str, Any]
    ) -> dict[str, Any]:
        return {"binary": op, "left": left, "right": right}

    def zscore(arg: dict[str, Any], window: int) -> dict[str, Any]:
        return {"zscore": True, "arg": arg, "window": window}

    returns = field("returns")
    close = field("close")
    open_price = field("open")
    log_volume = unary("signed_log1p", field("volume"))
    log_amount = unary("signed_log1p", field("amount"))
    return_3 = roll("sum", returns, 3)
    return_5 = roll("sum", returns, 5)
    return_10 = roll("sum", returns, 10)
    return_20 = roll("sum", returns, 20)
    return_60 = roll("sum", returns, 60)
    return_120 = roll("sum", returns, 120)
    gap = binary("div", open_price, {"lag": 1, "arg": close})
    intraday = binary("div", close, open_price)
    amihud_20 = roll(
        "mean", binary("div", unary("abs", returns), field("amount")), 20
    )
    volume_shock_20 = zscore(log_volume, 20)
    amount_shock_60 = zscore(log_amount, 60)
    range_position_60 = {"range_position": True, "arg": close, "window": 60}

    library: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "liquidity_reversal": [
            ("short_reversal_3", unary("neg", return_3)),
            ("short_reversal_5", unary("neg", return_5)),
            ("short_reversal_10", unary("neg", return_10)),
            ("volume_shock_reversal", binary("mul", unary("neg", return_5), volume_shock_20)),
            ("illiquidity_shock_reversal", binary("mul", unary("neg", return_5), amihud_20)),
            ("gap_reversal", unary("neg", gap)),
        ],
        "information_diffusion": [
            ("momentum_120_ex_recent_20", binary("sub", return_120, return_20)),
            ("momentum_60_ex_recent_5", binary("sub", return_60, return_5)),
            ("lagged_momentum_20", {"lag": 5, "arg": return_20}),
            ("overnight_gap_continuation", gap),
            ("intraday_continuation", intraday),
            ("gap_intraday_divergence", binary("sub", gap, intraday)),
        ],
        "benchmark_rebalancing": [
            ("medium_reversal_30_ex_10", binary("sub", roll("sum", returns, 30), return_10)),
            ("momentum_120_ex_recent_20", binary("sub", return_120, return_20)),
            ("flow_confirmed_momentum", binary("mul", return_20, amount_shock_60)),
            ("slow_fast_momentum", binary("sub", return_60, return_10)),
            ("liquidity_persistence", binary("div", roll("mean", log_amount, 5), roll("mean", log_amount, 60))),
            ("risk_adjusted_momentum", binary("div", return_60, roll("std", returns, 20))),
        ],
        "crowding_unwind": [
            ("max_return_reversal_20", unary("neg", roll("max", returns, 20))),
            ("crowded_momentum_unwind", unary("neg", binary("mul", return_20, zscore(log_volume, 60)))),
            ("volume_price_divergence", binary("mul", return_10, unary("neg", volume_shock_20))),
            ("volatility_compression", unary("neg", binary("div", roll("std", returns, 5), roll("std", returns, 60)))),
            ("range_position_reversal", unary("neg", range_position_60)),
            ("drawdown_flow_rebound", binary("mul", unary("neg", {"drawdown": True, "arg": close, "window": 60}), volume_shock_20)),
        ],
        "order_splitting": [
            ("flow_confirmed_momentum", binary("mul", return_20, amount_shock_60)),
            ("volume_confirmed_momentum", binary("mul", return_10, volume_shock_20)),
            ("persistent_intraday_pressure", roll("mean", intraday, 10)),
            ("slow_fast_momentum", binary("sub", return_60, return_10)),
        ],
        "risk_budgeting": [
            ("low_volatility_20", unary("neg", roll("std", returns, 20))),
            ("volatility_compression", unary("neg", binary("div", roll("std", returns, 5), roll("std", returns, 60)))),
            ("drawdown_60", {"drawdown": True, "arg": close, "window": 60}),
            ("risk_adjusted_momentum", binary("div", return_60, roll("std", returns, 20))),
        ],
        "volatility_feedback": [
            ("low_volatility_20", unary("neg", roll("std", returns, 20))),
            ("downside_pressure_20", roll("mean", binary("sub", returns, unary("abs", returns)), 20)),
            ("max_return_reversal_20", unary("neg", roll("max", returns, 20))),
            ("drawdown_flow_rebound", binary("mul", unary("neg", {"drawdown": True, "arg": close, "window": 60}), volume_shock_20)),
        ],
    }
    return copy.deepcopy(library.get(archetype_id, []))


def structured_predictive_seeds(
    archetype_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Broader, still bounded and causal A-share OHLCV hypotheses.

    These extend the v1 economic seed set without forming a free parameter grid.  Each
    expression represents a distinct mechanism and all rolling inputs end at t.  The
    direction continues to be learned on train only.
    """

    def field(name: str) -> dict[str, Any]:
        return {"field": name}

    def roll(op: str, arg: dict[str, Any], window: int) -> dict[str, Any]:
        return {"rolling": op, "arg": arg, "window": window}

    def unary(op: str, arg: dict[str, Any]) -> dict[str, Any]:
        return {"unary": op, "arg": arg}

    def binary(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {"binary": op, "left": left, "right": right}

    def zscore(arg: dict[str, Any], window: int) -> dict[str, Any]:
        return {"zscore": True, "arg": arg, "window": window}

    returns = field("returns")
    close = field("close")
    open_price = field("open")
    high = field("high")
    low = field("low")
    amount_log = unary("signed_log1p", field("amount"))
    volume_log = unary("signed_log1p", field("volume"))
    r5 = roll("sum", returns, 5)
    r10 = roll("sum", returns, 10)
    r20 = roll("sum", returns, 20)
    r60 = roll("sum", returns, 60)
    r120 = roll("sum", returns, 120)
    r252 = roll("sum", returns, 252)
    vol20 = roll("std", returns, 20)
    vol60 = roll("std", returns, 60)
    amount20 = roll("mean", amount_log, 20)
    amount120 = roll("mean", amount_log, 120)
    volume20 = roll("mean", volume_log, 20)
    volume120 = roll("mean", volume_log, 120)
    # Ratios retain the same cross-sectional ordering as minus-one returns and avoid
    # introducing a constant node that the deliberately small DSL does not permit.
    gap = binary("div", open_price, {"lag": 1, "arg": close})
    intraday = binary("div", close, open_price)
    close_location = binary(
        "div",
        binary("sub", close, low),
        binary("sub", high, low),
    )
    downside = roll("mean", binary("sub", returns, unary("abs", returns)), 20)
    market_corr = {
        "corr": True,
        "left": returns,
        "right": field("market_return"),
        "window": 60,
    }

    common: dict[str, dict[str, Any]] = {
        "momentum_252_ex_recent_20": binary("sub", r252, r20),
        "momentum_120_ex_recent_10": binary("sub", r120, r10),
        "risk_adjusted_reversal_60": unary("neg", binary("div", r60, vol20)),
        "short_long_reversal_spread": binary("sub", unary("neg", r5), r60),
        "turnover_acceleration": binary("div", amount20, amount120),
        "volume_acceleration": binary("div", volume20, volume120),
        "price_volume_disagreement": binary("mul", r20, unary("neg", zscore(amount_log, 60))),
        "overnight_intraday_disagreement": binary("sub", gap, intraday),
        "close_location_pressure": roll("mean", close_location, 10),
        "low_beta_proxy": unary("neg", market_corr),
        "downside_risk_pressure": downside,
        "volatility_term_structure": unary("neg", binary("div", vol20, vol60)),
        "range_breakout_position": {"range_position": True, "arg": close, "window": 120},
        "drawdown_recovery_with_flow": binary(
            "mul",
            unary("neg", {"drawdown": True, "arg": close, "window": 120}),
            zscore(amount_log, 60),
        ),
    }
    family_names: dict[str, list[str]] = {
        "order_splitting": [
            "turnover_acceleration", "volume_acceleration", "close_location_pressure",
            "momentum_120_ex_recent_10", "price_volume_disagreement",
        ],
        "liquidity_reversal": [
            "risk_adjusted_reversal_60", "short_long_reversal_spread",
            "price_volume_disagreement", "overnight_intraday_disagreement",
            "drawdown_recovery_with_flow",
        ],
        "risk_budgeting": [
            "low_beta_proxy", "downside_risk_pressure", "volatility_term_structure",
            "risk_adjusted_reversal_60", "drawdown_recovery_with_flow",
        ],
        "information_diffusion": [
            "momentum_252_ex_recent_20", "momentum_120_ex_recent_10",
            "overnight_intraday_disagreement", "close_location_pressure",
            "turnover_acceleration",
        ],
        "crowding_unwind": [
            "short_long_reversal_spread", "price_volume_disagreement",
            "risk_adjusted_reversal_60", "drawdown_recovery_with_flow",
            "range_breakout_position",
        ],
        "benchmark_rebalancing": [
            "momentum_252_ex_recent_20", "turnover_acceleration",
            "volume_acceleration", "low_beta_proxy", "range_breakout_position",
        ],
        "volatility_feedback": [
            "downside_risk_pressure", "volatility_term_structure", "low_beta_proxy",
            "risk_adjusted_reversal_60", "drawdown_recovery_with_flow",
        ],
    }
    extended = structured_tradeable_seeds(archetype_id)
    extended.extend((name, common[name]) for name in family_names.get(archetype_id, []))
    deduplicated: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, expression in extended:
        deduplicated.setdefault(digest(expression), (name, expression))
    return copy.deepcopy(list(deduplicated.values()))


def structured_fundamental_seeds(
    archetype_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Causal PIT fundamentals plus economically explicit OHLCV interactions.

    Values are already aligned to the first session strictly after public disclosure.
    These are a bounded hypothesis library, not a parameter sweep; evolution may combine
    them, but validation/shadow outcomes never return to the generator.
    """

    def f(name: str) -> dict[str, Any]:
        return {"field": name}

    def roll(op: str, arg: dict[str, Any], window: int) -> dict[str, Any]:
        return {"rolling": op, "arg": arg, "window": window}

    def unary(op: str, arg: dict[str, Any]) -> dict[str, Any]:
        return {"unary": op, "arg": arg}

    def binary(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {"binary": op, "left": left, "right": right}

    quality = f("fund_quality_composite")
    growth = f("fund_growth_composite")
    safety = f("fund_balance_sheet_safety")
    accrual = f("fund_accrual_quality")
    book_to_price = f("fund_book_to_price")
    earnings_yield = f("fund_annualized_earnings_yield")
    roe = f("fund_roe")
    roic = f("fund_roic")
    debt = f("fund_debt_asset_ratio")
    cash_profit = f("fund_cash_to_profit")
    revenue_growth = f("fund_revenue_yoy")
    profit_growth = f("fund_net_profit_yoy")
    returns = f("returns")
    momentum_20 = roll("sum", returns, 20)
    momentum_120 = roll("sum", returns, 120)
    volatility_20 = roll("std", returns, 20)
    liquidity_20 = roll("mean", unary("signed_log1p", f("amount")), 20)
    low_beta = unary(
        "neg",
        {
            "corr": True,
            "left": returns,
            "right": f("market_return"),
            "window": 60,
        },
    )
    seeds: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "order_splitting": [
            ("quality_with_liquidity_confirmation", binary("mul", quality, liquidity_20)),
            ("cash_profit_with_turnover", binary("mul", cash_profit, liquidity_20)),
            ("growth_information_diffusion", binary("mul", growth, momentum_20)),
        ],
        "liquidity_reversal": [
            ("value_short_term_reversal", binary("mul", book_to_price, unary("neg", momentum_20))),
            ("quality_at_discount", binary("mul", quality, book_to_price)),
            ("earnings_yield_reversal", binary("mul", earnings_yield, unary("neg", momentum_20))),
        ],
        "risk_budgeting": [
            ("quality_composite", quality),
            ("balance_sheet_safety", safety),
            ("quality_low_beta", binary("mul", quality, low_beta)),
            ("roic_low_volatility", binary("div", roic, volatility_20)),
            ("negative_leverage", unary("neg", debt)),
        ],
        "information_diffusion": [
            ("growth_composite", growth),
            ("revenue_growth", revenue_growth),
            ("profit_growth", profit_growth),
            ("growth_medium_momentum", binary("mul", growth, momentum_120)),
            ("roe_with_momentum", binary("mul", roe, momentum_20)),
        ],
        "crowding_unwind": [
            ("book_to_price", book_to_price),
            ("annualized_earnings_yield", earnings_yield),
            ("value_against_momentum", binary("sub", book_to_price, momentum_120)),
            ("accrual_quality", accrual),
        ],
        "benchmark_rebalancing": [
            ("quality_value", binary("add", quality, book_to_price)),
            ("growth_quality", binary("add", growth, quality)),
            ("fundamental_strength_liquid", binary("mul", binary("add", quality, growth), liquidity_20)),
        ],
        "volatility_feedback": [
            ("safety_low_beta", binary("mul", safety, low_beta)),
            ("quality_per_volatility", binary("div", quality, volatility_20)),
            ("cash_quality_per_volatility", binary("div", cash_profit, volatility_20)),
            ("roic_minus_leverage", binary("sub", roic, debt)),
        ],
    }
    return copy.deepcopy(seeds.get(archetype_id, []))


def book_identity(cog_config: dict[str, Any]) -> str:
    """Hash of the effective book definition a parent pool was bred under.

    Two configs that price portfolios differently must not share search memory: a v1 run
    finishing after a v2 run would otherwise hand v2 parents selected under the old screen
    and construction, silently mixing search histories and corrupting trial accounting.
    """
    data = cog_config.get("data", {})
    return digest({
        "topQuantile": data.get("topQuantile"),
        "roundTripCost": data.get("roundTripCost"),
        "sizeNeutralise": data.get("sizeNeutralise"),
        "sizeNeutraliseBins": data.get("sizeNeutraliseBins"),
        "bookConstruction": data.get("bookConstruction"),
        "holdForPredictionHorizon": data.get("holdForPredictionHorizon"),
        "predictionHorizonTradingDays": data.get("predictionHorizonTradingDays"),
    })


def load_train_parents(
    state_directory: Path,
    hypotheses: list[dict[str, Any]],
    cog_config: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    path = state_directory / "train_parent_pool.json"
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return []
    # Refuse a pool bred under a different screen/book definition. Silently inheriting one
    # would mix search histories across config versions and make the trial accounting a lie.
    if str(payload.get("bookIdentitySha256") or "") != book_identity(cog_config):
        return []
    rows = payload.get("parents", [])
    by_archetype = {
        str(hypothesis["archetypeId"]): hypothesis for hypothesis in hypotheses
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        archetype_id = str(row.get("archetypeId", ""))
        hypothesis = by_archetype.get(archetype_id)
        if hypothesis is None:
            continue
        expression = row.get("expression")
        try:
            core.validate_expression(
                expression, autonomous.expression_config(cog_config)
            )
        except Exception:
            continue
        output.append(
            make_candidate(
                hypothesis,
                expression,
                0,
                [str(row.get("factorId", "previous_train_parent"))],
                "previous_train_parent",
            )
        )
        if len(output) >= limit:
            break
    return output


def initial_candidates(
    hypotheses: list[dict[str, Any]],
    config: dict[str, Any],
    cog_config: dict[str, Any],
    state_directory: Path,
    use_state: bool,
) -> list[dict[str, Any]]:
    variants = int(config["synthesis"]["initialVariantsPerQuestion"])
    novelty_epoch = int(config.get("_runtimeNoveltyEpoch", 0))
    seed = int(config["synthesis"]["randomSeed"]) + novelty_epoch * 1_000_003
    output: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        structured_library = config["synthesis"].get("structuredSeedLibrary")
        if structured_library in {
            "a_share_tradeable_v1",
            "a_share_predictive_v2",
            "a_share_fundamental_pit_v1",
        }:
            seed_builder = {
                "a_share_tradeable_v1": structured_tradeable_seeds,
                "a_share_predictive_v2": structured_predictive_seeds,
                "a_share_fundamental_pit_v1": structured_fundamental_seeds,
            }[structured_library]
            structured = seed_builder(str(hypothesis["archetypeId"]))
            maximum_structured = int(
                config["synthesis"].get("maximumStructuredSeedsPerQuestion", 0)
            )
            for seed_name, expression in structured[:maximum_structured]:
                core.validate_expression(
                    expression, autonomous.expression_config(cog_config)
                )
                output.append(
                    make_candidate(
                        hypothesis,
                        expression,
                        0,
                        [],
                        f"structured_seed:{seed_name}",
                    )
                )
        local_seed = seed + int(digest(hypothesis["hypothesisId"])[:8], 16)
        rng = random.Random(local_seed)
        agent = str(ARCHETYPES[hypothesis["archetypeId"]]["agent"])
        expression = autonomous.random_expression(agent, rng, cog_config)
        output.append(
            make_candidate(
                hypothesis,
                expression,
                0,
                [],
                "mechanism_grammar_seed",
            )
        )
        for _ in range(max(0, variants - 1)):
            expression = autonomous.mutate_expression(
                expression, agent, rng, cog_config
            )
            output.append(
                make_candidate(
                    hypothesis,
                    expression,
                    0,
                    [output[-1]["factorId"]],
                    "mechanism_grammar_variant",
                )
            )
    if use_state:
        output.extend(
            load_train_parents(
                state_directory,
                hypotheses,
                cog_config,
                int(config["synthesis"]["maximumPersistentParents"]),
            )
        )
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in output:
        if candidate["fingerprint"] in seen:
            continue
        seen.add(candidate["fingerprint"])
        deduplicated.append(candidate)
    return deduplicated


def long_only_portfolio(
    signal: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    cog_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Delegate to the single shared book definition in research_cogalpha_autonomous.

    Screen and full evaluation must price the identical strategy; keeping a second
    implementation here is exactly how a factor ends up screened as one portfolio and
    validated as another. Construction knobs are injected into cog_config["data"] by
    build_cog_config, so both stages read one source of truth.
    """
    net, turnover, weights, _book, _bench = autonomous.long_only_portfolio(
        signal, one_day, panel, cog_config
    )
    return net, turnover, weights


def objective_group(config: dict[str, Any]) -> str:
    """Metric family used for discovery selection, never for cost reporting."""
    mode = str(
        config.get("discoveryObjective", {}).get("mode", "costed_tradeability")
    )
    return "grossLongOnly" if mode == "gross_predictive" else "costedLongOnly"


def fast_screen(
    candidate: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    split: autonomous.Split,
    config: dict[str, Any],
    cog_config: dict[str, Any],
    accepted: list[FastResult],
) -> tuple[FastResult | None, str | None, dict[str, Any]]:
    safe, reason = perception.static_lint(candidate, cog_config)
    if not safe:
        return None, "STATIC_DSL_REJECTED", {"detail": reason}
    if not perception.prefix_invariant(candidate["expression"], panel):
        return None, "PREFIX_LEAKAGE", {}
    try:
        raw = core.evaluate_expression(candidate["expression"], panel).replace(
            [np.inf, -np.inf], np.nan
        )
    except Exception as exc:
        return None, "STATIC_DSL_REJECTED", {"detail": type(exc).__name__}
    train_mask = split.train
    eligible = panel["close"].notna() & target.notna()
    train_eligible = eligible.loc[train_mask]
    denominator = int(train_eligible.to_numpy().sum())
    available = int(
        raw.loc[train_mask].where(train_eligible).notna().to_numpy().sum()
    )
    nan_fraction = 1.0 - available / max(1, denominator)
    if nan_fraction > float(config["fastScreen"]["maximumNanFraction"]):
        return None, "TOO_MANY_NAN", {"nanFraction": round(nan_fraction, 8)}
    distinct = raw.loc[train_mask].where(train_eligible).nunique(axis=1)
    median_distinct = float(distinct.median()) if not distinct.empty else 0.0
    if median_distinct < float(
        config["fastScreen"]["minimumMedianCrossSectionDistinct"]
    ):
        return None, "INSUFFICIENT_CROSS_SECTION", {
            "medianDistinct": median_distinct
        }
    raw_rank_ic = raw.loc[train_mask].corrwith(
        target.loc[train_mask], axis=1, method="spearman"
    ).dropna()
    if len(raw_rank_ic) < int(config["fastScreen"]["minimumTrainIcDays"]):
        return None, "INSUFFICIENT_TRAIN_DAYS", {"days": len(raw_rank_ic)}
    direction = 1.0 if float(raw_rank_ic.mean()) >= 0.0 else -1.0
    signal = raw * direction
    rank_ic = signal.loc[train_mask].corrwith(
        target.loc[train_mask], axis=1, method="spearman"
    ).dropna()
    long_net, turnover, book_weights, book_return, benchmark = (
        autonomous.long_only_portfolio(signal, one_day, panel, cog_config)
    )
    gross_excess = book_return - benchmark
    held = book_weights.gt(0.0)
    overlap_lag = max(0, int(cog_config["data"].get("predictionHorizonTradingDays", 1)) - 1)
    rank_stats = autonomous.period_stats(rank_ic, train_mask, overlap_lag)
    gross_stats = autonomous.period_stats(gross_excess, train_mask, overlap_lag)
    net_stats = autonomous.period_stats(long_net, train_mask, overlap_lag)
    metrics = {
        "rankIc": rank_stats,
        "grossLongOnly": gross_stats,
        "costedLongOnly": net_stats,
        "directionFromTrain": direction,
        "nanFraction": round(nan_fraction, 8),
        "medianCrossSectionDistinct": round(median_distinct, 4),
        "expressionDepth": core.expression_depth(candidate["expression"]),
    }
    if abs(float(rank_stats["mean"] or 0.0)) < float(
        config["fastScreen"]["minimumAbsoluteRankIc"]
    ):
        return None, "WEAK_TRAIN_RANK_IC", metrics
    if float(rank_stats["irAnn"] or -99.0) < float(
        config["fastScreen"]["minimumRankIcIr"]
    ):
        return None, "WEAK_TRAIN_RANK_IC_IR", metrics
    if objective_group(config) == "grossLongOnly":
        if float(gross_stats["irAnn"] or -99.0) < float(
            config["fastScreen"].get("minimumGrossLongOnlyIr", 0.0)
        ):
            return None, "NEGATIVE_GROSS_TRAIN_IR", metrics
    elif float(net_stats["irAnn"] or -99.0) < float(
        config["fastScreen"]["minimumCostedLongOnlyIr"]
    ):
        return None, "NEGATIVE_COSTED_TRAIN_IR", metrics
    maximum_similarity = -1.0
    nearest_factor = None
    for other in accepted:
        daily = signal.loc[train_mask].corrwith(
            other.signal.loc[train_mask], axis=1, method="spearman"
        )
        similarity = abs(float(daily.mean())) if daily.notna().any() else 0.0
        if similarity > maximum_similarity:
            maximum_similarity = similarity
            nearest_factor = other.candidate["factorId"]
    if maximum_similarity > float(
        config["fastScreen"]["maximumBehaviorCorrelation"]
    ):
        return None, "DUPLICATE_BEHAVIOR", {
            **metrics,
            "behaviorCorrelation": round(maximum_similarity, 8),
            "nearestFactorId": nearest_factor,
        }
    behavior = {
        "fingerprint": digest(
            {
                "direction": direction,
                "rankIcSign": int(math.copysign(1, float(rank_stats["mean"]))),
                "coverageBucket": round(1.0 - nan_fraction, 2),
                "turnoverBucket": round(float(turnover.loc[train_mask].mean()), 2),
                "eventDensityBucket": round(
                    float(held.loc[train_mask].mean().mean()), 2
                ),
            }
        ),
        "nearestFactorId": nearest_factor,
        "maximumAbsoluteTrainRankCorrelation": round(
            max(0.0, maximum_similarity), 8
        ),
    }
    selected_return_stats = (
        gross_stats if objective_group(config) == "grossLongOnly" else net_stats
    )
    fitness = (
        float(rank_stats["irAnn"] or -3.0)
        + max(-3.0, float(selected_return_stats["irAnn"] or -3.0))
        - float(config["fastScreen"]["complexityPenaltyPerDepth"])
        * float(metrics["expressionDepth"])
    )
    metrics["trainFitness"] = round(fitness, 8)
    return (
        FastResult(candidate, signal, long_net, metrics, fitness, behavior),
        None,
        metrics,
    )


def choose_operator(rng: random.Random, config: dict[str, Any]) -> str:
    operators = list(config["synthesis"]["operators"])
    weights = [
        float(config["synthesis"]["operatorWeights"][operator])
        for operator in operators
    ]
    return rng.choices(operators, weights=weights, k=1)[0]


class _SeedParent:
    """Breeding stand-in for a screened-out candidate.

    evolve_candidates only reads ``.candidate``, so a screened-out expression can seed the
    next generation without being promoted, scored or recorded as accepted anywhere.
    """

    __slots__ = ("candidate", "fitness")

    def __init__(self, candidate: dict[str, Any]) -> None:
        self.candidate = candidate
        self.fitness = float("-inf")


def evolve_candidates(
    parents: list[FastResult],
    hypotheses: list[dict[str, Any]],
    generation: int,
    config: dict[str, Any],
    cog_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not parents:
        return []
    novelty_epoch = int(config.get("_runtimeNoveltyEpoch", 0))
    seed = (
        int(config["synthesis"]["randomSeed"])
        + generation * 1009
        + novelty_epoch * 1_000_003
    )
    rng = random.Random(seed)
    hypothesis_map = {
        hypothesis["hypothesisId"]: hypothesis for hypothesis in hypotheses
    }
    children: list[dict[str, Any]] = []
    per_parent = int(config["synthesis"]["childrenPerParent"])
    for index, parent in enumerate(parents):
        hypothesis = hypothesis_map[parent.candidate["hypothesisId"]]
        agent = str(parent.candidate["agent"])
        for child_index in range(per_parent):
            operator = choose_operator(rng, config)
            expression = copy.deepcopy(parent.candidate["expression"])
            parent_ids = [parent.candidate["factorId"]]
            if operator == "mutation":
                expression = autonomous.mutate_expression(
                    expression, agent, rng, cog_config
                )
            elif operator == "crossover":
                mate = parents[(index + child_index + 1) % len(parents)]
                expression = {
                    "binary": rng.choice(["add", "sub", "mul"]),
                    "left": expression,
                    "right": copy.deepcopy(mate.candidate["expression"]),
                }
                parent_ids.append(mate.candidate["factorId"])
            elif operator == "refinement":
                expression, changed = core.replace_first_window(
                    expression, rng.choice(cog_config["search"]["allowedWindows"])
                )
                if not changed:
                    expression = {"unary": "tanh", "arg": expression}
            else:
                expression = autonomous.random_expression(agent, rng, cog_config)
            children.append(
                make_candidate(
                    hypothesis,
                    expression,
                    generation,
                    parent_ids,
                    f"evolution_{operator}",
                )
            )
    return children


def evaluate_member(
    candidate: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    split: autonomous.Split,
    cog_config: dict[str, Any],
) -> tuple[autonomous.Evaluation | None, str | None]:
    return autonomous.evaluate_candidate(
        candidate, panel, target, one_day, split, cog_config
    )


def counter_candidate(
    primary: FastResult,
    hypothesis: dict[str, Any],
    config: dict[str, Any],
    cog_config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config["synthesis"]["randomSeed"]) + int(
        digest(primary.candidate["factorId"] + "_counter")[:8], 16
    )
    rng = random.Random(seed)
    counter_agent = str(ARCHETYPES[hypothesis["archetypeId"]]["counterAgent"])
    expression = autonomous.random_expression(counter_agent, rng, cog_config)
    candidate = autonomous.candidate_record(
        counter_agent,
        expression,
        primary.candidate["generation"],
        [primary.candidate["factorId"]],
        "concrete",
        "counter_mechanism",
        hypothesis=autonomous.ROLE_HYPOTHESES[counter_agent],
        rationale=(
            "Competing mechanism control for "
            + str(hypothesis["falsifiablePrediction"])
        ),
    )
    candidate["id"] = "counter_" + digest(
        {
            "primary": primary.candidate["factorId"],
            "expression": expression,
        }
    )[:20]
    candidate["factorId"] = candidate["id"]
    return candidate


def placebo_candidate(
    primary: FastResult,
    config: dict[str, Any],
    cog_config: dict[str, Any],
) -> dict[str, Any]:
    lag = int(config["synthesis"]["placeboLagTradingDays"])
    expression = {
        "lag": lag,
        "arg": copy.deepcopy(primary.candidate["expression"]),
    }
    try:
        core.validate_expression(
            expression, autonomous.expression_config(cog_config)
        )
    except Exception:
        expression = {"lag": lag, "arg": {"field": "returns"}}
    candidate = autonomous.candidate_record(
        primary.candidate["agent"],
        expression,
        primary.candidate["generation"],
        [primary.candidate["factorId"]],
        "concrete",
        "causal_delayed_placebo",
        hypothesis=primary.candidate["hypothesis"],
        rationale=(
            "Causally delayed placebo with the same broad information family but "
            "without the proposed timing."
        ),
    )
    candidate["id"] = "placebo_" + digest(
        {
            "primary": primary.candidate["factorId"],
            "expression": expression,
        }
    )[:20]
    candidate["factorId"] = candidate["id"]
    return candidate


def metric_value(
    evaluation: autonomous.Evaluation | None,
    period: str,
    group: str,
    field: str,
    default: float = -99.0,
) -> float:
    if evaluation is None:
        return default
    value = evaluation.summary["periods"][period][group].get(field)
    return default if value is None else float(value)


def failure_condition_audit(
    primary: autonomous.Evaluation,
    panel: dict[str, pd.DataFrame],
    split: autonomous.Split,
    config: dict[str, Any],
    hypothesis: dict[str, Any],
) -> dict[str, Any]:
    market_volatility = (
        panel["market_abs_return"].median(axis=1).replace([np.inf, -np.inf], np.nan)
    )
    train_values = market_volatility.loc[split.train].dropna()
    threshold = (
        float(
            train_values.quantile(
                float(config["fullEvaluation"]["failureConditionTrainQuantile"])
            )
        )
        if len(train_values)
        else None
    )
    if threshold is None:
        return {
            "condition": hypothesis["failureCondition"],
            "threshold": None,
            "status": "insufficient_train_state",
        }
    validation_high = split.validation & market_volatility.ge(threshold)
    validation_normal = split.validation & market_volatility.lt(threshold)
    return {
        "condition": hypothesis["failureCondition"],
        "stateVariable": "market_abs_return",
        "thresholdFrozenOn": "train_only",
        "threshold": round(threshold, 10),
        "highState": autonomous.period_stats(primary.long_net, validation_high),
        "normalState": autonomous.period_stats(
            primary.long_net, validation_normal
        ),
        "usedForGeneration": False,
    }


def purged_walk_forward_audit(
    primary: autonomous.Evaluation,
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    split: autonomous.Split,
    config: dict[str, Any],
    cog_config: dict[str, Any],
) -> dict[str, Any]:
    """Expanding-window audit with a label-horizon purge before each test fold."""
    dates = pd.DatetimeIndex(panel["close"].index)
    usable_dates = dates[~split.shadow.reindex(dates, fill_value=False).to_numpy()]
    folds = int(config["fullEvaluation"]["purgedWalkForwardFolds"])
    purge = int(config["fullEvaluation"]["purgeTradingDays"])
    minimum_train = int(cog_config["data"]["minimumTrainingTradingDays"])
    remaining = len(usable_dates) - minimum_train
    if remaining < folds * 20:
        return {
            "status": "insufficient_dates",
            "folds": [],
            "purgeTradingDays": purge,
            "labelHorizonTradingDays": int(
                cog_config["data"]["predictionHorizonTradingDays"]
            ),
        }
    fold_size = remaining // folds
    raw = primary.signal * float(
        primary.summary.get("directionFromTrain", 1.0)
    )
    rows: list[dict[str, Any]] = []
    for fold in range(folds):
        test_start_position = minimum_train + fold * fold_size
        test_end_position = (
            len(usable_dates)
            if fold == folds - 1
            else test_start_position + fold_size
        )
        train_end_position = test_start_position - purge
        if train_end_position <= 0:
            continue
        train_dates = usable_dates[:train_end_position]
        test_dates = usable_dates[test_start_position:test_end_position]
        raw_train_ic = raw.loc[train_dates].corrwith(
            target.loc[train_dates], axis=1, method="spearman"
        ).dropna()
        if len(raw_train_ic) < 120:
            continue
        direction = 1.0 if float(raw_train_ic.mean()) >= 0.0 else -1.0
        fold_signal = raw * direction
        long_net, _fold_turnover, _fold_weights, fold_book, fold_benchmark = (
            autonomous.long_only_portfolio(
                fold_signal, one_day, panel, cog_config
            )
        )
        test_mask = pd.Series(False, index=dates)
        test_mask.loc[test_dates] = True
        rank_ic = fold_signal.corrwith(target, axis=1, method="spearman")
        rows.append(
            {
                "fold": fold + 1,
                "train": [
                    train_dates[0].date().isoformat(),
                    train_dates[-1].date().isoformat(),
                    len(train_dates),
                ],
                "test": [
                    test_dates[0].date().isoformat(),
                    test_dates[-1].date().isoformat(),
                    len(test_dates),
                ],
                "purgeTradingDays": purge,
                "directionFromFoldTrain": direction,
                "rankIc": autonomous.period_stats(rank_ic, test_mask),
                "grossLongOnly": autonomous.period_stats(
                    fold_book - fold_benchmark, test_mask
                ),
                "costedLongOnly": autonomous.period_stats(long_net, test_mask),
            }
        )
    fold_group = objective_group(config)
    positive = sum(
        1
        for row in rows
        if float(row[fold_group].get("irAnn") or -99.0) > 0.0
    )
    required = int(
        config["fullEvaluation"]["minimumPositiveWalkForwardFolds"]
    )
    return {
        "status": "passed" if positive >= required else "failed",
        "folds": rows,
        "positiveCostedFolds": positive,
        "positiveObjectiveFolds": positive,
        "objectiveMetric": fold_group,
        "requiredPositiveCostedFolds": required,
        "purgeTradingDays": purge,
        "labelHorizonTradingDays": int(
            cog_config["data"]["predictionHorizonTradingDays"]
        ),
        "validationOrShadowUsedForFit": False,
    }


def full_bundle_evaluation(
    fast: FastResult,
    hypothesis: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    split: autonomous.Split,
    config: dict[str, Any],
    cog_config: dict[str, Any],
) -> tuple[dict[str, Any], autonomous.Evaluation | None]:
    counter = counter_candidate(fast, hypothesis, config, cog_config)
    placebo = placebo_candidate(fast, config, cog_config)
    primary_evaluation, primary_reason = evaluate_member(
        fast.candidate, panel, target, one_day, split, cog_config
    )
    counter_evaluation, counter_reason = evaluate_member(
        counter, panel, target, one_day, split, cog_config
    )
    placebo_evaluation, placebo_reason = evaluate_member(
        placebo, panel, target, one_day, split, cog_config
    )
    bundle_id = "bundle_" + digest(
        {
            "primary": fast.candidate["expression"],
            "counter": counter["expression"],
            "placebo": placebo["expression"],
        }
    )[:20]
    state_history = [
        "DRAFT",
        "STATIC_VALIDATED",
        "LEAKAGE_VALIDATED",
        "FAST_SCREENED",
    ]
    reasons: list[str] = []
    if (
        primary_evaluation is None
        or counter_evaluation is None
        or placebo_evaluation is None
    ):
        reasons.append("FULL_EVALUATION_FAILED")
    else:
        primary_rank = metric_value(
            primary_evaluation, "validation", "rankIc", "mean"
        )
        comparison_group = objective_group(config)
        primary_ir = metric_value(
            primary_evaluation, "validation", comparison_group, "irAnn"
        )
        counter_ir = metric_value(
            counter_evaluation, "validation", comparison_group, "irAnn"
        )
        placebo_ir = metric_value(
            placebo_evaluation, "validation", comparison_group, "irAnn"
        )
        if primary_rank <= float(
            config["fullEvaluation"]["minimumValidationRankIc"]
        ):
            reasons.append("VALIDATION_RANK_IC_FAILED")
        if comparison_group == "grossLongOnly":
            if primary_ir <= float(
                config["fullEvaluation"].get("minimumValidationGrossIr", 0.0)
            ):
                reasons.append("VALIDATION_GROSS_IR_FAILED")
        elif primary_ir <= float(
            config["fullEvaluation"]["minimumValidationCostedIr"]
        ):
            reasons.append("VALIDATION_COSTED_IR_FAILED")
        if (
            primary_ir - counter_ir
            <= float(
                config["fullEvaluation"][
                    "minimumPrimaryAdvantageOverCounterIr"
                ]
            )
        ):
            reasons.append("COUNTER_NOT_BEATEN")
        if (
            primary_ir - placebo_ir
            <= float(
                config["fullEvaluation"][
                    "minimumPrimaryAdvantageOverPlaceboIr"
                ]
            )
        ):
            reasons.append("PLACEBO_NOT_BEATEN")
    walk_forward = (
        purged_walk_forward_audit(
            primary_evaluation,
            panel,
            target,
            one_day,
            split,
            config,
            cog_config,
        )
        if primary_evaluation is not None
        else {"status": "primary_evaluation_failed", "folds": []}
    )
    if walk_forward.get("status") != "passed":
        reasons.append("PURGED_WALK_FORWARD_FAILED")
    if reasons:
        state_history.append("REJECTED")
        status = "REJECTED"
    else:
        state_history.append("HISTORICALLY_VALIDATED")
        status = "HISTORICALLY_VALIDATED"
    bundle = {
        "schemaVersion": "perception_xalpha_factor_bundle_v1",
        "status": status,
        "bundleId": bundle_id,
        "factorId": fast.candidate["factorId"],
        "hypothesisId": hypothesis["hypothesisId"],
        "ticketId": hypothesis["ticketId"],
        "archetypeId": hypothesis["archetypeId"],
        "stateHistory": state_history,
        "primary": {
            "candidate": fast.candidate,
            "fastTrainMetrics": fast.metrics,
            "behavioralFingerprint": fast.behavior,
            "fullMetrics": (
                primary_evaluation.summary["periods"]
                if primary_evaluation is not None
                else None
            ),
            "evaluationFailure": primary_reason,
        },
        "counter": {
            "candidate": counter,
            "fullMetrics": (
                counter_evaluation.summary["periods"]
                if counter_evaluation is not None
                else None
            ),
            "evaluationFailure": counter_reason,
        },
        "placebo": {
            "candidate": placebo,
            "fullMetrics": (
                placebo_evaluation.summary["periods"]
                if placebo_evaluation is not None
                else None
            ),
            "evaluationFailure": placebo_reason,
        },
        "failureCondition": (
            failure_condition_audit(
                primary_evaluation, panel, split, config, hypothesis
            )
            if primary_evaluation is not None
            else {
                "condition": hypothesis["failureCondition"],
                "status": "primary_evaluation_failed",
            }
        ),
        "purgedWalkForward": walk_forward,
        "rejectionReasons": reasons,
        "validationOrShadowUsedForGeneration": False,
        "historicalResultCanPromote": False,
        "selectionObjective": objective_group(config),
        "costMetricsAreDiagnosticNotSelectionGate": (
            objective_group(config) == "grossLongOnly"
        ),
    }
    return bundle, primary_evaluation


def apply_project_pbo(
    bundles: list[dict[str, Any]],
    primary_evaluations: list[autonomous.Evaluation],
    split: autonomous.Split,
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(primary_evaluations) < 2:
        return {"pbo": None, "reason": "need at least two Stage-2 primary factors"}
    common = sorted(
        set.intersection(
            *(
                set(
                    (
                        evaluation.gross_excess
                        if objective_group(config) == "grossLongOnly"
                        else evaluation.long_net
                    ).loc[split.train].dropna().index
                )
                for evaluation in primary_evaluations
            )
        )
    )
    if not common:
        return {"pbo": None, "reason": "no common train dates"}
    matrix = [
        (
            evaluation.gross_excess
            if objective_group(config) == "grossLongOnly"
            else evaluation.long_net
        ).reindex(common).fillna(0.0).tolist()
        for evaluation in primary_evaluations
    ]
    result = og.combinatorial_symmetric_pbo(matrix, n_blocks=8)
    pbo_value = result.get("pbo")
    if pbo_value is not None and float(pbo_value) > float(
        config["fullEvaluation"]["pboMaximum"]
    ):
        for bundle in bundles:
            if bundle["status"] == "HISTORICALLY_VALIDATED":
                bundle["status"] = "REJECTED"
                bundle["stateHistory"].append("REJECTED")
                bundle["rejectionReasons"].append("PROJECT_PBO_FAILED")
    return result


def render_report(result: dict[str, Any]) -> str:
    universe_kind = result["dataAudit"]["universe"].get(
        "universeKind", "unknown"
    )
    lines = [
        "# Perception-XAlpha Autonomous Factor Discovery v2",
        "",
        f"- run_id: `{result['runId']}`",
        f"- status: `{result['status']}`",
        f"- data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}`",
        (
            f"- universe: `{result['dataAudit']['symbols']}` instruments "
            f"(`{universe_kind}`)"
        ),
        f"- research questions: `{len(result['researchPlan']['questions'])}`",
        f"- discovery selection objective: `{result['candidateAudit'].get('selectionObjective')}`",
        (
            "- primary candidates generated / fast-screened / Stage-2: "
            f"`{result['candidateAudit']['generated']}` / "
            f"`{result['candidateAudit']['fastScreenPassed']}` / "
            f"`{result['candidateAudit']['stage2Bundles']}`"
        ),
        (
            "- credible historical research factors after Counter/Placebo/walk-forward/PBO/DSR: "
            f"`{result['candidateAudit']['historicallyValidated']}`"
        ),
        f"- PBO: `{result['guards']['pbo']}`",
        f"- quick DSR diagnostic: `{result['guards']['quickDsr']}`",
        "- costs: `reported as a separate stress test; never deleted from artifacts`",
        "- automatic trading changes: `[]`",
        "",
        "## Research plan",
        "",
        "| priority | phenomenon | mechanism archetype | score |",
        "|---:|---|---|---:|",
    ]
    for row in result["researchPlan"]["questions"]:
        lines.append(
            f"| {row['priority']} | {row['phenomenonId']} | "
            f"{row['archetypeId']} | {row['priorityScore']} |"
        )
    lines.extend(
        [
            "",
            "## Full factor bundles",
            "",
            "| factor | archetype | state | validation gross IR | validation net IR | turnover/day | cost/day | counter IR | placebo IR | rejection |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for bundle in result["factorBundles"]:
        comparison_group = result["candidateAudit"].get(
            "selectionObjective", "costedLongOnly"
        )
        primary_metrics = bundle["primary"].get("fullMetrics") or {}
        counter_metrics = bundle["counter"].get("fullMetrics") or {}
        placebo_metrics = bundle["placebo"].get("fullMetrics") or {}
        primary_ir = (
            primary_metrics.get("validation", {})
            .get("costedLongOnly", {})
            .get("irAnn")
        )
        primary_gross_ir = (
            primary_metrics.get("validation", {})
            .get("grossLongOnly", {})
            .get("irAnn")
        )
        turnover_mean = (
            primary_metrics.get("validation", {}).get("turnover", {}).get("mean")
        )
        cost_mean = (
            primary_metrics.get("validation", {}).get("costDrag", {}).get("mean")
        )
        counter_ir = (
            counter_metrics.get("validation", {})
            .get(comparison_group, {})
            .get("irAnn")
        )
        placebo_ir = (
            placebo_metrics.get("validation", {})
            .get(comparison_group, {})
            .get("irAnn")
        )
        lines.append(
            f"| {bundle['factorId']} | {bundle['archetypeId']} | "
            f"{bundle['status']} | {primary_gross_ir} | {primary_ir} | "
            f"{turnover_mean} | {cost_mean} | {counter_ir} | "
            f"{placebo_ir} | {', '.join(bundle['rejectionReasons'])} |"
        )
    lines.extend(
        [
            "",
            "## Rejection ledger",
            "",
            f"`{result['candidateAudit']['rejectedReasons']}`",
            "",
            "## Boundary",
            "",
            "Evolution saw train-only fast-screen metrics. Validation, Counter, Placebo, "
            "failure-condition and shadow results were quarantined. No historical result "
            "can create an instruction, modify configuration or promote itself. A gross "
            "research factor is not the same thing as a costed tradeable strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def input_fingerprint(
    config_path: Path,
    config: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    universe_audit: dict[str, Any],
) -> str:
    close = panel["close"]
    research_fields = sorted(key for key in panel if key.startswith("fund_"))
    research_field_fingerprint = {
        field: {
            str(key): None if pd.isna(value) else round(float(value), 8)
            for key, value in panel[field].iloc[-1].items()
        }
        for field in research_fields
    }
    auxiliary_sources = {}
    for name in ("research_ashare_universe.py", "research_ashare_fundamentals.py"):
        path = ROOT / "scripts" / name
        if path.exists():
            auxiliary_sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "configSha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "sourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "codeVersion": CODE_VERSION,
        "auxiliarySourceSha256": auxiliary_sources,
        "universeAudit": universe_audit,
        "start": close.index.min().isoformat(),
        "end": close.index.max().isoformat(),
        "shape": list(close.shape),
        "lastClose": {
            str(key): None if pd.isna(value) else round(float(value), 8)
            for key, value in close.iloc[-1].items()
        },
        "researchFieldLastValues": research_field_fingerprint,
    }
    return digest(payload)


def run_cycle(
    config_path: Path = DEFAULT_CONFIG,
    use_state: bool = True,
    force: bool = False,
    maximum_candidates: int | None = None,
) -> tuple[dict[str, Any], Path | None]:
    config = load_json(config_path)
    validate_config(config)
    perception_config, cog_config = load_base_configs(config)
    raw_panel, universe_audit = build_configured_panel(config, cog_config)
    if not universe_audit.get("historicalValidationEligible", False):
        raise RuntimeError(
            "research universe failed closed: incomplete coverage or insufficient "
            f"eligible symbols ({universe_audit})"
        )
    panel = perception.enrich_panel(raw_panel, perception_config)
    close = panel["close"]
    if close.empty:
        raise RuntimeError("no eligible daily A-share instrument panel")
    split = autonomous.make_split(close.index, cog_config)
    target, one_day = autonomous.target_frames(panel, cog_config)
    fingerprint = input_fingerprint(
        config_path, config, panel, universe_audit
    )
    state_directory = ROOT / config["registry"]["stateDirectory"]
    registry_path = ROOT / config["registry"]["sqlitePath"]
    connection: sqlite3.Connection | None = None
    if use_state:
        state_directory.mkdir(parents=True, exist_ok=True)
        # Single instance per state directory. Two concurrent runs share one SQLite registry
        # and one parent pool, so the later finisher overwrites the other's search memory --
        # observed when a v1 and a v2 cycle ran together against the same directory.
        lock_path = state_directory / "research.lock"
        try:
            lock_handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"another research run holds {lock_path}; concurrent cycles would corrupt "
                "the shared registry and parent pool. Remove the lock only if no run is live."
            )
        os.write(lock_handle, json.dumps({
            "pid": os.getpid(),
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "config": str(config.get("schemaVersion", "unknown")),
        }).encode("utf-8"))
        os.close(lock_handle)
        atexit.register(lambda: lock_path.unlink(missing_ok=True))
        connection = sqlite3.connect(registry_path)
        initialize_registry(connection)
        if (
            not force
            and config["registry"]["skipWhenDataFingerprintUnchanged"]
            and connection.execute(
                "SELECT 1 FROM run_index WHERE input_hash=? AND status='complete' "
                "LIMIT 1",
                (fingerprint,),
            ).fetchone()
        ):
            connection.close()
            return {
                "status": "no_new_data",
                "inputFingerprint": fingerprint,
                "automaticTradingChanges": [],
            }, None
    novelty_epoch = completed_cycle_count(connection)
    known_factor_ids = previous_factor_ids(connection)
    # The epoch is derived only from completed historical research cycles. It changes the
    # deterministic grammar path when new market data arrive, so a daily loop does not
    # regenerate the same formulas forever. It contains no validation/shadow information.
    config["_runtimeNoveltyEpoch"] = novelty_epoch
    generated_at = datetime.now(timezone.utc)
    cycle_id = (
        f"cycle_{generated_at:%Y%m%dT%H%M%SZ}_{fingerprint[:10]}"
    )
    output = ROOT / config["registry"]["outputRoot"] / cycle_id
    output.mkdir(parents=True, exist_ok=False)
    candidate_manifest_path = output / "candidate_manifest.jsonl"
    atomic_json(
        output / "run_status.json",
        {
            "schemaVersion": "perception_xalpha_active_cycle_v1",
            "status": "running_research_only_not_trading",
            "runId": cycle_id,
            "startedAt": generated_at.isoformat(),
            "candidateBudget": int(
                config["synthesis"]["maximumPrimaryCandidatesPerCycle"]
            ),
            "evaluatedCandidates": 0,
            "orders": [],
            "automaticTradingChanges": [],
        },
    )
    quality_audit = perception.data_quality_audit(panel, perception_config)
    scores = perception.detector_scores(panel, perception_config)
    tickets, ticket_audit = perception.build_tickets(
        scores,
        perception_config,
        fingerprint,
        digest(config),
    )
    historical_counts = previous_archetype_counts(connection)
    cycle_request, research_plan, hypotheses = build_research_plan(
        tickets, config, historical_counts, cycle_id
    )
    candidates = initial_candidates(
        hypotheses, config, cog_config, state_directory, use_state
    )
    budget = int(config["synthesis"]["maximumPrimaryCandidatesPerCycle"])
    if maximum_candidates is not None:
        budget = min(budget, max(1, int(maximum_candidates)))
    evaluated_fingerprints: set[str] = set()
    accepted: list[FastResult] = []
    rejected_rows: list[dict[str, Any]] = []
    near_miss: list[tuple[float, dict[str, Any]]] = []
    generated_count = 0

    def evaluate_rows(rows: list[dict[str, Any]]) -> None:
        nonlocal generated_count
        for candidate in rows:
            if generated_count >= budget:
                return
            if candidate["fingerprint"] in evaluated_fingerprints:
                continue
            evaluated_fingerprints.add(candidate["fingerprint"])
            generated_count += 1
            fast, reason, audit = fast_screen(
                candidate,
                panel,
                target,
                one_day,
                split,
                config,
                cog_config,
                accepted,
            )
            if fast is None:
                # Keep a train-only ranking of screened-out candidates so evolution can still
                # breed when nothing clears the screen. Without this the search dies at
                # generation 1 whenever the screen is strict (observed: 12 candidates, 0
                # survivors, generations 2..N never ran) -- an evolutionary algorithm that
                # cannot evolve. Fitness here is |train rank-IC t|, which is train-only and
                # therefore does not breach the validation/shadow quarantine.
                # fast_screen reports metrics["rankIc"] as a FLAT period_stats block
                # ({n, mean, t, irAnn}) already restricted to train -- there is no nested
                # "train" key. Reading one used to leave near_miss permanently empty, so the
                # fallback below never fired and evolution stayed dead while appearing wired.
                rank_block = audit.get("rankIc") if isinstance(audit, dict) else None
                if isinstance(rank_block, dict) and rank_block.get("t") is not None:
                    near_miss.append((abs(float(rank_block["t"])), candidate))
                rejected_rows.append(
                    {
                        "rejectionId": "rejection_"
                        + digest(
                            {
                                "cycleId": cycle_id,
                                "factorId": candidate["factorId"],
                                "reason": reason,
                            }
                        )[:20],
                        "cycleId": cycle_id,
                        "factorId": candidate["factorId"],
                        "stage": "fast_screen",
                        "reason": reason,
                        "audit": audit,
                        "immutable": True,
                    }
                )
            else:
                accepted.append(fast)
            append_jsonl_record(
                candidate_manifest_path,
                {
                    "schemaVersion": "perception_xalpha_candidate_manifest_v1",
                    "status": "research_only_not_a_trade_signal",
                    "runId": cycle_id,
                    "ordinal": generated_count,
                    "factorId": candidate["factorId"],
                    "fingerprint": candidate["fingerprint"],
                    "archetypeId": candidate["archetypeId"],
                    "agent": candidate["agent"],
                    "generation": candidate["generation"],
                    "parents": candidate["parents"],
                    "source": candidate["source"],
                    "expression": candidate["expression"],
                    "expressionText": expression_text(candidate["expression"]),
                    "fastScreenPassed": fast is not None,
                    "fastScreenRejection": reason,
                    "fastTrainMetrics": audit,
                    "orders": [],
                    "automaticTradingChanges": [],
                },
            )
            atomic_json(
                output / "run_status.json",
                {
                    "schemaVersion": "perception_xalpha_active_cycle_v1",
                    "status": "running_research_only_not_trading",
                    "runId": cycle_id,
                    "startedAt": generated_at.isoformat(),
                    "candidateBudget": budget,
                    "evaluatedCandidates": generated_count,
                    "fastScreenPassed": len(accepted),
                    "orders": [],
                    "automaticTradingChanges": [],
                },
            )

    evaluate_rows(candidates)
    generation_audit: list[dict[str, Any]] = []
    for generation in range(1, int(config["synthesis"]["generationsPerCycle"]) + 1):
        pool_size = int(config["synthesis"]["parentPoolSize"])
        parents = sorted(accepted, key=lambda item: item.fitness, reverse=True)[:pool_size]
        seeded = 0
        minimum_parents = int(config["synthesis"].get("minimumParentsToContinue", 0))
        if (
            bool(config["synthesis"].get("seedParentsFromBestScreened", False))
            and len(parents) < max(minimum_parents, 1)
        ):
            for _score, candidate in sorted(near_miss, key=lambda item: -item[0]):
                if len(parents) >= max(minimum_parents, 1):
                    break
                parents.append(_SeedParent(candidate))
                seeded += 1
        generation_audit.append(
            {
                "generation": generation,
                "parents": [parent.candidate["factorId"] for parent in parents],
                "seededFromScreenedOut": seeded,
                "selectionData": "train_only",
                "validationFeedbackUsed": False,
                "shadowFeedbackUsed": False,
            }
        )
        if not parents or generated_count >= budget:
            break
        evaluate_rows(
            evolve_candidates(
                parents, hypotheses, generation, config, cog_config
            )
        )
    selected = sorted(accepted, key=lambda item: item.fitness, reverse=True)[
        : int(config["synthesis"]["maximumStage2Bundles"])
    ]
    hypothesis_map = {
        hypothesis["hypothesisId"]: hypothesis for hypothesis in hypotheses
    }
    bundles: list[dict[str, Any]] = []
    primary_evaluations: list[autonomous.Evaluation] = []
    for fast in selected:
        bundle, primary_evaluation = full_bundle_evaluation(
            fast,
            hypothesis_map[fast.candidate["hypothesisId"]],
            panel,
            target,
            one_day,
            split,
            config,
            cog_config,
        )
        bundles.append(bundle)
        if primary_evaluation is not None:
            primary_evaluations.append(primary_evaluation)
    pbo = apply_project_pbo(bundles, primary_evaluations, split, config)
    # Preserve the pre-DSR outcome as a candidate tier. This allows the daily research
    # loop to produce useful hypotheses without lying that each one has survived the
    # project-wide multiple-testing burden. The credible tier below remains stricter.
    for bundle in bundles:
        bundle["preMultipleTestingStatus"] = bundle["status"]
    for bundle in bundles:
        for reason in bundle["rejectionReasons"]:
            rejected_rows.append(
                {
                    "rejectionId": "rejection_"
                    + digest(
                        {
                            "cycleId": cycle_id,
                            "factorId": bundle["factorId"],
                            "reason": reason,
                            "stage": "full_evaluation",
                        }
                    )[:20],
                    "cycleId": cycle_id,
                    "factorId": bundle["factorId"],
                    "stage": "full_evaluation",
                    "reason": reason,
                    "immutable": True,
                }
            )
    total_trials = (
        int(config["fullEvaluation"]["priorProjectTrials"])
        + generated_count
    )
    selected_metric_group = objective_group(config)
    best_validation = (
        max(
            primary_evaluations,
            key=lambda item: metric_value(
                item, "validation", selected_metric_group, "irAnn"
            ),
        )
        if primary_evaluations
        else None
    )
    quick_dsr = (
        og.deflated_significance_note(
            total_trials,
            metric_value(
                best_validation, "validation", selected_metric_group, "irAnn", 0.0
            ),
            int(
                metric_value(
                    best_validation, "validation", selected_metric_group, "n", 0.0
                )
            ),
        )
        if best_validation is not None
        else {"flag": "insufficient", "expected_max_noise_sharpe": None}
    )
    if quick_dsr.get("flag") != "exceeds_noise_max":
        for bundle in bundles:
            if bundle["status"] != "HISTORICALLY_VALIDATED":
                continue
            bundle["status"] = "REJECTED"
            bundle["stateHistory"].append("REJECTED")
            bundle["rejectionReasons"].append("MULTIPLE_TESTING_DSR_FAILED")
            rejected_rows.append(
                {
                    "rejectionId": "rejection_"
                    + digest(
                        {
                            "cycleId": cycle_id,
                            "factorId": bundle["factorId"],
                            "reason": "MULTIPLE_TESTING_DSR_FAILED",
                            "stage": "multiple_testing",
                        }
                    )[:20],
                    "cycleId": cycle_id,
                    "factorId": bundle["factorId"],
                    "stage": "multiple_testing",
                    "reason": "MULTIPLE_TESTING_DSR_FAILED",
                    "immutable": True,
                }
            )
    rejection_counts: dict[str, int] = {}
    for row in rejected_rows:
        reason = str(row["reason"])
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    unknown_reasons = set(rejection_counts) - REJECTION_REASONS
    if unknown_reasons:
        raise AssertionError(
            f"unregistered rejection reasons: {sorted(unknown_reasons)}"
        )
    validated = [
        bundle
        for bundle in bundles
        if bundle["status"] == "HISTORICALLY_VALIDATED"
    ]
    target_count = int(
        config.get("discoveryObjective", {}).get(
            "targetCredibleFactorsPerCycle", len(validated)
        )
    )
    credible_factors = []
    for bundle in validated[:target_count]:
        metrics = bundle["primary"].get("fullMetrics") or {}
        credible_factors.append(
            {
                "factorId": bundle["factorId"],
                "archetypeId": bundle["archetypeId"],
                "expression": bundle["primary"]["candidate"]["expression"],
                "directionFromTrain": bundle["primary"]["fastTrainMetrics"].get(
                    "directionFromTrain"
                ),
                "train": metrics.get("train"),
                "validation": metrics.get("validation"),
                "shadow": metrics.get("shadow"),
                "purgedWalkForward": bundle.get("purgedWalkForward"),
                "selectionObjective": selected_metric_group,
                "costStressDidNotGateDiscovery": (
                    selected_metric_group == "grossLongOnly"
                ),
                "status": "historical_research_factor_not_a_trade_signal",
            }
        )
    daily_factor_candidates = []
    for bundle in bundles:
        if bundle.get("preMultipleTestingStatus") != "HISTORICALLY_VALIDATED":
            continue
        metrics = bundle["primary"].get("fullMetrics") or {}
        daily_factor_candidates.append(
            {
                "factorId": bundle["factorId"],
                "archetypeId": bundle["archetypeId"],
                "expression": bundle["primary"]["candidate"]["expression"],
                "directionFromTrain": bundle["primary"]["fastTrainMetrics"].get(
                    "directionFromTrain"
                ),
                "train": metrics.get("train"),
                "validation": metrics.get("validation"),
                "shadow": metrics.get("shadow"),
                "purgedWalkForward": bundle.get("purgedWalkForward"),
                "selectionObjective": selected_metric_group,
                "remainingGuards": bundle.get("rejectionReasons", []),
                "isNewFactor": bundle["factorId"] not in known_factor_ids,
                "status": "research_candidate_not_yet_credible_not_a_trade_signal",
            }
        )
    daily_factor_candidates.sort(
        key=lambda row: float(
            (row.get("validation") or {})
            .get(selected_metric_group, {})
            .get("irAnn")
            or -99.0
        ),
        reverse=True,
    )
    daily_factor_candidates.sort(key=lambda row: not bool(row["isNewFactor"]))
    daily_factor_candidates = daily_factor_candidates[:target_count]
    result = {
        "schemaVersion": "perception_xalpha_autonomous_result_v2",
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": cycle_id,
        "generatedAt": generated_at.isoformat(),
        "codeVersion": CODE_VERSION,
        "inputFingerprint": fingerprint,
        "dataAudit": {
            "start": close.index.min().date().isoformat(),
            "end": close.index.max().date().isoformat(),
            "days": int(len(close)),
            "symbols": int(len(close.columns)),
            "barInterval": "1d",
            "quality": quality_audit,
            "universe": universe_audit,
        },
        "splitAudit": split.audit,
        "ticketAudit": ticket_audit,
        "researchCycleRequest": cycle_request,
        "researchPlan": research_plan,
        "mechanismHypotheses": hypotheses,
        "candidateAudit": {
            "generated": generated_count,
            "fastScreenPassed": len(accepted),
            "stage2Bundles": len(bundles),
            "historicallyValidated": len(validated),
            "dailyResearchCandidates": len(daily_factor_candidates),
            "rejected": len(rejected_rows),
            "rejectedReasons": rejection_counts,
            "generationAudit": generation_audit,
            "provider": "deterministic_local_grammar",
            "remoteApiUsed": False,
            "arbitraryPythonExecuted": False,
            "validationOrShadowFedBack": False,
            "selectionObjective": selected_metric_group,
            "targetCredibleFactorsPerCycle": int(
                config.get("discoveryObjective", {}).get(
                    "targetCredibleFactorsPerCycle", 0
                )
            ),
            "noveltyEpoch": novelty_epoch,
            "previouslyKnownFactorCount": len(known_factor_ids),
            "newDailyResearchCandidates": sum(
                int(bool(row["isNewFactor"])) for row in daily_factor_candidates
            ),
        },
        "factorBundles": bundles,
        "credibleResearchFactors": credible_factors,
        "dailyResearchCandidates": daily_factor_candidates,
        "guards": {
            "pbo": pbo,
            "quickDsr": quick_dsr,
            "quickDsrIsFullDsr": False,
            "multipleTestingMetric": selected_metric_group,
            "priorProjectTrials": int(
                config["fullEvaluation"]["priorProjectTrials"]
            ),
            "currentCycleTrials": generated_count,
            "totalTrials": total_trials,
            "sameHistoricalWindowCannotPromote": True,
            "humanApprovalRequired": True,
        },
        "orders": [],
        "automaticTradingChanges": [],
        "verdict": (
            "Historical controls were passed, but the candidates remain research-only "
            "and require a separately preregistered fresh-forward study."
            if validated
            else "No candidate passed the full Primary/Counter/Placebo and project "
            "overfitting guards."
        ),
    }
    atomic_json(output / "result.json", result)
    atomic_json(output / "research_cycle_request.json", cycle_request)
    atomic_json(output / "research_plan.json", research_plan)
    atomic_json(
        output / "phenomenon_tickets.json", {"tickets": tickets}
    )
    atomic_json(
        output / "mechanism_hypotheses.json", {"hypotheses": hypotheses}
    )
    atomic_json(output / "factor_bundles.json", {"bundles": bundles})
    atomic_json(
        output / "credible_research_factors.json",
        {
            "schemaVersion": "perception_xalpha_credible_research_factors_v1",
            "status": "historical_research_only_not_trade_signals",
            "runId": cycle_id,
            "selectionObjective": selected_metric_group,
            "targetCount": target_count,
            "actualCount": len(credible_factors),
            "factors": credible_factors,
            "orders": [],
            "automaticTradingChanges": [],
        },
    )
    atomic_json(
        output / "daily_factor_candidates.json",
        {
            "schemaVersion": "perception_xalpha_daily_factor_candidates_v1",
            "status": "research_candidates_not_trade_signals",
            "runId": cycle_id,
            "selectionObjective": selected_metric_group,
            "targetCount": target_count,
            "actualCount": len(daily_factor_candidates),
            "candidates": daily_factor_candidates,
            "credibleCount": len(credible_factors),
            "orders": [],
            "automaticTradingChanges": [],
        },
    )
    manifest_rows = [
        json.loads(line)
        for line in candidate_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (output / "candidate_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ordinal",
                "factorId",
                "archetypeId",
                "agent",
                "generation",
                "source",
                "expressionText",
                "fastScreenPassed",
                "fastScreenRejection",
                "parents",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), ensure_ascii=False)
                        if key == "parents"
                        else row.get(key)
                    )
                    for key in writer.fieldnames
                }
            )
    atomic_json(
        output / "run_status.json",
        {
            "schemaVersion": "perception_xalpha_active_cycle_v1",
            "status": "complete_research_only_not_trading",
            "runId": cycle_id,
            "startedAt": generated_at.isoformat(),
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "candidateBudget": budget,
            "evaluatedCandidates": generated_count,
            "fastScreenPassed": len(accepted),
            "stage2Bundles": len(bundles),
            "dailyResearchCandidates": len(daily_factor_candidates),
            "credibleResearchFactors": len(credible_factors),
            "orders": [],
            "automaticTradingChanges": [],
        },
    )
    atomic_json(
        output / "shadow_candidate.json",
        {
            "schemaVersion": "perception_xalpha_autonomous_shadow_v1",
            "status": "research_only_not_a_trade_signal",
            "runId": cycle_id,
            "eligibleFactorIds": [
                bundle["factorId"] for bundle in validated
            ],
            "orders": [],
            "automaticTradingChanges": [],
            "warning": (
                "Historical research only. No factor is connected to observation, "
                "positioning or execution."
            ),
        },
    )
    (output / "report.md").write_text(
        render_report(result) + "\n", encoding="utf-8"
    )
    if use_state and connection is not None:
        insert_entity(connection, "research_cycles", cycle_id, cycle_request)
        insert_entity(
            connection, "research_plans", research_plan["planId"], research_plan
        )
        for hypothesis in hypotheses:
            insert_entity(
                connection,
                "mechanism_hypotheses",
                hypothesis["hypothesisId"],
                hypothesis,
            )
        for bundle in bundles:
            insert_entity(
                connection, "factor_bundles", bundle["bundleId"], bundle
            )
            candidate = bundle["primary"]["candidate"]
            observation = {
                "observationId": "factor_observation_"
                + digest({"cycleId": cycle_id, "factorId": bundle["factorId"]})[:20],
                "cycleId": cycle_id,
                "factorId": bundle["factorId"],
                "fingerprint": candidate.get("fingerprint"),
                "archetypeId": bundle.get("archetypeId"),
                "expression": candidate.get("expression"),
                "preMultipleTestingStatus": bundle.get("preMultipleTestingStatus"),
                "finalStatus": bundle.get("status"),
                "rejectionReasons": bundle.get("rejectionReasons", []),
                "metrics": bundle["primary"].get("fullMetrics"),
                "immutable": True,
            }
            insert_entity(
                connection,
                "factor_observations",
                observation["observationId"],
                observation,
            )
        for fast in accepted:
            experiment = {
                "experimentId": "experiment_"
                + digest(
                    {
                        "cycleId": cycle_id,
                        "factorId": fast.candidate["factorId"],
                    }
                )[:20],
                "cycleId": cycle_id,
                "factorId": fast.candidate["factorId"],
                "candidate": fast.candidate,
                "fastTrainMetrics": fast.metrics,
                "behavioralFingerprint": fast.behavior,
                "containsValidationMetrics": False,
                "containsShadowMetrics": False,
                "immutable": True,
            }
            insert_entity(
                connection,
                "experiments",
                experiment["experimentId"],
                experiment,
            )
        for rejection in rejected_rows:
            insert_entity(
                connection,
                "rejection_events",
                rejection["rejectionId"],
                rejection,
            )
        connection.execute(
            "INSERT INTO run_index "
            "(run_id, created_at, input_hash, status, output_path, payload) "
            "VALUES (?, ?, ?, 'complete', ?, ?)",
            (
                cycle_id,
                generated_at.isoformat(),
                fingerprint,
                str(output),
                canonical(result["candidateAudit"]),
            ),
        )
        connection.commit()
        atomic_json(
            state_directory / "cumulative_factor_catalog.json",
            cumulative_factor_catalog(connection),
        )
        parent_rows = sorted(
            accepted, key=lambda item: item.fitness, reverse=True
        )[: int(config["synthesis"]["maximumPersistentParents"])]
        atomic_json(
            state_directory / "train_parent_pool.json",
            {
                "schemaVersion": "perception_xalpha_train_parent_pool_v1",
                "status": "train_only_research_memory",
                # Parents bred under one book definition are not interchangeable with another:
                # a v1 run finishing after a v2 run would otherwise hand v2 a pool selected
                # under the old screen and construction. Readers must match this exactly.
                "bookIdentitySha256": book_identity(cog_config),
                "sourceRunId": cycle_id,
                "containsValidationMetrics": False,
                "containsShadowMetrics": False,
                "parents": [
                    {
                        "factorId": row.candidate["factorId"],
                        "archetypeId": row.candidate["archetypeId"],
                        "expression": row.candidate["expression"],
                        "fastTrainMetrics": row.metrics,
                    }
                    for row in parent_rows
                ],
            },
        )
        connection.close()
    return result, output


def status(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    registry_path = ROOT / config["registry"]["sqlitePath"]
    if not registry_path.exists():
        return {"status": "not_initialized", "registry": str(registry_path)}
    connection = sqlite3.connect(registry_path)
    initialize_registry(connection)
    counts = {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in APPEND_ONLY_TABLES
    }
    latest = connection.execute(
        "SELECT run_id, created_at, status, output_path FROM run_index "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    connection.close()
    return {
        "status": "ready",
        "registry": str(registry_path),
        "counts": counts,
        "latest": (
            {
                "runId": latest[0],
                "createdAt": latest[1],
                "status": latest[2],
                "output": latest[3],
            }
            if latest
            else None
        ),
    }


def queue(config_path: Path, limit: int) -> dict[str, Any]:
    state = status(config_path)
    latest = state.get("latest")
    if not latest:
        return {"status": "empty", "questions": []}
    path = Path(latest["output"]) / "research_plan.json"
    plan = load_json(path)
    return {
        "status": "ready",
        "runId": latest["runId"],
        "questions": plan.get("questions", [])[: max(1, limit)],
    }


def factor(config_path: Path, factor_id: str) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    registry_path = ROOT / config["registry"]["sqlitePath"]
    if not registry_path.exists():
        return {"status": "not_found", "factorId": factor_id}
    connection = sqlite3.connect(registry_path)
    initialize_registry(connection)
    matches = []
    for (payload_text,) in connection.execute(
        "SELECT payload FROM factor_bundles ORDER BY created_at DESC"
    ):
        payload = json.loads(payload_text)
        if payload.get("factorId") == factor_id:
            matches.append(payload)
    connection.close()
    return {
        "status": "found" if matches else "not_found",
        "factorId": factor_id,
        "bundles": matches,
    }


def self_test() -> None:
    config = load_json(DEFAULT_CONFIG)
    validate_config(config)
    connection = sqlite3.connect(":memory:")
    initialize_registry(connection)
    payload = {"cycleId": "cycle_test", "immutable": True}
    insert_entity(connection, "research_cycles", "cycle_test", payload)
    failed_closed = False
    try:
        connection.execute(
            "UPDATE research_cycles SET payload='changed' "
            "WHERE entity_id='cycle_test'"
        )
    except sqlite3.DatabaseError:
        failed_closed = True
    if not failed_closed:
        raise AssertionError("append-only research cycle accepted an update")
    connection.close()
    if "PREFIX_LEAKAGE" not in REJECTION_REASONS:
        raise AssertionError("required rejection reason missing")
    print("Perception-XAlpha autonomous v2 self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--no-state", action="store_true")
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--maximum-candidates", type=int, default=None)
    subparsers.add_parser("status")
    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("--limit", type=int, default=10)
    factor_parser = subparsers.add_parser("factor")
    factor_parser.add_argument("--factor-id", required=True)
    subparsers.add_parser("self-test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "run"
    if command == "self-test":
        self_test()
        return 0
    if command == "status":
        print(json.dumps(status(args.config), ensure_ascii=False, indent=2))
        return 0
    if command == "queue":
        print(
            json.dumps(
                queue(args.config, args.limit), ensure_ascii=False, indent=2
            )
        )
        return 0
    if command == "factor":
        print(
            json.dumps(
                factor(args.config, args.factor_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result, output = run_cycle(
        args.config,
        use_state=not getattr(args, "no_state", False),
        force=getattr(args, "force", False),
        maximum_candidates=getattr(args, "maximum_candidates", None),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "runId": result.get("runId"),
                "output": str(output) if output else None,
                "generated": result.get("candidateAudit", {}).get("generated"),
                "fastScreenPassed": result.get("candidateAudit", {}).get(
                    "fastScreenPassed"
                ),
                "stage2Bundles": result.get("candidateAudit", {}).get(
                    "stage2Bundles"
                ),
                "historicallyValidated": result.get("candidateAudit", {}).get(
                    "historicallyValidated"
                ),
                "automaticTradingChanges": result.get(
                    "automaticTradingChanges", []
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
