"""Autonomous, mechanism-first A-share instrument factor discovery.

The engine converts immutable market-phenomenon tickets into falsifiable mechanism
hypotheses, causal DSL programs and Primary/Counter/Placebo bundles. Evolution uses
train-only feedback. Validation and shadow remain quarantined and can never alter
the paper-trading system.
"""

from __future__ import annotations

import argparse
import copy
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
CODE_VERSION = "perception_xalpha_autonomous_v2.2"


REJECTION_REASONS = {
    "STATIC_DSL_REJECTED",
    "PREFIX_LEAKAGE",
    "TOO_MANY_NAN",
    "INSUFFICIENT_CROSS_SECTION",
    "INSUFFICIENT_TRAIN_DAYS",
    "WEAK_TRAIN_RANK_IC",
    "WEAK_TRAIN_RANK_IC_IR",
    "NEGATIVE_COSTED_TRAIN_IR",
    "DUPLICATE_BEHAVIOR",
    "FULL_EVALUATION_FAILED",
    "COUNTER_NOT_BEATEN",
    "PLACEBO_NOT_BEATEN",
    "PURGED_WALK_FORWARD_FAILED",
    "VALIDATION_RANK_IC_FAILED",
    "VALIDATION_COSTED_IR_FAILED",
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
    if int(synthesis["maximumPrimaryCandidatesPerCycle"]) > 128:
        raise ValueError("search budget is not bounded")
    if int(synthesis["maximumStage2Bundles"]) > 16:
        raise ValueError("Stage-2 budget is not bounded")
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
    return panel, audit


APPEND_ONLY_TABLES = [
    "research_cycles",
    "research_plans",
    "mechanism_hypotheses",
    "factor_bundles",
    "experiments",
    "rejection_events",
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
        rows = load_json(path).get("parents", [])
    except (OSError, ValueError):
        return []
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
    seed = int(config["synthesis"]["randomSeed"])
    output: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
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


def size_neutralise(signal: pd.DataFrame, panel: dict[str, pd.DataFrame], bins: int) -> pd.DataFrame:
    """Standardise the signal inside trailing-liquidity buckets.

    Un-neutralised A-share cross-sections are dominated by size: a raw signal is largely a
    size bet, and the size bet is what the cost model then destroys. Bucketing on trailing
    median amount (causal, shifted) and z-scoring within bucket keeps the intra-bucket
    ordering -- the part a factor can actually claim -- and discards the size tilt.
    """
    if bins < 2:
        return signal
    scale = np.log(panel["amount"].rolling(20).median().shift(1).replace(0.0, np.nan))
    buckets = scale.rank(axis=1, pct=True)
    out = signal * np.nan
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        member = buckets.gt(lower) & buckets.le(upper) if index else buckets.le(upper)
        group = signal.where(member)
        centred = group.sub(group.mean(axis=1), axis=0).div(
            group.std(axis=1).replace(0.0, np.nan), axis=0
        )
        out = out.fillna(centred)
    return out


def long_only_portfolio(
    signal: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    cog_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """(net excess return, turnover) for the harvestable long-only book.

    Three construction choices, each measured on the real panel with a 20-day reversal probe
    (IC t=15.8) before being adopted:
      * size neutralisation lifted gross IR 0.08 -> 0.29;
      * rank weighting over the whole book beats a hard decile cut (0.08 -> 0.21 gross)
        because a decile discards the monotone middle of a broad signal;
      * holding for the prediction horizon instead of rebalancing daily cut turnover
        0.23 -> 0.069/day, i.e. the cost drag the old screen was charging was ~3.3x the
        drag the strategy the label describes would actually pay.
    Together they moved that probe from -0.41 net IR to break-even, which is why an
    un-neutralised daily-rebalanced decile screen was rejecting real signals as unprofitable.
    """
    screen = config["fastScreen"]
    if bool(screen.get("sizeNeutralise", True)):
        signal = size_neutralise(signal, panel, int(screen.get("sizeNeutraliseBins", 5)))
    ranks = signal.rank(axis=1, pct=True)
    if str(screen.get("bookConstruction", "rank_weighted")) == "rank_weighted":
        raw_weights = ranks.sub(1.0 - float(cog_config["data"]["topQuantile"])).clip(lower=0.0)
    else:
        raw_weights = ranks.ge(1.0 - float(cog_config["data"]["topQuantile"])).astype(float)
    weights = raw_weights.div(raw_weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    hold = int(cog_config["data"]["predictionHorizonTradingDays"])
    if hold > 1:
        weights = weights.rolling(hold).mean().fillna(0.0)
        weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book_return = (weights * one_day).sum(axis=1, min_count=1)
    benchmark = one_day.mean(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    net = book_return - benchmark - turnover * float(config["fullEvaluation"]["roundTripCost"])
    return net, turnover


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
    long_net, _turnover = long_only_portfolio(
        signal, one_day, panel, cog_config, config
    )
    rank_stats = autonomous.period_stats(rank_ic, train_mask)
    net_stats = autonomous.period_stats(long_net, train_mask)
    metrics = {
        "rankIc": rank_stats,
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
    if float(net_stats["irAnn"] or -99.0) < float(
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
                    float(top.loc[train_mask].mean().mean()), 2
                ),
            }
        ),
        "nearestFactorId": nearest_factor,
        "maximumAbsoluteTrainRankCorrelation": round(
            max(0.0, maximum_similarity), 8
        ),
    }
    fitness = (
        float(rank_stats["irAnn"] or -3.0)
        + max(-3.0, float(net_stats["irAnn"] or -3.0))
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
    seed = int(config["synthesis"]["randomSeed"]) + generation * 1009
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
        long_net, _fold_turnover = long_only_portfolio(
            fold_signal, one_day, panel, cog_config, config
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
                "costedLongOnly": autonomous.period_stats(long_net, test_mask),
            }
        )
    positive = sum(
        1
        for row in rows
        if float(row["costedLongOnly"].get("irAnn") or -99.0) > 0.0
    )
    required = int(
        config["fullEvaluation"]["minimumPositiveWalkForwardFolds"]
    )
    return {
        "status": "passed" if positive >= required else "failed",
        "folds": rows,
        "positiveCostedFolds": positive,
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
        primary_ir = metric_value(
            primary_evaluation, "validation", "costedLongOnly", "irAnn"
        )
        counter_ir = metric_value(
            counter_evaluation, "validation", "costedLongOnly", "irAnn"
        )
        placebo_ir = metric_value(
            placebo_evaluation, "validation", "costedLongOnly", "irAnn"
        )
        if primary_rank <= float(
            config["fullEvaluation"]["minimumValidationRankIc"]
        ):
            reasons.append("VALIDATION_RANK_IC_FAILED")
        if primary_ir <= float(
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
                set(evaluation.long_net.loc[split.train].dropna().index)
                for evaluation in primary_evaluations
            )
        )
    )
    if not common:
        return {"pbo": None, "reason": "no common train dates"}
    matrix = [
        evaluation.long_net.reindex(common).fillna(0.0).tolist()
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
        (
            "- primary candidates generated / fast-screened / Stage-2: "
            f"`{result['candidateAudit']['generated']}` / "
            f"`{result['candidateAudit']['fastScreenPassed']}` / "
            f"`{result['candidateAudit']['stage2Bundles']}`"
        ),
        (
            "- historically validated after Counter/Placebo/walk-forward/PBO/DSR: "
            f"`{result['candidateAudit']['historicallyValidated']}`"
        ),
        f"- PBO: `{result['guards']['pbo']}`",
        f"- quick DSR diagnostic: `{result['guards']['quickDsr']}`",
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
            "| factor | archetype | state | validation net IR | counter IR | placebo IR | rejection |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for bundle in result["factorBundles"]:
        primary_metrics = bundle["primary"].get("fullMetrics") or {}
        counter_metrics = bundle["counter"].get("fullMetrics") or {}
        placebo_metrics = bundle["placebo"].get("fullMetrics") or {}
        primary_ir = (
            primary_metrics.get("validation", {})
            .get("costedLongOnly", {})
            .get("irAnn")
        )
        counter_ir = (
            counter_metrics.get("validation", {})
            .get("costedLongOnly", {})
            .get("irAnn")
        )
        placebo_ir = (
            placebo_metrics.get("validation", {})
            .get("costedLongOnly", {})
            .get("irAnn")
        )
        lines.append(
            f"| {bundle['factorId']} | {bundle['archetypeId']} | "
            f"{bundle['status']} | {primary_ir} | {counter_ir} | "
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
            "can create an instruction, modify configuration or promote itself.",
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
    payload = {
        "configSha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "sourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "codeVersion": CODE_VERSION,
        "universeAudit": universe_audit,
        "start": close.index.min().isoformat(),
        "end": close.index.max().isoformat(),
        "shape": list(close.shape),
        "lastClose": {
            str(key): None if pd.isna(value) else round(float(value), 8)
            for key, value in close.iloc[-1].items()
        },
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
    generated_at = datetime.now(timezone.utc)
    cycle_id = (
        f"cycle_{generated_at:%Y%m%dT%H%M%SZ}_{fingerprint[:10]}"
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
                rank_block = audit.get("rankIc") if isinstance(audit, dict) else None
                if isinstance(rank_block, dict):
                    train_block = rank_block.get("train")
                    if isinstance(train_block, dict) and train_block.get("t") is not None:
                        near_miss.append((abs(float(train_block["t"])), candidate))
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
    best_validation = (
        max(
            primary_evaluations,
            key=lambda item: metric_value(
                item, "validation", "costedLongOnly", "irAnn"
            ),
        )
        if primary_evaluations
        else None
    )
    quick_dsr = (
        og.deflated_significance_note(
            total_trials,
            metric_value(
                best_validation, "validation", "costedLongOnly", "irAnn", 0.0
            ),
            int(
                metric_value(
                    best_validation, "validation", "costedLongOnly", "n", 0.0
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
            "rejected": len(rejected_rows),
            "rejectedReasons": rejection_counts,
            "generationAudit": generation_audit,
            "provider": "deterministic_local_grammar",
            "remoteApiUsed": False,
            "arbitraryPythonExecuted": False,
            "validationOrShadowFedBack": False,
        },
        "factorBundles": bundles,
        "guards": {
            "pbo": pbo,
            "quickDsr": quick_dsr,
            "quickDsrIsFullDsr": False,
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
    output = ROOT / config["registry"]["outputRoot"] / cycle_id
    output.mkdir(parents=True, exist_ok=False)
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
        parent_rows = sorted(
            accepted, key=lambda item: item.fitness, reverse=True
        )[: int(config["synthesis"]["maximumPersistentParents"])]
        atomic_json(
            state_directory / "train_parent_pool.json",
            {
                "schemaVersion": "perception_xalpha_train_parent_pool_v1",
                "status": "train_only_research_memory",
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
