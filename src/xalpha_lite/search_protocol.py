"""Bounded AlphaBench-inspired orchestration for symbolic factor search.

Search paradigms may propose and order candidates using train-only evidence. They
never judge factor validity: the existing PIT, Counter, Placebo, purged
walk-forward, PBO and DSR pipeline remains the only acceptance path.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


PARADIGMS = ("coe", "tot", "ea")
OPERATORS = {
    "temporal_zscore",
    "rolling_refinement",
    "causal_lag_mutation",
    "mechanism_crossover_mul",
    "mechanism_crossover_add",
}


def validate_search_protocol(protocol: dict[str, Any], candidate_budget: int) -> None:
    """Fail closed when a search protocol can inspect held-out evidence or expand unboundedly."""
    if protocol.get("schema_version") != "alphabench_search_protocol_v1":
        raise ValueError("unexpected AlphaBench-inspired search protocol schema")
    if protocol.get("status") != "research_only_not_trading":
        raise ValueError("search protocol must remain research-only")
    if protocol.get("proposal_role") != "candidate_generation_and_queue_ordering_only":
        raise ValueError("proposal layer may only generate or order candidates")
    if protocol.get("absolute_factor_judgement_allowed") is not False:
        raise ValueError("absolute zero-shot factor judgement is prohibited")
    if protocol.get("selection_evidence") != "train_only":
        raise ValueError("candidate search must use train-only evidence")
    if (
        protocol.get("validation_feedback_allowed") is not False
        or protocol.get("shadow_feedback_allowed") is not False
    ):
        raise ValueError("validation and shadow feedback must remain quarantined")
    if protocol.get("final_judge") != "pit_counter_placebo_purged_pbo_dsr":
        raise ValueError("the deterministic falsification pipeline must remain the final judge")
    schedule = list(protocol.get("generation_schedule", []))
    if not schedule or any(item not in PARADIGMS for item in schedule):
        raise ValueError("generation schedule contains an unknown search paradigm")
    arms = protocol.get("arms", {})
    if set(arms) != set(PARADIGMS):
        raise ValueError("CoE, ToT and EA arms are all required")
    if not 1 <= int(candidate_budget) <= 512:
        raise ValueError("candidate budget must remain bounded")
    for paradigm, arm in arms.items():
        if not 1 <= int(arm.get("frontier_size", 0)) <= 24:
            raise ValueError(f"{paradigm} frontier is unbounded")
        if not 1 <= int(arm.get("children_per_parent", 0)) <= 8:
            raise ValueError(f"{paradigm} branching is unbounded")
        weights = arm.get("operator_weights", {})
        if not weights or set(weights) - OPERATORS:
            raise ValueError(f"{paradigm} contains an unknown operator")
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-12:
            raise ValueError(f"{paradigm} operator weights must sum to one")


def paradigm_for_generation(generation: int, protocol: dict[str, Any]) -> str:
    schedule = list(protocol["generation_schedule"])
    if generation < 1 or generation > len(schedule):
        raise ValueError("generation is outside the preregistered schedule")
    return str(schedule[generation - 1])


def _train_key(
    candidate: dict[str, Any],
    scores: dict[str, tuple[float, float, Any, Any]],
) -> tuple[float, str]:
    return (-float(scores[candidate["factor_id"]][0]), str(candidate["factor_id"]))


def pairwise_train_frontier(
    candidates: list[dict[str, Any]],
    scores: dict[str, tuple[float, float, Any, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return the train-best frontier through deterministic pairwise comparisons."""
    pool = [candidate for candidate in candidates if candidate["factor_id"] in scores]
    selected: list[dict[str, Any]] = []
    comparisons = 0
    while pool and len(selected) < limit:
        champion = 0
        for index in range(1, len(pool)):
            comparisons += 1
            if _train_key(pool[index], scores) < _train_key(pool[champion], scores):
                champion = index
        selected.append(pool.pop(champion))
    return selected, comparisons


def select_frontier(
    candidates: list[dict[str, Any]],
    scores: dict[str, tuple[float, float, Any, Any]],
    paradigm: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    limit = int(protocol["arms"][paradigm]["frontier_size"])
    if paradigm != "coe":
        return pairwise_train_frontier(candidates, scores, limit)
    ordered = sorted(
        (candidate for candidate in candidates if candidate["factor_id"] in scores),
        key=lambda candidate: _train_key(candidate, scores),
    )
    selected: list[dict[str, Any]] = []
    seen_phenomena: set[str] = set()
    for candidate in ordered:
        phenomenon = str(candidate.get("phenomenon", ""))
        if phenomenon in seen_phenomena:
            continue
        seen_phenomena.add(phenomenon)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected, 0


def choose_operator(
    protocol: dict[str, Any],
    paradigm: str,
    generation: int,
    parent_index: int,
    child_index: int,
    random_seed: int,
) -> str:
    """Choose from frozen weights with a stable local seed, never a global RNG."""
    material = f"{random_seed}|{paradigm}|{generation}|{parent_index}|{child_index}"
    seed = int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    weights = protocol["arms"][paradigm]["operator_weights"]
    operators = sorted(weights)
    return str(rng.choices(operators, [float(weights[item]) for item in operators], k=1)[0])


def empty_benchmark() -> dict[str, dict[str, Any]]:
    return {
        paradigm: {
            "generated": 0,
            "unique": 0,
            "fast_screen_passed": 0,
            "comparable_children": 0,
            "parent_improvements": 0,
            "stage2_candidates": 0,
            "validation_survivors": 0,
        }
        for paradigm in ("seed", *PARADIGMS)
    }


def record_candidate(
    benchmark: dict[str, dict[str, Any]],
    paradigm: str,
    *,
    unique: bool,
    fast_screen_passed: bool,
    child_score: float | None = None,
    parent_score: float | None = None,
) -> None:
    row = benchmark.setdefault(paradigm, empty_benchmark()["seed"])
    row["generated"] += 1
    row["unique"] += int(unique)
    row["fast_screen_passed"] += int(fast_screen_passed)
    if child_score is not None and parent_score is not None:
        row["comparable_children"] += 1
        row["parent_improvements"] += int(child_score > parent_score)


def record_stage2(
    benchmark: dict[str, dict[str, Any]], paradigm: str, survived: bool
) -> None:
    row = benchmark.setdefault(paradigm, empty_benchmark()["seed"])
    row["stage2_candidates"] += 1
    row["validation_survivors"] += int(survived)


def finalise_benchmark(
    benchmark: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for paradigm, row in benchmark.items():
        generated = int(row["generated"])
        unique = int(row["unique"])
        comparable = int(row["comparable_children"])
        stage2 = int(row["stage2_candidates"])
        output[paradigm] = {
            **row,
            "duplicate_rate": 1.0 - unique / max(1, generated),
            "fast_screen_pass_rate": row["fast_screen_passed"] / max(1, unique),
            "parent_improvement_rate": row["parent_improvements"] / max(1, comparable),
            "stage2_survival_rate": row["validation_survivors"] / max(1, stage2),
        }
    return output
