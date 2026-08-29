from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from xalpha_lite.discovery import (
    bounded_generate_audited,
    build_panel,
    make_split,
    run_discovery,
)
from xalpha_lite.search_protocol import (
    pairwise_train_frontier,
    select_frontier,
    validate_search_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    sys.path.insert(0, str(ROOT / "examples"))
    from run_synthetic import make_data

    prices, statements = make_data(days=650, symbols=20)
    config = json.loads(
        (ROOT / "configs" / "alphabench_search.json").read_text(encoding="utf-8")
    )
    config.update(
        {
            "minimum_train_days": 300,
            "validation_days": 100,
            "shadow_days": 100,
            "candidate_budget": 24,
            "stage2_budget": 4,
            "placebo_repetitions": 7,
            "pbo_blocks": 4,
        }
    )
    return prices, statements, config


def test_absolute_factor_judge_and_held_out_feedback_fail_closed() -> None:
    _, _, config = _inputs()
    protocol = copy.deepcopy(config["search_protocol"])
    protocol["absolute_factor_judgement_allowed"] = True
    try:
        validate_search_protocol(protocol, config["candidate_budget"])
    except ValueError as error:
        assert "zero-shot" in str(error)
    else:
        raise AssertionError("absolute factor judgement did not fail closed")

    protocol = copy.deepcopy(config["search_protocol"])
    protocol["validation_feedback_allowed"] = True
    try:
        validate_search_protocol(protocol, config["candidate_budget"])
    except ValueError as error:
        assert "quarantined" in str(error)
    else:
        raise AssertionError("validation feedback did not fail closed")


def test_pairwise_and_coe_frontiers_are_deterministic_and_mechanism_aware() -> None:
    _, _, config = _inputs()
    candidates = [
        {"factor_id": "b", "phenomenon": "p1"},
        {"factor_id": "a", "phenomenon": "p1"},
        {"factor_id": "c", "phenomenon": "p2"},
    ]
    scores = {
        "a": (2.0, 1.0, None, None),
        "b": (1.0, 1.0, None, None),
        "c": (0.5, 1.0, None, None),
    }
    first, comparisons = pairwise_train_frontier(candidates, scores, 2)
    second, _ = pairwise_train_frontier(list(reversed(candidates)), scores, 2)
    assert comparisons > 0
    assert [row["factor_id"] for row in first] == ["a", "b"]
    assert [row["factor_id"] for row in second] == ["a", "b"]

    coe, comparisons = select_frontier(
        candidates, scores, "coe", config["search_protocol"]
    )
    assert comparisons == 0
    assert [row["factor_id"] for row in coe] == ["a", "c"]


def test_held_out_labels_cannot_change_search_lineage() -> None:
    prices, statements, config = _inputs()
    panel = build_panel(prices, statements, config)
    split = make_split(panel["close"].index, config)
    horizon = int(config["prediction_horizon_days"])
    target = panel["open"].shift(-(horizon + 1)) / panel["open"].shift(-1) - 1.0
    first, _, first_audit = bounded_generate_audited(panel, target, split, config)
    contaminated = target.copy()
    contaminated.loc[~split.train] = 1_000_000.0
    second, _, second_audit = bounded_generate_audited(
        panel, contaminated, split, config
    )
    assert [row["factor_id"] for row in first] == [row["factor_id"] for row in second]
    assert first_audit["generation_audit"] == second_audit["generation_audit"]
    assert all(
        row["selection_evidence"] == "train_only"
        and row["validation_feedback_used"] is False
        and row["shadow_feedback_used"] is False
        for row in first_audit["generation_audit"]
    )


def test_search_protocol_is_audited_but_never_bypasses_validation() -> None:
    prices, statements, config = _inputs()
    result = run_discovery(prices, statements, config)
    audit = result["search_protocol"]
    assert audit["enabled"] is True
    assert audit["absolute_factor_judgement_allowed"] is False
    assert audit["final_judge"] == "pit_counter_placebo_purged_pbo_dsr"
    assert set(audit["benchmark"]) == {"seed", "coe", "tot", "ea"}
    assert sum(row["generated"] for row in audit["benchmark"].values()) >= 24
    assert result["orders"] == []
    assert result["automatic_trading_changes"] == []
    assert all("guards" in row for row in result["candidates"])
