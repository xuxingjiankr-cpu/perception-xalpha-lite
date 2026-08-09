from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from xalpha_lite.decision import (
    block_subsample_dates,
    chronological_research_partitions,
    fit_logistic_probability_model,
    fit_pairwise_topk_weights,
    fit_pairwise_weight_ensemble,
    fit_ridge_score_model,
    probability_metrics,
    project_bounded_simplex,
)


def _synthetic_rank_book() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(20260809)
    dates = pd.bdate_range("2022-01-03", periods=100)
    symbols = [f"S{index:03d}" for index in range(80)]
    useful = pd.DataFrame(
        rng.uniform(size=(len(dates), len(symbols))), index=dates, columns=symbols
    )
    factors = {"feature_01": useful}
    for index in range(2, 5):
        factors[f"feature_{index:02d}"] = pd.DataFrame(
            rng.uniform(size=useful.shape), index=dates, columns=symbols
        )
    outcomes = useful * 0.04 + pd.DataFrame(
        rng.normal(scale=0.002, size=useful.shape), index=dates, columns=symbols
    )
    return factors, outcomes


def test_bounded_simplex_projection_is_feasible() -> None:
    projected = project_bounded_simplex([2.0, -1.0, 0.2, 0.3], 0.05, 0.60)
    assert np.isclose(projected.sum(), 1.0)
    assert (projected >= 0.05 - 1e-12).all()
    assert (projected <= 0.60 + 1e-12).all()


def test_pairwise_fit_is_training_only_and_rewards_planted_information() -> None:
    factors, outcomes = _synthetic_rank_book()
    fit_dates = outcomes.index[:80]
    options = {
        "minimum_weight": 0.02,
        "maximum_weight": 0.60,
        "l2_to_prior": 0.01,
        "top_count": 5,
        "boundary_count": 10,
        "minimum_cross_section": 40,
    }
    fitted = fit_pairwise_topk_weights(factors, outcomes, fit_dates, **options)
    changed = outcomes.copy()
    changed.loc[outcomes.index[80:]] = 1_000.0
    unchanged = fit_pairwise_topk_weights(factors, changed, fit_dates, **options)
    np.testing.assert_allclose(fitted["weights"], unchanged["weights"], atol=1e-12)
    assert fitted["weight_map"]["feature_01"] > max(fitted["weights"][1:])
    assert fitted["audit"]["used_day_count"] == len(fit_dates)
    assert fitted["audit"]["label_table_only"] is True


def test_block_ensemble_is_deterministic_and_never_emits_orders() -> None:
    factors, outcomes = _synthetic_rank_book()
    fit_dates = outcomes.index[:80]
    first = block_subsample_dates(
        fit_dates, block_length=10, fraction=0.75, seed=9, replica=0
    )
    second = block_subsample_dates(
        fit_dates, block_length=10, fraction=0.75, seed=9, replica=0
    )
    assert first.equals(second)
    ensemble = fit_pairwise_weight_ensemble(
        factors,
        outcomes,
        fit_dates,
        replicas=3,
        block_length=10,
        subsample_fraction=0.75,
        random_seed=9,
        minimum_weight=0.02,
        maximum_weight=0.60,
        l2_to_prior=0.01,
        top_count=5,
        boundary_count=10,
        minimum_cross_section=40,
    )
    assert ensemble["replica_weights"].shape == (3, 4)
    assert ensemble["orders"] == []
    assert ensemble["automatic_trading_changes"] == []


def test_five_research_blocks_have_full_purges() -> None:
    dates = pd.bdate_range("2020-01-01", periods=800)
    result = chronological_research_partitions(
        dates,
        calibration_days=50,
        audit_days=50,
        validation_days=50,
        shadow_days=50,
        purge_days=10,
        minimum_fit_days=500,
    )
    names = ("fit", "calibration", "audit", "validation", "shadow")
    assert all(set(result[left]).isdisjoint(result[right]) for left in names for right in names if left != right)
    for left, right in zip(names, names[1:]):
        gap = dates.get_loc(result[right][0]) - dates.get_loc(result[left][-1]) - 1
        assert gap >= 10
    assert result["partition_audit"]["shadow_is_fit_input"] is False


def test_independent_calibrators_and_probability_diagnostics() -> None:
    rng = np.random.default_rng(77)
    features = rng.normal(size=(1_200, 3))
    latent = 1.4 * features[:, 0] - 0.7 * features[:, 1]
    events = (latent + rng.normal(scale=0.7, size=len(latent)) > 0).astype(int)
    model = fit_logistic_probability_model(
        features[:800], events[:800], feature_names=["a", "b", "c"], l2=0.01
    )
    probabilities = model.predict_proba(features[800:])
    metrics = probability_metrics(events[800:], probabilities)
    assert metrics["auc"] is not None and metrics["auc"] > 0.80
    assert 0 <= metrics["ece"] <= 1
    score = features[:800, 0]
    outcomes = 0.02 * score + rng.normal(scale=0.002, size=len(score))
    ridge = fit_ridge_score_model(score, outcomes, l2=0.001)
    assert ridge.slope > 0


def test_decision_module_has_no_execution_path() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src/xalpha_lite/decision.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "submit_order",
        "broker_client",
        "build_decision(",
        "strategy_overlay",
        "position_sizing",
    )
    assert all(token not in source for token in forbidden)


def test_public_research_content_is_english_only() -> None:
    root = Path(__file__).resolve().parents[1]
    cjk = re.compile(r"[\u3400-\u9fff]")
    included = {".md", ".html", ".py", ".json", ".toml"}
    violations = []
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix not in included
            or any(part.startswith(".") for part in path.relative_to(root).parts)
            or "outputs" in path.parts
            or "data" in path.parts
        ):
            continue
        if cjk.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(root)))
    assert violations == []
