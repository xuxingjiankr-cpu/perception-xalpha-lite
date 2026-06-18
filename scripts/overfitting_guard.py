"""Overfitting-defense toolkit (López de Prado): PBO via CSCV + purged/embargoed splits.

Open-ended "improve profitability" searches are overfitting machines: try enough
configs and one will look great on history by luck. This module quantifies that risk
so a "found edge" must clear it before anyone trusts it.

- combinatorial_symmetric_pbo(perf_matrix): Probability of Backtest Overfitting
  (Bailey-Borwein-López de Prado-Zhu 2014). Given a matrix of per-period performance
  for many configs, it splits the timeline every which way into in-sample / out-of-
  sample halves, picks the IS-best config each way, and measures how often that config
  lands BELOW the OOS median. PBO ~0.5 => the search is noise (the winner doesn't
  persist); PBO ~0 => a genuinely dominant config that holds up out-of-sample.

- purged_train_test_splits(n, k, embargo): time-series CV splits that PURGE the test
  fold from train and add an EMBARGO gap, to stop look-ahead leakage in any future
  signal validation (López de Prado, Advances in Financial ML, ch.7).

Pure functions, no I/O, no orders. Intended to be imported by the research scripts and
the evolution pipeline before any candidate is promoted toward live.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Iterator


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def combinatorial_symmetric_pbo(
    perf_matrix: list[list[float]],
    *,
    n_blocks: int = 8,
    metric: Callable[[list[float]], float] | None = None,
) -> dict[str, Any]:
    """perf_matrix[i][t] = performance of config i in period t (rows=configs, cols=periods).

    Returns {"pbo", "n_splits", "n_configs", "n_blocks"}. PBO is the fraction of IS/OOS
    splits where the IS-best config's OOS performance ranks at or below the median.
    """
    metric = metric or _mean
    n_configs = len(perf_matrix)
    if n_configs < 2:
        return {"pbo": None, "reason": "need >= 2 configs", "n_configs": n_configs}
    t_cols = len(perf_matrix[0])
    if any(len(row) != t_cols for row in perf_matrix):
        raise ValueError("all configs must have the same number of periods")
    # even block count, each block non-empty
    s = max(2, min(n_blocks, t_cols))
    if s % 2 == 1:
        s -= 1
    if s < 2 or t_cols < s:
        return {"pbo": None, "reason": "not enough periods for the requested blocks",
                "n_configs": n_configs, "n_periods": t_cols}
    # partition column indices into s contiguous blocks
    bounds = [round(i * t_cols / s) for i in range(s + 1)]
    blocks = [list(range(bounds[i], bounds[i + 1])) for i in range(s)]
    blocks = [b for b in blocks if b]
    s = len(blocks)
    if s < 2 or s % 2 == 1:
        s -= s % 2
        blocks = blocks[:s]
    logits_below = 0
    total = 0
    for is_block_ids in itertools.combinations(range(s), s // 2):
        is_cols: list[int] = []
        for b in is_block_ids:
            is_cols.extend(blocks[b])
        oos_cols = [c for b in range(s) if b not in is_block_ids for c in blocks[b]]
        is_perf = [metric([row[c] for c in is_cols]) for row in perf_matrix]
        oos_perf = [metric([row[c] for c in oos_cols]) for row in perf_matrix]
        best = max(range(n_configs), key=lambda i: is_perf[i])
        # relative rank of the IS-best in the OOS distribution, in (0,1)
        rank = sum(1 for v in oos_perf if v < oos_perf[best])
        w = (rank + 1) / (n_configs + 1)
        logit = math.log(w / (1 - w)) if 0 < w < 1 else (1.0 if w >= 1 else -1.0)
        if logit <= 0:
            logits_below += 1
        total += 1
    return {
        "pbo": round(logits_below / total, 4) if total else None,
        "n_splits": total, "n_configs": n_configs, "n_blocks": s,
        "interpretation": "pbo~0.5 => search is noise (winner does not persist); "
                          "pbo~0 => dominant config holds out-of-sample",
    }


def purged_train_test_splits(n_samples: int, n_splits: int = 5, embargo: int = 0) -> Iterator[tuple[list[int], list[int]]]:
    """Yield (train_idx, test_idx). Each test fold is a contiguous block; the train set
    excludes the test fold AND an `embargo` of samples on each side (purge + embargo) to
    prevent look-ahead leakage across adjacent (autocorrelated) observations."""
    if n_samples <= 0 or n_splits < 2:
        return
    bounds = [round(i * n_samples / n_splits) for i in range(n_splits + 1)]
    for i in range(n_splits):
        test = list(range(bounds[i], bounds[i + 1]))
        if not test:
            continue
        lo = max(0, test[0] - embargo)
        hi = min(n_samples, test[-1] + 1 + embargo)
        train = [j for j in range(n_samples) if j < lo or j >= hi]
        yield train, test


def deflated_significance_note(n_trials: int, observed_sharpe: float, n_obs: int) -> dict[str, Any]:
    """Quick multiple-testing sanity flag: the expected MAXIMUM Sharpe from n_trials of
    pure noise (Bailey-López de Prado). If observed <= expected-max-under-null, the
    result is consistent with luck. Not a full DSR -- the evolution pipeline owns that."""
    if n_trials < 1 or n_obs < 2:
        return {"flag": "insufficient", "expected_max_noise_sharpe": None}
    euler = 0.5772156649
    e_max = (math.sqrt(2 * math.log(n_trials)) if n_trials > 1 else 0.0)
    e_max -= (euler / math.sqrt(2 * math.log(n_trials))) if n_trials > 1 else 0.0
    # convert per-trial annualization-agnostic: treat observed_sharpe on the same scale
    looks_like_luck = observed_sharpe <= e_max
    return {
        "n_trials": n_trials, "observed_sharpe": round(observed_sharpe, 4),
        "expected_max_noise_sharpe": round(e_max, 4),
        "flag": "consistent_with_luck" if looks_like_luck else "exceeds_noise_max",
    }
