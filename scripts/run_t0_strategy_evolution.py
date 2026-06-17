"""Paper-only T+0 strategy parameter evolution.

This script runs offline replays over historical quote snapshots and writes a
bounded strategy overlay for the intraday ETF paper agent. It never calls the
paper-trading API and never changes execution locks. The live/paper agent will
only apply an overlay whose status is approved_for_paper_auto_apply and whose
strategy paths are allowlisted in configs/t0_intraday_paper_agent.json.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float, load_json, write_json


DEFAULT_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "t0_strategy_evolution"
DEFAULT_REPLAY_OUT = ROOT / "outputs" / "t0_replay"
EVOLUTION_GATE_VERSION = "dm_hln_mcs_spa_v2"


PARAM_SPACE = [
    {"path": "entry_momentum_pct", "type": "float", "lo": 0.0012, "hi": 0.0028},
    {"path": "exit_momentum_pct", "type": "float", "lo": -0.0020, "hi": -0.0005},
    {"path": "entry_score_threshold", "type": "int", "lo": 48, "hi": 72},
    {"path": "loss_exit_score_threshold", "type": "int", "lo": 65, "hi": 90},
    {"path": "profit_exit_score_threshold", "type": "int", "lo": 58, "hi": 84},
    {"path": "min_profit_exit_pct", "type": "float", "lo": 0.002, "hi": 0.006},
    {"path": "min_hold_minutes", "type": "int", "lo": 8, "hi": 35},
    {"path": "loss_review_after_minutes", "type": "int", "lo": 10, "hi": 40},
    {"path": "profit_trailing_drawdown_pct", "type": "float", "lo": -0.012, "hi": -0.004},
    {"path": "deceleration_exit_threshold", "type": "float", "lo": -0.004, "hi": -0.001},
    {"path": "cross_etf_divergence_threshold", "type": "float", "lo": 0.0015, "hi": 0.0035},
    {"path": "consolidation.max_range_pct", "type": "float", "lo": 0.0020, "hi": 0.0040},
    {"path": "consolidation.breakout_buffer_pct", "type": "float", "lo": 0.0005, "hi": 0.0020},
    {"path": "indicators.rolling_vwap.require_price_above_for_entry", "type": "bool", "lo": 0.0, "hi": 1.0},
    {"path": "indicators.intraday_atr.stop_multiplier", "type": "float", "lo": 1.0, "hi": 1.8},
    {"path": "indicators.intraday_atr.min_stop_pct", "type": "float", "lo": 0.0015, "hi": 0.0035},
    {"path": "indicators.bollinger_squeeze.squeeze_bandwidth_pct", "type": "float", "lo": 0.0040, "hi": 0.0080},
    {"path": "indicators.bollinger_squeeze.breakout_buffer_pct", "type": "float", "lo": 0.0003, "hi": 0.0012},
    {"path": "market_correlation_stress.avg_abs_corr_threshold", "type": "float", "lo": 0.60, "hi": 0.85},
    {"path": "bracket.risk_per_trade_pct", "type": "float", "lo": 0.0020, "hi": 0.0045},
    {"path": "bracket.risk_per_trade_pct_chaos_day", "type": "float", "lo": 0.0010, "hi": 0.0022},
    {"path": "bracket.target1_r_multiple", "type": "float", "lo": 0.8, "hi": 1.4},
    {"path": "bracket.target2_r_multiple", "type": "float", "lo": 1.6, "hi": 2.6},
    {"path": "bracket.reentry_cooldown_minutes", "type": "int", "lo": 15, "hi": 60},
]


# One-sided upper-tail Student-t critical values (P(T<=t)=1-alpha), df 1..30.
_T_CRIT_ONE_SIDED = {
    0.05: {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
           8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761,
           15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725, 21: 1.721,
           22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703, 28: 1.701,
           29: 1.699, 30: 1.697},
    0.10: {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415,
           8: 1.397, 9: 1.383, 10: 1.372, 11: 1.363, 12: 1.356, 13: 1.350, 14: 1.345,
           15: 1.341, 16: 1.337, 17: 1.333, 18: 1.330, 19: 1.328, 20: 1.325, 21: 1.323,
           22: 1.321, 23: 1.319, 24: 1.318, 25: 1.316, 26: 1.315, 27: 1.314, 28: 1.313,
           29: 1.311, 30: 1.310},
}


def _t_critical(df: int, alpha: float) -> float:
    table = _T_CRIT_ONE_SIDED.get(alpha, _T_CRIT_ONE_SIDED[0.05])
    if df <= 0:
        return float("inf")
    if df in table:
        return table[df]
    return 1.645 if alpha == 0.05 else 1.282  # large-df normal approximation


def diebold_mariano_hln(
    base_days: dict[str, Any],
    sel_days: dict[str, Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample
    correction, h=1) on the per-day PnL differential d_t = pnl_selected - pnl_baseline.

    H0: candidate not better (E[d] <= 0); reject (candidate significantly better)
    when DM* exceeds the upper-tail Student-t critical value with n-1 df.
    Heavy tails / autocorrelation distort DM (arXiv:2605.16866, 2409.12662), so
    this is used as a conservative ADDITIONAL gate, never the sole criterion.
    """
    shared = sorted(set(base_days) & set(sel_days))
    d = [as_float(sel_days[k]) - as_float(base_days[k]) for k in shared]
    n = len(d)
    result = {
        "n_days": n, "alpha": alpha, "dm_star": None, "t_critical": None,
        "significant": False, "mean_diff": None,
        "reason": "insufficient_days" if n < 2 else "ok",
    }
    if n < 2:
        return result
    dbar = sum(d) / n
    gamma0 = sum((x - dbar) ** 2 for x in d) / n  # autocovariance estimator at lag 0
    result["mean_diff"] = round(dbar, 4)
    if gamma0 <= 0:
        result["reason"] = "zero_variance_differential"
        return result
    dm = dbar / math.sqrt(gamma0 / n)
    hln_factor = math.sqrt((n - 1) / n)  # HLN correction for h=1
    dm_star = dm * hln_factor
    df = n - 1
    crit = _t_critical(df, alpha)
    result.update({
        "dm_star": round(dm_star, 4),
        "t_critical": crit,
        "df": df,
        "significant": dm_star >= crit,
    })
    return result


def _stationary_bootstrap_indices(n: int, avg_block: float, rng: random.Random) -> list[int]:
    """One stationary-bootstrap (Politis-Romano) resample of time indices 0..n-1."""
    if n <= 0:
        return []
    p = 1.0 / max(1.0, avg_block)
    idx = [rng.randrange(n)]
    for _ in range(n - 1):
        if rng.random() < p:
            idx.append(rng.randrange(n))
        else:
            idx.append((idx[-1] + 1) % n)
    return idx


def model_confidence_set(
    loss_by_model: dict[str, list[float]],
    *,
    alpha: float = 0.10,
    n_boot: int = 1000,
    avg_block: float = 3.0,
    seed: int = 20260615,
) -> dict[str, Any]:
    """Hansen-Lunde-Nason (2011) Model Confidence Set via stationary bootstrap,
    range statistic T_R. Lower loss = better. Returns the surviving set, each
    model's MCS p-value, and elimination order.

    Tiny samples have almost no power: the MCS keeps (almost) all models, which
    correctly means "cannot single out a best model yet". Used as a conservative
    gate (auto-apply only if baseline is EXCLUDED from the MCS).
    """
    names = [k for k, v in loss_by_model.items() if isinstance(v, list) and len(v) >= 2]
    result: dict[str, Any] = {"mcs_set": list(names), "p_values": {}, "eliminated_order": [],
                              "alpha": alpha, "n_obs": 0, "reason": "ok"}
    if len(names) < 2:
        result["reason"] = "fewer_than_two_models"
        return result
    n = min(len(loss_by_model[k]) for k in names)
    if n < 2:
        result["reason"] = "insufficient_obs"
        return result
    result["n_obs"] = n
    L = {k: loss_by_model[k][:n] for k in names}

    rng = random.Random(seed)
    boot_idx = [_stationary_bootstrap_indices(n, avg_block, rng) for _ in range(n_boot)]

    def col_mean(vals: list[float], idx: list[int]) -> float:
        return sum(vals[i] for i in idx) / len(idx)

    alive = list(names)
    p_running = 0.0
    while len(alive) > 1:
        means = {k: sum(L[k]) / n for k in alive}
        # Per-pair dbar_ij, bootstrap-mean series, and bootstrap sd of dbar_ij.
        pairs = [(i, j) for ii, i in enumerate(alive) for j in alive[ii + 1:]]
        dbar_p: dict[tuple, float] = {}
        sd_p: dict[tuple, float] = {}
        bootm_p: dict[tuple, list[float]] = {}
        for (i, j) in pairs:
            dbar = means[i] - means[j]
            bms = [col_mean(L[i], bi) - col_mean(L[j], bi) for bi in boot_idx]
            mb = sum(bms) / n_boot
            var = sum((x - mb) ** 2 for x in bms) / n_boot
            dbar_p[(i, j)] = dbar
            sd_p[(i, j)] = math.sqrt(var) if var > 1e-18 else 0.0
            bootm_p[(i, j)] = bms
        # Studentized range statistic T_R = max |dbar_ij / sd_ij|
        def tstat(pair: tuple) -> float:
            sd = sd_p[pair]
            return abs(dbar_p[pair] / sd) if sd > 0 else 0.0
        t_range = max((tstat(p) for p in pairs), default=0.0)
        # Bootstrap null: T_R* = max |(dbar*_ij - dbar_ij)/sd_ij|, p = P(T_R* >= T_R)
        ge = 0
        for b in range(n_boot):
            tmax_b = 0.0
            for pair in pairs:
                sd = sd_p[pair]
                if sd <= 0:
                    continue
                tb = abs((bootm_p[pair][b] - dbar_p[pair]) / sd)
                if tb > tmax_b:
                    tmax_b = tb
            if tmax_b >= t_range:
                ge += 1
        p_val = ge / n_boot if n_boot else 1.0
        p_running = max(p_running, p_val)
        if p_val > alpha:
            break  # cannot reject equal predictive ability -> remaining set is the MCS
        # eliminate worst: highest average relative (signed) loss t-stat
        tbar_i: dict[str, float] = {}
        for i in alive:
            num = 0.0
            cnt = 0
            for j in alive:
                if i == j:
                    continue
                pair = (i, j) if (i, j) in dbar_p else (j, i)
                sign = 1.0 if (i, j) in dbar_p else -1.0
                sd = sd_p[pair]
                num += (sign * dbar_p[pair] / sd) if sd > 0 else 0.0
                cnt += 1
            tbar_i[i] = num / cnt if cnt else 0.0
        worst = max(tbar_i, key=lambda k: tbar_i[k])
        result["p_values"][worst] = round(p_running, 4)
        result["eliminated_order"].append(worst)
        alive.remove(worst)
    for k in alive:
        result["p_values"].setdefault(k, 1.0)
    result["mcs_set"] = list(alive)
    return result


def _legacy_reality_check_spa_v1(
    benchmark_losses: list[float],
    alt_losses: dict[str, list[float]],
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    avg_block: float = 3.0,
    seed: int = 20260616,
) -> dict[str, Any]:
    """Studentized White Reality Check / Hansen SPA (conservative, no recentering).

    Tests H0: no alternative is superior to the benchmark (baseline), correcting
    for data-snooping across all alternatives. d_{k,t} = L_bench,t - L_alt_k,t
    (positive mean -> alternative better). Rejects (a snooping-robust superior
    model exists) when the bootstrap p-value < alpha. Conservative variant is
    chosen deliberately: it errs toward NOT auto-applying.
    """
    names = [k for k, v in alt_losses.items() if isinstance(v, list) and len(v) >= 2]
    result: dict[str, Any] = {"p_value": 1.0, "t_max": None, "n_obs": 0,
                              "reject": False, "best_alt": None, "alpha": alpha, "reason": "ok"}
    if not names or len(benchmark_losses) < 2:
        result["reason"] = "insufficient_models_or_obs"
        return result
    n = min([len(benchmark_losses)] + [len(alt_losses[k]) for k in names])
    if n < 2:
        result["reason"] = "insufficient_obs"
        return result
    result["n_obs"] = n
    bench = benchmark_losses[:n]
    rng = random.Random(seed)
    boot_idx = [_stationary_bootstrap_indices(n, avg_block, rng) for _ in range(n_boot)]

    def bmean(vals: list[float], idx: list[int]) -> float:
        return sum(vals[i] for i in idx) / len(idx)

    # Non-studentized White Reality Check statistic V = max(0, max_k dbar_k).
    # Avoiding division by a bootstrap sd makes it robust to near-degenerate
    # variance (candidates almost identical to baseline), which a studentized
    # form false-rejects on — the safe choice for an auto-apply gate.
    dbar: dict[str, float] = {}
    bdiff: dict[str, list[float]] = {}
    for k in names:
        d = [bench[t] - alt_losses[k][t] for t in range(n)]
        dbar[k] = sum(d) / n
        alt_k = alt_losses[k][:n]
        bdiff[k] = [bmean(bench, bi) - bmean(alt_k, bi) for bi in boot_idx]

    v_obs = max([0.0] + [dbar[k] for k in names])
    best_alt = max(names, key=lambda k: dbar[k]) if names else None
    ge = 0
    for b in range(n_boot):
        vmax = 0.0
        for k in names:
            vb = bdiff[k][b] - dbar[k]  # recenter by empirical mean (White RC)
            if vb > vmax:
                vmax = vb
        if vmax >= v_obs:
            ge += 1
    p = ge / n_boot if n_boot else 1.0
    result.update({
        "p_value": round(p, 4),
        "t_max": round(v_obs, 6),
        "best_alt": best_alt,
        "reject": p < alpha,
    })
    return result


def reality_check_spa(
    benchmark_losses: list[float],
    alt_losses: dict[str, list[float]],
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    avg_block: float = 3.0,
    seed: int = 20260616,
) -> dict[str, Any]:
    """Studentized Hansen SPA-lite reality check.

    Replacement for the legacy conservative White-RC implementation above.
    Positive d = benchmark loss - candidate loss means the candidate is better.
    """
    names = [k for k, v in alt_losses.items() if isinstance(v, list) and len(v) >= 2]
    result: dict[str, Any] = {
        "method": "studentized_hansen_spa_lite",
        "p_value": 1.0,
        "t_max": None,
        "n_obs": 0,
        "reject": False,
        "best_alt": None,
        "alpha": alpha,
        "reason": "ok",
        "active_alternatives": [],
        "candidate_stats": {},
    }
    if not names or len(benchmark_losses) < 2:
        result["reason"] = "insufficient_models_or_obs"
        return result
    n = min([len(benchmark_losses)] + [len(alt_losses[k]) for k in names])
    if n < 2:
        result["reason"] = "insufficient_obs"
        return result
    result["n_obs"] = n
    bench = benchmark_losses[:n]
    rng = random.Random(seed)
    boot_idx = [_stationary_bootstrap_indices(n, avg_block, rng) for _ in range(n_boot)]

    diffs: dict[str, list[float]] = {}
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    stats: dict[str, float] = {}
    for k in names:
        d = [bench[t] - alt_losses[k][t] for t in range(n)]
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / max(1, n - 1)
        sd = math.sqrt(var)
        diffs[k] = d
        means[k] = mean
        sds[k] = sd
        stats[k] = (float("inf") if mean > 0 else 0.0) if sd <= 1e-12 else math.sqrt(n) * mean / sd

    active: list[str] = []
    for k in names:
        sd = sds[k]
        trim_threshold = -sd / math.sqrt(n) if sd > 1e-12 else 0.0
        if means[k] >= trim_threshold:
            active.append(k)
    if not active:
        active = names[:]

    v_obs = max([0.0] + [stats[k] for k in active])
    best_alt = max(active, key=lambda k: means[k]) if active else None
    ge = 0
    for b in range(n_boot):
        vmax = 0.0
        for k in active:
            sd = sds[k]
            if sd <= 1e-12:
                vb = 0.0
            else:
                sample_mean = sum(diffs[k][i] for i in boot_idx[b]) / n
                vb = math.sqrt(n) * (sample_mean - means[k]) / sd
            if vb > vmax:
                vmax = vb
        if vmax >= v_obs:
            ge += 1
    p = ge / n_boot if n_boot else 1.0
    result.update({
        "p_value": round(p, 4),
        "t_max": "inf" if math.isinf(v_obs) else round(v_obs, 6),
        "best_alt": best_alt,
        "reject": p < alpha,
        "active_alternatives": active,
        "candidate_stats": {
            k: {
                "mean_diff": round(means[k], 6),
                "sd_diff": round(sds[k], 6),
                "t_stat": "inf" if math.isinf(stats[k]) else round(stats[k], 6),
            } for k in names
        },
    })
    return result


def deflated_sharpe_diagnostic(
    baseline_days: dict[str, Any],
    selected_days: dict[str, Any],
    *,
    n_trials: int,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Approximate deflated-Sharpe diagnostic on daily PnL differentials."""
    shared = sorted(set(baseline_days or {}) & set(selected_days or {}))
    result: dict[str, Any] = {
        "method": "approx_deflated_sharpe_on_daily_pnl_diff",
        "significant": False,
        "alpha": alpha,
        "n_days": len(shared),
        "n_trials": max(1, int(n_trials)),
        "reason": "ok",
    }
    if len(shared) < 8:
        result["reason"] = "insufficient_shared_days"
        return result
    diffs = [as_float(selected_days[d]) - as_float(baseline_days[d]) for d in shared]
    mean = sum(diffs) / len(diffs)
    var = sum((x - mean) ** 2 for x in diffs) / max(1, len(diffs) - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        result.update({
            "mean_diff": round(mean, 4),
            "std_diff": round(sd, 4),
            "p_value": 0.0 if mean > 0 else 1.0,
            "significant": mean > 0,
            "reason": "zero_variance",
        })
        return result
    sr = mean / sd
    t_stat = sr * math.sqrt(len(diffs))
    trials = max(2, int(n_trials))
    selection_bias_t = NormalDist().inv_cdf(1.0 - 1.0 / trials)
    deflated_t = t_stat - selection_bias_t
    p_value = 1.0 - NormalDist().cdf(deflated_t)
    result.update({
        "mean_diff": round(mean, 4),
        "std_diff": round(sd, 4),
        "sharpe_diff_daily": round(sr, 4),
        "t_stat": round(t_stat, 4),
        "selection_bias_t": round(selection_bias_t, 4),
        "deflated_t": round(deflated_t, 4),
        "p_value": round(p_value, 4),
        "significant": bool(mean > 0 and p_value < alpha),
    })
    return result


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)
    return dst


def flatten_overlay(obj: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            out.extend(flatten_overlay(val, path))
        else:
            out.append((path, val))
    return out


def get_nested(obj: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_nested(obj: dict[str, Any], dotted_path: str, value: Any) -> None:
    cur = obj
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


def normalize_param(value: Any, spec: dict[str, Any]) -> float:
    typ = spec["type"]
    if typ == "bool":
        return 1.0 if bool(value) else 0.0
    lo = as_float(spec["lo"])
    hi = as_float(spec["hi"])
    val = as_float(value, (lo + hi) / 2.0)
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def denormalize_param(x: float, spec: dict[str, Any]) -> Any:
    x = max(0.0, min(1.0, x))
    typ = spec["type"]
    if typ == "bool":
        return bool(x >= 0.5)
    lo = as_float(spec["lo"])
    hi = as_float(spec["hi"])
    val = lo + x * (hi - lo)
    if typ == "int":
        return int(round(val))
    return round(val, 6)


def overlay_from_vector(vec: list[float]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    for x, spec in zip(vec, PARAM_SPACE):
        set_nested(overlay, spec["path"], denormalize_param(x, spec))
    return overlay


def vector_from_strategy(strategy: dict[str, Any]) -> list[float]:
    vec: list[float] = []
    for spec in PARAM_SPACE:
        default_val = denormalize_param(0.5, spec)
        vec.append(normalize_param(get_nested(strategy, spec["path"], default_val), spec))
    return vec


def candidate_overlays() -> list[dict[str, Any]]:
    """Predefined candidate overlays. No search over live outcomes.

    The cs_ne_* candidates are a fixed-budget, deterministic trial inspired by
    recent cs.NE themes:
    - mixed categorical/continuous black-box optimization: combine discrete
      toggles and continuous thresholds;
    - dynamic-environment local EA: mutate only near the current policy;
    - multi-objective evolutionary selection: keep separate precision, risk,
      and diversity candidates instead of one opaque optimizer;
    - CMA-ES stopping-criteria caution: small fixed budget, no repeated search
      until more paper data arrives.
    """
    base = [
        {
            "name": "baseline_current",
            "description": "Current locked config.",
            "overlay": {},
        },
        {
            "name": "precision_entry_gate",
            "description": "Fewer entries: require stronger momentum and higher entry score.",
            "overlay": {
                "entry_momentum_pct": 0.0020,
                "entry_score_threshold": 60,
                "cross_etf_divergence_threshold": 0.0025,
                "consolidation": {"breakout_buffer_pct": 0.0015},
                "bracket": {"risk_per_trade_pct": 0.0035, "risk_per_trade_pct_chaos_day": 0.0018},
            },
        },
        {
            "name": "patient_exit",
            "description": "Avoid immediate loss selling unless sell_score confirmation is stronger.",
            "overlay": {
                "min_hold_minutes": 20,
                "loss_review_after_minutes": 25,
                "loss_exit_score_threshold": 82,
                "profit_exit_score_threshold": 75,
                "profit_trailing_drawdown_pct": -0.007,
                "deceleration_exit_threshold": -0.003,
            },
        },
        {
            "name": "balanced_precision",
            "description": "Moderately stricter entries plus more patient exits.",
            "overlay": {
                "entry_momentum_pct": 0.0018,
                "entry_score_threshold": 58,
                "min_hold_minutes": 15,
                "loss_review_after_minutes": 20,
                "loss_exit_score_threshold": 78,
                "profit_exit_score_threshold": 72,
                "bracket": {"risk_per_trade_pct": 0.0035, "risk_per_trade_pct_chaos_day": 0.0018},
            },
        },
        {
            "name": "tight_risk_cut",
            "description": "Lower risk budget while keeping signal logic mostly unchanged.",
            "overlay": {
                "entry_score_threshold": 55,
                "bracket": {"risk_per_trade_pct": 0.0030, "risk_per_trade_pct_chaos_day": 0.0015},
                "indicators": {"intraday_atr": {"stop_multiplier": 1.4}},
            },
        },
        {
            "name": "trend_quality",
            "description": "Require better quality breakouts and stricter correlation stress.",
            "overlay": {
                "entry_score_threshold": 60,
                "consolidation": {"max_range_pct": 0.0025, "breakout_buffer_pct": 0.0015},
                "indicators": {
                    "bollinger_squeeze": {"squeeze_bandwidth_pct": 0.005, "breakout_buffer_pct": 0.0008}
                },
                "market_correlation_stress": {"avg_abs_corr_threshold": 0.68},
            },
        },
    ]
    base.extend([
        {
            "name": "cs_ne_local_mutation_entry_plus",
            "description": "Dynamic-EA style small local mutation: slightly stricter entry, same exit.",
            "overlay": {
                "entry_momentum_pct": 0.0017,
                "entry_score_threshold": 56,
                "consolidation": {"breakout_buffer_pct": 0.0012},
                "indicators": {"bollinger_squeeze": {"breakout_buffer_pct": 0.0007}},
            },
        },
        {
            "name": "cs_ne_local_mutation_exit_plus",
            "description": "Dynamic-EA style small local mutation: more confirmation before selling losers.",
            "overlay": {
                "min_hold_minutes": 15,
                "loss_review_after_minutes": 20,
                "loss_exit_score_threshold": 76,
                "deceleration_exit_threshold": -0.0025,
            },
        },
        {
            "name": "cs_ne_mixed_categorical_vwap_relaxed",
            "description": "Mixed categorical/continuous trial: relax VWAP hard gate, compensate with stronger score gate.",
            "overlay": {
                "entry_score_threshold": 64,
                "entry_momentum_pct": 0.0022,
                "indicators": {"rolling_vwap": {"require_price_above_for_entry": False}},
                "market_correlation_stress": {"avg_abs_corr_threshold": 0.68},
            },
        },
        {
            "name": "cs_ne_quality_diversity_gold_hk",
            "description": "Quality-diversity proxy: broader breakout criteria but lower risk budget.",
            "overlay": {
                "entry_score_threshold": 54,
                "cross_etf_divergence_threshold": 0.0018,
                "consolidation": {"max_range_pct": 0.0028},
                "bracket": {"risk_per_trade_pct": 0.0028, "risk_per_trade_pct_chaos_day": 0.0014},
            },
        },
        {
            "name": "cs_ne_risk_first_low_budget",
            "description": "Risk-first candidate: preserve signals but reduce position risk and widen ATR stop sanity.",
            "overlay": {
                "entry_score_threshold": 55,
                "bracket": {"risk_per_trade_pct": 0.0025, "risk_per_trade_pct_chaos_day": 0.0012},
                "indicators": {"intraday_atr": {"stop_multiplier": 1.5, "min_stop_pct": 0.0025}},
            },
        },
        {
            "name": "cs_ne_patient_profit_capture",
            "description": "Profit-capture mutation: avoid early profit exits unless drawdown confirmation is stronger.",
            "overlay": {
                "profit_exit_score_threshold": 78,
                "min_profit_exit_pct": 0.004,
                "profit_trailing_drawdown_pct": -0.008,
                "bracket": {"target1_r_multiple": 1.2, "target2_r_multiple": 2.2},
            },
        },
    ])
    return base


def validate_allowed_paths(base_cfg: dict[str, Any], overlay: dict[str, Any]) -> list[str]:
    allowed = set(base_cfg.get("self_iteration", {}).get("allowed_strategy_paths", []))
    if not allowed:
        return []
    return [path for path, _ in flatten_overlay(overlay) if path not in allowed]


def write_candidate_config(base_cfg: dict[str, Any], overlay: dict[str, Any], path: Path) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["self_iteration"] = copy.deepcopy(cfg.get("self_iteration", {}))
    cfg["self_iteration"]["auto_apply_changes"] = False
    deep_merge(cfg["strategy"], overlay)
    write_json(path, cfg)


def _date_allowed(trade_date: str, date_filter: str | None, start_date: str | None, end_date: str | None) -> bool:
    if date_filter and trade_date != date_filter:
        return False
    if start_date and trade_date < start_date:
        return False
    if end_date and trade_date > end_date:
        return False
    return True


def prepare_daily_quote_cache(
    quotes_path: str | None,
    *,
    date_filter: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Split a large replay jsonl into per-day files and return the cache dir.

    Replaying all ETF quotes repeatedly is dominated by scanning the same large
    monthly jsonl for every candidate. This one-time split keeps candidate
    evaluation deterministic while making each replay read only the needed dates.
    """
    if not quotes_path:
        return {"quotes_arg": None, "cache_used": False, "reason": "no_quotes_path"}
    source = Path(quotes_path)
    if not source.exists():
        return {"quotes_arg": str(source), "cache_used": False, "reason": "source_missing"}
    if source.is_dir():
        return {"quotes_arg": str(source), "cache_used": True, "cache_dir": str(source), "reason": "source_is_directory"}

    cache_base = cache_root or (DEFAULT_REPLAY_OUT / "quote_cache")
    cache_dir = cache_base / f"{source.stem}_by_date"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    source_sig = {
        "source": str(source.resolve()),
        "size": source.stat().st_size,
        "mtime": source.stat().st_mtime,
        "date_filter": date_filter,
        "start_date": start_date,
        "end_date": end_date,
    }
    existing = load_json(manifest_path) if manifest_path.exists() else {}
    existing_dates = existing.get("dates", []) if isinstance(existing.get("dates"), list) else []
    if existing.get("source_signature") == source_sig and all((cache_dir / f"{d}.jsonl").exists() for d in existing_dates):
        return {
            "quotes_arg": str(cache_dir),
            "cache_used": True,
            "cache_dir": str(cache_dir),
            "dates": existing_dates,
            "reason": "cache_hit",
        }

    for stale_file in cache_dir.glob("*.jsonl"):
        stale_file.unlink()
    handles: dict[str, Any] = {}
    counts: dict[str, int] = {}
    try:
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict) or not obj.get("timestamp"):
                    continue
                trade_date = str(obj["timestamp"])[:10]
                if not _date_allowed(trade_date, date_filter, start_date, end_date):
                    continue
                if trade_date not in handles:
                    out_path = cache_dir / f"{trade_date}.jsonl"
                    handles[trade_date] = out_path.open("w", encoding="utf-8")
                    counts[trade_date] = 0
                handles[trade_date].write(line if line.endswith("\n") else line + "\n")
                counts[trade_date] += 1
    finally:
        for handle in handles.values():
            handle.close()

    dates = sorted(counts)
    write_json(manifest_path, {
        "source_signature": source_sig,
        "created_at": datetime.now().astimezone().isoformat(),
        "dates": dates,
        "rows_by_date": counts,
    })
    return {
        "quotes_arg": str(cache_dir),
        "cache_used": True,
        "cache_dir": str(cache_dir),
        "dates": dates,
        "rows_by_date": counts,
        "reason": "cache_created",
    }


def _session_fraction_from_timestamp(ts_text: str) -> float:
    try:
        hour = int(ts_text[11:13])
        minute = int(ts_text[14:16])
    except (TypeError, ValueError):
        return 1.0
    now_min = hour * 60 + minute
    morning_start = 9 * 60 + 30
    morning_end = 11 * 60 + 30
    afternoon_start = 13 * 60
    afternoon_end = 15 * 60
    if now_min <= morning_start:
        elapsed = 0
    elif now_min <= morning_end:
        elapsed = now_min - morning_start
    elif now_min <= afternoon_start:
        elapsed = 120
    elif now_min <= afternoon_end:
        elapsed = 120 + now_min - afternoon_start
    else:
        elapsed = 240
    return max(0.05, min(1.0, elapsed / 240.0))


def _passes_dynamic_replay_gate(row: dict[str, Any], dyn_cfg: dict[str, Any]) -> bool:
    name = str(row.get("name") or "")
    for keyword in dyn_cfg.get("name_exclude_keywords", []):
        if keyword and str(keyword) in name:
            return False
    price = as_float(row.get("currentPrice"), 0.0)
    bid = as_float(row.get("bidPrice1"), 0.0)
    ask = as_float(row.get("askPrice1"), 0.0)
    if price <= as_float(dyn_cfg.get("min_price"), 0.3) or bid <= 0 or ask <= bid:
        return False
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid if mid > 0 else float("inf")
    if spread > as_float(dyn_cfg.get("max_spread_pct"), 0.004):
        return False
    min_amount = as_float(dyn_cfg.get("min_amount_yuan"), 50_000_000.0)
    fraction = _session_fraction_from_timestamp(str(row.get("timestamp") or ""))
    return as_float(row.get("amount"), 0.0) >= min_amount * fraction


def prepare_dynamic_gate_cache(
    daily_quotes_dir: str | None,
    dyn_cfg: dict[str, Any],
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Keep the full intraday history for codes that pass the live universe gates.

    Eligibility is evaluated at every snapshot, but once a code is eligible on a
    date all of its rows for that date are retained. That preserves exit safety
    and indicator warm-up while avoiding deep calculations on permanently
    illiquid ETFs.
    """
    if not daily_quotes_dir:
        return {"quotes_arg": daily_quotes_dir, "gate_cache_used": False, "reason": "no_daily_cache"}
    source_dir = Path(daily_quotes_dir)
    if not source_dir.is_dir():
        return {"quotes_arg": daily_quotes_dir, "gate_cache_used": False, "reason": "source_not_directory"}
    cache_base = cache_root or (DEFAULT_REPLAY_OUT / "quote_cache")
    cache_dir = cache_base / f"{source_dir.name}_dynamic_gate"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    source_manifest = load_json(source_dir / "manifest.json") if (source_dir / "manifest.json").exists() else {}
    gate_signature = {
        "source_dir": str(source_dir.resolve()),
        "source_manifest": source_manifest,
        "dynamic_gate": {
            "min_amount_yuan": dyn_cfg.get("min_amount_yuan"),
            "max_spread_pct": dyn_cfg.get("max_spread_pct"),
            "min_price": dyn_cfg.get("min_price"),
            "name_exclude_keywords": dyn_cfg.get("name_exclude_keywords", []),
        },
    }
    existing = load_json(manifest_path) if manifest_path.exists() else {}
    existing_dates = existing.get("dates", []) if isinstance(existing.get("dates"), list) else []
    if existing.get("gate_signature") == gate_signature and all((cache_dir / f"{d}.jsonl").exists() for d in existing_dates):
        return {
            "quotes_arg": str(cache_dir),
            "gate_cache_used": True,
            "cache_dir": str(cache_dir),
            "dates": existing_dates,
            "eligible_codes_by_date": existing.get("eligible_codes_by_date", {}),
            "rows_by_date": existing.get("rows_by_date", {}),
            "reason": "gate_cache_hit",
        }

    for stale_file in cache_dir.glob("*.jsonl"):
        stale_file.unlink()
    eligible_counts: dict[str, int] = {}
    rows_by_date: dict[str, int] = {}
    for source in sorted(source_dir.glob("*.jsonl")):
        trade_date = source.stem[:10]
        eligible_codes: set[str] = set()
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and _passes_dynamic_replay_gate(row, dyn_cfg):
                    eligible_codes.add(str(row.get("stockCode", "")).zfill(6))
        out_path = cache_dir / f"{trade_date}.jsonl"
        written = 0
        with source.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if str(row.get("stockCode", "")).zfill(6) in eligible_codes:
                    out.write(line if line.endswith("\n") else line + "\n")
                    written += 1
        eligible_counts[trade_date] = len(eligible_codes)
        rows_by_date[trade_date] = written

    dates = sorted(eligible_counts)
    write_json(manifest_path, {
        "gate_signature": gate_signature,
        "created_at": datetime.now().astimezone().isoformat(),
        "dates": dates,
        "eligible_codes_by_date": eligible_counts,
        "rows_by_date": rows_by_date,
        "policy": "retain_all_intraday_rows_for_any_code_eligible_during_date",
    })
    return {
        "quotes_arg": str(cache_dir),
        "gate_cache_used": True,
        "cache_dir": str(cache_dir),
        "dates": dates,
        "eligible_codes_by_date": eligible_counts,
        "rows_by_date": rows_by_date,
        "reason": "gate_cache_created",
    }


def run_replay(
    config_path: Path,
    label: str,
    date_filter: str | None,
    quotes_path: str | None = None,
    timeout_seconds: int = 120,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "replay_t0_decisions.py"),
        "--config", str(config_path),
        "--label", label,
        "--output-detail", "summary",
    ]
    if date_filter:
        cmd.extend(["--date", date_filter])
    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])
    if quotes_path:
        cmd.extend(["--quotes", quotes_path])
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return False, (stdout + "\n" + stderr + f"\nreplay_timeout_after_seconds={timeout_seconds}").strip()
    return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr).strip()


def load_replay_summary(label: str) -> dict[str, Any]:
    path = DEFAULT_REPLAY_OUT / f"{label}_summary.json"
    return load_json(path) if path.exists() else {}


def summarize_candidate(name: str, overlay: dict[str, Any], summary: dict[str, Any], ok: bool, log: str) -> dict[str, Any]:
    trades = summary.get("trades", []) if isinstance(summary.get("trades"), list) else []
    pnl_values = [as_float(t.get("pnl")) for t in trades if isinstance(t, dict)]
    winning = sum(1 for x in pnl_values if x > 0)
    losing = sum(1 for x in pnl_values if x < 0)
    per_day = summary.get("per_day", {}) or {}
    entries = sum(int(as_float(d.get("entries"))) for d in per_day.values() if isinstance(d, dict))
    per_day_pnl = {k: round(as_float(v.get("pnl")), 2) for k, v in per_day.items() if isinstance(v, dict)}
    max_day_loss = min(list(per_day_pnl.values()) or [0.0])
    distinct_days = len(per_day_pnl)
    total_pnl = as_float(summary.get("total_pnl"))
    open_count = len(summary.get("open_positions_at_end", {}) or {})
    win_rate = winning / len(pnl_values) if pnl_values else 0.0
    # Precision-first paper objective: prefer less realized loss, fewer loss exits,
    # no unresolved open position, and enough actual entries to avoid "do nothing" wins.
    score = (
        total_pnl
        - 0.8 * abs(min(max_day_loss, 0.0))
        - 250.0 * losing
        + 150.0 * winning
        + 300.0 * win_rate
        - 100.0 * open_count
        - 20.0 * max(entries - 2, 0)
    )
    return {
        "candidate": name,
        "ok": ok,
        "overlay": overlay,
        "rounds_total": int(as_float(summary.get("rounds_total"))),
        "entries": entries,
        "trades": len(pnl_values),
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "max_day_loss": round(max_day_loss, 2),
        "open_positions_at_end": open_count,
        "distinct_days": distinct_days,
        "per_day_pnl": per_day_pnl,
        "objective_score": round(score, 2),
        "replay_log_tail": log[-2000:],
    }


def evaluate_candidate(
    base_cfg: dict[str, Any],
    cfg_dir: Path,
    prefix: str,
    name: str,
    overlay: dict[str, Any],
    date_filter: str | None,
    quotes_path: str | None = None,
    replay_timeout_seconds: int = 120,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    bad_paths = validate_allowed_paths(base_cfg, overlay)
    if bad_paths:
        return {
            "candidate": name,
            "ok": False,
            "overlay": overlay,
            "rounds_total": 0,
            "entries": 0,
            "trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "max_day_loss": 0.0,
            "open_positions_at_end": 0,
            "objective_score": -1_000_000.0,
            "replay_log_tail": f"invalid overlay paths: {bad_paths}",
            "invalid_overlay_paths": bad_paths,
        }
    cfg_path = cfg_dir / f"{prefix}_{name}.json"
    write_candidate_config(base_cfg, overlay, cfg_path)
    label = f"{prefix}_{name}"
    ok, log = run_replay(cfg_path, label, date_filter, quotes_path, replay_timeout_seconds, start_date, end_date)
    summary = load_replay_summary(label) if ok else {}
    return summarize_candidate(name, overlay, summary, ok, log)


def cma_es_blackbox_candidates(
    base_cfg: dict[str, Any],
    cfg_dir: Path,
    prefix: str,
    date_filter: str | None,
    quotes_path: str | None,
    replay_timeout_seconds: int,
    generations: int,
    population_size: int,
    seed: int,
    sigma0: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Run a bounded diagonal CMA-ES black-box optimization over allowlisted strategy params.

    This is deliberately offline and small-budget. It is a real distributional
    optimizer: samples a population, evaluates by replay, selects elites, and
    adapts the search mean and per-dimension variance across generations.
    """
    rng = random.Random(seed)
    mean = vector_from_strategy(base_cfg.get("strategy", {}))
    dim = len(mean)
    variance = [1.0 for _ in range(dim)]
    rows: list[dict[str, Any]] = []
    mu = max(2, population_size // 2)

    for gen in range(generations):
        gen_rows: list[dict[str, Any]] = []
        sigma = sigma0 * (0.85 ** gen)
        for idx in range(population_size):
            vec = [
                max(0.0, min(1.0, mean[j] + rng.gauss(0.0, sigma * (variance[j] ** 0.5))))
                for j in range(dim)
            ]
            overlay = overlay_from_vector(vec)
            name = f"cmaes_g{gen + 1:02d}_i{idx + 1:02d}"
            row = evaluate_candidate(
                base_cfg, cfg_dir, prefix, name, overlay, date_filter, quotes_path,
                replay_timeout_seconds, start_date, end_date
            )
            row["optimizer"] = "bounded_diagonal_cma_es"
            row["generation"] = gen + 1
            gen_rows.append(row)
            rows.append(row)

        elites = sorted(
            [r for r in gen_rows if r.get("ok")],
            key=lambda r: as_float(r.get("objective_score")),
            reverse=True,
        )[:mu]
        if not elites:
            break

        elite_vecs = [vector_from_overlay(row.get("overlay", {}), base_cfg.get("strategy", {})) for row in elites]
        new_mean = [sum(v[j] for v in elite_vecs) / len(elite_vecs) for j in range(dim)]
        new_variance: list[float] = []
        for j in range(dim):
            centered = [(v[j] - new_mean[j]) ** 2 for v in elite_vecs]
            # Keep non-zero exploration while shrinking noisy dimensions.
            new_variance.append(max(0.05, min(1.5, sum(centered) / len(centered) + 0.10 * variance[j])))
        mean = [0.65 * mean[j] + 0.35 * new_mean[j] for j in range(dim)]
        variance = new_variance

    return rows


def vector_from_overlay(overlay: dict[str, Any], base_strategy: dict[str, Any]) -> list[float]:
    merged = copy.deepcopy(base_strategy)
    deep_merge(merged, overlay)
    return vector_from_strategy(merged)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = [
        "candidate", "optimizer", "generation", "ok", "rounds_total", "entries", "trades", "winning_trades",
        "losing_trades", "win_rate", "total_pnl", "max_day_loss",
        "open_positions_at_end", "objective_score",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["candidates"]
    lines = [
        "# T0 Strategy Evolution Report",
        "",
        "Paper trading research only. Not live-ready, not a formal strategy, and not investment advice.",
        "",
        f"- Created at: {report['created_at']}",
        f"- Status: `{report['status']}`",
        f"- Selected candidate: `{report.get('selected_candidate')}`",
        f"- Decision reason: {report.get('decision_reason')}",
        "",
        "## Candidate Replay Results",
        "",
        "| candidate | optimizer | entries | trades | win_rate | total_pnl | max_day_loss | open | score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: as_float(x.get("objective_score")), reverse=True):
        lines.append(
            f"| {row['candidate']} | {row.get('optimizer', 'fixed_grid')} | {row['entries']} | {row['trades']} | {row['win_rate']:.2%} | "
            f"{row['total_pnl']:.2f} | {row['max_day_loss']:.2f} | {row['open_positions_at_end']} | "
            f"{row['objective_score']:.2f} |"
        )
    lines.extend([
        "",
        "## Applied Overlay",
        "",
        "```json",
        json.dumps(report.get("strategy_overlay", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Guardrails",
        "",
        "- Only allowlisted `strategy` paths can be auto-applied by the agent.",
        "- `mode`, `execution_enabled`, `risk`, shared execution guard, and order submission locks are not writable through this overlay.",
        "- Evidence gates can downgrade the result to `diagnostic_only`; in that case the agent ignores the overlay.",
        "- CMA-ES candidates are evaluated offline by replay only; they never call the paper-trading API.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-only T0 strategy evolution via offline replay")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--quotes", default=None, help="optional replay quote jsonl path")
    parser.add_argument("--replay-timeout-seconds", type=int, default=120)
    parser.add_argument("--date", default=None, help="optional YYYY-MM-DD replay subset")
    parser.add_argument("--start-date", default=None, help="optional inclusive validation start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="optional inclusive validation end date YYYY-MM-DD")
    parser.add_argument("--fast-top-k", type=int, default=0,
                        help="two-stage mode: screen all candidates on --stage1-date, validate baseline+top K on final dates")
    parser.add_argument("--stage1-date", default=None, help="single date for two-stage fast screening")
    parser.add_argument("--dynamic-gate-cache", action="store_true",
                        help="replay only codes passing live dynamic-universe liquidity/spread/price gates")
    parser.add_argument("--label-prefix", default=None)
    parser.add_argument("--optimizer", choices=["fixed", "cmaes", "both"], default=None)
    parser.add_argument("--cma-generations", type=int, default=None)
    parser.add_argument("--cma-population", type=int, default=None)
    parser.add_argument("--cma-seed", type=int, default=None)
    parser.add_argument("--cma-sigma", type=float, default=None)
    parser.add_argument("--no-update-latest-overlay", action="store_true",
                        help="write run-scoped research outputs without touching the agent-facing latest overlay")
    args = parser.parse_args()

    base_cfg = load_json(Path(args.config))
    si = base_cfg.get("self_iteration", {})
    out_dir = DEFAULT_OUT_DIR
    cfg_dir = out_dir / "candidate_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.label_prefix or f"evolution_{stamp}"
    rows: list[dict[str, Any]] = []
    stage1_rows: list[dict[str, Any]] = []
    invalid: dict[str, list[str]] = {}
    cache_meta = prepare_daily_quote_cache(
        args.quotes,
        date_filter=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    quotes_arg = cache_meta.get("quotes_arg")
    dynamic_gate_meta: dict[str, Any] = {"gate_cache_used": False, "reason": "disabled"}
    if args.dynamic_gate_cache:
        dyn_cfg = base_cfg.get("dynamic_universe", {}) if isinstance(base_cfg.get("dynamic_universe"), dict) else {}
        dynamic_gate_meta = prepare_dynamic_gate_cache(quotes_arg, dyn_cfg)
        quotes_arg = dynamic_gate_meta.get("quotes_arg")

    optimizer_cfg = si.get("blackbox_optimizer", {}) if isinstance(si.get("blackbox_optimizer", {}), dict) else {}
    optimizer_mode = args.optimizer or str(optimizer_cfg.get("mode", "both"))

    if optimizer_mode in {"fixed", "both"}:
        fixed_candidates = candidate_overlays()
    elif optimizer_mode == "cmaes":
        fixed_candidates = [candidate_overlays()[0]]
    else:
        fixed_candidates = []

    if args.fast_top_k > 0 and not args.stage1_date:
        stage_dates = cache_meta.get("dates", []) if isinstance(cache_meta.get("dates"), list) else []
        args.stage1_date = stage_dates[-1] if stage_dates else args.date

    if args.fast_top_k > 0 and args.stage1_date:
        stage1_prefix = f"{prefix}_stage1"
        if fixed_candidates:
            for cand in fixed_candidates:
                name = cand["name"]
                row = evaluate_candidate(
                    base_cfg,
                    cfg_dir,
                    stage1_prefix,
                    name,
                    cand["overlay"],
                    args.stage1_date,
                    quotes_arg,
                    args.replay_timeout_seconds,
                )
                row["optimizer"] = "fixed_grid"
                row["screening_stage"] = "stage1"
                if row.get("invalid_overlay_paths"):
                    invalid[name] = row["invalid_overlay_paths"]
                stage1_rows.append(row)

        if optimizer_mode in {"cmaes", "both"}:
            cma_stage1_rows = cma_es_blackbox_candidates(
                base_cfg=base_cfg,
                cfg_dir=cfg_dir,
                prefix=stage1_prefix,
                date_filter=args.stage1_date,
                quotes_path=quotes_arg,
                replay_timeout_seconds=args.replay_timeout_seconds,
                generations=args.cma_generations or int(as_float(optimizer_cfg.get("generations"), 2)),
                population_size=args.cma_population or int(as_float(optimizer_cfg.get("population_size"), 4)),
                seed=args.cma_seed or int(as_float(optimizer_cfg.get("seed"), 20260614)),
                sigma0=args.cma_sigma if args.cma_sigma is not None else as_float(optimizer_cfg.get("sigma0"), 0.22),
            )
            for row in cma_stage1_rows:
                row["screening_stage"] = "stage1"
                if row.get("invalid_overlay_paths"):
                    invalid[row["candidate"]] = row["invalid_overlay_paths"]
            stage1_rows.extend(cma_stage1_rows)

        stage1_ok = [r for r in stage1_rows if r.get("ok")]
        baseline_stage1 = next((r for r in stage1_ok if r["candidate"] == "baseline_current"), None)
        top_rows = sorted(
            [r for r in stage1_ok if r["candidate"] != "baseline_current"],
            key=lambda x: as_float(x.get("objective_score")),
            reverse=True,
        )[:max(0, args.fast_top_k)]
        final_specs: list[dict[str, Any]] = []
        if baseline_stage1:
            final_specs.append({"name": "baseline_current", "overlay": {}, "optimizer": "fixed_grid"})
        final_specs.extend({
            "name": r["candidate"],
            "overlay": r.get("overlay", {}) if isinstance(r.get("overlay"), dict) else {},
            "optimizer": r.get("optimizer", "stage1_selected"),
            "generation": r.get("generation"),
        } for r in top_rows)

        final_prefix = f"{prefix}_final"
        for spec in final_specs:
            row = evaluate_candidate(
                base_cfg,
                cfg_dir,
                final_prefix,
                spec["name"],
                spec["overlay"],
                args.date,
                quotes_arg,
                args.replay_timeout_seconds,
                args.start_date,
                args.end_date,
            )
            row["optimizer"] = spec.get("optimizer", "stage1_selected")
            row["generation"] = spec.get("generation")
            row["screening_stage"] = "final_validation"
            rows.append(row)

    elif fixed_candidates:
        for cand in fixed_candidates:
            name = cand["name"]
            row = evaluate_candidate(
                base_cfg,
                cfg_dir,
                prefix,
                name,
                cand["overlay"],
                args.date,
                quotes_arg,
                args.replay_timeout_seconds,
                args.start_date,
                args.end_date,
            )
            row["optimizer"] = "fixed_grid"
            if row.get("invalid_overlay_paths"):
                invalid[name] = row["invalid_overlay_paths"]
            rows.append(row)

    if args.fast_top_k <= 0 and optimizer_mode in {"cmaes", "both"}:
        cma_rows = cma_es_blackbox_candidates(
            base_cfg=base_cfg,
            cfg_dir=cfg_dir,
            prefix=prefix,
            date_filter=args.date,
            quotes_path=quotes_arg,
            replay_timeout_seconds=args.replay_timeout_seconds,
            generations=args.cma_generations or int(as_float(optimizer_cfg.get("generations"), 2)),
            population_size=args.cma_population or int(as_float(optimizer_cfg.get("population_size"), 4)),
            seed=args.cma_seed or int(as_float(optimizer_cfg.get("seed"), 20260614)),
            sigma0=args.cma_sigma if args.cma_sigma is not None else as_float(optimizer_cfg.get("sigma0"), 0.22),
            start_date=args.start_date,
            end_date=args.end_date,
        )
        for row in cma_rows:
            if row.get("invalid_overlay_paths"):
                invalid[row["candidate"]] = row["invalid_overlay_paths"]
        rows.extend(cma_rows)

    rows_ok = [r for r in rows if r.get("ok")]
    baseline = next((r for r in rows_ok if r["candidate"] == "baseline_current"), None)
    selected = max(rows_ok, key=lambda x: as_float(x.get("objective_score"))) if rows_ok else None

    status = "diagnostic_only"
    decision_reason = "no_valid_replay"
    selected_overlay: dict[str, Any] = {}
    dm_result: dict[str, Any] = {"reason": "no_valid_replay", "significant": False}
    mcs_result: dict[str, Any] = {"reason": "no_valid_replay", "mcs_set": []}
    spa_result: dict[str, Any] = {"reason": "no_valid_replay", "reject": False, "p_value": 1.0}
    dsr_result: dict[str, Any] = {"reason": "no_valid_replay", "significant": False}
    if baseline and selected:
        min_trades = int(as_float(si.get("min_replay_trades_for_auto_apply"), 2))
        min_entries = int(as_float(si.get("min_candidate_entries"), 1))
        min_distinct_days = int(as_float(si.get("min_distinct_days"), 0))
        min_improvement = as_float(si.get("min_score_improvement"), 100)
        max_day_loss_worsening_allowed = as_float(si.get("max_day_loss_worsening_allowed"), 0.0)
        improvement = as_float(selected.get("objective_score")) - as_float(baseline.get("objective_score"))
        max_day_loss_worsening = as_float(baseline.get("max_day_loss")) - as_float(selected.get("max_day_loss"))
        selected_overlay = selected.get("overlay", {}) if isinstance(selected.get("overlay"), dict) else {}

        # Small-sample selection-robustness guards (multiple-testing + per-day dominance).
        # When the best-of-N candidate is chosen, its apparent edge is inflated by
        # selection; require it to clear a multiplicity-scaled bar (deflated-improvement
        # idea) AND to not regress baseline on any single shared day (Majority/walk-
        # forward idea). These only make the gate stricter; they never auto-apply more.
        n_tested = len([r for r in rows_ok if r["candidate"] != "baseline_current"])
        selection_penalty_coef = as_float(si.get("selection_penalty_log_coef"), 0.5)
        multiplicity_factor = 1.0 + selection_penalty_coef * math.log(max(1, n_tested))
        effective_min_improvement = min_improvement * multiplicity_factor
        max_per_day_regression_allowed = as_float(si.get("max_per_day_regression_allowed"), 0.0)
        base_days = baseline.get("per_day_pnl", {}) if isinstance(baseline.get("per_day_pnl"), dict) else {}
        sel_days = selected.get("per_day_pnl", {}) if isinstance(selected.get("per_day_pnl"), dict) else {}
        shared_days = sorted(set(base_days) & set(sel_days))
        worst_day_regression = max(
            (as_float(base_days[d]) - as_float(sel_days[d]) for d in shared_days),
            default=0.0,
        )
        # Diebold-Mariano (HLN-corrected) test on per-day PnL differential vs baseline.
        require_dm = bool(si.get("require_diebold_mariano_significant", True))
        dm_alpha = as_float(si.get("dm_one_sided_alpha"), 0.05)
        dm_result = diebold_mariano_hln(base_days, sel_days, alpha=dm_alpha)
        # Model Confidence Set across ALL candidates (family-wise multiple comparison).
        # Loss = -pnl over days shared by every OK candidate; auto-apply only if
        # baseline is statistically EXCLUDED from the MCS (dominated by the set).
        require_mcs = bool(si.get("require_baseline_excluded_from_mcs", True))
        mcs_alpha = as_float(si.get("mcs_alpha"), 0.10)
        mcs_n_boot = int(as_float(si.get("mcs_n_boot"), 500))
        all_day_keys = [set(r.get("per_day_pnl", {}) or {}) for r in rows_ok]
        mcs_shared = sorted(set.intersection(*all_day_keys)) if all_day_keys else []
        if len(mcs_shared) >= 2:
            loss_by_model = {
                r["candidate"]: [-as_float((r.get("per_day_pnl", {}) or {}).get(d)) for d in mcs_shared]
                for r in rows_ok
            }
            mcs_result = model_confidence_set(loss_by_model, alpha=mcs_alpha, n_boot=mcs_n_boot)
        else:
            mcs_result = {"mcs_set": [r["candidate"] for r in rows_ok], "reason": "insufficient_shared_days", "n_obs": len(mcs_shared)}
        baseline_in_mcs = "baseline_current" in mcs_result.get("mcs_set", [])
        # SPA / White Reality Check: snooping-robust test that SOME candidate beats
        # baseline (benchmark). This is the rigorous replacement for the ad-hoc
        # log-multiplicity penalty above (kept as defense-in-depth).
        require_spa = bool(si.get("require_spa_reject", True))
        spa_alpha = as_float(si.get("spa_alpha"), 0.05)
        spa_n_boot = int(as_float(si.get("spa_n_boot"), 500))
        if len(mcs_shared) >= 2 and "baseline_current" in {r["candidate"] for r in rows_ok}:
            bench_loss = [-as_float((baseline.get("per_day_pnl", {}) or {}).get(d)) for d in mcs_shared]
            alt_loss = {
                r["candidate"]: [-as_float((r.get("per_day_pnl", {}) or {}).get(d)) for d in mcs_shared]
                for r in rows_ok if r["candidate"] != "baseline_current"
            }
            spa_result = reality_check_spa(bench_loss, alt_loss, alpha=spa_alpha, n_boot=spa_n_boot)
        else:
            spa_result = {"reject": False, "reason": "insufficient_shared_days", "p_value": 1.0}
        # Deflated Sharpe: repeated-search penalty on selected-vs-baseline daily
        # PnL differentials. This is another overfit guard, not a profitability
        # claim, and tiny samples fail closed.
        require_dsr = bool(si.get("require_deflated_sharpe_significant", True))
        dsr_alpha = as_float(si.get("deflated_sharpe_alpha"), 0.10)
        dsr_result = deflated_sharpe_diagnostic(base_days, sel_days, n_trials=max(1, n_tested), alpha=dsr_alpha)
        if selected["candidate"] == "baseline_current":
            decision_reason = "baseline_ranked_best"
            selected_overlay = {}
        elif int(selected.get("trades", 0)) < min_trades:
            decision_reason = f"selected_trades_below_min:{selected.get('trades')}<{min_trades}"
        elif int(selected.get("entries", 0)) < min_entries:
            decision_reason = f"selected_entries_below_min:{selected.get('entries')}<{min_entries}"
        elif int(selected.get("distinct_days", 0)) < min_distinct_days:
            decision_reason = f"sample_distinct_days_below_min:{selected.get('distinct_days')}<{min_distinct_days}"
        elif improvement < min_improvement:
            decision_reason = f"score_improvement_below_min:{improvement:.2f}<{min_improvement:.2f}"
        elif improvement < effective_min_improvement:
            decision_reason = (
                f"improvement_below_multiplicity_adjusted_bar:"
                f"{improvement:.2f}<{effective_min_improvement:.2f}(N={n_tested})"
            )
        elif worst_day_regression > max_per_day_regression_allowed:
            decision_reason = (
                f"selected_regresses_a_single_day_vs_baseline:"
                f"{worst_day_regression:.2f}>{max_per_day_regression_allowed:.2f}"
            )
        elif as_float(selected.get("total_pnl")) < as_float(baseline.get("total_pnl")):
            decision_reason = "selected_total_pnl_worse_than_baseline"
        elif max_day_loss_worsening > max_day_loss_worsening_allowed:
            decision_reason = (
                f"selected_max_day_loss_worsening_too_large:"
                f"{max_day_loss_worsening:.2f}>{max_day_loss_worsening_allowed:.2f}"
            )
        elif int(selected.get("losing_trades", 0)) > int(baseline.get("losing_trades", 0)):
            decision_reason = "selected_has_more_losing_trades_than_baseline"
        elif int(selected.get("open_positions_at_end", 0)) > int(baseline.get("open_positions_at_end", 0)):
            decision_reason = "selected_leaves_more_open_positions_than_baseline"
        elif require_dm and not dm_result.get("significant"):
            decision_reason = (
                f"diebold_mariano_not_significant:"
                f"dm*={dm_result.get('dm_star')}<crit={dm_result.get('t_critical')}"
                f"(n={dm_result.get('n_days')},{dm_result.get('reason')})"
            )
        elif require_mcs and baseline_in_mcs:
            decision_reason = (
                f"baseline_in_model_confidence_set:"
                f"mcs_size={len(mcs_result.get('mcs_set', []))},n_obs={mcs_result.get('n_obs')}"
            )
        elif require_spa and not spa_result.get("reject"):
            decision_reason = (
                f"spa_reality_check_not_significant:"
                f"p={spa_result.get('p_value')}>={spa_alpha}({spa_result.get('reason')})"
            )
        elif require_dsr and not dsr_result.get("significant"):
            decision_reason = (
                f"deflated_sharpe_not_significant:"
                f"p={dsr_result.get('p_value')}>={dsr_alpha}({dsr_result.get('reason')})"
            )
        else:
            status = "approved_for_paper_auto_apply"
            decision_reason = f"selected_improved_objective_by:{improvement:.2f}"

    selection_diagnostics = {
        "max_day_loss_worsening_allowed": as_float(si.get("max_day_loss_worsening_allowed"), 0.0),
        "selection_penalty_log_coef": as_float(si.get("selection_penalty_log_coef"), 0.5),
        "max_per_day_regression_allowed": as_float(si.get("max_per_day_regression_allowed"), 0.0),
        "candidates_tested": len([r for r in rows if r.get("ok") and r.get("candidate") != "baseline_current"]),
        "diebold_mariano": dm_result,
        "require_diebold_mariano_significant": bool(si.get("require_diebold_mariano_significant", True)),
        "model_confidence_set": {k: mcs_result.get(k) for k in ("mcs_set", "n_obs", "reason", "alpha")},
        "require_baseline_excluded_from_mcs": bool(si.get("require_baseline_excluded_from_mcs", True)),
        "spa_reality_check": {k: spa_result.get(k) for k in ("p_value", "reject", "best_alt", "n_obs", "reason")},
        "require_spa_reject": bool(si.get("require_spa_reject", True)),
        "deflated_sharpe": dsr_result,
        "require_deflated_sharpe_significant": bool(si.get("require_deflated_sharpe_significant", True)),
    }

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "gate_version": EVOLUTION_GATE_VERSION,
        "paper_trading_only": True,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "investment_advice": False,
        "status": status,
        "decision_reason": decision_reason,
        "selected_candidate": selected.get("candidate") if selected else None,
        "baseline_candidate": baseline,
        "selected_candidate_metrics": selected,
        "auto_apply_risk_limits": selection_diagnostics,
        "selection_diagnostics": selection_diagnostics,
        "strategy_overlay": selected_overlay if status == "approved_for_paper_auto_apply" else {},
        "suggested_strategy_overlay": selected_overlay,
        "optimizer_mode": optimizer_mode,
        "quote_cache": cache_meta,
        "dynamic_gate_cache": dynamic_gate_meta,
        "validation_date_filter": args.date,
        "validation_start_date": args.start_date,
        "validation_end_date": args.end_date,
        "two_stage_fast_screen": {
            "enabled": args.fast_top_k > 0,
            "stage1_date": args.stage1_date,
            "fast_top_k": args.fast_top_k,
            "stage1_candidates": stage1_rows,
        },
        "cma_es": {
            "enabled": optimizer_mode in {"cmaes", "both"},
            "generations": args.cma_generations or int(as_float(optimizer_cfg.get("generations"), 2)),
            "population_size": args.cma_population or int(as_float(optimizer_cfg.get("population_size"), 4)),
            "seed": args.cma_seed or int(as_float(optimizer_cfg.get("seed"), 20260614)),
            "sigma0": args.cma_sigma if args.cma_sigma is not None else as_float(optimizer_cfg.get("sigma0"), 0.22),
            "param_space_size": len(PARAM_SPACE),
            "implementation": "bounded_diagonal_cma_es_offline_replay",
        },
        "invalid_overlay_paths": invalid,
        "candidates": rows,
        "note": "Offline replay over historical snapshots; small sample is not a profitability claim.",
    }

    write_json(out_dir / f"{prefix}_summary.json", report)
    write_csv_rows(out_dir / f"{prefix}_candidate_results.csv", rows)
    write_markdown(out_dir / f"{prefix}_summary.md", report)
    if not args.no_update_latest_overlay:
        write_json(out_dir / "latest_strategy_overlay.json", report)
        write_markdown(out_dir / "latest_strategy_evolution.md", report)

    print(json.dumps({
        "status": status,
        "selected_candidate": report["selected_candidate"],
        "decision_reason": decision_reason,
        "overlay": None if args.no_update_latest_overlay else str(out_dir / "latest_strategy_overlay.json"),
        "live_ready": False,
        "formal_strategy_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
