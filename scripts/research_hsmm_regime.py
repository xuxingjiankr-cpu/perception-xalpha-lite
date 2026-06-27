"""Post-publication validation of Liu & Wang's three-state CSI 300 HSMM strategy.

Research-only.  The published 2005-2016 parameters are frozen before the clean
2017+ test.  At every close, a right-censored explicit-duration Viterbi decoder
uses observations available through that close only; the resulting position is
applied to the next close-to-close return.

The paper strategy is bear=-1, sidewalk=0, bull=+1.  A long-only variant
(bear/sidewalk=0, bull=+1) is also reported because the production ETF system is
long-only.  Neither variant is connected to live trading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import logser

from run_etf_paper_trading_agent import ROOT


CST = timezone(timedelta(hours=8))
OUT_DIR = ROOT / "outputs" / "edge_research" / "hsmm_regime"
SOURCE_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid=1.000300&klt=101&fqt=0&beg=20050408&end=20500101&lmt=10000"
    "&fields1=f1,f2,f3,f4,f5,f6"
    "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
)

# Published full-sample estimates, April 2005 through May 2016.
# Returns are log percentage returns, exactly as in the paper.
PUBLISHED_MEANS = np.array([-0.510, -0.020, 0.622], dtype=float)
PUBLISHED_STDS = np.array([3.113, 1.156, 1.440], dtype=float)
PUBLISHED_DURATION_MEANS = np.array([26.00, 204.29, 27.80], dtype=float)
PUBLISHED_TRANSITION = np.array(
    [
        [0.0, 0.0002, 0.9998],
        [0.4956, 0.0, 0.5044],
        [0.7408, 0.2592, 0.0],
    ],
    dtype=float,
)
PUBLISHED_STATE_COUNTS = np.array([572.0, 1430.0, 695.0], dtype=float)
STATE_NAMES = ("bear", "sidewalk", "bull")


@dataclass(frozen=True)
class DailyData:
    dates: list[str]
    closes: np.ndarray


def fetch_csi300_daily(timeout: float = 30.0) -> DailyData:
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    klines = ((payload.get("data") or {}).get("klines") or [])
    rows: list[tuple[str, float]] = []
    for line in klines:
        cols = str(line).split(",")
        if len(cols) < 3:
            continue
        try:
            close = float(cols[2])
        except (TypeError, ValueError):
            continue
        if close > 0:
            rows.append((cols[0], close))
    if len(rows) < 2500:
        raise RuntimeError(f"insufficient CSI 300 history: {len(rows)} rows")
    return DailyData([x[0] for x in rows], np.asarray([x[1] for x in rows], dtype=float))


def logseries_p_from_mean(target_mean: float) -> float:
    """Infer the paper's one-parameter logarithmic duration from its reported mean."""
    if target_mean <= 1.0:
        raise ValueError("log-series mean must exceed one")

    def mean_at(p: float) -> float:
        return -p / ((1.0 - p) * math.log1p(-p))

    return float(brentq(lambda p: mean_at(p) - target_mean, 1e-8, 1.0 - 1e-12))


def published_parameters() -> dict[str, np.ndarray]:
    duration_p = np.asarray(
        [logseries_p_from_mean(x) for x in PUBLISHED_DURATION_MEANS], dtype=float
    )
    pi = PUBLISHED_STATE_COUNTS / PUBLISHED_STATE_COUNTS.sum()
    return {
        "means": PUBLISHED_MEANS.copy(),
        "stds": PUBLISHED_STDS.copy(),
        "duration_p": duration_p,
        "transition": PUBLISHED_TRANSITION.copy(),
        "pi": pi,
    }


def decode_expanding_right_censored(
    observations: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    max_duration: int | None = None,
) -> np.ndarray:
    """Return the last state of the best path for every expanding prefix.

    Completed segments use the log-series duration PMF.  The final segment at
    every prefix is right-censored and uses P(D >= current_age), matching the
    paper's endpoint treatment.  No future observation enters a prefix state.
    """
    x = np.asarray(observations, dtype=float)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=np.int16)
    means = np.asarray(params["means"], dtype=float)
    stds = np.maximum(np.asarray(params["stds"], dtype=float), 1e-6)
    duration_p = np.asarray(params["duration_p"], dtype=float)
    trans = np.asarray(params["transition"], dtype=float)
    pi = np.asarray(params["pi"], dtype=float)
    k_count = len(means)
    if trans.shape != (k_count, k_count):
        raise ValueError("transition shape mismatch")
    d_cap = min(n, max_duration or n)

    log_a = np.full_like(trans, -np.inf, dtype=float)
    positive = trans > 0
    log_a[positive] = np.log(trans[positive])
    log_pi = np.log(np.maximum(pi, 1e-300))

    log_emit = np.empty((k_count, n), dtype=float)
    for k in range(k_count):
        z = (x - means[k]) / stds[k]
        log_emit[k] = -0.5 * z * z - math.log(stds[k]) - 0.5 * math.log(2.0 * math.pi)
    cumulative = np.concatenate(
        [np.zeros((k_count, 1), dtype=float), np.cumsum(log_emit, axis=1)], axis=1
    )

    durations = np.arange(1, d_cap + 1, dtype=int)
    log_pmf = np.vstack([logser.logpmf(durations, p) for p in duration_p])
    log_survival = np.vstack([logser.logsf(durations - 1, p) for p in duration_p])

    completed = np.full((n, k_count), -np.inf, dtype=float)
    best_transition = np.full((n, k_count), -np.inf, dtype=float)
    prefix_states = np.zeros(n, dtype=np.int16)

    for t in range(n):
        d_max = min(d_cap, t + 1)
        ds = durations[:d_max]
        starts = t - ds + 1
        prev_ends = starts - 1
        censored_state_scores = np.full(k_count, -np.inf, dtype=float)
        for k in range(k_count):
            base = np.empty(d_max, dtype=float)
            at_start = prev_ends < 0
            base[at_start] = log_pi[k]
            if np.any(~at_start):
                base[~at_start] = best_transition[prev_ends[~at_start], k]
            segment_ll = cumulative[k, t + 1] - cumulative[k, starts]
            completed[t, k] = float(
                np.max(base + log_pmf[k, :d_max] + segment_ll)
            )
            censored_state_scores[k] = float(
                np.max(base + log_survival[k, :d_max] + segment_ll)
            )
        for destination in range(k_count):
            best_transition[t, destination] = float(
                np.max(completed[t] + log_a[:, destination])
            )
        prefix_states[t] = int(np.argmax(censored_state_scores))
    return prefix_states


def positions_from_states(states: np.ndarray, mapping: tuple[float, float, float]) -> np.ndarray:
    """Shift a close-t state to the t+1 return, preventing same-close execution."""
    raw = np.asarray([mapping[int(s)] for s in states], dtype=float)
    positions = np.zeros_like(raw)
    if len(raw) > 1:
        positions[1:] = raw[:-1]
    return positions


def ma50_positions(closes: np.ndarray) -> np.ndarray:
    signal = np.zeros(len(closes) - 1, dtype=float)
    for observation_idx in range(len(signal)):
        close_idx = observation_idx + 1
        if close_idx >= 49:
            signal[observation_idx] = float(
                closes[close_idx] >= np.mean(closes[close_idx - 49 : close_idx + 1])
            )
    out = np.zeros_like(signal)
    if len(signal) > 1:
        out[1:] = signal[:-1]
    return out


def strategy_returns(
    asset_returns: np.ndarray,
    positions: np.ndarray,
    mask: np.ndarray,
    *,
    one_way_cost: float,
    short_borrow_annual: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(asset_returns[mask], dtype=float)
    p = np.asarray(positions[mask], dtype=float)
    if len(r) != len(p):
        raise ValueError("return/position length mismatch")
    changes = np.empty_like(p)
    if len(p):
        changes[0] = abs(p[0])  # every OOS evaluation starts flat
        if len(p) > 1:
            changes[1:] = np.abs(np.diff(p))
    borrowing = np.maximum(-p, 0.0) * (short_borrow_annual / 252.0)
    net = p * r - changes * one_way_cost - borrowing
    return net, p


def metrics(returns: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    p = np.asarray(positions, dtype=float)
    if len(r) == 0:
        return {}
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    years = len(r) / 252.0
    total = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / sd * math.sqrt(252.0)) if sd > 0 else 0.0
    changes = np.empty_like(p)
    changes[0] = abs(p[0]) if len(p) else 0.0
    if len(p) > 1:
        changes[1:] = np.abs(np.diff(p))
    active = p != 0
    return {
        "days": int(len(r)),
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": sd * math.sqrt(252.0),
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)),
        "calmar": cagr / abs(float(np.min(drawdown))) if np.min(drawdown) < 0 else None,
        "turnover_units": float(np.sum(changes)),
        "position_changes": int(np.sum(changes > 0)),
        "long_fraction": float(np.mean(p > 0)),
        "short_fraction": float(np.mean(p < 0)),
        "flat_fraction": float(np.mean(p == 0)),
        "active_win_rate": float(np.mean(r[active] > 0)) if np.any(active) else None,
    }


def circular_block_bootstrap_ci(
    values: np.ndarray,
    *,
    block: int = 20,
    n_boot: int = 3000,
    seed: int = 2827838,
) -> list[float]:
    x = np.asarray(values, dtype=float)
    if len(x) < block * 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    needed = math.ceil(len(x) / block)
    samples = np.empty(n_boot, dtype=float)
    offsets = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, len(x), size=needed)
        idx = ((starts[:, None] + offsets[None, :]) % len(x)).ravel()[: len(x)]
        samples[b] = float(np.mean(x[idx]) * 252.0)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def evaluate_period(
    name: str,
    start: str,
    end: str,
    return_dates: np.ndarray,
    asset_returns: np.ndarray,
    position_sets: dict[str, np.ndarray],
) -> dict[str, Any]:
    mask = (return_dates >= start) & (return_dates <= end)
    if not np.any(mask):
        raise ValueError(f"no observations for {name}: {start}..{end}")
    definitions = {
        "buy_hold": (position_sets["buy_hold"], 0.0006, 0.0),
        "ma50_long_flat": (position_sets["ma50_long_flat"], 0.0006, 0.0),
        "paper_long_short_gross": (position_sets["paper_long_short"], 0.0, 0.0),
        "paper_long_short_net": (position_sets["paper_long_short"], 0.0006, 0.03),
        "paper_long_short_stress": (position_sets["paper_long_short"], 0.0015, 0.06),
        "paper_long_flat_net": (position_sets["paper_long_flat"], 0.0006, 0.0),
        "paper_long_flat_stress": (position_sets["paper_long_flat"], 0.0015, 0.0),
    }
    strategies: dict[str, Any] = {}
    daily: dict[str, np.ndarray] = {}
    for strategy, (positions, cost, borrow) in definitions.items():
        net, used_pos = strategy_returns(
            asset_returns,
            positions,
            mask,
            one_way_cost=cost,
            short_borrow_annual=borrow,
        )
        daily[strategy] = net
        m = metrics(net, used_pos)
        m["annual_mean_ci95_block20"] = circular_block_bootstrap_ci(net)
        strategies[strategy] = m
    benchmark = daily["buy_hold"]
    for strategy, values in daily.items():
        strategies[strategy]["annual_excess_vs_buy_hold_ci95_block20"] = (
            circular_block_bootstrap_ci(values - benchmark)
        )
    return {
        "name": name,
        "start": str(return_dates[mask][0]),
        "end": str(return_dates[mask][-1]),
        "strategies": strategies,
        "_daily": daily,
        "_mask": mask,
    }


def fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}%"


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Three-State HSMM Strategy — Post-Publication Validation",
        "",
        "Status: `diagnostic_only / no live gating / no parameter refit on 2017+`",
        "",
        "## Method",
        "",
        f"- Source: CSI 300 daily index, Eastmoney, {result['data']['start']} to {result['data']['end']}.",
        "- Frozen parameters: Liu & Wang (2017), estimated on 2005-04-08 to 2016-05-13.",
        "- Clean primary OOS: 2017-01-01 onward.",
        "- 2014-2016 is shown only as a contaminated parameter-fidelity check, not validation.",
        "- Signal timing: state at close t; position earns close t to close t+1 return.",
        "- Decoder: expanding-prefix, explicit log-series durations, right-censored final state.",
        "- Net cost: 6 bps per one-way turnover; long/short also pays 3% annual short borrow.",
        "- Stress cost: 15 bps per one-way turnover; long/short pays 6% annual short borrow.",
        "",
    ]
    for period in result["periods"]:
        lines.extend(
            [
                f"## {period['name']} ({period['start']} → {period['end']})",
                "",
                "| Strategy | CAGR | Sharpe | Max DD | Turnover | Long | Short |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy, m in period["strategies"].items():
            lines.append(
                f"| {strategy} | {fmt_pct(m['cagr'])} | {m['sharpe']:.3f} | "
                f"{fmt_pct(m['max_drawdown'])} | {m['turnover_units']:.1f} | "
                f"{fmt_pct(m['long_fraction'])} | {fmt_pct(m['short_fraction'])} |"
            )
        lines.append("")
    v = result["verdict"]
    lines.extend(
        [
            "## Locked verdict",
            "",
            f"- Published-period parameter fidelity: `{v['replication_fidelity']}`.",
            f"- Paper long/short post-publication replication: `{v['paper_long_short']}`.",
            f"- Production-compatible long/flat regime filter: `{v['long_only']}`.",
            f"- Overall: `{v['overall_status']}`.",
            "",
            "The paper's headline return cannot be transferred to the current system unless the "
            "post-publication long/short result survives costs and the long-only variant improves "
            "risk-adjusted performance. Short selling remains outside the production mandate.",
            "",
            "## Limitations",
            "",
            "- Published component, transition, and average-duration estimates are used; the authors' "
            "original R `hsmm` object and duration parameters were not published.",
            "- The paper's training-split parameter object was not published; therefore the reported "
            "2014-2016 result cannot be reproduced exactly from the article alone.",
            "- Log-series duration parameters are inferred from the reported average durations.",
            "- Eastmoney prices replace the paper's Wind data.",
            "- One index and one post-publication path do not establish universal alpha.",
            "- Results do not include futures margin, short-sale availability, tax, or market impact.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path = OUT_DIR) -> dict[str, Any]:
    data = fetch_csi300_daily()
    closes = data.closes
    dates = np.asarray(data.dates)
    log_returns_pct = 100.0 * np.diff(np.log(closes))
    simple_returns = closes[1:] / closes[:-1] - 1.0
    return_dates = dates[1:]

    params = published_parameters()
    states = decode_expanding_right_censored(log_returns_pct, params)
    if len(states) != len(simple_returns):
        raise AssertionError("state/return alignment mismatch")

    position_sets = {
        "buy_hold": np.ones_like(simple_returns),
        "ma50_long_flat": ma50_positions(closes),
        "paper_long_short": positions_from_states(states, (-1.0, 0.0, 1.0)),
        "paper_long_flat": positions_from_states(states, (0.0, 0.0, 1.0)),
    }
    periods_raw = [
        evaluate_period(
            "paper_period_parameter_check_not_oos",
            "2014-01-01",
            "2016-05-13",
            return_dates,
            simple_returns,
            position_sets,
        ),
        evaluate_period(
            "post_publication_oos",
            "2017-01-01",
            "2099-12-31",
            return_dates,
            simple_returns,
            position_sets,
        ),
        evaluate_period(
            "recent_market_subset",
            "2021-01-01",
            "2099-12-31",
            return_dates,
            simple_returns,
            position_sets,
        ),
    ]

    calibration = periods_raw[0]["strategies"]["paper_long_short_gross"]
    primary = periods_raw[1]["strategies"]
    ls = primary["paper_long_short_net"]
    lf = primary["paper_long_flat_net"]
    bh = primary["buy_hold"]
    ls_ci = ls["annual_mean_ci95_block20"]
    ls_pass = bool(
        ls["cagr"] > 0
        and ls["sharpe"] >= 0.75
        and math.isfinite(ls_ci[0])
        and ls_ci[0] > 0
    )
    dd_improvement = (
        (abs(bh["max_drawdown"]) - abs(lf["max_drawdown"]))
        / abs(bh["max_drawdown"])
        if bh["max_drawdown"] < 0
        else 0.0
    )
    lf_pass = bool(
        lf["cagr"] > 0
        and lf["sharpe"] >= bh["sharpe"]
        and dd_improvement >= 0.20
    )
    replication_fidelity = bool(
        abs(calibration["cagr"] - 0.3759) <= 0.10
        and abs(calibration["sharpe"] - 1.14) <= 0.30
    )
    verdict = {
        "replication_fidelity": "adequate" if replication_fidelity else "low",
        "paper_reported": {
            "annual_return": 0.3759,
            "sharpe": 1.14,
            "max_drawdown": -0.2134,
        },
        "parameter_check_observed": {
            "cagr": calibration["cagr"],
            "sharpe": calibration["sharpe"],
            "max_drawdown": calibration["max_drawdown"],
        },
        "paper_long_short": "pass" if ls_pass else "fail",
        "long_only": "pass" if lf_pass else "fail",
        "long_only_max_drawdown_improvement_vs_buy_hold": dd_improvement,
        "criteria": {
            "paper_long_short": "net CAGR>0, Sharpe>=0.75, block-bootstrap annual mean CI lower>0",
            "long_only": "net CAGR>0, Sharpe>=buy-hold, max drawdown improves >=20%",
        },
        "overall_status": (
            "candidate_for_additional_shadow_validation"
            if ls_pass or lf_pass
            else (
                "post_publication_validation_failed_low_replication_fidelity"
                if not replication_fidelity
                else "post_publication_validation_failed"
            )
        ),
    }

    result = {
        "schemaVersion": "hsmm_regime_post_publication_validation_v1",
        "generatedAt": datetime.now(CST).isoformat(),
        "status": "diagnostic_only",
        "live_config_modified": False,
        "paper": {
            "title": "Decoding Chinese Stock Market Returns: Three-State Hidden Semi-Markov Model",
            "authors": "Zhenya Liu; Shixuan Wang",
            "doi": "10.1016/j.pacfin.2017.06.007",
            "ssrn": "2827838",
        },
        "data": {
            "source": "Eastmoney CSI 300 index 000300.SH",
            "source_url": SOURCE_URL,
            "start": data.dates[0],
            "end": data.dates[-1],
            "price_rows": len(data.dates),
            "return_rows": len(simple_returns),
        },
        "frozen_parameters": {
            "means_log_return_pct": params["means"].tolist(),
            "stds_log_return_pct": params["stds"].tolist(),
            "duration_means_days": PUBLISHED_DURATION_MEANS.tolist(),
            "inferred_logseries_p": params["duration_p"].tolist(),
            "transition": params["transition"].tolist(),
            "initial_probabilities": params["pi"].tolist(),
        },
        "state_frequency_full_history": {
            STATE_NAMES[i]: float(np.mean(states == i)) for i in range(3)
        },
        "periods": [
            {
                key: value
                for key, value in period.items()
                if not key.startswith("_")
            }
            for period in periods_raw
        ],
        "verdict": verdict,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hsmm_post_publication_validation.json"
    md_path = output_dir / "hsmm_post_publication_validation.md"
    csv_path = output_dir / "hsmm_post_publication_daily.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report_markdown(result), encoding="utf-8")

    primary_mask = periods_raw[1]["_mask"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "date",
                "asset_return",
                "state",
                "paper_long_short_position",
                "paper_long_short_net_return",
                "paper_long_flat_position",
                "paper_long_flat_net_return",
                "buy_hold_return",
            ]
        )
        ls_daily, ls_pos = strategy_returns(
            simple_returns,
            position_sets["paper_long_short"],
            primary_mask,
            one_way_cost=0.0006,
            short_borrow_annual=0.03,
        )
        lf_daily, lf_pos = strategy_returns(
            simple_returns,
            position_sets["paper_long_flat"],
            primary_mask,
            one_way_cost=0.0006,
        )
        bh_daily, _ = strategy_returns(
            simple_returns,
            position_sets["buy_hold"],
            primary_mask,
            one_way_cost=0.0006,
        )
        state_for_return = np.empty_like(states)
        state_for_return[0] = states[0]
        state_for_return[1:] = states[:-1]
        for dt, ar, st, p_ls, r_ls, p_lf, r_lf, r_bh in zip(
            return_dates[primary_mask],
            simple_returns[primary_mask],
            state_for_return[primary_mask],
            ls_pos,
            ls_daily,
            lf_pos,
            lf_daily,
            bh_daily,
        ):
            writer.writerow(
                [
                    dt,
                    f"{ar:.10f}",
                    STATE_NAMES[int(st)],
                    f"{p_ls:.1f}",
                    f"{r_ls:.10f}",
                    f"{p_lf:.1f}",
                    f"{r_lf:.10f}",
                    f"{r_bh:.10f}",
                ]
            )

    print(md_path.read_text(encoding="utf-8"))
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
