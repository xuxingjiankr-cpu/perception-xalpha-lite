"""Causal exit-timing research on actual filled trades from the 60-day replay.

The study freezes entries and quantities to the existing ``entry_logic_v2`` replay,
then compares the current unified timing exit with three preregistered trailing/reversal
families. Emergency and hard-stop exits are excluded and can never be delayed. A trigger
seen on bar t fills at the bid on bar t+1; if no trigger fires, the rule falls back to
the current replay exit. The ex-post path high is used only to score exit quality.

The 2026-05-21..2026-06-18 test window has already been reused elsewhere in this
project. Therefore every result remains ``diagnostic_only`` and needs genuinely new
forward data before any live change.

STRICTLY OFFLINE. No broker calls, order submissions, live-config edits, or overlay
writes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import overfitting_guard as og
from run_etf_paper_trading_agent import ROOT, as_float


QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
DEFAULT_REPLAY_SUMMARY = (
    ROOT
    / "outputs"
    / "entry_logic_v2_ablation"
    / "v2_pullback_06pct_10min"
    / "entrylv2_v2_pullback_06pct_10min_summary.json"
)
OUT_DIR = ROOT / "outputs" / "exit_timing_actual_trades"
TRAIN_END = "2026-05-20"
ROUND_TRIP_COMMISSION = 0.0006
MIN_HOLD_BARS = 2
VOL_LOOKBACK_RETURNS = 6

# Fixed before this script's results are observed. This is a mechanism comparison,
# not a parameter search.
VARIANTS = (
    "current_baseline",
    "fixed_trail_0p6",
    "vol_trail_2sigma",
    "vol_trail_2sigma_confirm2",
    "stale_peak_reversal_30m",
)

LITERATURE = [
    {
        "title": "When Do Stop-Loss Rules Stop Losses?",
        "finding": "Stops are conditional tools: momentum can make them useful, while a random walk does not.",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338",
    },
    {
        "title": "Optimal Asset Liquidation Using Limit Order Book Information",
        "finding": "Adverse supply-demand imbalance defines a favorable sell region, but latency rapidly erodes its value.",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2113827",
    },
    {
        "title": "The Price Impact of Order Book Events",
        "finding": "Short-horizon price changes relate more robustly to order-flow imbalance than raw volume.",
        "url": "https://arxiv.org/abs/1011.6402",
    },
    {
        "title": "Volatility-Managed Portfolios",
        "finding": "Predictable realized volatility motivates adaptive rather than fixed risk reduction.",
        "url": "https://doi.org/10.1111/jofi.12513",
    },
]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def pair_filled_lots(order_lifecycle: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """FIFO-pair filled buys and sells, returning unified timing exits only."""

    queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    lots: list[dict[str, Any]] = []
    diagnostics = {
        "filled_buys": 0,
        "filled_sells": 0,
        "timing_lots": 0,
        "hard_exit_slices_excluded": 0,
        "unmatched_sell_quantity": 0,
    }
    filled = [
        row
        for row in order_lifecycle
        if str(row.get("status")) == "filled"
        and row.get("fill_time")
        and as_float(row.get("filled_qty"), 0.0) > 0
        and str(row.get("side")) in {"buy", "sell"}
    ]
    filled.sort(key=lambda row: parse_time(str(row["fill_time"])))

    for row in filled:
        code = str(row.get("stockCode", "")).zfill(6)
        qty = int(as_float(row.get("filled_qty"), 0.0))
        if row["side"] == "buy":
            diagnostics["filled_buys"] += 1
            queues[code].append(
                {
                    "remaining": qty,
                    "entry_time": str(row["fill_time"]),
                    "entry_price": as_float(row.get("fill_price"), 0.0),
                    "entry_reason": str(row.get("reason") or ""),
                }
            )
            continue

        diagnostics["filled_sells"] += 1
        remaining = qty
        while remaining > 0 and queues[code]:
            entry = queues[code][0]
            take = min(remaining, int(entry["remaining"]))
            reason = str(row.get("reason") or "")
            if reason == "unified_sell_score_exit":
                lots.append(
                    {
                        "stockCode": code,
                        "quantity": take,
                        "entry_time": entry["entry_time"],
                        "entry_price": entry["entry_price"],
                        "entry_reason": entry["entry_reason"],
                        "baseline_exit_time": str(row["fill_time"]),
                        "baseline_exit_price": as_float(row.get("fill_price"), 0.0),
                        "baseline_exit_reason": reason,
                    }
                )
                diagnostics["timing_lots"] += 1
            else:
                diagnostics["hard_exit_slices_excluded"] += 1
            entry["remaining"] -= take
            remaining -= take
            if entry["remaining"] <= 0:
                queues[code].popleft()
        diagnostics["unmatched_sell_quantity"] += remaining
    return lots, diagnostics


def load_quotes(codes: set[str], path: Path = QUOTES) -> dict[str, list[dict[str, Any]]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            code = str(row.get("stockCode", "")).zfill(6)
            if code not in codes or not row.get("quote_ok") or row.get("isSuspended"):
                continue
            price = as_float(row.get("currentPrice"), 0.0)
            if price <= 0 or not row.get("timestamp"):
                continue
            by_code[code].append(
                {
                    "time": parse_time(str(row["timestamp"])),
                    "price": price,
                    "bid": as_float(row.get("bidPrice1"), 0.0),
                    "spread": max(0.0, as_float(row.get("spread_pct"), 0.0008)),
                }
            )
    for rows in by_code.values():
        rows.sort(key=lambda row: row["time"])
    return by_code


def build_path(lot: dict[str, Any], quotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entry_time = parse_time(str(lot["entry_time"]))
    exit_time = parse_time(str(lot["baseline_exit_time"]))
    times = [row["time"] for row in quotes]
    lo = bisect.bisect_right(times, entry_time)
    hi = bisect.bisect_left(times, exit_time)
    path = [
        {
            "time": entry_time,
            "price": as_float(lot["entry_price"]),
            "bid": as_float(lot["entry_price"]),
            "spread": 0.0008,
        }
    ]
    path.extend(quotes[lo:hi])
    path.append(
        {
            "time": exit_time,
            "price": as_float(lot["baseline_exit_price"]),
            "bid": as_float(lot["baseline_exit_price"]),
            "spread": 0.0008,
        }
    )
    deduped: list[dict[str, Any]] = []
    for row in path:
        if deduped and row["time"] == deduped[-1]["time"]:
            deduped[-1] = row
        else:
            deduped.append(row)
    return deduped


def sell_fill_price(bar: dict[str, Any]) -> float:
    bid = as_float(bar.get("bid"), 0.0)
    if bid > 0:
        return bid
    price = as_float(bar.get("price"), 0.0)
    spread = max(0.0, as_float(bar.get("spread"), 0.0008))
    return price * (1.0 - spread / 2.0)


def _recent_sigma(prices: list[float], end: int) -> float | None:
    window = prices[max(0, end - VOL_LOOKBACK_RETURNS) : end + 1]
    returns = [
        math.log(window[i] / window[i - 1])
        for i in range(1, len(window))
        if window[i] > 0 and window[i - 1] > 0
    ]
    return statistics.stdev(returns) if len(returns) >= 3 else None


def simulate_exit(path: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """Return a causal exit: trigger on bar t, fill at the bid on bar t+1."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if len(path) < 2:
        raise ValueError("path requires entry and baseline exit")
    if variant == "current_baseline":
        return {
            "exit_index": len(path) - 1,
            "exit_price": as_float(path[-1]["price"]),
            "triggered": False,
            "reason": "current_baseline_exit",
        }

    prices = [as_float(row["price"]) for row in path]
    peak = prices[0]
    peak_index = 0
    for i in range(1, len(path) - 1):
        current = prices[i]
        if current > peak:
            peak, peak_index = current, i
        if i < MIN_HOLD_BARS or peak <= prices[0]:
            continue

        trigger = False
        reason = ""
        if variant == "fixed_trail_0p6":
            trigger = current <= peak * (1.0 - 0.006)
            reason = "fixed_peak_trail"
        elif variant in {"vol_trail_2sigma", "vol_trail_2sigma_confirm2"}:
            sigma = _recent_sigma(prices, i)
            if sigma is None:
                continue
            trail = min(0.012, max(0.003, 2.0 * sigma))
            trigger = current <= peak * (1.0 - trail)
            if variant.endswith("confirm2"):
                trigger = trigger and i >= 2 and prices[i] < prices[i - 1] < prices[i - 2]
            reason = f"volatility_trail_{trail:.6f}"
        elif variant == "stale_peak_reversal_30m":
            rolling = statistics.mean(prices[max(0, i - 2) : i + 1])
            trigger = (
                i - peak_index >= 6
                and current <= peak * (1.0 - 0.003)
                and i >= 2
                and current < prices[i - 2]
                and current <= rolling
            )
            reason = "stale_peak_30m_and_reversal"

        if trigger:
            fill_index = i + 1
            return {
                "exit_index": fill_index,
                "exit_price": sell_fill_price(path[fill_index]),
                "triggered": True,
                "reason": reason,
            }

    return {
        "exit_index": len(path) - 1,
        "exit_price": as_float(path[-1]["price"]),
        "triggered": False,
        "reason": "fallback_to_current_baseline_exit",
    }


def evaluate_lot(lot: dict[str, Any], path: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    entry = as_float(lot["entry_price"])
    full_prices = [as_float(row["price"]) for row in path]
    full_low, full_high = min(full_prices), max(full_prices)
    out: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        result = simulate_exit(path, variant)
        idx = int(result["exit_index"])
        exit_price = as_float(result["exit_price"])
        causal_peak = max(full_prices[: idx + 1])
        percentile = (
            (exit_price - full_low) / (full_high - full_low) * 100.0
            if full_high > full_low
            else 50.0
        )
        result.update(
            {
                "net_return_pct": (exit_price / entry - 1.0 - ROUND_TRIP_COMMISSION) * 100.0,
                "exit_percentile": percentile,
                "shortfall_to_full_high_bps": (exit_price / full_high - 1.0) * 10_000.0,
                "giveback_from_causal_peak_bps": (exit_price / causal_peak - 1.0) * 10_000.0,
                "holding_bars": idx,
                "exit_time": path[idx]["time"].isoformat(),
            }
        )
        out[variant] = result
    return out


def trade_stats(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    rows = [row["variants"][variant] for row in records]
    if not rows:
        return {"trades": 0}
    rets = [as_float(row["net_return_pct"]) for row in rows]
    ordered = sorted(rets)
    p5 = ordered[max(0, math.ceil(len(ordered) * 0.05) - 1)]
    return {
        "trades": len(rows),
        "mean_net_return_pct": round(statistics.mean(rets), 4),
        "median_net_return_pct": round(statistics.median(rets), 4),
        "win_rate": round(sum(value > 0 for value in rets) / len(rets), 4),
        "p5_net_return_pct": round(p5, 4),
        "mean_exit_percentile": round(
            statistics.mean(as_float(row["exit_percentile"]) for row in rows), 2
        ),
        "mean_shortfall_to_full_high_bps": round(
            statistics.mean(as_float(row["shortfall_to_full_high_bps"]) for row in rows), 2
        ),
        "mean_giveback_from_causal_peak_bps": round(
            statistics.mean(as_float(row["giveback_from_causal_peak_bps"]) for row in rows), 2
        ),
        "mean_holding_bars": round(statistics.mean(as_float(row["holding_bars"]) for row in rows), 2),
        "trigger_rate": round(sum(bool(row["triggered"]) for row in rows) / len(rows), 4),
    }


def daily_metric(records: list[dict[str, Any]], variant: str, key: str) -> dict[str, float]:
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_day[str(row["baseline_exit_time"])[:10]].append(as_float(row["variants"][variant][key]))
    return {day: statistics.mean(values) for day, values in sorted(by_day.items())}


def paired_day_test(
    records: list[dict[str, Any]], variant: str, key: str, *, higher_is_better: bool = True
) -> dict[str, Any]:
    base = daily_metric(records, "current_baseline", key)
    candidate = daily_metric(records, variant, key)
    days = sorted(set(base) & set(candidate))
    sign = 1.0 if higher_is_better else -1.0
    diffs = [(candidate[day] - base[day]) * sign for day in days]
    if len(diffs) < 2:
        return {"days": len(diffs), "verdict": "insufficient"}
    mean_diff = statistics.mean(diffs)
    std = statistics.stdev(diffs)
    t_stat = mean_diff / std * math.sqrt(len(diffs)) if std > 0 else None
    return {
        "days": len(diffs),
        "mean_improvement": round(mean_diff, 4),
        "day_clustered_t": round(t_stat, 3) if t_stat is not None else None,
        "days_improved": sum(value > 0 for value in diffs),
    }


def _split(records: list[dict[str, Any]], which: str) -> list[dict[str, Any]]:
    if which == "train":
        return [row for row in records if str(row["baseline_exit_time"])[:10] <= TRAIN_END]
    if which == "test":
        return [row for row in records if str(row["baseline_exit_time"])[:10] > TRAIN_END]
    return records


def build_result(
    records: list[dict[str, Any]], diagnostics: dict[str, int], summary_path: Path
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for window in ("full", "train", "test"):
        sample = _split(records, window)
        windows[window] = {
            variant: {
                "stats": trade_stats(sample, variant),
                "vs_baseline_net_return": paired_day_test(sample, variant, "net_return_pct"),
                "vs_baseline_exit_percentile": paired_day_test(sample, variant, "exit_percentile"),
                "vs_baseline_high_shortfall": paired_day_test(
                    sample, variant, "shortfall_to_full_high_bps"
                ),
            }
            for variant in VARIANTS
        }

    train_candidates = [variant for variant in VARIANTS if variant != "current_baseline"]
    selected = max(
        train_candidates,
        key=lambda name: windows["train"][name]["stats"].get("mean_net_return_pct", float("-inf")),
    )
    overall_train_winner = max(
        VARIANTS,
        key=lambda name: windows["train"][name]["stats"].get("mean_net_return_pct", float("-inf")),
    )
    all_days = sorted(daily_metric(records, "current_baseline", "net_return_pct"))
    daily_by_variant = {
        variant: daily_metric(records, variant, "net_return_pct") for variant in VARIANTS
    }
    matrix = [[daily_by_variant[variant][day] for day in all_days] for variant in VARIANTS]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8)
    selected_daily = matrix[VARIANTS.index(selected)]
    daily_sharpe = (
        statistics.mean(selected_daily) / statistics.pstdev(selected_daily)
        if len(selected_daily) > 1 and statistics.pstdev(selected_daily) > 0
        else 0.0
    )
    dsr = og.deflated_significance_note(
        n_trials=len(VARIANTS) - 1, observed_sharpe=daily_sharpe, n_obs=len(selected_daily)
    )

    test_cmp = windows["test"][selected]
    test_stats = test_cmp["stats"]
    base_test = windows["test"]["current_baseline"]["stats"]
    gates = {
        "selected_on_train_not_test": True,
        "test_net_return_above_baseline": test_stats.get("mean_net_return_pct", -math.inf)
        > base_test.get("mean_net_return_pct", math.inf),
        "test_exit_percentile_above_baseline": test_stats.get("mean_exit_percentile", -math.inf)
        > base_test.get("mean_exit_percentile", math.inf),
        "test_high_shortfall_less_negative": test_stats.get(
            "mean_shortfall_to_full_high_bps", -math.inf
        )
        > base_test.get("mean_shortfall_to_full_high_bps", math.inf),
        "test_day_clustered_t_at_least_1p96": as_float(
            test_cmp["vs_baseline_net_return"].get("day_clustered_t"), -math.inf
        )
        >= 1.96,
        "pbo_below_0p25": pbo.get("pbo") is not None and as_float(pbo.get("pbo")) < 0.25,
        "deflated_sharpe_exceeds_noise": dsr.get("flag") == "exceeds_noise_max",
        "clean_unseen_forward_window": False,
    }
    return {
        "research_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "diagnostic_only",
        "edge_validated": False,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "paper_trading_only": True,
        "order_submit_calls_made": False,
        "live_config_modified": False,
        "source_quotes": str(QUOTES),
        "source_replay_summary": str(summary_path),
        "train_test_split": f"train<= {TRAIN_END} < test",
        "sample": {
            "paired_timing_exit_lots": len(records),
            "full_exit_days": len(all_days),
            "train_exit_days": len(
                daily_metric(_split(records, "train"), "current_baseline", "net_return_pct")
            ),
            "test_exit_days": len(
                daily_metric(_split(records, "test"), "current_baseline", "net_return_pct")
            ),
            **diagnostics,
        },
        "methodology": {
            "entry_population": "actual filled buys from the entry_logic_v2 60-day replay",
            "exit_population": "FIFO-paired unified_sell_score_exit fills only",
            "hard_exits": "excluded; emergency/kill/EOD exits are never delayed",
            "execution": "signal on bar t; alternative sell at next bar bid; fallback to current replay exit",
            "cost": f"actual/next-bid fill plus {ROUND_TRIP_COMMISSION:.4%} round-trip commission",
            "ex_post_high_usage": "evaluation metric only; never visible to a trigger",
            "variants_preregistered": list(VARIANTS),
        },
        "windows": windows,
        "selection": {
            "rule": "best non-baseline candidate chosen by TRAIN mean net return; baseline remains eligible as the benchmark winner",
            "selected_nonbaseline_candidate": selected,
            "overall_train_winner": overall_train_winner,
            "pbo": pbo,
            "pbo_read": (
                "low PBO reflects the current baseline consistently winning across splits; "
                "it is not evidence that the non-baseline candidate has edge"
                if overall_train_winner == "current_baseline"
                else "low PBO indicates the overall train winner tends to persist"
            ),
            "deflated_sharpe_note": dsr,
            "promotion_gates": gates,
            "all_numeric_gates_passed": all(
                value for key, value in gates.items() if key != "clean_unseen_forward_window"
            ),
            "verdict": "not_promotable_reused_oos_requires_new_forward_data",
        },
        "limitations": [
            "The 2026-05-21..2026-06-18 test window has already informed prior project research.",
            "Entry events come from a replay configuration selected using the same 60-day history.",
            "Five-minute snapshots cannot identify the true intrabar high or guarantee passive fills.",
            "Yahoo history has top-of-book only, so order-book imbalance liquidation is not backtested.",
            "Earlier exits are evaluated holding-by-holding; portfolio slots and later entries are not resimulated.",
        ],
        "literature": LITERATURE,
        "records": records,
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Actual-Trade Exit Timing Research — 60-day causal replay",
        "",
        f"Status: **{result['status']}** | live ready: **{str(result['live_ready']).lower()}**",
        "",
        f"Paired timing-exit lots: {result['sample']['paired_timing_exit_lots']} | "
        f"train/test exit days: {result['sample']['train_exit_days']}/{result['sample']['test_exit_days']}",
        "",
        "## OOS comparison",
        "",
        "| variant | trades | mean net | win | exit percentile | high shortfall | trigger | day Δ net / t |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for variant in VARIANTS:
        node = result["windows"]["test"][variant]
        stats = node["stats"]
        diff = node["vs_baseline_net_return"]
        lines.append(
            f"| {variant} | {stats.get('trades', 0)} | "
            f"{stats.get('mean_net_return_pct', 0):+.4f}% | {stats.get('win_rate', 0):.2%} | "
            f"{stats.get('mean_exit_percentile', 0):.2f} | "
            f"{stats.get('mean_shortfall_to_full_high_bps', 0):+.1f} bp | "
            f"{stats.get('trigger_rate', 0):.2%} | "
            f"{diff.get('mean_improvement', 0):+.4f}% / {diff.get('day_clustered_t')} |"
        )
    selection = result["selection"]
    lines += [
        "",
        "## Decision",
        "",
        f"- Overall TRAIN winner: **{selection['overall_train_winner']}**.",
        f"- Best non-baseline candidate on TRAIN: **{selection['selected_nonbaseline_candidate']}**.",
        f"- PBO: `{selection['pbo'].get('pbo')}`.",
        f"- PBO read: {selection['pbo_read']}.",
        f"- Deflated-Sharpe screen: `{selection['deflated_sharpe_note'].get('flag')}`.",
        f"- Verdict: **{selection['verdict']}**.",
        "- No live config, overlay, broker state, or execution lock was changed.",
        "",
        "## Interpretation",
        "",
        "Selling at the exact high is not causal. A defensible rule must improve the exit-price "
        "distribution and costed return on unseen days; the ex-post high is a score only.",
        "",
        "Historical L2 imbalance is unavailable in Yahoo. The most directly supported "
        "microstructure hypothesis—sell faster when supply-demand imbalance turns adverse—must "
        "be evaluated prospectively with the existing L2 collector.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in result["limitations"])
    lines += ["", "## Sources", ""]
    lines.extend(f"- [{item['title']}]({item['url']}): {item['finding']}" for item in result["literature"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Causal exit timing research on actual replay fills.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_REPLAY_SUMMARY)
    parser.add_argument("--quotes", type=Path, default=QUOTES)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    lots, diagnostics = pair_filled_lots(list(summary.get("order_lifecycle") or []))
    quote_map = load_quotes({lot["stockCode"] for lot in lots}, args.quotes)
    records: list[dict[str, Any]] = []
    skipped_short_path = 0
    for lot in lots:
        path = build_path(lot, quote_map.get(lot["stockCode"], []))
        if len(path) < 3:
            skipped_short_path += 1
            continue
        records.append({**lot, "variants": evaluate_lot(lot, path)})
    diagnostics["short_paths_skipped"] = skipped_short_path
    if not records:
        raise RuntimeError("no usable paired timing-exit paths")

    result = build_result(records, diagnostics, args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "actual_trade_exit_timing_60d.json"
    md_path = args.output_dir / "actual_trade_exit_timing_60d.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(report_markdown(result), encoding="utf-8")

    selected = result["selection"]["selected_nonbaseline_candidate"]
    oos = result["windows"]["test"]
    print(
        f"paired lots: {len(records)}; train/test days: "
        f"{result['sample']['train_exit_days']}/{result['sample']['test_exit_days']}"
    )
    for variant in VARIANTS:
        stats = oos[variant]["stats"]
        diff = oos[variant]["vs_baseline_net_return"]
        print(
            f"{variant:28s} OOS net={stats['mean_net_return_pct']:+.4f}% "
            f"exit_pct={stats['mean_exit_percentile']:.2f} "
            f"shortfall={stats['mean_shortfall_to_full_high_bps']:+.1f}bp "
            f"day_diff={diff.get('mean_improvement', 0):+.4f}% "
            f"t={diff.get('day_clustered_t')}"
        )
    print(
        f"overall_train_winner={result['selection']['overall_train_winner']}; "
        f"best_nonbaseline_on_train={selected}; pbo={result['selection']['pbo'].get('pbo')}; "
        f"verdict={result['selection']['verdict']}"
    )
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
