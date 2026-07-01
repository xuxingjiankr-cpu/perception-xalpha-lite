"""Forward L2 audit of block versus short-horizon sliced sell execution.

The experiment fixes a CNY 100,000 sell notional, rounds to 100-share lots, and
uses the displayed five bid levels to compute executable VWAP. It compares:

* ``block_now``: consume the current five-level bid book immediately;
* ``twap_4x_15s``: four equal slices at now/+5s/+10s/+15s;
* ``frontload_50``: 50% now, then three equal residual slices;
* ``conditional_urgency`` (primary): block immediately when idiosyncratic OBI <=
  -0.20 and microprice displacement <= -1 bp, otherwise use four-slice TWAP.

Only paths where every policy can be fully executed in displayed depth are compared,
so all results use an identical paired sample. This excludes thin books rather than
inventing invisible liquidity. Opportunities are sampled once per code per minute and
evidence is clustered by complete trading day. Fewer than 20 days cannot promote.

STRICTLY OFFLINE / SHADOW. No broker calls, orders, live config, or overlays.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, as_float
from research_l2_exit_timing import (
    DEPTH_DIR,
    EVENT_COOLDOWN_SECONDS,
    MIN_COMPLETE_DAYS,
    MIN_COMPLETE_ROWS,
    MIN_COMPLETE_SPAN_SECONDS,
    iter_snapshot_groups,
    parse_time,
)


OUT_DIR = ROOT / "outputs" / "l2_sell_slicing"
ORDER_BLOTTER = ROOT / "outputs" / "t0_intraday_agent" / "t0_order_blotter.csv"
TARGET_NOTIONAL = 100_000.0
WAIT_POLLS = 3
LOT_SIZE = 100
PRIMARY_POLICY = "conditional_urgency"
POLICIES = ("block_now", "twap_4x_15s", "frontload_50", "conditional_urgency")

SOURCES = [
    {
        "title": "Optimal Execution in a Limit Order Book and an Associated Microstructure Market Impact Model",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2610808",
        "use": "short-horizon execution costs depend on measured limit-order-book state, not size alone",
    },
    {
        "title": "Optimal Limit-versus-Market Order Slicing under a VWAP Benchmark",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2318890",
        "use": "slicing trades off immediate execution cost against schedule and non-execution risk",
    },
    {
        "title": "Optimal Execution Strategies in Limit Order Books with General Shape Functions",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1510104",
        "use": "book shape and resilience determine the impact of block market orders",
    },
]


@dataclass
class Accumulator:
    n: int = 0
    sum_improvement: float = 0.0
    sum_improvement2: float = 0.0
    improved: int = 0
    sum_block_slippage: float = 0.0

    def add(self, improvement_bps: float, block_slippage_bps: float) -> None:
        self.n += 1
        self.sum_improvement += improvement_bps
        self.sum_improvement2 += improvement_bps * improvement_bps
        self.improved += int(improvement_bps > 0)
        self.sum_block_slippage += block_slippage_bps

    def summary(self) -> dict[str, Any]:
        if not self.n:
            return {"opportunities": 0}
        mean = self.sum_improvement / self.n
        variance = max(0.0, self.sum_improvement2 / self.n - mean * mean)
        std = math.sqrt(variance)
        return {
            "opportunities": self.n,
            "mean_improvement_bps": round(mean, 4),
            "pooled_t_pseudo_only": round(mean / std * math.sqrt(self.n), 3) if std > 0 else None,
            "improvement_rate": round(self.improved / self.n, 4),
            "mean_block_slippage_bps": round(self.sum_block_slippage / self.n, 4),
        }


def sweep_sell_vwap(book: dict[str, Any], quantity: int) -> float | None:
    """Consume displayed bid levels. Return None when visible depth is insufficient."""
    if quantity <= 0:
        return None
    prices = list(book.get("bid_prices") or [])
    volumes = list(book.get("bid_volumes") or [])
    remaining = quantity
    proceeds = 0.0
    for raw_price, raw_volume in zip(prices, volumes):
        price = as_float(raw_price, 0.0)
        volume = int(as_float(raw_volume, 0.0))
        if price <= 0 or volume <= 0:
            continue
        take = min(remaining, volume)
        proceeds += take * price
        remaining -= take
        if remaining <= 0:
            return proceeds / quantity
    return None


def split_lots(total_quantity: int, weights: tuple[float, ...]) -> list[int]:
    """Allocate whole 100-share lots exactly across fixed policy weights."""
    total_lots = total_quantity // LOT_SIZE
    if total_lots < len(weights):
        return []
    raw = [total_lots * weight for weight in weights]
    lots = [int(value) for value in raw]
    remainder = total_lots - sum(lots)
    order = sorted(range(len(weights)), key=lambda idx: raw[idx] - lots[idx], reverse=True)
    for idx in order[:remainder]:
        lots[idx] += 1
    return [value * LOT_SIZE for value in lots]


def is_adverse(book: dict[str, Any]) -> bool:
    return (
        as_float(book.get("idiosyncratic_obi"), 0.0) <= -0.20
        and as_float(book.get("micro_dev_bps"), 0.0) <= -1.0
    )


def policy_vwap(path: list[dict[str, Any]], quantity: int, policy: str) -> float | None:
    if policy not in POLICIES or len(path) != WAIT_POLLS + 1:
        raise ValueError("unknown policy or invalid path")
    if policy == "block_now" or (policy == "conditional_urgency" and is_adverse(path[0])):
        return sweep_sell_vwap(path[0], quantity)
    if policy in {"twap_4x_15s", "conditional_urgency"}:
        quantities = split_lots(quantity, (0.25, 0.25, 0.25, 0.25))
    elif policy == "frontload_50":
        quantities = split_lots(quantity, (0.50, 1 / 6, 1 / 6, 1 / 6))
    else:  # pragma: no cover
        raise AssertionError(policy)
    if not quantities or sum(quantities) != quantity:
        return None
    proceeds = 0.0
    for book, slice_qty in zip(path, quantities):
        fill = sweep_sell_vwap(book, slice_qty)
        if fill is None:
            return None
        proceeds += fill * slice_qty
    return proceeds / quantity


def load_actual_sell_signals(path: Path = ORDER_BLOTTER) -> dict[str, list[dict[str, Any]]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_date
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (
                str(row.get("direction")) != "sell"
                or str(row.get("reason")) != "unified_sell_score_exit"
                or str(row.get("submit_ok")).lower() != "true"
                or not row.get("timestamp")
            ):
                continue
            timestamp = parse_time(str(row["timestamp"])).astimezone(ZoneInfo("Asia/Shanghai"))
            by_date[timestamp.date().isoformat()].append(
                {
                    "timestamp": timestamp,
                    "stockCode": str(row.get("stockCode") or "").zfill(6),
                    "quantity": int(as_float(row.get("quantity"), 0.0)),
                    "broker_order_id": str(row.get("broker_order_id") or ""),
                }
            )
    return by_date


def evaluate_actual_signals(
    signals: list[dict[str, Any]], books_by_code: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for signal in signals:
        serialized_signal = {**signal, "timestamp": signal["timestamp"].isoformat()}
        books = books_by_code.get(signal["stockCode"], [])
        if not books:
            records.append(
                {**serialized_signal, "path_matched": False, "paired": False, "reason": "no_l2_book"}
            )
            continue
        times = [book["time"] for book in books]
        start_index = bisect.bisect_right(times, signal["timestamp"]) - 1
        if start_index < 0:
            records.append(
                {
                    **serialized_signal,
                    "path_matched": False,
                    "paired": False,
                    "reason": "no_prior_l2_book",
                }
            )
            continue
        if start_index + WAIT_POLLS >= len(books):
            records.append(
                {
                    **serialized_signal,
                    "path_matched": False,
                    "paired": False,
                    "reason": "insufficient_future_l2_path",
                }
            )
            continue
        execution_path = books[start_index : start_index + WAIT_POLLS + 1]
        age = (signal["timestamp"] - execution_path[0]["time"]).total_seconds()
        elapsed = (execution_path[-1]["time"] - execution_path[0]["time"]).total_seconds()
        if age < 0 or age > 10.0:
            records.append(
                {
                    **serialized_signal,
                    "path_matched": False,
                    "paired": False,
                    "reason": "stale_initial_l2_book",
                    "book_age_seconds": round(age, 3),
                }
            )
            continue
        if elapsed <= 0 or elapsed > WAIT_POLLS * 10.0:
            records.append(
                {
                    **serialized_signal,
                    "path_matched": False,
                    "paired": False,
                    "reason": "invalid_l2_path_elapsed",
                    "path_elapsed_seconds": round(elapsed, 3),
                }
            )
            continue
        fills = {
            policy: policy_vwap(execution_path, signal["quantity"], policy)
            for policy in POLICIES
        }
        if fills["block_now"] is None:
            records.append(
                {
                    **serialized_signal,
                    "path_matched": True,
                    "paired": False,
                    "reason": "block_exceeds_visible_depth",
                }
            )
            continue
        block = as_float(fills["block_now"])
        records.append(
            {
                **serialized_signal,
                "path_matched": True,
                "paired": all(value is not None for value in fills.values()),
                "book_age_seconds": round(age, 3),
                "fills": fills,
                "improvement_bps": {
                    policy: round((as_float(fill) / block - 1.0) * 10_000.0, 4)
                    if fill is not None
                    else None
                    for policy, fill in fills.items()
                },
            }
        )
    return records


def process_day(path: Path, actual_signals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    actual_signals = actual_signals or []
    actual_codes = {signal["stockCode"] for signal in actual_signals}
    actual_books: dict[str, list[dict[str, Any]]] = defaultdict(list)
    history: dict[str, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=WAIT_POLLS + 1)
    )
    last_opportunity: dict[str, datetime] = {}
    accum: dict[str, Accumulator] = defaultdict(Accumulator)
    candidate_paths = 0
    paired_paths = 0
    row_count = 0
    snapshot_count = 0
    first_time: datetime | None = None
    last_time: datetime | None = None

    for group in iter_snapshot_groups(path):
        eligible = [
            row
            for row in group
            if row.get("is_fresh")
            and str(row.get("asset_class") or "").lower() != "bond"
            and as_float(row.get("midpoint"), 0.0) > 0
            and row.get("obi") is not None
            and row.get("bid_prices")
            and row.get("bid_volumes")
        ]
        if not eligible:
            continue
        snapshot_count += 1
        row_count += len(eligible)
        common_obi = statistics.median(as_float(row["obi"]) for row in eligible)

        for row in eligible:
            code = str(row.get("code") or "").zfill(6)
            event_time = parse_time(str(row.get("source_quote_time") or row["collected_at"]))
            first_time = event_time if first_time is None or event_time < first_time else first_time
            last_time = event_time if last_time is None or event_time > last_time else last_time
            previous = last_opportunity.get(code)
            opportunity = previous is None or (
                event_time - previous
            ).total_seconds() >= EVENT_COOLDOWN_SECONDS
            if opportunity:
                last_opportunity[code] = event_time
            book = {
                "time": event_time,
                "midpoint": as_float(row["midpoint"]),
                "bid1": as_float(row.get("bid1"), 0.0),
                "bid_prices": list(row.get("bid_prices") or []),
                "bid_volumes": list(row.get("bid_volumes") or []),
                "idiosyncratic_obi": as_float(row["obi"]) - common_obi,
                "micro_dev_bps": as_float(row.get("micro_dev_bps"), 0.0),
                "opportunity": opportunity,
            }
            if code in actual_codes:
                actual_books[code].append(book)
            books = history[code]
            books.append(book)
            if len(books) < WAIT_POLLS + 1:
                continue
            execution_path = list(books)
            start = execution_path[0]
            elapsed = (execution_path[-1]["time"] - start["time"]).total_seconds()
            if not start["opportunity"] or elapsed <= 0 or elapsed > WAIT_POLLS * 10.0:
                continue
            candidate_paths += 1
            quantity = int(TARGET_NOTIONAL / start["midpoint"] // LOT_SIZE) * LOT_SIZE
            if quantity < LOT_SIZE:
                continue
            fills = {
                policy: policy_vwap(execution_path, quantity, policy)
                for policy in POLICIES
            }
            # Identical paired sample: skip rather than invent liquidity beyond level 5.
            if any(value is None for value in fills.values()):
                continue
            paired_paths += 1
            block = as_float(fills["block_now"])
            bid1 = as_float(start["bid1"])
            block_slippage = (block / bid1 - 1.0) * 10_000.0 if bid1 > 0 else 0.0
            for policy, fill in fills.items():
                improvement = (as_float(fill) / block - 1.0) * 10_000.0
                accum[policy].add(improvement, block_slippage)

    span = (last_time - first_time).total_seconds() if first_time and last_time else 0.0
    complete = row_count >= MIN_COMPLETE_ROWS and span >= MIN_COMPLETE_SPAN_SECONDS
    return {
        "date": path.stem.removeprefix("depth_"),
        "path": str(path),
        "rows": row_count,
        "snapshots": snapshot_count,
        "span_seconds": round(span, 1),
        "complete_session": complete,
        "candidate_paths": candidate_paths,
        "paired_depth_paths": paired_paths,
        "paired_depth_coverage": round(paired_paths / candidate_paths, 4)
        if candidate_paths
        else 0.0,
        "policies": {policy: accum[policy].summary() for policy in POLICIES},
        "actual_sell_signals": evaluate_actual_signals(actual_signals, actual_books),
    }


def combine_days(days: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for policy in POLICIES:
        daily = [
            as_float(day["policies"][policy].get("mean_improvement_bps"))
            for day in days
            if day["policies"][policy].get("opportunities", 0) > 0
        ]
        opportunities = sum(
            int(day["policies"][policy].get("opportunities", 0)) for day in days
        )
        if not daily:
            combined[policy] = {"opportunities": 0, "independent_days": 0}
            continue
        mean = statistics.mean(daily)
        std = statistics.stdev(daily) if len(daily) > 1 else 0.0
        combined[policy] = {
            "opportunities": opportunities,
            "independent_days": len(daily),
            "mean_daily_improvement_bps": round(mean, 4),
            "day_clustered_t": round(mean / std * math.sqrt(len(daily)), 3)
            if std > 0
            else None,
            "positive_days": sum(value > 0 for value in daily),
            "daily_means_bps": [round(value, 4) for value in daily],
        }
    return combined


def summarize_actual_signals(days: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for day in days for record in day.get("actual_sell_signals", [])]
    matched = [record for record in records if record.get("path_matched")]
    paired = [record for record in records if record.get("paired")]
    rejected_reasons: dict[str, int] = defaultdict(int)
    for record in records:
        if not record.get("paired"):
            rejected_reasons[str(record.get("reason") or "unpaired_policy")] += 1
    return {
        "eligible_signals": len(records),
        "path_matched_signals": len(matched),
        "paired_signals": len(paired),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "mean_improvement_bps": {
            policy: round(
                statistics.mean(record["improvement_bps"][policy] for record in paired), 4
            )
            if paired
            else None
            for policy in POLICIES
        },
        "records": records,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# L2 sell slicing — block versus 15-second TWAP",
        "",
        f"Status: **{result['status']}** | complete days: "
        f"{result['complete_days']}/{MIN_COMPLETE_DAYS} required",
        "",
        f"Target notional: CNY {TARGET_NOTIONAL:,.0f}; identical paired sample with full visible-depth execution.",
        "",
        "| policy | paired paths | improvement vs block | day t | positive days |",
        "|---|--:|--:|--:|--:|",
    ]
    for policy in POLICIES:
        row = result["combined"][policy]
        lines.append(
            f"| {policy} | {row.get('opportunities', 0)} | "
            f"{row.get('mean_daily_improvement_bps', 0):+.4f} bp | "
            f"{row.get('day_clustered_t')} | {row.get('positive_days', 0)} |"
        )
    coverage = [day["paired_depth_coverage"] for day in result["days"] if day["complete_session"]]
    primary = result["combined"][PRIMARY_POLICY]
    actual = result["actual_sell_signal_check"]
    lines += [
        "",
        "## Decision",
        "",
        f"- Primary policy: **{PRIMARY_POLICY}**.",
        f"- Primary improvement: **{primary.get('mean_daily_improvement_bps', 0):+.4f} bp**.",
        f"- Mean paired visible-depth coverage: **{statistics.mean(coverage):.2%}**."
        if coverage
        else "- No complete-day coverage.",
        f"- Statistical readiness: **{result['readiness']}**.",
        f"- Edge validated: **{str(result['edge_validated']).lower()}**.",
        "- No live execution behavior was changed.",
        "",
        "## Actual strategy sell-signal overlap",
        "",
        f"- Eligible / L2-path matched / fully paired signals: "
        f"**{actual['eligible_signals']} / {actual['path_matched_signals']} / "
        f"{actual['paired_signals']}**.",
        f"- Rejected reasons: **{actual['rejected_reasons']}**.",
        f"- Conditional-urgency mean improvement: "
        f"**{actual['mean_improvement_bps'].get(PRIMARY_POLICY)} bp**.",
        "- This is a consistency check only; a handful of orders has no statistical power.",
        "",
        "## Limitations",
        "",
        "- Five displayed levels omit hidden liquidity and any impact beyond level 5.",
        "- Reusing displayed depth does not model our own order depleting future liquidity.",
        "- Generic sampled opportunities are not the strategy's actual sell decisions.",
        "- Only three complete days are available; pooled paths are not independent evidence.",
        "- A live implementation would require broker-confirmed partial-fill handling and the existing shared lock.",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- [{item['title']}]({item['url']}): {item['use']}." for item in SOURCES)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="L2 block-versus-sliced sell execution audit.")
    parser.add_argument("--depth-dir", type=Path, default=DEPTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--order-blotter", type=Path, default=ORDER_BLOTTER)
    args = parser.parse_args()

    signals_by_date = load_actual_sell_signals(args.order_blotter)
    all_days = []
    for path in sorted(args.depth_dir.glob("depth_*.jsonl")):
        print(f"processing {path.name} ...", flush=True)
        trade_date = path.stem.removeprefix("depth_")
        day = process_day(path, signals_by_date.get(trade_date, []))
        all_days.append(day)
        print(
            f"  rows={day['rows']} candidates={day['candidate_paths']} "
            f"paired={day['paired_depth_paths']} coverage={day['paired_depth_coverage']:.2%} "
            f"complete={day['complete_session']}",
            flush=True,
        )
    complete = [day for day in all_days if day["complete_session"]]
    combined = combine_days(complete)
    actual_check = summarize_actual_signals(complete)
    readiness = (
        "ready_for_statistical_testing"
        if len(complete) >= MIN_COMPLETE_DAYS
        else "insufficient_forward_days"
    )
    result = {
        "research_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "diagnostic_only",
        "readiness": readiness,
        "edge_validated": False,
        "live_ready": False,
        "formal_strategy_allowed": False,
        "paper_trading_only": True,
        "order_submit_calls_made": False,
        "live_config_modified": False,
        "target_notional": TARGET_NOTIONAL,
        "wait_polls": WAIT_POLLS,
        "primary_policy": PRIMARY_POLICY,
        "complete_days": len(complete),
        "minimum_complete_days": MIN_COMPLETE_DAYS,
        "days": all_days,
        "combined": combined,
        "actual_sell_signal_check": actual_check,
        "sources": SOURCES,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "l2_sell_slicing.json"
    md_path = args.output_dir / "l2_sell_slicing.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(result), encoding="utf-8")
    print(render(result))
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
