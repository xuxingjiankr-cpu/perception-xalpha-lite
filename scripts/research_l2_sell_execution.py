"""Forward shadow audit of sell order placement using real five-level ETF books.

This asks a different question from sell-signal timing: once a discretionary sell
decision exists, should execution cross the bid immediately or briefly provide
liquidity?

Preregistered policies (15-second / three-poll timeout):

* ``aggressive_bid``: sell immediately at the current best bid.
* ``ask_then_cross``: post at current ask; conservatively count a fill only if a later
  observed best bid reaches that limit, otherwise cross the bid at timeout.
* ``midpoint_then_cross``: same, with a midpoint sell limit.
* ``half_midpoint_then_cross``: half at current bid and half with midpoint-then-cross.
* ``conditional_midpoint`` (primary): cross immediately when idiosyncratic OBI <=
  -0.20 and microprice displacement <= -1 bp; otherwise midpoint-then-cross.

The conservative fill proxy deliberately ignores trades at the ask that do not move
the displayed bid. It therefore understates passive fill probability. It still cannot
observe queue position or partial fills, so every result remains diagnostic.

Opportunities are sampled once per code per 60 seconds. Evidence is clustered by
complete trading day and refuses promotion below 20 days.

STRICTLY OFFLINE / SHADOW. No broker calls, orders, live config, or overlays.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


OUT_DIR = ROOT / "outputs" / "l2_sell_execution"
WAIT_POLLS = 3
PRIMARY_POLICY = "conditional_midpoint"
POLICIES = (
    "aggressive_bid",
    "ask_then_cross",
    "midpoint_then_cross",
    "half_midpoint_then_cross",
    "conditional_midpoint",
)

SOURCES = [
    {
        "title": "Optimal Order Placement in Limit Order Markets",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2155218",
        "use": "limit-versus-market placement depends on order flow, queue sizes, fees and execution risk",
    },
    {
        "title": "Optimal Placement in a Limit Order Book",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2318220",
        "use": "optimal placement is threshold-like and can choose among market and displayed limit prices",
    },
    {
        "title": "The Negative Drift of a Limit Order Fill",
        "url": "https://arxiv.org/abs/2407.16527",
        "use": "passive fills coincide with adverse price moves, so naive spread capture is overstated",
    },
]


@dataclass
class PolicyAccumulator:
    n: int = 0
    sum_improvement: float = 0.0
    sum_improvement2: float = 0.0
    improved: int = 0
    passive_attempts: int = 0
    passive_fills: int = 0
    timeouts: int = 0

    def add(self, outcome: dict[str, Any]) -> None:
        improvement = as_float(outcome["improvement_bps"])
        self.n += 1
        self.sum_improvement += improvement
        self.sum_improvement2 += improvement * improvement
        self.improved += int(improvement > 0)
        self.passive_attempts += int(outcome.get("passive_attempted", False))
        self.passive_fills += int(outcome.get("passive_filled", False))
        self.timeouts += int(outcome.get("timed_out", False))

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
            "passive_attempts": self.passive_attempts,
            "passive_fills": self.passive_fills,
            "timeouts": self.timeouts,
            "conservative_passive_fill_rate": round(
                self.passive_fills / self.passive_attempts, 4
            )
            if self.passive_attempts
            else None,
            "timeout_rate": round(self.timeouts / self.passive_attempts, 4)
            if self.passive_attempts
            else None,
        }


def is_adverse(start: dict[str, Any]) -> bool:
    return (
        as_float(start.get("idiosyncratic_obi"), 0.0) <= -0.20
        and as_float(start.get("micro_dev_bps"), 0.0) <= -1.0
    )


def _passive_leg(
    start: dict[str, Any], future: list[dict[str, Any]], *, limit_price: float
) -> dict[str, Any]:
    baseline_bid = as_float(start["bid"])
    if baseline_bid <= 0 or limit_price <= 0 or not future:
        raise ValueError("valid start book and future snapshots required")
    for offset, row in enumerate(future, start=1):
        if as_float(row.get("bid"), 0.0) >= limit_price:
            return {
                "proceeds": limit_price,
                "passive_attempted": True,
                "passive_filled": True,
                "timed_out": False,
                "fill_after_polls": offset,
            }
    return {
        "proceeds": as_float(future[-1]["bid"]),
        "passive_attempted": True,
        "passive_filled": False,
        "timed_out": True,
        "fill_after_polls": len(future),
    }


def execute_policy(
    start: dict[str, Any], future: list[dict[str, Any]], policy: str
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    baseline_bid = as_float(start["bid"])
    ask = as_float(start["ask"])
    midpoint = as_float(start["midpoint"])
    if baseline_bid <= 0 or ask <= 0 or midpoint <= 0:
        raise ValueError("invalid start book")

    if policy == "aggressive_bid" or (policy == "conditional_midpoint" and is_adverse(start)):
        outcome = {
            "proceeds": baseline_bid,
            "passive_attempted": False,
            "passive_filled": False,
            "timed_out": False,
            "fill_after_polls": 0,
            "route": "aggressive_due_to_adverse_book"
            if policy == "conditional_midpoint"
            else "aggressive_baseline",
        }
    elif policy == "ask_then_cross":
        outcome = _passive_leg(start, future, limit_price=ask)
        outcome["route"] = "ask_then_cross"
    elif policy in {"midpoint_then_cross", "conditional_midpoint"}:
        outcome = _passive_leg(start, future, limit_price=midpoint)
        outcome["route"] = (
            "conditional_midpoint_nonadverse"
            if policy == "conditional_midpoint"
            else "midpoint_then_cross"
        )
    elif policy == "half_midpoint_then_cross":
        passive = _passive_leg(start, future, limit_price=midpoint)
        outcome = {
            **passive,
            "proceeds": 0.5 * baseline_bid + 0.5 * as_float(passive["proceeds"]),
            "route": "half_aggressive_half_midpoint",
        }
    else:  # pragma: no cover
        raise AssertionError(policy)

    proceeds = as_float(outcome["proceeds"])
    outcome["improvement_bps"] = (proceeds / baseline_bid - 1.0) * 10_000.0
    return outcome


def process_day(path: Path) -> dict[str, Any]:
    history: dict[str, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=WAIT_POLLS + 1)
    )
    last_opportunity: dict[str, datetime] = {}
    accum: dict[str, PolicyAccumulator] = defaultdict(PolicyAccumulator)
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
            and as_float(row.get("bid1"), 0.0) > 0
            and as_float(row.get("ask1"), 0.0) > 0
            and as_float(row.get("midpoint"), 0.0) > 0
            and row.get("obi") is not None
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
            prior_opportunity = last_opportunity.get(code)
            opportunity = prior_opportunity is None or (
                event_time - prior_opportunity
            ).total_seconds() >= EVENT_COOLDOWN_SECONDS
            if opportunity:
                last_opportunity[code] = event_time

            book = {
                "time": event_time,
                "bid": as_float(row["bid1"]),
                "ask": as_float(row["ask1"]),
                "midpoint": as_float(row["midpoint"]),
                "idiosyncratic_obi": as_float(row["obi"]) - common_obi,
                "micro_dev_bps": as_float(row.get("micro_dev_bps"), 0.0),
                "opportunity": opportunity,
            }
            books = history[code]
            books.append(book)
            if len(books) < WAIT_POLLS + 1:
                continue
            start = books[0]
            future = list(books)[1:]
            elapsed = (future[-1]["time"] - start["time"]).total_seconds()
            if not start["opportunity"] or elapsed <= 0 or elapsed > WAIT_POLLS * 10.0:
                continue
            for policy in POLICIES:
                accum[policy].add(execute_policy(start, future, policy))

    span = (last_time - first_time).total_seconds() if first_time and last_time else 0.0
    complete = row_count >= MIN_COMPLETE_ROWS and span >= MIN_COMPLETE_SPAN_SECONDS
    return {
        "date": path.stem.removeprefix("depth_"),
        "path": str(path),
        "rows": row_count,
        "snapshots": snapshot_count,
        "span_seconds": round(span, 1),
        "complete_session": complete,
        "policies": {policy: accum[policy].summary() for policy in POLICIES},
    }


def combine_days(days: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for policy in POLICIES:
        daily = [
            as_float(day["policies"][policy].get("mean_improvement_bps"))
            for day in days
            if day["policies"][policy].get("opportunities", 0) > 0
        ]
        events = sum(
            int(day["policies"][policy].get("opportunities", 0)) for day in days
        )
        fills = sum(int(day["policies"][policy].get("passive_fills", 0)) for day in days)
        attempts = sum(
            int(day["policies"][policy].get("passive_attempts", 0)) for day in days
        )
        if not daily:
            combined[policy] = {"opportunities": 0, "independent_days": 0}
            continue
        mean = statistics.mean(daily)
        std = statistics.stdev(daily) if len(daily) > 1 else 0.0
        combined[policy] = {
            "opportunities": events,
            "independent_days": len(daily),
            "mean_daily_improvement_bps": round(mean, 4),
            "day_clustered_t": round(mean / std * math.sqrt(len(daily)), 3)
            if std > 0
            else None,
            "positive_days": sum(value > 0 for value in daily),
            "daily_means_bps": [round(value, 4) for value in daily],
            "conservative_passive_fill_rate": round(fills / attempts, 4)
            if attempts
            else None,
        }
    return combined


def render(result: dict[str, Any]) -> str:
    lines = [
        "# L2 sell execution — passive versus aggressive forward shadow",
        "",
        f"Status: **{result['status']}** | complete days: "
        f"{result['complete_days']}/{MIN_COMPLETE_DAYS} required",
        "",
        "All values are proceeds improvement versus immediately crossing the current bid.",
        "Passive fills require a later displayed bid to reach the limit; otherwise the policy crosses at timeout.",
        "",
        "| policy | opportunities | improvement | day t | positive days | conservative fill |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for policy in POLICIES:
        row = result["combined"][policy]
        lines.append(
            f"| {policy} | {row.get('opportunities', 0)} | "
            f"{row.get('mean_daily_improvement_bps', 0):+.4f} bp | "
            f"{row.get('day_clustered_t')} | {row.get('positive_days', 0)} | "
            f"{row.get('conservative_passive_fill_rate')} |"
        )
    primary = result["combined"][PRIMARY_POLICY]
    lines += [
        "",
        "## Decision",
        "",
        f"- Primary policy: **{PRIMARY_POLICY}**.",
        f"- Primary mean improvement: **{primary.get('mean_daily_improvement_bps', 0):+.4f} bp**.",
        f"- Statistical readiness: **{result['readiness']}**.",
        f"- Edge validated: **{str(result['edge_validated']).lower()}**.",
        "- No live execution behavior was changed.",
        "",
        "## Limitations",
        "",
        "- Displayed bid crossing is a conservative fill proxy, not a broker-confirmed fill.",
        "- Queue position, partial fills, hidden liquidity and order-size capacity are unavailable.",
        "- Generic sampled opportunities are not the strategy's actual sparse sell signals.",
        "- Three complete days cannot establish stability across market regimes.",
        "- Any future paper implementation must preserve the pre-sell position check and shared execution lock.",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- [{item['title']}]({item['url']}): {item['use']}." for item in SOURCES)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="L2 sell order-placement shadow audit.")
    parser.add_argument("--depth-dir", type=Path, default=DEPTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    all_days = []
    for path in sorted(args.depth_dir.glob("depth_*.jsonl")):
        print(f"processing {path.name} ...", flush=True)
        day = process_day(path)
        all_days.append(day)
        print(
            f"  rows={day['rows']} snapshots={day['snapshots']} "
            f"span={day['span_seconds']:.0f}s complete={day['complete_session']}",
            flush=True,
        )
    complete = [day for day in all_days if day["complete_session"]]
    combined = combine_days(complete)
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
        "wait_polls": WAIT_POLLS,
        "opportunity_cooldown_seconds": EVENT_COOLDOWN_SECONDS,
        "primary_policy": PRIMARY_POLICY,
        "complete_days": len(complete),
        "minimum_complete_days": MIN_COMPLETE_DAYS,
        "days": all_days,
        "combined": combined,
        "sources": SOURCES,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "l2_sell_execution.json"
    md_path = args.output_dir / "l2_sell_execution.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(result), encoding="utf-8")
    print(render(result))
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
