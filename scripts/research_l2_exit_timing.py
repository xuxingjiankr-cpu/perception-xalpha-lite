"""Forward L2 exit-timing audit using real five-level A-share ETF books.

Primary preregistered hypothesis
--------------------------------
For a held long position, sell immediately at bid when both:

* idiosyncratic OBI = own OBI - same-snapshot cross-sectional median <= -0.20; and
* microprice displacement <= -1.0 bp.

The counterfactual is to wait 1/3/6/12/60 collector polls (roughly
5/15/30/60/300 seconds) and then sell at the later bid. The 300-second horizon
matches the current trading-agent cadence. Positive ``sell_now_advantage_bps`` means
the trigger sold at a higher price. Signals have a 60-second per-code cooldown so
thousands of near-identical snapshots are not treated as independent trades.

The script streams one day at a time and reports day-clustered evidence. It refuses
promotion below 20 complete trading days. Existing data are forward observations, but
there are currently too few independent days for a conclusion.

STRICTLY OFFLINE / SHADOW. No broker calls, no orders, no live config, no overlays.
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
from typing import Any, Iterable

from run_etf_paper_trading_agent import ROOT, as_float


DEPTH_DIR = ROOT / "outputs" / "l2_depth"
OUT_DIR = ROOT / "outputs" / "l2_exit_timing"
HORIZONS = (1, 3, 6, 12, 60)
MIN_COMPLETE_DAYS = 20
MIN_COMPLETE_SPAN_SECONDS = 19_000
MIN_COMPLETE_ROWS = 50_000
EVENT_COOLDOWN_SECONDS = 60.0

TRIGGERS = (
    "raw_obi_adverse",
    "idiosyncratic_obi_adverse",
    "microprice_adverse",
    "combined_adverse",  # preregistered primary
)
PRIMARY_TRIGGER = "combined_adverse"

SOURCES = [
    {
        "title": "Optimal Asset Liquidation Using Limit Order Book Information",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2113827",
        "use": "low supply-demand imbalance defines a sell region in an optimal stopping problem",
    },
    {
        "title": "The Price Impact of Order Book Events",
        "url": "https://arxiv.org/abs/1011.6402",
        "use": "order-flow imbalance is robustly linked to short-horizon price changes",
    },
    {
        "title": "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141",
        "use": "stationary order-flow features carry information at multiple short horizons",
    },
]


@dataclass
class Accumulator:
    n: int = 0
    sum_adv: float = 0.0
    sum_adv2: float = 0.0
    wins: int = 0
    sum_forward_mid: float = 0.0

    def add(self, advantage_bps: float, forward_mid_bps: float) -> None:
        self.n += 1
        self.sum_adv += advantage_bps
        self.sum_adv2 += advantage_bps * advantage_bps
        self.wins += int(advantage_bps > 0)
        self.sum_forward_mid += forward_mid_bps

    def summary(self) -> dict[str, Any]:
        if not self.n:
            return {"events": 0}
        mean = self.sum_adv / self.n
        variance = max(0.0, self.sum_adv2 / self.n - mean * mean)
        std = math.sqrt(variance)
        return {
            "events": self.n,
            "mean_sell_now_advantage_bps": round(mean, 4),
            "pooled_t_pseudo_only": round(mean / std * math.sqrt(self.n), 3) if std > 0 else None,
            "win_rate": round(self.wins / self.n, 4),
            "mean_forward_mid_return_bps": round(self.sum_forward_mid / self.n, 4),
        }


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def sell_now_advantage_bps(bid_now: float, bid_later: float) -> float:
    """Positive means selling now yields higher proceeds than waiting."""
    if bid_now <= 0 or bid_later <= 0:
        raise ValueError("bid prices must be positive")
    return (bid_now / bid_later - 1.0) * 10_000.0


def classify_triggers(obi: float, idiosyncratic_obi: float, micro_dev_bps: float) -> set[str]:
    out: set[str] = set()
    if obi <= -0.30:
        out.add("raw_obi_adverse")
    if idiosyncratic_obi <= -0.20:
        out.add("idiosyncratic_obi_adverse")
    if micro_dev_bps <= -1.0:
        out.add("microprice_adverse")
    if idiosyncratic_obi <= -0.20 and micro_dev_bps <= -1.0:
        out.add("combined_adverse")
    return out


def iter_snapshot_groups(path: Path) -> Iterable[list[dict[str, Any]]]:
    current_key: str | None = None
    group: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = str(row.get("collected_at") or "")
            if not key:
                continue
            if current_key is not None and key != current_key:
                yield group
                group = []
            current_key = key
            group.append(row)
    if group:
        yield group


def process_day(path: Path) -> dict[str, Any]:
    states: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=max(HORIZONS) + 1))
    last_event: dict[tuple[str, str], datetime] = {}
    accum: dict[tuple[str, int], Accumulator] = defaultdict(Accumulator)
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
            and as_float(row.get("bid1"), 0.0) > 0
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
            midpoint = as_float(row.get("midpoint"), 0.0)
            bid = as_float(row.get("bid1"), 0.0)
            obi = as_float(row.get("obi"), 0.0)
            idio = obi - common_obi
            micro = as_float(row.get("micro_dev_bps"), 0.0)
            raw_triggers = classify_triggers(obi, idio, micro)
            events: set[str] = set()
            for trigger in raw_triggers:
                previous = last_event.get((code, trigger))
                if previous is None or (event_time - previous).total_seconds() >= EVENT_COOLDOWN_SECONDS:
                    events.add(trigger)
                    last_event[(code, trigger)] = event_time

            history = states[code]
            current = {
                "time": event_time,
                "midpoint": midpoint,
                "bid": bid,
                "events": events,
            }
            for horizon in HORIZONS:
                if len(history) < horizon:
                    continue
                prior = history[-horizon]
                elapsed = (event_time - prior["time"]).total_seconds()
                # A nominal poll is ~5 seconds. Reject gaps/lunch boundaries and stale books.
                if elapsed <= 0 or elapsed > max(20.0, horizon * 10.0):
                    continue
                forward_mid_bps = (midpoint / prior["midpoint"] - 1.0) * 10_000.0
                advantage = sell_now_advantage_bps(prior["bid"], bid)
                for trigger in prior["events"]:
                    accum[(trigger, horizon)].add(advantage, forward_mid_bps)
            history.append(current)

    span = (last_time - first_time).total_seconds() if first_time and last_time else 0.0
    complete = row_count >= MIN_COMPLETE_ROWS and span >= MIN_COMPLETE_SPAN_SECONDS
    return {
        "date": path.stem.removeprefix("depth_"),
        "path": str(path),
        "rows": row_count,
        "snapshots": snapshot_count,
        "span_seconds": round(span, 1),
        "complete_session": complete,
        "metrics": {
            trigger: {
                f"+{horizon}_polls": accum[(trigger, horizon)].summary()
                for horizon in HORIZONS
            }
            for trigger in TRIGGERS
        },
    }


def combine_days(days: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for trigger in TRIGGERS:
        combined[trigger] = {}
        for horizon in HORIZONS:
            key = f"+{horizon}_polls"
            daily = [
                as_float(day["metrics"][trigger][key].get("mean_sell_now_advantage_bps"))
                for day in days
                if day["metrics"][trigger][key].get("events", 0) > 0
            ]
            event_count = sum(
                int(day["metrics"][trigger][key].get("events", 0)) for day in days
            )
            if not daily:
                combined[trigger][key] = {"events": 0, "independent_days": 0}
                continue
            mean = statistics.mean(daily)
            std = statistics.stdev(daily) if len(daily) > 1 else 0.0
            combined[trigger][key] = {
                "events": event_count,
                "independent_days": len(daily),
                "mean_daily_sell_now_advantage_bps": round(mean, 4),
                "day_clustered_t": round(mean / std * math.sqrt(len(daily)), 3)
                if std > 0
                else None,
                "positive_days": sum(value > 0 for value in daily),
                "daily_means_bps": [round(value, 4) for value in daily],
            }
    return combined


def render(result: dict[str, Any]) -> str:
    lines = [
        "# L2 adverse-imbalance exit timing — forward shadow audit",
        "",
        f"Status: **{result['status']}** | complete days: "
        f"{result['complete_days']}/{MIN_COMPLETE_DAYS} required",
        "",
        "Positive advantage means selling immediately at bid beat waiting and selling at the later bid.",
        "Commission cancels because both alternatives execute one sell.",
        "",
        "## Primary preregistered trigger",
        "",
        "`idiosyncratic OBI <= -0.20 AND microprice displacement <= -1.0 bp`",
        "",
        "| wait | events | days | sell-now advantage | day t | positive days |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    primary = result["combined"][PRIMARY_TRIGGER]
    for horizon in HORIZONS:
        row = primary[f"+{horizon}_polls"]
        lines.append(
            f"| ~{horizon * 5}s | {row.get('events', 0)} | "
            f"{row.get('independent_days', 0)} | "
            f"{row.get('mean_daily_sell_now_advantage_bps', 0):+.4f} bp | "
            f"{row.get('day_clustered_t')} | {row.get('positive_days', 0)} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Statistical readiness: **{result['readiness']}**.",
        f"- Edge validated: **{str(result['edge_validated']).lower()}**.",
        "- Pooled event counts are not treated as independent evidence; the gate uses trading days.",
        "- This result cannot change the live sell engine, sizing, or execution mode.",
        "",
        "## Limitations",
        "",
        "- Five-level snapshots are quotes, not message-level order flow; cancellations between polls are unseen.",
        "- The trigger is evaluated generically across T0 ETFs, not only while the strategy holds a position.",
        "- Three complete days cannot establish regime robustness or survive a proper forward/OOS test.",
        "- A future implementation must confirm fills and model latency before recording any P&L.",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- [{item['title']}]({item['url']}): {item['use']}." for item in SOURCES)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Forward L2 exit-timing audit.")
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
        "primary_trigger": PRIMARY_TRIGGER,
        "event_cooldown_seconds": EVENT_COOLDOWN_SECONDS,
        "horizons_polls": list(HORIZONS),
        "complete_days": len(complete),
        "minimum_complete_days": MIN_COMPLETE_DAYS,
        "days": all_days,
        "combined": combined,
        "sources": SOURCES,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "l2_exit_timing.json"
    md_path = args.output_dir / "l2_exit_timing.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(result), encoding="utf-8")
    print(render(result))
    print(f"outputs: {json_path} | {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
