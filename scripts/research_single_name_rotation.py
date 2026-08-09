"""What a one-name, one-day rotation actually earns, before anyone publishes a curve of it.

The proposed dashboard holds the single best-ranked name and rotates every session. Two
measured facts from this project decide whether that is worth publishing: the gross ceiling of
price/volume factors at a one-day horizon is about 3.84 bps/day, and a full round trip costs
30 bps. Cost is charged on realised turnover, so everything depends on how often the top name
actually changes — which is an empirical question, not an assumption.

This measures it, over the requested week and over a long enough window to mean anything, and
prices the same factors at the frozen record's ten-name / ten-session settings for contrast.
Entry and exit follow the record's convention: signal at close t, buy at open t+1, sell at the
open of the session the hold ends on.

    python scripts/research_single_name_rotation.py --sessions 400
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from xalpha_lite.book import select_names
from xalpha_lite.forward import composite_signal
from xalpha_lite.universe import sealed_bar_limits

from forward_record_framework import DRAFT_SPEC, OUT_DIR, prepare

ROUND_TRIP = 0.003
INDEX_CODES = {"sh.000016": "SSE50 (onshore A50 proxy)", "sh.000300": "CSI300"}


def rotation(
    signal: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    book_size: int,
    hold: int,
) -> pd.DataFrame:
    """Open-to-open returns of a top-N book, against the universe it was selected from.

    Three things this has to get right, each of which inflates the result badly if skipped:

    *Executability.* The top-ranked name on a momentum-flavoured composite is exactly the one
    likely to open limit-locked, and a locked bar cannot be bought. Those legs are dropped —
    the position was never opened. Measured on this panel they carry +6.05% against +0.38%
    for legs that could actually be traded, so keeping them is not a rounding error.

    *Benchmark.* A long-only book carries the market. Reporting it against zero credits the
    factor for beta. The comparison is the equal-weight eligible universe over the same bars.

    *Overlap.* At ``hold`` > 1 consecutive rows share sessions, so their returns are not a
    sequence that may be compounded. Non-overlapping rows are flagged for that purpose.
    """
    open_, sessions = panel["open"], list(signal.index)
    limit_up, limit_down = sealed_bar_limits(panel)
    rows, previous = [], set()
    for index in range(len(sessions) - hold - 1):
        as_of = sessions[index]
        names = select_names(signal.loc[as_of], book_size)
        if len(names) < book_size:
            continue
        entry_bar, exit_bar = sessions[index + 1], sessions[index + 1 + hold]
        legs, blocked = [], 0
        for symbol in names:
            if bool(limit_up.loc[entry_bar, symbol]) or bool(limit_down.loc[entry_bar, symbol]):
                blocked += 1
                continue
            entry, exit_ = open_.loc[entry_bar, symbol], open_.loc[exit_bar, symbol]
            if np.isfinite(entry) and np.isfinite(exit_) and entry > 0:
                legs.append(float(exit_ / entry - 1.0))
        if not legs:
            continue
        window = (open_.loc[exit_bar] / open_.loc[entry_bar].replace(0.0, np.nan) - 1.0)
        window = window.where(eligible.loc[as_of]).replace([np.inf, -np.inf], np.nan).dropna()
        benchmark = float(window.mean()) if len(window) else np.nan
        held = set(names)
        churn = len(held - previous) / float(len(held))
        previous = held
        gross = float(np.mean(legs))
        rows.append(
            {
                "as_of": as_of,
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "names": names,
                "filled": len(legs),
                "blocked_by_limit": blocked,
                "gross": gross,
                "benchmark": benchmark,
                "excess": gross - benchmark,
                "churn": churn,
                "cost": churn * ROUND_TRIP,
                "net": gross - benchmark - churn * ROUND_TRIP,
                "non_overlapping": index % hold == 0,
            }
        )
    return pd.DataFrame(rows)


def index_series(start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    bs.login()
    frames = {}
    try:
        for code in INDEX_CODES:
            result = bs.query_history_k_data_plus(
                code, "date,open,close", start_date=start, end_date=end, frequency="d"
            )
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            frame = pd.DataFrame(rows, columns=["date", "open", "close"])
            frame["date"] = pd.to_datetime(frame["date"])
            frames[code] = frame.set_index("date").astype(float)
    finally:
        bs.logout()
    return frames


def describe(frame: pd.DataFrame, label: str, hold: int) -> dict:
    """Per-day figures divide by the holding length; compounding uses only disjoint windows."""
    if frame.empty:
        return {"label": label, "n": 0}
    net, disjoint = frame["net"], frame.loc[frame["non_overlapping"], "net"]
    return {
        "label": label,
        "n": int(len(frame)),
        "hold": hold,
        "gross_bps_per_day": round(float(frame["gross"].mean()) * 1e4 / hold, 2),
        "benchmark_bps_per_day": round(float(frame["benchmark"].mean()) * 1e4 / hold, 2),
        "excess_bps_per_day": round(float(frame["excess"].mean()) * 1e4 / hold, 2),
        "cost_bps_per_day": round(float(frame["cost"].mean()) * 1e4 / hold, 2),
        "net_bps_per_day": round(float(net.mean()) * 1e4 / hold, 2),
        "mean_churn": round(float(frame["churn"].mean()), 3),
        "hit_rate": round(float((net > 0).mean()), 3),
        "period_vol_pct": round(float(net.std()) * 100, 2),
        "blocked_legs": int(frame["blocked_by_limit"].sum()),
        "compounded_net_pct": round(float((1.0 + disjoint).prod() - 1.0) * 100, 2),
        "compounded_windows": int(len(disjoint)),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=400)
    args = parser.parse_args()

    panel, eligible = prepare(args.sessions)
    signal = composite_signal(DRAFT_SPEC, panel, eligible)
    print(f"panel: {panel['close'].shape[0]} sessions x {panel['close'].shape[1]} symbols\n")

    holds = {"top1_hold1": (1, 1), "top10_hold1": (10, 1), "top10_hold10": (10, 10)}
    variants = {
        name: rotation(signal, panel, eligible, size, hold)
        for name, (size, hold) in holds.items()
    }

    print(f"{'variant':<14} {'n':>4} {'gross':>8} {'bench':>8} {'excess':>8} {'cost':>7} "
          f"{'net':>8} {'churn':>7} {'hit':>6} {'blocked':>8} {'cum%':>8}")
    summary = {}
    for name, frame in variants.items():
        stats = describe(frame, name, holds[name][1])
        summary[name] = stats
        if not stats["n"]:
            continue
        print(
            f"{name:<14} {stats['n']:>4} {stats['gross_bps_per_day']:>8.2f} "
            f"{stats['benchmark_bps_per_day']:>8.2f} {stats['excess_bps_per_day']:>8.2f} "
            f"{stats['cost_bps_per_day']:>7.2f} {stats['net_bps_per_day']:>8.2f} "
            f"{stats['mean_churn']:>7.3f} {stats['hit_rate']:>6.3f} "
            f"{stats['blocked_legs']:>8} {stats['compounded_net_pct']:>7.2f}%"
        )
    print("\nbps/day; net = gross - eligible-universe mean - realised turnover cost.")
    print("cum% compounds only non-overlapping windows. blocked = legs unbuyable at entry.\n")

    top1 = variants["top1_hold1"]
    week = top1.tail(5)
    print("last five sessions, one name held one day:")
    for row in week.itertuples(index=False):
        print(
            f"  {row.as_of.date()} -> buy {row.entry_bar.date()} sell {row.exit_bar.date()}  "
            f"{row.names[0]:<10} gross {row.gross*100:+6.2f}%  bench {row.benchmark*100:+5.2f}%  "
            f"cost {row.cost*1e4:5.1f}bp  net {row.net*100:+6.2f}%"
        )
    print(f"  week net (compounded): {((1+week['net']).prod()-1)*100:+.2f}%")

    first, last = top1["entry_bar"].min(), top1["exit_bar"].max()
    indices = index_series(str(first.date()), str(last.date()))
    print("\nindex over the same span, open to open:")
    index_summary = {}
    for code, label in INDEX_CODES.items():
        frame = indices.get(code)
        if frame is None or frame.empty:
            continue
        window = frame.loc[(frame.index >= first) & (frame.index <= last), "open"]
        full = float(window.iloc[-1] / window.iloc[0] - 1.0) * 100
        week_window = frame.loc[(frame.index >= week["entry_bar"].min()) & (frame.index <= week["exit_bar"].max()), "open"]
        week_return = float(week_window.iloc[-1] / week_window.iloc[0] - 1.0) * 100 if len(week_window) > 1 else float("nan")
        index_summary[code] = {"label": label, "full_pct": round(full, 2), "week_pct": round(week_return, 2)}
        print(f"  {code} {label:<26} full {full:+7.2f}%   last week {week_return:+6.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "single_name_rotation_study.json").write_text(
        json.dumps({"variants": summary, "indices": index_summary}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
