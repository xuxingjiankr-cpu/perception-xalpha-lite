"""Daily series for the one-name rotation dashboard: strategy, benchmark, index.

The dashboard shows a single-name book rotated daily against the onshore index. Two things
about it have to survive contact with the page, or the page is worse than nothing.

*The backfill is a backtest, and an in-sample one.* The four factors were chosen from a
456-candidate search whose data overlaps the backfilled window, and the selection step alone
is worth about 3 bps/day on this panel. Every backfilled row is therefore stamped
``phase: backtest`` so the renderer can grey it out. Only rows produced after the
specification was frozen are ``phase: live``.

*A long-only single name carries the market.* The strategy column is reported net of the
equal-weight eligible universe and net of realised turnover cost, so it is an excess, not a
raw return. The index columns are raw, because that is what an index is — and the page has to
say which is which.

    python scripts/daily_rotation_series.py backfill --sessions 400
    python scripts/daily_rotation_series.py append
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from xalpha_lite.book import select_names
from xalpha_lite.forward import composite_signal, freeze_spec, load_spec

from forward_record_framework import DRAFT_SPEC, OUT_DIR, prepare
from research_single_name_rotation import INDEX_CODES, ROUND_TRIP, index_series, rotation

PUBLIC = Path(__file__).resolve().parents[1] / "perception-xalpha-lite"
SERIES = PUBLIC / "docs" / "data" / "rotation.jsonl"
PICKS = PUBLIC / "docs" / "data" / "published_picks.jsonl"
LATEST = PUBLIC / "docs" / "data" / "next_pick.json"
SPEC_PATH = OUT_DIR / "single_name_rotation_v3.spec.json"

SPEC_V3 = dict(
    DRAFT_SPEC,
    name="single_name_rotation_v3",
    book_size=1,
    holding_days=1,
    rationale=(
        "The most concentrated, fastest-rotating form of the same four factors: hold the single "
        "best-ranked eligible name, rotate every session. Frozen so that the live curve published "
        "on the dashboard is a forward record rather than another unregistered variant."
    ),
    scoring_rule=(
        "Open-to-open return of the single held name, minus the equal-weight eligible-universe "
        "return over the same two bars, minus 30 bps whenever the name changes. A name sealed at "
        "the limit on the entry session is not bought and the session is skipped."
    ),
)
SPEC_V3.pop("predecessor", None)


def read_series() -> list[dict]:
    if not SERIES.exists():
        return []
    rows = []
    for line in SERIES.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_series(rows: list[dict]) -> None:
    SERIES.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted({row["date"]: row for row in rows}.values(), key=lambda row: row["date"])
    SERIES.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8"
    )


def index_returns(first: pd.Timestamp, last: pd.Timestamp) -> dict[str, pd.Series]:
    frames = index_series(str(first.date()), str(last.date()))
    out = {}
    for code, frame in frames.items():
        if frame is None or frame.empty:
            continue
        out[code] = frame["open"].pct_change().dropna()
    return out


def build_rows(sessions: int, phase: str, spec_sha: str, only_last: bool = False) -> list[dict]:
    panel, eligible = prepare(sessions)
    signal = composite_signal(SPEC_V3, panel, eligible)
    frame = rotation(signal, panel, eligible, 1, 1)
    if frame.empty:
        return []
    if only_last:
        frame = frame.tail(1)

    indices = index_returns(frame["entry_bar"].min(), frame["exit_bar"].max())
    rows = []
    for record in frame.itertuples(index=False):
        row = {
            "date": str(record.exit_bar.date()),
            "as_of": str(record.as_of.date()),
            "phase": phase,
            "spec_sha256": spec_sha[:16],
            "strategy_net": round(float(record.net), 6),
            "strategy_gross": round(float(record.gross), 6),
            "universe": round(float(record.benchmark), 6),
            "cost": round(float(record.cost), 6),
            "rotated": bool(record.churn > 0),
        }
        for code in INDEX_CODES:
            series = indices.get(code)
            if series is None:
                continue
            window = series.loc[(series.index > record.entry_bar) & (series.index <= record.exit_bar)]
            row[code.replace(".", "_")] = round(float((1.0 + window).prod() - 1.0), 6) if len(window) else None
        rows.append(row)
    return rows


def cmd_freeze(args) -> int:
    frozen = freeze_spec(SPEC_V3, SPEC_PATH)
    print(json.dumps({k: frozen[k] for k in ("name", "frozen_at", "spec_sha256")}, indent=2))
    return 0


def cmd_backfill(args) -> int:
    spec = load_spec(SPEC_PATH)
    rows = build_rows(args.sessions, "backtest", spec["spec_sha256"])
    live = {row["date"] for row in read_series() if row.get("phase") == "live"}
    rows = [row for row in rows if row["date"] not in live]
    write_series(rows + [row for row in read_series() if row["date"] in live])
    print(f"backfilled {len(rows)} sessions -> {SERIES}")
    print(f"  {rows[0]['date']} .. {rows[-1]['date']}" if rows else "  (empty)")
    return 0


def cmd_append(args) -> int:
    """Append the newest matured session as a live row. Never rewrites an existing date."""
    spec = load_spec(SPEC_PATH)
    rows = build_rows(args.sessions, "live", spec["spec_sha256"], only_last=True)
    if not rows:
        print("no matured session to append")
        return 1
    existing = read_series()
    known = {row["date"] for row in existing}
    fresh = [row for row in rows if row["date"] not in known]
    if not fresh:
        print(f"{rows[-1]['date']} already recorded; series unchanged")
        return 0
    write_series(existing + fresh)
    print(f"appended {[row['date'] for row in fresh]}")
    return 0


def cmd_publish(args) -> int:
    """Publish the name to be held on the next session, before that session happens.

    This is the part that carries the evidential weight. A record scored after the fact always
    invites the question of whether the rule was adjusted once the outcome was visible; a pick
    timestamped publicly before the market opens cannot be. The file is append-only and each
    session is written once, so a published pick can be shown to be wrong but never edited.
    """
    spec = load_spec(SPEC_PATH)
    panel, eligible = prepare(args.sessions)
    signal = composite_signal(spec, panel, eligible)
    as_of = signal.index.max()
    names = select_names(signal.loc[as_of], int(spec["book_size"]))
    if not names:
        print("no eligible name on the latest session")
        return 1

    entry = {
        "as_of": str(as_of.date()),
        "buy_at": "open of the next trading session",
        "symbol": names[0],
        "composite_rank": round(float(signal.loc[as_of, names[0]]), 6),
        "close_on_as_of": round(float(panel["close"].loc[as_of, names[0]]), 4),
        "eligible_names": int(eligible.loc[as_of].sum()),
        "spec_sha256": spec["spec_sha256"],
        "published_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_forward_record_not_trading",
        "orders": [],
    }

    PICKS.parent.mkdir(parents=True, exist_ok=True)
    published = []
    if PICKS.exists():
        for line in PICKS.read_text(encoding="utf-8").splitlines():
            try:
                published.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if any(row.get("as_of") == entry["as_of"] for row in published):
        print(f"{entry['as_of']} already published; append-only file unchanged")
        return 0
    with PICKS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    LATEST.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"published {entry['as_of']} -> hold {entry['symbol']} from the next open")
    return 0


def cmd_summary(args) -> int:
    rows = read_series()
    if not rows:
        print("series is empty")
        return 1
    frame = pd.DataFrame(rows)
    for phase in ("backtest", "live"):
        part = frame[frame["phase"] == phase]
        if part.empty:
            print(f"{phase:<9} (none)")
            continue
        net = part["strategy_net"].astype(float)
        print(
            f"{phase:<9} n={len(part):>4}  {part['date'].min()}..{part['date'].max()}  "
            f"net={net.mean()*1e4:+7.2f}bps/day  cum={(1+net).prod()-1:+7.2%}  "
            f"hit={(net>0).mean():.3f}"
        )
    for code in INDEX_CODES:
        key = code.replace(".", "_")
        if key in frame:
            series = pd.to_numeric(frame[key], errors="coerce").dropna()
            print(f"  {code:<12} cum={(1+series).prod()-1:+7.2%} over {len(series)} sessions")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, handler in (("freeze", cmd_freeze), ("backfill", cmd_backfill),
                          ("append", cmd_append), ("publish", cmd_publish),
                          ("summary", cmd_summary)):
        command = sub.add_parser(name)
        command.add_argument("--sessions", type=int, default=400)
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
