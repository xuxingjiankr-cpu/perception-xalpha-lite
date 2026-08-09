"""Compute and publish the next session's pick on a runner, from public data only.

The picks were previously computed on the machine that holds the historical panel and pushed
here. That makes the record depend on one desktop being switched on, which is a poor property
for something whose whole claim is that it was published before the outcome.

This does the same work from nothing but the frozen specification and a public data source, so
the record continues whether or not any particular machine is awake — and, more usefully, so
that anyone can rerun it and get the same name. A pick nobody else can reproduce is a claim,
not a record.

Per-symbol latency dominates: one query costs about 1.2 s from a runner regardless of how many
bars it returns, so the work is parallelised across processes, each with its own session.

    python tools/publish_from_source.py --workers 8 --sessions 260
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xalpha_lite.book import select_names  # noqa: E402
from xalpha_lite.discovery import build_panel  # noqa: E402
from xalpha_lite.forward import composite_signal, load_spec  # noqa: E402
from xalpha_lite.universe import apply_sealed_bar_limits, point_in_time_eligibility  # noqa: E402

DATA = ROOT / "docs" / "data"
SPEC_PATH = DATA / "single_name_rotation_v3.spec.json"
PICKS = DATA / "published_picks.jsonl"
LATEST = DATA / "next_pick.json"
FIELDS = "date,open,high,low,close,volume,amount,isST,tradestatus"
EMPTY_STATEMENTS = pd.DataFrame(columns=["symbol", "report_date", "notice_date", "update_date"])


def _login():
    import baostock as bs

    # baostock chatters on stdout at login; the runner log should carry our lines, not its.
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        bs.login()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    return bs


def universe_codes(day: date, lookback: int = 12) -> tuple[list[str], date]:
    """Membership as of the most recent session on or before ``day``.

    The listing endpoint answers for a trading date; asked about a weekend or a holiday it
    returns an empty set rather than an error, which silently becomes an empty universe. Walk
    back until a real session answers, and say which one did.
    """
    bs = _login()
    try:
        for offset in range(lookback):
            probe = day - timedelta(days=offset)
            result = bs.query_all_stock(day=str(probe))
            codes = []
            while result.error_code == "0" and result.next():
                code, status = result.get_row_data()[:2]
                # Indices and funds share the namespace; only the equity boards are wanted.
                if str(status) == "1" and code[:5] in ("sh.60", "sh.68", "sz.00", "sz.30"):
                    codes.append(code)
            if codes:
                return sorted(codes), probe
        raise RuntimeError(f"no trading session with listings in the {lookback} days to {day}")
    finally:
        bs.logout()


def _fetch_chunk(payload: tuple[list[str], str, str]) -> list[dict]:
    codes, start, end = payload
    bs = _login()
    rows = []
    try:
        for code in codes:
            result = bs.query_history_k_data_plus(
                code, FIELDS, start_date=start, end_date=end, frequency="d", adjustflag="1"
            )
            while result.error_code == "0" and result.next():
                values = result.get_row_data()
                rows.append(
                    {
                        "date": values[0],
                        "symbol": code.replace("sh.", "SH_").replace("sz.", "SZ_"),
                        "open": values[1], "high": values[2], "low": values[3],
                        "close": values[4], "volume": values[5], "amount": values[6],
                        "is_st": values[7], "trade_status": values[8],
                    }
                )
    finally:
        bs.logout()
    return rows


def fetch_prices(codes: list[str], start: str, end: str, workers: int) -> pd.DataFrame:
    if not codes:
        raise ValueError("no symbols to fetch")
    workers = max(1, int(workers))
    size = max(1, len(codes) // workers)
    chunks = [(codes[i:i + size], start, end) for i in range(0, len(codes), size)]
    began = time.time()
    with Pool(processes=min(workers, len(chunks))) as pool:
        collected = pool.map(_fetch_chunk, chunks)
    rows = [row for chunk in collected for row in chunk]
    print(f"fetched {len(rows)} bars for {len(codes)} symbols in {time.time()-began:.0f}s "
          f"across {len(chunks)} workers")
    frame = pd.DataFrame(rows)
    for column in ("open", "high", "low", "close", "volume", "amount", "trade_status"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_st"] = pd.to_numeric(frame["is_st"], errors="coerce").fillna(0).gt(0)
    return frame.dropna(subset=["close"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=260)
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args()

    spec = load_spec(SPEC_PATH)
    end = date.today()
    start = end - timedelta(days=int(args.sessions * 1.55))
    codes, listing_day = universe_codes(end)
    if not codes:
        print("empty universe; refusing to publish")
        return 1
    print(f"universe: {len(codes)} equity codes as of {listing_day}; window {start} .. {end}")

    prices = fetch_prices(codes, str(start), str(end), args.workers)
    panel = apply_sealed_bar_limits(
        build_panel(prices, EMPTY_STATEMENTS, {"require_neutralization_data": False})
    )
    eligible = point_in_time_eligibility(panel, **spec.get("universe", {}))
    signal = composite_signal(spec, panel, eligible)

    as_of = signal.index.max()
    names = select_names(signal.loc[as_of], int(spec["book_size"]))
    if not names:
        print("no eligible name on the latest session")
        return 1

    entry = {
        "as_of": str(pd.Timestamp(as_of).date()),
        "buy_at": "open of the next trading session",
        "symbol": names[0],
        "composite_rank": round(float(signal.loc[as_of, names[0]]), 6),
        "close_on_as_of": round(float(panel["close"].loc[as_of, names[0]]), 4),
        "eligible_names": int(eligible.loc[as_of].sum()),
        "spec_sha256": spec["spec_sha256"],
        "published_at": datetime.now(timezone.utc).isoformat(),
        "computed_by": "github-actions-from-source",
        "status": "research_only_forward_record_not_trading",
        "orders": [],
    }
    print(json.dumps(entry, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    published = []
    if PICKS.exists():
        for line in PICKS.read_text(encoding="utf-8").splitlines():
            try:
                published.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    prior = next((row for row in published if row.get("as_of") == entry["as_of"]), None)
    if prior:
        # Never rewrite a published pick. If an independent recomputation disagrees with one
        # already on the record, that is a finding about reproducibility and must be visible.
        if prior.get("symbol") != entry["symbol"]:
            print(f"DISAGREEMENT: {entry['as_of']} already published as {prior['symbol']}, "
                  f"recomputed as {entry['symbol']}")
            return 2
        print(f"{entry['as_of']} already published as {prior['symbol']}; reproduced exactly")
        return 0

    with PICKS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    LATEST.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"published {entry['as_of']} -> hold {entry['symbol']} from the next open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
