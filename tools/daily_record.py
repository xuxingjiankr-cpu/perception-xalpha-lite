"""The whole daily record, computed on a runner from public data.

Previously this work happened on the machine holding the historical panel and was pushed here.
That made a record whose entire claim is "published before the outcome" depend on one desktop
staying awake, and it made the picks unreproducible by anyone else.

Everything now runs from the frozen specification and a public source. The useful consequence
is not that a laptop can be switched off; it is that a reader can run this file and get the
same name. A pick nobody can recompute is a claim, not a record.

Three jobs, in order:

``pick``    select the name for the next session and publish it before that session opens
``record``  score every session whose holding window has fully elapsed and append it
``chart``   redraw

    python tools/daily_record.py --workers 8

Why a v4 specification exists: v3 named a universe rule but never said where the universe came
from, so two faithful implementations of it disagreed on 614 of ~4,700 eligible names. A
specification whose digest does not determine the answer is not doing its job. v3 keeps the one
pick it published; v4 pins the source, the board scope and the window, and everything else is
identical.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xalpha_lite.book import select_names  # noqa: E402
from xalpha_lite.discovery import build_panel  # noqa: E402
from xalpha_lite.forward import composite_signal, load_spec  # noqa: E402
from xalpha_lite.universe import (  # noqa: E402
    apply_sealed_bar_limits,
    point_in_time_eligibility,
    sealed_bar_limits,
)

DATA = ROOT / "docs" / "data"
SPEC_PATH = DATA / "single_name_rotation_v4.spec.json"
PICKS = DATA / "published_picks.jsonl"
LATEST = DATA / "next_pick.json"
SERIES = DATA / "rotation.jsonl"
FIELDS = "date,open,high,low,close,volume,amount,isST,tradestatus"
EMPTY_STATEMENTS = pd.DataFrame(columns=["symbol", "report_date", "notice_date", "update_date"])
INDEX_CODES = {"sh.000016": "SSE50 (onshore A50 proxy)", "sh.000300": "CSI300"}
# Two gates against publishing a name chosen from whichever symbols happened to answer.
MINIMUM_COVERAGE = 0.98          # symbols returning any bars
MINIMUM_SESSION_COVERAGE = 0.95  # of those, symbols quoting on the session being ranked
# The client has no timeout of its own, so a half-open socket blocks the worker forever and the
# retry logic below never gets a turn. A run was killed at 110 minutes having never finished a
# fetch that took 36 the day before, with no error, because nothing ever raised.
SOCKET_TIMEOUT_SECONDS = 45
FETCH_BUDGET_SECONDS = 55 * 60


def _login():
    import baostock as bs

    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)

    # baostock prints a banner on login; the runner log should carry our lines, not its.
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        bs.login()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    return bs


def universe_codes(day: date, boards: tuple[str, ...], lookback: int = 12) -> tuple[list[str], date]:
    """Listed codes as of the most recent session on or before ``day``.

    The listing endpoint answers for a trading date and returns an empty set for a weekend
    rather than an error, which silently becomes an empty universe. Walk back to a real session
    and report which one answered.
    """
    bs = _login()
    try:
        for offset in range(lookback):
            probe = day - timedelta(days=offset)
            result = bs.query_all_stock(day=str(probe))
            codes = []
            while result.error_code == "0" and result.next():
                code, status = result.get_row_data()[:2]
                if str(status) == "1" and code[:5] in boards:
                    codes.append(code)
            if codes:
                return sorted(codes), probe
        raise RuntimeError(f"no session with listings in the {lookback} days to {day}")
    finally:
        bs.logout()


def _query_one(bs, code: str, start: str, end: str, adjust: str, attempts: int = 3):
    """Bars for one symbol, retried on an empty answer; may hand back a fresh session.

    Under parallel load this source returns an empty result set with ``error_code == "0"``:
    a throttle indistinguishable from a stock with no history. Worse, a worker's session can
    die outright, after which every remaining symbol in its contiguous slice returns empty —
    the failures come back as runs of consecutive codes, which is what gave this away. Retrying
    inside the dead session cannot help, so the session is rebuilt before the last attempt.

    Returns ``(rows, bs)``; the caller must keep the session handed back.
    """
    for attempt in range(attempts):
        try:
            result = bs.query_history_k_data_plus(
                code, FIELDS, start_date=start, end_date=end, frequency="d", adjustflag=adjust
            )
            rows = []
            while result.error_code == "0" and result.next():
                v = result.get_row_data()
                rows.append({"date": v[0], "symbol": code.replace("sh.", "SH_").replace("sz.", "SZ_"),
                             "open": v[1], "high": v[2], "low": v[3], "close": v[4],
                             "volume": v[5], "amount": v[6], "is_st": v[7], "trade_status": v[8]})
            if rows:
                return rows, bs
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
        if attempt == attempts - 2:
            try:
                bs.logout()
            except Exception:
                pass
            bs = _login()
    return [], bs


def _fetch_chunk(payload: tuple[list[str], str, str, str]) -> tuple[list[dict], list[str]]:
    codes, start, end, adjust = payload
    bs = _login()
    rows, empty = [], []
    try:
        for code in codes:
            found, bs = _query_one(bs, code, start, end, adjust)
            rows.extend(found)
            if not found:
                empty.append(code)
    finally:
        bs.logout()
    return rows, empty


def fetch_prices(codes: list[str], start: str, end: str, adjust: str, workers: int) -> pd.DataFrame:
    if not codes:
        raise ValueError("no symbols to fetch")
    workers = max(1, int(workers))
    size = max(1, len(codes) // workers)
    chunks = [(codes[i:i + size], start, end, adjust) for i in range(0, len(codes), size)]
    began = time.time()
    with Pool(processes=min(workers, len(chunks))) as pool:
        collected = pool.map(_fetch_chunk, chunks)
    rows = [row for chunk, _ in collected for row in chunk]
    empty = [code for _, missing in collected for code in missing]
    covered = len(codes) - len(empty)
    print(f"fetched {len(rows)} bars for {covered}/{len(codes)} symbols in "
          f"{time.time()-began:.0f}s across {len(chunks)} workers")
    if empty:
        # Stragglers are almost always one worker's dead tail. A serial sweep on a fresh
        # session recovers them for the price of a few minutes.
        print(f"{len(empty)} symbols empty after the parallel pass; sweeping serially")
        bs, recovered = _login(), []
        try:
            for code in list(empty):
                # The sweep is the unbounded part: one symbol at a time, with backoff. On a
                # degraded source it will happily eat the whole job. Stop and let the coverage
                # gate decide, rather than being killed with nothing written.
                if time.time() - began > FETCH_BUDGET_SECONDS:
                    print(f"sweep abandoned after {FETCH_BUDGET_SECONDS//60} minutes with "
                          f"{len(empty) - len(recovered)} symbols outstanding")
                    break
                found, bs = _query_one(bs, code, start, end, adjust, attempts=4)
                if found:
                    rows.extend(found)
                    recovered.append(code)
        finally:
            bs.logout()
        empty = [code for code in empty if code not in set(recovered)]
        covered = len(codes) - len(empty)
        print(f"swept: recovered {len(recovered)}, still empty {len(empty)}"
              + (f", e.g. {empty[:8]}" if empty else ""))
    coverage = covered / float(len(codes))
    if coverage < MINIMUM_COVERAGE:
        # Publishing from a partial universe would publish a name selected from whichever
        # symbols happened to answer. Refusing is the only honest option.
        raise SystemExit(
            f"symbol coverage {coverage:.3%} is below the {MINIMUM_COVERAGE:.1%} floor; "
            f"refusing to select a name from an incomplete universe"
        )
    frame = pd.DataFrame(rows)
    for column in ("open", "high", "low", "close", "volume", "amount", "trade_status"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_st"] = pd.to_numeric(frame["is_st"], errors="coerce").fillna(0).gt(0)
    return frame.dropna(subset=["close"])


def index_frames(start: str, end: str) -> dict[str, pd.Series]:
    bs = _login()
    out = {}
    try:
        for code in INDEX_CODES:
            result = bs.query_history_k_data_plus(
                code, "date,open", start_date=start, end_date=end, frequency="d")
            dates, opens = [], []
            while result.error_code == "0" and result.next():
                day, value = result.get_row_data()
                dates.append(pd.Timestamp(day))
                opens.append(float(value))
            out[code] = pd.Series(opens, index=pd.DatetimeIndex(dates)).sort_index()
    finally:
        bs.logout()
    return out


def eligibility_funnel(panel: dict[str, pd.DataFrame], rules: dict, as_of: pd.Timestamp) -> dict:
    """Names surviving each condition on one session, so a disagreement can be located.

    Two faithful implementations of v3 differed by 614 names and nothing in the record said
    which condition dropped them. Printing the funnel makes the next disagreement a one-line
    diagnosis instead of an investigation.
    """
    close, amount = panel["close"], panel["amount"]
    observed = close.notna() & close.gt(0.0)
    window = int(rules.get("trailing_amount_window", 60))
    trailing = amount.rolling(window, min_periods=max(2, window // 3)).median().shift(1)
    steps = {
        "listed_in_panel": observed,
        "+traded_today": observed & panel["volume"].fillna(0).gt(0) & amount.fillna(0).gt(0),
        "+seasoned": observed.cumsum().shift(1).ge(int(rules.get("minimum_prior_observations", 0))),
        "+liquid": trailing.ge(float(rules.get("minimum_trailing_median_amount", 0.0))),
        "+not_st": ~panel["is_st"].fillna(False).astype(bool),
    }
    running, funnel = None, {}
    for label, mask in steps.items():
        running = mask.loc[as_of] if running is None else (running & mask.loc[as_of])
        funnel[label] = int(running.sum())
    return funnel


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_state(spec: dict, workers: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    source = spec["data_source"]
    end = date.today()
    start = end - timedelta(days=int(int(source["sessions"]) * 1.55))
    codes, listing_day = universe_codes(end, tuple(source["boards"]))
    print(f"universe: {len(codes)} codes as of {listing_day}; window {start} .. {end}")
    prices = fetch_prices(codes, str(start), str(end), str(source["adjust_flag"]), workers)
    panel = apply_sealed_bar_limits(
        build_panel(prices, EMPTY_STATEMENTS, {"require_neutralization_data": False}))
    eligible = point_in_time_eligibility(panel, **spec["universe"])
    signal = composite_signal(spec, panel, eligible)
    as_of = signal.index.max()
    funnel = eligibility_funnel(panel, spec["universe"], as_of)
    print(f"eligibility funnel on {as_of.date()}: {funnel}")
    quoting = funnel["listed_in_panel"] / float(len(codes))
    if quoting < MINIMUM_SESSION_COVERAGE:
        raise SystemExit(
            f"only {funnel['listed_in_panel']}/{len(codes)} listed symbols quote on "
            f"{as_of.date()} ({quoting:.1%}); the panel is incomplete, refusing to rank"
        )
    return panel, eligible, signal


def publish_pick(spec: dict, panel: dict, eligible: pd.DataFrame, signal: pd.DataFrame,
                 dry_run: bool) -> int:
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
        "computed_by": "github-actions",
        "status": "research_only_forward_record_not_trading",
        "orders": [],
    }
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    if dry_run:
        print("dry run: nothing written")
        return 0

    prior = next((row for row in read_jsonl(PICKS) if row.get("as_of") == entry["as_of"]
                  and row.get("spec_sha256") == entry["spec_sha256"]), None)
    if prior:
        # A published pick is never rewritten. A recomputation that disagrees with one already
        # on the record is a fact about reproducibility and has to stay visible.
        if prior.get("symbol") != entry["symbol"]:
            print(f"DISAGREEMENT: {entry['as_of']} published as {prior['symbol']}, "
                  f"recomputed as {entry['symbol']}")
            return 2
        print(f"{entry['as_of']} already published as {prior['symbol']}; reproduced exactly")
        return 0
    PICKS.parent.mkdir(parents=True, exist_ok=True)
    with PICKS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    LATEST.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"published {entry['as_of']} -> hold {entry['symbol']} from the next open")
    return 0


def append_records(spec: dict, panel: dict, eligible: pd.DataFrame, signal: pd.DataFrame,
                   dry_run: bool) -> int:
    """Score published picks whose holding window has elapsed, and append them once each.

    Only picks that were actually published are scored. Recomputing what the rule *would* have
    said today and scoring that instead would quietly turn the forward record back into a
    backtest.
    """
    open_, sessions = panel["open"], list(panel["close"].index)
    position = {session: index for index, session in enumerate(sessions)}
    limit_up, limit_down = sealed_bar_limits(panel)
    hold, cost = int(spec["holding_days"]), float(spec["round_trip_cost"])
    known = {row["as_of"] for row in read_jsonl(SERIES)}
    indices = index_frames(str(sessions[0].date()), str(sessions[-1].date()))

    # Only this specification's own picks, and one row per session. 2026-08-07 was published
    # twice — once by the retired v3 record and once by v4 — and both matured into the same
    # trade, which was scored twice and would have compounded twice on the very first live day.
    # The prediction log is append-only and keeps both; the scored series is derived, and a
    # derived series that counts one trade twice is simply wrong.
    digest = spec["spec_sha256"]
    picks = [row for row in read_jsonl(PICKS) if row.get("spec_sha256") == digest]
    picks = sorted({row["as_of"]: row for row in picks}.values(), key=lambda row: row["as_of"])

    fresh, previous, seen = [], None, set(known)
    for pick in picks:
        as_of = pd.Timestamp(pick["as_of"])
        index = position.get(as_of)
        if index is None or index + hold + 1 >= len(sessions):
            continue
        symbol = pick["symbol"]
        entry_bar, exit_bar = sessions[index + 1], sessions[index + 1 + hold]
        was = previous
        previous = symbol
        if pick["as_of"] in seen or symbol not in open_.columns:
            continue
        seen.add(pick["as_of"])
        if bool(limit_up.loc[entry_bar, symbol]) or bool(limit_down.loc[entry_bar, symbol]):
            print(f"{pick['as_of']}: {symbol} sealed at the limit on {entry_bar.date()}; not bought")
            continue
        entry_price, exit_price = open_.loc[entry_bar, symbol], open_.loc[exit_bar, symbol]
        if not (np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price > 0):
            continue
        gross = float(exit_price / entry_price - 1.0)
        window = (open_.loc[exit_bar] / open_.loc[entry_bar].replace(0.0, np.nan) - 1.0)
        window = window.where(eligible.loc[as_of]).replace([np.inf, -np.inf], np.nan).dropna()
        benchmark = float(window.mean()) if len(window) else float("nan")
        rotated = was is not None and was != symbol
        row = {
            "date": str(exit_bar.date()), "as_of": pick["as_of"], "phase": "live",
            "spec_sha256": pick["spec_sha256"][:16], "symbol": symbol,
            "strategy_gross": round(gross, 6), "universe": round(benchmark, 6),
            "cost": round(cost if rotated else 0.0, 6),
            "strategy_net": round(gross - benchmark - (cost if rotated else 0.0), 6),
            "rotated": rotated,
        }
        for code in INDEX_CODES:
            series = indices.get(code)
            if series is None:
                continue
            span = series.loc[(series.index > entry_bar) & (series.index <= exit_bar)]
            row[code.replace(".", "_")] = (
                round(float(span.iloc[-1] / series.loc[entry_bar] - 1.0), 6)
                if len(span) and entry_bar in series.index else None
            )
        fresh.append(row)

    if not fresh:
        print("no newly matured session to record")
        return 0
    print(f"appending {len(fresh)} live session(s): {[row['date'] for row in fresh]}")
    if dry_run:
        return 0
    with SERIES.open("a", encoding="utf-8") as handle:
        for row in fresh:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = load_spec(SPEC_PATH)
    panel, eligible, signal = build_state(spec, args.workers)
    status = publish_pick(spec, panel, eligible, signal, args.dry_run)
    if status not in (0,):
        return status
    return append_records(spec, panel, eligible, signal, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
