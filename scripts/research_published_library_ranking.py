"""Rank the published factor libraries on the training window, report on the untouched test one.

This regenerates the table on the front page of the public repository, which until now existed
only as numbers in a commit message. A repository whose subject is reproducibility should not
carry a headline table nobody can recompute, so this exists to (a) check the published rows
still reproduce and (b) extend the table, since two of its five rows are the same factor
published twice and it therefore shows four distinct behaviours, not five.

Method, matching what the table claims. Every factor in the vendored libraries is evaluated on
a point-in-time A-share panel, ranked **only on the training window** by the mean net-of-cost
excess of a ten-name equal-weight book, then reported on the test window it never touched. Ten
sessions held, entry and exit at the open after the signal, 30 bps round trip charged on
realised turnover, legs sealed at the limit on entry dropped rather than priced, and returns
stated as an excess over the equal-weight eligible universe over the same bars.

    python scripts/research_published_library_ranking.py --split 2024-12-31 --sessions 900
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from xalpha_lite.book import select_names
from xalpha_lite.universe import sealed_bar_limits

from forward_record_framework import OUT_DIR, VENDOR, prepare

BOOK, HOLD, BIG_MOVE = 10, 10, 0.10


def factor_keys() -> list[str]:
    root = Path(VENDOR) / "src" / "factors" / "zoo"
    return sorted(
        f"{zoo.name}/{path.stem}"
        for zoo in root.iterdir() if zoo.is_dir() and not zoo.name.startswith("_")
        for path in zoo.glob("*.py") if not path.stem.startswith("_")
    )


def load_factor(key: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    zoo, name = key.split("/")
    try:
        signal = importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(panel)
    except Exception:
        return None
    if not isinstance(signal, pd.DataFrame) or signal.shape != panel["close"].shape:
        return None
    return signal.replace([np.inf, -np.inf], np.nan)


def forward_returns(panel: dict, limits) -> pd.DataFrame:
    """Open-to-open return over the holding period, NaN where the position could not be opened.

    Computed once for the whole panel rather than per factor. The per-symbol loop this replaces
    made a full 456-factor scan a seven-hour job, which is another way of saying nobody would
    ever have rerun it — and a table nobody reruns is the problem being fixed here.
    """
    open_ = panel["open"]
    entry, exit_ = open_.shift(-1), open_.shift(-(HOLD + 1))
    forward = exit_ / entry.replace(0.0, np.nan) - 1.0
    limit_up, limit_down = limits
    sealed = (limit_up | limit_down).shift(-1, fill_value=False)
    return forward.where(~sealed).replace([np.inf, -np.inf], np.nan)


def book_series(signal: pd.DataFrame, eligible: pd.DataFrame,
                forward: pd.DataFrame) -> pd.DataFrame:
    """Per-session excess of a ten-name book over the universe it was selected from.

    Rebalanced every session, so consecutive rows overlap and the mean is a per-hold figure
    rather than something that may be compounded. Costs are excluded deliberately and stated as
    such; charging realised turnover on an overlapping series is ambiguous enough that three
    defensible conventions gave three different answers, and an unambiguous gross number that
    anyone can recompute is worth more than a net one that nobody can.
    """
    masked = signal.where(eligible)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    picked = ranks.le(BOOK) & masked.notna()
    enough = picked.sum(axis=1).ge(BOOK)

    book = forward.where(picked)
    universe = forward.where(eligible)
    frame = pd.DataFrame({
        "gross": book.mean(axis=1),
        "universe": universe.mean(axis=1),
        "big_move": book.gt(BIG_MOVE).sum(axis=1) / book.notna().sum(axis=1).replace(0, np.nan),
        "universe_big_move": universe.gt(BIG_MOVE).sum(axis=1) / universe.notna().sum(axis=1).replace(0, np.nan),
    })
    frame = frame.loc[enough & frame["gross"].notna() & frame["universe"].notna()]
    frame["net"] = frame["gross"] - frame["universe"]
    return frame.reset_index(names="as_of")


def summarise(frame: pd.DataFrame) -> dict:
    """No drawdown here, on purpose.

    Consecutive rows are overlapping ten-session holds, so compounding them is not a portfolio
    path and the "max drawdown" it produces is an artefact — the first version of this printed
    -100% for a factor that merely lost money slowly. The worst single hold is well defined for
    this construction and is reported instead.
    """
    net = frame["net"]
    base = float(frame["universe_big_move"].mean())
    return {
        "n": int(len(frame)),
        "net_pct": round(float(net.mean()) * 100, 4),
        "gross_pct": round(float(frame["gross"].mean()) * 100, 4),
        "universe_pct": round(float(frame["universe"].mean()) * 100, 4),
        "big_move_ratio": round(float(frame["big_move"].mean()) / base, 3) if base else None,
        "worst_hold_pct": round(float(net.min()) * 100, 2),
        "hit_rate": round(float((net > 0).mean()), 3),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=900)
    parser.add_argument("--split", default="2024-12-31")
    parser.add_argument("--limit", type=int, default=0, help="scan only the first N factors")
    args = parser.parse_args()

    panel, eligible = prepare(args.sessions)
    forward = forward_returns(panel, sealed_bar_limits(panel))
    split = pd.Timestamp(args.split)
    sessions = panel["close"].index
    print(f"panel {len(sessions)} sessions x {panel['close'].shape[1]} symbols | "
          f"train <= {split.date()} < test | train {int((sessions <= split).sum())} "
          f"test {int((sessions > split).sum())}")

    keys = factor_keys()
    if args.limit:
        keys = keys[: args.limit]
    print(f"factors found: {len(keys)}")

    results, began, unusable = [], time.time(), 0
    for position, key in enumerate(keys, 1):
        signal = load_factor(key, panel)
        if signal is None:
            unusable += 1
            continue
        frame = book_series(signal, eligible, forward)
        if frame.empty:
            unusable += 1
            continue
        train, test = frame[frame["as_of"] <= split], frame[frame["as_of"] > split]
        if len(train) < 60 or len(test) < 60:
            unusable += 1
            continue
        results.append({"factor": key,
                        "train": summarise(train), "test": summarise(test)})
        if position % 50 == 0:
            print(f"  {position}/{len(keys)}  {time.time()-began:.0f}s  usable {len(results)}")

    # Ranked on the training window only. The test columns are untouched by this ordering.
    results.sort(key=lambda row: row["train"]["net_pct"], reverse=True)
    print(f"\nscanned {len(keys)}, usable {len(results)}, unusable {unusable}, "
          f"{time.time()-began:.0f}s\n")
    print(f"{'#':>2} {'factor':<24} {'TRAIN exc':>10} {'TEST exc':>9} {'>10% odds':>10} {'worst':>8} {'hit':>6}")
    for rank, row in enumerate(results[:12], 1):
        t = row["test"]
        print(f"{rank:>2} {row['factor']:<24} {row['train']['net_pct']:>9.3f}% "
              f"{t['net_pct']:>8.3f}% {str(t['big_move_ratio']) + 'x':>10} "
              f"{t['worst_hold_pct']:>7.1f}% {t['hit_rate']:>6.3f}")

    universe = results[0]["test"]["universe_pct"] if results else None
    print(f"\neligible universe over the test window: {universe:+.3f}% per hold")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "published_library_ranking.json").write_text(
        json.dumps({"split": str(split.date()), "sessions": int(len(sessions)),
                    "scanned": len(keys), "usable": len(results),
                    "universe_test_pct": universe, "ranking": results},
                   ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
