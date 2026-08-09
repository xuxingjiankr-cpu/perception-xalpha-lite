"""Frozen-specification forward records.

Every gate in this library is a defence against reusing the same history twice. None of them
removes the underlying problem: once a rule has been chosen by looking at a sample, that
sample can no longer test it. The only clean test is data that did not exist when the rule
was written, which means committing to the rule first and waiting.

That is all this module does, and the discipline is entirely in what it refuses:

* :func:`freeze_spec` will not overwrite an existing specification. A changed rule is a new
  specification with a new hash — that is the point of freezing one.
* :func:`load_spec` recomputes the digest and rejects a file edited after freezing, so the
  record is tamper-evident rather than merely append-only.
* :func:`append_prediction` writes one entry per specification and session and never rewrites
  a line.
* :func:`score_log` scores only entries whose holding window has fully elapsed, so an
  in-progress position can never be counted as a result.

Factors live in the specification as DSL expressions, not as names pointing at some private
library, which makes a published record reproducible by whoever reads it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import dsl
from .book import select_names, universe_benchmark
from .universe import sealed_bar_limits

STAMP_KEYS = ("frozen_at", "spec_sha256")
RESEARCH_ONLY = "research_only_forward_record_not_trading"
MINIMUM_READABLE_SAMPLE = 60


def spec_digest(spec: dict[str, Any]) -> str:
    """SHA-256 over the rules, excluding the freezing stamp.

    The digest identifies the *rule set*, not the moment it was written down, so two people
    who commit to the same rules independently produce the same hash.
    """
    payload = {key: value for key, value in spec.items() if key not in STAMP_KEYS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_spec(
    spec: dict[str, Any], allowed_fields: set[str], allowed_windows: set[int]
) -> None:
    """Reject a specification that is incomplete, or whose factors are not evaluable."""
    required = {"name", "factors", "book_size", "holding_days", "round_trip_cost"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"specification missing keys: {sorted(missing)}")
    if not isinstance(spec["factors"], dict) or not spec["factors"]:
        raise ValueError("factors must be a non-empty mapping of name to DSL expression")
    for name, expression in spec["factors"].items():
        try:
            dsl.validate(expression, allowed_fields, allowed_windows)
        except ValueError as error:
            raise ValueError(f"factor {name!r}: {error}") from error
    if int(spec["book_size"]) < 1 or int(spec["holding_days"]) < 1:
        raise ValueError("book_size and holding_days must be positive")
    if float(spec["round_trip_cost"]) < 0.0:
        raise ValueError("round_trip_cost must not be negative")


def freeze_spec(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    """Write the specification once. An existing file is never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already holds a frozen specification; a changed rule needs a new one"
        )
    frozen = dict(spec)
    frozen.setdefault("status", RESEARCH_ONLY)
    frozen["frozen_at"] = datetime.now(timezone.utc).isoformat()
    frozen["spec_sha256"] = spec_digest(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frozen


def load_spec(path: Path) -> dict[str, Any]:
    """Load a frozen specification, refusing one whose contents no longer match its digest."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = spec.get("spec_sha256")
    if not recorded:
        raise ValueError(f"{path} carries no digest; it was not written by freeze_spec")
    actual = spec_digest(spec)
    if actual != recorded:
        raise ValueError(
            f"{path} was modified after freezing: recorded {recorded[:12]}, actual {actual[:12]}"
        )
    return spec


def composite_signal(
    spec: dict[str, Any], panel: dict[str, pd.DataFrame], eligible: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Cross-sectional rank average of the frozen factors, ranked within the eligible set."""
    ranks: list[pd.DataFrame] = []
    for expression in spec["factors"].values():
        values = dsl.evaluate(expression, panel).replace([np.inf, -np.inf], np.nan)
        if eligible is not None:
            values = values.where(eligible.reindex_like(values).fillna(False))
        ranks.append(values.rank(axis=1, pct=True))
    combination = str(spec.get("combination", "equal_weight_rank_average"))
    if combination != "equal_weight_rank_average":
        raise ValueError(f"unsupported combination: {combination}")
    return sum(ranks) / float(len(ranks))


def build_prediction(
    spec: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    eligible: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Select the book from the last available session without scoring anything."""
    signal = composite_signal(spec, panel, eligible)
    close = panel["close"]
    as_of = pd.Timestamp(as_of) if as_of is not None else signal.index.max()
    if as_of not in signal.index:
        raise KeyError(f"{as_of.date()} is not a session in the panel")
    picks = select_names(signal.loc[as_of], int(spec["book_size"]))
    if not picks:
        raise ValueError(f"no eligible names on {as_of.date()}; nothing to record")
    members = int(eligible.loc[as_of].sum()) if eligible is not None else int(close.loc[as_of].notna().sum())
    return {
        "spec_name": spec["name"],
        "spec_sha256": spec["spec_sha256"],
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "data_as_of": str(pd.Timestamp(as_of).date()),
        "eligible_names": members,
        "picks": [
            {
                "symbol": symbol,
                "composite_rank": round(float(signal.loc[as_of, symbol]), 6),
                "close": round(float(close.loc[as_of, symbol]), 4),
            }
            for symbol in picks
        ],
        "status": RESEARCH_ONLY,
        "orders": [],
    }


def read_log(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def append_prediction(entry: dict[str, Any], path: Path) -> bool:
    """Append one entry. Returns ``False`` when this session is already recorded."""
    path = Path(path)
    for existing in read_log(path):
        if (
            existing.get("data_as_of") == entry["data_as_of"]
            and existing.get("spec_sha256") == entry["spec_sha256"]
        ):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return True


def _executable_exit(
    sessions: list[pd.Timestamp], target: int, symbol: str, limit_down: pd.DataFrame
) -> pd.Timestamp | None:
    """First session at or after the target on which the position could be sold.

    A limit-down seal means the exit did not happen; scoring it at that price credits a fill
    nobody could get. Carrying the position forward keeps the loss the position actually took.
    """
    for index in range(target, len(sessions)):
        session = sessions[index]
        if not bool(limit_down.get(symbol, pd.Series(dtype=bool)).get(session, False)):
            return session
    return None


def score_log(
    spec: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    log_path: Path,
    eligible: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score only entries whose holding window has fully elapsed.

    Entry and exit both use the opening price of a session strictly after the signal, so no
    part of the return is available at decision time. Legs sealed at the limit on the entry
    session are dropped — the position was never opened — and a leg sealed limit-down at exit
    is carried to the first session it could be sold on.
    """
    horizon = int(spec["holding_days"])
    close, open_ = panel["close"], panel["open"]
    limit_up, limit_down = sealed_bar_limits(panel)
    sessions = list(close.index)
    position = {session: index for index, session in enumerate(sessions)}

    matured: list[dict[str, Any]] = []
    pending = skipped = 0
    for entry in read_log(log_path):
        if entry.get("spec_sha256") != spec.get("spec_sha256"):
            skipped += 1
            continue
        as_of = pd.Timestamp(entry["data_as_of"])
        index = position.get(as_of)
        if index is None:
            skipped += 1
            continue
        if index + horizon + 1 >= len(sessions):
            pending += 1
            continue
        entry_bar = sessions[index + 1]
        legs, unbuyable = [], 0
        for pick in entry["picks"]:
            symbol = pick["symbol"]
            if symbol not in open_.columns:
                continue
            if bool(limit_up.loc[entry_bar, symbol]) or bool(limit_down.loc[entry_bar, symbol]):
                unbuyable += 1
                continue
            exit_bar = _executable_exit(sessions, index + horizon + 1, symbol, limit_down)
            if exit_bar is None:
                continue
            entry_price = open_.loc[entry_bar, symbol]
            exit_price = open_.loc[exit_bar, symbol]
            if not (np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price > 0):
                continue
            legs.append(float(exit_price / entry_price - 1.0))
        if not legs:
            skipped += 1
            continue
        exit_bar = sessions[index + horizon + 1]
        universe = (open_.loc[exit_bar] / open_.loc[entry_bar].replace(0.0, np.nan) - 1.0)
        if eligible is not None:
            universe = universe.where(eligible.loc[as_of])
        universe = universe.replace([np.inf, -np.inf], np.nan).dropna()
        matured.append(
            {
                "data_as_of": entry["data_as_of"],
                "book": float(np.mean(legs)),
                "universe": float(universe.mean()) if len(universe) else float("nan"),
                "legs": len(legs),
                "dropped_unbuyable": unbuyable,
                "hit_rate": float(np.mean([leg > 0 for leg in legs])),
                "big_move_rate": float(np.mean([leg > 0.10 for leg in legs])),
            }
        )

    report: dict[str, Any] = {
        "spec_name": spec.get("name"),
        "spec_sha256": spec.get("spec_sha256"),
        "status": RESEARCH_ONLY,
        "matured_entries": len(matured),
        "pending_entries": pending,
        "skipped_entries": skipped,
        "orders": [],
    }
    if not matured:
        report["verdict"] = "nothing_matured_yet"
        return report

    frame = pd.DataFrame(matured)
    excess = frame["book"] - frame["universe"]
    net = excess - float(spec["round_trip_cost"])
    report.update(
        {
            "first_as_of": frame["data_as_of"].min(),
            "last_as_of": frame["data_as_of"].max(),
            "book_mean_pct": round(float(frame["book"].mean()) * 100, 3),
            "universe_mean_pct": round(float(frame["universe"].mean()) * 100, 3),
            "excess_mean_pct": round(float(excess.mean()) * 100, 3),
            "net_of_cost_mean_pct": round(float(net.mean()) * 100, 3),
            "hit_rate": round(float(frame["hit_rate"].mean()), 4),
            "big_move_rate": round(float(frame["big_move_rate"].mean()), 4),
            "dropped_unbuyable_legs": int(frame["dropped_unbuyable"].sum()),
            "verdict": (
                "insufficient_forward_sample"
                if len(frame) < MINIMUM_READABLE_SAMPLE
                else "sample_sufficient_for_first_read"
            ),
        }
    )
    return report


__all__ = [
    "append_prediction",
    "build_prediction",
    "composite_signal",
    "freeze_spec",
    "load_spec",
    "read_log",
    "score_log",
    "spec_digest",
    "universe_benchmark",
    "validate_spec",
]
