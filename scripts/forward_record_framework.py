"""Forward record v2, running on the published framework instead of private helpers.

The v1 record (``scripts/forward_record.py``, spec ``b19bbc74``) stays exactly where it is.
Its factors are names pointing at a vendored zoo, so re-expressing them as portable DSL
changes the specification digest — and a frozen record that can be restated is not a record.
v1 therefore continues untouched and v2 starts alongside it, on the same four factors, with
three properties v1 did not have:

* the factors travel *inside* the specification as DSL expressions, so anyone reading the
  published record can recompute the picks without this repository;
* the specification file is tamper-evident, not merely append-only;
* limit-sealed legs are enforced at scoring rather than only declared in the universe rules.

``verify`` proves the migration is faithful rather than approximate: it recomputes every
factor both ways on the real panel and requires the cross-sectional ranks to agree.

    python scripts/forward_record_framework.py verify
    python scripts/forward_record_framework.py freeze
    python scripts/forward_record_framework.py log
    python scripts/forward_record_framework.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from xalpha_lite.discovery import build_panel
from xalpha_lite.forward import (
    append_prediction,
    build_prediction,
    freeze_spec,
    load_spec,
    score_log,
    validate_spec,
)
from xalpha_lite.universe import (
    apply_sealed_bar_limits,
    eligibility_summary,
    point_in_time_eligibility,
)

from run_etf_paper_trading_agent import ROOT

OUT_DIR = ROOT / "outputs" / "forward_record"
BARS = ROOT / "data" / "market" / "ashare_research" / "baostock_pit_adjusted" / "bars_1d_backward_adjusted"
VENDOR = ROOT / "scripts" / "vendor" / "vibe_factors"
SPEC_PATH = OUT_DIR / "published_library_net_positive_v2.spec.json"
LOG_PATH = OUT_DIR / "published_library_net_positive_v2.predictions.jsonl"
EMPTY_STATEMENTS = pd.DataFrame(columns=["symbol", "report_date", "notice_date", "update_date"])


def _corr_close_time(window: int) -> dict:
    return {"corr": True, "left": {"field": "close"}, "right": {"field": "session_index"}, "window": window}


def _rsqr(window: int) -> dict:
    correlation = _corr_close_time(window)
    return {"binary": "mul", "left": correlation, "right": correlation}


def _illiq() -> dict:
    """Amihud (2002): trailing mean of |return| per unit of dollar volume.

    The zoo applies a cross-sectional z-score afterwards. It is a per-row positive affine
    map, so it cannot change within-row ordering, and the record ranks cross-sectionally
    anyway. ``verify`` checks that claim on the real panel rather than trusting it.
    """
    dollar_volume = {"binary": "mul", "left": {"field": "close"}, "right": {"field": "volume"}}
    impact = {
        "binary": "div",
        "left": {"unary": "abs", "arg": {"field": "returns"}},
        "right": dollar_volume,
    }
    return {"rolling": "mean", "arg": impact, "window": 21}


def _cma() -> dict:
    """Fama-French (2015) investment proxy: negative 60-day change in log average volume."""
    log_average_volume = {
        "unary": "signed_log1p",
        "arg": {"rolling": "mean", "arg": {"field": "volume"}, "window": 60},
    }
    growth = {
        "binary": "sub",
        "left": log_average_volume,
        "right": {"lag": 60, "arg": log_average_volume},
    }
    return {"unary": "neg", "arg": growth}


DRAFT_SPEC = {
    "name": "published_library_net_positive_v2",
    "predecessor": {
        "name": "published_library_net_positive_v1",
        "spec_sha256": "b19bbc74139765dd6a38609b322c4e1c820dbed2b9cdbf9fc02ed12c984d906c",
        "relationship": "same four factors, expressed portably; v1 continues unmodified",
    },
    "rationale": (
        "Four factors from a 456-candidate published-library search whose ten-name book "
        "remained net-positive after 30bps costs on the untouched test window. A deflated "
        "Sharpe test at that trial count returns consistent_with_luck, so this record exists "
        "to settle the question with data that did not exist when the rules were written."
    ),
    "factors": {
        "qlib158/rsqr60": _rsqr(60),
        "qlib158/rsqr30": _rsqr(30),
        "academic/illiq": _illiq(),
        "academic/cma": _cma(),
    },
    "windows": [21, 30, 60],
    "combination": "equal_weight_rank_average",
    "book_size": 10,
    "holding_days": 10,
    "round_trip_cost": 0.003,
    "universe": {
        "trailing_amount_window": 60,
        "minimum_trailing_median_amount": 30_000_000.0,
        "minimum_prior_observations": 120,
        "exclude_st": True,
        "require_normal_trade_status": True,
    },
    "scoring_rule": (
        "Book return from the open after the signal to the open ten sessions later, minus the "
        "eligible-universe mean over the same bars, minus one round trip at the frozen rate. "
        "Legs sealed at the limit on the entry session are dropped; a leg sealed limit-down at "
        "exit is carried to the first session it could be sold on."
    ),
}


def load_prices(sessions: int) -> pd.DataFrame:
    """BaoStock adjusted bars as the long frame ``build_panel`` expects.

    Restricted to the most recent ``sessions`` bars. A-share style turns over fast enough
    that a longer window buys stale regime rather than statistical power, and the whole panel
    at full depth does not fit comfortably in memory as a long frame.
    """
    frames = []
    for path in sorted(BARS.glob("*.jsonl")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if len(rows) < 260:
            continue
        frame = pd.DataFrame(rows)
        if not {"dt", "open", "high", "low", "close", "vol", "amount"}.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["dt"].astype(str).str[:10], errors="coerce")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").tail(sessions)
        frames.append(
            pd.DataFrame(
                {
                    "date": frame["date"],
                    "symbol": path.stem,
                    "open": pd.to_numeric(frame["open"], errors="coerce"),
                    "high": pd.to_numeric(frame["high"], errors="coerce"),
                    "low": pd.to_numeric(frame["low"], errors="coerce"),
                    "close": pd.to_numeric(frame["close"], errors="coerce"),
                    "volume": pd.to_numeric(frame["vol"], errors="coerce"),
                    "amount": pd.to_numeric(frame["amount"], errors="coerce"),
                    "is_st": pd.to_numeric(frame.get("isST", 0), errors="coerce").fillna(0).gt(0),
                    "trade_status": pd.to_numeric(frame.get("tradeStatus", 1), errors="coerce").fillna(1),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"no usable bars under {BARS}")
    combined = pd.concat(frames, ignore_index=True)
    # Each symbol contributed its own final bars, and a name delisted years ago ends on a
    # different date than an active one. Trimming to the last N sessions of the *market*
    # calendar is what keeps the panel a panel; trimming per symbol leaves a long sparse
    # union in which most sessions hold a handful of names.
    calendar = np.sort(combined["date"].unique())[-int(sessions):]
    return combined[combined["date"].isin(calendar)].reset_index(drop=True)


def prepare(sessions: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panel = apply_sealed_bar_limits(
        build_panel(load_prices(sessions), EMPTY_STATEMENTS, {"require_neutralization_data": False})
    )
    eligible = point_in_time_eligibility(panel, **DRAFT_SPEC["universe"])
    return panel, eligible


def zoo_signal(key: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import importlib

    zoo, name = key.split("/")
    return importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(panel)


def cmd_verify(args) -> int:
    from xalpha_lite import dsl

    panel, eligible = prepare(args.sessions)
    validate_spec(DRAFT_SPEC, set(panel), {int(w) for w in DRAFT_SPEC["windows"]})
    print(f"panel: {panel['close'].shape[0]} sessions x {panel['close'].shape[1]} symbols")
    print(f"universe: {eligibility_summary(eligible)}")

    failures = 0
    for key, expression in DRAFT_SPEC["factors"].items():
        ported = dsl.evaluate(expression, panel).replace([np.inf, -np.inf], np.nan)
        original = zoo_signal(key, panel).replace([np.inf, -np.inf], np.nan)
        both = ported.where(eligible).notna() & original.where(eligible).notna()
        rows = both.sum(axis=1).ge(20)
        left = ported.where(eligible & both).rank(axis=1, pct=True).loc[rows]
        right = original.where(eligible & both).rank(axis=1, pct=True).loc[rows]
        gap = (left - right).abs().to_numpy()
        worst = float(np.nanmax(gap)) if np.isfinite(gap).any() else float("nan")
        agree = worst < 1e-9
        failures += 0 if agree else 1
        print(
            f"  {'MATCH ' if agree else 'DIFFER'} {key:<20} rows={int(rows.sum()):>4} "
            f"cells={int(both.to_numpy().sum()):>7} max_rank_gap={worst:.2e}"
        )
    print(
        "faithful: DSL specification reproduces every v1 factor rank"
        if not failures
        else f"NOT faithful: {failures} factor(s) differ; do not freeze"
    )
    return 0 if not failures else 1


def cmd_freeze(args) -> int:
    frozen = freeze_spec(DRAFT_SPEC, SPEC_PATH)
    print(json.dumps({k: frozen[k] for k in ("name", "frozen_at", "spec_sha256")}, indent=2))
    return 0


def cmd_log(args) -> int:
    spec = load_spec(SPEC_PATH)
    panel, eligible = prepare(args.sessions)
    validate_spec(spec, set(panel), {int(w) for w in spec["windows"]})
    entry = build_prediction(spec, panel, eligible)
    appended = append_prediction(entry, LOG_PATH)
    print(json.dumps({
        "data_as_of": entry["data_as_of"],
        "appended": appended,
        "eligible_names": entry["eligible_names"],
        "picks": [pick["symbol"] for pick in entry["picks"]],
    }, indent=2))
    return 0


def cmd_score(args) -> int:
    spec = load_spec(SPEC_PATH)
    panel, eligible = prepare(args.sessions)
    report = score_log(spec, panel, LOG_PATH, eligible)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{spec['name']}.scorecard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, handler in (("verify", cmd_verify), ("freeze", cmd_freeze),
                          ("log", cmd_log), ("score", cmd_score)):
        command = sub.add_parser(name)
        command.add_argument("--sessions", type=int, default=500)
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
