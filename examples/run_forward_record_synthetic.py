"""Freeze a specification, log a book, and score it — the whole forward-record contract.

Runs on synthetic prices with no drift, so the honest answer is "no edge", and the point of
the example is the machinery rather than the number. Watch four refusals in the output:

1. freezing twice raises, because a changed rule is a different rule;
2. an edited specification file fails to load, because the digest no longer matches;
3. the same session logged twice appends nothing;
4. a book still inside its holding window is counted as pending, never as a result.

Run: python examples/run_forward_record_synthetic.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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
from xalpha_lite.discovery import build_panel

SYMBOLS = [f"S{index:02d}" for index in range(30)]
DRAFT = {
    "name": "synthetic_reversal_demo",
    "rationale": "Demonstration only: a 20-day reversal on prices with no drift.",
    "factors": {
        "reversal_20": {
            "unary": "neg",
            "arg": {"rolling": "mean", "arg": {"field": "returns"}, "window": 20},
        },
        "trend_quality_60": {
            "corr": True,
            "left": {"field": "close"},
            "right": {"field": "amount"},
            "window": 60,
        },
    },
    "combination": "equal_weight_rank_average",
    "book_size": 10,
    "holding_days": 10,
    "round_trip_cost": 0.003,
    "universe": {
        "trailing_amount_window": 60,
        "minimum_trailing_median_amount": 1_000_000.0,
        "minimum_prior_observations": 120,
    },
}


def synthetic_prices(days: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(days, len(SYMBOLS))), axis=0))
    volume = rng.uniform(2e5, 9e5, size=(days, len(SYMBOLS)))
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, len(SYMBOLS)),
            "symbol": np.tile(SYMBOLS, days),
            "close": close.ravel(),
            "volume": volume.ravel(),
        }
    )
    frame["open"] = frame["close"] / (1.0 + rng.normal(0.0, 0.004, len(frame)))
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.006
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.994
    frame["amount"] = frame["close"] * frame["volume"]
    return frame


def refuses(action, *expected) -> str:
    try:
        action()
    except expected as error:
        return f"refused: {type(error).__name__}: {str(error).splitlines()[0][:88]}"
    return "NOT REFUSED — the contract is broken"


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="xalpha_forward_"))
    spec_path, log_path = workspace / "spec.json", workspace / "record.jsonl"

    panel = apply_sealed_bar_limits(
        build_panel(
            synthetic_prices(),
            pd.DataFrame(columns=["symbol", "report_date", "notice_date", "update_date"]),
            {"require_neutralization_data": False},
        )
    )
    spec = freeze_spec(DRAFT, spec_path)
    validate_spec(spec, set(panel), {3, 5, 10, 20, 30, 60, 120})
    eligible = point_in_time_eligibility(panel, **spec["universe"])

    print(f"frozen  {spec['spec_sha256'][:16]}  factors={list(spec['factors'])}")
    print(f"universe {eligibility_summary(eligible)}")
    print(refuses(lambda: freeze_spec(DRAFT, spec_path), FileExistsError))

    edited = json.loads(spec_path.read_text(encoding="utf-8"))
    edited["book_size"] = 3
    tampered = workspace / "tampered.json"
    tampered.write_text(json.dumps(edited), encoding="utf-8")
    print(refuses(lambda: load_spec(tampered), ValueError))

    sessions = list(panel["close"].index)
    for date in sessions[-40::5]:
        append_prediction(build_prediction(spec, panel, eligible, as_of=date), log_path)
    repeat = build_prediction(spec, panel, eligible, as_of=sessions[-40])
    print(f"duplicate session appended: {append_prediction(repeat, log_path)}  (must be False)")

    report = score_log(spec, panel, log_path, eligible)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(
        f"\n{report['matured_entries']} matured, {report['pending_entries']} still held. "
        f"Verdict: {report['verdict']}. Orders emitted: {report['orders']}."
    )


if __name__ == "__main__":
    main()
