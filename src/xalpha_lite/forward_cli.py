"""Command-line entry point for frozen-specification forward records.

    xalpha-forward freeze --draft draft.json --spec outputs/spec.json
    xalpha-forward log    --prices prices.csv --spec outputs/spec.json --log outputs/log.jsonl
    xalpha-forward score  --prices prices.csv --spec outputs/spec.json --log outputs/log.jsonl

``freeze`` runs once. ``log`` runs on a schedule and appends the day's book. ``score`` reads
the record and reports only what has fully matured; it is safe to run at any time because it
cannot count an open position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .discovery import build_panel
from .forward import (
    append_prediction,
    build_prediction,
    freeze_spec,
    load_spec,
    score_log,
    validate_spec,
)
from .universe import apply_sealed_bar_limits, eligibility_summary, point_in_time_eligibility

DEFAULT_WINDOWS = {3, 5, 10, 20, 21, 30, 60, 120}
EMPTY_STATEMENTS = pd.DataFrame(columns=["symbol", "report_date", "notice_date", "update_date"])


def _panel(prices: Path, fundamentals: Path | None) -> dict[str, pd.DataFrame]:
    statements = pd.read_csv(fundamentals) if fundamentals else EMPTY_STATEMENTS
    panel = build_panel(
        pd.read_csv(prices), statements, {"require_neutralization_data": False}
    )
    return apply_sealed_bar_limits(panel)


def _prepared(args) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    spec = load_spec(args.spec)
    panel = _panel(args.prices, args.fundamentals)
    # A specification preregisters the windows it is allowed to use; the default set is a
    # fallback for specs that do not declare one, never a licence to widen a frozen rule.
    windows = {int(value) for value in spec.get("windows", DEFAULT_WINDOWS)}
    validate_spec(spec, set(panel), windows)
    eligible = point_in_time_eligibility(panel, **spec.get("universe", {}))
    return spec, panel, eligible


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    freeze = sub.add_parser("freeze", help="commit a specification; never overwrites")
    freeze.add_argument("--draft", required=True, type=Path)
    freeze.add_argument("--spec", required=True, type=Path)

    for name, help_text in (("log", "append today's book"), ("score", "report matured entries")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--prices", required=True, type=Path)
        command.add_argument("--spec", required=True, type=Path)
        command.add_argument("--log", required=True, type=Path)
        command.add_argument("--fundamentals", type=Path, default=None)

    args = parser.parse_args()

    if args.mode == "freeze":
        draft = json.loads(args.draft.read_text(encoding="utf-8"))
        frozen = freeze_spec(draft, args.spec)
        _emit({"frozen_at": frozen["frozen_at"], "spec_sha256": frozen["spec_sha256"],
               "spec": str(args.spec.resolve())})
        return 0

    spec, panel, eligible = _prepared(args)

    if args.mode == "log":
        entry = build_prediction(spec, panel, eligible)
        appended = append_prediction(entry, args.log)
        _emit({
            "data_as_of": entry["data_as_of"],
            "appended": appended,
            "picks": [pick["symbol"] for pick in entry["picks"]],
            "universe": eligibility_summary(eligible),
        })
        return 0

    _emit(score_log(spec, panel, args.log, eligible))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
