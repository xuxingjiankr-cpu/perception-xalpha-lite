"""Daily forward accumulation for the Decision Scoring System (offline, read-only on
trading; write-only to outputs/decision_scores/).

The live agent records a decision_score per run (when decision_scoring.enabled), but
with NULL outcomes -- the future isn't known at decision time. This job runs AFTER the
close: for each day's decision_scores file that still has unenriched records, it fills
the outcome fields (returns/MAE/MFE/mistake_type) from that day's logged minute quotes,
then regenerates the effectiveness report over all accumulated days.

This is how the scorer's predictive power gets validated FORWARD (post-2026-06-21) without
re-fitting any weights on the single historical backtest. NOT alpha, never drives live
sizing; the report stays sample_insufficient until enough enriched outcomes accumulate.

Run: py -3.13 scripts/run_decision_score_daily.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_etf_paper_trading_agent import ROOT
import decision_scoring as ds
import run_decision_score_report as rep
import run_decision_probability_forward as probability_forward

SCORE_DIR = ROOT / "outputs" / "decision_scores"
AGENT_OUT = ROOT / "outputs" / "t0_intraday_agent"


def minute_quotes_for(date_compact: str) -> Path | None:
    """decision_scores_YYYYMMDD -> outputs/t0_intraday_agent/minute_quotes_YYYY-MM-DD.jsonl."""
    if len(date_compact) != 8:
        return None
    iso = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
    p = AGENT_OUT / f"minute_quotes_{iso}.jsonl"
    return p if p.exists() else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not SCORE_DIR.exists():
        print("no decision_scores yet.")
        return
    if not any(AGENT_OUT.glob("minute_quotes_*.jsonl")):
        print("no minute_quotes yet.")
        return
    price_index, closes = ds.build_price_index(AGENT_OUT)
    enriched_days = 0
    for jsonl in sorted(SCORE_DIR.glob("decision_scores_*.jsonl")):
        import json
        recs = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
        if not recs:
            continue
        deduplicated: dict[str, dict] = {}
        for index, record in enumerate(recs):
            key = str(record.get("decision_id") or f"missing_{index}")
            deduplicated[key] = record
        recs = list(deduplicated.values())
        date_compact = jsonl.stem.replace("decision_scores_", "")
        quotes = minute_quotes_for(date_compact)
        if not quotes:
            print(f"  {date_compact}: no minute_quotes to enrich from yet (skipped)")
            continue
        ds.enrich_from_price_index(recs, price_index, closes)
        iso = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
        ds.write_scores(recs, iso)
        n_out = sum(1 for r in recs if r.get("realized_return") is not None)
        n_probability = sum(1 for r in recs if r.get("probability_outcome") is not None)
        print(f"  {date_compact}: enriched {n_out} executed / {n_probability} probability outcomes / {len(recs)} records")
        enriched_days += 1

    records = rep.load_records(None)
    report = rep.build_report(records)
    from datetime import datetime
    out = SCORE_DIR / f"score_effectiveness_report_{datetime.now().strftime('%Y%m%d')}.md"
    out.write_text(report, encoding="utf-8")
    probability_forward.refresh_reports(records)
    print(f"enriched {enriched_days} day(s); report over {len(records)} decisions -> {out}")


if __name__ == "__main__":
    main()
