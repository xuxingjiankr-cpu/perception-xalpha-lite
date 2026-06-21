"""Chronological, prefix-only historical paper replay for the frozen decision scorer.

This is deliberately named ``pseudo_forward``: the engine never sees future bars while
running, but the scorer and strategy were designed after this historical period existed.
Results are therefore diagnostic and can never replace post-2026-06-21 real forward data.

No broker calls, no live config writes, no overlay writes, no order submission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from build_t0_replay_quotes_from_minute_data import build_replay_directory, passes_dynamic_gate
from run_etf_paper_trading_agent import ROOT, as_float


DEFAULT_START = "2026-05-06"
DEFAULT_END = "2026-06-18"
LIVE_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
RESEARCH_CONFIG = ROOT / "configs" / "t0_intraday_research_safe.json"
T0_POOL = ROOT / "outputs" / "edge_research" / "t0_etf_confirmed_pool_latest.jsonl"
SCORER = ROOT / "scripts" / "decision_scoring.py"
MINUTE_ROOT = ROOT / "data" / "market" / "eastmoney" / "minute"
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_score_pseudo_forward" / "20260506_20260618"
YAHOO_QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_frozen_config(live: dict[str, Any], research: dict[str, Any], *,
                        source_commit: str, scorer_sha256: str) -> tuple[dict[str, Any], str]:
    """Copy the live decision rules into an offline-only, non-promotable replay config."""
    cfg = copy.deepcopy(live)
    cfg["mode"] = "paper_research"
    cfg["execution_enabled"] = False
    cfg["governance"] = {
        "operating_mode": "pseudo_forward",
        "risk_posture": "diagnostic_only",
        "research_baseline": False,
        "contest_mode": False,
        "purpose": "chronological historical diagnostic; never live or alpha evidence",
    }
    cfg["self_iteration"] = copy.deepcopy(cfg.get("self_iteration", {}))
    cfg["self_iteration"].update({
        "enabled": False,
        "auto_apply_changes": False,
        "requires_human_approval": True,
    })
    cfg["replay_execution"] = copy.deepcopy(research.get("replay_execution", {}))
    cfg["replay_execution"]["evaluate_strategy_locks_offline"] = True
    cfg["decision_scoring"] = {
        "enabled": True,
        "record_only": True,
        "sample_origin": "pseudo_forward",
        "scorer_version": f"frozen_at_{source_commit[:12]}",
        "scorer_sha256": scorer_sha256,
        "config_sha256": None,
        "weights_refit_allowed": False,
    }
    hash_payload = copy.deepcopy(cfg)
    hash_payload["decision_scoring"].pop("config_sha256", None)
    cfg_hash = canonical_sha256(hash_payload)
    cfg["decision_scoring"]["config_sha256"] = cfg_hash
    return cfg, cfg_hash


def clear_matching(directory: Path, patterns: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def combine_quote_dirs(sources: list[Path], destination: Path) -> list[str]:
    clear_matching(destination, ("*.jsonl",))
    dates: list[str] = []
    for source in sources:
        for path in sorted(source.glob("????-??-??.jsonl")):
            target = destination / path.name
            if target.exists():
                raise RuntimeError(f"duplicate replay date: {path.name}")
            shutil.copy2(path, target)
            dates.append(path.stem)
    return sorted(dates)


def build_contaminated_yahoo_fallback(source: Path, destination: Path, *,
                                      start_date: str, end_date: str,
                                      allowed_codes: set[str], dyn_cfg: dict[str, Any]) -> dict[str, Any]:
    """Recover chronological May coverage, while explicitly preserving its contamination.

    The old Yahoo archive was constructed from a universe that used final-day turnover.
    Cumulative bar amount below is point-in-time, but the prior universe deletion cannot be
    undone; every emitted row is therefore tagged ``contaminated_full_day``.
    """
    clear_matching(destination, ("*.jsonl",))
    handles: dict[str, Any] = {}
    cumulative: dict[tuple[str, str], tuple[float, float]] = {}
    rows_by_date: Counter[str] = Counter()
    codes_by_date: dict[str, set[str]] = {}
    try:
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                timestamp = str(row.get("timestamp") or "")
                day = timestamp[:10]
                code = str(row.get("stockCode") or "").zfill(6)
                if day < start_date or day > end_date or code not in allowed_codes:
                    continue
                key = (day, code)
                prev_volume, prev_amount = cumulative.get(key, (0.0, 0.0))
                minute_volume = max(0.0, as_float(row.get("volume")))
                minute_amount = max(0.0, as_float(row.get("amount")))
                cumulative[key] = (prev_volume + minute_volume, prev_amount + minute_amount)
                row["minute_volume"] = minute_volume
                row["minute_amount"] = minute_amount
                row["volume"] = cumulative[key][0]
                row["amount"] = cumulative[key][1]
                row["source"] = "yahoo_5m_historical_fallback"
                row["liquidity_source"] = "contaminated_full_day"
                row["synthetic_order_book"] = True
                row["snapshot_interval_minutes"] = 5
                if not passes_dynamic_gate(row, dyn_cfg):
                    continue
                if day not in handles:
                    handles[day] = (destination / f"{day}.jsonl").open("w", encoding="utf-8")
                handles[day].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows_by_date[day] += 1
                codes_by_date.setdefault(day, set()).add(code)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "task": "build_contaminated_yahoo_fallback",
        "start_date": start_date,
        "end_date": end_date,
        "rows_by_date": dict(sorted(rows_by_date.items())),
        "eligible_codes_by_date": {day: len(codes) for day, codes in sorted(codes_by_date.items())},
        "rows_written": sum(rows_by_date.values()),
        "loaded_symbols": len(set().union(*codes_by_date.values())) if codes_by_date else 0,
        "configured_universe": len(allowed_codes),
        "liquidity_source": "contaminated_full_day",
        "full_day_universe_deletion_unrecoverable": True,
    }
    (destination / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else None


def load_score_records(score_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(score_dir.glob("decision_scores_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def render_report(manifest: dict[str, Any], replay: dict[str, Any],
                  scores: list[dict[str, Any]]) -> str:
    days = sorted((replay.get("per_day") or {}).keys())
    directional = [
        row for row in scores
        if str(row.get("decision_type")) in {"BUY", "SELL"}
        and row.get("realized_return") is not None
    ]
    xs = [as_float(row.get("total_score")) for row in directional]
    ys = [as_float(row.get("realized_return")) for row in directional]
    corr = pearson(xs, ys)
    daily = [as_float(replay["per_day"][day].get("net_pnl")) for day in days]
    winning_days = sum(1 for value in daily if value > 0)
    source_summaries = manifest.get("source_summaries", [])
    rows_by_date = manifest.get("rows_by_date", {})
    codes_by_date = manifest.get("eligible_codes_by_date", {})
    decision_counts = Counter(str(row.get("decision_type")) for row in scores)
    buckets: dict[str, list[float]] = {}
    for row in directional:
        buckets.setdefault(str(row.get("score_bucket")), []).append(as_float(row.get("realized_return")))

    lines = [
        "# Frozen Decision-Score Pseudo-Forward Report",
        "",
        f"Trust level: `{replay.get('result_trust_level') or manifest.get('status')}`",
        f"Sample type: `{manifest.get('sample_origin', 'pseudo_forward')}` (chronological bars, but strategy/scorer post-date the sample)",
        "Weights refit allowed: `false`",
        "",
        "## Frozen inputs",
        "",
        f"- source commit: `{manifest.get('source_commit')}`",
        f"- scorer SHA256: `{manifest.get('scorer_sha256')}`",
        f"- config SHA256: `{manifest.get('config_sha256')}`",
        f"- requested window: {manifest.get('start_date')}..{manifest.get('end_date')}",
        f"- replayed trading days: {len(days)}",
        f"- confirmed T0 universe records: {manifest.get('confirmed_t0_records')}",
        "",
        "## Portfolio replay",
        "",
        f"- gross realized P&L before path costs: {as_float(replay.get('gross_realized_pnl')):,.2f}",
        f"- total equity P&L: {as_float(replay.get('total_pnl')):,.2f}",
        f"- ending equity: {as_float(replay.get('total_equity')):,.2f}",
        f"- max drawdown: {as_float(replay.get('max_drawdown')):.4%}",
        f"- completed trades: {int(as_float(replay.get('trade_count')))}",
        f"- transaction cost: {as_float(replay.get('transaction_cost_total')):,.2f}",
        f"- winning days: {winning_days}/{len(days)}",
        f"- median daily net P&L: {median(daily):,.2f}" if daily else "- median daily net P&L: n/a",
        f"- result trust from replay: `{replay.get('result_trust_level')}`",
        "",
        "## Frozen-score diagnostic",
        "",
        f"- recorded decisions: {len(scores)} ({dict(decision_counts)})",
        f"- directional BUY/SELL outcomes: {len(directional)} across "
        f"{len({row.get('date') for row in directional})} days",
        f"- total-score/return Pearson correlation: {corr:.4f}" if corr is not None else
        "- total-score/return Pearson correlation: insufficient variation/sample",
        "",
        "| score bucket | directional outcomes | mean return |",
        "|---|---:|---:|",
    ]
    for bucket in ("A", "B", "C", "D", "E"):
        values = buckets.get(bucket, [])
        lines.append(f"| {bucket} | {len(values)} | {mean(values):.4%} |" if values else f"| {bucket} | 0 | - |")
    lines.extend([
        "",
        "## Data coverage",
        "",
        "| date | replay rows | point-in-time eligible codes |",
        "|---|---:|---:|",
    ])
    for day in sorted(rows_by_date):
        lines.append(f"| {day} | {rows_by_date[day]} | {codes_by_date.get(day, 0)} |")
    lines.extend([
        "",
        "Source builds:",
        "",
    ])
    for item in source_summaries:
        lines.append(
            f"- {item.get('start_date')}..{item.get('end_date')}: loaded "
            f"{item.get('loaded_symbols')}/{item.get('configured_universe')} symbols, "
            f"rows={item.get('rows_written')}"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- May source coverage is incomplete and expands through the month; missing instruments fail closed.",
        "- The historical order book is synthetic, so fill probability and queue position are not observed.",
        "- The current strategy and scorer were created after this sample existed; chronological replay removes row look-ahead but not researcher-selection bias.",
        "- The T0 confirmed pool is a current master, not a point-in-time May master; survivor bias remains.",
        "- HOLD/SKIP are excluded from directional score claims; they are not treated as synthetic long/short trades.",
        "- No parameter or score weight may be changed from this result. Only post-2026-06-21 real forward data can support refitting.",
        "",
        "## Verdict",
        "",
        "This run is useful for checking chronological decisions, cash/position accounting and whether the frozen score is directionally coherent. "
        "It is not clean alpha evidence and cannot promote or reweight the live strategy.",
        "",
    ])
    if manifest.get("full_day_liquidity_used"):
        lines.insert(lines.index("## Verdict"),
                     "- Yahoo fallback used a final-day-turnover-filtered universe; this entire companion run is `contaminated`.")
    return "\n".join(lines)


def execute_replay(*, frozen_path: Path, quotes_dir: Path, replay_dir: Path,
                   score_dir: Path, start_date: str, end_date: str, label: str) -> dict[str, Any]:
    clear_matching(score_dir, ("decision_scores_*.jsonl", "decision_scores_*.csv"))
    cmd = [
        sys.executable, str(ROOT / "scripts" / "replay_t0_decisions.py"),
        "--config", str(frozen_path), "--quotes", str(quotes_dir),
        "--start-date", start_date, "--end-date", end_date,
        "--label", label, "--output-detail", "full", "--decision-scores",
        "--output-dir", str(replay_dir), "--decision-score-output-dir", str(score_dir),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    replay_dir.mkdir(parents=True, exist_ok=True)
    (replay_dir / "replay.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (replay_dir / "replay.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"replay failed ({completed.returncode}); see {replay_dir / 'replay.stderr.log'}")
    return load_json(replay_dir / f"{label}_summary.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated chronological pseudo-forward ETF replay.")
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-build", action="store_true", help="reuse existing per-day quote cache")
    parser.add_argument("--include-contaminated-yahoo-fallback", action="store_true",
                        help="also run a full May companion using the known-contaminated Yahoo universe")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.output_dir).resolve()
    if ROOT.resolve() not in out.parents:
        raise RuntimeError(f"output must stay inside workspace: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_commit = current_commit()
    scorer_sha = sha256_bytes(SCORER.read_bytes())
    frozen, config_sha = build_frozen_config(
        load_json(LIVE_CONFIG), load_json(RESEARCH_CONFIG),
        source_commit=source_commit, scorer_sha256=scorer_sha,
    )
    frozen_path = out / "frozen_pseudo_forward_config.json"
    frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    quotes_dir = out / "quotes"
    source_summaries: list[dict[str, Any]] = []
    if not args.skip_build:
        may_dir = out / "build_2026-05"
        june_dir = out / "build_2026-06"
        source_summaries.append(build_replay_directory(
            config_path=frozen_path, minute_dir=MINUTE_ROOT / "2026-05" / "etf",
            universe_file=T0_POOL, output_dir=may_dir,
            start_date=args.start_date, end_date=min(args.end_date, "2026-05-31"), dynamic_gate=True,
            snapshot_interval_minutes=5,
        ))
        source_summaries.append(build_replay_directory(
            config_path=frozen_path, minute_dir=MINUTE_ROOT / "2026-06" / "etf",
            universe_file=T0_POOL, output_dir=june_dir,
            start_date=max(args.start_date, "2026-06-01"), end_date=args.end_date, dynamic_gate=True,
            snapshot_interval_minutes=5,
        ))
        dates = combine_quote_dirs([may_dir, june_dir], quotes_dir)
    else:
        dates = sorted(path.stem for path in quotes_dir.glob("????-??-??.jsonl"))
        for source in (out / "build_2026-05", out / "build_2026-06"):
            if (source / "manifest.json").exists():
                source_summaries.append(load_json(source / "manifest.json"))
    if not dates:
        raise RuntimeError("no replay quote dates were built")

    rows_by_date: dict[str, int] = {}
    codes_by_date: dict[str, int] = {}
    for summary in source_summaries:
        rows_by_date.update(summary.get("rows_by_date", {}))
        codes_by_date.update(summary.get("eligible_codes_by_date", {}))
    confirmed_t0_records = sum(1 for line in T0_POOL.read_text(encoding="utf-8").splitlines() if line.strip())
    manifest = {
        "task": "decision_score_pseudo_forward",
        "status": "diagnostic_only",
        "sample_origin": "pseudo_forward",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "dates": dates,
        "source_commit": source_commit,
        "scorer_sha256": scorer_sha,
        "config_sha256": config_sha,
        "confirmed_t0_records": confirmed_t0_records,
        "rows_by_date": dict(sorted(rows_by_date.items())),
        "eligible_codes_by_date": dict(sorted(codes_by_date.items())),
        "source_summaries": source_summaries,
        "weights_refit_allowed": False,
        "live_config_modified": False,
        "broker_calls": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    replay_dir = out / "replay"
    score_dir = out / "scores"
    replay_summary = execute_replay(
        frozen_path=frozen_path, quotes_dir=quotes_dir, replay_dir=replay_dir,
        score_dir=score_dir, start_date=args.start_date, end_date=args.end_date,
        label="pseudo_forward",
    )
    report = render_report(manifest, replay_summary, load_score_records(score_dir))
    report_path = out / "pseudo_forward_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"report: {report_path}")

    if args.include_contaminated_yahoo_fallback:
        allowed_codes = {
            str(json.loads(line).get("code") or json.loads(line).get("stockCode") or "").zfill(6)
            for line in T0_POOL.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        fallback_dir = out / "build_yahoo_may_contaminated"
        fallback_summary = build_contaminated_yahoo_fallback(
            YAHOO_QUOTES, fallback_dir, start_date=args.start_date,
            end_date=min(args.end_date, "2026-05-31"), allowed_codes=allowed_codes,
            dyn_cfg=frozen.get("dynamic_universe", {}),
        )
        full_quotes = out / "quotes_with_yahoo_may_contaminated"
        full_dates = combine_quote_dirs([fallback_dir, out / "build_2026-06"], full_quotes)
        contaminated_manifest = copy.deepcopy(manifest)
        contaminated_manifest.update({
            "status": "contaminated",
            "sample_origin": "pseudo_forward_with_contaminated_yahoo_may",
            "dates": full_dates,
            "full_day_liquidity_used": True,
            "source_summaries": [fallback_summary, source_summaries[-1]],
            "rows_by_date": {
                **fallback_summary.get("rows_by_date", {}),
                **source_summaries[-1].get("rows_by_date", {}),
            },
            "eligible_codes_by_date": {
                **fallback_summary.get("eligible_codes_by_date", {}),
                **source_summaries[-1].get("eligible_codes_by_date", {}),
            },
        })
        (out / "manifest_with_yahoo_may_contaminated.json").write_text(
            json.dumps(contaminated_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        contaminated_replay = execute_replay(
            frozen_path=frozen_path, quotes_dir=full_quotes,
            replay_dir=out / "replay_with_yahoo_may_contaminated",
            score_dir=out / "scores_with_yahoo_may_contaminated",
            start_date=args.start_date, end_date=args.end_date,
            label="pseudo_forward_contaminated",
        )
        contaminated_report = render_report(
            contaminated_manifest, contaminated_replay,
            load_score_records(out / "scores_with_yahoo_may_contaminated"),
        )
        contaminated_path = out / "pseudo_forward_yahoo_may_contaminated_report.md"
        contaminated_path.write_text(contaminated_report, encoding="utf-8")
        print(contaminated_report)
        print(f"contaminated companion report: {contaminated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
