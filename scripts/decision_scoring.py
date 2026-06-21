"""Decision Scoring System (minimal, non-invasive, RECORD-ONLY).

Every ETF decision -- BUY / SELL / HOLD / SKIP -- gets an auditable `decision_score`
at the moment it is made: a 0-100 score built from interpretable sub-scores, each with
a written reason, plus data/execution audit flags. Later, outcome fields (returns, MAE,
MFE, mistake_type) are attached so we can ask: do high-score decisions actually do
better? which sub-score predicts? which is noise?

This is NOT alpha, NOT an auto-trading signal, and MUST NOT drive live sizing/execution.
It is a measurement/replay layer to validate whether the scoring logic has predictive
power. With insufficient samples, NO definitive conclusion is allowed (see the report).

Pure functions + JSONL/CSV writers; no broker calls, no order side effects. Designed to
be called from a guarded hook in run_t0_intraday_agent / replay (write-only).
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT, as_float

OUT_DIR = ROOT / "outputs" / "decision_scores"
VERSION_REGISTRY = ROOT / "configs" / "research" / "decision_score_version_registry.json"

# (min, max) for each sub-score; risk_penalty is NEGATIVE.
SCORE_RANGES: dict[str, tuple[float, float]] = {
    "market_regime_score": (0, 20),
    "relative_strength_score": (0, 20),
    "liquidity_score": (0, 15),
    "entry_quality_score": (0, 15),
    "execution_score": (0, 10),
    "counterfactual_score": (0, 10),
    "risk_penalty": (-20, 0),
}

# Field order for the CSV (decision fields, then outcome fields).
DECISION_FIELDS = [
    "decision_id", "date", "timestamp", "etf_code", "etf_name", "decision_type",
    "iteration_id", "sample_origin", "scorer_version", "weights_version",
    "outcome_model_version", "pipeline_version", "shadow_candidate_version",
    "scorer_sha256", "config_sha256",
    "signal_name", "market_regime",
    "market_regime_score", "relative_strength_score", "liquidity_score",
    "entry_quality_score", "risk_penalty", "execution_score", "counterfactual_score",
    "total_score", "score_bucket",
    "position_size_suggestion", "decision_reason", "risk_notes", "counterfactual_reason",
    "market_regime_reason", "relative_strength_reason", "liquidity_reason",
    "entry_quality_reason", "risk_penalty_reason", "execution_reason",
    "data_quality_flag", "execution_model_flag",
]
OUTCOME_FIELDS = [
    "fill_status", "entry_price", "exit_price", "exit_reason", "holding_period",
    "return_1d", "return_3d", "return_5d", "return_10d", "realized_return",
    "max_adverse_excursion", "max_favorable_excursion", "was_profitable", "was_stopped",
    "counterfactual_return", "mistake_type", "post_review_comment",
]
ALL_FIELDS = DECISION_FIELDS + OUTCOME_FIELDS


def active_version_metadata() -> dict[str, Any]:
    try:
        registry = json.loads(VERSION_REGISTRY.read_text(encoding="utf-8"))
        active = registry.get("active", {}) if isinstance(registry, dict) else {}
    except Exception:
        active = {}
    return {
        "iteration_id": active.get("iterationId") or "DSI-UNREGISTERED",
        "scorer_version": active.get("scorerLogicVersion") or "DSCORE-UNREGISTERED",
        "weights_version": active.get("weightsVersion") or "DWEIGHTS-UNREGISTERED",
        "outcome_model_version": active.get("outcomeModelVersion") or "DOUTCOME-UNREGISTERED",
        "pipeline_version": active.get("pipelineVersion") or "DPIPE-UNREGISTERED",
        "shadow_candidate_version": active.get("shadowCandidateVersion"),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def config_fingerprint(cfg: dict[str, Any]) -> str:
    stable = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_bucket(total: float) -> str:
    """A:85-100 B:70-85 C:55-70 D:40-55 E:0-40 (upper bound exclusive, A includes 100)."""
    if total >= 85:
        return "A"
    if total >= 70:
        return "B"
    if total >= 55:
        return "C"
    if total >= 40:
        return "D"
    return "E"


def total_from_subscores(sub: dict[str, Any]) -> float:
    """Sum the 7 components (risk_penalty negative), clamp to [0, 100]."""
    raw = sum(as_float(sub.get(k), 0.0) for k in SCORE_RANGES)
    return round(clamp(raw, 0, 100), 2)


def compute_audit_flags(ctx: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Return (data_quality_flag, execution_model_flag, problems).

    Any serious data problem -> data_quality_flag cannot be 'clean'. same_snapshot_fill or
    cost-not-in-path -> execution_model_flag 'contaminated'. optimistic execution model ->
    'optimistic'. Only sample-insufficient/diagnostic -> 'diagnostic_only'."""
    problems: list[str] = []
    if ctx.get("used_full_day_amount"):
        problems.append("full_day_amount_lookahead")
    if ctx.get("missing_price"):
        problems.append("missing_price")
    if ctx.get("t0_t1_handled") is False:
        problems.append("t0_t1_unhandled")
    if ctx.get("marked_to_market") is False:
        problems.append("not_marked_to_market")
    if ctx.get("cost_in_path") is False:
        problems.append("cost_not_in_path")
    if ctx.get("used_contaminated_backtest"):
        problems.append("contaminated_backtest_source")

    data_quality_flag = "contaminated" if problems else (
        "diagnostic_only" if ctx.get("diagnostic_only") else "clean")

    exec_problems: list[str] = []
    if ctx.get("same_snapshot_fill"):
        exec_problems.append("same_snapshot_fill")
    if ctx.get("cost_in_path") is False:
        exec_problems.append("cost_not_in_path")
    if exec_problems:
        execution_model_flag = "contaminated"
    elif ctx.get("execution_optimistic"):
        execution_model_flag = "optimistic"
    else:
        execution_model_flag = "clean"
    return data_quality_flag, execution_model_flag, problems + exec_problems


# --- sub-score helpers: each returns (score, reason) -------------------------------

def _market_regime(ctx: dict[str, Any]) -> tuple[float, str]:
    # Prefer cross-sectional BREADTH (fraction of names up) -- always available at decision
    # time, independent of whether the correlation-stress gate is enabled in the config.
    breadth = ctx.get("market_breadth_up_frac")
    if breadth is not None:
        breadth = float(breadth)
        if breadth >= 0.60:
            return 18.0, f"strong breadth: {breadth*100:.0f}% of ranked names up -> supportive"
        if breadth >= 0.45:
            return 12.0, f"mixed breadth: {breadth*100:.0f}% up -> neutral"
        if breadth >= 0.30:
            return 7.0, f"soft breadth: {breadth*100:.0f}% up -> cautious"
        return 3.0, f"weak breadth: {breadth*100:.0f}% up -> risk-off"
    regime = str(ctx.get("market_regime") or "unknown")
    corr_ok = ctx.get("correlation_stress_ok")
    if regime == "trend_up":
        return 17.0, f"regime={regime}; supportive"
    if ctx.get("broad_market_not_declining") is False or corr_ok is False:
        return 4.0, f"regime={regime}; weak/stressed market"
    return 11.0, f"regime={regime}; no breadth signal -> neutral default"


def _relative_strength(ctx: dict[str, Any]) -> tuple[float, str]:
    pct = ctx.get("cross_sectional_percentile")  # 0 = strongest, 1 = weakest (rank/N)
    if pct is None:
        return 9.0, "no cross-sectional rank available; neutral RS"
    pct = float(pct)
    if pct <= 0.20:
        return 18.0, f"top {pct*100:.0f}% by cross-sectional strength"
    if pct <= 0.50:
        return 13.0, f"top {pct*100:.0f}% (mid-strong) cross-sectionally"
    return 6.0, f"bottom {(1-pct)*100:.0f}% cross-sectionally (weak)"


def _liquidity(ctx: dict[str, Any]) -> tuple[float, str]:
    amount = as_float(ctx.get("amount"), 0.0)
    spread = ctx.get("spread_pct")
    if amount <= 0 or spread is None:
        return 2.0, "missing turnover/spread -> liquidity uncertain"
    spread = float(spread)
    if amount >= 100_000_000 and spread <= 0.0015:
        return 14.0, f"deep turnover ({amount/1e8:.1f}e8) and tight spread ({spread*1e4:.0f}bp)"
    if amount >= 30_000_000 and spread <= 0.004:
        return 9.0, f"adequate turnover ({amount/1e8:.2f}e8), spread {spread*1e4:.0f}bp"
    return 4.0, f"thin turnover ({amount/1e8:.2f}e8) or wide spread ({spread*1e4:.0f}bp)"


def _entry_quality(ctx: dict[str, Any]) -> tuple[float, str]:
    dt = str(ctx.get("decision_type") or "").upper()
    reason = str(ctx.get("decision_reason") or "")
    if dt in ("SELL", "HOLD", "SKIP"):
        # score the EXIT / wait quality, not entry
        if dt == "SELL":
            return (11.0, f"exit on {reason or 'sell_score'}") if "score" in reason or "stop" in reason \
                else (8.0, f"exit: {reason or 'sell'}")
        return 7.0, f"{dt.lower()}: {reason or 'no qualifying entry / waiting'}"
    # BUY: entry quality by signal type
    if any(k in reason for k in ("pullback", "consolidation", "回踩", "low")):
        return 13.0, f"constructive/pullback entry: {reason}"
    if any(k in reason for k in ("orb_breakout", "momentum", "bollinger")):
        return 9.0, f"standard momentum/breakout entry: {reason}"
    conv = ctx.get("alpha101_conviction")
    if conv is not None and float(conv) >= 0.6:
        return 11.0, f"high intraday conviction {float(conv):.2f}"
    return 5.0, f"weak/late entry signal: {reason or 'n/a'}"


def _execution(ctx: dict[str, Any]) -> tuple[float, str]:
    style = str(ctx.get("execution_style") or "")
    spread = ctx.get("spread_pct")
    has_stop = ctx.get("has_stop")
    if ctx.get("same_snapshot_fill"):
        # replay fills at the decision snapshot: execution is genuinely NOT assessable,
        # so score it NEUTRAL (not a floor) -- otherwise it just lowers every backtest score.
        return 5.0, "execution not assessable under same-snapshot replay fills (neutral)"
    if spread is not None and float(spread) <= 0.0015 and (style == "passive" or has_stop):
        return 9.0, f"clean fillability ({style or 'n/a'}), tight spread, stop defined"
    if spread is not None and float(spread) <= 0.004:
        return 6.0, f"moderate execution ({style or 'n/a'}), spread {float(spread)*1e4:.0f}bp"
    return 3.0, "execution unclear or data-limited"


def _counterfactual(ctx: dict[str, Any]) -> tuple[float, str]:
    dt = str(ctx.get("decision_type") or "").upper()
    # heuristic proxy for "would not trading be better": strong, well-ranked entries score high
    rs = ctx.get("cross_sectional_percentile")
    if dt in ("HOLD", "SKIP"):
        return 6.0, "no trade taken; counterfactual ~neutral (waiting preserves optionality)"
    if rs is not None and float(rs) <= 0.20:
        return 8.0, "skipping a top-ranked candidate would likely forgo a real opportunity"
    if rs is not None and float(rs) > 0.5:
        return 3.0, "weak candidate; not trading is plausibly better"
    return 5.0, "counterfactual roughly neutral on available evidence"


def _risk_penalty(ctx: dict[str, Any]) -> tuple[float, str]:
    penalty = 0.0
    notes: list[str] = []
    held_same_sector = int(as_float(ctx.get("held_same_sector_count"), 0))
    if held_same_sector >= 2:
        penalty -= 6; notes.append(f"theme concentration ({held_same_sector} same-sector held)")
    conv = ctx.get("alpha101_conviction")
    chg = as_float(ctx.get("change_pct"), 0.0)
    if chg >= 5.0:
        penalty -= 5; notes.append(f"extended (+{chg:.1f}% on day) -> chase risk")
    atr = as_float(ctx.get("atr_pct"), 0.0)
    if atr >= 0.02:
        penalty -= 4; notes.append(f"high intraday volatility (atr {atr*100:.1f}%)")
    if ctx.get("execution_optimistic") or ctx.get("same_snapshot_fill"):
        penalty -= 5; notes.append("optimistic execution model")
    if ctx.get("used_full_day_amount") or ctx.get("missing_price"):
        penalty -= 6; notes.append("data contamination/lookahead")
    if ctx.get("broad_market_not_declining") is False:
        penalty -= 4; notes.append("weak broad market")
    penalty = clamp(penalty, -20, 0)
    return penalty, ("; ".join(notes) if notes else "no material risk flags")


def score_decision(ctx: dict[str, Any]) -> dict[str, Any]:
    """Build a full decision_score record from a flat context dict. Outcome fields are
    null until enriched later (replay/post-review)."""
    subs: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for key, fn in (
        ("market_regime_score", _market_regime), ("relative_strength_score", _relative_strength),
        ("liquidity_score", _liquidity), ("entry_quality_score", _entry_quality),
        ("execution_score", _execution), ("counterfactual_score", _counterfactual),
        ("risk_penalty", _risk_penalty),
    ):
        score, reason = fn(ctx)
        lo, hi = SCORE_RANGES[key]
        subs[key] = round(clamp(score, lo, hi), 2)
        reasons[key] = reason or "n/a"
    total = total_from_subscores(subs)
    dq_flag, ex_flag, _problems = compute_audit_flags(ctx)
    rec = {
        "decision_id": ctx.get("decision_id"),
        "date": ctx.get("date"), "timestamp": ctx.get("timestamp"),
        "etf_code": ctx.get("etf_code"), "etf_name": ctx.get("etf_name"),
        "decision_type": str(ctx.get("decision_type") or "").upper() or "SKIP",
        "iteration_id": ctx.get("iteration_id"),
        "sample_origin": ctx.get("sample_origin"),
        "scorer_version": ctx.get("scorer_version"),
        "weights_version": ctx.get("weights_version"),
        "outcome_model_version": ctx.get("outcome_model_version"),
        "pipeline_version": ctx.get("pipeline_version"),
        "shadow_candidate_version": ctx.get("shadow_candidate_version"),
        "scorer_sha256": ctx.get("scorer_sha256"),
        "config_sha256": ctx.get("config_sha256"),
        "signal_name": ctx.get("signal_name"), "market_regime": ctx.get("market_regime"),
        **subs,
        "total_score": total, "score_bucket": score_bucket(total),
        "position_size_suggestion": ctx.get("position_size_suggestion"),
        "decision_reason": ctx.get("decision_reason"),
        "risk_notes": reasons["risk_penalty"],
        "counterfactual_reason": reasons["counterfactual_score"],
        "market_regime_reason": reasons["market_regime_score"],
        "relative_strength_reason": reasons["relative_strength_score"],
        "liquidity_reason": reasons["liquidity_score"],
        "entry_quality_reason": reasons["entry_quality_score"],
        "risk_penalty_reason": reasons["risk_penalty"],
        "execution_reason": reasons["execution_score"],
        "data_quality_flag": dq_flag, "execution_model_flag": ex_flag,
    }
    for f in OUTCOME_FIELDS:
        rec[f] = None
    return rec


def classify_mistake(rec: dict[str, Any]) -> str:
    """Best-effort attribution from score + outcome. Returns UNKNOWN when undecidable."""
    realized = rec.get("realized_return")
    if realized is None:
        return "UNKNOWN"
    realized = float(realized)
    total = as_float(rec.get("total_score"), 0.0)
    rs = as_float(rec.get("relative_strength_score"), 0.0)
    liq = as_float(rec.get("liquidity_score"), 0.0)
    eq = as_float(rec.get("entry_quality_score"), 0.0)
    risk_pen = as_float(rec.get("risk_penalty"), 0.0)
    mfe = rec.get("max_favorable_excursion")
    if realized < 0:
        if bool(rec.get("was_stopped")) is False and mfe is not None and float(mfe) > 0.005:
            return "EXIT_FAILED"
        if liq <= 5:
            return "LIQUIDITY_TRAP"
        if rs >= 16:
            return "RS_FALSE_STRENGTH"
        if eq <= 6:
            return "ENTRY_TOO_LATE"
        if risk_pen > -4:
            return "RISK_UNDERPENALIZED"
        if total >= 70:
            return "GOOD_TRADE_BAD_OUTCOME"
        return "UNKNOWN"
    # profitable
    if total < 55:
        return "BAD_TRADE_LUCKY_PROFIT"
    return "UNKNOWN"


def _minute_of(timestamp: str) -> int | None:
    try:
        hh, mm = str(timestamp).split(":")[:2]
        return int(hh) * 60 + int(mm)
    except Exception:
        return None


def build_price_index(quotes_path: Path) -> tuple[dict, dict]:
    """From a minute-quote JSONL build code -> {date -> {minute: price}} and code ->
    {date -> close}. Used to compute forward outcomes for scored decisions."""
    by: dict[str, dict[str, dict[int, float]]] = {}
    root = Path(quotes_path)
    sources = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    if root.is_dir() and root.name == "t0_intraday_agent":
        sources = sorted(root.glob("minute_quotes_*.jsonl"))
    for source in sources:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    q = json.loads(line)
                except Exception:
                    continue
                code = str(q.get("stockCode", "")).zfill(6)
                ts = str(q.get("timestamp", ""))
                price = as_float(q.get("currentPrice"), 0.0)
                if not code or len(ts) < 16 or price <= 0:
                    continue
                minute = _minute_of(ts[11:16])
                if minute is None:
                    continue
                by.setdefault(code, {}).setdefault(ts[:10], {})[minute] = price
    closes: dict[str, dict[str, float]] = {}
    for code, days in by.items():
        closes[code] = {d: mm[max(mm)] for d, mm in days.items() if mm}
    return by, closes


def enrich_from_quotes(records: list[dict[str, Any]], quotes_path: Path) -> list[dict[str, Any]]:
    """Attach outcomes using only prices after the decision timestamp.

    BUY/SELL receive a directional realized return. HOLD/SKIP are not synthetic trades;
    their raw next-snapshot-to-close move is kept separately as counterfactual_return.
    """
    by, closes = build_price_index(quotes_path)
    return enrich_from_price_index(records, by, closes)


def enrich_from_price_index(records: list[dict[str, Any]], by: dict, closes: dict) -> list[dict[str, Any]]:
    """Attach outcomes from a prebuilt index so the daily job scans quote history once."""
    for r in records:
        code, date = str(r.get("etf_code") or "").zfill(6), str(r.get("date"))
        minute = _minute_of(str(r.get("timestamp") or ""))
        daymap = by.get(code, {}).get(date)
        if not code or minute is None or not daymap:
            continue
        eligible_minutes = [m for m in sorted(daymap) if m > minute]
        entry = daymap[eligible_minutes[0]] if eligible_minutes else None
        fwd_path = [daymap[m] for m in eligible_minutes]
        if not entry or entry <= 0 or not fwd_path:
            continue
        decision_type = str(r.get("decision_type") or "").upper()
        side = 1.0 if decision_type == "BUY" else (-1.0 if decision_type == "SELL" else 0.0)
        raw_path_returns = [price / entry - 1.0 for price in fwd_path]
        r["entry_price"] = round(entry, 4)
        if side:
            signed_path = [side * value for value in raw_path_returns]
            r["max_favorable_excursion"] = round(max(signed_path), 5)
            r["max_adverse_excursion"] = round(min(signed_path), 5)
            r["realized_return"] = round(signed_path[-1], 5)
            r["was_profitable"] = r["realized_return"] > 0
        else:
            r["counterfactual_return"] = round(raw_path_returns[-1], 5)
        code_dates = sorted(closes.get(code, {}))
        if date in code_dates:
            i = code_dates.index(date)
            for nd, field in ((1, "return_1d"), (3, "return_3d"), (5, "return_5d"), (10, "return_10d")):
                if i + nd < len(code_dates):
                    r[field] = round(closes[code][code_dates[i + nd]] / entry - 1.0, 5)
        r["mistake_type"] = classify_mistake(r) if side else "NOT_A_TRADE"
    return records


def write_scores(records: list[dict[str, Any]], date: str, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """Append/write the day's decision scores to JSONL and CSV. Idempotent per call
    (overwrites the day file). Returns (jsonl_path, csv_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(date).replace("-", "")
    jsonl = out_dir / f"decision_scores_{stamp}.jsonl"
    csvp = out_dir / f"decision_scores_{stamp}.csv"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with csvp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    return jsonl, csvp


def append_score(record: dict[str, Any], date: str, out_dir: Path = OUT_DIR) -> Path:
    """Append a single decision score to the day's JSONL (used by the live/replay hook)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(date).replace("-", "")
    jsonl = out_dir / f"decision_scores_{stamp}.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl


def context_from_decision(cfg: dict[str, Any], decision: dict[str, Any], *,
                          trade_date: str, timestamp: str) -> dict[str, Any]:
    """Best-effort extraction of a scoring context from a build_decision output. Pulls the
    decision_type/reason, the selected candidate's fields, regime/liquidity, and marks the
    execution/data trust flags conservatively (replay/agent are NOT mark-to-market clean)."""
    sm = decision.get("state_machine", {}) if isinstance(decision.get("state_machine"), dict) else {}
    action = str(sm.get("action") or "hold").upper()
    decision_type = {"BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD"}.get(action, "SKIP")
    orders = decision.get("orders", []) or []
    order = next((o for o in orders if o.get("direction") == ("buy" if action == "BUY" else "sell")), None)
    ranked = decision.get("ranked", []) or []
    code = str((order or {}).get("stockCode") or (ranked[0] if ranked else {}).get("stockCode") or "").zfill(6) or None
    market_row = next(
        (quote for quote in ranked if str(quote.get("stockCode", "")).zfill(6) == code),
        ranked[0] if ranked else {},
    )
    n = len(ranked) or 1
    pos = next((i for i, q in enumerate(ranked) if str(q.get("stockCode", "")).zfill(6) == code), None)
    pct = (pos / n) if pos is not None else None
    sector = decision.get("sector_diversification", {}) if isinstance(decision.get("sector_diversification"), dict) else {}
    held_counts = sector.get("held_sector_counts", {}) if isinstance(sector.get("held_sector_counts"), dict) else {}
    # cross-sectional breadth = fraction of ranked names up (regime signal, always present)
    ups = sum(1 for q in ranked if as_float(q.get("change_pct"), 0) > 0 or as_float(q.get("momentum"), 0) > 0)
    breadth = (ups / len(ranked)) if ranked else None
    versions = active_version_metadata()
    scoring_cfg = cfg.get("decision_scoring", {}) if isinstance(cfg.get("decision_scoring"), dict) else {}
    return {
        "decision_id": f"{trade_date}_{timestamp}_{code or action}",
        "date": trade_date, "timestamp": timestamp,
        "etf_code": code, "etf_name": market_row.get("name") or (order or {}).get("name"),
        "decision_type": decision_type,
        "iteration_id": scoring_cfg.get("iteration_id") or versions["iteration_id"],
        "sample_origin": scoring_cfg.get("sample_origin") or "forward_live",
        "scorer_version": scoring_cfg.get("scorer_version") or versions["scorer_version"],
        "weights_version": scoring_cfg.get("weights_version") or versions["weights_version"],
        "outcome_model_version": scoring_cfg.get("outcome_model_version") or versions["outcome_model_version"],
        "pipeline_version": scoring_cfg.get("pipeline_version") or versions["pipeline_version"],
        "shadow_candidate_version": scoring_cfg.get("shadow_candidate_version") or versions["shadow_candidate_version"],
        "scorer_sha256": scoring_cfg.get("scorer_sha256") or versions["scorer_sha256"],
        "config_sha256": scoring_cfg.get("config_sha256") or config_fingerprint(cfg),
        "market_breadth_up_frac": breadth,
        "signal_name": sm.get("reason"), "decision_reason": sm.get("reason"),
        "market_regime": (decision.get("market_correlation_stress", {}) or {}).get("regime")
                         or ("stress" if (decision.get("market_correlation_stress", {}) or {}).get("stressed") else "neutral"),
        "broad_market_not_declining": (decision.get("market_correlation_stress", {}) or {}).get("broad_market_not_declining"),
        "correlation_stress_ok": not (decision.get("market_correlation_stress", {}) or {}).get("stressed", False),
        "cross_sectional_percentile": pct,
        "amount": market_row.get("amount"), "spread_pct": market_row.get("spread_pct"),
        "alpha101_conviction": market_row.get("alpha101_conviction"),
        "change_pct": market_row.get("change_pct"), "atr_pct": market_row.get("atr_pct"),
        "execution_style": (order or {}).get("execution_style"),
        "has_stop": bool((order or {}).get("bracket")),
        "position_size_suggestion": (order or {}).get("quantity"),
        "held_same_sector_count": max(held_counts.values()) if held_counts else 0,
        # trust flags: agent/replay fills are IDEALIZED (optimistic), and these scores are
        # diagnostic. The replay sets same_snapshot_fill=True (-> execution contaminated);
        # callers that know the data source uses full-day-amount gating should set
        # used_full_day_amount=True (-> data contaminated). We do NOT blanket-flag
        # not-marked-to-market, so the flags stay discriminating.
        "execution_optimistic": True,
        "cost_in_path": True,
        "missing_price": code is None,
        "diagnostic_only": True,
    }
