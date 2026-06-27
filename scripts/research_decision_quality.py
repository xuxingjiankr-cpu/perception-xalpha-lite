"""Decision-quality replay -- answers the three review questions on accumulated decision
scores, with DAY-CLUSTERED statistics so a favorable-market-day effect cannot masquerade
as score skill (the DSI-0008 lesson: pooled high>low was just good days, not selection).

Q1 BUY    : do high-score BUY signals beat low-score after cost? are high scores
            intercepted (capacity-blocked) less often? which interceptions were costly?
Q2 ORDERS : does the score separate good BUY / SELL *timing* (entry & exit price)?
Q3 HOLDS  : does the score separate good hold/sell timing? what should have been sold
            but was not, and how to sell at better moments?

Thresholds are data-driven tertiles (top vs bottom third of the observed score range),
not a fixed >=71, because the fixed high band is usually empty. NOT alpha, never gates
orders; diagnostic only, and refuses conclusions below MIN_DAYS distinct trading days.

Run: py -3.13 scripts/research_decision_quality.py [--scores DIR] [--label NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import numpy as np

from run_etf_paper_trading_agent import ROOT, as_float

COST = 0.0006                 # 万三 commission x2 round-trip (ETF: no stamp duty); passive fills ~no slippage
N_BOOT = 20_000
SEED = 20260622
MIN_DAYS = 10                 # below this, no directional conclusion is allowed
DEFAULT_SCORES = ROOT / "outputs" / "decision_score_pseudo_forward" / "20260506_20260618" / "scores"
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_quality"

Row = dict[str, Any]


def load_records(scores_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(scores_dir.glob("decision_scores_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _days(rows: list[Row]) -> int:
    return len({str(r.get("date")) for r in rows})


def tertile_cuts(values: list[float]) -> tuple[float, float] | None:
    """Bottom-third and top-third cut points of the observed scores."""
    clean = [v for v in values if v is not None]
    if len(clean) < 6:
        return None
    lo, hi = float(np.quantile(clean, 1 / 3)), float(np.quantile(clean, 2 / 3))
    if lo == hi:
        return None
    return lo, hi


def group_stats(rows: list[Row], value_fn: Callable[[Row], float | None]) -> dict[str, Any]:
    vals = [value_fn(r) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": len(rows), "days": _days(rows), "mean": None, "win_rate": None}
    arr = np.array(vals, dtype=float)
    return {
        "count": len(rows),
        "days": _days(rows),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "mean_mae": _avg(rows, "max_adverse_excursion"),
        "mean_mfe": _avg(rows, "max_favorable_excursion"),
    }


def _avg(rows: list[Row], field: str) -> float | None:
    vals = [as_float(r.get(field)) for r in rows if r.get(field) is not None]
    return float(mean(vals)) if vals else None


def day_paired_diff(rows: list[Row], value_fn: Callable[[Row], float | None],
                    high_pred: Callable[[Row], bool], low_pred: Callable[[Row], bool]) -> dict[str, Any]:
    """Mean within-day (high day-mean - low day-mean) over days that have both groups.
    This is the fake-high-score control: it removes the market-day component."""
    per_day: dict[str, float] = {}
    for day in sorted({str(r.get("date")) for r in rows}):
        day_rows = [r for r in rows if str(r.get("date")) == day]
        hi = [value_fn(r) for r in day_rows if high_pred(r) and value_fn(r) is not None]
        lo = [value_fn(r) for r in day_rows if low_pred(r) and value_fn(r) is not None]
        if hi and lo:
            per_day[day] = mean(hi) - mean(lo)
    return {
        "paired_days": len(per_day),
        "mean_diff": float(mean(per_day.values())) if per_day else None,
        "per_day": per_day,
    }


def cluster_bootstrap(rows: list[Row], value_fn: Callable[[Row], float | None],
                      high_pred: Callable[[Row], bool], low_pred: Callable[[Row], bool],
                      n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Resample whole DAYS (clusters), recompute pooled high-minus-low each time.

    Vectorized: precompute each day's (sum,count) of high/low values ONCE, then a
    bootstrap draw is just summing those per sampled day -- O(n_boot * n_days), not
    O(n_boot * n_rows). Pooled mean over sampled days = sum(sums)/sum(counts)."""
    days = sorted({str(r.get("date")) for r in rows})
    n = len(days)
    if n == 0:
        return {"n_boot": 0, "ci_95": [None, None], "p_diff_le_zero": None}
    idx = {d: i for i, d in enumerate(days)}
    hsum = np.zeros(n); hcnt = np.zeros(n); lsum = np.zeros(n); lcnt = np.zeros(n)
    for r in rows:
        v = value_fn(r)
        if v is None:
            continue
        i = idx[str(r.get("date"))]
        if high_pred(r):
            hsum[i] += v; hcnt[i] += 1
        elif low_pred(r):
            lsum[i] += v; lcnt[i] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))             # sampled day indices
    hs = hsum[draws].sum(1); hc = hcnt[draws].sum(1)
    ls = lsum[draws].sum(1); lc = lcnt[draws].sum(1)
    ok = (hc > 0) & (lc > 0)
    diffs = (hs[ok] / hc[ok]) - (ls[ok] / lc[ok])
    if diffs.size == 0:
        return {"n_boot": 0, "ci_95": [None, None], "p_diff_le_zero": None}
    return {
        "n_boot": int(diffs.size),
        "ci_95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "median": float(np.median(diffs)),
        "p_diff_le_zero": float(np.mean(diffs <= 0)),
    }


def high_low_block(rows: list[Row], value_fn: Callable[[Row], float | None],
                   score_fn: Callable[[Row], float | None]) -> dict[str, Any] | None:
    scored = [r for r in rows if score_fn(r) is not None and value_fn(r) is not None]
    cuts = tertile_cuts([score_fn(r) for r in scored])
    if not cuts:
        return None
    lo_cut, hi_cut = cuts
    high_pred = lambda r: score_fn(r) is not None and score_fn(r) >= hi_cut
    low_pred = lambda r: score_fn(r) is not None and score_fn(r) <= lo_cut
    high = [r for r in scored if high_pred(r)]
    low = [r for r in scored if low_pred(r)]
    hs, ls = group_stats(high, value_fn), group_stats(low, value_fn)
    pooled = (hs["mean"] - ls["mean"]) if hs["mean"] is not None and ls["mean"] is not None else None
    paired = day_paired_diff(scored, value_fn, high_pred, low_pred)
    boot = cluster_bootstrap(scored, value_fn, high_pred, low_pred)
    return {
        "score_cuts": {"low_max": round(lo_cut, 2), "high_min": round(hi_cut, 2)},
        "high": hs, "low": ls,
        "pooled_high_minus_low": pooled,
        "same_day_paired": paired,
        "day_cluster_bootstrap": boot,
        "verdict": {
            "pooled_high_better": bool(pooled is not None and pooled > 0),
            "same_day_paired_high_better": bool(paired["mean_diff"] is not None and paired["mean_diff"] > 0),
            "cluster_ci_excludes_zero": bool(boot["ci_95"][0] is not None and boot["ci_95"][0] > 0),
        },
    }


# ---- outcome accessors -------------------------------------------------------
def buy_net(r: Row) -> float | None:
    raw = r.get("realized_return")
    if raw is None:
        raw = r.get("counterfactual_return")
    return as_float(raw) - COST if raw is not None else None


def sell_quality(r: Row) -> float | None:
    # SELL realized_return is signed (+ = price fell after we sold = good exit).
    return as_float(r.get("realized_return")) if r.get("realized_return") is not None else None


def hold_to_close(r: Row) -> float | None:
    return as_float(r.get("counterfactual_return")) if r.get("counterfactual_return") is not None else None


def total_score(r: Row) -> float | None:
    return as_float(r.get("total_score")) if r.get("total_score") is not None else None


def sell_score(r: Row) -> float | None:
    for k in ("sell_score", "total_score"):
        if r.get(k) is not None:
            return as_float(r.get(k))
    return None


# ---- the three sections ------------------------------------------------------
def analyze_q1_buys(records: list[Row]) -> dict[str, Any]:
    executed = [r for r in records if str(r.get("decision_type")) == "BUY"
                and r.get("realized_return") is not None]
    candidates = [r for r in records if str(r.get("decision_type")) == "BUY_CANDIDATE"
                  and r.get("counterfactual_return") is not None]
    buys = executed + candidates
    block = high_low_block(buys, buy_net, total_score)

    interception: dict[str, Any] = {"candidate_rows_present": bool(candidates)}
    if candidates:
        cuts = tertile_cuts([total_score(r) for r in buys if total_score(r) is not None])
        if cuts:
            lo_cut, hi_cut = cuts
            def rate(group: list[Row]) -> float | None:
                n = len(group)
                return (sum(1 for r in group if str(r.get("decision_type")) == "BUY_CANDIDATE") / n) if n else None
            high_grp = [r for r in buys if total_score(r) is not None and total_score(r) >= hi_cut]
            low_grp = [r for r in buys if total_score(r) is not None and total_score(r) <= lo_cut]
            interception.update({
                "high_interception_rate": rate(high_grp),
                "low_interception_rate": rate(low_grp),
            })
        # costly interceptions: per day, did the best blocked candidate beat the trade?
        costly: list[dict[str, Any]] = []
        for day in sorted({str(r.get("date")) for r in buys}):
            traded = [buy_net(r) for r in executed if str(r.get("date")) == day and buy_net(r) is not None]
            blocked = [(buy_net(r), r) for r in candidates if str(r.get("date")) == day and buy_net(r) is not None]
            if not traded or not blocked:
                continue
            best_net, best_row = max(blocked, key=lambda x: x[0])
            traded_mean = mean(traded)
            if best_net - traded_mean > 0.003:
                costly.append({"date": day, "blocked_code": best_row.get("etf_code"),
                               "blocked_net": round(best_net, 4), "traded_mean_net": round(traded_mean, 4),
                               "gap": round(best_net - traded_mean, 4),
                               "blocked_reason": best_row.get("candidate_rejection_reason")})
        interception["costly_interception_days"] = len(costly)
        interception["worst_costly"] = sorted(costly, key=lambda x: -x["gap"])[:8]
    return {"executed_buys": len(executed), "candidates": len(candidates),
            "separation": block, "interception": interception}


def analyze_q2_orders(records: list[Row]) -> dict[str, Any]:
    exec_buys = [r for r in records if str(r.get("decision_type")) == "BUY"
                 and r.get("realized_return") is not None]
    exec_sells = [r for r in records if str(r.get("decision_type")) == "SELL"
                  and r.get("realized_return") is not None]
    return {
        "buy_timing": high_low_block(exec_buys, lambda r: as_float(r.get("realized_return")), total_score),
        "buy_n": len(exec_buys), "buy_days": _days(exec_buys),
        "sell_timing": high_low_block(exec_sells, sell_quality, total_score),
        "sell_n": len(exec_sells), "sell_days": _days(exec_sells),
        # regret = how much price rose AFTER we sold (sold-too-early magnitude)
        "sell_mean_rise_after": _neg(_avg(exec_sells, "max_adverse_excursion")),
    }


def analyze_q3_holds(records: list[Row]) -> dict[str, Any]:
    holds = [r for r in records if str(r.get("decision_type")) == "HOLD"
             and r.get("counterfactual_return") is not None]
    exec_sells = [r for r in records if str(r.get("decision_type")) == "SELL"
                  and r.get("realized_return") is not None]
    # hypothesis: a HIGH sell-side score should mean "should have sold" -> LOWER hold-to-close
    block = high_low_block(holds, hold_to_close, sell_score)
    # should-have-sold: held but fell >=0.5% into the close. Dedupe by (date, code) so the
    # same falling position counted across many snapshots is ONE position-day, not many.
    worst_by_posday: dict[tuple[str, str], Row] = {}
    for r in holds:
        v = hold_to_close(r)
        if v is None or v > -0.005:
            continue
        key = (str(r.get("date")), str(r.get("etf_code")))
        if key not in worst_by_posday or v < hold_to_close(worst_by_posday[key]):
            worst_by_posday[key] = r
    missed = list(worst_by_posday.values())
    hold_posdays = len({(str(r.get("date")), str(r.get("etf_code"))) for r in holds})
    missed_sorted = sorted(missed, key=lambda r: hold_to_close(r))[:10]
    # sold-too-early whipsaw: sold, then price rose >=0.5% after
    whips = [r for r in exec_sells if r.get("max_adverse_excursion") is not None
             and -as_float(r.get("max_adverse_excursion")) >= 0.005]
    return {
        "hold_n": len(holds), "hold_days": _days(holds),
        "score_predicts_hold_to_close": block,
        "missed_sell_count": len(missed),
        "hold_position_days": hold_posdays,
        "missed_sell_share": round(len(missed) / hold_posdays, 3) if hold_posdays else None,
        "worst_missed_sells": [{"date": r.get("date"), "code": r.get("etf_code"),
                                "hold_to_close": round(hold_to_close(r), 4),
                                "sell_score": r.get("sell_score")} for r in missed_sorted],
        "sell_whipsaw_count": len(whips),
        "sell_whipsaw_share": round(len(whips) / len(exec_sells), 3) if exec_sells else None,
    }


def _neg(v: float | None) -> float | None:
    return -v if v is not None else None


# ---- reporting ---------------------------------------------------------------
def _pct(v: float | None) -> str:
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def _block_md(title: str, block: dict[str, Any] | None) -> list[str]:
    if not block:
        return [f"**{title}**: insufficient/degenerate scores (cannot form tertiles).", ""]
    h, l = block["high"], block["low"]
    p, b, v = block["same_day_paired"], block["day_cluster_bootstrap"], block["verdict"]
    return [
        f"**{title}** (score cuts low<= {block['score_cuts']['low_max']}, high>= {block['score_cuts']['high_min']})",
        "",
        "| group | n | days | mean | win | MAE | MFE |",
        "|---|--:|--:|--:|--:|--:|--:|",
        f"| high | {h['count']} | {h['days']} | {_pct(h['mean'])} | {_pct(h.get('win_rate'))} | {_pct(h.get('mean_mae'))} | {_pct(h.get('mean_mfe'))} |",
        f"| low | {l['count']} | {l['days']} | {_pct(l['mean'])} | {_pct(l.get('win_rate'))} | {_pct(l.get('mean_mae'))} | {_pct(l.get('mean_mfe'))} |",
        "",
        f"- pooled high-minus-low: {_pct(block['pooled_high_minus_low'])}",
        f"- same-day paired ({p['paired_days']} days): {_pct(p['mean_diff'])}  ← market-day-neutral",
        f"- day-cluster 95% CI: [{_pct(b['ci_95'][0])}, {_pct(b['ci_95'][1])}]  | P(diff<=0)={_pct(b.get('p_diff_le_zero'))}",
        f"- verdict: pooled_better=`{v['pooled_high_better']}` paired_better=`{v['same_day_paired_high_better']}` "
        f"ci_excludes_zero=`{v['cluster_ci_excludes_zero']}`",
        "",
    ]


def render(result: dict[str, Any]) -> str:
    days = result["trading_days"]
    enough = days >= MIN_DAYS
    q1, q2, q3 = result["q1_buys"], result["q2_orders"], result["q3_holds"]
    lines = [
        f"# Decision-Quality Replay -- {result['label']}",
        "",
        f"Source: `{result['scores_dir']}` | records: {result['records']} | trading days: {days}",
        f"Status: **{'enough_days_but_still_cluster_gated' if enough else 'SAMPLE_INSUFFICIENT (<%d days): diagnostic only, NO conclusion' % MIN_DAYS}**",
        "",
        "All high/low splits use data-driven tertiles and report the SAME-DAY PAIRED diff "
        "and a DAY-CLUSTERED CI. A high score is only 'real' if it wins **within** a day and "
        "the clustered CI excludes zero -- otherwise it is just riding favorable market days.",
        "",
        "## Q1 -- BUY decisions: do high scores buy better, and are they intercepted less?",
        "",
    ]
    lines += _block_md("High vs low BUY signal (net of cost)", q1["separation"])
    lines.append(f"- executed buys: {q1['executed_buys']} | no-trade candidates: {q1['candidates']}")
    inter = q1["interception"]
    if not inter["candidate_rows_present"]:
        lines += ["- **interception**: this dataset has NO candidate rows "
                  "(generated before record_ranked_candidates / attribution fix). "
                  "Regenerate the replay with candidates to analyze who was blocked and why.", ""]
    else:
        lines += [
            f"- interception rate -- high scores: {_pct(inter.get('high_interception_rate'))} | "
            f"low scores: {_pct(inter.get('low_interception_rate'))}",
            f"- costly capacity interceptions (blocked high-net candidate beat the trade by >0.3%): "
            f"{inter.get('costly_interception_days')} day(s)",
        ]
        for c in inter.get("worst_costly", []):
            lines.append(f"    - {c['date']} {c['blocked_code']}: blocked {_pct(c['blocked_net'])} vs "
                         f"traded {_pct(c['traded_mean_net'])} (gap {_pct(c['gap'])}, reason={c['blocked_reason']})")
        lines.append("")
    lines += [
        "## Q2 -- planned orders: does the score separate buy/sell TIMING?",
        "",
    ]
    lines += _block_md("BUY entry timing (forward return)", q2["buy_timing"])
    lines += _block_md("SELL exit timing (+=price fell after sell=good)", q2["sell_timing"])
    lines += [f"- mean price rise AFTER our sells (sold-too-early magnitude): {_pct(q2['sell_mean_rise_after'])}", ""]
    lines += [
        "## Q3 -- holdings: does the score time exits, and what should have been sold?",
        "",
    ]
    lines += _block_md("Hold-to-close return by SELL-side score (high score should => lower hold return)",
                       q3["score_predicts_hold_to_close"])
    lines += [
        f"- 'should have sold' (held but fell >=0.5% into close, deduped by position-day): "
        f"{q3['missed_sell_count']} of {q3['hold_position_days']} position-days ({_pct(q3['missed_sell_share'])})",
    ]
    for m in q3["worst_missed_sells"][:6]:
        lines.append(f"    - {m['date']} {m['code']}: held to {_pct(m['hold_to_close'])} (sell_score={m['sell_score']})")
    lines += [
        f"- sell whipsaw (sold, then price rose >=0.5%): {q3['sell_whipsaw_count']} of {q2['sell_n']} sells "
        f"({_pct(q3['sell_whipsaw_share'])})",
        "",
        "## How to act (subject to more days)",
        "",
        "- Promote a score to a gate ONLY when same-day-paired diff is positive AND the day-cluster CI "
        "excludes zero. Pooled-only separation is the market-day trap.",
        "- Q3 levers for 'sell at better moments': separate a hard stop from timing sells; require a "
        "2-snapshot confirmation for timing sells; raise the sell threshold in strong-breadth tape; "
        "scale out (partial) then trail the rest. Validate SELL score on its own label (15m/30m/close edge).",
        "",
        "_Diagnostic only. No code path, gate, or sizing is changed by this report._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Three-question decision-quality replay.")
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--label", default="multi-day replay")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    scores_dir = Path(args.scores)
    records = load_records(scores_dir)
    result = {
        "label": args.label,
        "scores_dir": str(scores_dir),
        "records": len(records),
        "trading_days": _days(records),
        "min_days_for_conclusion": MIN_DAYS,
        "q1_buys": analyze_q1_buys(records),
        "q2_orders": analyze_q2_orders(records),
        "q3_holds": analyze_q3_holds(records),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.label.replace(" ", "_").replace("/", "_")
    (out / f"decision_quality_{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(result)
    (out / f"decision_quality_{stem}.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\noutputs: {out / ('decision_quality_' + stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
