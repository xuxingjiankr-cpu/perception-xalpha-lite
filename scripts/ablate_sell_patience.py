"""Sell-patience ablation -- a different sell strategy than the ones already tested this week.

Already tested and rejected: exit-price timing (bounce-before-sell: noise, TEST t 0.34-0.99
across 5 params, research_exit_bounce_60d.py) and sell-CONFIRMATION delay/breadth-gate/scale-out
(sell_logic_v2: PBO 0.77, noise, ablate_sell_logic_v2.py). Both probed HOW to time or gate a
sell once the score engine wants to fire. This probes something upstream and simpler: does the
sell engine simply fire too willingly -- too short a mandatory hold, too low a score bar -- given
this week's repeated finding that hold-to-close beats every stop tested and 56.5% of sells in
the June decision-quality review were premature ("卖飞").

Two independent, purely config-level levers (no new code -- min_hold_minutes and the
loss/profit exit_score_threshold already exist in evaluate_exit_for_code /
score_unified_sell):
  A. min_hold_minutes: baseline 10 -> 20 / 30 / 45. Blocks the TIMING sell (not hard
     stops/kill/emergency, which are unconditional) from firing before the position has had
     time to develop.
  B. exit_score_threshold (+10 / +20 on both loss and profit thresholds): requires a
     stronger score before the timing sell fires, i.e. more independent negative evidence.

Same 60d replay + purged OOS (train<=2026-05-20<test) + PBO/DSR as every other ablation this
week. Diagnostic/offline only -- writes nothing to the live config.

Run: py -3.13 scripts/ablate_sell_patience.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

from run_etf_paper_trading_agent import ROOT, as_float
import overfitting_guard as og

QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
BASE_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
OUT = ROOT / "outputs" / "sell_patience_ablation"
REPLAY = ROOT / "scripts" / "replay_t0_decisions.py"
TRAIN_END = "2026-05-20"
WHIPSAW_TH = 0.005

# variant -> {"min_hold_minutes": override, "threshold_delta": added to both loss/profit thresholds}
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "hold20": {"min_hold_minutes": 20},
    "hold30": {"min_hold_minutes": 30},
    "hold45": {"min_hold_minutes": 45},
    "thr_plus10": {"threshold_delta": 10},
    "thr_plus20": {"threshold_delta": 20},
    "hold30_thr_plus10": {"min_hold_minutes": 30, "threshold_delta": 10},
}


def _rows(scores_dir: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(scores_dir.glob("decision_scores_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                out.append(r)
    return out


def _window(rows: list[dict], which: str) -> list[dict]:
    if which == "train":
        return [r for r in rows if str(r.get("date")) <= TRAIN_END]
    if which == "test":
        return [r for r in rows if str(r.get("date")) > TRAIN_END]
    return rows


def sell_metrics(rows: list[dict]) -> dict:
    sells = [r for r in rows if str(r.get("decision_type")) == "SELL"
             and r.get("realized_return") is not None]
    holds = [r for r in rows if str(r.get("decision_type")) == "HOLD"
             and r.get("counterfactual_return") is not None]
    rise = [-as_float(r.get("max_adverse_excursion")) for r in sells
            if r.get("max_adverse_excursion") is not None]
    whips = [x for x in rise if x >= WHIPSAW_TH]
    worst: dict[tuple, float] = {}
    for r in holds:
        v = as_float(r.get("counterfactual_return"))
        if v > -WHIPSAW_TH:
            continue
        key = (str(r.get("date")), str(r.get("etf_code")))
        worst[key] = min(v, worst.get(key, 0.0))
    posdays = len({(str(r.get("date")), str(r.get("etf_code"))) for r in holds})
    return {
        "sells": len(sells),
        "whipsaw_share": round(len(whips) / len(sells), 4) if sells else None,
        "mean_rise_after": round(mean(rise), 5) if rise else None,
        "missed_sell_share": round(len(worst) / posdays, 4) if posdays else None,
    }


def run_variant(name: str, ov: dict) -> dict:
    vdir = OUT / name
    (vdir / "scores").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    strat = cfg["strategy"]
    strat["entry_logic_v2"] = {"enabled": False}   # isolate: this ablation is sell-side only
    strat.setdefault("t0_entry_eligibility", {})["enabled"] = False  # yahoo quotes have no asset_class
    if "min_hold_minutes" in ov:
        strat["min_hold_minutes"] = ov["min_hold_minutes"]
    if "threshold_delta" in ov:
        d = ov["threshold_delta"]
        strat["loss_exit_score_threshold"] = as_float(strat.get("loss_exit_score_threshold"), 70) + d
        strat["profit_exit_score_threshold"] = as_float(strat.get("profit_exit_score_threshold"), 65) + d
    cfg_path = vdir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] {ov or '(baseline)'} -> running replay ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(REPLAY), "--config", str(cfg_path), "--quotes", str(QUOTES),
         "--label", f"patience_{name}", "--decision-scores",
         "--decision-score-output-dir", str(vdir / "scores"),
         "--output-dir", str(vdir), "--output-detail", "summary"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    sp = vdir / f"patience_{name}_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    per_day = summary.get("per_day", {}) if isinstance(summary.get("per_day"), dict) else {}
    dates = sorted(per_day)
    daily_pnl = [as_float(per_day[d].get("net_pnl")) for d in dates]
    rows = _rows(vdir / "scores")
    res = {
        "name": name, "override": ov,
        "total_pnl": as_float(summary.get("total_pnl")),
        "max_drawdown": as_float(summary.get("max_drawdown")),
        "daily_pnl": daily_pnl, "dates": dates,
        "daily_sharpe": round(mean(daily_pnl) / pstdev(daily_pnl), 4) if len(daily_pnl) > 1 and pstdev(daily_pnl) else None,
        "full": sell_metrics(rows), "train": sell_metrics(_window(rows, "train")), "test": sell_metrics(_window(rows, "test")),
        "replay_ok": proc.returncode == 0,
        "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else None,
    }
    print(f"[{name}] pnl={res['total_pnl']:.0f} maxDD={res['max_drawdown']:.4f} "
          f"whipsaw full/test={res['full']['whipsaw_share']}/{res['test']['whipsaw_share']} "
          f"missed_test={res['test']['missed_sell_share']}", flush=True)
    return res


def scorecard(results: list[dict]) -> str:
    base = next((r for r in results if r["name"] == "baseline"), results[0])
    lines = [
        "# Sell-Patience Ablation -- min_hold_minutes / exit-score-threshold sweep, 60d + purged OOS + PBO/DSR",
        "",
        f"Quotes: `{QUOTES.name}` | purged split: train<= {TRAIN_END} < test | whipsaw thr {WHIPSAW_TH:.1%}",
        "",
        "Different question than the two sell probes already rejected this week (bounce-exit "
        "timing: noise; sell_logic_v2 confirmation/breadth: PBO 0.77 noise). This tests whether "
        "the sell engine simply fires too early/too easily -- longer mandatory hold, or a higher "
        "score bar before the timing sell is allowed to trigger.",
        "",
        "| variant | total PnL | maxDD | dSharpe | whipsaw full | whipsaw TEST | missed TEST |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        f, t = r["full"], r["test"]
        lines.append(
            f"| {r['name']} | {r['total_pnl']:.0f} | {r['max_drawdown']:.4f} | {r['daily_sharpe']} | "
            f"{_p(f['whipsaw_share'])} | {_p(t['whipsaw_share'])} | {_p(t['missed_sell_share'])} |")
    matrix = [r["daily_pnl"] for r in results if len(r["daily_pnl"]) == len(base["daily_pnl"]) and r["daily_pnl"]]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 else {"pbo": None}
    ranked = [r for r in results if r["name"] != "baseline" and r["total_pnl"] is not None]
    best = max(ranked, key=lambda r: r["total_pnl"], default=None)
    dsr = (og.deflated_significance_note(n_trials=len(VARIANTS) - 1,
                                         observed_sharpe=best["daily_sharpe"] or 0.0, n_obs=len(best["daily_pnl"]))
           if best and best["daily_sharpe"] is not None else {"note": "n/a"})
    lines += [
        "",
        f"- baseline total PnL {base['total_pnl']:.0f}, whipsaw full/test {_p(base['full']['whipsaw_share'])}/{_p(base['test']['whipsaw_share'])}, "
        f"missed test {_p(base['test']['missed_sell_share'])}",
        f"- **PBO** (winner persists OOS?): {pbo.get('pbo')} "
        f"({'noise: winner does NOT hold' if (pbo.get('pbo') or 0) >= 0.5 else 'holds OOS' if pbo.get('pbo') is not None else 'n/a'})",
        f"- best non-baseline (by total PnL): {best['name'] if best else 'n/a'} "
        f"(PnL {best['total_pnl']:.0f} vs baseline {base['total_pnl']:.0f})" if best else "- best: n/a",
        f"- **Deflated-Sharpe note**: {dsr}",
        "",
        "## Read",
        "",
        "A variant only advances if it improves total PnL/drawdown AND does not materially worsen "
        "missed-sell (holding a name that keeps falling) AND survives PBO. Otherwise the sell engine's "
        "trigger cadence was not the binding factor.",
        "",
        "_Offline ablation. No live config changed._",
        "",
    ]
    return "\n".join(lines)


def _p(v) -> str:
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_variant(name, ov) for name, ov in VARIANTS.items()]
    (OUT / "ablation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = scorecard(results)
    (OUT / "ablation_scorecard.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"outputs: {OUT / 'ablation_scorecard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
