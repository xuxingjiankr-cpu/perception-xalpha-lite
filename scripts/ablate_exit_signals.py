"""Exit-signal component ablation -- is the sell DECISION firing on anti-predictive triggers?

Motivation: 56.5% of sells in the June decision-quality review were premature; the exit-bounce
probe showed sell candidates are HIGHER 30min later more than half the time; and the strongest
validated pattern in this data is SHORT-TERM REVERSAL (drops bounce). The unified sell score's
flow components (momentum_reversal 20 + acceleration_negative 15 + bid_pressure_negative 15 =
50 of 145 max) fire exactly on short-term weakness -- i.e. possibly selling INTO the bounce.
The entry side had the mirror-image finding (momentum_strength anti-predictive at entry).
This ablation zeroes each exit component through the full 60d replay, purged OOS + PBO/DSR,
mirroring ablate_entry_signals.py.

PREREGISTERED promotion rule (fixed before running): a variant may be enabled live ONLY if
  (1) TEST-window PnL (dates > 2026-05-20) beats baseline's TEST PnL,
  (2) full-period total PnL >= baseline's,
  (3) TEST mean rise-after-sell (premature-sell severity) is not worse than baseline,
  (4) maxDD not worse than baseline by more than 3 percentage points,
  (5) PBO < 0.5.
User pre-authorized enabling directly on a pass (2026-07-03); hard stops / emergency stop /
kill switch are untouched by exit_score_weights and remain in force regardless.

Run: py -3.13 scripts/ablate_exit_signals.py
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
OUT = ROOT / "outputs" / "exit_signal_ablation"
REPLAY = ROOT / "scripts" / "replay_t0_decisions.py"
TRAIN_END = "2026-05-20"
WHIPSAW_TH = 0.005

# variant -> exit_score_weights keys to ZERO. structure_break kept as a sanity control
# (expected to hurt if the ablation methodology is sound).
VARIANTS: dict[str, list[str]] = {
    "baseline": [],
    "drop_momentum_reversal": ["momentum_reversal"],
    "drop_acceleration_negative": ["acceleration_negative"],
    "drop_bid_pressure_negative": ["bid_pressure_negative"],
    "drop_flow3": ["momentum_reversal", "acceleration_negative", "bid_pressure_negative"],
    "drop_profit_drawdown": ["profit_drawdown"],
    "drop_vwap_breakdown": ["vwap_breakdown"],
    "drop_liquidity_deterioration": ["liquidity_deterioration"],
    "control_drop_structure_break": ["structure_break"],
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
    rise = [-as_float(r.get("max_adverse_excursion")) for r in sells
            if r.get("max_adverse_excursion") is not None]
    whips = [x for x in rise if x >= WHIPSAW_TH]
    return {
        "sells": len(sells),
        "whipsaw_share": round(len(whips) / len(sells), 4) if sells else None,
        "mean_rise_after": round(mean(rise), 5) if rise else None,
    }


def run_variant(name: str, zero_keys: list[str]) -> dict:
    vdir = OUT / name
    (vdir / "scores").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    strat = cfg["strategy"]
    strat["entry_logic_v2"] = {"enabled": False}   # isolate: sell-side ablation only
    strat.setdefault("t0_entry_eligibility", {})["enabled"] = False  # yahoo quotes lack asset_class
    w = strat.setdefault("exit_score_weights", {})
    for k in zero_keys:
        w[k] = 0
    cfg_path = vdir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] zeroed {zero_keys or '(none)'} -> running replay ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(REPLAY), "--config", str(cfg_path), "--quotes", str(QUOTES),
         "--label", f"exit_{name}", "--decision-scores",
         "--decision-score-output-dir", str(vdir / "scores"),
         "--output-dir", str(vdir), "--output-detail", "summary"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    sp = vdir / f"exit_{name}_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    per_day = summary.get("per_day", {}) if isinstance(summary.get("per_day"), dict) else {}
    dates = sorted(per_day)
    daily_pnl = [as_float(per_day[d].get("net_pnl")) for d in dates]
    test_pnl = sum(as_float(per_day[d].get("net_pnl")) for d in dates if d > TRAIN_END)
    rows = _rows(vdir / "scores")
    res = {
        "name": name, "zeroed": zero_keys,
        "total_pnl": as_float(summary.get("total_pnl")),
        "test_pnl": round(test_pnl, 0),
        "max_drawdown": as_float(summary.get("max_drawdown")),
        "daily_pnl": daily_pnl, "dates": dates,
        "daily_sharpe": round(mean(daily_pnl) / pstdev(daily_pnl), 4) if len(daily_pnl) > 1 and pstdev(daily_pnl) else None,
        "full": sell_metrics(rows), "train": sell_metrics(_window(rows, "train")), "test": sell_metrics(_window(rows, "test")),
        "replay_ok": proc.returncode == 0,
        "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else None,
    }
    print(f"[{name}] pnl={res['total_pnl']:.0f} testPnl={res['test_pnl']:.0f} maxDD={res['max_drawdown']:.4f} "
          f"sells={res['full']['sells']} rise-after test={res['test']['mean_rise_after']}", flush=True)
    return res


def promotion_check(r: dict, base: dict) -> tuple[bool, list[str]]:
    """The preregistered rule from the docstring, applied verbatim."""
    reasons = []
    if not (r["test_pnl"] > base["test_pnl"]):
        reasons.append(f"test_pnl {r['test_pnl']:.0f} <= baseline {base['test_pnl']:.0f}")
    if not (r["total_pnl"] >= base["total_pnl"]):
        reasons.append(f"total_pnl {r['total_pnl']:.0f} < baseline {base['total_pnl']:.0f}")
    ra, rb = r["test"].get("mean_rise_after"), base["test"].get("mean_rise_after")
    if ra is not None and rb is not None and ra > rb:
        reasons.append(f"test rise-after {ra} worse than baseline {rb}")
    if r["max_drawdown"] < base["max_drawdown"] - 0.03:
        reasons.append(f"maxDD {r['max_drawdown']:.4f} worse than baseline {base['max_drawdown']:.4f} - 0.03")
    return (not reasons), reasons


def scorecard(results: list[dict]) -> tuple[str, dict | None]:
    base = next((r for r in results if r["name"] == "baseline"), results[0])
    lines = [
        "# Exit-Signal Component Ablation -- 60d replay, purged OOS, PBO/DSR",
        "",
        f"Quotes `{QUOTES.name}` | purged train<= {TRAIN_END} < test | whipsaw thr {WHIPSAW_TH:.1%}",
        "",
        "| variant | total PnL | TEST PnL | maxDD | dSharpe | sells | rise-after TEST | whipsaw TEST |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        t = r["test"]
        lines.append(
            f"| {r['name']} | {r['total_pnl']:.0f} | {r['test_pnl']:.0f} | {r['max_drawdown']:.4f} | "
            f"{r['daily_sharpe']} | {r['full']['sells']} | {_p(t['mean_rise_after'])} | {_p(t['whipsaw_share'])} |")
    matrix = [r["daily_pnl"] for r in results if len(r["daily_pnl"]) == len(base["daily_pnl"]) and r["daily_pnl"]]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 else {"pbo": None}
    pbo_ok = pbo.get("pbo") is not None and pbo["pbo"] < 0.5
    candidates = [r for r in results if r["name"] not in ("baseline", "control_drop_structure_break")]
    passers = [(r, promotion_check(r, base)) for r in candidates]
    passing = [r for r, (ok, _) in passers if ok]
    best = max(passing, key=lambda r: r["test_pnl"], default=None)
    dsr = (og.deflated_significance_note(n_trials=len(VARIANTS) - 1,
                                         observed_sharpe=best["daily_sharpe"] or 0.0, n_obs=len(best["daily_pnl"]))
           if best and best["daily_sharpe"] is not None else {"note": "n/a"})
    control = next((r for r in results if r["name"] == "control_drop_structure_break"), None)
    lines += [
        "",
        f"- **PBO**: {pbo.get('pbo')} ({'holds OOS' if pbo_ok else 'noise: winner does NOT hold' if pbo.get('pbo') is not None else 'n/a'})",
        f"- sanity control (drop structure_break, expected to hurt): PnL {control['total_pnl']:.0f} vs baseline {base['total_pnl']:.0f}" if control else "",
        "- preregistered-rule results:",
    ]
    for r, (ok, reasons) in passers:
        lines.append(f"  - {r['name']}: {'PASS' if ok else 'fail -- ' + '; '.join(reasons)}")
    winner = best if (best and pbo_ok) else None
    lines += [
        f"- **Deflated-Sharpe note (best passer)**: {dsr}",
        "",
        "## Verdict",
        "",
        (f"**{winner['name']}** passes the full preregistered rule including PBO; eligible to enable "
         f"(zeroing {winner['zeroed']})." if winner else
         "No variant passes the full preregistered rule (incl. PBO<0.5). The exit components stay as they are."),
        "",
        "_Hard stops, emergency stop and kill switch are outside exit_score_weights and unaffected either way._",
        "",
    ]
    return "\n".join(lines), winner


def _p(v) -> str:
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_variant(n, ks) for n, ks in VARIANTS.items()]
    (OUT / "ablation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report, winner = scorecard(results)
    (OUT / "ablation_scorecard.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    if winner:
        (OUT / "promotion_candidate.json").write_text(
            json.dumps({"name": winner["name"], "zeroed": winner["zeroed"]}, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
