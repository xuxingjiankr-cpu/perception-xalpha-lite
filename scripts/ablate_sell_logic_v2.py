"""Sell-logic v2 ablation over the 60-day replay, with purged OOS split + PBO/DSR.

For each config variant it runs the offline replay (no broker, no orders), then scores the
sell quality (whipsaw, rise-after-sell, missed-sell) on full / train / test windows and the
PnL/drawdown from the replay summary. Finally it computes combinatorial PBO across days and a
deflated-Sharpe note, so a winner that is just multiple-testing noise is flagged.

Diagnostic/offline only -- writes nothing to the live config or live state. Nothing here flips
sell_logic_v2 on in production; that needs forward-shadow validation first.

Run: py -3.13 scripts/ablate_sell_logic_v2.py
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
OUT = ROOT / "outputs" / "sell_v2_ablation"
REPLAY = ROOT / "scripts" / "replay_t0_decisions.py"
TRAIN_END = "2026-05-20"          # purged split: train <= this, test strictly after
WHIPSAW_TH = 0.005

# Each variant overrides strategy.sell_logic_v2. baseline = OFF (current production behavior).
VARIANTS: dict[str, dict | None] = {
    "baseline_off": {"enabled": False},
    "v2_full": {"enabled": True, "min_independent_negative_components": 2,
                "confirmation_snapshots": 2, "confirmation_max_gap_minutes": 7,
                "strong_tape_breadth": 0.65, "strong_tape_threshold_delta": 10,
                "scale_out_fraction": 0.5},
    "v2_confirm2_only": {"enabled": True, "min_independent_negative_components": 1,
                         "confirmation_snapshots": 2, "scale_out_fraction": 1.0,
                         "strong_tape_breadth": 1.01},
    "v2_components2_only": {"enabled": True, "min_independent_negative_components": 2,
                            "confirmation_snapshots": 1, "scale_out_fraction": 1.0,
                            "strong_tape_breadth": 1.01},
    "v2_scaleout_only": {"enabled": True, "min_independent_negative_components": 1,
                         "confirmation_snapshots": 1, "scale_out_fraction": 0.5,
                         "strong_tape_breadth": 1.01},
    "v2_breadth_only": {"enabled": True, "min_independent_negative_components": 1,
                        "confirmation_snapshots": 1, "scale_out_fraction": 1.0,
                        "strong_tape_breadth": 0.65, "strong_tape_threshold_delta": 10},
    "v2_confirm3_comp2": {"enabled": True, "min_independent_negative_components": 2,
                          "confirmation_snapshots": 3, "scale_out_fraction": 0.5,
                          "strong_tape_breadth": 0.7, "strong_tape_threshold_delta": 8},
}


# ---- sell-quality metrics (no bootstrap; direct shares) -----------------------
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
        "missed_sell_posdays": len(worst),
        "missed_sell_share": round(len(worst) / posdays, 4) if posdays else None,
    }


def run_variant(name: str, override: dict) -> dict:
    vdir = OUT / name
    (vdir / "scores").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["strategy"]["sell_logic_v2"] = override
    cfg_path = vdir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] running replay ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(REPLAY), "--config", str(cfg_path), "--quotes", str(QUOTES),
         "--label", f"v2_{name}", "--decision-scores",
         "--decision-score-output-dir", str(vdir / "scores"),
         "--output-dir", str(vdir), "--output-detail", "summary"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    summary_path = vdir / f"v2_{name}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    per_day = summary.get("per_day", {}) if isinstance(summary.get("per_day"), dict) else {}
    dates = sorted(per_day)
    daily_pnl = [as_float(per_day[d].get("net_pnl")) for d in dates]
    rows = _rows(vdir / "scores")
    result = {
        "name": name, "override": override,
        "total_pnl": as_float(summary.get("total_pnl")),
        "max_drawdown": as_float(summary.get("max_drawdown")),
        "daily_pnl": daily_pnl, "dates": dates,
        "daily_sharpe": round(mean(daily_pnl) / pstdev(daily_pnl), 4) if len(daily_pnl) > 1 and pstdev(daily_pnl) else None,
        "full": sell_metrics(rows),
        "train": sell_metrics(_window(rows, "train")),
        "test": sell_metrics(_window(rows, "test")),
        "replay_ok": proc.returncode == 0,
    }
    print(f"[{name}] pnl={result['total_pnl']:.0f} maxDD={result['max_drawdown']:.4f} "
          f"whipsaw full/test={result['full']['whipsaw_share']}/{result['test']['whipsaw_share']}",
          flush=True)
    return result


def scorecard(results: list[dict]) -> str:
    base = next((r for r in results if r["name"] == "baseline_off"), results[0])
    lines = [
        "# Sell-Logic v2 Ablation -- 60d replay + purged OOS + PBO/DSR",
        "",
        f"Quotes: `{QUOTES.name}` | purged split: train<= {TRAIN_END} < test | whipsaw thr {WHIPSAW_TH:.1%}",
        "",
        "| variant | total PnL | maxDD | dSharpe | whipsaw full | whipsaw TEST | rise-after TEST | missed TEST |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        f, t = r["full"], r["test"]
        lines.append(
            f"| {r['name']} | {r['total_pnl']:.0f} | {r['max_drawdown']:.4f} | {r['daily_sharpe']} | "
            f"{_p(f['whipsaw_share'])} | {_p(t['whipsaw_share'])} | {_p(t['mean_rise_after'])} | {_p(t['missed_sell_share'])} |")
    # PBO across days (rows=variants, cols=days net pnl)
    matrix = [r["daily_pnl"] for r in results if len(r["daily_pnl"]) == len(base["daily_pnl"]) and r["daily_pnl"]]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 else {"pbo": None}
    # DSR on the best-by-test-whipsaw variant's daily PnL
    ranked = [r for r in results if r["name"] != "baseline_off" and r["test"]["whipsaw_share"] is not None]
    best = min(ranked, key=lambda r: (r["test"]["whipsaw_share"], -(r["total_pnl"]))) if ranked else None
    dsr = (og.deflated_significance_note(n_trials=len(VARIANTS) - 1,
                                         observed_sharpe=best["daily_sharpe"] or 0.0,
                                         n_obs=len(best["daily_pnl"]))
           if best and best["daily_sharpe"] is not None else {"note": "n/a"})
    lines += [
        "",
        f"- baseline whipsaw full/test: {_p(base['full']['whipsaw_share'])} / {_p(base['test']['whipsaw_share'])}; "
        f"baseline missed test: {_p(base['test']['missed_sell_share'])}",
        f"- **PBO** (winner persists OOS?): {pbo.get('pbo')} "
        f"({'noise: winner does NOT hold' if (pbo.get('pbo') or 0) >= 0.5 else 'holds OOS' if pbo.get('pbo') is not None else 'n/a'})",
        f"- **best-by-test-whipsaw**: {best['name'] if best else 'n/a'} "
        f"(test whipsaw {_p(best['test']['whipsaw_share']) if best else 'n/a'}, "
        f"PnL {best['total_pnl']:.0f} vs baseline {base['total_pnl']:.0f})" if best else "- best: n/a",
        f"- **Deflated-Sharpe note**: {dsr}",
        "",
        "## Read",
        "",
        "A variant only advances to forward-shadow if it cuts TEST whipsaw materially vs baseline, "
        "does NOT worsen TEST missed-sell or total PnL, AND survives PBO (winner persists out-of-sample, "
        "pbo well below 0.5). Otherwise the sell levers are noise on this data. No live flip from this run.",
        "",
        "_Offline ablation. sell_logic_v2 stays `enabled:false` in the live config._",
        "",
    ]
    return "\n".join(lines)


def _p(v) -> str:
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_variant(name, ov) for name, ov in VARIANTS.items()]
    (OUT / "ablation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = scorecard(results)
    (OUT / "ablation_scorecard.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"outputs: {OUT / 'ablation_scorecard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
