"""Entry-logic-v2 (pullback-entry) ablation over the 60-day replay, with purged OOS split +
PBO/DSR. Tests the actual config-gated mechanism (apply_entry_logic_v2 in
run_t0_intraday_agent.py) through the full decision/order/fill pipeline -- not the standalone
percentile probe in research_timing.py / research_timing_60d.py. Source hypothesis: pullback
entries beat breakout entries 10/10 live days (t=4.80) and replicate directionally on a 60d
purged-OOS holdout (test=21d, 17/21 days favoring pullback, t=1.75, below |t|=2 significance).

Diagnostic/offline only -- writes nothing to the live config. entry_logic_v2 stays
`enabled:false` in production regardless of this run's outcome; flipping it needs a clean
PBO/DSR pass here AND a forward-shadow period, per repo convention (mirrors
ablate_sell_logic_v2.py / ablate_entry_signals.py).

Run: py -3.13 scripts/ablate_entry_logic_v2.py
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
OUT = ROOT / "outputs" / "entry_logic_v2_ablation"
REPLAY = ROOT / "scripts" / "replay_t0_decisions.py"
TRAIN_END = "2026-05-20"          # same purged split as the other ablations

# Each variant overrides strategy.entry_logic_v2. baseline_off = current production (no wait).
VARIANTS: dict[str, dict] = {
    "baseline_off": {"enabled": False},
    "v2_pullback_04pct_15min": {"enabled": True, "pullback_frac": 0.004, "max_wait_minutes": 15},
    "v2_pullback_06pct_10min": {"enabled": True, "pullback_frac": 0.006, "max_wait_minutes": 10},
    "v2_pullback_03pct_20min": {"enabled": True, "pullback_frac": 0.003, "max_wait_minutes": 20},
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


def buy_stats(rows: list[dict], which: str) -> dict:
    buys = [r for r in rows if str(r.get("decision_type")) == "BUY" and r.get("realized_return") is not None]
    if which == "train":
        buys = [r for r in buys if str(r.get("date")) <= TRAIN_END]
    elif which == "test":
        buys = [r for r in buys if str(r.get("date")) > TRAIN_END]
    nets = [as_float(r.get("realized_return")) - 0.0006 for r in buys]  # 万三 round-trip
    return {
        "buys": len(buys),
        "mean_buy_net": round(mean(nets), 5) if nets else None,
        "win_rate": round(sum(1 for x in nets if x > 0) / len(nets), 3) if nets else None,
    }


def run_variant(name: str, override: dict) -> dict:
    vdir = OUT / name
    (vdir / "scores").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["strategy"]["entry_logic_v2"] = override
    # yahoo_60d_quotes.jsonl carries no asset_class field, so the live
    # block_unknown_asset_class t0_entry_eligibility gate would reject every candidate before
    # entry_score is even reached (precedent: same fix used for the concentration-frontier
    # replay). Disabling it here only affects this offline replay config copy, not live.
    cfg["strategy"].setdefault("t0_entry_eligibility", {})["enabled"] = False
    cfg_path = vdir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] running replay ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(REPLAY), "--config", str(cfg_path), "--quotes", str(QUOTES),
         "--label", f"entrylv2_{name}", "--decision-scores",
         "--decision-score-output-dir", str(vdir / "scores"),
         "--output-dir", str(vdir), "--output-detail", "summary"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    sp = vdir / f"entrylv2_{name}_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    per_day = summary.get("per_day", {}) if isinstance(summary.get("per_day"), dict) else {}
    dates = sorted(per_day)
    daily_pnl = [as_float(per_day[d].get("net_pnl")) for d in dates]
    rows = _rows(vdir / "scores")
    res = {
        "name": name, "override": override,
        "total_pnl": as_float(summary.get("total_pnl")),
        "max_drawdown": as_float(summary.get("max_drawdown")),
        "daily_pnl": daily_pnl, "dates": dates,
        "daily_sharpe": round(mean(daily_pnl) / pstdev(daily_pnl), 4) if len(daily_pnl) > 1 and pstdev(daily_pnl) else None,
        "full": buy_stats(rows, "full"), "train": buy_stats(rows, "train"), "test": buy_stats(rows, "test"),
        "replay_ok": proc.returncode == 0,
        "stderr_tail": proc.stderr[-2000:] if proc.returncode != 0 else None,
    }
    print(f"[{name}] pnl={res['total_pnl']:.0f} maxDD={res['max_drawdown']:.4f} "
          f"buys={res['full']['buys']} mean_buy_net full/test={res['full']['mean_buy_net']}/{res['test']['mean_buy_net']}",
          flush=True)
    return res


def scorecard(results: list[dict]) -> str:
    base = next((r for r in results if r["name"] == "baseline_off"), results[0])
    lines = [
        "# Entry-Logic v2 (Pullback-Entry) Ablation -- 60d replay + purged OOS + PBO/DSR",
        "",
        f"Quotes: `{QUOTES.name}` | purged split: train<= {TRAIN_END} < test",
        "",
        "Source: research_timing.py found pullback-entry beat breakout-entry 10/10 live days "
        "(day-clustered t=4.80, even after a realistic round-trip cost); research_timing_60d.py "
        "replicated the direction on an independent 60d purged-OOS sample (test=21d, 17/21 days "
        "favoring pullback, t=1.75 -- below conventional significance). This ablation tests the "
        "actual config-gated mechanism (apply_entry_logic_v2), not the standalone probe.",
        "",
        "| variant | total PnL | maxDD | dSharpe | buys full | mean buy-net full | mean buy-net TEST | win TEST |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        f, t = r["full"], r["test"]
        lines.append(
            f"| {r['name']} | {r['total_pnl']:.0f} | {r['max_drawdown']:.4f} | {r['daily_sharpe']} | "
            f"{f['buys']} | {_p(f['mean_buy_net'])} | {_p(t['mean_buy_net'])} | {_p(t['win_rate'])} |")
    matrix = [r["daily_pnl"] for r in results if len(r["daily_pnl"]) == len(base["daily_pnl"]) and r["daily_pnl"]]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 else {"pbo": None}
    ranked = [r for r in results if r["name"] != "baseline_off" and r["test"]["mean_buy_net"] is not None]
    best = max(ranked, key=lambda r: r["test"]["mean_buy_net"], default=None)
    dsr = (og.deflated_significance_note(n_trials=len(VARIANTS) - 1,
                                         observed_sharpe=best["daily_sharpe"] or 0.0, n_obs=len(best["daily_pnl"]))
           if best and best["daily_sharpe"] is not None else {"note": "n/a"})
    lines += [
        "",
        f"- baseline total PnL {base['total_pnl']:.0f}, mean buy-net full/test "
        f"{_p(base['full']['mean_buy_net'])}/{_p(base['test']['mean_buy_net'])}",
        f"- **PBO** (winner persists OOS?): {pbo.get('pbo')} "
        f"({'noise: winner does NOT hold' if (pbo.get('pbo') or 0) >= 0.5 else 'holds OOS' if pbo.get('pbo') is not None else 'n/a'})",
        f"- best non-baseline (by TEST buy-net): {best['name'] if best else 'n/a'} "
        f"(PnL {best['total_pnl']:.0f} vs baseline {base['total_pnl']:.0f})" if best else "- best: n/a",
        f"- **Deflated-Sharpe note**: {dsr}",
        "",
        "## Read",
        "",
        "A variant only advances to forward-shadow if it raises TEST mean buy-net AND does not "
        "worsen total PnL/drawdown AND survives PBO (winner persists out-of-sample, pbo well "
        "below 0.5). The standalone probe's effect was directional but sub-significance (t=1.75) "
        "on 60d OOS -- this run is the harder bar: full pipeline, position sizing, capacity limits, "
        "and competing entry candidates all included, not just a percentile/return probe.",
        "",
        "_Offline ablation. entry_logic_v2 stays `enabled:false` in the live config regardless of outcome._",
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
