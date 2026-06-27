"""Entry-signal ablation -- test whether DROPPING the IC-audited anti-predictive entry
components (momentum_strength, acceleration_positive: both had stable NEGATIVE cross-sectional
IC) improves the strategy on the 60-day replay, with purged OOS + PBO/DSR.

Each variant zeroes one or more entry_score_weights, runs the offline replay (no broker), and
is scored on net PnL / daily Sharpe / drawdown / buy quality. PBO across days and a deflated-
Sharpe note flag a winner that is just multiple-testing noise (the v2/RS lesson: plausible !=
profitable). Offline only; flips no live weight.

Run: py -3.13 scripts/ablate_entry_signals.py
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
OUT = ROOT / "outputs" / "entry_ablation"
REPLAY = ROOT / "scripts" / "replay_t0_decisions.py"
TRAIN_END = "2026-05-20"

# variant -> entry_score_weights keys to ZERO out (anti-predictive per IC audit)
VARIANTS: dict[str, list[str]] = {
    "baseline": [],
    "drop_momentum_strength": ["momentum_strength"],
    "drop_acceleration": ["acceleration_positive"],
    "drop_both": ["momentum_strength", "acceleration_positive"],
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


def run_variant(name: str, zero_keys: list[str]) -> dict:
    vdir = OUT / name
    (vdir / "scores").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    w = cfg["strategy"]["entry_score_weights"]
    for k in zero_keys:
        if k in w:
            w[k] = 0
    cfg_path = vdir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] zeroed {zero_keys or '(none)'} -> running replay ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(REPLAY), "--config", str(cfg_path), "--quotes", str(QUOTES),
         "--label", f"entry_{name}", "--decision-scores",
         "--decision-score-output-dir", str(vdir / "scores"),
         "--output-dir", str(vdir), "--output-detail", "summary"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    sp = vdir / f"entry_{name}_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    per_day = summary.get("per_day", {}) if isinstance(summary.get("per_day"), dict) else {}
    dates = sorted(per_day)
    daily_pnl = [as_float(per_day[d].get("net_pnl")) for d in dates]
    rows = _rows(vdir / "scores")
    res = {
        "name": name, "zeroed": zero_keys,
        "total_pnl": as_float(summary.get("total_pnl")),
        "max_drawdown": as_float(summary.get("max_drawdown")),
        "daily_pnl": daily_pnl, "dates": dates,
        "daily_sharpe": round(mean(daily_pnl) / pstdev(daily_pnl), 4) if len(daily_pnl) > 1 and pstdev(daily_pnl) else None,
        "full": buy_stats(rows, "full"), "train": buy_stats(rows, "train"), "test": buy_stats(rows, "test"),
        "replay_ok": proc.returncode == 0,
    }
    print(f"[{name}] pnl={res['total_pnl']:.0f} maxDD={res['max_drawdown']:.4f} "
          f"buys={res['full']['buys']} mean_buy_net full/test={res['full']['mean_buy_net']}/{res['test']['mean_buy_net']}", flush=True)
    return res


def scorecard(results: list[dict]) -> str:
    base = next((r for r in results if r["name"] == "baseline"), results[0])
    lines = [
        "# Entry-Signal Ablation -- drop IC-anti-predictive components (60d, purged OOS, PBO/DSR)",
        "",
        f"Quotes `{QUOTES.name}` | purged train<= {TRAIN_END} < test. Zeroing momentum_strength / "
        "acceleration_positive (both had stable NEGATIVE cross-sectional IC).",
        "",
        "| variant | total PnL | maxDD | dSharpe | buys | mean buy-net full | mean buy-net TEST | win TEST |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        f, t = r["full"], r["test"]
        lines.append(
            f"| {r['name']} | {r['total_pnl']:.0f} | {r['max_drawdown']:.4f} | {r['daily_sharpe']} | "
            f"{f['buys']} | {_p(f['mean_buy_net'])} | {_p(t['mean_buy_net'])} | {_p(t['win_rate'])} |")
    matrix = [r["daily_pnl"] for r in results if len(r["daily_pnl"]) == len(base["daily_pnl"]) and r["daily_pnl"]]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 else {"pbo": None}
    best = max((r for r in results if r["name"] != "baseline"), key=lambda r: r["total_pnl"], default=None)
    dsr = (og.deflated_significance_note(n_trials=len(VARIANTS) - 1,
                                         observed_sharpe=best["daily_sharpe"] or 0.0, n_obs=len(best["daily_pnl"]))
           if best and best["daily_sharpe"] is not None else {"note": "n/a"})
    lines += [
        "",
        f"- baseline total PnL {base['total_pnl']:.0f}, mean buy-net full/test {_p(base['full']['mean_buy_net'])}/{_p(base['test']['mean_buy_net'])}",
        f"- **PBO**: {pbo.get('pbo')} ({'noise: winner does NOT hold' if (pbo.get('pbo') or 0)>=0.5 else 'holds OOS' if pbo.get('pbo') is not None else 'n/a'})",
        f"- best non-baseline: {best['name'] if best else 'n/a'} (PnL {best['total_pnl']:.0f} vs {base['total_pnl']:.0f})" if best else "- best: n/a",
        f"- **Deflated-Sharpe**: {dsr}",
        "",
        "## Read",
        "",
        "Dropping an anti-predictive entry component should HELP if that component truly added noise. "
        "It only advances if it raises TEST buy-net AND total PnL AND survives PBO. Otherwise the "
        "component was not the binding factor (entry is gated/capacity-limited so weight changes may "
        "barely move the realized book). Offline only; no live weight changed.",
        "",
    ]
    return "\n".join(lines)


def _p(v) -> str:
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_variant(n, ks) for n, ks in VARIANTS.items()]
    (OUT / "entry_ablation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = scorecard(results)
    (OUT / "entry_ablation_scorecard.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
