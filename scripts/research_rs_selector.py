"""Day-neutral to-close relative-strength selector -- prototype + honest gate.

Builds on the IC audit: the OOS-stable POSITIVE to-close signals were vwap_dist,
orb_breakout, range_pos, chg_from_open. Here we combine them into a cross-sectional
composite, and at ONE decision time per day (default 10:00, after the opening range forms)
rank the universe, go long the top tertile to the close, and compare to the bottom tertile.
One observation per (day, code) keeps the day cluster honest (to-close returns within a day
are otherwise perfectly correlated).

Reports: top-tertile to-close return NET of cost, day-neutral top-minus-bottom spread, the
composite's daily cross-sectional IC, a day-cluster bootstrap CI, purged train/test, and a
deflated-Sharpe note over the signals tried. IC != PnL after cost (the v2 lesson), so the
spread must clear cost AND hold OOS to matter.

Diagnostic only. Run: py -3.13 scripts/research_rs_selector.py [--entry-min 600]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT
from research_signal_ic import load_bars, signals_at, TRAIN_END
import overfitting_guard as og

OUT = ROOT / "outputs" / "rs_selector"
COST = 0.0006                       # 万三 commission x2 round-trip (ETF: no stamp duty)
COMPOSITE_SIGNALS = ["vwap_dist", "orb_breakout", "range_pos", "chg_from_open"]
MIN_CODES = 10


def _idx_at_or_after(bars, minute):
    for k, b in enumerate(bars):
        if b[0] >= minute:
            return k
    return None


def panel_for_day(series, date, codes, entry_min):
    """Return list of (code, composite_z, to_close_net) for one day's decision."""
    raw: list[tuple[str, dict, float]] = []
    for c in codes:
        bars = series[(date, c)]
        i = _idx_at_or_after(bars, entry_min)
        if i is None or i < 2 or i >= len(bars) - 1:
            continue
        sg = signals_at(bars, i)
        entry_px = bars[i][1]
        close_px = bars[-1][1]
        if entry_px <= 0 or close_px <= 0:
            continue
        to_close = close_px / entry_px - 1.0
        raw.append((c, sg, to_close))
    if len(raw) < MIN_CODES:
        return []
    # cross-sectional z-score each component, average available -> composite
    comp_z: dict[str, float] = {}
    for s in COMPOSITE_SIGNALS:
        vals = [(c, sg.get(s)) for c, sg, _ in raw if sg.get(s) is not None]
        if len(vals) < MIN_CODES:
            continue
        arr = np.array([v for _, v in vals], dtype=float)
        mu, sd = arr.mean(), arr.std()
        if sd == 0:
            continue
        for (c, v) in vals:
            comp_z.setdefault(c, [0.0, 0])
            comp_z[c][0] += (v - mu) / sd
            comp_z[c][1] += 1
    out = []
    for c, sg, tc in raw:
        z = comp_z.get(c)
        if z and z[1] > 0:
            out.append((c, z[0] / z[1], tc))
    return out


def analyze(entry_min: int) -> dict:
    series = load_bars()
    dates = sorted({d for d, _ in series})
    codes_by_date: dict[str, list[str]] = {}
    for (d, c) in series:
        codes_by_date.setdefault(d, []).append(c)

    daily = []   # per day: dict(date, top_net, bottom_net, spread, ic, n)
    for d in dates:
        panel = panel_for_day(series, d, codes_by_date[d], entry_min)
        if len(panel) < MIN_CODES:
            continue
        comps = np.array([x[1] for x in panel])
        rets = np.array([x[2] for x in panel])
        lo, hi = np.quantile(comps, 1 / 3), np.quantile(comps, 2 / 3)
        top = rets[comps >= hi]
        bot = rets[comps <= lo]
        if top.size == 0 or bot.size == 0:
            continue
        # cross-sectional IC (composite vs to-close) for the day
        rc = np.argsort(np.argsort(comps)).astype(float)
        rr = np.argsort(np.argsort(rets)).astype(float)
        ic = float(np.corrcoef(rc, rr)[0, 1]) if rc.std() and rr.std() else None
        daily.append({
            "date": d, "n": len(panel),
            "top_net": float(top.mean()) - COST,
            "bottom_net": float(bot.mean()) - COST,
            "spread": float(top.mean() - bot.mean()),
            "ic": ic,
        })

    def stats(rows: list[dict]) -> dict:
        if not rows:
            return {"days": 0}
        spread = np.array([r["spread"] for r in rows])
        top = np.array([r["top_net"] for r in rows])
        ics = np.array([r["ic"] for r in rows if r["ic"] is not None])
        sh = float(spread.mean() / spread.std()) if spread.std() else None
        return {
            "days": len(rows),
            "top_net_mean": round(float(top.mean()), 5),
            "spread_mean": round(float(spread.mean()), 5),
            "spread_t": round(float(spread.mean() / spread.std() * np.sqrt(len(spread))), 2) if spread.std() else None,
            "spread_sharpe_daily": round(sh, 4) if sh is not None else None,
            "mean_ic": round(float(ics.mean()), 4) if ics.size else None,
            "ic_t": round(float(ics.mean() / ics.std() * np.sqrt(ics.size)), 2) if ics.size and ics.std() else None,
            "win_days": round(float(np.mean(spread > 0)), 3),
        }

    train = [r for r in daily if r["date"] <= TRAIN_END]
    test = [r for r in daily if r["date"] > TRAIN_END]

    # day-cluster bootstrap of the spread mean
    rng = np.random.default_rng(20260622)
    sp = np.array([r["spread"] for r in daily])
    boot = ([float(sp[rng.integers(0, len(sp), len(sp))].mean()) for _ in range(20000)]
            if len(sp) else [])
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else [None, None]
    p_le0 = float(np.mean(np.array(boot) <= 0)) if boot else None

    # PBO across the candidate signals (rows=signals, cols=days IC) + composite
    perf = []
    sig_list = COMPOSITE_SIGNALS + ["composite"]
    day_index = [r["date"] for r in daily]
    # rebuild per-signal daily IC to feed PBO
    per_sig_daily: dict[str, list[float]] = {s: [] for s in sig_list}
    for d in day_index:
        panel = panel_for_day(series, d, codes_by_date[d], entry_min)
        comps = {s: [] for s in COMPOSITE_SIGNALS}
        rets = []
        zc = []
        for c, z, tc in panel:
            rets.append(tc); zc.append(z)
        # composite IC
        if len(zc) >= MIN_CODES:
            rc = np.argsort(np.argsort(np.array(zc))).astype(float)
            rr = np.argsort(np.argsort(np.array(rets))).astype(float)
            per_sig_daily["composite"].append(float(np.corrcoef(rc, rr)[0, 1]) if rc.std() and rr.std() else 0.0)
        else:
            per_sig_daily["composite"].append(0.0)
        # individual signals
        sgmap = {c: signals_at(series[(d, c)], _idx_at_or_after(series[(d, c)], entry_min))
                 for c, _, _ in panel}
        for s in COMPOSITE_SIGNALS:
            xs, ys = [], []
            for c, _, tc in panel:
                v = sgmap[c].get(s)
                if v is not None:
                    xs.append(v); ys.append(tc)
            if len(xs) >= MIN_CODES:
                rc = np.argsort(np.argsort(np.array(xs))).astype(float)
                rr = np.argsort(np.argsort(np.array(ys))).astype(float)
                per_sig_daily[s].append(float(np.corrcoef(rc, rr)[0, 1]) if rc.std() and rr.std() else 0.0)
            else:
                per_sig_daily[s].append(0.0)
    matrix = [per_sig_daily[s] for s in sig_list]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 else {"pbo": None}
    full = stats(daily)
    dsr = og.deflated_significance_note(n_trials=len(sig_list),
                                        observed_sharpe=full.get("spread_sharpe_daily") or 0.0,
                                        n_obs=full.get("days", 0))
    return {
        "entry_min": entry_min, "cost": COST, "composite_signals": COMPOSITE_SIGNALS,
        "full": full, "train": stats(train), "test": stats(test),
        "spread_day_cluster_ci": ci, "spread_p_le_zero": p_le0,
        "pbo": pbo, "deflated_sharpe": dsr,
    }


def render(r: dict) -> str:
    f, tr, te = r["full"], r["train"], r["test"]
    ci = r["spread_day_cluster_ci"]
    return "\n".join([
        f"# Day-Neutral To-Close RS Selector -- entry {r['entry_min']//60:02d}:{r['entry_min']%60:02d}, cost {r['cost']:.2%}",
        "",
        f"Composite of {r['composite_signals']} (cross-sectional z-avg). Top vs bottom tertile, "
        "held to close; one obs per (day,code). Purged train/test.",
        "",
        "| window | days | top net | top-bottom spread | spread t | daily Sharpe | IC | IC t | win days |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
        f"| full | {f['days']} | {_p(f['top_net_mean'])} | {_p(f['spread_mean'])} | {f['spread_t']} | {f['spread_sharpe_daily']} | {f['mean_ic']} | {f['ic_t']} | {_p(f['win_days'])} |",
        f"| train | {tr['days']} | {_p(tr.get('top_net_mean'))} | {_p(tr.get('spread_mean'))} | {tr.get('spread_t')} | {tr.get('spread_sharpe_daily')} | {tr.get('mean_ic')} | {tr.get('ic_t')} | {_p(tr.get('win_days'))} |",
        f"| TEST | {te['days']} | {_p(te.get('top_net_mean'))} | {_p(te.get('spread_mean'))} | {te.get('spread_t')} | {te.get('spread_sharpe_daily')} | {te.get('mean_ic')} | {te.get('ic_t')} | {_p(te.get('win_days'))} |",
        "",
        f"- spread day-cluster 95% CI: [{_p(ci[0])}, {_p(ci[1])}] | P(spread<=0)={_p(r['spread_p_le_zero'])}",
        f"- **PBO**: {r['pbo'].get('pbo')} ({'noise' if (r['pbo'].get('pbo') or 0)>=0.5 else 'holds OOS' if r['pbo'].get('pbo') is not None else 'n/a'})",
        f"- **Deflated-Sharpe**: {r['deflated_sharpe']}",
        "",
        "## Read",
        "",
        "Tradeable only if: top net return > 0 AND top-bottom spread clears with CI excluding zero "
        "AND it HOLDS in TEST AND PBO well below 0.5. `top net` already deducts round-trip cost; if "
        "it is <=0 the to-close edge does not survive cost even though the raw IC was positive.",
        "",
        "_Diagnostic only; no signal, weight, or gate changed._",
        "",
    ])


def _p(v) -> str:
    return f"{v:.4%}" if isinstance(v, (int, float)) else "n/a"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-min", type=int, default=600)  # 10:00
    args = ap.parse_args()
    r = analyze(args.entry_min)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rs_selector_{args.entry_min}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(r)
    (OUT / f"rs_selector_{args.entry_min}.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
