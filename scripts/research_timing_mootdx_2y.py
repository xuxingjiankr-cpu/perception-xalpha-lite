"""Two-year out-of-time validation of the pullback-entry finding, on mootdx TDX 5-min bars
(data/market/mootdx/bars_5m/, ~490 trading days 2024-06..2026-07, backfilled 2026-07-03).

entry_logic_v2 (pullback entry, live since 2026-07-02) was validated on: 10 live days
(t=4.80), a 60d Yahoo purged split (test t=1.75), and a 60d full-pipeline ablation
(PBO=0.0143). ALL of that evidence lives inside 2026-03-23..2026-07-02. This script asks the
harder question: does pullback>breakout hold on ~440 trading days that NO tuning ever saw
(everything before 2026-03-23)?

Universe note: the 18 backfilled codes are the codes the agents actually trade (rebalance
universe + T0 legacy + OBI/liquid T0) -- narrower cross-section than the 612-code Yahoo probe
(top-decile pick =~ 2 names/round), but it is the population the live gate actually operates
on. Cost: flat 15.5bp round-trip (2x measured median half-spread + commission), conservative
for these liquid names; the pullback-breakout DIFF is cost-invariant anyway (same cost both
legs).

Eras reported separately, day-clustered:
  pre_tuning : dates < 2026-03-23  (never seen by any fit -- the decisive sample)
  tuning_era : 2026-03-23..2026-07-03 (overlaps all prior evidence; sanity only)

Decision rule (preregistered, per standing user authorization): if pre_tuning shows the
pullback advantage DIRECTIONALLY REVERSED (mean diff < 0) or fill-rate collapse, flip
strategy.entry_logic_v2.enabled back to false. If it holds (mean diff > 0, majority of days
favoring), the live flag stands with upgraded evidence. STRICTLY offline; this script itself
changes nothing.

Run: py -3.13 scripts/research_timing_mootdx_2y.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

from run_etf_paper_trading_agent import ROOT
from research_timing import pullback_entry

BARS_DIR = ROOT / "data" / "market" / "mootdx" / "bars_5m"
OUT_DIR = ROOT / "outputs" / "research_timing"
TUNING_START = "2026-03-23"     # everything from here on influenced some fit/choice

WINDOW_BARS = 4                  # 20 min momentum window (same as prior probes)
PULLBACK_FRAC = 0.006            # the LIVE parameterization (0.6% / 10 min = 2 bars)
PULLBACK_WAIT_BARS = 2
TOP_FRAC = 0.10
COST_RT = 0.00155                # 15.5bp flat round-trip, cancels in the diff
DECISION_TIMES = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]


def load() -> dict[str, dict[str, dict[str, list]]]:
    """date -> code -> {'times': [...], 'close': [...], 'amount': [...]}"""
    by_date: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(
        lambda: {"times": [], "close": [], "amount": []}))
    for path in sorted(BARS_DIR.glob("*.jsonl")):
        code = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            dt = str(r.get("dt", ""))
            d, t = dt[:10], dt[11:16]
            px = r.get("close")
            if not d or not t or not px or px <= 0:
                continue
            node = by_date[d][code]
            node["times"].append(t)
            node["close"].append(float(px))
            node["amount"].append(float(r.get("amount") or 0.0))
    return by_date


def analyze_day(day_data: dict[str, dict[str, list]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"fwd_breakout": [], "fwd_pullback": [], "filled": []}
    idx = {c: {t: i for i, t in enumerate(n["times"])} for c, n in day_data.items()}
    for dt in DECISION_TIMES:
        recent = []
        for c, n in day_data.items():
            i = idx[c].get(dt)
            if i is None or i < WINDOW_BARS or i + 3 > len(n["close"]):
                continue
            p_now, p_past = n["close"][i], n["close"][i - WINDOW_BARS]
            if p_now and p_past:
                recent.append((c, i, p_now / p_past - 1.0))
        if len(recent) < 8:
            continue
        recent.sort(key=lambda x: x[2], reverse=True)
        k = max(1, int(len(recent) * TOP_FRAC))
        for c, i, _ in recent[:k]:
            path = day_data[c]["close"][i:]
            if len(path) < 3:
                continue
            entry_b = path[0]
            out["fwd_breakout"].append((path[-1] / entry_b - 1.0 - COST_RT) * 100)
            pull = pullback_entry(path, PULLBACK_FRAC, PULLBACK_WAIT_BARS)
            out["filled"].append(1.0 if pull else 0.0)
            if pull:
                entry_p, j = pull
                rem = path[j:]
                out["fwd_pullback"].append((rem[-1] / entry_p - 1.0 - COST_RT) * 100)
    return out


def day_clustered(rows: list[dict], key_a: str, key_b: str) -> dict:
    diffs = [r[key_a] - r[key_b] for r in rows
             if r.get(key_a) is not None and r.get(key_b) is not None]
    if len(diffs) < 2:
        return {"days": len(diffs), "verdict": "insufficient"}
    arr = np.array(diffs)
    t = float(arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr))) if arr.std(ddof=1) else None
    return {"days": len(diffs), "mean_diff_pct": round(float(arr.mean()), 4),
            "day_clustered_t": round(t, 2) if t is not None else None,
            "days_favoring_pullback": sum(1 for d in diffs if d > 0),
            "share_favoring": round(sum(1 for d in diffs if d > 0) / len(diffs), 3)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    by_date = load()
    rows = []
    for d in sorted(by_date):
        b = analyze_day(by_date[d])
        if not b["fwd_breakout"]:
            continue
        def m(k):
            v = b.get(k, [])
            return round(sum(v) / len(v), 4) if v else None
        rows.append({"date": d, "n": len(b["fwd_breakout"]),
                      "breakout": m("fwd_breakout"), "pullback": m("fwd_pullback"),
                      "fill_rate": m("filled")})

    pre = [r for r in rows if r["date"] < TUNING_START]
    tun = [r for r in rows if r["date"] >= TUNING_START]
    fill_pre = [r["fill_rate"] for r in pre if r["fill_rate"] is not None]

    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "mootdx TDX 5-min bars, 18 traded codes",
        "status": "diagnostic_only", "paper_trading_only": True,
        "params": {"pullback_frac": PULLBACK_FRAC, "wait_bars": PULLBACK_WAIT_BARS,
                    "window_bars": WINDOW_BARS, "cost_rt": COST_RT},
        "usable_days": len(rows), "pre_tuning_days": len(pre), "tuning_era_days": len(tun),
        "pre_tuning": day_clustered(pre, "pullback", "breakout"),
        "tuning_era": day_clustered(tun, "pullback", "breakout"),
        "pre_tuning_fill_rate": round(float(np.mean(fill_pre)), 3) if fill_pre else None,
        "pre_tuning_abs": {
            "breakout_mean": round(float(np.mean([r["breakout"] for r in pre])), 4) if pre else None,
            "pullback_mean": round(float(np.mean([r["pullback"] for r in pre if r["pullback"] is not None])), 4) if pre else None,
        },
        "per_day": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "timing_mootdx_2y.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"usable days: {len(rows)} (pre-tuning {len(pre)} / tuning-era {len(tun)})")
    print(f"PRE-TUNING (never seen): {result['pre_tuning']}")
    print(f"  abs: breakout {result['pre_tuning_abs']['breakout_mean']}% vs pullback {result['pre_tuning_abs']['pullback_mean']}% | fill {result['pre_tuning_fill_rate']}")
    print(f"TUNING ERA (sanity)    : {result['tuning_era']}")
    print(f"log: {OUT_DIR / 'timing_mootdx_2y.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
