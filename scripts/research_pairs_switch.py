"""Same-index ETF pairs / long-only SWITCH pre-test (KBPT idea, retail-adapted) -- now with
a PLUGGABLE real bid-ask spread, because the headline 'edge' is a bid-ask-bounce mirage until
you charge the spread you actually cross.

A-share retail can't short ETFs, so the long-only adaptation: among ETFs tracking the SAME
index (near-identical twins), always hold the relatively cheaper one and switch when the
log-spread is stretched (z-band). The common index move cancels (market-neutral), isolating
the spread's mean-reversion.

COST per switch = commission (万六, sell held + buy other) + the bid-ask HALF-SPREAD crossed on
BOTH legs (sell at bid, buy at ask). The half-spread is pluggable:
  * default: a FLAT assumption in bps, swept here to find the breakeven (no data needed);
  * real: supply --spread-file (per code/date/minute) and fills use the actual spread.
The data's synthetic 8bps book is NOT used -- that was the whole problem.

Verdict needs: TEST excess > 0 net of commission AND real half-spread, t>~2, PBO<0.5.
With real same-index ETF half-spreads (~1-3bps/leg) and ~8 switches/day, expect it to vanish.

Run: py -3.13 scripts/research_pairs_switch.py [--spread-file PATH] [--default-half-spread-bps X]
"""

from __future__ import annotations

import argparse
import json
import sys
import collections
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT
from research_signal_ic import load_bars, TRAIN_END
import overfitting_guard as og

OUT = ROOT / "outputs" / "pairs_switch"
UNIVERSE_DIR = ROOT / "data" / "market" / "eastmoney" / "universe"
COMMISSION_RT = 0.0006        # 万六: commission only (sell held + buy other)
MIN_CORR = 0.90
WINDOW = 6
ENTRY_BANDS = [0.75, 1.0, 1.5, 2.0]
HALF_SPREAD_SWEEP_BPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]   # per-leg half-spread, bps
MIN_BARS_DAY = WINDOW + 4


# ------------------------------------------------------------------ spread model
class SpreadModel:
    """Returns the per-leg HALF-spread (fraction) to cross for a code at a minute.

    Default: flat `default_bps` for every leg. To use REAL spreads, build the index from a
    tick/L2 export (JoinQuant/MyQuant): JSONL lines of either
        {"code": "510300", "date": "2026-06-18", "minute": 570, "half_spread_bps": 1.4}
    or {"code","date","minute","bid","ask"} (half-spread = (ask-bid)/2/mid).
    Missing (code,date,minute) falls back to `default_bps` so partial data still runs."""

    def __init__(self, default_bps: float = 0.0, index: dict | None = None):
        self.default = default_bps / 1e4
        self.index = index or {}

    def half(self, code: str, date: str, minute: int) -> float:
        v = self.index.get((code, date, minute))
        return (v / 1e4) if v is not None else self.default


def load_spread_model(path: str | None, default_bps: float) -> SpreadModel:
    if not path:
        return SpreadModel(default_bps)
    p = Path(path)
    if not p.exists():
        print(f"[warn] spread-file {p} not found; using flat {default_bps}bps")
        return SpreadModel(default_bps)
    index: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        code = str(r.get("code", "")).zfill(6)
        date = str(r.get("date", ""))
        minute = int(r.get("minute"))
        if "half_spread_bps" in r:
            hs = float(r["half_spread_bps"])
        elif r.get("bid") and r.get("ask"):
            bid, ask = float(r["bid"]), float(r["ask"])
            mid = (bid + ask) / 2.0
            hs = (ask - bid) / 2.0 / mid * 1e4 if mid > 0 else default_bps
        else:
            continue
        index[(code, date, minute)] = hs
    print(f"[spread-file] loaded {len(index)} (code,date,minute) real half-spreads")
    return SpreadModel(default_bps, index)


# ------------------------------------------------------------------ twins
def code_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for f in sorted(UNIVERSE_DIR.glob("eastmoney_universe_etf_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                names[str(r["stockCode"]).zfill(6)] = str(r.get("name", ""))
            except Exception:
                continue
    return names


def twin_pairs(bars, names) -> list[tuple[str, str]]:
    groups = collections.defaultdict(list)
    for c in sorted({c for (_, c) in bars}):
        nm = names.get(c)
        if nm:
            groups[nm.split("ETF")[0]].append(c)
    pairs = []
    for members in groups.values():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs


def aligned_day(bars, a, b, date):
    da, db = bars.get((date, a)), bars.get((date, b))
    if not da or not db:
        return None
    pa = {m: p for m, p, _ in da}
    pb = {m: p for m, p, _ in db}
    common = sorted(set(pa) & set(pb))
    if len(common) < MIN_BARS_DAY:
        return None
    return common, [pa[m] for m in common], [pb[m] for m in common]


def pair_corr(bars, a, b, dates) -> float | None:
    ra, rb = [], []
    for d in dates:
        al = aligned_day(bars, a, b, d)
        if not al:
            continue
        _, A, B = al
        ra += [A[k] / A[k - 1] - 1 for k in range(1, len(A))]
        rb += [B[k] / B[k - 1] - 1 for k in range(1, len(B))]
    if len(ra) < 50:
        return None
    ra, rb = np.array(ra), np.array(rb)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else None


def switch_day(minutes, A, B, code_a, code_b, date, entry, spreads: SpreadModel):
    """Long-only intraday switch; cost = commission + half-spread crossed on both legs."""
    A, B = np.array(A), np.array(B)
    s = np.log(A) - np.log(B)
    held, excess, switches = "A", 0.0, 0
    for t in range(1, len(s)):
        lo = max(0, t - WINDOW)
        w = s[lo:t]
        target = held
        if len(w) >= WINDOW and w.std() > 0:
            z = (s[t - 1] - w.mean()) / w.std()
            if z > entry:
                target = "B"
            elif z < -entry:
                target = "A"
        switched = target != held
        cost = 0.0
        if switched:
            mins = minutes[t]
            held_code = code_a if held == "A" else code_b
            new_code = code_a if target == "A" else code_b
            cost = COMMISSION_RT + spreads.half(held_code, date, mins) + spreads.half(new_code, date, mins)
            switches += 1
        held = target
        ra, rb = A[t] / A[t - 1] - 1, B[t] / B[t - 1] - 1
        held_ret = ra if held == "A" else rb
        excess += (held_ret - cost) - 0.5 * (ra + rb)
    return excess, switches


def run(bars, pairs, dates, entry, spreads):
    by_day = collections.defaultdict(list)
    sw = pd = 0
    for a, b in pairs:
        for d in dates:
            al = aligned_day(bars, a, b, d)
            if not al:
                continue
            minutes, A, B = al
            ex, n = switch_day(minutes, A, B, a, b, d, entry, spreads)
            by_day[d].append(ex)
            sw += n
            pd += 1
    daily = {d: float(np.mean(v)) for d, v in by_day.items() if v}
    return daily, sw, pd


def stats(daily, dates_sorted):
    vals = np.array([daily[d] for d in dates_sorted if d in daily])
    if vals.size == 0:
        return {"days": 0, "mean": None, "t": None}
    return {"days": int(vals.size), "mean": float(vals.mean()),
            "t": round(float(vals.mean() / vals.std() * np.sqrt(vals.size)), 2) if vals.std() else None}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread-file", default="")
    ap.add_argument("--default-half-spread-bps", type=float, default=None,
                    help="if set, run ONE model at this half-spread (else sweep)")
    args = ap.parse_args()

    bars = load_bars()
    dates = sorted({d for (d, _) in bars})
    names = code_names()
    cand = twin_pairs(bars, names)
    print(f"candidate twin pairs: {len(cand)} | confirming corr>{MIN_CORR} ...")
    pairs = [(a, b) for a, b in cand if (pair_corr(bars, a, b, dates) or 0) >= MIN_CORR]
    print(f"confirmed twin pairs: {len(pairs)}\n")

    hs_grid = ([args.default_half_spread_bps] if args.default_half_spread_bps is not None
               else HALF_SPREAD_SWEEP_BPS)
    print("Breakeven sweep -- TEST excess/day (net of commission + per-leg half-spread):")
    print("half_spread\\entry  " + "  ".join(f"z={e}" for e in ENTRY_BANDS))
    grid = {}
    for hs in hs_grid:
        spreads = load_spread_model(args.spread_file or None, hs)
        row = []
        for e in ENTRY_BANDS:
            daily, sw, pd = run(bars, pairs, dates, e, spreads)
            test = stats({d: daily[d] for d in daily if d > TRAIN_END}, dates)
            grid[(hs, e)] = {"daily": daily, "test": test, "sw_per_pairday": round(sw / pd, 2) if pd else None}
            row.append(f"{test['mean']*100:+.3f}%(t{test['t']})" if test["mean"] is not None else "n/a")
        print(f"  {hs:>4.1f}bps/leg     " + "  ".join(f"{c:>16}" for c in row))

    # PBO across entry bands at the most realistic flat spread tried (max in grid)
    hs_ref = max(hs_grid)
    common = sorted(set.intersection(*[set(grid[(hs_ref, e)]["daily"]) for e in ENTRY_BANDS])) if pairs else []
    matrix = [[grid[(hs_ref, e)]["daily"][d] for d in common] for e in ENTRY_BANDS] if common else []
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=8) if len(matrix) >= 2 and len(common) >= 4 else {"pbo": None}

    out = {"confirmed_pairs": len(pairs), "commission_round_trip": COMMISSION_RT,
           "window": WINDOW, "date_span": [dates[0], dates[-1]] if dates else None,
           "spread_file": args.spread_file or None,
           "grid": {f"hs{hs}_z{e}": {"test_mean": grid[(hs, e)]["test"]["mean"],
                                     "test_t": grid[(hs, e)]["test"]["t"],
                                     "sw_per_pairday": grid[(hs, e)]["sw_per_pairday"]}
                    for hs in hs_grid for e in ENTRY_BANDS},
           "pbo_entrybands_at_hs": {"hs_bps": hs_ref, "pbo": pbo.get("pbo")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs_switch.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nswitches/pair-day at z=1.0: {grid[(hs_grid[0],1.0)]['sw_per_pairday']}")
    print("Read: find the half-spread column where TEST excess crosses <=0 -- that is the "
          "real bid-ask at which the 'edge' is gone. Plug --spread-file with real tick spreads "
          "to settle it. -> " + str(OUT / "pairs_switch.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
