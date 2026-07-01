"""Research: extend research_timing.py's pullback-entry-vs-breakout finding (10 live days,
day-clustered t=4.80, 10/10 days favoring pullback even after a realistic round-trip cost) to
the 60-trading-day Yahoo 5-min history (outputs/t0_replay/yahoo_60d_quotes.jsonl,
2026-03-23..2026-06-18, 612 codes), so the purged-OOS minimum (>=20 days, train/test split) can
actually be checked instead of waited on live-snapshot accumulation day by day.

Same entry/exit logic as research_timing.py (range_percentile, pullback_entry, trailing_exit),
adapted to 5-min bars instead of 1-min eastmoney snapshots: window/pullback_wait/trail params
are expressed in BAR units here (5 min/bar), not minutes. Cost uses the per-quote real
spread_pct field already present in the Yahoo data (bid/ask), not a flat assumption.

Purged split mirrors the repo convention used in ablate_entry_signals.py / ablate_sell_logic_v2.py:
train <= 2026-05-20, test > 2026-05-20. Day-clustered t-stats computed separately on train and
test so the live-sample finding (formed on data the model never saw) can be checked for an
out-of-sample replication, not just refit.

STRICTLY offline. NO orders, NO broker calls, NO agent/live config touched. diagnostic_only.
Run: py -3.13 scripts/research_timing_60d.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT
from research_timing import range_percentile, pullback_entry, trailing_exit

QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
OUT_DIR = ROOT / "outputs" / "research_timing"
TRAIN_TEST_SPLIT = "2026-05-20"   # repo convention: train <= split < test

# bar-unit params (5-min bars). window=4 bars=20min, pullback_wait=3 bars=15min match the
# live (1-min) study's minute settings as closely as integer bars allow.
WINDOW_BARS = 4
PULLBACK_FRAC = 0.004
PULLBACK_WAIT_BARS = 3
TRAIL_FRAC = 0.006
TOP_FRAC = 0.10
MIN_AMOUNT = 50_000_000.0
DECISION_BAR_TIMES = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]


def load() -> dict[str, dict[str, dict[str, list]]]:
    """date -> code -> {'times': [...], 'price': [...], 'amount': [...], 'spread_pct': [...]}"""
    by_date: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(
        lambda: {"times": [], "price": [], "amount": [], "spread_pct": []}))
    with QUOTES.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("quote_ok") or r.get("isSuspended"):
                continue
            ts = r.get("timestamp", "")
            d, t = ts[:10], ts[11:16]
            px = r.get("currentPrice")
            if not d or not t or not px or px <= 0:
                continue
            node = by_date[d][r["stockCode"]]
            node["times"].append(t)
            node["price"].append(float(px))
            node["amount"].append(float(r.get("amount") or 0.0))
            node["spread_pct"].append(float(r.get("spread_pct") or 0.0008))


    return by_date


def analyze_day(day_data: dict[str, dict[str, list]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {
        "entry_pct_breakout": [], "entry_pct_pullback": [], "pullback_filled": [],
        "fwd_ret_breakout": [], "fwd_ret_pullback": [],
        "exit_pct_hold": [], "exit_pct_trail": [], "ret_hold": [], "ret_trail": [],
    }
    # index by time for each code
    idx = {c: {t: i for i, t in enumerate(n["times"])} for c, n in day_data.items()}
    for dt in DECISION_BAR_TIMES:
        eligible = []
        for c, n in day_data.items():
            i = idx[c].get(dt)
            if i is None or i < WINDOW_BARS:
                continue
            if sum(n["amount"][max(0, i - WINDOW_BARS):i + 1]) / (WINDOW_BARS + 1) < MIN_AMOUNT / 48:
                continue
            eligible.append((c, i))
        if len(eligible) < 20:
            continue
        recent = []
        for c, i in eligible:
            n = day_data[c]
            p_now, p_past = n["price"][i], n["price"][i - WINDOW_BARS]
            if p_now and p_past:
                recent.append((c, i, p_now / p_past - 1.0))
        if len(recent) < 20:
            continue
        recent.sort(key=lambda x: x[2], reverse=True)
        k = max(1, int(len(recent) * TOP_FRAC))
        for c, i, _ in recent[:k]:
            n = day_data[c]
            path = n["price"][i:]
            spread = n["spread_pct"][i] if i < len(n["spread_pct"]) else 0.0008
            cost = spread + 0.0006   # real half-spread*2 proxy (spread_pct is full bid-ask) + commission
            if len(path) < 3:
                continue
            lo, hi = min(path), max(path)
            entry_b = path[0]
            pb = range_percentile(entry_b, lo, hi)
            if pb is not None:
                out["entry_pct_breakout"].append(pb * 100)
                out["fwd_ret_breakout"].append((path[-1] / entry_b - 1.0 - cost) * 100)
            pull = pullback_entry(path, PULLBACK_FRAC, PULLBACK_WAIT_BARS)
            out["pullback_filled"].append(1.0 if pull else 0.0)
            if pull:
                entry_p, j = pull
                rem = path[j:]
                lo2, hi2 = min(rem), max(rem)
                pp = range_percentile(entry_p, lo2, hi2)
                if pp is not None:
                    out["entry_pct_pullback"].append(pp * 100)
                    out["fwd_ret_pullback"].append((rem[-1] / entry_p - 1.0 - cost) * 100)
            eh = range_percentile(path[-1], lo, hi)
            if eh is not None:
                out["exit_pct_hold"].append(eh * 100)
                out["ret_hold"].append((path[-1] / entry_b - 1.0 - cost) * 100)
            ex_price, _ = trailing_exit(path, TRAIL_FRAC)
            et = range_percentile(ex_price, lo, hi)
            if et is not None:
                out["exit_pct_trail"].append(et * 100)
                out["ret_trail"].append((ex_price / entry_b - 1.0 - cost) * 100)
    return out


def day_clustered(rows: list[dict], key_a: str, key_b: str) -> dict:
    diffs = [r[key_a] - r[key_b] for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    if len(diffs) < 2:
        return {"days": len(diffs), "verdict": "insufficient"}
    arr = np.array(diffs)
    t = float(arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr))) if arr.std(ddof=1) else None
    return {
        "days": len(diffs), "mean_diff_pct": round(float(arr.mean()), 4),
        "std_pct": round(float(arr.std(ddof=1)), 4),
        "day_clustered_t": round(t, 2) if t is not None else None,
        "days_favoring_first": sum(1 for d in diffs if d > 0),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    by_date = load()
    rows = []
    for d in sorted(by_date):
        b = analyze_day(by_date[d])
        if not any(b.values()):
            continue
        def m(k):
            v = b.get(k, [])
            return round(sum(v) / len(v), 4) if v else None
        rows.append({"date": d, "n": len(b.get("fwd_ret_breakout", [])),
                      "breakout_fwd": m("fwd_ret_breakout"), "pullback_fwd": m("fwd_ret_pullback"),
                      "hold_ret": m("ret_hold"), "trail_ret": m("ret_trail"),
                      "pullback_fill": m("pullback_filled")})

    train = [r for r in rows if r["date"] <= TRAIN_TEST_SPLIT]
    test = [r for r in rows if r["date"] > TRAIN_TEST_SPLIT]

    result = {
        "research_version": 1, "generated_at": datetime.now().astimezone().isoformat(),
        "source": "yahoo_60d_quotes.jsonl (5-min bars, 2026-03-23..2026-06-18)",
        "paper_trading_only": True, "status": "diagnostic_only", "edge_validated": False,
        "live_ready": False, "formal_strategy_allowed": False, "order_submit_calls_made": False,
        "usable_days": len(rows), "train_days": len(train), "test_days": len(test),
        "train_test_split": f"train<= {TRAIN_TEST_SPLIT} < test",
        "pullback_vs_breakout": {
            "train": day_clustered(train, "pullback_fwd", "breakout_fwd"),
            "test": day_clustered(test, "pullback_fwd", "breakout_fwd"),
        },
        "hold_vs_trail": {
            "train": day_clustered(train, "hold_ret", "trail_ret"),
            "test": day_clustered(test, "hold_ret", "trail_ret"),
        },
        "per_day": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "timing_60d_purged.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"usable days: {len(rows)} (train {len(train)} / test {len(test)})")
    print(f"pullback-breakout TRAIN: {result['pullback_vs_breakout']['train']}")
    print(f"pullback-breakout TEST : {result['pullback_vs_breakout']['test']}")
    print(f"hold-trail TRAIN: {result['hold_vs_trail']['train']}")
    print(f"hold-trail TEST : {result['hold_vs_trail']['test']}")
    print(f"log: {OUT_DIR / 'timing_60d_purged.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
