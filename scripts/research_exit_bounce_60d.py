"""Research: sell-side mirror of the pullback-entry finding. entry_logic_v2 waits for a small
DIP before buying a breakout; this probes whether waiting for a small BOUNCE before selling a
sell-signal candidate gets a better exit price, using the same 60-day Yahoo 5-min history and
the same day-clustered purged train/test split as research_timing_60d.py.

Candidate proxy: at each decision bar, the BOTTOM-momentum names (worst recent return) stand in
for "positions the sell engine would be exiting now" (mirrors research_timing_60d.py's TOP-
momentum proxy for buy candidates -- neither script replays the live agent's actual state, both
are a standalone probe on the observable universe).

Two exits compared, both costed identically (cost cancels in the diff, same as the entry study):
  IMMEDIATE : sell at the signal price (bar 0) -- this is what the agent does today.
  BOUNCE    : wait up to `wait` bars for a `bounce_frac` rise above the signal price and sell
              there; if no bounce appears, fail-open and sell at whatever price exists at the
              end of the wait window (never refuses to exit -- selling stays unconditional).

Day-clustered t-stat on (bounce_realized_price / signal_price - 1), which equals the net-return
difference between BOUNCE and IMMEDIATE once the flat cost cancels.

STRICTLY offline. NO orders, NO broker calls, NO agent/live config touched. diagnostic_only.
Run: py -3.13 scripts/research_exit_bounce_60d.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np

from run_etf_paper_trading_agent import ROOT
from research_timing_60d import load, TRAIN_TEST_SPLIT, WINDOW_BARS, MIN_AMOUNT, DECISION_BAR_TIMES

OUT_DIR = ROOT / "outputs" / "research_timing"
TOP_FRAC = 0.10          # bottom decile by recent momentum = sell-candidate proxy
BOUNCE_FRAC = 0.006      # mirrors the winning entry-side pullback_frac (0.6%)
WAIT_BARS = 2            # mirrors the winning entry-side 10min wait (2 x 5min bars)
FWD_BARS_TO_JUDGE = 6    # how far past the exit we look, purely to sanity-check "did it keep falling"


def analyze_day(day_data: dict[str, dict[str, list]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {
        "immediate_ret": [], "bounce_ret": [], "bounce_filled": [],
        "kept_falling_after_immediate": [],
    }
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
        recent.sort(key=lambda x: x[2])   # ascending: worst movers first = sell-candidate proxy
        k = max(1, int(len(recent) * TOP_FRAC))
        for c, i, _ in recent[:k]:
            n = day_data[c]
            path = n["price"][i:]
            if len(path) < WAIT_BARS + 2:
                continue
            signal_px = path[0]
            # IMMEDIATE: sell at signal price -> net return relative to signal is 0 (cost cancels in the diff)
            out["immediate_ret"].append(0.0)
            # BOUNCE: wait up to WAIT_BARS for a rise, else fail-open at end of wait
            target = signal_px * (1 + BOUNCE_FRAC)
            filled = False
            for j in range(1, min(len(path), WAIT_BARS + 1)):
                if path[j] >= target:
                    out["bounce_ret"].append((path[j] / signal_px - 1.0) * 100)
                    out["bounce_filled"].append(1.0)
                    filled = True
                    break
            if not filled:
                j = min(WAIT_BARS, len(path) - 1)
                out["bounce_ret"].append((path[j] / signal_px - 1.0) * 100)
                out["bounce_filled"].append(0.0)
            # sanity check: did price keep falling after the (hypothetical) immediate exit point?
            far = path[min(FWD_BARS_TO_JUDGE, len(path) - 1)]
            out["kept_falling_after_immediate"].append(1.0 if far < signal_px else 0.0)
    return out


def day_clustered(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if len(vals) < 2:
        return {"days": len(vals), "verdict": "insufficient"}
    arr = np.array(vals)
    t = float(arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr))) if arr.std(ddof=1) else None
    return {
        "days": len(vals), "mean_pct": round(float(arr.mean()), 4),
        "std_pct": round(float(arr.std(ddof=1)), 4),
        "day_clustered_t": round(t, 2) if t is not None else None,
        "days_positive": sum(1 for v in vals if v > 0),
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
        rows.append({"date": d, "n": len(b.get("bounce_ret", [])),
                      "bounce_ret_mean": m("bounce_ret"), "bounce_fill_rate": m("bounce_filled"),
                      "kept_falling_share": m("kept_falling_after_immediate")})

    train = [r for r in rows if r["date"] <= TRAIN_TEST_SPLIT]
    test = [r for r in rows if r["date"] > TRAIN_TEST_SPLIT]

    result = {
        "research_version": 1, "generated_at": datetime.now().astimezone().isoformat(),
        "source": "yahoo_60d_quotes.jsonl (5-min bars, 2026-03-23..2026-06-18)",
        "paper_trading_only": True, "status": "diagnostic_only", "edge_validated": False,
        "live_ready": False, "formal_strategy_allowed": False, "order_submit_calls_made": False,
        "params": {"bounce_frac": BOUNCE_FRAC, "wait_bars": WAIT_BARS, "top_frac": TOP_FRAC},
        "usable_days": len(rows), "train_days": len(train), "test_days": len(test),
        "bounce_vs_immediate_pct": {   # this IS the bounce-vs-immediate net-return diff (cost cancels)
            "train": day_clustered(train, "bounce_ret_mean"),
            "test": day_clustered(test, "bounce_ret_mean"),
        },
        "kept_falling_after_immediate_exit": {
            "train": day_clustered(train, "kept_falling_share"),
            "test": day_clustered(test, "kept_falling_share"),
        },
        "per_day": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exit_bounce_60d.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"usable days: {len(rows)} (train {len(train)} / test {len(test)})")
    print(f"bounce-vs-immediate TRAIN: {result['bounce_vs_immediate_pct']['train']}")
    print(f"bounce-vs-immediate TEST : {result['bounce_vs_immediate_pct']['test']}")
    print(f"kept-falling-after-immediate-exit TRAIN: {result['kept_falling_after_immediate_exit']['train']}")
    print(f"kept-falling-after-immediate-exit TEST : {result['kept_falling_after_immediate_exit']['test']}")
    print(f"log: {OUT_DIR / 'exit_bounce_60d.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
