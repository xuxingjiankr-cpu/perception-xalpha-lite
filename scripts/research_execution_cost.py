"""Execution-cost reality check (SHADOW / research).

Applies the impact-cost literature (Almgren-Chriss; Obizhaeva-Wang square-root
impact) to OUR actual order sizes and ETF liquidity, to answer one question:
is MARKET IMPACT or the BID-ASK SPREAD our dominant trading cost? The answer
decides whether the right lever is gradual execution (Almgren/Obizhaeva) or
passive limit orders that earn the spread (Avellaneda-Stoikov).

No orders, no account/skill calls, no agent state. Reads the quote log only.
Run: py -3.13 scripts/research_execution_cost.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUOTES = ROOT / "outputs" / "t0_intraday_agent" / "minute_quotes.jsonl"

# Representative order sizes (shares) we actually submit, and a small one.
ORDER_SIZES = [20500, 81300, 114600]
SHARES_PER_LOT = 100  # Eastmoney volume is in 手 (lots of 100 shares)
IMPACT_Y = 1.0        # square-root impact coefficient (order ~1, Obizhaeva-Wang/empirical)


def load() -> dict[str, dict[str, Any]]:
    by_code: dict[str, dict[str, Any]] = defaultdict(lambda: {"vol_by_day": defaultdict(float), "px": [], "spreads": []})
    for line in QUOTES.read_text(encoding="utf-8").splitlines():
        try:
            q = json.loads(line)
        except Exception:
            continue
        c = str(q.get("stockCode", "")).zfill(6)
        day = str(q.get("timestamp", ""))[:10]
        v, p = q.get("volume"), q.get("currentPrice")
        bid, ask = q.get("bidPrice1"), q.get("askPrice1")
        if v not in (None, 0) and day:
            by_code[c]["vol_by_day"][day] = max(by_code[c]["vol_by_day"][day], float(v))  # cumulative -> day max
        if p not in (None, 0):
            by_code[c]["px"].append(float(p))
        if bid not in (None, 0) and ask not in (None, 0) and float(ask) > 0:
            mid = (float(bid) + float(ask)) / 2.0
            if mid > 0:
                by_code[c]["spreads"].append((float(ask) - float(bid)) / mid)
    return by_code


def daily_sigma(px: list[float]) -> float:
    if len(px) < 3:
        return 0.01
    rets = [px[i] / px[i - 1] - 1.0 for i in range(1, len(px)) if px[i - 1] > 0]
    if not rets:
        return 0.01
    m = sum(rets) / len(rets)
    sd = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
    # snapshots are ~5 min; ~48 per session -> scale per-step sigma to daily
    return sd * math.sqrt(48)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = load()
    print("=== execution-cost reality check (SHADOW) ===")
    print(f"{'ETF':7} {'ADV(shares)':>14} {'spread_bps':>10} {'sigma_d':>8} | impact for our order sizes (bps)")
    print("-" * 86)
    for c in sorted(data):
        d = data[c]
        vol_days = list(d["vol_by_day"].values())
        adv_lots = (sum(vol_days) / len(vol_days)) if vol_days else 0.0
        adv_shares = adv_lots * SHARES_PER_LOT
        spread = (sum(d["spreads"]) / len(d["spreads"])) if d["spreads"] else None
        spread_bps = spread * 1e4 if spread else None
        sig = daily_sigma(d["px"])
        impacts = []
        for q in ORDER_SIZES:
            if adv_shares > 0:
                part = q / adv_shares
                impact_bps = IMPACT_Y * sig * math.sqrt(part) * 1e4  # sqrt-law temporary impact
                impacts.append(f"{q//1000}k:{impact_bps:.2f}")
            else:
                impacts.append(f"{q//1000}k:?")
        sp = f"{spread_bps:.1f}" if spread_bps is not None else "?"
        print(f"{c:7} {adv_shares:14,.0f} {sp:>10} {sig*100:7.2f}% | " + "  ".join(impacts))
    print("-" * 86)
    print("verdict: if impact (bps) << spread (bps), the dominant cost is the SPREAD, not impact.")
    print("  -> gradual execution (Almgren/Obizhaeva) saves little; the lever is PASSIVE LIMIT ORDERS")
    print("     that EARN the spread (Avellaneda-Stoikov), plus trading persistent (slow) signals (GP).")


if __name__ == "__main__":
    main()
