"""Cross-ETF pair relative-value research (SHADOW / research only).

Tier-1 research direction #2: ETFs with highly overlapping holdings should be
cointegrated, so their price spread mean-reverts. When the spread deviates,
fade it. Pairs: 513050<->513330 (China internet, overlapping baskets) and
513500<->513100 (S&P vs Nasdaq, correlated). Data-self-sufficient: uses the
ETF price history we already collect (minute_quotes.jsonl). No external feed.

Strictly research: NO orders, NO skill/account calls, NO agent state, NO locks.
It reads the existing quote log and prints mean-reversion diagnostics.

Run: py -3.13 scripts/research_cross_etf_pairs.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUOTES = ROOT / "outputs" / "t0_intraday_agent" / "minute_quotes.jsonl"

PAIRS = [
    ("513050", "513330", "中概互联 / 恒生互联(成分重叠)"),
    ("513500", "513100", "标普 / 纳指(相关)"),
]


def _parse_ts(value: Any):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def load_aligned(path: Path) -> list[dict[str, Any]]:
    """Group per-ETF snapshots into FETCH ROUNDS (all ETFs of one run land within
    a few seconds). A new round starts on a code repeat or a >30s gap — the agent
    logs each ETF ~1-2s apart, so second-resolution bucketing would wrongly split
    them."""
    if not path.exists():
        return []
    rounds: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    last_dt = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            q = json.loads(line)
        except Exception:
            continue
        code = str(q.get("stockCode", "")).zfill(6)
        px = q.get("currentPrice")
        ts = q.get("timestamp")
        if not code or px in (None, 0):
            continue
        dt = _parse_ts(ts)
        gap = (dt - last_dt).total_seconds() if dt and last_dt else 0.0
        if cur and (code in cur or gap > 30):
            rounds.append(cur)
            cur = {}
        cur[code] = float(px)
        cur["ts"] = str(ts)[:19]
        if dt:
            last_dt = dt
    if cur:
        rounds.append(cur)
    return rounds


def ou_mean_reversion(spread: list[float]) -> dict[str, Any]:
    """Ornstein-Uhlenbeck style test: regress d(spread) on lagged spread.
    coef<0 => mean-reverting; half-life = -ln(2)/coef. Also reports current z."""
    n = len(spread)
    if n < 5:
        return {"points": n, "ready": False}
    x = spread[:-1]
    dy = [spread[i + 1] - spread[i] for i in range(n - 1)]
    mx = sum(x) / len(x)
    mdy = sum(dy) / len(dy)
    cov = sum((x[i] - mx) * (dy[i] - mdy) for i in range(len(x)))
    var = sum((xi - mx) ** 2 for xi in x)
    coef = cov / var if var > 0 else 0.0
    mean = sum(spread) / n
    sd = (sum((s - mean) ** 2 for s in spread) / n) ** 0.5
    z = (spread[-1] - mean) / sd if sd > 0 else 0.0
    half_life = (-math.log(2) / coef) if coef < 0 else None
    return {
        "points": n, "ready": n >= 40,
        "ou_coef": round(coef, 5),
        "mean_reverting": coef < 0,
        "half_life_steps": round(half_life, 1) if half_life else None,
        "spread_mean": round(mean, 5), "spread_sd": round(sd, 5),
        "current_z": round(z, 2),
        "interpretation": ("mean_reverting(fadeable)" if coef < -1e-4 else
                           ("trending/diverging" if coef > 1e-4 else "no_signal")),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = load_aligned(QUOTES)
    print(f"=== cross-ETF pair relative value (SHADOW, no trading) — {len(rows)} quote rounds ===")
    if not rows:
        print("no quote history yet")
        return
    for a, b, label in PAIRS:
        spread = [math.log(r[a]) - math.log(r[b]) for r in rows if a in r and b in r and r[a] > 0 and r[b] > 0]
        res = ou_mean_reversion(spread)
        tag = "READY" if res.get("ready") else f"thin(n={res.get('points')}, need>=40)"
        if res.get("points", 0) >= 5:
            print(f"  {a}<->{b} {label}: {res['interpretation']} | OU_coef={res.get('ou_coef')} "
                  f"half_life={res.get('half_life_steps')} steps | current_z={res.get('current_z')} | {tag}")
        else:
            print(f"  {a}<->{b} {label}: insufficient overlap (n={res.get('points')})")
    print("note: log-price spread; OU_coef<0 => reverting. Trust needs >=40 aligned points / multiple days.")


if __name__ == "__main__":
    main()
