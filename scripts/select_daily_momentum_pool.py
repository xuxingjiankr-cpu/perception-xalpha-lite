"""Nightly DAILY-momentum ETF pool selector (offline research/selection; no broker, no orders).

Ranks the liquid ETF universe by MULTI-DAY close momentum (default 20 trading days) using
free Yahoo daily bars, and writes the top-N (default 20) to a pool file the agent reads next
session to RESTRICT its tradeable universe (see select_t0_universe._load_daily_momentum_pool).

Why daily, not intraday: our IC audit showed intraday momentum REVERTS (negative IC), while
DAILY momentum has the right sign. So pool selection on the daily horizon is the correct use
of momentum; the intraday entry weight is deliberately NOT cranked.

Candidate set = today's gate-passing names (latest_dynamic_universe.json) when available
(already liquid, fewer fetches), else the full money-fund-excluded universe jsonl.

Run nightly after close: py -3.13 scripts/select_daily_momentum_pool.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from run_etf_paper_trading_agent import ROOT, as_float

LOOKBACK_DAYS = 20
TOP_N = 20
MIN_BARS = LOOKBACK_DAYS + 2
CST = timezone(timedelta(hours=8))
GATE_UNIVERSE = ROOT / "outputs" / "t0_intraday_agent" / "latest_dynamic_universe.json"
UNIVERSE_DIR = ROOT / "data" / "market" / "eastmoney" / "universe"
OUT = ROOT / "outputs" / "t0_intraday_agent" / "daily_momentum_pool.json"


def yahoo_symbol(code: str) -> str:
    return f"{code}.SS" if code[:1] in ("5", "6") else f"{code}.SZ"


def candidate_codes() -> list[tuple[str, str]]:
    """(code, name) candidates -- prefer today's gate-passing liquid set."""
    if GATE_UNIVERSE.exists():
        try:
            d = json.loads(GATE_UNIVERSE.read_text(encoding="utf-8"))
            sel = d.get("selected") if isinstance(d, dict) else None
            if sel:
                seen: dict[str, str] = {}
                for r in sel:
                    c = str(r.get("stockCode", "")).zfill(6)
                    if c:
                        seen.setdefault(c, str(r.get("name") or c))
                if len(seen) >= 30:
                    return sorted(seen.items())
        except Exception:
            pass
    # fallback: latest money-fund-excluded universe jsonl
    files = sorted(UNIVERSE_DIR.glob("eastmoney_universe_etf_*.jsonl"))
    seen = {}
    if files:
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                c = str(r.get("stockCode", "")).zfill(6)
                if c:
                    seen.setdefault(c, str(r.get("name") or c))
            except Exception:
                continue
    return sorted(seen.items())


def fetch_closes(code: str, timeout: float = 15.0) -> list[float]:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(code)}"
           f"?interval=1d&range=2mo")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    res = (d.get("chart", {}).get("result") or [None])[0]
    if not res or "timestamp" not in res:
        return []
    q = res["indicators"]["quote"][0].get("close") or []
    return [float(p) for p in q if p and p > 0]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cands = candidate_codes()
    if not cands:
        print("no candidate universe found; aborting (agent keeps full universe).")
        return 1
    print(f"ranking {len(cands)} ETFs by {LOOKBACK_DAYS}d momentum ...")
    ranked: list[tuple[float, str, str]] = []
    ok = 0
    for n, (code, name) in enumerate(cands):
        try:
            closes = fetch_closes(code)
        except Exception:
            closes = []
        if len(closes) >= MIN_BARS:
            mom = closes[-1] / closes[-1 - LOOKBACK_DAYS] - 1.0
            ranked.append((mom, code, name))
            ok += 1
        if n % 100 == 0:
            print(f"  {n}/{len(cands)} ok={ok}", flush=True)
        time.sleep(0.12)
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:TOP_N]
    today = datetime.now(CST).strftime("%Y-%m-%d")
    payload = {
        "date": today,
        "lookback_days": LOOKBACK_DAYS,
        "top_n": TOP_N,
        "ranked_candidates": len(ranked),
        "codes": [c for _, c, _ in top],
        "detail": [{"stockCode": c, "name": nm, "momentum_pct": round(m * 100, 3)} for m, c, nm in top],
        "generated_at": datetime.now(CST).isoformat(),
        "note": "DAILY-momentum leader pool; agent restricts tradeable universe to these (fail-open). "
                "Competition tilt -- revert with the bold-play knobs after July.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== top {len(top)} of {len(ranked)} by {LOOKBACK_DAYS}d momentum -> {OUT} ===")
    for m, c, nm in top:
        print(f"  {c} {nm}: {m*100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
