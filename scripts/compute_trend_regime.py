"""Nightly market trend-regime computer (Gayed-style trend-conditioned exposure).

Fetches the broad-market index (沪深300, 510300) daily from Yahoo, compares close to its N-day
moving average, and writes a deploy factor the T0 agent uses to scale position sizing:
  uptrend (close >= MA)   -> full exposure  (deploy_factor = up)
  downtrend (close <  MA)  -> de-risk        (deploy_factor = down)
This keeps the up-market upside but trims drawdowns in down-markets -- risk-managed BETA, not
alpha. A small hysteresis band avoids whipsaw on tiny MA crossings.

Fail-open: the agent uses deploy_factor=1.0 if this file is missing/stale (never halts).
Run nightly: py -3.13 scripts/compute_trend_regime.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from run_etf_paper_trading_agent import ROOT

CST = timezone(timedelta(hours=8))
INDEX = "510300"           # 沪深300 broad-market proxy
MA_WINDOW = 50
HYSTERESIS = 0.01          # 1% band: downtrend only if close < MA*(1-band); reduces whipsaw
UP_FACTOR = 1.0            # full exposure in uptrend
DOWN_FACTOR = 0.3          # de-risk to ~30% sizing in downtrend
OUT = ROOT / "outputs" / "t0_intraday_agent" / "trend_regime.json"


def fetch_daily_closes(code: str, timeout: float = 20.0) -> list[float]:
    sym = f"{code}.SS" if code[:1] in ("5", "6") else f"{code}.SZ"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=6mo"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    res = (d.get("chart", {}).get("result") or [None])[0]
    if not res:
        return []
    q = res["indicators"]["quote"][0].get("close") or []
    return [float(p) for p in q if p and p > 0]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    closes = fetch_daily_closes(INDEX)
    if len(closes) < MA_WINDOW + 1:
        print(f"insufficient history ({len(closes)}); leaving trend file untouched (agent fail-opens).")
        return 1
    close = closes[-1]
    ma = sum(closes[-MA_WINDOW:]) / MA_WINDOW
    # hysteresis: stay/declare uptrend unless clearly below the MA band
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    if close >= ma:
        regime = "uptrend"
    elif close < ma * (1 - HYSTERESIS):
        regime = "downtrend"
    else:
        regime = prev.get("regime", "uptrend")   # in the band -> hold previous
    deploy_factor = UP_FACTOR if regime == "uptrend" else DOWN_FACTOR
    payload = {
        "date": datetime.now(CST).strftime("%Y-%m-%d"),
        "index": INDEX, "ma_window": MA_WINDOW,
        "close": round(close, 4), "ma": round(ma, 4),
        "pct_vs_ma": round((close / ma - 1) * 100, 2),
        "regime": regime, "deploy_factor": deploy_factor,
        "generated_at": datetime.now(CST).isoformat(),
        "note": "Gayed-style trend filter; agent scales max_position_pct by deploy_factor. "
                "Risk-managed beta, not alpha. Revert by removing strategy.trend_deployment.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trend regime: {regime} (沪深300 close {close:.3f} vs {MA_WINDOW}d MA {ma:.3f}, "
          f"{payload['pct_vs_ma']:+.1f}%) -> deploy_factor {deploy_factor} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
