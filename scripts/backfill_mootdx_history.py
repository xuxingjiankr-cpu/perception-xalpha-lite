"""Backfill deep intraday history from TDX servers via mootdx (found via HKUDS/Vibe-Trading's
data-source list, 2026-07-03). Free, no key. Materially deeper than our other free sources:
5-min bars reach back ~13 months (Yahoo caps at 60 days) and 1-min bars ~4.5 months.

Purpose: research data only -- lets the 60-day replay validations (entry_logic_v2 pullback
entry, trend filter, exit ablations) be re-checked across a full year and multiple regimes
instead of waiting on forward-day accumulation. Writes one JSONL per code per frequency to
data/market/mootdx/bars_{freq}/{code}.jsonl with rows {dt, open, high, low, close, vol, amount}.

Idempotent: re-running refreshes each file completely (TDX pagination is depth-from-now, so
appending would duplicate). NO broker calls, NO orders, NO live config.

Run: py -3.13 scripts/backfill_mootdx_history.py [--freq 5m|1m] [--codes 510300,159915]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from run_etf_paper_trading_agent import ROOT

OUT_BASE = ROOT / "data" / "market" / "mootdx"
PAGE = 800

# core research universe: rebalance agent + T0 legacy inventory + OBI/L2 T0 names
DEFAULT_CODES = [
    "510300", "510500", "159915", "588000", "518880", "511010", "513500", "513100",  # rebalance
    "588030", "159546",                                                              # t0 legacy
    "159934", "159941", "159920", "513050", "513180", "159985", "501018", "510050",  # OBI/liquid
]
FREQ_MAP = {"5m": 0, "1m": 8, "1d": 9}   # TDX frequency ids used by mootdx bars()
MASTER = ROOT / "outputs" / "edge_research" / "t0_etf_master_latest.jsonl"


def master_codes() -> list[str]:
    """All non-money ETFs from the audited master (daily-frequency zoo scans)."""
    codes = []
    for line in MASTER.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("is_money_like"):
            continue
        code = str(r.get("code") or "").zfill(6)
        if len(code) == 6 and code.isdigit():
            codes.append(code)
    return sorted(set(codes))


def fetch_all(client: Any, symbol: str, frequency: int, max_pages: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(max_pages):
        for attempt in range(3):
            try:
                df = client.bars(symbol=symbol, frequency=frequency, offset=PAGE, start=page * PAGE)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2.0)
        if df is None or not len(df):
            return rows
        for dt, r in df.iterrows():
            rows.append({
                "dt": str(dt), "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "vol": float(r.get("vol") or 0.0), "amount": float(r.get("amount") or 0.0),
            })
        if len(df) < PAGE:
            return rows
        time.sleep(0.3)   # be polite to the free TDX servers
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freq", default="5m", choices=sorted(FREQ_MAP))
    ap.add_argument("--codes", default=",".join(DEFAULT_CODES))
    ap.add_argument("--from-master", action="store_true",
                    help="ignore --codes; use every non-money ETF from the audited master")
    ap.add_argument("--max-pages", type=int, default=60,
                    help="pagination cap per code (daily: 2 pages = ~6.5 years)")
    args = ap.parse_args()

    from mootdx.quotes import Quotes
    client = Quotes.factory(market="std")

    out_dir = OUT_BASE / f"bars_{args.freq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = master_codes() if args.from_master else [c.strip() for c in args.codes.split(",") if c.strip()]
    summary = []
    for code in codes:
        try:
            rows = fetch_all(client, code, FREQ_MAP[args.freq], max_pages=args.max_pages)
        except Exception as exc:
            print(json.dumps({"code": code, "error": str(exc)[:200]}, ensure_ascii=False))
            summary.append({"code": code, "rows": 0, "error": True})
            continue
        rows.sort(key=lambda r: r["dt"])
        # dedupe on dt (pagination overlap safety)
        seen: set[str] = set()
        unique = [r for r in rows if not (r["dt"] in seen or seen.add(r["dt"]))]
        path = out_dir / f"{code}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in unique:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        first = unique[0]["dt"] if unique else None
        last = unique[-1]["dt"] if unique else None
        print(json.dumps({"code": code, "rows": len(unique), "first": first, "last": last}, ensure_ascii=False))
        summary.append({"code": code, "rows": len(unique), "first": first, "last": last})
    (OUT_BASE / f"backfill_{args.freq}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
