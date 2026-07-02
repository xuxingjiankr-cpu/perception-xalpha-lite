"""Holdings-book depth watcher -- five-level sina depth for the codes the trading agents
actually HOLD or trade, so the forward L2 exit-timing evidence (research_l2_exit_timing.py:
sell-now-on-adverse-idiosyncratic-OBI beat waiting by ~+0.6bp@15-30s on early days) can ever
apply to the real book. The confirmed-T0 depth collector (collect_l2_depth.py) deliberately
excludes T+1 domestic-equity ETFs (T67 invariant: its universe is product-level confirmed T0
only, feeding the OBI paper contest) -- but the rebalance agent's book IS mostly domestic
(510300/510500/159915/588000), which is exactly where the 2026-07-02 late-sell happened, and
selling yesterday's T+1 inventory intraday is always allowed, so exit timing applies there too.

Watches the union of: the rebalance agent's configured universe, the T0 agent's
legacy_inventory_codes, and (fail-open) the current daily momentum pool. Writes the same row
schema as collect_l2_depth.py to a SEPARATE directory (outputs/holdings_depth/) so the OBI
experiment's data lineage stays clean.

Data-only: NO broker calls, NO orders, NO account state, NO live config. Reuses the audited
fetch/parse/freshness machinery from collect_l2_depth.

Run: py -3.13 scripts/collect_holdings_depth.py [--once] [--poll-seconds 5] [--until 15:00]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from run_etf_paper_trading_agent import ROOT
from collect_l2_depth import (
    CODE_RE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_QUOTE_AGE_SECONDS,
    append_jsonl,
    coverage_record,
    fetch_depth,
    in_continuous_session,
    is_xshg_session,
    now_cn,
)

OUT_DIR = ROOT / "outputs" / "holdings_depth"
REBALANCE_CONFIG = ROOT / "configs" / "etf_paper_trading_agent.json"
T0_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"


def load_holdings_universe() -> list[dict[str, Any]]:
    """Union of rebalance-agent universe + T0 legacy inventory + daily momentum pool.
    Every source is fail-open (missing file -> skipped) except the rebalance config,
    which is the anchor and must exist."""
    selected: dict[tuple[str, str], dict[str, Any]] = {}

    def add(code: str, exchange: str, name: str, source: str) -> None:
        code = str(code).zfill(6)
        exchange = str(exchange).upper()
        if not CODE_RE.fullmatch(code) or exchange not in {"SH", "SZ"}:
            return
        selected.setdefault((exchange, code), {
            "code": code, "exchange": exchange, "name": name,
            "asset_class": "holdings_watch", "confirmation_basis": source,
        })

    cfg = json.loads(REBALANCE_CONFIG.read_text(encoding="utf-8"))
    for u in cfg.get("universe", []):
        add(u.get("stockCode"), u.get("exchange"), str(u.get("name") or ""), "rebalance_universe")

    try:
        t0 = json.loads(T0_CONFIG.read_text(encoding="utf-8"))
        for code in t0.get("strategy", {}).get("legacy_inventory_codes", []) or []:
            exchange = "SH" if str(code).startswith(("5", "6")) else "SZ"
            add(code, exchange, "", "t0_legacy_inventory")
        pool_ref = t0.get("strategy", {}).get("daily_momentum_pool_file")
        if pool_ref:
            pool_path = Path(pool_ref) if Path(pool_ref).is_absolute() else ROOT / pool_ref
            if pool_path.exists():
                pool = json.loads(pool_path.read_text(encoding="utf-8"))
                for item in pool.get("codes", []) or []:
                    code = item.get("code") if isinstance(item, dict) else item
                    exchange = "SH" if str(code).startswith(("5", "6")) else "SZ"
                    add(code, exchange, "", "daily_momentum_pool")
    except Exception:
        pass  # fail-open: the static rebalance universe is the required core

    if not selected:
        raise RuntimeError("holdings-watch universe is empty; refusing to run without codes")
    return [selected[key] for key in sorted(selected)]


def collect_once(universe: list[dict[str, Any]], *, batch_size: int, timeout: float,
                 max_quote_age_seconds: int, write: bool = True) -> dict[str, Any]:
    moment = now_cn()
    rows, fetch_meta = fetch_depth(
        universe, batch_size=batch_size, timeout=timeout,
        max_quote_age_seconds=max_quote_age_seconds, collected_at=moment,
    )
    coverage = coverage_record(universe, rows, fetch_meta, moment)
    fresh_rows = [row for row in rows if row.get("is_fresh")]
    if write:
        trade_date = moment.strftime("%Y-%m-%d")
        append_jsonl(OUT_DIR / f"depth_{trade_date}.jsonl", fresh_rows)
        append_jsonl(OUT_DIR / f"coverage_{trade_date}.jsonl", [coverage])
    return {**coverage, "written_depth_rows": len(fresh_rows) if write else 0}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--until", default="15:00")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-quote-age-seconds", type=int, default=DEFAULT_MAX_QUOTE_AGE_SECONDS)
    args = parser.parse_args()

    universe = load_holdings_universe()
    moment = now_cn()
    if not is_xshg_session(moment.date()):
        print(json.dumps({"status": "skipped_non_trading_day",
                           "trade_date": moment.strftime("%Y-%m-%d"),
                           "expected_count": len(universe)}, ensure_ascii=False))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    until_h, until_m = (int(x) for x in str(args.until).split(":"))
    print(json.dumps({"status": "start", "codes": len(universe),
                       "sources": sorted({u['confirmation_basis'] for u in universe})}, ensure_ascii=False))
    while True:
        moment = now_cn()
        if (moment.hour, moment.minute) >= (until_h, until_m):
            break
        if in_continuous_session(moment):
            try:
                result = collect_once(universe, batch_size=args.batch_size,
                                      timeout=args.timeout_seconds,
                                      max_quote_age_seconds=args.max_quote_age_seconds)
                print(json.dumps({"t": moment.strftime("%H:%M:%S"),
                                   "rows": result.get("written_depth_rows")}, ensure_ascii=False))
            except Exception as exc:  # network hiccups: log and keep polling
                print(json.dumps({"t": moment.strftime("%H:%M:%S"), "error": str(exc)[:200]}))
        if args.once:
            break
        time.sleep(max(1.0, float(args.poll_seconds)))
    print(json.dumps({"status": "end"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
