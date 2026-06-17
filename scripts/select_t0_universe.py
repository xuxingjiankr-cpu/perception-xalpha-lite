"""Dynamic T0 universe selection from the full-market ETF snapshot.

We now collect the full A-share ETF cross-section (~1,400 names, money-market
funds excluded) via scripts/collect_eastmoney_full_market.py. This module turns
that wide, cheap snapshot into a small TRADABLE candidate universe for the T0
agent: apply liquidity/tradability gates, then rank cross-sectionally -- the
breadth that finally makes rank-based selection meaningful (it is noise at N=7)
-- and take the top-N. Pure data-driven, no hand-picked thematic core.

Scope decision (with the user): the full universe INCLUDING T+1 equity ETFs.
A T+1 name that gets bought simply becomes an overnight carry -- the broker
reports 0 same-day sellable quantity, so the agent's pre-sell broker check drops
any same-day exit and the position is carried until sellable. That is consistent
with the market-driven (not time-driven) exit policy already in place.

The ranking uses ONLY intraday, single-snapshot fields (no multi-day history is
required, so it works from day one): cross-sectional z-score of intraday return
plus Alpha#101 conviction (close-open)/(high-low). The deeper, multi-day rank()
alphas (adv20, long correlations) come online later as snapshot history builds.

Safety: NO broker calls and NO order side effects. Reads snapshot files and
writes a JSON universe file the agent consumes. Held codes and the original
static seed are always unioned back in by resolve_agent_universe so a position
can never be orphaned out of the universe (which would block its exit).

CLI: py -3.13 scripts/select_t0_universe.py [--config configs/t0_intraday_paper_agent.json]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT, as_float, load_json


SESSION_MINUTES = 240.0  # A-share continuous session length (09:30-11:30 + 13:00-15:00)


def _market_to_exchange(market: Any) -> str:
    """Eastmoney market code -> exchange. 1 = Shanghai, 0 = Shenzhen."""
    return "SH" if str(market).strip() == "1" else "SZ"


def _session_minutes_elapsed(hhmm_dt: datetime) -> float:
    """Minutes of continuous trading elapsed by the given China-time timestamp,
    accounting for the 11:30-13:00 lunch break. Clamped to [0, SESSION_MINUTES]."""
    m = hhmm_dt.hour * 60 + hhmm_dt.minute
    open_am, close_am = 9 * 60 + 30, 11 * 60 + 30
    open_pm, close_pm = 13 * 60, 15 * 60
    if m <= open_am:
        return 0.0
    if m <= close_am:
        return float(m - open_am)
    if m <= open_pm:
        return 120.0
    if m <= close_pm:
        return 120.0 + float(m - open_pm)
    return SESSION_MINUTES


def latest_snapshot_path(snapshot_dir: Path, trade_date: str) -> Path | None:
    day_dir = snapshot_dir / trade_date
    if not day_dir.exists():
        return None
    files = sorted(day_dir.glob("*_etf.csv.gz"))
    return files[-1] if files else None


def load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8")
        return list(csv.DictReader(text))


def load_excluded_money_codes(universe_dir: Path, trade_date: str) -> set[str]:
    """Belt-and-suspenders: drop any code the collector flagged as a money fund."""
    out: set[str] = set()
    if not universe_dir.exists():
        return out
    stamp = trade_date.replace("-", "")
    for pat in (f"*excluded_money_funds*{stamp}*.jsonl", "*excluded_money_funds*.jsonl"):
        for f in universe_dir.glob(pat):
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                code = str(rec.get("stockCode", "")).zfill(6)
                if code:
                    out.add(code)
        if out:
            break
    return out


def _spread_pct(row: dict[str, Any]) -> float | None:
    bid = as_float(row.get("bidPrice1"), 0.0)
    ask = as_float(row.get("askPrice1"), 0.0)
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else None


def passes_gates(row: dict[str, Any], gates: dict[str, Any], min_amount_effective: float,
                 excluded: set[str]) -> tuple[bool, str]:
    code = str(row.get("stockCode", "")).zfill(6)
    if code in excluded:
        return False, "money_fund_excluded"
    name = str(row.get("name", ""))
    for kw in gates.get("name_exclude_keywords", ["货币", "现金", "理财"]):
        if kw and kw in name:
            return False, "name_excluded"
    price = as_float(row.get("currentPrice"), 0.0)
    if price <= as_float(gates.get("min_price", 0.3), 0.3):
        return False, "price_too_low"
    spread = _spread_pct(row)
    if spread is None:
        return False, "no_valid_book"
    if spread > as_float(gates.get("max_spread_pct", 0.004), 0.004):
        return False, "spread_too_wide"
    amount = as_float(row.get("amount"), 0.0)
    if amount < min_amount_effective:
        return False, "illiquid"
    return True, "ok"


def _conviction(row: dict[str, Any]) -> float:
    """Alpha#101 from the snapshot's own OHLC: (close-open)/(high-low+eps)."""
    cur = as_float(row.get("currentPrice"), 0.0)
    op = as_float(row.get("open"), 0.0)
    hi = as_float(row.get("high"), 0.0)
    lo = as_float(row.get("low"), 0.0)
    rng = hi - lo
    if cur <= 0 or op <= 0 or rng <= 0:
        return 0.0
    return max(-1.0, min(1.0, (cur - op) / (rng + 1e-9)))


def _zscores(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = var ** 0.5
    if sd <= 0:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def select_dynamic_universe(snapshot_dir: Path, universe_dir: Path, trade_date: str,
                            dyn_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return {"selected": [...], "meta": {...}} or an empty selection on failure."""
    # top_n <= 0 means "all eligible after gates". This keeps the source universe
    # as the full ETF cross-section with money-market funds removed, while the
    # existing liquidity/spread gates still prevent untradeable names entering
    # the agent's ranking.
    top_n = int(as_float(dyn_cfg.get("top_n", 25), 25))
    gates = dyn_cfg
    path = latest_snapshot_path(snapshot_dir, trade_date)
    if path is None:
        return {"selected": [], "meta": {"ok": False, "reason": "no_snapshot_for_date", "trade_date": trade_date}}
    rows = load_snapshot_rows(path)
    if not rows:
        return {"selected": [], "meta": {"ok": False, "reason": "empty_snapshot", "snapshot": path.name}}

    # session fraction from the snapshot's collection time scales the liquidity gate
    # (cumulative amount grows through the day, so a fixed full-day floor is scaled).
    try:
        stamp = datetime.fromisoformat(str(rows[0].get("collected_at"))).astimezone(ZoneInfo("Asia/Shanghai"))
        frac = max(0.05, _session_minutes_elapsed(stamp) / SESSION_MINUTES)
    except Exception:
        frac = 1.0
    min_amount_full = as_float(dyn_cfg.get("min_amount_yuan", 50_000_000), 50_000_000)
    min_amount_effective = min_amount_full * frac

    excluded = load_excluded_money_codes(universe_dir, trade_date)
    extra_exclude = {str(c).zfill(6) for c in dyn_cfg.get("exclude_codes", [])}

    eligible: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    for row in rows:
        code = str(row.get("stockCode", "")).zfill(6)
        if code in extra_exclude:
            rejects["config_excluded"] = rejects.get("config_excluded", 0) + 1
            continue
        ok, reason = passes_gates(row, gates, min_amount_effective, excluded)
        if not ok:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        eligible.append(row)

    if not eligible:
        return {"selected": [], "meta": {"ok": False, "reason": "no_eligible_after_gates",
                                         "snapshot": path.name, "rejects": rejects}}

    momentum = [as_float(r.get("change_pct"), 0.0) for r in eligible]
    conviction = [_conviction(r) for r in eligible]
    mom_w = as_float(dyn_cfg.get("momentum_weight", 1.0), 1.0)
    conv_w = as_float(dyn_cfg.get("conviction_weight", 0.5), 0.5)
    zmom = _zscores(momentum)
    zconv = _zscores(conviction)
    scored = []
    for i, r in enumerate(eligible):
        composite = mom_w * zmom[i] + conv_w * zconv[i]
        scored.append((composite, r, momentum[i], conviction[i]))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    selected_rows = scored if top_n <= 0 else scored[:top_n]
    for composite, r, mom, conv in selected_rows:
        selected.append({
            "stockCode": str(r.get("stockCode", "")).zfill(6),
            "exchange": _market_to_exchange(r.get("market")),
            "name": r.get("name"),
            "asset_class": "dynamic",
            "inclusion": "ranked",
            "rank_score": round(composite, 4),
            "change_pct": round(mom, 3),
            "conviction": round(conv, 4),
            "amount": as_float(r.get("amount"), 0.0),
        })
    meta = {
        "ok": True, "trade_date": trade_date, "snapshot": path.name,
        "eligible": len(eligible), "selected": len(selected),
        "top_n": top_n,
        "selection_scope": "all_eligible_after_gates" if top_n <= 0 else "top_n_after_gates",
        "session_fraction": round(frac, 3), "min_amount_effective": round(min_amount_effective, 0),
        "rejects": rejects,
    }
    return {"selected": selected, "meta": meta}


def _held_codes_from_state(state: dict[str, Any], trade_date: str) -> set[str]:
    out: set[str] = set()
    inv = state.get("t0_inventory_by_date", {})
    if isinstance(inv, dict):
        day = inv.get(trade_date, {})
        if isinstance(day, dict):
            for code, node in day.items():
                if not isinstance(node, dict):
                    continue
                bought = as_float(node.get("buy_quantity_submitted"), 0.0)
                sold = as_float(node.get("sell_quantity_submitted"), 0.0)
                if bought - sold > 0:
                    out.add(str(code).zfill(6))
    return out


def resolve_agent_universe(cfg: dict[str, Any], state: dict[str, Any], trade_date: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Universe the agent should trade this run. When dynamic_universe is enabled:
    data-driven top-N UNION (original static seed + today's held codes) so a held
    position is never orphaned (orphaning would block its exit). Falls back to the
    static cfg universe if selection yields nothing."""
    dyn_cfg = cfg.get("dynamic_universe", {}) if isinstance(cfg.get("dynamic_universe"), dict) else {}
    static_universe = cfg.get("universe", [])
    if not bool(dyn_cfg.get("enabled", False)):
        return static_universe, {"mode": "static"}

    snapshot_dir = ROOT / dyn_cfg.get("snapshot_dir", "data/market/eastmoney/full_market/snapshots")
    universe_dir = ROOT / dyn_cfg.get("universe_dir", "data/market/eastmoney/universe")
    result = select_dynamic_universe(snapshot_dir, universe_dir, trade_date, dyn_cfg)
    selected = result["selected"]
    out_path = dyn_cfg.get("output")
    if selected and out_path:
        p = ROOT / out_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"selected": selected, "meta": result["meta"]}, ensure_ascii=False, indent=2), encoding="utf-8")

    if not selected:
        # fall back to a previously written file, else the static universe
        if out_path and (ROOT / out_path).exists():
            try:
                prev = json.loads((ROOT / out_path).read_text(encoding="utf-8")).get("selected", [])
            except Exception:
                prev = []
            if prev:
                merged = _union_universe(prev, static_universe, state, trade_date)
                return merged, {"mode": "dynamic_stale_file", "selection_meta": result["meta"], "count": len(merged)}
        return static_universe, {"mode": "static_fallback", "selection_meta": result["meta"]}

    merged = _union_universe(selected, static_universe, state, trade_date)
    return merged, {"mode": "dynamic", "selection_meta": result["meta"], "count": len(merged),
                    "ranked": len(selected)}


def _union_universe(selected: list[dict[str, Any]], static_universe: list[dict[str, Any]],
                    state: dict[str, Any], trade_date: str) -> list[dict[str, Any]]:
    by_code: dict[str, dict[str, Any]] = {}
    for item in selected:
        by_code[str(item.get("stockCode", "")).zfill(6)] = dict(item)
    # static seed: preserve original asset_class (gold/cross_border/exit_only_t1...)
    # so asset-class-specific logic still applies; included for EXIT safety only.
    for item in static_universe:
        code = str(item.get("stockCode", "")).zfill(6)
        if code not in by_code:
            seed = dict(item)
            seed["inclusion"] = "static_seed"
            by_code[code] = seed
    # today's held codes from state not otherwise present -> add minimal entry
    for code in _held_codes_from_state(state, trade_date):
        if code not in by_code:
            exch = "SH" if code.startswith(("5", "6")) else "SZ"
            by_code[code] = {"stockCode": code, "exchange": exch, "name": code,
                             "asset_class": "dynamic", "inclusion": "held_exit_safety"}
    return list(by_code.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Select the dynamic T0 ETF universe from the full-market snapshot.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "t0_intraday_paper_agent.json"))
    ap.add_argument("--date", default=datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    cfg = load_json(Path(args.config))
    dyn_cfg = cfg.get("dynamic_universe", {})
    snapshot_dir = ROOT / dyn_cfg.get("snapshot_dir", "data/market/eastmoney/full_market/snapshots")
    universe_dir = ROOT / dyn_cfg.get("universe_dir", "data/market/eastmoney/universe")
    result = select_dynamic_universe(snapshot_dir, universe_dir, args.date, dyn_cfg)
    meta = result["meta"]
    print(f"=== dynamic T0 universe selection {args.date} ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    for item in result["selected"][:30]:
        print(f"  {item['stockCode']} {item['exchange']} {item.get('name')}: "
              f"score={item['rank_score']} chg={item['change_pct']}% conv={item['conviction']} amount={item['amount']:.0f}")


if __name__ == "__main__":
    main()
