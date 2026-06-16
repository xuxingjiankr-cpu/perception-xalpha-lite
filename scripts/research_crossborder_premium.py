"""Cross-border ETF premium research collector (SHADOW / research only).

Tier-1 research (see review): US-tracking QDII ETFs (513500 S&P, 513100 Nasdaq)
trade while the US cash market is closed, so their intraday price is driven by
US index FUTURES + USD/CNH, plus a creation/redemption premium that QDII quota
limits make persistent. This tool builds the aligned reference history needed to
test whether the intraday premium MEAN-REVERTS.

It is strictly research: NO orders, NO account/skill calls, NO agent state, NO
locks. It only fetches free market data (Sina) and appends a log. The
mean-reversion analysis activates once enough aligned history accumulates.

Run: py -3.13 scripts/research_crossborder_premium.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "crossborder_premium"
LOG_PATH = OUT_DIR / "premium_log.jsonl"

# ETF -> US index futures (Sina hf_ symbol). 513500 tracks S&P500 (ES),
# 513100 tracks Nasdaq-100 (NQ). FX leg is offshore USD/CNH (fx_susdcnh).
PAIRS = [
    {"etf": "513500", "etf_sym": "sh513500", "name": "标普500ETF", "fut": "hf_ES", "fut_name": "S&P500_fut"},
    {"etf": "513100", "etf_sym": "sh513100", "name": "纳指ETF", "fut": "hf_NQ", "fut_name": "Nasdaq100_fut"},
]
FX_SYM = "fx_susdcnh"


def now_cn() -> datetime:
    return datetime.now(tz=ZoneInfo("Asia/Shanghai"))


def _sina(symbols: list[str], timeout: float = 8.0) -> dict[str, list[str]]:
    url = "https://hq.sinajs.cn/list=" + ",".join(symbols)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("gbk", errors="replace")
    out: dict[str, list[str]] = {}
    for line in text.strip().split("\n"):
        if "hq_str_" not in line or '="' not in line:
            continue
        sym = line.split("hq_str_")[1].split("=")[0]
        out[sym] = line.split('="', 1)[1].strip().strip(";").strip('"').split(",")
    return out


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def collect_snapshot() -> dict[str, Any]:
    """One aligned snapshot of ETF + futures + CNH levels (raw levels only)."""
    syms = [p["etf_sym"] for p in PAIRS] + [p["fut"] for p in PAIRS] + [FX_SYM]
    raw = _sina(syms)
    cnh_rec = raw.get(FX_SYM, [])
    cnh = _f(cnh_rec[1]) if len(cnh_rec) > 1 else 0.0  # [1]=current offshore USD/CNH
    rows = []
    for p in PAIRS:
        etf = raw.get(p["etf_sym"], [])
        fut = raw.get(p["fut"], [])
        if len(etf) < 4 or len(fut) < 9:
            continue
        # Sina ETF: [1]open [2]prevClose [3]current ; Sina hf_ future: [0]current [8]prev settle
        rows.append({
            "etf": p["etf"], "etf_name": p["name"],
            "etf_price": _f(etf[3]), "etf_prevclose": _f(etf[2]),
            "fut": p["fut_name"], "fut_price": _f(fut[0]), "fut_prevsettle": _f(fut[8]),
            "cnh": cnh,
        })
    return {"timestamp": now_cn().isoformat(), "trade_date": now_cn().strftime("%Y-%m-%d"), "rows": rows}


def snapshot_premium(row: dict[str, Any], cnh_prevclose: float | None) -> dict[str, Any]:
    """Rough current premium vs previous closes (for live display only).

    premium_drift = etf_change - (futures_change + fx_change). Positive => ETF is
    rich vs what futures+FX justify (a candidate to fade); negative => cheap.
    """
    etf_chg = row["etf_price"] / row["etf_prevclose"] - 1.0 if row["etf_prevclose"] > 0 else 0.0
    fut_chg = row["fut_price"] / row["fut_prevsettle"] - 1.0 if row["fut_prevsettle"] > 0 else 0.0
    fx_chg = (row["cnh"] / cnh_prevclose - 1.0) if cnh_prevclose and cnh_prevclose > 0 else 0.0
    fair_chg = fut_chg + fx_chg
    return {
        "etf": row["etf"], "etf_change_pct": round(etf_chg * 100, 3),
        "fair_change_pct": round(fair_chg * 100, 3), "fut_change_pct": round(fut_chg * 100, 3),
        "fx_change_pct": round(fx_chg * 100, 3),
        "premium_drift_pct": round((etf_chg - fair_chg) * 100, 3),
    }


def load_log() -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    rows = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def analyze_mean_reversion(history: list[dict[str, Any]], etf: str, *, min_points: int = 40, min_days: int = 8,
                           discount_threshold_pct: float = 0.3, roundtrip_cost_pct: float = 0.12) -> dict[str, Any]:
    """LONG-ONLY, COST-AWARE test of the premium edge.

    We cannot short, so the only executable trade is the DISCOUNT side: when the
    ETF is cheap vs its futures+FX fair value (premium_drift < 0), BUY and profit
    as it converges UP. This is a clean long-only single-instrument trade (the
    'fair value' is a synthetic reference, not a leg you must short).

    GPY (Garleanu-Panageas-Yu 2019) caveat: the ABSOLUTE QDII premium is largely a
    STRUCTURAL 'implicit tax' (quota-constrained access to overseas assets) and does
    NOT revert -- do not trade the level. We therefore measure premium_drift as the
    INTRADAY deviation from each day's open (the structural level differences out),
    so we trade only the cyclical/sentiment deviation (LST), not the structural tax.

    Cost note: round-trip default 0.12% reflects the MEASURED cross-border ETF cost
    (513500/513100 spread ~4-5 bps one-way + ~3-5 bps impact at our size), not the
    wider HK-ETF spread. Market impact is small for us (orders ~0.02% of ADV); the
    spread dominates, so passive limit execution would lower this further.

    Reports: (1) full drift<->next-return correlation (diagnostic);
    (2) DISCOUNT-side conditional edge: average forward ETF return after entering
    at a discount beyond discount_threshold_pct; (3) that edge NET of round-trip
    cost. A premium-side (drift>0) signal is logged but flagged NOT executable.
    """
    by_day: dict[str, list[dict[str, Any]]] = {}
    for snap in history:
        for r in snap.get("rows", []):
            if r.get("etf") != etf:
                continue
            by_day.setdefault(snap["trade_date"], []).append({"t": snap["timestamp"], **r})
    drifts: list[float] = []
    next_excess: list[float] = []
    disc_fwd: list[float] = []   # forward ETF return after a DISCOUNT entry (long-only P&L proxy)
    days_used = 0
    dthr = discount_threshold_pct / 100.0
    for day, recs in by_day.items():
        recs = [r for r in recs if r["etf_price"] > 0 and r["fut_price"] > 0]
        if len(recs) < 3:
            continue
        days_used += 1
        e0, f0, c0 = recs[0]["etf_price"], recs[0]["fut_price"], recs[0]["cnh"] or 1.0
        series = []
        for r in recs:
            etf_ret = r["etf_price"] / e0 - 1.0
            fut_ret = r["fut_price"] / f0 - 1.0
            fx_ret = (r["cnh"] / c0 - 1.0) if c0 else 0.0
            series.append({"etf_ret": etf_ret, "drift": etf_ret - fut_ret - fx_ret, "px": r["etf_price"]})
        for i in range(len(series) - 1):
            drifts.append(series[i]["drift"])
            next_excess.append(series[i + 1]["etf_ret"] - series[i]["etf_ret"])
            # discount entry: ETF cheap vs fair -> long-only forward P&L = next-step ETF price return
            if series[i]["drift"] <= -dthr:
                disc_fwd.append(series[i + 1]["px"] / series[i]["px"] - 1.0)
    n = len(drifts)
    result = {"etf": etf, "points": n, "days": days_used, "ready": n >= min_points and days_used >= min_days}
    if n >= 3:
        md, mn = sum(drifts) / n, sum(next_excess) / n
        cov = sum((drifts[i] - md) * (next_excess[i] - mn) for i in range(n)) / n
        sd_d = (sum((x - md) ** 2 for x in drifts) / n) ** 0.5
        sd_n = (sum((x - mn) ** 2 for x in next_excess) / n) ** 0.5
        corr = cov / (sd_d * sd_n) if sd_d > 0 and sd_n > 0 else 0.0
        result["drift_next_return_corr"] = round(corr, 4)
        nd = len(disc_fwd)
        avg_disc = (sum(disc_fwd) / nd) if nd else None
        hit = (sum(1 for x in disc_fwd if x > 0) / nd) if nd else None
        net = (avg_disc * 100 - roundtrip_cost_pct) if avg_disc is not None else None
        result.update({
            "discount_entries": nd,
            "discount_avg_forward_pct": round(avg_disc * 100, 4) if avg_disc is not None else None,
            "discount_hit_rate": round(hit, 3) if hit is not None else None,
            "roundtrip_cost_pct": roundtrip_cost_pct,
            "discount_edge_net_of_cost_pct": round(net, 4) if net is not None else None,
            "executable_long_only": "discount_side_only (cannot short the premium side)",
            "interpretation": ("tradable_discount_edge" if (net is not None and net > 0)
                               else "discount_edge_below_cost" if net is not None else "insufficient_discount_entries"),
            "note": "long-only buys the discount; needs net>0 AND >=8 days to trust",
        })
    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = collect_snapshot()
    if not snap["rows"]:
        print("no data (market data unavailable)")
        return
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")

    cnh_prev = None  # rough display only; analysis uses session-open instead
    print(f"=== cross-border premium snapshot {snap['timestamp']} (SHADOW, no trading) ===")
    for r in snap["rows"]:
        d = snapshot_premium(r, cnh_prev)
        print(f"  {r['etf']} {r['etf_name']}: etf={d['etf_change_pct']}% fut={d['fut_change_pct']}% "
              f"premium_drift={d['premium_drift_pct']}%  (etf_px={r['etf_price']} {r['fut']}={r['fut_price']} cnh={r['cnh']})")

    history = load_log()
    print(f"=== mean-reversion study (history points={len(history)}) ===")
    for p in PAIRS:
        a = analyze_mean_reversion(history, p["etf"])
        if a.get("ready"):
            print(f"  {p['etf']} [DISCOUNT side, long-only]: net_of_cost={a.get('discount_edge_net_of_cost_pct')}% "
                  f"(avg_fwd={a.get('discount_avg_forward_pct')}% - cost={a.get('roundtrip_cost_pct')}%, "
                  f"hit={a.get('discount_hit_rate')}, entries={a.get('discount_entries')}) -> {a.get('interpretation')}")
        else:
            print(f"  {p['etf']}: NOT READY (points={a['points']}, days={a['days']}; need >=40 points / >=8 days)")
    print(f"log: {LOG_PATH}")


if __name__ == "__main__":
    main()
