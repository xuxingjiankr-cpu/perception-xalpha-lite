"""Signal IC / SNR audit -- which intraday signals carry STABLE cross-sectional
forward information, and which are noise to drop.

For each candidate price/volume signal we compute, point-in-time (no lookahead), the
cross-sectional Information Coefficient: at every (date, minute) panel, the Spearman rank
correlation between the signal across ETFs and their forward return at horizon h. We report
mean IC, IC stability IR=mean/std, a t-stat, and the purged out-of-sample IC -- so a signal
only counts if its IC is non-trivial AND holds in the test window with the same sign.

Honest scope: the free Yahoo 5m data is price/volume only -- book-based signals (bid
pressure, true spread) are synthetic in replay and are NOT audited here. Cross-sectional IC
measures SELECTION power (which ETF), the question the universe ranker actually faces.

Diagnostic only. Run: py -3.13 scripts/research_signal_ic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from run_etf_paper_trading_agent import ROOT

QUOTES = ROOT / "outputs" / "t0_replay" / "yahoo_60d_quotes.jsonl"
OUT = ROOT / "outputs" / "signal_ic"
TRAIN_END = "2026-05-20"
MIN_CODES_PER_PANEL = 8          # need a cross-section to rank
ORB_BARS = 6                     # ~first 30 min defines the opening range
FWD_HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "to_close": None}


def load_bars() -> dict[tuple[str, str], list[tuple[int, float, float]]]:
    """(date, code) -> time-ordered [(minute, price, volume)] in Shanghai time."""
    series: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    with QUOTES.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                q = json.loads(line)
            except Exception:
                continue
            ts = str(q.get("timestamp", ""))
            code = str(q.get("stockCode", "")).zfill(6)
            price = float(q.get("currentPrice") or 0.0)
            if len(ts) < 16 or price <= 0 or not code:
                continue
            date = ts[:10]
            try:
                minute = int(ts[11:13]) * 60 + int(ts[14:16])
            except Exception:
                continue
            series.setdefault((date, code), []).append((minute, price, float(q.get("volume") or 0.0)))
    for key in series:
        series[key].sort()
    return series


def _kf_local_linear_trend(prices: list[float]) -> tuple[float | None, float | None]:
    """Causal local-linear-trend Kalman filter in log-price (Benhamou 2018 style):
    state [level, velocity], F=[[1,1],[0,1]], H=[1,0]. Returns the filtered (velocity,
    residual) at the last point -- velocity = de-lagged per-bar log-return (denoised
    momentum), residual = log price minus filtered level (denoised deviation/reversal).
    Fixed, UNTUNED noise (R ~ (0.2%/bar)^2, Q_vel/R ~ 1/16 => ~4-bar smoothing) to avoid
    fitting the filter to this 60d sample."""
    import math
    if len(prices) < 3:
        return None, None
    R, Qv = 4e-6, 2.5e-7
    lvl, vel = math.log(max(prices[0], 1e-9)), 0.0
    p00, p01, p10, p11 = 1e-2, 0.0, 0.0, 1e-4
    for p in prices[1:]:
        z = math.log(max(p, 1e-9))
        lvl = lvl + vel                                  # predict level
        a00, a01 = p00 + p10, p01 + p11                  # F P
        np00, np01 = a00 + a01, a01                      # (F P) F^T
        np10, np11 = p10 + p11, p11
        p00, p01, p10, p11 = np00, np01, np10, np11 + Qv  # + Q
        S = p00 + R
        k0, k1 = p00 / S, p10 / S
        y = z - lvl
        lvl, vel = lvl + k0 * y, vel + k1 * y            # update
        np00, np01 = (1 - k0) * p00, (1 - k0) * p01
        np10, np11 = p10 - k1 * p00, p11 - k1 * p01
        p00, p01, p10, p11 = np00, np01, np10, np11
    return vel, math.log(max(prices[-1], 1e-9)) - lvl


def signals_at(bars: list[tuple[int, float, float]], i: int) -> dict[str, float]:
    """Point-in-time signals from bars[:i+1] only (no lookahead)."""
    m, p, _ = bars[i]
    prices = [b[1] for b in bars[:i + 1]]
    vols = [b[2] for b in bars[:i + 1]]
    out: dict[str, float] = {}

    def ret(k: int) -> float | None:
        return prices[-1] / prices[-1 - k] - 1.0 if len(prices) > k and prices[-1 - k] > 0 else None

    out["mom_5m"] = ret(1)
    out["mom_15m"] = ret(3)
    out["mom_30m"] = ret(6)
    r1, r2 = ret(1), (prices[-2] / prices[-3] - 1.0 if len(prices) > 2 and prices[-3] > 0 else None)
    out["acceleration"] = (r1 - r2) if (r1 is not None and r2 is not None) else None
    out["reversal_5m"] = (-r1) if r1 is not None else None
    out["chg_from_open"] = prices[-1] / prices[0] - 1.0 if prices[0] > 0 else None
    # cumulative VWAP distance (uses volume when present, else simple mean)
    if sum(vols) > 0:
        vwap = sum(pp * vv for pp, vv in zip(prices, vols)) / sum(vols)
    else:
        vwap = sum(prices) / len(prices)
    out["vwap_dist"] = p / vwap - 1.0 if vwap > 0 else None
    # ORB breakout (only meaningful after the opening range is formed)
    if len(prices) > ORB_BARS:
        orb_hi = max(prices[:ORB_BARS])
        out["orb_breakout"] = p / orb_hi - 1.0 if orb_hi > 0 else None
    else:
        out["orb_breakout"] = None
    # position in the day's range so far
    lo, hi = min(prices), max(prices)
    out["range_pos"] = (p - lo) / (hi - lo) if hi > lo else None
    # volume surge vs running mean
    avgv = sum(vols) / len(vols)
    out["vol_surge"] = vols[-1] / avgv if avgv > 0 else None
    # Kalman local-linear-trend: de-lagged velocity (denoised momentum) and residual
    kf_vel, kf_resid = _kf_local_linear_trend(prices)
    out["kf_velocity"] = kf_vel
    out["kf_resid"] = kf_resid
    return out


def fwd_return(bars: list[tuple[int, float, float]], i: int, h: int | None) -> float | None:
    p = bars[i][1]
    if p <= 0:
        return None
    j = (len(bars) - 1) if h is None else (i + h)
    if j <= i or j >= len(bars):
        return None
    pj = bars[j][1]
    return pj / p - 1.0 if pj > 0 else None


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < MIN_CODES_PER_PANEL:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def audit() -> dict:
    series = load_bars()
    dates = sorted({d for d, _ in series})
    codes_by_date: dict[str, list[str]] = {}
    for (d, c) in series:
        codes_by_date.setdefault(d, []).append(c)

    sig_names = ["mom_5m", "mom_15m", "mom_30m", "acceleration", "reversal_5m",
                 "chg_from_open", "vwap_dist", "orb_breakout", "range_pos", "vol_surge",
                 "kf_velocity", "kf_resid"]
    # per (signal, horizon) -> list of (date, panel_IC)
    panels: dict[tuple[str, str], list[tuple[str, float]]] = {}

    for d in dates:
        codes = codes_by_date[d]
        # all minutes present this day (union across codes), evaluate panels at each
        minutes = sorted({b[0] for c in codes for b in series[(d, c)]})
        # index of each minute within each code's bar list
        for minute in minutes:
            # collect per-code signals + forward returns at this minute
            sig_vals: dict[str, list[float]] = {s: [] for s in sig_names}
            fwd_vals: dict[str, list[float]] = {h: [] for h in FWD_HORIZONS}
            present = 0
            cross: list[tuple[dict, dict]] = []
            for c in codes:
                bars = series[(d, c)]
                idx = next((k for k, b in enumerate(bars) if b[0] == minute), None)
                if idx is None or idx < 2:
                    continue
                sg = signals_at(bars, idx)
                fw = {h: fwd_return(bars, idx, hb) for h, hb in FWD_HORIZONS.items()}
                cross.append((sg, fw))
                present += 1
            if present < MIN_CODES_PER_PANEL:
                continue
            for s in sig_names:
                for h in FWD_HORIZONS:
                    xs, ys = [], []
                    for sg, fw in cross:
                        if sg.get(s) is not None and fw.get(h) is not None:
                            xs.append(sg[s]); ys.append(fw[h])
                    ic = spearman(np.array(xs), np.array(ys)) if len(xs) >= MIN_CODES_PER_PANEL else None
                    if ic is not None:
                        panels.setdefault((s, h), []).append((d, ic))

    def stats(rows: list[tuple[str, float]]) -> dict:
        ics = np.array([v for _, v in rows], dtype=float)
        if ics.size == 0:
            return {"n": 0}
        ir = float(ics.mean() / ics.std()) if ics.std() > 0 else None
        return {
            "n_panels": int(ics.size),
            "mean_ic": round(float(ics.mean()), 4),
            "ir": round(ir, 3) if ir is not None else None,
            "t_stat": round(ir * np.sqrt(ics.size), 2) if ir is not None else None,
            "hit_rate": round(float(np.mean(ics > 0)), 3),
        }

    result: dict = {"signals": {}, "train_end": TRAIN_END, "horizons": list(FWD_HORIZONS)}
    for (s, h), rows in panels.items():
        train = [r for r in rows if r[0] <= TRAIN_END]
        test = [r for r in rows if r[0] > TRAIN_END]
        result["signals"].setdefault(s, {})[h] = {
            "full": stats(rows), "train": stats(train), "test": stats(test),
        }
    return result


def render(result: dict) -> str:
    lines = [
        "# Signal IC / SNR Audit -- 60d cross-sectional (point-in-time)",
        "",
        f"Purged split: train<= {result['train_end']} < test. Cross-sectional Spearman IC per "
        f"(date,minute) panel; price/volume signals only (book signals synthetic in replay).",
        "",
        "Keep a signal only if |mean IC| is non-trivial, |t|>~2, AND test IC keeps the same sign "
        "with comparable magnitude. IR=mean/std of panel ICs (stability).",
        "",
    ]
    for s in sorted(result["signals"]):
        lines += [f"## {s}", "",
                  "| horizon | mean IC | IR | t | hit | TEST IC | TEST t | sign holds |",
                  "|---|--:|--:|--:|--:|--:|--:|:--:|"]
        for h in result["horizons"]:
            cell = result["signals"][s].get(h)
            if not cell or cell["full"].get("n_panels", 0) == 0:
                continue
            f, t = cell["full"], cell["test"]
            holds = (f.get("mean_ic") is not None and t.get("mean_ic") is not None
                     and np.sign(f["mean_ic"]) == np.sign(t["mean_ic"]) and abs(t["mean_ic"]) >= 0.01)
            lines.append(
                f"| {h} | {f.get('mean_ic')} | {f.get('ir')} | {f.get('t_stat')} | {f.get('hit_rate')} | "
                f"{t.get('mean_ic')} | {t.get('t_stat')} | {'yes' if holds else 'no'} |")
        lines.append("")
    lines += [
        "## Read",
        "",
        "- |mean IC| < ~0.02 with |t|<2 OR test-sign flip => **noise, drop the signal**.",
        "- A signal that survives (stable IC, sign holds OOS) is a candidate to KEEP/up-weight; "
        "everything else just adds variance to the composite score.",
        "- Cross-sectional IC = selection power (which ETF). A real positive IC here is the basis "
        "for a day-neutral relative-strength selector (research direction #2).",
        "",
        "_Diagnostic only; changes no signal, weight, or gate._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "signal_ic_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(result)
    (OUT / "signal_ic_audit.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"outputs: {OUT / 'signal_ic_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
