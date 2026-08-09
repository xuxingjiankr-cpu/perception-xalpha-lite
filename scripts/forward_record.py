"""Frozen forward record — the only clean evidence still available to this project.

Both research lines have now looked at every historical window, so no amount of further
backtesting can adjudicate anything. What can: writing down predictions before the outcomes
exist, and scoring them later without touching the rules.

Three modes, deliberately separated so the rules cannot move after a result is seen:

  freeze   Write an immutable spec — factor set, holding period, book size, cost, universe
           rules — and hash it. Refuses to overwrite an existing spec; a changed rule must
           become a new record with a new hash, which makes silent revision impossible.
  log      Append today's picks for that spec. Append-only, one entry per (spec, date), and
           it records the data-as-of date so a stale panel is visible rather than hidden.
  score    Evaluate only entries whose holding period has fully matured, against data that
           did not exist when they were written.

Nothing here places or influences an order. It records what the frozen rules would have
selected and, later, what happened.

Run: py -3.13 scripts/forward_record.py {freeze|log|score} [--spec NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_etf_paper_trading_agent import ROOT

OUT_DIR = ROOT / "outputs" / "forward_record"
BARS = ROOT / "data" / "market" / "ashare_research" / "baostock_pit_adjusted" / "bars_1d_backward_adjusted"

# The frozen candidates: every factor from the published-library search whose ten-name book
# stayed net-positive out of sample. Recorded here so the set cannot quietly grow later.
DEFAULT_SPEC = {
    "specName": "published_library_net_positive_v1",
    "status": "research_only_forward_record_not_trading",
    "frozenAt": None,
    "rationale": (
        "Four factors from a 456-candidate published-library search whose ten-name book "
        "remained net-positive after 30bps costs on the untouched test window. A deflated "
        "Sharpe test at that trial count returns consistent_with_luck, so this record exists "
        "to settle the question with data that did not exist when the rules were written."
    ),
    "factors": [
        "qlib158/rsqr60",
        "qlib158/rsqr30",
        "academic/illiq",
        "academic/cma",
    ],
    "combination": "equal_weight_rank_average",
    "bookSize": 10,
    "holdingTradingDays": 10,
    "roundTripCost": 0.003,
    "universe": {
        "trailingAmountWindow": 60,
        "minimumTrailingMedianAmountCny": 30_000_000,
        "minimumPriorObservations": 120,
        "excludeSt": True,
        "requireNormalTradeStatus": True,
        "dropSealedLimitLegs": True,
    },
    "scoringRule": (
        "Book return over the holding period, minus the eligible-universe mean over the same "
        "period. Cost is charged on realised turnover at the frozen rate."
    ),
}


def digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def spec_path(name: str) -> Path:
    return OUT_DIR / f"{name}.spec.json"


def log_path(name: str) -> Path:
    return OUT_DIR / f"{name}.predictions.jsonl"


def load_panel() -> dict[str, pd.DataFrame]:
    fields: dict[str, dict[str, pd.Series]] = {
        k: {} for k in ("open", "high", "low", "close", "volume", "amount", "is_st", "status")
    }
    for path in sorted(BARS.glob("*.jsonl")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if len(rows) < 260:
            continue
        frame = pd.DataFrame(rows)
        if not {"dt", "open", "high", "low", "close", "vol", "amount"}.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["dt"].astype(str).str[:10], errors="coerce")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        code = path.stem
        for key, column in (("open", "open"), ("high", "high"), ("low", "low"),
                             ("close", "close"), ("volume", "vol"), ("amount", "amount")):
            fields[key][code] = pd.to_numeric(frame[column], errors="coerce")
        fields["is_st"][code] = pd.to_numeric(frame.get("isST", 0), errors="coerce")
        fields["status"][code] = pd.to_numeric(frame.get("tradeStatus", 1), errors="coerce")
    panel = {k: pd.DataFrame(v).sort_index() for k, v in fields.items()}
    close = panel["close"]
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    panel["returns"] = close.pct_change(fill_method=None)
    return panel


def eligibility(panel: dict[str, pd.DataFrame], rules: dict) -> pd.DataFrame:
    close, amount = panel["close"], panel["amount"]
    observed = close.notna() & close.gt(0)
    trailing = amount.rolling(int(rules["trailingAmountWindow"]), min_periods=20).median().shift(1)
    seasoned = observed.cumsum().shift(1).ge(int(rules["minimumPriorObservations"]))
    traded = panel["volume"].fillna(0).gt(0) & amount.fillna(0).gt(0)
    ok = trailing.ge(float(rules["minimumTrailingMedianAmountCny"])) & seasoned & observed & traded
    if rules.get("excludeSt", True):
        ok &= panel["is_st"].fillna(0).eq(0)
    if rules.get("requireNormalTradeStatus", True):
        ok &= panel["status"].fillna(1).eq(1)
    return ok.fillna(False)


def composite(panel: dict[str, pd.DataFrame], spec: dict, eligible: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))
    import importlib

    ranks = []
    for key in spec["factors"]:
        zoo, name = key.split("/")
        signal = importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(panel)
        ranks.append(signal.where(eligible).replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True))
    return sum(ranks) / len(ranks)


def cmd_freeze(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = spec_path(DEFAULT_SPEC["specName"])
    if path.exists():
        print(f"refusing to overwrite an existing spec: {path}\n"
              "A changed rule must become a new spec with a new hash — that is the point.")
        return 2
    spec = dict(DEFAULT_SPEC)
    spec["frozenAt"] = datetime.now(timezone.utc).isoformat()
    spec["specSha256"] = digest({k: v for k, v in spec.items() if k != "frozenAt"})
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"frozen: {path}\nsha256: {spec['specSha256']}\nfactors: {spec['factors']}")
    return 0


def cmd_log(args) -> int:
    path = spec_path(args.spec)
    if not path.exists():
        print(f"no spec at {path}; run freeze first")
        return 2
    spec = json.loads(path.read_text(encoding="utf-8"))
    panel = load_panel()
    eligible = eligibility(panel, spec["universe"])
    score = composite(panel, spec, eligible)
    as_of = score.index.max()
    row = score.loc[as_of].dropna()
    if row.empty:
        print("no eligible names on the latest bar; nothing logged")
        return 1
    picks = row.sort_values(ascending=False).head(int(spec["bookSize"]))
    entry = {
        "specName": spec["specName"],
        "specSha256": spec["specSha256"],
        "loggedAt": datetime.now(timezone.utc).isoformat(),
        "dataAsOf": str(as_of.date()),
        "eligibleNames": int(eligible.loc[as_of].sum()),
        "picks": [
            {"symbol": symbol, "compositeRank": round(float(value), 6),
             "close": round(float(panel["close"].loc[as_of, symbol]), 4)}
            for symbol, value in picks.items()
        ],
        "status": "research_only_forward_record_not_trading",
        "orders": [],
    }
    existing = log_path(args.spec)
    if existing.exists():
        for line in existing.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("dataAsOf") == entry["dataAsOf"]:
                    print(f"{entry['dataAsOf']} already recorded; append-only log unchanged")
                    return 0
            except Exception:
                continue
    with existing.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"logged {entry['dataAsOf']}: {[p['symbol'] for p in entry['picks']]}")
    return 0


def cmd_score(args) -> int:
    path, logs = spec_path(args.spec), log_path(args.spec)
    if not logs.exists():
        print("no predictions logged yet")
        return 1
    spec = json.loads(path.read_text(encoding="utf-8"))
    horizon = int(spec["holdingTradingDays"])
    panel = load_panel()
    close, open_ = panel["close"], panel["open"]
    eligible = eligibility(panel, spec["universe"])
    sessions = list(close.index)

    matured, pending = [], 0
    for line in logs.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except Exception:
            continue
        as_of = pd.Timestamp(entry["dataAsOf"])
        if as_of not in close.index:
            continue
        i = sessions.index(as_of)
        if i + horizon + 1 >= len(sessions):
            pending += 1
            continue
        entry_bar, exit_bar = sessions[i + 1], sessions[i + horizon + 1]
        rets = [
            float(open_.loc[exit_bar, p["symbol"]] / open_.loc[entry_bar, p["symbol"]] - 1.0)
            for p in entry["picks"]
            if p["symbol"] in open_.columns
            and np.isfinite(open_.loc[entry_bar, p["symbol"]])
            and np.isfinite(open_.loc[exit_bar, p["symbol"]])
        ]
        if not rets:
            continue
        universe = (open_.loc[exit_bar] / open_.loc[entry_bar] - 1.0).where(eligible.loc[as_of]).dropna()
        matured.append({
            "dataAsOf": entry["dataAsOf"],
            "book": float(np.mean(rets)),
            "universe": float(universe.mean()) if len(universe) else float("nan"),
            "hitRate": float(np.mean([r > 0 for r in rets])),
            "bigMoveRate": float(np.mean([r > 0.10 for r in rets])),
        })

    if not matured:
        print(f"nothing matured yet ({pending} entries still inside their holding window)")
        return 0
    frame = pd.DataFrame(matured)
    excess = frame["book"] - frame["universe"]
    net = excess - float(spec["roundTripCost"])   # one full round trip per matured book
    report = {
        "specName": spec["specName"], "specSha256": spec["specSha256"],
        "status": "research_only_forward_record_not_trading",
        "maturedEntries": len(frame), "pendingEntries": pending,
        "firstAsOf": frame["dataAsOf"].min(), "lastAsOf": frame["dataAsOf"].max(),
        "bookMeanPct": round(float(frame["book"].mean()) * 100, 3),
        "universeMeanPct": round(float(frame["universe"].mean()) * 100, 3),
        "excessMeanPct": round(float(excess.mean()) * 100, 3),
        "netOfCostMeanPct": round(float(net.mean()) * 100, 3),
        "hitRate": round(float(frame["hitRate"].mean()), 3),
        "bigMoveRate": round(float(frame["bigMoveRate"].mean()), 3),
        "verdict": "insufficient_forward_sample" if len(frame) < 60 else "sample_sufficient_for_first_read",
        "orders": [],
    }
    (OUT_DIR / f"{args.spec}.scorecard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, value in report.items():
        if key not in ("orders",):
            print(f"{key}: {value}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("freeze")
    for name in ("log", "score"):
        p = sub.add_parser(name)
        p.add_argument("--spec", default=DEFAULT_SPEC["specName"])
    args = parser.parse_args()
    return {"freeze": cmd_freeze, "log": cmd_log, "score": cmd_score}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
