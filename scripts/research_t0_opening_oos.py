"""OOS research for opening behavior in officially confirmed T+0 ETFs.

Fixed, pre-declared rules are evaluated on 2026-03-23..05-20 (development)
and the untouched 2026-05-21..06-18 holdout.  This script is diagnostic only:
it neither optimizes thresholds nor writes a trading configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
CN = timezone(timedelta(hours=8))
QUOTES = ROOT / "data" / "research" / "t0_opening" / "yahoo_60d_confirmed_t0_raw_5m.jsonl"
OUT_JSON = ROOT / "outputs" / "edge_research" / "opening_oos_20260323_20260618.json"
OUT_MD = ROOT / "outputs" / "edge_research" / "opening_oos_20260323_20260618.md"
TRAIN_END = "2026-05-20"
OOS_START = "2026-05-21"
OOS_END = "2026-06-18"
BASE_COST_BPS = 12.0
STRESS_COST_BPS = 20.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def max_drawdown(returns: list[float]) -> float:
    equity = peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    return drawdown


def rolling_reference(values: list[float], minimum: int = 5, lookback: int = 20) -> float | None:
    history = values[-lookback:]
    return statistics.median(history) if len(history) >= minimum else None


def load_days(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[(str(row["stockCode"]), str(row["trade_date"]))].append(row)
    opening_amount_history: dict[str, list[float]] = defaultdict(list)
    features: list[dict[str, Any]] = []
    for (code, date), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.sort(key=lambda row: row["timestamp"])
        morning = [row for row in rows if row["timestamp"][11:16] <= "11:30"]
        if len(rows) < 40 or not morning:
            continue
        first = morning[0]
        prev_close = first.get("prev_close")
        if not prev_close or float(prev_close) <= 0:
            continue

        def at_or_before(clock: str) -> dict[str, Any] | None:
            chosen = [row for row in morning if row["timestamp"][11:16] <= clock]
            return chosen[-1] if chosen else None

        ten = at_or_before("09:40")
        thirty = at_or_before("10:00")
        if ten is None or thirty is None:
            continue
        first_price, ten_price, thirty_price = float(first["close"]), float(ten["close"]), float(thirty["close"])
        close = float(rows[-1]["close"])
        amount_10m = float(ten.get("cumulative_amount") or 0.0)
        reference = rolling_reference(opening_amount_history[code])
        amount_surge = amount_10m / reference if reference and reference > 0 else None
        opening_amount_history[code].append(amount_10m)
        gap = first_price / float(prev_close) - 1.0
        if gap > 0.01:
            state = "gap_up_high"
        elif -0.005 <= gap <= 0.005:
            state = "gap_flat"
        elif gap < -0.005:
            state = "gap_down"
        else:
            state = "gap_other"
        after_30 = [float(row["close"]) for row in rows if row["timestamp"][11:16] >= "10:00"]
        features.append({
            "date": date,
            "code": code,
            "exchange": first["exchange"],
            "asset_class": first.get("asset_class") or "unknown",
            "state": state,
            "gap": gap,
            "first_price": first_price,
            "ten_price": ten_price,
            "thirty_price": thirty_price,
            "close": close,
            "ret_10m": ten_price / first_price - 1.0,
            "ret_30m": thirty_price / first_price - 1.0,
            "first_to_close": close / first_price - 1.0,
            "ten_to_close": close / ten_price - 1.0,
            "thirty_to_close": close / thirty_price - 1.0,
            "after_30_max_capture": max(after_30) / thirty_price - 1.0,
            "close_range_amplitude": max(float(row["close"]) for row in rows) / min(float(row["close"]) for row in rows) - 1.0,
            "amount_10m": amount_10m,
            "amount_surge_vs_prior20": amount_surge,
            "daily_turnover": float(rows[-1].get("cumulative_amount") or 0.0),
        })
    return features


RULES: dict[str, tuple[str, Callable[[dict[str, Any]], bool], str]] = {
    "all_10m_baseline": ("ten_price", lambda row: True, "all eligible ETF-days, enter after 10 minutes"),
    "gap_up_chase_5m": ("first_price", lambda row: row["gap"] > 0.01, "first 5m close gap > 1%"),
    "gap_up_confirmed_10m": (
        "ten_price",
        lambda row: row["gap"] > 0.01 and row["ret_10m"] > 0
        and row["amount_surge_vs_prior20"] is not None and row["amount_surge_vs_prior20"] >= 1.5,
        "gap > 1%, positive 10m return, 10m amount >= 1.5x prior-20 median",
    ),
    "gap_down_reversal_10m": ("ten_price", lambda row: row["gap"] < -0.005 and row["ret_10m"] > 0, "gap < -0.5%, then positive 10m reversal"),
    "flat_momentum_10m": ("ten_price", lambda row: abs(row["gap"]) <= 0.005 and row["ret_10m"] > 0.002, "gap within +/-0.5%, 10m momentum > 0.2%"),
}


def selected_returns(rows: list[dict[str, Any]], rule_name: str, cost_bps: float) -> list[dict[str, Any]]:
    entry_field, predicate, _ = RULES[rule_name]
    result = []
    for row in rows:
        if not predicate(row):
            continue
        gross = row["close"] / row[entry_field] - 1.0
        result.append({**row, "gross_return": gross, "net_return": gross - cost_bps / 10_000.0})
    return result


def daily_returns(trades: list[dict[str, Any]]) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        grouped[trade["date"]].append(trade["net_return"])
    return [(date, statistics.fmean(values)) for date, values in sorted(grouped.items())]


def bootstrap_mean_ci(values: list[float], reps: int = 2000, seed: int = 43) -> dict[str, Any]:
    if len(values) < 5:
        return {"ci95": None, "probability_mean_positive": None}
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(reps)]
    return {
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "probability_mean_positive": sum(value > 0 for value in means) / reps,
    }


def metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    net = [trade["net_return"] for trade in trades]
    gross = [trade["gross_return"] for trade in trades]
    daily = daily_returns(trades)
    daily_values = [value for _, value in daily]
    positives = sum(value for value in net if value > 0)
    negatives = -sum(value for value in net if value < 0)
    daily_std = statistics.stdev(daily_values) if len(daily_values) > 1 else 0.0
    capacity = statistics.median([trade["daily_turnover"] for trade in trades]) * 0.01 if trades else None
    return {
        "trades": len(trades),
        "trading_days": len(daily),
        "gross_mean_pct": round(statistics.fmean(gross) * 100, 4) if gross else None,
        "net_mean_pct": round(statistics.fmean(net) * 100, 4) if net else None,
        "net_median_pct": round(statistics.median(net) * 100, 4) if net else None,
        "win_rate": round(sum(value > 0 for value in net) / len(net), 4) if net else None,
        "profit_factor": round(positives / negatives, 4) if negatives else None,
        "average_trade_pct": round(statistics.fmean(net) * 100, 4) if net else None,
        "tail_loss_p05_pct": round((percentile(net, 0.05) or 0.0) * 100, 4) if net else None,
        "daily_sharpe": round(statistics.fmean(daily_values) / daily_std * math.sqrt(252), 4) if daily_std > 0 else None,
        "max_drawdown_pct": round(max_drawdown(daily_values) * 100, 4) if daily_values else None,
        "capacity_estimate_cny_at_1pct_median_daily_turnover": round(capacity, 2) if capacity is not None else None,
        "bootstrap_daily_mean": bootstrap_mean_ci(daily_values),
        "cagr": None,
        "sortino": None,
        "turnover": len(trades),
    }


def state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state in ("gap_up_high", "gap_flat", "gap_down", "gap_other"):
        chosen = [row for row in rows if row["state"] == state]
        result[state] = {
            "samples": len(chosen),
            "first_to_close_mean_pct": round(statistics.fmean(row["first_to_close"] for row in chosen) * 100, 4) if chosen else None,
            "after_30_max_capture_mean_pct": round(statistics.fmean(row["after_30_max_capture"] for row in chosen) * 100, 4) if chosen else None,
            "amplitude_mean_pct": round(statistics.fmean(row["close_range_amplitude"] for row in chosen) * 100, 4) if chosen else None,
        }
    return result


def evaluate(rows: list[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in RULES:
        trades = selected_returns(rows, name, cost_bps)
        by_class = {
            asset: metrics([trade for trade in trades if trade["asset_class"] == asset])
            for asset in sorted({trade["asset_class"] for trade in trades})
        }
        result[name] = {"definition": RULES[name][2], "overall": metrics(trades), "by_asset_class": by_class}
    return result


def fixed_rule_promotion(oos: dict[str, Any]) -> dict[str, Any]:
    alternatives = [name for name in RULES if name != "all_10m_baseline"]
    threshold = 0.05 / len(alternatives)
    decisions: dict[str, Any] = {}
    for name in alternatives:
        row = oos[name]["overall"]
        ci = row["bootstrap_daily_mean"]["ci95"]
        probability = row["bootstrap_daily_mean"]["probability_mean_positive"]
        p_one_sided = 1.0 - probability if probability is not None else None
        passed = bool(row["trades"] >= 30 and ci and ci[0] > 0 and p_one_sided is not None and p_one_sided < threshold)
        decisions[name] = {
            "passed": passed,
            "minimum_trades": 30,
            "bonferroni_one_sided_alpha": threshold,
            "one_sided_bootstrap_p": p_one_sided,
            "requires_positive_ci_lower_bound": True,
        }
    return decisions


def build_report(features: list[dict[str, Any]], source_path: Path = QUOTES) -> dict[str, Any]:
    train = [row for row in features if row["date"] <= TRAIN_END]
    oos = [row for row in features if OOS_START <= row["date"] <= OOS_END]
    train_eval = evaluate(train, BASE_COST_BPS)
    oos_eval = evaluate(oos, BASE_COST_BPS)
    stress_eval = evaluate(oos, STRESS_COST_BPS)
    months = sorted({row["date"][:7] for row in features})
    walk_forward = {
        month: evaluate([row for row in features if row["date"].startswith(month)], BASE_COST_BPS)
        for month in months
    }
    promotion = fixed_rule_promotion(oos_eval)
    any_pass = any(row["passed"] for row in promotion.values())
    return {
        "schemaVersion": "t0_opening_oos_research_v1",
        "generatedAt": datetime.now(CN).isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "edgeValidated": any_pass,
        "liveReady": False,
        "formalStrategyAllowed": False,
        "data": {
            "source": str(source_path),
            "dateStart": min(row["date"] for row in features),
            "dateEnd": max(row["date"] for row in features),
            "ETFDaySamples": len(features),
            "instruments": len({row["code"] for row in features}),
            "train": {"start": min(row["date"] for row in train), "end": TRAIN_END, "samples": len(train)},
            "oos": {"start": OOS_START, "end": OOS_END, "samples": len(oos)},
            "pointInTime": True,
            "fullDayLiquiditySelection": False,
        },
        "costModel": {
            "baseRoundtripBps": BASE_COST_BPS,
            "stressRoundtripBps": STRESS_COST_BPS,
            "stampTaxBps": 0.0,
            "note": "all-in round-trip research assumption; historical book depth is unavailable",
        },
        "openingState": {"train": state_summary(train), "oos": state_summary(oos)},
        "trainResults": train_eval,
        "oosResults": oos_eval,
        "oosStressCostResults": stress_eval,
        "monthlyWalkForward": walk_forward,
        "promotionGates": promotion,
        "conclusion": {
            "researchQuestion": "Do fixed opening-gap, momentum or reversal rules create post-cost OOS edge in confirmed T+0 ETFs?",
            "dataRange": f"{min(row['date'] for row in features)}..{max(row['date'] for row in features)}; holdout {OOS_START}..{OOS_END}",
            "sampleSize": len(features),
            "method": "fixed thresholds, point-in-time opening amount reference, base/stress costs, untouched holdout and monthly stability",
            "result": "at least one fixed rule passed" if any_pass else "no fixed rule passed the OOS multiple-testing gate",
            "edgeExists": any_pass,
            "edgeAfterCosts": any_pass,
            "applicableETFClasses": sorted({row["asset_class"] for row in features}),
            "notApplicable": ["SZ ETFs pending product-level T+0 verification", "auction execution", "strategies requiring historical IOPV or order-book depth"],
            "recommendStrategyEntry": False,
            "recommendedParameters": None,
            "risks": ["60 trading days", "survivorship bias", "5-minute close is not an executable auction price", "Yahoo volume quality", "no historical spread/depth"],
            "nextResearch": "extend the point-in-time dataset, collect real spread/IOPV and repeat a purged walk-forward test before any shadow gate",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Confirmed T+0 ETF Opening Research — OOS",
        "",
        f"Data: {report['data']['dateStart']}..{report['data']['dateEnd']} | ETF-days: {report['data']['ETFDaySamples']} | instruments: {report['data']['instruments']}",
        f"Holdout: {report['data']['oos']['start']}..{report['data']['oos']['end']} | base cost: {BASE_COST_BPS:.0f} bps round-trip",
        "",
        "Diagnostic only. No trading configuration was changed.",
        "",
        "## OOS fixed-rule results",
        "",
        "| Rule | Trades | Net mean | Win rate | Daily Sharpe | Max drawdown | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in report["oosResults"].items():
        row = item["overall"]
        gate = report["promotionGates"].get(name, {}).get("passed")
        lines.append(
            f"| {name} | {row['trades']} | {row['net_mean_pct']}% | {row['win_rate']} | "
            f"{row['daily_sharpe']} | {row['max_drawdown_pct']}% | {'PASS' if gate else 'FAIL' if gate is not None else 'benchmark'} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        f"- {report['conclusion']['result']}.",
        "- The existing Yahoo replay was not used because its same-day full-turnover universe gate leaks end-of-day information into opening research.",
        "- Results remain non-promotable even if a point estimate is positive; the dataset lacks historical spread, depth and IOPV.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", default=str(QUOTES))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()
    features = load_days(Path(args.quotes))
    if not features:
        raise RuntimeError("no eligible ETF-day features")
    report = build_report(features, Path(args.quotes))
    json_path, md_path = Path(args.out_json), Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
