"""Forward quoted-spread and passive-fill research for confirmed T+0 ETFs.

The script reads already-recorded live quote snapshots only. It never calls a
broker or quote API. Passive fills are reported under two explicit proxies:
trade-touch (optimistic) and ask-cross (conservative). Neither proxy knows queue
position, so both remain diagnostic and cannot justify live execution changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CN = ZoneInfo("Asia/Shanghai")
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "forward_execution_friction_preregistered.json"
)
DEFAULT_OUT = ROOT / "outputs" / "edge_research" / "forward_execution_friction"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_china_time(row: dict[str, Any]) -> datetime | None:
    source = row.get("source_quote_time")
    if source:
        text = str(source).replace(" ", "T")
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=CN)
            return parsed.astimezone(CN)
        except ValueError:
            pass
    raw = row.get("timestamp")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=CN) if parsed.tzinfo is None else parsed.astimezone(CN)


def load_confirmed_master(path: Path, allowed_classes: set[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("t0_confirmed")
                and not row.get("is_money_like")
                and str(row.get("asset_class")) in allowed_classes
            ):
                output[str(row["code"]).zfill(6)] = row
    if not output:
        raise RuntimeError("empty confirmed T0 master")
    return output


def normalized_quote(row: dict[str, Any], master: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    code = str(row.get("stockCode", "")).zfill(6)
    timestamp = parse_china_time(row)
    if code not in master or timestamp is None:
        return None
    try:
        bid = float(row.get("bidPrice1") or 0.0)
        ask = float(row.get("askPrice1") or 0.0)
        current = float(row.get("currentPrice") or 0.0)
        previous = float(row.get("prevClose") or 0.0)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask < bid or current <= 0 or previous <= 0:
        return None
    midpoint = (bid + ask) / 2.0
    return {
        "timestamp": timestamp,
        "trade_date": timestamp.date().isoformat(),
        "code": code,
        "bid": bid,
        "ask": ask,
        "mid": midpoint,
        "current": current,
        "prev_close": previous,
        "spread_bps": (ask - bid) / midpoint * 10_000.0,
        "asset_class": master[code].get("asset_class"),
        "name": row.get("name") or master[code].get("name") or code,
    }


def load_forward_quotes(config: dict[str, Any]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    data_cfg = config["data"]
    master_path = ROOT / str(data_cfg["master"])
    master = load_confirmed_master(master_path, set(data_cfg["assetClasses"]))
    directory = ROOT / str(data_cfg["quoteDirectory"])
    paths = sorted(directory.glob(str(data_cfg["filePattern"])))
    by_code_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_read = valid_rows = 0
    used_paths: list[str] = []
    for path in paths:
        if path.stem[-10:] < str(data_cfg["startDate"]):
            continue
        used_paths.append(str(path))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows_read += 1
                row = normalized_quote(json.loads(line), master)
                if row is None or row["trade_date"] < str(data_cfg["startDate"]):
                    continue
                by_code_day[(row["trade_date"], row["code"])].append(row)
                valid_rows += 1
    for key in by_code_day:
        unique: dict[datetime, dict[str, Any]] = {}
        for row in by_code_day[key]:
            unique[row["timestamp"]] = row
        by_code_day[key] = sorted(unique.values(), key=lambda item: item["timestamp"])
    return by_code_day, {
        "master": str(master_path),
        "files": used_paths,
        "rowsRead": rows_read,
        "validT0BookRows": valid_rows,
        "codeDays": len(by_code_day),
    }


def first_at_or_after(
    rows: list[dict[str, Any]], target: datetime, maximum_delay_minutes: float | None = None
) -> dict[str, Any] | None:
    for row in rows:
        if row["timestamp"] >= target:
            if (
                maximum_delay_minutes is not None
                and (row["timestamp"] - target).total_seconds() > maximum_delay_minutes * 60
            ):
                return None
            return row
    return None


def passive_fill_proxies(
    rows: list[dict[str, Any]], index: int, window_minutes: int
) -> dict[str, Any]:
    initial = rows[index]
    deadline = initial["timestamp"] + timedelta(minutes=window_minutes)
    future = [
        row for row in rows[index + 1 :]
        if initial["timestamp"] < row["timestamp"] <= deadline
    ]
    touch_rows = [row for row in future if row["current"] <= initial["bid"]]
    cross_rows = [row for row in future if row["ask"] <= initial["bid"]]
    return {
        "touch_fill": bool(touch_rows),
        "touch_fill_time": touch_rows[0]["timestamp"] if touch_rows else None,
        "conservative_fill": bool(cross_rows),
        "conservative_fill_time": cross_rows[0]["timestamp"] if cross_rows else None,
    }


def execution_observation(
    rows: list[dict[str, Any]], index: int, config: dict[str, Any]
) -> dict[str, Any] | None:
    execution = config["execution"]
    initial = rows[index]
    future_target = initial["timestamp"] + timedelta(
        minutes=int(execution["genericExitHorizonMinutes"])
    )
    exit_quote = first_at_or_after(rows[index + 1 :], future_target, maximum_delay_minutes=10)
    if exit_quote is None:
        return None
    fills = passive_fill_proxies(
        rows, index, int(execution["passiveFillWindowMinutes"])
    )
    mid_return = exit_quote["mid"] / initial["mid"] - 1.0
    aggressive_return = exit_quote["bid"] / initial["ask"] - 1.0
    aggressive_cost_bps = (mid_return - aggressive_return) * 10_000.0
    result = {
        "trade_date": initial["trade_date"],
        "timestamp": initial["timestamp"].isoformat(),
        "code": initial["code"],
        "spread_bps": initial["spread_bps"],
        "exit_spread_bps": exit_quote["spread_bps"],
        "mid_return_30m": mid_return,
        "aggressive_return_30m": aggressive_return,
        "aggressive_cost_bps": aggressive_cost_bps,
        "touch_fill": fills["touch_fill"],
        "conservative_fill": fills["conservative_fill"],
    }
    for label, filled, fill_time in (
        ("touch", fills["touch_fill"], fills["touch_fill_time"]),
        ("conservative", fills["conservative_fill"], fills["conservative_fill_time"]),
    ):
        result[f"{label}_passive_return_30m"] = (
            exit_quote["bid"] / initial["bid"] - 1.0 if filled else None
        )
        if filled and fill_time is not None:
            post_target = fill_time + timedelta(minutes=10)
            post_quote = first_at_or_after(rows, post_target, maximum_delay_minutes=10)
            result[f"{label}_post_fill_mid_return_10m"] = (
                post_quote["mid"] / initial["bid"] - 1.0 if post_quote else None
            )
        else:
            result[f"{label}_post_fill_mid_return_10m"] = None
    return result


def sample_execution_observations(
    by_code_day: dict[tuple[str, str], list[dict[str, Any]]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    bucket_minutes = int(config["sampling"]["bucketMinutes"])
    output: list[dict[str, Any]] = []
    for rows in by_code_day.values():
        used_buckets: set[int] = set()
        for index, row in enumerate(rows):
            minute = row["timestamp"].hour * 60 + row["timestamp"].minute
            bucket = minute // bucket_minutes
            if bucket in used_buckets:
                continue
            used_buckets.add(bucket)
            observation = execution_observation(rows, index, config)
            if observation is not None:
                output.append(observation)
    return output


def describe(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": round(float(np.mean(finite)), 6),
        "median": round(float(np.median(finite)), 6),
        "p25": round(float(np.quantile(finite, 0.25)), 6),
        "p75": round(float(np.quantile(finite, 0.75)), 6),
        "p90": round(float(np.quantile(finite, 0.90)), 6),
    }


def summarize_execution(observations: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    count = len(observations)
    touch = [row for row in observations if row["touch_fill"]]
    conservative = [row for row in observations if row["conservative_fill"]]
    budget = float(config["execution"]["breakEvenExecutionBudgetBps"])
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_day[row["trade_date"]].append(row)
    return {
        "days": len(by_day),
        "observations": count,
        "spreadBps": describe([row["spread_bps"] for row in observations]),
        "aggressiveRoundTripCostBps30m": describe(
            [row["aggressive_cost_bps"] for row in observations]
        ),
        "aggressiveWithinBreakEvenBudgetRate": round(
            sum(row["aggressive_cost_bps"] <= budget for row in observations) / count, 6
        )
        if count
        else None,
        "touchFillRate10m": round(len(touch) / count, 6) if count else None,
        "conservativeFillRate10m": round(len(conservative) / count, 6) if count else None,
        "touchPostFillMidReturn10mBps": describe(
            [
                row["touch_post_fill_mid_return_10m"] * 10_000.0
                for row in touch
                if row["touch_post_fill_mid_return_10m"] is not None
            ]
        ),
        "conservativePostFillMidReturn10mBps": describe(
            [
                row["conservative_post_fill_mid_return_10m"] * 10_000.0
                for row in conservative
                if row["conservative_post_fill_mid_return_10m"] is not None
            ]
        ),
        "byDay": {
            day: {
                "observations": len(rows),
                "medianSpreadBps": round(float(np.median([row["spread_bps"] for row in rows])), 4),
                "medianAggressiveCostBps": round(
                    float(np.median([row["aggressive_cost_bps"] for row in rows])), 4
                ),
                "touchFillRate": round(sum(row["touch_fill"] for row in rows) / len(rows), 4),
                "conservativeFillRate": round(
                    sum(row["conservative_fill"] for row in rows) / len(rows), 4
                ),
            }
            for day, rows in sorted(by_day.items())
        },
    }


def forward_strategy_shadow(
    by_code_day: dict[tuple[str, str], list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    strategy = config["strategyShadow"]
    minimum_codes = int(strategy["requiresMinimumCodes"])
    threshold = float(strategy["trainingDispersionThreshold"])
    by_day: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for (day, code), rows in by_code_day.items():
        by_day[day][code] = rows
    days: list[dict[str, Any]] = []
    for day, code_rows in sorted(by_day.items()):
        signal_rows: dict[str, dict[str, Any]] = {}
        cutoff = datetime.fromisoformat(f"{day}T{strategy['signalCutoffTime']}:00").replace(tzinfo=CN)
        for code, rows in code_rows.items():
            first = rows[0] if rows else None
            if first and first["timestamp"] <= cutoff:
                signal_rows[code] = first
        if len(signal_rows) < minimum_codes:
            days.append(
                {
                    "trade_date": day,
                    "eligible": False,
                    "reason": "insufficient_cross_section",
                    "codesAtSignal": len(signal_rows),
                }
            )
            continue
        overnight = {
            code: row["current"] / row["prev_close"] - 1.0
            for code, row in signal_rows.items()
        }
        median = float(np.median(list(overnight.values())))
        residual = {code: value - median for code, value in overnight.items()}
        dispersion = float(np.std(list(residual.values()), ddof=1))
        if dispersion <= threshold:
            days.append(
                {
                    "trade_date": day,
                    "eligible": False,
                    "reason": "dispersion_below_training_threshold",
                    "codesAtSignal": len(signal_rows),
                    "dispersion": dispersion,
                }
            )
            continue
        selected = sorted(residual, key=residual.get)[: int(strategy["maxAssets"])]
        trades: list[dict[str, Any]] = []
        for code in selected:
            rows = code_rows[code]
            signal = signal_rows[code]
            later = [row for row in rows if row["timestamp"] > signal["timestamp"]]
            if not later:
                continue
            entry = later[0]
            exit_quote = rows[-1]
            if exit_quote["timestamp"] <= entry["timestamp"]:
                continue
            fills = passive_fill_proxies(rows, rows.index(entry), int(config["execution"]["passiveFillWindowMinutes"]))
            trades.append(
                {
                    "code": code,
                    "signalTime": signal["timestamp"].isoformat(),
                    "entryTime": entry["timestamp"].isoformat(),
                    "exitTime": exit_quote["timestamp"].isoformat(),
                    "midReturn": exit_quote["mid"] / entry["mid"] - 1.0,
                    "aggressiveReturn": exit_quote["bid"] / entry["ask"] - 1.0,
                    "passiveTouchFill": fills["touch_fill"],
                    "passiveTouchReturn": exit_quote["bid"] / entry["bid"] - 1.0
                    if fills["touch_fill"]
                    else None,
                    "passiveConservativeFill": fills["conservative_fill"],
                    "passiveConservativeReturn": exit_quote["bid"] / entry["bid"] - 1.0
                    if fills["conservative_fill"]
                    else None,
                }
            )
        days.append(
            {
                "trade_date": day,
                "eligible": bool(trades),
                "reason": "evaluated" if trades else "no_complete_trades",
                "codesAtSignal": len(signal_rows),
                "dispersion": dispersion,
                "selected": selected,
                "trades": trades,
                "midPortfolioReturn": float(np.mean([trade["midReturn"] for trade in trades]))
                if trades
                else None,
                "aggressivePortfolioReturn": float(
                    np.mean([trade["aggressiveReturn"] for trade in trades])
                )
                if trades
                else None,
                "touchFillRate": float(np.mean([trade["passiveTouchFill"] for trade in trades]))
                if trades
                else None,
                "conservativeFillRate": float(
                    np.mean([trade["passiveConservativeFill"] for trade in trades])
                )
                if trades
                else None,
            }
        )
    evaluated = [day for day in days if day.get("eligible")]
    return {
        "forwardCalendarDays": len(days),
        "evaluatedStrategyDays": len(evaluated),
        "minimumDaysForConclusion": int(config["sampling"]["minimumForwardDaysForConclusion"]),
        "status": "diagnostic_only"
        if len(evaluated) >= int(config["sampling"]["minimumForwardDaysForConclusion"])
        else "insufficient_forward_execution_days",
        "days": days,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict[str, Any]) -> str:
    execution = report["executionSummary"]
    strategy = report["strategyShadow"]
    lines = [
        "# Forward ETF Execution-Friction Audit",
        "",
        "Status: `diagnostic_only / recorded quotes only / no broker calls`",
        "",
        f"- Forward dates observed: {execution['days']}",
        f"- Sampled code/bucket observations: {execution['observations']}",
        f"- Median quoted spread: {execution['spreadBps'].get('median')} bps",
        f"- Median aggressive 30-minute round-trip implementation shortfall: "
        f"{execution['aggressiveRoundTripCostBps30m'].get('median')} bps",
        f"- Aggressive observations within the 8.3 bps edge budget: "
        f"{execution.get('aggressiveWithinBreakEvenBudgetRate', 0):.1%}",
        f"- Passive bid touch-fill proxy (10m): {execution.get('touchFillRate10m', 0):.1%}",
        f"- Conservative ask-cross fill proxy (10m): "
        f"{execution.get('conservativeFillRate10m', 0):.1%}",
        f"- Touch-fill median post-fill 10-minute markout: "
        f"{execution['touchPostFillMidReturn10mBps'].get('median')} bps",
        f"- Conservative-fill median post-fill 10-minute markout: "
        f"{execution['conservativePostFillMidReturn10mBps'].get('median')} bps",
        "",
        "## Daily coverage",
        "",
        "| Date | Observations | Median spread | Median aggressive cost | Touch fill | Conservative fill |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for day, row in execution["byDay"].items():
        lines.append(
            f"| {day} | {row['observations']} | {row['medianSpreadBps']:.2f} bps | "
            f"{row['medianAggressiveCostBps']:.2f} bps | {row['touchFillRate']:.1%} | "
            f"{row['conservativeFillRate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Overnight-reversal forward shadow",
            "",
            f"- Evaluated strategy days: {strategy['evaluatedStrategyDays']} / "
            f"{strategy['minimumDaysForConclusion']} required.",
            f"- Status: `{strategy['status']}`.",
            "",
            "## Interpretation",
            "",
            "- Touch and ask-cross are fill proxies, not actual queue-aware fills.",
            "- A passive order that fills after price falls may suffer adverse selection; fill rate alone is not an edge.",
            "- Coverage changes materially by day, so the strategy shadow cannot be compared as a stable universe yet.",
            "- No result changes live order style, entry gating, sizing, overlays or execution locks.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    by_code_day, data_audit = load_forward_quotes(config)
    observations = sample_execution_observations(by_code_day, config)
    execution_summary = summarize_execution(observations, config)
    strategy_shadow = forward_strategy_shadow(by_code_day, config)
    report = {
        "schemaVersion": "forward_execution_friction_result_v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "diagnostic_only",
        "dataAudit": data_audit,
        "executionSummary": execution_summary,
        "strategyShadow": strategy_shadow,
        "verdict": strategy_shadow["status"],
        "liveChanges": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "forward_execution_friction_result.json", report)
    write_csv(output_dir / "forward_execution_observations.csv", observations)
    (output_dir / "forward_execution_friction_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(
        {
            "status": report["status"],
            "verdict": report["verdict"],
            "executionSummary": execution_summary,
            "strategyDays": strategy_shadow["evaluatedStrategyDays"],
            "output": str(output_dir / "forward_execution_friction_report.md"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
