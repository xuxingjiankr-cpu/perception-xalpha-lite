"""Paper-competition-only cross-sectional frontier ranker.

The ranker combines price state with fresh point-in-time market microstructure:
short momentum, acceleration, causal VWAP location, five-level order-book
imbalance after removing its common component, microprice displacement, IOPV
premium residual, and quoted spread.

It changes candidate ordering only. It cannot create orders, bypass checks,
change sizing, write overlays, or call a broker. If fresh depth coverage is
insufficient, it leaves the baseline momentum ranking in force.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CN = ZoneInfo("Asia/Shanghai")


def as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text else ""


def parse_time(row: dict[str, Any]) -> datetime | None:
    for key in ("collected_at", "timestamp", "source_quote_time"):
        text = str(row.get(key) or "").strip()
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=CN)
            return parsed.astimezone(CN)
        except ValueError:
            continue
    day = str(row.get("date") or row.get("trade_date") or "").strip()
    clock = str(row.get("ts") or "").strip()
    if day and clock:
        try:
            return datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=CN)
        except ValueError:
            return None
    return None


def load_jsonl_tail(path: Path, max_rows: int = 60_000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = deque(handle, maxlen=max_rows)
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def rows_for_day(
    relative_patterns: list[str],
    trade_date: str,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in relative_patterns:
        path = ROOT / pattern.format(trade_date=trade_date)
        if path.exists():
            rows.extend(load_jsonl_tail(path, max_rows=max_rows))
    return rows


def latest_fresh_depth(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    max_age_seconds: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    rejected: dict[str, int] = defaultdict(int)
    for row in rows:
        code = normalize_code(row.get("code") or row.get("stockCode"))
        timestamp = parse_time(row)
        if not code or timestamp is None:
            rejected["missing_code_or_time"] += 1
            continue
        age = (now - timestamp).total_seconds()
        if timestamp.date() != now.date() or age < -5 or age > max_age_seconds:
            rejected["stale"] += 1
            continue
        if row.get("is_fresh") is False:
            rejected["source_marked_stale"] += 1
            continue
        obi = as_number(row.get("obi"))
        micro = as_number(row.get("micro_dev_bps"))
        half_spread = as_number(row.get("half_spread_bps"))
        if obi is None or micro is None:
            rejected["missing_microstructure"] += 1
            continue
        normalized = {
            **row,
            "code": code,
            "_timestamp": timestamp,
            "_age_seconds": age,
            "obi": obi,
            "micro_dev_bps": micro,
            "half_spread_bps": half_spread,
        }
        if code not in latest or timestamp > latest[code][0]:
            latest[code] = (timestamp, normalized)
    selected = {code: pair[1] for code, pair in latest.items()}
    return selected, {
        "inputRows": len(rows),
        "freshCodes": len(selected),
        "rejected": dict(rejected),
    }


def premium_residuals(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    max_age_seconds: float,
    minimum_history: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_code: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in rows:
        code = normalize_code(row.get("code") or row.get("stockCode"))
        timestamp = parse_time(row)
        premium = as_number(row.get("premium_pct"))
        if not code or timestamp is None or premium is None:
            continue
        if timestamp.date() != now.date() or timestamp > now:
            continue
        by_code[code].append((timestamp, premium))
    residuals: dict[str, dict[str, Any]] = {}
    stale = 0
    for code, observations in by_code.items():
        observations.sort(key=lambda item: item[0])
        latest_time, latest_premium = observations[-1]
        age = (now - latest_time).total_seconds()
        if age < -5 or age > max_age_seconds:
            stale += 1
            continue
        if len(observations) < minimum_history:
            continue
        history = [value for _, value in observations[:-1]]
        if not history:
            continue
        intraday_median = float(median(history))
        residuals[code] = {
            "latestPremiumPct": latest_premium,
            "intradayMedianPremiumPct": intraday_median,
            "discountResidualBps": (intraday_median - latest_premium) * 100.0,
            "observations": len(observations),
            "ageSeconds": age,
        }
    return residuals, {
        "inputRows": len(rows),
        "codesWithHistory": len(residuals),
        "staleCodes": stale,
    }


def percentile_scores(
    values: dict[str, float | None],
    *,
    higher_is_better: bool = True,
) -> dict[str, float]:
    finite = sorted(
        (value, code)
        for code, value in values.items()
        if value is not None and math.isfinite(value)
    )
    if len(finite) <= 1:
        return {code: 0.5 for code in values}
    scores: dict[str, float] = {}
    cursor = 0
    while cursor < len(finite):
        end = cursor + 1
        while end < len(finite) and finite[end][0] == finite[cursor][0]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        score = average_rank / (len(finite) - 1)
        for _, code in finite[cursor:end]:
            scores[code] = score
        cursor = end
    if not higher_is_better:
        scores = {code: 1.0 - score for code, score in scores.items()}
    return {code: scores.get(code, 0.5) for code in values}


def enrich_quotes(
    quotes: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    premium_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    now: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or {}
    enabled = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode") or "shadow")
    if not enabled:
        return quotes, {
            "enabled": False,
            "mode": mode,
            "appliedToRanking": False,
            "reason": "disabled",
        }
    now_cn = now.astimezone(CN)
    depth, depth_meta = latest_fresh_depth(
        depth_rows,
        now=now_cn,
        max_age_seconds=float(cfg.get("max_depth_age_seconds", 120)),
    )
    premium, premium_meta = premium_residuals(
        premium_rows,
        now=now_cn,
        max_age_seconds=float(cfg.get("max_premium_age_seconds", 180)),
        minimum_history=int(cfg.get("minimum_premium_history", 10)),
    )
    allowed_classes = {
        str(value)
        for value in cfg.get(
            "allowed_asset_classes",
            ["gold_etf", "cross_border_etf", "hk_etf"],
        )
    }
    eligible_codes = {
        normalize_code(row.get("stockCode"))
        for row in quotes
        if str(row.get("asset_class") or "") in allowed_classes
    }
    depth_covered = eligible_codes & set(depth)
    coverage = len(depth_covered) / len(eligible_codes) if eligible_codes else 0.0
    minimum_coverage = float(cfg.get("minimum_depth_coverage", 0.50))
    data_ready = bool(eligible_codes and coverage >= minimum_coverage)
    applied = bool(data_ready and mode == "active_rerank")

    latest_obis = [
        float(depth[code]["obi"]) for code in sorted(depth_covered)
    ]
    common_obi = float(median(latest_obis)) if latest_obis else 0.0
    by_code = {
        normalize_code(row.get("stockCode")): row for row in quotes
    }
    raw: dict[str, dict[str, float | None]] = {}
    for code, quote in by_code.items():
        depth_row = depth.get(code)
        premium_row = premium.get(code)
        spread_pct = as_number(quote.get("spread_pct"))
        raw[code] = {
            "momentum": as_number(quote.get("momentum")),
            "acceleration": as_number(quote.get("acceleration")),
            "vwap_distance": as_number(quote.get("vwap_distance_pct")),
            "idiosyncratic_obi": (
                float(depth_row["obi"]) - common_obi if depth_row else None
            ),
            "microprice": (
                as_number(depth_row.get("micro_dev_bps")) if depth_row else None
            ),
            "relative_discount": (
                as_number(premium_row.get("discountResidualBps"))
                if premium_row
                else None
            ),
            "spread_cost": (
                as_number(depth_row.get("half_spread_bps"))
                if depth_row
                else (spread_pct * 5_000.0 if spread_pct is not None else None)
            ),
        }

    directions = {
        "momentum": True,
        "acceleration": True,
        "vwap_distance": True,
        "idiosyncratic_obi": True,
        "microprice": True,
        "relative_discount": True,
        "spread_cost": False,
    }
    ranks = {
        feature: percentile_scores(
            {code: values[feature] for code, values in raw.items()},
            higher_is_better=higher,
        )
        for feature, higher in directions.items()
    }
    default_weights = {
        "momentum": 0.25,
        "acceleration": 0.15,
        "vwap_distance": 0.10,
        "idiosyncratic_obi": 0.20,
        "microprice": 0.10,
        "relative_discount": 0.15,
        "spread_cost": 0.05,
    }
    configured = cfg.get("feature_weights")
    weights = (
        {
            key: float(configured.get(key, value))
            for key, value in default_weights.items()
        }
        if isinstance(configured, dict)
        else default_weights
    )
    weight_total = sum(max(0.0, value) for value in weights.values()) or 1.0

    enriched: list[dict[str, Any]] = []
    for quote in quotes:
        code = normalize_code(quote.get("stockCode"))
        depth_row = depth.get(code)
        premium_row = premium.get(code)
        components = {
            feature: ranks[feature].get(code, 0.5) for feature in weights
        }
        score = 100.0 * sum(
            max(0.0, weights[feature]) * components[feature]
            for feature in weights
        ) / weight_total
        out = dict(quote)
        out["frontier_rank_score"] = round(score, 4)
        out["frontier_rank_ready"] = applied
        out["frontier_policy"] = {
            "schemaVersion": "frontier_competition_policy_v1",
            "mode": mode,
            "appliedToRanking": applied,
            "score": round(score, 4),
            "components": {
                key: round(value, 6) for key, value in components.items()
            },
            "rawFeatures": raw.get(code, {}),
            "commonObi": round(common_obi, 8),
            "depthAgeSeconds": (
                round(float(depth_row["_age_seconds"]), 3)
                if depth_row
                else None
            ),
            "premium": premium_row,
            "estimatedAggressiveRoundTripCostBps": (
                round(
                    2.0 * float(depth_row["half_spread_bps"])
                    + float(cfg.get("execution_buffer_bps", 4.0)),
                    4,
                )
                if depth_row and depth_row.get("half_spread_bps") is not None
                else None
            ),
            "evidenceStatus": (
                "experimental_paper_competition_not_validated_alpha"
            ),
        }
        enriched.append(out)
    return enriched, {
        "schemaVersion": "frontier_competition_policy_run_v1",
        "enabled": True,
        "mode": mode,
        "paperCompetitionOnly": bool(cfg.get("paper_competition_only", True)),
        "appliedToRanking": applied,
        "reason": (
            "fresh_depth_coverage_ready"
            if applied
            else ("shadow_mode" if data_ready else "insufficient_fresh_depth_coverage")
        ),
        "eligibleCodes": len(eligible_codes),
        "depthCoveredCodes": len(depth_covered),
        "depthCoverage": round(coverage, 6),
        "minimumDepthCoverage": minimum_coverage,
        "commonObi": round(common_obi, 8),
        "depth": depth_meta,
        "premium": premium_meta,
        "weights": weights,
        "alphaValidated": False,
    }


def apply_frontier_policy(
    quotes: list[dict[str, Any]],
    strategy_config: dict[str, Any],
    trade_date: str,
    *,
    now: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = strategy_config.get("frontier_competition_policy", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return quotes, {
            "enabled": False,
            "appliedToRanking": False,
            "reason": "disabled",
        }
    depth_rows = rows_for_day(
        list(
            cfg.get(
                "depth_files",
                ["outputs/l2_depth/depth_{trade_date}.jsonl"],
            )
        ),
        trade_date,
        max_rows=int(cfg.get("maximum_depth_rows", 60_000)),
    )
    premium_rows = rows_for_day(
        list(
            cfg.get(
                "premium_files",
                [
                    "data/research/etf_iopv/iopv_{trade_date}.jsonl",
                    "outputs/iopv_premium/iopv_{trade_date}.jsonl",
                ],
            )
        ),
        trade_date,
        max_rows=int(cfg.get("maximum_premium_rows", 30_000)),
    )
    return enrich_quotes(quotes, depth_rows, premium_rows, cfg, now=now)


def rank_quotes(
    quotes: list[dict[str, Any]],
    strategy_config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = strategy_config.get("frontier_competition_policy", {})
    active = bool(
        isinstance(cfg, dict)
        and cfg.get("enabled", False)
        and str(cfg.get("mode")) == "active_rerank"
        and any(row.get("frontier_rank_ready") for row in quotes)
    )
    def value(row: dict[str, Any], key: str) -> float:
        number = as_number(row.get(key))
        return number if number is not None else -999.0

    if not active:
        return sorted(
            quotes,
            key=lambda row: value(row, "momentum"),
            reverse=True,
        )
    return sorted(
        quotes,
        key=lambda row: (
            value(row, "frontier_rank_score"),
            value(row, "momentum"),
            str(row.get("stockCode") or ""),
        ),
        reverse=True,
    )
