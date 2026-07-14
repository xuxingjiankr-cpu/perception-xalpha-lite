"""CogAlpha-inspired ETF factor research, safely isolated from trading.

The paper's useful idea is an automated loop that generates, audits, evolves and
tests interpretable factor code.  This implementation deliberately narrows the code
surface to a causal expression DSL: an LLM may propose expressions, but it cannot
execute arbitrary Python or touch the paper agent.  Selection uses training data,
one frozen ensemble is selected on validation, and the historical test is report-only.

This script never imports the trading agent, broker client, order builder or overlays.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))

import overfitting_guard as og  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "research" / "cogalpha_etf_v1_preregistered.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "edge_research" / "cogalpha_etf"
FORBIDDEN_PATH_PARTS = {
    "configs/t0_intraday_paper_agent.json",
    "configs/etf_paper_trading_agent.json",
    "configs/etf_paper_trading_agent_execute.json",
    "latest_strategy_overlay.json",
    "decision_probability_v1.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "cogalpha_etf_research_v1":
        raise ValueError("unexpected CogAlpha config schema")
    safety = config.get("safety", {})
    forbidden_true = [key for key, value in safety.items() if key.startswith("may") and value is not False]
    if forbidden_true:
        raise ValueError(f"research safety flags must all be false: {forbidden_true}")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("outputStatus must remain diagnostic_only")
    output_root = str(safety.get("allowedOutputRoot", "")).replace("\\", "/")
    if output_root != "outputs/edge_research/cogalpha_etf":
        raise ValueError("output root is not isolated")
    data = config["data"]
    if not (data["trainEnd"] < data["validationStart"] <= data["validationEnd"] < data["sealedTestStart"]):
        raise ValueError("chronological split is invalid")
    if config["promotion"].get("historicalRunCanPromote") is not False:
        raise ValueError("historical CogAlpha run must not promote")
    if config["generator"]["maximumCandidateCount"] > 256:
        raise ValueError("candidate count exceeds preregistered multiple-testing bound")
    feedback = config["generator"].get("adaptiveFeedback")
    if feedback:
        if (
            feedback.get("feedbackData") != "train_only"
            or feedback.get("validationFeedbackAllowed") is not False
            or feedback.get("testFeedbackAllowed") is not False
            or feedback.get("previousRunTestFeedbackAllowed") is not False
        ):
            raise ValueError("adaptive feedback must remain train-only")
    if int(config["selection"].get("priorResearchTrials", 0)) < 0:
        raise ValueError("priorResearchTrials cannot be negative")


def build_panel(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_cfg = config["data"]
    bars_root = ROOT / data_cfg["barsRoot"]
    minimum_obs = int(data_cfg["minimumObservationsPerSymbol"])
    minimum_amount = float(data_cfg["minimumMedianDailyAmountCny"])
    fields: dict[str, dict[str, pd.Series]] = {
        key: {} for key in ("open", "high", "low", "close", "volume", "amount")
    }
    for path in sorted(bars_root.glob("*.jsonl")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except (ValueError, TypeError):
                continue
        if len(rows) < minimum_obs:
            continue
        frame = pd.DataFrame(rows)
        required = {"dt", "open", "high", "low", "close", "vol", "amount"}
        if not required.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["dt"].astype(str).str[:10], errors="coerce")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        if len(frame) < minimum_obs or float(pd.to_numeric(frame["amount"], errors="coerce").median()) < minimum_amount:
            continue
        code = path.stem
        for field in fields:
            source = "vol" if field == "volume" else field
            fields[field][code] = pd.to_numeric(frame[source], errors="coerce")
    panel = {field: pd.DataFrame(values).sort_index() for field, values in fields.items()}
    close = panel["close"]
    panel["vwap"] = (panel["amount"] / panel["volume"].replace(0.0, np.nan)).combine_first(close)
    panel["returns"] = close.pct_change(fill_method=None)
    return panel


# Expression nodes are dictionaries with one of: field, unary, binary, rolling,
# lag, corr, zscore, drawdown, range_position.  No arbitrary code is evaluated.
def expression_depth(expr: dict[str, Any]) -> int:
    if "field" in expr:
        return 1
    children = []
    for key in ("arg", "left", "right"):
        if isinstance(expr.get(key), dict):
            children.append(expr[key])
    return 1 + max((expression_depth(child) for child in children), default=0)


def validate_expression(expr: Any, config: dict[str, Any]) -> None:
    if not isinstance(expr, dict) or len(expr) == 0:
        raise ValueError("expression must be a non-empty object")
    if expression_depth(expr) > int(config["generator"]["maximumExpressionDepth"]):
        raise ValueError("expression is too deep")
    allowed_fields = set(config["generator"]["rawInputs"])
    windows = set(int(value) for value in config["generator"]["allowedWindows"])

    def walk(node: dict[str, Any]) -> None:
        kinds = [key for key in ("field", "unary", "binary", "rolling", "lag", "corr", "zscore", "drawdown", "range_position") if key in node]
        if len(kinds) != 1:
            raise ValueError(f"expression node must have exactly one operator: {node}")
        kind = kinds[0]
        if kind == "field":
            if node["field"] not in allowed_fields:
                raise ValueError(f"field not allowed: {node['field']}")
            return
        if kind == "unary":
            if node["unary"] not in {"abs", "neg", "tanh", "signed_log1p"}:
                raise ValueError("unary operator not allowed")
            walk(node["arg"])
            return
        if kind == "binary":
            if node["binary"] not in {"add", "sub", "mul", "div"}:
                raise ValueError("binary operator not allowed")
            walk(node["left"])
            walk(node["right"])
            return
        if kind == "lag":
            lag = int(node["lag"])
            if lag < 0 or lag > 252:
                raise ValueError("lag must be past-only")
            walk(node["arg"])
            return
        window = int(node.get("window", 0))
        if window not in windows:
            raise ValueError(f"window not preregistered: {window}")
        if kind == "rolling":
            if node["rolling"] not in {"mean", "std", "min", "max", "median", "sum"}:
                raise ValueError("rolling operator not allowed")
            walk(node["arg"])
        elif kind == "corr":
            walk(node["left"])
            walk(node["right"])
        elif kind in {"zscore", "drawdown", "range_position"}:
            walk(node["arg"])

    walk(expr)


def evaluate_expression(expr: dict[str, Any], panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "field" in expr:
        return panel[expr["field"]].copy()
    if "unary" in expr:
        value = evaluate_expression(expr["arg"], panel)
        operator = expr["unary"]
        if operator == "abs":
            return value.abs()
        if operator == "neg":
            return -value
        if operator == "tanh":
            return pd.DataFrame(np.tanh(value), index=value.index, columns=value.columns)
        return np.sign(value) * np.log1p(value.abs())
    if "binary" in expr:
        left = evaluate_expression(expr["left"], panel)
        right = evaluate_expression(expr["right"], panel)
        operator = expr["binary"]
        if operator == "add":
            return left + right
        if operator == "sub":
            return left - right
        if operator == "mul":
            return left * right
        return left / right.replace(0.0, np.nan)
    if "lag" in expr:
        return evaluate_expression(expr["arg"], panel).shift(int(expr["lag"]))
    if "rolling" in expr:
        value = evaluate_expression(expr["arg"], panel)
        window = int(expr["window"])
        rolling = value.rolling(window, min_periods=max(2, window // 2))
        return getattr(rolling, expr["rolling"])()
    if "corr" in expr:
        left = evaluate_expression(expr["left"], panel)
        right = evaluate_expression(expr["right"], panel)
        return left.rolling(int(expr["window"]), min_periods=int(expr["window"])).corr(right)
    value = evaluate_expression(expr["arg"], panel)
    window = int(expr["window"])
    minimum = max(2, window // 2)
    if "zscore" in expr:
        mean = value.rolling(window, min_periods=minimum).mean()
        std = value.rolling(window, min_periods=minimum).std().replace(0.0, np.nan)
        return (value - mean) / std
    if "drawdown" in expr:
        high = value.rolling(window, min_periods=minimum).max().replace(0.0, np.nan)
        return value / high - 1.0
    low = value.rolling(window, min_periods=minimum).min()
    high = value.rolling(window, min_periods=minimum).max()
    return (value - low) / (high - low).replace(0.0, np.nan)


def seed_candidates() -> list[dict[str, Any]]:
    f = lambda name: {"field": name}
    roll = lambda op, arg, window: {"rolling": op, "arg": arg, "window": window}
    binary = lambda op, left, right: {"binary": op, "left": left, "right": right}
    unary = lambda op, arg: {"unary": op, "arg": arg}
    ret = f("returns")
    volume = f("volume")
    close = f("close")
    amount = f("amount")
    candle_range = binary("div", binary("sub", f("high"), f("low")), close)
    body = binary("div", binary("sub", f("close"), f("open")), binary("sub", f("high"), f("low")))
    price_volume_corr = {"corr": True, "left": ret, "right": unary("signed_log1p", volume), "window": 20}
    seeds = [
        ("market_cycle", "trend_gap_20_60", binary("div", roll("mean", close, 20), roll("mean", close, 60))),
        ("volatility_regime", "vol_ratio_5_20", binary("div", roll("std", ret, 5), roll("std", ret, 20))),
        ("tail_risk", "negative_return_pressure_20", roll("mean", unary("abs", binary("sub", ret, unary("abs", ret))), 20)),
        ("crash_predictor", "drawdown_60", {"drawdown": True, "arg": close, "window": 60}),
        ("liquidity", "range_per_amount_10", binary("div", roll("mean", candle_range, 10), roll("mean", amount, 10))),
        ("order_imbalance_proxy", "body_volume_pressure_10", roll("mean", binary("mul", body, unary("signed_log1p", volume)), 10)),
        ("price_volume_coherence", "return_volume_corr_20", price_volume_corr),
        ("volume_structure", "volume_zscore_20", {"zscore": True, "arg": unary("signed_log1p", volume), "window": 20}),
        ("daily_trend", "return_sum_20", roll("sum", ret, 20)),
        ("reversal", "negative_return_5", unary("neg", roll("sum", ret, 5))),
        ("range_volatility", "range_mean_10", roll("mean", candle_range, 10)),
        ("lag_response", "lagged_return_5", {"lag": 5, "arg": ret}),
        ("volatility_asymmetry", "downside_vs_total_20", binary("div", roll("std", binary("sub", ret, unary("abs", ret)), 20), roll("std", ret, 20))),
        ("drawdown", "drawdown_recovery_20", binary("add", {"drawdown": True, "arg": close, "window": 60}, roll("sum", ret, 20))),
        ("fractal_proxy", "multi_scale_vol_ratio", binary("div", roll("std", ret, 10), roll("std", ret, 60))),
        ("regime_gating", "trend_stability_gate", binary("div", roll("mean", ret, 20), roll("std", ret, 20))),
        ("stability", "range_stability_20", unary("neg", roll("std", candle_range, 20))),
        ("bar_shape", "body_position_10", roll("mean", body, 10)),
        ("creative", "range_position_60", {"range_position": True, "arg": close, "window": 60}),
        ("composite", "trend_volume_coherence", binary("mul", roll("sum", ret, 20), price_volume_corr)),
        ("herding_proxy", "cross_pressure_20", binary("mul", roll("mean", body, 20), {"zscore": True, "arg": volume, "window": 20})),
    ]
    return [
        {"id": f"seed_{index:02d}_{name}", "agent": agent, "generation": 0, "parents": [], "expression": expr}
        for index, (agent, name, expr) in enumerate(seeds, start=1)
    ]


def replace_first_window(expr: dict[str, Any], new_window: int) -> tuple[dict[str, Any], bool]:
    result = copy.deepcopy(expr)
    if "window" in result:
        result["window"] = new_window
        return result, True
    for key in ("arg", "left", "right"):
        if isinstance(result.get(key), dict):
            child, changed = replace_first_window(result[key], new_window)
            if changed:
                result[key] = child
                return result, True
    return result, False


def evolve(parents: list[dict[str, Any]], config: dict[str, Any], generation: int) -> list[dict[str, Any]]:
    rng = random.Random(int(config["generator"]["randomSeed"]) + generation)
    windows = list(config["generator"]["allowedWindows"])
    children = []
    for index, parent in enumerate(parents):
        changed, ok = replace_first_window(parent["expression"], rng.choice(windows))
        if ok:
            children.append({
                "id": f"g{generation:02d}_m{index:02d}", "agent": parent["agent"],
                "generation": generation, "parents": [parent["id"]], "expression": changed,
            })
        children.append({
            "id": f"g{generation:02d}_t{index:02d}", "agent": parent["agent"],
            "generation": generation, "parents": [parent["id"]],
            "expression": {"unary": "tanh", "arg": copy.deepcopy(parent["expression"])},
        })
        mate = parents[(index + 1) % len(parents)]
        children.append({
            "id": f"g{generation:02d}_c{index:02d}", "agent": "composite",
            "generation": generation, "parents": [parent["id"], mate["id"]],
            "expression": {"binary": "mul", "left": copy.deepcopy(parent["expression"]), "right": copy.deepcopy(mate["expression"])},
        })
    return children


def parse_ollama_candidates(raw: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        raise ValueError("Ollama response did not contain a JSON array")
    rows = json.loads(match.group(0))
    output = []
    maximum = int(config["generator"]["optionalOllama"]["maximumGeneratedCandidates"])
    allowed_agents = {agent for agents in config["generator"]["agentHierarchy"].values() for agent in agents}
    for index, row in enumerate(rows[:maximum]):
        if not isinstance(row, dict) or row.get("agent") not in allowed_agents:
            continue
        validate_expression(row.get("expression"), config)
        output.append({
            "id": f"ollama_{index:02d}", "agent": row["agent"], "generation": 0,
            "parents": [], "expression": row["expression"],
            "rationale": str(row.get("rationale", ""))[:500],
        })
    return output


def generate_with_ollama(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = config["generator"]["optionalOllama"]
    if not provider.get("enabled"):
        return [], {"status": "disabled"}
    model = str(provider.get("model", "")).strip()
    if not model:
        return [], {"status": "skipped", "reason": "no_local_model_preregistered"}
    prompt = {
        "role": "Generate diverse, economically interpretable past-only ETF OHLCV alpha expressions.",
        "constraints": {
            "output": "JSON array only: [{agent,rationale,expression}]",
            "operators": ["field", "unary", "binary", "rolling", "lag", "corr", "zscore", "drawdown", "range_position"],
            "fields": config["generator"]["rawInputs"],
            "windows": config["generator"]["allowedWindows"],
            "future_data": "forbidden; lag must be >=0; no labels or future returns",
            "agent_hierarchy": config["generator"]["agentHierarchy"],
        },
    }
    body = json.dumps({
        "model": model,
        "prompt": canonical(prompt),
        "stream": False,
        "options": {"temperature": float(provider["temperature"])},
    }).encode("utf-8")
    request = urllib.request.Request(
        provider["baseUrl"].rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=int(provider["timeoutSeconds"])) as response:
            payload = json.loads(response.read().decode("utf-8"))
        candidates = parse_ollama_candidates(str(payload.get("response", "")), config)
        return candidates, {"status": "ok", "model": model, "accepted": len(candidates)}
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return [], {"status": "failed_closed", "model": model, "reason": str(exc)[:300]}


@dataclass
class Evaluation:
    candidate: dict[str, Any]
    signal: pd.DataFrame
    direction: float
    ic: pd.Series
    rank_ic: pd.Series
    long_short: pd.Series
    long_net: pd.Series
    hit: pd.Series


def evaluate_signal(candidate: dict[str, Any], signal: pd.DataFrame, panel: dict[str, pd.DataFrame], config: dict[str, Any]) -> Evaluation | None:
    data_cfg = config["data"]
    close = panel["close"]
    valid = close.notna() & (close > 0)
    forward = close.shift(-2) / close.shift(-1) - 1.0
    signal = signal.replace([np.inf, -np.inf], np.nan).where(valid)
    minimum_cross_section = int(data_cfg["minimumCrossSection"])
    usable = signal.notna().sum(axis=1) >= minimum_cross_section
    if int(usable.sum()) < int(config["selection"]["minimumTrainRankIcDays"]):
        return None
    train_mask = usable & (signal.index <= pd.Timestamp(data_cfg["trainEnd"]))
    raw_rank_ic = signal[train_mask].corrwith(forward[train_mask], axis=1, method="spearman").dropna()
    if len(raw_rank_ic) < int(config["selection"]["minimumTrainRankIcDays"]):
        return None
    direction = 1.0 if float(raw_rank_ic.mean()) >= 0 else -1.0
    oriented = signal * direction
    rank_ic = oriented[usable].corrwith(forward[usable], axis=1, method="spearman").dropna()
    ic = oriented[usable].corrwith(forward[usable], axis=1, method="pearson").dropna()
    ranks = oriented.rank(axis=1, pct=True)
    top = ranks.ge(1.0 - float(data_cfg["topQuantile"])) & usable.values[:, None]
    bottom = ranks.le(float(data_cfg["topQuantile"])) & usable.values[:, None]
    top_weights = top.div(top.sum(axis=1), axis=0).fillna(0.0)
    bottom_weights = bottom.div(bottom.sum(axis=1), axis=0).fillna(0.0)
    top_return = (top_weights * forward).sum(axis=1)
    bottom_return = (bottom_weights * forward).sum(axis=1)
    benchmark = forward.where(valid).mean(axis=1)
    turnover = top_weights.diff().abs().sum(axis=1) / 2.0
    long_short = (top_return - bottom_return)[usable]
    long_net = ((top_return - benchmark) - turnover * float(data_cfg["roundTripCost"]))[usable]
    hit = (top_return > benchmark)[usable].astype(float)
    return Evaluation(candidate, oriented, direction, ic, rank_ic, long_short, long_net, hit)


def period_mask(index: pd.Index, period: str, config: dict[str, Any]) -> np.ndarray:
    data = config["data"]
    if period == "train":
        return index <= pd.Timestamp(data["trainEnd"])
    if period == "validation":
        return (index >= pd.Timestamp(data["validationStart"])) & (index <= pd.Timestamp(data["validationEnd"]))
    mask = index >= pd.Timestamp(data["sealedTestStart"])
    if data.get("sealedTestEnd"):
        mask &= index <= pd.Timestamp(data["sealedTestEnd"])
    return mask


def series_stats(series: pd.Series, period: str, config: dict[str, Any]) -> dict[str, Any]:
    values = series[period_mask(series.index, period, config)].dropna()
    if len(values) < 2:
        return {"n": int(len(values)), "mean": None, "t": None, "irAnn": None}
    std = float(values.std(ddof=1))
    mean = float(values.mean())
    return {
        "n": int(len(values)),
        "mean": round(mean, 8),
        "t": round(mean / std * math.sqrt(len(values)), 4) if std > 0 else None,
        "irAnn": round(mean / std * math.sqrt(244), 4) if std > 0 else None,
    }


def summarize(evaluation: Evaluation, config: dict[str, Any]) -> dict[str, Any]:
    periods = {}
    for period in ("train", "validation", "test"):
        periods[period] = {
            "ic": series_stats(evaluation.ic, period, config),
            "rankIc": series_stats(evaluation.rank_ic, period, config),
            "longShort": series_stats(evaluation.long_short, period, config),
            "longNet": series_stats(evaluation.long_net, period, config),
            "topDecileHitRate": series_stats(evaluation.hit, period, config),
        }
    return {
        "id": evaluation.candidate["id"],
        "agent": evaluation.candidate.get("agent"),
        "generation": evaluation.candidate.get("generation"),
        "parents": evaluation.candidate.get("parents", []),
        "directionFromTrain": evaluation.direction,
        "expression": evaluation.candidate.get("expression"),
        "periods": periods,
    }


def train_fitness(summary: dict[str, Any]) -> float:
    metrics = summary["periods"]["train"]
    rank_ir = metrics["rankIc"].get("irAnn") or -99.0
    ic_ir = metrics["ic"].get("irAnn") or -99.0
    long_ir = metrics["longNet"].get("irAnn") or -99.0
    return float(rank_ir) + 0.35 * float(ic_ir) + 0.25 * max(-2.0, float(long_ir))


def validation_fitness(summary: dict[str, Any]) -> float:
    metrics = summary["periods"]["validation"]
    rank_ir = metrics["rankIc"].get("irAnn") or -99.0
    long_ir = metrics["longNet"].get("irAnn") or -99.0
    return float(rank_ir) + 0.75 * float(long_ir)


def load_fixed_baselines(panel: dict[str, pd.DataFrame], config: dict[str, Any]) -> list[Evaluation]:
    output = []
    for baseline in config["fixedBaselines"]:
        try:
            module = importlib.import_module(f"src.factors.zoo.{baseline['zoo']}.{baseline['name']}")
            signal = module.compute(panel)
            candidate = {
                "id": f"baseline_{baseline['zoo']}_{baseline['name']}",
                "agent": "fixed_baseline", "generation": -1, "parents": [], "expression": None,
            }
            evaluated = evaluate_signal(candidate, signal, panel, config)
            if evaluated:
                output.append(evaluated)
        except Exception:
            continue
    return output


def ensemble_evaluation(elites: list[Evaluation], panel: dict[str, pd.DataFrame], config: dict[str, Any]) -> Evaluation | None:
    if not elites:
        return None
    ranks = [item.signal.rank(axis=1, pct=True) for item in elites]
    signal = sum(ranks) / len(ranks)
    candidate = {
        "id": "cogalpha_frozen_ensemble", "agent": "frozen_ensemble", "generation": "selected_on_validation",
        "parents": [item.candidate["id"] for item in elites], "expression": None,
    }
    return evaluate_signal(candidate, signal, panel, config)


def input_fingerprint(config_path: Path, config: dict[str, Any]) -> str:
    bars = ROOT / config["data"]["barsRoot"]
    manifest = []
    for path in sorted(bars.glob("*.jsonl")):
        stat = path.stat()
        manifest.append((path.name, stat.st_size, stat.st_mtime_ns))
    return digest({"config": digest(load_json(config_path)), "bars": manifest})


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# CogAlpha ETF Research Report", "",
        f"- Status: `{result['status']}`",
        f"- Run ID: `{result['runId']}`",
        f"- Data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}` | "
        f"{result['dataAudit']['symbols']} symbols | {result['dataAudit']['days']} calendar rows",
        f"- Generated/evaluated: `{result['candidateAudit']['generated']}` / `{result['candidateAudit']['evaluated']}`",
        f"- Ollama: `{result['candidateAudit']['ollama']}`", "",
        "This is an offline research artifact. It cannot change orders, sizing, risk gates, overlays or build_decision().", "",
        "## Frozen selection", "",
        f"Selected parents: `{result['selection']['parents']}`",
        f"Validation passed preregistered baseline gate: `{result['selection']['validationGatePassed']}`", "",
        "| model | period | RankIC IR | net long-only IR | net mean/day | hit rate |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["comparison"]:
        for period in ("train", "validation", "test"):
            metrics = row["periods"][period]
            lines.append(
                f"| {row['id']} | {period} | {metrics['rankIc'].get('irAnn')} | "
                f"{metrics['longNet'].get('irAnn')} | {metrics['longNet'].get('mean')} | "
                f"{metrics['topDecileHitRate'].get('mean')} |"
            )
    lines += [
        "", "## Multiple-testing and credibility", "",
        f"- Candidate-level PBO: `{result['guards']['pbo']}`",
        f"- Deflated-significance note: `{result['guards']['dsr']}`",
        f"- Test window pristine: `false` — {result['limitations']['testContaminationWarning']}",
        f"- Historical run promotable: `false`", "",
        "## Verdict", "",
        result["verdict"], "",
        "## Skipped checks", "",
        *[f"- {key}: {value}" for key, value in result["skippedChecks"].items()], "",
        "Even a positive historical test would remain a hypothesis until at least 20 new independent forward days and 200 candidate events pass the frozen gates.",
    ]
    return "\n".join(lines) + "\n"


def run(config_path: Path, output_root: Path, maximum_candidates: int | None = None) -> tuple[dict[str, Any], Path]:
    config = load_json(config_path)
    validate_config(config)
    if maximum_candidates is not None:
        config = copy.deepcopy(config)
        config["generator"]["maximumCandidateCount"] = min(
            int(config["generator"]["maximumCandidateCount"]), int(maximum_candidates)
        )
    panel = build_panel(config)
    if panel.get("close") is None or panel["close"].empty:
        raise RuntimeError("no eligible ETF daily panel")

    candidates = seed_candidates()
    ollama_candidates, ollama_status = generate_with_ollama(config)
    candidates.extend(ollama_candidates)
    seen = {digest(candidate["expression"]) for candidate in candidates}
    evaluated: list[Evaluation] = []
    summaries: list[dict[str, Any]] = []

    def evaluate_new(rows: list[dict[str, Any]]) -> None:
        for candidate in rows:
            if len(evaluated) >= int(config["generator"]["maximumCandidateCount"]):
                break
            try:
                validate_expression(candidate["expression"], config)
                signal = evaluate_expression(candidate["expression"], panel)
                item = evaluate_signal(candidate, signal, panel, config)
            except Exception:
                item = None
            if item is not None:
                evaluated.append(item)
                summaries.append(summarize(item, config))

    evaluate_new(candidates)
    for generation in range(1, int(config["generator"]["generations"]) + 1):
        ranked = sorted(zip(evaluated, summaries), key=lambda pair: train_fitness(pair[1]), reverse=True)
        parents = [pair[0].candidate for pair in ranked[: int(config["generator"]["parentPoolSize"])]]
        children = []
        for child in evolve(parents, config, generation):
            fingerprint = digest(child["expression"])
            if fingerprint not in seen:
                seen.add(fingerprint)
                children.append(child)
        evaluate_new(children)

    ranked_train = sorted(zip(evaluated, summaries), key=lambda pair: train_fitness(pair[1]), reverse=True)
    train_elites = ranked_train[: int(config["generator"]["elitePoolSize"])]
    ranked_validation = sorted(train_elites, key=lambda pair: validation_fitness(pair[1]), reverse=True)
    maximum_ensemble = int(config["selection"]["ensembleMaximumFactors"])
    frozen_elites = [pair[0] for pair in ranked_validation[:maximum_ensemble]]
    ensemble = ensemble_evaluation(frozen_elites, panel, config)
    if ensemble is None:
        raise RuntimeError("could not build frozen CogAlpha ensemble")
    ensemble_summary = summarize(ensemble, config)

    baselines = load_fixed_baselines(panel, config)
    baseline_summaries = [summarize(item, config) for item in baselines]
    best_baseline_validation = max(
        (row["periods"]["validation"]["longNet"].get("irAnn") or -99.0 for row in baseline_summaries),
        default=-99.0,
    )
    ensemble_validation = ensemble_summary["periods"]["validation"]["longNet"].get("irAnn") or -99.0
    gate_passed = ensemble_validation > 0 and ensemble_validation > best_baseline_validation

    development_end = pd.Timestamp(config["data"]["validationEnd"])
    common_dates = (
        sorted(
            date
            for date in set.intersection(*(set(item.long_short.index) for item in evaluated))
            if date <= development_end
        )
        if evaluated else []
    )
    performance_matrix = [item.long_short.reindex(common_dates).fillna(0.0).tolist() for item in evaluated]
    pbo = og.combinatorial_symmetric_pbo(performance_matrix, n_blocks=8) if len(performance_matrix) >= 2 else {"pbo": None}
    test_net_ir = ensemble_summary["periods"]["test"]["longNet"].get("irAnn") or 0.0
    prior_trials = int(config["selection"].get("priorResearchTrials", 0))
    total_trials = prior_trials + len(evaluated)
    dsr = og.deflated_significance_note(total_trials, test_net_ir, ensemble_summary["periods"]["test"]["longNet"].get("n") or 1)

    close = panel["close"]
    fingerprint = input_fingerprint(config_path, config)
    run_label = re.sub(r"[^a-z0-9_]+", "_", str(config.get("runLabel", "cogalpha")).lower()).strip("_") or "cogalpha"
    run_id = f"{run_label}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{fingerprint[:10]}"
    output = output_root / run_id
    verdict = (
        "Historical validation beats the fixed baselines, but this remains diagnostic-only because the test window is research-contaminated and the multiple-testing/forward gates still apply."
        if gate_passed
        else "The frozen CogAlpha ensemble did not beat the fixed validation baselines after costs. No trading integration is justified."
    )
    result = {
        "schemaVersion": "cogalpha_etf_research_result_v1",
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": run_id,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputFingerprint": fingerprint,
        "configFingerprint": digest(config),
        "dataAudit": {
            "start": close.index.min().date().isoformat(), "end": close.index.max().date().isoformat(),
            "days": int(len(close.index)), "symbols": int(len(close.columns)), "barInterval": "1d",
        },
        "candidateAudit": {
            "generated": len(seen), "evaluated": len(evaluated), "ollama": ollama_status,
            "arbitraryPythonExecuted": False, "strictlyPastOnlyDsl": True,
        },
        "selection": {
            "parents": ensemble.candidate["parents"],
            "usedTestForSelection": False,
            "validationGatePassed": gate_passed,
            "bestBaselineValidationNetIr": best_baseline_validation,
            "ensembleValidationNetIr": ensemble_validation,
        },
        "comparison": baseline_summaries + [ensemble_summary],
        "candidateSummaries": summaries,
        "guards": {
            "pbo": pbo,
            "dsr": dsr,
            "currentRunTrials": len(evaluated),
            "priorResearchTrials": prior_trials,
            "nTrials": total_trials,
        },
        "limitations": {"testContaminationWarning": config["data"]["testContaminationWarning"]},
        "skippedChecks": {
            "etfCategoryStability": "skipped: local point-in-time ETF category history is incomplete; current-name classification would add survivorship leakage",
            "minuteReplay": "skipped at discovery stage: only a validation-passing daily candidate may enter a separate minute replay",
            "forwardShadow": "not started: historical validation gate did not pass",
        },
        "verdict": verdict,
        "automaticTradingChanges": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output / "report.md", render_report(result))
    atomic_write(output / "frozen_candidate.json", json.dumps({
        "schemaVersion": "cogalpha_etf_frozen_candidate_v1",
        "status": "research_only_not_a_trade_signal",
        "runId": run_id,
        "parents": ensemble.candidate["parents"],
        "expressions": [item.candidate["expression"] for item in frozen_elites],
        "mayPromoteAutomatically": False,
    }, ensure_ascii=False, indent=2) + "\n")
    return result, output


def self_test() -> None:
    config = load_json(DEFAULT_CONFIG)
    validate_config(config)
    panel = {
        field: pd.DataFrame({"510300": np.arange(1.0, 41.0), "510500": np.arange(2.0, 42.0)})
        for field in config["generator"]["rawInputs"]
    }
    expr = {"zscore": True, "arg": {"field": "close"}, "window": 10}
    validate_expression(expr, config)
    full = evaluate_expression(expr, panel)
    prefix = {key: value.iloc[:25].copy() for key, value in panel.items()}
    short = evaluate_expression(expr, prefix)
    if not full.iloc[:25].equals(short):
        raise AssertionError("prefix invariance failed")
    rejected = False
    try:
        validate_expression({"lag": -1, "arg": {"field": "close"}}, config)
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("future lag was not rejected")
    if any(config["safety"].get(key) for key in config["safety"] if key.startswith("may")):
        raise AssertionError("research config can mutate trading")
    print("cogalpha self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-candidates", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result, output = run(args.config.resolve(), args.output_root.resolve(), args.maximum_candidates)
    print(json.dumps({
        "status": result["status"], "run_id": result["runId"], "output": str(output),
        "validation_gate_passed": result["selection"]["validationGatePassed"],
        "automatic_trading_changes": result["automaticTradingChanges"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
