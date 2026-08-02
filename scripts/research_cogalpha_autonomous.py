"""Continuous CogAlpha-style ETF alpha discovery, permanently research-only.

The engine can search without an external API by using a diverse, role-aware factor
grammar.  If a local Ollama model is explicitly configured, it also receives train-only
fitness feedback and may propose semantic mutations.  Validation and shadow quarantine
data are never returned to either generator.  The only strategy artifact is a shadow
ETF ranking with an empty order list.
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "vibe_factors"))

import overfitting_guard as og  # noqa: E402
import research_cogalpha_etf as core  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "research" / "cogalpha_autonomous_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "edge_research" / "cogalpha_autonomous"
CODE_VERSION = "cogalpha_autonomous_v1.0"


ROLE_HYPOTHESES: dict[str, dict[str, str]] = {
    "market_cycle": {
        "mechanism": "Slow-moving allocation flows create persistent trend and cycle structure.",
        "forcedTrader": "benchmark trackers and allocation rebalancers",
        "persistence": "Flows are split across sessions because liquidity is finite.",
    },
    "volatility_regime": {
        "mechanism": "Risk budgets contract after volatility rises and expand after it normalizes.",
        "forcedTrader": "volatility-targeting and risk-control portfolios",
        "persistence": "Risk estimates and mandate adjustments update with a lag.",
    },
    "tail_risk": {
        "mechanism": "Asymmetric downside pressure reveals constrained risk absorption.",
        "forcedTrader": "leveraged holders and stop-driven sellers",
        "persistence": "Deleveraging is staged and can spill into following sessions.",
    },
    "crash_predictor": {
        "mechanism": "Drawdown depth and unstable ranges proxy fragility before forced liquidation.",
        "forcedTrader": "margin-constrained and drawdown-limited portfolios",
        "persistence": "Constraints bind at different thresholds across participants.",
    },
    "liquidity": {
        "mechanism": "Price impact relative to traded value measures scarce liquidity provision.",
        "forcedTrader": "liquidity demanders and inventory-limited market makers",
        "persistence": "Dealer inventory and creation-redemption capacity normalize gradually.",
    },
    "order_imbalance_proxy": {
        "mechanism": "Signed candle pressure times volume proxies persistent directional demand.",
        "forcedTrader": "execution algorithms splitting parent orders",
        "persistence": "Large parent orders are intentionally distributed through time.",
    },
    "price_volume_coherence": {
        "mechanism": "Return-volume coherence separates informed moves from low-conviction noise.",
        "forcedTrader": "informed traders and passive liquidity suppliers",
        "persistence": "Information is incorporated progressively under limited depth.",
    },
    "volume_structure": {
        "mechanism": "Abnormal volume identifies changes in participation and information arrival.",
        "forcedTrader": "institutional execution and event-driven participants",
        "persistence": "Participation changes persist while an event is being processed.",
    },
    "daily_trend": {
        "mechanism": "Underreaction and order splitting can make recent direction persist.",
        "forcedTrader": "slow allocators and benchmark-sensitive portfolios",
        "persistence": "Capital is deployed over multiple sessions.",
    },
    "reversal": {
        "mechanism": "Temporary liquidity shocks can overshoot fundamental value and mean revert.",
        "forcedTrader": "urgent liquidity demanders",
        "persistence": "The discount closes as liquidity providers rebuild inventory.",
    },
    "range_volatility": {
        "mechanism": "Intraday range reveals disagreement and the cost of immediacy.",
        "forcedTrader": "short-horizon liquidity takers",
        "persistence": "Disagreement and inventory risk cluster in time.",
    },
    "lag_response": {
        "mechanism": "Delayed response to past returns proxies gradual information diffusion.",
        "forcedTrader": "attention-constrained and rule-based investors",
        "persistence": "Different participants react at different speeds.",
    },
    "volatility_asymmetry": {
        "mechanism": "Downside volatility has a different risk-budget impact than upside volatility.",
        "forcedTrader": "loss-sensitive and volatility-controlled portfolios",
        "persistence": "Asymmetric risk limits create serially correlated flows.",
    },
    "drawdown": {
        "mechanism": "Drawdown-recovery geometry separates exhausted selling from continuing stress.",
        "forcedTrader": "drawdown-limited holders and contrarian liquidity providers",
        "persistence": "Recovery depends on gradual inventory transfer.",
    },
    "fractal_proxy": {
        "mechanism": "Cross-scale volatility ratios proxy changes in persistence and market roughness.",
        "forcedTrader": "participants operating at heterogeneous horizons",
        "persistence": "Horizon mismatch creates multi-scale dependence.",
    },
    "regime_gating": {
        "mechanism": "The payoff to direction changes with trend stability and volatility regime.",
        "forcedTrader": "trend followers and mean-reversion liquidity providers",
        "persistence": "Strategy populations adjust only after regime evidence accumulates.",
    },
    "stability": {
        "mechanism": "Stable price formation makes a directional signal more reliable than noisy jumps.",
        "forcedTrader": "risk-sensitive systematic allocators",
        "persistence": "Stable execution conditions persist across adjacent sessions.",
    },
    "bar_shape": {
        "mechanism": "Candle geometry summarizes the location and intensity of intraday pressure.",
        "forcedTrader": "close-sensitive execution and auction participants",
        "persistence": "Unfinished imbalances can carry to the next session.",
    },
    "creative": {
        "mechanism": "Bounded combinations may expose nonlinear interactions missed by single factors.",
        "forcedTrader": "heterogeneous systematic strategies",
        "persistence": "Different constraints interact nonlinearly across regimes.",
    },
    "composite": {
        "mechanism": "Independent price, volume and risk evidence can improve signal precision.",
        "forcedTrader": "multiple constrained participant groups",
        "persistence": "Overlapping slow mechanisms reinforce one another.",
    },
    "herding_proxy": {
        "mechanism": "Aligned price and volume pressure proxies crowded participation.",
        "forcedTrader": "crowded systematic and thematic flows",
        "persistence": "Crowds build gradually and unwind under common constraints.",
    },
}


@dataclass
class Split:
    train: pd.Series
    validation: pd.Series
    shadow: pd.Series
    audit: dict[str, Any]


@dataclass
class Evaluation:
    candidate: dict[str, Any]
    signal: pd.DataFrame
    ic: pd.Series
    rank_ic: pd.Series
    long_net: pd.Series
    hit: pd.Series
    mutual_information: dict[str, float | None]
    summary: dict[str, Any]
    fitness: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def expression_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "generator": {
            "rawInputs": config["search"]["allowedInputs"],
            "allowedWindows": config["search"]["allowedWindows"],
            "maximumExpressionDepth": config["search"]["maximumExpressionDepth"],
        }
    }


def all_agents(config: dict[str, Any]) -> list[str]:
    return [agent for agents in config["search"]["agentHierarchy"].values() for agent in agents]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schemaVersion") != "cogalpha_autonomous_research_v1":
        raise ValueError("unexpected autonomous CogAlpha schema")
    safety = config.get("safety", {})
    if any(value is not False for key, value in safety.items() if key.startswith("may")):
        raise ValueError("all research mutation permissions must remain false")
    if safety.get("outputStatus") != "diagnostic_only":
        raise ValueError("output status must remain diagnostic_only")
    if safety.get("allowedOutputRoot") != "outputs/edge_research/cogalpha_autonomous":
        raise ValueError("autonomous output root must remain isolated")
    data = config["data"]
    horizon = int(data["predictionHorizonTradingDays"])
    if int(data["purgeTradingDays"]) < horizon:
        raise ValueError("purge must be at least the maximum label horizon")
    if int(data["validationTradingDays"]) < 60 or int(data["shadowQuarantineTradingDays"]) < 60:
        raise ValueError("validation and shadow quarantine are too short")
    search = config["search"]
    if int(search["maximumEvaluatedCandidatesPerRun"]) > 512:
        raise ValueError("per-run search budget is not bounded")
    if set(search["guidanceModes"]) != {"light", "moderate", "creative", "divergent", "concrete"}:
        raise ValueError("the five preregistered guidance modes must remain fixed")
    if set(all_agents(config)) != set(ROLE_HYPOTHESES):
        raise ValueError("agent hierarchy and hypothesis registry differ")
    fitness = config["fitness"]
    if (
        fitness["feedbackData"] != "train_only"
        or fitness["validationFeedbackAllowed"] is not False
        or fitness["shadowFeedbackAllowed"] is not False
        or fitness["previousRunValidationFeedbackAllowed"] is not False
        or fitness["previousRunShadowFeedbackAllowed"] is not False
    ):
        raise ValueError("adaptive feedback must remain train-only")
    promotion = config["promotion"]
    if promotion["historicalOrAutonomousRunCanPromote"] is not False:
        raise ValueError("autonomous historical research cannot promote")
    if promotion["mayConnectToTradingAutomatically"] is not False:
        raise ValueError("trading connection must remain disabled")


def make_split(index: pd.DatetimeIndex, config: dict[str, Any]) -> Split:
    dates = pd.DatetimeIndex(sorted(pd.unique(index)))
    data = config["data"]
    validation_days = int(data["validationTradingDays"])
    shadow_days = int(data["shadowQuarantineTradingDays"])
    purge_days = int(data["purgeTradingDays"])
    minimum_train = int(data["minimumTrainingTradingDays"])
    required = minimum_train + validation_days + shadow_days + 2 * purge_days
    if len(dates) < required:
        raise RuntimeError(f"need at least {required} trading dates, found {len(dates)}")
    shadow_start_pos = len(dates) - shadow_days
    validation_end_pos = shadow_start_pos - purge_days - 1
    validation_start_pos = validation_end_pos - validation_days + 1
    train_end_pos = validation_start_pos - purge_days - 1
    train_end = dates[train_end_pos]
    validation_start = dates[validation_start_pos]
    validation_end = dates[validation_end_pos]
    shadow_start = dates[shadow_start_pos]
    as_series = pd.Series(index=index, dtype=bool)
    train = pd.Series(index <= train_end, index=index)
    validation = pd.Series((index >= validation_start) & (index <= validation_end), index=index)
    shadow = pd.Series(index >= shadow_start, index=index)
    as_series.loc[:] = train | validation | shadow
    if bool((train & validation).any() or (train & shadow).any() or (validation & shadow).any()):
        raise AssertionError("split overlap")
    return Split(
        train=train,
        validation=validation,
        shadow=shadow,
        audit={
            "train": [dates[0].date().isoformat(), train_end.date().isoformat(), int(train.sum())],
            "validation": [validation_start.date().isoformat(), validation_end.date().isoformat(), int(validation.sum())],
            "shadowQuarantine": [shadow_start.date().isoformat(), dates[-1].date().isoformat(), int(shadow.sum())],
            "purgeTradingDays": purge_days,
            "labelHorizonTradingDays": int(data["predictionHorizonTradingDays"]),
        },
    )


def candidate_record(
    agent: str,
    expression: dict[str, Any],
    generation: int,
    parents: list[str],
    guidance: str,
    source: str,
    rationale: str = "",
    hypothesis: dict[str, str] | None = None,
) -> dict[str, Any]:
    fingerprint = core.digest(expression)
    return {
        "id": f"g{generation:02d}_{agent}_{fingerprint[:10]}",
        "fingerprint": fingerprint,
        "agent": agent,
        "generation": generation,
        "parents": parents,
        "guidance": guidance,
        "source": source,
        "hypothesis": copy.deepcopy(hypothesis or ROLE_HYPOTHESES[agent]),
        "rationale": rationale or ROLE_HYPOTHESES[agent]["mechanism"],
        "expression": expression,
    }


def f(name: str) -> dict[str, Any]:
    return {"field": name}


def roll(op: str, arg: dict[str, Any], window: int) -> dict[str, Any]:
    return {"rolling": op, "arg": arg, "window": window}


def binary(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"binary": op, "left": left, "right": right}


def unary(op: str, arg: dict[str, Any]) -> dict[str, Any]:
    return {"unary": op, "arg": arg}


def random_expression(agent: str, rng: random.Random, config: dict[str, Any]) -> dict[str, Any]:
    windows = list(map(int, config["search"]["allowedWindows"]))
    short = rng.choice(windows[:5])
    long = rng.choice([window for window in windows if window >= max(20, short)])
    returns = f("returns")
    close = f("close")
    volume = unary("signed_log1p", f("volume"))
    amount = unary("signed_log1p", f("amount"))
    candle_range = binary("div", binary("sub", f("high"), f("low")), close)
    body = binary("div", binary("sub", f("close"), f("open")), binary("sub", f("high"), f("low")))
    templates: dict[str, list[dict[str, Any]]] = {
        "market_cycle": [binary("div", roll("mean", close, short), roll("mean", close, long)), binary("sub", roll("sum", returns, long), roll("sum", returns, short))],
        "volatility_regime": [binary("div", roll("std", returns, short), roll("std", returns, long)), {"zscore": True, "arg": roll("std", returns, short), "window": long}],
        "tail_risk": [unary("neg", roll("std", binary("sub", returns, unary("abs", returns)), long)), roll("min", returns, long)],
        "crash_predictor": [{"drawdown": True, "arg": close, "window": long}, binary("mul", {"drawdown": True, "arg": close, "window": long}, roll("std", returns, short))],
        "liquidity": [binary("div", roll("mean", candle_range, short), roll("mean", amount, long)), binary("div", unary("abs", returns), amount)],
        "order_imbalance_proxy": [roll("mean", binary("mul", body, volume), short), binary("mul", body, {"zscore": True, "arg": volume, "window": long})],
        "price_volume_coherence": [{"corr": True, "left": returns, "right": volume, "window": long}, binary("mul", roll("sum", returns, short), {"zscore": True, "arg": volume, "window": long})],
        "volume_structure": [{"zscore": True, "arg": volume, "window": long}, binary("div", roll("mean", volume, short), roll("mean", volume, long))],
        "daily_trend": [roll("sum", returns, short), binary("div", roll("mean", close, short), roll("mean", close, long))],
        "reversal": [unary("neg", roll("sum", returns, short)), unary("neg", {"range_position": True, "arg": close, "window": long})],
        "range_volatility": [roll("mean", candle_range, short), binary("div", roll("std", candle_range, short), roll("std", candle_range, long))],
        "lag_response": [{"lag": short, "arg": returns}, {"corr": True, "left": returns, "right": {"lag": short, "arg": returns}, "window": long}],
        "volatility_asymmetry": [binary("div", roll("std", binary("sub", returns, unary("abs", returns)), long), roll("std", returns, long)), binary("sub", roll("std", returns, short), roll("std", unary("abs", returns), short))],
        "drawdown": [binary("add", {"drawdown": True, "arg": close, "window": long}, roll("sum", returns, short)), unary("neg", roll("std", {"drawdown": True, "arg": close, "window": long}, short))],
        "fractal_proxy": [binary("div", roll("std", returns, short), roll("std", returns, long)), binary("div", roll("sum", unary("abs", returns), short), unary("abs", roll("sum", returns, short)))],
        "regime_gating": [binary("div", roll("mean", returns, long), roll("std", returns, long)), binary("mul", roll("sum", returns, short), unary("neg", roll("std", returns, long)))],
        "stability": [unary("neg", roll("std", candle_range, long)), binary("div", unary("abs", roll("mean", returns, long)), roll("std", returns, long))],
        "bar_shape": [roll("mean", body, short), binary("mul", body, {"range_position": True, "arg": close, "window": long})],
        "creative": [unary("tanh", binary("mul", {"range_position": True, "arg": close, "window": long}, {"zscore": True, "arg": volume, "window": long})), unary("tanh", binary("div", roll("sum", returns, short), roll("mean", candle_range, long)))],
        "composite": [binary("mul", roll("sum", returns, short), {"corr": True, "left": returns, "right": volume, "window": long}), binary("mul", {"range_position": True, "arg": close, "window": long}, unary("neg", roll("std", candle_range, short)))],
        "herding_proxy": [binary("mul", roll("mean", body, short), {"zscore": True, "arg": volume, "window": long}), binary("mul", {"corr": True, "left": returns, "right": volume, "window": long}, roll("sum", returns, short))],
    }
    expression = copy.deepcopy(rng.choice(templates[agent]))
    if rng.random() < 0.35:
        expression = unary(rng.choice(["tanh", "signed_log1p"]), expression)
    return expression


def mutate_expression(expr: dict[str, Any], agent: str, rng: random.Random, config: dict[str, Any]) -> dict[str, Any]:
    choice = rng.randrange(5)
    if choice == 0:
        changed, ok = core.replace_first_window(expr, rng.choice(config["search"]["allowedWindows"]))
        return changed if ok else unary("tanh", copy.deepcopy(expr))
    if choice == 1:
        return unary(rng.choice(["tanh", "signed_log1p", "neg"]), copy.deepcopy(expr))
    if choice == 2:
        return {"zscore": True, "arg": copy.deepcopy(expr), "window": rng.choice(config["search"]["allowedWindows"])}
    other = random_expression(agent, rng, config)
    return binary(rng.choice(["add", "sub", "mul", "div"]), copy.deepcopy(expr), other)


def grammar_initial(config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(int(config["search"]["randomSeed"]))
    agents = all_agents(config)
    guidance = config["search"]["guidanceModes"]
    output = []
    for index in range(int(config["search"]["initialGrammarCandidates"])):
        agent = agents[index % len(agents)]
        output.append(candidate_record(agent, random_expression(agent, rng, config), 0, [], guidance[index % len(guidance)], "local_grammar"))
    return output


def grammar_children(parents: list[Evaluation], generation: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(int(config["search"]["randomSeed"]) + 1009 * generation)
    output = []
    count = int(config["search"]["childrenPerParent"])
    guidance = config["search"]["guidanceModes"]
    for index, parent in enumerate(parents):
        for child_index in range(count):
            if child_index == 1 and len(parents) > 1:
                mate = parents[(index + 1 + rng.randrange(len(parents) - 1)) % len(parents)]
                expression = binary(rng.choice(["add", "sub", "mul"]), copy.deepcopy(parent.candidate["expression"]), copy.deepcopy(mate.candidate["expression"]))
                parent_ids = [parent.candidate["id"], mate.candidate["id"]]
                agent = "composite"
            elif child_index == 2:
                agent = parent.candidate["agent"]
                expression = random_expression(agent, rng, config)
                parent_ids = [parent.candidate["id"]]
            else:
                agent = parent.candidate["agent"]
                expression = mutate_expression(parent.candidate["expression"], agent, rng, config)
                parent_ids = [parent.candidate["id"]]
            output.append(candidate_record(agent, expression, generation, parent_ids, guidance[(generation + index + child_index) % len(guidance)], "train_feedback_grammar"))
    return output


def parse_llm_candidates(raw: str, generation: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        raise ValueError("local model response contains no JSON array")
    rows = json.loads(match.group(0))
    allowed = set(all_agents(config))
    maximum = int(config["search"]["ollama"]["maximumCandidatesPerGeneration"])
    output = []
    for row in rows[:maximum]:
        if not isinstance(row, dict) or row.get("agent") not in allowed:
            continue
        hypothesis = row.get("hypothesis") or {}
        if not all(str(hypothesis.get(key, "")).strip() for key in ("mechanism", "forcedTrader", "persistence")):
            continue
        core.validate_expression(row.get("expression"), expression_config(config))
        output.append(candidate_record(
            row["agent"], row["expression"], generation, [str(value) for value in row.get("parents", [])],
            str(row.get("guidance", "creative")), "local_ollama", str(row.get("rationale", "")), hypothesis,
        ))
    return output


def ollama_generate(generation: int, feedback: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = config["search"]["ollama"]
    model = os.environ.get(str(provider["modelEnvironmentVariable"]), str(provider.get("defaultModel", ""))).strip()
    if not provider.get("enabled"):
        return [], {"status": "disabled"}
    if not model:
        return [], {"status": "skipped_no_local_model", "environmentVariable": provider["modelEnvironmentVariable"]}
    rng = random.Random(int(config["search"]["randomSeed"]) + generation)
    prompt = {
        "task": "Generate novel, economically interpretable, strictly past-only A-share ETF alpha programs.",
        "generation": generation,
        "output": "JSON array only; each item has agent, hypothesis{mechanism,forcedTrader,persistence}, rationale, guidance, parents, expression.",
        "allowedAgents": all_agents(config),
        "guidanceModes": config["search"]["guidanceModes"],
        "dsl": {
            "inputs": config["search"]["allowedInputs"],
            "windows": config["search"]["allowedWindows"],
            "nodeKinds": ["field", "unary", "binary", "rolling", "lag", "corr", "zscore", "drawdown", "range_position"],
            "futureData": "forbidden; lag must be nonnegative",
        },
        "trainOnlyFeedback": feedback,
        "forbidden": ["validation metrics", "shadow metrics", "future returns in features", "trading orders", "file or network access"],
    }
    body = json.dumps({
        "model": model,
        "prompt": core.canonical(prompt),
        "stream": False,
        "options": {"temperature": rng.choice(provider["temperatureChoices"])},
    }).encode("utf-8")
    request = urllib.request.Request(provider["baseUrl"].rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=int(provider["timeoutSeconds"])) as response:
            payload = json.loads(response.read().decode("utf-8"))
        candidates = parse_llm_candidates(str(payload.get("response", "")), generation, config)
        return candidates, {"status": "ok", "model": model, "accepted": len(candidates)}
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return [], {"status": "failed_closed", "model": model, "reason": str(exc)[:300]}


def period_stats(series: pd.Series, mask: pd.Series) -> dict[str, Any]:
    values = series[mask.reindex(series.index, fill_value=False)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return {"n": int(len(values)), "mean": None, "t": None, "irAnn": None}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return {
        "n": int(len(values)),
        "mean": round(mean, 8),
        "t": round(mean / std * math.sqrt(len(values)), 4) if std > 0 else None,
        "irAnn": round(mean / std * math.sqrt(244), 4) if std > 0 else None,
    }


def discrete_mutual_information(signal: pd.DataFrame, target: pd.DataFrame, mask: pd.Series) -> float | None:
    x = signal.rank(axis=1, pct=True).loc[mask].stack().rename("x")
    y = target.rank(axis=1, pct=True).loc[mask].stack().rename("y")
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 1000:
        return None
    x_bin = np.minimum((joined["x"].to_numpy() * 10).astype(int), 9)
    y_bin = np.minimum((joined["y"].to_numpy() * 10).astype(int), 9)
    return round(float(mutual_info_score(x_bin, y_bin) / math.log(10)), 8)


def tradability_frames(panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(buyable, sellable) per bar under A-share microstructure.

    A bar that never leaves a single price (high == low) is a sealed session: on the A-share
    boards that is a locked price limit (10% main / 20% STAR-ChiNext / 30% BJ), so it is
    board-agnostic and needs no per-board limit arithmetic. Sealed-up cannot be bought into
    (the queue never fills); sealed-down cannot be sold out of. A bar with no volume is a
    halt. Both stay in the panel as observations -- only the ability to TRANSACT is denied,
    which is what the label must respect.
    """
    high, low, close = panel["high"], panel["low"], panel["close"]
    volume = panel.get("volume")
    previous_close = close.shift(1)
    sealed = (high == low) & high.notna() & low.notna()
    halted = volume.fillna(0.0).le(0.0) if volume is not None else False
    buyable = ~((sealed & close.gt(previous_close)) | halted)
    sellable = ~((sealed & close.lt(previous_close)) | halted)
    return buyable, sellable


def size_neutralise(signal: pd.DataFrame, panel: dict[str, pd.DataFrame], bins: int) -> pd.DataFrame:
    """Standardise the signal inside trailing-liquidity buckets.

    An un-neutralised A-share cross-section is largely a size bet, and the size bet is what
    the cost model then destroys. Bucketing on trailing median amount (causal, shifted) and
    z-scoring within bucket keeps the intra-bucket ordering a factor can actually claim.
    """
    if bins < 2 or "amount" not in panel:
        return signal
    scale = np.log(panel["amount"].rolling(20).median().shift(1).replace(0.0, np.nan))
    buckets = scale.rank(axis=1, pct=True)
    out = signal * np.nan
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        member = buckets.le(upper) if index == 0 else (buckets.gt(lower) & buckets.le(upper))
        group = signal.where(member)
        centred = group.sub(group.mean(axis=1), axis=0).div(
            group.std(axis=1).replace(0.0, np.nan), axis=0
        )
        out = out.fillna(centred)
    return out


def long_only_portfolio(
    signal: pd.DataFrame,
    one_day: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """THE book definition: (net, turnover, weights, book_return, benchmark).

    Every stage -- fast screen, purged walk-forward, Primary/Counter/Placebo, DSR/PBO -- must
    price the same strategy, otherwise a factor is screened as one portfolio and validated as
    a different one. Construction knobs live in config["data"] and DEFAULT TO THE LEGACY
    behaviour (equal-weight top decile, daily rebalance, no neutralisation) so no pipeline
    silently changes; the A-share config opts in explicitly.
    """
    data = config["data"]
    if bool(data.get("sizeNeutralise", False)):
        signal = size_neutralise(signal, panel, int(data.get("sizeNeutraliseBins", 5)))
    ranks = signal.rank(axis=1, pct=True)
    quantile = float(data["topQuantile"])
    if str(data.get("bookConstruction", "top_decile")) == "rank_weighted":
        raw_weights = ranks.sub(1.0 - quantile).clip(lower=0.0)
    else:
        raw_weights = ranks.ge(1.0 - quantile).astype(float)
    weights = raw_weights.div(raw_weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    hold = int(data.get("predictionHorizonTradingDays", 1)) if bool(
        data.get("holdForPredictionHorizon", False)
    ) else 1
    if hold > 1:
        weights = weights.rolling(hold).mean().fillna(0.0)
        weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book_return = (weights * one_day).sum(axis=1, min_count=1)
    benchmark = one_day.mean(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    net = book_return - benchmark - turnover * float(data["roundTripCost"])
    return net, turnover, weights, book_return, benchmark


def target_frames(panel: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Labels are entered at the next open and exited at the horizon open.

    Untradeable legs are dropped (NaN) rather than priced: a limit-locked or halted entry
    could not have been bought and a limit-locked exit could not have been sold, so keeping
    those samples would credit the factor with returns no account could have realised. This
    is the classic A-share backtest inflation channel for momentum-family signals.
    """
    horizon = int(config["data"]["predictionHorizonTradingDays"])
    open_price = panel["open"]
    target = open_price.shift(-(horizon + 1)) / open_price.shift(-1) - 1.0
    one_day = open_price.shift(-2) / open_price.shift(-1) - 1.0
    buyable, sellable = tradability_frames(panel)
    entry_ok = buyable.shift(-1)
    target = target.where(entry_ok & sellable.shift(-(horizon + 1)))
    one_day = one_day.where(entry_ok & sellable.shift(-2))
    return target, one_day


def evaluate_candidate(
    candidate: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    one_day: pd.DataFrame,
    split: Split,
    config: dict[str, Any],
) -> tuple[Evaluation | None, str | None]:
    try:
        if candidate["agent"] not in ROLE_HYPOTHESES:
            return None, "unknown_agent"
        hypothesis = candidate.get("hypothesis") or {}
        if not all(str(hypothesis.get(key, "")).strip() for key in ("mechanism", "forcedTrader", "persistence")):
            return None, "missing_economic_hypothesis"
        core.validate_expression(candidate["expression"], expression_config(config))
        raw = core.evaluate_expression(candidate["expression"], panel).replace([np.inf, -np.inf], np.nan)
    except Exception as exc:
        return None, f"invalid_expression:{type(exc).__name__}"
    eligible = panel["close"].notna() & target.notna()
    train_eligible = eligible.loc[split.train]
    denominator = int(train_eligible.to_numpy().sum())
    available = int(raw.loc[split.train].where(train_eligible).notna().to_numpy().sum())
    nan_fraction = 1.0 - available / max(1, denominator)
    if nan_fraction > float(config["search"]["maximumNanFraction"]):
        return None, "too_many_nan"
    distinct = raw.loc[split.train].where(train_eligible).nunique(axis=1)
    if distinct.empty or float(distinct.median()) < float(config["search"]["minimumMedianCrossSectionDistinct"]):
        return None, "insufficient_cross_section_variation"
    raw_train_rank_ic = raw.loc[split.train].corrwith(target.loc[split.train], axis=1, method="spearman").dropna()
    if len(raw_train_rank_ic) < 120:
        return None, "insufficient_train_ic_days"
    direction = 1.0 if float(raw_train_rank_ic.mean()) >= 0 else -1.0
    signal = raw * direction
    ic = signal.corrwith(target, axis=1, method="pearson").dropna()
    rank_ic = signal.corrwith(target, axis=1, method="spearman").dropna()
    long_net, turnover, weights, top_return, benchmark = long_only_portfolio(
        signal, one_day, panel, config
    )
    hit = (top_return > benchmark).astype(float)
    masks = {"train": split.train, "validation": split.validation, "shadow": split.shadow}
    mi = {period: discrete_mutual_information(signal, target, mask) for period, mask in masks.items()}
    periods = {}
    for period, mask in masks.items():
        periods[period] = {
            "ic": period_stats(ic, mask),
            "rankIc": period_stats(rank_ic, mask),
            "costedLongOnly": period_stats(long_net, mask),
            "hitRate": period_stats(hit, mask),
            "mutualInformation": mi[period],
        }
    summary = {
        "id": candidate["id"],
        "fingerprint": candidate["fingerprint"],
        "agent": candidate["agent"],
        "generation": candidate["generation"],
        "parents": candidate["parents"],
        "guidance": candidate["guidance"],
        "source": candidate["source"],
        "directionFromTrain": direction,
        "expressionDepth": core.expression_depth(candidate["expression"]),
        "nanFractionTrain": round(nan_fraction, 6),
        "periods": periods,
    }
    weights_cfg = config["fitness"]["weights"]
    train = periods["train"]
    rank_ir = train["rankIc"]["irAnn"] or -99.0
    ic_ir = train["ic"]["irAnn"] or -99.0
    net_ir = train["costedLongOnly"]["irAnn"] or -99.0
    mutual = train["mutualInformation"] or 0.0
    fitness = (
        float(weights_cfg["rankIcIr"]) * float(rank_ir)
        + float(weights_cfg["icIr"]) * float(ic_ir)
        + float(weights_cfg["mutualInformation"]) * 100.0 * float(mutual)
        + float(weights_cfg["costedLongOnlyIr"]) * max(-3.0, float(net_ir))
        - float(weights_cfg["complexityPenaltyPerDepth"]) * float(summary["expressionDepth"])
    )
    summary["trainFitness"] = round(fitness, 8)
    return Evaluation(candidate, signal, ic, rank_ic, long_net, hit, mi, summary, fitness), None


def choose_parents(evaluated: list[Evaluation], config: dict[str, Any]) -> tuple[list[Evaluation], dict[str, Any]]:
    ranked = sorted(evaluated, key=lambda item: item.fitness, reverse=True)
    if not ranked:
        return [], {"qualified": 0, "fallback": True}
    percentile = float(config["fitness"]["qualifiedPercentile"]) / 100.0
    keys = [
        ("rankIc", "mean"), ("rankIc", "irAnn"), ("ic", "mean"), ("ic", "irAnn")
    ]
    cutoffs = {}
    for metric, field in keys:
        values = [item.summary["periods"]["train"][metric].get(field) for item in ranked]
        clean = [float(value) for value in values if value is not None]
        cutoffs[f"{metric}.{field}"] = float(np.quantile(clean, percentile)) if clean else math.inf
    mi_values = [item.summary["periods"]["train"]["mutualInformation"] for item in ranked]
    mi_clean = [float(value) for value in mi_values if value is not None]
    cutoffs["mutualInformation"] = float(np.quantile(mi_clean, percentile)) if mi_clean else math.inf
    qualified = []
    for item in ranked:
        train = item.summary["periods"]["train"]
        if (
            (train["rankIc"]["mean"] or -99.0) >= max(float(config["fitness"]["minimumAbsoluteTrainRankIc"]), cutoffs["rankIc.mean"])
            and (train["rankIc"]["irAnn"] or -99.0) >= max(float(config["fitness"]["minimumTrainRankIcIr"]), cutoffs["rankIc.irAnn"])
            and (train["ic"]["mean"] or -99.0) >= cutoffs["ic.mean"]
            and (train["ic"]["irAnn"] or -99.0) >= cutoffs["ic.irAnn"]
            and (train["mutualInformation"] or -99.0) >= max(float(config["fitness"]["minimumTrainMutualInformation"]), cutoffs["mutualInformation"])
        ):
            qualified.append(item)
    pool_size = int(config["search"]["parentPoolSize"])
    selected = qualified[:pool_size]
    fallback = len(selected) < min(4, pool_size)
    if fallback:
        used = {item.candidate["fingerprint"] for item in selected}
        selected.extend(item for item in ranked if item.candidate["fingerprint"] not in used)
    return selected[:pool_size], {
        "qualified": len(qualified),
        "fallback": fallback,
        "cutoffs": {key: round(value, 8) if math.isfinite(value) else None for key, value in cutoffs.items()},
    }


def train_feedback(evaluated: list[Evaluation]) -> dict[str, Any]:
    ranked = sorted(evaluated, key=lambda item: item.fitness, reverse=True)
    def row(item: Evaluation) -> dict[str, Any]:
        return {
            "id": item.candidate["id"],
            "agent": item.candidate["agent"],
            "hypothesis": item.candidate["hypothesis"],
            "expression": item.candidate["expression"],
            "trainMetrics": item.summary["periods"]["train"],
            "trainFitness": item.summary["trainFitness"],
        }
    return {"bestValid": [row(item) for item in ranked[:2]], "worstValid": [row(item) for item in ranked[-2:]]}


def fit_ridge_ensemble(
    elites: list[Evaluation], target: pd.DataFrame, split: Split, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    maximum = int(config["ensemble"]["maximumFactors"])
    chosen = sorted(elites, key=lambda item: item.fitness, reverse=True)[:maximum]
    if not chosen:
        raise RuntimeError("no train elites for ensemble")
    fill = float(config["ensemble"]["missingRankFill"])
    features = [item.signal.rank(axis=1, pct=True).fillna(fill) for item in chosen]
    train_arrays = [frame.loc[split.train].to_numpy(dtype=float) for frame in features]
    x_train = np.stack(train_arrays, axis=2).reshape(-1, len(features))
    y_train = target.loc[split.train].to_numpy(dtype=float).reshape(-1)
    valid = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    x_train = x_train[valid]
    y_train = y_train[valid]
    maximum_rows = int(config["ensemble"]["maximumTrainingRows"])
    if len(y_train) > maximum_rows:
        rng = np.random.default_rng(int(config["search"]["randomSeed"]))
        selected = np.sort(rng.choice(len(y_train), size=maximum_rows, replace=False))
        x_train = x_train[selected]
        y_train = y_train[selected]
    model = Ridge(alpha=float(config["ensemble"]["ridgeAlpha"]), fit_intercept=True)
    model.fit(x_train, y_train)
    all_x = np.stack([frame.to_numpy(dtype=float) for frame in features], axis=2)
    predictions = model.predict(all_x.reshape(-1, len(features))).reshape(features[0].shape)
    signal = pd.DataFrame(predictions, index=features[0].index, columns=features[0].columns)
    return signal, {
        "method": "ridge_train_only",
        "parents": [item.candidate["id"] for item in chosen],
        "parentFingerprints": [item.candidate["fingerprint"] for item in chosen],
        "coefficients": [round(float(value), 10) for value in model.coef_],
        "intercept": round(float(model.intercept_), 10),
        "trainingRows": int(len(y_train)),
        "usedValidationForFit": False,
        "usedShadowForFit": False,
    }


def fixed_baselines(panel: dict[str, pd.DataFrame], target: pd.DataFrame, one_day: pd.DataFrame, split: Split, config: dict[str, Any]) -> list[Evaluation]:
    output = []
    for zoo, name in (("qlib158", "cntp30"), ("qlib158", "cntd30"), ("academic", "high52w")):
        try:
            module = importlib.import_module(f"src.factors.zoo.{zoo}.{name}")
            expression = {"field": "close"}
            candidate = candidate_record("composite", expression, -1, [], "concrete", "fixed_baseline", rationale=f"Fixed baseline {zoo}/{name}")
            candidate["id"] = f"baseline_{zoo}_{name}"
            candidate["fingerprint"] = core.digest({"zoo": zoo, "name": name})
            signal = module.compute(panel)
            original = core.evaluate_expression
            try:
                core.evaluate_expression = lambda _expr, _panel: signal
                item, _ = evaluate_candidate(candidate, panel, target, one_day, split, config)
            finally:
                core.evaluate_expression = original
            if item:
                output.append(item)
        except Exception:
            continue
    return output


def load_persistent_parents(state_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = state_root / "parent_pool.json"
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except (ValueError, OSError):
        return []
    output = []
    for row in payload.get("parents", [])[: int(config["search"]["maximumPersistentParents"])]:
        try:
            core.validate_expression(row["expression"], expression_config(config))
            output.append(candidate_record(
                row["agent"], row["expression"], 0, [row.get("id", "previous_train_parent")],
                row.get("guidance", "moderate"), "previous_train_parent", row.get("rationale", ""), row.get("hypothesis"),
            ))
        except Exception:
            continue
    return output


def latest_shadow_ranking(signal: pd.DataFrame, config: dict[str, Any], run_id: str) -> dict[str, Any]:
    row = signal.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    count = int(config["data"]["latestShadowRankingCount"])
    ranked = [
        {"rank": index, "stockCode": str(code)[-6:], "researchScore": round(float(value), 10)}
        for index, (code, value) in enumerate(row.head(count).items(), start=1)
    ]
    return {
        "schemaVersion": "cogalpha_autonomous_shadow_strategy_v1",
        "status": "research_only_not_a_trade_signal",
        "runId": run_id,
        "asOfDate": signal.index[-1].date().isoformat(),
        "strategy": {
            "signalAt": "day_t_close",
            "hypotheticalEntry": "day_t_plus_1_open",
            "holdingHorizonTradingDays": int(config["data"]["predictionHorizonTradingDays"]),
            "portfolio": "daily top-decile research tranche",
            "roundTripCost": float(config["data"]["roundTripCost"]),
        },
        "rankedEtfs": ranked,
        "orders": [],
        "automaticTradingChanges": [],
        "warning": "Observation only. It is neither a BUY instruction nor connected to the paper agent.",
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Autonomous CogAlpha ETF Research", "",
        f"- Status: `{result['status']}`",
        f"- Run ID: `{result['runId']}`",
        f"- Data: `{result['dataAudit']['start']}..{result['dataAudit']['end']}` | {result['dataAudit']['symbols']} ETFs",
        f"- Split: `{result['splitAudit']}`",
        f"- Candidates generated/evaluated/rejected: `{result['candidateAudit']['generated']}` / `{result['candidateAudit']['evaluated']}` / `{result['candidateAudit']['rejected']}`",
        f"- Generator: `{result['candidateAudit']['provider']}`", "",
        "This loop discovers factor programs and persists train-only research memory. It cannot place orders or modify any trading decision, position, risk gate or overlay.", "",
        "## Frozen train-only ensemble", "",
        f"Parents: `{result['ensemble']['parents']}`", "",
        "| model | period | RankIC mean | RankIC IR | costed net IR | net mean/day | hit rate | MI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparison"]:
        for period in ("train", "validation", "shadow"):
            metrics = row["periods"][period]
            lines.append(
                f"| {row['id']} | {period} | {metrics['rankIc']['mean']} | {metrics['rankIc']['irAnn']} | "
                f"{metrics['costedLongOnly']['irAnn']} | {metrics['costedLongOnly']['mean']} | "
                f"{metrics['hitRate']['mean']} | {metrics['mutualInformation']} |"
            )
    lines += [
        "", "## Guards", "",
        f"- Validation gate passed: `{result['validation']['gatePassed']}`",
        f"- Candidate PBO: `{result['guards']['pbo']}`",
        f"- Quick DSR warning: `{result['guards']['quickDsr']}`",
        f"- Total recorded CogAlpha-family trials: `{result['guards']['totalTrials']}`",
        f"- Validation/shadow fed back to generator: `false`",
        f"- Automatic promotion: `false`", "",
        "## Verdict", "", result["verdict"], "",
        "## Limitations", "",
        f"- {result['limitations']['survivorshipWarning']}",
        "- Existing historical windows have already been inspected by prior research and are not pristine final OOS.",
        "- The built-in grammar is autonomous but is not equivalent to the paper's gpt-oss-120B semantic reasoning. Ollama augmentation remains optional and local-only.",
        "- Daily cross-sectional alpha does not establish minute-level T0 execution edge; a separate causal replay would be required.", "",
    ]
    return "\n".join(lines)


def run(
    config_path: Path,
    output_root: Path,
    maximum_candidates: int | None = None,
    generations: int | None = None,
    use_state: bool = True,
    force: bool = False,
) -> tuple[dict[str, Any], Path | None]:
    config = load_json(config_path)
    validate_config(config)
    if maximum_candidates is not None:
        config = copy.deepcopy(config)
        config["search"]["maximumEvaluatedCandidatesPerRun"] = min(int(maximum_candidates), int(config["search"]["maximumEvaluatedCandidatesPerRun"]))
    if generations is not None:
        config = copy.deepcopy(config)
        config["search"]["generationsPerRun"] = min(int(generations), int(config["search"]["generationsPerRun"]))
    panel = core.build_panel(config)
    close = panel.get("close")
    if close is None or close.empty:
        raise RuntimeError("no eligible ETF daily panel")
    split = make_split(close.index, config)
    target, one_day = target_frames(panel, config)
    config_input = core.input_fingerprint(config_path, config)
    fingerprint = core.digest({"input": config_input, "version": CODE_VERSION})
    state_root = ROOT / config["continuousIteration"]["stateDirectory"]
    latest_path = state_root / "latest.json"
    if use_state and not force and config["continuousIteration"]["skipWhenDataFingerprintUnchanged"] and latest_path.exists():
        latest = load_json(latest_path)
        if latest.get("inputFingerprint") == fingerprint:
            return {"status": "no_new_data", "runId": latest.get("runId"), "inputFingerprint": fingerprint}, None

    run_id = f"cogalpha_autonomous_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{fingerprint[:10]}"
    initial = []
    for row in core.seed_candidates():
        initial.append(candidate_record(row["agent"], row["expression"], 0, [], "light", "preregistered_seed"))
    initial.extend(grammar_initial(config))
    if use_state:
        initial.extend(load_persistent_parents(state_root, config))
    llm_initial, llm_audit = ollama_generate(0, {"bestValid": [], "worstValid": []}, config)
    initial.extend(llm_initial)

    maximum = int(config["search"]["maximumEvaluatedCandidatesPerRun"])
    seen: set[str] = set()
    evaluated: list[Evaluation] = []
    rejected: dict[str, int] = {}
    provider_audit = [llm_audit]

    def evaluate_rows(rows: list[dict[str, Any]]) -> None:
        for candidate in rows:
            if len(evaluated) >= maximum:
                return
            fingerprint_value = candidate["fingerprint"]
            if fingerprint_value in seen:
                rejected["duplicate_expression"] = rejected.get("duplicate_expression", 0) + 1
                continue
            seen.add(fingerprint_value)
            item, reason = evaluate_candidate(candidate, panel, target, one_day, split, config)
            if item is None:
                rejected[reason or "unknown"] = rejected.get(reason or "unknown", 0) + 1
            else:
                evaluated.append(item)

    evaluate_rows(initial)
    qualification_audit = []
    for generation in range(1, int(config["search"]["generationsPerRun"]) + 1):
        parents, audit = choose_parents(evaluated, config)
        audit["generation"] = generation
        qualification_audit.append(audit)
        if not parents or len(evaluated) >= maximum:
            break
        feedback = train_feedback(evaluated)
        rows = grammar_children(parents, generation, config)
        llm_rows, llm_status = ollama_generate(generation, feedback, config)
        provider_audit.append(llm_status)
        rows.extend(llm_rows)
        evaluate_rows(rows)
    if len(evaluated) < 2:
        raise RuntimeError("autonomous search produced fewer than two valid candidates")

    train_ranked = sorted(evaluated, key=lambda item: item.fitness, reverse=True)
    ensemble_signal, ensemble_meta = fit_ridge_ensemble(train_ranked[: int(config["search"]["elitePoolSize"])], target, split, config)
    ensemble_candidate = candidate_record("composite", {"field": "close"}, 999, ensemble_meta["parents"], "concrete", "ridge_train_only", rationale="Train-only Ridge combination of autonomous alpha programs.")
    ensemble_candidate["id"] = "cogalpha_autonomous_frozen_ensemble"
    original = core.evaluate_expression
    try:
        core.evaluate_expression = lambda _expr, _panel: ensemble_signal
        ensemble_eval, reason = evaluate_candidate(ensemble_candidate, panel, target, one_day, split, config)
    finally:
        core.evaluate_expression = original
    if ensemble_eval is None:
        raise RuntimeError(f"could not evaluate train-only ensemble: {reason}")
    baselines = fixed_baselines(panel, target, one_day, split, config)
    best_baseline_validation = max((item.summary["periods"]["validation"]["costedLongOnly"]["irAnn"] or -99.0 for item in baselines), default=-99.0)
    ensemble_validation = ensemble_eval.summary["periods"]["validation"]["costedLongOnly"]["irAnn"] or -99.0
    validation_gate = (
        ensemble_validation > 0.0
        and ensemble_validation > best_baseline_validation
        and (ensemble_eval.summary["periods"]["validation"]["rankIc"]["mean"] or -99.0) > 0.0
    )

    common = sorted(set.intersection(*(set(item.long_net.loc[split.train].dropna().index) for item in evaluated)))
    matrix = [item.long_net.reindex(common).fillna(0.0).tolist() for item in evaluated]
    pbo = og.combinatorial_symmetric_pbo(matrix, n_blocks=int(config["multipleTesting"]["pboBlocks"])) if common else {"pbo": None}
    previous_trials = 0
    ledger_path = state_root / "trial_ledger.jsonl"
    if use_state and ledger_path.exists():
        previous_trials = sum(1 for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip())
    total_trials = int(config["multipleTesting"]["priorCogAlphaFamilyTrials"]) + previous_trials + len(evaluated)
    shadow_ir = ensemble_eval.summary["periods"]["shadow"]["costedLongOnly"]["irAnn"] or 0.0
    shadow_n = ensemble_eval.summary["periods"]["shadow"]["costedLongOnly"]["n"] or 1
    quick_dsr = og.deflated_significance_note(total_trials, shadow_ir, shadow_n)
    verdict = (
        "The frozen ensemble passed the contaminated historical validation proxy, but it remains a shadow hypothesis. It cannot change trading and must accumulate fresh forward evidence."
        if validation_gate
        else "The train-only ensemble did not beat the fixed validation baselines after costs. The loop may continue searching on future weekly data, but this run supplies no trading edge."
    )
    result = {
        "schemaVersion": "cogalpha_autonomous_result_v1",
        "status": "diagnostic_only_research_only_not_promotable",
        "runId": run_id,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputFingerprint": fingerprint,
        "dataAudit": {
            "start": close.index.min().date().isoformat(),
            "end": close.index.max().date().isoformat(),
            "days": int(len(close.index)),
            "symbols": int(len(close.columns)),
            "barInterval": "1d",
        },
        "splitAudit": split.audit,
        "candidateAudit": {
            "generated": len(seen) + rejected.get("duplicate_expression", 0),
            "evaluated": len(evaluated),
            "rejected": sum(rejected.values()),
            "rejectedReasons": rejected,
            "provider": "local grammar active; local Ollama optional",
            "providerRuns": provider_audit,
            "qualification": qualification_audit,
            "arbitraryPythonExecuted": False,
            "strictlyPastOnlyDsl": True,
        },
        "ensemble": ensemble_meta,
        "comparison": [item.summary for item in baselines] + [ensemble_eval.summary],
        "validation": {
            "gatePassed": validation_gate,
            "ensembleCostedNetIr": ensemble_validation,
            "bestFixedBaselineCostedNetIr": best_baseline_validation,
            "usedForGeneratorFeedback": False,
        },
        "guards": {
            "pbo": pbo,
            "quickDsr": quick_dsr,
            "quickDsrIsFullDsr": False,
            "priorFamilyTrials": int(config["multipleTesting"]["priorCogAlphaFamilyTrials"]),
            "previousAutonomousTrials": previous_trials,
            "currentRunTrials": len(evaluated),
            "totalTrials": total_trials,
        },
        "limitations": {"survivorshipWarning": config["data"]["survivorshipWarning"]},
        "verdict": verdict,
        "automaticTradingChanges": [],
    }
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    shadow_strategy = latest_shadow_ranking(ensemble_signal, config, run_id)
    atomic_write(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output / "report.md", render_report(result) + "\n")
    atomic_write(output / "shadow_strategy_candidate.json", json.dumps(shadow_strategy, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output / "train_elites.json", json.dumps({
        "schemaVersion": "cogalpha_autonomous_train_elites_v1",
        "status": "research_only",
        "runId": run_id,
        "selectionData": "train_only",
        "elites": [item.candidate for item in train_ranked[: int(config["search"]["elitePoolSize"])]],
    }, ensure_ascii=False, indent=2) + "\n")

    if use_state:
        persistent = train_ranked[: int(config["search"]["maximumPersistentParents"])]
        atomic_write(state_root / "parent_pool.json", json.dumps({
            "schemaVersion": "cogalpha_autonomous_parent_pool_v1",
            "status": "train_only_research_memory",
            "sourceRunId": run_id,
            "validationOrShadowMetricsStored": False,
            "parents": [item.candidate for item in persistent],
        }, ensure_ascii=False, indent=2) + "\n")
        append_jsonl(ledger_path, [{
            "schemaVersion": "cogalpha_autonomous_trial_v1",
            "runId": run_id,
            "candidate": item.candidate,
            "trainMetrics": item.summary["periods"]["train"],
            "trainFitness": item.summary["trainFitness"],
            "containsValidationMetrics": False,
            "containsShadowMetrics": False,
        } for item in evaluated])
        append_jsonl(state_root / "run_registry.jsonl", [{
            "runId": run_id,
            "inputFingerprint": fingerprint,
            "generatedAt": result["generatedAt"],
            "validationGatePassed": validation_gate,
            "researchOnly": True,
        }])
        atomic_write(latest_path, json.dumps({
            "schemaVersion": "cogalpha_autonomous_latest_v1",
            "runId": run_id,
            "inputFingerprint": fingerprint,
            "output": str(output),
            "researchOnly": True,
        }, ensure_ascii=False, indent=2) + "\n")
    return result, output


def self_test() -> None:
    config = load_json(DEFAULT_CONFIG)
    validate_config(config)
    index = pd.bdate_range("2020-01-01", periods=1100)
    rng = np.random.default_rng(7)
    columns = [f"51{index:04d}"[-6:] for index in range(40)]
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (len(index), len(columns))), index=index, columns=columns)
    close = 100.0 * (1.0 + returns).cumprod()
    panel = {
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]) * (1.0 + rng.normal(0.0, 0.001, close.shape)),
        "high": close * 1.005,
        "low": close * 0.995,
        "volume": pd.DataFrame(rng.lognormal(15, 0.3, close.shape), index=index, columns=columns),
        "amount": pd.DataFrame(rng.lognormal(19, 0.3, close.shape), index=index, columns=columns),
    }
    panel["vwap"] = panel["close"].copy()
    panel["returns"] = panel["close"].pct_change(fill_method=None)
    split = make_split(index, config)
    target, one_day = target_frames(panel, config)
    candidate = grammar_initial(config)[0]
    item, reason = evaluate_candidate(candidate, panel, target, one_day, split, config)
    if item is None:
        raise AssertionError(f"valid grammar candidate rejected: {reason}")
    full = core.evaluate_expression(candidate["expression"], panel)
    prefix_panel = {key: value.iloc[:800].copy() for key, value in panel.items()}
    prefix = core.evaluate_expression(candidate["expression"], prefix_panel)
    if not np.allclose(full.iloc[:800].to_numpy(), prefix.to_numpy(), equal_nan=True):
        raise AssertionError("factor expression is not prefix invariant")
    bad = copy.deepcopy(config)
    bad["fitness"]["shadowFeedbackAllowed"] = True
    try:
        validate_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("shadow feedback was not rejected")
    print("autonomous CogAlpha self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-candidates", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result, output = run(
        args.config.resolve(), args.output_root.resolve(), args.maximum_candidates,
        args.generations, not args.no_state, args.force,
    )
    print(json.dumps({
        "status": result["status"],
        "run_id": result.get("runId"),
        "output": str(output) if output else None,
        "validation_gate_passed": result.get("validation", {}).get("gatePassed"),
        "automatic_trading_changes": result.get("automaticTradingChanges", []),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
