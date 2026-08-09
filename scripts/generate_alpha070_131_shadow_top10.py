"""Generate an exploratory next-session Top10 for the fixed Alpha070/131 15% policy.

The 15% policy was selected after viewing historical results, so this artifact is not clean
forward evidence and cannot place an order.  Expected gross return and probabilities are
fit only on the frozen V6 calibration block; the latest row is scored after all fitting.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))

import research_cogalpha_autonomous as autonomous  # noqa: E402
import research_perception_xalpha_autonomous as perception  # noqa: E402
import research_perception_xalpha_horizon_precision_v3 as precision  # noqa: E402
import research_perception_xalpha_twelve_factor_utility_weights_v6 as v6  # noqa: E402
import research_twelve_factor_alpha070_131_ablation as ablation  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "twelve_factor_alpha070_131_ablation_v3.json"
)
OUTPUT_ROOT = (
    ROOT / "outputs" / "edge_research" / "twelve_factor_alpha070_131_shadow_top10"
)
POLICY_NAME = "alpha070_alpha131_each_15pct"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_up_probability(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> IsotonicRegression:
    rows = v6.stack_score_outcome(score, returns, dates)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(
        rows["score"].to_numpy(float),
        rows["return"].gt(0.0).to_numpy(float),
        sample_weight=rows["sampleWeight"].to_numpy(float),
    )
    return model


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Alpha070/Alpha131 exploratory shadow Top10",
        "",
        "> Research-only. Post-selection exploratory policy; not clean forward evidence or an order.",
        "",
        f"- signal date: `{result['signalDate']}`",
        f"- intended entry: `{result['intendedEntryDate']}` open",
        f"- intended exit: `{result['intendedExitDate']}` open, subject to sellability",
        "- policy: Alpha070 15%, Alpha131 15%, remaining eleven factors 70% proportional",
        "- estimates: frozen historical calibration block only",
        "",
        "| rank | security | name | close | expected gross return | probability up | probability net positive | severe loss probability |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["top10"]:
        lines.append(
            f"| {row['rank']} | {row['securityId']} | {row['name']} | "
            f"{row['close']:.3f} | {row['expectedGrossReturn']:.3%} | "
            f"{row['probabilityUp']:.2%} | {row['probabilityNetPositive']:.2%} | "
            f"{row['probabilitySevereLoss']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The estimates are conditional model outputs, not guaranteed returns. The selected "
            "15% policy missed its preregistered historical significance threshold and remains exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, entry_date: str, exit_date: str) -> dict[str, Any]:
    config = load_json(config_path)
    ablation.validate_config(config)
    factors, frozen_weights, v6_config = ablation.frozen_factors_and_weights(config)
    base = load_json(ROOT / str(v6_config["baseResearchConfig"]))
    _, cog_config = perception.load_base_configs(base)
    panel, panel_audit = perception.build_configured_panel(base, cog_config)
    print(
        f"panel_ready symbols={panel['close'].shape[1]} sessions={panel['close'].shape[0]}",
        flush=True,
    )
    ranks, _audit = v6.compute_factor_ranks(panel, factors)
    alpha131 = str(config["factorContract"]["alpha131Key"])
    zoo, name = alpha131.split("/", 1)
    inputs = v6.build_factor_inputs(panel)
    raw = importlib.import_module(f"src.factors.zoo.{zoo}.{name}").compute(inputs)
    ranks[alpha131] = (
        raw.reindex_like(panel["close"])
        * float(config["factorContract"]["alpha131Direction"])
    ).where(panel["eligible"]).rank(axis=1, pct=True).astype("float32")
    keys = [str(item["factorKey"]) for item in factors]
    weights = ablation.policy_weights(
        keys,
        frozen_weights,
        str(config["factorContract"]["alpha070Key"]),
        alpha131,
    )[POLICY_NAME]
    common = panel["eligible"].copy().fillna(False)
    for rank in ranks.values():
        common &= rank.notna()
    score = ablation.weighted_score(ranks, weights, common)
    returns, execution_eligible, _exit_delay = precision.executable_horizon_return(
        panel,
        int(v6_config["data"]["holdingTradingDays"]),
        int(v6_config["data"]["maximumExitDelayTradingDays"]),
    )
    returns = returns.where(execution_eligible)
    split = autonomous.make_split(panel["close"].index, cog_config)
    partitions = v6.training_partitions(split.train, v6_config)
    calibrators = v6.fit_calibrators(
        score, returns, partitions["calibration"], v6_config
    )
    predictions = v6.predict_frames(score, calibrators, v6_config)
    up_model = fit_up_probability(score, returns, partitions["calibration"])
    signal_date = score.index.max()
    current = score.loc[signal_date].dropna().sort_values(ascending=False).head(10)
    names = v6.name_map(base)
    output: list[dict[str, Any]] = []
    for position, (security_id, factor_score) in enumerate(current.items(), start=1):
        expected = float(predictions["expectedReturn"].loc[signal_date, security_id])
        net_loss = float(predictions["netLossProbability"].loc[signal_date, security_id])
        severe = float(predictions["severeLossProbability"].loc[signal_date, security_id])
        probability_up = float(up_model.predict([float(factor_score)])[0])
        output.append(
            {
                "rank": position,
                "securityId": str(security_id),
                "name": names.get(str(security_id), ""),
                "close": round(float(panel["close"].loc[signal_date, security_id]), 4),
                "factorScore": round(float(factor_score), 8),
                "expectedGrossReturn": round(expected, 8),
                "probabilityUp": round(probability_up, 8),
                "probabilityNetPositive": round(1.0 - net_loss, 8),
                "probabilitySevereLoss": round(severe, 8),
            }
        )
    result = {
        "schemaVersion": "alpha070_131_shadow_top10_v1",
        "status": "research_only_post_selection_exploratory_not_trading",
        "generatedAt": datetime.now().astimezone().isoformat(),
        "signalDate": str(signal_date.date()),
        "intendedEntryDate": entry_date,
        "intendedExitDate": exit_date,
        "policy": POLICY_NAME,
        "weights": weights,
        "calibrationDates": [
            str(partitions["calibration"].min().date()),
            str(partitions["calibration"].max().date()),
        ],
        "candidateCount": int(common.loc[signal_date].sum()),
        "top10": output,
        "panelAudit": panel_audit,
        "historicalPolicySignificancePassed": False,
        "eligibleForTrading": False,
        "orders": [],
        "automaticTradingChanges": [],
    }
    root = OUTPUT_ROOT / str(signal_date.date())
    atomic_text(root / "shadow_top10.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    atomic_text(root / "shadow_top10.md", render_report(result))
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--entry-date", required=True)
    parser.add_argument("--exit-date", required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.entry_date, args.exit_date)
    print(json.dumps({"signalDate": result["signalDate"], "top10": result["top10"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
