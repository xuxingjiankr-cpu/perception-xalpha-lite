#!/usr/bin/env python3
"""Does qlib158/vstd60 rank what its label claims?

vstd60 is ts_std(volume,60) / volume - a 60-session volume standard deviation
divided by TODAY's volume, not by the 60-session mean. That denominator is the
whole question. A coefficient of variation divides by the mean and measures
stability; dividing by the current observation makes the quantity move inversely
with today's volume, so a quiet session scores high whatever the trailing
dispersion did.

This matters because vstd60 is a weighted member of the frozen twelve (0.0960,
direction +1), so the book is spending a tenth of its weight on whatever this
actually is. The VWAP basis defect in #16 was the same shape of problem: a factor
input that was not the quantity its name implied.

The audit is deliberately narrow. It reports cross-sectional rank correlations
against a true coefficient of variation, against inverse current volume, against
price volatility and against volume level, per session, with a day-clustered t.
It says what vstd60 co-moves with. It cannot and does not say anything about
returns, edge or weights - #12 already established the book has no day-neutral
selection edge at any horizon, which caps what any single-factor result means.

Research only. Emits no orders and touches no trading path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_horizon_cost_frontier_v1 as frontier  # noqa: E402
from research_horizon_cost_frontier_v1 import panel_cache, guarded, perception  # noqa: E402


def read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(c: dict) -> None:
    if c.get("schemaVersion") != "vstd60_specification_audit_v1":
        raise ValueError("wrong_schema")
    if c.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("not_research_only")
    if any(v for k, v in c["safety"].items() if k.startswith("may")):
        raise ValueError("mutation_permission_granted")
    if c["safety"]["outputStatus"] != "diagnostic_only_not_an_order":
        raise ValueError("output_not_marked_diagnostic")
    h = c["preregisteredHypothesis"]
    if h["historicalRunCanPromote"] or not h["cannotEstablishAnyTradingConclusion"]:
        raise ValueError("promotion_path_present")
    if not h["thisIsASpecificationAuditNotAnEdgeClaim"]:
        raise ValueError("audit_must_not_claim_edge")
    if not c["output"]["ordersAlwaysEmpty"]:
        raise ValueError("orders_not_pinned_empty")


def ts_std(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).std()


def ts_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).mean()


def vendor_vstd60(volume: pd.DataFrame) -> pd.DataFrame:
    """The shipped formula, reproduced exactly: std over the CURRENT observation."""
    return ts_std(volume, 60) / (volume + 1e-12)


def true_cv60(volume: pd.DataFrame) -> pd.DataFrame:
    """What the label describes: dispersion over its own mean."""
    return ts_std(volume, 60) / ts_mean(volume, 60).replace(0.0, np.nan)


def comparators(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    volume = panel["volume"].replace(0.0, np.nan)
    returns = panel["returns"]
    return {
        "trueCoefficientOfVariation60": true_cv60(volume),
        "inverseCurrentVolume": 1.0 / volume,
        "realizedVolatility20": ts_std(returns, 20),
        "logCurrentVolume": np.log(volume),
    }


def daily_rank_correlation(
    left: pd.DataFrame, right: pd.DataFrame, eligible: pd.DataFrame, minimum: int
) -> pd.Series:
    """Spearman across names, one value per session.

    Ranking inside each session is what makes this a cross-sectional statement;
    a pooled correlation would be dominated by the level differences between
    symbols that the book never trades on.
    """
    a = left.where(eligible)
    b = right.where(eligible)
    both = a.notna() & b.notna()
    counts = both.sum(axis=1)
    ra = a.where(both).rank(axis=1)
    rb = b.where(both).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    numerator = (ra * rb).sum(axis=1, min_count=1)
    denominator = np.sqrt((ra ** 2).sum(axis=1, min_count=1) * (rb ** 2).sum(axis=1, min_count=1))
    out = numerator / denominator.replace(0.0, np.nan)
    return out.where(counts >= minimum)


def day_clustered_t(daily: pd.Series) -> float | None:
    values = daily.dropna().to_numpy(dtype=float)
    if values.size < 3:
        return None
    standard_error = float(values.std(ddof=1)) / float(np.sqrt(values.size))
    scale = float(np.abs(values).mean())
    if standard_error <= max(1e-15, scale * 1e-12):
        return None
    return float(values.mean() / standard_error)


def build_verdict(per_window: dict, margin: float) -> dict:
    """Falsified only if inverse volume beats the true CV on BOTH windows."""
    beats = {}
    for window, rows in per_window.items():
        cv = abs(rows["trueCoefficientOfVariation60"]["meanCorrelation"])
        inv = abs(rows["inverseCurrentVolume"]["meanCorrelation"])
        beats[window] = {
            "absCorrWithTrueCv": cv,
            "absCorrWithInverseVolume": inv,
            "inverseVolumeWins": inv > cv,
            "marginClearsThreshold": (inv - cv) > margin,
        }
    windows = list(beats)
    all_win = bool(windows) and all(beats[w]["inverseVolumeWins"] for w in windows)
    all_decisive = bool(windows) and all(beats[w]["marginClearsThreshold"] for w in windows)
    if all_win and all_decisive:
        decision = "label_falsified_vstd60_ranks_inverse_current_volume_not_volume_stability"
    elif all_win:
        decision = "label_doubtful_inverse_volume_leads_but_not_by_the_preregistered_margin"
    else:
        decision = "label_not_falsified_on_both_windows"
    return {
        "decision": decision,
        "byWindow": beats,
        "eligibleForTrading": False,
        "orders": [],
        "researchOnly": True,
        "establishesNoEdge": True,
        "note": "A correlation audit constrains what the factor measures, never whether it predicts returns.",
    }


def run(config_path, run_id: str) -> dict:
    c = read(config_path)
    validate_config(c)
    out = ROOT / c["output"]["root"] / run_id
    out.mkdir(parents=True, exist_ok=False)

    fc = read(ROOT / "configs/research/horizon_cost_frontier_v1.json")
    frozen, source, _ = guarded.load_frozen_config(
        {"basePrecisionConfig": fc["basePrecisionConfig"]}
    )
    base = read(ROOT / frozen["baseResearchConfig"])
    _, cog = perception.load_base_configs(base)
    panel, _ = panel_cache.build_configured_panel_cached(base, cog)

    weight = next(
        (f["weight"] for f in frozen["frozenFactors"] if f["factorKey"] == "qlib158/vstd60"),
        None,
    )
    if weight is None:
        raise ValueError("vstd60_not_in_frozen_book")

    eligible = panel["eligible"]
    subject = vendor_vstd60(panel["volume"].replace(0.0, np.nan))
    others = comparators(panel)
    minimum = int(c["evaluation"]["minimumNamesPerSession"])

    # splitAudit stores each window as [start, end, sessions], and the shadow
    # window is keyed shadowQuarantine. Resolve both rather than assuming.
    splits = source["splitAudit"]
    aliases = {"shadow": "shadowQuarantine"}
    windows = {}
    for name in c["evaluation"]["windows"]:
        key = name if name in splits else aliases.get(name)
        if key not in splits:
            raise ValueError("unknown_evaluation_window:" + name)
        span = splits[key]
        start, end = pd.Timestamp(span[0]), pd.Timestamp(span[1])
        mask = (eligible.index >= start) & (eligible.index <= end)
        if not mask.any():
            raise ValueError("empty_evaluation_window:" + name)
        windows[name] = mask

    per_window = {}
    for name, mask in windows.items():
        rows = {}
        for label, other in others.items():
            daily = daily_rank_correlation(
                subject.loc[mask], other.loc[mask], eligible.loc[mask], minimum
            )
            rows[label] = {
                "meanCorrelation": float(daily.mean()) if daily.notna().any() else None,
                "medianCorrelation": float(daily.median()) if daily.notna().any() else None,
                "dayClusteredT": day_clustered_t(daily),
                "sessions": int(daily.notna().sum()),
            }
        per_window[name] = rows

    verdict = build_verdict(per_window, float(c["evaluation"]["decisiveMarginInCorrelation"]))
    result = {
        "runId": run_id,
        "researchOnly": True,
        "status": c["safety"]["outputStatus"],
        "eligibleForTrading": False,
        "orders": [],
        "subject": "qlib158/vstd60",
        "vendorFormula": "ts_std(volume,60) / volume",
        "frozenWeight": weight,
        "frozenDirection": next(
            f["direction"] for f in frozen["frozenFactors"] if f["factorKey"] == "qlib158/vstd60"
        ),
        "correlations": per_window,
        "verdict": verdict,
        "dataRange": [str(eligible.index[0].date()), str(eligible.index[-1].date())],
        "knownLimitations": c["knownLimitations"],
    }
    (out / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    for window, rows in per_window.items():
        print(f"\n[{window}]")
        for label, row in rows.items():
            print(
                f"  {label:32s} mean={row['meanCorrelation']:+.4f} "
                f"median={row['medianCorrelation']:+.4f} t={row['dayClusteredT']:+.1f} "
                f"n={row['sessions']}"
            )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=ROOT / "configs/research/vstd60_specification_audit_v1.json"
    )
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    run(a.config, a.run_id)
