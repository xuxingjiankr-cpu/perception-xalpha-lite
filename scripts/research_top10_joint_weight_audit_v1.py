#!/usr/bin/env python3
"""Frozen six-policy weight experiment; no trading or automatic publication.

Rank at t close BEFORE examining any future tradability. Train only on completely
matured labels. Returns are t+1 open to a sellable open at/after t+2 (A-share T+1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import panel_cache  # noqa: E402
import research_stock_top10_win_capture_weights_v2 as existing  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/research/top10_joint_weight_audit_v1.json"


def save(path, obj):
    path.write_text(json.dumps(existing.safe(obj), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config):
    if config["schemaVersion"] != "top10_joint_weight_audit_v1":
        raise ValueError("wrong schema")
    if config["status"] != "research_only_shadow_only_not_trading":
        raise ValueError("research only")
    if any(v for k, v in config["safety"].items() if k.startswith("may")):
        raise ValueError("mutation permissions forbidden")
    if config["safety"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders forbidden")
    t, e = config["training"], config["evaluation"]
    if t["purgeSessions"] < 1 + e["holdingSessions"] + e["maximumExitDelay"]:
        raise ValueError("purge shorter than maximum label maturity")
    if e["topCount"] != 10 or e["roundTripCost"] <= 0:
        raise ValueError("Top10 and positive costs required")
    if not e["neverPromoteFromThisStudy"] or not e["historicalWindowsAlreadyViewed"]:
        raise ValueError("historical reuse must be disclosed")
    if len(config["candidates"]) != 6:
        raise ValueError("six preregistered candidates only")
    if not 0 <= t["minimumWeight"] < 1 / 16 < t["maximumWeight"] <= 1:
        raise ValueError("infeasible weight bounds")
    for name, spec in config["candidates"].items():
        a = np.array(spec["targets"])
        if len(a) != 3 or np.any(a < 0) or not np.isclose(a.sum(), 0 if name == "decorrelated" else 1):
            raise ValueError("invalid target mixture")
    root = (ROOT / config["outputRoot"]).resolve()
    if not root.is_relative_to((ROOT / "outputs/edge_research").resolve()):
        raise ValueError("output outside research directory")


def transform(x, volatility, neutral):
    """Cross-sectional transformation uses ONLY today's eligible feature rows."""
    x = (x.astype(float) - 0.5) * np.sqrt(12.0)
    if not neutral:
        return x
    z = volatility.astype(float) - np.mean(volatility)
    xc = x - np.mean(x, axis=0)
    return xc - np.outer(z, z @ xc / max(float(z @ z), 1e-12))


def choose(x, weights, top=10):
    """No future labels or execution masks accepted by this interface."""
    score = x @ weights
    return np.argsort(-score, kind="stable")[:top]


def day_statistics(x, returns, prior, config):
    """Day-equal moments; 60% focus on prior Top100, 40% full cross section.

    Neighbourhood is chosen before looking at future returns. Targets are centred
    within each TRAIN day, so a general market rally cannot stand in for alpha.
    """
    t, e = config["training"], config["evaluation"]
    neighbourhood = choose(x, prior, t["baselineNeighbourhoodCount"])
    mask = np.isfinite(returns)
    if mask.sum() < t["minimumCrossSection"]:
        return None
    p = np.full(len(x), (1 - t["baselineNeighbourhoodMass"]) / len(x))
    p[neighbourhood] += t["baselineNeighbourhoodMass"] / len(neighbourhood)
    xx, yy, p = x[mask], returns[mask], p[mask]
    p /= p.sum()
    targets = np.column_stack([
        np.clip(yy / t["returnScale"], -t["returnClip"], t["returnClip"]),
        (yy > 0).astype(float), -(yy <= e["tailThreshold"]).astype(float),
    ])
    # Scale all three tasks to unit training-day spread; no test target is used.
    targets -= targets.mean(axis=0)
    targets /= np.maximum(targets.std(axis=0), 0.05)
    xc = xx - p @ xx
    return xc.T @ (p[:, None] * xc), xc.T @ (p[:, None] * targets)


def training_indices(first, config):
    """End is exclusive; even maximum-delay labels end before test begins."""
    t = config["training"]
    end = first - t["purgeSessions"]
    return np.arange(max(0, end - t["lookbackSessions"]), max(0, end))


def fit(moments, indices, prior, spec, config):
    valid = [moments[i] for i in indices if moments[i] is not None]
    if len(valid) < config["training"]["minimumDays"]:
        raise RuntimeError(f"insufficient_mature_training_days:{len(valid)}")
    g = np.mean([item[0] for item in valid], axis=0)
    b = np.mean([item[1] for item in valid], axis=0) @ np.array(spec["targets"])
    t = config["training"]
    shrink = t["covarianceDiagonalShrinkage"]
    g = (1 - shrink) * g + shrink * np.diag(np.diag(g))
    lam = t["l2Prior"]
    def objective(w):
        return float(w @ g @ w - 2 * b @ w + lam * np.sum((w - prior) ** 2))
    def gradient(w):
        return 2 * g @ w - 2 * b + 2 * lam * (w - prior)
    result = minimize(objective, np.full(len(prior), 1 / len(prior)), jac=gradient,
                      method="SLSQP", bounds=[(t["minimumWeight"], t["maximumWeight"])] * len(prior),
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                   "jac": lambda w: np.ones_like(w)},
                      options={"maxiter": 200, "ftol": 1e-10})
    if not result.success or abs(result.x.sum() - 1) > 1e-7:
        raise RuntimeError(f"optimizer_failed:{result.message}")
    return result.x


def basket_row(date, period, name, x, w, returns, config):
    picks = choose(x, w)
    selected = returns[picks]
    valid = np.isfinite(selected)
    fill = int(valid.sum())
    cost = config["evaluation"]["roundTripCost"]
    tail = config["evaluation"]["tailThreshold"]
    universe = returns[np.isfinite(returns)]
    # No replacement of unfilled/unresolved names. Fixed-slot cash proxy separately
    # exposes selection attrition; unresolved exits are NOT claimed to be cash fills.
    return {"date": date, "period": period, "policy": name, "selected": len(picks),
            "resolved": fill, "missing": len(picks) - fill,
            "gross": float(np.mean(selected[valid])) if fill else np.nan,
            "net": float(np.mean(selected[valid]) - cost) if fill else np.nan,
            "up": float(np.mean(selected[valid] > 0)) if fill else np.nan,
            "net_up": float(np.mean(selected[valid] > cost)) if fill else np.nan,
            "tail": float(np.mean(selected[valid] <= tail)) if fill else np.nan,
            "slot_net_proxy": float(np.nansum(selected - cost) / 10),
            "excess": float(np.mean(selected[valid]) - np.mean(universe)) if fill else np.nan}


def hac(values, lags):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return {"t": None, "p_one_sided": 1.0}
    z = x - x.mean()
    v = float(z @ z / len(z))
    for k in range(1, min(lags, len(z) - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * float(z[k:] @ z[:-k] / len(z))
    if v <= 1e-16:
        return {"t": None, "p_one_sided": 1.0}
    stat = float(x.mean() / np.sqrt(v / len(z)))
    return {"t": stat, "p_one_sided": float(norm.sf(stat))}


def summarize(frame):
    return {"days": len(frame), "selected": int(frame.selected.sum()),
            "resolved": int(frame.resolved.sum()),
            "fill_fraction": float(frame.resolved.sum() / max(1, frame.selected.sum())),
            **{col: float(frame[col].mean()) for col in
               ["gross", "net", "up", "net_up", "tail", "slot_net_proxy", "excess"]}}


def compare(rows, config):
    summary, checks = {}, {}
    candidates = list(config["candidates"])
    for period in rows.period.unique():
        subset = rows[rows.period == period]
        summary[period] = {n: summarize(f) for n, f in subset.groupby("policy")}
        base = subset[subset.policy == "current_guarded16"].set_index("date")
        for name in candidates:
            item = subset[subset.policy == name].set_index("date")
            paired = item.join(base, lsuffix="_new", rsuffix="_old").dropna(subset=["net_new", "net_old"])
            delta = {m: paired[m + "_new"] - paired[m + "_old"] for m in ["net", "up", "tail"]}
            stats = {m: hac(v, config["evaluation"]["hacLags"]) for m, v in delta.items()}
            monthly = pd.DataFrame(delta)
            monthly.index = pd.to_datetime(monthly.index)
            monthly = monthly.resample("MS").mean()
            checks[(period, name)] = {"delta": {k: float(v.mean()) for k, v in delta.items()},
                                     "hac": stats, "months": len(monthly),
                                     "joint_improved_months": int(((monthly.net > 0) & (monthly.up > 0)).sum()),
                                     "monthly": monthly.reset_index().to_dict("records")}
        # Intersection-union: both return AND win must improve. Holm across six policies.
        ordered = sorted(candidates, key=lambda n: max(checks[(period, n)]["hac"][m]["p_one_sided"] for m in ["net", "up"]))
        running = 0.0
        for i, name in enumerate(ordered):
            c = checks[(period, name)]
            p = max(c["hac"][m]["p_one_sided"] for m in ["net", "up"])
            running = max(running, min(1.0, (len(ordered) - i) * p))
            c["joint_holm_p"] = running
            s = summary[period][name]
            c["passes_reject_only_checks"] = bool(
                running < config["evaluation"]["holmFamilyAlpha"] and
                c["delta"]["tail"] <= 0 and s["net"] > 0 and
                s["fill_fraction"] >= config["evaluation"]["minimumFillFraction"] and
                c["joint_improved_months"] > c["months"] / 2)
    return summary, {p: {n: checks[(p, n)] for n in candidates} for p in summary}


def build_data(config):
    read = existing.load_json
    ranking = read(ROOT / config["sourceRankingConfig"])
    existing.sixteen.validate_config(ranking)
    interaction = read(ROOT / ranking["sourceInteractionConfig"])
    existing.interaction_model.validate_config(interaction)
    source = read(ROOT / interaction["sourceForecastConfig"])
    guarded_config = read(ROOT / source["guardedWeightsConfig"])
    frozen, _, _ = existing.guarded.load_frozen_config(guarded_config)
    base = read(ROOT / frozen["baseResearchConfig"])
    _, cog = existing.perception.load_base_configs(base)
    key, _ = panel_cache.cache_key(base, cog)
    panel, panel_audit = panel_cache.build_configured_panel_cached(base, cog)
    ranks, _, factor_audit = panel_cache.build_rank_book_cached(panel, frozen, key)
    print("building_point_in_time_fundamental_families", flush=True)
    families, support, fundamental_audit = existing.fundamental_stage.build_family_scores(
        panel, read(ROOT / interaction["fundamentalMechanismConfig"]))
    contexts = existing.interaction_model.build_market_context_ranks(panel, interaction)
    for f in list(ranks.values()) + list(contexts.values()):
        support &= f.notna()
    interactions = existing.interaction_model.build_interaction_ranks(families, contexts, interaction, support)
    features = {**ranks, **{"interaction/" + k: v for k, v in interactions.items()}}
    for f in features.values():
        support &= f.notna()
    volatility = panel["close"].pct_change(fill_method=None).rolling(20, min_periods=20).std().where(support).rank(axis=1, pct=True)
    support &= volatility.notna()
    e = config["evaluation"]
    y, entry_eligible, delay = existing.precision.executable_horizon_return(panel, e["holdingSessions"], e["maximumExitDelay"])
    prior = existing.prior_research.prior_weights(frozen, list(features))
    price_prior = pd.Series({f["factorKey"]: f["weight"] for f in frozen["frozenFactors"]})
    ic = existing.rolling.factor_daily_ic(ranks, y)
    adaptive, _ = existing.guarded.guarded_weight_path(ic, price_prior, guarded_config)
    guarded_weights = pd.DataFrame(index=y.index, columns=list(features), dtype=float)
    for k in features:
        guarded_weights[k] = adaptive[k] * .75 if k in adaptive else .0625
    partitions = existing.discrimination.fixed_partitions(y.index, read(ROOT / interaction["modelTemplate"]))
    return panel, features, support, volatility, y, entry_eligible, delay, prior, guarded_weights, partitions, {
        "panelKey": key, "panelAudit": panel_audit, "fundamentalAudit": fundamental_audit,
        "factorAudit": factor_audit, "fullFactorsRequired": 16, "imputationAllowed": False,
        "sourceConfigHashes": {p: digest(ROOT / p) for p in [config["sourceRankingConfig"],
            ranking["sourceInteractionConfig"], interaction["sourceForecastConfig"],
            interaction["fundamentalMechanismConfig"], interaction["modelTemplate"],
            source["guardedWeightsConfig"], frozen["baseResearchConfig"]]}}


def run(config_path, run_id):
    config = existing.load_json(config_path)
    validate(config)
    out = ROOT / config["outputRoot"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"runId": run_id, "startedAt": datetime.now(timezone.utc).isoformat(),
                "status": "running", "researchOnly": True, "orders": [], "gitCommit": commit,
                "configSha256": digest(config_path), "scriptSha256": digest(Path(__file__)),
                "gitStatus": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "python": sys.version, "platform": platform.platform(), "config": config}
    save(out / "manifest.json", manifest)
    try:
        panel, features, support, vol, y, entry, delay, prior, adaptive, partitions, audit = build_data(config)
        save(out / "data_audit.json", audit)
        dates, symbols = y.index, y.columns
        keys = list(features)
        first_eval = dates.get_loc(partitions["audit"][0])
        start = max(0, first_eval - config["training"]["lookbackSessions"] - config["training"]["purgeSessions"])
        arrays = {k: f.to_numpy(dtype=np.float32) for k, f in features.items()}
        support_a, ya, va = support.to_numpy(), y.to_numpy(), vol.to_numpy()
        moments = {False: [None] * len(dates), True: [None] * len(dates)}
        daily = {}
        print(f"preparing_day_equal_moments dates={len(dates)-start} factors={len(keys)}", flush=True)
        for i in range(start, len(dates)):
            ids = np.flatnonzero(support_a[i])
            if len(ids) < config["training"]["minimumCrossSection"]:
                continue
            x = np.column_stack([arrays[k][i, ids] for k in keys])
            for neutral in [False, True]:
                xx = transform(x, va[i, ids], neutral)
                moments[neutral][i] = day_statistics(xx, ya[i, ids], prior, config)
            daily[i] = (ids, x)
        rows, weight_log = [], []
        candidate_weights = {}
        count = 0
        for period in ["audit", "validation", "shadow"]:
            period_dates = partitions[period]
            period_end = dates.get_loc(period_dates[-1])
            for date in period_dates:
                i = dates.get_loc(date)
                if i not in daily:
                    continue
                if count % config["training"]["refitSessions"] == 0:
                    tr = training_indices(i, config)
                    for name, spec in config["candidates"].items():
                        candidate_weights[name] = fit(moments[spec["neutralizeVolatility"]], tr, prior, spec, config)
                        weight_log.append({"testStart": date, "trainStart": dates[tr[0]],
                                           "trainEnd": dates[tr[-1]], "policy": name,
                                           "weights": dict(zip(keys, candidate_weights[name]))})
                    print(f"refit date={date.date()} train_end={dates[tr[-1]].date()}", flush=True)
                count += 1
                ids, x = daily[i]
                neutral_x = {n: transform(x, va[i, ids], n) for n in [False, True]}
                outcomes = ya[i, ids].copy()
                # Labels crossing period end are censored, never borrowed from next period.
                exit_i = i + 1 + config["evaluation"]["holdingSessions"] + delay.iloc[i, ids].to_numpy()
                outcomes[~(exit_i <= period_end)] = np.nan
                controls = {"frozen16": prior, "equal16": np.full(len(prior), 1 / len(prior)),
                            "current_guarded16": adaptive.iloc[i].to_numpy(dtype=float)}
                for name, w in {**controls, **candidate_weights}.items():
                    n = config["candidates"].get(name, {}).get("neutralizeVolatility", False)
                    rows.append(basket_row(date, period, name, neutral_x[n], w, outcomes, config))
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "daily_baskets.csv", index=False, encoding="utf-8")
        save(out / "weight_path.json", weight_log)
        summary, checks = compare(frame, config)
        # Explicit invalid hindsight ceiling on identical period dates and label rules.
        hindsight = {}
        for period in summary:
            indices = np.array([dates.get_loc(d) for d in partitions[period]])
            # Use all evaluated days when a single short period cannot meet minimumDays;
            # this is a clearly labelled future-leaking ceiling, never a deployable model.
            all_eval = np.array(sorted(dates.get_loc(d) for d in frame.date.unique()))
            pool = all_eval[all_eval + config["training"]["purgeSessions"] < len(dates)]
            for name, spec in config["candidates"].items():
                w = fit(moments[spec["neutralizeVolatility"]], pool, prior, spec, config)
                rr = []
                for i in indices:
                    if i not in daily:
                        continue
                    ids, x = daily[i]
                    outcomes = ya[i, ids].copy()
                    exit_i = i + 2 + delay.iloc[i, ids].to_numpy()
                    outcomes[~(exit_i <= indices[-1])] = np.nan
                    rr.append(basket_row(dates[i], period, name, transform(x, va[i, ids], spec["neutralizeVolatility"]), w, outcomes, config))
                hs = summarize(pd.DataFrame(rr))
                hindsight.setdefault(period, {})[name] = {"invalid_hindsight": hs,
                    "trailing_net": summary[period][name]["net"],
                    "hindsight_minus_trailing_net": hs["net"] - summary[period][name]["net"]}
        latest = []
        last = max(daily)
        tr = training_indices(last, config)
        for name, spec in config["candidates"].items():
            w = fit(moments[spec["neutralizeVolatility"]], tr, prior, spec, config)
            ids, x = daily[last]
            picked = choose(transform(x, va[last, ids], spec["neutralizeVolatility"]), w)
            latest.append({"policy": name, "signalDate": dates[last], "weights": dict(zip(keys, w)),
                           "researchOnlyTop10": list(symbols[ids[picked]])})
        survivors = [n for n in config["candidates"] if all(checks[p][n]["passes_reject_only_checks"] for p in config["evaluation"]["primaryPeriods"])]
        result = {"runId": run_id, "status": "diagnostic_only", "researchOnly": True, "orders": [],
                  "gitCommit": commit, "dataRange": [dates[0], dates[-1]], "symbolCount": len(symbols),
                  "latestCompleteSupport": len(daily[last][0]), "factorKeys": keys,
                  "summary": summary, "comparisons": checks, "hindsightExposure": hindsight,
                  "latestWeights": latest, "rejectOnlySurvivors": survivors, "mayPromote": False,
                  "newPolicyTrials": 6, "cumulativeTrialLowerBound": config["evaluation"]["historicalTrialCountLowerBound"] + 6,
                  "DSR": {"value": None, "reason": "No full historical trial-return ledger; never substitute a quick noise flag for a calibrated DSR."},
                  "PBO": {"value": None, "reason": "Not a clean held-out search matrix; causal walk-forward and paired HAC-Holm reported instead."},
                  "limitations": ["All historical windows already viewed; these tests may reject, not validate or promote.",
                    "SH/SZ supported PIT universe, not complete BJ coverage; remaining source coverage/survivorship bias is possible.",
                    "Strict full16 support differs from imputed dashboard universe; all policies use identical support here.",
                    "Fixed 30bp costs are assumptions, not observed fills; auction queues and price impact are unmodelled.",
                    "Unfilled/unresolved picks remain selected; no future-aware replacement. Resolved-only metrics can still have attrition bias.",
                    "Per-cohort returns are not a capital-constrained equity curve or an annual return.",
                    "Existing factor selection reused history; causal weights do not remove that earlier selection bias.",
                    "Vol-neutral policies change score preprocessing as a declared ablation, not weights alone.",
                    "No per-stock calibrated probabilities are produced by this weight study."]}
        save(out / "result.json", result)
        lines = ["# Top10 joint weight audit V1", "", "Research-only; no trading, dashboard or frozen-forward changes.",
                 f"Run: `{run_id}`. Data: {dates[0].date()} to {dates[-1].date()}; {len(symbols)} symbols.", "",
                 "Next open entry, following sellable open exit (T+1); 30bp round-trip cost. All 16 factors required.", "",
                 "| Period | Policy | Gross/pick | Net/pick | Up rate | Tail <= -3% | Resolved |", "|---|---|---:|---:|---:|---:|---:|"]
        for p, books in summary.items():
            for n, s in books.items():
                lines.append(f"| {p} | {n} | {s['gross']:.4%} | {s['net']:.4%} | {s['up']:.2%} | {s['tail']:.2%} | {s['fill_fraction']:.2%} |")
        lines += ["", f"Reject-only survivors: {survivors}. No promotion is permitted.", "", "## Limitations", ""]
        lines += ["- " + item for item in result["limitations"]]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.update(status="completed", finishedAt=datetime.now(timezone.utc).isoformat())
        save(out / "manifest.json", manifest)
        print(json.dumps({"output": str(out), "survivors": survivors}), flush=True)
        return result
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}:{exc}")
        save(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    run(args.config, args.run_id)
