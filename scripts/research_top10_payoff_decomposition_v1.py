"""Fixed-count, purged historical payoff research. No publication or trading."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score

import research_ashare_pit_panel_v2 as pit
import research_top10_joint_weight_audit_v1 as audit_lib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/top10_payoff_decomposition_v1.json"
HEADS = ["direct_return", "up_probability", "payoff_decomposition"]
POLICIES = ["guarded16", "frozen16", "equal16"] + HEADS
save = audit_lib.save


def validate(c):
    if c["schemaVersion"] != "top10_payoff_decomposition_v1" or c["status"] != "research_only_shadow_only_not_trading":
        raise ValueError("research-only schema required")
    if any(v for k, v in c["safety"].items() if k.startswith("may")) or c["safety"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("mutation permission forbidden")
    t, e = c["training"], c["evaluation"]
    if t["purgeSessions"] < e["holdingSessions"] + e["maximumExitDelay"] + 1:
        raise ValueError("insufficient label maturity purge")
    if c["policies"] != POLICIES or e["topCount"] != 10 or e["roundTripCost"] <= 0:
        raise ValueError("frozen policies, Top10 and positive cost required")
    if not e["neverPromote"] or not e["historicalWindowsAlreadyViewed"]:
        raise ValueError("historical reuse cannot promote")
    if t["trainSessions"] < t["minimumTrainDays"] or t["calibrationSessions"] < t["minimumCalibrationDays"]:
        raise ValueError("insufficient window lengths")
    if not (ROOT / c["outputRoot"]).resolve().is_relative_to((ROOT / "outputs/edge_research").resolve()):
        raise ValueError("output must be isolated research")


def dependency_hashes():
    # Capture before execution, including dynamically imported vendored factors.
    return {str(p.relative_to(ROOT)): pit.digest(p) for p in sorted((ROOT / "scripts").rglob("*.py"))}


def opening_masks(panel, c):
    """No same-session HIGH/LOW/CLOSE/AMOUNT/VOLUME determines an opening fill.

    Date-level verified status is assumed available at open, not timestamp-proven.
    Prices at/near limits are conservatively unavailable; queues remain unmodelled.
    """
    op = panel["open"]
    lim = pd.DataFrame(.1, index=op.index, columns=op.columns)
    for sid in op.columns:
        code = str(sid).split(".")[-1]
        if code.startswith(("688", "689")):
            lim[sid] = .2
        elif code.startswith(("300", "301")):
            lim.loc[op.index >= "2020-08-24", sid] = .2
    old_main_st = panel["is_st"].eq(1) & lim.eq(.1)
    old_main_st.loc[op.index >= "2026-07-06"] = False
    lim = lim.mask(old_main_st, .05)
    ratio = op.div(panel["preclose"].where(panel["preclose"].gt(0))) - 1
    known = (op.gt(0) & panel["trade_status"].eq(1)
             & panel["membership"].fillna(False).astype(bool)
             & panel["is_st"].isin([0, 1]) & ratio.notna())
    clearance = c["evaluation"]["openLimitBuffer"]
    return (known & panel["is_st"].eq(0) & ratio.lt(lim - clearance),
            known & ratio.gt(-lim + clearance))


def execution_labels(panel, c):
    buy, sell = opening_masks(panel, c)
    op = panel["open"].to_numpy(dtype=float)
    entry = np.zeros(op.shape, dtype=bool)
    entry[:-1] = buy.to_numpy()[1:]
    y, delay = np.full_like(op, np.nan), np.full_like(op, np.nan)
    chosen = np.full_like(op, np.nan)
    for d in range(c["evaluation"]["maximumExitDelay"] + 1):
        offset = 1 + c["evaluation"]["holdingSessions"] + d
        n = len(op) - offset
        if n <= 0:
            continue
        take = entry[:n] & np.isnan(chosen[:n]) & sell.to_numpy()[offset:]
        chosen[:n] = np.where(take, op[offset:], chosen[:n])
        delay[:n] = np.where(take, d, delay[:n])
    y[:-1] = chosen[:-1] / op[1:] - 1
    frame = lambda a: pd.DataFrame(a, index=panel["open"].index, columns=panel["open"].columns)
    return frame(y), frame(entry), frame(delay)


def split_indices(first, c):
    t = c["training"]
    ce = first - t["purgeSessions"]
    cs = ce - t["calibrationSessions"]
    te = cs - t["purgeSessions"]
    ts = te - t["trainSessions"]
    if ts < 0:
        raise RuntimeError("insufficient_mature_history")
    return np.arange(ts, te), np.arange(cs, ce)


def gather(daily, indices, c):
    xs, ys, ws, used = [], [], [], []
    for i in indices:
        if i not in daily:
            continue
        row = daily[i]
        valid = np.isfinite(row["y"])
        ids = np.flatnonzero(valid)
        if len(ids) < c["training"]["minimumCrossSection"]:
            continue
        # Subsampling depends on index/seed, never on return magnitude or class.
        if len(ids) > c["training"]["maximumRowsPerDay"]:
            ids = np.sort(np.random.default_rng(c["training"]["seed"] ^ int(i)).choice(
                ids, c["training"]["maximumRowsPerDay"], replace=False))
        xs.append(row["x"][ids]); ys.append(row["y"][ids])
        ws.append(np.full(len(ids), 1 / len(ids)))
        used.append(int(i))
    if not xs:
        raise RuntimeError("empty_training_block")
    weights = np.concatenate(ws)
    weights /= weights.mean()
    return np.concatenate(xs), np.concatenate(ys), weights, used


def weighted_moments(x, w):
    mean = np.average(x, weights=w, axis=0)
    std = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
    return mean, np.maximum(std, 1e-6)


def raw_predict(m, x):
    z = (x - m["mean"]) / m["std"]
    logit = m["up"].decision_function(z)
    p = m["platt"].predict_proba(logit[:, None])[:, 1] if "platt" in m else m["up"].predict_proba(z)[:, 1]
    gain = np.clip(m["gain"].predict(z) * m["scale"], 0, m["cap"])
    loss = np.clip(m["loss"].predict(z) * m["scale"], 0, m["cap"])
    return {"logit": logit, "p": p, "gain": gain, "loss": loss,
            "direct_return": np.clip(m["direct"].predict(z) * m["scale"], -m["cap"], m["cap"]),
            "payoff_decomposition": p * gain - (1 - p) * loss}


def fit_models(daily, train, cal, c, hindsight=False):
    x, y, w, used_train = gather(daily, train, c)
    cx, cy, cw, used_cal = gather(daily, cal, c)
    t = c["training"]
    if len(used_train) < t["minimumTrainDays"] or len(used_cal) < t["minimumCalibrationDays"]:
        raise RuntimeError(f"insufficient_usable_fit_days:{len(used_train)}/{len(used_cal)}")
    if not hindsight and (max(used_train) + t["purgeSessions"] >= min(used_cal)):
        raise RuntimeError("train_calibration_overlap")
    mean, std = weighted_moments(x, w)
    cap = max(float(np.quantile(np.abs(y), t["targetClipQuantile"])), .001)
    target = np.clip(y, -cap, cap) / t["returnScale"]
    z = (x - mean) / std
    up = y > 0
    if min(up.sum(), (~up).sum()) < 100 or len(np.unique(cy > 0)) != 2:
        raise RuntimeError("insufficient_classes")
    m = {"mean": mean, "std": std, "cap": cap, "scale": t["returnScale"],
         "up": LogisticRegression(C=t["logisticC"], max_iter=500),
         "direct": Ridge(alpha=t["ridgeAlpha"]), "gain": Ridge(alpha=t["ridgeAlpha"]),
         "loss": Ridge(alpha=t["ridgeAlpha"])}
    m["up"].fit(z, up, sample_weight=w)
    m["direct"].fit(z, target, sample_weight=w)
    m["gain"].fit(z[up], target[up], sample_weight=w[up])
    m["loss"].fit(z[~up], -target[~up], sample_weight=w[~up])
    raw = raw_predict(m, cx)
    m["platt"] = LogisticRegression(C=1., max_iter=500).fit(
        raw["logit"][:, None], cy > 0, sample_weight=cw)
    if max(int(m[k].n_iter_.max()) for k in ["up", "platt"]) >= 500:
        raise RuntimeError("logistic_fit_not_converged")
    raw = raw_predict(m, cx)
    for name in ["direct_return", "payoff_decomposition"]:
        m[name + "_cal"] = Ridge(alpha=t["ridgeAlpha"]).fit(
            raw[name][:, None] / m["scale"], cy / m["scale"], sample_weight=cw)
    meta = {"trainDays": len(used_train), "calibrationDays": len(used_cal),
            "trainRows": len(y), "calibrationRows": len(cy),
            "trainFirst": min(used_train), "trainLast": max(used_train),
            "calibrationFirst": min(used_cal), "calibrationLast": max(used_cal),
            "cap": cap, "upFraction": float(up.mean()), "invalidHindsight": hindsight,
            "returnCalibrationSlopes": {n: float(m[n + "_cal"].coef_[0]) for n in
                                         ["direct_return", "payoff_decomposition"]}}
    return m, meta


def predict(m, x):
    out = raw_predict(m, x)
    out["raw_payoff"] = out["payoff_decomposition"].copy()
    for name in ["direct_return", "payoff_decomposition"]:
        out[name] = np.clip(m[name + "_cal"].predict(out[name][:, None] / m["scale"]) * m["scale"], -m["cap"], m["cap"])
    out["up_probability"] = out["p"]
    return out


def top10(score):
    if not np.isfinite(score).all() or len(score) < 10:
        raise RuntimeError("invalid_rank_support")
    return np.argsort(-score, kind="stable")[:10]


def period_outcomes(row, i, end, c):
    y = row["y"].copy()
    exit_index = i + 1 + c["evaluation"]["holdingSessions"] + row["delay"]
    y[~(exit_index <= end)] = np.nan
    return y


def record_picks(row, i, date, end, period, policy, scores, pred, c):
    # Future information is accessed only AFTER the ten choices are fixed.
    picks = top10(scores)
    y = period_outcomes(row, i, end, c)
    records = []
    for rank, k in enumerate(picks, 1):
        if np.isfinite(y[k]):
            state = "resolved"
        elif i + 1 > end:
            state = "pending_entry"
        elif not row["entry"][k]:
            state = "unfilled_entry"
        elif i + 1 + c["evaluation"]["holdingSessions"] + c["evaluation"]["maximumExitDelay"] > end:
            state = "pending_exit_maturity"
        else:
            state = "entered_unresolved_exit"
        records.append({"date": str(date.date()), "period": period, "policy": policy,
                        "rank": rank, "securityId": row["symbols"][k], "state": state,
                        "gross": y[k], "net": y[k] - c["evaluation"]["roundTripCost"],
                        "pUp": pred["p"][k], "gainIfUp": pred["gain"][k],
                        "lossIfNonUp": pred["loss"][k], "rawPayoff": pred["raw_payoff"][k],
                        "expectedGross": pred["payoff_decomposition"][k],
                        "score": float(scores[k]), "researchOnly": True})
    return records


def stats(picks, c):
    r = picks.gross.dropna().to_numpy()
    cost = c["evaluation"]["roundTripCost"]
    wins, losses = r[r > 0], r[r < 0]
    mean = lambda x: float(np.mean(x)) if len(x) else np.nan
    daily = picks.groupby("date").gross.mean()
    return {"days": picks.date.nunique(), "selected": len(picks), "resolved": len(r),
            "resolvedFraction": len(r) / max(1, len(picks)), "states": picks.state.value_counts().to_dict(),
            "gross": mean(daily.dropna()), "net": mean(daily.dropna()) - cost,
            "win": mean(r > 0), "netWin": mean(r > cost), "tail": mean(r <= c["evaluation"]["tailThreshold"]),
            "meanWin": mean(wins), "meanLoss": mean(losses),
            "payoffRatio": mean(wins) / -mean(losses) if len(losses) and len(wins) else np.nan,
            "worst5PercentMean": mean(r[r <= np.quantile(r, .05)]) if len(r) else np.nan}


def paired(picks, candidate, control, c):
    by = picks.groupby(["date", "policy"]).gross.mean().unstack()
    diff = (by[candidate] - by[control]).dropna()
    monthly = diff.groupby(pd.to_datetime(diff.index).to_period("M")).mean()
    return {"deltaNet": float(diff.mean()), "hac": audit_lib.hac(diff, c["evaluation"]["hacLags"]),
            "improvedMonths": int((monthly > 0).sum()), "months": len(monthly),
            "monthlyDelta": {str(k): float(v) for k, v in monthly.items()}}


def build_data(c, out):
    ex = audit_lib.existing
    read = ex.load_json
    ranking = read(ROOT / c["sourceRankingConfig"])
    interaction = read(ROOT / ranking["sourceInteractionConfig"])
    source = read(ROOT / interaction["sourceForecastConfig"])
    guarded_config = read(ROOT / source["guardedWeightsConfig"])
    frozen, _, _ = ex.guarded.load_frozen_config(guarded_config)
    base = read(ROOT / frozen["baseResearchConfig"])
    panel, pa = pit.build_panel(base["assetUniverse"], c["panel"])
    save(out / "panel_audit.json", pa)
    source_paths = [c["sourceRankingConfig"], ranking["sourceInteractionConfig"],
                    interaction["sourceForecastConfig"], source["guardedWeightsConfig"],
                    frozen["baseResearchConfig"], interaction["fundamentalMechanismConfig"], interaction["modelTemplate"]]
    ranks, _, fa = ex.rolling.compute_rank_book(panel, frozen)
    families, support, funda = ex.fundamental_stage.build_family_scores(panel, read(ROOT / interaction["fundamentalMechanismConfig"]))
    contexts = ex.interaction_model.build_market_context_ranks(panel, interaction)
    for f in list(ranks.values()) + list(contexts.values()):
        support &= f.notna()
    interactions = ex.interaction_model.build_interaction_ranks(families, contexts, interaction, support)
    features = {**ranks, **{"interaction/" + k: v for k, v in interactions.items()}}
    for f in features.values():
        support &= f.notna()
    y, entry, delay = execution_labels(panel, c)
    prior_price = pd.Series({r["factorKey"]: r["weight"] for r in frozen["frozenFactors"]})
    adaptive, _ = ex.guarded.guarded_weight_path(ex.rolling.factor_daily_ic(ranks, y), prior_price, guarded_config)
    prior = np.array([prior_price[k] * .75 if k in prior_price else .0625 for k in features])
    dates = panel["close"].index
    partitions = ex.discrimination.fixed_partitions(dates, read(ROOT / interaction["modelTemplate"]))
    first_eval = dates.get_loc(partitions["audit"][0])
    first_train = split_indices(first_eval, c)[0][0]
    daily = {}
    for i in range(first_train, len(dates)):
        ids = np.flatnonzero(support.iloc[i].to_numpy())
        if len(ids) < c["training"]["minimumCrossSection"]:
            continue
        x = np.column_stack([f.iloc[i, ids].to_numpy() for f in features.values()])
        aw = np.array([adaptive.iloc[i][k] * .75 if k in prior_price else .0625 for k in features])
        daily[i] = {"x": x, "symbols": np.array(panel["close"].columns)[ids],
                    "y": y.iloc[i, ids].to_numpy(), "entry": entry.iloc[i, ids].to_numpy(),
                    "delay": delay.iloc[i, ids].to_numpy(),
                    "controls": {"guarded16": x @ aw, "frozen16": x @ prior, "equal16": x.mean(axis=1)}}
    meta = {"panel": {k: v for k, v in pa.items() if k not in ["sourceFiles", "daily", "skippedFiles"]},
            "factors": fa, "fundamentals": funda, "fullSixteenRequired": True,
            "sourceConfigHashes": {p: pit.digest(ROOT / p) for p in source_paths},
            "featureKeys": list(features), "eligibleCompleteDays": len(daily)}
    save(out / "data_audit.json", meta)
    return dates, partitions, daily, meta


def run(path, run_id):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(c)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("unsafe run id")
    out = ROOT / c["outputRoot"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"runId": run_id, "researchOnly": True, "status": "running", "orders": [],
                "startedAt": datetime.now(timezone.utc).isoformat(), "config": c,
                "configSha256": pit.digest(path), "gitCommit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "gitStatus": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "python": sys.version, "platform": platform.platform(),
                "sourceHashes": dependency_hashes()}
    save(out / "manifest.json", manifest)
    try:
        dates, partitions, daily, meta = build_data(c, out)
        gc.collect()
        records, folds, diagnostic, buckets = [], [], [], []
        first = dates.get_loc(partitions["audit"][0])
        last_fit, models = None, None
        for period in ["audit", "validation", "shadow"]:
            end = dates.get_loc(partitions[period][-1])
            for date in partitions[period]:
                i = dates.get_loc(date)
                if i not in daily:
                    continue
                if models is None or i - last_fit >= c["training"]["refitSessions"]:
                    tr, cal = split_indices(i, c)
                    models, fold = fit_models(daily, tr, cal, c)
                    fold.update(testStart=str(date.date()), testIndex=i)
                    folds.append(fold); last_fit = i
                    save(out / "folds.json", folds)
                    print(f"payoff_refit {date.date()} train={fold['trainRows']} cal={fold['calibrationRows']}", flush=True)
                row = daily[i]
                pred = predict(models, row["x"])
                scores = {**row["controls"], **{k: pred[k] for k in HEADS}}
                for name in POLICIES:
                    records.extend(record_picks(row, i, date, end, period, name, scores[name], pred, c))
                yy = period_outcomes(row, i, end, c)
                valid = np.isfinite(yy)
                if valid.sum() > 20:
                    labels, p = yy[valid] > 0, pred["p"][valid]
                    clipped = np.clip(p, 1e-8, 1 - 1e-8)
                    diagnostic.append({"date": str(date.date()), "period": period,
                                       "rows": int(valid.sum()), "brier": brier_score_loss(labels, p),
                                       "logLoss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
                                       "auc": roc_auc_score(labels, p) if len(np.unique(labels)) == 2 else np.nan,
                                       **{k + "_mse": float(np.mean((pred[k][valid] - yy[valid]) ** 2))
                                          for k in ["direct_return", "payoff_decomposition"]},
                                       **{k + "_rank_ic": float(pd.Series(pred[k][valid]).corr(pd.Series(yy[valid]), method="spearman"))
                                          for k in ["direct_return", "payoff_decomposition"]}})
                    bins = np.minimum((p * 10).astype(int), 9)
                    for bin_id in np.unique(bins):
                        mask = bins == bin_id
                        buckets.append({"period": period, "date": str(date.date()), "bucket": int(bin_id),
                                        "count": int(mask.sum()), "sumPredicted": float(p[mask].sum()),
                                        "wins": int(labels[mask].sum())})
        picks = pd.DataFrame(records)
        if picks.empty:
            raise RuntimeError("no_evaluable_choices")
        picks.to_csv(out / "daily_picks.csv", index=False, encoding="utf-8")
        pd.DataFrame(diagnostic).to_csv(out / "prediction_diagnostics.csv", index=False)
        bucket_frame = pd.DataFrame(buckets).groupby(["period", "bucket"])[["count", "sumPredicted", "wins"]].sum().reset_index()
        bucket_frame["meanProbability"] = bucket_frame.sumPredicted / bucket_frame["count"]
        bucket_frame["hitRate"] = bucket_frame.wins / bucket_frame["count"]
        bucket_frame.to_csv(out / "probability_buckets.csv", index=False)
        ece = {p: float(np.average(np.abs(b.hitRate - b.meanProbability), weights=b["count"]))
               for p, b in bucket_frame.groupby("period")}
        summary, comparisons = {}, {}
        for period, block in picks.groupby("period"):
            summary[period] = {k: stats(b, c) for k, b in block.groupby("policy")}
            tests = {k: paired(block, k, "guarded16", c) for k in HEADS}
            running = 0.
            for n, k in enumerate(sorted(HEADS, key=lambda v: tests[v]["hac"]["p_one_sided"])):
                running = max(running, min(1., (len(HEADS) - n) * tests[k]["hac"]["p_one_sided"]))
                tests[k]["holmP"] = running
            tests["payoff_vs_direct"] = paired(block, "payoff_decomposition", "direct_return", c)
            comparisons[period] = tests
        # Deliberately invalid fit/calibration on ALL evaluated matured outcomes.
        # Report only in a clearly labelled side table; it never selects the rolling model.
        pool = np.array([i for i in daily if i >= first and i + c["training"]["purgeSessions"] < len(dates)])
        hindsight_model, _ = fit_models(daily, pool, pool, c, hindsight=True)
        hs_rows = []
        for period in ["audit", "validation", "shadow"]:
            end = dates.get_loc(partitions[period][-1])
            for date in partitions[period]:
                i = dates.get_loc(date)
                if i not in daily:
                    continue
                row = daily[i]; pr = predict(hindsight_model, row["x"])
                for name in HEADS:
                    hs_rows.extend(record_picks(row, i, date, end, period, name, pr[name], pr, c))
        hs = pd.DataFrame(hs_rows)
        hindsight = {p: {n: {"invalidHindsightNet": stats(b, c)["net"], "rollingNet": summary[p][n]["net"],
                             "gap": stats(b, c)["net"] - summary[p][n]["net"]}
                         for n, b in block.groupby("policy")} for p, block in hs.groupby("period")}
        checks = {}
        for p in ["validation", "shadow"]:
            m = summary[p]["payoff_decomposition"]
            a, b = comparisons[p]["payoff_decomposition"], comparisons[p]["payoff_vs_direct"]
            checks[p] = bool(m["net"] > 0 and m["resolvedFraction"] >= c["evaluation"]["minimumResolvedFraction"]
                             and a["holmP"] < c["evaluation"]["alpha"] and b["hac"]["p_one_sided"] < c["evaluation"]["alpha"]
                             and a["improvedMonths"] > a["months"] / 2 and b["improvedMonths"] > b["months"] / 2
                             and m["tail"] <= min(summary[p][n]["tail"] for n in ["guarded16", "direct_return"]))
        board = lambda s: "STAR" if str(s).split(".")[-1].startswith(("688", "689")) else (
            "ChiNext" if str(s).split(".")[-1].startswith(("300", "301")) else "Main")
        picks["board"] = picks.securityId.map(board)
        picks["month"] = picks.date.str[:7]
        slices = {key: [{"period": p, "policy": n, key: val, **stats(b, c)}
                        for (p, n, val), b in picks.groupby(["period", "policy", key])]
                  for key in ["month", "board"]}
        result = {"runId": run_id, "status": "diagnostic_only", "researchOnly": True, "orders": [],
                  "mayPromote": False, "gitCommit": manifest["gitCommit"], "data": meta,
                  "summary": summary, "comparisons": comparisons, "hindsightExposure": hindsight,
                  "probabilityECE": ece,
                  "rejectOnlyChecks": checks, "survivesRejection": all(checks.values()),
                  "slices": slices, "newRankPolicyTrials": 3, "cumulativeTrialLowerBound": c["evaluation"]["historicalTrialCountLowerBound"] + 3,
                  "DSR": None, "PBO": None,
                  "skipped": ["DSR/PBO: no complete historical trial ledger or clean selection matrix.",
                              "No live Top10 publication; data access blocked and research cannot promote.",
                              "BJ, auction queues, exact limit ticks, actual fills and capacity are unavailable."],
                  "limitations": meta.get("panel", {}).get("limitations", []) + [
                      "Previously viewed history; old factor-selection bias remains despite causal fitting.",
                      "Observed-result metrics can have attrition bias; entered-unresolved exits are never cash.",
                      "Fixed 30bp is an assumption. Cohort returns are not account return or annualised equity.",
                      "Opening-proxy revision differs from old report; all comparisons here use the same new support/execution.",
                      "Calibration and target clipping can attenuate right tails; reported realised returns are not clipped."]}
        save(out / "result.json", result)
        lines = ["# Top10 payoff decomposition V1", "", "RESEARCH ONLY — no orders, publication or promotion.",
                 f"Run: {run_id}. Commit: {manifest['gitCommit']}", "",
                 "All policies select ten names on identical support BEFORE observing opening fills.", "",
                 "| Period | Policy | Win | Gross/pick | Net/pick | Tail <= -3% | Resolved |", "|---|---|---:|---:|---:|---:|---:|"]
        for p, blocks in summary.items():
            for n, b in blocks.items():
                lines.append(f"| {p} | {n} | {b['win']:.2%} | {b['gross']:.4%} | {b['net']:.4%} | {b['tail']:.2%} | {b['resolvedFraction']:.2%} |")
        lines += ["", f"Rejection checks: {checks}; may promote: FALSE.", "", "## Limitations", ""] + ["- " + s for s in result["limitations"] + result["skipped"]]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if dependency_hashes() != manifest["sourceHashes"]:
            raise RuntimeError("source_code_changed_during_run")
        manifest.update(status="completed", finishedAt=datetime.now(timezone.utc).isoformat())
        save(out / "manifest.json", manifest)
        print(json.dumps({"output": str(out), "checks": checks, "mayPromote": False}), flush=True)
        return result
    except Exception as exc:
        manifest.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
        save(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    run(args.config, args.run_id)
