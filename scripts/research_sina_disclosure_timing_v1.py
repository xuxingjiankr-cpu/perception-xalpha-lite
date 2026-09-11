"""One preregistered historical statement-recency ablation; no trading/inference service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

import research_sina_fundamental_top10_v1 as base

ROOT = base.ROOT
ARMS = ["equal_families", "pooled_payoff", "disclosure_timing", "lagged_timing_counter"]
NEW_ARMS = ARMS[2:]
read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
save = base.save


def validate(c):
    if c["schemaVersion"] != "sina_disclosure_timing_v1" or c["researchOnly"] is not True:
        raise ValueError("research_only_required")
    if c["arms"] != ARMS or c["outputRoot"] != "outputs/edge_research/sina_disclosure_timing_v1":
        raise ValueError("isolated_fixed_arms_required")
    if any(v for k, v in c["safety"].items() if k.startswith("may")) or not c["safety"]["ordersAlwaysEmpty"]:
        raise ValueError("mutation_forbidden")
    f = c["features"]
    if (f["halfLifeSessions"] != [5, 20] or f["counterLagSessions"] != 20 or f["baselineFamilyMaxAgeSessions"] != 130
            or f["knownDisclosureAgeExpires"] is not False
            or f["newFeatureCount"] != 10 or f["modelFeatureCount"] != 22):
        raise ValueError("new_preregistration_required_for_changed_features")
    if c["comparison"]["mayPromote"] or not c["comparison"]["historicalWindowsAlreadyViewed"]:
        raise ValueError("reused_history_cannot_promote")


def normalized_hashes(rows):
    result = {}
    for r in rows:
        key = r["path"].replace("\\", "/")
        if key in result:
            raise ValueError("duplicate_input_hash:" + key)
        result[key] = r["sha256"]
    return result


def reference_contract(c):
    cfg_path = ROOT / c["baseConfig"]
    ref = ROOT / "outputs/edge_research/sina_fundamental_top10_v1" / c["referenceRun"]
    for p, expected in [(cfg_path, c["baseConfigSha256"]), (Path(base.__file__), c["baseModuleSha256"]),
                        (ref / "manifest.json", c["referenceManifestSha256"])]:
        if base.pit.digest(p) != expected:
            raise ValueError("frozen_reference_changed:" + str(p))
    bc, manifest = read(cfg_path), read(ref / "manifest.json")
    base.validate(bc)
    if bc != manifest["config"] or read(ref / "status.json")["state"] != "completed_diagnostic_only":
        raise ValueError("reference_not_identical_completed_training")
    for rel, h in manifest["dependencies"].items():
        if base.pit.digest(ROOT / rel) != h:
            raise ValueError("frozen_dependency_changed:" + rel)
    if (manifest["numpy"], manifest["pandas"], manifest["sklearn"]) != (np.__version__, pd.__version__, sklearn.__version__):
        raise ValueError("reference_runtime_changed")
    artifacts = [ref / n for n in ["manifest.json", "status.json", "input_hashes.json", "historical_picks.csv"]]
    artifacts += sorted((ref / "models").glob("fold_*.json"))
    if len(artifacts) <= 4:
        raise ValueError("missing_frozen_models")
    frozen_hashes = {str(p.relative_to(ROOT)): base.pit.digest(p) for p in artifacts}
    frozen_hashes.update(manifest["dependencies"])
    frozen_hashes[str(cfg_path.relative_to(ROOT))] = base.pit.digest(cfg_path)
    return bc, ref, frozen_hashes


def restore_model(artifact):
    """Allowlisted numeric coefficients only; no pickle, eval or dynamic classes."""
    if artifact.get("researchOnly") is not True or artifact.get("onlineInferenceAllowed") is not False:
        raise ValueError("research_numeric_artifact_required")
    p = artifact["parameters"]
    keys = {"mean", "std", "cap", "scale", "up", "direct", "gain", "loss", "platt", "direct_return_cal",
            "payoff_decomposition_cal", "tail", "tail_cal"}
    if set(p) != keys:
        raise ValueError("unexpected_model_fields")
    model = {k: np.asarray(p[k], dtype=float) for k in ["mean", "std"]}
    n = len(model["mean"])
    if model["mean"].shape != (n,) or model["std"].shape != (n,) or (model["std"] <= 0).any():
        raise ValueError("invalid_standardizer")
    for k in ("cap", "scale"):
        if not base.source.finite(p[k], True):
            raise ValueError("invalid_return_scale")
        model[k] = p[k]
    for k in keys - {"mean", "std", "cap", "scale"}:
        d = p[k]
        logistic = k in {"up", "platt", "tail", "tail_cal"}
        cls = LogisticRegression if logistic else Ridge
        if set(d) != {"class", "coef", "intercept"} or d["class"] != cls.__name__:
            raise ValueError("untrusted_model_class_or_fields")
        obj = cls()
        dim = 1 if k in {"platt", "tail_cal", "direct_return_cal", "payoff_decomposition_cal"} else n
        coef, intercept = np.asarray(d["coef"], float), np.asarray(d["intercept"], float)
        if coef.shape != ((1, dim) if logistic else (dim,)) or intercept.shape != ((1,) if logistic else ()):
            raise ValueError("invalid_coefficient_shape")
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError("nonfinite_coefficient")
        obj.coef_, obj.intercept_, obj.n_features_in_ = coef, intercept, dim
        if logistic:
            obj.classes_ = np.array([False, True])
        model[k] = obj
    if not np.isfinite(model["mean"]).all() or not np.isfinite(model["std"]).all():
        raise ValueError("invalid_standardizer")
    return model


def disclosure_age(rows, sessions, sid, families):
    events, audit = base.fundamentals.causal_fundamental_records_for_symbol(rows, sessions, sid, families)
    age = np.full(len(sessions), np.nan, np.float32)
    for j, event in enumerate(events):
        start = int(sessions.get_loc(event["eventDate"]))
        stop = int(sessions.get_loc(events[j + 1]["eventDate"])) if j + 1 < len(events) else len(sessions)
        # Knowing when a disclosure happened does not expire with its numeric
        # family's 130-session carry limit. This matters for the lagged counter.
        age[start:stop] = np.arange(stop - start)
    # No archive/no known disclosure is unknown, NOT "nothing happened".
    return age, audit


def timing_matrix(original, ages, half_lives):
    if original.ndim != 2 or original.shape[1] != 12 or len(original) != len(ages):
        raise ValueError("timing_feature_shape_mismatch")
    if not np.isfinite(original).all() or not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError("unknown_age_or_feature_no_support_dropping_allowed")
    decay = np.exp2(-np.asarray(ages, float)[:, None] / np.asarray(half_lives, float))
    interaction = (original[:, :4, None] * decay[:, None, :]).reshape(len(original), 8)
    return np.column_stack([original, decay, interaction])


def add_timing(daily, sessions, bc, c, input_hashes, out):
    fc = read(ROOT / bc["fundamentalConfig"])
    sids = sorted(set(np.concatenate([r["symbols"] for r in daily.values()])))
    ages = np.full((len(sessions), len(sids)), np.nan, np.float32)
    counts = {}
    for j, sid in enumerate(sids):
        path = ROOT / fc["fundamentals"]["root"] / (sid.replace(".", "_") + ".jsonl")
        rows, h = base.readiness.read_rows(path)
        if input_hashes.get(path.relative_to(ROOT).as_posix()) != h:
            raise ValueError("disclosure_input_changed:" + sid)
        ages[:, j], detail = disclosure_age(rows, sessions, sid, fc["families"])
        for k, v in detail.items():
            counts[k] = counts.get(k, 0) + v
        if (j + 1) % 500 == 0:
            save(out / "status.json", {"researchOnly": True, "state": "aligning_disclosures", "symbols": j + 1, "totalSymbols": len(sids), "orders": []})
            print(f"disclosure_alignment {j+1}/{len(sids)}", flush=True)
    lookup, enriched = {sid: j for j, sid in enumerate(sids)}, {}
    lag = c["features"]["counterLagSessions"]
    for i, r in daily.items():
        if i < lag:
            raise ValueError("missing_counter_history")
        ids = [lookup[sid] for sid in r["symbols"]]
        now, old = ages[i, ids], ages[i - lag, ids]
        original = r["features"]["interaction_model"]
        features = {**r["features"], "pooled_payoff": original,
                    "disclosure_timing": timing_matrix(original, now, c["features"]["halfLifeSessions"]),
                    "lagged_timing_counter": timing_matrix(original, old, c["features"]["halfLifeSessions"])}
        enriched[i] = {**r, "features": features, "disclosureAge": now}
    save(out / "disclosure_audit.json", {"researchOnly": True, "symbols": len(sids), "counts": counts,
         "unknownAgesImputed": 0, "rowsRemovedFromPriorSupport": 0, "availability": c["features"]["availability"]})
    return enriched


def verify_baseline(generated, reference):
    expected = reference.loc[reference.policy.isin(["equal_families", "interaction_model"])].copy()
    expected["policy"] = expected.policy.replace({"interaction_model": "pooled_payoff"})
    actual = generated.loc[generated.policy.isin(ARMS[:2])]
    keys = ["date", "policy", "rank", "securityId", "state"]
    sort = ["date", "policy", "rank"]
    expected, actual = (d.sort_values(sort).reset_index(drop=True) for d in [expected, actual])
    if not expected[keys].equals(actual[keys]):
        raise ValueError("frozen_baseline_selection_or_support_mismatch")
    for k in ["gross", "net", "pUp", "pTail", "expectedGross", "score"]:
        if not np.allclose(expected[k].to_numpy(float), actual[k].to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError("frozen_baseline_numeric_mismatch:" + k)
    return {"researchOnly": True, "matchedRows": len(actual), "identicalRankAndSecurity": True,
            "probabilityAndPayoffAbsoluteTolerance": 1e-12}


def probability_rows(row, date, predictions, bc):
    good = np.isfinite(row["y"])
    y = row["y"][good]
    output = []
    for arm in ARMS:
        p = predictions[arm]
        for task, truth, values in [("up", y > 0, p["p"]), ("tail", y <= bc["evaluation"]["tailThreshold"], p["pTail"])]:
            if len(values) != len(row["y"]) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError("invalid_probabilities_no_support_filtering")
            m = base.probability_metrics(truth, values[good])
            output.append({"researchOnly": True, "date": str(date.date()), "policy": arm, "task": task,
                           "eligible": len(row["y"]), **{k: v for k, v in m.items() if k != "buckets"}})
    return output


def joint_payoffs(frame, bc):
    counts = frame.groupby(["date", "policy"]).net.count().unstack().reindex(columns=ARMS)
    dates = counts.index[counts.eq(10).all(axis=1)]
    out = {"dates": list(dates), "jointCompleteDays": len(dates), "arms": {}, "researchOnly": True}
    for a in ARMS:
        r = frame.loc[frame.date.isin(dates) & frame.policy.eq(a)]
        out["arms"][a] = {"picks": len(r), "grossUp": float(r.gross.gt(0).mean()), "grossMean": float(r.gross.mean()),
                          "netMean": float(r.net.mean()), "tail": float(r.gross.le(bc["evaluation"]["tailThreshold"]).mean())}
    out["warning"] = "Same resolved dates, still conditional on future observability; not account PnL."
    return out


def selection_exposure(frame, sessions, bc):
    means = frame.groupby(["date", "policy"]).net.mean().unstack().reindex(columns=ARMS)
    counts = frame.groupby(["date", "policy"]).net.count().unstack().reindex(columns=ARMS)
    means = means.where(counts.eq(10)).dropna()
    observations = []
    for date in means.index:
        end = int(sessions.get_loc(pd.Timestamp(date))) - bc["training"]["purgeSessions"]
        prior = means.loc[means.index.isin(sessions[max(0, end - 63):end].strftime("%Y-%m-%d"))]
        if len(prior) >= bc["evaluation"]["minimumTrailingSelectorDays"]:
            observations.append((date, str(prior.mean().idxmax())))
    if not observations:
        return {"status": "insufficient_joint_complete_selector_days", "researchOnly": True}
    # Hindsight and trailing choice measured on exactly the same supported dates.
    paired = means.loc[[date for date, _ in observations]]
    hindsight = str(paired.mean().idxmax())
    trailing = np.array([means.loc[date, selected] for date, selected in observations])
    return {"researchOnly": True, "pairedDays": len(paired), "hindsightArm_NOT_VALID_STRATEGY": hindsight,
            "hindsightNet": float(paired[hindsight].mean()), "trailingOnlyNet": float(trailing.mean()),
            "overfittingExposure": float(paired[hindsight].mean() - trailing.mean())}


def evaluate(sessions, daily, bc, c, ref, out):
    """Real fitting, reference reproduction and reporting entry; input loader separate."""
    start = int(sessions.searchsorted(bc["evaluation"]["startDate"]))
    end = len(sessions) - bc["training"]["purgeSessions"]
    picks, metrics, folds = [], [], []
    original_picks = pd.read_csv(ref / "historical_picks.csv")
    for first in range(start, end, bc["training"]["refitSessions"]):
        block = [i for i in range(first, min(first + bc["training"]["refitSessions"], end)) if i in daily]
        if not block:
            continue
        artifact = read(ref / "models" / f"fold_{first}.json")
        if artifact["firstSignalIndex"] != first or artifact["researchOnly"] is not True:
            raise ValueError("wrong_reference_fold")
        models = {"equal_families": restore_model(artifact["arms"]["fundamental_model"]),
                  "pooled_payoff": restore_model(artifact["arms"]["interaction_model"])}
        train, cal = base.payoff.split_indices(first, bc)
        meta = {}
        for a in NEW_ARMS:
            with threadpool_limits(limits=1):
                models[a], meta[a] = base.fit(base.model_daily(daily, a), train, cal, bc)
        save(out / "models" / f"fold_{first}.json", {"researchOnly": True, "firstSignalIndex": first,
             "arms": {a: base.model_parameters(models[a]) for a in NEW_ARMS}})
        block_picks = []
        for i in block:
            row = daily[i]
            predictions = {a: base.predict(models[a], row["features"]["fundamental_model" if a == "equal_families" else a]) for a in ARMS}
            for a in ARMS:
                p = predictions[a]
                score = row["features"]["fundamental_model"].mean(axis=1) if a == "equal_families" else p["payoff_decomposition"] - bc["evaluation"]["roundTripCost"]
                chosen = base.payoff.top10(score)
                rr = base.payoff.record_picks(row, i, sessions[i], len(sessions) - 1, "reused_history", a, score, p, bc)
                for r, k in zip(rr, chosen):
                    r.update(pTail=float(p["pTail"][k]), expectedNet=r["expectedGross"] - bc["evaluation"]["roundTripCost"],
                             disclosureAge=int(row["disclosureAge"][k]))
                block_picks.extend(rr)
            metrics.extend(probability_rows(row, sessions[i], predictions, bc))
        block_frame = pd.DataFrame(block_picks)
        verify_baseline(block_frame, original_picks.loc[original_picks.date.isin(block_frame.date)])
        picks.extend(block_picks)
        folds.append({"firstSignalIndex": first, "firstSignalDate": str(sessions[first].date()), "newModels": meta})
        save(out / "folds.json", folds)
        save(out / "status.json", {"researchOnly": True, "state": "training_walk_forward", "folds": len(folds), "picks": len(picks), "orders": []})
        print(f"timing_fold {sessions[first].date()} folds={len(folds)} picks={len(picks)}", flush=True)
    if not picks:
        raise RuntimeError("no_valid_training_folds")
    frame, pm = pd.DataFrame(picks), pd.DataFrame(metrics)
    reproduction = verify_baseline(frame, original_picks)
    base.source.atomic_text(out / "historical_picks.csv", frame.to_csv(index=False))
    base.source.atomic_text(out / "same_support_probability_metrics.csv", pm.to_csv(index=False))
    tc = read(ROOT / bc["targetsConfig"])
    comparisons = {control: base.targets.assess_top10(frame, tc, "disclosure_timing", control, "reused_history") for control in ["equal_families", "pooled_payoff", "lagged_timing_counter"]}
    coverage = base.coverage_gate(comparisons, frame, sessions, daily, start, end)
    summary_metrics = []
    for (a, task), group in pm.groupby(["policy", "task"]):
        summary_metrics.append({"policy": a, "task": task, "days": len(group), "resolvedLabels": int(group["count"].sum()),
                               **group[["brier", "logLoss", "auc", "ece"]].mean().to_dict()})
    return {"researchOnly": True, "mayPromote": False, "orders": [], "state": "completed_diagnostic_only", "folds": len(folds),
            "baselineReproduction": reproduction, "evaluationCoverage": coverage,
            "arms": {a: base.summary(frame.loc[frame.policy.eq(a)], bc) for a in ARMS},
            "jointCompleteComparison": joint_payoffs(frame, bc), "sameSupportDailyMeanProbabilityMetrics": summary_metrics,
            "comparisons": comparisons, "hindsightVsTrailing": selection_exposure(frame, sessions, bc)}


def run(config_path, run_id):
    c = read(config_path)
    validate(c)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid_run_id")
    bc, ref, frozen = reference_contract(c)
    out = ROOT / c["outputRoot"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"researchOnly": True, "runId": run_id, "createdAt": base.source.now(), "config": c,
                "configSha256": base.pit.digest(config_path), "codeSha256": base.pit.digest(Path(__file__)),
                "codeCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "codeDirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "frozenReferenceHashes": frozen, "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "orders": []}
    save(out / "manifest.json", manifest)
    try:
        sessions, daily, audit = base.load_data(bc, out)
        inputs = normalized_hashes(read(out / "input_hashes.json"))
        if inputs != normalized_hashes(read(ref / "input_hashes.json")):
            raise ValueError("different_input_support_from_frozen_reference")
        daily = add_timing(daily, sessions, bc, c, inputs, out)
        result = evaluate(sessions, daily, bc, c, ref, out)
        for p, h in frozen.items():
            if base.pit.digest(ROOT / p) != h:
                raise ValueError("frozen_artifact_changed_during_training:" + p)
        if base.pit.digest(config_path) != manifest["configSha256"] or base.pit.digest(Path(__file__)) != manifest["codeSha256"]:
            raise ValueError("experiment_code_or_config_changed_during_training")
        result.update(runId=run_id, codeCommit=manifest["codeCommit"], dataAudit={k: v for k, v in audit.items() if k != "dailyCoverage"},
                      trials={**c["trials"], "globalLowerBoundAfterRun": c["trials"]["globalPriorLowerBound"] + len(ARMS),
                              "newPerMechanismFamily": {f: len(ARMS) for f in read(ROOT / bc["fundamentalConfig"])["families"]}},
                      dsr={"value": None, "reason": "incomplete_global_and_family_trial_history"},
                      limitations=["Reused historical windows and current vendor statement vintages; not fresh OOS.",
                                   "Availability age is not consensus earnings surprise or verified investor attention.",
                                   "Both controls reproduced exactly; unknown age cannot remove unfavorable samples.",
                                   "SH/SZ qualified subset only; missing historical BJ status and recent support remain excluded.",
                                   "Unresolved legs remain in denominators. Conditional complete-day payoffs are not account PnL.",
                                   "Same-label probability comparisons use daily means, not independent stock-row significance.",
                                   "No event timestamp/queue replay, capacity certification, neutrality or placebo distribution.",
                                   "No hyperparameter scan, old twelve-factor reweighting, forward-model edits or trading."])
        save(out / "result.json", result)
        lines = ["# Statement-recency historical ablation", "", "Research-only; no trading, promotion or current-stock recommendation.", "",
                 f"Run: `{run_id}`; commit: `{manifest['codeCommit']}`.", "",
                 f"Jointly complete comparison dates: {result['jointCompleteComparison']['jointCompleteDays']}.", "",
                 "| Arm | Gross-up rate | Mean gross/pick | Mean net/pick | Tail <= -3% |", "|---|---:|---:|---:|---:|"]
        for a, r in result["jointCompleteComparison"]["arms"].items():
            lines.append(f"| {a} | {r['grossUp']:.2%} | {r['grossMean']:.4%} | {r['netMean']:.4%} | {r['tail']:.2%} |")
        lines += ["", "Conditional on resolution, not a deployable portfolio. See result.json for all-selected bounds, strict gates, identical-label calibration and hindsight/trailing exposure.", ""]
        lines += ["- " + s for s in result["limitations"]]
        base.source.atomic_text(out / "report.md", "\n".join(lines) + "\n")
        save(out / "status.json", {"researchOnly": True, "state": result["state"], "folds": result["folds"], "orders": []})
        print(json.dumps({"runId": run_id, "state": result["state"], "folds": result["folds"], "mayPromote": False}), flush=True)
        return result
    except Exception as exc:
        save(out / "status.json", {"researchOnly": True, "state": "failed_closed", "error": f"{type(exc).__name__}:{exc}", "orders": []})
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/research/sina_disclosure_timing_v1.json"))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run(args.config, args.run_id)
