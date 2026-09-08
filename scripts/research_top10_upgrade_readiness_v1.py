"""Audit the new input contract and user targets; no collection, fitting or orders.

--diagnostic-picks can measure an EXISTING historical run, never certify it as a
corrected-input result. CLI exits 2 when prerequisites are missing, even if the
diagnostic numeric targets happen to pass. All output lives in a fresh run dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import research_top10_strict_inputs_v1 as strict
from research_top10_joint_weight_audit_v1 import save

ROOT = strict.ROOT
DEFAULT = ROOT / "configs/research/top10_strict_upgrade_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(c):
    if c.get("schemaVersion") != "top10_strict_upgrade_v1" or c.get("status") != "research_only_shadow_only_not_trading":
        raise ValueError("research-only schema required")
    expected = {"mayTrade", "mayPublishDashboard", "mayChangeExistingConfig", "mayChangeForwardRecord",
                "mayPromoteAutomatically", "mayStretchProbabilities"}
    safety = c.get("safety", {})
    if not expected.issubset(safety) or any(safety[k] is not False for k in expected) or safety.get("ordersAlwaysEmpty") is not True:
        raise ValueError("unsafe_or_incomplete_permissions")
    t = c["targets"]
    if (t["topCount"] != 10 or t["dailyGuarantee"] is not False or t["roundTripCost"] != .003
            or t["pairedMeanNetReturnLift"] != .01 or t["realizedTop10UpRateGreaterThan"] != .5
            or t["individualPredictedUpProbabilityGreaterThan"] != .5
            or t["requiredPairedResolvedDayFraction"] != 1):
        raise ValueError("frozen_target_contract_changed")
    if t["minimumIndependentDays"] < 20 or t["blockBootstrapSessions"] < 7 or t["bootstrapReplications"] < 1000:
        raise ValueError("weak_target_uncertainty_contract")
    if not 0 < t["lowerConfidenceQuantile"] <= .05:
        raise ValueError("invalid_confidence_quantile")
    d = c["priceData"]
    if (d["requireOfficialSameSessionRawPair"] is not True
            or d["allowCrossProviderScaleInference"] is not False
            or d["allowProxyOrMissingFactorImputation"] is not False
            or d["requiredContiguousFactorInputSessions"] != strict.STRICT_HISTORY_SESSIONS):
        raise ValueError("strict_price_contract_changed")
    if c["outputRoot"] != "outputs/edge_research/top10_strict_upgrade_v1":
        raise ValueError("isolated_output_namespace_required")
    if not (ROOT / c["outputRoot"]).resolve().is_relative_to((ROOT / "outputs/edge_research").resolve()):
        raise ValueError("research_output_only")
    return c


def lower_bound(values, t):
    """Circular moving-block bootstrap of equal-weighted paired DAY metrics."""
    x = np.asarray(values, dtype=float)
    if len(x) < t["minimumIndependentDays"] or not np.isfinite(x).all():
        return None
    n, length = len(x), t["blockBootstrapSessions"]
    rng = np.random.default_rng(t["seed"])
    starts = rng.integers(0, n, size=(t["bootstrapReplications"], (n + length - 1) // length))
    ids = ((starts[..., None] + np.arange(length)) % n).reshape(len(starts), -1)[:, :n]
    return float(np.quantile(x[ids].mean(axis=1), t["lowerConfidenceQuantile"]))


def assess_top10(picks, c, candidate, baseline, period):
    """Fixed quota, gross-up != net-up, same dates; unresolved is NEVER zero PnL."""
    t = c["targets"]
    block = picks.loc[picks.period.eq(period) & picks.policy.isin([candidate, baseline])].copy()
    if candidate == baseline or block.empty:
        raise ValueError("invalid_comparison")
    books = {}
    for name in [candidate, baseline]:
        frame = block.loc[block.policy.eq(name)].copy()
        if frame.empty or frame.duplicated(["date", "securityId"]).any():
            raise ValueError("missing_policy_or_duplicate_pick")
        grouped = frame.groupby("date", sort=True)
        if not grouped.size().eq(10).all() or not grouped["rank"].apply(lambda s: set(s) == set(range(1, 11))).all():
            raise ValueError("not_exactly_ten_preselected_names")
        p = frame.pUp.to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("invalid_probabilities_no_fallback")
        resolved = frame.state.eq("resolved")
        if not np.isfinite(frame.loc[resolved, ["gross", "net"]].to_numpy(float)).all():
            raise ValueError("nonfinite_resolved_outcome")
        if not np.allclose(frame.loc[resolved, "net"], frame.loc[resolved, "gross"] - t["roundTripCost"], atol=1e-10, rtol=0):
            raise ValueError("inconsistent_return_cost_basis")
        frame["resolved"] = resolved
        # Unknown exits may not enter return means. A daily return needs ALL ten.
        frame["gross"] = frame.gross.where(resolved)
        frame["net"] = frame.net.where(resolved)
        frame["win"] = frame.gross.gt(0) & resolved
        frame["tail"] = frame.gross.le(t["tailThreshold"]) & resolved
        day = frame.groupby("date", sort=True).agg(resolved=("resolved", "sum"), gross=("gross", "mean"),
                    net=("net", "mean"), win=("win", "mean"), tail=("tail", "mean"))
        day.loc[day.resolved.ne(10), ["gross", "net", "tail"]] = np.nan
        books[name] = (frame, day)
    cf, cd = books[candidate]
    _, bd = books[baseline]
    if not cd.index.equals(bd.index):
        raise ValueError("comparison_day_support_mismatch")
    complete = cd.resolved.eq(10) & bd.resolved.eq(10)
    good, control = cd.loc[complete], bd.loc[complete]
    resolved = cf.loc[cf.resolved]
    y, p = resolved.gross.gt(0).to_numpy(), resolved.pUp.to_numpy(float)
    ece = 0.
    buckets = []
    for k in range(10):
        sel = np.minimum((p * 10).astype(int), 9) == k
        if sel.any():
            hit, pred = float(y[sel].mean()), float(p[sel].mean())
            ece += float(sel.sum()) / len(p) * abs(hit - pred)
            buckets.append({"lower": k / 10, "upper": (k + 1) / 10, "count": int(sel.sum()),
                            "predicted": pred, "hitRate": hit})
    lift = good.net - control.net
    win_lower, lift_lower = lower_bound(good.win, t), lower_bound(lift, t)
    mean = lambda s: float(s.mean()) if len(s) else None
    requirements = {
        "allTenIndividualProbabilitiesAboveHalf": bool(cf.pUp.gt(.5).all()),
        "enoughIndependentDays": int(complete.sum()) >= t["minimumIndependentDays"],
        "allPairedDaysResolved": bool(complete.all()),
        "top10WinLowerBoundAboveHalf": win_lower is not None and win_lower > .5,
        "pairedNetLiftAtLeastOnePercentagePoint": bool(len(lift)) and float(lift.mean()) >= .01,
        "pairedNetLiftLowerBoundAboveZero": lift_lower is not None and lift_lower > 0,
        "netReturnPositive": bool(len(good)) and float(good.net.mean()) > 0,
        "tailRateNotWorse": bool(len(good)) and float(good["tail"].mean()) <= float(control["tail"].mean()),
    }
    return {"researchOnly": True, "period": period, "candidate": candidate, "baseline": baseline,
            "signalDays": len(cd), "pairedCompleteDays": int(complete.sum()),
            "selectedNamesPerDay": 10, "candidateResolvedPicks": len(resolved),
            "winRateOnResolvedPicks": float(y.mean()) if len(y) else None,
            "allSelectedWinRateBounds": [float(cf.win.mean()), float((cf.win | ~cf.resolved).mean())],
            "pairedCompleteDayWinRate": mean(good.win), "pairedCompleteDayGross": mean(good.gross),
            "pairedCompleteDayNet": mean(good.net), "pairedNetLift": mean(lift),
            "winRateLowerBound": win_lower, "pairedLiftLowerBound": lift_lower,
            "daysWithAtLeastSixKnownWinners": int((cd.win > .5).sum()),
            "predictedProbabilityRange": [float(cf.pUp.min()), float(cf.pUp.max())],
            "brier": float(brier_score_loss(y, p)) if len(y) else None,
            "logLoss": float(log_loss(y, p, labels=[False, True])) if len(y) else None,
            "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "ece": ece if len(y) else None, "probabilityBuckets": buckets,
            "requirements": requirements, "numericalTargetsMet": all(requirements.values()),
            "mayPromote": False, "orders": [], "dailyGuarantee": False,
            "limitations": ["Paired-complete returns are conditional on all legs resolving; excluded days remain counted.",
                            "Pooled-pick calibration is descriptive; uncertainty uses blocks of days, not independent stocks.",
                            "No multiplicity correction or fresh OOS certificate is implied by these intervals.",
                            "One percentage point is a cohort return lift, not an account or daily guaranteed return."]}


def run(config_path, run_id, diagnostic_picks=None, period="shadow", candidate="payoff_decomposition", baseline="guarded16"):
    c = validate(json.loads(Path(config_path).read_text(encoding="utf-8")))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("unsafe_run_id")
    out = ROOT / c["outputRoot"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(config_path), Path(__file__), Path(strict.__file__)]
    hashes = {str(p.resolve()): digest(p) for p in files}
    manifest = {"researchOnly": True, "runId": run_id, "startedAt": datetime.now(timezone.utc).isoformat(),
                "gitCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "gitDirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "hashes": hashes, "status": "running", "orders": [],
                "diagnosticComparison": {"path": str(diagnostic_picks) if diagnostic_picks else None,
                                         "period": period, "candidate": candidate, "baseline": baseline}}
    save(out / "manifest.json", manifest)
    try:
        d = c["priceData"]
        prices = strict.audit_prices(ROOT / d["adjustedRoot"], ROOT / d["rawCompanionRoot"],
                                     d["startDate"], d["endDate"], d["minimumCrossSection"])
        save(out / "price_audit.json", prices)
        target = None
        if diagnostic_picks is not None:
            old_hash = digest(diagnostic_picks)
            target = assess_top10(pd.read_csv(diagnostic_picks), c, candidate, baseline, period)
            if old_hash != digest(diagnostic_picks):
                raise strict.ContractError("diagnostic_input_changed_during_read")
            target.update(sourcePath=str(Path(diagnostic_picks).resolve()), sourceSha256=old_hash,
                          correctedInputModel=False, evidenceUse="existing_historical_diagnostic_only")
            save(out / "existing_run_target_diagnostic.json", target)
        blockers = []
        if not prices["daysWithMinimumPairedCrossSection"]:
            blockers.append("no_day_with_300_verified_same_source_raw_adjusted_pairs")
        if prices["fileErrors"]:
            blockers.append("invalid_price_files_see_price_audit")
        # A price audit cannot silently turn quarterly ratios into verified news.
        blockers.append("timestamped_stock_event_archive_and_expectation_vintages_not_yet_qualified")
        blockers.append("frozen_factor_formula_and_internal_missing_value_semantics_not_yet_certified")
        result = {"researchOnly": True, "status": "blocked_data_prerequisites", "runId": run_id,
                  "blockers": blockers, "newModelTrained": False, "modelsPromoted": 0,
                  "targetGuarantee": False, "orders": [], "mayPublishDashboard": False,
                  "existingHistoricalTargetsMet": target["numericalTargetsMet"] if target else None,
                  "exitCode": 2}
        save(out / "result.json", result)
        lines = ["# Top10 strict-input upgrade readiness", "", "RESEARCH ONLY. No fitting, publication or orders.", "",
                 f"Run: {run_id}; commit: {manifest['gitCommit']}; dirty: {manifest['gitDirty']}", "",
                 f"Price files: {prices['filesScanned']}; audited rows: {prices['rows']}; verified VWAP pairs: {prices['verifiedPairedVwapRows']}.",
                 f"Legacy raw VWAP outside adjusted daily range: {prices['legacyRawVwapOutsideAdjustedRangeRows']} rows.", "",
                 "## Blockers", ""] + ["- " + s for s in blockers]
        if target:
            lines += ["", "## Existing run only — not a new corrected model", "",
                      f"Numeric targets met: {target['numericalTargetsMet']}; paired complete days: {target['pairedCompleteDays']}/{target['signalDays']}.",
                      f"Resolved-pick up rate: {target['winRateOnResolvedPicks']}; paired net lift: {target['pairedNetLift']}."]
        lines += ["", "All of >50% and +1 percentage point are rejection targets, not guarantees.",
                  "No price-volume rescan or new model search was started.", ""]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        if hashes != {str(p.resolve()): digest(p) for p in files}:
            raise strict.ContractError("code_or_config_changed_during_run")
        manifest.update(status="completed_audit_blocked_model", finishedAt=datetime.now(timezone.utc).isoformat(), exitCode=2)
        save(out / "manifest.json", manifest)
        print(json.dumps({"output": str(out), **result}), flush=True)
        return result
    except Exception as exc:
        manifest.update(status="failed_closed", error=f"{type(exc).__name__}:{exc}", exitCode=1)
        save(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--diagnostic-picks", type=Path)
    parser.add_argument("--period", default="shadow")
    parser.add_argument("--candidate", default="payoff_decomposition")
    parser.add_argument("--baseline", default="guarded16")
    args = parser.parse_args()
    result = run(args.config, args.run_id, args.diagnostic_picks, args.period, args.candidate, args.baseline)
    raise SystemExit(result["exitCode"])
