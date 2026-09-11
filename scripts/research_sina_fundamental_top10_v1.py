"""Versioned, historical-only Sina/fundamental Top10 comparison. Never trades.

No legacy factor implementation, forecast, cache or forward record is modified.
This is a diagnostic new book, NOT a corrected or promoted legacy twelve book.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import platform
import re
import subprocess

import exchange_calendars as xc
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

import audit_sina_fundamental_readiness_v1 as readiness
import collect_sina_research_daily_v1 as source
import research_ashare_pit_panel_v2 as pit
import research_fundamental_mechanism_families as fundamentals
import research_top10_payoff_decomposition_v1 as payoff
import research_top10_upgrade_readiness_v1 as targets

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["equal_families", "fundamental_model", "context_model", "interaction_model", "lagged_interaction_counter"]
CONTEXTS = ["momentum20", "reversal5", "low_volatility20", "volume5_vs20"]


def save(path, obj):
    source.atomic_json(path, fundamentals.json_safe(obj))


def validate(c):
    if c["schemaVersion"] != "sina_fundamental_top10_v1" or c["researchOnly"] is not True:
        raise ValueError("research_only_required")
    if c["arms"] != ARMS or any(v for k, v in c["safety"].items() if k.startswith("may")):
        raise ValueError("frozen_arms_and_no_mutation_required")
    if c["safety"].get("ordersAlwaysEmpty") is not True:
        raise ValueError("orders_forbidden")
    e, t = c["evaluation"], c["training"]
    if (e["topCount"] != 10 or e["roundTripCost"] != .003 or e["holdingSessions"] != 1
            or not e["excludeCorporateActionLegs"] or not e["historicalWindowsAlreadyViewed"] or not e["neverPromote"]):
        raise ValueError("fixed_execution_contract_required")
    if t["purgeSessions"] < 1 + e["holdingSessions"] + e["maximumExitDelay"]:
        raise ValueError("purge_shorter_than_label_maturity")
    if not (ROOT / c["outputRoot"]).resolve().is_relative_to((ROOT / "outputs/edge_research").resolve()):
        raise ValueError("isolated_output_required")


def indexed(rows, sid):
    out = {}
    for r in rows:
        if r.get("securityId") != sid or source.iso_date(r.get("dt")) != r.get("dt"):
            raise ValueError("invalid_identity_or_date:" + sid)
        if r["dt"] in out:
            raise ValueError("duplicate_date:" + sid)
        out[r["dt"]] = r
    return out


def close_value(actual, expected):
    """Scalar equivalent of finite np.isclose(rtol=1e-8, atol=1e-8).

    Avoid constructing millions of temporary NumPy arrays during input checks.
    This changes no model, threshold, candidate or successful finite comparison.
    """
    return (source.finite(actual) and source.finite(expected)
            and abs(actual - expected) <= 1e-8 + 1e-8 * abs(expected))


def symbol_frame(raw_rows, adj_rows, status_rows, master, sessions):
    """Same-vendor price arithmetic; status-only exact-date cross-provider join.

    Opening eligibility never consults that session's high/low/close/volume.
    Unknown or changed adjustment basis is unresolved, not an imputed return.
    """
    sid = master["securityId"]
    raw, adj, status = (indexed(rows, sid) for rows in (raw_rows, adj_rows, status_rows))
    if set(raw) != set(adj):
        raise ValueError("raw_adjusted_dates_mismatch:" + sid)
    listing = pd.to_datetime(master.get("listingDate"), errors="coerce")
    if master.get("pointInTimeMembership") is not True or pd.isna(listing):
        raise ValueError("unverified_membership:" + sid)
    data = []
    for dt in sessions.strftime("%Y-%m-%d"):
        row = {"dt": dt}
        member = dt >= str(listing.date()) and (not master.get("delistingDate") or dt <= master["delistingDate"])
        row["membership"] = bool(member)
        s = status.get(dt, {})
        if readiness.trusted_status(s, sid):
            row.update(is_st=s["isST"], trade_status=s["tradeStatus"])
        r, a = raw.get(dt), adj.get(dt)
        if r is not None:
            if (r.get("source") != source.SOURCE or a.get("source") != source.SOURCE
                    or r.get("adjustment") != "none_raw_sina"
                    or a.get("adjustment") != "same_vendor_hfq_factor_effective_date_asof_no_future_factor"
                    or any(z.get("volumeUnit") != "shares" or z.get("amountUnit") != "CNY" for z in (r, a))):
                raise ValueError("source_or_units_mismatch:" + sid)
            factor_date = source.iso_date(a["factorEffectiveDate"]) if a.get("factorEffectiveDate") is not None else None
            factor = a.get("factor")
            factor_ok = source.finite(factor, True) and factor_date is not None and factor_date <= dt
            if factor_ok:
                row["factor"] = factor
                if source.finite(r.get("open"), True):
                    row["raw_open"] = r["open"]
                if source.finite(r.get("close"), True):
                    row["raw_close"] = r["close"]
            if r.get("priceBasisValid") is True and a.get("priceBasisValid") is True:
                if not factor_ok:
                    raise ValueError("future_or_missing_adjustment_factor:" + sid)
                for k in source.OHLC:
                    if not source.finite(r.get(k), True) or not close_value(a[k], r[k] * factor):
                        raise ValueError("adjustment_arithmetic_mismatch:" + sid)
                if not (r["low"] <= min(r["open"], r["close"]) <= max(r["open"], r["close"]) <= r["high"]):
                    raise ValueError("invalid_ohlc:" + sid)
                if (not source.finite(r.get("amount"), True) or not source.finite(r.get("volume"), True)
                        or r["volume"] != a["volume"] or r["amount"] != a["amount"]):
                    raise ValueError("flow_mismatch:" + sid)
                vwap = r["amount"] / r["volume"]
                tol = .0001 + r["high"] * 1e-5
                if not r["low"] - tol <= vwap <= r["high"] + tol or not close_value(a["vwap"], vwap * factor):
                    raise ValueError("invalid_vwap_basis:" + sid)
                row.update(close=a["close"], volume=r["volume"], amount=r["amount"])
        data.append(row)
    frame = pd.DataFrame(data).set_index("dt").reindex(columns=[
        "membership", "is_st", "trade_status", "factor", "raw_open", "raw_close", "close", "volume", "amount"])
    frame.index = sessions
    return frame


def feature_arrays(frame, statements, sid, fc, c):
    """Bounded disclosure intervals reset missing fields instead of old-value fill."""
    events, audit = fundamentals.causal_fundamental_records_for_symbol(statements, frame.index, sid, fc["families"])
    family = np.full((len(frame), 4), np.nan, dtype=np.float32)
    for i, event in enumerate(events):
        start = int(frame.index.get_loc(event["eventDate"]))
        nxt = int(frame.index.get_loc(events[i + 1]["eventDate"])) if i + 1 < len(events) else len(frame)
        end = min(nxt, start + c["features"]["maxFundamentalAgeSessions"] + 1)
        for j, spec in enumerate(fc["families"].values()):
            values = [event[d["id"]] for d in spec["candidates"]]
            if all(source.finite(v) for v in values):
                family[start:end, j] = np.mean(values)
    p, v, f = frame.close, frame.volume, c["features"]
    r = p.pct_change(fill_method=None)
    contexts = np.column_stack([
        p.div(p.shift(f["momentumWindow"])) - 1,
        -(p.div(p.shift(f["reversalWindow"])) - 1),
        -r.rolling(f["volatilityWindow"], min_periods=f["volatilityWindow"]).std(),
        v.rolling(f["volumeShortWindow"], min_periods=f["volumeShortWindow"]).mean()
        / v.rolling(f["volumeLongWindow"], min_periods=f["volumeLongWindow"]).mean() - 1])
    # No price padding across missing sessions, including momentum endpoints.
    contexts[~p.notna().rolling(21, min_periods=21).sum().eq(21).to_numpy()] = np.nan
    lag = np.full_like(family, np.nan)
    n = f["counterLagSessions"]
    if n < len(family):
        lag[n:] = family[:-n]
    return np.column_stack([family, contexts, lag]).astype(np.float32), len(events), audit


def labels(frame, sid, c):
    """t close -> t+1 open -> t+2 or delayed sellable open. No lookahead fills."""
    op, fac = frame.raw_open.to_numpy(float), frame.factor.to_numpy(float)
    prev = frame.raw_close.shift(1).to_numpy(float)
    same = (frame.factor.eq(frame.factor.shift(1)) & frame.factor.notna()).to_numpy()
    lim = np.full(len(frame), .1)
    code = sid.split(".")[1]
    if code.startswith("688"):
        lim[:] = .2
    elif code.startswith(("300", "301")):
        lim[frame.index >= "2020-08-24"] = .2
    # Only non-ST rows enter the eligible execution subset; later ST exits stay
    # unresolved instead of guessing a historical rule transition or queue fill.
    known = (frame.membership & frame.trade_status.eq(1) & frame.is_st.eq(0)).to_numpy(bool)
    ratio = np.divide(op, prev, out=np.full_like(op, np.nan), where=prev > 0) - 1
    known &= same & np.isfinite(ratio) & np.isfinite(op) & (op > 0)
    buy = known & (ratio < lim - c["evaluation"]["openLimitBuffer"])
    sell = known & (ratio > -lim + c["evaluation"]["openLimitBuffer"])
    entry, y, delay = np.zeros(len(frame), bool), np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    entry[:-1] = buy[1:]
    for i in np.flatnonzero(entry):
        for j in range(i + 2, min(len(frame), i + 3 + c["evaluation"]["maximumExitDelay"])):
            # A changed/unknown factor anywhere in a leg invalidates that label.
            if not np.isfinite(fac[i + 1:j + 1]).all() or not (fac[i + 1:j + 1] == fac[i + 1]).all():
                break
            if sell[j]:
                y[i], delay[i] = op[j] / op[i + 1] - 1, j - i - 2
                break
    return y, entry, delay


def rank_features(raw):
    """One identical complete cross-section for every arm; no return filtering."""
    if not np.isfinite(raw).all() or raw.shape[1] != 12:
        raise ValueError("incomplete_feature_support")
    ranks = pd.DataFrame(raw).rank(pct=True).to_numpy(np.float32) - .5
    f, ctx, lag = ranks[:, :4], ranks[:, 4:8], ranks[:, 8:12]
    # Positive earnings*volume, growth*momentum, quality*low-vol, cash*reversal.
    match = ctx[:, [3, 0, 2, 1]]
    current = np.column_stack([f, ctx, f * match])
    counter = np.column_stack([f, ctx, lag * match])
    return {"fundamental_model": f, "context_model": current[:, :8],
            "interaction_model": current, "lagged_interaction_counter": counter}


def load_data(c, out):
    origin = ROOT / "outputs/edge_research/sina_daily_v1" / c["sourceRun"]
    manifest = json.loads((origin / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((origin / "result.json").read_text(encoding="utf-8"))
    if status["state"] not in ("completed", "completed_with_gaps") or status["attemptedSymbols"] != status["requestedSymbols"]:
        raise ValueError("full_collection_not_terminal")
    audit_path = ROOT / "outputs/edge_research/sina_fundamental_readiness_v1" / c["inputAuditRun"] / "result.json"
    input_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if input_audit["collectorManifestSha256"] != pit.digest(origin / "manifest.json"):
        raise ValueError("audit_source_mismatch")
    fc = json.loads((ROOT / c["fundamentalConfig"]).read_text(encoding="utf-8"))
    if len(fc["families"]) != 4:
        raise ValueError("four_frozen_families_required")
    sessions = xc.get_calendar("XSHG", start=c["panel"]["startDate"], end=c["panel"]["endDate"]).sessions
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    master, master_hash = readiness.read_rows(ROOT / c["masterPath"])
    if {"path": c["masterPath"], "sha256": master_hash} not in manifest["contract"]["masterSources"]:
        raise ValueError("master_not_in_frozen_collection_contract")
    if (input_audit["familyConfigSha256"] != pit.digest(ROOT / c["fundamentalConfig"])
            or input_audit["familyModuleSha256"] != pit.digest(Path(fundamentals.__file__))):
        raise ValueError("fundamental_contract_changed_since_audit")
    audited_hashes = {r["path"].replace("\\", "/"): r["sha256"] for r in input_audit["sourceFiles"]}
    master = sorted([r for r in master if r.get("exchange") in ("SH", "SZ") and r.get("pointInTimeMembership") is True], key=lambda r: r["securityId"])
    symbols = [r["securityId"] for r in master]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate_master_identity")
    shape = (len(sessions), len(symbols))
    features = np.full((*shape, 12), np.nan, np.float32)
    outcomes, delays = np.full(shape, np.nan), np.full(shape, np.nan)
    entries, eligible = np.zeros(shape, bool), np.zeros(shape, bool)
    records, counts, inputs = [], Counter(), [{"path": c["masterPath"], "sha256": master_hash}]
    price_root = ROOT / manifest["config"]["dataRoot"] / c["sourceRun"]
    for j, row in enumerate(master):
        sid, stem = row["securityId"], row["securityId"].replace(".", "_")
        record_path = origin / "symbols" / (stem + ".json")
        item = {"securityId": sid}
        if not record_path.exists():
            item["skip"] = "no_collected_symbol"
        else:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["state"] != "collected":
                item["skip"] = "collector_failed"
            else:
                status_path = ROOT / c["statusRoot"] / (stem + ".jsonl")
                funda_path = ROOT / fc["fundamentals"]["root"] / (stem + ".jsonl")
                if not status_path.exists() or not funda_path.exists():
                    item["skip"] = "missing_dated_status_or_fundamentals"
                else:
                    source.verify_cached(record, price_root)
                    pieces = []
                    for p in [price_root / "raw" / (stem + ".jsonl"), price_root / "adjusted" / (stem + ".jsonl"), status_path, funda_path]:
                        rr, hh = readiness.read_rows(p)
                        if p in (status_path, funda_path) and audited_hashes.get(p.relative_to(ROOT).as_posix()) != hh:
                            raise ValueError("dated_input_changed_since_full_audit:" + sid)
                        pieces.append(rr)
                        inputs.append({"path": str(p.relative_to(ROOT)), "sha256": hh})
                    frame = symbol_frame(*pieces[:3], row, sessions)
                    features[:, j], ne, fa = feature_arrays(frame, pieces[3], sid, fc, c)
                    small_panel = {k: frame[[k]].rename(columns={k: sid}) for k in ["close", "amount", "volume", "membership", "is_st", "trade_status"]}
                    eligible[:, j] = pit.eligibility(small_panel, c["panel"])[sid].to_numpy()
                    outcomes[:, j], entries[:, j], delays[:, j] = labels(frame, sid, c)
                    item.update(events=ne, eligibleDays=int(eligible[:, j].sum()), datedStatusRows=int(frame.trade_status.notna().sum()), eventAudit=fa)
                    counts["loadedSymbols"] += 1
                    counts["disclosureEvents"] += ne
                    counts["datedStatusRows"] += item["datedStatusRows"]
        counts[item.get("skip", "loaded")] += 1
        records.append(item)
        if (j + 1) % 100 == 0:
            save(out / "status.json", {"researchOnly": True, "state": "building_panel", "loaded": j + 1, "symbols": len(master), "orders": []})
            print(f"qualified_input {j+1}/{len(master)}", flush=True)
    support = eligible & np.isfinite(features).all(axis=2)
    daily, coverage = {}, []
    for i, date in enumerate(sessions):
        ids = np.flatnonzero(support[i])
        coverage.append({"date": str(date.date()), "eligible": int(eligible[i].sum()), "complete": len(ids)})
        if len(ids) >= c["panel"]["minimumCrossSection"]:
            arms = rank_features(features[i, ids])
            daily[i] = {"features": arms, "symbols": np.asarray(symbols)[ids], "y": outcomes[i, ids],
                        "entry": entries[i, ids], "delay": delays[i, ids]}
    save(out / "input_hashes.json", inputs)
    source.atomic_jsonl(out / "symbol_quality.jsonl", records)
    meta = {"counts": dict(counts), "masterSymbols": len(master), "requestedCollectionUniverse": status["requestedSymbols"],
            "qualifiedExchanges": ["SH", "SZ"], "beijingSkipped": "no_verified_historical_status_and_membership",
            "dataRange": [str(sessions[0].date()), str(sessions[-1].date())], "dailyCoverage": coverage,
            "completeDays": len(daily), "currentVendorVintagesNotHistoricallyArchived": True,
            "researchOnly": True, "noLegacyFactorFormulaUsed": True, "tradingCertification": False}
    save(out / "data_audit.json", meta)
    del features, outcomes, delays, entries, eligible, support
    gc.collect()
    return sessions, daily, meta


def model_daily(daily, arm):
    return {i: {**r, "x": r["features"][arm]} for i, r in daily.items()}


def fit(daily, train, cal, c):
    model, meta = payoff.fit_models(daily, train, cal, c)
    x, y, w, _ = payoff.gather(daily, train, c)
    cx, cy, cw, _ = payoff.gather(daily, cal, c)
    z = (x - model["mean"]) / model["std"]
    cz = (cx - model["mean"]) / model["std"]
    tail, ctail = y <= c["evaluation"]["tailThreshold"], cy <= c["evaluation"]["tailThreshold"]
    if len(np.unique(tail)) != 2 or len(np.unique(ctail)) != 2:
        raise RuntimeError("insufficient_tail_classes")
    model["tail"] = LogisticRegression(C=c["training"]["logisticC"], max_iter=500).fit(z, tail, sample_weight=w)
    model["tail_cal"] = LogisticRegression(C=1., max_iter=500).fit(model["tail"].decision_function(cz)[:, None], ctail, sample_weight=cw)
    if any(model[k].n_iter_.max() >= 500 for k in ("tail", "tail_cal")):
        raise RuntimeError("tail_fit_not_converged")
    meta["upCoefficientsStandardized"] = model["up"].coef_[0].tolist()
    meta["returnCoefficientsStandardized"] = model["direct"].coef_.tolist()
    meta["featureMean"] = model["mean"].tolist()
    meta["featureStd"] = model["std"].tolist()
    return model, meta


def predict(model, x):
    p = payoff.predict(model, x)
    z = (x - model["mean"]) / model["std"]
    p["pTail"] = model["tail_cal"].predict_proba(model["tail"].decision_function(z)[:, None])[:, 1]
    return p


def model_parameters(model):
    """Research-only portable numeric artifact; no executable pickle payload."""
    out = {}
    for k, v in model.items():
        if hasattr(v, "coef_"):
            out[k] = {"class": type(v).__name__, "coef": np.asarray(v.coef_).tolist(),
                      "intercept": np.asarray(v.intercept_).tolist()}
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (int, float, str)):
            out[k] = v
    return {"researchOnly": True, "onlineInferenceAllowed": False, "parameters": out}


def probability_metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[mask].astype(int), np.asarray(p)[mask]
    if not len(y):
        return {"count": 0}
    buckets, ece = [], 0.
    for b in range(10):
        use = np.minimum((10 * p).astype(int), 9) == b
        if use.any():
            hit, mean = float(y[use].mean()), float(p[use].mean())
            ece += use.mean() * abs(hit - mean)
            buckets.append({"lower": b / 10, "count": int(use.sum()), "meanProbability": mean, "hitRate": hit})
    return {"count": len(y), "brier": float(brier_score_loss(y, p)), "logLoss": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None, "ece": ece, "buckets": buckets}


def summary(picks, c):
    groups = picks.groupby("date", sort=True)
    if not groups.size().eq(10).all():
        raise ValueError("not_exactly_ten_selections")
    resolved = picks.state.eq("resolved")
    n, k = len(picks), int(resolved.sum())
    good = picks.loc[resolved]
    complete = groups.state.apply(lambda s: s.eq("resolved").all())
    days = picks.loc[picks.date.isin(complete[complete].index)].groupby("date").net.mean()
    return {"selected": n, "resolved": k, "resolutionFraction": k / n, "signalDays": len(groups),
            "completeDays": int(complete.sum()), "netPerPickOnCompleteDays": float(days.mean()),
            "grossUpOnResolvedPicks": float(good.gross.gt(0).mean()), "netUpOnResolvedPicks": float(good.net.gt(0).mean()),
            "tailOnResolvedPicks": float(good.gross.le(c["evaluation"]["tailThreshold"]).mean()),
            "allSelectedUpBounds": [int(good.gross.gt(0).sum()) / n, (int(good.gross.gt(0).sum()) + n - k) / n],
            "states": picks.state.value_counts().to_dict(),
            "upCalibration": probability_metrics(good.gross.gt(0).to_numpy(), good.pUp.to_numpy()),
            "tailCalibration": probability_metrics(good.gross.le(c["evaluation"]["tailThreshold"]).to_numpy(), good.pTail.to_numpy()),
            "yearNetOnCompleteDays": days.groupby(pd.to_datetime(days.index).year).mean().to_dict(),
            "warning": "Conditional cohort statistics, not portfolio/account PnL; unresolved selections not discarded for acceptance."}


def selection_exposure(picks, sessions, c):
    mean = picks.groupby(["date", "policy"]).net.mean().unstack().reindex(columns=ARMS)
    count = picks.groupby(["date", "policy"]).net.count().unstack().reindex(columns=ARMS)
    # No arm can gain from a different subset of resolved days.
    mean = mean.where(count.eq(10)).dropna()
    if mean.empty:
        return {"status": "no_joint_complete_days"}
    hindsight = str(mean.mean().idxmax())
    observations = []
    for date in mean.index:
        i = int(sessions.get_loc(pd.Timestamp(date)))
        end = i - c["training"]["purgeSessions"]
        start = max(0, end - 63)
        prior = mean.loc[mean.index.isin(sessions[start:end].strftime("%Y-%m-%d"))]
        if len(prior) < c["evaluation"]["minimumTrailingSelectorDays"]:
            continue
        selected = str(prior.mean().idxmax())
        observations.append({"date": date, "trailingArm": selected, "trailingNet": mean.loc[date, selected], "hindsightNet": mean.loc[date, hindsight]})
    if not observations:
        return {"status": "insufficient_trailing_selector_days"}
    r = pd.DataFrame(observations)
    return {"hindsightArm_NOT_VALID_STRATEGY": hindsight, "jointCompleteDays": len(mean), "pairedSelectorDays": len(r),
            "trailingOnlyNet": float(r.trailingNet.mean()), "hindsightNet": float(r.hindsightNet.mean()),
            "overfittingExposure": float((r.hindsightNet - r.trailingNet).mean()),
            "details": observations, "historicalSelectionIsDiagnosticOnly": True}


def coverage_gate(comparisons, frame, sessions, daily, start, end):
    expected = {str(sessions[i].date()) for i in range(start, end) if i in daily}
    predicted = set(frame.date)
    missing = sorted(expected - predicted)
    absent_support = [str(sessions[i].date()) for i in range(start, end) if i not in daily]
    for result in comparisons.values():
        result["requirements"]["allCommonSupportDaysPredicted"] = not missing
        result["requirements"]["allRequestedSessionsHaveCommonSupport"] = not absent_support
        result["numericalTargetsMet"] = all(result["requirements"].values())
    return {"requestedSignalDays": end - start, "commonSupportDays": len(expected),
            "predictedSignalDays": len(predicted), "missingPredictions": missing,
            "noCommonSupportDates": absent_support, "researchOnly": True}


def run(path, run_id):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(c)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid_run_id")
    out = ROOT / c["outputRoot"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    dependencies = [Path(__file__), Path(readiness.__file__), Path(source.__file__), Path(fundamentals.__file__),
                    Path(pit.__file__), Path(payoff.__file__), Path(targets.__file__), ROOT / c["fundamentalConfig"],
                    ROOT / c["targetsConfig"]]
    manifest = {"researchOnly": True, "runId": run_id, "createdAt": source.now(), "config": c,
                "configSha256": pit.digest(path), "codeCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "codeDirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "dependencies": {str(p.relative_to(ROOT)): pit.digest(p) for p in dependencies},
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "orders": [], "historicalOnly": True}
    save(out / "manifest.json", manifest)
    try:
        sessions, daily, audit = load_data(c, out)
        start = int(sessions.searchsorted(c["evaluation"]["startDate"]))
        # Full maximum-maturity buffer; no pretending pending labels are zero cash.
        end = len(sessions) - c["training"]["purgeSessions"]
        picks, fold_meta, skipped = [], [], []
        for first in range(start, end, c["training"]["refitSessions"]):
            train, cal = payoff.split_indices(first, c)
            block = [i for i in range(first, min(first + c["training"]["refitSessions"], end)) if i in daily]
            if not block:
                skipped.append({"first": first, "reason": "no_common_complete_cross_section"})
                continue
            models, views, meta = {}, {}, {}
            try:
                for arm in ARMS[1:]:
                    views[arm] = model_daily(daily, arm)
                    with threadpool_limits(limits=1):
                        models[arm], meta[arm] = fit(views[arm], train, cal, c)
            except RuntimeError as exc:
                skipped.append({"first": first, "reason": str(exc)})
                continue
            fold_meta.append({"firstSignalDate": str(sessions[first].date()), "lastSignalDate": str(sessions[block[-1]].date()), "arms": meta})
            save(out / "folds.json", fold_meta)
            save(out / "models" / f"fold_{first}.json", {"researchOnly": True, "firstSignalIndex": first,
                 "arms": {a: model_parameters(model) for a, model in models.items()}})
            for i in block:
                predictions = {a: predict(models[a], daily[i]["features"][a]) for a in ARMS[1:]}
                predictions["equal_families"] = predictions["fundamental_model"]
                for arm in ARMS:
                    p = predictions[arm]
                    score = daily[i]["features"]["fundamental_model"].mean(axis=1) if arm == "equal_families" else p["payoff_decomposition"] - c["evaluation"]["roundTripCost"]
                    chosen = payoff.top10(score)
                    rr = payoff.record_picks(daily[i], i, sessions[i], len(sessions) - 1, "reused_history", arm, score, p, c)
                    for r, k in zip(rr, chosen):
                        r["pTail"] = float(p["pTail"][k])
                        r["expectedNet"] = r["expectedGross"] - c["evaluation"]["roundTripCost"]
                    picks.extend(rr)
            save(out / "status.json", {"researchOnly": True, "state": "training_walk_forward", "lastFold": str(sessions[first].date()), "folds": len(fold_meta), "picks": len(picks), "orders": []})
            print(f"trained_fold {sessions[first].date()} folds={len(fold_meta)} picks={len(picks)}", flush=True)
        if not picks:
            raise RuntimeError("no_valid_training_folds:" + json.dumps(skipped))
        frame = pd.DataFrame(picks)
        source.atomic_text(out / "historical_picks.csv", frame.to_csv(index=False))
        tc = json.loads((ROOT / c["targetsConfig"]).read_text(encoding="utf-8"))
        comparisons = {a: targets.assess_top10(frame, tc, a, "equal_families", "reused_history") for a in ARMS[1:]}
        comparisons["interaction_increment"] = targets.assess_top10(frame, tc, "interaction_model", "context_model", "reused_history")
        comparisons["current_vs_lagged"] = targets.assess_top10(frame, tc, "interaction_model", "lagged_interaction_counter", "reused_history")
        evaluation_coverage = coverage_gate(comparisons, frame, sessions, daily, start, end)
        result = {"researchOnly": True, "runId": run_id, "state": "completed_diagnostic_only", "mayPromote": False, "orders": [],
                  "dataAudit": {k: v for k, v in audit.items() if k != "dailyCoverage"}, "folds": len(fold_meta), "skippedFolds": skipped,
                  "evaluationCoverage": evaluation_coverage,
                  "arms": {a: summary(frame.loc[frame.policy.eq(a)], c) for a in ARMS}, "comparisons": comparisons,
                  "hindsightVsTrailing": selection_exposure(frame, sessions, c),
                  "trials": {**c["trials"], "globalLowerBoundAfterRun": c["trials"]["globalPriorLowerBound"] + len(ARMS),
                             "newPerMechanismFamily": {k: len(ARMS) for k in ["earnings_innovation", "growth_acceleration", "quality", "cash_flow_quality"]}},
                  "dsr": {"value": None, "reason": "complete_global_and_family_search_history_unavailable_no_significance_certificate"},
                  "limitations": ["Retrospective vendor vintages and reused history; not fresh OOS.",
                                  "SH/SZ audited subset, not all A-shares. BJ lacks verified status.",
                                  "Date-level vendor status assumed available at open; no queue or intraday timestamp reconstruction.",
                                  "Corporate-action/unknown-status legs unresolved, no rank-11 replacement.",
                                  "Fundamental date availability is conservative, but complete historical revision vintages are unavailable.",
                                  "Four fundamental mechanism scores plus fixed contexts/interactions, NOT the previous twelve price-volume factors.",
                                  "Equal-family book probabilities supplied by the same causal fundamental model, not independent fitting.",
                                  "No industry neutrality, capacity certification or independent placebo distribution in this diagnostic."]}
        save(out / "result.json", result)
        lines = ["# Sina / fundamental Top10 historical training", "", "Research-only. No trading, promotion or dashboard publication.", "",
                 "| Arm | Selected / resolved | Complete days | Gross up (resolved) | Net/pick (complete days) | Tail (resolved) |",
                 "|---|---:|---:|---:|---:|---:|"]
        for arm, r in result["arms"].items():
            lines.append(f"| {arm} | {r['selected']} / {r['resolved']} | {r['completeDays']} | {r['grossUpOnResolvedPicks']:.2%} | {r['netPerPickOnCompleteDays']:.4%} | {r['tailOnResolvedPicks']:.2%} |")
        lines += ["", "These are conditional cohort outcomes, not deployable forecasts or portfolio returns.", "", "## Selection exposure", "", "```json", json.dumps(fundamentals.json_safe(result["hindsightVsTrailing"] | {"details": "see result.json"}), indent=2), "```", ""]
        lines += ["- " + s for s in result["limitations"]]
        source.atomic_text(out / "report.md", "\n".join(lines) + "\n")
        save(out / "status.json", {"researchOnly": True, "state": result["state"], "folds": len(fold_meta), "orders": []})
        print(json.dumps({"runId": run_id, "state": result["state"], "folds": len(fold_meta), "mayPromote": False}), flush=True)
        return result
    except Exception as exc:
        save(out / "status.json", {"researchOnly": True, "state": "failed_closed", "error": f"{type(exc).__name__}:{exc}", "orders": []})
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/research/sina_fundamental_top10_v1.json"))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run(args.config, args.run_id)
