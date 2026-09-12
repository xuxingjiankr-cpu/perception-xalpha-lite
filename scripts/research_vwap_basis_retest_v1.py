"""Paired, reject-only correction audit of RESEARCH_LOG #12 and #14. No factor search."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import pandas as pd

import research_horizon_cost_frontier_v1 as frontier
import research_tail_exclusion_screen_v1 as tail
from research_horizon_cost_frontier_v1 import panel_cache, precision, guarded, perception

ROOT = precision.ROOT
OLD, NEW = "legacy_amount_volume_v1", "archive_vwap_v2"
AFFECTED = "alpha101/alpha_094"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    precision.atomic_write(path, precision.canonical(precision.json_safe(value)) + "\n")


def frame_digest(frame):
    h = hashlib.sha256()
    h.update(precision.canonical({"dates": list(map(str, frame.index)), "symbols": list(map(str, frame.columns)), "dtypes": list(map(str, frame.dtypes))}).encode())
    for start in range(0, len(frame), 64):
        h.update(np.ascontiguousarray(frame.iloc[start:start + 64].to_numpy()).tobytes())
    return h.hexdigest()


def price_basis_audit(panel, inputs, *, require_consistent=False):
    v = inputs["vwap"]
    fields = [panel[k] for k in ("close", "open", "high", "low")]
    valid = v.notna() & panel["close"].gt(0) & panel["low"].gt(0)
    for f in fields:
        valid &= np.isfinite(f)
    tolerance = panel["high"].abs() * 1e-5 + .0001
    bad = valid & ((v < panel["low"] - tolerance) | (v > panel["high"] + tolerance) | ~np.isfinite(v))
    n = int(bad.to_numpy().sum())
    if require_consistent and n:
        raise ValueError(f"vwap_not_on_ohlc_price_basis:outside_adjusted_range={n}")
    proxy = sum(fields) / 4
    proxy_match = valid & (v - proxy).abs().le(tolerance)
    return {**v.attrs["factorInputBasisAudit"], "auditedCells": int(valid.to_numpy().sum()),
            "outsideAdjustedRange": n, "matchingOhlc4ProxyCells": int(proxy_match.to_numpy().sum()),
            "trueTransactionVwapCertified": False,
            "warning": "A same-session OHLC envelope is a scale sanity check, not a proof of vendor vintage or executable VWAP. Preserved panel fields may include upstream fallbacks."}


def common_rank_support(panel, old, new):
    if set(old) != set(new) or len(old) != 12 or AFFECTED not in old:
        raise ValueError("not_same_frozen_twelve")
    common = panel["eligible"].copy()
    old_available = common & False
    new_available = common & False
    audit = []
    for key in old:
        if not old[key].index.equals(new[key].index) or not old[key].columns.equals(new[key].columns):
            raise ValueError("rank_axes_mismatch:" + key)
        equal = old[key].equals(new[key])
        if key != AFFECTED and not equal:
            raise ValueError("unexpected_change_outside_single_vwap_factor:" + key)
        old_available |= np.isfinite(old[key])
        new_available |= np.isfinite(new[key])
        differences = ~(old[key].eq(new[key]) | (old[key].isna() & new[key].isna()))
        audit.append({"factorKey": key, "unchanged": equal, "changedCells": int(differences.to_numpy().sum())})
    rv = panel["returns"].rolling(20, min_periods=10).std()
    # Positive frozen/equal weights use the original available-factor arithmetic.
    # Requiring ALL twelve finite or reranking would change the experiment in
    # addition to the VWAP correction. Refuse rather than shrink eligibility.
    if (common & ~(old_available & new_available & np.isfinite(rv))).to_numpy().any():
        raise ValueError("original_eligibility_not_fully_supported_by_both_composites_and_rv20")
    counts = common.sum(axis=1)
    if not counts.ge(10).any():
        raise ValueError("no_joint_rank_support")
    return common, {"researchOnly": True, "factorChanges": audit, "supportSha256": frame_digest(common),
                    "originalEligibleCells": int(panel["eligible"].to_numpy().sum()), "commonCells": int(common.to_numpy().sum()),
                    "droppedCells": int((panel["eligible"] & ~common).to_numpy().sum()),
                    "rule": "original_eligibility_unchanged_both_composites_and_rv20_finite_no_reranking",
                    "dates": [str(common.index[0].date()), str(common.index[-1].date())], "symbols": len(common.columns)}


def paired_ranks(ranks, common):
    # Preserve every pre-existing rank value and missing-value convention.
    return {k: r.where(common) for k, r in ranks.items()}


def conclusions(old_frontier, new_frontier, old_tail, new_tail, c):
    hrows, frontier_pass = [], []
    for book in c["compositeBooks"]:
        by_h = {}
        for p in c["tailRequiredWindows"]:
            old = {(r["book"], r["holdingTradingDays"]): r for r in old_frontier["periods"][p]["headline"]}
            for r in new_frontier["periods"][p]["headline"]:
                if r["book"] != book:
                    continue
                before = old[(book, r["holdingTradingDays"])]
                v, t = r["meanExcessPerPick"], r["excessDayClusteredT"]
                good = v is not None and t is not None and v > 0 and t >= c["frontierMinimumPositiveTBothWindows"]
                by_h.setdefault(r["holdingTradingDays"], []).append(good)
                hrows.append({"book": book, "horizon": r["holdingTradingDays"], "period": p,
                              "oldExcess": before["meanExcessPerPick"], "newExcess": v,
                              "oldT": before["excessDayClusteredT"], "newT": t, "positiveSignificant": good,
                              "oldGross": before["meanGrossPerPick"], "newGross": r["meanGrossPerPick"]})
        frontier_pass += [{"book": book, "horizon": h} for h, flags in by_h.items() if len(flags) == 2 and all(flags)]
    def cells(report):
        return {(r["book"], r["period"], r["holdingTradingDays"]): r for r in report["cells"] if r["bucket"] == report["bucketCount"]}
    old_cells, new_cells = cells(old_tail), cells(new_tail)
    trows, tail_pass, secondary = [], [], []
    for book in c["compositeBooks"]:
        flags = []
        for p in c["tailRequiredWindows"]:
            for h in c["tailRequiredHorizons"]:
                before, after = old_cells[(book, p, h)], new_cells[(book, p, h)]
                rv = new_cells[("realized_volatility_20", p, h)]
                numbers = [after["excessSevereLossRate"], rv["excessSevereLossRate"], after["meanExcessReturn"], rv["meanExcessReturn"]]
                good = all(x is not None and np.isfinite(x) for x in numbers) and numbers[0] > numbers[1] and numbers[2] < numbers[3]
                flags.append(good)
                trows.append({"book": book, "period": p, "horizon": h, "oldTailExcess": before["excessSevereLossRate"],
                              "newTailExcess": numbers[0], "vol20TailExcess": numbers[1],
                              "oldExcludedReturnExcess": before["meanExcessReturn"], "newExcludedReturnExcess": numbers[2],
                              "vol20ExcludedReturnExcess": numbers[3], "beatsBothAxes": good})
        if len(flags) == 4 and all(flags):
            tail_pass.append(book)
    for p in c["tailRequiredWindows"]:
        for h in c["tailRequiredHorizons"]:
            vstd, rv = new_cells[("single/qlib158/vstd60", p, h)], new_cells[("realized_volatility_20", p, h)]
            secondary.append({"period": p, "horizon": h, "volumeVstd60TailExcess": vstd["excessSevereLossRate"],
                              "returnVol20TailExcess": rv["excessSevereLossRate"], "volumeVstd60ExcludedReturnExcess": vstd["meanExcessReturn"],
                              "returnVol20ExcludedReturnExcess": rv["meanExcessReturn"]})
    return {"researchOnly": True, "mayPromote": False, "eligibleForTrading": False, "orders": [],
            "study12": {"standingConclusionRejected": bool(frontier_pass), "qualifyingComposites": frontier_pass, "cells": hrows},
            "study14": {"standingConclusionRejected": bool(tail_pass), "qualifyingComposites": tail_pass, "cells": trows},
            "vstd60Diagnostic": {"formula": "std(volume,60)/current_volume (volume variability, NOT price-return volatility)", "cells": secondary},
            "limitations": ["Already-viewed windows; reject-only, never promotion.",
                            "Original next-session eligibility/execution rules and conditional resolved-return accounting unchanged; not a fresh execution audit.",
                            "The legacy day-clustered t does not fully adjust overlapping multi-session outcomes or all historical multiple testing.",
                            "VWAP remains an adjusted OHLC4 proxy, not true transaction VWAP. Alpha094 formula discrepancy and other internal missing-value behavior are not changed.",
                            "Only one of twelve frozen factors references VWAP. No reweighting or sweep beyond the two existing studies."]}


def report_text(r):
    lines = ["# VWAP price-basis correction: paired rejection audit", "", "Research-only. No promotion, trading eligibility, refit or reweighting.", "",
             f"Run `{r['runId']}`; code `{r['codeCommit']}`.", "",
             f"Common eligible rank cells: {r['support']['commonCells']:,}; excluded from original support: {r['support']['droppedCells']:,}.", "",
             "## #12 — day-neutral return excess", "", f"Standing conclusion rejected: {r['study12']['standingConclusionRejected']}.", "",
             "| Book | h | Window | Old excess bp | New excess bp | Old t | New t |", "|---|---:|---|---:|---:|---:|---:|"]
    def fmt(v, scale=1):
        return "null" if v is None else f"{v*scale:.3f}"
    for x in r["study12"]["cells"]:
        lines.append(f"| {x['book']} | {x['horizon']} | {x['period']} | {fmt(x['oldExcess'],1e4)} | {fmt(x['newExcess'],1e4)} | {fmt(x['oldT'])} | {fmt(x['newT'])} |")
    lines += ["", "## #14 — excluded worst decile", "", f"Standing conclusion rejected: {r['study14']['standingConclusionRejected']}.", "",
              "Higher tail excess AND lower excluded-return excess are better. Both must win in all four window/horizon cells.", "",
              "| Book | h | Window | Old/new/RV20 tail excess pp | Old/new/RV20 excluded-return excess bp |", "|---|---:|---|---:|---:|"]
    for x in r["study14"]["cells"]:
        lines.append(f"| {x['book']} | {x['horizon']} | {x['period']} | " + " / ".join(fmt(x[k],100) for k in ["oldTailExcess", "newTailExcess", "vol20TailExcess"]) + " | " + " / ".join(fmt(x[k],1e4) for k in ["oldExcludedReturnExcess", "newExcludedReturnExcess", "vol20ExcludedReturnExcess"]) + " |")
    lines += ["", "## One cheap diagnostic: VSTD60 vs RV20", "", r["vstd60Diagnostic"]["formula"], "",
              "| h | Window | VSTD60/RV20 tail excess pp | VSTD60/RV20 excluded return bp |", "|---:|---|---:|---:|"]
    for x in r["vstd60Diagnostic"]["cells"]:
        lines.append(f"| {x['horizon']} | {x['period']} | {fmt(x['volumeVstd60TailExcess'],100)} / {fmt(x['returnVol20TailExcess'],100)} | {fmt(x['volumeVstd60ExcludedReturnExcess'],1e4)} / {fmt(x['returnVol20ExcludedReturnExcess'],1e4)} |")
    lines += ["", "## Limits", ""] + ["- " + s for s in r["limitations"]]
    return "\n".join(lines) + "\n"


def run(config_path, run_id):
    c = read(config_path)
    if (c["schemaVersion"] != "vwap_basis_retest_v1" or c.get("supportVersion") != "preserve_original_eligibility_v2"
            or c["sameSupport"] != "original_eligibility_unchanged" or c["rankWithinCommonSupport"]
            or not c["researchOnly"] or c["mayPromote"] or c["mayTrade"] or c["orders"]
            or c["basisVersions"] != [OLD, NEW] or c["onlyAffectedFactor"] != AFFECTED or c["frontierMinimumPositiveTBothWindows"] != 2.
            or c["compositeBooks"] != ["frozen_prior", "equal_weight"] or c["tailRequiredHorizons"] != [1, 5]
            or c["tailRequiredWindows"] != ["validation", "shadow"] or c["outputRoot"] != "outputs/edge_research/vwap_basis_retest_v1"):
        raise ValueError("fixed_reject_only_contract_required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid_run_id")
    out = ROOT / c["outputRoot"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    status = lambda state, **fields: save(out / "status.json", {"researchOnly": True, "state": state, "orders": [], **fields})
    try:
        fc, tc = read(ROOT / c["frontierConfig"]), read(ROOT / c["tailConfig"])
        frontier.validate_config(fc); tail.validate_config(tc)
        if fc["basePrecisionConfig"] != tc["basePrecisionConfig"]:
            raise ValueError("different_frozen_books")
        frozen, source, frozen_sha = guarded.load_frozen_config({"basePrecisionConfig": fc["basePrecisionConfig"]})
        base = read(ROOT / frozen["baseResearchConfig"])
        _, cog = perception.load_base_configs(base)
        protected = [ROOT / c["frontierConfig"], ROOT / c["tailConfig"], ROOT / fc["basePrecisionConfig"],
                     ROOT / frozen["baseResearchConfig"], ROOT / frozen["frozenSourceSummary"]]
        # Pin the actual evaluator bytes as well as Git metadata; another local
        # research task must not silently change code during a paired run.
        protected += [Path(m.__file__).resolve() for m in (frontier, tail, precision, guarded, perception, panel_cache)]
        protected += [Path(__file__).resolve(), ROOT / "scripts/research_perception_xalpha_rolling_health_v4.py"]
        for item in frozen["frozenFactors"]:
            zoo, factor = item["factorKey"].split("/")
            protected.append(ROOT / "scripts/vendor/vibe_factors/src/factors/zoo" / zoo / (factor + ".py"))
        for cfg in (fc, tc):
            protected += [p for p in (ROOT / cfg["output"]["root"]).rglob("*") if p.is_file()]
        hashes = {str(p.relative_to(ROOT)): panel_cache._sha256_file(p) for p in protected}
        panel_key, cache_inputs = panel_cache.cache_key(base, cog)
        manifest = {"researchOnly": True, "runId": run_id, "config": c, "configSha256": panel_cache._sha256_file(Path(config_path)),
                    "codeCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "codeDirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                    "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                    "panelKey": panel_key, "panelInputs": cache_inputs, "frozenConfigSha256": frozen_sha,
                    "protectedHashes": hashes, "createdAt": pd.Timestamp.now(tz="UTC").isoformat(), "orders": []}
        save(out / "manifest.json", manifest)
        status("loading_identical_panel")
        panel, pa = panel_cache.build_configured_panel_cached(base, cog)
        original_eligible = frame_digest(panel["eligible"])
        audit = {}
        for basis in (OLD, NEW):
            inputs = precision.build_factor_inputs(panel, vwap_basis=basis)
            audit[basis] = price_basis_audit(panel, inputs, require_consistent=(basis == NEW))
            del inputs
        save(out / "input_basis_audit.json", audit)
        versions, audits = {}, {}
        for basis in (OLD, NEW):
            status("building_rank_book", basis=basis)
            ranks, static, fa = panel_cache.build_rank_book_cached(panel, frozen, panel_key, vwap_basis=basis)
            versions[basis], audits[basis] = ranks, fa
            del static
        common, support = common_rank_support(panel, versions[OLD], versions[NEW])
        save(out / "identical_support_audit.json", support)
        common.to_parquet(out / "common_support.parquet")
        pd.DataFrame({"date": common.index, "originalEligible": panel["eligible"].sum(axis=1).values,
                      "pairedEligible": common.sum(axis=1).values}).to_csv(out / "support_by_date.csv", index=False)
        summaries = {OLD: {}, NEW: {}}
        for basis in (OLD, NEW):
            ranks = paired_ranks(versions.pop(basis), common)
            gc.collect()
            shared = {"panel": panel, "panelAudit": pa, "ranks": ranks, "factorAudit": audits[basis], "commonSupport": common,
                      "audit": {"version": basis, "supportSha256": support["supportSha256"], "originalEligibilitySha256": original_eligible,
                                "researchOnly": True, "sameOutcomesAndExecutionRules": True, "priceBasisAudit": audit[basis]}}
            for study, module, cfg in [("frontier", frontier, c["frontierConfig"]), ("tail", tail, c["tailConfig"])]:
                status("evaluating", basis=basis, study=study)
                print(f"study_start {basis} {study}", flush=True)
                prepared = {**shared, "output": str((out / basis / study).relative_to(ROOT))}
                summaries[basis][study] = module.run(ROOT / cfg, run_id + "_" + basis + "_" + study, paired_inputs=prepared)
                print(f"study_complete {basis} {study}", flush=True)
            del ranks, shared, prepared
            gc.collect()
        if frame_digest(panel["eligible"]) != original_eligible or frame_digest(common) != support["supportSha256"]:
            raise ValueError("support_mutated_during_paired_runs")
        if panel_cache.cache_key(base, cog)[0] != panel_key:
            raise ValueError("data_sources_changed_during_paired_runs")
        for path, h in hashes.items():
            if panel_cache._sha256_file(ROOT / path) != h:
                raise ValueError("protected_artifact_changed:" + path)
        result = conclusions(summaries[OLD]["frontier"], summaries[NEW]["frontier"], summaries[OLD]["tail"], summaries[NEW]["tail"], c)
        result.update(runId=run_id, codeCommit=manifest["codeCommit"], support=support, protectedArtifactsUnchanged=True,
                      dataRange=[str(panel["close"].index[0].date()), str(panel["close"].index[-1].date())],
                      evaluationWindows=source["splitAudit"],
                      hypothesesCounted={"newPriceBasisCorrection": 1, "frontierCompositeCells": 24, "tailCompositeCells": 8,
                                         "newWeightsFitted": 0, "newFactorsSearched": 0})
        save(out / "result.json", result)
        precision.atomic_write(out / "report.md", report_text(result))
        status("completed_reject_only", study12Rejected=result["study12"]["standingConclusionRejected"], study14Rejected=result["study14"]["standingConclusionRejected"])
        return result
    except Exception as exc:
        status("failed_closed", error=f"{type(exc).__name__}:{exc}")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/research/vwap_basis_retest_v2.json")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    run(a.config, a.run_id)
