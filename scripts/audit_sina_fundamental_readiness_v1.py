"""Offline input audit only: frozen fundamental families + collected Sina prices.

No training, predictions, price-volume search, metadata carry-forward or trading.
Partial collection is explicitly NOT a stock-selection universe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

import exchange_calendars as xc

import collect_sina_research_daily_v1 as source
import research_fundamental_mechanism_families as mechanism

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    payload = path.read_bytes()
    rows = [source.strict_json(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if source.sha(path.read_bytes()) != source.sha(payload):
        raise ValueError("input_changed_during_read:" + str(path))
    return rows, source.sha(payload)


def trusted_status(row, sid):
    return (row.get("securityId") == sid
            and row.get("source") == "baostock_query_history_k_data_plus"
            and row.get("pointInTimeStatus") is True
            and row.get("pointInTimeStatusSource") in (None, "")
            and type(row.get("isST")) is int and row["isST"] in (0, 1)
            and type(row.get("tradeStatus")) is int and row["tradeStatus"] in (0, 1))


def summarize_fundamentals(rows, sessions, sid, config):
    """Reuse frozen transforms; newest disclosure completeness, NOT a score."""
    events, audit = mechanism.causal_fundamental_records_for_symbol(rows, sessions, sid, config["families"])
    age = None
    complete = {family: False for family in config["families"]}
    latest = events[-1] if events else None
    if latest:
        age = len(sessions) - 1 - int(sessions.get_loc(latest["eventDate"]))
        for family, spec in config["families"].items():
            complete[family] = (age <= config["fundamentals"]["maximumSignalAgeTradingDays"]
                                and all(source.finite(latest.get(c["id"])) for c in spec["candidates"]))
    return {"events": len(events), "latestEventDate": str(latest["eventDate"].date()) if latest else None,
            "latestEventAgeSessions": age, "latestFamilyComplete": complete,
            "allFamiliesComplete": all(complete.values()), "dateAudit": audit}


def run(args):
    for value in (args.price_run, args.run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("invalid_run_id")
    output = ROOT / "outputs/edge_research/sina_fundamental_readiness_v1" / args.run_id
    if output.exists():
        raise ValueError("audit_run_exists")
    origin = ROOT / "outputs/edge_research/sina_daily_v1" / args.price_run
    manifest_bytes = (origin / "manifest.json").read_bytes()
    manifest = source.strict_json(manifest_bytes.decode("utf-8"))
    cfg = manifest["config"]
    sessions = xc.get_calendar("XSHG", start=cfg["startDate"], end=cfg["endDate"]).sessions
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    config_path = ROOT / "configs/research/fundamental_mechanism_families_v1.json"
    config_bytes = config_path.read_bytes()
    fundamental_cfg = source.strict_json(config_bytes.decode("utf-8"))
    keys = manifest["contract"]["symbols"]
    # Freeze the available collector records at invocation; never silently broaden
    # a partial SH/BJ prefix to the whole market when collection grows concurrently.
    records = {p.stem.replace("_", ".", 1): p.read_bytes() for p in sorted((origin / "symbols").glob("*.json"))}
    rows, totals, family = [], Counter(), Counter()
    hashes, daily_status = [], Counter()
    price_root = ROOT / cfg["dataRoot"] / args.price_run
    for n, sid in enumerate(keys, 1):
        item = {"securityId": sid, "hasCollectedPrices": sid in records}
        path = ROOT / fundamental_cfg["fundamentals"]["root"] / (sid.replace(".", "_") + ".jsonl")
        if path.exists():
            statements, checksum = read_rows(path)
            hashes.append({"path": str(path.relative_to(ROOT)), "sha256": checksum})
            fundamental = summarize_fundamentals(statements, sessions, sid, fundamental_cfg)
            item["fundamentals"] = fundamental
            totals["fundamentalFiles"] += 1
            totals["statementRows"] += len(statements)
            totals["causalStatementEvents"] += fundamental["events"]
            totals["latestAllFamiliesCompleteSymbols"] += fundamental["allFamiliesComplete"]
            family.update(k for k, ok in fundamental["latestFamilyComplete"].items() if ok)
        else:
            totals["missingFundamentalFiles"] += 1
        if sid in records:
            record = source.strict_json(records[sid].decode("utf-8"))
            source.verify_cached(record, price_root)
            hashes.append({"path": str((origin / "symbols" / (sid.replace(".", "_") + ".json")).relative_to(ROOT)),
                           "sha256": source.sha(records[sid]), "priceFiles": record["files"]})
            if record["state"] == "collected":
                dates = set(record["validDates"])
                totals["collectedPriceSymbols"] += 1
                totals["validPriceRows"] += record["validAdjustedRows"]
                status_path = ROOT / "data/market/ashare_research/baostock_pit_adjusted/bars_1d_backward_adjusted" / (sid.replace(".", "_") + ".jsonl")
                if status_path.exists():
                    status_rows, checksum = read_rows(status_path)
                    hashes.append({"path": str(status_path.relative_to(ROOT)), "sha256": checksum,
                                   "usedFor": "date_local_ST_and_trade_status_only_no_prices"})
                    seen = set()
                    matched = []
                    for row in status_rows:
                        dt = row.get("dt")
                        if dt in seen:
                            raise ValueError("duplicate_status_date:" + sid)
                        seen.add(dt)
                        if dt in dates and trusted_status(row, sid):
                            matched.append(dt)
                            daily_status[dt] += 1
                    totals["priceRowsWithVerifiedDatedStatus"] += len(matched)
                    totals["cutoffPriceSymbolsWithVerifiedStatus"] += cfg["endDate"] in matched
                    item["matchedStatusRows"] = len(matched)
                    item["latestMatchedStatusDate"] = max(matched) if matched else None
                else:
                    totals["collectedSymbolsWithoutStatusFile"] += 1
        rows.append(item)
        if n % 500 == 0:
            print(f"readiness_audited {n}/{len(keys)}", flush=True)
    result = {"schemaVersion": "sina_fundamental_readiness_v1", "researchOnly": True,
              "status": "input_audit_only_no_model_trained", "runId": args.run_id,
              "sourcePriceRun": args.price_run, "codeCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "codeDirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "dataRange": [cfg["startDate"], cfg["endDate"]], "requestedSymbols": len(keys),
              "frozenCollectedSymbolsByExchange": dict(Counter(sid[:2] for sid in records)),
              "counts": dict(totals), "latestCompleteFamilySymbols": dict(family),
              "dailyPricesWithVerifiedDatedStatus": dict(sorted(daily_status.items())),
              "collectorManifestSha256": source.sha(manifest_bytes), "familyConfigSha256": source.sha(config_bytes),
              "familyModuleSha256": source.sha(Path(mechanism.__file__).read_bytes()),
              "sourceFiles": hashes, "orders": [], "readyForTraining": False, "mayPromote": False,
              "limitations": ["Partial sorted-symbol collection is not a representative stock-selection universe.",
                              "Historical status matched by exact date only; active status and non-ST are not assumed for missing rows.",
                              "Status overlap count includes active and inactive/ST rows, not final eligible picks.",
                              "Latest-disclosure completeness is an input audit, not predictive value or earnings surprise versus consensus.",
                              "Fundamental families use frozen notice/update alignment and transforms; no fitting or outcomes examined.",
                              "Membership/seasoning, factor formula certification and historical-vintage limits still require separate review."]}
    source.atomic_json(output / "result.json", result)
    source.atomic_jsonl(output / "symbol_audit.jsonl", rows)
    text = "# Sina / fundamental research input audit\n\nResearch-only; no model trained or promoted.\n\n"
    text += f"Price cutoff: {cfg['endDate']}. Requested names: {len(keys)}.\n\n"
    text += "```json\n" + json.dumps({"counts": dict(totals), "familyCompleteness": dict(family)}, indent=2) + "\n```\n\n"
    text += "\n".join("- " + s for s in result["limitations"]) + "\n"
    source.atomic_text(output / "report.md", text)
    print(json.dumps({k: result[k] for k in ("runId", "status", "counts", "latestCompleteFamilySymbols", "readyForTraining")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-run", required=True)
    parser.add_argument("--run-id", required=True)
    run(parser.parse_args())
