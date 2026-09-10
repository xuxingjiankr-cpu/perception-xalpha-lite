"""Isolated, resumable Sina raw daily bars + dated HFQ factors; never trades.

Only a bundled, hashed decoder is executed, never downloaded JavaScript.
Raw turnover is retained in CNY/shares; adjusted VWAP uses a date-local factor.
This verifies a price basis, NOT historical vendor vintages or tradability.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_right
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from collect_ashare_research_daily import atomic_json, atomic_jsonl, atomic_text

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "sina_hisdata_klc2_and_hfq"
BASE = "https://finance.sina.com.cn/realstock/company/"
OHLC = ("open", "high", "low", "close")


class DataError(ValueError):
    pass


class AccessDenied(DataError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise DataError("duplicate_json_key")
            out[key] = value
        return out
    def reject(value):
        raise DataError("nonfinite_json_constant:" + value)
    return json.JSONDecoder(object_pairs_hook=pairs, parse_constant=reject).decode(text)


def parse_assignment(text, variable):
    """Accept one JSON literal assignment and comments, never active JS suffixes."""
    match = re.match(r"\s*var\s+" + re.escape(variable) + r"\s*=\s*", text)
    if not match:
        raise DataError("source_symbol_assignment_mismatch")
    body = text[match.end():]
    _, end = json.JSONDecoder().raw_decode(body)
    tail = body[end:]
    if not re.fullmatch(r"\s*;?\s*(?:/\*[^*]*(?:\*(?!/)[^*]*)*\*/\s*)*", tail):
        raise DataError("unexpected_executable_response_suffix")
    return strict_json(body[:end])


def iso_date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise DataError("invalid_date")
    return value


def finite(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value) and (not positive or value > 0))


def parse_factors(text, symbol):
    obj = parse_assignment(text, symbol + "hfq")
    if not isinstance(obj, dict) or not isinstance(obj.get("data"), list):
        raise DataError("missing_factor_table")
    if obj.get("total") != len(obj["data"]) or not obj["data"]:
        raise DataError("factor_count_mismatch_or_empty")
    result = {}
    for row in obj["data"]:
        dt = iso_date(row["d"])
        value = float(row["f"])
        if isinstance(row["f"], bool) or not finite(value, True) or dt in result:
            raise DataError("invalid_or_duplicate_factor")
        result[dt] = value
    return sorted(result.items())


def decode_bars(text, symbol, decoder):
    encoded = parse_assignment(text, "KLC_K2_" + symbol)
    if not isinstance(encoded, str) or not encoded:
        raise DataError("missing_encoded_prices")
    rows = decoder.call("d", encoded, timeout=10000)
    if not isinstance(rows, list) or not rows:
        raise DataError("empty_decoded_prices")
    return rows


def normalize(rows, factors, sid, sessions, cfg):
    """No fill of missing bars, no inferred factor=1, no price/flow source splice."""
    session_set = set(sessions)
    start, end = cfg["startDate"], cfg["endDate"]
    factor_dates = [dt for dt, _ in factors]
    seen, raw_out, adj_out, reasons = set(), [], [], Counter()
    for row in rows:
        stamp = row.get("date")
        if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T00:00:00\.000Z", stamp):
            raise DataError("unexpected_vendor_date_timezone")
        dt = iso_date(stamp[:10])
        if dt in seen:
            raise DataError("duplicate_price_date:" + dt)
        seen.add(dt)
        if not start <= dt <= end:
            continue
        errors = []
        if dt not in session_set:
            errors.append("not_exchange_session")
        values = {k: row.get(k) for k in (*OHLC, "volume", "amount")}
        if not all(finite(v, True) for v in values.values()):
            errors.append("invalid_or_inactive_ohlcv_amount")
        elif not row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]:
            errors.append("invalid_ohlc_order")
        raw_vwap = row["amount"] / row["volume"] if not errors else None
        if raw_vwap is not None:
            tol = cfg["rawVwapAbsoluteToleranceCny"] + row["high"] * cfg["rawVwapRelativeTolerance"]
            if not row["low"] - tol <= raw_vwap <= row["high"] + tol:
                errors.append("transaction_vwap_outside_raw_range")
        pos = bisect_right(factor_dates, dt) - 1
        factor = factors[pos][1] if pos >= 0 else None
        factor_dt = factor_dates[pos] if pos >= 0 else None
        raw_errors = list(errors)
        if factor is None:
            errors.append("missing_date_local_factor")
        elif not errors and not all(finite(row[k] * factor, True) for k in OHLC):
            errors.append("nonfinite_adjusted_price")
        valid = not errors
        common = {"dt": dt, "securityId": sid, "source": SOURCE, "researchOnly": True,
                  "volumeUnit": "shares", "amountUnit": "CNY", "pointInTimeStatus": False,
                  "isST": None, "tradeStatus": None, "historicalVintageVerified": False}
        raw = {**common, **{k: v if finite(v) else None for k, v in values.items()},
               "vwap": raw_vwap, "priceBasisValid": not raw_errors,
               "rejectedReasons": raw_errors, "adjustment": "none_raw_sina",
               "vendorExtra": {k: row[k] for k in ("prevclose", "postVol", "postAmt")
                               if k in row and finite(row[k])}}
        adj = {**common, **{k: row[k] * factor if valid else None for k in OHLC},
               "volume": raw["volume"], "amount": raw["amount"],
               "vwap": raw_vwap * factor if valid else None,
               "adjustment": cfg["adjustment"], "factor": factor, "factorEffectiveDate": factor_dt,
               "priceBasisValid": valid, "rejectedReasons": errors}
        raw_out.append(raw)
        adj_out.append(adj)
        reasons.update(errors)
    raw_out.sort(key=lambda r: r["dt"])
    adj_out.sort(key=lambda r: r["dt"])
    if not raw_out:
        raise DataError("no_prices_in_requested_range")
    observed = {r["dt"] for r in raw_out}
    missing = sum(raw_out[0]["dt"] <= dt <= raw_out[-1]["dt"] and dt not in observed for dt in sessions)
    report = {"rawRows": len(raw_out), "validAdjustedRows": sum(r["priceBasisValid"] for r in adj_out),
              "rejectedReasons": dict(reasons), "firstDate": raw_out[0]["dt"],
              "lastDate": raw_out[-1]["dt"], "missingBarsBetweenObservedEndpoints": missing,
              "fresh": raw_out[-1]["dt"] == sessions[-1],
              "freshPriceBasisValid": adj_out[-1]["dt"] == sessions[-1] and adj_out[-1]["priceBasisValid"]}
    return raw_out, adj_out, report


def load_universe(paths):
    """Union, including known delisted names; no whole-file price filter."""
    universe, sources = {}, []
    for relative in paths:
        path = ROOT / relative
        content = path.read_bytes()
        sources.append({"path": relative, "sha256": sha(content)})
        local = set()
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = strict_json(line)
            sid = row["securityId"]
            if not re.fullmatch(r"(?:SH|SZ|BJ)\.\d{6}", sid) or sid in local:
                raise DataError("invalid_or_duplicate_master_id")
            local.add(sid)
            if sid == "SH.689009":  # CDR uses a different Sina decoder, not an A-share.
                continue
            if sid not in universe or row.get("pointInTimeMembership") is True:
                universe[sid] = row
    return universe, sources


@contextmanager
def single_instance(path):
    """OS lock is released on crash; a stale PID never grants an unsafe takeover."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DataError("collector_already_running") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class PublicClient:
    def __init__(self, cfg):
        import requests
        self.session = requests.Session()
        self.cfg, self.last = cfg, 0.0

    def get(self, url):
        time.sleep(max(0, self.cfg["requestIntervalSeconds"] - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        response = self.session.get(url, timeout=(self.cfg["connectTimeoutSeconds"], self.cfg["readTimeoutSeconds"]),
                                    allow_redirects=False)
        if response.status_code in (401, 403, 429):
            raise AccessDenied(f"provider_access_denied_http_{response.status_code}")
        if response.status_code != 200:
            raise DataError(f"provider_http_{response.status_code}")
        if len(response.content) > 8_000_000:
            raise DataError("oversized_provider_payload")
        text = response.content.decode("utf-8")
        if any(word in text.lower() for word in ("captcha", "access denied", "访问频繁", "访问受限")):
            raise AccessDenied("provider_access_challenge")
        return text


def verify_cached(record, data):
    for relative, checksum in record["files"].items():
        path = (data / relative).resolve()
        if not path.is_relative_to(data.resolve()) or sha(path.read_bytes()) != checksum:
            raise DataError("resume_artifact_hash_mismatch")


def normalizer_hash():
    source = Path(__file__).read_text(encoding="utf-8")
    names = {"strict_json", "parse_assignment", "iso_date", "finite", "parse_factors", "decode_bars", "normalize"}
    segments = [ast.get_source_segment(source, node) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(segments) != len(names):
        raise DataError("normalizer_definition_missing")
    return sha((json.dumps([SOURCE, BASE, OHLC]) + "\n" + "\n".join(segments)).encode("utf-8"))


def seed_contract(seed_id, cfg, contract):
    """New run only: reuse identical-definition data, never relax --resume."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", seed_id):
        raise DataError("invalid_seed_run_id")
    origin = ROOT / cfg["outputRoot"] / seed_id
    payload = (origin / "manifest.json").read_bytes()
    previous = strict_json(payload.decode("utf-8"))
    old = previous["contract"]
    if previous.get("researchOnly") is not True or previous.get("orders") != [] or previous.get("mayPromote") is not False:
        raise DataError("unsafe_seed_manifest")
    a = {k: v for k, v in cfg.items() if k != "masterPaths"}
    b = {k: v for k, v in previous["config"].items() if k != "masterPaths"}
    if a != b:
        raise DataError("seed_config_mismatch")
    for key in ("symbols", "sessionsSha256", "decoderSha256", "atomicHelperSha256", "dependencies"):
        if old[key] != contract[key]:
            raise DataError("seed_contract_mismatch:" + key)
    if [r["sha256"] for r in old["masterSources"]] != [r["sha256"] for r in contract["masterSources"]]:
        raise DataError("seed_master_content_changed")
    legacy = old.get("codeCommit") == "12282088d427b2a1b76efce6f511a126c37a55a5" and old.get("scriptSha256") == "69f70cf37ff3b3ff8b71dd73bb65bb52ef2d8ed28fedecea77d0c5ace1cf799f"
    expected_normalizer = "cffca9294a5b4d4ffa72b0294bed49d891e44c82e07d023c77f41b56455c4461" if legacy else old.get("normalizerSha256")
    if expected_normalizer != contract["normalizerSha256"]:
        raise DataError("seed_normalizer_changed")
    state = strict_json((origin / "status.json").read_text(encoding="utf-8"))
    if state.get("failClosedReason", "").startswith("AccessDenied:"):
        raise DataError("access_denied_seed_cannot_restart")
    return {"seedRunId": seed_id, "seedManifestSha256": sha(payload),
            "seedCodeCommit": old["codeCommit"], "seedStatusAtInspection": state.get("state"),
            "masterContentIdentical": True, "newRunIsNotOldRunContinuation": True}


def copy_seed(seed, cfg, data, output, keys):
    """Copy, not hard-link. Preserve original runs and validate every reused byte."""
    if not seed:
        return 0
    origin = ROOT / cfg["outputRoot"] / seed["seedRunId"]
    source_data = ROOT / cfg["dataRoot"] / seed["seedRunId"]
    count = 0
    for path in sorted((origin / "symbols").glob("*.json")):
        content = path.read_bytes()
        record = strict_json(content.decode("utf-8"))
        sid = record["securityId"]
        stem = sid.replace(".", "_")
        if sid not in keys or path.name != stem + ".json":
            raise DataError("seed_symbol_mismatch")
        if record["state"] != "collected":
            raise DataError("failed_seed_requires_separate_review")
        expected = {f"provider_payloads/{stem}_raw.js.txt", f"provider_payloads/{stem}_hfq.js.txt",
                    f"raw/{stem}.jsonl", f"adjusted/{stem}.jsonl"}
        if set(record["files"]) != expected:
            raise DataError("seed_missing_required_artifact")
        verify_cached(record, source_data)
        target_record = output / "symbols" / path.name
        if target_record.exists():
            continue
        for relative in sorted(expected):
            target = data / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".seed_tmp")
            shutil.copyfile(source_data / relative, temp)
            os.replace(temp, target)
        verify_cached(record, data)
        record["seedProvenance"] = {**seed, "originalRecordSha256": sha(content)}
        atomic_json(target_record, record)
        count += 1
    return count


def run(args):
    import exchange_calendars as xc
    import pandas as pd
    import py_mini_racer
    import requests
    from akshare.stock.cons import hk_js_decode

    cfg_bytes = (ROOT / args.config).read_bytes()
    cfg = strict_json(cfg_bytes.decode("utf-8"))
    if (cfg["schemaVersion"] != "sina_daily_research_v1" or cfg["researchOnly"] is not True
            or any(cfg[k] is not False for k in ("mayTrainAutomatically", "mayPublishDashboard", "mayTrade"))
            or cfg["requestIntervalSeconds"] < 2):
        raise DataError("invalid_research_only_config")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise DataError("invalid_run_id")
    for key, prefix in (("dataRoot", "data/market/ashare_research/sina_daily_v1"),
                        ("outputRoot", "outputs/edge_research/sina_daily_v1")):
        if (ROOT / cfg[key]).resolve() != (ROOT / prefix).resolve():
            raise DataError("output_must_be_isolated")
    calendar = xc.get_calendar("XSHG", start=cfg["startDate"], end=cfg["endDate"])
    if calendar.session_close(pd.Timestamp(cfg["endDate"])) > pd.Timestamp.now(tz="UTC"):
        raise DataError("end_session_not_closed")
    sessions = [str(d.date()) for d in calendar.sessions_in_range(cfg["startDate"], cfg["endDate"])]
    universe, sources = load_universe(cfg["masterPaths"])
    if args.symbols:
        keys = args.symbols.split(",")
        if len(keys) != len(set(keys)) or not set(keys) <= set(universe):
            raise DataError("invalid_or_unknown_requested_symbol")
    else:
        keys = sorted(universe)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    contract = {"configSha256": sha(cfg_bytes), "codeCommit": commit, "codeDirty": dirty,
                "scriptSha256": sha(Path(__file__).read_bytes()),
                "normalizerSha256": normalizer_hash(),
                "atomicHelperSha256": sha((ROOT / "scripts/collect_ashare_research_daily.py").read_bytes()),
                "decoderSha256": sha(hk_js_decode.encode()), "masterSources": sources,
                "symbols": sorted(keys), "sessionsSha256": sha("\n".join(sessions).encode()),
                "dependencies": {p: importlib.metadata.version(p) for p in
                                 ("requests", "akshare", "py-mini-racer", "exchange-calendars")}}
    data, output = ROOT / cfg["dataRoot"] / args.run_id, ROOT / cfg["outputRoot"] / args.run_id
    with single_instance(ROOT / cfg["dataRoot"] / "collector.lock"):
        manifest_path = output / "manifest.json"
        seed = None
        if manifest_path.exists():
            old = strict_json(manifest_path.read_text(encoding="utf-8"))
            if not args.resume or old["contract"] != contract:
                raise DataError("resume_contract_mismatch_or_run_exists")
            seed = old.get("seed")
            if getattr(args, "seed_run", None) and (not seed or args.seed_run != seed["seedRunId"]):
                raise DataError("resume_seed_mismatch")
            previous_status = output / "status.json"
            if previous_status.exists():
                reason = strict_json(previous_status.read_text(encoding="utf-8")).get("failClosedReason", "")
                if reason.startswith("AccessDenied:"):
                    raise DataError("access_denied_run_cannot_resume")
        else:
            if args.resume or data.exists() or output.exists():
                raise DataError("new_run_requires_empty_independent_directories")
            if getattr(args, "seed_run", None):
                if args.seed_run == args.run_id:
                    raise DataError("seed_must_be_another_run")
                seed = seed_contract(args.seed_run, cfg, contract)
            atomic_json(manifest_path, {"schemaVersion": cfg["schemaVersion"], "runId": args.run_id,
                        "startedAt": now(), "contract": contract, "config": cfg, "researchOnly": True,
                        "historicalVintageVerified": False, "orders": [], "mayPromote": False, "seed": seed})
            atomic_jsonl(data / "universe.jsonl", (universe[sid] for sid in sorted(keys)))
        if seed:
            current_seed = seed_contract(seed["seedRunId"], cfg, contract)
            if current_seed["seedManifestSha256"] != seed["seedManifestSha256"]:
                raise DataError("seed_manifest_changed")
            copy_seed(seed, cfg, data, output, set(keys))
        decoder = py_mini_racer.MiniRacer()
        decoder.eval(hk_js_decode)
        client = PublicClient(cfg)
        records, daily, failures = [], Counter(), 0
        status = {"runId": args.run_id, "researchOnly": True, "pid": os.getpid(), "state": "running",
                  "requestedSymbols": len(keys), "dataRange": [sessions[0], sessions[-1]],
                  "orders": [], "mayPromote": False, "readyForTraining": False,
                  "limitations": ["Current vendor vintage, not archived historical factor releases.",
                                  "Historical ST/tradestatus unknown; no carry-forward or trade eligibility.",
                                  "Union of known PIT and current names; missing/delisted/BJ history may be incomplete.",
                                  "Daily aggregate VWAP, not an executable fill; post-session turnover preserved.",
                                  "No factor training, no signal or return claim, no production input replacement."]}

        def checkpoint():
            status.update(updatedAt=now(), attemptedSymbols=len(records),
                          reusedSymbols=sum("seedProvenance" in r for r in records),
                          successfulSymbols=sum(r["state"] == "collected" for r in records),
                          failedSymbols=sum(r["state"] != "collected" for r in records),
                          validAdjustedRows=sum(r.get("validAdjustedRows", 0) for r in records),
                          freshValidPriceSymbols=sum(r.get("freshPriceBasisValid", False) for r in records),
                          errors=dict(Counter(r["error"] for r in records if "error" in r)))
            atomic_json(output / "status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)

        checkpoint()
        try:
            for sid in sorted(keys):
                status["activeSymbol"] = sid
                stem, symbol = sid.replace(".", "_"), sid.lower().replace(".", "")
                record_path = output / "symbols" / f"{stem}.json"
                if record_path.exists():
                    record = strict_json(record_path.read_text(encoding="utf-8"))
                    if record.get("securityId") != sid:
                        raise DataError("resume_symbol_mismatch")
                    verify_cached(record, data)
                else:
                    record = {"securityId": sid, "state": "failed", "files": {}, "collectedAt": now()}
                    try:
                        url = BASE + symbol + "/hisdata_klc2/klc_kl.js"
                        text = client.get(url)
                        raw_path = f"provider_payloads/{stem}_raw.js.txt"
                        atomic_text(data / raw_path, text)
                        record["files"][raw_path] = sha((data / raw_path).read_bytes())
                        rows = decode_bars(text, symbol, decoder)
                        factor_url = BASE + symbol + "/hfq.js"
                        text = client.get(factor_url)
                        factor_path = f"provider_payloads/{stem}_hfq.js.txt"
                        atomic_text(data / factor_path, text)
                        record["files"][factor_path] = sha((data / factor_path).read_bytes())
                        factors = parse_factors(text, symbol)
                        raw, adj, audit = normalize(rows, factors, sid, sessions, cfg)
                        for kind, items in (("raw", raw), ("adjusted", adj)):
                            path = f"{kind}/{stem}.jsonl"
                            atomic_jsonl(data / path, items)
                            record["files"][path] = sha((data / path).read_bytes())
                        record.update(audit, state="collected", sourceUrls=[url, factor_url],
                                      validDates=[r["dt"] for r in adj if r["priceBasisValid"]])
                        failures = 0
                    except AccessDenied as exc:
                        record["error"] = str(exc)
                        atomic_json(record_path, record)
                        records.append(record)
                        raise
                    except requests.RequestException as exc:
                        failures += 1
                        record["error"] = "transport_failure:" + type(exc).__name__
                        record["detail"] = str(exc)[:300]
                    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                        failures = 0
                        record["error"] = str(exc)[:250]
                    atomic_json(record_path, record)
                records.append(record)
                daily.update(record.get("validDates", []))
                if len(records) % 25 == 0 or len(records) == len(keys):
                    checkpoint()
                if failures >= cfg["maximumConsecutiveTransportFailures"]:
                    raise DataError("source_unavailable_consecutive_transport_failures")
            status["state"] = "completed_with_gaps" if any(r["state"] != "collected" for r in records) else "completed"
        except Exception as exc:
            status.update(state="blocked", failClosedReason=f"{type(exc).__name__}:{exc}")
        finally:
            client.session.close()
            checkpoint()
            atomic_json(output / "result.json", {**status,
                        "priceCoverageReadyForReview": status["freshValidPriceSymbols"] >= cfg["minimumFreshPriceSymbolsForReview"],
                        "dailyValidPrices": dict(sorted(daily.items())),
                        "rejectedReasons": dict(sum((Counter(r.get("rejectedReasons", {})) for r in records), Counter())),
                        "symbols": [{k: v for k, v in r.items() if k not in ("files", "validDates")} for r in records]})
        return 0 if status["state"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/sina_daily_v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--symbols", help="Optional comma-separated master IDs for a pilot, e.g. SH.600000,SZ.000001")
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed symbol artifacts only, identical contract required")
    parser.add_argument("--seed-run", help="New independent run using hash-verified identical-definition files from an earlier run")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"researchOnly": True, "state": "blocked", "reason": f"{type(exc).__name__}:{exc}"}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
