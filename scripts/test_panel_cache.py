#!/usr/bin/env python3
"""Correctness tests for the research panel cache.

A stale or lossy panel cache would silently corrupt every downstream experiment,
so these tests concentrate on the ways a cache is allowed to fail: it must miss,
never serve something that is not exactly what was built.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import panel_cache  # noqa: E402


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def synthetic_panel() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=25)
    columns = ["SH.600000", "SZ.000001", "SZ.300750"]
    rng = np.random.default_rng(20260904)
    close = pd.DataFrame(
        rng.normal(10.0, 1.0, size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    volume = pd.DataFrame(
        rng.integers(1000, 9000, size=(len(dates), len(columns))).astype(float),
        index=dates,
        columns=columns,
    )
    eligible = pd.DataFrame(True, index=dates, columns=columns)
    eligible.iloc[3, 1] = False
    is_st = pd.DataFrame(False, index=dates, columns=columns)
    returns = close.pct_change(fill_method=None)
    trade_status = pd.DataFrame("1", index=dates, columns=columns, dtype=object)
    return {
        "close": close,
        "volume": volume,
        "eligible": eligible,
        "is_st": is_st,
        "returns": returns,
        "trade_status": trade_status,
    }


def synthetic_base(master: Path, bars: Path) -> tuple[dict, dict]:
    base = {
        "assetUniverse": {
            "kind": "ashare",
            "masterPath": str(master.relative_to(ROOT).as_posix()),
            "barsRoot": str(bars.relative_to(ROOT).as_posix()),
            "exchanges": ["SH", "SZ"],
            "maximumSuspensionFraction": 0.2,
        }
    }
    cog = {"data": {"minimumObservationsPerSymbol": 250}}
    return base, cog


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="panel_cache_test_", dir=ROOT / "outputs"))
    try:
        bars = workspace / "bars"
        bars.mkdir(parents=True)
        (bars / "SH_600000.jsonl").write_text('{"dt":"2025-01-02"}\n', encoding="utf-8")
        (bars / "SZ_000001.jsonl").write_text('{"dt":"2025-01-02"}\n', encoding="utf-8")
        master = workspace / "master.jsonl"
        master.write_text('{"securityId":"SH.600000"}\n', encoding="utf-8")
        cache_root = workspace / "cache"

        signature_before = panel_cache._tree_signature(bars)
        check("tree signature counts every bar file", signature_before["files"] == 2)
        time.sleep(0.01)
        (bars / "SZ_000001.jsonl").write_text(
            '{"dt":"2025-01-02"}\n{"dt":"2025-01-03"}\n', encoding="utf-8"
        )
        signature_after = panel_cache._tree_signature(bars)
        check(
            "appending a session changes the tree signature",
            signature_before["signature"] != signature_after["signature"],
        )
        (bars / "SZ_300750.jsonl").write_text('{"dt":"2025-01-02"}\n', encoding="utf-8")
        check(
            "adding a symbol changes the tree signature",
            panel_cache._tree_signature(bars)["signature"] != signature_after["signature"],
        )

        base, cog = synthetic_base(master, bars)
        key, inputs = panel_cache.cache_key(base, cog)
        key_again, _ = panel_cache.cache_key(base, cog)
        check("the key is deterministic for identical inputs", key == key_again)
        check(
            "the key covers the builder source code",
            bool(inputs.get("builderSources")),
        )
        changed_cog = {"data": {"minimumObservationsPerSymbol": 500}}
        key_config, _ = panel_cache.cache_key(base, changed_cog)
        check("changing the data config changes the key", key_config != key)
        changed_base = json.loads(json.dumps(base))
        changed_base["assetUniverse"]["maximumSuspensionFraction"] = 0.5
        key_universe, _ = panel_cache.cache_key(changed_base, cog)
        check("changing the universe config changes the key", key_universe != key)
        time.sleep(0.01)
        (bars / "SH_600000.jsonl").write_text(
            '{"dt":"2025-01-02"}\n{"dt":"2025-01-03"}\n', encoding="utf-8"
        )
        key_data, _ = panel_cache.cache_key(base, cog)
        check("changing the underlying bars changes the key", key_data != key)

        panel = synthetic_panel()
        audit = {"status": "diagnostic_only_research_only", "orders": []}
        check(
            "an empty cache misses",
            panel_cache.load(key_data, cache_root) is None,
        )
        check(
            "storing verifies the round trip and succeeds",
            panel_cache.store(key_data, inputs, panel, audit, cache_root) is True,
        )
        loaded = panel_cache.load(key_data, cache_root)
        check("a stored panel loads back", loaded is not None)
        cached_panel, cached_audit = loaded
        check(
            "every field survives the round trip exactly",
            set(cached_panel) == set(panel)
            and all(cached_panel[f].equals(panel[f]) for f in panel),
        )
        check(
            "boolean and object dtypes are preserved",
            cached_panel["eligible"].dtypes.eq(bool).all()
            and cached_panel["trade_status"].dtypes.eq(object).all(),
        )
        check(
            "NaN placement is preserved",
            cached_panel["returns"].isna().equals(panel["returns"].isna()),
        )
        check("the audit round-trips", cached_audit == audit)

        check(
            "a different key never reads another key's panel",
            panel_cache.load(key, cache_root) is None,
        )

        cache_dir = cache_root / key_data[:16]
        manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["key"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        check(
            "a manifest whose key no longer matches is a miss",
            panel_cache.load(key_data, cache_root) is None,
        )

        manifest["key"] = key_data
        manifest["fields"]["close"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        check(
            "a field whose checksum no longer matches is a miss",
            panel_cache.load(key_data, cache_root) is None,
        )

        manifest["fields"]["close"] = panel_cache._frame_checksum(panel["close"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        check(
            "restoring the checksum makes it readable again",
            panel_cache.load(key_data, cache_root) is not None,
        )
        (cache_dir / "volume.parquet").unlink()
        check(
            "a missing field file is a miss rather than a partial panel",
            panel_cache.load(key_data, cache_root) is None,
        )
        manifest_path.write_text("{ not json", encoding="utf-8")
        check(
            "an unreadable manifest is a miss rather than an exception",
            panel_cache.load(key_data, cache_root) is None,
        )

        bad = dict(panel)
        bad["broken"] = "not a dataframe"
        check(
            "storing refuses a payload that is not a frame",
            panel_cache.store("f" * 64, inputs, bad, audit, cache_root) is False,
        )
        check(
            "a refused store leaves no staging directory behind",
            not any(p.name.endswith(".staging") for p in cache_root.iterdir()),
        )

        source = (ROOT / "scripts" / "panel_cache.py").read_text(encoding="utf-8").lower()
        forbidden = ("submitorder", "cancelorder", "build_decision(", "latest_strategy_overlay")
        check(
            "source has no trading mutation path",
            not any(term in source for term in forbidden),
        )
        print("ALL PANEL CACHE TESTS PASSED")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
