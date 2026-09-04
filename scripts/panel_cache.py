#!/usr/bin/env python3
"""Content-addressed cache for the PIT-adjusted A-share research panel.

Building the panel re-parses roughly ten million JSON objects out of 5.5 GB of
per-symbol jsonl on every run, which dominates the wall time and the memory
footprint of every experiment.  The underlying data only advances once per
trading day, so every experiment after the first on a given day rebuilds an
identical panel.

This module caches the built panel as parquet under a key derived from the data
files, the panel configuration and the builder source code.  A stale cache would
silently corrupt research, so the key covers every input that can change the
panel, the round trip is verified at write time, and any mismatch, corruption or
unreadable field falls back to a normal rebuild rather than serving a guess.

Research infrastructure only: it moves no data of its own, and cannot trade,
promote or alter any configuration.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "panel_cache"
CACHE_FORMAT_VERSION = "panel_cache_v1"

# Changing any of these changes how the panel is built, so their bytes are part
# of the key.  A code edit must invalidate every cached panel.
BUILDER_SOURCES = (
    "research_ashare_universe.py",
    "research_ashare_fundamentals.py",
    "research_perception_xalpha_autonomous.py",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_signature(root: Path) -> dict[str, Any]:
    """Size and mtime of every bar file. Stat-only, so it stays cheap at 5k files."""
    if not root.exists():
        return {"root": str(root), "files": 0, "signature": "absent"}
    entries = []
    count = 0
    for path in sorted(root.rglob("*.jsonl")):
        stat = path.stat()
        entries.append(
            f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}"
        )
        count += 1
    return {
        "root": str(root),
        "files": count,
        "signature": _sha256_bytes("\n".join(entries).encode("utf-8")),
    }


def cache_key(base: dict[str, Any], cog_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Derive the cache key from every input that can change the built panel."""
    universe = base.get("assetUniverse", {"kind": "etf"})
    data_config = cog_config.get("data", {})
    inputs: dict[str, Any] = {
        "cacheFormatVersion": CACHE_FORMAT_VERSION,
        "assetUniverse": universe,
        "dataConfig": data_config,
        "builderSources": {
            name: _sha256_file(ROOT / "scripts" / name)
            for name in BUILDER_SOURCES
            if (ROOT / "scripts" / name).exists()
        },
    }
    master_ref = universe.get("masterPath")
    if master_ref:
        master_path = ROOT / master_ref
        inputs["master"] = {
            "path": str(master_path),
            "sha256": _sha256_file(master_path) if master_path.exists() else "absent",
        }
    bars_ref = universe.get("barsRoot")
    if bars_ref:
        inputs["bars"] = _tree_signature(ROOT / bars_ref)
    fundamental = universe.get("fundamentalData")
    if fundamental:
        inputs["fundamentalConfig"] = fundamental
        fundamental_ref = fundamental.get("root")
        if fundamental_ref:
            inputs["fundamentalTree"] = _tree_signature(ROOT / fundamental_ref)
    key = _sha256_bytes(
        json.dumps(inputs, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    )
    return key, inputs


def _frame_checksum(frame: pd.DataFrame) -> str:
    """Cheap structural fingerprint; catches truncated or reordered parquet."""
    return _sha256_bytes(
        json.dumps(
            {
                "shape": list(frame.shape),
                "columns": [str(c) for c in frame.columns[:64]],
                "index_first": str(frame.index[0]) if len(frame.index) else "",
                "index_last": str(frame.index[-1]) if len(frame.index) else "",
                "dtypes": sorted({str(d) for d in frame.dtypes}),
            },
            sort_keys=True,
        ).encode("utf-8")
    )


def load(key: str, cache_root: Path | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]] | None:
    """Return the cached panel and audit, or None on any miss, mismatch or damage."""
    root = Path(cache_root or DEFAULT_CACHE_ROOT) / key[:16]
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("key") != key:
            return None
        if manifest.get("cacheFormatVersion") != CACHE_FORMAT_VERSION:
            return None
        panel: dict[str, pd.DataFrame] = {}
        for field, expected in manifest["fields"].items():
            frame = pd.read_parquet(root / f"{field}.parquet")
            if _frame_checksum(frame) != expected:
                return None
            panel[field] = frame
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return panel, audit


def store(
    key: str,
    inputs: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    audit: dict[str, Any],
    cache_root: Path | None = None,
) -> bool:
    """Write the panel, verifying every field round-trips before publishing it."""
    root = Path(cache_root or DEFAULT_CACHE_ROOT) / key[:16]
    staging = root.with_name(root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        fields: dict[str, str] = {}
        for field, frame in panel.items():
            if not isinstance(frame, pd.DataFrame):
                return False
            target = staging / f"{field}.parquet"
            frame.to_parquet(target)
            # Serving a lossy field silently would corrupt every later experiment,
            # so the round trip is proven here rather than assumed.
            reloaded = pd.read_parquet(target)
            if not reloaded.equals(frame):
                return False
            fields[field] = _frame_checksum(frame)
        (staging / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, default=str), encoding="utf-8"
        )
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "cacheFormatVersion": CACHE_FORMAT_VERSION,
                    "key": key,
                    "keyInputs": inputs,
                    "fields": fields,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        staging.rename(root)
        return True
    except Exception:
        return False
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def build_configured_panel_cached(
    base: dict[str, Any],
    cog_config: dict[str, Any],
    *,
    cache_root: Path | None = None,
    enabled: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Drop-in replacement for perception.build_configured_panel with a cache."""
    import research_perception_xalpha_autonomous as perception

    if not enabled:
        return perception.build_configured_panel(base, cog_config)
    try:
        key, inputs = cache_key(base, cog_config)
    except Exception:
        return perception.build_configured_panel(base, cog_config)
    cached = load(key, cache_root)
    if cached is not None:
        if verbose:
            print(f"panel_cache hit key={key[:16]}", flush=True)
        return cached
    if verbose:
        print(f"panel_cache miss key={key[:16]} building", flush=True)
    panel, audit = perception.build_configured_panel(base, cog_config)
    stored = store(key, inputs, panel, audit, cache_root)
    if verbose:
        print(f"panel_cache {'stored' if stored else 'store_failed'} key={key[:16]}", flush=True)
    return panel, audit
