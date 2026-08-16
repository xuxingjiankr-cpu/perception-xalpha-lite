from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools" / "literature_registry.py"
    spec = importlib.util.spec_from_file_location("literature_registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_literature_registry_is_valid_and_rendered_deterministically() -> None:
    registry = _module()
    entries = registry.load_registry()
    assert len(entries) >= 20
    assert any(entry["status"] == "partially_mapped" for entry in entries)
    assert any(entry["status"] == "out_of_scope" for entry in entries)
    assert (ROOT / "docs" / "data" / "literature_registry.json").read_text(
        encoding="utf-8"
    ) == registry.render_json(entries)
    assert (ROOT / "docs" / "USER_SUPPLIED_LITERATURE.md").read_text(
        encoding="utf-8"
    ) == registry.render_markdown(entries)


def test_claimed_public_mappings_are_resolvable() -> None:
    registry = _module()
    entries = registry.load_registry()
    claimed = [
        entry
        for entry in entries
        if entry["status"] in {"implemented", "partially_mapped"}
    ]
    assert claimed
    assert all(entry["repositorySurfaces"] for entry in claimed)
