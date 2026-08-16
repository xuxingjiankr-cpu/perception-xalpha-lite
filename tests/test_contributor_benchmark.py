from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools" / "contributor_benchmark.py"
    spec = importlib.util.spec_from_file_location("contributor_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contributor_benchmark_is_valid_and_rendered_deterministically() -> None:
    benchmark = _module()
    rows = benchmark.load_benchmark()
    assert len(rows) >= 5
    assert all(row["score"] <= 100 for row in rows)
    assert all(row["futureDataRole"] in {"none", "offline_label_only"} for row in rows)
    assert (ROOT / "docs" / "data" / "contributor_benchmark.json").read_text(
        encoding="utf-8"
    ) == benchmark.render_json(rows)
    assert (ROOT / "docs" / "CONTRIBUTOR_BENCHMARK.md").read_text(
        encoding="utf-8"
    ) == benchmark.render_markdown(rows)
