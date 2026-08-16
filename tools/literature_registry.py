"""Validate and render the auditable user-supplied literature registry."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SOURCE = ROOT / "research" / "literature_registry.json"
SCHEMA = ROOT / "research" / "literature_registry.schema.json"
ALLOWED_STATUS = {
    "implemented",
    "partially_mapped",
    "extension_candidate",
    "reference_only",
    "data_gated",
    "deferred",
    "out_of_scope",
}
STATUS_ORDER = {
    "implemented": 0,
    "partially_mapped": 1,
    "extension_candidate": 2,
    "data_gated": 3,
    "deferred": 4,
    "reference_only": 5,
    "out_of_scope": 6,
}
STATUS_MEANING = {
    "implemented": "A matching bounded public callable exists; no alpha claim follows.",
    "partially_mapped": "Some auditable primitives exist, but the paper's full architecture does not.",
    "extension_candidate": "A direct, testable extension of an existing primitive; not implemented.",
    "reference_only": "Research context or a future benchmark; no implementation is claimed.",
    "data_gated": "The mechanism needs event types or data granularity not present in the package.",
    "deferred": "Deliberately postponed until a simpler prerequisite survives falsification.",
    "out_of_scope": "Reviewed and excluded because no defensible financial mechanism was specified.",
}
REQUIRED_ENTRY = {
    "paperId",
    "title",
    "year",
    "url",
    "domain",
    "status",
    "systemMapping",
    "implementationBoundary",
    "requiredData",
    "nextFalsifiableStep",
    "repositorySurfaces",
}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]+$")


def _text(value: Any, field: str, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} characters")
    return value.strip()


def _resolve_surface(reference: str) -> None:
    if ":" in reference and not reference.startswith(("http://", "https://")):
        module_name, function_name = reference.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name, None)
        if not callable(function):
            raise ValueError(f"repository surface is not callable: {reference}")
        return
    path = ROOT / reference
    if not path.is_file():
        raise ValueError(f"repository surface does not exist: {reference}")


def validate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_ENTRY - entry.keys()
    unknown = entry.keys() - REQUIRED_ENTRY
    if missing:
        raise ValueError(f"literature entry missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"literature entry has unknown fields: {sorted(unknown)}")
    paper_id = _text(entry["paperId"], "paperId")
    if ID_PATTERN.fullmatch(paper_id) is None:
        raise ValueError(f"invalid paperId: {paper_id}")
    _text(entry["title"], f"{paper_id}.title", 8)
    if not isinstance(entry["year"], int) or not 1900 <= entry["year"] <= 2100:
        raise ValueError(f"{paper_id}.year is invalid")
    if not _text(entry["url"], f"{paper_id}.url").startswith("https://"):
        raise ValueError(f"{paper_id}.url must use https")
    _text(entry["domain"], f"{paper_id}.domain", 3)
    if entry["status"] not in ALLOWED_STATUS:
        raise ValueError(f"{paper_id}.status is invalid")
    for field in (
        "systemMapping",
        "implementationBoundary",
        "requiredData",
        "nextFalsifiableStep",
    ):
        _text(entry[field], f"{paper_id}.{field}", 10)
    surfaces = entry["repositorySurfaces"]
    if not isinstance(surfaces, list) or not all(isinstance(item, str) for item in surfaces):
        raise ValueError(f"{paper_id}.repositorySurfaces must be a string array")
    for surface in surfaces:
        _resolve_surface(surface)
    if entry["status"] in {"implemented", "partially_mapped"} and not surfaces:
        raise ValueError(f"{paper_id} claims public mapping without a resolvable surface")
    return entry


def load_registry() -> list[dict[str, Any]]:
    if not SCHEMA.is_file():
        raise ValueError("literature registry schema is missing")
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "xalpha_literature_registry_v1":
        raise ValueError("unsupported literature registry schemaVersion")
    _text(payload.get("purpose"), "purpose", 20)
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("literature registry must contain entries")
    validated = [validate_entry(entry) for entry in entries]
    ids = [entry["paperId"] for entry in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate paperId in literature registry")
    return sorted(
        validated,
        key=lambda entry: (STATUS_ORDER[entry["status"]], -entry["year"], entry["paperId"]),
    )


def render_json(entries: list[dict[str, Any]]) -> str:
    counts = Counter(entry["status"] for entry in entries)
    payload = {
        "schemaVersion": "xalpha_public_literature_map_v1",
        "meaning": "traceability_and_implementation_boundaries_not_empirical_validation",
        "statusDefinitions": STATUS_MEANING,
        "statusCounts": {status: counts.get(status, 0) for status in STATUS_ORDER},
        "entries": entries,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(entries: list[dict[str, Any]]) -> str:
    counts = Counter(entry["status"] for entry in entries)
    active = [entry for entry in entries if entry["status"] != "out_of_scope"]
    excluded = [entry for entry in entries if entry["status"] == "out_of_scope"]
    lines = [
        "# User-Supplied Literature Registry",
        "",
        "> Machine-validated traceability, not a claim of implementation quality, predictive",
        "> value, or trading readiness. A citation is evidence for a method boundary—not alpha.",
        "",
        "This registry records which submitted papers map to public code, which require new data",
        "or falsification, and which were reviewed but excluded. CI resolves every claimed callable",
        "or repository path and regenerates this page from `research/literature_registry.json`.",
        "",
        "## Status contract",
        "",
        "| Status | Count | Meaning |",
        "|---|---:|---|",
    ]
    for status in STATUS_ORDER:
        lines.append(f"| `{status}` | {counts.get(status, 0)} | {STATUS_MEANING[status]} |")
    lines.extend(
        [
            "",
            "## Research map",
            "",
            "| Paper | Domain | Status | Public mapping and boundary | Next falsifiable step |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in active:
        paper = f"[{_cell(entry['title'])}]({entry['url']}) ({entry['year']})"
        mapping = f"{entry['systemMapping']} **Boundary:** {entry['implementationBoundary']}"
        lines.append(
            f"| {paper} | {_cell(entry['domain'])} | `{entry['status']}` | "
            f"{_cell(mapping)} | {_cell(entry['nextFalsifiableStep'])} |"
        )
    lines.extend(
        [
            "",
            "## Reviewed and excluded",
            "",
            "These records are retained to prevent impressive terminology from being recycled into",
            "a factor without a causal market mechanism.",
            "",
            "| Paper | Domain | Why it is excluded |",
            "|---|---|---|",
        ]
    )
    for entry in excluded:
        paper = f"[{_cell(entry['title'])}]({entry['url']}) ({entry['year']})"
        lines.append(
            f"| {paper} | {_cell(entry['domain'])} | "
            f"{_cell(entry['implementationBoundary'])} |"
        )
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- `partially_mapped` does not mean the cited architecture was reproduced.",
            "- `extension_candidate` does not authorize implementation, tuning, or promotion.",
            "- Cross-domain evidence cannot validate a financial feature by analogy.",
            "- No paper in this registry changes an order, position, risk gate, or execution path.",
            "- Empirical results and proprietary factor weights are intentionally absent.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate metadata and mappings")
    parser.add_argument("--render", action="store_true", help="write deterministic public artifacts")
    args = parser.parse_args()
    if not args.check and not args.render:
        parser.error("select --check, --render, or both")
    entries = load_registry()
    if args.render:
        json_path = ROOT / "docs" / "data" / "literature_registry.json"
        markdown_path = ROOT / "docs" / "USER_SUPPLIED_LITERATURE.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(render_json(entries), encoding="utf-8")
        markdown_path.write_text(render_markdown(entries), encoding="utf-8")
    print(f"validated {len(entries)} literature registry entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
