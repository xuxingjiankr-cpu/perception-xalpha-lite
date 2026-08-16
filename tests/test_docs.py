"""Guards on what the documentation claims, not on what the code does.

These exist because of a failure the rest of the suite could not have caught. A specification
was retired and superseded; the code and the data were both updated and every test passed,
while the README went on naming the retired one as the rule producing the live record — in its
first screen, for a day.

For a project whose whole claim is that a published pick is exactly what a named rule produced,
naming the wrong rule is worse than most bugs, and nothing mechanical was watching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "data" / "specs.json"
# The reader-facing surfaces. A retired specification may be discussed in the changelog and in
# the research notes; it may not be presented here as the rule now in force.
FRONT_FACING = ("README.md", "docs/README_CN.md", "docs/index.html")
SEARCHED = FRONT_FACING + ("CHANGELOG.md",)
DIGEST = re.compile(r"\b[0-9a-f]{8,}\b")
# Specification digests are always presented as code in these documents. Scanning the raw text
# instead sweeps up DOIs and URL fragments — the first run of this guard flagged the digits of
# a journal DOI — so the code spans are extracted first.
CODE_SPAN = re.compile(r"`([^`\n]+)`|<code>([^<]+)</code>")


def registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["specs"]


def cited(paths: tuple[str, ...]) -> dict[str, set[str]]:
    """Every digest-shaped token inside a code span, mapped to the files citing it."""
    found: dict[str, set[str]] = {}
    for name in paths:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        spans = [backtick or tag for backtick, tag in CODE_SPAN.findall(text)]
        for span in spans:
            for token in DIGEST.findall(span):
                found.setdefault(token, set()).add(name)
    return found


def test_every_registered_spec_file_matches_its_recorded_digest() -> None:
    for entry in registry():
        if not entry.get("file"):
            continue
        spec = json.loads((ROOT / entry["file"]).read_text(encoding="utf-8"))
        assert spec["spec_sha256"] == entry["sha256"], f"{entry['file']} disagrees with the registry"
        assert spec["name"] == entry["name"]


def test_documents_cite_only_known_specifications() -> None:
    known = [entry["sha256"] for entry in registry()]
    tokens = cited(SEARCHED)
    assert tokens, "no digest-shaped token found; the guard would pass on anything"
    unknown = {
        token: sorted(files)
        for token, files in tokens.items()
        if not any(digest.startswith(token) for digest in known)
    }
    assert not unknown, (
        f"documents cite digests that are in no registry entry: {unknown}. "
        f"Add them to docs/data/specs.json, or correct the citation."
    )


def test_no_reader_facing_document_cites_a_retired_specification() -> None:
    retired = [entry["sha256"] for entry in registry() if entry["status"] == "retired"]
    assert retired, "no retired specification in the registry; this guard is untested"
    tokens = cited(FRONT_FACING)
    offences = {
        token: sorted(files)
        for token, files in tokens.items()
        if any(digest.startswith(token) for digest in retired)
    }
    assert not offences, (
        f"a retired specification is presented as current: {offences}. "
        f"A reader checking the record against that hash would find it does not match."
    )


@pytest.mark.parametrize("status", ["live", "retired"])
def test_registry_records_where_a_specification_runs(status: str) -> None:
    entries = [entry for entry in registry() if entry["status"] == status]
    assert entries, f"no specification is {status}"
    for entry in entries:
        if status == "live":
            assert entry["runs_on"], f"{entry['name']} is live but names no host"
        else:
            assert entry["runs_on"] is None, f"{entry['name']} is retired but still names a host"


def test_live_record_block_is_machine_written_and_still_marked() -> None:
    """The daily job rewrites this block; without the markers it silently stops.

    Two sentences in these files were written as facts about one particular day and had to be
    corrected within a week — a superseded digest, and "the solid span is empty" on the morning
    the first live session landed. The block exists so those numbers are never typed by hand
    again, which only works while the markers are there to find.
    """
    for name in ("README.md", "docs/README_CN.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert text.count("<!-- LIVE-RECORD:BEGIN -->") == 1, f"{name}: opening marker"
        assert text.count("<!-- LIVE-RECORD:END -->") == 1, f"{name}: closing marker"
        head, _, rest = text.partition("<!-- LIVE-RECORD:BEGIN -->")
        body, _, _ = rest.partition("<!-- LIVE-RECORD:END -->")
        assert "Verdict" in body or "结论" in body, f"{name}: block was emptied"
        assert text.index("<!-- LIVE-RECORD:BEGIN -->") < text.index("<!-- LIVE-RECORD:END -->")
