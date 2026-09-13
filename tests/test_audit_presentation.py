from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_walkthrough_matches_recomputed_results():
    renderer = _module(ROOT / "examples/render_audit_walkthrough.py")
    for name, expected in renderer.render_assets().items():
        assert (ROOT / name).read_text(encoding="utf-8") == expected, name


def test_walkthrough_hash_and_static_no_tracking_contract():
    page = (ROOT / "docs/demo.html").read_text(encoding="utf-8")
    report = (ROOT / "docs/examples/audit-cases.json").read_text(encoding="utf-8")
    assert hashlib.sha256(report.encode()).hexdigest() in page
    assert len(re.findall(r'<article class="case"', page)) == 3
    assert 'max="45"' in page and 'elapsed / 15' in page
    assert "not live inference" in page
    assert not re.search(r'<script[^>]+src=|\bfetch\(|XMLHttpRequest|sendBeacon|localStorage', page)
    assert 'id="pause"' in page and 'id="reset"' in page
    assert 'aria-live="polite"' in page


def test_first_screen_links_to_demo_without_exposing_empirical_picks():
    for name in ("README.md", "docs/README_CN.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        first_screen = "\n".join(text.splitlines()[:45])
        assert "demo.html" in first_screen
        assert "audit-demo.svg" in first_screen
        assert "LIVE-RECORD" not in text
        assert "SZ_300804" not in text
        assert "DATA_PROVENANCE.md" in text


def test_provenance_does_not_claim_the_whole_repository_is_synthetic():
    for name in ("README.md", "README_PYPI.md", "docs/index.html", "docs/ARCHITECTURE.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for wrong in ("No empirical result is published here", "Every number above",
                      "Every number in the package", "No empirical research artifact is committed"):
            assert wrong not in text, (name, wrong)
        assert "DATA_PROVENANCE.md" in text


def test_tutorial_numbers_match_the_generated_report():
    report = json.loads((ROOT / "docs/examples/audit-cases.json").read_text())
    noise = report["cases"][0]
    tutorial = (ROOT / "docs/tutorials/01-noise-selection.md").read_text()
    for key in ("best_sharpe_ann", "dsr_statistic", "pbo", "hindsight_evaluation_bps", "train_selected_evaluation_bps"):
        assert str(noise[key]) in tutorial


def test_new_markdown_and_demo_local_links_resolve():
    paths = [ROOT / n for n in ("README.md", "docs/README_CN.md", "docs/ARCHITECTURE.md",
             "docs/DATA_PROVENANCE.md", "docs/RESEARCH_RECORD.md", "docs/RESEARCH_RECORD_CN.md")]
    paths += list((ROOT / "docs/tutorials").glob("*.md"))
    paths += [ROOT / "docs/demo.html"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        links = re.findall(r'\]\(([^)]+)\)', text) + re.findall(r'(?:href|src)="([^"]+)"', text)
        for link in links:
            if re.match(r"[a-z]+:|#", link):
                continue
            file = link.split("#", 1)[0]
            assert (path.parent / file).exists(), (str(path.relative_to(ROOT)), link)


def test_record_writer_still_updates_relocated_pages_without_touching_readme(tmp_path, monkeypatch):
    writer = _module(ROOT / "tools/daily_record.py")
    expected = (ROOT / "docs/RESEARCH_RECORD.md", ROOT / "docs/RESEARCH_RECORD_CN.md")
    assert writer.READMES == expected
    pages = tuple(tmp_path / p.name for p in expected)
    for source, dest in zip(expected, pages):
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    original = (ROOT / "README.md").read_bytes()
    monkeypatch.setattr(writer, "READMES", pages)
    def fake_rows(path):
        if path == writer.PICKS:
            return [{"as_of": "2020-01-01", "symbol": "TOY_A"}]
        assert path == writer.SERIES
        return [{"phase": "live", "date": "2020-01-02", "strategy_net": -.01, "strategy_gross": -.005}]
    monkeypatch.setattr(writer, "read_jsonl", fake_rows)
    writer.refresh_readmes({})
    for path in pages:
        text = path.read_text(encoding="utf-8")
        assert "TOY_A" in text and "insufficient_forward_sample" in text
        assert text.count(writer.BEGIN) == text.count(writer.END) == 1
    assert (ROOT / "README.md").read_bytes() == original
