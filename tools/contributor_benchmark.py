"""Validate and render the public research-engineering contribution benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xalpha_lite.dsl import validate as validate_expression  # noqa: E402


REQUIRED = {
    "schemaVersion",
    "submissionId",
    "title",
    "contributor",
    "kind",
    "paper",
    "hypothesis",
    "causalBoundary",
    "counterfactual",
    "testSelector",
    "prefixCausalityTested",
    "adversarialTested",
}
ALLOWED = REQUIRED | {
    "implementation",
    "expression",
    "counterExpression",
    "allowedFields",
    "allowedWindows",
    "reproductionExample",
    "futureDataRole",
}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]+$")


def _text(value: Any, field: str, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} characters")
    return value.strip()


def _resolve_callable(reference: str) -> None:
    if ":" not in reference:
        raise ValueError("implementation must use module:function syntax")
    module_name, function_name = reference.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"implementation is not callable: {reference}")


def _validate_test_selector(selector: str | None) -> bool:
    if selector is None:
        return False
    path_text, separator, test_name = selector.partition("::")
    if not separator or not test_name.startswith("test_"):
        raise ValueError("testSelector must use path.py::test_name")
    path = ROOT / path_text
    if not path.is_file():
        raise ValueError(f"test file does not exist: {path_text}")
    source = path.read_text(encoding="utf-8")
    if re.search(rf"^def {re.escape(test_name)}\s*\(", source, re.MULTILINE) is None:
        raise ValueError(f"test function does not exist: {selector}")
    return True


def validate_card(card: dict[str, Any], source: Path) -> dict[str, Any]:
    missing = REQUIRED - card.keys()
    unknown = card.keys() - ALLOWED
    if missing:
        raise ValueError(f"{source.name}: missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{source.name}: unknown fields: {sorted(unknown)}")
    if card["schemaVersion"] != "xalpha_contributor_submission_v1":
        raise ValueError(f"{source.name}: unsupported schemaVersion")
    submission_id = _text(card["submissionId"], "submissionId")
    if ID_PATTERN.fullmatch(submission_id) is None or source.stem != submission_id:
        raise ValueError(f"{source.name}: filename and submissionId must match")
    _text(card["title"], "title", 4)
    _text(card["contributor"], "contributor")
    _text(card["hypothesis"], "hypothesis", 20)
    _text(card["causalBoundary"], "causalBoundary", 20)
    _text(card["counterfactual"], "counterfactual", 10)
    if card["kind"] not in {"mechanism", "factor"}:
        raise ValueError(f"{source.name}: kind must be mechanism or factor")
    paper = card["paper"]
    if not isinstance(paper, dict):
        raise ValueError(f"{source.name}: paper must be an object")
    _text(paper.get("title"), "paper.title")
    paper_url = _text(paper.get("url"), "paper.url")
    if not paper_url.startswith("https://"):
        raise ValueError(f"{source.name}: paper.url must use https")
    if not isinstance(paper.get("year"), int) or paper["year"] < 1900:
        raise ValueError(f"{source.name}: paper.year is invalid")

    if card["kind"] == "mechanism":
        _resolve_callable(_text(card.get("implementation"), "implementation"))
        if card.get("expression") is not None or card.get("counterExpression") is not None:
            raise ValueError(f"{source.name}: mechanism cards cannot contain DSL expressions")
    else:
        fields = set(card.get("allowedFields", []))
        windows = {int(value) for value in card.get("allowedWindows", [])}
        if not fields or not windows:
            raise ValueError(f"{source.name}: factor cards require allowedFields and allowedWindows")
        validate_expression(card.get("expression"), fields, windows)
        validate_expression(card.get("counterExpression"), fields, windows)
        if card.get("implementation") is not None:
            raise ValueError(f"{source.name}: factor cards use DSL, not implementation")

    test_exists = _validate_test_selector(card.get("testSelector"))
    example = card.get("reproductionExample")
    example_exists = isinstance(example, str) and (ROOT / example).is_file()
    if example is not None and not example_exists:
        raise ValueError(f"{source.name}: reproductionExample does not exist: {example}")
    if card.get("futureDataRole", "none") not in {"none", "offline_label_only"}:
        raise ValueError(f"{source.name}: invalid futureDataRole")
    if not isinstance(card["prefixCausalityTested"], bool):
        raise ValueError(f"{source.name}: prefixCausalityTested must be boolean")
    if not isinstance(card["adversarialTested"], bool):
        raise ValueError(f"{source.name}: adversarialTested must be boolean")

    score = 20  # traceable paper
    score += 15  # explicit hypothesis and point-in-time boundary
    score += 20  # callable mechanism or validated DSL primary/counter pair
    score += 15  # falsification statement and counter implementation/description
    score += 15 if card["prefixCausalityTested"] and test_exists else 0
    score += 10 if card["adversarialTested"] and test_exists else 0
    score += 5 if example_exists else 0
    tier = (
        "research-grade"
        if score >= 90 and card["prefixCausalityTested"] and card["adversarialTested"]
        else "causal-tested"
        if card["prefixCausalityTested"] and test_exists
        else "reproducible-implementation"
        if score >= 60
        else "documented"
    )
    return {
        "submissionId": submission_id,
        "title": card["title"],
        "contributor": card["contributor"],
        "kind": card["kind"],
        "score": score,
        "tier": tier,
        "paper": paper,
        "implementation": card.get("implementation"),
        "testSelector": card.get("testSelector"),
        "prefixCausalityTested": card["prefixCausalityTested"],
        "adversarialTested": card["adversarialTested"],
        "reproductionExample": example,
        "futureDataRole": card.get("futureDataRole", "none"),
    }


def load_benchmark() -> list[dict[str, Any]]:
    sources = sorted((ROOT / "benchmark" / "submissions").glob("*.json"))
    if not sources:
        raise ValueError("no contributor benchmark submissions found")
    rows = [validate_card(json.loads(path.read_text(encoding="utf-8")), path) for path in sources]
    ids = [row["submissionId"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate submissionId")
    return sorted(rows, key=lambda row: (-row["score"], row["submissionId"]))


def render_json(rows: list[dict[str, Any]]) -> str:
    payload = {
        "schemaVersion": "xalpha_contributor_benchmark_v1",
        "meaning": "research_engineering_completeness_not_predictive_performance",
        "scoring": {
            "paperTraceability": 20,
            "hypothesisAndCausalBoundary": 15,
            "validatedImplementation": 20,
            "falsificationDesign": 15,
            "prefixCausalityTest": 15,
            "adversarialTest": 10,
            "reproductionExample": 5,
        },
        "entries": rows,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Contributor Benchmark",
        "",
        "> This leaderboard measures research-engineering completeness, not returns, alpha,",
        "> statistical significance, or deployment readiness. A high score cannot authorize trading.",
        "",
        "Each card is machine-checked against a real callable or audited DSL expression, a cited",
        "paper, an explicit point-in-time boundary, and repository test/example paths.",
        "",
        "| Rank | Contribution | Kind | Score | Tier | Prefix causal test | Adversarial test |",
        "|---:|---|---|---:|---|:---:|:---:|",
    ]
    for rank, row in enumerate(rows, 1):
        paper = row["paper"]
        title = f"[{row['title']}]({paper['url']})"
        lines.append(
            f"| {rank} | {title} | {row['kind']} | {row['score']}/100 | {row['tier']} | "
            f"{'yes' if row['prefixCausalityTested'] else 'no'} | "
            f"{'yes' if row['adversarialTested'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Scoring contract",
            "",
            "- Paper traceability: 20",
            "- Explicit hypothesis and causal boundary: 15",
            "- Resolvable implementation or validated primary/counter DSL pair: 20",
            "- Falsification design: 15",
            "- Executable prefix-causality test: 15",
            "- Executable adversarial test: 10",
            "- Reproduction example: 5",
            "",
            "CI regenerates this page from `benchmark/submissions/*.json` and rejects stale or",
            "unresolvable entries. See [CONTRIBUTING.md](../CONTRIBUTING.md) to submit one.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate every submission")
    parser.add_argument("--render", action="store_true", help="write deterministic public artifacts")
    args = parser.parse_args()
    if not args.check and not args.render:
        parser.error("select --check, --render, or both")
    rows = load_benchmark()
    if args.render:
        json_path = ROOT / "docs" / "data" / "contributor_benchmark.json"
        markdown_path = ROOT / "docs" / "CONTRIBUTOR_BENCHMARK.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(render_json(rows), encoding="utf-8")
        markdown_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"validated {len(rows)} contributor benchmark entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
