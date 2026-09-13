"""Render the public 45-second walkthrough from a freshly computed synthetic report.

The output is a guided explanation of recorded results, not a video of live computation.
Use --check in CI to reject drift between the example and the published presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xalpha_lite.audit_cases import build_cases, markdown_report  # noqa: E402


def render_assets() -> dict[str, str]:
    report, _ = build_cases()
    noise, pit, basis = report["cases"]
    raw = json.dumps(report, indent=2, allow_nan=False) + "\n"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    panels = [
        ("01 / SELECTION", f"A Sharpe of {noise['best_sharpe_ann']:.2f}. From noise.",
         f"Best full-sample Sharpe: {noise['best_sharpe_ann']:.2f}",
         f"DSR statistic: {noise['dsr_statistic']:.3f} | PBO: {noise['pbo']:.3f}",
         "Audit the search, not just its winner.",
         "64 zero-mean random variants. DSR and CSCV use the existing library. "
         "A single positive realization does not change the zero-mean process.",
         "tutorials/01-noise-selection.md"),
        ("02 / AVAILABILITY", "The report existed. The information did not.",
         f"Wrong-date rows: {pit['mismatched_rows']} / {pit['rows']}",
         f"First available: {pit['first_disclosed_value_available']}",
         "Publication time, not period end.",
         "The PIT aligner waits until after disclosure. The March restatement must not "
         "rewrite January features. This case tests timing, not trading returns.",
         "tutorials/02-disclosure-timing.md"),
        ("03 / PRICE BASIS", "One numerator. Four different ranks.",
         f"Proxy / cash price: up to {basis['max_archive_to_cash_ratio']:.2f}x",
         f"Changed ranks: {basis['rank_changes']} / {basis['rows']} | identical names",
         "A coherent proxy is still a proxy.",
         "Raw cash turnover divided by raw volume is not an adjusted price. Preserve "
         "the declared OHLC4 proxy; never relabel it as true transaction VWAP.",
         "tutorials/03-price-basis.md"),
    ]
    # Titles and numbers all originate here, after a real invocation of the example.
    cards = []
    for index, (label, title, metric, audited, takeaway, explanation, link) in enumerate(panels):
        cards.append(f'''<article class="case" id="case-{index}" aria-labelledby="title-{index}">
  <p class="eyebrow">{html.escape(label)}</p><h2 id="title-{index}">{html.escape(title)}</h2>
  <p class="metric">{html.escape(metric)}</p><p class="audit">{html.escape(audited)}</p>
  <h3>{html.escape(takeaway)}</h3><p>{html.escape(explanation)}</p>
  <a href="https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/docs/{link}">Read the reproducible case &rarr;</a>
</article>''')
    template = (ROOT / "examples" / "audit_walkthrough.html").read_text(encoding="utf-8")
    page = template.replace("<!-- CASES -->", "\n".join(cards)).replace("<!-- REPORT_HASH -->", digest)
    page = page.replace("<!-- REPORT -->", html.escape(markdown_report(report)))
    # SVG is a static README preview, not a fabricated screenshot of the CLI.
    preview = '''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="400" viewBox="0 0 1100 400" role="img" aria-labelledby="title desc">
<title id="title">Three synthetic audits. A 45-second guided walkthrough.</title>
<desc id="desc">Actual example outputs: random-strategy selection, disclosure timing, and inconsistent price bases. Synthetic demonstrations only, not investment evidence.</desc>
<rect width="1100" height="400" rx="18" fill="#0b1526"/>
<g font-family="Arial, sans-serif">
<text x="36" y="42" fill="#69dcc6" font-size="14" letter-spacing="2">PERCEPTION-XALPHA LITE / SYNTHETIC AUDIT LAB</text>
<text x="36" y="91" fill="#f1f5fb" font-size="32" font-weight="700">Discover factors. Audit the evidence.</text>
<text x="36" y="121" fill="#acbdd5" font-size="17">Three reproducible mistakes. One research-only toolkit.</text>
'''
    for i, (label, _, metric, audited, *_rest) in enumerate(panels):
        x = 36 + 346 * i
        preview += f'''<rect x="{x}" y="153" width="332" height="130" rx="10" fill="#142239" stroke="#2e4561"/>
<text x="{x+16}" y="184" fill="#69dcc6" font-size="13">{html.escape(label)}</text>
<text x="{x+16}" y="222" fill="#f1f5fb" font-size="16">{html.escape(metric)}</text>
<text x="{x+16}" y="250" fill="#acbdd5" font-size="12">{html.escape(audited)}</text>
'''
    preview += '''<text x="36" y="331" fill="#f1f5fb" font-size="19">&#9654; Open the 45-second walkthrough</text>
<text x="36" y="365" fill="#acbdd5" font-size="14">No account. No data upload. No trading or profitability claim.</text></g></svg>
'''
    return {"docs/demo.html": page, "docs/audit-demo.svg": preview,
            "docs/examples/audit-cases.json": raw,
            "docs/examples/audit-cases.md": markdown_report(report)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in render_assets().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale synthetic presentation: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"{'checked' if args.check else 'rendered'} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
