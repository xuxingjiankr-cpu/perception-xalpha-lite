# Data provenance and claim boundaries

The repository is a research toolkit, not a performance product. It contains both software
demonstrations and previously published empirical materials; calling the entire repository
"synthetic" or "methods only, with no securities or performance tables" would be inaccurate.

| Material | Provenance | Permitted interpretation |
|---|---|---|
| `xalpha-audit-cases`, `examples/run_audit_cases.py`, `docs/demo.html`, `docs/examples/audit-cases.*` | Generated IID returns, fictional TOY symbols, synthetic disclosures and weekday calendar | Reproduce specific data/statistical mistakes; not real-market evidence |
| `xalpha-demo` / Colab full-loop example | Deterministic generated prices and fundamentals | Test pipeline mechanics, never certify alpha |
| `docs/RESEARCH_RECORD*.md`, rotation files in `docs/data/`, historical charts and existing research pages | Previously published historical or sampled forward observations, with original caveats | Empirical records, NOT synthetic or a continuous certified equity curve |
| User-supplied local CSVs | The user's chosen vendor, vintage and permissions | The user must audit provenance; software cannot infer missing vintages |

The new tutorials import no private data, forecasts, symbols, factor weights or study results.
Existing public records are preserved separately so changing the landing page does not erase
unfavorable evidence or interrupt its publication. Moving those records is not recertification.

A tool implementing a paper is not evidence that a trading strategy works. Corrected input
timing, a consistent price basis or a statistical diagnostic cannot independently establish
predictive power. DSR is not a posterior probability of making money.

The new walkthrough makes no network requests, collects no telemetry and performs no
inference. Its numbers are computed by the runnable example and checked against the published
presentation in CI. Its duration describes the guide, not the runtime of the research engine.
