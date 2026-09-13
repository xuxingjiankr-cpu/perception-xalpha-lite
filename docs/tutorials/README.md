# Three mistakes you can reproduce

[Tool overview](../../README.md) · [45-second guide](../demo.html) · [Sample report](../examples/audit-cases.md)

These are synthetic teaching cases, not new factor research. Every symbol is fictional and
every input is generated locally. No API keys, credentials, private results or vendor downloads.

## Run from a checkout

```bash
git clone https://github.com/xuxingjiankr-cpu/perception-xalpha-lite.git
cd perception-xalpha-lite
python -m pip install -e .
python examples/run_audit_cases.py --output-dir outputs/audit-cases
```

Windows users can substitute `py -3.13` for `python`. Alternatively, use the
[Colab notebook](https://colab.research.google.com/github/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/examples/perception_xalpha_quickstart.ipynb).
Local execution and the static guide need no account; running Colab may require a Google account.

| Read | Compare | Reused implementation |
|---|---|---|
| [01 — Selecting noise](01-noise-selection.md) | Hindsight versus train-only selection on identical evaluation dates | `discovery.pbo`, `discovery.deflated_sharpe_ratio` |
| [02 — Disclosure timing](02-disclosure-timing.md) | Reporting-period alignment versus actual disclosure/revision availability | `pit.align_point_in_time_fundamentals` |
| [03 — Price basis](03-price-basis.md) | Raw cash VWAP versus a declared adjusted OHLC4 proxy on identical symbols | Isolated teaching adapter; no discovery-loader change |

## Inspect the output

```text
outputs/audit-cases/
  noise_returns.csv             # all 64 variants, not just the selected winner
  disclosures.csv               # original and revised synthetic statements
  pit_alignment.csv             # both alignments on every synthetic session
  price_basis_inputs.csv        # raw cash fields, adjusted OHLC, explicit basis metadata
  price_basis_comparison.csv    # both scores/ranks on the same four names
  report.json                  # results, assumptions and limitations
  report.md                    # human-readable report
  manifest.json                # source/library hashes, input/report hashes, environment
```

The fixed seed is part of the example, not a tunable research parameter. The report includes
positive realized noise returns as well as failed checks; no seed is searched to force a verdict.
Approximate statistical diagnostics are not a certificate that a strategy is real.

## Rebuild the public presentation

```bash
python examples/render_audit_walkthrough.py
python examples/render_audit_walkthrough.py --check
python -m pytest tests/test_audit_cases.py tests/test_audit_presentation.py -q
```

The renderer recomputes the cases, then produces the static preview, guided HTML and sample
reports. CI rejects stale assets. The guide takes 45 seconds to explain recorded outputs;
it does not pretend to run a research engine in the browser. No tracking or uploads are added.

For the larger synthetic factor search, use the existing `xalpha-demo`. For your own data,
start with the [data doctor](../DATA_DOCTOR.md) and [evidence contract](../EVIDENCE_LAB.md).
