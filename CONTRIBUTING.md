# Contributing Research Mechanisms

Perception-XAlpha Lite accepts small, auditable contributions that make a research claim more
testable. It does not accept broker connectivity, order placement, automatic promotion, private
performance screenshots, or a new method whose only justification is a better backtest.

## Two contribution paths

### Paper-backed mechanism

Use the [mechanism proposal](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/issues/new?template=mechanism-proposal.yml)
before writing substantial code. A proposal identifies:

1. the paper and exact claim being implemented;
2. the economic or statistical hypothesis;
3. which inputs exist at decision time;
4. the counterfactual or placebo that could falsify the claim;
5. the output role: feature, offline label, risk diagnostic, calibrator, or validation tool;
6. what the implementation explicitly does not claim.

Implementations belong in `src/xalpha_lite/`, use no network or execution interface, and must
have deterministic tests. A future-dependent outcome may exist only as an offline label.

### Contributor benchmark entry

Use the [benchmark submission template](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/issues/new?template=benchmark-submission.yml),
then add one JSON card under `benchmark/submissions/`. The benchmark ranks research-engineering
completeness—not returns—and verifies that cited code and tests actually exist.

```bash
python tools/contributor_benchmark.py --check --render
python -m pytest tests -q
python -m ruff check src tests tools examples
```

The generated files must remain unchanged after a second render:

```bash
python tools/contributor_benchmark.py --check --render
git diff --exit-code -- docs/CONTRIBUTOR_BENCHMARK.md docs/data/contributor_benchmark.json
```

## Pull-request contract

- One mechanism or factor family per pull request.
- Every paper URL must resolve to the publisher, DOI, arXiv, SSRN, or author manuscript.
- Feature code is strictly past-only and passes a prefix-invariance test.
- Offline labels are named and isolated from the feature namespace.
- A primary claim ships with a counter, placebo, or adversarial test.
- Validation and shadow results reject; they never refit or select direction.
- Trial-count implications are stated.
- No empirical security recommendation, order, position, or private result is committed.
- Documentation states limitations at the same prominence as capabilities.

The project may reject a technically correct implementation when its causal boundary cannot be
made explicit. That is a research-integrity decision, not a judgment on the underlying paper.
