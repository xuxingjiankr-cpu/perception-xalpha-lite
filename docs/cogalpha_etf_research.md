# CogAlpha ETF research integration

## Boundary

This is an offline factor-discovery system inspired by *Cognitive Alpha Mining via
LLM-Driven Code-Based Evolution*. It is not an online inference model and is not an
order policy. It cannot read or write the live/paper agent configuration, strategy
overlay, position sizing, risk gates, broker state, or `build_decision()`.

The historical run can only produce a `diagnostic_only` candidate. Promotion is
permanently disabled in the preregistered configuration. A separate change, separate
approval, frozen replay and at least 20 new forward trading days would be required
before any trading integration could be discussed.

## Why this implementation differs from the paper

The paper used `gpt-oss-120b` on an H100. This workstation has a 6 GB RTX 4050 and no
installed Ollama model. Reproducing the paper's compute budget is therefore neither
possible nor proportionate.

Instead, the implementation preserves the research logic while reducing attack and
overfit surfaces:

1. Twenty-one preregistered factor roles produce interpretable seed expressions.
2. Mutation and crossover operate on a bounded expression tree.
3. An optional Ollama provider may propose more expression trees later.
4. Generated text is parsed as a safe JSON DSL. Arbitrary Python is never executed.
5. Every expression is checked for allowed fields, allowed windows, maximum depth and
   non-negative lag.
6. Fitness and orientation use training data only.
7. Validation selects one frozen ensemble from the training elites.
8. The 2026 historical test is report-only and cannot select or mutate a candidate.
9. PBO and a deflated-significance warning account for the candidate search burden.

## Timing contract

- Feature timestamp: close of trading day `t`.
- Earliest assumed action: trading day `t+1`.
- Graded return proxy: close `t+1` to close `t+2`.
- Rolling and lagged features use current or prior rows only.
- A label is never exposed to the candidate generator or expression evaluator.
- The same expression evaluated on a prefix must reproduce the corresponding prefix
  of the full-sample result.

This is conservative with respect to close-generated daily factors. It is not a claim
that the proxy matches intraday ETF fills. A candidate that survives here still needs
the existing minute replay with bid/ask and fill constraints.

## Data and split

- Source: local mootdx ETF daily bars.
- Eligibility: at least 250 observations and median daily amount of CNY 30 million.
- Train: through 2024-12-31.
- Validation: 2025-01-01 through 2025-12-31.
- Historical test: 2026-01-01 onward.
- Cost: 15.5 bps round trip multiplied by portfolio turnover.

The 2026 window has already been viewed by other research in this repository. It is
not a pristine final OOS window. Even a positive result remains historical evidence,
not permission to trade.

## Outputs

Each run is isolated under:

`outputs/edge_research/cogalpha_etf/<run_id>/`

Files:

- `result.json`: full candidate audit, metrics, guards and verdict.
- `report.md`: compact human-readable comparison.
- `frozen_candidate.json`: research-only expressions selected before opening the test.

The candidate artifact explicitly says `research_only_not_a_trade_signal` and
`mayPromoteAutomatically=false`.

## Running

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_cogalpha_etf.py --self-test
py -3.13 scripts/research_cogalpha_etf.py
```

For a quick smoke run:

```powershell
py -3.13 scripts/research_cogalpha_etf.py --maximum-candidates 24
```

The optional Ollama provider is disabled. Enabling it requires a separately
preregistered local model name. Installing a model, changing prompts, candidate caps,
windows or selection metrics creates a new research version; it must not silently
rewrite this experiment.

## Evidence gate

A candidate is not eligible for further consideration unless it:

1. beats the fixed baselines on validation after costs;
2. remains positive on the historical test without test-time selection;
3. is not consistent with luck after the full candidate count is considered;
4. does not concentrate all gains in one month or one ETF category;
5. survives the existing minute replay and execution-cost model;
6. accumulates at least 20 new independent forward days and 200 events;
7. passes all replay invariants; and
8. receives separate human approval in a separate change.

Until then, the honest expected effect on automatic-trading success rate is unknown.

## Adaptive rerun 2026-07-15

The preregistered adaptive rerun used four train-feedback generations, a different
random seed and a maximum of 160 evaluated candidates. Parent selection was recomputed
from training fitness after every generation. Validation, historical test and the
previous run's test result were prohibited as feedback.

The run also carried the first run's 83 evaluated candidates into the multiple-testing
trial count, for 243 total CogAlpha trials. The frozen ensemble increased training
RankIC IR to 2.7484 but produced net long-only IR of -0.4003 in training, -2.2156 in
validation and -1.1914 in the historical test. PBO was 0.6857 and the deflated-
significance check remained consistent with luck.

Interpretation: the adaptive loop learned to optimize correlation more aggressively,
but did not learn a cost-harvestable long-only ETF edge. No third historical search is
authorized on the same window; doing so would only increase selection contamination.
