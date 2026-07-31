# Perception-XAlpha Autonomous Factor Discovery v2

## What changed

The MVP converted recurring market anomalies into eight fixed factor templates. That
was an automated validator, not an autonomous discovery system. Version 2 makes the
research loop generative while keeping the executable language and search budget
strictly bounded.

The loop is:

1. detect recurring, past-only market phenomena;
2. seal immutable `PhenomenonTicket` records;
3. let the `ResearchDirector` select the most useful unanswered research questions;
4. generate falsifiable market-mechanism hypotheses;
5. synthesize causal DSL programs;
6. evolve only from train-only fast-screen feedback;
7. require a Primary/Counter/Placebo bundle;
8. run purged historical validation with explicit 15.5 bps round-trip cost;
9. append every experiment and rejection to SQLite;
10. retain train-only parents for the next weekly cycle.

Validation and shadow metrics never enter the generator or persistent parent pool.

## Seven mechanism archetypes

| Archetype | Economic mechanism | Typical counter |
|---|---|---|
| `order_splitting` | parent orders are spread over time | short-horizon reversal |
| `liquidity_reversal` | urgent flow temporarily dislocates price | persistent trend |
| `risk_budgeting` | volatility-sensitive mandates adjust exposure slowly | stable trend |
| `information_diffusion` | information reaches assets at different speeds | contemporaneous noise |
| `crowding_unwind` | crowded positions unwind under common constraints | continued herding |
| `benchmark_rebalancing` | index and allocation flows persist around rebalance demand | idiosyncratic reversal |
| `volatility_feedback` | price shocks change risk budgets and future liquidity | unconditional direction |

Each archetype specifies supported phenomena, a causal input/operator boundary, a
forced trader, persistence mechanism, a falsifiable prediction, a counter mechanism,
and an observable failure condition.

## Factor bundles

Every Stage-2 candidate is evaluated as a bundle:

- **Primary**: the proposed mechanism expression.
- **Counter**: a viable competing economic mechanism.
- **Placebo**: a causally delayed version of the Primary expression.
- **Failure condition**: a preregistered state in which the mechanism is expected to
  weaken.

A Primary factor is not considered historically validated unless it beats both its
Counter and Placebo after costs in the same validation sample. Passing historical
validation still cannot create a trade signal or promotion.

## Two-stage evaluation

### Stage 1 — train-only fast screen

- DSL/static validation;
- prefix-invariance leakage test;
- missingness and cross-sectional variation;
- train RankIC and costed long-only performance;
- expression-complexity penalty;
- behavioral duplicate rejection;
- bounded train-only evolutionary feedback.

### Stage 2 — purged historical evaluation

- frozen train/validation/shadow split;
- label-horizon purge;
- five expanding walk-forward folds, each with a full label-horizon purge;
- next-open target timing inherited from CogAlpha;
- 15.5 bps round-trip cost;
- Primary vs Counter vs Placebo;
- failure-condition diagnostics;
- PBO and multiple-testing ledger;
- no validation/shadow feedback to synthesis.

## Rejection reasons

The system records explicit reasons including:

`STATIC_DSL_REJECTED`, `PREFIX_LEAKAGE`, `TOO_MANY_NAN`,
`INSUFFICIENT_CROSS_SECTION`, `INSUFFICIENT_TRAIN_DAYS`,
`WEAK_TRAIN_RANK_IC`, `WEAK_TRAIN_RANK_IC_IR`, `NEGATIVE_COSTED_TRAIN_IR`,
`DUPLICATE_BEHAVIOR`, `COUNTER_NOT_BEATEN`, `PLACEBO_NOT_BEATEN`,
`VALIDATION_RANK_IC_FAILED`, `VALIDATION_COSTED_IR_FAILED`, and
`PROJECT_PBO_FAILED`.

Rejections are evidence. They are never silently discarded.

## Persistent research memory

SQLite stores immutable research cycles, plans, hypotheses, bundles, experiments and
rejections. Update/delete triggers protect append-only tables. The only cross-cycle
adaptive state is a bounded `train_parent_pool.json`, containing expressions and
train-only fast-screen metrics. It contains no validation or shadow metrics.

## CLI

```powershell
py -3.13 scripts/research_perception_xalpha_autonomous.py run
py -3.13 scripts/research_perception_xalpha_autonomous.py run --no-state --maximum-candidates 12
py -3.13 scripts/research_perception_xalpha_autonomous.py status
py -3.13 scripts/research_perception_xalpha_autonomous.py queue --limit 10
py -3.13 scripts/research_perception_xalpha_autonomous.py factor --factor-id factor_xxx
py -3.13 scripts/research_perception_xalpha_autonomous.py self-test
```

The weekly runner is `scripts/run_perception_xalpha_autonomous_weekly.ps1`.

## Permanent boundary

This system does not import the paper agent or broker, does not write trading
configuration or overlays, and does not call `build_decision()`. Its shadow artifact
contains no executable instruction. Historical validation is a research state, not
permission to trade. A separate preregistered forward protocol and explicit human
approval remain mandatory.
