# AlphaBench-inspired search protocol

Perception-XAlpha Lite can compare three bounded symbolic-search paradigms without
letting any of them grade its own work:

- **Chain of Exploration (CoE)** retains one train-best anchor per economic
  phenomenon before sequential refinement, preserving mechanism diversity.
- **Tree of Thought (ToT)** uses deterministic pairwise train comparisons and a
  narrow, preregistered branching frontier.
- **Evolutionary search (EA)** applies bounded mutation, refinement and crossover
  to a train-selected population.

The design is inspired by *AlphaBench: A Benchmark for LLM-Driven Alpha Mining*
(ICLR 2026). It is not a reproduction of that paper's data, model results or
reported performance.

## One search space, one budget, one judge

Every arm uses the same allowlisted JSON DSL, point-in-time input panel, candidate
budget and train-only score. The generation schedule, frontier sizes, branching
limits and operator weights are frozen in `configs/alphabench_search.json`.

An LLM adapter may eventually propose a valid DSL expression or order a research
queue. It may not execute generated code, inspect validation or shadow outcomes,
make an absolute factor-validity judgement, promote a factor, or connect to an
order path.

All candidates still pass through the ordinary deterministic pipeline:

1. static DSL and point-in-time validation;
2. train-only scoring and behavioral-correlation pruning;
3. Primary/Counter/Placebo evaluation;
4. purged walk-forward validation;
5. factor-relative and project PBO;
6. cumulative-trial Deflated Sharpe Ratio;
7. separately frozen forward research before any empirical claim.

## What is measured

The result artifact reports per-arm search diagnostics:

- generated and unique candidates;
- duplicate rate;
- train fast-screen pass rate;
- child-versus-parent train improvement rate;
- Stage-2 count and final validation-survival rate.

These measure how efficiently a paradigm explores the declared grammar. They are
not estimates of return, probability of profit or deployability. A zero-survivor
result is valid.

## Run

Use the standard CLI with the explicit protocol configuration:

```bash
xalpha-lite \
  --prices data/prices.csv \
  --fundamentals data/fundamentals.csv \
  --config configs/alphabench_search.json \
  --output outputs/alphabench-research.json
```

The output remains `diagnostic_only_research_only_not_trading`, with empty
`orders` and `automatic_trading_changes` arrays.

## Reference

*AlphaBench: A Benchmark for LLM-Driven Alpha Mining*, ICLR 2026:
https://proceedings.iclr.cc/paper_files/paper/2026/file/4f3820576130a8f796ddbf204c841487-Paper-Conference.pdf
