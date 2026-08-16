# Contributor Benchmark

> This leaderboard measures research-engineering completeness, not returns, alpha,
> statistical significance, or deployment readiness. A high score cannot authorize trading.

Each card is machine-checked against a real callable or audited DSL expression, a cited
paper, an explicit point-in-time boundary, and repository test/example paths.

| Rank | Contribution | Kind | Score | Tier | Prefix causal test | Adversarial test |
|---:|---|---|---:|---|:---:|:---:|
| 1 | [Fixed-window Dynamic Mode Decomposition residual](https://doi.org/10.1017/S0022112010001217) | mechanism | 90/100 | causal-tested | yes | no |
| 2 | [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742) | mechanism | 75/100 | reproducible-implementation | no | no |
| 3 | [Generic early-warning statistics](https://doi.org/10.1038/nature08227) | mechanism | 75/100 | reproducible-implementation | no | no |
| 4 | [Causal two-state Hidden Markov filter](https://doi.org/10.2307/1912559) | mechanism | 75/100 | reproducible-implementation | no | no |
| 5 | [Fixed-grid simplified LPPLS diagnostics](https://doi.org/10.1016/j.physa.2013.04.012) | mechanism | 75/100 | reproducible-implementation | no | no |
| 6 | [Triple-barrier offline labels](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | mechanism | 75/100 | reproducible-implementation | no | no |

## Scoring contract

- Paper traceability: 20
- Explicit hypothesis and causal boundary: 15
- Resolvable implementation or validated primary/counter DSL pair: 20
- Falsification design: 15
- Executable prefix-causality test: 15
- Executable adversarial test: 10
- Reproduction example: 5

CI regenerates this page from `benchmark/submissions/*.json` and rejects stale or
unresolvable entries. See [CONTRIBUTING.md](../CONTRIBUTING.md) to submit one.
