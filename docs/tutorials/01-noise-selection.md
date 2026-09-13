# 01 — A high Sharpe after searching noise

[Run all cases](README.md) · [Generated report](../examples/audit-cases.md)

## Problem

Reporting the best backtest without its search history hides selection. This fixture creates
64 independent zero-mean Gaussian daily return series, not stock signals. It keeps every
variant, fixes one seed, and declares all 64 trials. Annualized Sharpe uses 244 sessions.

## Minimal reproduction

```bash
python examples/run_audit_cases.py --output-dir outputs/audit-cases
```

Inspect `noise_returns.csv` and the `noise_selection` case in `report.json`.

## What the tool checks

The **existing** `pbo()` and `deflated_sharpe_ratio()` functions audit the full candidate
family. A separate deliberately wrong control picks the best Sharpe on the final 252 rows;
the causal control picks using only the first 504. Both are evaluated on those **same final
252 rows**, with the same units and no cost model. This isolates selection timing.

The frozen example reports a full-sample best annualized Sharpe of 1.2402, DSR statistic
0.531984 and CSCV/PBO 0.5857. Hindsight selection returns 14.381803 bps/day in the evaluation
window; train-only selection returns 6.892377. **Both can be positive on one draw even
though the generating process has zero expected return.** These numbers are generated
teaching outputs, not empirical strategy results.

## Interpretation and limits

DSR is an approximate evidence statistic against a search-adjusted Sharpe threshold, **not
a posterior probability that a strategy is profitable**. CSCV is a resampling diagnostic, not
chronological forward validation. No costs, fills, persistence or market behavior are modeled.
A different seed can produce a different apparent winner; this example is not a power study.

## Paper → implementation

Bailey and López de Prado motivate correcting Sharpe evidence for selection and non-normality:
[author-hosted DSR paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
Bailey et al. examine how an in-sample winner ranks out of sample:
[author-hosted PBO paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
The example calls the library implementations; it does not claim exact coverage for arbitrary
dependent candidate families. See [the implementation map](../PAPERS.md).
