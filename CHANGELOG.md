# Changelog

## Unreleased — Data contracts become executable

- `xalpha-doctor` audits price and fundamental inputs before discovery. Duplicate bars,
  impossible OHLC, missing notice dates, insufficient chronological capacity, absent
  neutralization controls and missing tradability fields fail closed in machine-readable JSON.
- `xalpha-demo` runs the installed package end to end without a checkout or private dataset. It
  emits deterministic synthetic inputs, the doctor report, frozen config and full research
  result in one directory.
- The daily forward-record summary is rendered from language templates, keeping generated
  figures out of source code while preserving the English and Chinese reader views.
- The user-supplied literature registry machine-checks paper metadata, claimed public-code
  mappings, implementation boundaries and next falsifiable steps. Black--Litterman and
  Hoeffding now have executable contribution cards; unrelated papers remain visibly excluded.

## 0.5.0 — The record publishes itself

The daily record now runs on GitHub Actions from public data, and publishes the name **before**
the session it applies to.

- `tools/daily_record.py` — selects the name, scores every published pick whose holding window
  has elapsed, and redraws. Only picks that were actually published are scored; recomputing
  what the rule would say today would quietly turn the forward record back into a backtest.
- `tools/audit_returns.py` — audit somebody else's backtest in their own fork: CSCV probability
  of backtest overfitting, deflated Sharpe against the declared trial count, and White's Reality
  Check across the family. The example that ships with it is 24 variants of pure noise whose
  best has an annualised Sharpe of 1.11 — against 1.18 expected from noise at that trial count.
- Specification v4 (`dea0e608`) pins the data provider, adjustment convention, board scope,
  window and listing filter. v3 named a universe rule without saying where the universe came
  from, and two faithful implementations disagreed on 614 of ~4,700 eligible names; v3 keeps the
  one pick it published and is retired rather than edited.
- Coverage gates. Under parallel load the data source answers a throttled query with an empty
  result set and `error_code "0"`, indistinguishable from a symbol with no history — two runs
  over the same 5,202 codes differed by 143,028 bars. Empty answers are retried on a rebuilt
  session and swept serially; below 98% symbol coverage or 95% quoting on the ranked session the
  run aborts rather than selecting a name from whichever symbols replied.
- Chinese README, anchor navigation, and the first tagged releases.

## 0.4.0 — Forward records

- `forward.py` — frozen specifications with four refusals: no overwrite, digest verified on
  load, one entry per session, and scoring only fully elapsed holding windows. Factors travel
  inside the specification as portable DSL so a published record is reproducible by its reader.
- `universe.py` — `sealed_bar_limits` infers limit-locked sessions from the bars. `build_panel`
  previously defaulted the flags to `False`, so a panel built from plain OHLCV backtested fills
  on locked boards; those legs carry +6.05% against +0.38% for legs that could be traded.
  `point_in_time_eligibility` decides membership only from prior sessions.
- `book.py` — long-only top-N construction sharing the tranche and turnover accounting with the
  neutral book, so a difference between them is construction and never charging.
- `session_index` in the panel, making trend-quality factors (`ts_corr(close, t, w)²`)
  expressible in the DSL at all.

## 0.3.0 and earlier

Point-in-time alignment, the audited factor DSL, bounded synthesis with purged walk-forward,
counterfactual and placebo controls, PBO and deflated Sharpe, the evidence lab (stationary
bootstrap, Reality Check, Romano–Wolf step-down), and the decision toolkit.
