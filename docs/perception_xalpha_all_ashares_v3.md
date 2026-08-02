# Perception-XAlpha all-A v3 preregistration

Status: `research_only / shadow_only / not_trading`.

## Single hypothesis

The v2 smoke found persistent cross-sectional Rank IC but negative costed top-decile
returns. V3 tests one mechanism-level explanation: a weak monotone ranking may be
harvestable by a broad, fully-invested linear rank tilt with materially lower turnover,
while deterministic A-share factor seeds may cover economically plausible OHLCV
mechanisms that random grammar misses.

This is not a clean OOS test. The historical window through 2026-07-31 has already been
inspected. No v3 historical result can promote, alter a trading configuration, create an
order, or enter a BUY/SELL gate. A survivor must be frozen and validated on at least 60
new trading days.

## Frozen research choices

- Universe: current discoverable SH/SZ/BJ A shares; known survivorship/ST limitation.
- Signal time: close of day t, using only information available through t.
- Hypothetical entry: next open; label horizon: 10 trading days.
- Round-trip cost: 30 bps.
- Book: size-neutralised, fully-invested linear rank tilt, overlapping 10-day cohorts.
- Search: at most 128 candidates and 12 Stage-2 bundles.
- Structured seeds: at most six per selected mechanism question from
  `a_share_tradeable_v1`; remaining candidates use the existing causal DSL grammar.
- Selection feedback: train only. Validation and shadow remain quarantined.
- Prior recorded research trials: 715.

## Structured factor families

- short-horizon reversal and liquidity-shock reversal;
- momentum excluding the most recent interval;
- overnight/intraday information diffusion;
- flow-confirmed momentum and liquidity persistence;
- crowding unwind, MAX-return reversal and range-position reversal;
- low volatility, volatility compression and drawdown/flow interaction.

Every expression is generated inside the existing DSL. Arbitrary Python, hosted-model
generation, broker access and production configuration writes remain forbidden.

## Required evidence

A historical candidate must have positive validation costed IR, pass at least three of
five purged walk-forward folds, beat its Counter and Placebo under the identical book,
and survive project PBO/DSR diagnostics. Reports must separate gross return, turnover,
cost drag and net return. Historical success creates only a frozen forward hypothesis.

