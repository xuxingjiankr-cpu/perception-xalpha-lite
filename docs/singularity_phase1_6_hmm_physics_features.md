# Singularity Phase 1.6: HMM physics auxiliary-feature audit

## Corrected research question

Phase 1.6 does not ask whether LPPLS or Koopman/DMD is an independent trading
edge. It asks whether their fixed, past-only diagnostics could later improve:

- HMM regime interpretation;
- HMM state-transition risk;
- calibrated 5/10-bar turning probability; or
- the distinction between stable trends and reversal-prone instability.

They cannot become BUY/SELL gates, online inference, position-sizing inputs or
independent trading signals in this phase.

## Gate 0 precedes modeling

No user-specified or independently documented business-event break date exists
in the current research record. An algorithm may report distribution-shift
candidates, but using the best candidate in a post-jump HMM would be ex-post
selection.

The checked-in configuration therefore fixes:

- `breakDate.value = null`;
- `breakDate.status = break_date_not_preregistered`;
- exploratory monthly candidate scanning that cannot use profitability or OOS
  model metrics; and
- `plannedModels.*Enabled = false`.

Until a break date and any time-decay half-life are separately preregistered,
Gate 0 may produce audits only. It must not fit or compare post-jump,
time-decay, multivariate or physics-augmented HMMs.

## Gate 0 data audit

The audit reuses the Phase 2A mootdx 5-minute source, universe, causal turning
labels and fixed LPPLS/DMD implementations. It reports:

- history, continuity, missing bars and adjustment/source-time limitations;
- exploratory pre/post comparisons for each candidate break;
- independent turning events and label-rate changes;
- fixed current-product-pool and survivorship bias;
- LPPLS and DMD availability by month, symbol, ETF category and intraday slot;
- the DMD complete-window bias found in Phase 2A;
- feature correlations with ordinary momentum, volatility and range proxies;
  and
- whether enough data exist in principle for each planned HMM path.

Exploratory candidate breaks are not selected, ranked by trading performance or
passed into a model.

## Auxiliary-feature audit contract

LPPLS rows retain a fit-success flag and failure reason. Critical time is a
distribution-derived time-to-critical-point diagnostic, never a point
forecast.

DMD rows retain validity and missing-ratio fields. Invalid DMD rows are not
silently dropped or forcibly imputed. This makes the time-of-day coverage loss
visible before any HMM study.

Any standardization and winsorization statistics are fitted only through the
fixed training-statistics cutoff. Their presence in Gate 0 is a feasibility
audit, not authorization to model.

## Required artifacts

Each immutable run directory contains:

- `data_audit.json` and `data_audit.md`;
- `post_jump_audit.json`;
- `coverage_bias_report.json`;
- `feature_missingness_report.json`;
- `label_distribution_report.json`;
- `feature_correlation_report.json`;
- `break_date_risk_report.md`;
- `phase1_6_gate0_result.json`; and
- `phase1_6_gate0_report.md`.

All artifacts are research-only. Phase 1, Phase 1.5 and Phase 2A outputs are
read-only inputs or historical references and are never overwritten.

## Conditions for a separate modeling run

A later commit may preregister and run the minimum HMM comparison only after:

1. a break date is supplied from information independent of model performance,
   or the post-jump path is explicitly abandoned;
2. the time-decay half-life, state count, observations and preprocessing are
   frozen;
3. DMD missingness is handled without hiding coverage loss or manufacturing
   observations;
4. all variants use identical OOS trading-date clusters; and
5. Phase 1.5 remains hash-pinned and excluded from tuning.

Passing Gate 0 would authorize a historical experiment only, never production
or forward-task integration.

## User-preregistered 2026-06-12 follow-up

The user subsequently fixed the break date at 2026-06-12. Only 15 trading days
exist from that date through the available 2026-07-03 endpoint, so a
post-jump train/calibration/OOS split remains impossible.

The historical model study therefore tests the existing frozen 3-state HMM
with auxiliary covariates:

- LPPLS is compared with `HMM + EWS` on the full-session same-sample
  population.
- DMD is compared on its explicitly separate complete-feature late-session
  population.
- DMD results are not extrapolated to the full session.
- No hidden-state emission, transition matrix or state duration is modified.

The LPPLS 5-bar result is retained as a frozen forward-retest hypothesis when
aggregate metrics improve but the trading-day-clustered confidence interval
or high-risk sample gate fails. At least 20 genuinely new trading days after
2026-07-03 are required before re-evaluation, with no historical retuning.
