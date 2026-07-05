# Singularity Phase 1.5 forward shadow plan

## 1. Phase 1 conclusion

Phase 1 remains `research_only / shadow_only`.

- Generic EWS produced the most credible incremental information. The 5-bar
  comparison improved Brier and AUC in 6/6 walk-forward folds; the 10-bar
  comparison jointly improved them in 4/6 folds.
- The HMM alone added little and was inconsistent. It remains an ablation
  input and is not promoted as a standalone model.
- The 20- and 30-bar results were unstable. The 30-bar high-probability tail
  was especially small and over-optimistic.
- The historical evaluation window had already been reused by other studies.
  The product universe can also contain coverage/survivorship bias.

No Phase 1 probability, `singularity_score`, or research action hint is
connected to trading.

## 2. Why Phase 2 is deferred

LPPLS, Koopman/DMD, Hawkes processes, SOC avalanche features and surrogate ML
would add model-selection degrees of freedom before the simpler EWS hypothesis
has passed a clean forward test. That would increase overfitting risk without
resolving the reused-window and source-coverage limitations.

Phase 1.5 therefore adds monitoring infrastructure only. It does not add a
feature, label, model, threshold, optimizer or trading rule.

## 3. Forward objective

Starting with the first eligible session after 2026-07-05, each completed
session is reconstructed from the point-in-time Eastmoney full-market snapshot
archive. The Phase 1 universe, feature definitions, EWS normalization, HMM
parameters, logistic coefficients and Platt calibration are frozen before the
first forward session.

Predictions are reconstructed post-close but each row uses only information
available through its completed five-minute bar. Future prices are written
only to a separate outcome file.

Primary horizons are 5 and 10 bars. Horizons 20 and 30 remain observational.
Horizon 60 stays null because it crosses the A-share session.

## 4. Frozen parameter inventory

The following cannot change within version `SP15-001`:

- fixed 50-symbol Phase 1 universe;
- 5-minute same-session bars;
- all Phase 1 base features;
- 12-bar EWS window and its four equally weighted components;
- EWS means and standard deviations fitted through 2026-05-20;
- the existing three-state HMM, its transition/emission parameters and
  session-open reset;
- local-extrema lookback, move threshold, volatility multiplier, breakout,
  trend, chase and round-trip-cost label parameters;
- active horizons 5/10/20/30 and null horizon 60;
- baseline/HMM/EWS/HMM+EWS feature sets;
- logistic and Platt-calibration specifications;
- probability bins and the 0.35 research high-risk threshold;
- data-quality thresholds and the prospective start date.

The checked-in frozen-model JSON is hash-pinned by the forward config using a
canonical-JSON SHA-256 that is stable across LF/CRLF checkouts. The daily
runner fails closed if either the model or Phase 1 config hash changes.
It contains no fitting path. Changing any item requires a new version and a
new preregistration.

## 5. Daily output

Each eligible session writes:

`outputs/edge_research/singularity_phase1_5/daily/YYYY-MM-DD/`

- `turning_probabilities.jsonl`: feature-time forecasts only; no outcome
  labels, orders or position instructions;
- `turning_outcomes.jsonl`: separate post-close same-session labels and
  forward-path diagnostics;
- `daily_result.json`: data-quality audit and metrics by horizon/variant;
- `daily_report.md`: concise primary-horizon report.

Daily metrics include Brier, LogLoss, AUC, ECE, probability buckets, false
positive rate and high-risk sample count. Two explicitly counterfactual
diagnostics are also reported:

- chase-failure residual rate after hypothetically excluding high-risk chase
  observations; and
- a sell-efficiency proxy for high-risk reversal-down observations.

Neither diagnostic is an order simulation or a proposed gate. Counts are
always reported so a lower rate cannot be credited to mechanically discarding
most observations.

## 6. Weekly/cumulative validation report

`forward_validation_report.json` and `.md` aggregate:

- independent forward trading days;
- full-period, daily and ISO-week metrics;
- baseline versus EWS and HMM+EWS deltas;
- the fraction of days/weeks with lower Brier and LogLoss and higher AUC;
- high-risk bucket count, predicted probability, realized hit rate and
  optimism;
- equal forecast-population checks;
- chase-failure and sell-efficiency diagnostics; and
- remaining sample days.

Before 20 independent sessions the status is always
`insufficient_forward_days`. Reports never change model parameters or enable a
trade path.

## 7. Conditions for reopening Phase 2 discussion

All of the following are required:

1. at least 20 new independent forward sessions;
2. stable EWS or HMM+EWS improvement over baseline at both 5 and 10 bars;
3. Brier, LogLoss and AUC improve in a majority of eligible daily or weekly
   comparisons;
4. each primary horizon has at least 100 high-risk observations and high-risk
   probability optimism is no more than 10 percentage points;
5. every compared variant forecasts the same population, proving that a result
   is not caused by mechanically reducing observations;
6. the full replay invariant suite and T88 causal test still pass; and
7. a separate human review explicitly approves a new preregistration.

Even satisfying these conditions permits discussion only. It does not permit
automatic implementation, production integration, BUY/SELL gating, position
sizing or model promotion.

## 8. Automation

The post-close runner is appended to the existing offline
`run_research_suite.ps1`. The existing Windows task `ETF Research Suite Daily`
runs that suite Monday through Friday at 16:30 KST. The runner selects only
snapshot directories later than 2026-07-05 and is idempotent for a matching
input hash. A changed snapshot for an already sealed forward day fails closed
instead of revising history.

The first scheduled eligible run is 2026-07-06. A weekend/no-data invocation
writes `waiting_for_first_forward_session` and does not create a forward day.
The existing task is configured as `Interactive only`, so the Windows user
session and machine must be available at run time; Phase 1.5 does not alter
that operating-system credential policy.
