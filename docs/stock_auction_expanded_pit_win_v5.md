# Expanded PIT Auction Win Classifier V5

## Status

Research-only and shadow-only. This experiment cannot create orders, modify a
trading configuration, write an overlay, alter a risk gate, or promote itself.
The historical evaluation window has already been viewed, so a passing result
can justify only a separately preregistered fresh-forward study.

## Hypothesis

The earlier auction classifiers compressed the prior-close stock information
into one cross-sectional rank. V5 tests one frozen representation change:
retain the twenty individual point-in-time price, fundamental, and
price-fundamental interaction ranks, and allow the existing nonlinear model to
learn interactions between them and the opening-auction features.

The twenty inputs are the twelve price characteristics, four direct
fundamental mechanisms, and four preregistered price-fundamental interactions
from `price_fundamental_interaction_20`. Every input is shifted by one full
trading session before it is joined to the auction row. No same-session close,
high, low, volume, or amount is permitted.

This is motivated by evidence that stock-return prediction is a low
signal-to-noise problem in which nonlinear interactions among multiple
characteristics can matter. It does not imply that a larger feature set is
automatically better; the sole candidate must beat a same-day, same-support,
same-count control under purged walk-forward evaluation. See Gu, Kelly, and
Xiu, *Empirical Asset Pricing via Machine Learning*, Review of Financial
Studies (2020):
https://academic.oup.com/rfs/article/33/5/2223/5758276

## Frozen evaluation

- Reuse the V4 probability-up objective and model hyperparameters.
- Reuse the 504-day training lookback, ten-day outer purge, refit cadence, and
  63-day train-only reliability window.
- Reuse the one-sided Wilson reliability requirement and the tail ceiling.
- Permit zero to ten selections per day.
- Compare against the prior-close control on exactly the same day, support,
  and selection count.
- Evaluate win rate, gross return, configured-friction return, severe-loss
  rate, and walk-forward block consistency.

Selective prediction is a legitimate way to trade coverage for reliability,
but selection must be calibrated without using the test labels. Relevant
methodological references include SelectiveNet
(https://proceedings.mlr.press/v97/geifman19a.html) and One-Sided Prediction
(https://proceedings.mlr.press/v130/gangrade21a.html).

## Interpretation rule

A historical pass is not a trading approval. V1 through V4 and the current
window have already influenced the research path, increasing multiple-testing
and researcher-degree-of-freedom risk. V5 can be rejected on this history; if
it passes, it can only be frozen and evaluated on future, unseen trading days.

