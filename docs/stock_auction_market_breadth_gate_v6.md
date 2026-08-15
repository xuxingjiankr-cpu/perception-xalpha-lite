# Opening-Auction Market Breadth Gate V6

## Status

Research-only and shadow-only. V6 cannot create an order or modify any trading,
position, risk, overlay, or execution setting. The historical window has been
viewed repeatedly; a historical pass can justify only a separately frozen
fresh-forward study.

## Hypothesis

A long-only Top10 process contains two different questions:

1. Which stocks are strongest relative to the cross-section?
2. Is the absolute intraday market opportunity sufficiently favorable?

V5 improved mean return and tail loss but did not materially improve the
probability of a positive stock return. V6 leaves the V5 stock model unchanged
and adds one day-level gate using information observable by 09:25:

- fraction of eligible stocks opening above the prior close;
- median opening gap and cross-sectional gap IQR;
- fractions with gaps above +1% and below -1%;
- prior-session intraday positive breadth.

The day label is whether more than half of the eligible cross-section closes
above its opening price. It is available only to historical training rows and
is never written back into a feature.

## Frozen selective rule

The L2 logistic model is fit on the base segment with training-only median
imputation and standardization. On the separate 63-day calibration segment,
the raw-probability median defines the high-opportunity cohort. The cohort is
enabled only when it contains at least 20 days and its 90% one-sided Wilson
lower bound exceeds 50%. A failed cohort causes the entire following test block
to abstain. No test label can change the model, threshold, or gate.

The selected stocks are compared with the prior-close control on the exact
same dates, support, and daily count. V4's minimum win-rate lift, coverage,
tail-loss, return, percentile, and block-stability checks remain in force.

The design follows the selective-prediction principle that a model may abstain
when its train-only evidence is weak, while explicitly auditing coverage and a
matched control. Methodological background: Geifman and El-Yaniv,
*SelectiveNet* (https://proceedings.mlr.press/v97/geifman19a.html), and
Gangrade et al., *Selective Classification via One-Sided Prediction*
(https://proceedings.mlr.press/v130/gangrade21a.html).

