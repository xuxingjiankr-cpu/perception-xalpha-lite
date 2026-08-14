# Opening-auction multi-task utility V2

V1 directly ranked rare large winners.  It increased average gross return but
reduced the positive-return rate and selected roughly three times as many severe
losses.  V2 preregisters exactly one correction: three shallow nonlinear heads
estimate an up day, conditional return and a loss of at least 3%, then combine
their within-day ranks with fixed 55% / 30% / 15% weights.

The nonlinear choice follows the evidence in Gu, Kelly and Xiu, *Empirical Asset
Pricing via Machine Learning*, that shallow trees can capture predictor
interactions in low-signal return data while deeper learning need not help:
<https://academic.oup.com/rfs/article/33/5/2223/5758276>.  The utility is a small,
auditable application of decision-focused ranking rather than an assertion that
multi-task learning creates information.

Every outer block uses a 504-session trailing window and a 10-session purge.  The
last 63 training sessions are used only for isotonic probability calibration and
positive-slope return calibration, with a separate two-session inner purge.  Test
data never change the model or utility weights.

The exact same 09:25 feature-complete support is used for the prior-close control
and candidate.  Same-session high, low, close, volume and amount are forbidden as
features.  Daily open remains a non-executable approximation after the auction is
observed, so a historical pass can only justify real auction and first-minute
forward data collection.  The study is research-only and cannot trade.

## Recorded V2 result

`run_20260814_auction_multitask_utility_v2` was closer but still rejected.  On
3,200 fixed Top10 observations, the candidate moved win rate from 51.59% to
52.16%, mean gross return from 0.0413% to 0.2743%, and mean cross-sectional return
percentile from 51.37% to 53.08%.  The win-rate gain was below the frozen one
percentage-point minimum, severe losses edged up from 3.22% to 3.31%, and mean
return remained approximately -0.0257% after the configured 30 bp friction.
Recent blocks were stronger, but selecting those blocks after seeing them would
be regime cherry-picking.  The frozen historical candidate therefore remains an
unvalidated forward hypothesis, not a trading model.
