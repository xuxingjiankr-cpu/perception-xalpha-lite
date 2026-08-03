# Perception-XAlpha V7: volatility-scaled Top3 intervals

## Status

Research-only / shadow-only / not a trading signal. No trading configuration,
position, order, risk gate, strategy overlay, or production decision path may
be read or changed.

## Why this is a separate counted trial

V6 correctly produced no Top3 selection, but also revealed a structural
limitation: one fold-level residual offset was added to every stock, so lower-
bound ranking was identical to point-forecast ranking. V7 changes exactly one
mechanism and counts it as a new research trial. It does not retune V6's market
or residual Ridge models, time splits, quantiles, costs, reliability thresholds,
or evaluation gates.

## Frozen conditional scale

For stock `i` at signal date `t`:

`scale(i,t) = sqrt(10) * abs(stock_volatility_20(i,t))`

The scale is clipped to the 5th and 95th percentiles estimated only on the
base-fit segment; missing values use the base-fit median. Calibration scores
are signed total-return residuals divided by this scale. Date-equal-weighted
10th and 90th standardized-residual quantiles form stock-specific lower and
upper bounds after multiplication by each prediction row's frozen scale.

The chronology remains:

`base fit -> 10-day purge -> 63-day calibration -> 10-day purge -> 63-day reliability audit -> 10-day outer purge -> prediction block`

Audit labels may enable or disable later intervals but cannot fit either Ridge
model, the volatility floor/cap, or the standardized residual bounds.

## Decision rule

Every V2 Top10 name receives a point estimate and stock-specific interval. At
most three are selected by descending lower bound, and only when each lower
bound is strictly above the inherited 0.30% round-trip cost and the complete V6
reliability gate passed. Zero selections is valid. Profit is never guaranteed.

Validation and shadow must satisfy every unchanged V6 success condition. V7
also reports whether conditional ranking actually differs from point ranking
and whether any apparent gain comes only from reduced coverage. Historical
success cannot promote or connect the result to trading.

## Frozen historical result (2026-08-03)

Run: `run_20260803_preregistered_conditional_top3_v7`  
Data: 2019-10-09 through 2026-08-03  
Prediction rows: 57,050  
Result: **rejected for trading; latest eligible Top3 count is zero.**

The conditional scale was active rather than cosmetic: it changed the Top3 set
on 123/126 validation days and all 114 comparable shadow days. Nevertheless it
made the return ranking worse.

| Period | Top3 method | 10d mean | Win rate | Tail-loss rate | Costed cumulative |
|---|---|---:|---:|---:|---:|
| Validation | V6 point ranking | 2.0113% | 65.08% | 8.73% | 22.35% |
| Validation | V7 conditional lower bound | 1.4480% | 63.49% | 8.73% | 15.08% |
| Shadow | V6 point ranking | 0.1476% | 40.35% | 34.21% | -0.97% |
| Shadow | V7 conditional lower bound | -0.4692% | 44.35% | 34.78% | -8.10% |

Relative to the identical V6 point book, V7 lost 0.5632 percentage points per
validation signal (HAC t = -1.3667) and 0.5889 points per shadow signal (HAC
t = -0.4138). The small shadow win-rate increase came with negative mean return,
worse tail rate and materially worse costed performance; it is not a usable
win-rate improvement.

No one of the 55 rolling folds passed every inherited reliability condition.
The validation Top10 interval covered 69.77% rather than the required 75%; in
shadow it covered 66.96%, with a 27.56% lower-bound violation rate. Mean interval
width expanded from 11.43 percentage points in validation to 22.13 in shadow.

On 2026-08-03, all ten lower bounds were negative, ranging from -7.44% to
-20.48%, so none cleared the 0.30% cost threshold. These wide bounds are not a
reason to shrink the scale after seeing the result. They show that recent
volatility alone does not repair the unstable V6 point forecast and that
low-volatility reranking discards profitable right-tail exposure in this sample.

V7 therefore ends this branch of research. A V8 that merely adjusts the scale
multiplier, clipping percentiles or confidence level on the same viewed windows
would be parameter fishing. Further work should require a new economic mechanism
for the forecast center or genuinely fresh forward data, not narrower intervals
chosen after failure.
