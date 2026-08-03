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
