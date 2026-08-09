# Twelve-factor Alpha070/Alpha131 incremental study V3

> **Preregistered research-only ablation. It does not change trading or select a policy
> for deployment.**

The existing model contains twelve price-volume factors. `gtja191/alpha_070` is already
the first member, oriented negatively, with a frozen V6 weight of approximately 2.20%.
Adding another copy would silently double-count the same signal and is forbidden.

`gtja191/alpha_131` is the only new factor. It combines the cross-sectional rank of the
one-day VWAP change with the time-series rank of the correlation between close and
50-day average volume. It is evaluated with positive orientation and the same causal
liquidity neutralisation as the baseline. No industry-neutral claim is made.

Three policies are fixed before their Top10 outcomes are read:

1. the exact frozen weighted twelve-factor baseline;
2. the baseline scaled by 12/13 plus Alpha131 at 1/13; and
3. a symmetric Alpha070/Alpha131 pair at 1/14 each, with the other eleven weights
   proportionally scaled to 12/14.

Every policy selects exactly ten names on the same date and eligible universe. Outcomes
use close `t` scoring, buyable open `t+1`, sellable open `t+2`, A-share T+1 constraints
and 30 bps round-trip cost. Alpha131 is useful only if it raises both gross-up and
net-positive probability, lowers net-loss probability, raises mean net return and
improves the matched-control spread relative to the twelve-factor baseline. Mean-return
improvement additionally requires paired day-clustered HAC t >= 2.0.

The source twelve factors and their V6 weights previously failed trading readiness, and
the historical window has been viewed. This study can reject Alpha131; it cannot validate
the baseline, choose a forward policy, or connect either policy to trading.
