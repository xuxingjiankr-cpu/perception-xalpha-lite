# Perception-XAlpha V8: calibrated market-opportunity gate

Status: **preregistered research-only / shadow-only / not a trading signal**.

## Question

Can the frozen V2 cross-sectional Ridge ranker keep choosing the same Top3 stocks,
while a separate probability model avoids ten-session market environments in which a
long-only A-share basket is unlikely to clear the fixed 30 bp round-trip cost?

V8 is not another stock-return model and does not change V2 ranking. Its only increment
is the replacement of V2's deterministic trend-and-breadth gate with one calibrated
market-opportunity probability. This separates relative stock selection from the
absolute long/cash decision.

## Frozen design before the historical run

- Horizon: 10 trading days; signal at day `t` close, hypothetical entry at `t+1` open.
- Candidate pool: frozen V2 Top50; selected names: frozen V2 ranks 1-3.
- Label: equal-weight eligible-A-share future 10-day return strictly above 0.30%.
- Past-only features: 20/60-day market return, 20-day market volatility and 20-day
  breadth. No stock future return enters the model.
- Model: L2 logistic regression, `C=0.1`, training-median imputation and training-only
  standardization.
- Calibration: Platt logistic regression on a separate 63-day calibration segment.
- Reliability audit: a later, never-fit 63-day segment.
- Outer rolling fit: 756 days, minimum 504, refit every 21 days, with a 10-day purge.
- Nested chronology: base fit -> 10-day purge -> 63-day calibration -> 10-day purge
  -> 63-day audit -> outer 10-day purge -> prediction block.
- Selection threshold: calibrated probability >= 60%, and only when every audit
  reliability check passes. Zero selections are valid; the policy never fills a quota.

The reliability audit must beat the frozen base-rate forecast on Brier and LogLoss,
have AUC >= 0.55, ECE <= 0.15, positive calibration slope, and at least 10 audit dates
in the >=60% bucket with realized hit rate >=60%.

## Evidence required

Validation and shadow must independently have at least 20 signal days, 8 independent
10-day events and 15% calendar coverage. The Top3 basket must have win rate >=60%,
positive mean return and positive costed cumulative return. Each rank slot must have at
least 10 observations, positive mean and win rate above 50%.

To prevent an abstention illusion, the same frozen V2 Top3 basket is compared between
gated and excluded dates. The gated cohort must have higher mean return with HAC
`t >= 1.65`, no lower win rate and no worse 3% tail-loss rate. Aggregate calibrated
probability metrics must also beat the frozen prior and satisfy AUC/ECE thresholds.

## Interpretation boundary

The validation and shadow windows have already been inspected by earlier V1-V7 work.
Even a pass is only a forward-shadow hypothesis; it cannot change production, place an
order, alter a risk gate or claim guaranteed profit. A failure ends this exact V8
specification. It must not be repaired by searching probability thresholds or adding
models on the same evaluation windows.

## Result

Not run at preregistration time.
