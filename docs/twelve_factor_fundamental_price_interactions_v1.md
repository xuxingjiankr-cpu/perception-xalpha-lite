# Point-in-time fundamental by price-volume interactions V1

Status: `research-only / shadow-only / not trading`.

## Purpose

Financial statements and market behaviour answer different questions. Fundamentals
describe an issuer's operating state; price and volume describe how quickly the market
is processing that state. This study tests four constrained conjunction factors instead
of blindly enumerating every fundamental-by-price formula.

## Frozen mechanism interactions

1. **Earnings innovation × amount surprise**: a positive filing accompanied by unusual
   same-session trading activity is a limited-attention / information-recognition test.
2. **Growth acceleration × 20-session momentum**: improving growth plus gradual price
   diffusion is a constrained post-announcement-drift test.
3. **Quality × low 20-session volatility**: high accounting quality with stable realised
   prices attempts to separate durable quality from speculative quality.
4. **Cash-flow quality × five-session reversal**: cash-supported profitability after a
   short drawdown is a quality-at-temporary-discount test.

Every component is converted to a same-date cross-sectional rank. The conjunction is
the square root of the two rank products and is reranked cross-sectionally. No outcome,
future price, future volume, historical winner or fitted interaction weight enters the
factor definition.

The amount-surprise denominator uses only the preceding 20 sessions and ends at `t-1`;
the signal-date amount is observed at the signal-date close. Momentum, volatility and
reversal end at the same signal-date close. Fundamental availability remains the first
market session strictly after `max(noticeDate, updateDate)`.

## Ablation and acceptance

The current twelve-factor features and fixed guarded Top10 form the baseline. Four
single-interaction models are explanatory ablations. They cannot be used as a historical
winner menu. The sole primary candidate adds all four interaction ranks to the twelve
price-factor ranks.

All six models use identical stock-date support and the same selected Top10. The model
family, regularisation, fit/calibration periods and probability calibration are frozen.
Probability stretching and hyperparameter search are forbidden.

The primary candidate must improve both validation and shadow on:

- a majority of gross-up AUC/Brier/LogLoss overall;
- a majority of severe-loss AUC/Brier/LogLoss overall;
- a majority of gross-up AUC/Brier/LogLoss in the fixed Top10;
- expected-return MAE overall; and
- expected-return MAE in the fixed Top10.

Historical windows are already viewed. A complete pass can only justify a separate
60-session fresh-forward preregistration; it cannot alter the dashboard or trading.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_twelve_factor_fundamental_price_interactions_v1.py `
  --run-id run_YYYYMMDD_fixed_interactions_v1
```

Artifacts are isolated under:

`outputs/edge_research/twelve_factor_fundamental_price_interactions_v1/<run_id>/`

