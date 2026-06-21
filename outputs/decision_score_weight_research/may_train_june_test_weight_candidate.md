# Decision-Score Weight Research: May Train / June Test

Status: `diagnostic_only / candidate_not_promotable`

- May training universe uses the known final-day-turnover-contaminated Yahoo fallback.
- June test rows are point-in-time, but the scorer design post-dates the sample.
- BUY decisions only; target is next-snapshot-to-close return minus 14bps.
- One fixed ridge candidate (`alpha=10`), day-balanced. No hyperparameter search.
- No live config, score range, strategy parameter or execution lock was changed.

## Sample

- train: 95 BUY decisions / 18 days
- test: 60 BUY decisions / 13 days
- formal minimum: 20 test days

## June fixed-test comparison

| model | Pearson | Spearman | top-minus-bottom net return |
|---|---:|---:|---:|
| frozen total score | 0.091159 | 0.129425 | 0.002658 |
| May ridge candidate | -0.205216 | -0.154043 | -0.001234 |

## Data-derived shadow hypothesis

- Keep the existing frozen component weights unchanged.
- Shadow-tag BUY decisions with `total_score >= 71.0`.
- June observations: 28 BUY decisions; cost-adjusted mean 0.2235%.
- May-low-threshold comparison group mean: -0.0423%.
- This is a forward hypothesis only; `trade_gate_enabled=false`.

## Candidate standardized coefficients

| component | active in May | coefficient |
|---|---|---:|
| market_regime_score | true | -0.00146603 |
| relative_strength_score | true | +0.00070168 |
| liquidity_score | true | -0.00213819 |
| entry_quality_score | true | -0.00187772 |
| execution_score | false | +0.00000000 |
| counterfactual_score | true | +0.00057835 |
| risk_penalty | true | +0.00472078 |

## Verdict

- test_days_gate: `False`
- candidate_improves_both_metrics: `False`
- promotion_allowed: `False`
- These workdays can prioritize hypotheses, but cannot authorize weight changes.
- Keep the existing scorer frozen; compare this one preregistered candidate only on new real-forward days.
