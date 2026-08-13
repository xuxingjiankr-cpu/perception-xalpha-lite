# Sixteen-factor walk-forward weight audit V1

> Research-only. Hindsight weights are an invalid ceiling and are never published.

- run: `run_20260813_sixteen_factor_walkforward_weights_v1`
- data: `2019-01-02..2026-08-12`
- decision: `reject_learned_weights_keep_current_fixed_ranking`
- walk-forward blocks improving gross: `6/6`
- selection overfit exposure: `-0.024259%` per pick

## Latest trailing weights

| factor | prior | learned | replica min | replica max |
|---|---:|---:|---:|---:|
| gtja191/alpha_070 | 8.229% | 12.540% | 11.950% | 13.227% |
| gtja191/alpha_052 | 7.316% | 4.065% | 3.349% | 4.556% |
| qlib158/vstd60 | 7.197% | 4.378% | 3.721% | 4.587% |
| gtja191/alpha_097 | 6.305% | 11.377% | 10.673% | 11.834% |
| academic/retskew | 6.181% | 6.966% | 6.206% | 7.774% |
| gtja191/alpha_145 | 6.024% | 2.690% | 2.252% | 3.253% |
| qlib158/cord30 | 5.961% | 3.080% | 2.268% | 4.028% |
| gtja191/alpha_063 | 5.865% | 2.639% | 1.993% | 3.452% |
| alpha101/alpha_029 | 5.639% | 5.969% | 5.361% | 7.090% |
| alpha101/alpha_094 | 5.481% | 5.992% | 5.664% | 7.109% |
| alpha101/alpha_088 | 5.432% | 2.375% | 1.936% | 2.523% |
| qlib158/min5 | 5.370% | 2.407% | 2.040% | 2.592% |
| interaction/earnings_volume_confirmation | 6.250% | 14.348% | 13.673% | 14.803% |
| interaction/growth_momentum_confirmation | 6.250% | 15.000% | 14.876% | 15.000% |
| interaction/quality_low_volatility | 6.250% | 3.212% | 2.361% | 3.696% |
| interaction/cash_quality_reversal | 6.250% | 2.961% | 2.634% | 3.901% |

## Identical-date comparison

| method | mean gross | mean net | up rate | severe loss | max drawdown net |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0749% | -0.2251% | 51.45% | 3.53% | -55.84% |
| trailing | 0.1379% | -0.1621% | 51.14% | 4.38% | -47.89% |
| hindsight | 0.1136% | -0.1864% | 51.07% | 4.95% | -50.68% |

The latest learned weights use only the trailing window ending before the frozen purge. They remain a historical challenger because all evaluation windows have already been viewed.
