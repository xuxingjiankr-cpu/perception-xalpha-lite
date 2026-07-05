# Koopman minute-direction research

## Scope

This experiment redefines Koopman/DMD as a next-minute directional context
model. It does not reuse the earlier late-session turning-risk conclusion.

The source is the local one-minute mootdx archive for 18 ETFs from 2026-02-06
through 2026-07-03. A fixed 24-minute, delay-3 linear DMD operator is rebuilt
after every completed same-session minute. After the causal warm-up, forecasts
are available from 10:00 through 14:59.

The implementation is fixed before the walk-forward test:

- no kernel or deep Koopman;
- no cross-session windows;
- no DMD parameter search;
- fixed standardization through 2026-04-27;
- next-minute direction is stored in a separate offline label table;
- exact-zero next-minute returns are neutral and excluded from the binary fit;
- the walk-forward test begins on 2026-04-28;
- variants forecast exactly the same rows.

## Historical result

Run `koopman_minute_direction_20260705_v1` used 45 walk-forward trading days
and 122,738 non-neutral next-minute observations.

| Variant | Brier | LogLoss | AUC | ECE | Accuracy |
|---|---:|---:|---:|---:|---:|
| causal baseline | 0.248069 | 0.689648 | 0.562955 | 0.020972 | 54.8494% |
| baseline + Koopman | 0.248040 | 0.689579 | 0.563148 | 0.020541 | 54.8290% |

The candidate improved all four preregistered probability metrics, 4/5 folds,
3/4 months, and all six ETF categories. The trading-date-clustered Brier
delta was -0.00002939 with a 95% interval of
[-0.00004653, -0.00001229].

The statistical gate therefore passed, but the effect size is extremely
small. Raw Koopman sign accuracy was 51.97%, raw forecast/return correlation
was 0.0302, and the calibrated candidate's hard direction accuracy was
slightly lower than the baseline. This is not evidence that increasing its
weight will increase trading profit.

## Integration boundary

The result permits a frozen forward-shadow hypothesis only. It does not permit
Koopman to:

- generate an independent BUY or SELL;
- amplify a BUY score;
- change position sizing;
- alter the SELL path or risk gates;
- bypass the three execution locks;
- write the existing decision-probability artifact.

Any trading use requires a separate preregistration that includes economic
materiality, transaction costs, the existing decision population, and fresh
forward data. The present paper configuration and agent remain byte-identical.
