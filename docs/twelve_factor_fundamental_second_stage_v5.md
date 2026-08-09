# Point-in-time fundamental second-stage study V5

## Preregistered question

Can four economically distinct, point-in-time fundamental families improve next-session Top10 ranking after the frozen thirteen-price-factor model has selected a Top100 candidate pool?

This is a historical rejection study. It is research-only, shadow-only, cannot place orders, cannot alter trading configuration, and cannot promote itself.

## Frozen construction

1. The first stage is the previously selected thirteen-price-factor score: Alpha070 and Alpha131 receive 15% each; the other eleven weights are proportionally scaled to 70%.
2. At market close `t`, the first stage selects exactly 100 stocks without access to future outcomes.
3. Fundamental disclosures become available only on the first market session strictly after `max(noticeDate, updateDate)`. `reportDate` is used only for fiscal chronology and same-period comparison.
4. Each of the four existing mechanism families is an equal-rank composite of its four preregistered candidates. Every stock must have all four families; missing values fail closed.
5. The executable outcome is next buyable open to the following sellable open, with a maximum five-session execution delay and 30 bps round-trip cost.

## Preregistered ablations

- Price baseline: the original price score within the identical complete-support Top100.
- Fundamental only: equal rank across earnings innovation, growth acceleration, quality, and cash-flow quality.
- Fixed blend: 50% price rank and 50% equal-family fundamental rank.
- Regularized five-feature model: price rank plus the four family ranks, using fixed Ridge and L2 logistic specifications and frozen calibration windows.

The regularized model is the only preregistered candidate for the acceptance decision. The other policies are explanatory ablations, not a menu from which the best historical result may be selected.

## Isolation and decision rule

The base-fit, calibration, audit, validation, and shadow periods are frozen in the configuration. Audit is the only acceptance period; validation and shadow are reject-only. Comparisons use the same daily Top100 candidate pool and the same complete-fundamental support, so any apparent improvement cannot be attributed solely to removing stocks with missing fundamentals.

All preregistered accuracy, Top10 outcome, probability-spread, monotonicity, and coverage gates must pass. A historical pass would still remain unvalidated and require 60 fresh forward sessions after a separately frozen version. A failure rejects the V5 model without replacing the existing shadow artifact.

## Known limitations

- Historical windows have already been viewed in earlier studies.
- The source contains retrievable statement versions, not every historical restatement vintage.
- Historical industry membership is unavailable; five trailing-liquidity bins provide only approximate size neutralisation.
- Daily bars do not reproduce opening-auction queue priority.
- No result from this study is eligible for trading.
