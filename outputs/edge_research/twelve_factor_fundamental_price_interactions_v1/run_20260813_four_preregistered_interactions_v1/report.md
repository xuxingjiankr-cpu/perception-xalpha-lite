# PIT fundamental by price-volume interaction study V1

> Research-only fixed-selection historical rejection study. Not trading.

- run: `run_20260813_four_preregistered_interactions_v1`
- data: `2019-01-02..2026-08-12`
- universe: `4710` PIT SH/SZ stocks
- selector: unchanged guarded twelve-factor Top10
- primary candidate: baseline plus all four preregistered interaction ranks
- verdict: `reject_fundamental_increment_for_current_forecast`
- eligible for trading: `False`

## Primary comparison

| period | scope | model | up AUC | up Brier | tail AUC | return MAE | p(up) spread |
|---|---|---|---:|---:|---:|---:|---:|
| audit | overall | baseline | 0.5349194008647296 | 0.24966058 | 0.6649235289917502 | 0.02086375 | 0.004892364832214512 |
| audit | overall | all_interactions | 0.5346972247216467 | 0.24965810 | 0.676108236759146 | 0.02086370 | 0.005260271563970299 |
| audit | fixedTop10 | baseline | 0.5021014573480115 | 0.25070509 | 0.6198677457560204 | 0.01671057 | 0.0006379946981255915 |
| audit | fixedTop10 | all_interactions | 0.498769508107068 | 0.25070105 | 0.654510461902882 | 0.01671014 | 0.0007664408035789165 |
| validation | overall | baseline | 0.5261257152595202 | 0.24985097 | 0.7248167556651994 | 0.01917504 | 0.005000096702607345 |
| validation | overall | all_interactions | 0.5260487070506241 | 0.24984888 | 0.7389222895511072 | 0.01917530 | 0.005300327770881609 |
| validation | fixedTop10 | baseline | 0.49306318122661463 | 0.25083136 | 0.46293311845286056 | 0.01094377 | 0.0006977481280415485 |
| validation | fixedTop10 | all_interactions | 0.5003533227590695 | 0.25082412 | 0.5472289372369952 | 0.01094018 | 0.0007559236303213071 |
| shadow | overall | baseline | 0.5031397077892883 | 0.24935437 | 0.6396710530475637 | 0.02424853 | 0.0049400932357804225 |
| shadow | overall | all_interactions | 0.5035567633954676 | 0.24935343 | 0.6539851282582789 | 0.02424838 | 0.005286173623136379 |
| shadow | fixedTop10 | baseline | 0.4504937420216278 | 0.25018971 | 0.607901868492369 | 0.01624848 | 0.0006761405994928338 |
| shadow | fixedTop10 | all_interactions | 0.4638680615673558 | 0.25018392 | 0.6323023004666137 | 0.01624546 | 0.000776074927518077 |

## Individual interaction ablations

Individual rows are diagnostics, not a menu from which a historical winner may be selected.

| period | interaction | overall up AUC | fixed Top10 up AUC | overall tail AUC | fixed Top10 return MAE |
|---|---|---:|---:|---:|---:|
| audit | earnings_volume_confirmation | 0.5349847242181062 | 0.5022432424220942 | 0.6653090707920208 | 0.01671015 |
| audit | growth_momentum_confirmation | 0.5347971506220508 | 0.5060916944329104 | 0.6651717503914106 | 0.01670987 |
| audit | quality_low_volatility | 0.5351030014228931 | 0.4966326044905358 | 0.6758174488141382 | 0.01671088 |
| audit | cash_quality_reversal | 0.534599164513945 | 0.5049979238614153 | 0.6653079026831191 | 0.01671000 |
| validation | earnings_volume_confirmation | 0.5261855425306838 | 0.49383845778428503 | 0.7251394618399789 | 0.01094390 |
| validation | growth_momentum_confirmation | 0.526257480462582 | 0.4930860582397918 | 0.7248896372191138 | 0.01094409 |
| validation | quality_low_volatility | 0.5260953745675959 | 0.4940341833414673 | 0.7392776301468358 | 0.01094338 |
| validation | cash_quality_reversal | 0.526018419557105 | 0.49629392386530014 | 0.7246188153181792 | 0.01094095 |
| shadow | earnings_volume_confirmation | 0.5032718087299395 | 0.4512273854415799 | 0.640025129996664 | 0.01624846 |
| shadow | growth_momentum_confirmation | 0.5035456039113587 | 0.4561158960965279 | 0.6399838572707817 | 0.01624638 |
| shadow | quality_low_volatility | 0.503199344930129 | 0.4504105957673666 | 0.6545753113375433 | 0.01624810 |
| shadow | cash_quality_reversal | 0.5029571588918192 | 0.4583143808783179 | 0.6388829075638369 | 0.01624877 |

All models use identical rows and the same selected Top10. A wider probability range is not evidence unless calibration and discrimination also improve in validation and shadow.
Historical windows are already viewed. Even a pass could only justify a separately preregistered 60-session fresh-forward challenger.
Orders remain `[]`; dashboard, ranking and trading files remain unchanged.
