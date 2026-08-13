# Twelve-factor forecast plus PIT fundamentals V1

> Research-only fixed-selection rejection study. No ranking or trading output was changed.

- run: `run_20260813_complete12_plus_pit_fundamentals_v1`
- data: `2019-01-02..2026-08-12`
- universe: `4710` PIT SH/SZ stocks
- causal statement events: `114897`
- baseline: 12 complete price-factor ranks
- candidate: the same 12 ranks plus 4 PIT fundamental-family ranks
- selection: held fixed to the current guarded twelve-factor score
- verdict: `reject_fundamental_increment_for_current_forecast`
- eligible for trading: `False`

## Forecast comparison

| period | scope | model | up AUC | up Brier | up LogLoss | up ECE | tail AUC | return MAE | p(up) spread |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| audit | overall | baseline | 0.5339008157595335 | 0.24965857 | 0.69246419 | 0.00765609 | 0.6649093525858756 | 0.02086096 | 0.0052405362778053 |
| audit | overall | candidate | 0.5339926925430823 | 0.24965834 | 0.69246373 | 0.00765615 | 0.6665064209053398 | 0.02086075 | 0.005240183202879304 |
| audit | fixedTop10 | baseline | 0.5020406923162617 | 0.25069420 | 0.69453574 | 0.04376459 | 0.6136004737465457 | 0.01671362 | 0.0007116704505116027 |
| audit | fixedTop10 | candidate | 0.501554572062264 | 0.25069494 | 0.69453721 | 0.04376324 | 0.6531780497433873 | 0.01672061 | 0.0007172865662859031 |
| validation | overall | baseline | 0.5263430972215868 | 0.24984790 | 0.69284291 | 0.00055124 | 0.7248070842412896 | 0.01917288 | 0.005352561431374457 |
| validation | overall | candidate | 0.5263590426171767 | 0.24984783 | 0.69284279 | 0.00055105 | 0.7264766203320678 | 0.01917313 | 0.005364556627287624 |
| validation | fixedTop10 | baseline | 0.4927327354807223 | 0.25081845 | 0.69478425 | 0.05121850 | 0.4669621273166801 | 0.01094480 | 0.0007678735345564204 |
| validation | fixedTop10 | candidate | 0.4940519765739385 | 0.25081732 | 0.69478200 | 0.05121550 | 0.49310591816635324 | 0.01093884 | 0.0007746519695431636 |
| shadow | overall | baseline | 0.5030714002514258 | 0.24935444 | 0.69185587 | 0.02419430 | 0.6393718876588712 | 0.02425020 | 0.005304459440176934 |
| shadow | overall | candidate | 0.5031570896373203 | 0.24935435 | 0.69185569 | 0.02419447 | 0.641592717599594 | 0.02424994 | 0.0053041197050286756 |
| shadow | fixedTop10 | baseline | 0.45288297409260536 | 0.25018758 | 0.69352241 | 0.01385894 | 0.610397946084724 | 0.01625588 | 0.0007490632355934214 |
| shadow | fixedTop10 | candidate | 0.45175805418201204 | 0.25018814 | 0.69352351 | 0.01385664 | 0.6211361737677528 | 0.01625542 | 0.0007495993399056493 |

## Interpretation

The same stocks, dates and current Top10 are used for both forecast heads. Any metric change therefore comes from the four fundamental inputs, not from skipping names or difficult sessions.
Historical audit, validation and shadow windows have already been viewed. They can reject this increment but cannot validate or promote it. A pass only permits a separately preregistered 60-session fresh-forward challenger.
PBO and DSR are not reported because this run evaluates one preregistered forecast challenger and does not change realised selections or returns. Multiple testing is controlled by a single fixed candidate and simultaneous validation-and-shadow gates.

Orders remain `[]`; dashboard and trading files remain unchanged.
