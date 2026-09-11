# Statement-recency conditioning: one historical ablation

Research-only. This preregistration does not change any trading, dashboard, old
twelve-factor ranking, forward ledger or frozen model. No automatic promotion.

## Why this experiment

The completed Sina fundamental comparison did not improve Top10 payoff or hit
rate. A specific untested omission is statement availability age: a recent
disclosure and a several-month-old disclosure have the same carried family
values. This experiment tests that omission, not another broad weight search.

Limited attention is an economic motivation, not proof of Chinese-stock alpha.
[Hirshleifer, Lim and Teoh (2009)](https://bpb-us-e2.wpmucdn.com/sites.uci.edu/dist/c/362/files/2020/07/Driven-to-Distraction-Extraneous-Events-and-Underreaction-to-Earnings-News.pdf)
study competing announcements and earnings-news underreaction. We do NOT have
historical analyst consensus or attention-load records, so this is not a
replication or a consensus-surprise factor. Counterevidence matters too:
[Christensen, Timmermann and Veliyev (2026)](https://arxiv.org/abs/2601.08962)
report post-announcement trading returns consistent with efficient price
formation in their post-2016 sample. The two decay constants below are our
frozen hypothesis, not parameters validated by either paper.

## Frozen contract

- Config: `configs/research/sina_disclosure_timing_v1.json`.
- Reference: `run_20260912_frozen_fundamental_training_v1_r2`. Source/config/code,
  reference models, selected stocks and outcomes must reproduce. No overwrites.
- Source history 2019-01-02 to 2026-09-08, qualified PIT SH/SZ subset, same
  2024-01-02-start evaluation and maximum-maturity cutoff. Missing status days
  remain missing; BJ cannot be certified from the existing inputs.
- Existing four financial mechanism families, four fixed contexts and four
  interactions unchanged. These are NOT the legacy twelve price-volume factors.
- Age counts sessions after the existing causal function's accepted advancing
  statement event: first session strictly after `max(noticeDate, updateDate)`.
  `reportDate` orders fiscal reports; it cannot make a statement observable.
- Decays `2**(-age/5)` and `2**(-age/20)`, plus each current family rank times each
  decay: exactly ten new features, 22 total. No windows/half-life search.
- The baseline's family values expire after 130 sessions, unchanged. Knowing
  the date of a past accepted disclosure does not expire; the lagged counter
  can therefore have age >130 without carrying stale financial values. Before
  any known disclosure, age is unknown and is an error, not zero/no news.
  Adding these features cannot drop any original stock-date support.
- Counter uses the same base twelve features and the recency values known 20
  sessions earlier. It has identical capacity; it is not a permutation p-value.
- Same 252 training sessions, 63 calibration sessions, seven-session purges,
  21-session refits, seeds, daily subsampling, ridge/logistic regularization and
  train-only clipping/standardization as the original config. Only two new
  model arms are fitted. Numeric reference models are loaded by class allowlist.
- Exactly ten choices per supported signal day BEFORE looking at outcomes.
  Buy t+1 open, exit no earlier than t+2 open, maximum five extra exit sessions,
  round-trip cost 30bp. No rank-11 substitution for unfilled/unresolved picks.

## Four arms and reporting

1. Equal fundamental families: untouched ranking; descriptive probabilities
   from the frozen fundamental model, not separately trained probabilities.
2. Pooled payoff: untouched frozen interaction model.
3. Disclosure timing: the one primary candidate.
4. Lagged timing counter: the one capacity-matched control.

Candidate must be compared with BOTH equal families and pooled payoff, not just
the weaker trained control. Unresolved selections and absent requested days
continue to fail strict acceptance. A numerical historical pass still cannot
promote. Current historical windows are reused, not a new clean holdout.

Outputs in `outputs/edge_research/sina_disclosure_timing_v1/<run_id>/`:
manifest/config/code/environment and reference hashes; input hashes; input and
disclosure audits; new numeric fold models; historical Top10 selections; daily
same-support Brier/LogLoss/AUC/ECE for up and tail on identical resolved eligible
stock-date labels; all-selected bounds; all-four joint-complete-day payoff table;
strict paired block-bootstrap gates; hindsight versus trailing-63-day net payoff
on identical matured dates. No millions-of-rows-as-independent significance.

Multiplicity is recorded conservatively: prior global lower bound 465 plus four
reported policy comparisons (two reused controls, two newly fitted arms); each
financial family gains four comparisons. DSR remains null because the complete
historical search ledger is unavailable. No fabricated certificate. A negative
result ends this version; changing parameters requires a new preregistration.

## Verification and use

T173 exercises disclosure timing and future-prefix invariance, unknown-age
rejection, numeric model round-trip, frozen baseline reproduction, actual
fitting/calibration/reporting, shared probability support, unresolved Top10
accounting and output isolation. Full replay invariants must also pass.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
py -3.13 scripts/research_sina_disclosure_timing_v1.py --run-id <unique_run_id>
```

Do not use this historical tool for online inference, current-stock trade
recommendations, or tuning the frozen forward programs.
