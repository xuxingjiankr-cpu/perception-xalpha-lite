# PIT fundamental catalyst V5 preregistration

Status: `research-only / shadow-only / not trading`.

## Why this is the next useful change

The earlier fundamental miner had broad statement coverage, but it mostly fed slow
fundamental *levels* into the same grammar as price/volume factors. Its cleanest adjusted
PIT rerun generated 256 candidates and retained zero credible factors. The later frozen
twelve-factor price/volume book also failed external precision tests. Reweighting those
same inputs again is therefore a low-value research path.

This study tests a different information set: what changed in a newly disclosed filing
relative to the issuer's own previously available filing. That event is not present in
pre-disclosure OHLCV. The economic hypothesis is information diffusion after an earnings
improvement, with a small frozen price/volume component used only to rank entry timing.

## Fixed causal timeline

For every statement row:

1. Take `max(NOTICE_DATE, UPDATE_DATE)`.
2. Make the row available on the first market date strictly after that date.
3. Collapse multiple historical statements exposed on the same safe date to the latest
   report period. This prevents an IPO or endpoint history dump from becoming a series of
   fake events.
4. Skip a late row whose report period does not advance. The public endpoint does not
   contain the original/restated value pair needed to model that revision honestly.
5. Compare the new row only with the preceding causally available report.
6. Form the signal after the safe session closes, enter at the next buyable open, and
   attempt the exit after five sessions. A locked exit carries for at most five additional
   sessions.

The label is never written into the event feature table. Each train/validation/shadow
period drops its tail by the full holding plus exit-delay window.

## Frozen event score

Seven changes are compressed with fixed `tanh` scales:

- growth, 45%: revenue YoY acceleration and parent-profit YoY acceleration;
- quality, 30%: ROE, gross-margin and net-margin improvement;
- cash, 15%: operating-cash/net-profit improvement;
- safety, 10%: lower debt/assets.

An event must contain growth plus at least one other group. Only a strictly positive
composite enters the candidate pool. The event score is standardized inside three causal
trailing-liquidity buckets. This reduces a simple size bet, but it is explicitly not a
substitute for unavailable historical industry classifications.

## Four preregistered arms

All arms use exactly the same candidate pool, Top10 cap and daily selection count:

1. Primary: 75% catalyst-change rank + 25% frozen twelve-factor timing rank.
2. Ablation: catalyst-change rank only.
3. Counter: absolute fundamental level from the same filing.
4. Counter: frozen twelve-factor timing rank only.

No arm is selected after looking at train, validation or shadow. The four definitions add
four trials to the research burden. In addition, 200 deterministic cross-sectional
permutations retain the same candidates and number selected each day; a single placebo
draw is not treated as evidence.

## Fixed rejection rule

The primary must pass separately in both validation and shadow:

- at least 20 signal days and 98% resolved observations;
- daily net win rate above 50% after 30 bps round-trip cost;
- positive daily mean net return;
- higher daily mean net return than both counters;
- no higher severe-loss rate than the timing counter; and
- permutation empirical `p <= 0.10`.

These external dates were viewed by earlier studies, so even a full pass cannot authorize
trading. It can only justify a separately preregistered fresh-forward shadow study.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_perception_xalpha_pit_fundamental_catalyst_v5.py
```

Artifacts are written atomically under:

`outputs/edge_research/perception_xalpha_pit_fundamental_catalyst_v5/<run_id>/`

The result and every report contain `orders: []`; the script has no broker, production
decision, overlay, position, sizing, risk-gate or execution-lock path.

## First full historical result (2026-08-07)

The full adjusted PIT run covered 2019-01-02 through 2026-08-05, 4,706 stocks and
104,521 causal issuer-change events. It rejected the hypothesis.

The external numbers looked superficially attractive: the primary Top10 had a 63.89%
validation daily net win rate and 66.67% in shadow. They were not ranking alpha. A
same-pool random selection also benefited from those reporting-season windows; the
primary permutation p-values were 0.139 and 0.557. In shadow, the primary mean net return
(0.770%) was below the absolute fundamental-level counter (0.916%), while its severe-loss
rate (25.57%) was above the frozen timing counter (20.98%). Training was outright negative
after cost (42.80% daily net win, -0.287% mean net return).

The clean interpretation is that the *positive filing-event candidate pool* happened to
perform well in the two recent windows, but this particular catalyst score could not rank
the winners inside that pool. It is therefore not connected to stock selection or trading.
