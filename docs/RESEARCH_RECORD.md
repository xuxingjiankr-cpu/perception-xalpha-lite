# Existing empirical research record

[Tool overview](../README.md) · [Data provenance](DATA_PROVENANCE.md)

This material was moved from the project homepage to separate empirical observations from
synthetic teaching examples. It is preserved, not newly validated or recertified. Historical
comparisons and sampled forward observations are **not synthetic**, do not establish an
executable edge, and must be read with their original limitations. No new private research
results were added in this presentation update. The marked block remains machine-updated.

## The pick is published before the session it applies to

[![Rotation record](daily-rotation.svg)](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/#live)

A frozen specification (`dea0e608`) picks one name out of ~4,700 eligible and commits it
here, timestamped, **before that market opens**. Since 2026-08-22 the runner publishes
**weekly**, not nightly: the specification rotates every session, so the record samples
roughly one session in five and understates the turnover - and the cost - the specification
itself would incur. The file is append-only, and the
whole thing — fetch, select, score, redraw — runs on a GitHub runner from public data, so a
reader can rerun it and get the same name.

<!-- LIVE-RECORD:BEGIN -->
| the live record, as of 2026-09-02 | |
|---|--:|
| Trades scored | **8** |
| Compounded excess over the eligible universe | **-4.40%** |
| Mean gross per trade | -0.38% |
| Trades that rose | 5/8 |
| Next name, published in advance | `SZ_300804` |
| Verdict | `insufficient_forward_sample` |

Each row is one holding_days=1 trade. The runner publishes weekly while the
specification rotates every session, so these trades are sampled, not consecutive,
and compounding them is not a continuous equity curve. Below 60 scored
trades the verdict does not change, whatever the numbers do. This block is rewritten
by the weekly job, not by hand.
<!-- LIVE-RECORD:END -->

Two separate records run, and they answer different questions:

| record | book | held | where it runs |
|---|---|---|---|
| `dea0e608` — the chart above | 1 name | 1 session | GitHub Actions, published in advance |
| `b19bbc74` / `c0768449` | 10 names | 10 sessions | the author's machine, from the 456-candidate search |

That ordering is the entire claim. A record scored afterwards always invites the question of
whether the rule moved once the outcome was visible. One committed in advance cannot — it can
be shown to be wrong, but it cannot be edited.

**The dashed span on the left proves nothing, and is drawn that way deliberately.** Those
factors were chosen from a 456-candidate search on data that overlaps it, and on this panel the
selection step alone is worth ~3 bps/day. An in-sample curve that beats every index is what a
selected strategy always looks like. Only the solid span is evidence, the chart states how many
sessions it contains, and a handful of them settles nothing — the scorecard reports
`insufficient_forward_sample` below sixty and will keep doing so for months.

Two numbers people routinely conflate, over the backtested span:

| | cumulative |
|---|--:|
| Strategy, **raw** net of cost — comparable to an index | ≈ +60% |
| Eligible universe, equal weight | +24.7% |
| CSI300 | +20.3% |
| SSE50 (onshore A50 proxy) | +8.6% |
| Strategy, **excess over universe** — the research metric | +18.5% |

Plotting an excess against a raw index is the oldest trick in the genre, so the chart draws
them as separate lines and says which question each answers. A GitHub Action re-fetches the
index series daily and refuses to redraw if any committed benchmark return has drifted by more
than 5 bps.

## It runs the record it describes

This is not a methods library sitting next to the research. The frozen forward records for the
A-share project it was built for run on this package — `build_panel`, `point_in_time_eligibility`,
`long_only_book`, `score_log` — the one-name rotation on a GitHub runner, the ten-name record on
the author's machine. The one-name runner has appended weekly since 2026-08-22.

Pointing it at real work is what found the gaps. Four, in one sitting:

| found by using it | what it was |
|---|---|
| `build_panel` defaulted `limit_up`/`limit_down` to `False` | plain OHLCV silently backtested fills on **limit-locked boards**, worth +6.05% a leg against +0.38% for executable ones |
| no point-in-time universe rule | membership had to be hand-rolled, which is where survivorship gets in |
| only a dollar-neutral book existed | the construction nearly everyone actually trades — long-only top-N — could not be expressed |
| the DSL had no time trend | an entire published family (`ts_corr(close, t, w)²`) was inexpressible |

All four are now in the library. The migration was checked rather than assumed: every factor
recomputed both ways across 400 sessions × 5,169 symbols agrees to a maximum cross-sectional
rank difference of **0.00e+00**, and the first book the ported spec produced shares nine of ten
names with the one the original pipeline produced.

The record's value is entirely in what it refuses — `freeze_spec` will not overwrite,
`load_spec` rejects a file edited after freezing, a session already logged appends nothing, and
`score_log` counts only fully elapsed holding windows. CI asserts all four still fire, because
a forward record whose guards stop firing is just a backtest.

```bash
python examples/run_forward_record_synthetic.py
```

That example returns **+0.32% net on prices with no drift** and reports it as
`insufficient_forward_sample` at n=6. Which is the point: the number is noise, and the
framework says so instead of printing it as a result.

## What survives so far

Every factor in four published libraries (Kakushadze 101, GTJA 191, Qlib 158, academic) over a
point-in-time A-share panel, ranked **only on the training window** by the mean excess of a
ten-name book, then reported on the untouched test window. 900 sessions, split at 2024-12-31,
rebalanced every session, ten sessions held, entry and exit at the open after the signal,
limit-locked legs dropped rather than priced. Excess is against the equal-weight eligible
universe over the same bars, which returned **+0.85%** per hold on the test window.

| # | factor | TRAIN excess | TEST excess | >10% odds | worst hold | hit |
|---|---|--:|--:|--:|--:|--:|
| 1 | `academic/hml` | +2.20% | −0.04% | 1.01× | −17.5% | 49.2% |
| 2 | `qlib158/imin60` | +0.81% | −0.73% | 0.53× | −11.6% | 36.7% |
| 3 | `gtja191/alpha_144` | +0.73% | **+0.17%** | 1.16× | −7.5% | 46.5% |
| 4 | `academic/cma` | +0.67% | **+0.26%** | 0.72× | −9.4% | 48.9% |
| 5 | `qlib158/vsumn20` | +0.66% | **+0.40%** | 0.86× | −6.4% | 50.5% |

Costs are excluded, deliberately and visibly. On an overlapping series three defensible
turnover conventions give three different answers; a gross figure anyone can recompute is worth
more than a net one nobody can. At 30 bps a round trip, a book that fully rotates each hold
gives back 0.30% of the numbers above. The full ranking of all 456 is in
[`docs/data/library_ranking.json`](data/library_ranking.json).

### What came through both gates

The table above ranks on training data alone, which is the honest experiment. A fair question
is what happened to those names afterwards. Of the **thirty** highest-ranked on the training
window, **two** cleared both out-of-sample tests — a positive excess *and* a better-than-even
chance of a large gain:

| training rank | factor | what it measures | TEST excess | >10% odds | worst hold | hit |
|--:|---|---|--:|--:|--:|--:|
| 3 | `gtja191/alpha_144` | mean \|return\| per unit turnover over 20 days, **counted only on down days** — downside price impact | **+0.17%** | **1.16×** | −7.5% | 46.5% |
| 30 | `academic/illiq` | Amihud (2002): mean \|return\| per unit turnover over 21 days — price impact, all days | **+0.14%** | **1.18×** | −7.8% | 47.9% |

**Two out of thirty is the yield**, and it is stated that way rather than as a top five, because
there is no third. Filling the row count would have meant ranking all 456 by what happened on
the test window — which is the hindsight bias this project [measured at +2.00 against −1.24
bps/day](#why-the-gates-are-this-strict). A table assembled that way tells you nothing you could
have acted on.

**The two survivors are the same idea.** Both are mean absolute return per unit of turnover:
Amihud's illiquidity, and the same quantity restricted to down days. That coherence is worth
something — the survivors are not two unrelated flukes but one economic mechanism, the
illiquidity premium, appearing twice. It also means they are **not two independent pieces of
evidence**, and a trial ledger that counted them as two would be overstating its own breadth in
exactly the way the duplicate pair above does.

Across all 456, 79 factors (17.3%) clear both conditions against roughly 10.7% expected if the
two were independent and unrelated to skill. Real enrichment — and selecting those 79 by their
test outcome would still be hindsight, which is why the count appears here and the names do not.

### Stated as a portfolio rather than as an excess

Excess over the eligible universe is the research metric because it strips out the market. It is
also not what a holder experiences, and the universe is not something anyone can actually buy —
equal-weighting four thousand A-shares is not a portfolio. So, in raw terms, on the test window:

| | per ten-session hold | probability of rising |
|---|--:|--:|
| the training-window top ten, equal weight | **+0.61%** | **61.4%** |
| eligible universe, equal weight | +0.85% | 62.2% |

The book makes money and rises three holds in five. It also trails the universe it was drawn
from on both counts, which is the whole finding: **the return is real and the skill is not
demonstrated.** Reporting only the first line would be the same move as plotting an excess
against a raw index, run in the opposite direction.

### What pinning the data source turned out to be worth

A provider outage broke the 17:00 run on 2026-08-12 and again on 2026-08-13. Adding a fallback
source is the obvious fix, and shipping it without checking would have been the same mistake
that retired the previous specification. So
the same frozen rule was run over panels built from two different providers of backward-adjusted
A-share bars — baostock, which the specification names, and TDX as the candidate fallback — and
compared on the single thing that matters: **which name each one picks.**

| | |
|---|--:|
| Sessions compared | 140 |
| **Sessions where both picked the same name** | **57.9%** |
| Where they differed, the other provider's pick ranked (in the primary) | **median 4th** of ~4,700 |

The fallback was not adopted. **A source that changes the answer on 42% of sessions is not a
fallback, it is a different rule.** [The data is published](data/source_reconciliation.json).

Neither day became a gap in the end: the second scheduled attempt, at 21:00 Asia/Shanghai,
succeeded both times and published each pick before the session it applied to. That is what the
backstop is for, and it is a better answer than a second data source — a later attempt runs the
same rule against the same source, while a fallback runs a different rule and hopes nobody
checks.

**The disagreements are near-ties, not divergence.** A median rank of 4 means the two providers
are usually choosing between candidates the composite cannot separate. Where the leader is
clear they agree without exception: over the eight most recent sessions both pick the same name,
which leads the runner-up 0.945 to 0.872.

Two things follow, and the second is more important than the first.

**Naming the provider in the specification was necessary, not pedantic.** A reader who follows
the spec uses the same source and gets the same names, so the record reproduces exactly — which
is what `dea0e608` was frozen to guarantee after its predecessor failed to. Had the source been left
open, 42% of published picks would be unreproducible by an equally careful reader.

**A one-name book is the most fragile object this rule can produce.** On close to half of all
sessions the winner is decided in the fourth decimal place, where the difference between two
honest vendors of the same adjusted prices is enough to change it. That is a property of holding
one name, not of the data: with a median disagreement rank of 4, a ten-name book contains the
other provider's pick almost every time. The ten-name record is the more meaningful of the two,
and the one-name chart on this page should be read knowing what decides it.

### Selection works, and it is not enough

The interesting number is not in the table. Across all 456 factors, training-window excess
predicts test-window excess with a Spearman **ρ = +0.48** (p < 0.001) — training-side ranking
carries real information, and anyone claiming this is all noise is wrong.

Then look at what the information buys:

| | mean TEST excess per hold | share beating the universe |
|---|--:|--:|
| all 456 factors | −0.71% | 21.3% |
| the training-window top 10 | **−0.24%** | 30.0% |

Choosing cleanly on the training window is worth **+0.47 percentage points per hold** against
picking at random. It is also still **negative**. The best ten of 456, selected without a
glance at the test window, went on to underperform the universe they were drawn from.

That is the honest shape of the result, and it is neither of the two stories usually told. The
signal is real. It is smaller than what decay and concentration take away, before a single
basis point of cost is charged.

Two more things in the table are worth more than the ranking:

- **Row 1 is the whole problem in one line.** `hml` leads the training window by a mile at
  +2.20% and lands at −0.04% out of sample. The fifth-placed factor, at a third of its training
  excess, is the best of the five on test. Training rank and test rank are correlated across the
  full 456 and nearly unrelated at the top, which is exactly where everyone selects.
- **The libraries contain duplicates, and counting them twice inflates the trial ledger.**
  `gtja191/alpha_120` and `alpha101/alpha_042` are character-for-character identical formulas
  published under different names, and this run reproduces them to the digit — TRAIN +0.358%,
  TEST −0.338%, worst hold −9.5%, both. A deflated-Sharpe denominator should count distinct
  *behaviours*, not files.

### The table this replaces could not be reproduced

An earlier version of this section reported a different top five, with different numbers, and
nothing on disk could regenerate it. Rebuilding it from the description recovered the universe
benchmark exactly (+0.820% against the published +0.82%, which pins the panel, window and
split) and matched no individual factor under any of three rebalancing and cost conventions —
one factor came out with the opposite sign.

Continuing to vary the convention until the numbers agreed would have been fitting the method
to the answer, which is the failure this repository exists to name. So the table was replaced
by one a committed script regenerates.

The four factors under forward observation were chosen by that unreproducible ranking. Under
this one they rank **4th, 30th, 57th and 234th of 456** — `academic/cma`, `academic/illiq`,
`qlib158/rsqr60` and `qlib158/rsqr30`, the last below the median. That does not weaken the
[forward record](#the-pick-is-published-before-the-session-it-applies-to): a rule frozen before
the data existed is tested by what happens next, not by how it was picked. It does mean the
story about *why* those four were chosen cannot be checked, and it is a third independent
demonstration that this ranking is not stable across defensible choices.

## Why the gates are this strict

Four biases, each measured on a real equity panel while building this pipeline, and each one
large enough to invent a strategy on its own:

![Four measured biases. Factors chosen with hindsight report +2.00 bps/day against -1.24 when chosen on trailing data. Limit-locked legs priced as fillable carry +6.05% forward return against +0.38% for tradeable ones. A universe filtered on whole history admits 391 names in the first year against 77. Overlapping labels scored as independent give a t-statistic of -5.79 on pure noise against -2.25.](measured-corrections.png)

| flaw | what it reports | what survives | unit |
|---|--:|--:|---|
| factors chosen with hindsight | **+2.00** | −1.24 | bps/day, same panel and cost |
| limit-locked legs priced as fillable | **+6.05** | +0.38 | % forward return of those legs |
| universe filtered on whole history | **391** | 77 | eligible names, first year |
| overlapping labels scored as independent | **−5.79** | −2.25 | t-statistic on pure noise |

The first row is the one worth sitting with. Same data, same cost model, same construction —
only the rule for choosing factors differs, and the gap is about 3 bps/day. That is larger
than most published equity-factor results, which means a pipeline that cannot audit its own
selection step cannot tell a discovery from an artifact of choosing.

The mechanism is reproducible on synthetic data with no signal in it at all, in ten seconds:

```bash
python examples/selection_artifact.py          # IR 4.53 manufactured from pure noise
python examples/make_corrections_figure.py     # regenerates the chart above
```
