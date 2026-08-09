<h1>
  <img src="docs/logo.svg" alt="" height="76" align="left" />
  Perception-XAlpha Lite
</h1>

[![PyPI](https://img.shields.io/pypi/v/perception-xalpha-lite?color=0073B7&logo=pypi&logoColor=white)](https://pypi.org/project/perception-xalpha-lite/)
[![ci](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research Status](https://img.shields.io/badge/status-research--only-7C3AED)](#research-integrity-contract)
[![Point-in-Time](https://img.shields.io/badge/data-point--in--time-0891B2)](#point-in-time-data-contract)
[![Safe DSL](https://img.shields.io/badge/factor%20language-audited%20DSL-059669)](src/xalpha_lite/dsl.py)
[![License](https://img.shields.io/badge/license-MIT-111827)](LICENSE)

**English** | [简体中文](docs/README_CN.md)

[Audit your backtest](#audit-your-own-backtest) &middot; [Live record](#the-pick-is-published-before-the-session-it-applies-to) &middot; [Measured biases](#why-the-gates-are-this-strict) &middot; [What survives](#what-survives-so-far) &middot; [Quick start](#quick-start) &middot; [Wiki](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/wiki)


> **Quantitative discovery is a multiple-comparisons problem disguised as an optimization
> problem.** Finds fewer factors, on purpose.

A research framework that generates formulaic factors, backtests them on point-in-time data
with real costs, and then tries to prove its own findings wrong before believing them.

Perception-XAlpha Lite separates **hypothesis generation** from **evidence
acceptance**. It can synthesize bounded symbolic factors, but every candidate must survive
chronological isolation, counterfactual controls, permutation tests, and multiple-testing
corrections. Historical evidence can produce a forward-shadow hypothesis; it can never
produce an order. It is deliberately **not** a trading engine.

## Research notes

- **[Wiki](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/wiki)** — the measurements behind each design decision, including the rejected hypotheses
- **[Research roadmap](https://github.com/users/xuxingjiankr-cpu/projects/1)** — what is being tested, what was rejected, what is waiting on forward data
- **[Site](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/)** — the same material, rendered

## Audit your own backtest

```bash
pip install perception-xalpha-lite
```

Fork this, drop your daily returns into `audit/returns.csv`, say how many variants you actually
tried in `audit/audit.json`, and push. The audit runs **in your fork** and writes the verdict to
the run summary. Your returns never leave your repository — this one never sees them.

```bash
python tools/audit_returns.py
```

It asks three questions that are not the same question:

| | |
|---|---|
| **Was the winner picked by luck?** | CSCV splits the sample many ways, takes the best variant in-sample, and checks where it lands out of sample. Above 0.5 and selection is doing the work. |
| **Is the Sharpe big enough to survive the search that found it?** | The deflated Sharpe discounts what you observed by the best you'd expect from that many trials on noise. |
| **Does anything beat zero once the whole family is counted?** | White's Reality Check, with a stationary bootstrap that keeps the serial dependence. |

The example that ships with it is **24 variants of pure random noise**. The best has a Sharpe of
**1.11 annualised** — a number most people would trade. At 24 trials, noise is expected to
produce **1.18**. Verdict: `SELECTION IS DOING THE WORK`.

The trial count is the number of variants you *tried*, including every one you deleted. Not the
number in the file. Understating it is the most common way a backtest passes a test it should
fail, and the tool says so when the two numbers match.

## The pick is published before the session it applies to

[![Daily rotation record](docs/daily-rotation.svg)](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/#live)

Each evening a frozen specification (`dea0e608`) picks one name out of ~4,700 eligible and
commits it here, timestamped, **before that market opens**. The file is append-only, and the
whole thing — fetch, select, score, redraw — runs on a GitHub runner from public data, so a
reader can rerun it and get the same name.

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
selected strategy always looks like. Only the solid span is evidence, and today it is empty.

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
the author's machine, both appended daily.

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
[`docs/data/library_ranking.json`](docs/data/library_ranking.json).

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

## What it actually does

It is a full research loop, not only a critic:

| stage | what runs |
|---|---|
| **Generate** | bounded symbolic factors from an audited DSL — no `eval`, no subprocess, no network |
| **Backtest** | point-in-time panel, disclosure-date alignment, industry- and size-neutral book, explicit turnover and cost |
| **Control** | counterfactual and placebo variants of every candidate, chronological isolation, purged walk-forward |
| **Discount** | PBO by combinatorially symmetric cross-validation, deflated Sharpe against the full trial count |
| **Record** | immutable audit trail — commit hash, config hash, candidate lineage, every trial counted |

The backtester is the point as much as the search is. A candidate is priced the same way at
every stage, so a factor cannot be screened as one portfolio and validated as another.

**Current status, stated plainly: nothing has graduated.** Candidates are generated and
fully evaluated; none has yet cleared the counterfactual, walk-forward and multiple-testing
gates together. That is the gates doing their job on a price-and-volume factor library, not
the engine failing to run — and unlike most backtests, this one can tell you the exact count
and the reason each candidate died. A framework that reports a hit rate of zero is being
more informative than one that never counted.

This public repository contains **methods only** — no proprietary data, empirical factor
weights, securities, performance tables, or private research conclusions. **It makes no
profitability claim, and it never will.** Every number above and in `examples/` comes from
synthetic data generated by the repository itself.

## Why the gates are this strict

Four biases, each measured on a real equity panel while building this pipeline, and each one
large enough to invent a strategy on its own:

![Four measured biases. Factors chosen with hindsight report +2.00 bps/day against -1.24 when chosen on trailing data. Limit-locked legs priced as fillable carry +6.05% forward return against +0.38% for tradeable ones. A universe filtered on whole history admits 391 names in the first year against 77. Overlapping labels scored as independent give a t-statistic of -5.79 on pure noise against -2.25.](docs/measured-corrections.png)

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


## Why this architecture exists

Quantitative discovery is a multiple-comparisons problem disguised as an optimization
problem. Three ideas anchor the design:

1. **Backtest selection requires an explicit false-discovery model.** Bailey et al. propose
   Combinatorially Symmetric Cross-Validation to estimate the Probability of Backtest
   Overfitting (PBO), because a conventional holdout can be unreliable after extensive
   strategy selection ([paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)).
2. **A Sharpe ratio is not evidence without a trial ledger.** The Deflated Sharpe Ratio
   corrects for selection bias, non-normal returns, and the number of attempted variants
   ([paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).
3. **Prediction quality and decision quality are different objectives.** Decision-focused
   learning evaluates the downstream decision induced by a prediction rather than relying
   only on point-prediction error ([PMLR](https://proceedings.mlr.press/v119/elmachtoub20a.html)).
4. **The winning candidate inherits the full search history.** The Evidence Lab combines
   dependence-preserving resampling, White's Reality Check, and Romano–Wolf step-down
   inference so a searched winner is never tested as if it had been specified in advance.

The framework therefore optimizes only inside a preregistered research envelope and treats
validation/shadow windows as reject-only evidence.

## System architecture

```mermaid
flowchart LR
    subgraph DATA["Causal Data Plane"]
        A["OHLCV + disclosures"] --> B["Point-in-time alignment"]
        B --> C["Tradability + neutralization controls"]
    end
    subgraph DISCOVERY["Autonomous Discovery Plane"]
        C --> D["Audited factor DSL"]
        D --> E["Economic seeds"]
        E --> F["Bounded mutation + crossover"]
        F --> G["Immutable birth certificates"]
    end
    subgraph VALIDATION["Falsification Plane"]
        G --> H["Purged walk-forward"]
        H --> I["Primary / Counter / Placebo"]
        I --> J["PBO + Deflated Sharpe"]
        J --> Q["Reality Check + step-down max-t"]
    end
    subgraph DECISION["Optional Decision Research"]
        G --> K["Top-K pairwise objective"]
        K --> L["Block-replica stability"]
        L --> M["Independent probability calibration"]
    end
    J --> N["Sealed research artifact"]
    M --> N
    N -. "never automatic" .-> O["Fresh forward shadow"]
```

## Scientific method map

The default discovery pipeline and the optional primitives library are intentionally
distinguished. Availability in the library means **testable**, not **validated alpha**.

| Research layer | Implementation | Scientific basis | Enforced boundary |
|---|---|---|---|
| Disclosure timing | `align_point_in_time_fundamentals()` | Information must not exist before publication | Missing disclosure dates fail closed |
| Symbolic search | audited JSON DSL + bounded generator | Interpretable formulaic alpha search; related to AlphaForge | No `eval`, subprocess, network, or arbitrary Python |
| Neutral portfolios | industry/size residualization | Cross-sectional factor research | Zero-investment research proxy; not an executable short book |
| Regime inference | causal two-state HMM | Hamilton regime switching | Forward filtering only; full-path Viterbi is not online-safe |
| Transition risk | EWS + BOCPD | Critical slowing and Bayesian run-length inference | Risk diagnostic, not directional alpha |
| Local dynamics | DMD/Koopman residual | Linear operator approximation of nonlinear dynamics | Past-window residual; not a universal price predictor |
| Bubble feasibility | simplified LPPLS | Stable reduced-parameter calibration | Fixed grid and stability distribution; no “exact crash date” claim |
| View shrinkage | Black–Litterman posterior | Equilibrium prior plus uncertain views | Shrinkage cannot manufacture information |
| Offline outcomes | Triple Barrier | Profit/loss/time labeling | Future-dependent labels are forbidden as features |
| Finite-sample risk | Hoeffding lower bound | Distribution-free concentration | Dependence and effective sample size still require audit |
| Validation | purge + walk-forward + controls | Temporal isolation and falsification | Validation/shadow never select parents or directions |
| Search correction | CSCV/PBO + DSR | Backtest-overfitting and selection-bias control | Every attempted candidate counts, including failures |
| Family evidence | stationary bootstrap + Reality Check + Romano–Wolf + BH/BY | Dependence-aware data-snooping and multiple-testing inference | A historical rejection is not a deployment decision |
| Decision ranking | bounded Top-K pairwise loss | Decision-focused learning | Fit block only; cannot create absent signal |
| Forecast reliability | Brier, LogLoss, AUC, ECE | Strictly proper scoring-rule framework | Calibration block is independent and frozen |

See the full [literature and implementation map](docs/PAPERS.md).

## Mathematical core

### 1. Industry- and size-neutral factor book

For a cross-sectional factor rank vector `r_t` and control matrix `X_t`, the research
portfolio is the normalized residual of the cross-sectional projection:

```math
\tilde r_t = (I - X_t(X_t^\top X_t)^{-1}X_t^\top)r_t,
\qquad
w_t = \frac{\tilde r_t}{\lVert \tilde r_t \rVert_1}.
```

This removes the constant, point-in-time log market capitalization, and industry dummy
exposures before evaluating the factor. It does not imply that the short leg is executable.

### 2. Decision-focused Top-K weighting

For each fit date, positive names are compared with names immediately below the Top-K
boundary. The bounded simplex solves:

```math
\min_{w \in \Delta_{[l,u]}}
\frac{1}{T}\sum_t\frac{1}{|P_t|}
\sum_{(i,j)\in P_t}\log\!\left(1+e^{-w^\top(x_{ti}-x_{tj})}\right)
+ \lambda\lVert w-w_0\rVert_2^2.
```

Each date receives equal mass. Lower/upper bounds and the prior `w_0` are preregistered;
complete chronological block replicas expose weight instability.

### 3. Independent probability calibration

Composite scores and loss events are calibrated only after weight fitting. Reliability is
reported with Brier score, LogLoss, ROC AUC, expected calibration error, and probability
bucket hit rates. Proper scoring rules are used because they reward honest probabilistic
forecasts rather than arbitrary confidence ([Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437)).

## Research lifecycle

```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> StaticValidated: DSL + causality + safety checks
    StaticValidated --> HistoricalValidated: purged WF + controls
    HistoricalValidated --> ForwardShadow: preregistered survivor
    Draft --> Rejected
    StaticValidated --> Rejected
    HistoricalValidated --> Rejected
    ForwardShadow --> Retired
    ForwardShadow --> ResearchCandidate: independent forward evidence
    ResearchCandidate --> [*]
```

`ResearchCandidate` is still not an execution state. This package defines no state that can
submit a trade.

## Point-in-time data contract

### Prices

```text
date,symbol,open,high,low,close,volume,amount,industry,market_cap,
is_st,is_suspended,is_delisted,limit_up,limit_down
```

Signals are formed at close `t` and evaluated from the next session's open. Industry and
market capitalization are point-in-time controls. Feasibility flags describe the execution
session, not information observed later in that session.

### Fundamentals

```text
symbol,report_date,notice_date,update_date,eps_ytd,bps,roe,roic,
gross_margin,cash_to_profit,revenue_yoy,profit_yoy,debt_ratio,
current_ratio,quick_ratio
```

`notice_date` is mandatory. A statement becomes visible on the first market session strictly
after `max(notice_date, update_date)`. Missing disclosure timestamps are rejected rather than
imputed.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -e .
python examples/selection_artifact.py          # why this framework exists, in ~10s
python examples/run_synthetic.py
python examples/run_decision_tools_synthetic.py
python examples/run_evidence_lab_synthetic.py
```

Run the factor-discovery CLI on local data:

```bash
xalpha-lite \
  --prices data/prices.csv \
  --fundamentals data/fundamentals.csv \
  --config configs/example.json \
  --output outputs/result.json
```

Stress-test a searched candidate family against a frozen benchmark:

```bash
xalpha-evidence \
  --performance candidate_performance.csv \
  --benchmark benchmark \
  --date-column date \
  --output outputs/evidence.json
```

See the [Dependence-aware Evidence Lab](docs/EVIDENCE_LAB.md) for its input contract,
equations, preregistration requirements, and paper-to-code boundaries.

## Repository map

```text
src/xalpha_lite/
├── pit.py                 # disclosure-aware point-in-time alignment
├── dsl.py                 # allowlisted causal expression language
├── discovery.py           # generation, neutral books, controls, PBO/DSR
├── universe.py            # point-in-time membership, sealed-bar limit detection
├── book.py                # long-only top-N book on the shared cost engine
├── forward.py             # frozen specifications, tamper-evident forward records
├── decision.py            # Top-K weights, block replicas, calibration
├── evidence.py            # stationary bootstrap, Reality Check, step-down inference
├── evidence_cli.py        # searched-family evidence command line interface
├── forward_cli.py         # freeze / log / score command line interface
├── paper_mechanisms.py    # auditable research primitives
└── cli.py                 # research-only command line interface

docs/
├── PAPERS.md              # literature-to-code traceability
├── EVIDENCE_LAB.md        # dependence-aware family-level inference
└── DECISION_TOOLKIT.md    # decision-research API and constraints
```

## Reproducibility contract

Every discovery run records:

- source commit;
- configuration hash;
- price and fundamental data hashes;
- Python, NumPy, pandas, and platform versions;
- immutable factor IDs, parent IDs, operators, and expression hashes;
- the complete attempted-candidate count, including rejected candidates;
- a SHA-256 commitment for undisclosed shadow metrics.

The output contract always includes:

```json
{
  "status": "diagnostic_only_research_only_not_trading",
  "orders": [],
  "automatic_trading_changes": []
}
```

No empirical research artifact is committed to this repository.

## Research integrity contract

- No broker client, account credentials, order submission, strategy overlay, or production
  BUY/SELL gate exists in this package.
- Validation and shadow outcomes may reject a model but may not refit it.
- Future-dependent labels remain in offline label tables and never enter features.
- Candidate direction and parent selection use training data only.
- Failed candidates remain part of the multiple-testing trial count.
- A statistically empty result is valid; the framework never forces a survivor.
- Historical evidence alone cannot authorize deployment.

## Limitations

- The examples are synthetic and make no profitability claim.
- Public financial endpoints may not preserve every historical restatement vintage.
- A contemporary security master creates survivorship bias unless replaced by a genuine
  point-in-time membership history.
- A zero-investment research portfolio is not directly executable in a long-only cash market.
- Equal overlapping tranches approximate a holding horizon; they do not model queue priority,
  borrow availability, market impact, or venue-specific capacity.
- HMM, EWS, BOCPD, DMD, LPPLS, Black–Litterman, and Triple Barrier are minimal auditable
  primitives, not production solvers and not claims of predictive power.
- Stationary-bootstrap inference relies on a defensible weak-stationarity approximation;
  no resampling procedure repairs contaminated data or an incomplete trial ledger.

## License

MIT. Research and educational use only. No investment advice.

See [CHANGELOG.md](CHANGELOG.md) for method-only releases.
