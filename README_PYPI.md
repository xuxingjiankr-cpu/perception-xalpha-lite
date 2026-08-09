# Perception-XAlpha Lite

**Quantitative discovery is a multiple-comparisons problem disguised as an optimization
problem.** This finds fewer factors, on purpose.

A research framework that generates formulaic factors, backtests them on point-in-time data
with real costs, and then tries to prove its own findings wrong before believing them. It is
deliberately **not** a trading engine: no broker client, no order path, and CI asserts that
mechanically on every commit.

- **Repository, wiki and live record:** https://github.com/xuxingjiankr-cpu/perception-xalpha-lite
- **中文说明:** https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/docs/README_CN.md

```bash
pip install perception-xalpha-lite
```

## Audit a backtest for overfitting

Give it daily returns for the variants you tried, and say how many you actually tried —
including the ones you deleted.

```python
import pandas as pd
from xalpha_lite.discovery import pbo, deflated_sharpe_ratio
from xalpha_lite.evidence import white_reality_check

returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
sharpes = list(returns.mean() / returns.std(ddof=1))
best = (returns.mean() / returns.std(ddof=1)).idxmax()

print(pbo(returns))                                        # CSCV overfitting probability
print(deflated_sharpe_ratio(returns[best], sharpes, 250))  # against 250 declared trials
print(white_reality_check(returns))                        # family-wide null
```

On **24 variants of pure random noise**, the best has an annualised Sharpe of **1.11** — a
number most people would trade. At 24 trials, noise is expected to produce **1.18**. PBO comes
back 0.64, deflated Sharpe probability 0.46. The verdict is that selection is doing the work.

## Four biases, measured on a real equity panel

![Four measured biases](https://raw.githubusercontent.com/xuxingjiankr-cpu/perception-xalpha-lite/main/docs/measured-corrections.png)

| flaw | reports | survives | unit |
|---|--:|--:|---|
| factors chosen with hindsight | **+2.00** | −1.24 | bps/day, same panel and cost |
| limit-locked legs priced as fillable | **+6.05** | +0.38 | % forward return of those legs |
| universe filtered on whole history | **391** | 77 | eligible names, first year |
| overlapping labels scored as independent | **−5.79** | −2.25 | t-statistic on pure noise |

The first row is the one to sit with. Same data, same cost model, same construction — only the
rule for *choosing* factors differs, and the gap is about 3 bps/day, larger than most published
equity-factor results. A pipeline that cannot audit its own selection step cannot tell a
discovery from an artifact of choosing.

Reproducible on synthetic data with no signal in it, in ten seconds:

```bash
python examples/selection_artifact.py    # IR 4.53 manufactured from pure noise
```

## What is in the box

| module | what it does |
|---|---|
| `pit` | disclosure-aware point-in-time alignment; a value appears only after `max(notice_date, update_date)`, and rows without a disclosure date are rejected rather than imputed |
| `dsl` | allowlisted causal expression language — no `eval`, no subprocess, no network |
| `discovery` | bounded synthesis, neutral books, purged walk-forward, counterfactual and placebo controls, PBO and deflated Sharpe |
| `universe` | point-in-time membership, and limit-locked sessions inferred from the bars themselves |
| `book` | long-only top-N and dollar-neutral books sharing one cost engine |
| `forward` | frozen specifications: no overwrite, digest verified on load, one entry per session, scoring only fully elapsed windows |
| `evidence` | stationary bootstrap, White's Reality Check, Romano–Wolf step-down, BH/BY |
| `decision` | Top-K pairwise weighting, block replicas, independent probability calibration |

Command line: `xalpha-lite`, `xalpha-evidence`, `xalpha-forward`.

## Current status, stated plainly

Nothing has graduated. Candidates are generated and fully evaluated; none has cleared the
counterfactual, walk-forward and multiple-testing gates together. That is the gates working on
a price-and-volume factor library, not the engine failing to run — and unlike most backtests,
this one reports the exact count and the reason each candidate died.

Every number in the package and its examples comes from synthetic data it generates itself.
**It makes no profitability claim and never will.**

## Limitations

Examples are synthetic. Public financial endpoints may not preserve every restatement vintage.
A contemporary security master creates survivorship bias unless replaced by genuine
point-in-time membership. A zero-investment research portfolio is not executable in a long-only
cash market. Equal overlapping tranches approximate a holding horizon and model neither queue
priority nor market impact. Stationary-bootstrap inference assumes weak stationarity, and no
resampling procedure repairs contaminated data or an incomplete trial ledger.

MIT. Research and educational use only. No investment advice.
