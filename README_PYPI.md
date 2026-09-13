# Perception-XAlpha Lite

**Quantitative discovery is a multiple-comparisons problem disguised as an optimization
problem.** This finds fewer factors, on purpose.

A research framework that generates formulaic factors, backtests them on point-in-time data
with real costs, and then tries to prove its own findings wrong before believing them. It is
deliberately **not** a trading engine: no broker client, no order path, and CI asserts that
mechanically on every commit.

- **Repository and documentation:** https://github.com/xuxingjiankr-cpu/perception-xalpha-lite
- **中文说明:** https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/docs/README_CN.md
- **Colab quickstart:** https://colab.research.google.com/github/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/examples/perception_xalpha_quickstart.ipynb
- **Contributor benchmark:** https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/docs/CONTRIBUTOR_BENCHMARK.md

```bash
pip install perception-xalpha-lite
xalpha-demo --output-dir xalpha-demo-output
```

The installed demo creates deterministic synthetic prices and point-in-time fundamentals,
runs the complete falsification loop, and writes inspectable inputs, readiness and result
artifacts. Nothing in it is a performance claim.

Paper-backed mechanisms and audited DSL factors can be proposed through the repository's issue
templates. The contributor leaderboard measures reproducibility and causal-test coverage—not
returns, alpha, or deployment readiness.

Audit local data before spending compute on discovery:

```bash
xalpha-doctor \
  --prices data/prices.csv \
  --fundamentals data/fundamentals.csv \
  --config configs/example.json \
  --output outputs/data_readiness.json
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

## Reproducible audit cases

The current GitHub source adds three synthetic cases: noise selection, disclosure timing,
and price-basis consistency. [Try the guide](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/demo.html)
or [reproduce the examples](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/docs/tutorials/README.md).
Install from GitHub main for these new cases; an older PyPI build may not include them.

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
| `doctor` | fail-closed schema, disclosure-timing, tradability and leakage-risk preflight |
| `synthetic` | deterministic zero-setup data for the installed full-loop demonstration |

Command line: `xalpha-lite`, `xalpha-evidence`, `xalpha-forward`, `xalpha-doctor`, `xalpha-demo`.

## Current status, stated plainly

Nothing has graduated. Candidates are generated and fully evaluated; none has cleared the
counterfactual, walk-forward and multiple-testing gates together. That is the gates working on
a price-and-volume factor library, not the engine failing to run — and unlike most backtests,
this one reports the exact count and the reason each candidate died.

The installed full-loop demo uses synthetic data. Previously published empirical records
on GitHub are separately labeled and are not synthetic or certified performance. See the
[data provenance policy](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/docs/DATA_PROVENANCE.md).
**It makes no profitability claim and never will.**

## Limitations

The full-loop demonstration is synthetic. Public financial endpoints may not preserve every restatement vintage.
A contemporary security master creates survivorship bias unless replaced by genuine
point-in-time membership. A zero-investment research portfolio is not executable in a long-only
cash market. Equal overlapping tranches approximate a holding horizon and model neither queue
priority nor market impact. Stationary-bootstrap inference assumes weak stationarity, and no
resampling procedure repairs contaminated data or an incomplete trial ledger.

MIT. Research and educational use only. No investment advice.
