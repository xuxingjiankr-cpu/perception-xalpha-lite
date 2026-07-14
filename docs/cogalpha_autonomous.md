# Autonomous CogAlpha ETF research loop

## Scope

This module continuously searches for interpretable A-share ETF factors. It is a
research system, not an order-generating trading system. It cannot read or write the
paper-agent configuration, strategy overlay, position sizing, risk gates or order path.
Every emitted strategy artifact contains an empty `orders` array.

The implementation follows the useful structure of CogAlpha:

1. seven research levels containing twenty-one distinct factor roles;
2. five guidance modes for search diversity;
3. explicit economic hypotheses for every candidate;
4. program generation, quality checks, train-fitness feedback, mutation and crossover;
5. qualified and elite pools;
6. a train-only Ridge combination of elite factors;
7. persistent research memory for future weekly generations.

It does not claim to reproduce the paper's gpt-oss-120B/H100 experiment. The local
machine currently has no Ollama model installed. The built-in grammar search therefore
provides the always-available autonomous engine. If a local model is installed later,
set `COGALPHA_OLLAMA_MODEL`; the same loop will add semantic proposals without using a
remote API.

## Research protocol

The rolling split is computed only from complete daily bars:

- training: all eligible dates before the first purge;
- purge: at least the ten-session prediction horizon;
- validation: 126 trading days;
- second purge: ten trading days;
- shadow quarantine: the latest 126 trading days.

Only training metrics can choose parents, generate feedback, mutate factors, fit Ridge,
or enter persistent parent memory. Validation is a gate and report. Shadow quarantine
is report-only. Neither is included in a future generation prompt or trial ledger.

The target is the return from the next session's open through the open after ten full
trading sessions. IC, ICIR, RankIC, RankICIR and discretized mutual information measure
predictive content. A non-overlapping next-session open-to-open return proxy supplies the
costed long-only metric used in fitness. The declared round-trip cost is 15.5 basis
points.

Factor programs use a bounded, past-only expression language. Negative lags, imports,
file access, network access, arbitrary Python and order functions cannot be represented.
Generated programs are rejected for excessive missing values, insufficient cross-section
variation, missing economic mechanism or invalid structure.

## Continuous iteration

The persistent state lives only under:

`outputs/edge_research/cogalpha_autonomous/state/`

It contains:

- `parent_pool.json`: train-selected factor programs, with no validation/shadow metrics;
- `trial_ledger.jsonl`: every autonomous trial and its train metrics;
- `run_registry.jsonl`: run identity and research verdict for audit;
- `latest.json`: last processed data fingerprint.

If the daily-bar fingerprint has not changed, the weekly job exits without creating new
trials. This prevents repeated searches against identical data from silently inflating
the factor zoo.

Run manually:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_cogalpha_autonomous.py
```

Smoke test without persistent research memory:

```powershell
py -3.13 scripts/research_cogalpha_autonomous.py --maximum-candidates 24 --generations 1 --no-state --force
```

Local semantic augmentation, after separately installing an appropriate Ollama model:

```powershell
$env:COGALPHA_OLLAMA_MODEL = "the-explicitly-approved-local-model"
py -3.13 scripts/research_cogalpha_autonomous.py
```

The scheduled runner is `scripts/run_cogalpha_autonomous_weekly.ps1`. The intended
Windows task name is `CogAlpha_Autonomous_Research_Weekly`, Saturday at 09:00 local time.

## Output and promotion boundary

Each run is sealed in:

`outputs/edge_research/cogalpha_autonomous/<run_id>/`

The directory contains the full result, report, train elites and
`shadow_strategy_candidate.json`. The latter describes a research portfolio of daily
top-decile tranches held for ten sessions and lists the latest twenty ETF scores. It is
not connected to the T0 observation pool or trading agent.

No autonomous run can promote itself. Discussion of a separate integration requires at
least sixty fresh forward trading days, two hundred independent forward events, positive
costed validation and forward results, full DSR/PBO, stability across months and ETF
families, all replay invariants, and separate human approval. Daily cross-sectional edge
would still need an independent minute-level causal replay before it could influence T0.
