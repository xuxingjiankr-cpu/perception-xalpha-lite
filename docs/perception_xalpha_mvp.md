# Perception-XAlpha MVP

## What this adds

The existing autonomous CogAlpha loop already owns the bounded factor DSL,
train-only evolution, purged chronological split, costed evaluation, PBO
diagnostics and shadow isolation. Perception-XAlpha adds the missing
market-perception layer from the supplied architecture:

1. detect recurring return, volume, range, correlation and CUSUM anomalies;
2. aggregate them into immutable `Phenomenon Ticket` records;
3. generate four bounded factor families from those tickets;
4. pass every factor through the existing deterministic CogAlpha evaluator;
5. persist ticket, factor, experiment, validation and relationship lineage in
   an append-only SQLite registry.

It does not create a new trading system and does not call an LLM or remote API.
The generator emits JSON expressions in the existing causal DSL. Arbitrary
Python, negative lags, imports, subprocesses, file/network access and trading
functions are unrepresentable and fail closed.

## Phenomenon tickets

Each ticket records the detector, affected assets, first/last observation,
baseline and observed residual scale, independent trading days, event count,
data snapshot hash and config hash. A ticket is admitted only after the
preregistered recurrence and data-quality gates pass. SQLite triggers reject
updates and deletes to append-only research entities.

Tickets are evidence that a recurring anomaly exists; they are not evidence
that it is tradable.

## Four generated factor families

- `price_volume_coupling`: price impact, volume surprise and return-volume
  coherence;
- `lead_lag`: causal lag response and return autocorrelation;
- `conditional_beta`: separate upside/downside market beta proxies;
- `shock_response`: response to lagged market shocks and short-term recovery.

All market-wide fields are computed from the completed daily cross-section and
then broadcast to symbols at the same timestamp. Signal at day `t` is evaluated
using hypothetical execution from day `t+1` open, as in the base CogAlpha
protocol.

## Validation and status

Candidate states are:

`DRAFT → STATIC_VALIDATED → LEAKAGE_VALIDATED → HISTORICALLY_VALIDATED`

or `REJECTED`. Historical success may create a research `BORN` record, but
neither `BORN` nor this historical run may become `SHADOW` automatically.
`SHADOW` requires a separate forward protocol, at least 60 fresh trading days
and explicit human approval.

Validation/shadow results are never supplied to the generator. The historical
window cannot promote a factor, regardless of its score.

## Running

Smoke, without persistent registry writes:

```powershell
py -3.13 scripts/research_perception_xalpha.py --no-state --maximum-candidates 8 --force
```

Normal weekly research run:

```powershell
py -3.13 scripts/research_perception_xalpha.py
```

The wrapper is `scripts/run_perception_xalpha_weekly.ps1`. It is intentionally
separate from the paper-trading scheduler.
