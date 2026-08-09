# Point-in-time fundamental mechanism families V1

> **Preregistered research-only / shadow-only protocol. It cannot place orders, alter a
> trading decision, or promote a factor.**

## Why this study exists

The completed price-volume programme tested 456 Alpha101/GTJA191/Qlib158/academic
expressions on the clean PIT-adjusted A-share panel. Its honest rolling books were net
negative at 1, 5, 10 and 20 sessions after 30 bps round-trip cost. The apparent
hindsight edge exceeded the honest edge by roughly 3 bps/day. The twelve-factor weight
fit is therefore frozen rather than refined.

This protocol tests the remaining orthogonal information source: issuer fundamentals.
It does not reopen price-volume search and does not mix the 456 trials into fundamental
multiple-testing ledgers.

## Frozen hypotheses

Four economic mechanisms are evaluated independently:

1. **Earnings innovation** — newly reported or accelerating profitability.
2. **Growth acceleration** — revenue and operating-efficiency inflections.
3. **Quality** — persistent margins and capital efficiency.
4. **Cash-flow quality** — the degree to which operating cash supports accounting profit.

Candidate definitions and robust scales are fully enumerated in
`configs/research/fundamental_mechanism_families_v1.json`. Candidate ranks are equally
weighted inside each family; the four family ranks receive exactly 25% each. Missing
components are not reweighted: a family score requires every frozen component, and the
primary score requires all four families. Historical performance never selects a
candidate or fits a weight in the primary book.

## Point-in-time contract

- `noticeDate` is mandatory.
- Availability is the first market session strictly after
  `max(noticeDate, updateDate)`.
- `reportDate` is used only to order fiscal periods and find the same fiscal period one
  year earlier. It is never an availability timestamp.
- Multiple records exposed on one date collapse to the latest fiscal report.
- A non-advancing late restatement is skipped because the unavailable original vintage
  cannot be reconstructed safely.
- Signals are formed after the safe disclosure session closes. A hypothetical position
  enters at the following buyable open and exits at the sellable open 20 sessions later.

## Evaluation and the required overfit-exposure number

All policies use the same clean SH/SZ panel, tradability mask and 30 bps round-trip cost.
For every family, the report includes:

- the frozen equal-candidate family book;
- a hindsight selector that chooses the full-sample best component;
- an annual trailing selector that chooses one component using only prior observations;
- both selected books evaluated on the same trailing-OOS dates; and
- `hindsight net - trailing net`, in bps/day, as selection-overfit exposure.

DSR trial counts are reported separately for each mechanism family. The four-family
integration is a separate single frozen hypothesis. These counts are never pooled with
the exhausted price-volume search.

## Forward shadow contract

The version `fundamental_mechanism_families_v1_20260809` is immutable. Starting with the
first panel date strictly after 2026-08-07, each run writes one idempotent daily snapshot
and appends it to an immutable JSONL ledger. The snapshot contains the equal-family rank
and its four family contributions; it contains no claim of validated expected return,
loss probability or trade action.

At least 60 independent new trading days are required before a review. Forward outcomes
cannot refit this version. Any parameter change requires a new preregistration and a new
artifact namespace.

## Interpretation boundary

Historical results can reject a family or justify continuing its frozen shadow record;
they cannot validate it for trading. The first clean evidence is post-2026-08-07. The
validated-factor count remains zero until a separately reviewed forward result says
otherwise.
