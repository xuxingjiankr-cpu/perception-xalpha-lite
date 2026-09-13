# 02 — The financial statement was not available yet

[Run all cases](README.md) · [Generated report](../examples/audit-cases.md)

## Problem

A statement dated at a reporting-period end is not necessarily public on that date.
Backdating a later revision is an additional, separate source of future information.

The fixture has one fictional company: EPS 1.0 announced January 3, EPS 4.0 for the January
period announced February 14, and a revision to 2.5 available March 3. Dates use synthetic
weekdays rather than an exchange holiday calendar. No prices or investment returns are used.

## Minimal reproduction

```bash
python examples/run_audit_cases.py --output-dir outputs/audit-cases
```

Read `disclosures.csv` beside `pit_alignment.csv`.

## Wrong control → existing guard

The deliberately wrong control takes the latest statement for each reporting period and
forward-fills it from `report_date`. The existing `align_point_in_time_fundamentals()` waits:

```text
first usable session = first session strictly after max(notice_date, update_date)
```

The controls differ on 26 of 50 rows. On February 3 the wrong table already shows the
future revision, 2.5; the safe table still shows 1.0. The new disclosure first appears on
February 17, and the revision on March 4. A causal test adds the revision and verifies that
earlier rows do not change; missing disclosure dates must raise an error.

## Interpretation and limits

The test establishes a timestamp boundary on this fixture, not predictive power. Real
research needs an exchange calendar, appropriate intraday availability and historical
vendor vintages. If a vendor replaced the original disclosure, a correct aligner cannot
recover it. This conservative next-session rule does not assert optimal execution timing.

## Research basis → implementation

Arnott, Harvey and Markowitz's backtesting protocol motivates disciplined experiment
design and explicit controls against contaminated inference:
[primary publication record](https://scholars.duke.edu/publication/1518184).
The exact next-session disclosure/revision rule above is **this toolkit's conservative
engineering contract**, not a formula claimed to come from the paper. See
[`pit.py`](../../src/xalpha_lite/pit.py) and [the data contract](../DATA_DOCTOR.md).
