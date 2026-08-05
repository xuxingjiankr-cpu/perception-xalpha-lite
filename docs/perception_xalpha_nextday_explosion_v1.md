# Perception-XAlpha next-session explosion V1

This is an isolated historical research model. It does not read a trading config, call a
broker, create an order, change a position, or feed `build_decision()`.

## Question

Can information available after an A-share session closes identify a very small set of
stocks with both unusually high immediate upside and controlled downside?

The model deliberately does **not** promise a profitable trade or a price-limit event. A
zero-name result is valid.

## Executable clock

1. Features stop at day-t close.
2. Entry is day-t+1 open, only if the session is buyable.
3. Exit is day-t+2 open, only if the session is sellable.
4. A sealed-up entry and sealed-down exit are excluded from labels.
5. The price-limit label uses the next session's high and the board-specific 10%/20%
   limit. This is an offline outcome and never enters a feature.

The model therefore distinguishes a chart that touched its limit from a position that an
account could have entered and exited.

## Heads

- executable return regression;
- probability of executable return at least 5%;
- probability of a next-session board-normalised price-limit touch;
- probability of a non-positive executable return;
- probability of a loss no better than -3%;
- conditional tenth-percentile return.

Every head has a base-fit segment, a purged calibration segment and a never-fit reliability
audit. An unreliable probability head falls back to its base-rate prior. An unreliable
return or quantile head falls back to its corresponding base-fit prior. Selection is
disabled unless every head passes.

## Candidate coverage

A fixed past-only rank composite selects 500 candidates from the clean PIT universe. The
training-period recall of strong-gain and limit-touch events is reported and must exceed
the preregistered floor. This prevents a superficially accurate model from hiding most
events in the discarded universe.

## Selection

At most three stocks may pass. Each must satisfy all of:

- expected executable return at least 1%;
- non-positive-return probability at most 30%;
- strong-gain probability at least 15%;
- limit-touch probability at least 3%;
- severe-loss probability at most 12%;
- predicted tenth-percentile return at least -4%.

These thresholds were frozen before the first historical run. Validation and shadow may
reject them but may not tune them. Historical success still cannot connect the model to
trading; it would only justify a separately preregistered forward shadow study.

## Run

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3.13 scripts/research_perception_xalpha_nextday_explosion.py
```

Artifacts are written under
`outputs/edge_research/perception_xalpha_nextday_explosion_v1/<run_id>/`.
