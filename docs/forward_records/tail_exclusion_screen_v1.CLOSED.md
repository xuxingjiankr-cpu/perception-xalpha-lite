# tail_exclusion_screen_v1 - CLOSED 2026-09-12

Spec `1620e0883b64cdd398ac37120cfaab223b78a702ebe38edb7c720cee6af38998`, frozen
2026-09-04, holding one recorded session (2026-09-04).

The record is abandoned, not failed. It was opened to settle RESEARCH_LOG #13 on
fresh sessions. Before enough sessions accumulated, the ablation that #13 always
needed was run, and it falsified the finding on the historical windows: a plain
twenty-session realised-volatility sort ranks tail risk better than the sixteen-factor
composite on BOTH windows, and the slice volatility excludes costs nothing to drop
(-6.1 bps at h=1, -12.3 at h=5) while the slice the composite excludes gives up
return (+6.7 and +32.9 bps).

Continuing to accumulate sixty sessions would have tested whether a dominated method
holds up out of sample. Even a pass would not have made it usable, because the free
alternative was already better on both axes.

The spec and its single prediction stay here unmodified. A frozen record is not
deleted because its conclusion changed; it is marked and left in place.

If the volatility ranking is ever worth a forward record of its own, that is a NEW
specification with its own digest, not an edit of this one.

See RESEARCH_LOG #13 and #14.
