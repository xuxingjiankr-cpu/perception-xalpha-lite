# Break-date selection risk

- Status: `break_date_not_preregistered`
- Selected break date: `null`
- Source: no user-specified or independently documented business-event date was supplied.
- Consequence: post-jump HMM evaluation is blocked.

The dates below are exploratory distribution-shift candidates. They were ranked without profitability or model OOS metrics, but they are still selected with hindsight and cannot be used for formal claims.

| rank | candidate | distribution-shift score |
|---:|---|---:|
| 1 | 2025-05-06 | 0.713815 |
| 2 | 2025-09-01 | 0.677154 |
| 3 | 2025-08-01 | 0.672683 |
| 4 | 2025-06-03 | 0.505119 |
| 5 | 2025-01-02 | 0.393331 |

A later post-jump experiment requires a new immutable config that states the break date and independent rationale before inspecting post-break model results. Reusing the best candidate above would retain selection bias.
