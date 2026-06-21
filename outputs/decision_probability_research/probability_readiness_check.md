# Probability Readiness Check

Final status: `research-valid start / trade-invalid probability`

- Brier vs neutral 50%: - vs -
- LogLoss vs neutral 50%: - vs -
- ECE <= 5.00%: `False`
- full + rolling AUC >= 0.60: `False`
- both proper scores beat neutral: `False`

## Full and rolling diagnostics

| window | n | Brier | LogLoss | AUC | ECE | MCE |
|---|---:|---:|---:|---:|---:|---:|
| full forward | 0 | - | - | - | - | - |
| latest 20 trading days | 0 | - | - | - | - | - |
| latest 50 signals | 0 | - | - | - | - | - |

| tier | minimum outcomes | minimum days | sample pass | statistical pass | action |
|---|---:|---:|---:|---:|---|
| shadow_warmup | 50 | 20 | false | false | report only |
| research_validated | 100 | 40 | false | false | consider simulated filtering |
| gate_candidate | 200 | 60 | false | false | recommendation report only |
| position_candidate | 300 | 60 | false | false | requires stable EV bins |

Even if every statistical gate passes, this pipeline cannot enable live filtering or sizing.
