# Thirteen-factor individual discrimination V4

This is a research-only test of whether retaining all thirteen factor ranks can distinguish
stocks inside the Top10 better than calibrating only the final scalar composite rank.

The existing scalar model compresses the latest Top10 into scores from roughly 0.9976 to
1.0000. Ridge therefore produces nearly identical expected returns and isotonic probability
calibration places several names on the same step. Multiplying scores or applying a lower
temperature would only manufacture confidence and cannot improve ordering or calibration.

V4 fixes the model before outcomes are read. It fits one regularized return head and three
regularized probability heads on the frozen base-fit dates, uses the frozen independent
calibration dates for return calibration and Platt probability calibration, and treats audit,
validation and shadow as reject-only. Every row receives equal day weight; the large base-fit
block is deterministically capped at 600 rows per day for bounded memory, with no outcome-based
sampling.

The multivariate model is useful only if the audit block improves AUC, Brier and LogLoss for
both gross-up and net-positive probabilities, improves Top10 mean net return and net win rate,
increases probability dispersion, and does not reduce score-decile monotonicity. Dispersion
alone is explicitly insufficient.

The 15% Alpha070 / 15% Alpha131 policy was selected after historical results were viewed. Even
if every V4 historical gate passes, the result remains a post-selection hypothesis requiring a
separately frozen 60-day forward record. It cannot create orders or alter trading configuration.
