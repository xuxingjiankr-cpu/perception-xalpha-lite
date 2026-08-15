# Selective win classifier V4

V4 tests one hypothesis only: if the objective is next-session positive return,
the return-magnitude head must not dilute the ordering target. Stocks are ordered
solely by the raw, out-of-fold probability-up head. The expected-return head is
reported for diagnostics but has zero ranking weight.

The reject option follows the selective-classification risk/coverage framework:

- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
  formalizes abstention as a trade-off between predictive risk and coverage.
- [Selective Classification via One-Sided Prediction](https://proceedings.mlr.press/v130/gangrade21a.html)
  motivates one-sided control when false positives are the error of interest.
- [Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276)
  supports regularized nonlinear interactions but also emphasizes the low
  signal-to-noise ratio of stock returns.

For each purged walk-forward refit, the model is trained on the base portion of the
504-day window. The final 63 training days form a calibration segment. Raw scores are
converted to ten within-day percentile bins. A bin is eligible only when its fixed
90% one-sided Wilson lower bound for positive-return frequency exceeds 50%, with at
least 1,000 calibration observations. A separately calibrated severe-loss probability
must not exceed 8%. Test labels never determine eligibility.

The candidate may select zero to ten stocks. Its result is compared with the
prior-close control on exactly the same dates and with exactly the same number of
stocks, exposing any apparent accuracy gain caused only by lower coverage.

This is historical research on a window that has already been viewed. It cannot be
promoted, connected to trading, or used to change orders, positions, risk gates,
execution locks or production configuration. At most, a successful result can justify
a new frozen forward experiment.
