# Literature-to-code traceability

Perception-XAlpha Lite treats academic literature as a source of **testable mechanisms and
research constraints**, not as proof that an implementation has alpha. This document records:

1. the claim supported by each reference;
2. the corresponding public implementation;
3. the boundary the implementation must not cross.

No empirical research result is included.

## Statistical validation and research integrity

| Method | Evidence from the literature | Public implementation | Explicit boundary |
|---|---|---|---|
| Probability of Backtest Overfitting | CSCV estimates the probability that selection among many backtests produces an in-sample winner that underperforms out of sample | `pbo()` | PBO cannot repair contaminated data, invalid labels, or an understated trial count |
| Deflated Sharpe Ratio | DSR corrects apparent Sharpe evidence for multiple testing and non-normal returns | `deflated_sharpe_ratio()` | All attempted candidates, including failures, must enter the trial ledger |
| Purged chronological validation | Overlapping outcome horizons can leak information across adjacent train/test samples | `make_split()` and purged walk-forward folds | Purge length must cover the maximum label horizon |
| Counterfactual and placebo controls | A candidate should outperform mechanism-specific alternatives and a permutation distribution | Primary/Counter/Placebo evaluation | One lucky placebo draw is insufficient; the empirical distribution is required |
| Proper probability scoring | Strictly proper scores incentivize honest probabilistic forecasts | `probability_metrics()` | AUC alone is insufficient; Brier, LogLoss, and calibration error are reported separately |
| Stationary bootstrap | Geometrically distributed circular blocks preserve weak time dependence under resampling | `stationary_bootstrap_indices()` and mean intervals | Block length must be frozen and weak stationarity must be defensible |
| Reality Check | The best member of a searched family must be tested against the joint data-snooping null | `white_reality_check()` | Global rejection does not identify an executable strategy |
| Step-down max-t | Joint resampling and studentized step-down tests control family-wise error more powerfully than single-step correction | `romano_wolf_stepdown()` | Candidate family and benchmark must be frozen before testing |
| False discovery rate | BH controls FDR under its dependence conditions; BY adds a conservative arbitrary-dependence correction | `benjamini_hochberg_qvalues()` and `benjamini_yekutieli_qvalues()` | FDR is complementary to, not a replacement for, family-wise inference |

### Primary references

- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2015).
  *The Probability of Backtest Overfitting*. Journal of Computational Finance.
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) ·
  [DOI](https://doi.org/10.2139/ssrn.2326253)
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
  Journal of Portfolio Management, 40(5), 94–107.
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) ·
  [DOI](https://doi.org/10.2139/ssrn.2460551)
- Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation*. Journal of the American Statistical Association,
  102(477), 359–378. [DOI](https://doi.org/10.1198/016214506000001437)
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*.
  Journal of the American Statistical Association, 89(428), 1303–1313.
  [DOI](https://doi.org/10.1080/01621459.1994.10476870)
- White, H. (2000). *A Reality Check for Data Snooping*. Econometrica, 68(5),
  1097–1126. [DOI](https://doi.org/10.1111/1468-0262.00152)
- Romano, J. P., & Wolf, M. (2005). *Stepwise Multiple Testing as Formalized
  Data Snooping*. Econometrica, 73(4), 1237–1282.
  [DOI](https://doi.org/10.1111/j.1468-0262.2005.00615.x)
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate:
  A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal
  Statistical Society: Series B, 57(1), 289–300.
  [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- Benjamini, Y., & Yekutieli, D. (2001). *The Control of the False Discovery
  Rate in Multiple Testing under Dependency*. Annals of Statistics, 29(4),
  1165–1188. [DOI](https://doi.org/10.1214/aos/1013699998)

The complete method contract and command-line workflow are documented in
[EVIDENCE_LAB.md](EVIDENCE_LAB.md).

## Autonomous formulaic factor discovery

| Research question | Literature contribution | Public implementation | Deliberate simplification |
|---|---|---|---|
| How can a system explore formulaic factors while retaining interpretability? | AlphaForge uses a generative-predictive architecture for formulaic factor generation and combination | allowlisted JSON DSL, economic seeds, bounded mutation/crossover | No neural generator; every expression is statically inspectable |
| How can factor lineage remain reproducible? | Evolutionary and grammar-guided research motivates explicit generation histories | immutable factor ID, parent IDs, operator, generation, expression hash | No claim that lineage quality implies predictive quality |
| How should redundant candidates be handled? | Diverse factor sets reduce repeated variants of the same behavior | train-only behavioral-correlation pruning | Validation/shadow correlations never select the parent pool |

### Primary references

- Shi, H., Song, W., Zhang, X., Shi, J., Luo, C., Ao, X., Arian, H., & Seco, L.
  (2024). *AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha
  Factors*. [arXiv:2406.18394](https://arxiv.org/abs/2406.18394)
- Zhang, T., et al. (2020). *AutoAlpha: An Efficient Hierarchical Evolutionary
  Algorithm for Mining Alpha Factors in Quantitative Investment*.
  [arXiv:2002.08245](https://arxiv.org/abs/2002.08245)

## Decision-focused ranking and calibration

Prediction error is not identical to downstream decision loss. The optional decision toolkit
therefore fits a bounded Top-K pairwise objective after the factor definitions and directions
are frozen.

```math
\min_{w \in \Delta_{[l,u]}}
\frac{1}{T}\sum_t\frac{1}{|P_t|}
\sum_{(i,j)\in P_t}\log(1+e^{-w^\top(x_{ti}-x_{tj})})
+ \lambda\lVert w-w_0\rVert_2^2.
```

| Method | Public implementation | Boundary |
|---|---|---|
| Decision-focused learning | `fit_pairwise_topk_weights()` | This is a transparent pairwise surrogate, not an implementation of the full SPO optimizer |
| Weight sensitivity | `fit_pairwise_weight_ensemble()` | Complete-block replicas measure instability; they are not a posterior distribution |
| Expected-outcome calibration | `fit_ridge_score_model()` | Fitted only on the independent calibration block |
| Event-probability calibration | `fit_logistic_probability_model()` | Ordinary and severe loss events are separate models |
| Reliability evaluation | `probability_metrics()` | Validation/shadow may reject but never refit the calibrator |

### Primary references

- Elmachtoub, A. N., Liang, J. C. N., & McNellis, R. (2020). *Decision Trees
  for Decision-Making under the Predict-then-Optimize Framework*. Proceedings of
  Machine Learning Research, 119, 2858–2867.
  [PMLR](https://proceedings.mlr.press/v119/elmachtoub20a.html)
- Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation*. [DOI](https://doi.org/10.1198/016214506000001437)
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*.
  Journal of the American Statistical Association, 89(428), 1303–1313.
  [DOI](https://doi.org/10.1080/01621459.1994.10476870)

## Regime and transition primitives

| Mechanism | Literature claim | Public implementation | Boundary |
|---|---|---|---|
| Hidden Markov regime model | Time-series parameters may depend on an unobserved discrete-state Markov process | `causal_two_state_hmm_filter()` | Forward filtering only; full-path Viterbi labels use future observations |
| Early-warning signals | Rising variance and autocorrelation may accompany critical slowing before some transitions | `early_warning_features()` | A generic warning is not directional return alpha |
| Bayesian online change-point detection | Run-length posteriors support online inference of abrupt generative changes | `bocpd_change_probability()` | Hazard assumptions and observation models must be frozen |

### Primary references

- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2), 357–384.
  [DOI](https://doi.org/10.2307/1912559)
- Scheffer, M., et al. (2009). *Early-Warning Signals for Critical Transitions*.
  Nature, 461, 53–59. [DOI](https://doi.org/10.1038/nature08227)
- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint
  Detection*. [arXiv:0710.3742](https://arxiv.org/abs/0710.3742)

## Dynamical-systems and bubble-feasibility primitives

| Mechanism | Literature claim | Public implementation | Boundary |
|---|---|---|---|
| Dynamic Mode Decomposition | DMD approximates dynamics through the eigendecomposition of a fitted linear operator and connects to Koopman analysis | `causal_dmd_residual()` | Fixed past window; linear consistency and rank deficiency require audit |
| LPPLS | Reduced-parameter calibration can improve the numerical stability of log-periodic power-law fitting | `simplified_lppls_features()` | Fixed grid, residuals, and `t_c` stability distribution; no exact crash-date forecast |

### Primary references

- Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L., & Kutz, J. N.
  (2014). *On Dynamic Mode Decomposition: Theory and Applications*.
  Journal of Computational Dynamics, 1(2), 391–421.
  [arXiv:1312.0041](https://arxiv.org/abs/1312.0041)
- Filimonov, V., & Sornette, D. (2013). *A Stable and Robust Calibration
  Scheme of the Log-Periodic Power Law Model*. Physica A, 392(17), 3698–3707.
  [arXiv:1108.0099](https://arxiv.org/abs/1108.0099) ·
  [DOI](https://doi.org/10.1016/j.physa.2013.04.012)

## Portfolio views, offline labels, and finite-sample bounds

| Mechanism | Public implementation | Boundary |
|---|---|---|
| Black–Litterman view shrinkage | `black_litterman_posterior()` | Stabilizes uncertain views; cannot create predictive content |
| Triple Barrier labeling | `triple_barrier_labels()` | Uses future paths by design and belongs only in an offline label table |
| Hoeffding lower bound | `hoeffding_lower_bound()` | Independence assumptions and effective sample size remain explicit limitations |

### Primary references

- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization*.
  Financial Analysts Journal, 48(5), 28–43.
  [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/1992/faj-v48-n5-28) ·
  [DOI](https://doi.org/10.2469/faj.v48.n5.28)
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Triple Barrier and purged validation are represented as offline research tools.
- Hoeffding, W. (1963). *Probability Inequalities for Sums of Bounded Random
  Variables*. Journal of the American Statistical Association, 58(301), 13–30.
  [DOI](https://doi.org/10.1080/01621459.1963.10500830)

## Related directions not claimed as implemented

| Direction | Public-package boundary | Reference |
|---|---|---|
| Instrumented PCA | PIT characteristics and neutral portfolios are available; full IPCA estimation is not included | Kelly, Pruitt & Su, *Characteristics Are Covariances* ([published article](https://www.sciencedirect.com/science/article/pii/S0304405X19301151)) |
| Conditional factor timing | Block-replica weights are available; validation-driven or online timing is forbidden | *Factor Timing with Portfolio Characteristics* ([article](https://academic.oup.com/raps/article/14/1/84/7191017)) |
| Neural formula generation | The generator is bounded and symbolic; it does not reproduce AlphaForge's neural architecture | Shi et al., [arXiv:2406.18394](https://arxiv.org/abs/2406.18394) |
| Gaussian-process ensembles | Probability calibration is included; Gaussian-process forecasting is not | *Ensemble Gaussian Process Regression for Time Series Forecasting* ([arXiv:2212.01048](https://arxiv.org/abs/2212.01048)) |

## Non-claims

- A cited paper does not validate this implementation on a new market or dataset.
- A mechanism primitive is not a production model.
- Better calibration is not a guarantee of profit.
- Lower turnover or fewer selections is not automatically alpha.
- Historical validation is not fresh forward evidence.
- This repository publishes no empirical factor result and defines no execution path.
