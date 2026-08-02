# Research mechanisms and literature map

This repository contains small, auditable implementations of mechanisms used in
the parent research project. Inclusion means “available for falsification”, not
“proven alpha”.

| Mechanism | Lite implementation | Research role | Reference |
|---|---|---|---|
| Probability of Backtest Overfitting | CSCV-style `pbo()` | Penalise selection from many candidates | Bailey et al., *The Probability of Backtest Overfitting* |
| Deflated Sharpe Ratio | `deflated_sharpe_ratio()` | Adjust Sharpe evidence for many trials and non-normal returns | Bailey & López de Prado, *The Deflated Sharpe Ratio* |
| Purged chronological validation | `make_split()` and walk-forward folds | Keep overlapping label windows out of adjacent samples | López de Prado, *Advances in Financial Machine Learning* |
| Hidden Markov Model | `causal_two_state_hmm_filter()` | Forward-filtered regime probabilities; no Viterbi look-ahead | Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series* |
| Early-warning statistics | `early_warning_features()` | Variance/autocorrelation/skew changes before transitions | Scheffer et al. (2009), *Early-warning signals for critical transitions* |
| Bayesian online change point detection | `bocpd_change_probability()` | Online transition-risk gate | Adams & MacKay (2007), *Bayesian Online Changepoint Detection* |
| Dynamic Mode Decomposition / Koopman | `causal_dmd_residual()` | Past-window linear dynamics and unexpected residual | Tu et al. (2014), *On Dynamic Mode Decomposition* |
| LPPLS | `simplified_lppls_features()` | Fixed-grid bubble-fit residual and tc stability, never a single magic crash date | Filimonov & Sornette (2013), *A Stable and Robust Calibration Scheme of the LPPLS Model* |
| Black–Litterman | `black_litterman_posterior()` | Shrink noisy views toward a prior before allocation research | Black & Litterman (1992), *Global Portfolio Optimization* |
| Triple Barrier | `triple_barrier_labels()` | Offline profit/loss/time labels; explicitly forbidden as features | López de Prado (2018) |
| Stop-loss conditionality | Triple-barrier diagnostics | Test whether exits help only under serial dependence | Kaminski & Lo (2008), *When Do Stop-Loss Rules Stop Losses?* |
| Hoeffding bound | `hoeffding_lower_bound()` | Conservative finite-sample hit-rate bound | Hoeffding (1963), *Probability Inequalities for Sums of Bounded Random Variables* |

## Important boundaries

- HMM parameters must be estimated on training data and frozen before validation.
- EWS, BOCPD, DMD and LPPLS features use only the prefix ending at time t.
- LPPLS grids and windows must be preregistered; the implementation reports fit
  stability rather than cherry-picking a visually attractive critical time.
- Triple-barrier output uses the future by design and belongs only in an offline
  label table.
- Black–Litterman improves the treatment of views; it does not create predictive
  information by itself.
- PBO/DSR reduce false discoveries but cannot repair survivorship bias or bad data.

