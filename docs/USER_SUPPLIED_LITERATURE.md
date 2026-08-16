# User-Supplied Literature Registry

> Machine-validated traceability, not a claim of implementation quality, predictive
> value, or trading readiness. A citation is evidence for a method boundary—not alpha.

This registry records which submitted papers map to public code, which require new data
or falsification, and which were reviewed but excluded. CI resolves every claimed callable
or repository path and regenerates this page from `research/literature_registry.json`.

## Status contract

| Status | Count | Meaning |
|---|---:|---|
| `implemented` | 0 | A matching bounded public callable exists; no alpha claim follows. |
| `partially_mapped` | 3 | Some auditable primitives exist, but the paper's full architecture does not. |
| `extension_candidate` | 6 | A direct, testable extension of an existing primitive; not implemented. |
| `data_gated` | 1 | The mechanism needs event types or data granularity not present in the package. |
| `deferred` | 3 | Deliberately postponed until a simpler prerequisite survives falsification. |
| `reference_only` | 8 | Research context or a future benchmark; no implementation is claimed. |
| `out_of_scope` | 6 | Reviewed and excluded because no defensible financial mechanism was specified. |

## Research map

| Paper | Domain | Status | Public mapping and boundary | Next falsifiable step |
|---|---|---|---|---|
| [AI-Trader: Benchmarking Autonomous Agents in Real-Time Financial Markets](https://arxiv.org/abs/2512.10971) (2025) | forward evaluation | `partially_mapped` | Frozen specifications and tamper-evident forward records implement the uncontaminated-evaluation principle without an agent. **Boundary:** The package contains no autonomous news agent, broker access, order path or live-trading benchmark. | Compare preregistered model cards on the same forward ledger while keeping all outputs research-only. |
| [Cognitive Alpha Mining via LLM-Driven Code-Based Evolution](https://arxiv.org/abs/2511.18850) (2025) | factor discovery | `partially_mapped` | The repository implements bounded mutation, crossover, immutable lineage and fail-closed evaluation. **Boundary:** No LLM is permitted to emit or execute arbitrary code; this is not a CogAlpha reproduction. | Submit LLM proposals through the same allowlisted DSL and compare novelty and rejection rates without changing acceptance gates. |
| [AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors](https://arxiv.org/abs/2406.18394) (2024) | factor discovery | `partially_mapped` | Bounded symbolic generation, lineage and train-only diversity pruning cover the auditable symbolic-search layer. **Boundary:** The public package does not reproduce AlphaForge's neural generator or claim its empirical performance. | Compare a frozen grammar family with economic seeds under one cumulative trial ledger. |
| [Continuous Hidden Markov Models for Equity Returns: Heavy-Tail Emission Families and Regime-Conditional Value-at-Risk](https://arxiv.org/abs/2606.23492) (2026) | regime risk | `extension_candidate` | Directly motivates replacing Gaussian-only emissions with preregistered heavy-tail families. **Boundary:** The existing causal HMM is a fixed two-state Gaussian filter; heavy-tail estimation and VaR are absent. | Hold the state count fixed and test Student-t emissions against Gaussian emissions on calibration and tail-risk metrics. |
| [Observed Fisher Information in Hidden Markov Models - Application to a Noisy Gaussian Random Walk](https://arxiv.org/abs/2606.02118) (2026) | regime uncertainty | `extension_candidate` | Motivates reporting parameter uncertainty instead of treating fitted HMM parameters as known. **Boundary:** Observed Fisher information and parameter confidence intervals are not implemented. | Quantify whether parameter uncertainty changes state-probability calibration on untouched periods. |
| [Generalized Stochastic Resilience for Early Warning Signals Based on Koopman Operator](https://doi.org/10.1007/s11071-025-12072-5) (2026) | transition diagnostics | `extension_candidate` | Provides a principled bridge between the existing EWS and DMD/Koopman primitives. **Boundary:** Generalized stochastic resilience is not implemented; EWS and DMD remain separate diagnostics. | Test incremental transition calibration versus EWS alone on the same causal windows. |
| [Finite-Sample Borel--Cantelli Inequalities under Mixing Conditions](https://arxiv.org/abs/2604.23791) (2026) | dependent concentration | `extension_candidate` | Motivates replacing independent-trial confidence claims when observations are serially dependent. **Boundary:** The current Hoeffding primitive exposes but does not correct its independence limitation. | Compare nominal and dependence-adjusted coverage on frozen block-dependent simulations. |
| [Fuk-Nagaev Inequality in Smooth Banach Spaces: Optimum Bounds for Distributions of Heavy-Tailed Martingales](https://arxiv.org/abs/2512.10012) (2025) | heavy-tail concentration | `extension_candidate` | Motivates a heavy-tail and martingale-aware alternative to bounded independent Hoeffding intervals. **Boundary:** No Fuk-Nagaev bound or martingale-difference diagnostic is implemented. | Validate finite-sample coverage under preregistered heavy-tail simulations before market use. |
| [Decoding Chinese Stock Market Returns: Three-State Hidden Semi-Markov Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2827838) (2017) | regime duration | `extension_candidate` | Motivates explicit state-duration tests for Chinese equity regimes. **Boundary:** The public HMM has geometric state durations and two states; no HSMM is implemented. | Compare frozen two-state HMM and three-state HSMM probabilities under purged chronological validation. |
| [Quantifying Reflexivity in Financial Markets: Towards a Prediction of Flash Crashes](https://arxiv.org/abs/1201.3572) (2012) | event intensity | `data_gated` | Motivates a Hawkes endogeneity diagnostic for clustered high-risk market events. **Boundary:** No Hawkes estimator is included because daily or bar data cannot identify event excitation reliably. | Audit event-time coverage and recover excitation on synthetic processes before any market hypothesis test. |
| [Predicting Critical Transitions with Machine Learning Trained on Surrogates of Historical Data](https://doi.org/10.1038/s42005-025-02172-4) (2025) | transition diagnostics | `deferred` | Motivates surrogate-trained transition classifiers after simple causal diagnostics are exhausted. **Boundary:** Cross-domain surrogate ML is not implemented and cannot validate a financial signal by analogy. | Show that surrogate-trained predictions remain calibrated under held-out real regimes and simulator perturbations. |
| [Deep LPPLS: Forecasting of Temporal Critical Points in Natural, Engineering and Financial Systems](https://arxiv.org/abs/2405.12803) (2024) | critical-transition modeling | `deferred` | The fixed-grid classic LPPLS primitive supplies a transparent feasibility baseline. **Boundary:** Deep LPPLS is deliberately excluded until classic LPPLS shows stable incremental value. | First require stable classic LPPLS residual and critical-time distributions across periods and asset groups. |
| [The Self-Organized Criticality Paradigm in Economics and Finance](https://arxiv.org/abs/2407.10284) (2024) | complex systems | `deferred` | Supplies a hypothesis vocabulary for clustered activity, avalanches and endogenous fragility. **Boundary:** No SOC score or avalanche process is represented as a validated financial feature. | Test scaling and alternative generative explanations before defining any SOC-derived feature. |
| [Multifactor Timing with Deep Learning](https://doi.org/10.1093/jjfinec/nbag006) (2026) | factor timing | `reference_only` | Provides a multitask alternative for testing whether shared structure improves factor-sign forecasts. **Boundary:** No LSTM, multitask neural network or deep timing policy is implemented. | Compare multitask predictions with equal-weight and linear timing baselines under the same cost model. |
| [Vector-Quantized Discrete Latent Factors Meet Financial Priors: Dynamic Cross-Sectional Stock Ranking Prediction for Portfolio Construction](https://arxiv.org/abs/2605.13407) (2026) | latent factor ranking | `reference_only` | Motivates future tests of discrete latent states alongside explicit financial priors. **Boundary:** Vector quantization, mixture-of-experts routing and learned latent factors are not implemented. | Test whether a frozen discrete-state representation adds OOS ranking value over explicit factors on identical support. |
| [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739) (2025) | financial time-series foundation models | `reference_only` | Provides a candidate representation benchmark for candlestick and volatility forecasting. **Boundary:** No Kronos weights, tokenizer or inference path ships in this repository. | Freeze one public checkpoint and compare zero-shot features with price-only baselines under purged walk-forward. |
| [Meta-Learning the Optimal Mixture of Strategies for Online Portfolio Selection](https://arxiv.org/abs/2505.03659) (2025) | online strategy mixtures | `reference_only` | Motivates comparison of frozen ensembles with adaptive mixtures under regime drift. **Boundary:** No online learner changes strategy weights in this research-only package. | Measure whether an online mixture beats equal weight after costs without reading validation or shadow outcomes. |
| [Factor Timing with Portfolio Characteristics](https://academic.oup.com/raps/article/14/1/84/7191017) (2024) | factor timing | `reference_only` | Motivates preregistered conditional factor timing rather than validation-driven weight switching. **Boundary:** Block-replica weight sensitivity is implemented, but online factor timing is forbidden. | Fit timing rules only on rolling training blocks and report the gap from hindsight factor selection. |
| [Tipping Point Detection and Early Warnings in Climate, Ecological, and Human Systems](https://doi.org/10.5194/esd-15-1117-2024) (2024) | early-warning review | `reference_only` | Frames failure modes and validation requirements for generic early-warning indicators. **Boundary:** Evidence from nonfinancial systems is methodological context, not financial validation. | Audit false alarms, lead time and regime specificity rather than reporting EWS correlation alone. |
| [Empirical Asset Pricing via Ensemble Gaussian Process Regression](https://arxiv.org/abs/2212.01048) (2022) | nonlinear asset pricing | `reference_only` | Provides a nonlinear probabilistic benchmark for cross-sectional forecasting. **Boundary:** Probability calibration is available, but Gaussian-process forecasting is not implemented. | Compare calibrated GPR predictions with linear and symbolic candidates on identical dates and names. |
| [Characteristics Are Covariances: A Unified Model of Risk and Return](https://www.sciencedirect.com/science/article/pii/S0304405X19301151) (2019) | conditional factor models | `reference_only` | Motivates conditioning latent exposures on point-in-time characteristics. **Boundary:** Instrumented PCA estimation is not included; neutralization is not IPCA. | Benchmark frozen IPCA factors against explicit characteristic factors on the same investible universe. |

## Reviewed and excluded

These records are retained to prevent impressive terminology from being recycled into
a factor without a causal market mechanism.

| Paper | Domain | Why it is excluded |
|---|---|---|
| [Fractional Excitations in Kitaev Quasi-One-Dimensional Chain](https://arxiv.org/abs/2606.20309) (2026) | condensed-matter physics | Physical terminology is not transferred into factor definitions by analogy. |
| [Simplex Faces and Quadratic Toric Ideals of Lattice Polytopes](https://arxiv.org/abs/2606.20430) (2026) | discrete geometry | Mathematical sophistication alone is not a finance mechanism. |
| [Establishing an Omega(sqrt(d)) Complexity Lower Bound for PDMP Samplers and How to Break It: A Sub-sqrt(d) Algorithm for Gaussian-Tailed Targets](https://arxiv.org/abs/2606.19909) (2026) | Monte Carlo theory | Sampling complexity theory is not presented as alpha generation. |
| [Quantile of Means: A Bonus-Free Ensemble Method for Minimax Optimal Reinforcement Learning](https://arxiv.org/abs/2606.20107) (2026) | reinforcement learning | An RL confidence construction is not treated as a trading factor without a separately specified decision process. |
| [Reading Weakly, Acting Strongly: A Static Parity Horizon and Its Dynamical Bypass in the Monitored Lipkin--Meshkov--Glick Model](https://arxiv.org/abs/2606.24928) (2026) | quantum information | No quantum-information objective is used in validation. |
| [Existence of Weak Solutions of the Surface Beris--Edwards Model](https://arxiv.org/abs/2607.01638) (2026) | continuum physics | The PDE model is not repurposed as a price predictor. |

## Non-claims

- `partially_mapped` does not mean the cited architecture was reproduced.
- `extension_candidate` does not authorize implementation, tuning, or promotion.
- Cross-domain evidence cannot validate a financial feature by analogy.
- No paper in this registry changes an order, position, risk gate, or execution path.
- Empirical results and proprietary factor weights are intentionally absent.
