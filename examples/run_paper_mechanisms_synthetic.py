"""Exercise the paper-backed primitives on deterministic synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from xalpha_lite.paper_mechanisms import (
    black_litterman_posterior,
    bocpd_change_probability,
    causal_dmd_residual,
    causal_two_state_hmm_filter,
    early_warning_features,
    hoeffding_lower_bound,
    simplified_lppls_features,
    triple_barrier_labels,
)


def main() -> None:
    index = pd.bdate_range("2025-01-01", periods=180)
    returns = pd.Series(0.002 * np.sin(np.arange(180) / 9), index=index)
    price = 100 * (1 + returns).cumprod()
    features = pd.DataFrame({"return": returns, "lag": returns.shift(1).fillna(0.0)})
    hmm = causal_two_state_hmm_filter(
        returns,
        (-0.002, 0.002),
        (0.005, 0.005),
        np.array([[0.96, 0.04], [0.04, 0.96]]),
    )
    posterior_mean, _ = black_litterman_posterior(
        np.array([0.03, 0.02]),
        np.array([[0.04, 0.01], [0.01, 0.03]]),
        np.array([[1.0, -1.0]]),
        np.array([0.01]),
        np.array([[0.02]]),
    )
    output = {
        "status": "synthetic_research_mechanics_only",
        "orders": [],
        "rows": len(index),
        "ewsCompleteRows": int(early_warning_features(returns, 20).dropna().shape[0]),
        "hmmProbabilitySumAtEnd": float(hmm.iloc[-1].sum()),
        "bocpdProbabilityAtEnd": float(bocpd_change_probability(returns).iloc[-1]),
        "dmdCompleteRows": int(causal_dmd_residual(features, 20).dropna().shape[0]),
        "lpplsCompleteRows": int(
            simplified_lppls_features(price, windows=(40, 80)).dropna().shape[0]
        ),
        "blackLittermanPosteriorMean": posterior_mean.tolist(),
        "hoeffdingLowerBound": hoeffding_lower_bound(60, 100),
        "tripleBarrierLabeledRows": int(
            triple_barrier_labels(price, 0.02, 0.02, 10)["barrier_label"].notna().sum()
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
