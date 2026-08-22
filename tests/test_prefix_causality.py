from __future__ import annotations
import numpy as np
import pandas as pd
from xalpha_lite.paper_mechanisms import (
    bocpd_change_probability,
    causal_two_state_hmm_filter,
)

def test_hmm_filter_is_prefix_causal() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    series = pd.Series(np.sin(np.arange(100) / 10), index=dates)
    means = (-0.2, 0.2)
    sds = (0.5, 0.5)
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    
    full = causal_two_state_hmm_filter(series, means, sds, transition)
    
    # Test multiple prefix lengths
    for k in [10, 50, 90]:
        prefix = causal_two_state_hmm_filter(series.iloc[:k], means, sds, transition)
        pd.testing.assert_frame_equal(full.iloc[:k], prefix)

def test_bocpd_is_prefix_causal() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    series = pd.Series(np.sin(np.arange(100) / 10), index=dates)
    
    full = bocpd_change_probability(series)
    
    # Test multiple prefix lengths
    for k in [10, 50, 90]:
        prefix = bocpd_change_probability(series.iloc[:k])
        pd.testing.assert_series_equal(full.iloc[:k], prefix)
