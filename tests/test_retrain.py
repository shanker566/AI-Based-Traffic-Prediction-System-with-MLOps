import os
import pytest
import numpy as np

def test_promotion_rule_approval():
    """Verify promotion passes when candidate RMSE improves baseline and R2 exceeds threshold."""
    baseline_rmse = 8.2154
    candidate_rmse = 7.5000
    candidate_r2 = 0.9200
    tolerance = 0.05
    min_r2 = 0.85

    rmse_pass = candidate_rmse <= (baseline_rmse * (1.0 + tolerance))
    r2_pass = candidate_r2 >= min_r2
    smoke_pass = True

    should_promote = rmse_pass and r2_pass and smoke_pass
    assert should_promote is True

def test_promotion_rule_rejection_high_rmse():
    """Verify promotion rejects candidate when RMSE degrades significantly."""
    baseline_rmse = 8.2154
    candidate_rmse = 12.5000  # Large error increase
    candidate_r2 = 0.9100
    tolerance = 0.05
    min_r2 = 0.85

    rmse_pass = candidate_rmse <= (baseline_rmse * (1.0 + tolerance))
    should_promote = rmse_pass and (candidate_r2 >= min_r2)
    assert should_promote is False

def test_promotion_rule_rejection_low_r2():
    """Verify promotion rejects candidate when R2 falls below threshold."""
    baseline_rmse = 8.2154
    candidate_rmse = 8.1000
    candidate_r2 = 0.7200  # Poor correlation / fit
    min_r2 = 0.85

    r2_pass = candidate_r2 >= min_r2
    assert r2_pass is False
