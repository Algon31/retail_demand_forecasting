"""
Tests for Feature Engineering and Temporal Leakage Prevention.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.pipeline import add_calendar_features, add_lag_and_rolling_features

def test_temporal_leakage_prevention():
    """
    Critical test:
    Verify that perturbing future actual sales does NOT change historical lag
    or rolling features for prior dates.
    """
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    sales_a = [100.0 + i for i in range(30)]
    
    df_base = pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * 30,
        "family": ["BEVERAGES"] * 30,
        "sales": sales_a
    })
    
    # Copy with future sales on Day 25 changed dramatically (from 125 to 9999)
    sales_perturbed = sales_a.copy()
    sales_perturbed[24] = 9999.0
    df_perturbed = pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * 30,
        "family": ["BEVERAGES"] * 30,
        "sales": sales_perturbed
    })
    
    feat_base = add_lag_and_rolling_features(df_base)
    feat_perturbed = add_lag_and_rolling_features(df_perturbed)
    
    # Historical feature on day 24 (index 23) must be strictly identical!
    # Day 24's lag_1 uses day 23's sales. The perturbation happened on day 25.
    for col in ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28"]:
        val_base = feat_base.loc[23, col]
        val_pert = feat_perturbed.loc[23, col]
        if pd.notna(val_base):
            assert val_base == pytest.approx(val_pert), f"Leakage detected in {col} on prior date!"

def test_calendar_features():
    """Verify weekend flags and calendar extractions."""
    df = pd.DataFrame({
        "date": ["2025-01-04", "2025-01-05", "2025-01-06"]  # Saturday, Sunday, Monday
    })
    cal = add_calendar_features(df)
    assert list(cal["is_weekend"]) == [1, 1, 0]
    assert list(cal["day_of_week"]) == [5, 6, 0]
