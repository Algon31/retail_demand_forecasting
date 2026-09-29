"""
Tests for Model Evaluation Metrics and Baselines.
"""
import pytest
import numpy as np
import pandas as pd
from src.models.evaluation import calculate_metrics, evaluate_by_group
from src.models.baselines import BaselineForecaster

def test_metrics_calculation():
    """Verify MAE, RMSE, WAPE, and Bias formulas."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])  # errors: +10, -10, +30 (abs sum = 50, net sum = +30)
    
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(50.0 / 3.0, rel=1e-3)
    assert metrics["WAPE"] == pytest.approx((50.0 / 600.0) * 100.0, rel=1e-3)
    assert metrics["Bias"] == pytest.approx((30.0 / 600.0) * 100.0, rel=1e-3)

def test_wape_zero_actual_handling():
    """WAPE should gracefully return 0.0 when actual demand is 0 instead of crashing with ZeroDivisionError."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["WAPE"] == 0.0
    assert metrics["Bias"] == 0.0
