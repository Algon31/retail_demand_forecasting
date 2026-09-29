"""
Baseline and Statistical Forecasting Models.
Provides benchmark comparisons for machine learning models.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.models.evaluation import calculate_metrics

class BaselineForecaster:
    """
    Evaluates baseline forecasting methods:
    1. Naive (yesterday's sales / lag_1)
    2. Seasonal Naive (same day last week / lag_7)
    3. 4-Week Seasonal Moving Average (average of last 4 same weekdays)
    """
    
    @staticmethod
    def evaluate_baselines(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Computes baseline metrics on a dataframe containing historical lags."""
        metrics_dict = {}
        
        y_true = df["sales"].values
        
        # 1. Naive (lag_1)
        if "lag_1" in df.columns:
            metrics_dict["Naive_Lag1"] = calculate_metrics(y_true, df["lag_1"].fillna(y_true.mean()).values)
            
        # 2. Seasonal Naive (lag_7)
        if "lag_7" in df.columns:
            metrics_dict["Seasonal_Naive_Lag7"] = calculate_metrics(y_true, df["lag_7"].fillna(y_true.mean()).values)
            
        # 3. 4-week weekday average (lag_7, lag_14, lag_28)
        if all(col in df.columns for col in ["lag_7", "lag_14", "lag_28"]):
            ma_pred = (df["lag_7"] + df["lag_14"] + df["lag_28"]) / 3.0
            metrics_dict["Moving_Average_4W"] = calculate_metrics(y_true, ma_pred.fillna(y_true.mean()).values)
            
        return metrics_dict

class StatisticalETSForecaster:
    """
    Holt-Winters / Exponential Smoothing time-series forecaster
    for store-family aggregated series.
    """
    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted = None

    def fit(self, series: pd.Series):
        try:
            self.model = ExponentialSmoothing(
                series,
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                trend="add",
                initialization_method="estimated"
            )
            self.fitted = self.model.fit()
        except Exception:
            # Fallback to simple exponential smoothing
            self.model = ExponentialSmoothing(series, trend="add")
            self.fitted = self.model.fit()
        return self

    def forecast(self, steps: int = 14) -> np.ndarray:
        if self.fitted is None:
            raise ValueError("Model must be fitted before forecasting.")
        preds = self.fitted.forecast(steps)
        return np.maximum(0.0, preds.values)
