"""
Time-Series Evaluation Metrics and Rolling-Origin Backtesting.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes industry-standard retail demand forecasting metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - WAPE: Weighted Absolute Percentage Error (%)
    - Bias: Forecast Bias (%)
    - RMSLE: Root Mean Squared Logarithmic Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip negative predictions to 0
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    
    sum_actual = float(np.sum(y_true))
    sum_abs_err = float(np.sum(np.abs(y_true - y_pred)))
    sum_err = float(np.sum(y_pred - y_true))
    
    # WAPE (undefined if sum_actual is 0)
    wape = (sum_abs_err / sum_actual * 100.0) if sum_actual > 0 else 0.0
    bias = (sum_err / sum_actual * 100.0) if sum_actual > 0 else 0.0
    
    # RMSLE
    log_true = np.log1p(np.maximum(0.0, y_true))
    log_pred = np.log1p(np.maximum(0.0, y_pred))
    rmsle = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))
    
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE": round(wape, 2),
        "Bias": round(bias, 2),
        "RMSLE": round(rmsle, 4)
    }

def evaluate_by_group(
    df: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "predicted",
    group_col: str = "family"
) -> pd.DataFrame:
    """
    Evaluates forecast performance across sub-segments (e.g. per category or per store).
    """
    results = []
    for group_name, group_data in df.groupby(group_col):
        m = calculate_metrics(group_data[actual_col].values, group_data[pred_col].values)
        m[group_col] = group_name
        m["observations"] = len(group_data)
        m["total_actual_volume"] = round(float(group_data[actual_col].sum()), 1)
        results.append(m)
        
    res_df = pd.DataFrame(results)
    # Sort by WAPE ascending
    return res_df.sort_values("WAPE").reset_index(drop=True)
