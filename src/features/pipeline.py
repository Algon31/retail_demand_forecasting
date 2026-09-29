"""
Feature Pipeline: Generates temporal, lag, rolling statistical,
and contextual promotional/holiday features with strict leakage prevention.
"""
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from src.config import settings

FEATURE_COLUMNS = [
    "store_nbr",
    "family_encoded",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "onpromotion",
    "oil_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date and calendar indicators."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag and rolling window features.
    Enforces STRICT leakage prevention by applying .shift(1) on grouped targets.
    """
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    grouped = df.groupby(["store_nbr", "family"])["sales"]
    
    # Historical Lags
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)
    df["lag_28"] = grouped.shift(28)
    
    # Rolling Statistics (all shifted by 1 day)
    shifted = grouped.shift(1)
    df["rolling_mean_7"] = shifted.rolling(7, min_periods=1).mean()
    df["rolling_mean_14"] = shifted.rolling(14, min_periods=1).mean()
    df["rolling_mean_28"] = shifted.rolling(28, min_periods=1).mean()
    
    df["rolling_std_7"] = shifted.rolling(7, min_periods=2).std().fillna(0.0)
    df["rolling_std_28"] = shifted.rolling(28, min_periods=2).std().fillna(0.0)
    
    return df

def build_features_dataset(
    df: pd.DataFrame,
    family_mapping: dict = None,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Transforms cleaned sales records into a complete feature dataset for ML.
    """
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    
    # Encode categorical product family
    if family_mapping is None:
        unique_families = sorted(df["family"].unique().tolist())
        family_mapping = {f: i for i, f in enumerate(unique_families)}
        
    df["family_encoded"] = df["family"].map(family_mapping)
    
    if drop_na:
        # Drop warm-up rows where lag_28 is null
        df = df.dropna(subset=["lag_28", "rolling_mean_28"]).reset_index(drop=True)
        
    return df, family_mapping

def save_features(df: pd.DataFrame, output_dir: Path = settings.FEATURES_DATA_DIR):
    """Saves engineered features table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    # Also save a sample CSV for easy inspection
    df.head(1000).to_csv(output_dir / "features_sample.csv", index=False)
    print(f"Engineered features saved: {out_path} ({len(df):,} rows)")
