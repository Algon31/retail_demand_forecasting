"""
Data Cleaning and Preprocessing:
Performs missing value imputation, holiday formatting, store merging,
and produces the consolidated processed dataset.
"""
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from src.config import settings
from src.data.loader import load_raw_data

def clean_and_consolidate(
    raw_dfs: Dict[str, pd.DataFrame],
    output_dir: Path = settings.PROCESSED_DATA_DIR
) -> pd.DataFrame:
    """
    Cleans raw data and joins store metadata, oil prices, and holiday indicators.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train = raw_dfs["train"].copy()
    stores = raw_dfs["stores"].copy()
    oil = raw_dfs["oil"].copy()
    holidays = raw_dfs["holidays"].copy()
    
    # 1. Clean oil prices (forward-fill and back-fill missing weekend/holiday prices)
    if not oil.empty and "dcoilwtico" in oil.columns:
        oil["date"] = pd.to_datetime(oil["date"])
        oil = oil.sort_values("date").reset_index(drop=True)
        oil["oil_price"] = oil["dcoilwtico"].ffill().bfill()
        oil = oil[["date", "oil_price"]]
    else:
        oil = pd.DataFrame(columns=["date", "oil_price"])
        
    # 2. Process holidays: flag whether a date is a national/regional holiday
    # Only keep non-transferred holidays or transfer days
    if not holidays.empty and "transferred" in holidays.columns:
        holidays["date"] = pd.to_datetime(holidays["date"])
        valid_holidays = holidays[~holidays["transferred"]].copy()
        valid_holidays["is_holiday"] = 1
        holiday_summary = valid_holidays.groupby("date")["is_holiday"].max().reset_index()
    else:
        holiday_summary = pd.DataFrame(columns=["date", "is_holiday"])
        
    # 3. Clean sales data
    train["date"] = pd.to_datetime(train["date"])
    train["sales"] = train["sales"].clip(lower=0.0)
    train["onpromotion"] = train["onpromotion"].fillna(0).astype(int)
    
    # 4. Merges
    merged = train.merge(stores, on="store_nbr", how="left")
    
    if not oil.empty:
        merged = merged.merge(oil, on="date", how="left")
        merged["oil_price"] = merged["oil_price"].ffill().bfill().fillna(70.0)
    else:
        merged["oil_price"] = 70.0
        
    if not holiday_summary.empty:
        merged = merged.merge(holiday_summary, on="date", how="left")
        merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)
    else:
        merged["is_holiday"] = 0
        
    # Sort chronologically by store, family, and date
    merged = merged.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    
    # Save processed outputs
    csv_path = output_dir / "clean_sales.csv"
    merged.to_csv(csv_path, index=False)
    print(f"Cleaned dataset saved: {csv_path} ({len(merged):,} rows)")
    
    return merged

if __name__ == "__main__":
    raw_data = load_raw_data()
    clean_df = clean_and_consolidate(raw_data)
