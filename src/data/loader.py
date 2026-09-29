"""
Data Loader: Responsible for reading raw CSV files, parsing date columns,
validating schemas, and providing clean data structures.
"""
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from src.config import settings
from src.data.generator import generate_synthetic_data

def load_raw_data(data_dir: Path = settings.RAW_DATA_DIR, auto_generate: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Loads all raw files. If files do not exist and auto_generate is True,
    generates sample datasets.
    """
    train_path = data_dir / "train.csv"
    stores_path = data_dir / "stores.csv"
    oil_path = data_dir / "oil.csv"
    holidays_path = data_dir / "holidays_events.csv"
    tx_path = data_dir / "transactions.csv"
    inventory_path = data_dir / "inventory.csv"
    
    if not train_path.exists():
        if auto_generate:
            print(f"Raw data not found in {data_dir}. Generating synthetic Kaggle Store Sales + Inventory dataset...")
            return generate_synthetic_data(output_dir=data_dir)
        else:
            raise FileNotFoundError(f"Missing required file: {train_path}")
            
    print(f"Loading raw datasets from {data_dir}...")
    train = pd.read_csv(train_path, parse_dates=["date"])
    stores = pd.read_csv(stores_path)
    oil = pd.read_csv(oil_path, parse_dates=["date"]) if oil_path.exists() else pd.DataFrame(columns=["date", "dcoilwtico"])
    holidays = pd.read_csv(holidays_path, parse_dates=["date"]) if holidays_path.exists() else pd.DataFrame()
    transactions = pd.read_csv(tx_path, parse_dates=["date"]) if tx_path.exists() else pd.DataFrame()
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    
    return {
        "train": train,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
        "inventory": inventory
    }

def validate_raw_data(dfs: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """
    Runs Phase 1 data quality checks:
    - Null key checks
    - Duplicate detection
    - Negative sales check
    """
    train = dfs["train"]
    checks = {}
    
    # Check no nulls in critical primary keys
    checks["no_null_date"] = bool(train["date"].notna().all())
    checks["no_null_store"] = bool(train["store_nbr"].notna().all())
    checks["no_null_family"] = bool(train["family"].notna().all())
    
    # Check duplicates on (date, store_nbr, family)
    duplicates = train.duplicated(subset=["date", "store_nbr", "family"]).sum()
    checks["zero_duplicates"] = bool(duplicates == 0)
    
    # Negative sales check
    negative_sales = (train["sales"] < 0).sum()
    checks["no_negative_sales"] = bool(negative_sales == 0)
    
    return checks
