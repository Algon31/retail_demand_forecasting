"""
Tests for Data Ingestion, Schemas, and Data Quality Validation.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.loader import validate_raw_data
from src.data.cleaner import clean_and_consolidate

def test_data_validation_assertions():
    """Verify that quality assertions pass on clean data and fail on dirty data."""
    clean_train = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "store_nbr": [1] * 10,
        "family": ["BEVERAGES"] * 10,
        "sales": [100.0] * 10,
        "onpromotion": [0] * 10
    })
    
    checks = validate_raw_data({"train": clean_train})
    assert checks["no_null_date"] is True
    assert checks["no_null_store"] is True
    assert checks["no_null_family"] is True
    assert checks["zero_duplicates"] is True
    assert checks["no_negative_sales"] is True

def test_duplicate_and_negative_detection():
    """Ensure duplicate records and negative sales are flagged."""
    dirty_train = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        "store_nbr": [1, 1],
        "family": ["BEVERAGES", "BEVERAGES"],
        "sales": [-5.0, 50.0],
        "onpromotion": [0, 0]
    })
    
    checks = validate_raw_data({"train": dirty_train})
    assert checks["zero_duplicates"] is False
    assert checks["no_negative_sales"] is False
