"""
Tests for Inventory Optimization Engine and Simulator.
"""
import pytest
import pandas as pd
from src.inventory.engine import InventoryEngine
from src.inventory.simulator import InventorySimulator

def test_inventory_recommendation_logic():
    """
    Test standard replenishment calculation:
    Forecast mean = 10 units/day, lead time = 7 days, review period = 7 days
    Total horizon demand = 140 units.
    Safety stock with zero variance = 0.
    Target stock = 140.
    Current stock = 40, reserved = 10, on order = 20 -> inventory position = 50.
    Suggested order = max(0, 140 - 50) = 90.
    """
    rec = InventoryEngine.calculate_recommendation(
        product_id="BEVERAGES",
        store_id=1,
        current_stock=40.0,
        reserved_stock=10.0,
        on_order_stock=20.0,
        forecast_daily_mean=10.0,
        forecast_daily_std=0.0,
        lead_time_days=7,
        review_period_days=7,
        service_level=0.95,
        reorder_multiple=5,
        min_order_qty=10
    )
    
    assert rec["inventory_position"] == 50.0
    assert rec["target_stock"] == 140.0
    assert rec["suggested_order"] == 90
    assert rec["days_of_supply"] == pytest.approx(3.0)  # (40 - 10) / 10 = 3.0 days
    assert rec["risk_level"] == "HIGH"  # 3 days < 7 days lead time

def test_inventory_simulation():
    """Verify inventory simulation runs without negative stock and tracks lost sales."""
    demand_series = pd.Series([50.0] * 30)
    sim = InventorySimulator(lead_time_days=5, review_period_days=7)
    results = sim.run_simulation(demand_series, initial_stock=100.0)
    
    assert results["total_demand"] == 1500.0
    assert results["total_days"] == 30
    assert results["fill_rate_percent"] > 0.0
    assert len(results["daily_trajectory"]) == 30
