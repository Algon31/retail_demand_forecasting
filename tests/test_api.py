"""
Integration Tests for FastAPI Endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_stores_and_products_catalog():
    r_stores = client.get("/stores")
    assert r_stores.status_code == 200
    assert len(r_stores.json()) > 0
    
    r_prods = client.get("/products")
    assert r_prods.status_code == 200
    assert "BEVERAGES" in r_prods.json()

def test_forecast_endpoint():
    payload = {
        "store_id": 1,
        "product_id": "BEVERAGES",
        "horizon_days": 14,
        "scenario_promo_multiplier": 1.0
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 14
    assert data["summary_total_predicted_units"] > 0
    assert data["predictions"][0]["lower_80"] <= data["predictions"][0]["predicted_sales"]

def test_inventory_risks_endpoint():
    response = client.get("/inventory/risks?service_level=0.95")
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated_skus"] > 0
    assert "critical_stockouts" in data

def test_simulation_endpoint():
    payload = {
        "store_id": 1,
        "product_id": "BEVERAGES",
        "initial_stock": 300.0,
        "lead_time_days": 7,
        "review_period_days": 7,
        "service_level": 0.95,
        "policy_type": "dynamic_ml"
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fill_rate_percent"] > 0
    assert len(data["daily_trajectory"]) > 0
