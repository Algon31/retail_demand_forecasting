"""
Pydantic Request and Response Schemas for the FastAPI service.
"""
from typing import List, Optional, Dict, Any
from datetime import date
from pydantic import BaseModel, Field

# Health Schema
class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    model_loaded: bool

# Store & Product Catalog
class StoreItem(BaseModel):
    store_nbr: int
    city: str
    state: str
    type: str
    cluster: int

class ProductCatalogResponse(BaseModel):
    products: List[str]
    stores: List[StoreItem]

# Forecast Schemas
class ForecastRequest(BaseModel):
    store_id: int = Field(..., gt=0, description="Store number")
    product_id: str = Field(..., min_length=1, description="Product family category")
    horizon_days: int = Field(default=14, ge=1, le=90, description="Forecast horizon in days")
    scenario_promo_multiplier: float = Field(default=1.0, ge=0.0, le=5.0, description="Promotional intensity multiplier")
    scenario_oil_price: Optional[float] = Field(default=None, description="Simulated oil price")

class DailyForecastPoint(BaseModel):
    date: str
    store_nbr: int
    family: str
    predicted_sales: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    onpromotion: int

class ForecastResponse(BaseModel):
    store_id: int
    product_id: str
    forecast_horizon: int
    predictions: List[DailyForecastPoint]
    model_version: str = "v1.0"
    summary_total_predicted_units: float
    summary_daily_mean: float

# Inventory Schemas
class ReplenishmentRequest(BaseModel):
    store_id: Optional[int] = None
    product_id: Optional[str] = None
    service_level: float = Field(default=0.95, ge=0.50, le=0.999)
    lead_time_days: Optional[int] = Field(default=None, ge=1, le=60)
    review_period_days: Optional[int] = Field(default=None, ge=1, le=60)

class ReplenishmentRecommendationItem(BaseModel):
    product_id: str
    store_id: int
    current_stock: float
    reserved_stock: float
    on_order_stock: float
    inventory_position: float
    forecast_daily_mean: float
    lead_time_days: int
    safety_stock: float
    reorder_point: float
    target_stock: float
    days_of_supply: float
    suggested_order: int
    unit_cost: float
    order_value: float
    risk_level: str

class InventoryRisksResponse(BaseModel):
    total_evaluated_skus: int
    critical_stockouts: int
    high_risks: int
    medium_risks: int
    low_risks: int
    overstocked: int
    total_order_units_needed: int
    total_capital_required: float
    items: List[ReplenishmentRecommendationItem]

# Explainability Schemas
class ExplainRequest(BaseModel):
    store_id: int
    product_id: str
    date_str: Optional[str] = None

class FeatureImpact(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    impact: str

class ExplainResponse(BaseModel):
    store_id: int
    product_id: str
    base_value: float
    predicted_value: float
    top_drivers: List[FeatureImpact]

# Simulation Schemas
class SimulationRequest(BaseModel):
    store_id: int = 1
    product_id: str = "BEVERAGES"
    initial_stock: float = 250.0
    lead_time_days: int = 7
    review_period_days: int = 7
    service_level: float = 0.95
    policy_type: str = Field(default="dynamic_ml", pattern="^(dynamic_ml|static_heuristic)$")

class SimulationResponse(BaseModel):
    policy: str
    total_days: int
    total_demand: float
    total_fulfilled: float
    total_lost_sales: float
    fill_rate_percent: float
    stockout_days: int
    stockout_rate_percent: float
    average_inventory: float
    total_holding_cost: float
    total_order_cost: float
    lost_sales_revenue: float
    orders_count: int
    daily_trajectory: List[Dict[str, Any]]
