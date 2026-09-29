"""
Data Schemas and Validation Models for Retail Demand Forecasting.
"""
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field

class StoreSalesRecord(BaseModel):
    id: Optional[int] = None
    date: date
    store_nbr: int = Field(gt=0, description="Store identifier")
    family: str = Field(description="Product family / category")
    sales: float = Field(ge=0.0, description="Daily sales unit volume")
    onpromotion: int = Field(ge=0, description="Number of items on promotion")

class StoreMetadata(BaseModel):
    store_nbr: int
    city: str
    state: str
    type: str
    cluster: int

class OilPriceRecord(BaseModel):
    date: date
    dcoilwtico: Optional[float] = None

class HolidayEventRecord(BaseModel):
    date: date
    type: str
    locale: str
    locale_name: str
    description: str
    transferred: bool

class InventoryRecord(BaseModel):
    product_id: str
    store_id: int
    current_stock: float
    reserved_stock: float = 0.0
    on_order_stock: float = 0.0
    lead_time_days: int = 7
    safety_stock: float = 20.0
    unit_cost: float = 10.0
    selling_price: float = 15.0
    reorder_multiple: int = 5
    min_order_qty: int = 10
    last_updated: date
