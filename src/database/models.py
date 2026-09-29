"""
SQLAlchemy Database Models for Retail Demand and Inventory Platform.
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Store(Base):
    __tablename__ = "stores"
    
    store_nbr = Column(Integer, primary_key=True, index=True)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    type = Column(String(10), nullable=False)
    cluster = Column(Integer, nullable=False)

class DailySale(Base):
    __tablename__ = "daily_sales"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    store_nbr = Column(Integer, nullable=False, index=True)
    family = Column(String(100), nullable=False, index=True)
    sales = Column(Float, nullable=False)
    onpromotion = Column(Integer, default=0)
    oil_price = Column(Float, nullable=True)
    is_holiday = Column(Integer, default=0)
    
    __table_args__ = (
        Index("idx_store_family_date", "store_nbr", "family", "date"),
    )

class InventoryItem(Base):
    __tablename__ = "inventory"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(100), nullable=False, index=True)
    store_id = Column(Integer, nullable=False, index=True)
    current_stock = Column(Float, nullable=False, default=0.0)
    reserved_stock = Column(Float, default=0.0)
    on_order_stock = Column(Float, default=0.0)
    lead_time_days = Column(Integer, default=7)
    safety_stock = Column(Float, default=20.0)
    unit_cost = Column(Float, default=10.0)
    selling_price = Column(Float, default=15.0)
    reorder_multiple = Column(Integer, default=5)
    min_order_qty = Column(Integer, default=10)
    last_updated = Column(Date, default=date.today)
    
    __table_args__ = (
        Index("idx_inv_product_store", "product_id", "store_id", unique=True),
    )

class ForecastRecord(Base):
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    store_nbr = Column(Integer, nullable=False, index=True)
    family = Column(String(100), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    predicted_sales = Column(Float, nullable=False)
    lower_80 = Column(Float, nullable=True)
    upper_80 = Column(Float, nullable=True)
    lower_95 = Column(Float, nullable=True)
    upper_95 = Column(Float, nullable=True)
    model_version = Column(String(50), default="v1.0")

class ReplenishmentRecommendation(Base):
    __tablename__ = "replenishment_recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    store_id = Column(Integer, nullable=False, index=True)
    product_id = Column(String(100), nullable=False, index=True)
    current_stock = Column(Float, nullable=False)
    inventory_position = Column(Float, nullable=False)
    safety_stock = Column(Float, nullable=False)
    reorder_point = Column(Float, nullable=False)
    suggested_order = Column(Integer, nullable=False)
    risk_level = Column(String(50), nullable=False)
    order_value = Column(Float, default=0.0)
