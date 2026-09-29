"""
Inventory Intelligence & Replenishment Endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd

from src.database.session import get_db
from src.database.models import InventoryItem
from src.inventory.engine import InventoryEngine
from api.routes.forecast import get_forecaster, get_features_dataset
from api.schemas.models import (
    ReplenishmentRequest,
    ReplenishmentRecommendationItem,
    InventoryRisksResponse
)

router = APIRouter(tags=["Inventory Intelligence"])

@router.post("/inventory/recommendations", response_model=List[ReplenishmentRecommendationItem])
def get_recommendations(
    req: ReplenishmentRequest,
    db: Session = Depends(get_db)
):
    """
    Computes replenishment recommendations and reorder quantities across inventory items.
    """
    query = db.query(InventoryItem)
    if req.store_id:
        query = query.filter(InventoryItem.store_id == req.store_id)
    if req.product_id:
        query = query.filter(InventoryItem.product_id == req.product_id)
        
    items = query.all()
    if not items:
        return []
        
    features_df = get_features_dataset()
    forecaster = get_forecaster()
    
    results = []
    for item in items:
        lead_time = req.lead_time_days or item.lead_time_days
        review_period = req.review_period_days or 7
        
        # Project 14 days ahead to compute daily mean and std
        try:
            preds_df = forecaster.forecast_recursive(
                historical_df=features_df,
                store_nbr=item.store_id,
                family=item.product_id,
                horizon_days=lead_time + review_period
            )
            d_mean = float(preds_df["predicted_sales"].mean())
            d_std = float(preds_df["predicted_sales"].std())
        except Exception:
            d_mean = 120.0
            d_std = 25.0
            
        rec = InventoryEngine.calculate_recommendation(
            product_id=item.product_id,
            store_id=item.store_id,
            current_stock=item.current_stock,
            reserved_stock=item.reserved_stock,
            on_order_stock=item.on_order_stock,
            forecast_daily_mean=d_mean,
            forecast_daily_std=d_std,
            lead_time_days=lead_time,
            review_period_days=review_period,
            service_level=req.service_level,
            unit_cost=item.unit_cost,
            reorder_multiple=item.reorder_multiple,
            min_order_qty=item.min_order_qty
        )
        results.append(ReplenishmentRecommendationItem(**rec))
        
    # Sort with highest risk first
    risk_priority = {"CRITICAL_STOCKOUT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "OVERSTOCKED": 4}
    results.sort(key=lambda x: (risk_priority.get(x.risk_level, 5), -x.suggested_order))
    return results

@router.get("/inventory/risks", response_model=InventoryRisksResponse)
def get_inventory_risks(
    service_level: float = 0.95,
    db: Session = Depends(get_db)
):
    """
    Returns inventory risk breakdown across the company, highlighting stockouts and order requirements.
    """
    req = ReplenishmentRequest(service_level=service_level)
    items = get_recommendations(req, db)
    
    crit = sum(1 for i in items if i.risk_level == "CRITICAL_STOCKOUT")
    high = sum(1 for i in items if i.risk_level == "HIGH")
    med = sum(1 for i in items if i.risk_level == "MEDIUM")
    low = sum(1 for i in items if i.risk_level == "LOW")
    over = sum(1 for i in items if i.risk_level == "OVERSTOCKED")
    
    total_order_units = sum(i.suggested_order for i in items)
    total_capital = sum(i.order_value for i in items)
    
    return InventoryRisksResponse(
        total_evaluated_skus=len(items),
        critical_stockouts=crit,
        high_risks=high,
        medium_risks=med,
        low_risks=low,
        overstocked=over,
        total_order_units_needed=total_order_units,
        total_capital_required=round(total_capital, 2),
        items=items
    )
