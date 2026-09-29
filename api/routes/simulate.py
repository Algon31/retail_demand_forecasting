"""
Inventory and Demand Scenario Simulation endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd

from src.database.session import get_db
from src.database.models import DailySale, InventoryItem
from src.inventory.simulator import InventorySimulator
from api.schemas.models import SimulationRequest, SimulationResponse

router = APIRouter(tags=["Scenario Simulation"])

@router.post("/simulate", response_model=SimulationResponse)
def simulate_policy(
    req: SimulationRequest,
    db: Session = Depends(get_db)
):
    """
    Simulates inventory replenishment trajectory over historical daily sales.
    """
    sales_records = (
        db.query(DailySale)
        .filter(DailySale.store_nbr == req.store_id, DailySale.family == req.product_id)
        .order_by(DailySale.date.asc())
        .all()
    )
    if not sales_records or len(sales_records) < 30:
        raise HTTPException(status_code=400, detail="Insufficient sales history for simulation (minimum 30 days required).")
        
    demand_series = pd.Series([s.sales for s in sales_records])
    
    # Retrieve cost economics if available
    inv_item = (
        db.query(InventoryItem)
        .filter(InventoryItem.store_id == req.store_id, InventoryItem.product_id == req.product_id)
        .first()
    )
    unit_cost = inv_item.unit_cost if inv_item else 10.0
    selling_price = inv_item.selling_price if inv_item else 15.0
    reorder_multiple = inv_item.reorder_multiple if inv_item else 5
    min_order_qty = inv_item.min_order_qty if inv_item else 10
    
    sim = InventorySimulator(
        lead_time_days=req.lead_time_days,
        review_period_days=req.review_period_days,
        service_level=req.service_level,
        unit_cost=unit_cost,
        selling_price=selling_price,
        reorder_multiple=reorder_multiple,
        min_order_qty=min_order_qty
    )
    
    results = sim.run_simulation(
        demand_series=demand_series,
        initial_stock=req.initial_stock,
        policy_type=req.policy_type
    )
    
    return SimulationResponse(**results)
