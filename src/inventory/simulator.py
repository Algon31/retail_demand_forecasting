"""
Inventory Policy Simulator:
Backtests replenishment strategies over historical or projected daily demand.
Tracks stockouts, lost sales, inventory holding costs, and order frequencies.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.inventory.engine import InventoryEngine

class InventorySimulator:
    """
    Simulates inventory replenishment dynamics day-by-day.
    """
    def __init__(
        self,
        lead_time_days: int = 7,
        review_period_days: int = 7,
        service_level: float = 0.95,
        holding_cost_annual_rate: float = 0.20,
        order_cost_fixed: float = 50.0,
        unit_cost: float = 10.0,
        selling_price: float = 15.0,
        reorder_multiple: int = 5,
        min_order_qty: int = 10
    ):
        self.lead_time_days = lead_time_days
        self.review_period_days = review_period_days
        self.service_level = service_level
        self.holding_cost_daily_rate = (holding_cost_annual_rate / 365.0) * unit_cost
        self.order_cost_fixed = order_cost_fixed
        self.unit_cost = unit_cost
        self.selling_price = selling_price
        self.reorder_multiple = reorder_multiple
        self.min_order_qty = min_order_qty

    def run_simulation(
        self,
        demand_series: pd.Series,
        initial_stock: float = 200.0,
        policy_type: str = "dynamic_ml"
    ) -> Dict[str, Any]:
        """
        Runs daily time-step inventory simulation.
        
        policy_type:
          - "dynamic_ml": Reorders based on rolling forecast and dynamic safety stock.
          - "static_heuristic": Reorders when stock < fixed static threshold.
        """
        demands = demand_series.values
        n_days = len(demands)
        
        on_hand = float(initial_stock)
        on_order_pipeline = []  # list of (arrival_day, quantity)
        
        daily_records = []
        
        total_demand = 0.0
        total_fulfilled = 0.0
        total_lost_sales = 0.0
        stockout_days = 0
        total_holding_cost = 0.0
        total_order_cost = 0.0
        orders_placed_count = 0
        
        # Historical window for standard deviation
        recent_demands = list(demands[:14])
        
        for day in range(n_days):
            daily_demand = float(demands[day])
            total_demand += daily_demand
            
            # 1. Receive incoming purchase orders scheduled for today
            arrived_qty = sum(qty for arr_day, qty in on_order_pipeline if arr_day == day)
            on_hand += arrived_qty
            on_order_pipeline = [(arr_day, qty) for arr_day, qty in on_order_pipeline if arr_day > day]
            
            # 2. Demand fulfillment
            sales_fulfilled = min(on_hand, daily_demand)
            unmet_demand = daily_demand - sales_fulfilled
            
            if unmet_demand > 0:
                stockout_days += 1
                total_lost_sales += unmet_demand
                
            on_hand -= sales_fulfilled
            total_fulfilled += sales_fulfilled
            
            # 3. Holding cost for remaining inventory
            daily_holding = on_hand * self.holding_cost_daily_rate
            total_holding_cost += daily_holding
            
            # 4. Replenishment Review Check
            order_qty_placed = 0
            if day % self.review_period_days == 0:
                recent_demands.append(daily_demand)
                d_mean = float(np.mean(recent_demands[-14:]))
                d_std = float(np.std(recent_demands[-14:])) if len(recent_demands) > 1 else d_mean * 0.25
                
                pipeline_qty = sum(qty for _, qty in on_order_pipeline)
                inventory_position = on_hand + pipeline_qty
                
                if policy_type == "dynamic_ml":
                    # Dynamic safety stock and target level
                    ss = InventoryEngine.calculate_safety_stock(d_std, self.lead_time_days, self.service_level)
                    rop = (d_mean * self.lead_time_days) + ss
                    target_stock = (d_mean * (self.lead_time_days + self.review_period_days)) + ss
                    
                    if inventory_position < rop:
                        raw_qty = max(0.0, target_stock - inventory_position)
                        qty = max(float(self.min_order_qty), raw_qty)
                        if self.reorder_multiple > 1:
                            qty = int(np.ceil(qty / self.reorder_multiple) * self.reorder_multiple)
                        order_qty_placed = int(qty)
                else:
                    # Static baseline policy (simple fixed threshold)
                    static_rop = d_mean * self.lead_time_days * 1.2
                    if inventory_position < static_rop:
                        order_qty_placed = int(max(float(self.min_order_qty), d_mean * 14))
                        
                if order_qty_placed > 0:
                    on_order_pipeline.append((day + self.lead_time_days, order_qty_placed))
                    total_order_cost += self.order_cost_fixed
                    orders_placed_count += 1
                    
            daily_records.append({
                "day": day,
                "demand": round(daily_demand, 1),
                "fulfilled": round(sales_fulfilled, 1),
                "lost_sales": round(unmet_demand, 1),
                "ending_on_hand": round(on_hand, 1),
                "arrived_orders": round(arrived_qty, 1),
                "order_placed": order_qty_placed
            })
            
        fill_rate = (total_fulfilled / total_demand * 100.0) if total_demand > 0 else 100.0
        avg_inventory = float(np.mean([r["ending_on_hand"] for r in daily_records]))
        lost_sales_revenue = total_lost_sales * self.selling_price
        
        return {
            "policy": policy_type,
            "total_days": n_days,
            "total_demand": round(total_demand, 1),
            "total_fulfilled": round(total_fulfilled, 1),
            "total_lost_sales": round(total_lost_sales, 1),
            "fill_rate_percent": round(fill_rate, 2),
            "stockout_days": stockout_days,
            "stockout_rate_percent": round((stockout_days / n_days) * 100.0, 2),
            "average_inventory": round(avg_inventory, 1),
            "total_holding_cost": round(total_holding_cost, 2),
            "total_order_cost": round(total_order_cost, 2),
            "lost_sales_revenue": round(lost_sales_revenue, 2),
            "orders_count": orders_placed_count,
            "daily_trajectory": daily_records
        }
