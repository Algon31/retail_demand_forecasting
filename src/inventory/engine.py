"""
Inventory Optimization Engine:
Calculates safety stocks, reorder points (ROP), inventory positions,
risk classifications, and replenishment recommendations.
"""
from typing import Dict, List, Optional, Any
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

class InventoryEngine:
    """
    Decision-support engine for retail inventory management.
    """
    
    @staticmethod
    def get_z_score(service_level: float = 0.95) -> float:
        """Computes inverse standard normal multiplier z for target cycle service level."""
        service_level = max(0.50, min(0.999, service_level))
        return float(norm.ppf(service_level))

    @classmethod
    def calculate_safety_stock(
        cls,
        daily_demand_std: float,
        lead_time_days: int,
        service_level: float = 0.95
    ) -> float:
        """
        Calculates buffer safety stock:
        SS = z * sigma_d * sqrt(L)
        """
        z = cls.get_z_score(service_level)
        ss = z * daily_demand_std * math.sqrt(max(1, lead_time_days))
        return round(float(ss), 1)

    @classmethod
    def calculate_recommendation(
        cls,
        product_id: str,
        store_id: int,
        current_stock: float,
        reserved_stock: float,
        on_order_stock: float,
        forecast_daily_mean: float,
        forecast_daily_std: float,
        lead_time_days: int = 7,
        review_period_days: int = 7,
        service_level: float = 0.95,
        unit_cost: float = 10.0,
        reorder_multiple: int = 5,
        min_order_qty: int = 10
    ) -> Dict[str, Any]:
        """
        Computes comprehensive replenishment analysis for an individual item.
        """
        # 1. Inventory Position
        inventory_position = current_stock - reserved_stock + on_order_stock
        
        # 2. Safety stock
        safety_stock = cls.calculate_safety_stock(forecast_daily_std, lead_time_days, service_level)
        
        # 3. Lead time demand and ROP
        lead_time_demand = forecast_daily_mean * lead_time_days
        reorder_point = round(lead_time_demand + safety_stock, 1)
        
        # 4. Target Stock for periodic review (L + R)
        total_protection_days = lead_time_days + review_period_days
        protection_demand = forecast_daily_mean * total_protection_days
        target_stock = round(protection_demand + safety_stock, 1)
        
        # 5. Raw suggested order
        raw_order = max(0.0, target_stock - inventory_position)
        
        # 6. Apply supplier constraints (reorder multiples and MOQ)
        suggested_order = 0
        if raw_order > 0:
            qty = max(float(min_order_qty), raw_order)
            if reorder_multiple > 1:
                qty = math.ceil(qty / reorder_multiple) * reorder_multiple
            suggested_order = int(qty)
            
        # 7. Days of Supply (DoS)
        effective_stock = max(0.0, current_stock - reserved_stock)
        days_of_supply = round(effective_stock / forecast_daily_mean, 1) if forecast_daily_mean > 0 else 999.0
        
        # 8. Stockout Risk Classification
        if effective_stock <= 0:
            risk_level = "CRITICAL_STOCKOUT"
        elif days_of_supply < lead_time_days:
            risk_level = "HIGH"
        elif inventory_position < reorder_point:
            risk_level = "MEDIUM"
        elif days_of_supply > 45.0:
            risk_level = "OVERSTOCKED"
        else:
            risk_level = "LOW"
            
        estimated_order_cost = round(suggested_order * unit_cost, 2)
        
        return {
            "product_id": product_id,
            "store_id": store_id,
            "current_stock": round(current_stock, 1),
            "reserved_stock": round(reserved_stock, 1),
            "on_order_stock": round(on_order_stock, 1),
            "inventory_position": round(inventory_position, 1),
            "forecast_daily_mean": round(forecast_daily_mean, 2),
            "lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "lead_time_demand": round(lead_time_demand, 1),
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "target_stock": target_stock,
            "days_of_supply": days_of_supply,
            "suggested_order": suggested_order,
            "unit_cost": unit_cost,
            "order_value": estimated_order_cost,
            "risk_level": risk_level,
        }

    @classmethod
    def evaluate_batch(
        cls,
        inventory_df: pd.DataFrame,
        forecast_summary_df: pd.DataFrame,
        service_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Evaluates replenishment recommendations across an entire inventory table.
        """
        merged = inventory_df.merge(
            forecast_summary_df,
            left_on=["store_id", "product_id"],
            right_on=["store_nbr", "family"],
            how="left"
        )
        
        results = []
        for _, row in merged.iterrows():
            daily_mean = row.get("forecast_mean", 100.0)
            daily_std = row.get("forecast_std", daily_mean * 0.25)
            
            rec = cls.calculate_recommendation(
                product_id=str(row["product_id"]),
                store_id=int(row["store_id"]),
                current_stock=float(row.get("current_stock", 0.0)),
                reserved_stock=float(row.get("reserved_stock", 0.0)),
                on_order_stock=float(row.get("on_order_stock", 0.0)),
                forecast_daily_mean=float(daily_mean),
                forecast_daily_std=float(daily_std),
                lead_time_days=int(row.get("lead_time_days", 7)),
                review_period_days=int(row.get("review_period_days", 7)),
                service_level=service_level,
                unit_cost=float(row.get("unit_cost", 10.0)),
                reorder_multiple=int(row.get("reorder_multiple", 5)),
                min_order_qty=int(row.get("min_order_qty", 10))
            )
            results.append(rec)
            
        return pd.DataFrame(results)
