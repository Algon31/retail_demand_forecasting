"""
Analytics and Catalog routes: stores, products, historical sales trends.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd

from src.database.session import get_db
from src.database.models import Store, DailySale
from api.schemas.models import StoreItem, ProductCatalogResponse

router = APIRouter(tags=["Analytics & Catalog"])

@router.get("/stores", response_model=List[StoreItem])
def list_stores(db: Session = Depends(get_db)):
    """Returns all retail store locations and clusters."""
    stores = db.query(Store).order_by(Store.store_nbr).all()
    return stores

@router.get("/products", response_model=List[str])
def list_products(db: Session = Depends(get_db)):
    """Returns list of product family categories."""
    families = db.query(DailySale.family).distinct().order_by(DailySale.family).all()
    if not families:
        # Fallback to predefined list
        return [
            "BEVERAGES", "BREAD/BAKERY", "CLEANING", "DAIRY", "DELI",
            "GROCERY I", "MEATS", "PERSONAL CARE", "POULTRY", "PRODUCE"
        ]
    return [f[0] for f in families]

@router.get("/catalog", response_model=ProductCatalogResponse)
def get_catalog(db: Session = Depends(get_db)):
    """Returns full catalog of stores and products."""
    stores = db.query(Store).order_by(Store.store_nbr).all()
    families = list_products(db)
    return ProductCatalogResponse(products=families, stores=stores)

@router.get("/history/{store_id}/{product_id}")
def get_historical_sales(
    store_id: int,
    product_id: str,
    days: int = Query(default=60, ge=7, le=365),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Returns historical daily sales for a store and product family."""
    records = (
        db.query(DailySale)
        .filter(DailySale.store_nbr == store_id, DailySale.family == product_id)
        .order_by(DailySale.date.desc())
        .limit(days)
        .all()
    )
    records.reverse()
    return [
        {
            "date": r.date.strftime("%Y-%m-%d"),
            "sales": r.sales,
            "onpromotion": r.onpromotion,
            "is_holiday": r.is_holiday,
            "oil_price": r.oil_price
        }
        for r in records
    ]
