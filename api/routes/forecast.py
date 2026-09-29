"""
Demand Forecasting endpoints.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
from src.config import settings
from src.models.forecaster import RetailDemandForecaster
from api.schemas.models import ForecastRequest, ForecastResponse, DailyForecastPoint

router = APIRouter(tags=["Demand Forecasting"])

# Global cache for forecaster model and features dataset
_forecaster_instance = None
_features_cache = None

def get_forecaster() -> RetailDemandForecaster:
    global _forecaster_instance
    if _forecaster_instance is None:
        if not settings.MODEL_ARTIFACT_PATH.exists():
            raise HTTPException(status_code=503, detail="Forecasting model artifact not found. Please train model first.")
        _forecaster_instance = RetailDemandForecaster.load(settings.MODEL_ARTIFACT_PATH)
    return _forecaster_instance

def get_features_dataset() -> pd.DataFrame:
    global _features_cache
    if _features_cache is None:
        feat_path = settings.FEATURES_DATA_DIR / "features.parquet"
        if not feat_path.exists():
            clean_path = settings.PROCESSED_DATA_DIR / "clean_sales.csv"
            if clean_path.exists():
                _features_cache = pd.read_csv(clean_path)
            else:
                raise HTTPException(status_code=503, detail="Historical dataset not found.")
        else:
            _features_cache = pd.read_parquet(feat_path)
    return _features_cache

@router.post("/forecast", response_model=ForecastResponse)
def generate_forecast(
    req: ForecastRequest,
    forecaster: RetailDemandForecaster = Depends(get_forecaster)
):
    """
    Generates multi-horizon demand predictions using recursive XGBoost forecaster.
    Returns 80% and 95% uncertainty intervals and scenario adjustments.
    """
    features_df = get_features_dataset()
    
    try:
        preds_df = forecaster.forecast_recursive(
            historical_df=features_df,
            store_nbr=req.store_id,
            family=req.product_id,
            horizon_days=req.horizon_days,
            scenario_promo_multiplier=req.scenario_promo_multiplier,
            scenario_oil_price=req.scenario_oil_price
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    points = [DailyForecastPoint(**row) for row in preds_df.to_dict(orient="records")]
    total_sales = float(preds_df["predicted_sales"].sum())
    mean_sales = float(preds_df["predicted_sales"].mean())
    
    return ForecastResponse(
        store_id=req.store_id,
        product_id=req.product_id,
        forecast_horizon=req.horizon_days,
        predictions=points,
        model_version="v1.0-xgboost",
        summary_total_predicted_units=round(total_sales, 1),
        summary_daily_mean=round(mean_sales, 1)
    )

@router.get("/forecast/{store_id}/{product_id}", response_model=ForecastResponse)
def get_default_forecast(store_id: int, product_id: str):
    """Convenience endpoint returning standard 14-day forecast."""
    req = ForecastRequest(store_id=store_id, product_id=product_id, horizon_days=14)
    forecaster = get_forecaster()
    return generate_forecast(req, forecaster)
