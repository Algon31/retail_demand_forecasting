"""
Explainable AI endpoints with SHAP.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import pandas as pd

from src.explainability.shap_explainer import ModelExplainer
from api.routes.forecast import get_forecaster, get_features_dataset
from api.schemas.models import ExplainRequest, ExplainResponse, FeatureImpact

router = APIRouter(tags=["Explainability"])

_explainer_cache = None

def get_explainer() -> ModelExplainer:
    global _explainer_cache
    if _explainer_cache is None:
        forecaster = get_forecaster()
        _explainer_cache = ModelExplainer(forecaster.model)
    return _explainer_cache

@router.get("/explain/global")
def get_global_importance(
    sample_size: int = 200,
    explainer: ModelExplainer = Depends(get_explainer)
) -> List[Dict[str, Any]]:
    """
    Returns global feature importance ranked by mean absolute SHAP value.
    """
    features_df = get_features_dataset()
    sample_df = features_df.sample(n=min(sample_size, len(features_df)), random_state=42)
    return explainer.explain_global(sample_df)

@router.post("/explain", response_model=ExplainResponse)
def explain_prediction(
    req: ExplainRequest,
    explainer: ModelExplainer = Depends(get_explainer)
):
    """
    Returns local SHAP feature impact breakdown for a specific store and product.
    """
    features_df = get_features_dataset()
    filtered = features_df[
        (features_df["store_nbr"] == req.store_id) &
        (features_df["family"] == req.product_id)
    ]
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No historical records found for requested combination.")
        
    latest_row = filtered.tail(1)
    exp = explainer.explain_local(latest_row)
    
    drivers = [FeatureImpact(**d) for d in exp["top_drivers"]]
    
    return ExplainResponse(
        store_id=req.store_id,
        product_id=req.product_id,
        base_value=exp["base_value"],
        predicted_value=exp["predicted_value"],
        top_drivers=drivers
    )
