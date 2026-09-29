"""
Model Explainability Engine using SHAP (SHapley Additive exPlanations).
Provides global feature importances and local individual prediction contributions.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import shap
from src.features.pipeline import FEATURE_COLUMNS

class ModelExplainer:
    """
    Computes SHAP values for tree-based forecasting models.
    """
    def __init__(self, model):
        self.model = model
        self.feature_names = FEATURE_COLUMNS
        # Use TreeExplainer for XGBoost and Random Forest
        self.explainer = shap.TreeExplainer(model)
        
    def explain_global(self, X_sample: pd.DataFrame, max_features: int = 15) -> List[Dict[str, Any]]:
        """
        Calculates global feature importance ranking based on mean(|SHAP value|).
        """
        X_eval = X_sample[self.feature_names]
        shap_vals = self.explainer.shap_values(X_eval)
        
        # If list (multiclass), take first output, but for regression shap_vals is ndarray
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        
        feature_importance = []
        for name, score in zip(self.feature_names, mean_abs_shap):
            feature_importance.append({
                "feature": name,
                "importance": round(float(score), 4)
            })
            
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        return feature_importance[:max_features]

    def explain_local(self, single_row_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Explains an individual prediction: baseline value, prediction,
        and per-feature positive/negative impact.
        """
        X_row = single_row_df[self.feature_names].iloc[0:1]
        shap_vals = self.explainer.shap_values(X_row)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        base_value = float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0
        row_vals = X_row.iloc[0].to_dict()
        
        contributions = []
        for feat_name, shap_val in zip(self.feature_names, shap_vals[0]):
            contributions.append({
                "feature": feat_name,
                "feature_value": round(float(row_vals[feat_name]), 2),
                "shap_value": round(float(shap_val), 3),
                "impact": "increases_demand" if shap_val > 0 else "decreases_demand"
            })
            
        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        predicted_val = base_value + float(np.sum(shap_vals[0]))
        
        return {
            "base_value": round(base_value, 2),
            "predicted_value": round(max(0.0, predicted_val), 2),
            "top_drivers": contributions[:10]
        }
