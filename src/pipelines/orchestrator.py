"""
Pipeline Orchestrator:
Coordinates the end-to-end data processing, feature engineering,
model training, evaluation, and inventory optimization workflows.
"""
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.config import settings
from src.data.loader import load_raw_data, validate_raw_data
from src.data.cleaner import clean_and_consolidate
from src.features.pipeline import build_features_dataset, save_features
from src.models.baselines import BaselineForecaster
from src.models.forecaster import RetailDemandForecaster
from src.models.evaluation import calculate_metrics, evaluate_by_group
from src.explainability.shap_explainer import ModelExplainer
from src.inventory.engine import InventoryEngine
from src.database.session import seed_database_from_csv

def run_end_to_end_pipeline() -> Dict[str, Any]:
    """
    Executes all phases:
    1. Ingestion & Quality Checks
    2. Data Cleaning & Merges
    3. Feature Engineering
    4. Model Training & Evaluation (Baselines vs XGBoost)
    5. SHAP Feature Importance
    6. Inventory Optimization Evaluation
    7. Database Seeding
    """
    print("=" * 60)
    print("RETAIL DEMAND FORECASTING & INVENTORY INTELLIGENCE PIPELINE")
    print("=" * 60)
    
    # 1. Ingestion
    print("\n--- [Phase 1] Data Ingestion & Quality Validation ---")
    raw_dfs = load_raw_data(auto_generate=True)
    quality_checks = validate_raw_data(raw_dfs)
    print(f"Data Quality Checks: {quality_checks}")
    
    # 2. Cleaning & Consolidation
    print("\n--- [Phase 1.2] Cleaning & Merges ---")
    clean_df = clean_and_consolidate(raw_dfs)
    
    # 3. Feature Engineering
    print("\n--- [Phase 2] Feature Engineering ---")
    features_df, family_mapping = build_features_dataset(clean_df)
    save_features(features_df)
    
    # 4. Model Training & Evaluation
    print("\n--- [Phase 3] Model Training & Chronological Evaluation ---")
    # Chronological Split (e.g., last 60 days as test, prior 60 days as validation, rest as train)
    features_df["date"] = pd.to_datetime(features_df["date"])
    max_date = features_df["date"].max()
    test_cutoff = max_date - pd.Timedelta(days=60)
    val_cutoff = test_cutoff - pd.Timedelta(days=60)
    
    train_df = features_df[features_df["date"] < val_cutoff].copy()
    val_df = features_df[(features_df["date"] >= val_cutoff) & (features_df["date"] < test_cutoff)].copy()
    test_df = features_df[features_df["date"] >= test_cutoff].copy()
    
    print(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    # 4.1 Evaluate Baselines on Test Set
    baseline_metrics = BaselineForecaster.evaluate_baselines(test_df)
    print("\nBaseline Results on Test Period:")
    for b_name, b_m in baseline_metrics.items():
        print(f"  {b_name}: WAPE={b_m['WAPE']}%, MAE={b_m['MAE']}, Bias={b_m['Bias']}%")
        
    # 4.2 Train XGBoost Forecaster
    print("\nFitting XGBoost Forecaster...")
    forecaster = RetailDemandForecaster(model_type="xgboost")
    forecaster.fit(train_df, val_df=val_df, family_mapping=family_mapping)
    
    # 4.3 Predict on Test Set
    test_preds = forecaster.predict(test_df)
    test_df["predicted"] = test_preds
    test_df["actual"] = test_df["sales"].values
    
    ml_metrics = calculate_metrics(test_df["actual"].values, test_preds)
    print("\nXGBoost Results on Test Period:")
    print(f"  WAPE: {ml_metrics['WAPE']}%")
    print(f"  MAE:  {ml_metrics['MAE']}")
    print(f"  RMSE: {ml_metrics['RMSE']}")
    print(f"  Bias: {ml_metrics['Bias']}%")
    
    # Save Model Artifact
    forecaster.save(settings.MODEL_ARTIFACT_PATH)
    
    # 4.4 Category-level Evaluation
    group_eval_df = evaluate_by_group(test_df, actual_col="actual", pred_col="predicted", group_col="family")
    print("\nTop 5 Categories by Accuracy (Lowest WAPE):")
    print(group_eval_df[["family", "WAPE", "MAE", "Bias", "total_actual_volume"]].head(5).to_string(index=False))
    
    # 5. Explainable AI with SHAP
    print("\n--- [Phase 4] SHAP Model Explainability ---")
    explainer = ModelExplainer(forecaster.model)
    sample_eval = test_df.sample(n=min(300, len(test_df)), random_state=42)
    global_importance = explainer.explain_global(sample_eval, max_features=12)
    print("Top Feature Drivers (Global SHAP):")
    for feat in global_importance[:6]:
        print(f"  - {feat['feature']}: {feat['importance']}")
        
    # 6. Inventory Intelligence Evaluation
    print("\n--- [Phase 5] Inventory Intelligence & Replenishment ---")
    inv_df = raw_dfs["inventory"]
    # Compute 14-day daily forecast stats per (store, family)
    forecast_stats = (
        test_df.groupby(["store_nbr", "family"])["predicted"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "forecast_mean", "std": "forecast_std"})
    )
    
    recommendations_df = InventoryEngine.evaluate_batch(
        inventory_df=inv_df,
        forecast_summary_df=forecast_stats,
        service_level=0.95
    )
    
    risk_counts = recommendations_df["risk_level"].value_counts().to_dict()
    total_suggested_units = int(recommendations_df["suggested_order"].sum())
    total_order_investment = float(recommendations_df["order_value"].sum())
    
    print(f"Inventory Risk Profile: {risk_counts}")
    print(f"Total Recommended Orders: {total_suggested_units:,} units (${total_order_investment:,.2f})")
    
    # 7. Seed Database
    print("\n--- [Phase 6] Database Seeding ---")
    seed_database_from_csv()
    
    # Save metrics and summary
    summary_report = {
        "dataset_records": len(clean_df),
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "baseline_comparison": baseline_metrics,
        "ml_model_metrics": ml_metrics,
        "category_metrics": group_eval_df.to_dict(orient="records"),
        "global_shap_importance": global_importance,
        "inventory_risk_summary": risk_counts,
        "total_replenishment_units": total_suggested_units,
        "total_order_investment": round(total_order_investment, 2),
    }
    
    with open(settings.METRICS_PATH, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nPipeline summary report saved to {settings.METRICS_PATH}")
    
    return summary_report

if __name__ == "__main__":
    run_end_to_end_pipeline()
