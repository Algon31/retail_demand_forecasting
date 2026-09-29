"""
Machine Learning Demand Forecaster:
Supports Random Forest and XGBoost regression with chronological splitting,
multi-step recursive forecasting, and prediction intervals.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import settings
from src.features.pipeline import FEATURE_COLUMNS, add_calendar_features
from src.models.evaluation import calculate_metrics

class RetailDemandForecaster:
    """
    Production-grade demand forecaster for retail store/SKU combinations.
    """
    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None
        self.family_mapping: Dict[str, int] = {}
        self.inverse_family_mapping: Dict[int, str] = {}
        self.feature_names = FEATURE_COLUMNS
        self.residual_std: float = 15.0
        self.group_residuals: Dict[str, float] = {}

    def _init_model(self):
        if self.model_type == "xgboost":
            default_params = {
                "n_estimators": 400,
                "max_depth": 7,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1
            }
            default_params.update(self.params)
            self.model = XGBRegressor(**default_params)
        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 250,
                "max_depth": 14,
                "min_samples_leaf": 3,
                "random_state": 42,
                "n_jobs": -1
            }
            default_params.update(self.params)
            self.model = RandomForestRegressor(**default_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        family_mapping: Optional[Dict[str, int]] = None
    ):
        """Fits the regression model on engineered feature dataframe."""
        self._init_model()
        
        if family_mapping:
            self.family_mapping = family_mapping
        else:
            families = sorted(train_df["family"].unique().tolist())
            self.family_mapping = {f: i for i, f in enumerate(families)}
            
        self.inverse_family_mapping = {v: k for k, v in self.family_mapping.items()}
        
        X_train = train_df[self.feature_names]
        y_train = train_df["sales"].values
        
        if self.model_type == "xgboost" and val_df is not None:
            X_val = val_df[self.feature_names]
            y_val = val_df["sales"].values
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
            
        # Calculate residual std for prediction intervals
        preds_train = self.model.predict(X_train)
        residuals = y_train - preds_train
        self.residual_std = float(np.std(residuals))
        
        # Calculate per-family residual standard deviation
        train_df = train_df.copy()
        train_df["_resid"] = residuals
        for fam, grp in train_df.groupby("family"):
            self.group_residuals[fam] = float(np.std(grp["_resid"]))
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point predictions clipped at 0."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        preds = self.model.predict(X[self.feature_names])
        return np.maximum(0.0, preds)

    def forecast_recursive(
        self,
        historical_df: pd.DataFrame,
        store_nbr: int,
        family: str,
        horizon_days: int = 14,
        scenario_promo_multiplier: float = 1.0,
        scenario_oil_price: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generates recursive multi-step forecasts for a given (store, family).
        Updates lags dynamically using newly predicted sales to prevent future leakage.
        """
        # Filter historical series for this store & family
        series_hist = historical_df[
            (historical_df["store_nbr"] == store_nbr) &
            (historical_df["family"] == family)
        ].sort_values("date").copy()
        
        if len(series_hist) < 30:
            raise ValueError(f"Insufficient historical data for Store {store_nbr}, {family}")
            
        last_date = pd.to_datetime(series_hist["date"].max())
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]
        
        # Keep track of sales history: date -> sales
        sales_history = dict(zip(pd.to_datetime(series_hist["date"]), series_hist["sales"]))
        oil_history = dict(zip(pd.to_datetime(series_hist["date"]), series_hist["oil_price"]))
        last_oil = series_hist["oil_price"].iloc[-1] if "oil_price" in series_hist.columns else 70.0
        
        fam_code = self.family_mapping.get(family, 0)
        group_std = self.group_residuals.get(family, self.residual_std)
        
        forecast_rows = []
        
        for f_date in forecast_dates:
            # 1. Calendar features
            dow = f_date.dayofweek
            is_weekend = int(dow >= 5)
            
            # 2. Extract lagged sales from history (including earlier recursive predictions)
            lag_1 = sales_history.get(f_date - pd.Timedelta(days=1), series_hist["sales"].mean())
            lag_7 = sales_history.get(f_date - pd.Timedelta(days=7), series_hist["sales"].mean())
            lag_14 = sales_history.get(f_date - pd.Timedelta(days=14), series_hist["sales"].mean())
            lag_28 = sales_history.get(f_date - pd.Timedelta(days=28), series_hist["sales"].mean())
            
            # 3. Rolling window means
            sales_7 = [sales_history.get(f_date - pd.Timedelta(days=d), lag_1) for d in range(1, 8)]
            sales_14 = [sales_history.get(f_date - pd.Timedelta(days=d), lag_1) for d in range(1, 15)]
            sales_28 = [sales_history.get(f_date - pd.Timedelta(days=d), lag_1) for d in range(1, 29)]
            
            rolling_mean_7 = float(np.mean(sales_7))
            rolling_mean_14 = float(np.mean(sales_14))
            rolling_mean_28 = float(np.mean(sales_28))
            rolling_std_7 = float(np.std(sales_7))
            rolling_std_28 = float(np.std(sales_28))
            
            # 4. Contextual features
            oil_price = scenario_oil_price if scenario_oil_price is not None else last_oil
            # Promo assumptions (apply scenario promo multiplier)
            baseline_promo = 1 if (dow in [5, 6] or f_date.day in [1, 15]) else 0
            onpromotion = int(baseline_promo * scenario_promo_multiplier)
            
            # Holiday indicator
            is_holiday = 1 if (f_date.month == 12 and f_date.day == 25) or (f_date.month == 1 and f_date.day == 1) else 0
            
            # Feature row
            row_dict = {
                "store_nbr": store_nbr,
                "family_encoded": fam_code,
                "day_of_week": dow,
                "day_of_month": f_date.day,
                "month": f_date.month,
                "quarter": f_date.quarter,
                "week_of_year": int(f_date.isocalendar().week),
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "onpromotion": onpromotion,
                "oil_price": oil_price,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_28": lag_28,
                "rolling_mean_7": rolling_mean_7,
                "rolling_mean_14": rolling_mean_14,
                "rolling_mean_28": rolling_mean_28,
                "rolling_std_7": rolling_std_7,
                "rolling_std_28": rolling_std_28,
            }
            
            feat_df = pd.DataFrame([row_dict])[self.feature_names]
            pred_sales = float(self.predict(feat_df)[0])
            
            # Store in recursive sales history
            sales_history[f_date] = pred_sales
            
            # 80% and 95% uncertainty intervals (Z=1.28 and Z=1.96)
            lower_80 = max(0.0, pred_sales - 1.28 * group_std)
            upper_80 = pred_sales + 1.28 * group_std
            lower_95 = max(0.0, pred_sales - 1.96 * group_std)
            upper_95 = pred_sales + 1.96 * group_std
            
            forecast_rows.append({
                "date": f_date.strftime("%Y-%m-%d"),
                "store_nbr": store_nbr,
                "family": family,
                "predicted_sales": round(pred_sales, 2),
                "lower_80": round(lower_80, 2),
                "upper_80": round(upper_80, 2),
                "lower_95": round(lower_95, 2),
                "upper_95": round(upper_95, 2),
                "onpromotion": onpromotion,
            })
            
        return pd.DataFrame(forecast_rows)

    def save(self, filepath: Path = settings.MODEL_ARTIFACT_PATH):
        """Serializes forecaster state to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model artifact saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path = settings.MODEL_ARTIFACT_PATH) -> "RetailDemandForecaster":
        """Loads forecaster from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        return joblib.load(filepath)
