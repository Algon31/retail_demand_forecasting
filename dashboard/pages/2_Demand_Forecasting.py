"""
Demand Forecasting Page:
Interactive multi-horizon forecasting with uncertainty intervals and promotional scenario adjustment.
"""
import streamlit as st
import pandas as pd
import requests

from src.config import settings
from src.models.forecaster import RetailDemandForecaster
from dashboard.components.styles import CUSTOM_CSS
from dashboard.components.charts import plot_forecast_with_intervals
from dashboard.components.kpi_cards import render_kpi

st.set_page_config(page_title="Demand Forecasting - Retail Demand AI", page_icon="🔮", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🔮 Demand Forecasting Explorer")
st.markdown("Generate recursive multi-step daily demand forecasts with uncertainty bounds and scenario adjustments.")

# Load historical features
@st.cache_data
def load_data():
    clean_path = settings.PROCESSED_DATA_DIR / "clean_sales.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

hist_df = load_data()

if hist_df.empty:
    st.error("Processed sales dataset not found. Please run the training pipeline first.")
    st.stop()

# Controls row
c1, c2, c3, c4 = st.columns([1, 1.2, 1, 1.2])

with c1:
    stores = sorted(hist_df["store_nbr"].unique().tolist())
    selected_store = st.selectbox("Select Store", stores, index=0)
    
with c2:
    families = sorted(hist_df["family"].unique().tolist())
    selected_family = st.selectbox("Select Product Category", families, index=0)
    
with c3:
    horizon = st.selectbox("Forecast Horizon", [7, 14, 21, 30], index=1)
    
with c4:
    promo_mult = st.slider("Promo Intensity Multiplier", min_value=0.0, max_value=3.0, value=1.0, step=0.25)

# Generate Forecast (via API if running or local forecaster)
with st.spinner("Generating recursive multi-step demand predictions..."):
    forecast_df = pd.DataFrame()
    api_success = False
    
    try:
        payload = {
            "store_id": int(selected_store),
            "product_id": str(selected_family),
            "horizon_days": int(horizon),
            "scenario_promo_multiplier": float(promo_mult)
        }
        res = requests.post(f"{settings.API_URL}/forecast", json=payload, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            forecast_df = pd.DataFrame(data["predictions"])
            api_success = True
    except Exception:
        api_success = False
        
    if not api_success:
        # Fallback to direct model load
        if settings.MODEL_ARTIFACT_PATH.exists():
            forecaster = RetailDemandForecaster.load(settings.MODEL_ARTIFACT_PATH)
            forecast_df = forecaster.forecast_recursive(
                historical_df=hist_df,
                store_nbr=selected_store,
                family=selected_family,
                horizon_days=horizon,
                scenario_promo_multiplier=promo_mult
            )
            
if forecast_df.empty:
    st.error("Failed to generate forecast. Ensure model artifact exists.")
    st.stop()

# Forecast Summary Metrics
total_pred = forecast_df["predicted_sales"].sum()
daily_avg = forecast_df["predicted_sales"].mean()
min_pred = forecast_df["predicted_sales"].min()
max_pred = forecast_df["predicted_sales"].max()

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_kpi(f"{horizon}-Day Total Forecast", f"{total_pred:,.1f} units", "Projected Demand", "#4f46e5")
with m2:
    render_kpi("Daily Expected Average", f"{daily_avg:,.1f} units/day", "Baseline Demand", "#0ea5e9")
with m3:
    render_kpi("Projected Peak Day", f"{max_pred:,.1f} units", "Maximum Day", "#10b981")
with m4:
    render_kpi("Scenario Multiplier", f"{promo_mult:.2f}x", "Promotional Lift", "#8b5cf6")

st.markdown("---")

# Chart: Actual History + Forecast Fan Chart
recent_history = hist_df[
    (hist_df["store_nbr"] == selected_store) &
    (hist_df["family"] == selected_family)
].sort_values("date").tail(45)

fig_forecast = plot_forecast_with_intervals(
    hist_df=recent_history,
    forecast_df=forecast_df,
    title=f"Store {selected_store} — {selected_family} ({horizon}-Day Demand Forecast & Uncertainty Intervals)"
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Data Table & Export
st.subheader("Daily Prediction Breakdown")
cols_to_show = ["date", "predicted_sales", "lower_80", "upper_80", "lower_95", "upper_95", "onpromotion"]

st.dataframe(
    forecast_df[cols_to_show].style.format({
        "predicted_sales": "{:,.1f}",
        "lower_80": "{:,.1f}",
        "upper_80": "{:,.1f}",
        "lower_95": "{:,.1f}",
        "upper_95": "{:,.1f}",
    }),
    use_container_width=True
)

csv_data = forecast_df[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Export Forecast to CSV",
    data=csv_data,
    file_name=f"forecast_store_{selected_store}_{selected_family}_{horizon}d.csv",
    mime="text/csv"
)
