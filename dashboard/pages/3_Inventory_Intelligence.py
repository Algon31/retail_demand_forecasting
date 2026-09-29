"""
Inventory Intelligence & Replenishment Optimization Page:
Stockout risk alerts, dynamic reorder points (ROP), safety stock tuning, and purchase orders.
"""
import streamlit as st
import pandas as pd
import requests

from src.config import settings
from src.inventory.engine import InventoryEngine
from dashboard.components.styles import CUSTOM_CSS
from dashboard.components.kpi_cards import render_kpi

st.set_page_config(page_title="Inventory Intelligence - Retail Demand AI", page_icon="📦", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📦 Inventory Intelligence & Replenishment")
st.markdown("Automated safety stock optimization, reorder points, stockout risk alerts, and order recommendations.")

# Policy Configuration Controls
with st.expander("⚙️ Replenishment Policy & Service Level Settings", expanded=False):
    p1, p2, p3 = st.columns(3)
    with p1:
        service_level = st.select_slider(
            "Target Cycle Service Level (CSL)",
            options=[0.85, 0.90, 0.95, 0.98, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%"
        )
    with p2:
        review_days = st.slider("Review Period Interval (Days)", min_value=1, max_value=14, value=7)
    with p3:
        st.caption("Safety stock dynamically scales with service level: SS = z(CSL) × σ_d × √L")

# Load Inventory Data
@st.cache_data
def load_inventory_data():
    inv_file = settings.RAW_DATA_DIR / "inventory.csv"
    if inv_file.exists():
        return pd.read_csv(inv_file)
    return pd.DataFrame()

inv_raw = load_inventory_data()

if inv_raw.empty:
    st.error("Inventory database not found. Please run the training pipeline first.")
    st.stop()

# Filters
f1, f2, f3 = st.columns(3)
with f1:
    stores_list = ["All Stores"] + sorted(inv_raw["store_id"].unique().tolist())
    selected_store = st.selectbox("Filter Store", stores_list, index=0)
with f2:
    prod_list = ["All Categories"] + sorted(inv_raw["product_id"].unique().tolist())
    selected_prod = st.selectbox("Filter Product", prod_list, index=0)
with f3:
    risk_filter = st.selectbox("Filter Risk Level", ["All Risks", "CRITICAL_STOCKOUT", "HIGH", "MEDIUM", "LOW", "OVERSTOCKED"], index=0)

# Compute Recommendations
@st.cache_data
def get_recommendations_batch(service_lvl: float, r_days: int):
    # Fetch from API or local engine
    try:
        res = requests.get(f"{settings.API_URL}/inventory/risks?service_level={service_lvl}", timeout=3.0)
        if res.status_code == 200:
            data = res.json()
            return pd.DataFrame(data["items"])
    except Exception:
        pass
        
    # Local fallback
    clean_path = settings.PROCESSED_DATA_DIR / "clean_sales.csv"
    sales_df = pd.read_csv(clean_path)
    forecast_stats = (
        sales_df.groupby(["store_nbr", "family"])["sales"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "forecast_mean", "std": "forecast_std"})
    )
    return InventoryEngine.evaluate_batch(inv_raw, forecast_stats, service_level=service_lvl)

with st.spinner("Calculating replenishment orders and risk levels..."):
    recs_df = get_recommendations_batch(service_level, review_days)

# Apply UI Filters
filtered_df = recs_df.copy()
if selected_store != "All Stores":
    filtered_df = filtered_df[filtered_df["store_id"] == selected_store]
if selected_prod != "All Categories":
    filtered_df = filtered_df[filtered_df["product_id"] == selected_prod]
if risk_filter != "All Risks":
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

# Summary Cards
crit_count = (recs_df["risk_level"] == "CRITICAL_STOCKOUT").sum()
high_count = (recs_df["risk_level"] == "HIGH").sum()
total_units = int(filtered_df["suggested_order"].sum())
total_cost = float(filtered_df["order_value"].sum())

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Critical Stockouts", f"{crit_count}", "Immediate Action", "#b91c1c")
with k2:
    render_kpi("High Risk SKUs", f"{high_count}", "Stock < Lead Time", "#ef4444")
with k3:
    render_kpi("Recommended Order Units", f"{total_units:,}", "Filtered Selection", "#3b82f6")
with k4:
    render_kpi("Purchase Order Value", f"${total_cost:,.2f}", "Estimated Capital", "#8b5cf6")

st.markdown("---")

# Render Recommendations Table
st.subheader("Replenishment Decision Grid")

def format_risk(val):
    colors = {
        "CRITICAL_STOCKOUT": "background-color: #fee2e2; color: #991b1b; font-weight: bold",
        "HIGH": "background-color: #fef2f2; color: #dc2626; font-weight: bold",
        "MEDIUM": "background-color: #fefce8; color: #ca8a04; font-weight: bold",
        "LOW": "background-color: #f0fdf4; color: #16a34a; font-weight: bold",
        "OVERSTOCKED": "background-color: #f5f3ff; color: #7c3aed; font-weight: bold",
    }
    return colors.get(val, "")

display_cols = [
    "product_id", "store_id", "risk_level", "current_stock", "inventory_position",
    "forecast_daily_mean", "lead_time_days", "safety_stock", "reorder_point",
    "days_of_supply", "suggested_order", "unit_cost", "order_value"
]

styled_table = filtered_df[display_cols].style.applymap(format_risk, subset=["risk_level"]).format({
    "current_stock": "{:,.1f}",
    "inventory_position": "{:,.1f}",
    "forecast_daily_mean": "{:,.1f}",
    "safety_stock": "{:,.1f}",
    "reorder_point": "{:,.1f}",
    "days_of_supply": "{:,.1f}d",
    "suggested_order": "{:,}",
    "unit_cost": "${:,.2f}",
    "order_value": "${:,.2f}"
})

st.dataframe(styled_table, use_container_width=True)

# Order Export
c_dl1, c_dl2 = st.columns([1, 3])
with c_dl1:
    po_csv = filtered_df[filtered_df["suggested_order"] > 0][display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📑 Generate Purchase Order CSV",
        data=po_csv,
        file_name="purchase_order_recommendations.csv",
        mime="text/csv"
    )
