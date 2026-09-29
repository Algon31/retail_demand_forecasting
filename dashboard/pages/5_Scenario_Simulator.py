"""
Scenario Simulator Page:
Simulates operational inventory replenishment policies and demand scenarios.
"""
import streamlit as st
import pandas as pd
import requests

from src.config import settings
from src.inventory.simulator import InventorySimulator
from dashboard.components.styles import CUSTOM_CSS
from dashboard.components.charts import plot_simulation_trajectory
from dashboard.components.kpi_cards import render_kpi

st.set_page_config(page_title="Scenario Simulator - Retail Demand AI", page_icon="🧪", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧪 Operational Scenario Simulator")
st.markdown("Test replenishment policies, supplier lead time changes, and safety-stock sensitivity under simulated retail demand.")

# Load Sales Data for simulation
@st.cache_data
def load_sales_series():
    clean_path = settings.PROCESSED_DATA_DIR / "clean_sales.csv"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    return pd.DataFrame()

sales_df = load_sales_series()

if sales_df.empty:
    st.error("Historical dataset not found.")
    st.stop()

# Controls
st.subheader("Scenario Parameters")
c1, c2, c3, c4 = st.columns(4)

with c1:
    stores = sorted(sales_df["store_nbr"].unique().tolist())
    s_store = st.selectbox("Store", stores, index=0)
with c2:
    families = sorted(sales_df["family"].unique().tolist())
    s_family = st.selectbox("Category", families, index=0)
with c3:
    lead_time = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=21, value=7)
with c4:
    service_lvl = st.select_slider(
        "Service Level Target",
        options=[0.85, 0.90, 0.95, 0.98, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%"
    )

c5, c6, c7, c8 = st.columns(4)
with c5:
    review_period = st.slider("Review Period (Days)", min_value=1, max_value=14, value=7)
with c6:
    init_stock = st.number_input("Starting On-Hand Inventory", min_value=10.0, max_value=5000.0, value=250.0, step=50.0)
with c7:
    unit_cost = st.number_input("Unit Purchase Cost ($)", min_value=1.0, max_value=500.0, value=3.50, step=0.50)
with c8:
    selling_price = st.number_input("Selling Price ($)", min_value=1.0, max_value=500.0, value=5.00, step=0.50)

# Filter recent demand series (e.g. 60 days)
series_data = sales_df[
    (sales_df["store_nbr"] == s_store) &
    (sales_df["family"] == s_family)
].sort_values("date").tail(60)["sales"]

if len(series_data) < 20:
    st.error("Insufficient historical points for simulation.")
    st.stop()

# Run Simulations for both policies
sim = InventorySimulator(
    lead_time_days=lead_time,
    review_period_days=review_period,
    service_level=service_lvl,
    unit_cost=unit_cost,
    selling_price=selling_price
)

res_ml = sim.run_simulation(series_data, initial_stock=init_stock, policy_type="dynamic_ml")
res_static = sim.run_simulation(series_data, initial_stock=init_stock, policy_type="static_heuristic")

st.markdown("---")

# Comparative KPI Cards
st.subheader("Policy Comparison: Dynamic ML vs Static Heuristic")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    delta_fill = f"{res_ml['fill_rate_percent'] - res_static['fill_rate_percent']:+.1f}% vs Static"
    render_kpi("Dynamic ML Fill Rate", f"{res_ml['fill_rate_percent']:.1f}%", delta_fill, "#10b981")
with col_m2:
    delta_stockouts = f"{res_ml['stockout_days'] - res_static['stockout_days']:+d} days vs Static"
    render_kpi("ML Stockout Days", f"{res_ml['stockout_days']} days", delta_stockouts, "#10b981" if res_ml['stockout_days'] <= res_static['stockout_days'] else "#ef4444")
with col_m3:
    render_kpi("Average Inventory Level", f"{res_ml['average_inventory']:.0f} units", f"Static: {res_static['average_inventory']:.0f} units", "#3b82f6")
with col_m4:
    savings = res_static["total_holding_cost"] + res_static["lost_sales_revenue"] - (res_ml["total_holding_cost"] + res_ml["lost_sales_revenue"])
    render_kpi("Net Economic Benefit", f"${max(0.0, savings):,.2f}", "Holding & Stockout Savings", "#8b5cf6")

# Comparison Table
comp_table = pd.DataFrame([
    {
        "Metric": "Fill Rate (%)",
        "Dynamic ML Policy": f"{res_ml['fill_rate_percent']:.2f}%",
        "Static Heuristic Policy": f"{res_static['fill_rate_percent']:.2f}%",
    },
    {
        "Metric": "Stockout Frequency (Days)",
        "Dynamic ML Policy": f"{res_ml['stockout_days']} / {res_ml['total_days']}",
        "Static Heuristic Policy": f"{res_static['stockout_days']} / {res_static['total_days']}",
    },
    {
        "Metric": "Lost Sales (Units)",
        "Dynamic ML Policy": f"{res_ml['total_lost_sales']:.1f}",
        "Static Heuristic Policy": f"{res_static['total_lost_sales']:.1f}",
    },
    {
        "Metric": "Average Inventory (Units)",
        "Dynamic ML Policy": f"{res_ml['average_inventory']:.1f}",
        "Static Heuristic Policy": f"{res_static['average_inventory']:.1f}",
    },
    {
        "Metric": "Total Holding Cost ($)",
        "Dynamic ML Policy": f"${res_ml['total_holding_cost']:,.2f}",
        "Static Heuristic Policy": f"${res_static['total_holding_cost']:,.2f}",
    },
    {
        "Metric": "Lost Sales Opportunity Cost ($)",
        "Dynamic ML Policy": f"${res_ml['lost_sales_revenue']:,.2f}",
        "Static Heuristic Policy": f"${res_static['lost_sales_revenue']:,.2f}",
    },
    {
        "Metric": "Purchase Orders Placed",
        "Dynamic ML Policy": f"{res_ml['orders_count']}",
        "Static Heuristic Policy": f"{res_static['orders_count']}",
    },
])
st.table(comp_table)

# Visual Trajectories
st.subheader("Daily Stock & Demand Trajectory")
t_tab1, t_tab2 = st.tabs(["Dynamic ML Trajectory", "Static Heuristic Trajectory"])

with t_tab1:
    fig_ml = plot_simulation_trajectory(res_ml["daily_trajectory"], title="Dynamic ML Policy — Daily On-Hand Inventory & Stockouts")
    st.plotly_chart(fig_ml, use_container_width=True)
    
with t_tab2:
    fig_st = plot_simulation_trajectory(res_static["daily_trajectory"], title="Static Policy — Daily On-Hand Inventory & Stockouts")
    st.plotly_chart(fig_st, use_container_width=True)
