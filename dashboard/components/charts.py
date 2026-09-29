"""
Plotly Chart Builders for Streamlit Dashboard.
"""
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

CHART_THEME = {
    "font_family": "Plus Jakarta Sans, sans-serif",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(248,250,252,0.6)",
    "gridcolor": "#e2e8f0"
}

def plot_forecast_with_intervals(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str = "Demand Forecast with Prediction Intervals"
) -> go.Figure:
    """
    Renders actual history and future forecast with 80% & 95% uncertainty bands.
    """
    fig = go.Figure()
    
    # 1. Historical Actuals (last 45 days)
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(hist_df["date"]),
            y=hist_df["sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=4)
        ))
        
    if not forecast_df.empty:
        fc_dates = pd.to_datetime(forecast_df["date"])
        
        # 2. 95% Confidence Band (Outer Fan)
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["upper_95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["lower_95"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.12)",
            name="95% Confidence Interval",
            hoverinfo="skip"
        ))
        
        # 3. 80% Confidence Band (Inner Fan)
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["upper_80"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["lower_80"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.22)",
            name="80% Confidence Interval",
            hoverinfo="skip"
        ))
        
        # 4. Point Forecast Line
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["predicted_sales"],
            mode="lines+markers",
            name="XGBoost Forecast",
            line=dict(color="#4f46e5", width=3, dash="solid"),
            marker=dict(size=6, color="#3730a3")
        ))
        
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family=CHART_THEME["font_family"], color="#0f172a")),
        font=dict(family=CHART_THEME["font_family"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=CHART_THEME["gridcolor"]),
        yaxis=dict(title="Daily Units Sold", showgrid=True, gridcolor=CHART_THEME["gridcolor"]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_shap_waterfall(drivers: List[Dict[str, Any]], title: str = "SHAP Feature Impact on Forecast") -> go.Figure:
    """
    Horizontal bar chart showing features that push demand UP (green/indigo) vs DOWN (coral/amber).
    """
    df = pd.DataFrame(drivers).sort_values("shap_value", ascending=True)
    colors = ["#10b981" if v > 0 else "#ef4444" for v in df["shap_value"]]
    
    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.1f}" for v in df["shap_value"]],
        textposition="auto"
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family=CHART_THEME["font_family"])),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"]),
        xaxis=dict(title="SHAP Contribution to Predicted Units (±)", showgrid=True, gridcolor=CHART_THEME["gridcolor"]),
        margin=dict(l=140, r=40, t=40, b=40),
        height=380
    )
    return fig

def plot_inventory_risk_donut(risk_counts: Dict[str, int]) -> go.Figure:
    """
    Donut chart of inventory risk distribution.
    """
    color_map = {
        "CRITICAL_STOCKOUT": "#b91c1c",
        "HIGH": "#ef4444",
        "MEDIUM": "#f59e0b",
        "LOW": "#10b981",
        "OVERSTOCKED": "#8b5cf6"
    }
    
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = [color_map.get(k, "#64748b") for k in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    )])
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280
    )
    return fig

def plot_simulation_trajectory(trajectory_records: List[Dict[str, Any]], title: str = "Inventory Backtest Trajectory") -> go.Figure:
    """
    Simulated trajectory: On-hand inventory vs daily demand and lost sales markers.
    """
    df = pd.DataFrame(trajectory_records)
    
    fig = go.Figure()
    
    # 1. On Hand Stock Line
    fig.add_trace(go.Scatter(
        x=df["day"],
        y=df["ending_on_hand"],
        mode="lines",
        name="On-Hand Inventory",
        line=dict(color="#059669", width=2.5)
    ))
    
    # 2. Demand Bar
    fig.add_trace(go.Bar(
        x=df["day"],
        y=df["demand"],
        name="Realized Demand",
        marker=dict(color="rgba(148, 163, 184, 0.45)")
    ))
    
    # 3. Stockout Lost Sales Callouts
    stockouts = df[df["lost_sales"] > 0]
    if not stockouts.empty:
        fig.add_trace(go.Scatter(
            x=stockouts["day"],
            y=stockouts["lost_sales"],
            mode="markers",
            name="Stockout Lost Sales",
            marker=dict(symbol="x", size=10, color="#dc2626", line=dict(width=2))
        ))
        
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family=CHART_THEME["font_family"])),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(family=CHART_THEME["font_family"]),
        xaxis=dict(title="Simulation Day", showgrid=True, gridcolor=CHART_THEME["gridcolor"]),
        yaxis=dict(title="Units", showgrid=True, gridcolor=CHART_THEME["gridcolor"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig
