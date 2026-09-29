"""
Reusable KPI metric card components with HTML/CSS rendering.
"""
import streamlit as st

def render_kpi(label: str, value: str, delta: str = None, delta_color: str = "#10b981"):
    delta_html = f'<div class="kpi-delta" style="color: {delta_color}">{delta}</div>' if delta else ""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
