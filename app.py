"""
app.py
------
FinPilot AI — Main Streamlit Application Entry Point

Currently implemented:
  - Module 1: Upload CSV/Excel (with preview)

Coming in later modules:
  - Dataset Validation
  - Automatic Data Cleaning
  - Financial Analysis (KPIs)
  - Interactive Dashboard (Plotly)
  - Forecasting (Prophet)
  - AI Insights (Ollama + Qwen 3)
  - PDF & Excel Report Export
  - History (SQLite)
"""

import streamlit as st

from modules.upload import render_upload_page
from modules.validator import render_validation_page
from modules.cleaner import render_cleaning_page
from modules.kpi import render_analysis_page
from modules.charts import render_dashboard_page
from modules.forecast import render_forecast_page
from modules.insights import render_insights_page

st.set_page_config(
    page_title="FinPilot AI",
    page_icon="💹",
    layout="wide",
)

# --- Sidebar Navigation ---
st.sidebar.title("💹 FinPilot AI")
st.sidebar.caption("AI-Powered Financial Analysis Dashboard")

PAGES = [
    "Upload Dataset",
    "Data Quality",
    "Data Cleaning",
    "Financial Analysis",
    "Dashboard",
    "Forecast",
    "AI Insights",
    "Reports",
    "History",
    "Settings",
]

selected_page = st.sidebar.radio("Navigate", PAGES, index=0)

st.sidebar.markdown("---")
if "raw_dataset" in st.session_state:
    st.sidebar.success(f"Loaded: {st.session_state.get('raw_dataset_name', 'dataset')}")
else:
    st.sidebar.warning("No dataset loaded yet")

# --- Page Routing ---
st.title("FinPilot AI")

if selected_page == "Upload Dataset":
    render_upload_page()
elif selected_page == "Data Quality":
    render_validation_page()
elif selected_page == "Data Cleaning":
    render_cleaning_page()
elif selected_page == "Financial Analysis":
    render_analysis_page()
elif selected_page == "Dashboard":
    render_dashboard_page()
elif selected_page == "Forecast":
    render_forecast_page()
elif selected_page == "AI Insights":
    render_insights_page()
else:
    st.subheader(selected_page)
    st.info(f"'{selected_page}' will be built in an upcoming module. Stay tuned!")
