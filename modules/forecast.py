"""
forecast.py
-----------
Module 6: Forecasting

Uses Prophet (Meta/Facebook's open-source forecasting library) to predict
future Revenue, Expenses, or Cash Flow based on the cleaned dataset's
transaction history.

Flow:
  1. Build a continuous daily time series (missing days filled with 0) for
     whichever metric the user picks: Revenue, Expenses, or Cash Flow.
  2. Fit a Prophet model on that history.
  3. Predict forward for the chosen horizon: 30 / 90 / 180 / 365 days.
  4. Show an interactive forecast chart (historical + forecast + confidence
     interval band) and a forecast table for the future period only.

The fitted model is cached in session_state per (metric, dataset) so
switching horizons doesn't require re-fitting from scratch every time.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.kpi import ColumnMapping, KPIResult, detect_columns, compute_kpis
from modules.charts import THEMES, DEFAULT_THEME_NAME, DashboardTheme, _apply_common_layout

# Prophet (and its cmdstanpy backend) are very chatty by default — quiet them down
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

HORIZON_OPTIONS = {"30 Days": 30, "90 Days": 90, "180 Days": 180, "365 Days": 365}
METRIC_OPTIONS = ["Revenue", "Expenses", "Cash Flow"]

MIN_HISTORY_DAYS = 14  # below this, warn that forecasts may be unreliable


def _build_daily_series(df: pd.DataFrame, mapping: ColumnMapping, metric: str) -> pd.DataFrame:
    """
    Build a continuous daily time series (columns: ds, y) for the chosen metric.
    Days with no transactions are filled with 0 so Prophet sees a regular series.
    """
    work = df.copy()
    work["_parsed_date"] = pd.to_datetime(work[mapping.date_col], errors="coerce", format="mixed")
    work["_amount"] = pd.to_numeric(work[mapping.amount_col], errors="coerce").fillna(0)
    work["_is_revenue"] = work[mapping.type_col].astype(str).str.strip().isin(mapping.revenue_values)

    work = work.dropna(subset=["_parsed_date"])
    if work.empty:
        return pd.DataFrame(columns=["ds", "y"])

    if metric == "Revenue":
        daily = work[work["_is_revenue"]].groupby("_parsed_date")["_amount"].sum()
    elif metric == "Expenses":
        daily = work[~work["_is_revenue"]].groupby("_parsed_date")["_amount"].sum()
    else:  # Cash Flow
        rev = work[work["_is_revenue"]].groupby("_parsed_date")["_amount"].sum()
        exp = work[~work["_is_revenue"]].groupby("_parsed_date")["_amount"].sum()
        daily = rev.sub(exp, fill_value=0)

    if daily.empty:
        return pd.DataFrame(columns=["ds", "y"])

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)

    return pd.DataFrame({"ds": daily.index, "y": daily.values})


def _run_prophet_forecast(daily_df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Fit Prophet on daily_df and return the full forecast (historical + future)."""
    from prophet import Prophet  # imported lazily so the rest of the app works even if Prophet isn't installed yet

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(daily_df)
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast


def _forecast_chart(daily_df: pd.DataFrame, forecast: pd.DataFrame, metric: str, theme: DashboardTheme) -> go.Figure:
    """Plot historical actuals + forecast line + confidence interval band."""
    history_end = daily_df["ds"].max()
    future_part = forecast[forecast["ds"] > history_end]

    fig = go.Figure()

    # Confidence interval band (future only)
    fig.add_trace(go.Scatter(
        x=list(future_part["ds"]) + list(future_part["ds"][::-1]),
        y=list(future_part["yhat_upper"]) + list(future_part["yhat_lower"][::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Interval", showlegend=True,
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=daily_df["ds"], y=daily_df["y"], mode="lines", name=f"Historical {metric}",
        line=dict(color=theme.accent, width=2),
    ))

    # Forecast line (future only)
    fig.add_trace(go.Scatter(
        x=future_part["ds"], y=future_part["yhat"], mode="lines", name=f"Forecasted {metric}",
        line=dict(color=theme.revenue if metric == "Revenue" else theme.expense, width=3, dash="dash"),
    ))

    _apply_common_layout(fig, theme, 420)
    fig.update_layout(hovermode="x unified")
    return fig


def render_forecast_page() -> None:
    """Render the Streamlit UI for Module 6: Forecasting."""
    st.subheader("🔮 Forecast Future Revenue / Expenses / Cash Flow")

    df = st.session_state.get("cleaned_dataset")
    if df is None:
        df = st.session_state.get("raw_dataset")
    if df is None:
        st.warning("No dataset loaded yet. Please upload a dataset first on the 'Upload Dataset' page.")
        return

    mapping: Optional[ColumnMapping] = st.session_state.get("column_mapping")
    if mapping is None:
        st.info("Using auto-detected columns — visit 'Financial Analysis' first to confirm or adjust the mapping.")
        mapping = detect_columns(df)

    theme = THEMES[st.session_state.get("dashboard_theme_name", DEFAULT_THEME_NAME)]

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Metric to forecast", METRIC_OPTIONS)
    with col2:
        horizon_label = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()))
    periods = HORIZON_OPTIONS[horizon_label]

    daily_df = _build_daily_series(df, mapping, metric)

    if daily_df.empty or daily_df.shape[0] < 2:
        st.error(
            "Not enough dated transactions to build a forecast. "
            "Please check that your Date column was detected correctly on the 'Financial Analysis' page."
        )
        return

    history_days = daily_df.shape[0]
    if history_days < MIN_HISTORY_DAYS:
        st.warning(
            f"Only {history_days} day(s) of history found. Forecasts from such a short history "
            "may be unreliable — treat them as a rough estimate rather than a precise prediction."
        )

    # Cache the fitted forecast per (metric, dataset shape, horizon) so switching
    # options doesn't always trigger a fresh ~5-10s Prophet fit unnecessarily.
    cache_key = (metric, daily_df.shape[0], float(daily_df["y"].sum()), periods)
    if st.session_state.get("_forecast_cache_key") == cache_key:
        forecast = st.session_state["_forecast_cache_value"]
    else:
        with st.spinner(f"Training forecast model for {metric} ({horizon_label})..."):
            forecast = _run_prophet_forecast(daily_df, periods)
        st.session_state["_forecast_cache_key"] = cache_key
        st.session_state["_forecast_cache_value"] = forecast

    st.markdown("---")
    st.markdown(f"#### {metric} Forecast — Next {horizon_label}")
    st.plotly_chart(_forecast_chart(daily_df, forecast, metric, theme), width="stretch")

    # --- Forecast table (future only) ---
    history_end = daily_df["ds"].max()
    future_table = forecast[forecast["ds"] > history_end][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
    future_table["Date"] = future_table["Date"].dt.strftime("%Y-%m-%d")
    for col in ["Predicted", "Lower Bound", "Upper Bound"]:
        future_table[col] = future_table[col].round(2)

    st.markdown("#### Forecast Table")
    st.dataframe(future_table, width="stretch", hide_index=True)

    # --- Quick summary ---
    total_predicted = future_table["Predicted"].sum()
    avg_daily_predicted = future_table["Predicted"].mean()
    c1, c2 = st.columns(2)
    c1.metric(f"Total Predicted {metric} ({horizon_label})", f"₹{total_predicted:,.0f}")
    c2.metric("Average Predicted / Day", f"₹{avg_daily_predicted:,.0f}")

    st.session_state["forecast_result"] = forecast
    st.session_state["forecast_metric"] = metric

    st.success(f"✅ {metric} forecast ready for the next {horizon_label.lower()}.")
