"""
charts.py
---------
Module 5: Interactive Dashboard

Builds an interactive Plotly-powered dashboard on top of the KPIs computed
in Module 4 (Financial Analysis). All charts are interactive (zoom, hover,
pan) since they use Plotly rather than static matplotlib images.

Charts included:
  - Monthly Revenue Trend
  - Monthly Expense Trend
  - Cash Flow Trend
  - Revenue vs Expenses (grouped bar)
  - Monthly Profit Trend
  - Revenue by Category (pie)
  - Expense by Category (pie)
  - Top Expense Categories (horizontal bar)
  - Top Revenue Categories (horizontal bar)

Includes a Dashboard Theme selector — the chosen theme controls the color
palette and Plotly template used across every chart on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.kpi import KPIResult, ColumnMapping, detect_columns, compute_kpis


@dataclass(frozen=True)
class DashboardTheme:
    name: str
    revenue: str
    expense: str
    profit_positive: str
    profit_negative: str
    accent: str
    palette: List[str]
    plotly_template: str
    paper_bgcolor: str
    font_color: str


THEMES: dict[str, DashboardTheme] = {
    "Classic (Green/Red)": DashboardTheme(
        name="Classic (Green/Red)",
        revenue="#2ecc71", expense="#e74c3c",
        profit_positive="#2ecc71", profit_negative="#e74c3c",
        accent="#3498db",
        palette=["#2ecc71", "#3498db", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c", "#34495e"],
        plotly_template="plotly", paper_bgcolor="rgba(0,0,0,0)", font_color="#262730",
    ),
    "Ocean Blue": DashboardTheme(
        name="Ocean Blue",
        revenue="#00b4d8", expense="#0077b6",
        profit_positive="#00b4d8", profit_negative="#03045e",
        accent="#90e0ef",
        palette=["#03045e", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8", "#023e8a", "#0096c7", "#48cae4"],
        plotly_template="plotly", paper_bgcolor="rgba(0,0,0,0)", font_color="#262730",
    ),
    "Sunset": DashboardTheme(
        name="Sunset",
        revenue="#f9844a", expense="#d62828",
        profit_positive="#f9c74f", profit_negative="#d62828",
        accent="#f3722c",
        palette=["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f", "#e76f51", "#d62828", "#ee9b00"],
        plotly_template="plotly", paper_bgcolor="rgba(0,0,0,0)", font_color="#262730",
    ),
    "Forest": DashboardTheme(
        name="Forest",
        revenue="#40916c", expense="#6f4518",
        profit_positive="#40916c", profit_negative="#6f4518",
        accent="#74c69d",
        palette=["#1b4332", "#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2", "#6f4518", "#a9744f"],
        plotly_template="plotly", paper_bgcolor="rgba(0,0,0,0)", font_color="#262730",
    ),
    "Dark Mode": DashboardTheme(
        name="Dark Mode",
        revenue="#2ecc71", expense="#ff6b6b",
        profit_positive="#2ecc71", profit_negative="#ff6b6b",
        accent="#4dabf7",
        palette=["#2ecc71", "#4dabf7", "#ffd43b", "#da77f2", "#ff922b", "#63e6be", "#ff6b6b", "#adb5bd"],
        plotly_template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e8e8",
    ),
}

DEFAULT_THEME_NAME = "Classic (Green/Red)"


def _apply_common_layout(fig: go.Figure, theme: DashboardTheme, height: int) -> go.Figure:
    fig.update_layout(
        template=theme.plotly_template,
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.paper_bgcolor,
        font=dict(color=theme.font_color),
        legend=dict(font=dict(color=theme.font_color)),
    )
    return fig


def _line_chart(x: list, y: list, name: str, color: str, theme: DashboardTheme) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color, width=3)))
    _apply_common_layout(fig, theme, 350)
    fig.update_layout(hovermode="x unified")
    return fig


def _revenue_vs_expense_chart(kpis: KPIResult, theme: DashboardTheme) -> go.Figure:
    months = list(kpis.monthly_revenue.index)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=kpis.monthly_revenue.values, name="Revenue", marker_color=theme.revenue))
    fig.add_trace(go.Bar(x=months, y=kpis.monthly_expenses.values, name="Expenses", marker_color=theme.expense))
    _apply_common_layout(fig, theme, 380)
    fig.update_layout(barmode="group")
    return fig


def _cash_flow_chart(kpis: KPIResult, theme: DashboardTheme) -> go.Figure:
    months = list(kpis.monthly_cash_flow.index)
    colors = [theme.profit_positive if v >= 0 else theme.profit_negative for v in kpis.monthly_cash_flow.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=kpis.monthly_cash_flow.values, name="Cash Flow", marker_color=colors))
    _apply_common_layout(fig, theme, 350)
    return fig


def _profit_trend_chart(kpis: KPIResult, theme: DashboardTheme) -> go.Figure:
    months = list(kpis.monthly_cash_flow.index)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months, y=kpis.monthly_cash_flow.values, mode="lines+markers",
            name="Net Profit", line=dict(color=theme.accent, width=3), fill="tozeroy",
        )
    )
    _apply_common_layout(fig, theme, 350)
    fig.update_layout(hovermode="x unified")
    return fig


def _category_pie_chart(series: pd.Series, title: str, theme: DashboardTheme) -> go.Figure:
    fig = go.Figure()
    if series.empty:
        _apply_common_layout(fig, theme, 350)
        fig.update_layout(title=f"{title} — no data")
        return fig
    fig.add_trace(
        go.Pie(
            labels=series.index.tolist(), values=series.values.tolist(), hole=0.4,
            marker=dict(colors=theme.palette),
        )
    )
    _apply_common_layout(fig, theme, 350)
    return fig


def _top_categories_bar(series: pd.Series, color: str, theme: DashboardTheme) -> go.Figure:
    fig = go.Figure()
    if series.empty:
        _apply_common_layout(fig, theme, 350)
        return fig
    sorted_series = series.sort_values(ascending=True)  # ascending so biggest bar is on top in horizontal chart
    fig.add_trace(go.Bar(x=sorted_series.values, y=sorted_series.index.tolist(), orientation="h", marker_color=color))
    _apply_common_layout(fig, theme, 350)
    return fig


def render_dashboard_page() -> None:
    """Render the Streamlit UI for Module 5: Interactive Dashboard."""
    st.subheader("📈 Interactive Dashboard")

    df = st.session_state.get("cleaned_dataset")
    if df is None:
        df = st.session_state.get("raw_dataset")
    if df is None:
        st.warning("No dataset loaded yet. Please upload a dataset first on the 'Upload Dataset' page.")
        return

    # --- Theme selector ---
    theme_col, _ = st.columns([1, 2])
    with theme_col:
        selected_theme_name = st.selectbox(
            "🎨 Dashboard Theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(
                st.session_state.get("dashboard_theme_name", DEFAULT_THEME_NAME)
            ),
        )
    st.session_state["dashboard_theme_name"] = selected_theme_name
    theme = THEMES[selected_theme_name]

    # Reuse the KPI result + mapping computed in Financial Analysis if available,
    # so the dashboard always matches what the user confirmed there. Otherwise,
    # fall back to auto-detection so the dashboard still works standalone.
    mapping: ColumnMapping | None = st.session_state.get("column_mapping")
    kpis: KPIResult | None = st.session_state.get("kpi_result")

    if mapping is None or kpis is None:
        st.info("Using auto-detected columns — visit 'Financial Analysis' first to confirm or adjust the mapping.")
        mapping = detect_columns(df)
        kpis = compute_kpis(df, mapping)

    # --- Top KPI cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"₹{kpis.total_revenue:,.0f}")
    c2.metric("Expenses", f"₹{kpis.total_expenses:,.0f}")
    c3.metric("Profit", f"₹{kpis.net_profit:,.0f}")
    c4.metric("Profit Margin", f"{kpis.profit_margin:.1f}%")

    st.markdown("---")

    if kpis.monthly_revenue.empty:
        st.info("No dated transactions found — monthly trend charts need a valid Date column.")
    else:
        # --- Trend charts ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Monthly Revenue Trend")
            st.plotly_chart(
                _line_chart(list(kpis.monthly_revenue.index), kpis.monthly_revenue.values.tolist(), "Revenue", theme.revenue, theme),
                width="stretch",
            )
        with col2:
            st.markdown("##### Monthly Expense Trend")
            st.plotly_chart(
                _line_chart(list(kpis.monthly_expenses.index), kpis.monthly_expenses.values.tolist(), "Expenses", theme.expense, theme),
                width="stretch",
            )

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### Cash Flow Trend")
            st.plotly_chart(_cash_flow_chart(kpis, theme), width="stretch")
        with col4:
            st.markdown("##### Monthly Profit Trend")
            st.plotly_chart(_profit_trend_chart(kpis, theme), width="stretch")

        st.markdown("##### Revenue vs Expenses")
        st.plotly_chart(_revenue_vs_expense_chart(kpis, theme), width="stretch")

    st.markdown("---")
    st.markdown("#### Category Breakdown")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("##### Revenue by Category")
        st.plotly_chart(_category_pie_chart(kpis.revenue_by_category, "Revenue", theme), width="stretch")
    with col6:
        st.markdown("##### Expense by Category")
        st.plotly_chart(_category_pie_chart(kpis.expense_by_category, "Expenses", theme), width="stretch")

    col7, col8 = st.columns(2)
    with col7:
        st.markdown("##### Top Revenue Categories")
        st.plotly_chart(_top_categories_bar(kpis.revenue_by_category.head(5), theme.revenue, theme), width="stretch")
    with col8:
        st.markdown("##### Top Expense Categories")
        st.plotly_chart(_top_categories_bar(kpis.top_spending_categories, theme.expense, theme), width="stretch")
