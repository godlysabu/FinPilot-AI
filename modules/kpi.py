"""
kpi.py
------
Module 4: Financial Analysis

Takes the cleaned dataset (from Module 3) and:
  1. Auto-detects which columns represent Date, Amount, Category, and Type
     (Revenue vs Expense), with a manual override UI in case detection is wrong.
  2. Calculates core financial KPIs: Total Revenue, Total Expenses, Net Profit,
     Profit Margin, Average Transaction, Highest/Lowest Expense, Monthly
     Revenue/Expenses, Cash Flow, Monthly Growth, Revenue/Expense by Category,
     and Top Spending Categories.

Column detection is heuristic (name hints + content inspection) since
real-world spreadsheets rarely use exactly the same column names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger("finpilot.kpi")

DATE_COLUMN_HINTS = ["date", "dt", "time", "period"]
AMOUNT_COLUMN_HINTS = ["amount", "revenue", "expense", "price", "cost", "sales", "total", "value"]
CATEGORY_COLUMN_HINTS = ["category", "categ", "type_detail", "class"]
TYPE_COLUMN_HINTS = ["type", "flow", "kind"]

REVENUE_KEYWORDS = {"revenue", "income", "credit", "sales", "sale", "in"}
EXPENSE_KEYWORDS = {"expense", "cost", "debit", "spend", "expenditure", "out"}


@dataclass
class ColumnMapping:
    date_col: Optional[str]
    amount_col: Optional[str]
    category_col: Optional[str]
    type_col: Optional[str]
    revenue_values: List[str] = field(default_factory=list)


@dataclass
class KPIResult:
    total_revenue: float
    total_expenses: float
    net_profit: float
    profit_margin: float
    average_transaction: float
    highest_expense: float
    lowest_expense: float
    monthly_revenue: pd.Series
    monthly_expenses: pd.Series
    monthly_cash_flow: pd.Series
    monthly_growth_pct: pd.Series
    revenue_by_category: pd.Series
    expense_by_category: pd.Series
    top_spending_categories: pd.Series


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Prefer a column whose name hints at a date; fall back to the first column
    where most values successfully parse as dates."""
    name_matches = [c for c in df.columns if any(h in c.lower() for h in DATE_COLUMN_HINTS)]
    if name_matches:
        return name_matches[0]

    best_col, best_rate = None, 0.0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_col, best_rate = col, rate
    return best_col if best_rate > 0.5 else None


def _detect_amount_column(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    """Prefer a numeric column whose name hints at an amount; fall back to
    the first numeric column available."""
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    name_matches = [c for c in numeric_cols if any(h in c.lower() for h in AMOUNT_COLUMN_HINTS)]
    return name_matches[0] if name_matches else numeric_cols[0]


def _detect_type_column(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    """Prefer a column named like 'Type'; fall back to any text column whose
    unique values substantially overlap with revenue/expense keywords."""
    candidates = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]

    name_matches = [c for c in candidates if any(h in c.lower() for h in TYPE_COLUMN_HINTS)]
    if name_matches:
        return name_matches[0]

    for col in candidates:
        uniques = set(str(v).strip().lower() for v in df[col].dropna().unique())
        if uniques & (REVENUE_KEYWORDS | EXPENSE_KEYWORDS):
            return col
    return None


def _detect_category_column(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    """Prefer a column named like 'Category'; fall back to a text column with a
    reasonable number of unique values (not an ID, not free-form description)."""
    candidates = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]

    name_matches = [c for c in candidates if any(h in c.lower() for h in CATEGORY_COLUMN_HINTS)]
    if name_matches:
        return name_matches[0]

    best_col, best_score = None, -1
    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        n_total = df[col].notna().sum()
        if n_total == 0:
            continue
        # Reasonable "category-like" columns have several repeated values,
        # not near-unique values (like an ID) and not just 1 constant value.
        if 1 < n_unique <= max(2, n_total * 0.5):
            score = n_total / n_unique  # higher = more repetition = more "category-like"
            if score > best_score:
                best_col, best_score = col, score
    return best_col


def _guess_revenue_values(df: pd.DataFrame, type_col: Optional[str]) -> List[str]:
    """Given a detected Type column, guess which of its unique values represent Revenue."""
    if type_col is None:
        return []
    uniques = [str(v).strip() for v in df[type_col].dropna().unique()]
    revenue_values = [v for v in uniques if v.strip().lower() in REVENUE_KEYWORDS]
    if revenue_values:
        return revenue_values
    # Fallback: fuzzy contains-match against keywords
    return [v for v in uniques if any(kw in v.strip().lower() for kw in REVENUE_KEYWORDS)]


def detect_columns(df: pd.DataFrame) -> ColumnMapping:
    """Run all column-detection heuristics and return a best-guess ColumnMapping."""
    date_col = _detect_date_column(df)
    exclude = [c for c in [date_col] if c]

    amount_col = _detect_amount_column(df, exclude)
    exclude_with_amount = exclude + ([amount_col] if amount_col else [])

    type_col = _detect_type_column(df, exclude_with_amount)
    exclude_with_type = exclude_with_amount + ([type_col] if type_col else [])

    category_col = _detect_category_column(df, exclude_with_type)

    revenue_values = _guess_revenue_values(df, type_col)

    return ColumnMapping(
        date_col=date_col,
        amount_col=amount_col,
        category_col=category_col,
        type_col=type_col,
        revenue_values=revenue_values,
    )


def compute_kpis(df: pd.DataFrame, mapping: ColumnMapping) -> KPIResult:
    """
    Compute all financial KPIs given a finalized column mapping.
    Rows where Type isn't recognized as Revenue are treated as Expense.
    """
    work = df.copy()

    # Parse dates for monthly grouping
    work["_parsed_date"] = pd.to_datetime(work[mapping.date_col], errors="coerce", format="mixed")
    work["_amount"] = pd.to_numeric(work[mapping.amount_col], errors="coerce").fillna(0)
    work["_is_revenue"] = work[mapping.type_col].astype(str).str.strip().isin(mapping.revenue_values)

    revenue_rows = work[work["_is_revenue"]]
    expense_rows = work[~work["_is_revenue"]]

    total_revenue = float(revenue_rows["_amount"].sum())
    total_expenses = float(expense_rows["_amount"].sum())
    net_profit = total_revenue - total_expenses
    profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0.0

    average_transaction = float(work["_amount"].mean()) if not work.empty else 0.0
    highest_expense = float(expense_rows["_amount"].max()) if not expense_rows.empty else 0.0
    lowest_expense = float(expense_rows["_amount"].min()) if not expense_rows.empty else 0.0

    # Monthly breakdowns (only rows with a valid parsed date)
    dated = work.dropna(subset=["_parsed_date"]).copy()
    dated["_month"] = dated["_parsed_date"].dt.to_period("M").astype(str)

    monthly_revenue = dated[dated["_is_revenue"]].groupby("_month")["_amount"].sum().sort_index()
    monthly_expenses = dated[~dated["_is_revenue"]].groupby("_month")["_amount"].sum().sort_index()

    all_months = sorted(set(monthly_revenue.index) | set(monthly_expenses.index))
    monthly_revenue = monthly_revenue.reindex(all_months, fill_value=0.0)
    monthly_expenses = monthly_expenses.reindex(all_months, fill_value=0.0)
    monthly_cash_flow = monthly_revenue - monthly_expenses

    monthly_growth_pct = monthly_revenue.pct_change().fillna(0.0) * 100

    # Category breakdowns
    if mapping.category_col:
        revenue_by_category = revenue_rows.groupby(mapping.category_col)["_amount"].sum().sort_values(ascending=False)
        expense_by_category = expense_rows.groupby(mapping.category_col)["_amount"].sum().sort_values(ascending=False)
        top_spending_categories = expense_by_category.head(5)
    else:
        revenue_by_category = pd.Series(dtype=float)
        expense_by_category = pd.Series(dtype=float)
        top_spending_categories = pd.Series(dtype=float)

    return KPIResult(
        total_revenue=total_revenue,
        total_expenses=total_expenses,
        net_profit=net_profit,
        profit_margin=profit_margin,
        average_transaction=average_transaction,
        highest_expense=highest_expense,
        lowest_expense=lowest_expense,
        monthly_revenue=monthly_revenue,
        monthly_expenses=monthly_expenses,
        monthly_cash_flow=monthly_cash_flow,
        monthly_growth_pct=monthly_growth_pct,
        revenue_by_category=revenue_by_category,
        expense_by_category=expense_by_category,
        top_spending_categories=top_spending_categories,
    )


def render_analysis_page() -> None:
    """Render the Streamlit UI for Module 4: Financial Analysis."""
    st.subheader("📊 Financial Analysis")

    df = st.session_state.get("cleaned_dataset")
    source_label = "cleaned"
    if df is None:
        df = st.session_state.get("raw_dataset")
        source_label = "raw (not yet cleaned)"
    if df is None:
        st.warning("No dataset loaded yet. Please upload a dataset first on the 'Upload Dataset' page.")
        return

    if source_label != "cleaned":
        st.info("Using the raw dataset — visit 'Data Cleaning' first for more accurate results.")

    auto_mapping = detect_columns(df)

    st.markdown("#### Column Mapping")
    st.caption("FinPilot auto-detected these columns. Adjust if anything looks wrong.")

    all_cols = list(df.columns)
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox(
            "Date column", all_cols,
            index=all_cols.index(auto_mapping.date_col) if auto_mapping.date_col in all_cols else 0,
        )
        category_col = st.selectbox(
            "Category column", all_cols,
            index=all_cols.index(auto_mapping.category_col) if auto_mapping.category_col in all_cols else 0,
        )
    with col2:
        amount_col = st.selectbox(
            "Amount column", all_cols,
            index=all_cols.index(auto_mapping.amount_col) if auto_mapping.amount_col in all_cols else 0,
        )
        type_col = st.selectbox(
            "Type column (Revenue / Expense)", all_cols,
            index=all_cols.index(auto_mapping.type_col) if auto_mapping.type_col in all_cols else 0,
        )

    unique_type_values = sorted(str(v).strip() for v in df[type_col].dropna().unique())
    default_revenue_values = [v for v in unique_type_values if v in auto_mapping.revenue_values] or unique_type_values[:1]
    revenue_values = st.multiselect(
        "Which values in the Type column mean 'Revenue'? (everything else is treated as Expense)",
        unique_type_values,
        default=default_revenue_values,
    )

    mapping = ColumnMapping(
        date_col=date_col, amount_col=amount_col, category_col=category_col,
        type_col=type_col, revenue_values=revenue_values,
    )

    kpis = compute_kpis(df, mapping)

    st.markdown("---")
    st.markdown("#### Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{kpis.total_revenue:,.2f}")
    c2.metric("Total Expenses", f"₹{kpis.total_expenses:,.2f}")
    c3.metric("Net Profit", f"₹{kpis.net_profit:,.2f}")
    c4.metric("Profit Margin", f"{kpis.profit_margin:.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Average Transaction", f"₹{kpis.average_transaction:,.2f}")
    c6.metric("Highest Expense", f"₹{kpis.highest_expense:,.2f}")
    c7.metric("Lowest Expense", f"₹{kpis.lowest_expense:,.2f}")

    st.markdown("---")
    st.markdown("#### Monthly Revenue vs Expenses")
    monthly_df = pd.DataFrame({
        "Revenue": kpis.monthly_revenue,
        "Expenses": kpis.monthly_expenses,
        "Cash Flow": kpis.monthly_cash_flow,
        "Revenue Growth %": kpis.monthly_growth_pct.round(1),
    })
    st.dataframe(monthly_df, width="stretch")

    st.markdown("---")
    col_rev, col_exp = st.columns(2)
    with col_rev:
        st.markdown("#### Revenue by Category")
        if not kpis.revenue_by_category.empty:
            st.dataframe(kpis.revenue_by_category.rename("Revenue"), width="stretch")
        else:
            st.write("No revenue rows found with the current mapping.")
    with col_exp:
        st.markdown("#### Expense by Category")
        if not kpis.expense_by_category.empty:
            st.dataframe(kpis.expense_by_category.rename("Expense"), width="stretch")
        else:
            st.write("No expense rows found with the current mapping.")

    st.markdown("#### Top 5 Spending Categories")
    if not kpis.top_spending_categories.empty:
        st.dataframe(kpis.top_spending_categories.rename("Total Spent"), width="stretch")
    else:
        st.write("No expense categories found with the current mapping.")

    # Store for downstream modules (Dashboard, Forecasting, AI Insights, Reports)
    st.session_state["column_mapping"] = mapping
    st.session_state["kpi_result"] = kpis

    st.success("✅ Financial analysis complete. KPIs are ready for the Dashboard and Reports.")
