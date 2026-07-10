"""
cleaner.py
----------
Module 3: Automatic Data Cleaning

Takes the raw uploaded DataFrame and returns a cleaned copy, fixing the
issues surfaced by the validator (Module 2) automatically:

  - Remove duplicate rows
  - Remove fully empty rows
  - Fill / drop missing values
  - Trim extra whitespace from text columns
  - Standardize text casing (Title Case for category-like columns)
  - Strip currency symbols & thousands separators, convert to numeric
  - Standardize date columns to YYYY-MM-DD

Never mutates the original DataFrame — always works on a copy, so the
raw upload stays available if the user wants to compare before/after.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import streamlit as st

logger = logging.getLogger("finpilot.cleaner")

CURRENCY_SYMBOLS = ["₹", "$", "€", "£", "¥"]
DATE_COLUMN_HINTS = ["date", "dt", "time", "period"]

# Column name hints used to decide how a numeric column's missing values get filled.
# "Amount-like" columns (the actual transaction value) are filled with the column
# average so a missing value doesn't unfairly zero out a real transaction.
# Everything else (quantities, counts, discounts, tax, etc.) defaults to 0, since
# "no value recorded" for those usually genuinely means zero.
MEAN_FILL_HINTS = ["amount", "revenue", "expense", "price", "cost", "sales", "total", "profit", "value"]
ZERO_FILL_HINTS = ["qty", "quantity", "count", "discount", "tax", "unit", "number", "num"]

# Matches a currency symbol, thousands separators, and whitespace around numbers,
# e.g. "₹12,500", "$ 1,200.50", "12500 "
_CURRENCY_CLEAN_PATTERN = re.compile(
    "|".join(re.escape(sym) for sym in CURRENCY_SYMBOLS) + r"|,|\s"
)


@dataclass
class CleaningSummary:
    rows_before: int
    rows_after: int
    duplicate_rows_removed: int
    empty_rows_removed: int
    missing_values_filled: int
    missing_values_before: int
    currency_columns_cleaned: List[str]
    date_columns_standardized: List[str]
    text_columns_standardized: List[str]
    numeric_mean_filled_columns: List[str] = field(default_factory=list)
    numeric_zero_filled_columns: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _looks_like_currency_column(series: pd.Series) -> bool:
    """
    A text column is treated as a currency/numeric column if, after stripping
    currency symbols and commas, most of its non-null values parse as numbers.
    This catches columns where only a few rows happen to carry a ₹/$ symbol.
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    cleaned = sample.apply(lambda v: _CURRENCY_CLEAN_PATTERN.sub("", v))
    parsed = pd.to_numeric(cleaned, errors="coerce")
    parse_rate = parsed.notna().mean()
    return bool(parse_rate >= 0.6)  # at least 60% of values look numeric once cleaned


def _clean_currency_series(series: pd.Series) -> pd.Series:
    """Strip currency symbols, commas, and whitespace; convert to numeric (float)."""

    def _clean_value(v):
        if pd.isna(v):
            return pd.NA
        return _CURRENCY_CLEAN_PATTERN.sub("", str(v))

    cleaned = series.map(_clean_value)
    return pd.to_numeric(cleaned, errors="coerce")


def _standardize_date_series(series: pd.Series) -> pd.Series:
    """Parse a date-like column with mixed formats and standardize to YYYY-MM-DD strings."""
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.dt.strftime("%Y-%m-%d")


def _standardize_text_series(series: pd.Series) -> pd.Series:
    """Trim whitespace and apply consistent Title Case to a text column."""
    return series.astype(str).str.strip().str.title().replace("Nan", pd.NA)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    """
    Clean the given DataFrame and return (cleaned_df, CleaningSummary).
    The original df is never modified.
    """
    rows_before = df.shape[0]
    missing_before = int(df.isna().sum().sum())
    cleaned = df.copy()

    # 1. Remove fully empty rows
    empty_mask = cleaned.isna().all(axis=1)
    empty_rows_removed = int(empty_mask.sum())
    cleaned = cleaned.loc[~empty_mask].reset_index(drop=True)

    # 2. Remove duplicate rows
    duplicate_mask = cleaned.duplicated()
    duplicate_rows_removed = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].reset_index(drop=True)

    # 3. Standardize date columns
    date_columns_standardized = []
    for col in cleaned.columns:
        if any(hint in col.lower() for hint in DATE_COLUMN_HINTS):
            cleaned[col] = _standardize_date_series(cleaned[col])
            date_columns_standardized.append(col)

    # 4. Clean currency-like numeric columns
    currency_columns_cleaned = []
    for col in cleaned.columns:
        if col in date_columns_standardized:
            continue
        if _looks_like_currency_column(cleaned[col]) or pd.api.types.is_numeric_dtype(cleaned[col]):
            # Only re-clean if it's not already a clean numeric dtype
            if not pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = _clean_currency_series(cleaned[col])
                currency_columns_cleaned.append(col)

    # 5. Standardize remaining text columns (trim whitespace, Title Case)
    text_columns_standardized = []
    for col in cleaned.columns:
        if col in date_columns_standardized or col in currency_columns_cleaned:
            continue
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            cleaned[col] = _standardize_text_series(cleaned[col])
            text_columns_standardized.append(col)

    # 6. Fill missing values
    #    - Numeric "amount-like" columns (Amount, Revenue, Expense, Price, Cost...):
    #      fill with the column's average, so one missing value doesn't zero out
    #      what was likely a real transaction.
    #    - Numeric "count-like" columns (Quantity, Discount, Tax...) and any other
    #      numeric column not recognized as amount-like: fill with 0.
    #    - Text columns: fill with "Unknown"
    missing_filled = 0
    numeric_mean_filled_columns: List[str] = []
    numeric_zero_filled_columns: List[str] = []
    for col in cleaned.columns:
        col_missing = int(cleaned[col].isna().sum())
        if col_missing == 0:
            continue
        col_lower = col.lower()
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            is_amount_like = any(hint in col_lower for hint in MEAN_FILL_HINTS)
            is_count_like = any(hint in col_lower for hint in ZERO_FILL_HINTS)
            if is_amount_like and not is_count_like:
                col_mean = cleaned[col].mean()
                # If the entire column is missing, mean is NaN — fall back to 0
                fill_value = 0.0 if pd.isna(col_mean) else round(float(col_mean), 2)
                cleaned[col] = cleaned[col].fillna(fill_value)
                numeric_mean_filled_columns.append(col)
            else:
                cleaned[col] = cleaned[col].fillna(0)
                numeric_zero_filled_columns.append(col)
        else:
            cleaned[col] = cleaned[col].fillna("Unknown")
        missing_filled += col_missing

    rows_after = cleaned.shape[0]

    notes = []
    if empty_rows_removed:
        notes.append(f"Removed {empty_rows_removed} completely empty row(s).")
    if duplicate_rows_removed:
        notes.append(f"Removed {duplicate_rows_removed} duplicate row(s).")
    if currency_columns_cleaned:
        notes.append(f"Cleaned currency formatting in: {currency_columns_cleaned}.")
    if date_columns_standardized:
        notes.append(f"Standardized dates to YYYY-MM-DD in: {date_columns_standardized}.")
    if text_columns_standardized:
        notes.append(f"Standardized text casing/whitespace in: {text_columns_standardized}.")
    if numeric_mean_filled_columns:
        notes.append(f"Filled missing values with the column average in: {numeric_mean_filled_columns}.")
    if numeric_zero_filled_columns:
        notes.append(f"Filled missing values with 0 in: {numeric_zero_filled_columns}.")
    if not notes:
        notes.append("Dataset was already clean — no changes were necessary.")

    summary = CleaningSummary(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicate_rows_removed=duplicate_rows_removed,
        empty_rows_removed=empty_rows_removed,
        missing_values_filled=missing_filled,
        missing_values_before=missing_before,
        currency_columns_cleaned=currency_columns_cleaned,
        date_columns_standardized=date_columns_standardized,
        text_columns_standardized=text_columns_standardized,
        numeric_mean_filled_columns=numeric_mean_filled_columns,
        numeric_zero_filled_columns=numeric_zero_filled_columns,
        notes=notes,
    )

    return cleaned, summary



def render_cleaning_page() -> None:
    """Render the Streamlit UI for Module 3: Automatic Data Cleaning."""
    st.subheader("🧹 Automatic Data Cleaning")

    df = st.session_state.get("raw_dataset")
    if df is None:
        st.warning("No dataset loaded yet. Please upload a dataset first on the 'Upload Dataset' page.")
        return

    # Avoid recomputing the full cleaning pipeline on every Streamlit rerun —
    # only re-run it if the dataset itself has changed since last time.
    cache_key = (st.session_state.get("raw_dataset_name"), df.shape)
    if st.session_state.get("_cleaning_cache_key") == cache_key:
        cleaned_df = st.session_state["cleaned_dataset"]
        summary = st.session_state["cleaning_summary"]
    else:
        cleaned_df, summary = clean_dataset(df)
        st.session_state["_cleaning_cache_key"] = cache_key

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows Before", f"{summary.rows_before:,}")
    col2.metric("Rows After", f"{summary.rows_after:,}")
    col3.metric("Rows Removed", f"{summary.rows_before - summary.rows_after:,}")
    col4.metric("Values Filled", f"{summary.missing_values_filled:,}")

    st.markdown("---")
    st.markdown("#### Cleaning Summary")
    for note in summary.notes:
        st.write(f"- {note}")

    st.markdown("---")
    tab1, tab2 = st.tabs(["Before Cleaning", "After Cleaning"])
    with tab1:
        st.dataframe(df.head(15), width="stretch")
    with tab2:
        st.dataframe(cleaned_df.head(15), width="stretch")

    st.markdown("#### Data Types After Cleaning")
    dtype_df = pd.DataFrame(
        {"Column": cleaned_df.columns, "Data Type": cleaned_df.dtypes.astype(str).values}
    )
    st.dataframe(dtype_df, width="stretch", hide_index=True)

    # Store cleaned dataset for downstream modules (analysis, forecasting, etc.)
    st.session_state["cleaned_dataset"] = cleaned_df
    st.session_state["cleaning_summary"] = summary

    st.success("✅ Cleaned dataset is ready and stored for the next steps (Financial Analysis).")
