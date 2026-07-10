"""
validator.py
------------
Module 2: Dataset Validation

Runs a set of automatic checks against the raw uploaded dataset and
produces a Data Quality Report, without modifying the original data.

Checks performed:
  - Missing values (per column + total)
  - Duplicate rows
  - Fully empty rows
  - Invalid / unparseable dates (in likely date columns)
  - Currency symbols present in text/object columns (₹, $, €, £, commas)
  - Extra leading/trailing whitespace in text columns
  - Mixed data types within a single column
  - Presence of at least one numeric column (needed for financial analysis)

The report is returned as a dictionary so it can be displayed in
Streamlit and later stored in the SQLite history table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

logger = logging.getLogger("finpilot.validator")

CURRENCY_SYMBOLS = ["₹", "$", "€", "£", "¥"]

# Column name hints used to guess which columns should contain dates
DATE_COLUMN_HINTS = ["date", "dt", "time", "period"]


@dataclass
class QualityReport:
    rows_loaded: int
    total_columns: int
    missing_values_total: int
    missing_by_column: Dict[str, int]
    duplicate_rows: int
    empty_rows: int
    date_columns_checked: List[str]
    invalid_dates_total: int
    invalid_dates_by_column: Dict[str, int]
    currency_symbol_columns: List[str]
    whitespace_issue_columns: List[str]
    mixed_type_columns: List[str]
    has_numeric_column: bool
    dataset_ready: bool
    issues: List[str] = field(default_factory=list)


def _guess_date_columns(df: pd.DataFrame) -> List[str]:
    """Return columns whose name suggests they hold dates."""
    return [col for col in df.columns if any(hint in col.lower() for hint in DATE_COLUMN_HINTS)]


def _count_invalid_dates(series: pd.Series) -> int:
    """Count values that fail to parse as dates, ignoring nulls."""
    non_null = series.dropna()
    if non_null.empty:
        return 0
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return int(parsed.isna().sum())


def _is_text_like(series: pd.Series) -> bool:
    """
    True for columns holding text data, covering both the legacy 'object'
    dtype and pandas' newer dedicated string dtype (pandas >= 2.x/3.x).
    """
    return bool(pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))


def _has_currency_symbols(series: pd.Series) -> bool:
    """Check if a column's string values contain common currency symbols."""
    if not _is_text_like(series):
        return False
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    pattern = re.compile("|".join(re.escape(sym) for sym in CURRENCY_SYMBOLS))
    return bool(sample.str.contains(pattern).any())


def _has_whitespace_issues(series: pd.Series) -> bool:
    """Check if a text column has leading/trailing spaces or inconsistent casing."""
    if not _is_text_like(series):
        return False
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    has_padding = (sample != sample.str.strip()).any()
    return bool(has_padding)


def _has_mixed_types(series: pd.Series) -> bool:
    """Check if an object-dtype column contains more than one underlying Python type."""
    # Only legacy 'object' columns can hold mixed Python types; pandas' dedicated
    # string dtype enforces a single type by construction.
    if not pd.api.types.is_object_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    types_found = non_null.map(type).unique()
    return len(types_found) > 1


def run_validation(df: pd.DataFrame) -> QualityReport:
    """
    Run all validation checks against the DataFrame and return a QualityReport.
    This function does not modify df in any way.
    """
    issues: List[str] = []

    rows_loaded = df.shape[0]
    total_columns = df.shape[1]

    # Missing values
    missing_by_column = {
        col: int(count) for col, count in df.isna().sum().items() if count > 0
    }
    missing_values_total = int(df.isna().sum().sum())
    if missing_values_total > 0:
        issues.append(f"{missing_values_total} missing value(s) found across {len(missing_by_column)} column(s).")

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate row(s) found.")

    # Fully empty rows
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows > 0:
        issues.append(f"{empty_rows} completely empty row(s) found.")

    # Date columns
    date_columns_checked = _guess_date_columns(df)
    invalid_dates_by_column = {}
    for col in date_columns_checked:
        invalid_count = _count_invalid_dates(df[col])
        if invalid_count > 0:
            invalid_dates_by_column[col] = invalid_count
    invalid_dates_total = sum(invalid_dates_by_column.values())
    if invalid_dates_total > 0:
        issues.append(f"{invalid_dates_total} invalid date value(s) found in {list(invalid_dates_by_column.keys())}.")

    # Currency symbols
    currency_symbol_columns = [col for col in df.columns if _has_currency_symbols(df[col])]
    if currency_symbol_columns:
        issues.append(f"Currency symbols found in columns: {currency_symbol_columns}. These will need cleaning.")

    # Whitespace issues
    whitespace_issue_columns = [col for col in df.columns if _has_whitespace_issues(df[col])]
    if whitespace_issue_columns:
        issues.append(f"Extra whitespace found in columns: {whitespace_issue_columns}.")

    # Mixed types
    mixed_type_columns = [col for col in df.columns if _has_mixed_types(df[col])]
    if mixed_type_columns:
        issues.append(f"Mixed data types found in columns: {mixed_type_columns}.")

    # Numeric column presence (required for financial analysis later)
    has_numeric_column = bool(df.select_dtypes(include="number").shape[1] > 0)
    if not has_numeric_column:
        issues.append("No numeric column detected — at least one amount/value column is required.")

    # A dataset is "ready" if it has rows, has a numeric column, and isn't fully empty.
    # (Missing values, currency symbols, whitespace, etc. are fixable in the Cleaning module,
    # so they don't block readiness on their own.)
    dataset_ready = rows_loaded > 0 and has_numeric_column and empty_rows < rows_loaded

    return QualityReport(
        rows_loaded=rows_loaded,
        total_columns=total_columns,
        missing_values_total=missing_values_total,
        missing_by_column=missing_by_column,
        duplicate_rows=duplicate_rows,
        empty_rows=empty_rows,
        date_columns_checked=date_columns_checked,
        invalid_dates_total=invalid_dates_total,
        invalid_dates_by_column=invalid_dates_by_column,
        currency_symbol_columns=currency_symbol_columns,
        whitespace_issue_columns=whitespace_issue_columns,
        mixed_type_columns=mixed_type_columns,
        has_numeric_column=has_numeric_column,
        dataset_ready=dataset_ready,
        issues=issues,
    )


def render_validation_page() -> None:
    """Render the Streamlit UI for Module 2: Dataset Validation."""
    st.subheader("🔍 Data Quality Report")

    df = st.session_state.get("raw_dataset")
    if df is None:
        st.warning("No dataset loaded yet. Please upload a dataset first on the 'Upload Dataset' page.")
        return

    # Avoid recomputing the full validation on every Streamlit rerun (e.g. sidebar
    # clicks) — only re-run it if the dataset itself has changed since last time.
    cache_key = (st.session_state.get("raw_dataset_name"), df.shape)
    if st.session_state.get("_quality_report_cache_key") == cache_key:
        report = st.session_state["quality_report"]
    else:
        report = run_validation(df)
        st.session_state["_quality_report_cache_key"] = cache_key

    # --- Top summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows Loaded", f"{report.rows_loaded:,}")
    col2.metric("Missing Values", f"{report.missing_values_total:,}")
    col3.metric("Duplicate Rows", f"{report.duplicate_rows:,}")
    col4.metric("Invalid Dates", f"{report.invalid_dates_total:,}")

    st.markdown("---")

    if report.dataset_ready:
        st.success("✅ Dataset Ready: Yes — this dataset can proceed to cleaning and analysis.")
    else:
        st.error("❌ Dataset Ready: No — please resolve the issues below before proceeding.")

    # --- Detailed report ---
    st.markdown("#### Detailed Findings")

    if report.missing_by_column:
        st.write("**Missing values by column:**")
        st.dataframe(
            pd.DataFrame(
                {"Column": list(report.missing_by_column.keys()), "Missing Count": list(report.missing_by_column.values())}
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("✅ No missing values found.")

    st.write(f"**Empty rows:** {report.empty_rows}")
    st.write(f"**Duplicate rows:** {report.duplicate_rows}")

    if report.date_columns_checked:
        st.write(f"**Date columns checked:** {', '.join(report.date_columns_checked)}")
        if report.invalid_dates_by_column:
            st.write("**Invalid dates by column:**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": list(report.invalid_dates_by_column.keys()),
                        "Invalid Count": list(report.invalid_dates_by_column.values()),
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.write("✅ No invalid dates found.")
    else:
        st.write("No date-like columns were detected (looked for names containing: date, dt, time, period).")

    st.write(
        "**Currency symbols found in:** "
        + (", ".join(report.currency_symbol_columns) if report.currency_symbol_columns else "None")
    )
    st.write(
        "**Extra whitespace found in:** "
        + (", ".join(report.whitespace_issue_columns) if report.whitespace_issue_columns else "None")
    )
    st.write(
        "**Mixed data types found in:** "
        + (", ".join(report.mixed_type_columns) if report.mixed_type_columns else "None")
    )
    st.write(f"**Numeric column present:** {'Yes' if report.has_numeric_column else 'No'}")

    if report.issues:
        st.markdown("#### Summary of Issues")
        for issue in report.issues:
            st.write(f"- {issue}")
    else:
        st.markdown("#### ✅ No issues found — this dataset is clean and ready.")

    # Store the report in session state for later modules (cleaning, history, etc.)
    st.session_state["quality_report"] = report
