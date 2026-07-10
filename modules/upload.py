"""
upload.py
----------
Module 1: Dataset Upload

Handles:
- Accepting CSV / XLSX file uploads from the user
- Reading the file safely into a Pandas DataFrame
- Displaying a preview: rows, columns, column names, data types
- Basic file-level error handling (corrupted, empty, wrong format)

This module is intentionally self-contained so it can be imported
by app.py and, later, by the validation/cleaning modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger("finpilot.upload")
logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _save_uploaded_file(uploaded_file) -> Path:
    """Persist the uploaded file to the local uploads/ folder and return its path."""
    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info("Saved uploaded file to %s", save_path)
    return save_path


def _read_dataset(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Read a CSV or Excel file into a DataFrame.

    Returns None (and shows a Streamlit error) if the file can't be read.
    """
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            st.error(f"Unsupported file type: {suffix}. Please upload a CSV or Excel file.")
            return None
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a file that contains data.")
        return None
    except pd.errors.ParserError:
        st.error("The file could not be parsed. It may be corrupted or not a valid CSV.")
        return None
    except Exception as exc:  # noqa: BLE001 - surface any unexpected read error to the user
        logger.exception("Failed to read uploaded file")
        st.error(f"Could not read the file: {exc}")
        return None

    if df.empty:
        st.error("The uploaded dataset has no rows. Please upload a non-empty dataset.")
        return None

    return df


def render_upload_page() -> Optional[pd.DataFrame]:
    """
    Render the Streamlit UI for dataset upload and return the loaded DataFrame
    (or None if nothing has been successfully uploaded yet).
    """
    st.subheader("📁 Upload Dataset")
    st.write("Upload a CSV or Excel file containing your financial transactions.")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Maximum recommended size: 100,000 rows for smooth performance.",
    )

    if uploaded_file is None:
        st.info("No file uploaded yet.")
        return None

    file_path = _save_uploaded_file(uploaded_file)
    df = _read_dataset(file_path)

    if df is None:
        return None

    st.success(f"✅ Dataset loaded successfully: **{uploaded_file.name}**")

    # --- Dataset overview ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("File Size", f"{file_path.stat().st_size / 1024:.1f} KB")

    st.markdown("#### Column Names & Data Types")
    dtype_df = pd.DataFrame(
        {"Column": df.columns, "Data Type": df.dtypes.astype(str).values}
    )
    st.dataframe(dtype_df, width="stretch", hide_index=True)

    st.markdown("#### Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10), width="stretch")

    # Store in session state so later modules (validation, cleaning, etc.) can access it
    st.session_state["raw_dataset"] = df
    st.session_state["raw_dataset_name"] = uploaded_file.name

    return df
