"""
reports.py
----------
Module 8: PDF & Excel Report Export

Streamlit UI that ties together report/pdf.py and report/excel.py,
pulling from whatever KPIs, forecast, and AI insights have already been
computed elsewhere in the session.
"""

from __future__ import annotations

import streamlit as st

from modules.kpi import KPIResult, ColumnMapping, detect_columns, compute_kpis
from report.pdf import generate_pdf_report
from report.excel import generate_excel_report


def render_reports_page() -> None:
    """Render the Streamlit UI for Module 8: Reports."""
    st.subheader("📄 Export Reports")

    df = st.session_state.get("cleaned_dataset")
    if df is None:
        df = st.session_state.get("raw_dataset")
    if df is None:
        st.warning("No dataset loaded yet. Please upload a dataset first on the 'Upload Dataset' page.")
        return

    mapping: ColumnMapping | None = st.session_state.get("column_mapping")
    kpis: KPIResult | None = st.session_state.get("kpi_result")
    if mapping is None or kpis is None:
        st.info("Using auto-detected columns — visit 'Financial Analysis' first to confirm or adjust the mapping.")
        mapping = detect_columns(df)
        kpis = compute_kpis(df, mapping)

    company_name = st.text_input("Company Name (shown on the report)", value=st.session_state.get("company_name", "My Company"))
    st.session_state["company_name"] = company_name

    # --- Pull in optional forecast + AI insights, if the user has generated them ---
    forecast_metric = st.session_state.get("forecast_metric")
    forecast_df = st.session_state.get("forecast_result")
    forecast_table = None
    forecast_horizon_label = None

    if forecast_df is not None and forecast_metric is not None:
        cache_key = st.session_state.get("_forecast_cache_key")
        if cache_key is not None:
            history_len = cache_key[1]
            future_part = forecast_df.iloc[history_len:][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            future_part.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
            future_part["Date"] = future_part["Date"].dt.strftime("%Y-%m-%d")
            for col in ["Predicted", "Lower Bound", "Upper Bound"]:
                future_part[col] = future_part[col].round(2)
            forecast_table = future_part
            forecast_horizon_label = f"{future_part.shape[0]} days"
        st.success(f"✅ Including {forecast_metric} forecast from the Forecast page.")
    else:
        st.info("No forecast found — visit 'Forecast' first if you'd like it included in the reports.")

    ai_insights_text = st.session_state.get("ai_insights_text")
    if ai_insights_text:
        st.success("✅ Including AI-generated insights from the AI Insights page.")
    else:
        st.info("No AI insights found — visit 'AI Insights' first if you'd like them included in the PDF report.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📕 PDF Report")
        st.caption("Company info, KPIs, charts, forecast summary, and AI insights.")
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Building PDF report..."):
                pdf_bytes = generate_pdf_report(
                    company_name=company_name,
                    kpis=kpis,
                    forecast_metric=forecast_metric,
                    forecast_table=forecast_table,
                    forecast_horizon_label=forecast_horizon_label,
                    ai_insights_text=ai_insights_text,
                )
            st.session_state["_pdf_report_bytes"] = pdf_bytes
            st.success("PDF report ready.")

        if st.session_state.get("_pdf_report_bytes"):
            st.download_button(
                "⬇️ Download PDF Report",
                data=st.session_state["_pdf_report_bytes"],
                file_name=f"{company_name.replace(' ', '_')}_FinPilot_Report.pdf",
                mime="application/pdf",
            )

    with col2:
        st.markdown("#### 📗 Excel Report")
        st.caption("Clean dataset, KPI summary, monthly & category breakdown, forecast sheet.")
        if st.button("Generate Excel Report", type="primary"):
            with st.spinner("Building Excel report..."):
                excel_bytes = generate_excel_report(
                    cleaned_df=df,
                    kpis=kpis,
                    forecast_table=forecast_table,
                    forecast_metric=forecast_metric,
                )
            st.session_state["_excel_report_bytes"] = excel_bytes
            st.success("Excel report ready.")

        if st.session_state.get("_excel_report_bytes"):
            st.download_button(
                "⬇️ Download Excel Report",
                data=st.session_state["_excel_report_bytes"],
                file_name=f"{company_name.replace(' ', '_')}_FinPilot_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
