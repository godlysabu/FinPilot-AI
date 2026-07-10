"""
insights.py
-----------
Module 7: AI Financial Insights

Uses a locally running Ollama server (with a model like Qwen 3) to turn
the KPIs computed in Module 4 — and, if available, the forecast from
Module 6 — into plain-language business insights and recommendations.

Requires Ollama to be installed and running on the user's machine:
  1. Install Ollama: https://ollama.com
  2. Pull a model:   ollama pull qwen3:4b
  3. Ollama runs a local server automatically (default: http://localhost:11434)

No API key, subscription, or internet connection is needed once the
model has been downloaded once.
"""

from __future__ import annotations

import streamlit as st

from modules.kpi import KPIResult, ColumnMapping, detect_columns, compute_kpis
from ai.ollama_client import (
    DEFAULT_HOST, DEFAULT_MODEL, check_ollama_available, list_available_models, generate_insight,
)
from ai.prompt import build_insights_prompt


def render_insights_page() -> None:
    """Render the Streamlit UI for Module 7: AI Financial Insights."""
    st.subheader("🤖 AI Financial Insights")
    st.caption("Powered by a local Ollama model (e.g. Qwen 3) — runs entirely on your machine, no API key needed.")

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

    # --- Connection settings ---
    with st.expander("⚙️ Ollama Connection Settings", expanded=False):
        host = st.text_input("Ollama host", value=DEFAULT_HOST)
        model = st.text_input("Model name", value=DEFAULT_MODEL, help="Must match a model you've pulled, e.g. `ollama pull qwen3:4b`")
        if st.button("Test connection"):
            status = check_ollama_available(host)
            if status.success:
                st.success("✅ Ollama is reachable.")
                available = list_available_models(host)
                if available:
                    st.write("Available models:", ", ".join(available))
                else:
                    st.warning("Connected, but no models found. Pull one with `ollama pull qwen3:4b`.")
            else:
                st.error(status.error)

    # --- Optional: include forecast context if the user has run one ---
    forecast_metric = st.session_state.get("forecast_metric")
    forecast = st.session_state.get("forecast_result")
    forecast_total = None
    forecast_horizon_label = None
    include_forecast = False

    if forecast is not None and forecast_metric is not None:
        include_forecast = st.checkbox(
            f"Include the {forecast_metric} forecast from the Forecast page in the analysis", value=True
        )
        if include_forecast:
            # forecast_result stores the full history+future frame; approximate the
            # future-only total using the cached forecast/metric from session_state.
            history_len = st.session_state.get("_forecast_cache_key", (None, None))[1]
            if history_len is not None:
                future_part = forecast.iloc[history_len:]
                forecast_total = float(future_part["yhat"].sum())
                periods = future_part.shape[0]
                forecast_horizon_label = f"{periods} days"

    st.markdown("---")

    if st.button("✨ Generate AI Insights", type="primary"):
        prompt = build_insights_prompt(
            kpis,
            forecast_metric=forecast_metric if include_forecast else None,
            forecast_total=forecast_total,
            forecast_horizon_label=forecast_horizon_label,
        )

        with st.spinner(f"Asking {model} to analyze your financials... this may take a moment."):
            result = generate_insight(prompt, model=model, host=host)

        if result.success:
            st.session_state["ai_insights_text"] = result.text
        else:
            st.error(f"Could not generate insights: {result.error}")
            with st.expander("Setup help"):
                st.markdown(
                    "1. Install Ollama from **https://ollama.com**\n"
                    "2. Open a terminal and run: `ollama pull qwen3:4b`\n"
                    "3. Ollama runs its local server automatically — no need to start anything else\n"
                    "4. Come back here and click **Generate AI Insights** again"
                )

    if st.session_state.get("ai_insights_text"):
        st.markdown("#### 💡 Insights & Recommendations")
        st.markdown(st.session_state["ai_insights_text"])

        with st.expander("View the prompt sent to the model"):
            st.code(
                build_insights_prompt(
                    kpis,
                    forecast_metric=forecast_metric if include_forecast else None,
                    forecast_total=forecast_total,
                    forecast_horizon_label=forecast_horizon_label,
                ),
                language="text",
            )
