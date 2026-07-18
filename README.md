# FinPilot AI — AI-Powered Financial Analysis Dashboard

A fully free, open-source, offline-capable financial analysis dashboard built with Streamlit.

## Status: Modules 1–8 Complete ✅

Upload → Validate → Clean → Analyze → Dashboard → Forecast → AI Insights → Export Reports
is fully working end-to-end. Module 9 (History / SQLite) is in progress — the database
layer is built and tested, but not yet wired into the app UI.

## Setup

```bash
cd FinPilot_AI
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

### AI Insights (optional but recommended)

Module 7 (AI Insights) needs a local [Ollama](https://ollama.com) install:

```bash
ollama pull qwen3:4b
```

Ollama runs its own local server automatically — no API key, no internet required
once the model is pulled. Every other module works fine without this step.

## Try it

Sample datasets are included in `data/`:

- `sample_transactions.csv` — small, clean dataset, good for a first run
- `sample_messy.csv` — deliberately messy, good for testing Data Quality / Cleaning
- `sample_financial_150.csv` — 150 rows, 8 columns, realistic messiness (currency
  symbols, missing values, mixed casing, invalid dates, duplicates) — the main
  dataset used to test every module

Upload any of these on the "Upload Dataset" page to get started.

## Project Structure

```
FinPilot_AI/
├── app.py                     # Main Streamlit entry point + sidebar navigation
├── requirements.txt
├── modules/
│   ├── upload.py              # Module 1: CSV/Excel upload, preview
│   ├── validator.py           # Module 2: Data Quality Report
│   ├── cleaner.py             # Module 3: Automatic Data Cleaning
│   ├── kpi.py                 # Module 4: Financial Analysis (KPIs, column detection)
│   ├── charts.py              # Module 5: Interactive Dashboard (Plotly + themes)
│   ├── forecast.py            # Module 6: Forecasting (Prophet)
│   ├── insights.py            # Module 7: AI Insights (Ollama + Qwen 3)
│   └── reports.py             # Module 8: Reports page (PDF + Excel export UI)
├── ai/
│   ├── ollama_client.py       # Local Ollama HTTP client
│   └── prompt.py              # Builds the insights prompt from KPIs/forecast
├── report/
│   ├── pdf.py                 # PDF report generation (ReportLab + matplotlib charts)
│   └── excel.py               # Excel report generation (openpyxl, multi-sheet)
├── database/
│   └── database.py            # SQLite history layer (built, not yet wired into UI)
├── data/
│   ├── sample_transactions.csv
│   ├── sample_messy.csv
│   └── sample_financial_150.csv
├── uploads/                   # Uploaded files get saved here at runtime
└── assets/
```

## Roadmap (build order)

1. ✅ Upload CSV/Excel
2. ✅ Dataset Validation (data quality report)
3. ✅ Automatic Data Cleaning
4. ✅ Financial Analysis (KPIs)
5. ✅ Interactive Dashboard (Plotly, with theme selector)
6. ✅ Forecasting (Prophet)
7. ✅ AI Insights (Ollama + Qwen 3)
8. ✅ PDF & Excel Report Export
9. 🔄 History (SQLite) — database layer built, UI integration in progress

## Tech Stack (100% free & open-source)

Streamlit · Pandas · NumPy · Plotly · Prophet · Scikit-learn · Ollama · Qwen 3 · SQLite · ReportLab · OpenPyXL · Matplotlib
