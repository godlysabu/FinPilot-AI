# FinPilot AI — AI-Powered Financial Analysis Dashboard

A fully free, open-source, offline-capable financial analysis dashboard built with Streamlit.

## Status: Module 1 — Dataset Upload ✅

## Setup

```bash
cd FinPilot_AI
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Try it

A sample dataset is included at `data/sample_transactions.csv` — upload it on the
"Upload Dataset" page to see the preview in action.

## Project Structure

```
FinPilot_AI/
├── app.py                  # Main Streamlit entry point + sidebar navigation
├── requirements.txt
├── modules/
│   └── upload.py           # Module 1: CSV/Excel upload, validation of file type, preview
├── data/
│   └── sample_transactions.csv
├── uploads/                # Uploaded files get saved here at runtime
└── assets/
```

## Roadmap (build order)

1. ✅ Upload CSV/Excel
2. ✅ Dataset Validation (data quality report)
3. ✅ Automatic Data Cleaning
4. ✅ Financial Analysis (KPIs)
5. ✅ Interactive Dashboard (Plotly)
6. ✅ Forecasting (Prophet)
7. ✅ AI Insights (Ollama + Qwen 3)
8. ⬜ PDF & Excel Report Export
9. ⬜ History (SQLite)

## Tech Stack (100% free & open-source)

Streamlit · Pandas · NumPy · Plotly · Prophet · Scikit-learn · Ollama · Qwen 3 · SQLite · ReportLab · OpenPyXL
