"""
prompt.py
---------
Builds the prompt sent to the local LLM (via Ollama) to generate
business-language financial insights and recommendations.

Keeps prompt construction separate from the Ollama client so the prompt
wording can be iterated on independently of the networking code.
"""

from __future__ import annotations

from typing import Optional

from modules.kpi import KPIResult


def build_insights_prompt(
    kpis: KPIResult,
    forecast_metric: Optional[str] = None,
    forecast_total: Optional[float] = None,
    forecast_horizon_label: Optional[str] = None,
) -> str:
    """
    Build a structured prompt summarizing the KPIs (and optionally a forecast)
    so the model can generate grounded, specific insights rather than generic
    financial platitudes.
    """
    monthly_rev = kpis.monthly_revenue
    monthly_exp = kpis.monthly_expenses

    monthly_lines = []
    for month in monthly_rev.index:
        monthly_lines.append(
            f"  - {month}: Revenue ₹{monthly_rev[month]:,.0f}, Expenses ₹{monthly_exp[month]:,.0f}"
        )
    monthly_block = "\n".join(monthly_lines) if monthly_lines else "  (no monthly breakdown available)"

    top_expense_lines = [
        f"  - {cat}: ₹{amount:,.0f}" for cat, amount in kpis.top_spending_categories.items()
    ] or ["  (no category data available)"]

    top_revenue_lines = [
        f"  - {cat}: ₹{amount:,.0f}" for cat, amount in kpis.revenue_by_category.head(5).items()
    ] or ["  (no category data available)"]

    forecast_block = ""
    if forecast_metric and forecast_total is not None and forecast_horizon_label:
        forecast_block = (
            f"\nForecast:\n"
            f"  - Predicted {forecast_metric} for the next {forecast_horizon_label}: ₹{forecast_total:,.0f}\n"
        )

    prompt = f"""You are a professional financial analyst. Based on the following business
financial data, write 5 to 8 concise, specific insights in professional business
language, followed by 2 to 3 actionable recommendations.

Financial Summary:
  - Total Revenue: ₹{kpis.total_revenue:,.0f}
  - Total Expenses: ₹{kpis.total_expenses:,.0f}
  - Net Profit: ₹{kpis.net_profit:,.0f}
  - Profit Margin: {kpis.profit_margin:.1f}%
  - Average Transaction: ₹{kpis.average_transaction:,.0f}
  - Highest Expense: ₹{kpis.highest_expense:,.0f}
  - Lowest Expense: ₹{kpis.lowest_expense:,.0f}

Monthly Revenue vs Expenses:
{monthly_block}

Top Spending Categories:
{chr(10).join(top_expense_lines)}

Top Revenue Categories:
{chr(10).join(top_revenue_lines)}
{forecast_block}
Instructions:
  - Reference specific numbers from the data above wherever possible (e.g. "Revenue increased by X%", "Category Y contributes Z% of total expenses").
  - Keep each insight to 1-2 sentences.
  - Avoid generic advice not grounded in the numbers provided.
  - End with a short "Recommendations" section (2-3 bullet points).
  - Do not repeat the raw numbers verbatim in a list — synthesize them into insights.
"""
    return prompt
