# Omni-Retail Revenue & Churn Decision Engine

End-to-end customer churn propensity, SQL window feature engineering, retention ROI simulation, and executive brief generator across 25,000 commercial accounts.

## Highlights
- **SQL Feature Mart (`SQLite`):** Window functions (`NTILE`, `SUM() OVER`) computing RFM scores, 90-day revenue growth %, and regional revenue share.
- **Churn Scoring Model (`scikit-learn`):** Gradient Boosting classifier (`ROC-AUC: 0.7906`, `PR-AUC: 0.3383`) quantifying expected INR revenue at risk.
- **Interactive Executive App (`app/dashboard.py`):** Filter by region, plan tier, and risk tier; simulate retention campaign ROI; and generate concise retention briefs via `google-genai` (`gemini-flash-lite-latest`).

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_pipeline.py
streamlit run app/dashboard.py
```
