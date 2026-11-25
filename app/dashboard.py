import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Revenue & Churn Engine", layout="wide")
BASE = Path(__file__).resolve().parent.parent
df = pd.read_parquet(BASE / "artifacts/scored_customers.parquet")
metrics = json.loads((BASE / "artifacts/model_metrics.json").read_text())

st.title("Omni-Retail Revenue & Churn Decision Engine")
st.caption("SQL RFM Window Mart (25,000 Accounts) + Gradient Boosting Propensity + Retention ROI Simulator")

reg = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
plan = st.sidebar.multiselect("Plan Tier", sorted(df["plan_tier"].unique()), default=sorted(df["plan_tier"].unique()))
view = df[df["region"].isin(reg) & df["plan_tier"].isin(plan)]

total_rev = view["revenue_90d_inr"].sum()
total_risk = view["expected_revenue_at_risk_inr"].sum()
roc_auc = metrics["roc_auc"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered Accounts", f"{len(view):,}")
c2.metric("90d Revenue", f"INR {total_rev:,.0f}")
c3.metric("Revenue at Risk", f"INR {total_risk:,.0f}")
c4.metric("Model ROC-AUC", f"{roc_auc:.4f}")

save_rate = st.slider("Targeted Retention Save Rate (%)", 5, 50, 20)
cost_per_acct = st.number_input("Outreach Cost per High-Risk Account (INR)", min_value=100, max_value=25000, value=1500, step=100)
high_risk = view[view["risk_tier"] == "High"]
projected_saved = high_risk["expected_revenue_at_risk_inr"].sum() * (save_rate / 100.0)
campaign_cost = len(high_risk) * cost_per_acct
net_gain = projected_saved - campaign_cost
st.info(
    f"High-Risk Accounts: {len(high_risk):,} | "
    f"Projected Retained 90d Revenue: INR {projected_saved:,.0f} | "
    f"Campaign Cost: INR {campaign_cost:,.0f} | "
    f"Net Impact: INR {net_gain:,.0f}"
)

col_a, col_b = st.columns(2)
with col_a:
    by_plan = view.groupby("plan_tier", as_index=False)["expected_revenue_at_risk_inr"].sum()
    st.plotly_chart(px.bar(by_plan, x="plan_tier", y="expected_revenue_at_risk_inr", title="Revenue at Risk by Plan Tier (INR)"), use_container_width=True)
with col_b:
    feat_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in metrics["top_features"].items()])
    st.plotly_chart(px.bar(feat_df, x="importance", y="feature", orientation="h", title="Top Model Churn Drivers"), use_container_width=True)

if st.button("Generate Executive Retention Brief"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("Set GEMINI_API_KEY in .env to generate a live Gemini executive brief.")
    else:
        from google import genai
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        top_feats = metrics["top_features"]
        prompt = (
            f"Summarize in 3 crisp executive bullets for a VP of Revenue in India: "
            f"{len(high_risk)} high-risk accounts out of {len(view)}, "
            f"INR {total_risk:,.0f} revenue at risk, top drivers {top_feats}, "
            f"projected net impact INR {net_gain:,.0f} at {save_rate}% save rate."
        )
        resp = client.models.generate_content(model=model, contents=prompt)
        st.write(resp.text)
