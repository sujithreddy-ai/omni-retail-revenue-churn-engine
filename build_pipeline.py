import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DB_PATH = Path("data/commerce_analytics.db")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def main() -> None:
    rng = np.random.default_rng(42)
    n = 25000
    plans = rng.choice(["Starter", "Growth", "Scale", "Enterprise"], n, p=[0.42, 0.33, 0.17, 0.08])
    region = rng.choice(["South", "West", "North", "East"], n, p=[0.36, 0.28, 0.22, 0.14])
    channel = rng.choice(["Organic", "Paid Search", "Partner", "Outbound"], n, p=[0.35, 0.30, 0.20, 0.15])
    tenure = rng.integers(1, 37, n)
    orders_90d = rng.poisson(lam=np.where(plans == "Enterprise", 11, np.where(plans == "Scale", 7, 3)))
    days_inactive = rng.integers(1, 120, n)
    tickets_90d = rng.poisson(lam=1.4, size=n)
    discount = np.round(rng.uniform(0.0, 0.30, n), 2)
    aov = np.where(plans == "Enterprise", 48000, np.where(plans == "Scale", 19000, np.where(plans == "Growth", 7500, 2800)))
    rev_90d = np.round(aov * np.maximum(orders_90d, 1) * (1 - discount * 0.5) * rng.uniform(0.85, 1.15, n), 2)
    prev_rev_90d = np.round(rev_90d * rng.uniform(0.70, 1.35, n), 2)
    logit = -2.1 + 0.028 * (days_inactive - 35) + 0.42 * np.maximum(tickets_90d - 1, 0) - 0.045 * tenure - 0.12 * np.minimum(orders_90d, 10) + np.where(rev_90d < prev_rev_90d * 0.8, 0.85, -0.25) + np.where(plans == "Enterprise", -0.55, 0.0)
    churned = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"customer_id": [f"CUST-{100000 + i}" for i in range(n)], "region": region, "plan_tier": plans, "acquisition_channel": channel, "tenure_months": tenure, "orders_90d": orders_90d, "days_since_last_order": days_inactive, "support_tickets_90d": tickets_90d, "discount_rate": discount, "revenue_90d_inr": rev_90d, "prev_90d_revenue_inr": prev_rev_90d, "churned": churned})
    if DB_PATH.exists(): DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("customer_metrics", conn, index=False)
        mart = pd.read_sql_query("WITH enriched AS (SELECT *, ROUND((revenue_90d_inr - prev_90d_revenue_inr) / NULLIF(prev_90d_revenue_inr, 0) * 100.0, 2) AS rev_growth_pct, NTILE(4) OVER (ORDER BY days_since_last_order DESC) AS r_q, NTILE(4) OVER (ORDER BY orders_90d ASC) AS f_q, NTILE(4) OVER (ORDER BY revenue_90d_inr ASC) AS m_q, ROUND(revenue_90d_inr * 100.0 / SUM(revenue_90d_inr) OVER (PARTITION BY region), 4) AS region_rev_share_pct FROM customer_metrics) SELECT *, (r_q + f_q + m_q) AS rfm_score FROM enriched", conn)
    X = pd.get_dummies(mart[["region", "plan_tier", "acquisition_channel", "tenure_months", "orders_90d", "days_since_last_order", "support_tickets_90d", "discount_rate", "revenue_90d_inr", "prev_90d_revenue_inr", "rev_growth_pct", "rfm_score"]], drop_first=True)
    y = mart["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=180, max_depth=4, learning_rate=0.06, random_state=42).fit(X_tr, y_tr)
    probs_te = clf.predict_proba(X_te)[:, 1]
    mart["churn_probability"] = np.round(clf.predict_proba(X)[:, 1], 4)
    mart["risk_tier"] = pd.cut(mart["churn_probability"], bins=[-0.01, 0.30, 0.60, 1.01], labels=["Low", "Medium", "High"])
    mart["expected_revenue_at_risk_inr"] = np.round(mart["revenue_90d_inr"] * mart["churn_probability"], 2)
    mart.to_parquet(ARTIFACTS / "scored_customers.parquet", index=False)
    metrics = {"total_customers": int(len(mart)), "baseline_churn_rate": round(float(y.mean()), 4), "roc_auc": round(float(roc_auc_score(y_te, probs_te)), 4), "pr_auc": round(float(average_precision_score(y_te, probs_te)), 4), "total_90d_revenue_inr": round(float(mart["revenue_90d_inr"].sum()), 2), "total_revenue_at_risk_inr": round(float(mart["expected_revenue_at_risk_inr"].sum()), 2), "top_features": pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(8).round(4).to_dict()}
    (ARTIFACTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
