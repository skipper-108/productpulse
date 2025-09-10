"""
Very simple churn model:
• Label = 1 if recency > 90 days, else 0
• Features = recency, frequency, monetary
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_churn_model(rfm: pd.DataFrame):
    rfm = rfm.copy()
    rfm["churned"] = (rfm["recency"] > 90).astype(int)
    X = rfm[["recency", "frequency", "monetary"]]
    y = rfm["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    rfm["churn_prob"] = model.predict_proba(X)[:, 1]
    return model, auc, rfm[["user_id", "churn_prob"]].sort_values(
        "churn_prob", ascending=False
    )
