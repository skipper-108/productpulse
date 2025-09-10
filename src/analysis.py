"""Reusable analytic helpers for ProductPulse."""
import pandas as pd
import numpy as np
import scipy.stats as st
from .config import DB_URI
import sqlalchemy as sa



engine = sa.create_engine(DB_URI, future=True)

def rfm_segmentation():
    """Return classic Recency/ Frequency/ Monetary segmentation."""
    orders = pd.read_sql(
        "SELECT user_id, order_id, order_date, revenue FROM orders", engine,
        parse_dates=["order_date"],
    )
    snapshot = orders["order_date"].max()
    rfm = (
        orders.groupby("user_id")
        .agg(
            recency=("order_date", lambda x: (snapshot - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )

    # Quantile cut‑points
    quantiles = rfm.quantile([0.33, 0.66]).to_dict()
    def r_score(x): return 3 - int(x > quantiles["recency"][0.66]) - int(x > quantiles["recency"][0.33])
    def fm_score(x, col): return 1 + int(x > quantiles[col][0.33]) + int(x > quantiles[col][0.66])

    rfm["R"] = rfm["recency"].apply(r_score)
    rfm["F"] = rfm["frequency"].apply(lambda x: fm_score(x, "frequency"))
    rfm["M"] = rfm["monetary"].apply(lambda x: fm_score(x, "monetary"))
    rfm["segment"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    return rfm

def ab_test_summary(test_name: str):
    """Basic two‑sample proportions z‑test for binary conversion."""
    q = f"SELECT group_id, converted FROM experiments WHERE test_name = :tn"
    df = pd.read_sql(q, engine, params={"tn": test_name})
    g = df.groupby("group_id")["converted"].agg(["sum", "count"])
    g["cr"] = g["sum"] / g["count"]
    lift = g["cr"].diff().iloc[-1]
    z = st.norm.ppf(0.975)
    se = np.sqrt(
        g["cr"].iloc[0] * (1 - g["cr"].iloc[0]) / g["count"].iloc[0]
        + g["cr"].iloc[1] * (1 - g["cr"].iloc[1]) / g["count"].iloc[1]
    )
    pval = 2 * (1 - st.norm.cdf(abs(lift) / se))
    return g.assign(lift=lift, p_value=pval)


# ───────────────────────────────── Cohort Retention ────────────────────────────
def cohort_retention(events_df=None, freq="W"):
    """
    Returns a pivot table: cohort_week × age_week → retention %
    events_df: optional pre-loaded events DataFrame. If None, pulls from DB.
    """
    import pandas as pd
    if events_df is None:
        events_df = pd.read_sql("SELECT user_id, event_time FROM events", engine,
                                parse_dates=["event_time"])
    # signup = first event_time per user
    events_df["signup_week"] = events_df.groupby("user_id")["event_time"].transform(
        "min"
    ).dt.to_period(freq)
    events_df["event_week"] = events_df["event_time"].dt.to_period(freq)
    events_df["age"] = (
    (events_df["event_week"].dt.start_time - events_df["signup_week"].dt.start_time)
    .dt.days // 7
)

    ct = (
        events_df
        .drop_duplicates(subset=["user_id", "age"])
        .groupby(["signup_week", "age"])["user_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    retention = ct.divide(ct.iloc[:, 0], axis=0).round(3)  # % of cohort that returns
    return retention
