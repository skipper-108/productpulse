import streamlit as st, plotly.express as px
from src.analysis import rfm_segmentation, cohort_retention

st.set_page_config(page_title="ProductPulse Dashboard", layout="wide")
st.title("📊 ProductPulse – Demo Dashboard")

# ─── KPI cards ────────────────────────────────────────────────────────────────
rfm = rfm_segmentation()
total_rev, n_users = rfm["monetary"].sum(), rfm.shape[0]
c1, c2 = st.columns(2)
c1.metric("🛍️ Total Revenue", f"$ {total_rev:,.0f}")
c2.metric("👥 Active Users", f"{n_users:,}")

# ─── Page Tabs ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["RFM Segmentation", "Cohort Retention"])

with tab1:
    seg_counts = (
        rfm["segment"].value_counts()
        .rename_axis("segment")
        .reset_index(name="count")
    )
    fig = px.bar(seg_counts, x="segment", y="count",
                 labels={"segment": "Segment", "count": "Users"})
    st.plotly_chart(fig, use_container_width=True)
    st.header("Raw RFM sample")
    st.dataframe(rfm.head(50), use_container_width=True)

with tab2:
    st.subheader("Weekly Retention Heat-map")

    # Compute retention table
    retention = cohort_retention()

    # Plotly/Streamlit can’t serialize PeriodIndex → cast to string
    retention_plot = retention.copy()
    retention_plot.index = retention_plot.index.astype("string")

    fig = px.imshow(
        retention_plot,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Week age", y="Signup week", color="Retention %"),
    )
    st.plotly_chart(fig, use_container_width=True)


import src.modeling as mdl
if st.button("📉 Train churn model"):
    model, auc, probs = mdl.train_churn_model(rfm)
    st.success(f"Model trained · ROC-AUC = {auc:.3f}")
    st.dataframe(probs.head(20), use_container_width=True)
