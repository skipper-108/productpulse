"""ProductPulse Streamlit dashboard."""

# ── Make top-level `src.*` imports work on Streamlit Cloud ──────────
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ── Standard libs ───────────────────────────────────────────────────
import streamlit as st
import plotly.express as px

# ── Project modules ─────────────────────────────────────────────────
from src.analysis import rfm_segmentation, cohort_retention
import src.modeling as mdl

# ────────────────────────────────────────────────────────────────────
# Streamlit page config & title
st.set_page_config(page_title="ProductPulse Dashboard", layout="wide")
st.title("📊 ProductPulse – Demo Dashboard")

# ─── KPI cards ──────────────────────────────────────────────────────
rfm = rfm_segmentation()
total_rev = rfm["monetary"].sum()
n_users   = rfm.shape[0]

c1, c2 = st.columns(2)
c1.metric("🛍️ Total Revenue", f"$ {total_rev:,.0f}")
c2.metric("👥 Active Users",   f"{n_users:,}")

# ─── Tabs: RFM Segmentation · Cohort Retention ─────────────────────
tab_rfm, tab_cohort = st.tabs(["RFM Segmentation", "Cohort Retention"])

# ── Tab 1  RFM Segmentation ────────────────────────────────────────
with tab_rfm:
    seg_counts = (
        rfm["segment"]
        .value_counts()
        .rename_axis("segment")
        .reset_index(name="count")
    )
    fig_rfm = px.bar(
        seg_counts,
        x="segment", y="count",
        labels={"segment": "Segment", "count": "Users"},
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

    st.header("Raw RFM sample")
    st.dataframe(rfm.head(50), use_container_width=True)

    # ── Churn-model button ──────────────────────────────────────────
    if st.button("📉 Train churn model"):
        model, auc, probs = mdl.train_churn_model(rfm)
        st.success(f"Model trained · ROC-AUC = {auc:.3f}")
        st.dataframe(probs.head(20), use_container_width=True)

# ── Tab 2  Cohort Retention ────────────────────────────────────────
with tab_cohort:
    st.subheader("Weekly Retention Heat-map")

    retention = cohort_retention()
    # Plotly / Streamlit can’t JSON-serialize PeriodIndex
    retention_plot = retention.copy()
    retention_plot.index = retention_plot.index.astype("string")

    fig_cohort = px.imshow(
        retention_plot,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Week age", y="Signup week", color="Retention %")
    )
    st.plotly_chart(fig_cohort, use_container_width=True)
