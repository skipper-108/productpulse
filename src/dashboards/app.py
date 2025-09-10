"""ProductPulse Streamlit dashboard."""
# ──────────────────────────────────────────────────────────────────────
# Make  import src.*  work everywhere
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]      # …/productpulse
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DB_PATH = ROOT / "data" / "app.db"                      # SQLite store

# ── build SQLite on first boot ───────────────────────────────────────
import runpy, importlib, streamlit as st

if not DB_PATH.exists():
    st.info("Creating local database – first-time setup ⏳")
    # 1) generate dummy CSVs
    runpy.run_path(str(ROOT / "scripts" / "generate_dummy_data.py"))
    # 2) run ETL job to populate SQLite
    importlib.import_module("src.etl").main()
    st.success("SQLite ready – continuing…")

# ── libs ─────────────────────────────────────────────────────────────
import plotly.express as px

# ── project modules ─────────────────────────────────────────────────
from src.analysis import rfm_segmentation, cohort_retention
import src.modeling as mdl

# ── page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="ProductPulse Dashboard", layout="wide")
st.title("📊 ProductPulse – Demo Dashboard")

# ── KPI cards ───────────────────────────────────────────────────────
rfm = rfm_segmentation()
st.metric("🛍️ Total Revenue", f"$ {rfm['monetary'].sum():,.0f}")
st.metric("👥 Active Users",   f"{len(rfm):,}")

# ── tabs ────────────────────────────────────────────────────────────
tab_rfm, tab_cohort = st.tabs(["RFM Segmentation", "Cohort Retention"])

# ── RFM tab ─────────────────────────────────────────────────────────
with tab_rfm:
    seg_counts = (rfm["segment"]
                  .value_counts()
                  .rename_axis("segment")
                  .reset_index(name="count"))
    st.plotly_chart(
        px.bar(seg_counts,
               x="segment",
               y="count",
               labels={"segment": "Segment", "count": "Users"}),
        use_container_width=True,
    )
    st.dataframe(rfm.head(50), use_container_width=True)

    if st.button("📉 Train churn model"):
        model, auc, probs = mdl.train_churn_model(rfm)
        st.success(f"Model trained · ROC-AUC = {auc:.3f}")
        st.dataframe(probs.head(20), use_container_width=True)

# ── Cohort tab ──────────────────────────────────────────────────────
with tab_cohort:
    retention = cohort_retention()
    retention_plot = retention.copy()
    retention_plot.index = retention_plot.index.astype("string")  # JSON-safe
    st.plotly_chart(
        px.imshow(
            retention_plot,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Week age",
                        y="Signup week",
                        color="Retention %"),
        ),
        use_container_width=True,
    )
