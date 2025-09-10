<div align="center">

# 📊 ProductPulse  
**Full-stack product-analytics stack — Python · Streamlit · SQLite**

![Dashboard screenshot](docs/screenshot_dashboard.png)

</div>

---

## ✨  Key Features
| Area | Details |
|------|---------|
| **ETL pipeline** | `src/etl.py` — ingests raw CSV / API dumps → cleans & loads into SQLite + Parquet |
| **Synthetic data generator** | `scripts/generate_dummy_data.py` — Faker-based script that fabricates 1 000 users, 3 – 5 k orders, ~10 k events |
| **Dashboard (Streamlit + Plotly)** | KPI cards (revenue, active users) · RFM segmentation bar-chart · **Weekly cohort-retention heat-map** · 1-click churn-risk model (GradientBoosting, ROC-AUC ≈ 0.8) |
| **Testing & CI** | Pytest suite · GitHub Actions workflow (`.github/workflows/ci.yml`) runs tests on every push |
| **One-click deploy** | Works out-of-the-box on Streamlit Community Cloud or Render.com |
| **One-command local spin-up** | Clone → `generate_dummy_data.py` (optional) → `python -m src.etl` → Streamlit run |

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/<your-handle>/productpulse.git
cd productpulse

# 2. Create & activate virtual-env
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

# 3. Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Generate synthetic demo data
python scripts/generate_dummy_data.py

# 5. Run ETL
python -m src.etl

# 6. Launch Streamlit dashboard
set PYTHONPATH=%cd%               # Windows
# export PYTHONPATH=$PWD          # macOS / Linux
streamlit run src/dashboards/app.py
