"""ETL pipeline for ProductPulse."""

import pandas as pd
import sqlalchemy as sa
from .config import RAW_DIR, PROC_DIR, DB_URI

TABLE_MAP = {
    "orders": "orders.csv",
    "events": "events.csv",
    "ad_spend": "ad_spend.csv",
}

engine = sa.create_engine(DB_URI, echo=False)

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"\W+", "_", regex=True)
    )
    for c in df.select_dtypes("object"):
        if any(key in c for key in ("date", "time")):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

def run_etl() -> None:
    for name, file in TABLE_MAP.items():
        src_path = RAW_DIR / file
        if not src_path.exists():
            print(f"⚠️  {src_path} not found—skipping {name}")
            continue
        df = pd.read_csv(src_path)
        df = _standardize(df)
        df.to_parquet(PROC_DIR / f"{name}.parquet", index=False)
        df.to_sql(name, con=engine, if_exists="replace", index=False)
        print(f"✓ Loaded {name}  ({len(df):,} rows)")
    print("ETL complete.")

if __name__ == "__main__":
    run_etl()
