from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

# Ensure processed dir exists
PROC_DIR.mkdir(parents=True, exist_ok=True)

DB_URI = os.getenv("PP_DB_URI", f"sqlite:///{(PROC_DIR / 'productpulse.db').as_posix()}")
