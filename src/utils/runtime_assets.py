from pathlib import Path
import os
import gdown


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ANALYTICAL_DATA = DATA_DIR / "credit_applicants.parquet"
DATABASE_PATH = DATA_DIR / "credit_risk.db"


def ensure_runtime_assets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parquet_id = os.getenv("PARQUET_DRIVE_FILE_ID")
    database_id = os.getenv("DATABASE_DRIVE_FILE_ID")

    if not ANALYTICAL_DATA.exists():
        if not parquet_id:
            raise RuntimeError("PARQUET_DRIVE_FILE_ID is not configured.")

        gdown.download(
            id=parquet_id,
            output=str(ANALYTICAL_DATA),
            quiet=False,
        )

    if not DATABASE_PATH.exists():
        if not database_id:
            raise RuntimeError("DATABASE_DRIVE_FILE_ID is not configured.")

        gdown.download(
            id=database_id,
            output=str(DATABASE_PATH),
            quiet=False,
        )