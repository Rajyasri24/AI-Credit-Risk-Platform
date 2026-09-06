from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
SQL_DIR = PROJECT_ROOT / "sql"

APPLICATION_TRAIN = DATA_DIR / "application_train.csv"
APPLICATION_TEST = DATA_DIR / "application_test.csv"
BUREAU = DATA_DIR / "bureau.csv"
INSTALLMENTS = DATA_DIR / "installments_payments.csv"

ANALYTICAL_DATA = DATA_DIR / "credit_applicants.parquet"
DATABASE_PATH = DATA_DIR / "credit_risk.db"

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"
SURROGATE_PATH = MODELS_DIR / "surrogate_model.joblib"

RANDOM_STATE = 42