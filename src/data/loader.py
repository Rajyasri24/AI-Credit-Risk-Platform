from pathlib import Path

import pandas as pd

from src.utils.config import (
    APPLICATION_TRAIN,
    APPLICATION_TEST,
    BUREAU,
    INSTALLMENTS,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


REQUIRED_FILES = {
    "application_train.csv": APPLICATION_TRAIN,
    "application_test.csv": APPLICATION_TEST,
    "bureau.csv": BUREAU,
    "installments_payments.csv": INSTALLMENTS,
}


def validate_files():
    missing = [
        name
        for name, path in REQUIRED_FILES.items()
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required dataset files:\n- "
            + "\n- ".join(missing)
        )

    logger.info("All required Home Credit files found.")


def load_application_train():
    return pd.read_csv(APPLICATION_TRAIN)


def load_application_test():
    return pd.read_csv(APPLICATION_TEST)


def load_bureau():
    return pd.read_csv(BUREAU)


def load_installments():
    return pd.read_csv(INSTALLMENTS)