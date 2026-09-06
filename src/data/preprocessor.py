import numpy as np
import pandas as pd

from src.data.loader import (
    validate_files,
    load_application_train,
    load_bureau,
    load_installments,
)

from src.utils.config import ANALYTICAL_DATA
from src.utils.logger import get_logger


logger = get_logger(__name__)


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def clean_application(app):
    df = app.copy()

    # Home Credit sentinel value, not a real employment duration.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        365243,
        np.nan,
    )

    df["AGE_YEARS"] = (
        abs(df["DAYS_BIRTH"]) / 365.25
    )

    df["EMPLOYMENT_YEARS"] = (
        abs(df["DAYS_EMPLOYED"]) / 365.25
    )

    df["CREDIT_INCOME_RATIO"] = safe_ratio(
        df["AMT_CREDIT"],
        df["AMT_INCOME_TOTAL"],
    )

    df["ANNUITY_INCOME_RATIO"] = safe_ratio(
        df["AMT_ANNUITY"],
        df["AMT_INCOME_TOTAL"],
    )

    df["CREDIT_GOODS_RATIO"] = safe_ratio(
        df["AMT_CREDIT"],
        df["AMT_GOODS_PRICE"],
    )

    df["EMPLOYMENT_AGE_RATIO"] = safe_ratio(
        df["EMPLOYMENT_YEARS"],
        df["AGE_YEARS"],
    )

    return df


def aggregate_bureau(bureau):
    df = bureau.copy()

    df["IS_ACTIVE_CREDIT"] = (
        df["CREDIT_ACTIVE"]
        .eq("Active")
        .astype(int)
    )

    df["IS_CLOSED_CREDIT"] = (
        df["CREDIT_ACTIVE"]
        .eq("Closed")
        .astype(int)
    )

    df["HAS_OVERDUE"] = (
        df["AMT_CREDIT_SUM_OVERDUE"]
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    bureau_agg = (
        df.groupby("SK_ID_CURR")
        .agg(
            BUREAU_CREDIT_COUNT=(
                "SK_ID_BUREAU",
                "count",
            ),
            BUREAU_ACTIVE_COUNT=(
                "IS_ACTIVE_CREDIT",
                "sum",
            ),
            BUREAU_CLOSED_COUNT=(
                "IS_CLOSED_CREDIT",
                "sum",
            ),
            BUREAU_OVERDUE_COUNT=(
                "HAS_OVERDUE",
                "sum",
            ),
            BUREAU_CREDIT_SUM=(
                "AMT_CREDIT_SUM",
                "sum",
            ),
            BUREAU_DEBT_SUM=(
                "AMT_CREDIT_SUM_DEBT",
                "sum",
            ),
            BUREAU_OVERDUE_SUM=(
                "AMT_CREDIT_SUM_OVERDUE",
                "sum",
            ),
        )
        .reset_index()
    )

    return bureau_agg


def aggregate_installments(installments):
    df = installments.copy()

    # Positive value means actual payment happened after due date.
    df["PAYMENT_DELAY_DAYS"] = (
        df["DAYS_ENTRY_PAYMENT"]
        - df["DAYS_INSTALMENT"]
    ).clip(lower=0)

    df["IS_LATE_PAYMENT"] = (
        df["PAYMENT_DELAY_DAYS"] > 0
    ).astype(int)

    df["PAYMENT_RATIO"] = safe_ratio(
        df["AMT_PAYMENT"],
        df["AMT_INSTALMENT"],
    )

    installments_agg = (
        df.groupby("SK_ID_CURR")
        .agg(
            INSTALLMENT_COUNT=(
                "SK_ID_PREV",
                "count",
            ),
            AVG_PAYMENT_DELAY=(
                "PAYMENT_DELAY_DAYS",
                "mean",
            ),
            MAX_PAYMENT_DELAY=(
                "PAYMENT_DELAY_DAYS",
                "max",
            ),
            LATE_PAYMENT_RATE=(
                "IS_LATE_PAYMENT",
                "mean",
            ),
            AVG_PAYMENT_RATIO=(
                "PAYMENT_RATIO",
                "mean",
            ),
        )
        .reset_index()
    )

    return installments_agg


def build_analytical_dataset():
    validate_files()

    logger.info(
        "Building applicant-level analytical dataset."
    )

    app = load_application_train()
    bureau = load_bureau()
    installments = load_installments()

    app = clean_application(app)

    bureau_agg = aggregate_bureau(bureau)
    installments_agg = aggregate_installments(
        installments
    )

    df = app.merge(
        bureau_agg,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        installments_agg,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )

    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError(
            "Duplicate applicants detected after joins."
        )

    ANALYTICAL_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        ANALYTICAL_DATA,
        index=False,
    )

    logger.info(
        "Created analytical dataset: %s rows x %s columns",
        len(df),
        len(df.columns),
    )

    return df


if __name__ == "__main__":
    df = build_analytical_dataset()

    print("\nDataset created successfully")
    print("Shape:", df.shape)

    print(
        "Unique applicants:",
        df["SK_ID_CURR"].nunique(),
    )

    print(
        "Duplicate applicants:",
        df["SK_ID_CURR"]
        .duplicated()
        .sum(),
    )

    print(
        "Default rate:",
        round(
            df["TARGET"].mean() * 100,
            2,
        ),
        "%",
    )