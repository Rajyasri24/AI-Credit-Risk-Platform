import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from src.ml.evaluate import (
    build_surrogate_tree,
    calculate_band_performance,
    calculate_global_shap,
    evaluate_model,
    select_operating_threshold,
)

from src.utils.config import (
    ANALYTICAL_DATA,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SURROGATE_PATH,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


NUMERIC_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "EMPLOYMENT_AGE_RATIO",
    "BUREAU_CREDIT_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_OVERDUE_COUNT",
    "BUREAU_CREDIT_SUM",
    "BUREAU_DEBT_SUM",
    "BUREAU_OVERDUE_SUM",
    "INSTALLMENT_COUNT",
    "AVG_PAYMENT_DELAY",
    "MAX_PAYMENT_DELAY",
    "LATE_PAYMENT_RATE",
    "AVG_PAYMENT_RATIO",
]


CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]


MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def build_preprocessor():
    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def split_data(df):
    X = df[
        MODEL_FEATURES
    ]

    y = df[
        "TARGET"
    ]

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    (
        X_train,
        X_cal,
        y_train,
        y_cal,
    ) = train_test_split(
        X_dev,
        y_dev,
        test_size=0.125,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_cal,
        X_test,
        y_train,
        y_cal,
        y_test,
    )


def calibrated_probability(
    model,
    calibrator,
    X,
):
    margins = model.predict(
        X,
        output_margin=True,
    )

    return (
        calibrator
        .predict_proba(
            margins.reshape(
                -1,
                1,
            )
        )[:, 1]
    )


def train():
    if not ANALYTICAL_DATA.exists():
        raise FileNotFoundError(
            "Run preprocessing first."
        )

    df = pd.read_parquet(
        ANALYTICAL_DATA
    )

    (
        X_train,
        X_cal,
        X_test,
        y_train,
        y_cal,
        y_test,
    ) = split_data(df)

    logger.info(
        "Train=%s | Calibration=%s | Test=%s",
        len(X_train),
        len(X_cal),
        len(X_test),
    )

    preprocessor = (
        build_preprocessor()
    )

    X_train_t = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_cal_t = (
        preprocessor.transform(
            X_cal
        )
    )

    X_test_t = (
        preprocessor.transform(
            X_test
        )
    )

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        scale_pos_weight=(
            scale_pos_weight
        ),
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info(
        "Training XGBoost."
    )

    model.fit(
        X_train_t,
        y_train,
    )

    calibration_margin = (
        model.predict(
            X_cal_t,
            output_margin=True,
        )
    )

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    calibrator.fit(
        calibration_margin.reshape(
            -1,
            1,
        ),
        y_cal,
    )

    calibration_pd = (
        calibrated_probability(
            model,
            calibrator,
            X_cal_t,
        )
    )

    operating_threshold = (
        select_operating_threshold(
            y_cal,
            calibration_pd,
        )
    )

    low_threshold = float(
        np.quantile(
            calibration_pd,
            1 / 3,
        )
    )

    medium_threshold = float(
        np.quantile(
            calibration_pd,
            2 / 3,
        )
    )

    test_pd = (
        calibrated_probability(
            model,
            calibrator,
            X_test_t,
        )
    )

    metrics = evaluate_model(
        y_test,
        test_pd,
        operating_threshold,
    )

    band_performance = (
        calculate_band_performance(
            y_test,
            test_pd,
            low_threshold,
            medium_threshold,
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    global_shap = (
        calculate_global_shap(
            model,
            X_cal_t,
            feature_names,
        )
    )

    (
        surrogate,
        surrogate_info,
    ) = build_surrogate_tree(
        X_cal_t,
        X_test_t,
        calibration_pd,
        test_pd,
        feature_names,
        global_shap,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    joblib.dump(
        surrogate,
        SURROGATE_PATH,
    )

    metadata = {
        "model_features": (
            MODEL_FEATURES
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "calibrator": calibrator,
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "operating_threshold": (
            operating_threshold
        ),
        "risk_thresholds": {
            "low_max": low_threshold,
            "medium_max": (
                medium_threshold
            ),
        },
        "metrics": metrics,
        "band_performance": (
            band_performance
        ),
        "global_shap": (
            global_shap
        ),
        "global_surrogate_tree": (
            surrogate_info
        ),
        "split": {
            "train": len(X_train),
            "calibration": len(X_cal),
            "test": len(X_test),
        },
    }

    joblib.dump(
        metadata,
        METADATA_PATH,
    )

    print("\nMODEL TRAINING COMPLETE")
    print("=" * 50)

    print(
        f"Operating threshold: "
        f"{operating_threshold:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"PR-AUC lift: "
        f"{metrics['pr_auc_lift']:.2f}x"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Brier score: "
        f"{metrics['brier_calibrated']:.4f}"
    )

    print(
        "\nTEST-SET RISK BAND VALIDATION"
    )

    print(
        pd.DataFrame(
            band_performance
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    train()