import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from src.ml.evaluate import (
    calculate_global_shap,
    build_global_surrogate_tree,
)

from src.utils.config import (
    ANALYTICAL_DATA,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
    SURROGATE_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


NUMERIC_FEATURES = [
    # Financial profile
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",

    # Household profile
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",

    # Employment and demographic context
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",

    # External credit indicators
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    # Engineered financial ratios
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "EMPLOYMENT_AGE_RATIO",

    # Bureau credit history
    "BUREAU_CREDIT_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_OVERDUE_COUNT",
    "BUREAU_CREDIT_SUM",
    "BUREAU_DEBT_SUM",
    "BUREAU_OVERDUE_SUM",

    # Historical repayment behaviour
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


def split_data(df):
    X = df[
        MODEL_FEATURES
    ].copy()

    y = df[
        "TARGET"
    ].copy()

    # Reserve a completely untouched 20% final test set.
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

    # 12.5% of the remaining 80% = 10% of full data.
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


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
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
        ],
        remainder="drop",
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

    return calibrator.predict_proba(
        margins.reshape(-1, 1)
    )[:, 1]


def assign_band(
    probability,
    low_threshold,
    medium_threshold,
):
    if probability <= low_threshold:
        return "Low"

    if probability <= medium_threshold:
        return "Medium"

    return "High"


def calculate_band_performance(
    y_true,
    probability,
    low_threshold,
    medium_threshold,
):
    evaluation = pd.DataFrame(
        {
            "TARGET": np.asarray(y_true),
            "PD": probability,
        }
    )

    evaluation["RISK_BAND"] = [
        assign_band(
            p,
            low_threshold,
            medium_threshold,
        )
        for p in evaluation["PD"]
    ]

    summary = (
        evaluation
        .groupby("RISK_BAND")
        .agg(
            Applicants=(
                "TARGET",
                "size",
            ),
            Observed_Default_Rate=(
                "TARGET",
                "mean",
            ),
            Average_PD=(
                "PD",
                "mean",
            ),
        )
        .reindex(
            [
                "Low",
                "Medium",
                "High",
            ]
        )
    )

    summary[
        "Observed_Default_Rate"
    ] *= 100

    summary[
        "Average_PD"
    ] *= 100

    return (
        summary
        .round(3)
        .reset_index()
        .to_dict(
            orient="records"
        )
    )


def train():
    logger.info(
        "Loading analytical dataset."
    )

    if not ANALYTICAL_DATA.exists():
        raise FileNotFoundError(
            "credit_applicants.parquet was not found. "
            "Run preprocessing first:\n"
            "python -m src.data.preprocessor"
        )

    df = pd.read_parquet(
        ANALYTICAL_DATA
    )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Required model features are missing: "
            f"{missing_features}"
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

    logger.info(
        "Train default rate=%.4f | "
        "Calibration default rate=%.4f | "
        "Test default rate=%.4f",
        y_train.mean(),
        y_cal.mean(),
        y_test.mean(),
    )

    preprocessor = (
        build_preprocessor()
    )

    logger.info(
        "Fitting preprocessing pipeline."
    )

    X_train_t = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    X_cal_t = (
        preprocessor
        .transform(
            X_cal
        )
    )

    X_test_t = (
        preprocessor
        .transform(
            X_test
        )
    )

    negative = int(
        (y_train == 0).sum()
    )

    positive = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative / positive
    )

    logger.info(
        "scale_pos_weight=%.3f",
        scale_pos_weight,
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info(
        "Training XGBoost model."
    )

    model.fit(
        X_train_t,
        y_train,
    )

    # Dedicated 10% calibration set.
    logger.info(
        "Fitting Platt probability calibrator."
    )

    calibration_margin = model.predict(
        X_cal_t,
        output_margin=True,
    )

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    calibrator.fit(
        calibration_margin.reshape(-1, 1),
        y_cal,
    )

    calibration_pd = (
        calibrator
        .predict_proba(
            calibration_margin.reshape(
                -1,
                1,
            )
        )[:, 1]
    )

    # Data-derived Low / Medium / High thresholds.
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

    logger.info(
        "Risk thresholds | "
        "Low <= %.4f | "
        "Medium <= %.4f | "
        "High > %.4f",
        low_threshold,
        medium_threshold,
    )

    # Final untouched test evaluation.
    raw_test_pd = (
        model.predict_proba(
            X_test_t
        )[:, 1]
    )

    test_pd = calibrated_probability(
        model,
        calibrator,
        X_test_t,
    )

    binary_prediction = (
        test_pd >= 0.50
    ).astype(int)

    metrics = {
        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_pd,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                test_pd,
            )
        ),
        "precision_at_0_5": float(
            precision_score(
                y_test,
                binary_prediction,
                zero_division=0,
            )
        ),
        "recall_at_0_5": float(
            recall_score(
                y_test,
                binary_prediction,
                zero_division=0,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                y_test,
                binary_prediction,
                zero_division=0,
            )
        ),
        "brier_raw": float(
            brier_score_loss(
                y_test,
                raw_test_pd,
            )
        ),
        "brier_calibrated": float(
            brier_score_loss(
                y_test,
                test_pd,
            )
        ),
        "log_loss_calibrated": float(
            log_loss(
                y_test,
                test_pd,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_test,
                binary_prediction,
            )
            .tolist()
        ),
    }

    band_performance = (
        calculate_band_performance(
            y_true=y_test,
            probability=test_pd,
            low_threshold=low_threshold,
            medium_threshold=medium_threshold,
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    logger.info(
        "Calculating global SHAP importance."
    )

    shap_summary = (
        calculate_global_shap(
            model=model,
            X=X_test_t,
            feature_names=feature_names,
            sample_size=2000,
        )
    )

    logger.info(
        "Training Global Surrogate Decision Tree."
    )

    (
        surrogate_model,
        surrogate_information,
    ) = build_global_surrogate_tree(
        X_cal=X_cal_t,
        X_test=X_test_t,
        calibration_pd=calibration_pd,
        test_pd=test_pd,
        feature_names=feature_names,
        shap_summary=shap_summary,
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
        surrogate_model,
        SURROGATE_PATH,
    )

    metadata = {
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,

        "calibrator": calibrator,

        "scale_pos_weight": float(
            scale_pos_weight
        ),

        "risk_thresholds": {
            "low_max": low_threshold,
            "medium_max": medium_threshold,
        },

        "metrics": metrics,

        "band_performance": (
            band_performance
        ),

        "global_shap": (
            shap_summary
        ),

        "global_surrogate_tree": (
            surrogate_information
        ),

        "split": {
            "train": int(
                len(X_train)
            ),
            "calibration": int(
                len(X_cal)
            ),
            "test": int(
                len(X_test)
            ),
        },
    }

    joblib.dump(
        metadata,
        METADATA_PATH,
    )

    print(
        "\nMODEL TRAINING COMPLETE"
    )

    print("=" * 60)

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Calibration rows: "
        f"{len(X_cal):,}"
    )

    print(
        f"Test rows: "
        f"{len(X_test):,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.3f}"
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
        f"Precision @ 0.5: "
        f"{metrics['precision_at_0_5']:.4f}"
    )

    print(
        f"Recall @ 0.5: "
        f"{metrics['recall_at_0_5']:.4f}"
    )

    print(
        f"F1 @ 0.5: "
        f"{metrics['f1_at_0_5']:.4f}"
    )

    print(
        f"Brier raw: "
        f"{metrics['brier_raw']:.4f}"
    )

    print(
        f"Brier calibrated: "
        f"{metrics['brier_calibrated']:.4f}"
    )

    print(
        "\nDATA-DERIVED RISK THRESHOLDS"
    )

    print(
        f"Low    : PD <= "
        f"{low_threshold:.4f}"
    )

    print(
        f"Medium : "
        f"{low_threshold:.4f} < "
        f"PD <= "
        f"{medium_threshold:.4f}"
    )

    print(
        f"High   : PD > "
        f"{medium_threshold:.4f}"
    )

    print(
        "\nTEST-SET BAND PERFORMANCE"
    )

    print(
        pd.DataFrame(
            band_performance
        ).to_string(
            index=False
        )
    )

    print(
        "\nGLOBAL SURROGATE DECISION TREE"
    )

    print(
        "Fidelity R²:",
        round(
            surrogate_information[
                "fidelity_r2"
            ],
            4,
        ),
    )

    print(
        "Fidelity MAE:",
        round(
            surrogate_information[
                "fidelity_mae"
            ],
            4,
        ),
    )

    print(
        "\nTOP GLOBAL SHAP FEATURES"
    )

    for item in (
        shap_summary[:10]
    ):
        print(
            f"- {item['feature']}: "
            f"{item['importance']:.6f}"
        )

    print(
        "\nSaved model artifacts:"
    )

    print(
        f"- {MODEL_PATH.name}"
    )

    print(
        f"- {PREPROCESSOR_PATH.name}"
    )

    print(
        f"- {METADATA_PATH.name}"
    )

    print(
        f"- {SURROGATE_PATH.name}"
    )


if __name__ == "__main__":
    train()