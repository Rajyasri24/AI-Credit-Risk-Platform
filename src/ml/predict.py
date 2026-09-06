import joblib
import numpy as np
import pandas as pd
import shap

from src.utils.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
)


def load_artifacts():
    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    return (
        model,
        preprocessor,
        metadata,
    )


def assign_risk_band(
    probability,
    thresholds,
):
    if probability <= thresholds[
        "low_max"
    ]:
        return "Low"

    if probability <= thresholds[
        "medium_max"
    ]:
        return "Medium"

    return "High"


def clean_feature_name(name):
    name = name.replace(
        "numeric__",
        "",
    )

    name = name.replace(
        "categorical__",
        "",
    )

    name = name.replace(
        "missingindicator_",
        "Missing value indicator: ",
    )

    return (
        name
        .replace("_", " ")
        .strip()
    )


def predict_applicant(
    applicant,
):
    (
        model,
        preprocessor,
        metadata,
    ) = load_artifacts()

    if isinstance(
        applicant,
        pd.Series,
    ):
        applicant = (
            applicant
            .to_frame()
            .T
        )

    X = applicant[
        metadata["model_features"]
    ].copy()

    X_transformed = (
        preprocessor
        .transform(X)
    )

    raw_probability = (
        model
        .predict_proba(
            X_transformed
        )[0, 1]
    )

    margin = model.predict(
        X_transformed,
        output_margin=True,
    )

    probability = (
        metadata[
            "calibrator"
        ]
        .predict_proba(
            margin.reshape(
                -1,
                1,
            )
        )[0, 1]
    )

    risk_band = (
        assign_risk_band(
            probability,
            metadata[
                "risk_thresholds"
            ],
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    if hasattr(
        X_transformed,
        "toarray",
    ):
        shap_input = (
            X_transformed
            .toarray()
        )
    else:
        shap_input = np.asarray(
            X_transformed
        )

    explainer = (
        shap.TreeExplainer(
            model
        )
    )

    shap_values = np.asarray(
        explainer.shap_values(
            shap_input
        )
    )[0]

    top_indices = np.argsort(
        np.abs(
            shap_values
        )
    )[::-1][:5]

    drivers = []

    for index in top_indices:
        contribution = float(
            shap_values[index]
        )

        drivers.append(
            {
                "feature": clean_feature_name(
                    feature_names[
                        index
                    ]
                ),
                "shap_value": contribution,
                "direction": (
                    "Increases predicted risk"
                    if contribution > 0
                    else
                    "Reduces predicted risk"
                ),
            }
        )

    return {
        "raw_probability": float(
            raw_probability
        ),
        "default_probability": float(
            probability
        ),
        "risk_score": float(
            probability * 100
        ),
        "risk_band": risk_band,
        "drivers": drivers,
    }