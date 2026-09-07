import joblib
import numpy as np
import pandas as pd
import shap

from src.ml.evaluate import (
    readable_feature_name,
)

from src.utils.config import (
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)


def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(
            PREPROCESSOR_PATH
        ),
        joblib.load(
            METADATA_PATH
        ),
    )


def get_risk_band(
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
            applicant.to_frame().T
        )

    elif isinstance(
        applicant,
        dict,
    ):
        applicant = pd.DataFrame(
            [applicant]
        )

    X = applicant[
        metadata[
            "model_features"
        ]
    ]

    X_t = preprocessor.transform(
        X
    )

    margin = model.predict(
        X_t,
        output_margin=True,
    )

    probability = float(
        metadata[
            "calibrator"
        ]
        .predict_proba(
            margin.reshape(-1, 1)
        )[0, 1]
    )

    risk_band = get_risk_band(
        probability,
        metadata[
            "risk_thresholds"
        ],
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    X_shap = (
        X_t.toarray()
        if hasattr(
            X_t,
            "toarray",
        )
        else np.asarray(X_t)
    )

    values = (
        shap.TreeExplainer(
            model
        )
        .shap_values(
            X_shap
        )
    )

    if isinstance(
        values,
        list,
    ):
        values = values[-1]

    if values.ndim == 3:
        values = values[
            :,
            :,
            -1,
        ]

    values = values[0]

    top_indices = np.argsort(
        np.abs(values)
    )[::-1][:5]

    drivers = [
        {
            "feature": (
                readable_feature_name(
                    feature_names[index]
                )
            ),
            "shap_value": float(
                values[index]
            ),
            "direction": (
                "Increases risk"
                if values[index] > 0
                else "Reduces risk"
            ),
        }
        for index
        in top_indices
    ]

    return {
        "default_probability": (
            probability
        ),
        "risk_score": (
            probability * 100
        ),
        "risk_band": (
            risk_band
        ),
        "drivers": (
            drivers
        ),
    }