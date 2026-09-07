import numpy as np
import pandas as pd
import shap

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor, _tree

from src.utils.config import RANDOM_STATE


def readable_feature_name(name):
    name = (
        name.replace("numeric__", "")
        .replace("categorical__", "")
    )

    if name.startswith("missingindicator_"):
        return (
            "Missing value indicator: "
            + name.replace(
                "missingindicator_",
                "",
            ).replace("_", " ")
        )

    return name.replace("_", " ")


def select_operating_threshold(
    y_true,
    probabilities,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(
        np.argmax(f1_scores)
    )

    return float(
        thresholds[best_index]
    )


def evaluate_model(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    prevalence = float(
        np.mean(y_true)
    )

    pr_auc = float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": pr_auc,
        "default_prevalence": prevalence,
        "pr_auc_lift": float(
            pr_auc / prevalence
        ),
        "operating_threshold": float(
            threshold
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "brier_calibrated": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                predictions,
            ).tolist()
        ),
    }


def assign_risk_band(
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
    probabilities,
    low_threshold,
    medium_threshold,
):
    result = pd.DataFrame(
        {
            "TARGET": np.asarray(
                y_true
            ),
            "MODEL_PD": probabilities,
        }
    )

    result["RISK_BAND"] = [
        assign_risk_band(
            p,
            low_threshold,
            medium_threshold,
        )
        for p in probabilities
    ]

    summary = (
        result.groupby(
            "RISK_BAND"
        )
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
                "MODEL_PD",
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
        .reset_index()
    )

    summary[
        "Observed_Default_Rate"
    ] *= 100

    summary[
        "Average_PD"
    ] *= 100

    return (
        summary.round(3)
        .to_dict(
            orient="records"
        )
    )


def calculate_global_shap(
    model,
    X,
    feature_names,
    sample_size=2000,
):
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if X.shape[0] > sample_size:
        indices = rng.choice(
            X.shape[0],
            sample_size,
            replace=False,
        )
        X = X[indices]

    if hasattr(X, "toarray"):
        X = X.toarray()

    explainer = shap.TreeExplainer(
        model
    )

    values = explainer.shap_values(
        X
    )

    if isinstance(values, list):
        values = values[-1]

    if values.ndim == 3:
        values = values[:, :, -1]

    importance = np.mean(
        np.abs(values),
        axis=0,
    )

    summary = pd.DataFrame(
        {
            "encoded_feature": (
                feature_names
            ),
            "feature": [
                readable_feature_name(
                    name
                )
                for name
                in feature_names
            ],
            "importance": importance,
        }
    )

    return (
        summary.sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
        .to_dict(
            orient="records"
        )
    )


def _extract_rules(
    model,
    feature_names,
):
    tree = model.tree_
    rules = []

    def walk(node, conditions):
        if (
            tree.feature[node]
            != _tree.TREE_UNDEFINED
        ):
            feature = feature_names[
                tree.feature[node]
            ]

            threshold = (
                tree.threshold[node]
            )

            walk(
                tree.children_left[node],
                conditions
                + [
                    f"{feature} <= "
                    f"{threshold:.3f}"
                ],
            )

            walk(
                tree.children_right[node],
                conditions
                + [
                    f"{feature} > "
                    f"{threshold:.3f}"
                ],
            )

        else:
            rules.append(
                {
                    "conditions": conditions,
                    "predicted_pd": float(
                        tree.value[node]
                        .reshape(-1)[0]
                    ),
                    "samples": int(
                        tree.n_node_samples[
                            node
                        ]
                    ),
                }
            )

    walk(0, [])

    return sorted(
        rules,
        key=lambda x: x["samples"],
        reverse=True,
    )


def build_surrogate_tree(
    X_cal,
    X_test,
    calibration_pd,
    test_pd,
    feature_names,
    shap_summary,
    top_n=8,
):
    top_features = [
        item["encoded_feature"]
        for item
        in shap_summary[:top_n]
    ]

    index_map = {
        name: i
        for i, name
        in enumerate(feature_names)
    }

    selected_indices = [
        index_map[name]
        for name
        in top_features
    ]

    selected_names = [
        readable_feature_name(
            feature_names[i]
        )
        for i
        in selected_indices
    ]

    def select_columns(X):
        X_selected = X[
            :,
            selected_indices
        ]

        if hasattr(
            X_selected,
            "toarray",
        ):
            X_selected = (
                X_selected.toarray()
            )

        return X_selected

    X_cal_selected = (
        select_columns(X_cal)
    )

    X_test_selected = (
        select_columns(X_test)
    )

    surrogate = (
        DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=max(
                200,
                int(
                    len(calibration_pd)
                    * 0.01
                ),
            ),
            random_state=(
                RANDOM_STATE
            ),
        )
    )

    surrogate.fit(
        X_cal_selected,
        calibration_pd,
    )

    predicted = surrogate.predict(
        X_test_selected
    )

    info = {
        "selected_features": (
            selected_names
        ),
        "fidelity_r2": float(
            r2_score(
                test_pd,
                predicted,
            )
        ),
        "fidelity_mae": float(
            mean_absolute_error(
                test_pd,
                predicted,
            )
        ),
        "rules": _extract_rules(
            surrogate,
            selected_names,
        ),
    }

    return surrogate, info