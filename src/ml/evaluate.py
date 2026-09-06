import numpy as np
import shap

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor

from src.utils.config import RANDOM_STATE


def to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()

    return np.asarray(X)


def readable_feature_name(name):
    name = name.replace("numeric__", "")
    name = name.replace("categorical__", "")

    if "missingindicator_" in name:
        name = name.replace(
            "missingindicator_",
            "Missing value indicator: ",
        )

    return name.replace("_", " ").strip()


def calculate_global_shap(
    model,
    X,
    feature_names,
    sample_size=2000,
):
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_size = min(
        sample_size,
        X.shape[0],
    )

    indices = rng.choice(
        X.shape[0],
        size=sample_size,
        replace=False,
    )

    X_sample = to_dense(
        X[indices]
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    shap_values = np.asarray(
        shap_values
    )

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    ordering = np.argsort(
        mean_abs_shap
    )[::-1]

    results = []

    for index in ordering[:20]:
        results.append(
            {
                "feature": readable_feature_name(
                    feature_names[index]
                ),
                "encoded_feature": feature_names[index],
                "importance": float(
                    mean_abs_shap[index]
                ),
                "index": int(index),
            }
        )

    return results


def extract_tree_rules(
    tree_model,
    feature_names,
):
    tree = tree_model.tree_

    rules = []

    def recurse(
        node,
        conditions,
    ):
        feature_index = tree.feature[node]

        if feature_index >= 0:
            feature = readable_feature_name(
                feature_names[
                    feature_index
                ]
            )

            threshold = tree.threshold[
                node
            ]

            recurse(
                tree.children_left[node],
                conditions
                + [
                    f"{feature} <= "
                    f"{threshold:.3f}"
                ],
            )

            recurse(
                tree.children_right[node],
                conditions
                + [
                    f"{feature} > "
                    f"{threshold:.3f}"
                ],
            )

        else:
            prediction = float(
                tree.value[node]
                .reshape(-1)[0]
            )

            samples = int(
                tree.n_node_samples[node]
            )

            rules.append(
                {
                    "conditions": conditions,
                    "predicted_pd": prediction,
                    "samples": samples,
                }
            )

    recurse(
        0,
        [],
    )

    rules = sorted(
        rules,
        key=lambda item: item["samples"],
        reverse=True,
    )

    return rules


def build_global_surrogate_tree(
    X_cal,
    X_test,
    calibration_pd,
    test_pd,
    feature_names,
    shap_summary,
):
    top_features = shap_summary[:8]

    top_indices = [
        item["index"]
        for item in top_features
    ]

    top_feature_names = [
        item["encoded_feature"]
        for item in top_features
    ]

    X_cal_small = to_dense(
        X_cal[:, top_indices]
    )

    X_test_small = to_dense(
        X_test[:, top_indices]
    )

    surrogate = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=0.05,
        random_state=RANDOM_STATE,
    )

    surrogate.fit(
        X_cal_small,
        calibration_pd,
    )

    approximation = surrogate.predict(
        X_test_small
    )

    fidelity_r2 = r2_score(
        test_pd,
        approximation,
    )

    fidelity_mae = mean_absolute_error(
        test_pd,
        approximation,
    )

    rules = extract_tree_rules(
        surrogate,
        top_feature_names,
    )

    information = {
        "top_features": [
            readable_feature_name(name)
            for name in top_feature_names
        ],
        "top_feature_indices": top_indices,
        "fidelity_r2": float(
            fidelity_r2
        ),
        "fidelity_mae": float(
            fidelity_mae
        ),
        "rules": rules,
    }

    return surrogate, information