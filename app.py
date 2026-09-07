import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.ml.predict import load_artifacts, predict_applicant
from src.talk_to_data.nl_to_sql import ask_credit_data
from src.utils.config import ANALYTICAL_DATA, METADATA_PATH


st.set_page_config(
    page_title="AI Credit Risk Platform",
    page_icon="◈",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px;
        background: #ffffff;
    }

    .subtitle {
        color: #667085;
        font-size: 1rem;
        margin-top: -8px;
        margin-bottom: 22px;
    }

    .insight-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #fafafa;
    }

    .answer-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        background: #fafafa;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_parquet(ANALYTICAL_DATA)


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


@st.cache_resource
def load_model_assets():
    return load_artifacts()


df = load_data()
metadata = load_metadata()
model, preprocessor, model_metadata = load_model_assets()


if (
    "MODEL_PD" not in df.columns
    or "RISK_SCORE" not in df.columns
    or "RISK_BAND" not in df.columns
):
    X = df[model_metadata["model_features"]].copy()
    X_transformed = preprocessor.transform(X)

    margins = model.predict(
        X_transformed,
        output_margin=True,
    )

    probabilities = (
        model_metadata["calibrator"]
        .predict_proba(margins.reshape(-1, 1))[:, 1]
    )

    thresholds = model_metadata["risk_thresholds"]

    def assign_risk_band(probability):
        if probability <= thresholds["low_max"]:
            return "Low"
        if probability <= thresholds["medium_max"]:
            return "Medium"
        return "High"

    df["MODEL_PD"] = probabilities
    df["RISK_SCORE"] = probabilities * 100
    df["RISK_BAND"] = [
        assign_risk_band(probability)
        for probability in probabilities
    ]


with st.sidebar:
    st.title("AI Credit Risk Platform")

    page = st.radio(
        "Navigation",
        [
            "EDA",
            "Machine Learning",
            "Chatbot",
        ],
    )

    st.caption("Home Credit Default Risk")


if page == "EDA":
    st.title("Portfolio Intelligence")

    st.markdown(
        """
        <div class="subtitle">
        Business view of applicant risk, historical defaults,
        repayment behaviour and credit profile.
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_applicants = len(df)
    default_rate = df["TARGET"].mean() * 100
    total_credit = df["AMT_CREDIT"].sum()
    high_risk_share = df["RISK_BAND"].eq("High").mean() * 100
    high_risk_credit = (
        df.loc[df["RISK_BAND"] == "High", "AMT_CREDIT"].sum()
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Applicants",
        f"{total_applicants:,}",
    )

    k2.metric(
        "Observed Default Rate",
        f"{default_rate:.2f}%",
    )

    k3.metric(
        "Requested Credit Exposure",
        f"{total_credit / 1e9:.2f}B",
    )

    k4.metric(
        "High-Risk Portfolio",
        f"{high_risk_share:.1f}%",
    )

    k5.metric(
        "High-Risk Credit Exposure",
        f"{high_risk_credit / 1e9:.2f}B",
    )

    st.subheader("Key Business Insights")

    ext_non_default = df.loc[
        df["TARGET"] == 0,
        "EXT_SOURCE_2",
    ].median()

    ext_default = df.loc[
        df["TARGET"] == 1,
        "EXT_SOURCE_2",
    ].median()

    emp_non_default = df.loc[
        df["TARGET"] == 0,
        "EMPLOYMENT_YEARS",
    ].median()

    emp_default = df.loc[
        df["TARGET"] == 1,
        "EMPLOYMENT_YEARS",
    ].median()

    late_non_default = (
        df.loc[
            df["TARGET"] == 0,
            "LATE_PAYMENT_RATE",
        ].median()
        * 100
    )

    late_default = (
        df.loc[
            df["TARGET"] == 1,
            "LATE_PAYMENT_RATE",
        ].median()
        * 100
    )

    band_default_rates = (
        df.groupby("RISK_BAND")["TARGET"]
        .mean()
        .mul(100)
        .reindex(["Low", "Medium", "High"])
    )

    insights = [
        (
            "Portfolio Default Level",
            f"{default_rate:.2f}% of applicants historically defaulted.",
        ),
        (
            "External Credit Profile",
            f"Median EXT_SOURCE_2 is {ext_non_default:.3f} for non-defaults "
            f"and {ext_default:.3f} for defaults.",
        ),
        (
            "Employment Stability",
            f"Median employment history is {emp_non_default:.2f} years "
            f"for non-defaults versus {emp_default:.2f} years for defaults.",
        ),
        (
            "Repayment Behaviour",
            f"Median late-payment rate is {late_non_default:.2f}% for "
            f"non-defaults and {late_default:.2f}% for defaults.",
        ),
        (
            "Risk Segmentation",
            f"Observed default rates are "
            f"{band_default_rates['Low']:.2f}% in Low risk, "
            f"{band_default_rates['Medium']:.2f}% in Medium risk and "
            f"{band_default_rates['High']:.2f}% in High risk.",
        ),
    ]

    for title, text in insights:
        st.markdown(
            f"""
            <div class="insight-card">
            <b>{title}</b><br>
            {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    risk_counts = (
        df["RISK_BAND"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
    )

    fig1, ax1 = plt.subplots(figsize=(5.4, 4))
    risk_counts.plot(kind="bar", ax=ax1)
    ax1.set_title("Applicant Risk Classification")
    ax1.set_xlabel("Risk Level")
    ax1.set_ylabel("Applicants")
    ax1.tick_params(axis="x", rotation=0)
    fig1.tight_layout()

    risk_default = (
        df.groupby("RISK_BAND")["TARGET"]
        .mean()
        .mul(100)
        .reindex(["Low", "Medium", "High"])
    )

    fig2, ax2 = plt.subplots(figsize=(5.4, 4))
    risk_default.plot(kind="bar", ax=ax2)
    ax2.set_title("Observed Default Rate by Risk Level")
    ax2.set_xlabel("Risk Level")
    ax2.set_ylabel("Default Rate (%)")
    ax2.tick_params(axis="x", rotation=0)
    fig2.tight_layout()

    external_summary = (
        df.groupby("TARGET")[
            [
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
            ]
        ]
        .median()
        .T
    )

    external_summary.columns = [
        "Non-default",
        "Default",
    ]

    fig3, ax3 = plt.subplots(figsize=(5.4, 4))
    external_summary.plot(kind="bar", ax=ax3)
    ax3.set_title("External Credit Indicators")
    ax3.set_ylabel("Median Indicator")
    ax3.tick_params(axis="x", rotation=0)
    fig3.tight_layout()

    row1 = st.columns(3)

    with row1[0]:
        st.pyplot(fig1, use_container_width=True)

    with row1[1]:
        st.pyplot(fig2, use_container_width=True)

    with row1[2]:
        st.pyplot(fig3, use_container_width=True)

    repayment_summary = (
        df.groupby("TARGET")["LATE_PAYMENT_RATE"]
        .mean()
        .mul(100)
    )

    repayment_summary.index = [
        "Non-default",
        "Default",
    ]

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    repayment_summary.plot(kind="bar", ax=ax4)
    ax4.set_title("Historical Late-Payment Behaviour")
    ax4.set_ylabel("Average Late-Payment Rate (%)")
    ax4.tick_params(axis="x", rotation=0)
    fig4.tight_layout()

    employment = df[
        [
            "EMPLOYMENT_YEARS",
            "TARGET",
        ]
    ].dropna().copy()

    employment["EMPLOYMENT_BAND"] = pd.cut(
        employment["EMPLOYMENT_YEARS"],
        bins=[
            0,
            1,
            3,
            5,
            10,
            np.inf,
        ],
        labels=[
            "<1 year",
            "1–3 years",
            "3–5 years",
            "5–10 years",
            "10+ years",
        ],
        include_lowest=True,
    )

    employment_default = (
        employment.groupby(
            "EMPLOYMENT_BAND",
            observed=True,
        )["TARGET"]
        .mean()
        .mul(100)
    )

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    employment_default.plot(kind="bar", ax=ax5)
    ax5.set_title("Default Rate by Employment Tenure")
    ax5.set_ylabel("Observed Default Rate (%)")
    ax5.tick_params(axis="x", rotation=15)
    fig5.tight_layout()

    row2 = st.columns(2)

    with row2[0]:
        st.pyplot(fig4, use_container_width=True)

    with row2[1]:
        st.pyplot(fig5, use_container_width=True)

    with st.expander("Dataset Summary & Data Quality"):
        numeric_features = len(
            df.select_dtypes(include="number").columns
        )

        categorical_features = len(
            df.select_dtypes(exclude="number").columns
        )

        missing = (
            df.isna()
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Rows",
            f"{len(df):,}",
        )

        d2.metric(
            "Numeric Features",
            numeric_features,
        )

        d3.metric(
            "Categorical Features",
            categorical_features,
        )

        d4.metric(
            "Duplicate Applicants",
            int(
                df["SK_ID_CURR"]
                .duplicated()
                .sum()
            ),
        )

        st.markdown("**Feature Categories**")
        st.markdown(
            """
            - Applicant demographics
            - Financial characteristics
            - Employment information
            - External credit indicators
            - Bureau credit history
            - Historical repayment behaviour
            """
        )

        st.markdown("**Highest Missing-Value Fields**")

        st.dataframe(
            missing
            .head(10)
            .rename("Missing (%)")
            .round(2),
            use_container_width=True,
        )


elif page == "Machine Learning":
    st.title("Credit Risk Prediction")

    st.markdown(
        """
        <div class="subtitle">
        Predict applicant default probability and explain
        the result using business-readable risk drivers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = metadata["metrics"]

    st.subheader("Model Performance")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.3f}",
    )

    m2.metric(
        "PR-AUC",
        f"{metrics['pr_auc']:.3f}",
    )

    m3.metric(
        "Recall",
        f"{metrics['recall']:.3f}",
    )

    m4.metric(
        "Precision",
        f"{metrics['precision']:.3f}",
    )

    m5.metric(
        "F1 Score",
        f"{metrics['f1']:.3f}",
    )

    st.caption(
        "Performance is evaluated on the untouched 20% test set."
    )

    st.subheader("Risk-Level Validation")

    validation = pd.DataFrame(
    metadata["band_performance"]
    )

    validation = validation.rename(
    columns={
        "RISK_BAND": "Risk Band",
        "Observed_Default_Rate": "Observed Default Rate (%)",
        "Average_PD": "Average Predicted Default (%)",
    }
    )

    st.dataframe(
    validation,
    hide_index=True,
    use_container_width=True,
)

    st.caption(
    "Risk-level validation is evaluated on the untouched 20% test set."
)
    st.divider()

    st.subheader("Applicant Risk Assessment")

    applicant_id_text = st.text_input(
        "Applicant ID",
        placeholder="Example: 100002",
    )

    if st.button(
        "Predict Risk",
        type="primary",
    ):
        if not applicant_id_text.strip():
            st.warning(
                "Enter an Applicant ID."
            )

        else:
            try:
                applicant_id = int(
                    applicant_id_text
                )

            except ValueError:
                st.error(
                    "Applicant ID must be numeric."
                )

            else:
                matched = df[
                    df["SK_ID_CURR"]
                    == applicant_id
                ]

                if matched.empty:
                    st.error(
                        "Applicant ID not found."
                    )

                else:
                    applicant = matched.iloc[0]

                    prediction = predict_applicant(
                        applicant
                    )

                    r1, r2, r3 = st.columns(3)

                    r1.metric(
                        "Probability of Default",
                        f"{prediction['default_probability'] * 100:.2f}%",
                    )

                    r2.metric(
                        "Risk Score",
                        f"{prediction['risk_score']:.1f} / 100",
                    )

                    r3.metric(
                        "Risk Classification",
                        prediction["risk_band"],
                    )

                    st.subheader("Applicant Profile")

                    profile = pd.DataFrame(
                        {
                            "Indicator": [
                                "Annual Income",
                                "Requested Credit",
                                "Loan Annuity",
                                "Employment Years",
                                "Previous Bureau Credits",
                                "Outstanding Bureau Debt",
                                "Historical Late-Payment Rate",
                            ],
                            "Value": [
                                applicant["AMT_INCOME_TOTAL"],
                                applicant["AMT_CREDIT"],
                                applicant["AMT_ANNUITY"],
                                applicant["EMPLOYMENT_YEARS"],
                                applicant["BUREAU_CREDIT_COUNT"],
                                applicant["BUREAU_DEBT_SUM"],
                                applicant["LATE_PAYMENT_RATE"],
                            ],
                        }
                    )

                    st.dataframe(
                        profile,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.subheader("Why This Prediction?")

                    drivers = pd.DataFrame(
                        prediction["drivers"]
                    )

                    drivers["Impact"] = (
                        drivers["shap_value"]
                        .apply(
                            lambda value:
                            (
                                "Increases risk"
                                if value > 0
                                else
                                "Reduces risk"
                            )
                        )
                    )

                    explanation_table = drivers[
                        [
                            "feature",
                            "Impact",
                        ]
                    ].copy()

                    explanation_table.columns = [
                        "Risk Driver",
                        "Effect",
                    ]

                    st.dataframe(
                        explanation_table,
                        hide_index=True,
                        use_container_width=True,
                    )

                    plot_drivers = (
                        drivers
                        .sort_values(
                            "shap_value"
                        )
                    )

                    fig, ax = plt.subplots(
                        figsize=(8, 4)
                    )

                    ax.barh(
                        plot_drivers["feature"],
                        plot_drivers["shap_value"],
                    )

                    ax.axvline(
                        0,
                        linewidth=1,
                    )

                    ax.set_title(
                        "SHAP Explanation"
                    )

                    ax.set_xlabel(
                        "Contribution to Predicted Risk"
                    )

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    st.caption(
                        "Factors on the right increase predicted risk; "
                        "factors on the left reduce predicted risk."
                    )

    st.divider()

    st.subheader(
        "Key Risk Drivers Across the Portfolio"
    )

    global_shap = pd.DataFrame(
        metadata["global_shap"][:10]
    )

    global_shap = (
        global_shap
        .sort_values("importance")
    )

    fig_global, ax_global = plt.subplots(
        figsize=(8, 5)
    )

    ax_global.barh(
        global_shap["feature"],
        global_shap["importance"],
    )

    ax_global.set_title(
        "Global SHAP Feature Importance"
    )

    ax_global.set_xlabel(
        "Average Absolute SHAP Contribution"
    )

    fig_global.tight_layout()

    st.pyplot(
        fig_global,
        use_container_width=True,
    )

    st.subheader(
        "ML-Derived Business Rules"
    )

    rules = metadata[
        "global_surrogate_tree"
    ]["rules"][:3]

    for number, rule in enumerate(
        rules,
        start=1,
    ):
        conditions = " AND ".join(
            rule["conditions"]
        )

        predicted_pd = (
            rule["predicted_pd"]
            * 100
        )

        st.markdown(
            f"""
            **Rule {number}**

            When `{conditions}`

            Approximate default probability:
            **{predicted_pd:.2f}%**
            """
        )


else:
    st.title("Credit Portfolio Chatbot")

    st.markdown(
        """
        <div class="subtitle">
        Ask a business question directly, or use one of the
        ready-made questions below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    default_questions = [
        "What is the observed default rate?",
        "How many applicants are in each risk band?",
        "What is the observed default rate for each risk band?",
        "Compare historical late-payment behaviour across risk bands.",
        "What is the average requested credit amount by risk band?",
    ]

    selected_question = st.selectbox(
        "Ready-Made Questions",
        [""] + default_questions,
    )

    question = st.text_input(
        "Your Question",
        value=selected_question,
        placeholder=(
            "Example: Which risk level has the highest observed default rate?"
        ),
    )

    if st.button(
        "Ask",
        type="primary",
    ):
        if not question.strip():
            st.warning(
                "Enter a question."
            )

        else:
            try:
                with st.spinner(
                    "Analysing portfolio data..."
                ):
                    response = ask_credit_data(
                        question=question,
                        history=st.session_state[
                            "chat_history"
                        ],
                    )

                st.markdown(
                    f"""
                    <div class="answer-card">
                    <b>Answer</b><br><br>
                    {response["answer"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if len(response["result"]) > 1:
                    st.dataframe(
                        response["result"],
                        hide_index=True,
                        use_container_width=True,
                    )

                with st.expander(
                    "View Generated SQL"
                ):
                    st.code(
                        response["sql"],
                        language="sql",
                    )

                st.session_state[
                    "chat_history"
                ].append(
                    {
                        "question": question,
                        "answer": response[
                            "answer"
                        ],
                    }
                )

                st.session_state[
                    "chat_history"
                ] = (
                    st.session_state[
                        "chat_history"
                    ][-3:]
                )

            except Exception as error:
                st.error(
                    f"Unable to answer the question: {error}"
                )

    if st.session_state["chat_history"]:
        st.subheader(
            "Recent Questions"
        )

        for item in reversed(
            st.session_state[
                "chat_history"
            ]
        ):
            st.markdown(
                f"**{item['question']}**"
            )
            st.write(
                item["answer"]
            )