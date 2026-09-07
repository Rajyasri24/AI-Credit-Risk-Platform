# AI Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk decision-support platform built using the **Home Credit Default Risk** dataset.

The platform combines **portfolio analytics, machine learning, calibrated risk scoring, explainable AI, business-readable model rules, natural-language-to-SQL analytics, Streamlit, SQLite, and Docker** in a single modular application.

---

## Overview

Credit-risk analysis requires more than predicting whether an applicant may default. Decision-makers also need to:

- understand portfolio-level risk patterns,
- identify applicants with elevated default probability,
- understand the factors influencing each prediction,
- explore portfolio data without manually writing SQL,
- and obtain reproducible outputs through a deployable application.

The platform addresses these requirements through three user-facing modules:

### EDA
Portfolio intelligence, business insights, data quality and visual analysis.

### Machine Learning
Probability of default, model risk score, Low / Medium / High risk segmentation, SHAP explanations and ML-derived business rules.

### Chatbot
Natural-language portfolio questions translated into validated SQL and returned as business-readable answers.

The application is designed as a **credit-risk decision-support system**, rather than an autonomous loan approval or rejection engine.

---

## Key Capabilities

| Area | Capability |
|---|---|
| Data Preparation | Applicant, bureau and installment data integrated at applicant level |
| Portfolio Analytics | Business-focused EDA and portfolio KPIs |
| Default Prediction | XGBoost probability-of-default model |
| Class Imbalance | Stratified sampling and dynamic `scale_pos_weight` |
| Probability Calibration | Platt / sigmoid calibration |
| Operating Threshold | Selected on calibration data using F1 optimization |
| Risk Segmentation | Low, Medium and High model-derived risk bands |
| Explainability | Local and global SHAP |
| Business Rules | Interpretable patterns derived from a shallow surrogate tree |
| Talk-to-Data | Gemini-powered natural-language-to-SQL analytics |
| SQL Safety | SQLGlot validation and read-only database execution |
| Query Resilience | Deterministic SQL fallback for five core analytical questions |
| Interface | Streamlit |
| Deployment | Docker and Docker Compose |

---

# Dataset

The project uses the **Home Credit Default Risk** dataset.

The implementation uses four source tables:

```text
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
└── installments_payments.csv
