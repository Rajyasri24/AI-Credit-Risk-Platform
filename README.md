# AI Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk decision-support platform built using the **Home Credit Default Risk** dataset.

The platform combines **portfolio analytics, machine learning, calibrated risk scoring, explainable AI, business-readable model rules, natural-language-to-SQL analytics, Streamlit, SQLite, and Docker** in one modular application.

---

## Overview

Credit-risk analysis requires more than predicting whether an applicant may default. Decision-makers also need to:

- understand portfolio-level risk patterns,
- identify applicants with elevated default probability,
- explain why the model produced a given prediction,
- explore portfolio data without manually writing SQL,
- and obtain reproducible results through a deployable application.

The platform addresses these requirements through three user-facing modules:

- **EDA** — portfolio intelligence, business insights, data quality, repayment behaviour, employment stability and risk segmentation.
- **Machine Learning** — probability of default, risk score, Low / Medium / High risk classification, SHAP explanations and ML-derived business rules.
- **Chatbot** — natural-language portfolio questions translated into validated SQL and returned as business-readable answers.

> The application is designed as a **credit-risk decision-support platform**, not as an autonomous loan approval or rejection engine.

---

## Key Capabilities

| Area | Capability |
|---|---|
| Data Preparation | Application, bureau and installment data integrated at applicant level |
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
| User Interface | Streamlit |
| Deployment | Docker and Docker Compose |

---

## Dataset

The project uses the **Home Credit Default Risk** dataset.

The implementation uses four source tables:

```text
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
└── installments_payments.csv
```

### Source Table Roles

| Table | Purpose |
|---|---|
| `application_train.csv` | Applicant profile, financial information and historical default target |
| `application_test.csv` | Application records without the target field |
| `bureau.csv` | Historical bureau credit information |
| `installments_payments.csv` | Historical installment and repayment behaviour |

The raw dataset is intentionally excluded from the GitHub repository.

---

## Analytical Dataset

Supporting bureau and installment records contain multiple rows per applicant. These are aggregated using:

```text
SK_ID_CURR
```

before being merged with the application-level data.

The analytical design follows:

```text
One row = One applicant
```

### Final Analytical Dataset

| Property | Value |
|---|---:|
| Applicants | 307,511 |
| Analytical Fields | 140 |
| Unique Applicants | 307,511 |
| Duplicate Applicants | 0 |
| Historical Default Rate | 8.07% |

Generated analytical dataset:

```text
data/credit_applicants.parquet
```

---

## Data Preparation and Feature Engineering

The analytical layer combines applicant information, financial characteristics, employment history, bureau information and repayment behaviour.

### Representative Engineered Features

```text
CREDIT_INCOME_RATIO =
AMT_CREDIT / AMT_INCOME_TOTAL
```

```text
ANNUITY_INCOME_RATIO =
AMT_ANNUITY / AMT_INCOME_TOTAL
```

```text
CREDIT_GOODS_RATIO =
AMT_CREDIT / AMT_GOODS_PRICE
```

```text
AGE_YEARS =
|DAYS_BIRTH| / 365.25
```

```text
EMPLOYMENT_YEARS =
|DAYS_EMPLOYED| / 365.25
```

```text
EMPLOYMENT_AGE_RATIO =
EMPLOYMENT_YEARS / AGE_YEARS
```

The Home Credit sentinel value:

```text
DAYS_EMPLOYED = 365243
```

is treated as missing before deriving employment duration.

### Bureau Aggregates

Examples include:

- historical bureau credit count,
- active credit count,
- closed credit count,
- overdue credit count,
- total bureau credit,
- outstanding bureau debt,
- historical overdue amount.

### Repayment Aggregates

Examples include:

- installment count,
- average payment delay,
- maximum payment delay,
- historical late-payment rate,
- average payment ratio.

---

## Data Processing Flow

```text
Home Credit Source Tables
          │
          ▼
File Validation
          │
          ▼
Data Cleaning
          │
          ├──────── Application Features
          ├──────── Bureau Aggregation
          └──────── Installment Aggregation
          │
          ▼
Applicant-Level Merge
          │
          ▼
Feature Engineering
          │
          ▼
credit_applicants.parquet
```

---

## Exploratory Data Analysis

EDA was designed to answer **business-relevant credit-risk questions**, rather than only generating descriptive statistics.

The analysis focuses on:

- portfolio composition,
- historical default behaviour,
- external credit indicators,
- repayment behaviour,
- employment stability,
- model-derived risk segmentation.

### Key EDA Findings

#### 1. Portfolio Default Distribution

The historical default rate is approximately:

```text
8.07%
```

This confirms that default is a minority event and motivates imbalance-aware model training and evaluation.

#### 2. External Credit Indicators

External credit indicators show clear separation between historical outcomes.

For example:

```text
Median EXT_SOURCE_2

Non-default ≈ 0.574
Default     ≈ 0.440
```

Applicants with lower external credit indicators show comparatively higher historical default behaviour.

#### 3. Employment Stability

Median employment duration differs between historical outcomes:

```text
Non-default ≈ 4.63 years
Default     ≈ 3.37 years
```

Longer employment history therefore contributes useful information to the overall applicant risk profile.

#### 4. Historical Repayment Behaviour

Historical defaulters demonstrate higher late-payment behaviour:

```text
Median Late-Payment Rate

Non-default ≈ 1.61%
Default     ≈ 4.84%
```

Repayment behaviour therefore adds an important behavioural signal beyond application-level financial information.

#### 5. Risk Segmentation

Applicants grouped into Low, Medium and High model-risk segments show progressively different observed default behaviour.

This provides a portfolio-level interpretation layer in addition to individual probabilities.

---

## EDA Dashboard Views

The Streamlit EDA module contains five principal visualizations.

### Applicant Risk Classification

Shows the distribution of applicants across the model-derived Low, Medium and High risk segments.

### Observed Default Rate by Risk Level

Tests whether model risk segmentation translates into progressively higher observed historical default rates.

### External Credit Indicators

Compares external credit indicators between historical default and non-default applicants.

### Historical Late-Payment Behaviour

Shows the relationship between repayment behaviour and historical default outcomes.

### Default Rate by Employment Tenure

Examines whether employment stability is associated with differences in historical default behaviour.

---

## Machine Learning Approach

The prediction task is formulated as binary classification:

```text
TARGET = 0 → Non-default
TARGET = 1 → Default
```

### Selected Model — XGBoost

The final predictive model is:

```text
XGBoost Classifier
```

XGBoost was selected because the dataset consists of structured tabular credit data with:

- mixed numeric and categorical attributes,
- non-linear relationships,
- interactions between financial and behavioural features,
- missing information,
- significant class imbalance,
- and a requirement for explainable predictions.

Tree-based XGBoost models also integrate naturally with **SHAP**, supporting both applicant-level and portfolio-level explanations.

---

## Model Feature Strategy

The model uses a curated feature subset rather than automatically consuming every analytical field.

Selection is based on:

- prediction-time availability,
- business relevance,
- credit-risk domain coverage,
- data quality,
- interpretability,
- representation across application, bureau and repayment behaviour.

The complete analytical dataset remains available for EDA and Talk-to-Data analytics.

---

## Model Preprocessing

### Numeric Features

Numeric variables use:

```text
Median Imputation
+
Missing-Value Indicators
```

Missing-value indicators allow the model to learn when the absence of information itself has predictive value.

### Categorical Features

Categorical fields use:

```text
Missing Value → "Unknown"
One-Hot Encoding
handle_unknown = ignore
```

This creates a consistent training and inference pipeline while safely handling unseen categories.

---

## Train, Calibration and Test Strategy

The labelled data is divided using stratified sampling.

| Split | Purpose |
|---|---|
| 70% | XGBoost model training |
| 10% | Probability calibration, operating-threshold selection, risk-band definition and explanatory feature selection |
| 20% | Final untouched model evaluation |

The test set remains independent of model fitting, probability calibration and operating-threshold selection.

---

## Handling Class Imbalance

Historical defaults represent only approximately **8%** of the portfolio.

The pipeline therefore uses multiple imbalance-aware mechanisms.

### Stratified Sampling

The class distribution is maintained across training, calibration and test partitions.

### XGBoost Class Weighting

The minority default class receives additional learning importance using:

```text
scale_pos_weight =
Number of Non-default Applicants
/
Number of Default Applicants
```

This strengthens minority-class learning without creating synthetic applicant records.

### Operating Threshold

A generic probability threshold of `0.50` is not assumed.

Instead, the operating threshold is selected on the dedicated calibration set by optimizing F1.

Final selected threshold:

```text
0.1562
```

The selected threshold is frozen before final test-set evaluation.

---

## Final Model Performance

Evaluation is performed only on the untouched 20% test set.

| Metric | Result |
|---|---:|
| ROC-AUC | **0.7645** |
| PR-AUC | **0.2586** |
| PR-AUC Lift | **3.20×** |
| Precision | **0.2541** |
| Recall | **0.4109** |
| F1 Score | **0.3140** |
| Brier Score | **0.0672** |
| Operating Threshold | **0.1562** |

### Metric Interpretation

- **ROC-AUC** measures overall ranking capability.
- **PR-AUC** evaluates discrimination specifically for the minority default class.
- **Precision** measures how often positive default predictions are correct.
- **Recall** measures how many actual defaults are identified.
- **F1 Score** balances precision and recall at the selected operating threshold.
- **Brier Score** evaluates the quality of calibrated probability estimates.

---

## Risk-Level Validation

Risk segmentation is independently evaluated on the untouched test set.

| Risk Band | Applicants | Observed Default Rate | Average Predicted Default |
|---|---:|---:|---:|
| Low | 20,437 | **2.070%** | **2.089%** |
| Medium | 20,981 | **5.481%** | **5.531%** |
| High | 20,085 | **16.888%** | **16.547%** |

The observed default rate increases consistently from Low to Medium to High risk.

The close alignment between observed and predicted default rates also supports the interpretation of the calibrated probabilities.

---

## Probability Calibration and Risk Scoring

Raw XGBoost decision scores are calibrated using:

```text
Platt / Sigmoid Calibration
```

on the dedicated calibration partition.

The calibrated probability becomes the user-facing probability of default:

```text
MODEL_PD
```

A model risk score is then calculated as:

```text
Risk Score = MODEL_PD × 100
```

Example:

```text
Probability of Default = 0.0662

Risk Score = 6.62 / 100
```

This score represents model-predicted credit risk and is not intended to replicate an external bureau score.

---

## Risk Classification

Applicants are grouped into:

```text
Low
Medium
High
```

risk categories.

Risk-band thresholds are derived from the calibrated probability distribution rather than manually hard-coded business rules.

The bands therefore provide **relative portfolio risk segmentation** while the underlying calibrated probability remains available for more granular analysis.

---

## Explainable AI

The platform uses:

```text
SHAP
```

to explain model behaviour.

### Local Explainability

For an individual applicant, the Machine Learning interface provides:

- probability of default,
- risk score,
- risk classification,
- applicant financial profile,
- leading prediction drivers,
- direction of each driver's impact,
- SHAP contribution visualization.

The interface translates technical SHAP values into:

```text
Increases risk
```

or:

```text
Reduces risk
```

so model reasoning remains understandable to business users.

### Global Explainability

Global SHAP importance identifies the variables with the strongest overall influence on model predictions.

Leading portfolio-level drivers include:

- `EXT_SOURCE_2`
- `EXT_SOURCE_3`
- `EXT_SOURCE_1`
- `CREDIT_GOODS_RATIO`
- `BUREAU_DEBT_SUM`
- `AMT_GOODS_PRICE`
- `AMT_ANNUITY`
- `LATE_PAYMENT_RATE`
- `AGE_YEARS`
- `EMPLOYMENT_YEARS`

Global explanatory feature selection is performed using calibration data while preserving the final test set for evaluation.

---

## ML-Derived Business Rules

The final applicant prediction continues to come from XGBoost.

A shallow surrogate decision tree summarizes common XGBoost probability patterns using influential features.

```text
XGBoost Predictions
        │
        ▼
Important Features
        │
        ▼
Shallow Surrogate Tree
        │
        ▼
Business-Readable Rules
```

Rules are presented in the form:

```text
IF business conditions
THEN approximate probability of default
```

This adds an interpretable view of model behaviour without replacing the primary predictive model.

---

## Talk-to-Data

The platform includes a natural-language analytics assistant powered by **Gemini** through the Google GenAI SDK.

Users can ask portfolio questions in plain English rather than manually writing SQL.

Example questions include:

```text
What is the observed default rate?
```

```text
How many applicants are in each risk band?
```

```text
What is the observed default rate for each risk band?
```

```text
Compare historical late-payment behaviour across risk bands.
```

```text
What is the average requested credit amount by risk band?
```

---

## NL-to-SQL Workflow

```text
Natural-Language Question
          │
          ▼
Schema-Grounded Prompt
          │
          ▼
Gemini
          │
          ▼
SQL Generation
          │
          ▼
SQL Validation
          │
          ▼
Read-Only SQLite Execution
          │
          ▼
Query Result
          │
          ▼
Business-Readable Answer
```

---

## Prompt Engineering

The SQL-generation prompt contains:

- approved database schema,
- business definitions for relevant columns,
- SQL safety rules,
- distinction between historical outcomes and predicted probabilities,
- five representative few-shot query patterns,
- recent conversation context.

Two fields are explicitly distinguished:

```text
TARGET
```

represents the **observed historical default outcome**.

```text
MODEL_PD
```

represents the **model-predicted probability of default**.

This prevents historical outcomes and model predictions from being mixed in analytical responses.

---

## SQL Validation and Hallucination Control

Generated SQL is validated before execution.

The validation layer enforces:

- one SQL statement,
- `SELECT`-only execution,
- approved database table,
- approved schema columns,
- read-only database access,
- rejection of destructive operations.

Examples of blocked operations include:

```text
INSERT
UPDATE
DELETE
DROP
CREATE
ALTER
```

The chatbot operates against:

```text
credit_applicants
```

inside:

```text
data/credit_risk.db
```

Business answers are generated only after validated SQL executes and are grounded in returned database values.

---

## Controlled SQL Repair

If a generated query does not pass validation or execution, one bounded correction attempt is permitted.

The repair process receives:

- original question,
- generated SQL,
- validation or execution feedback,
- approved schema.

This maintains controlled behaviour without introducing unrestricted autonomous retry loops.

---

## Deterministic SQL Fallback

The five core analytical questions also have a deterministic SQL execution path.

```text
User Question
      │
      ▼
Gemini NL-to-SQL
      │
      ├──────── Standard analytical path
      │
      └──────── Core-question deterministic path
                         │
                         ▼
                Prevalidated SQL Template
                         │
                         ▼
                       SQLite
                         │
                         ▼
                Deterministic Answer
```

The deterministic path covers the five predefined portfolio questions using prevalidated read-only SQL.

This provides:

- flexible natural-language analytics for custom questions,
- deterministic execution for the platform's core analytical questions,
- consistent database-grounded results,
- no additional LLM provider or dependency.

---

## Prompt and Token Efficiency

The Talk-to-Data workflow minimizes unnecessary context by using:

- a curated analytical schema,
- concise field definitions,
- reusable few-shot examples,
- recent conversation turns only,
- database-side aggregation,
- one bounded repair attempt,
- query results rather than the complete dataset.

The complete applicant dataset is therefore never passed directly to the language model.

---

## Application Interface

The Streamlit application contains three primary sections.

### 1. EDA

Provides:

- portfolio KPIs,
- historical default analysis,
- business insights,
- risk segmentation,
- external credit analysis,
- repayment behaviour analysis,
- employment-tenure analysis,
- data-quality summary.

### 2. Machine Learning

Provides:

- final test-set model performance,
- test-set risk-band validation,
- applicant probability of default,
- risk score,
- Low / Medium / High classification,
- applicant financial profile,
- local SHAP explanation,
- global SHAP importance,
- ML-derived business rules.

### 3. Chatbot

Provides:

- five ready-made analytical questions,
- custom natural-language questions,
- validated SQL execution,
- readable business responses,
- supporting result tables,
- generated SQL visibility,
- deterministic SQL support for the five core queries.

---

## System Architecture

```text
                         HOME CREDIT DATA
                                │
                                ▼
                     Data Validation & Loading
                                │
                                ▼
                Cleaning + Aggregation + Engineering
                                │
                                ▼
                  Applicant-Level Analytical Dataset
                                │
              ┌─────────────────┴──────────────────┐
              │                                    │
              ▼                                    ▼
             EDA                              ML Pipeline
                                                   │
                                                   ▼
                                                XGBoost
                                                   │
                                  ┌────────────────┴──────────────┐
                                  │                               │
                                  ▼                               ▼
                           Platt Calibration                    SHAP
                                  │                               │
                                  ▼                               ▼
                       Probability of Default              Explanations
                                  │
                                  ▼
                        Risk Score + Risk Bands
                                  │
                                  │
                  ┌───────────────┴─────────────────┐
                  │                                 │
                  ▼                                 ▼
            Streamlit ML UI                  SQLite Database
                                                    │
                                    ┌───────────────┴──────────────┐
                                    │                              │
                                    ▼                              ▼
                              Gemini NL-to-SQL             Core SQL Fallback
                                    │                              │
                                    ▼                              │
                              SQL Validation                      │
                                    └───────────────┬──────────────┘
                                                    ▼
                                           Read-Only Execution
                                                    │
                                                    ▼
                                           Business-Readable Answer
                                                    │
                                                    ▼
                                              Streamlit Chatbot
```

---

## Major Design Decisions

| Decision | Selected Approach | Rationale |
|---|---|---|
| Predictive Model | XGBoost | Strong fit for structured tabular credit data and SHAP |
| Class Imbalance | `scale_pos_weight` | Improves minority-class learning without synthetic records |
| Data Split | 70 / 10 / 20 | Separates training, calibration and final evaluation |
| Probability Calibration | Platt calibration | Supports interpretable default probabilities |
| Operating Threshold | Calibration-set F1 optimization | Creates an imbalance-aware operating point |
| Risk Bands | Calibration-distribution thresholds | Provides relative portfolio segmentation |
| Explainability | SHAP | Supports local and global explanations |
| Business Rules | Shallow surrogate tree | Converts model behaviour into readable patterns |
| Analytical Database | SQLite | Lightweight and reproducible analytical storage |
| LLM | Gemini | Schema-grounded natural-language-to-SQL generation |
| SQL Validation | SQLGlot | Validates SQL before database execution |
| Core Query Resilience | Prevalidated SQL templates | Provides deterministic execution for key business questions |
| UI | Streamlit | Lightweight multi-section application |
| Deployment | Docker Compose | Reproducible application execution |

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data Storage | Parquet, SQLite |
| SQL Validation | SQLGlot |
| Generative AI | Gemini API, Google GenAI SDK |
| User Interface | Streamlit |
| Artifact Persistence | Joblib |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```text
credit_risk_platform/
├── data/
├── documents/
│   └── project_presentation.pdf
├── notebooks/
│   ├── eda.ipynb
│   └── eda.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py
│   │   ├── query_runner.py
│   │   └── prompt_templates.py
│   └── utils/
│       ├── logger.py
│       ├── config.py
│       ├── helpers.py
│       └── docker_utils.py
├── sql/
│   └── schema.sql
├── models/
│   ├── xgboost_model.joblib
│   ├── preprocessor.joblib
│   ├── model_metadata.joblib
│   └── surrogate_model.joblib
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rajyasri24/AI-Credit-Risk-Platform.git
cd AI-Credit-Risk-Platform
```

### 2. Create a Python Environment

#### Windows

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Add Dataset Files

Place the required Home Credit source files inside:

```text
data/
```

Required files:

```text
application_train.csv
application_test.csv
bureau.csv
installments_payments.csv
```

---

### 4. Configure Environment Variables

Create:

```text
.env
```

from:

```text
.env.example
```

Configure:

```env
GEMINI_API_KEY=your_api_key
LLM_MODEL=gemini-3.6-flash
```

The actual `.env` file is excluded from version control.

---

## Run Locally

### Build the Analytical Dataset

```bash
python -m src.data.preprocessor
```

Generated:

```text
data/credit_applicants.parquet
```

### Train the Model

```bash
python -m src.ml.train
```

Generated model artifacts are stored under:

```text
models/
```

### Build the Analytical Database

```bash
python -m src.talk_to_data.query_runner
```

Generated:

```text
data/credit_risk.db
```

### Start the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Docker

The complete application can be executed through Docker Compose.

Ensure:

- required Home Credit files are inside `data/`,
- `.env` contains the Gemini configuration,
- trained model artifacts are available inside `models/`.

Run:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

Stop:

```bash
docker compose down
```

---

## Generated Artifacts

### Processed Data

```text
data/credit_applicants.parquet
data/credit_risk.db
```

### Model Artifacts

```text
models/xgboost_model.joblib
models/preprocessor.joblib
models/model_metadata.joblib
models/surrogate_model.joblib
```

---

## Presentation

The final project presentation is maintained under:

```text
documents/project_presentation.pdf
```

---

## Repository

```text
https://github.com/Rajyasri24/AI-Credit-Risk-Platform
```

---

## Author
Rajyasri S

**Rajyasri S**  
M.Sc. Data Science  
CHRIST (Deemed to be University), Bengaluru
