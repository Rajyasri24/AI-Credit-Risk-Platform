# AI Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk decision-support platform built using the **Home Credit Default Risk** dataset.

The solution combines **data understanding, machine learning, explainable AI, natural-language-to-SQL analytics, a Streamlit interface, and Dockerized deployment** in one modular application.

---

## Overview

Credit-risk teams need more than a prediction score. They need to understand portfolio behaviour, identify higher-risk applicants, explain model decisions, and explore credit data without depending on technical analysts for every question.

This platform supports those requirements through three user-facing modules:

- **EDA** — portfolio intelligence and business insights
- **Machine Learning** — applicant default prediction, risk classification, and explainability
- **Chatbot** — natural-language portfolio analysis backed by validated SQL

The application is designed as a **decision-support platform** rather than an autonomous loan approval or rejection system.

---

## Key Capabilities

| Area | Capability |
|---|---|
| Data Understanding | Applicant, financial, bureau and repayment analysis |
| Data Quality | Missing-value analysis, duplicate checks and feature categorization |
| Business Insights | Five portfolio-level credit-risk insights with visualizations |
| Default Prediction | XGBoost-based probability-of-default model |
| Risk Scoring | Calibrated 0–100 model risk score |
| Risk Classification | Low, Medium and High risk bands |
| Explainable AI | Local and global SHAP explanations |
| Business Rules | ML-derived interpretable decision patterns |
| Talk-to-Data | Natural-language questions converted into SQL |
| SQL Safety | Schema validation and read-only execution |
| User Interface | Lightweight multi-section Streamlit application |
| Deployment | Docker and Docker Compose |

---

## Dataset

The project uses the **Home Credit Default Risk** dataset.

Required files:

```text
application_train.csv
application_test.csv
bureau.csv
installments_payments.csv
```

Place them inside:

```text
data/
```

Expected structure:

```text
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
└── installments_payments.csv
```

The raw dataset is intentionally **not stored in the GitHub repository**.

### Source Tables

| Dataset | Role |
|---|---|
| `application_train.csv` | Applicant profile, financial information and historical default target |
| `application_test.csv` | Application records without the target column |
| `bureau.csv` | Historical bureau credit information |
| `installments_payments.csv` | Historical installment and repayment behaviour |

---

## Analytical Dataset

The preprocessing pipeline converts the source tables into a single applicant-level analytical dataset.

| Property | Value |
|---|---:|
| Applicants | 307,511 |
| Analytical Fields | 140 |
| Unique Applicants | 307,511 |
| Duplicate Applicants | 0 |
| Observed Historical Default Rate | 8.07% |

The analytical design maintains:

```text
One row = One applicant
```

Supporting tables such as bureau and installment-payment records contain multiple records per applicant. These are aggregated at:

```text
SK_ID_CURR
```

before being joined to the main application data.

The resulting processed dataset is stored as:

```text
data/credit_applicants.parquet
```

---

## Feature Categories

The analytical data covers the major business dimensions required for credit-risk analysis.

### Applicant Profile

- age
- family size
- number of children
- education
- family status
- housing type

### Financial Profile

- annual income
- requested credit amount
- loan annuity
- goods price
- credit-to-income ratio
- annuity-to-income ratio

### Employment Profile

- employment duration
- occupation type
- organization type
- employment-to-age ratio

### External Credit Indicators

```text
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
```

### Bureau Credit History

- number of historical bureau credits
- active credit count
- closed credit count
- overdue credit count
- total historical credit
- historical debt
- historical overdue amount

### Historical Repayment Behaviour

- installment count
- average payment delay
- maximum payment delay
- historical late-payment rate
- average payment ratio

---

## Feature Engineering

The preprocessing pipeline creates interpretable features that retain clear credit-risk meaning.

### Credit-to-Income Ratio

```text
CREDIT_INCOME_RATIO =
AMT_CREDIT / AMT_INCOME_TOTAL
```

Represents the requested credit relative to annual income.

### Annuity-to-Income Ratio

```text
ANNUITY_INCOME_RATIO =
AMT_ANNUITY / AMT_INCOME_TOTAL
```

Represents repayment burden relative to income.

### Credit-to-Goods Ratio

```text
CREDIT_GOODS_RATIO =
AMT_CREDIT / AMT_GOODS_PRICE
```

Represents how much of the purchase value is financed through credit.

### Age

```text
AGE_YEARS =
|DAYS_BIRTH| / 365.25
```

### Employment Duration

```text
EMPLOYMENT_YEARS =
|DAYS_EMPLOYED| / 365.25
```

The Home Credit sentinel value:

```text
DAYS_EMPLOYED = 365243
```

is treated as missing rather than as a valid employment duration.

### Employment-to-Age Ratio

```text
EMPLOYMENT_AGE_RATIO =
EMPLOYMENT_YEARS / AGE_YEARS
```

---

## Data Processing Workflow

```text
Raw Home Credit Files
        │
        ▼
File Validation
        │
        ▼
Data Cleaning
        │
        ├── Application Features
        ├── Bureau Aggregation
        └── Installment Aggregation
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

The EDA module focuses on business-relevant portfolio behaviour rather than only descriptive statistics.

It includes:

- dataset summary
- data quality observations
- feature categorization
- five business insights
- five supporting visualizations

### Business Insight 1 — Portfolio Default Level

The observed historical default rate is approximately:

```text
8.07%
```

This confirms that the target variable is strongly imbalanced and requires imbalance-aware model training and evaluation.

### Business Insight 2 — External Credit Indicators

External credit indicators are lower among historical defaulters.

For example:

```text
EXT_SOURCE_2

Non-default median ≈ 0.574
Default median     ≈ 0.440
```

### Business Insight 3 — Employment Stability

Median employment duration differs across historical outcomes:

```text
Non-default ≈ 4.63 years
Default     ≈ 3.37 years
```

### Business Insight 4 — Repayment Behaviour

Historical defaulters show a higher late-payment rate.

```text
Median Late-Payment Rate

Non-default ≈ 1.6%
Default     ≈ 4.8%
```

### Business Insight 5 — Model Risk Segmentation

Applicants classified as Low, Medium and High risk show progressively different observed default behaviour, providing an interpretable portfolio segmentation layer.

### Supporting Visualizations

The EDA interface includes:

1. Applicant Risk Classification
2. Observed Default Rate by Risk Level
3. External Credit Indicators
4. Historical Late-Payment Behaviour
5. Default Rate by Employment Tenure

The charts are presented in a business-focused **3 + 2 dashboard layout**.

---

## Machine Learning Approach

The prediction task is formulated as binary classification.

```text
TARGET = 0 → Non-default
TARGET = 1 → Default
```

### Selected Model

The primary predictive model is:

```text
XGBoost Classifier
```

XGBoost was selected because the dataset is structured tabular credit data containing:

- numeric variables
- encoded categorical variables
- non-linear relationships
- feature interactions
- missing information
- significant class imbalance

XGBoost also integrates naturally with SHAP for explainability.

---

## Model Feature Strategy

The model does not automatically consume every analytical field.

A curated subset is selected using:

- prediction-time availability
- business relevance
- credit-risk domain coverage
- data quality
- interpretability
- representation across application, bureau and repayment behaviour

The broader analytical dataset remains available for EDA and Talk-to-Data analysis.

---

## Model Preprocessing

### Numeric Features

Numeric fields use:

```text
Median Imputation
```

Missing-value indicators are retained so that the model can learn whether missingness itself contains predictive information.

### Categorical Features

Categorical variables use:

```text
Missing value → "Unknown"
One-Hot Encoding
handle_unknown = ignore
```

This produces a consistent inference pipeline while allowing unseen categories to be handled safely.

---

## Train, Calibration and Test Strategy

The labelled dataset is split using stratified sampling.

| Split | Usage |
|---|---|
| 70% | Model training |
| 10% | Probability calibration and risk threshold definition |
| 20% | Final model evaluation |

The 20% test set is kept separate from model fitting and calibration.

---

## Handling Class Imbalance

The historical default rate is approximately:

```text
8.07%
```

Accuracy alone would therefore be misleading.

The implementation uses two primary controls.

### Stratified Sampling

The class distribution is maintained across train, calibration and test splits.

### XGBoost Class Weighting

The minority default class is weighted dynamically using:

```text
scale_pos_weight =
Number of Non-default Applicants
/
Number of Default Applicants
```

This approach improves minority-class learning without creating synthetic applicant records.

---

## Model Evaluation

The final evaluation is performed using the untouched 20% test set.

The following metrics are calculated:

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures overall ranking quality |
| PR-AUC | Focuses on the minority default class |
| Precision | Measures correctness of positive default predictions |
| Recall | Measures coverage of actual defaults |
| F1 Score | Balances precision and recall |
| Brier Score | Evaluates quality of predicted probabilities |
| Confusion Matrix | Shows classification outcomes |

The final trained model metrics are stored in:

```text
models/model_metadata.joblib
```

and the primary metrics are displayed directly in the **Machine Learning** section of the application.

---

## Probability Calibration

The system exposes probability of default directly to users, so probability quality is important in addition to classification performance.

The raw XGBoost output is calibrated using:

```text
Platt / Sigmoid Calibration
```

on the dedicated calibration split.

The calibrated probability is stored as:

```text
MODEL_PD
```

---

## Risk Score

The user-facing risk score is calculated as:

```text
Risk Score = MODEL_PD × 100
```

Example:

```text
Probability of Default = 0.124

Risk Score = 12.4 / 100
```

This is a **model-generated risk score**, not a bureau credit score.

---

## Risk Classification

Applicants are classified into:

```text
Low
Medium
High
```

risk categories.

Risk thresholds are derived from the calibrated probability distribution.

The categories therefore provide relative portfolio risk segmentation rather than hard-coded approval rules.

---

## Explainable AI

The application uses:

```text
SHAP
```

to explain model behaviour.

### Local Explainability

For an individual applicant, the system shows:

- probability of default
- risk score
- risk classification
- major contributing factors
- whether each factor increases or reduces predicted risk
- SHAP contribution visualization

Example interpretation:

```text
External Credit Indicator → Reduces predicted risk
Late-Payment Behaviour    → Increases predicted risk
Employment History        → Reduces predicted risk
```

This converts technical model contributions into language understandable to non-technical users.

### Global Explainability

Global SHAP feature importance identifies the variables that most strongly influence XGBoost predictions across the full portfolio.

This allows stakeholders to understand the dominant drivers of model behaviour at portfolio level.

---

## ML-Derived Business Rules

The primary prediction continues to come from XGBoost.

A shallow surrogate decision tree is used internally only to summarize common XGBoost decision patterns.

```text
Trained XGBoost
      │
      ▼
Model Probability Predictions
      │
      ▼
Influential Features
      │
      ▼
Shallow Decision Tree
      │
      ▼
Readable Decision Patterns
```

The resulting patterns are presented in the application as:

```text
IF business conditions
THEN approximate model probability of default
```

The surrogate tree does not replace the primary model and is not used for applicant-level final prediction.

---

## Talk-to-Data System

The platform includes an LLM-powered chatbot that allows users to ask analytical questions using plain English.

Users can either:

- select a ready-made question
- enter a custom business question

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

## Talk-to-Data Workflow

```text
Natural-Language Question
        │
        ▼
Prompt Construction
        │
        ▼
Gemini LLM
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

## LLM Integration

The Talk-to-Data component uses the **Gemini API** through the Google GenAI SDK.

The configured LLM model is supplied using:

```env
LLM_MODEL=gemini-3.6-flash
```

Direct SDK integration keeps the following components explicit:

- prompt construction
- schema grounding
- SQL validation
- bounded repair
- result summarization
- conversation context

---

## Prompt Engineering

The SQL-generation prompt includes:

- approved database schema
- business descriptions of important columns
- SQL safety constraints
- distinction between historical and predicted risk
- five representative few-shot query patterns
- recent conversation context

Two fields are deliberately separated:

```text
TARGET
```

represents the **observed historical default outcome**.

```text
MODEL_PD
```

represents the **model-predicted probability of default**.

This prevents observed performance and predicted risk from being mixed in analytical answers.

---

## Hallucination Control and SQL Safety

Generated SQL is never executed directly.

The validation layer checks:

- exactly one SQL statement
- SELECT-only access
- approved table name
- approved schema columns
- no destructive operations
- read-only database execution

Blocked operations include:

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

Detailed non-aggregate queries are bounded to prevent unnecessarily large responses.

Aggregate analytical queries operate across the required portfolio data.

---

## Bounded SQL Repair

If generated SQL fails validation or execution, the system permits one controlled correction attempt.

The repair prompt receives:

- the original user question
- rejected SQL
- validation or execution feedback
- approved database schema

There is no unrestricted autonomous retry loop.

---

## Result Grounding

Business responses are generated only after validated SQL has been executed.

The summarization prompt instructs the LLM to:

- use only returned query values
- avoid inventing causes
- distinguish observed historical outcomes from predicted risk
- avoid treating SQL output limits as analytical sample sizes

---

## Prompt and Token Optimization

The Talk-to-Data workflow keeps model context intentionally compact.

Token usage is controlled through:

- a curated analytical SQL schema
- concise field descriptions
- five reusable few-shot examples
- only recent conversation turns
- one bounded repair attempt
- database-side aggregation
- sending only query results to the LLM rather than the complete dataset

The full 307,511-row analytical dataset is therefore never passed directly to the language model.

---

## Application Interface

The Streamlit application contains three sections.

### 1. EDA

Provides:

- portfolio KPIs
- five business insights
- five supporting charts
- feature categorization
- data-quality summary

### 2. Machine Learning

Provides:

- model performance
- applicant probability of default
- risk score
- Low / Medium / High classification
- applicant financial profile
- local SHAP explanation
- global SHAP importance
- ML-derived business rules

### 3. Chatbot

Provides:

- ready-made analytical questions
- custom natural-language questions
- readable business answers
- supporting query results
- optional generated SQL view

---

## System Architecture

```text
Home Credit Source Data
        │
        ▼
Data Loading
        │
        ▼
Cleaning + Feature Engineering
        │
        ▼
Applicant-Level Analytical Dataset
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
      EDA                ML Training
                              │
                              ▼
                         XGBoost Model
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Calibration            SHAP
                    │                   │
                    ▼                   ▼
              Risk Score         Explanations
                    │
                    ▼
               Risk Band
        │
        └─────────────────────┐
                              │
                              ▼
                         SQLite Database
                              │
                              ▼
                        Gemini NL-to-SQL
                              │
                              ▼
                       SQL Validation
                              │
                              ▼
                       Business Answer
                              │
             ┌────────────────┴───────────────┐
             ▼                                ▼
        Machine Learning                   Chatbot
             │                                │
             └──────────────┬─────────────────┘
                            ▼
                       Streamlit UI
```

---

## Major Design Decisions

| Decision | Selected Approach | Reason |
|---|---|---|
| Predictive Model | XGBoost | Strong tabular-data performance and SHAP compatibility |
| Imbalance Handling | `scale_pos_weight` | Handles minority defaults without synthetic applicants |
| Data Split | 70 / 10 / 20 | Separates training, calibration and final evaluation |
| Probability Calibration | Platt calibration | Produces user-facing probability estimates |
| Explainability | SHAP | Supports local and global explanations |
| Business Rules | Shallow surrogate tree | Converts model behaviour into readable patterns |
| Analytical Database | SQLite | Lightweight and reproducible |
| LLM | Gemini | Supports schema-grounded NL-to-SQL generation |
| LLM Integration | Google GenAI SDK | Keeps validation and prompt logic transparent |
| Interface | Streamlit | Lightweight end-to-end demonstration |
| Deployment | Docker Compose | Reproducible one-command execution |

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
| Generative AI | Gemini API |
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

# Installation and Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Rajyasri24/AI-Credit-Risk-Platform.git
```

```bash
cd AI-Credit-Risk-Platform
```

---

## 2. Create a Virtual Environment

### Windows

```bash
py -3.11 -m venv .venv
```

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Dataset

Place the required Home Credit files inside:

```text
data/
```

Required:

```text
application_train.csv
application_test.csv
bureau.csv
installments_payments.csv
```

---

## 5. Configure Environment Variables

Create:

```text
.env
```

using:

```text
.env.example
```

Add:

```env
GEMINI_API_KEY=your_api_key
LLM_MODEL=gemini-3.6-flash
```

The real `.env` file must not be committed to Git.

---

# Running the Project

## Step 1 — Build the Analytical Dataset

```bash
python -m src.data.preprocessor
```

Generated file:

```text
data/credit_applicants.parquet
```

---

## Step 2 — Train the Model

```bash
python -m src.ml.train
```

Generated model artifacts are stored inside:

```text
models/
```

---

## Step 3 — Build the Analytical Database

```bash
python -m src.talk_to_data.query_runner
```

Generated database:

```text
data/credit_risk.db
```

---

## Step 4 — Start the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Docker Deployment

The application can be executed through Docker Compose.

Ensure that:

- the required Home Credit data files are inside `data/`
- `.env` contains the Gemini API credentials
- model artifacts are available inside `models/`

Run:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

Stop the application using:

```bash
docker compose down
```

---

# User Guide

## EDA

Open the **EDA** section to review:

- total applicants
- historical default rate
- requested credit exposure
- high-risk portfolio share
- five business insights
- five portfolio charts
- dataset summary
- data quality
- feature categories

---

## Machine Learning

Open **Machine Learning** and enter a valid:

```text
SK_ID_CURR
```

The application returns:

```text
Probability of Default
Risk Score
Risk Classification
```

The same page provides:

- applicant financial context
- local SHAP explanation
- portfolio-level SHAP importance
- ML-derived business rules

---

## Chatbot

Open **Chatbot**.

Either:

1. select a ready-made question, or
2. enter a custom portfolio question.

Example:

```text
What is the observed default rate for each risk band?
```

The application:

```text
Question
   ↓
NL-to-SQL
   ↓
SQL Validation
   ↓
Database Execution
   ↓
Readable Business Answer
```

Generated SQL can be viewed from the optional SQL section.

---

## Sample Talk-to-Data Output

### Question

```text
What is the observed default rate?
```

### SQL

```sql
SELECT
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS defaulted_applicants,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM credit_applicants;
```

### Business Result

```text
Observed Historical Default Rate: 8.07%
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Authentication for Gemini |
| `LLM_MODEL` | Gemini model used by Talk-to-Data |

Example:

```env
GEMINI_API_KEY=your_api_key
LLM_MODEL=gemini-3.6-flash
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

## Future Enhancements

Potential production extensions include:

- organization-defined credit policy thresholds
- user authentication and role-based access
- model monitoring and drift detection
- centralized model registry
- automated retraining workflows
- cloud-hosted analytical database
- additional credit-history sources
- portfolio monitoring dashboards
- persistent governed conversation history

---

## Repository

```text
https://github.com/Rajyasri24/AI-Credit-Risk-Platform
```

---

## Run Locally

```bash
streamlit run app.py
```

## Run with Docker

```bash
docker compose up --build
```
