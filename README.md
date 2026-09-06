# AI Credit Risk Platform

AI-powered credit risk decision-support platform built using the **Home Credit Default Risk** dataset.

The platform combines **exploratory data analysis, machine learning, explainable AI, natural-language querying, Streamlit, and Docker** in a single end-to-end application.

---

## Overview

The platform is designed to support credit-risk analysis at both portfolio and applicant level.

It enables users to:

- understand the credit portfolio through business-focused EDA
- predict applicant probability of default
- classify applicants into Low, Medium, and High risk
- explain individual predictions using SHAP
- identify global model risk drivers
- interact with portfolio data through natural-language questions
- execute validated read-only SQL queries through a chatbot
- run the complete application locally or through Docker

---

## Core Capabilities

| Module | Capability |
|---|---|
| EDA | Dataset summary, data quality, feature categorization, five business insights, five supporting charts |
| Machine Learning | Applicant default probability prediction using XGBoost |
| Risk Scoring | Calibrated probability converted to a 0–100 model risk score |
| Risk Classification | Low / Medium / High applicant risk bands |
| Explainable AI | Local and global SHAP-based explanations |
| Business Rules | ML-derived readable decision patterns |
| Talk-to-Data | Natural-language question to SQL to business insight |
| SQL Safety | Schema validation and read-only query execution |
| UI | Streamlit multi-section application |
| Deployment | Docker Compose and Streamlit deployment |

---

## Dataset

This project uses the **Home Credit Default Risk** dataset.

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

Raw dataset files are intentionally excluded from the repository.

---

## Analytical Dataset

The preprocessing pipeline creates an applicant-level analytical dataset.

| Property | Value |
|---|---:|
| Applicants | 307,511 |
| Analytical fields | 140 |
| Unique applicant IDs | 307,511 |
| Duplicate applicants | 0 |
| Historical default rate | 8.07% |

The analytical design maintains:

```text
One row = One applicant
```

Bureau and installment-payment records are aggregated by:

```text
SK_ID_CURR
```

before joining them to the application data.

---

## Feature Groups

The platform uses information from multiple credit-risk domains:

- applicant demographics
- financial characteristics
- employment information
- external credit indicators
- bureau credit history
- historical repayment behaviour
- engineered financial ratios

Key engineered features include:

```text
CREDIT_INCOME_RATIO
ANNUITY_INCOME_RATIO
CREDIT_GOODS_RATIO
EMPLOYMENT_AGE_RATIO
```

Historical features include:

```text
BUREAU_CREDIT_COUNT
BUREAU_ACTIVE_COUNT
BUREAU_OVERDUE_COUNT
BUREAU_DEBT_SUM
AVG_PAYMENT_DELAY
MAX_PAYMENT_DELAY
LATE_PAYMENT_RATE
AVG_PAYMENT_RATIO
```

---

## Business Insights

The EDA layer presents five business-focused findings.

1. **Portfolio Default Level**  
   The observed historical default rate is approximately **8.07%**, confirming a strongly imbalanced credit-risk problem.

2. **External Credit Indicators**  
   Historical defaulters show lower median values across external credit indicators.

3. **Employment Stability**  
   Historical defaulters show shorter median employment history than non-defaulters.

4. **Repayment Behaviour**  
   Historical defaulters show a higher historical late-payment rate.

5. **Risk Segmentation**  
   Model-generated Low, Medium, and High risk groups show progressively different observed default behaviour.

The Streamlit EDA page includes five supporting visualizations:

- Applicant Risk Classification
- Observed Default Rate by Risk Level
- External Credit Indicators
- Historical Late-Payment Behaviour
- Default Rate by Employment Tenure

---

## Machine Learning Pipeline

The prediction problem is formulated as binary classification.

```text
TARGET = 0 → Non-default
TARGET = 1 → Default
```

The primary model is:

```text
XGBoost Classifier
```

### Data Split

```text
70% Training
10% Calibration
20% Final Test
```

Stratified splitting is used to preserve the target distribution.

### Class Imbalance

Class imbalance is handled using XGBoost class weighting:

```text
scale_pos_weight =
number of non-default applicants
/
number of default applicants
```

This allows the model to learn the minority default class without generating synthetic applicants.

### Evaluation Metrics

The model is evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 Score
- Brier Score

---

## Probability Calibration and Risk Score

The XGBoost model output is calibrated using the dedicated calibration split.

The calibrated probability is stored as:

```text
MODEL_PD
```

The user-facing model risk score is:

```text
Risk Score = MODEL_PD × 100
```

Applicants are classified into:

```text
Low
Medium
High
```

risk segments using calibrated probability thresholds.

---

## Explainable AI

The platform uses **SHAP** to explain model behaviour.

### Local Explanation

For an individual applicant, the application shows:

- major contributing risk factors
- whether each factor increases or reduces predicted risk
- SHAP contribution visualization

### Global Explanation

The application also provides portfolio-level SHAP importance to show which variables most strongly influence model predictions overall.

### ML-Derived Rules

Simplified business-readable decision rules are derived from the trained model to summarize recurring model decision patterns.

These are displayed as interpretable rule conditions with approximate default probability.

---

## Talk-to-Data System

The chatbot enables a user to ask portfolio questions in plain English.

Example questions:

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

### Workflow

```text
User Question
      ↓
Gemini LLM
      ↓
SQL Generation
      ↓
SQL Validation
      ↓
Read-Only SQLite Execution
      ↓
Query Result
      ↓
Business-Readable Answer
```

The system distinguishes between:

```text
TARGET
```

which represents the historical observed outcome, and:

```text
MODEL_PD
```

which represents model-predicted probability of default.

---

## SQL Validation

Generated SQL is validated before execution.

The validation layer enforces:

- one SQL statement only
- SELECT queries only
- approved table only
- approved schema columns only
- no destructive SQL operations
- read-only SQLite execution
- bounded output for detailed row queries

Blocked operations include:

```text
INSERT
UPDATE
DELETE
DROP
CREATE
ALTER
```

The analytical table used by the chatbot is:

```text
credit_applicants
```

---

## System Workflow

```text
Home Credit Raw Data
        │
        ▼
Data Loading and Validation
        │
        ▼
Cleaning and Feature Engineering
        │
        ▼
Applicant-Level Analytical Dataset
        │
        ├───────────────┬─────────────────┐
        ▼               ▼                 ▼
       EDA         XGBoost Model         SQLite
                        │                 │
                        ▼                 ▼
              Probability Calibration   NL-to-SQL
                        │                 │
                        ▼                 ▼
                  Risk Score        SQL Validation
                        │                 │
                        ▼                 ▼
               Risk Classification   Query Result
                        │                 │
                        ▼                 ▼
                    SHAP            Business Answer
                        │                 │
                        └────────┬────────┘
                                 ▼
                           Streamlit UI
```

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
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data Query Layer | SQLite, SQLGlot |
| Generative AI | Gemini API |
| User Interface | Streamlit |
| Model Persistence | Joblib |
| Deployment | Docker, Docker Compose |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rajyasri24/AI-Credit-Risk-Platform.git
```

```bash
cd AI-Credit-Risk-Platform
```

### 2. Create a virtual environment

Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place:

```text
application_train.csv
application_test.csv
bureau.csv
installments_payments.csv
```

inside:

```text
data/
```

### 5. Configure environment variables

Create:

```text
.env
```

from:

```text
.env.example
```

Add:

```env
GEMINI_API_KEY=your_api_key
LLM_MODEL=gemini-3.6-flash
```

---

## Run the Pipeline

### Preprocess data

```bash
python -m src.data.preprocessor
```

### Train model

```bash
python -m src.ml.train
```

### Build analytical database

```bash
python -m src.talk_to_data.query_runner
```

### Start application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Docker

The complete application can be started with:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

Stop the application:

```bash
docker compose down
```

---

## Application Usage

### EDA

Use the EDA section to review:

- portfolio KPIs
- five business insights
- five visualizations
- feature categories
- data quality summary

### Machine Learning

Enter a valid:

```text
SK_ID_CURR
```

to view:

- probability of default
- model risk score
- Low / Medium / High classification
- applicant financial profile
- SHAP-based explanation
- portfolio-level model drivers
- ML-derived business rules

### Chatbot

The chatbot supports two modes:

- select a ready-made portfolio question
- enter a custom natural-language question

The generated SQL can be viewed from the optional SQL section.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API authentication |
| `LLM_MODEL` | Gemini model used for Talk-to-Data |

Example:

```env
GEMINI_API_KEY=your_api_key
LLM_MODEL=gemini-3.6-flash
```

Never commit the real `.env` file.

---

## Deployment

The platform supports:

```text
Local Python
Docker Compose
Streamlit
```

Application entry point:

```text
app.py
```

Docker entry point:

```bash
docker compose up --build
```

Streamlit entry point:

```bash
streamlit run app.py
```
