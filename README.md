AI Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk decision-support platform built on the Home Credit Default Risk dataset. The solution integrates portfolio EDA, XGBoost-based default prediction, calibrated risk scoring, SHAP explainability, business-readable model rules, natural-language-to-SQL analytics, Streamlit, SQLite, and Docker in one modular application.

Overview

Credit-risk teams need more than a binary prediction. They need to understand portfolio behaviour, identify higher-risk applicants, explain model decisions, and explore credit data without writing SQL for every question.

The platform addresses this through three user-facing modules:

EDA — portfolio intelligence, business insights, risk segmentation, data quality and visual analysis

Machine Learning — probability of default, risk score, Low/Medium/High risk classification, SHAP explanations and ML-derived business rules

Chatbot — natural-language portfolio questions translated into validated SQL and returned as readable business answers

The application is designed as a decision-support platform rather than an autonomous approval or rejection engine.

Key Capabilities

Area

Capability

Data Preparation

Application, bureau and installment data integrated at applicant level

EDA

Business-focused portfolio analysis with five supporting visualizations

Default Prediction

XGBoost probability-of-default model

Class Imbalance

Stratified sampling + dynamic scale_pos_weight

Probability Calibration

Platt / sigmoid calibration

Operating Threshold

Selected on calibration data by maximizing F1

Risk Segmentation

Low, Medium and High model-derived risk bands

Explainable AI

Local and global SHAP

Business Rules

Shallow surrogate tree for readable model patterns

Talk-to-Data

Gemini-powered schema-grounded NL-to-SQL

SQL Safety

Read-only execution with SQLGlot validation

Query Resilience

Deterministic SQL fallback for the five core analytical questions

User Interface

Streamlit

Deployment

Docker and Docker Compose

Dataset

The project uses the Home Credit Default Risk dataset.

Required source files:

data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
└── installments_payments.csv

The preprocessing pipeline converts the source data into a single applicant-level analytical dataset:

data/credit_applicants.parquet

Analytical Dataset Summary

Property

Value

Applicants

307,511

Analytical Fields

140

Unique Applicants

307,511

Duplicate Applicants

0

Historical Default Rate

8.07%

All supporting tables are aggregated by SK_ID_CURR before joining, preserving:

One row = One applicant

The raw dataset is excluded from the repository.

Data Preparation and Feature Engineering

The analytical layer combines application, bureau and repayment information.

Representative engineered features include:

AGE_YEARS

EMPLOYMENT_YEARS

CREDIT_INCOME_RATIO

ANNUITY_INCOME_RATIO

CREDIT_GOODS_RATIO

EMPLOYMENT_AGE_RATIO

bureau credit counts, active/closed history, debt and overdue aggregates

installment count, payment delay, late-payment rate and payment ratio

The Home Credit sentinel value:

DAYS_EMPLOYED = 365243

is treated as missing before employment duration is derived.

Processing Flow

Raw Home Credit Tables
        │
        ▼
Validation + Cleaning
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

Exploratory Data Analysis

The EDA layer focuses on portfolio-level business interpretation rather than descriptive statistics alone.

Key Business Insights

Portfolio default level: historical default rate is approximately 8.07%, confirming a strongly imbalanced target.

External credit profile: historical defaulters show lower median external credit indicators.

Employment stability: non-default applicants show longer median employment history.

Repayment behaviour: historical defaulters show higher late-payment behaviour.

Risk segmentation: observed default rates increase clearly from Low to Medium to High model risk.

EDA Dashboard

The Streamlit dashboard presents five core views:

Applicant Risk Classification

Observed Default Rate by Risk Level

External Credit Indicators

Historical Late-Payment Behaviour

Default Rate by Employment Tenure

These views connect portfolio behaviour directly with the features later used in prediction and explanation.

Machine Learning Approach

The prediction task is binary classification:

TARGET = 0 → Non-default
TARGET = 1 → Default

Selected Model — XGBoost

XGBoost was selected because the problem consists of structured tabular credit data with:

mixed numeric and categorical attributes

non-linear relationships

feature interactions

missing information

strong class imbalance

a requirement for model explainability

XGBoost also integrates naturally with SHAP for both local and global explanations.

Model Feature Strategy

The model uses a curated feature subset rather than automatically consuming every analytical field.

Selection is based on:

prediction-time availability

business relevance

credit-risk coverage

data quality

interpretability

representation across application, bureau and repayment behaviour

The wider analytical dataset remains available for EDA and Talk-to-Data.

Preprocessing

Numeric Features

Median Imputation
+ Missing-Value Indicators

Missingness indicators are retained so the model can learn whether missing information itself has predictive value.

Categorical Features

Missing → "Unknown"
One-Hot Encoding
handle_unknown = ignore

This keeps inference consistent while safely handling unseen categories.

Train, Calibration and Test Strategy

The labelled dataset is split using stratified sampling.

Split

Purpose

70%

XGBoost model training

10%

Probability calibration, operating-threshold selection, risk-band definition and SHAP-based explanatory feature selection

20%

Final untouched evaluation

The operating threshold is selected only on the calibration set by maximizing F1. The test set remains independent of fitting, calibration and threshold selection.

Handling Class Imbalance

The default class represents only about 8% of the labelled portfolio, so accuracy alone would not be meaningful.

The pipeline handles imbalance through:

stratified train/calibration/test splitting

dynamic XGBoost scale_pos_weight

calibration-set operating-threshold selection

This gives the minority default class greater learning importance without generating synthetic applicant records.

Final Model Evaluation

Evaluation is performed only on the untouched 20% test set.

Metric

Result

ROC-AUC

0.7645

PR-AUC

0.2586

PR-AUC Lift

3.20×

Precision

0.2541

Recall

0.4109

F1 Score

0.3140

Brier Score

0.0672

Operating Threshold

0.1562

Risk-Band Validation

Risk Band

Applicants

Observed Default Rate

Average Predicted Default

Low

20,437

2.070%

2.089%

Medium

20,981

5.481%

5.531%

High

20,085

16.888%

16.547%

The progression from Low to High risk demonstrates clear portfolio separation, while the close agreement between observed and predicted rates supports calibrated risk interpretation.

Probability Calibration and Risk Scoring

Raw XGBoost decision scores are calibrated using Platt / sigmoid calibration on the dedicated calibration split.

The user-facing model risk score is:

Risk Score = Probability of Default × 100

Applicants are grouped into:

Low
Medium
High

risk bands derived from the calibrated probability distribution.

These bands represent relative portfolio risk segments rather than hard-coded approval policies.

Explainable AI

The application uses SHAP to make predictions interpretable.

Local Explainability

For an individual applicant, the system shows:

probability of default

risk score

risk classification

major contributing factors

whether each factor increases or reduces predicted risk

SHAP contribution chart

Global Explainability

Global SHAP feature importance identifies the variables with the strongest influence across model predictions.

The current portfolio view highlights features such as:

EXT_SOURCE_2

EXT_SOURCE_3

EXT_SOURCE_1

CREDIT_GOODS_RATIO

BUREAU_DEBT_SUM

AMT_ANNUITY

LATE_PAYMENT_RATE

age and employment tenure

Global explanatory feature selection uses calibration data, preserving the test set for final evaluation.

ML-Derived Business Rules

A shallow surrogate decision tree summarizes common XGBoost decision patterns into readable rules:

IF business conditions
THEN approximate model probability of default

The XGBoost model remains the source of final applicant-level predictions.

Talk-to-Data

The platform includes a natural-language analytics assistant powered by Gemini.

Users can ask analytical questions such as:

What is the observed default rate?
How many applicants are in each risk band?
What is the observed default rate for each risk band?
Compare historical late-payment behaviour across risk bands.
What is the average requested credit amount by risk band?

NL-to-SQL Workflow

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

Prompt Engineering

The generation prompt includes:

approved analytical schema

business descriptions of key fields

distinction between observed outcome (TARGET) and predicted probability (MODEL_PD)

five representative few-shot query patterns

SQL safety constraints

recent conversation context only

This keeps the model grounded in the available data and limits unnecessary token usage.

SQL Safety and Result Grounding

Generated SQL is validated before execution.

The validation layer enforces:

exactly one SQL statement

SELECT-only access

approved table name

approved schema columns

no destructive operations

read-only SQLite execution

Blocked operations include:

INSERT
UPDATE
DELETE
DROP
CREATE
ALTER

Business answers are generated only after validated SQL executes successfully and are grounded in the returned values.

A single bounded repair attempt is available for a rejected query; there is no unrestricted autonomous retry loop.

Deterministic SQL Fallback Strategy

The five core analytical questions also have a deterministic SQL path.

User Question
      │
      ▼
Gemini NL-to-SQL
      │
      ├── Standard Path
      │       ↓
      │   Validated SQL
      │       ↓
      │    SQLite
      │       ↓
      │ Business Answer
      │
      └── Core-Query Fallback
              ↓
        Prevalidated SQL Template
              ↓
            SQLite
              ↓
      Deterministic Business Answer

The fallback covers the five predefined portfolio questions and uses only prevalidated read-only SQL.

This provides two complementary behaviours:

flexibility for custom analytical questions through Gemini

deterministic continuity for the platform's five core business questions

No second LLM provider or additional dependency is introduced.

Application Interface

1. EDA

Provides:

portfolio KPIs

five business insights

five supporting charts

feature categorization

data-quality summary

2. Machine Learning

Provides:

final test-set model performance

test-set risk-band validation

applicant probability of default

model risk score

Low / Medium / High risk classification

applicant profile

local SHAP explanation

global SHAP importance

ML-derived business rules

3. Chatbot

Provides:

five ready-made analytical questions

custom natural-language questions

readable business answers

supporting query results

optional generated SQL view

deterministic SQL support for the five core questions

System Architecture

Home Credit Source Data
        │
        ▼
Data Validation + Loading
        │
        ▼
Cleaning + Aggregation + Feature Engineering
        │
        ▼
Applicant-Level Analytical Dataset
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
       EDA                 ML Pipeline
                               │
                               ▼
                            XGBoost
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Platt Calibration           SHAP
                    │                     │
                    ▼                     ▼
           PD + Risk Score/Bands    Explanations
        │
        └──────────────────────┐
                               ▼
                         SQLite Database
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
             Gemini NL-to-SQL       Core SQL Fallback
                   │                       │
                   ▼                       │
             SQL Validation               │
                   └───────────┬───────────┘
                               ▼
                       Read-Only Execution
                               │
                               ▼
                      Business-Readable Answer
                               │
                               ▼
                         Streamlit Interface

Major Design Decisions

Decision

Selected Approach

Rationale

Predictive Model

XGBoost

Strong performance on structured tabular data and SHAP compatibility

Imbalance Handling

scale_pos_weight + stratified splits

Improves minority-default learning without synthetic records

Data Split

70 / 10 / 20

Separates training, calibration and final evaluation

Probability Calibration

Platt calibration

Supports interpretable user-facing default probabilities

Operating Threshold

Calibration-set F1 maximization

Provides an imbalance-aware classification operating point

Risk Bands

Calibration-distribution thresholds

Creates relative Low / Medium / High portfolio segmentation

Explainability

SHAP

Supports local and global explanations

Business Rules

Shallow surrogate tree

Converts model behaviour into readable patterns

Analytical Database

SQLite

Lightweight and reproducible

LLM

Gemini

Supports schema-grounded NL-to-SQL generation

SQL Validation

SQLGlot

Enforces safe analytical SQL

Core Query Resilience

Prevalidated SQL templates

Keeps five business-critical analytical paths deterministic

Interface

Streamlit

Lightweight end-to-end application

Deployment

Docker Compose

Reproducible execution

Technology Stack

Layer

Technologies

Programming

Python

Data Processing

Pandas, NumPy

Machine Learning

Scikit-learn, XGBoost

Explainability

SHAP

Data Storage

Parquet, SQLite

SQL Validation

SQLGlot

Generative AI

Gemini API, Google GenAI SDK

User Interface

Streamlit

Artifact Persistence

Joblib

Deployment

Docker, Docker Compose

Project Structure

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

Installation

1. Clone

git clone https://github.com/Rajyasri24/AI-Credit-Risk-Platform.git
cd AI-Credit-Risk-Platform

2. Create Environment

Windows:

py -3.11 -m venv .venv
.venv\Scripts\activate

Linux / macOS:

python3.11 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

3. Add Dataset

Place the required Home Credit files inside data/.

4. Configure Environment

Create .env from .env.example:

GEMINI_API_KEY=your_api_key
LLM_MODEL=gemini-3.6-flash

Run Locally

Build the analytical dataset:

python -m src.data.preprocessor

Train and persist the model:

python -m src.ml.train

Build the analytical database:

python -m src.talk_to_data.query_runner

Start the application:

streamlit run app.py

Open:

http://localhost:8501

Run with Docker

docker compose up --build

Open:

http://localhost:8501

Stop:

docker compose down

Generated Artifacts

data/credit_applicants.parquet
data/credit_risk.db

models/xgboost_model.joblib
models/preprocessor.joblib
models/model_metadata.joblib
models/surrogate_model.joblib

Repository

https://github.com/Rajyasri24/AI-Credit-Risk-Platform
