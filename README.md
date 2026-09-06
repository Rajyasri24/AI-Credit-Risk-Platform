## Major Design Decisions

The implementation was intentionally kept lightweight and modular so the complete platform can be understood, executed, and evaluated without unnecessary infrastructure.

| Decision | Choice | Reason |
|---|---|---|
| Predictive model | XGBoost | Strong performance on structured tabular data and direct compatibility with SHAP |
| Class imbalance | `scale_pos_weight` | Handles the minority default class without generating synthetic applicant records |
| Data split | 70% train / 10% calibration / 20% test | Separates model fitting, probability calibration, and final evaluation |
| Probability calibration | Platt / sigmoid calibration | Produces probabilities that are more suitable for user-facing default-risk estimates |
| Explainability | SHAP | Provides both applicant-level and portfolio-level model explanations |
| Business rules | Shallow surrogate decision tree | Converts model behaviour into simplified business-readable patterns |
| Analytical database | SQLite | Lightweight, reproducible and sufficient for the assignment-scale analytical workload |
| LLM integration | Gemini API | Supports schema-grounded natural-language-to-SQL generation and result summarization |
| LLM orchestration | Direct Google GenAI SDK | Keeps prompt construction, validation and repair logic explicit and easy to evaluate |
| UI | Streamlit | Provides a lightweight multi-section interface for the complete workflow |
| Deployment | Docker + Docker Compose | Enables reproducible one-command execution |

---

## Model Evaluation

The final model is evaluated on the untouched 20% test set.

The evaluation focuses on metrics suitable for an imbalanced credit-default problem.

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures overall ranking ability between default and non-default applicants |
| PR-AUC | Evaluates performance on the minority default class |
| Precision | Measures how many predicted defaults are actual defaults |
| Recall | Measures how many actual defaults are identified |
| F1 Score | Balances precision and recall |
| Brier Score | Evaluates probability calibration quality |

### Final Test Results

| Metric | Result |
|---|---:|
| ROC-AUC | `ADD_FINAL_VALUE` |
| PR-AUC | `ADD_FINAL_VALUE` |
| Precision | `ADD_FINAL_VALUE` |
| Recall | `ADD_FINAL_VALUE` |
| F1 Score | `ADD_FINAL_VALUE` |
| Calibrated Brier Score | `ADD_FINAL_VALUE` |

The probability calibration stage is evaluated separately from model ranking so that the probability shown to the user can be interpreted as a model-estimated probability of default rather than only a ranking score.

---

## Prompt Engineering and Hallucination Control

The Talk-to-Data component is designed as a constrained analytical workflow rather than an unrestricted chatbot.

### Prompt Grounding

The SQL-generation prompt provides:

- the approved SQLite table name
- available column names
- business descriptions for important fields
- distinction between historical outcomes and model predictions
- explicit SQL safety constraints
- representative few-shot examples
- limited recent conversation context

Two fields are deliberately distinguished:

```text
TARGET
```

represents the **observed historical default outcome**, while:

```text
MODEL_PD
```

represents the **model-predicted probability of default**.

This distinction prevents the LLM from interpreting predicted risk as an actual historical default rate.

### Few-Shot Query Patterns

The prompt includes at least five representative query patterns covering:

1. overall observed default rate
2. applicant distribution by risk band
3. observed default rate by risk band
4. grouped default analysis
5. historical repayment behaviour by risk band

### SQL Validation

LLM-generated SQL is validated before execution.

The validator ensures:

- exactly one SQL statement
- SELECT-only access
- approved table access
- approved schema columns
- no destructive SQL
- read-only SQLite execution

Detailed row-level queries are bounded to prevent unnecessarily large outputs.

### Bounded Repair

If generated SQL fails validation or execution, the system performs only **one controlled repair attempt** using:

- the original business question
- rejected SQL
- validation error
- approved schema

This prevents uncontrolled retry loops.

### Result Grounding

The final business response is generated only after successful SQL execution.

The LLM receives the validated query result and is instructed to:

- use only values returned by the database
- avoid inventing causes
- distinguish observed and predicted risk
- avoid interpreting SQL row limits as analytical sample size

---

## Prompt and Token Optimization

The Talk-to-Data workflow keeps LLM context intentionally compact.

Token usage is reduced through:

- a restricted analytical schema instead of the complete raw Home Credit schema
- concise business descriptions of approved fields
- a fixed set of representative few-shot examples
- only the most recent conversation turns
- one bounded SQL repair attempt
- result summarization based on returned rows instead of resending the full underlying dataset

The LLM never receives the complete 307,511-row portfolio.

SQL performs the analytical computation and only the resulting data is provided to the model for summarization.

---

## ML-Derived Rule Logic

The primary predictive model remains XGBoost.

A shallow global surrogate decision tree is used only to extract simplified decision patterns from the trained model.

The rule-generation process is:

```text
Trained XGBoost Model
        ↓
Model Probability Predictions
        ↓
Most Influential Model Features
        ↓
Shallow Decision Tree Approximation
        ↓
Readable IF / THEN Patterns
        ↓
Displayed as ML-Derived Business Rules
```

The surrogate does not replace the primary model and its output is not used as the final applicant prediction.

Its role is to make recurring model behaviour easier to interpret for business users.

### Example Rule Format

```text
IF
    Feature A <= threshold
AND
    Feature B > threshold

THEN
    Approximate model probability of default = XX%
```

Actual rules displayed in the application are derived from the trained model artifacts.

---

## Sample Application Outputs

### Applicant Risk Prediction

```text
Applicant ID
    ↓
Probability of Default
    ↓
Risk Score (0–100)
    ↓
Low / Medium / High Risk
```

Example presentation:

```text
Probability of Default: 12.4%
Risk Score: 12.4 / 100
Risk Classification: Medium
```

The displayed values for an actual applicant are calculated from the saved model artifacts at runtime.

### SHAP Explanation

For an applicant, the platform presents the strongest model drivers as:

```text
Feature                       Effect
------------------------------------------------
External Credit Indicator     Reduces predicted risk
Late-Payment Behaviour        Increases predicted risk
Employment History            Reduces predicted risk
```

A SHAP contribution chart accompanies the readable explanation.

### Talk-to-Data

Example user question:

```text
What is the observed default rate?
```

Example SQL structure:

```sql
SELECT
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS defaulted_applicants,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM credit_applicants;
```

Business output:

```text
The observed historical default rate is 8.07%.
```

The response is generated only after the SQL query has been validated and executed.

---

## Future Improvements

Potential extensions include:

- external policy-defined risk thresholds for production lending environments
- model monitoring and drift detection
- authentication and role-based access control
- additional portfolio datasets and credit-history sources
- persistent conversational history
- cloud-hosted analytical database for higher concurrency
- automated model retraining and model registry integration
