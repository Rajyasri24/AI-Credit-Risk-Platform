def schema_text(schema):
    descriptions = {
        "TARGET":
            "Observed historical default "
            "indicator: 1 default, 0 non-default.",

        "AMT_INCOME_TOTAL":
            "Applicant reported annual income.",

        "AMT_CREDIT":
            "Requested credit amount.",

        "AMT_ANNUITY":
            "Loan annuity amount.",

        "EMPLOYMENT_YEARS":
            "Applicant employment duration "
            "in years.",

        "EXT_SOURCE_1":
            "External credit-risk indicator.",

        "EXT_SOURCE_2":
            "External credit-risk indicator.",

        "EXT_SOURCE_3":
            "External credit-risk indicator.",

        "LATE_PAYMENT_RATE":
            "Historical share of installment "
            "payments made late.",

        "MODEL_PD":
            "Calibrated model probability "
            "of default from 0 to 1.",

        "RISK_SCORE":
            "Model probability of default "
            "expressed from 0 to 100.",

        "RISK_BAND":
            "Data-derived model risk segment: "
            "Low, Medium or High.",
    }

    lines = [
        "TABLE: credit_applicants",
        "",
        "COLUMNS:",
    ]

    for column, dtype in (
        schema.items()
    ):
        description = (
            descriptions.get(
                column,
                ""
            )
        )

        lines.append(
            f"- {column} ({dtype})"
            + (
                f": {description}"
                if description
                else ""
            )
        )

    return "\n".join(lines)


FEW_SHOT_EXAMPLES = """
Example 1
Question:
What is the observed default rate?

SQL:
SELECT
    ROUND(AVG(TARGET) * 100, 2)
        AS default_rate_pct
FROM credit_applicants;

Example 2
Question:
What is the observed default rate by income type?

SQL:
SELECT
    NAME_INCOME_TYPE,
    COUNT(*) AS applicants,
    ROUND(AVG(TARGET) * 100, 2)
        AS default_rate_pct
FROM credit_applicants
GROUP BY NAME_INCOME_TYPE
ORDER BY default_rate_pct DESC;

Example 3
Question:
How many applicants are in each risk band?

SQL:
SELECT
    RISK_BAND,
    COUNT(*) AS applicants
FROM credit_applicants
GROUP BY RISK_BAND
ORDER BY CASE RISK_BAND
    WHEN 'Low' THEN 1
    WHEN 'Medium' THEN 2
    WHEN 'High' THEN 3
END;

Example 4
Question:
Which income types have the highest observed
default rate among groups with at least
100 applicants?

SQL:
SELECT
    NAME_INCOME_TYPE,
    COUNT(*) AS applicants,
    ROUND(AVG(TARGET) * 100, 2)
        AS default_rate_pct
FROM credit_applicants
GROUP BY NAME_INCOME_TYPE
HAVING COUNT(*) >= 100
ORDER BY default_rate_pct DESC;

Example 5
Question:
Compare average historical late-payment rate
across model risk bands.

SQL:
SELECT
    RISK_BAND,
    ROUND(
        AVG(LATE_PAYMENT_RATE) * 100,
        2
    ) AS avg_late_payment_rate_pct
FROM credit_applicants
GROUP BY RISK_BAND
ORDER BY CASE RISK_BAND
    WHEN 'Low' THEN 1
    WHEN 'Medium' THEN 2
    WHEN 'High' THEN 3
END;
"""


def sql_generation_prompt(
    question,
    schema,
    history="",
):
    return f"""
You are the SQL generation component of a
credit-risk analytics application.

Convert the user's question into exactly one
read-only SQLite SELECT query.

STRICT RULES:
1. Use only the table and columns in the schema.
2. Never invent tables or columns.
3. Only SELECT queries are permitted.
4. Never INSERT, UPDATE, DELETE, DROP, ALTER,
   CREATE, ATTACH, PRAGMA or execute commands.
5. TARGET is the observed historical default
   outcome. TARGET=1 means default.
6. MODEL_PD is model-predicted probability.
7. RISK_SCORE = MODEL_PD * 100.
8. RISK_BAND is model-generated Low/Medium/High.
9. Do not confuse observed TARGET with model risk.
10. Use COUNT and minimum-support filtering when
    ranking categorical default rates if the
    question implies meaningful group comparison.
11. Return valid JSON only:
    {{"sql":"<SQL query>"}}

SCHEMA
------
{schema_text(schema)}

WORKING EXAMPLES
----------------
{FEW_SHOT_EXAMPLES}

RECENT CONVERSATION
-------------------
{history or "No previous conversation."}

USER QUESTION
-------------
{question}
"""


def repair_prompt(
    question,
    schema,
    invalid_sql,
    error,
):
    return f"""
Correct the invalid SQLite query.

Return exactly:
{{"sql":"<corrected SELECT query>"}}

Do not explain.

Original user question:
{question}

Schema:
{schema_text(schema)}

Invalid SQL:
{invalid_sql}

Validation or execution error:
{error}

Rules:
- SELECT only.
- One statement.
- Use only credit_applicants.
- Use only columns present in the schema.
"""


def result_summary_prompt(
    question,
    sql,
    result_text,
):
    return f"""
You are presenting a credit-risk analytics
query result to a business user.

Answer using ONLY the SQL result below.

Do not invent causes, recommendations,
statistics or facts not contained in the result.

Be concise.
Differentiate observed defaults from
model-predicted risk.

Question:
{question}

Validated SQL:
{sql}

Query result:
{result_text}

Write a clear business-readable answer in
2 to 4 sentences.
"""