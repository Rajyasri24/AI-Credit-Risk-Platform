import sqlite3

import joblib
import pandas as pd
import sqlglot
from sqlglot import exp

from src.utils.config import (
    ANALYTICAL_DATA,
    DATABASE_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
    SQL_DIR,
)


SQL_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",

    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",

    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",

    "AGE_YEARS",
    "EMPLOYMENT_YEARS",

    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",

    "BUREAU_CREDIT_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_OVERDUE_COUNT",
    "BUREAU_DEBT_SUM",

    "INSTALLMENT_COUNT",
    "AVG_PAYMENT_DELAY",
    "MAX_PAYMENT_DELAY",
    "LATE_PAYMENT_RATE",
    "AVG_PAYMENT_RATIO",

    "MODEL_PD",
    "RISK_SCORE",
    "RISK_BAND",
]


FORBIDDEN_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
)


def assign_risk_band(
    probability,
    thresholds,
):
    if probability <= thresholds[
        "low_max"
    ]:
        return "Low"

    if probability <= thresholds[
        "medium_max"
    ]:
        return "Medium"

    return "High"


def score_analytical_data():
    """
    Score the existing analytical dataset in memory.

    No additional scored parquet file is created.
    """

    if not ANALYTICAL_DATA.exists():
        raise FileNotFoundError(
            "credit_applicants.parquet was not found. "
            "Run preprocessing first."
        )

    required_artifacts = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH,
    ]

    missing_artifacts = [
        str(path)
        for path in required_artifacts
        if not path.exists()
    ]

    if missing_artifacts:
        raise FileNotFoundError(
            "Required model artifacts are missing:\n- "
            + "\n- ".join(
                missing_artifacts
            )
        )

    df = pd.read_parquet(
        ANALYTICAL_DATA
    )

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    model_features = metadata[
        "model_features"
    ]

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The analytical dataset is missing "
            "required model features: "
            f"{missing_features}"
        )

    X = df[
        model_features
    ].copy()

    X_transformed = (
        preprocessor.transform(
            X
        )
    )

    margins = model.predict(
        X_transformed,
        output_margin=True,
    )

    calibrator = metadata[
        "calibrator"
    ]

    probability = (
        calibrator
        .predict_proba(
            margins.reshape(
                -1,
                1,
            )
        )[:, 1]
    )

    thresholds = metadata[
        "risk_thresholds"
    ]

    df["MODEL_PD"] = probability

    df["RISK_SCORE"] = (
        probability * 100
    )

    df["RISK_BAND"] = [
        assign_risk_band(
            probability=p,
            thresholds=thresholds,
        )
        for p in probability
    ]

    return df


def build_database():
    """
    Build the SQLite analytical database.

    MODEL_PD, RISK_SCORE and RISK_BAND are
    calculated in memory before being written
    directly into SQLite.
    """

    df = score_analytical_data()

    missing_columns = [
        column
        for column in SQL_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot build SQLite database. "
            "Missing columns: "
            f"{missing_columns}"
        )

    sql_df = df[
        SQL_COLUMNS
    ].copy()

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        sql_df.to_sql(
            "credit_applicants",
            connection,
            if_exists="replace",
            index=False,
        )

        schema_file = (
            SQL_DIR / "schema.sql"
        )

        if schema_file.exists():
            with open(
                schema_file,
                "r",
                encoding="utf-8",
            ) as file:
                connection.executescript(
                    file.read()
                )

        connection.commit()

    finally:
        connection.close()


def get_schema():
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "SQLite database was not found. "
            "Run:\n"
            "python -m src.talk_to_data.query_runner"
        )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        cursor = connection.execute(
            """
            PRAGMA table_info(
                credit_applicants
            )
            """
        )

        rows = cursor.fetchall()

    finally:
        connection.close()

    if not rows:
        raise ValueError(
            "credit_applicants table "
            "does not exist."
        )

    return {
        row[1]: row[2]
        for row in rows
    }


def validate_sql(sql):
    if not sql or not sql.strip():
        raise ValueError(
            "Generated SQL is empty."
        )

    statements = sqlglot.parse(
        sql,
        read="sqlite",
    )

    if len(statements) != 1:
        raise ValueError(
            "Only one SQL statement "
            "is allowed."
        )

    parsed = statements[0]

    for forbidden_expression in (
        FORBIDDEN_EXPRESSIONS
    ):
        if parsed.find(
            forbidden_expression
        ):
            raise ValueError(
                "Only read-only SELECT "
                "queries are permitted."
            )

    if not parsed.find(
        exp.Select
    ):
        raise ValueError(
            "The query must contain SELECT."
        )

    tables = {
        table.name
        for table in parsed.find_all(
            exp.Table
        )
    }

    allowed_tables = {
        "credit_applicants"
    }

    if not tables:
        raise ValueError(
            "Query must reference "
            "credit_applicants."
        )

    if not tables.issubset(
        allowed_tables
    ):
        raise ValueError(
            "Query references an "
            "unauthorized table."
        )

    schema = get_schema()

    allowed_columns = set(
        schema.keys()
    )

    aliases = {
        alias.alias
        for alias in parsed.find_all(
            exp.Alias
        )
        if alias.alias
    }

    for column in parsed.find_all(
        exp.Column
    ):
        column_name = (
            column.name
        )

        if column_name == "*":
            continue

        if (
            column_name
            not in allowed_columns
            and column_name
            not in aliases
        ):
            raise ValueError(
                "Unknown column: "
                f"{column_name}"
            )

    return parsed


def add_safe_limit(
    parsed,
    limit=200,
):
    if parsed.args.get(
        "limit"
    ) is None:
        parsed = parsed.limit(
            limit
        )

    return parsed.sql(
        dialect="sqlite"
    )


def execute_query(sql):
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "SQLite database does not exist."
        )

    parsed = validate_sql(
        sql
    )

    safe_sql = add_safe_limit(
        parsed
    )

    connection = sqlite3.connect(
        f"file:{DATABASE_PATH}?mode=ro",
        uri=True,
    )

    try:
        result = pd.read_sql_query(
            safe_sql,
            connection,
        )

    finally:
        connection.close()

    return (
        safe_sql,
        result,
    )


if __name__ == "__main__":
    print(
        "Building credit-risk SQLite database..."
    )

    build_database()

    print(
        "Database built successfully:"
    )

    print(
        DATABASE_PATH
    )

    print(
        "\nAvailable schema:"
    )

    schema = get_schema()

    for column, dtype in (
        schema.items()
    ):
        print(
            f"- {column}: {dtype}"
        )