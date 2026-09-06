import sqlite3

from src.data.preprocessor import (
    build_analytical_dataset,
)

from src.ml.predict import (
    load_artifacts,
)

from src.talk_to_data.query_runner import (
    build_database,
)

from src.utils.config import (
    ANALYTICAL_DATA,
    DATABASE_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
)


REQUIRED_DATABASE_COLUMNS = {
    "SK_ID_CURR",
    "TARGET",
    "MODEL_PD",
    "RISK_SCORE",
    "RISK_BAND",
}


def database_is_ready():
    """
    Check whether the SQLite database exists
    and contains the current required risk fields.
    """

    if not DATABASE_PATH.exists():
        return False

    try:
        connection = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = connection.execute(
            """
            PRAGMA table_info(
                credit_applicants
            )
            """
        )

        columns = {
            row[1]
            for row in cursor.fetchall()
        }

        connection.close()

        return (
            REQUIRED_DATABASE_COLUMNS
            .issubset(columns)
        )

    except Exception:
        return False


def ensure_runtime_assets():
    # ------------------------------------------
    # 1. Processed analytical dataset
    # ------------------------------------------

    if not ANALYTICAL_DATA.exists():
        print(
            "Analytical dataset not found."
        )

        print(
            "Building analytical dataset..."
        )

        build_analytical_dataset()

    # ------------------------------------------
    # 2. Trained model artifacts
    # ------------------------------------------

    required_models = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH,
    ]

    missing_models = [
        str(path)
        for path in required_models
        if not path.exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "Model artifacts are missing.\n"
            "Train the model first:\n\n"
            "python -m src.ml.train\n\n"
            "Missing files:\n- "
            + "\n- ".join(
                missing_models
            )
        )

    # Confirm artifacts are readable.
    load_artifacts()

    # ------------------------------------------
    # 3. SQLite analytical database
    # ------------------------------------------

    if not database_is_ready():
        print(
            "SQLite database is missing "
            "or outdated."
        )

        print(
            "Building SQLite database..."
        )

        build_database()

    if not database_is_ready():
        raise RuntimeError(
            "SQLite database could not "
            "be prepared correctly."
        )

    print(
        "Runtime assets ready."
    )


if __name__ == "__main__":
    ensure_runtime_assets()