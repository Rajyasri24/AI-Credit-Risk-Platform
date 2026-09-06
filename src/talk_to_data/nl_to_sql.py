import json
import os

from dotenv import load_dotenv
from google import genai

from src.talk_to_data.prompt_templates import (
    sql_generation_prompt,
    repair_prompt,
    result_summary_prompt,
)

from src.talk_to_data.query_runner import (
    get_schema,
    execute_query,
)


load_dotenv()


class NLToSQLClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in .env"
            )

        self.model = os.getenv(
            "LLM_MODEL",
            "gemini-3.6-flash",
        )

        self.client = genai.Client(
            api_key=api_key
        )

    def _text_call(self, prompt):
        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        text = interaction.output_text

        if not text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return text.strip()

    def _json_call(self, prompt):
        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string"
                        }
                    },
                    "required": ["sql"],
                },
            },
        )

        text = interaction.output_text

        if not text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return json.loads(text)

    def generate_sql(
        self,
        question,
        schema,
        history="",
    ):
        response = self._json_call(
            sql_generation_prompt(
                question,
                schema,
                history,
            )
        )

        sql = response.get("sql")

        if not sql:
            raise ValueError(
                "Gemini did not return SQL."
            )

        return sql.strip()

    def repair_sql(
        self,
        question,
        schema,
        invalid_sql,
        error,
    ):
        response = self._json_call(
            repair_prompt(
                question,
                schema,
                invalid_sql,
                error,
            )
        )

        sql = response.get("sql")

        if not sql:
            raise ValueError(
                "Gemini repair did not return SQL."
            )

        return sql.strip()

    def summarize(
        self,
        question,
        sql,
        result_text,
    ):
        return self._text_call(
            result_summary_prompt(
                question,
                sql,
                result_text,
            )
        )


def format_history(history):
    if not history:
        return ""

    recent = history[-3:]

    lines = []

    for item in recent:
        lines.append(
            f"User: {item.get('question', '')}"
        )

        lines.append(
            f"Assistant: {item.get('answer', '')}"
        )

    return "\n".join(lines)


def ask_credit_data(
    question,
    history=None,
):
    schema = get_schema()

    client = NLToSQLClient()

    history_text = format_history(
        history or []
    )

    generated_sql = client.generate_sql(
        question=question,
        schema=schema,
        history=history_text,
    )

    repair_used = False

    try:
        validated_sql, result = execute_query(
            generated_sql
        )

    except Exception as first_error:
        repair_used = True

        repaired_sql = client.repair_sql(
            question=question,
            schema=schema,
            invalid_sql=generated_sql,
            error=str(first_error),
        )

        validated_sql, result = execute_query(
            repaired_sql
        )

    result_text = (
        result
        .head(20)
        .to_string(index=False)
    )

    answer = client.summarize(
        question=question,
        sql=validated_sql,
        result_text=result_text,
    )

    return {
        "answer": answer,
        "sql": validated_sql,
        "result": result,
        "repair_used": repair_used,
    }