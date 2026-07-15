from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.sql.generation import (
    GeneratedSQL,
    OpenAISQLGenerator,
    SQLGenerationError,
    SQLStructuredOutputError,
)
from src.sql.schema import APPROVED_SCHEMA


def test_generator_uses_only_question_schema_and_structured_output() -> None:
    client = Mock()
    parsed = GeneratedSQL(
        sql="SELECT TOP (5) se.ServiceType FROM dbo.ServiceEvents AS se ORDER BY se.ServiceDate DESC",
        intent_summary="Recent services",
    )
    client.parse.return_value = SimpleNamespace(output_parsed=parsed)
    generator = OpenAISQLGenerator(client, "model", APPROVED_SCHEMA, 100)
    assert generator.generate("What are recent services?") is parsed
    call = client.parse.call_args.kwargs
    assert call["input"] == "What are recent services?"
    assert call["text_format"] is GeneratedSQL
    assert "dbo.Clients" in call["instructions"]
    assert "ClientCode" not in call["instructions"]
    assert "SERVER" not in call["instructions"]


def test_generator_wraps_model_failure() -> None:
    client = Mock()
    client.parse.side_effect = RuntimeError("provider detail")
    with pytest.raises(SQLGenerationError):
        OpenAISQLGenerator(client, "model", APPROVED_SCHEMA, 100).generate("q")


def test_generator_rejects_invalid_structured_output() -> None:
    client = Mock()
    client.parse.return_value = SimpleNamespace(output_parsed=None)
    with pytest.raises(SQLStructuredOutputError):
        OpenAISQLGenerator(client, "model", APPROVED_SCHEMA, 100).generate("q")
