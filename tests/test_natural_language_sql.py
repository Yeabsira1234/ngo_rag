from unittest.mock import Mock

import pytest

from src.sql.generation import GeneratedSQL
from src.sql.models import SQLOperation, SQLQueryResult
from src.sql.natural_language import NaturalLanguageSQLService
from src.sql.validation import ApprovedSQL, SQLPrivacyError, SQLUnsafeQueryError
from src.sql.validation import SQLValidator
from src.sql.schema import APPROVED_SCHEMA


def test_approved_candidate_executes_once_without_exposing_sql() -> None:
    generator = Mock()
    generated = GeneratedSQL(sql="SELECT TOP (5) x", intent_summary="recent")
    generator.generate.return_value = generated
    validator = Mock()
    validator.validate.return_value = ApprovedSQL("SELECT TOP (5) safe", ())
    repository = Mock()
    expected = SQLQueryResult(SQLOperation.NATURAL_LANGUAGE_QUERY, ({"Count": 2},), False, True)
    repository.execute_generated.return_value = expected
    result = NaturalLanguageSQLService(generator, validator, repository).query("Question")
    assert result is expected
    repository.execute_generated.assert_called_once_with("SELECT TOP (5) safe", ())


def test_rejected_candidate_is_never_executed() -> None:
    generator, validator, repository = Mock(), Mock(), Mock()
    generator.generate.return_value = GeneratedSQL(sql="DELETE", intent_summary="bad")
    validator.validate.side_effect = SQLUnsafeQueryError("unsafe")
    with pytest.raises(SQLUnsafeQueryError):
        NaturalLanguageSQLService(generator, validator, repository).query("Question")
    repository.execute_generated.assert_not_called()


def test_sensitive_question_is_rejected_before_generation() -> None:
    generator, validator, repository = Mock(), Mock(), Mock()
    with pytest.raises(SQLPrivacyError):
        NaturalLanguageSQLService(generator, validator, repository).query(
            "Show every client and all their identifiers"
        )
    generator.generate.assert_not_called()
    repository.execute_generated.assert_not_called()


def test_open_client_language_regression_reaches_execution() -> None:
    generator = Mock()
    generator.generate.return_value = GeneratedSQL(
        sql=(
            "SELECT TOP (100) c.PreferredLanguage, COUNT_BIG(*) AS ClientCount "
            "FROM dbo.Clients AS c WHERE c.CaseStatus = ? "
            "GROUP BY c.PreferredLanguage "
            "ORDER BY ClientCount DESC, c.PreferredLanguage ASC"
        ),
        parameters=["Open"],
        intent_summary="Rank languages among open clients",
    )
    repository = Mock()
    repository.execute_generated.return_value = SQLQueryResult(
        SQLOperation.NATURAL_LANGUAGE_QUERY,
        ({"PreferredLanguage": "Amharic", "ClientCount": 3},),
        False,
        True,
    )
    service = NaturalLanguageSQLService(
        generator, SQLValidator(APPROVED_SCHEMA, 100), repository
    )
    result = service.query(
        "Which preferred languages are most common among open clients?"
    )
    assert result.rows[0]["PreferredLanguage"] == "Amharic"
    repository.execute_generated.assert_called_once()
