from unittest.mock import Mock

from src.agent.models import ToolExecutionStatus, ToolProvenance
from src.agent.tools import SQLQueryTool
from src.sql.models import SQLOperation, SQLQueryResult
from src.sql.repository import SQLConnectionError
from src.sql.validation import SQLUnsafeQueryError


def test_sql_tool_returns_typed_read_only_result() -> None:
    repository = Mock()
    repository.execute.return_value = SQLQueryResult(
        SQLOperation.LIST_OFFICES, ({"OfficeName": "Northern Virginia"},), False
    )
    result = SQLQueryTool(repository).execute(
        {"operation": "list_offices", "office_name": None, "language": None, "question": None}
    )
    assert result.status is ToolExecutionStatus.ANSWERED
    assert result.provenance is ToolProvenance.STRUCTURED_SQL_DATA
    assert "Northern Virginia" in result.answer
    repository.execute.assert_called_once_with(SQLOperation.LIST_OFFICES, {})


def test_sql_failure_returns_safe_tool_result() -> None:
    repository = Mock()
    repository.execute.side_effect = SQLConnectionError("server detail")
    result = SQLQueryTool(repository).execute(
        {"operation": "list_offices", "office_name": None, "language": None, "question": None}
    )
    assert result.status is ToolExecutionStatus.ERROR
    assert "server detail" not in result.answer


def test_natural_language_query_returns_approved_metadata_without_sql() -> None:
    repository = Mock()
    natural = Mock()
    natural.query.return_value = SQLQueryResult(
        SQLOperation.NATURAL_LANGUAGE_QUERY,
        ({"OfficeName": "North", "ProgramCount": 4},),
        False,
        True,
    )
    result = SQLQueryTool(repository, natural).execute({
        "operation": "natural_language_query",
        "office_name": None,
        "language": None,
        "question": "Which office has the most active programs?",
    })
    assert result.status is ToolExecutionStatus.ANSWERED
    assert '"query_approved": true' in result.answer
    assert "SELECT" not in result.answer


def test_unsafe_natural_language_sql_returns_safe_result() -> None:
    natural = Mock()
    natural.query.side_effect = SQLUnsafeQueryError("DELETE dbo.Cases")
    result = SQLQueryTool(Mock(), natural).execute({
        "operation": "natural_language_query",
        "office_name": None,
        "language": None,
        "question": "Delete all closed cases",
    })
    assert result.status is ToolExecutionStatus.ERROR
    assert "DELETE" not in result.answer
