from unittest.mock import Mock

from src.agent.models import ToolExecutionStatus, ToolProvenance
from src.agent.tools import SQLQueryTool
from src.sql.models import SQLOperation, SQLQueryResult
from src.sql.repository import SQLConnectionError


def test_sql_tool_returns_typed_read_only_result() -> None:
    repository = Mock()
    repository.execute.return_value = SQLQueryResult(
        SQLOperation.LIST_OFFICES, ({"OfficeName": "Northern Virginia"},), False
    )
    result = SQLQueryTool(repository).execute(
        {"operation": "list_offices", "office_name": None, "language": None}
    )
    assert result.status is ToolExecutionStatus.ANSWERED
    assert result.provenance is ToolProvenance.STRUCTURED_SQL_DATA
    assert "Northern Virginia" in result.answer
    repository.execute.assert_called_once_with(SQLOperation.LIST_OFFICES, {})


def test_sql_failure_returns_safe_tool_result() -> None:
    repository = Mock()
    repository.execute.side_effect = SQLConnectionError("server detail")
    result = SQLQueryTool(repository).execute(
        {"operation": "list_offices", "office_name": None, "language": None}
    )
    assert result.status is ToolExecutionStatus.ERROR
    assert "server detail" not in result.answer
