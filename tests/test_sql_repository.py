from unittest.mock import Mock

import pytest

from src.sql.models import SQLOperation
from src.sql.repository import (
    QUERIES,
    SQLInvalidParametersError,
    SQLServerRepository,
)


def _repository(rows, *, max_rows=2):
    cursor = Mock()
    cursor.description = [("Value",)]
    cursor.fetchmany.return_value = rows
    connection = Mock()
    connection.cursor.return_value = cursor
    repository = SQLServerRepository(
        Mock(return_value=connection), timeout_seconds=10, max_rows=max_rows
    )
    return repository, cursor


@pytest.mark.parametrize(
    "operation",
    [operation for operation in SQLOperation if operation is not SQLOperation.NATURAL_LANGUAGE_QUERY],
)
def test_each_operation_uses_its_predefined_select(operation) -> None:
    parameters = {}
    if operation in {
        SQLOperation.LIST_PROGRAMS_BY_OFFICE,
        SQLOperation.LIST_STAFF_BY_OFFICE,
    }:
        parameters = {"office_name": "Alexandria Community Office"}
    elif operation is SQLOperation.COUNT_CLIENTS_BY_LANGUAGE:
        parameters = {"language": "Amharic"}
    repository, cursor = _repository([])

    repository.execute(operation, parameters)

    query = cursor.execute.call_args.args[0]
    assert query == QUERIES[operation]
    assert query.lstrip().upper().startswith("SELECT")
    assert not any(word in query.upper() for word in (" INSERT ", " UPDATE ", " DELETE ", " DROP ", " EXEC "))


def test_user_value_is_parameterized() -> None:
    repository, cursor = _repository([])
    repository.execute(
        SQLOperation.LIST_PROGRAMS_BY_OFFICE,
        {"office_name": "Northern Virginia Office"},
    )
    query, value = cursor.execute.call_args.args
    assert "?" in query
    assert "Northern Virginia Office" not in query
    assert value == "Northern Virginia Office"


def test_query_timeout_is_assigned_to_connection_not_cursor() -> None:
    repository, cursor = _repository([])
    connection = repository.connection_factory.return_value

    repository.execute(SQLOperation.LIST_OFFICES, {})

    assert connection.timeout == 10
    assert "timeout" not in cursor.__dict__


def test_row_limit_and_truncation_are_enforced() -> None:
    repository, cursor = _repository([(1,), (2,), (3,)], max_rows=2)
    result = repository.execute(SQLOperation.LIST_OFFICES, {})
    cursor.fetchmany.assert_called_once_with(3)
    assert result.row_count == 2
    assert result.truncated is True


def test_generated_query_uses_parameters_and_independent_row_cap() -> None:
    repository, cursor = _repository([(1,), (2,), (3,)], max_rows=2)
    result = repository.execute_generated(
        "SELECT TOP (2) o.OfficeName FROM dbo.Offices AS o WHERE o.City = ?",
        ("Alexandria",),
    )
    cursor.execute.assert_called_once_with(
        "SELECT TOP (2) o.OfficeName FROM dbo.Offices AS o WHERE o.City = ?",
        "Alexandria",
    )
    cursor.fetchmany.assert_called_once_with(3)
    assert result.query_approved is True
    assert result.truncated is True


@pytest.mark.parametrize("parameters", [{}, {"office_name": ""}, {"extra": "x"}])
def test_required_parameters_are_validated(parameters) -> None:
    repository, _ = _repository([])
    with pytest.raises(SQLInvalidParametersError):
        repository.execute(SQLOperation.LIST_PROGRAMS_BY_OFFICE, parameters)
