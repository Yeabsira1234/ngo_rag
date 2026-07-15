import logging
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Any, Protocol

from src.sql.models import SQLOperation, SQLQueryResult

logger = logging.getLogger(__name__)


class SQLToolError(RuntimeError): pass
class SQLConfigurationError(SQLToolError): pass
class SQLConnectionError(SQLToolError): pass
class SQLTimeoutError(SQLToolError): pass
class SQLUnknownOperationError(SQLToolError): pass
class SQLInvalidParametersError(SQLToolError): pass
class SQLExecutionError(SQLToolError): pass


class Cursor(Protocol):
    timeout: int
    description: Any
    def execute(self, query: str, *parameters: Any) -> Any: ...
    def fetchmany(self, size: int) -> list[Any]: ...
    def close(self) -> None: ...


class Connection(Protocol):
    timeout: int
    def cursor(self) -> Cursor: ...
    def close(self) -> None: ...


QUERIES: dict[SQLOperation, str] = {
    SQLOperation.LIST_OFFICES: """SELECT OfficeID, OfficeName, City, StateCode, Phone, Email, IsActive FROM dbo.Offices WHERE IsActive = 1 ORDER BY OfficeName, OfficeID""",
    SQLOperation.LIST_PROGRAMS: """SELECT p.ProgramID, p.ProgramName, p.Category, o.OfficeName, p.StartDate, p.IsActive FROM dbo.Programs AS p INNER JOIN dbo.Offices AS o ON o.OfficeID = p.OfficeID WHERE p.IsActive = 1 ORDER BY p.ProgramName, p.ProgramID""",
    SQLOperation.COUNT_CASES_BY_STATUS: """SELECT CaseStatus, COUNT_BIG(*) AS CaseCount FROM dbo.Cases GROUP BY CaseStatus ORDER BY CaseStatus""",
    SQLOperation.LIST_PROGRAMS_BY_OFFICE: """SELECT p.ProgramID, p.ProgramName, p.Category, o.OfficeName, p.StartDate FROM dbo.Programs AS p INNER JOIN dbo.Offices AS o ON o.OfficeID = p.OfficeID WHERE o.OfficeName = ? AND p.IsActive = 1 ORDER BY p.ProgramName, p.ProgramID""",
    SQLOperation.LIST_STAFF_BY_OFFICE: """SELECT s.StaffID, s.FirstName, s.LastName, s.JobTitle, s.Department, o.OfficeName FROM dbo.Staff AS s INNER JOIN dbo.Offices AS o ON o.OfficeID = s.OfficeID WHERE o.OfficeName = ? AND s.IsActive = 1 ORDER BY s.LastName, s.FirstName, s.StaffID""",
    SQLOperation.LIST_OPEN_CASES: """SELECT c.CaseID, cl.ClientCode, p.ProgramName, c.OpenDate, c.PriorityLevel, c.CaseStatus FROM dbo.Cases AS c INNER JOIN dbo.Clients AS cl ON cl.ClientID = c.ClientID INNER JOIN dbo.Programs AS p ON p.ProgramID = c.ProgramID WHERE c.CloseDate IS NULL ORDER BY c.OpenDate DESC, c.CaseID DESC""",
    SQLOperation.COUNT_CLIENTS_BY_LANGUAGE: """SELECT PreferredLanguage, COUNT_BIG(*) AS ClientCount FROM dbo.Clients WHERE PreferredLanguage = ? GROUP BY PreferredLanguage ORDER BY PreferredLanguage""",
    SQLOperation.RECENT_SERVICE_EVENTS: """SELECT se.ServiceEventID, cl.ClientCode, se.ServiceType, se.ServiceDate, se.DurationMinutes, se.Outcome FROM dbo.ServiceEvents AS se INNER JOIN dbo.Cases AS c ON c.CaseID = se.CaseID INNER JOIN dbo.Clients AS cl ON cl.ClientID = c.ClientID ORDER BY se.ServiceDate DESC, se.ServiceEventID DESC""",
}


class SQLServerRepository:
    def __init__(self, connection_factory: Callable[[], Connection], *, timeout_seconds: int, max_rows: int) -> None:
        if timeout_seconds <= 0 or max_rows <= 0:
            raise SQLConfigurationError("SQL limits must be greater than zero.")
        self.connection_factory = connection_factory
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def execute(self, operation: SQLOperation, parameters: Mapping[str, Any]) -> SQLQueryResult:
        query = QUERIES.get(operation)
        if query is None:
            raise SQLUnknownOperationError("Unsupported SQL operation.")
        expected = {
            SQLOperation.LIST_PROGRAMS_BY_OFFICE: "office_name",
            SQLOperation.LIST_STAFF_BY_OFFICE: "office_name",
            SQLOperation.COUNT_CLIENTS_BY_LANGUAGE: "language",
        }.get(operation)
        if expected is None:
            if parameters:
                raise SQLInvalidParametersError("This operation accepts no parameters.")
            values: tuple[Any, ...] = ()
        else:
            if set(parameters) != {expected}:
                raise SQLInvalidParametersError(f"This operation requires {expected}.")
            value = parameters[expected]
            if not isinstance(value, str) or not value.strip() or len(value) > 150:
                raise SQLInvalidParametersError(f"{expected} must be a non-empty string.")
            values = (value.strip(),)
        return self._execute_query(operation, query, values)

    def execute_generated(
        self, sql: str, parameters: tuple[object, ...]
    ) -> SQLQueryResult:
        return self._execute_query(
            SQLOperation.NATURAL_LANGUAGE_QUERY,
            sql,
            parameters,
            query_approved=True,
        )

    def _execute_query(
        self,
        operation: SQLOperation,
        query: str,
        values: tuple[Any, ...],
        *,
        query_approved: bool | None = None,
    ) -> SQLQueryResult:
        started = time.monotonic()
        connection = cursor = None
        try:
            connection = self.connection_factory()
            connection.timeout = self.timeout_seconds
            cursor = connection.cursor()
            cursor.execute(query, *values)
            raw_rows = cursor.fetchmany(self.max_rows + 1)
            columns = [column[0] for column in cursor.description]
            rows = tuple(
                {name: _serialize(value) for name, value in zip(columns, row, strict=True)}
                for row in raw_rows[: self.max_rows]
            )
            result = SQLQueryResult(
                operation, rows, len(raw_rows) > self.max_rows, query_approved
            )
            logger.info("event=sql_query_completed operation=%s duration_ms=%d row_count=%d truncated=%s", operation.value, int((time.monotonic()-started)*1000), result.row_count, result.truncated)
            return result
        except SQLToolError:
            raise
        except Exception as error:
            name = type(error).__name__.lower()
            message = str(error).lower()
            logger.exception("event=sql_query_failed operation=%s duration_ms=%d error_type=%s", operation.value, int((time.monotonic()-started)*1000), type(error).__name__)
            if "timeout" in name or "timeout" in message:
                raise SQLTimeoutError("The database request timed out.") from error
            if connection is None:
                raise SQLConnectionError("The database is unavailable.") from error
            raise SQLExecutionError("The database request failed.") from error
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()


def _serialize(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
