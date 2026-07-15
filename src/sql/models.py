from dataclasses import dataclass
from enum import Enum
from typing import Any


class SQLOperation(str, Enum):
    LIST_OFFICES = "list_offices"
    LIST_PROGRAMS = "list_programs"
    COUNT_CASES_BY_STATUS = "count_cases_by_status"
    LIST_PROGRAMS_BY_OFFICE = "list_programs_by_office"
    LIST_STAFF_BY_OFFICE = "list_staff_by_office"
    LIST_OPEN_CASES = "list_open_cases"
    COUNT_CLIENTS_BY_LANGUAGE = "count_clients_by_language"
    RECENT_SERVICE_EVENTS = "recent_service_events"


@dataclass(frozen=True, slots=True)
class SQLQueryResult:
    operation: SQLOperation
    rows: tuple[dict[str, Any], ...]
    truncated: bool

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_model_output(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "source": "sql_query",
            "rows": list(self.rows),
        }
