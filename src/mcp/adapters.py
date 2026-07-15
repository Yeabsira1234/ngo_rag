import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.agent.tools import AgentTool
from src.mcp.schemas import (
    CitationOutput,
    DocumentSearchOutput,
    OrganizationInformationOutput,
    SQLInformationOutput,
    WeatherInformationOutput,
)

logger = logging.getLogger(__name__)


class MCPAdapterError(RuntimeError):
    """Safe adapter error exposed through the MCP protocol."""


@dataclass(slots=True)
class MCPAdapters:
    """Thin typed mappings over the existing application tool boundaries."""

    tools: dict[str, AgentTool]
    max_input_length: int

    @classmethod
    def from_tools(
        cls, tools: tuple[AgentTool, ...], max_input_length: int
    ) -> "MCPAdapters":
        if max_input_length <= 0:
            raise ValueError("max_input_length must be greater than zero.")
        by_name = {tool.definition.name: tool for tool in tools}
        required = {
            "document_search",
            "organization_info",
            "sql_query",
            "weather_information",
        }
        if set(by_name) != required:
            raise ValueError("MCP requires the four approved application tools.")
        return cls(by_name, max_input_length)

    def search_documents(self, question: str) -> DocumentSearchOutput:
        result = self._execute(
            "search_documents",
            "document_search",
            {"question": self._text(question, "question")},
        )
        return DocumentSearchOutput(
            answer=result.answer,
            status=result.status.value,
            citations=[
                CitationOutput(
                    source=item.source,
                    page_number=item.page_number,
                    chunk_index=item.chunk_index,
                    distance=item.distance,
                    source_relative_path=item.source_relative_path,
                    document_id=item.document_id,
                )
                for item in result.citations
            ],
        )

    def organization_information(
        self, category: str
    ) -> OrganizationInformationOutput:
        result = self._execute(
            "organization_information",
            "organization_info",
            {"category": category},
        )
        return OrganizationInformationOutput(
            answer=result.answer,
            status=result.status.value,
            category=result.category,
            source=result.source,
        )

    def sql_information(
        self,
        operation: str,
        office_name: str | None,
        language: str | None,
        question: str | None,
    ) -> SQLInformationOutput:
        arguments = {
            "operation": operation,
            "office_name": self._optional_text(office_name, "office_name"),
            "language": self._optional_text(language, "language"),
            "question": self._optional_text(question, "question"),
        }
        result = self._execute("sql_information", "sql_query", arguments)
        data = (
            self._structured_answer(result.answer)
            if result.status.value == "answered"
            else None
        )
        return SQLInformationOutput(
            answer=result.answer,
            status=result.status.value,
            operation=result.category,
            data=data,
            failure_category=result.failure_category,
        )

    def weather_information(self, city: str) -> WeatherInformationOutput:
        result = self._execute(
            "weather_information",
            "weather_information",
            {"city": self._text(city, "city")},
        )
        data = (
            self._structured_answer(result.answer)
            if result.status.value == "answered"
            else None
        )
        return WeatherInformationOutput(
            answer=result.answer,
            status=result.status.value,
            data=data,
            failure_category=result.failure_category,
        )

    def _execute(
        self,
        mcp_tool_name: str,
        application_tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        started = time.monotonic()
        try:
            result = self.tools[application_tool_name].execute(arguments)
        except (TypeError, ValueError) as error:
            duration_ms = round((time.monotonic() - started) * 1000)
            logger.warning(
                "event=mcp_tool_completed tool=%s duration_ms=%d status=rejected failure_category=invalid_input",
                mcp_tool_name,
                duration_ms,
            )
            raise MCPAdapterError("The MCP tool input was invalid.") from error
        except Exception as error:
            duration_ms = round((time.monotonic() - started) * 1000)
            logger.warning(
                "event=mcp_tool_completed tool=%s duration_ms=%d status=error failure_category=dependency_failure",
                mcp_tool_name,
                duration_ms,
            )
            raise MCPAdapterError("The MCP tool could not be completed safely.") from error
        duration_ms = round((time.monotonic() - started) * 1000)
        logger.info(
            "event=mcp_tool_completed tool=%s duration_ms=%d status=%s failure_category=%s",
            mcp_tool_name,
            duration_ms,
            result.status.value,
            result.failure_category or "none",
        )
        return result

    def _text(self, value: str, field: str) -> str:
        if not isinstance(value, str):
            raise MCPAdapterError(f"{field} must be text.")
        normalized = value.strip()
        if not normalized or len(normalized) > self.max_input_length:
            raise MCPAdapterError(
                f"{field} must be non-empty and within the configured size limit."
            )
        return normalized

    def _optional_text(self, value: str | None, field: str) -> str | None:
        return None if value is None else self._text(value, field)

    @staticmethod
    def _structured_answer(answer: str) -> dict[str, Any]:
        try:
            value = json.loads(answer)
        except (json.JSONDecodeError, TypeError) as error:
            raise MCPAdapterError(
                "The application returned invalid structured data."
            ) from error
        if not isinstance(value, dict):
            raise MCPAdapterError("The application returned invalid structured data.")
        return value
