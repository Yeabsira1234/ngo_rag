import json
from collections.abc import Mapping
from typing import Any, Protocol

from src.agent.models import (
    AgentCitation,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolProvenance,
)
from src.agent.organization_data import (
    SAMPLE_ORGANIZATION_INFO,
    OrganizationInfoCategory,
    SampleOrganizationInfo,
)
from src.rag_service import RAGService
from src.sql.models import SQLOperation
from src.sql.repository import SQLServerRepository, SQLToolError
from src.sql.generation import SQLGenerationError
from src.sql.natural_language import NaturalLanguageSQLService
from src.sql.validation import SQLValidationError


class AgentTool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    def execute(self, arguments: Mapping[str, Any]) -> ToolExecutionResult: ...


class ToolRegistry:
    """Validated registry of the tools exposed to the agent model."""

    def __init__(self, tools: tuple[AgentTool, ...]) -> None:
        tool_map = {tool.definition.name: tool for tool in tools}
        if len(tool_map) != len(tools):
            raise ValueError("Agent tool names must be unique.")
        self._tools = tool_map
        self.definitions = tuple(tool.definition for tool in tools)

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)


class DocumentSearchTool:
    """Expose the existing RAG workflow as an agent-callable tool."""

    NAME = "document_search"

    def __init__(self, rag_service: RAGService) -> None:
        self.rag_service = rag_service

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.NAME,
            description=(
                "Search the indexed internal document when the user asks about "
                "its policies, facts, procedures, or other document content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete document-related question.",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Mapping[str, Any]) -> ToolExecutionResult:
        if set(arguments) != {"question"}:
            raise ValueError("document_search requires only the question argument.")

        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("document_search question must be a non-empty string.")

        response = self.rag_service.answer(question.strip())
        return ToolExecutionResult(
            answer=response.answer,
            status=ToolExecutionStatus.from_rag_status(response.status),
            citations=tuple(
                AgentCitation.from_source_reference(citation)
                for citation in response.citations
            ),
            rag_llm_called=response.llm_called,
            source=self.NAME,
            provenance=ToolProvenance.DOCUMENT,
            category="document_answer",
        )


class OrganizationInfoTool:
    """Return one category of safe, fictional organization information."""

    NAME = "organization_info"
    NOT_FOUND_MESSAGE = (
        "No fictional sample organization information is available for that "
        "category."
    )

    def __init__(
        self,
        data: SampleOrganizationInfo = SAMPLE_ORGANIZATION_INFO,
    ) -> None:
        self.data = data

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.NAME,
            description=(
                "Look up one structured fact from the fictional sample "
                "organization directory. Use this for the organization name, "
                "support hours, contact email, office location, or service "
                "categories; do not use it for document policies."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [category.value for category in OrganizationInfoCategory],
                        "description": "The organization information category.",
                    }
                },
                "required": ["category"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Mapping[str, Any]) -> ToolExecutionResult:
        if set(arguments) != {"category"}:
            raise ValueError("organization_info requires only the category argument.")

        raw_category = arguments.get("category")
        if not isinstance(raw_category, str):
            raise ValueError("organization_info category must be a string.")

        try:
            category = OrganizationInfoCategory(raw_category)
        except ValueError:
            return ToolExecutionResult(
                answer=self.NOT_FOUND_MESSAGE,
                status=ToolExecutionStatus.NOT_FOUND,
                citations=(),
                rag_llm_called=False,
                source=self.NAME,
                provenance=ToolProvenance.STRUCTURED_ORGANIZATION_DATA,
                category=raw_category,
            )

        return ToolExecutionResult(
            answer=self.data.value_for(category),
            status=ToolExecutionStatus.ANSWERED,
            citations=(),
            rag_llm_called=False,
            source=self.NAME,
            provenance=ToolProvenance.STRUCTURED_ORGANIZATION_DATA,
            category=category.value,
        )


class SQLQueryTool:
    """Expose only predefined, parameterized, read-only SQL operations."""

    NAME = "sql_query"
    SAFE_ERROR_MESSAGE = (
        "The structured database request could not be completed safely."
    )

    POLICY_ERROR_MESSAGE = (
        "The database request could not be completed under the read-only data policy."
    )

    def __init__(
        self,
        repository: SQLServerRepository,
        natural_language_service: NaturalLanguageSQLService | None = None,
    ) -> None:
        self.repository = repository
        self.natural_language_service = natural_language_service

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.NAME,
            description=(
                "Query the structured NGO database using one predefined "
                "read-only operation. Use natural_language_query with question "
                "for rankings, comparisons, grouped counts, most-common questions, "
                "or any flexible aggregate question; answer these with one natural "
                "language operation rather than chaining predefined operations. "
                "Use recent_service_events for recent, latest, or 'most recent' service "
                "event requests. Use office_name "
                "only for office-filtered predefined operations and language only "
                "for count_clients_by_language."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [operation.value for operation in SQLOperation],
                    },
                    "office_name": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                    "question": {"type": ["string", "null"]},
                },
                "required": ["operation", "office_name", "language", "question"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Mapping[str, Any]) -> ToolExecutionResult:
        raw_operation = arguments.get("operation")
        if not isinstance(raw_operation, str):
            raise ValueError("sql_query operation must be a string.")
        try:
            operation = SQLOperation(raw_operation)
        except ValueError as error:
            raise ValueError("Unknown sql_query operation.") from error
        if set(arguments) != {"operation", "office_name", "language", "question"}:
            raise ValueError("sql_query received malformed arguments.")
        if operation is SQLOperation.NATURAL_LANGUAGE_QUERY:
            question = arguments.get("question")
            if (
                self.natural_language_service is None
                or not isinstance(question, str)
                or not question.strip()
                or arguments.get("office_name") is not None
                or arguments.get("language") is not None
            ):
                raise ValueError("natural_language_query requires only a question.")
            try:
                result = self.natural_language_service.query(question.strip())
            except (SQLGenerationError, SQLValidationError):
                return self._safe_error(
                    self.POLICY_ERROR_MESSAGE,
                    operation,
                    "sql_policy_rejection",
                )
            except SQLToolError:
                return self._safe_error(
                    self.SAFE_ERROR_MESSAGE,
                    operation,
                    "sql_execution_failure",
                )
            return self._answered(result, operation)
        parameters = {
            key: value
            for key, value in arguments.items()
            if key in {"office_name", "language"} and value is not None
        }
        try:
            result = self.repository.execute(operation, parameters)
        except SQLToolError:
            return self._safe_error(
                self.SAFE_ERROR_MESSAGE,
                operation,
                "sql_execution_failure",
            )
        return self._answered(result, operation)

    def _safe_error(
        self,
        message: str,
        operation: SQLOperation,
        failure_category: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            answer=message,
            status=ToolExecutionStatus.ERROR,
            citations=(),
            rag_llm_called=False,
            source=self.NAME,
            provenance=ToolProvenance.STRUCTURED_SQL_DATA,
            category=operation.value,
            failure_category=failure_category,
        )

    def _answered(self, result: Any, operation: SQLOperation) -> ToolExecutionResult:
        return ToolExecutionResult(
            answer=json.dumps(result.to_model_output(), default=str),
            status=ToolExecutionStatus.ANSWERED,
            citations=(),
            rag_llm_called=False,
            source=self.NAME,
            provenance=ToolProvenance.STRUCTURED_SQL_DATA,
            category=operation.value,
        )
