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

    def __init__(self, repository: SQLServerRepository) -> None:
        self.repository = repository

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.NAME,
            description=(
                "Query the structured NGO database using one predefined "
                "read-only operation. Use office_name only for office-filtered "
                "operations and language only for count_clients_by_language."
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
                },
                "required": ["operation", "office_name", "language"],
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
        if set(arguments) != {"operation", "office_name", "language"}:
            raise ValueError("sql_query received malformed arguments.")
        parameters = {
            key: value
            for key, value in arguments.items()
            if key != "operation" and value is not None
        }
        try:
            result = self.repository.execute(operation, parameters)
        except SQLToolError:
            return ToolExecutionResult(
                answer=self.SAFE_ERROR_MESSAGE,
                status=ToolExecutionStatus.ERROR,
                citations=(),
                rag_llm_called=False,
                source=self.NAME,
                provenance=ToolProvenance.STRUCTURED_SQL_DATA,
                category=operation.value,
            )
        return ToolExecutionResult(
            answer=json.dumps(result.to_model_output(), default=str),
            status=ToolExecutionStatus.ANSWERED,
            citations=(),
            rag_llm_called=False,
            source=self.NAME,
            provenance=ToolProvenance.STRUCTURED_SQL_DATA,
            category=operation.value,
        )
