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
