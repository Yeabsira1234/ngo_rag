from collections.abc import Mapping
from typing import Any, Protocol

from src.agent.models import AgentCitation, ToolDefinition, ToolExecutionResult
from src.rag_service import RAGService


class AgentTool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    def execute(self, arguments: Mapping[str, Any]) -> ToolExecutionResult: ...


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
            status=response.status,
            citations=tuple(
                AgentCitation.from_source_reference(citation)
                for citation in response.citations
            ),
            rag_llm_called=response.llm_called,
        )
