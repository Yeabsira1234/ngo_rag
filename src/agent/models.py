from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.rag_service import RAGStatus, SourceReference


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Model-facing definition of a callable application tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Provider-independent request from a model to execute a tool."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class AgentModelResponse:
    """Normalized model output consumed by the agent loop."""

    output_text: str
    tool_calls: tuple[ToolCall, ...]
    continuation_items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class AgentCitation:
    """Document citation preserved in the final agent response."""

    source: str
    page_number: int
    chunk_index: int
    distance: float

    @classmethod
    def from_source_reference(cls, citation: SourceReference) -> "AgentCitation":
        return cls(
            source=citation.source,
            page_number=citation.page_number,
            chunk_index=citation.chunk_index,
            distance=citation.distance,
        )


class AgentStatus(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    DOCUMENT_ANSWER = "document_answer"
    INVALID_QUESTION = "invalid_question"
    TOOL_ERROR = "tool_error"
    MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Final answer and provenance returned by the agent service."""

    answer: str
    status: AgentStatus
    citations: tuple[AgentCitation, ...]
    document_tool_used: bool


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Structured output produced by an application tool."""

    answer: str
    status: RAGStatus
    citations: tuple[AgentCitation, ...]
    rag_llm_called: bool

    def to_model_output(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status.value,
            "citations": [
                {
                    "source": citation.source,
                    "page_number": citation.page_number,
                    "chunk_index": citation.chunk_index,
                    "distance": citation.distance,
                }
                for citation in self.citations
            ],
            "llm_called": self.rag_llm_called,
        }
