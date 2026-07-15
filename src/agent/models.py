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
class PlanStep:
    """One validated, ordered application-tool step."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    purpose: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Internal bounded plan derived from model-selected tools."""

    tools_needed: bool
    steps: tuple[PlanStep, ...]
    combine_results: bool


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Typed tool evidence retained until final synthesis."""

    step: PlanStep
    result: "ToolExecutionResult"


@dataclass(frozen=True, slots=True)
class AgentModelResponse:
    """Normalized model output consumed by the agent loop."""

    output_text: str
    tool_calls: tuple[ToolCall, ...]
    continuation_items: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AgentModelInput:
    """Ordered conversation items sent through the model boundary."""

    items: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AgentCitation:
    """Document citation preserved in the final agent response."""

    source: str
    page_number: int
    chunk_index: int
    distance: float
    source_relative_path: str = ""
    document_id: str = ""

    @classmethod
    def from_source_reference(cls, citation: SourceReference) -> "AgentCitation":
        return cls(
            source=citation.source,
            page_number=citation.page_number,
            chunk_index=citation.chunk_index,
            distance=citation.distance,
            source_relative_path=citation.source_relative_path,
            document_id=citation.document_id,
        )


class AgentStatus(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    DOCUMENT_ANSWER = "document_answer"
    ORGANIZATION_ANSWER = "organization_answer"
    SQL_ANSWER = "sql_answer"
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
    tool_sources: tuple[str, ...] = ()


class ToolExecutionStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    INVALID_QUESTION = "invalid_question"
    NOT_FOUND = "not_found"
    ERROR = "error"

    @classmethod
    def from_rag_status(cls, status: RAGStatus) -> "ToolExecutionStatus":
        return cls(status.value)


class ToolProvenance(str, Enum):
    DOCUMENT = "document"
    STRUCTURED_ORGANIZATION_DATA = "structured_organization_data"
    STRUCTURED_SQL_DATA = "structured_sql_data"


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Structured output produced by an application tool."""

    answer: str
    status: ToolExecutionStatus
    citations: tuple[AgentCitation, ...]
    rag_llm_called: bool
    source: str
    provenance: ToolProvenance
    category: str | None = None
    failure_category: str | None = None

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
                    "source_relative_path": citation.source_relative_path,
                    "document_id": citation.document_id,
                }
                for citation in self.citations
            ],
            "llm_called": self.rag_llm_called,
            "source": self.source,
            "provenance": self.provenance.value,
            "category": self.category,
            "failure_category": self.failure_category,
        }
