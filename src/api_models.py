from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from src.rag_service import RAGResponse, RAGStatus, SourceReference


class HealthResponse(BaseModel):
    """Lightweight API liveness response."""

    status: Literal["ok"] = "ok"


class AskRequest(BaseModel):
    """Validated question submitted to the document assistant."""

    model_config = ConfigDict(extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class CitationResponse(BaseModel):
    """Serializable source reference returned by the API."""

    source: str
    page_number: int
    chunk_index: int
    distance: float
    source_relative_path: str
    document_id: str

    @classmethod
    def from_source_reference(
        cls,
        citation: SourceReference,
    ) -> "CitationResponse":
        return cls(
            source=citation.source,
            page_number=citation.page_number,
            chunk_index=citation.chunk_index,
            distance=citation.distance,
            source_relative_path=citation.source_relative_path,
            document_id=citation.document_id,
        )


class AskResponse(BaseModel):
    """Stable HTTP representation of a RAG response."""

    answer: str
    status: RAGStatus
    llm_called: bool
    citations: list[CitationResponse]

    @classmethod
    def from_rag_response(cls, response: RAGResponse) -> "AskResponse":
        return cls(
            answer=response.answer,
            status=response.status,
            llm_called=response.llm_called,
            citations=[
                CitationResponse.from_source_reference(citation)
                for citation in response.citations
            ],
        )
