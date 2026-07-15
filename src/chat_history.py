from dataclasses import dataclass
from enum import Enum
from typing import Literal

from src.rag_service import RAGResponse, RAGStatus, SourceReference


SAFE_REQUEST_ERROR_MESSAGE = (
    "The request could not be completed because a service is unavailable. "
    "Please try again later."
)


class MessageKind(str, Enum):
    STANDARD = "standard"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """User-visible browser-session message and citation data."""

    role: Literal["user", "assistant"]
    content: str
    kind: MessageKind = MessageKind.STANDARD
    citations: tuple[SourceReference, ...] = ()


def append_user_message(
    messages: list[ChatMessage],
    question: str,
) -> None:
    messages.append(ChatMessage(role="user", content=question))


def append_rag_response(
    messages: list[ChatMessage],
    response: RAGResponse,
) -> None:
    kind = (
        MessageKind.INSUFFICIENT_CONTEXT
        if response.status is RAGStatus.INSUFFICIENT_CONTEXT
        else MessageKind.STANDARD
    )
    messages.append(
        ChatMessage(
            role="assistant",
            content=response.answer,
            kind=kind,
            citations=response.citations,
        )
    )


def append_safe_error(messages: list[ChatMessage]) -> None:
    messages.append(
        ChatMessage(
            role="assistant",
            content=SAFE_REQUEST_ERROR_MESSAGE,
            kind=MessageKind.ERROR,
        )
    )


def format_citation(citation: SourceReference) -> str:
    return (
        f"{citation.source_relative_path or citation.source} · "
        f"Page {citation.page_number} · "
        f"Chunk {citation.chunk_index} · Distance {citation.distance:.4f}"
    )
