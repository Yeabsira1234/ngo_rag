from src.chat_history import (
    SAFE_REQUEST_ERROR_MESSAGE,
    MessageKind,
    append_rag_response,
    append_safe_error,
    append_user_message,
    format_citation,
)
from src.rag_service import RAGResponse, RAGStatus, SourceReference


def _citation() -> SourceReference:
    return SourceReference(
        source="sample_document.pdf",
        page_number=2,
        chunk_index=4,
        distance=0.4321,
    )


def test_format_citation_preserves_rendering_data() -> None:
    assert format_citation(_citation()) == (
        "sample_document.pdf · Page 2 · Chunk 4 · Distance 0.4321"
    )


def test_conversation_state_appends_visible_exchange() -> None:
    messages = []
    append_user_message(messages, "What is the policy?")
    append_rag_response(
        messages,
        RAGResponse(
            answer="The policy requires approval.",
            citations=(_citation(),),
            llm_called=True,
            status=RAGStatus.ANSWERED,
        ),
    )

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "What is the policy?"
    assert messages[1].citations == (_citation(),)


def test_insufficient_context_has_distinct_display_kind() -> None:
    messages = []
    append_rag_response(
        messages,
        RAGResponse(
            answer="Not enough relevant information.",
            citations=(),
            llm_called=False,
            status=RAGStatus.INSUFFICIENT_CONTEXT,
        ),
    )

    assert messages[0].kind is MessageKind.INSUFFICIENT_CONTEXT
    assert messages[0].citations == ()


def test_safe_failure_message_excludes_dependency_details() -> None:
    messages = []
    append_safe_error(messages)

    assert messages[0].kind is MessageKind.ERROR
    assert messages[0].content == SAFE_REQUEST_ERROR_MESSAGE
    assert "exception" not in messages[0].content.lower()
