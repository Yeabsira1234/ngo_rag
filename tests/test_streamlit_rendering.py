from streamlit.testing.v1 import AppTest


def _render_insufficient_context() -> None:
    from src.chat_history import ChatMessage, MessageKind
    from streamlit_app import render_message

    render_message(
        ChatMessage(
            role="assistant",
            content="Not enough relevant information.",
            kind=MessageKind.INSUFFICIENT_CONTEXT,
        )
    )


def _render_safe_error() -> None:
    from src.chat_history import ChatMessage, MessageKind
    from streamlit_app import render_message

    render_message(
        ChatMessage(
            role="assistant",
            content="The request could not be completed safely.",
            kind=MessageKind.ERROR,
        )
    )


def _render_answer_with_citation() -> None:
    from src.chat_history import ChatMessage
    from src.rag_service import SourceReference
    from streamlit_app import render_message

    render_message(
        ChatMessage(
            role="assistant",
            content="A grounded answer.",
            citations=(
                SourceReference(
                    source="sample_document.pdf",
                    page_number=2,
                    chunk_index=4,
                    distance=0.4321,
                ),
            ),
        )
    )


def _render_upload_error() -> None:
    from streamlit_app import render_safe_upload_error
    render_safe_upload_error(RuntimeError("C:/private/secret.pdf api-key-value"))


def _render_agent_tool_message() -> None:
    from src.chat_history import ChatMessage
    from streamlit_app import render_message
    render_message(ChatMessage(
        role="assistant",
        content="There are five open cases.",
        agent_status="document_answer",
        tools_used=("sql_query", "document_search"),
    ))


def test_insufficient_context_renders_as_warning() -> None:
    app = AppTest.from_function(
        _render_insufficient_context,
        default_timeout=10,
    ).run()

    assert len(app.warning) == 1
    assert app.warning[0].value == "Not enough relevant information."


def test_safe_failure_renders_as_error_without_traceback() -> None:
    app = AppTest.from_function(
        _render_safe_error,
        default_timeout=10,
    ).run()

    assert len(app.error) == 1
    assert app.error[0].value == (
        "The request could not be completed safely."
    )
    assert len(app.exception) == 0


def test_answer_renders_compact_citation() -> None:
    app = AppTest.from_function(
        _render_answer_with_citation,
        default_timeout=10,
    ).run()

    assert app.markdown[0].value == "A grounded answer."
    assert [caption.value for caption in app.caption] == [
        "Sources",
        "sample_document.pdf · Page 2 · Chunk 4 · Distance 0.4321",
    ]


def test_upload_failure_renders_safely_without_exception_details() -> None:
    app = AppTest.from_function(_render_upload_error, default_timeout=10).run()
    assert len(app.error) == 1
    assert app.error[0].value == (
        "The documents could not be ingested. Check the application log."
    )
    assert "secret.pdf" not in app.error[0].value
    assert len(app.exception) == 0


def test_agent_status_and_safe_tool_label_render_compactly() -> None:
    app = AppTest.from_function(_render_agent_tool_message, default_timeout=10).run()
    assert [caption.value for caption in app.caption] == [
        "Tools used: Structured database, Document search",
        "Status: document answer",
    ]
    assert "SELECT" not in app.markdown[0].value
