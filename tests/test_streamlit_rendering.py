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
