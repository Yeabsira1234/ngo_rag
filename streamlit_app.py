import logging
from dataclasses import dataclass

import streamlit as st

from src.application import build_rag_service
from src.chat_history import (
    ChatMessage,
    MessageKind,
    append_rag_response,
    append_safe_error,
    append_user_message,
    format_citation,
)
from src.config import Settings
from src.logging_config import configure_logging
from src.rag_service import RAGService


logger = logging.getLogger(__name__)
MESSAGES_KEY = "visible_chat_messages"
STARTUP_ERROR_MESSAGE = (
    "The document assistant could not start. Check the application logs "
    "or contact the administrator."
)


@dataclass(frozen=True, slots=True)
class StreamlitApplication:
    settings: Settings
    rag_service: RAGService


@st.cache_resource(show_spinner=False)
def initialize_application() -> StreamlitApplication:
    """Initialize and cache settings, logging, and the RAG service."""
    settings: Settings | None = None
    try:
        settings = Settings.from_env()
        configure_logging(
            settings.log_level,
            sensitive_values=(settings.openai_api_key,),
        )
        logger.info("event=streamlit_startup_started")
        rag_service = build_rag_service(settings)
    except Exception as error:
        sensitive_values = (
            (settings.openai_api_key,) if settings is not None else ()
        )
        configure_logging("ERROR", sensitive_values=sensitive_values)
        logger.exception(
            "event=streamlit_startup_failed error_type=%s",
            type(error).__name__,
        )
        raise

    logger.info("event=streamlit_startup_completed")
    return StreamlitApplication(settings=settings, rag_service=rag_service)


def render_message(message: ChatMessage) -> None:
    with st.chat_message(message.role):
        if message.kind is MessageKind.INSUFFICIENT_CONTEXT:
            st.warning(message.content)
        elif message.kind is MessageKind.ERROR:
            st.error(message.content)
        else:
            st.markdown(message.content)

        if message.citations:
            st.caption("Sources")
            for citation in message.citations:
                st.caption(format_citation(citation))


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Document")
        st.code(settings.document_path.name, language=None)
        st.caption("Models")
        st.write(f"Answer: `{settings.llm_model}`")
        st.write(f"Embeddings: `{settings.embedding_model}`")
        st.caption("Retrieval")
        st.write(f"Candidates: `{settings.retrieval_result_count}`")
        st.write(f"Maximum L2 distance: `{settings.retrieval_max_distance}`")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state[MESSAGES_KEY] = []
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="💬",
        layout="centered",
    )
    st.title("Document Assistant")
    st.caption(
        "Ask grounded questions about the indexed document. "
        "Answers include retrieval citations when relevant context is found."
    )

    try:
        application = initialize_application()
    except Exception:
        st.error(STARTUP_ERROR_MESSAGE)
        st.stop()

    render_sidebar(application.settings)
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []

    messages: list[ChatMessage] = st.session_state[MESSAGES_KEY]
    for message in messages:
        render_message(message)

    question = st.chat_input("Ask a question about the document")
    if not question:
        return

    append_user_message(messages, question)
    render_message(messages[-1])

    with st.spinner("Searching the document..."):
        try:
            response = application.rag_service.answer(question)
            append_rag_response(messages, response)
        except Exception as error:
            logger.error(
                "event=streamlit_request_failed error_type=%s "
                "question_length=%d",
                type(error).__name__,
                len(question),
            )
            append_safe_error(messages)

    render_message(messages[-1])


if __name__ == "__main__":
    main()
