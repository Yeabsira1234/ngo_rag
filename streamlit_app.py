import logging
from dataclasses import dataclass
from pathlib import PurePath
from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from src.application import build_agent_service, build_ingestion_service
from src.agent.service import AgentService
from src.chat_history import (
    ChatMessage,
    MessageKind,
    append_agent_response,
    append_safe_error,
    append_user_message,
    format_citation,
)
from src.config import Settings
from src.logging_config import configure_logging
from src.upload_workflow import UploadIngestionWorkflow, UploadInProgressError
from src.uploads import PDFUploadService, UploadValidationError


logger = logging.getLogger(__name__)
MESSAGES_KEY = "visible_chat_messages"
INGESTING_KEY = "document_ingestion_running"
AGENT_KEY = "session_agent_service"
STARTUP_ERROR_MESSAGE = (
    "The document assistant could not start. Check the application logs "
    "or contact the administrator."
)


@dataclass(frozen=True, slots=True)
class StreamlitApplication:
    settings: Settings
    upload_workflow: UploadIngestionWorkflow


@st.cache_resource(show_spinner=False)
def initialize_application() -> StreamlitApplication:
    """Initialize and cache stateless/shared upload infrastructure."""
    settings: Settings | None = None
    try:
        settings = Settings.from_env()
        configure_logging(
            settings.log_level,
            sensitive_values=(settings.openai_api_key,),
        )
        logger.info("event=streamlit_startup_started")
        upload_workflow = UploadIngestionWorkflow(
            PDFUploadService(
                settings.upload_directory,
                max_files=settings.max_upload_files,
                max_file_size_mb=settings.max_upload_file_size_mb,
            ),
            build_ingestion_service(settings),
        )
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
    return StreamlitApplication(
        settings=settings,
        upload_workflow=upload_workflow,
    )


def get_session_agent(
    settings: Settings,
    state: MutableMapping[str, Any] | None = None,
) -> AgentService:
    """Return one mutable agent and memory store per browser session."""
    session = st.session_state if state is None else state
    if AGENT_KEY not in session:
        session[AGENT_KEY] = build_agent_service(settings)
    return session[AGENT_KEY]


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
        if message.tools_used:
            labels = {
                "document_search": "Document search",
                "organization_info": "Organization information",
                "sql_query": "Structured database",
                "weather_information": "Live weather",
            }
            rendered = [labels.get(tool, "Application tool") for tool in message.tools_used]
            st.caption(f"Tools used: {', '.join(rendered)}")
        if message.agent_status:
            st.caption(f"Status: {message.agent_status.replace('_', ' ')}")


def render_safe_upload_error(error: Exception) -> None:
    """Log upload details while rendering only a safe browser message."""
    logger.exception(
        "event=browser_ingestion_failed error_type=%s",
        type(error).__name__,
    )
    st.error("The documents could not be ingested. Check the application log.")


def clear_session_history(
    agent: AgentService,
    state: MutableMapping[str, Any] | None = None,
) -> None:
    session = st.session_state if state is None else state
    session[MESSAGES_KEY] = []
    agent.clear_memory()


def submit_agent_question(
    messages: list[ChatMessage], agent: AgentService, question: str
) -> None:
    """Store one complete visible exchange while AgentService owns context."""
    append_user_message(messages, question)
    try:
        response = agent.answer(question)
        append_agent_response(messages, response)
    except Exception as error:
        logger.error(
            "event=streamlit_request_failed error_type=%s question_length=%d",
            type(error).__name__,
            len(question),
        )
        append_safe_error(messages)


def render_sidebar(settings: Settings, agent: AgentService) -> None:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Document")
        st.code(str(settings.documents_directory), language=None)
        st.caption("Models")
        st.write(f"Answer: `{settings.llm_model}`")
        st.write(f"Embeddings: `{settings.embedding_model}`")
        st.caption("Retrieval")
        st.write(f"Candidates: `{settings.retrieval_result_count}`")
        st.write(f"Maximum L2 distance: `{settings.retrieval_max_distance}`")

        if st.button("Clear chat history", use_container_width=True):
            clear_session_history(agent)
            st.rerun()


def render_document_management(application: StreamlitApplication) -> None:
    st.subheader("Document management")
    uploads = st.file_uploader(
        "Select PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            f"Up to {application.settings.max_upload_files} files, "
            f"{application.settings.max_upload_file_size_mb} MB each."
        ),
    )
    if uploads:
        st.caption(f"Selected files: {len(uploads)}")
        for upload in uploads:
            st.write(
                f"- {PurePath(upload.name).name or 'unnamed file'} "
                f"({len(upload.getvalue()):,} bytes)"
            )
    if INGESTING_KEY not in st.session_state:
        st.session_state[INGESTING_KEY] = False
    if not st.button(
        "Upload and ingest documents",
        disabled=not uploads or st.session_state[INGESTING_KEY],
        use_container_width=True,
    ):
        return
    st.session_state[INGESTING_KEY] = True
    try:
        with st.spinner("Validating and ingesting documents..."):
            try:
                result = application.upload_workflow.run(uploads)
            except UploadValidationError as error:
                st.error("The upload batch was rejected. No files were saved.")
                with st.expander("Validation errors"):
                    for message in error.messages:
                        st.write(f"- {message}")
                return
            except UploadInProgressError:
                st.warning("Another ingestion operation is already running.")
                return
            except Exception as error:
                render_safe_upload_error(error)
                return
    finally:
        st.session_state[INGESTING_KEY] = False
    ingestion = result.ingestion
    if ingestion is None:
        st.success(
            f"All {len(result.upload.unchanged_filenames)} files were unchanged; "
            "no ingestion was needed."
        )
        return
    st.success(
        f"Accepted {result.upload.uploaded_count} files "
        f"({len(result.upload.unchanged_filenames)} unchanged) and processed "
        f"{ingestion.processed_document_count} documents "
        f"({ingestion.failed_document_count} failed) into "
        f"{ingestion.chunk_count} chunks ({ingestion.page_count} pages)."
    )
    if ingestion.failures:
        with st.expander("Files that could not be processed"):
            for failure in ingestion.failures:
                st.write(f"- {failure.filename}: {failure.reason}")


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

    try:
        agent = get_session_agent(application.settings)
    except Exception as error:
        logger.exception(
            "event=streamlit_agent_startup_failed error_type=%s",
            type(error).__name__,
        )
        st.error(STARTUP_ERROR_MESSAGE)
        st.stop()

    render_sidebar(application.settings, agent)
    render_document_management(application)
    st.divider()
    st.subheader("Chat")
    st.caption(
        "In-session memory with document search, structured database queries, "
        "organization information, and direct answers."
    )
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []

    messages: list[ChatMessage] = st.session_state[MESSAGES_KEY]
    for message in messages:
        render_message(message)

    question = st.chat_input("Ask a question about the document")
    if not question:
        return

    temporary_user_message = ChatMessage(role="user", content=question)
    render_message(temporary_user_message)

    with st.spinner("Searching the document..."):
        submit_agent_question(messages, agent, question)

    render_message(messages[-1])


if __name__ == "__main__":
    main()
