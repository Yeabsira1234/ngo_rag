import logging

from src.application import build_rag_service
from src.chat_history import SAFE_REQUEST_ERROR_MESSAGE
from src.config import Settings
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

STARTUP_ERROR_MESSAGE = (
    "The document assistant could not start. Check the application log "
    "for details."
)
REQUEST_ERROR_MESSAGE = SAFE_REQUEST_ERROR_MESSAGE


def run() -> int:
    settings: Settings | None = None
    try:
        settings = Settings.from_env()
        configure_logging(
            settings.log_level,
            sensitive_values=(settings.openai_api_key,),
        )
        logger.info("event=chat_startup_started")
        rag_service = build_rag_service(settings)
    except Exception as error:
        sensitive_values = (
            (settings.openai_api_key,) if settings is not None else ()
        )
        configure_logging("ERROR", sensitive_values=sensitive_values)
        logger.exception(
            "event=chat_startup_failed error_type=%s",
            type(error).__name__,
        )
        print(STARTUP_ERROR_MESSAGE)
        return 1

    logger.info("event=chat_startup_completed")

    print("Document assistant is ready.")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            logger.info("event=chat_shutdown input_interrupted=true")
            return 0

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            logger.info("event=chat_shutdown input_interrupted=false")
            return 0

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            response = rag_service.answer(question)
        except Exception as error:
            logger.error(
                "event=chat_request_failed error_type=%s question_length=%d",
                type(error).__name__,
                len(question),
            )
            print(f"\n{REQUEST_ERROR_MESSAGE}\n")
            continue

        print("\nAnswer:")
        print(response.answer)

        if response.citations:
            print("\nSources retrieved:")

        for index, citation in enumerate(response.citations, start=1):
            print(
                f"{index}. {citation.source_relative_path or citation.source}, "
                f"page {citation.page_number}, "
                f"chunk {citation.chunk_index}, "
                f"distance {citation.distance:.4f}"
            )

        print()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
