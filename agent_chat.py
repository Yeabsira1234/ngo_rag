import logging

from src.agent.service import AgentService
from src.application import build_agent_service
from src.config import Settings
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)
STARTUP_ERROR_MESSAGE = (
    "The agent could not start. Check the application log for details."
)
REQUEST_ERROR_MESSAGE = (
    "The agent could not complete the request. Please try again later."
)


def run() -> int:
    settings: Settings | None = None
    try:
        settings = Settings.from_env()
        configure_logging(
            settings.log_level,
            sensitive_values=(settings.openai_api_key,),
        )
        logger.info("event=agent_cli_startup_started")
        agent = build_agent_service(settings)
    except Exception as error:
        sensitive_values = (
            (settings.openai_api_key,) if settings is not None else ()
        )
        configure_logging("ERROR", sensitive_values=sensitive_values)
        logger.exception(
            "event=agent_cli_startup_failed error_type=%s",
            type(error).__name__,
        )
        print(STARTUP_ERROR_MESSAGE)
        return 1

    logger.info("event=agent_cli_startup_completed")
    print("Document agent is ready.")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("Ask the agent: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0

        if not question:
            print(f"{AgentService.INVALID_QUESTION_MESSAGE}\n")
            continue

        try:
            response = agent.answer(question)
        except Exception as error:
            logger.error(
                "event=agent_cli_request_failed error_type=%s "
                "question_length=%d",
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
                f"{index}. {citation.source}, page {citation.page_number}, "
                f"chunk {citation.chunk_index}, "
                f"distance {citation.distance:.4f}"
            )
        print()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
