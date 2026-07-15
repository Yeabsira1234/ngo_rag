import logging

from src.application import build_ingestion_service
from src.config import Settings
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

INGESTION_ERROR_MESSAGE = (
    "Ingestion could not be completed. Check the application log for details."
)


def run() -> int:
    settings: Settings | None = None
    try:
        settings = Settings.from_env()
        configure_logging(
            settings.log_level,
            sensitive_values=(settings.openai_api_key,),
        )
        logger.info("event=ingestion_started")

        summary = build_ingestion_service(settings).ingest_directory(
            settings.documents_directory, settings.document_glob
        )
    except Exception as error:
        sensitive_values = (
            (settings.openai_api_key,) if settings is not None else ()
        )
        configure_logging("ERROR", sensitive_values=sensitive_values)
        logger.exception(
            "event=ingestion_failed error_type=%s",
            type(error).__name__,
        )
        print(INGESTION_ERROR_MESSAGE)
        return 1

    logger.info(
        "event=ingestion_completed document_count=%d failed_document_count=%d "
        "page_count=%d chunk_count=%d stale_chunk_count=%d",
        summary.processed_document_count,
        summary.failed_document_count,
        summary.page_count,
        summary.chunk_count,
        summary.stale_chunk_count,
    )
    print("Ingestion complete.")
    print(f"Discovered documents: {summary.discovered_document_count}")
    print(f"Successfully processed documents: {summary.processed_document_count}")
    print(f"Skipped or failed documents: {summary.skipped_document_count + summary.failed_document_count}")
    print(f"Total pages: {summary.page_count}")
    print(f"Stored {summary.chunk_count} chunks in ChromaDB.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
