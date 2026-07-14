import logging

from src.chunking.text_chunker import TextChunker
from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.loaders.pdf_loader import PDFLoader
from src.logging_config import configure_logging
from src.vectorstore.chroma_store import ChromaVectorStore

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
        logger.info(
            "event=ingestion_started document_name=%s",
            settings.document_path.name,
        )

        loader = PDFLoader()
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        embedding_service = OpenAIEmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        vector_store = ChromaVectorStore(
            collection_name=settings.chroma_collection_name,
            persist_directory=str(settings.chroma_persist_directory),
        )

        print("Loading PDF...")
        page_documents = loader.load(settings.document_path)
        logger.info(
            "event=document_loaded page_count=%d",
            len(page_documents),
        )

        print("Creating chunks...")
        chunks = chunker.split_documents(page_documents)
        logger.info(
            "event=document_chunked page_count=%d chunk_count=%d",
            len(page_documents),
            len(chunks),
        )

        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_service.embed_documents(
            [chunk.page_content for chunk in chunks]
        )

        vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
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
        "event=ingestion_completed page_count=%d chunk_count=%d",
        len(page_documents),
        len(chunks),
    )
    print("Ingestion complete.")
    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
