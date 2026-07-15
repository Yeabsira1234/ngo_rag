import logging

from src.chunking.text_chunker import TextChunker
from src.config import Settings
from src.discovery import PDFDocumentDiscovery
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
        logger.info("event=ingestion_started")

        loader = PDFLoader()
        discovery = PDFDocumentDiscovery()
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

        discovered = discovery.discover(
            settings.documents_directory, settings.document_glob
        )
        print(f"Discovered {len(discovered.documents)} PDF documents.")
        page_documents = []
        successful_documents = 0
        failed_documents = 0
        for document in discovered.documents:
            try:
                pages = loader.load(document)
            except Exception as error:
                failed_documents += 1
                logger.error(
                    "event=document_load_failed document_id=%s error_type=%s",
                    document.document_id,
                    type(error).__name__,
                )
                continue
            page_documents.extend(pages)
            successful_documents += 1
        if not successful_documents:
            raise RuntimeError("No discovered PDF could be processed.")

        print("Creating chunks...")
        chunks = chunker.split_documents(page_documents)
        if not chunks:
            raise RuntimeError("No text chunks were produced from the collection.")
        logger.info(
            "event=document_collection_chunked document_count=%d "
            "failed_document_count=%d page_count=%d chunk_count=%d",
            successful_documents,
            failed_documents,
            len(page_documents),
            len(chunks),
        )

        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_service.embed_documents(
            [chunk.page_content for chunk in chunks]
        )

        stale_count = vector_store.remove_stale_documents(
            {document.document_id for document in discovered.documents}
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
        "event=ingestion_completed document_count=%d failed_document_count=%d "
        "page_count=%d chunk_count=%d stale_chunk_count=%d",
        successful_documents,
        failed_documents,
        len(page_documents),
        len(chunks),
        stale_count,
    )
    print("Ingestion complete.")
    print(f"Successfully processed documents: {successful_documents}")
    print(f"Skipped or failed documents: {discovered.skipped_count + failed_documents}")
    print(f"Total pages: {len(page_documents)}")
    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
