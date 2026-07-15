import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.chunking.text_chunker import TextChunker
from src.discovery import PDFDocumentDiscovery
from src.documents import Document
from src.loaders.pdf_loader import PDFLoader
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class CollectionIngestionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IngestionFailure:
    filename: str
    reason: str = "The PDF could not be processed."


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    discovered_document_count: int
    processed_document_count: int
    failed_document_count: int
    page_count: int
    chunk_count: int
    skipped_document_count: int
    stale_chunk_count: int
    failures: tuple[IngestionFailure, ...] = ()


class CollectionIngestionService:
    def __init__(self, *, discovery: PDFDocumentDiscovery, loader: PDFLoader,
                 chunker: TextChunker, embedding_service: EmbeddingService,
                 vector_store: ChromaVectorStore) -> None:
        self.discovery = discovery
        self.loader = loader
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_directory(self, directory: str | Path, pattern: str = "*.pdf",
                         *, remove_stale: bool = False,
                         identity_namespace: str = "") -> IngestionSummary:
        discovered = self.discovery.discover(
            directory, pattern, identity_namespace=identity_namespace
        )
        pages: list[Document] = []
        failures: list[IngestionFailure] = []
        processed = 0
        for document in discovered.documents:
            try:
                document_pages = self.loader.load(document)
            except Exception as error:
                logger.error("event=document_load_failed document_id=%s error_type=%s",
                             document.document_id, type(error).__name__)
                failures.append(IngestionFailure(document.relative_path))
                continue
            pages.extend(document_pages)
            processed += 1
        if not processed:
            raise CollectionIngestionError("No discovered PDF could be processed.")
        chunks = self.chunker.split_documents(pages)
        if not chunks:
            raise CollectionIngestionError("No text chunks were produced from the collection.")
        embeddings = self.embedding_service.embed_documents(
            [chunk.page_content for chunk in chunks]
        )
        stale_count = 0
        if remove_stale:
            stale_count = self.vector_store.remove_stale_documents(
                {document.document_id for document in discovered.documents}
            )
        self.vector_store.add_documents(chunks, embeddings)
        summary = IngestionSummary(
            len(discovered.documents), processed, len(failures), len(pages),
            len(chunks), discovered.skipped_count, stale_count, tuple(failures)
        )
        logger.info("event=collection_ingestion_completed discovered=%d processed=%d failed=%d pages=%d chunks=%d stale_chunks=%d",
                    summary.discovered_document_count, summary.processed_document_count,
                    summary.failed_document_count, summary.page_count,
                    summary.chunk_count, summary.stale_chunk_count)
        return summary
