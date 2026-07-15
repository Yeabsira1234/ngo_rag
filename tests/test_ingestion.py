from unittest.mock import Mock

import pytest

from src.discovery import DiscoveredDocument, DiscoveryResult
from src.documents import Document, DocumentMetadata
from src.ingestion import CollectionIngestionError, CollectionIngestionService


def make_service(loader: Mock):
    discovery = Mock()
    documents = (
        DiscoveredDocument(Mock(), "bad.pdf", "bad"),
        DiscoveredDocument(Mock(), "good.pdf", "good"),
    )
    discovery.discover.return_value = DiscoveryResult(documents, 0)
    chunk = Document("chunk", DocumentMetadata("good.pdf", 1, 0, "good.pdf", "good"))
    chunker = Mock()
    chunker.split_documents.return_value = [chunk]
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1]]
    store = Mock()
    store.remove_stale_documents.return_value = 2
    return CollectionIngestionService(
        discovery=discovery, loader=loader, chunker=chunker,
        embedding_service=embeddings, vector_store=store
    ), store


def test_partial_processing_failure_returns_accurate_safe_summary() -> None:
    loader = Mock()
    loader.load.side_effect = [ValueError("private path"), [
        Document("page", DocumentMetadata("good.pdf", 1, document_id="good"))
    ]]
    service, store = make_service(loader)
    summary = service.ingest_directory("documents")
    assert summary.processed_document_count == 1
    assert summary.failed_document_count == 1
    assert summary.failures[0].filename == "bad.pdf"
    assert "private path" not in summary.failures[0].reason
    assert summary.stale_chunk_count == 0
    store.remove_stale_documents.assert_not_called()
    store.add_documents.assert_called_once()


def test_no_successful_document_fails_without_writing_vectors() -> None:
    loader = Mock()
    loader.load.side_effect = ValueError("bad")
    service, store = make_service(loader)
    with pytest.raises(CollectionIngestionError):
        service.ingest_directory("documents")
    store.add_documents.assert_not_called()
