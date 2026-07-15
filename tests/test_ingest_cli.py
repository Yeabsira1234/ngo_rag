from unittest.mock import Mock

import ingest
from src.config import Settings
from src.discovery import DiscoveredDocument, DiscoveryResult
from src.documents import Document, DocumentMetadata


def test_pdf_failure_returns_non_zero_exit_code_safely(
    monkeypatch,
    capsys,
) -> None:
    settings = Settings(openai_api_key="test-key")
    loader = Mock()
    loader.load.side_effect = FileNotFoundError("private/document/path.pdf")

    monkeypatch.setattr(ingest.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(ingest, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "PDFLoader", lambda: loader)
    monkeypatch.setattr(ingest, "OpenAIEmbeddingService", Mock())
    monkeypatch.setattr(ingest, "ChromaVectorStore", Mock())

    exit_code = ingest.run()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert ingest.INGESTION_ERROR_MESSAGE in output
    assert "private/document/path.pdf" not in output


def test_bad_pdf_does_not_block_valid_document(monkeypatch, capsys, tmp_path) -> None:
    settings = Settings(openai_api_key="test-key", documents_directory=tmp_path)
    good = DiscoveredDocument(tmp_path / "good.pdf", "good.pdf", "good-id")
    bad = DiscoveredDocument(tmp_path / "bad.pdf", "bad.pdf", "bad-id")
    discovery = Mock()
    discovery.discover.return_value = DiscoveryResult((bad, good), 0)
    loader = Mock()
    loader.load.side_effect = [ValueError("malformed"), [
        Document("valid text", DocumentMetadata(
            "good.pdf", 1, source_relative_path="good.pdf", document_id="good-id"
        ))
    ]]
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1, 0.2]]
    store = Mock()
    store.remove_stale_documents.return_value = 0
    monkeypatch.setattr(ingest.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(ingest, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "PDFDocumentDiscovery", lambda: discovery)
    monkeypatch.setattr(ingest, "PDFLoader", lambda: loader)
    monkeypatch.setattr(ingest, "OpenAIEmbeddingService", Mock(return_value=embeddings))
    monkeypatch.setattr(ingest, "ChromaVectorStore", Mock(return_value=store))

    assert ingest.run() == 0
    assert "Successfully processed documents: 1" in capsys.readouterr().out
    store.add_documents.assert_called_once()
    store.remove_stale_documents.assert_called_once_with({"bad-id", "good-id"})
