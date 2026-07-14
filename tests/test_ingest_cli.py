from unittest.mock import Mock

import ingest
from src.config import Settings


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
