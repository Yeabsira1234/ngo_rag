from unittest.mock import Mock

import ingest
from src.config import Settings
from src.ingestion import IngestionSummary


def test_pdf_failure_returns_non_zero_exit_code_safely(
    monkeypatch,
    capsys,
) -> None:
    settings = Settings(openai_api_key="test-key")
    service = Mock()
    service.ingest_directory.side_effect = FileNotFoundError("private/document/path.pdf")

    monkeypatch.setattr(ingest.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(ingest, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "build_ingestion_service", Mock(return_value=service))

    exit_code = ingest.run()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert ingest.INGESTION_ERROR_MESSAGE in output
    assert "private/document/path.pdf" not in output


def test_bad_pdf_does_not_block_valid_document(monkeypatch, capsys, tmp_path) -> None:
    settings = Settings(openai_api_key="test-key", documents_directory=tmp_path)
    service = Mock()
    service.ingest_directory.return_value = IngestionSummary(2, 1, 1, 1, 1, 0, 0)
    monkeypatch.setattr(ingest.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(ingest, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "build_ingestion_service", Mock(return_value=service))

    assert ingest.run() == 0
    assert "Successfully processed documents: 1" in capsys.readouterr().out
    service.ingest_directory.assert_called_once_with(tmp_path, "*.pdf")
