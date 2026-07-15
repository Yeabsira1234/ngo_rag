from unittest.mock import Mock

from src.ingestion import IngestionSummary
import pytest

from src.upload_workflow import UploadIngestionWorkflow, UploadInProgressError
from src.uploads import UploadSaveResult


def test_valid_batch_calls_ingestion_exactly_once() -> None:
    upload_service = Mock(upload_directory="uploads")
    upload_service.validate_and_save.return_value = UploadSaveResult(2, ("a.pdf", "b.pdf"), ())
    ingestion_service = Mock()
    summary = IngestionSummary(2, 1, 1, 2, 3, 0, 0)
    ingestion_service.ingest_directory.return_value = summary
    result = UploadIngestionWorkflow(upload_service, ingestion_service).run([Mock(), Mock()])
    ingestion_service.ingest_directory.assert_called_once_with(
        "uploads", "*.pdf", remove_stale=False,
        identity_namespace="browser_uploads"
    )
    assert result.ingestion == summary
    assert result.ingestion.failed_document_count == 1


def test_unchanged_batch_skips_reingestion() -> None:
    upload_service = Mock(upload_directory="uploads")
    upload_service.validate_and_save.return_value = UploadSaveResult(1, (), ("a.pdf",))
    ingestion_service = Mock()
    result = UploadIngestionWorkflow(upload_service, ingestion_service).run([Mock()])
    assert result.ingestion is None
    ingestion_service.ingest_directory.assert_not_called()


def test_concurrent_ingestion_is_rejected() -> None:
    workflow = UploadIngestionWorkflow(Mock(), Mock())
    workflow._lock.acquire()
    try:
        with pytest.raises(UploadInProgressError):
            workflow.run([Mock()])
    finally:
        workflow._lock.release()
