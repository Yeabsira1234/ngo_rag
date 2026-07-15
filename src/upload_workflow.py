from dataclasses import dataclass
from threading import Lock
from typing import Sequence

from src.ingestion import CollectionIngestionService, IngestionSummary
from src.uploads import PDFUploadService, UploadedFile, UploadSaveResult


class UploadInProgressError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class UploadIngestionSummary:
    upload: UploadSaveResult
    ingestion: IngestionSummary | None


class UploadIngestionWorkflow:
    def __init__(self, upload_service: PDFUploadService,
                 ingestion_service: CollectionIngestionService) -> None:
        self.upload_service = upload_service
        self.ingestion_service = ingestion_service
        self._lock = Lock()

    def run(self, uploads: Sequence[UploadedFile]) -> UploadIngestionSummary:
        if not self._lock.acquire(blocking=False):
            raise UploadInProgressError("An ingestion operation is already running.")
        try:
            saved = self.upload_service.validate_and_save(uploads)
            if not saved.saved_filenames:
                return UploadIngestionSummary(saved, None)
            ingestion = self.ingestion_service.ingest_directory(
                self.upload_service.upload_directory,
                "*.pdf",
                remove_stale=False,
                identity_namespace="browser_uploads",
            )
            return UploadIngestionSummary(saved, ingestion)
        finally:
            self._lock.release()
