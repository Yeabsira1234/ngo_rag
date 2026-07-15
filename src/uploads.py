import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Protocol, Sequence

import fitz


class UploadedFile(Protocol):
    name: str
    type: str | None
    def getvalue(self) -> bytes: ...


class UploadValidationError(ValueError):
    def __init__(self, messages: Sequence[str]) -> None:
        self.messages = tuple(messages)
        super().__init__("One or more uploaded files are invalid.")


@dataclass(frozen=True, slots=True)
class UploadSaveResult:
    uploaded_count: int
    saved_filenames: tuple[str, ...]
    unchanged_filenames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedUpload:
    filename: str
    content: bytes


class PDFUploadService:
    SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*\.pdf$", re.IGNORECASE)

    def __init__(self, upload_directory: Path, *, max_files: int,
                 max_file_size_mb: int) -> None:
        self.upload_directory = upload_directory
        self.max_files = max_files
        self.max_bytes = max_file_size_mb * 1024 * 1024

    def validate_and_save(self, uploads: Sequence[UploadedFile]) -> UploadSaveResult:
        validated = self._validate(uploads)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        unchanged: list[str] = []
        to_save: list[_ValidatedUpload] = []
        errors: list[str] = []
        for upload in validated:
            destination = self.upload_directory / upload.filename
            if destination.exists():
                if destination.read_bytes() == upload.content:
                    unchanged.append(upload.filename)
                else:
                    errors.append(f"{upload.filename}: a different file already exists.")
            else:
                to_save.append(upload)
        if errors:
            raise UploadValidationError(errors)
        temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=self.upload_directory))
        moved: list[Path] = []
        try:
            for upload in to_save:
                temporary_file = temporary / upload.filename
                temporary_file.write_bytes(upload.content)
            for upload in to_save:
                destination = self.upload_directory / upload.filename
                os.replace(temporary / upload.filename, destination)
                moved.append(destination)
        except Exception:
            for destination in moved:
                destination.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return UploadSaveResult(len(validated), tuple(item.filename for item in to_save), tuple(unchanged))

    def _validate(self, uploads: Sequence[UploadedFile]) -> list[_ValidatedUpload]:
        errors: list[str] = []
        if not uploads:
            raise UploadValidationError(("Select at least one PDF file.",))
        if len(uploads) > self.max_files:
            raise UploadValidationError((f"Select no more than {self.max_files} files.",))
        validated: list[_ValidatedUpload] = []
        seen: set[str] = set()
        for upload in uploads:
            name = upload.name
            if (PurePath(name).name != name or not self.SAFE_NAME.fullmatch(name)
                    or name.startswith(".")):
                errors.append(f"{PurePath(name).name or 'file'}: unsafe filename.")
                continue
            normalized = name.casefold()
            if normalized in seen:
                errors.append(f"{name}: duplicate filename in this batch.")
                continue
            seen.add(normalized)
            content = upload.getvalue()
            if Path(name).suffix.casefold() != ".pdf":
                errors.append(f"{name}: only PDF files are supported.")
            elif upload.type not in (None, "", "application/pdf", "application/x-pdf"):
                errors.append(f"{name}: invalid PDF content type.")
            elif not content:
                errors.append(f"{name}: file is empty.")
            elif len(content) > self.max_bytes:
                errors.append(f"{name}: file exceeds the size limit.")
            elif not self._readable_pdf(content):
                errors.append(f"{name}: file is not a readable PDF.")
            else:
                validated.append(_ValidatedUpload(name, content))
        if errors:
            raise UploadValidationError(errors)
        return validated

    @staticmethod
    def _readable_pdf(content: bytes) -> bool:
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                return len(pdf) > 0
        except Exception:
            return False
