from dataclasses import dataclass
from pathlib import Path

import fitz
import pytest

from src.uploads import PDFUploadService, UploadValidationError


@dataclass
class FakeUpload:
    name: str
    content: bytes
    type: str | None = "application/pdf"
    def getvalue(self) -> bytes:
        return self.content


def pdf_bytes(text: str = "fictional") -> bytes:
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), text)
    content = pdf.tobytes()
    pdf.close()
    return content


def service(tmp_path: Path, **kwargs) -> PDFUploadService:
    return PDFUploadService(
        tmp_path / "uploads",
        max_files=kwargs.get("max_files", 10),
        max_file_size_mb=kwargs.get("max_file_size_mb", 10),
    )


def test_multiple_valid_pdfs_are_saved_only_under_upload_directory(tmp_path: Path) -> None:
    result = service(tmp_path).validate_and_save([
        FakeUpload("one.pdf", pdf_bytes("one")),
        FakeUpload("two.pdf", pdf_bytes("two")),
    ])
    assert result.saved_filenames == ("one.pdf", "two.pdf")
    assert sorted(path.name for path in (tmp_path / "uploads").iterdir()) == ["one.pdf", "two.pdf"]


@pytest.mark.parametrize("upload", [
    FakeUpload("notes.txt", b"text", "text/plain"),
    FakeUpload("wrong.pdf", pdf_bytes(), "text/plain"),
    FakeUpload("empty.pdf", b""),
    FakeUpload("corrupt.pdf", b"%PDF-not-readable"),
    FakeUpload("../escape.pdf", pdf_bytes()),
    FakeUpload(".hidden.pdf", pdf_bytes()),
])
def test_invalid_upload_is_rejected_without_saving(tmp_path: Path, upload: FakeUpload) -> None:
    with pytest.raises(UploadValidationError):
        service(tmp_path).validate_and_save([upload])
    assert not (tmp_path / "uploads").exists()


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    upload_service = service(tmp_path)
    upload_service.max_bytes = 10
    with pytest.raises(UploadValidationError, match="invalid"):
        upload_service.validate_and_save([FakeUpload("large.pdf", pdf_bytes())])


def test_too_many_and_duplicate_names_are_rejected(tmp_path: Path) -> None:
    content = pdf_bytes()
    with pytest.raises(UploadValidationError):
        service(tmp_path, max_files=1).validate_and_save([
            FakeUpload("one.pdf", content), FakeUpload("two.pdf", content)
        ])
    with pytest.raises(UploadValidationError):
        service(tmp_path).validate_and_save([
            FakeUpload("same.pdf", content), FakeUpload("SAME.PDF", content)
        ])


def test_one_invalid_file_prevents_entire_batch_from_being_saved(tmp_path: Path) -> None:
    with pytest.raises(UploadValidationError):
        service(tmp_path).validate_and_save([
            FakeUpload("valid.pdf", pdf_bytes()), FakeUpload("bad.pdf", b"bad")
        ])
    assert not (tmp_path / "uploads").exists()


def test_existing_file_is_not_overwritten_and_identical_file_is_skipped(tmp_path: Path) -> None:
    upload_service = service(tmp_path)
    content = pdf_bytes("original")
    upload_service.validate_and_save([FakeUpload("guide.pdf", content)])
    unchanged = upload_service.validate_and_save([FakeUpload("guide.pdf", content)])
    assert unchanged.unchanged_filenames == ("guide.pdf",)
    with pytest.raises(UploadValidationError):
        upload_service.validate_and_save([FakeUpload("guide.pdf", pdf_bytes("different"))])
    assert (tmp_path / "uploads" / "guide.pdf").read_bytes() == content
