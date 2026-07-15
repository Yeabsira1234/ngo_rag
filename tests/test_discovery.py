from pathlib import Path

import pytest

from src.discovery import DocumentDiscoveryError, PDFDocumentDiscovery


def test_discovers_pdfs_in_stable_order_and_ignores_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_bytes(b"pdf")
    (tmp_path / "a.pdf").write_bytes(b"pdf")
    (tmp_path / ".hidden.pdf").write_bytes(b"pdf")
    (tmp_path / "empty.pdf").touch()
    (tmp_path / "notes.txt").write_text("not pdf")
    result = PDFDocumentDiscovery().discover(tmp_path)
    assert [item.relative_path for item in result.documents] == ["a.pdf", "b.pdf"]
    assert result.skipped_count == 2


def test_same_filename_in_different_folders_has_distinct_identity(tmp_path: Path) -> None:
    for folder in ("north", "south"):
        path = tmp_path / folder
        path.mkdir()
        (path / "guide.pdf").write_bytes(b"pdf")
    result = PDFDocumentDiscovery().discover(tmp_path, "**/*.pdf")
    assert len({item.document_id for item in result.documents}) == 2


def test_no_valid_documents_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "empty.pdf").touch()
    with pytest.raises(DocumentDiscoveryError, match="No valid PDF"):
        PDFDocumentDiscovery().discover(tmp_path)


def test_duplicate_paths_are_processed_once(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [pdf, pdf])
    result = PDFDocumentDiscovery().discover(tmp_path)
    assert [item.relative_path for item in result.documents] == ["guide.pdf"]
    assert result.skipped_count == 1
