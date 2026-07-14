from pathlib import Path

import fitz

from src.loaders.pdf_loader import PDFLoader


def test_load_returns_one_document_per_page_with_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "handbook.pdf"
    pdf = fitz.open()
    first_page = pdf.new_page()
    first_page.insert_text((72, 72), "First page content")
    second_page = pdf.new_page()
    second_page.insert_text((72, 72), "Second page content")
    pdf.save(pdf_path)
    pdf.close()

    documents = PDFLoader().load(pdf_path)

    assert len(documents) == 2
    assert "First page content" in documents[0].page_content
    assert documents[0].metadata.source == "handbook.pdf"
    assert documents[0].metadata.page_number == 1
    assert documents[0].metadata.chunk_index is None
    assert "Second page content" in documents[1].page_content
    assert documents[1].metadata.source == "handbook.pdf"
    assert documents[1].metadata.page_number == 2
