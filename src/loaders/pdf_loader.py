from pathlib import Path

import fitz

from src.documents import Document, DocumentMetadata


class PDFLoader:
    def load(self, pdf_path: str | Path) -> list[Document]:
        """Load a PDF as one document per page."""
        path = Path(pdf_path)

        with fitz.open(path) as pdf:
            return [
                Document(
                    page_content=page.get_text(),
                    metadata=DocumentMetadata(
                        source=path.name,
                        page_number=page_index,
                    ),
                )
                for page_index, page in enumerate(pdf, start=1)
            ]
