from pathlib import Path

import fitz

from src.documents import Document, DocumentMetadata
from src.discovery import DiscoveredDocument


class PDFLoader:
    """Extract text and source metadata from each page of a PDF."""

    def load(self, pdf_path: str | Path | DiscoveredDocument) -> list[Document]:
        """Load a PDF as one document per page."""
        discovered = pdf_path if isinstance(pdf_path, DiscoveredDocument) else None
        path = discovered.path if discovered else Path(pdf_path)

        with fitz.open(path) as pdf:
            return [
                Document(
                    page_content=page.get_text(),
                    metadata=DocumentMetadata(
                        source=path.name,
                        page_number=page_index,
                        source_relative_path=(
                            discovered.relative_path if discovered else path.name
                        ),
                        document_id=(
                            discovered.document_id if discovered else path.name
                        ),
                    ),
                )
                for page_index, page in enumerate(pdf, start=1)
            ]
