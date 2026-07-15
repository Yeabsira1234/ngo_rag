from dataclasses import replace
from itertools import count

from src.documents import Document


class TextChunker:
    """Split page documents into overlapping, metadata-preserving chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """Divide text into overlapping chunks."""
        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            return []

        chunks = []
        start = 0

        while start < len(cleaned_text):
            end = start + self.chunk_size
            chunk = cleaned_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Chunk documents while preserving their source and page metadata."""
        chunked_documents: list[Document] = []
        document_counters: dict[str, count] = {}
        for document in documents:
            chunk_indices = document_counters.setdefault(
                document.metadata.document_id, count()
            )
            text_chunks = self.split(document.page_content)
            chunked_documents.extend(
                Document(
                    page_content=text_chunk,
                    metadata=replace(
                        document.metadata,
                        chunk_index=next(chunk_indices),
                    ),
                )
                for text_chunk in text_chunks
            )

        return chunked_documents
