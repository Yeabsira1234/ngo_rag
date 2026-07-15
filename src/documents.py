from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata that identifies a document passage and its origin."""

    source: str
    page_number: int
    chunk_index: int | None = None
    source_relative_path: str = ""
    document_id: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source cannot be empty.")
        if self.page_number <= 0:
            raise ValueError("page_number must be greater than zero.")
        if self.chunk_index is not None and self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")
        if not self.source_relative_path:
            object.__setattr__(self, "source_relative_path", self.source)
        if not self.document_id:
            object.__setattr__(self, "document_id", self.source)

    def to_dict(self) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "source": self.source,
            "page_number": self.page_number,
            "source_relative_path": self.source_relative_path,
            "document_id": self.document_id,
        }
        if self.chunk_index is not None:
            metadata["chunk_index"] = self.chunk_index
        return metadata


@dataclass(frozen=True, slots=True)
class Document:
    """A piece of text together with metadata describing its origin."""

    page_content: str
    metadata: DocumentMetadata
